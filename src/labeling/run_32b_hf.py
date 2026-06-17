#!/usr/bin/env python3
"""
run_32b.py - run MedVLThinker-32B (think mode) on the SAME seed-42 500-question
slices the 7B gate used, so 7B and 32B labels align per idx for a cross-model
complementarity (cascade) check. bf16 sharded across 2 GPUs via device_map=auto.

Records per sample: idx, gold, pred, ok, parse_ok, opt_logprobs, gen_tokens,
latency_s, raw_output  -> ckpts/gate_32b/ckpt_<dataset>_think_norag.jsonl
Resumable. Mirrors gate_router.py exactly except model path + single arm (think).
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, re, json, random, time, torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "/data/dan/weights/MedVLThinker-32B-RL_m23k"     # <-- confirm this path exists
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts "
             "within <think> </think> tags before providing the answer.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
RETRY_CAPS = [HIGH_PX, HIGH_PX//4, HIGH_PX//16, HIGH_PX//64]
MAX_SNIP = 500

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--datasets", nargs="+",
                default=["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--ckpt_dir", default="ckpts/gate_32b")
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"  # single shard -> no suffix
os.makedirs(A.ckpt_dir, exist_ok=True)

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]

def subset(*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt = ""
        if qt.lower()==t: out.append(i)
    return out
DATASET_IDX = {
    "MedXpert-Reasoning":     lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type("understanding"),
    "PMC-VQA":                lambda: subset("pmcvqa","pmc"),
    "SLAKE":                  lambda: subset("slake"),
    "VQA-RAD":                lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                lambda: subset("pathvqa","path"),
}
def fixed_slice(idxs):                              # IDENTICAL to gate_router.py
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]
    return s[SHARD_K::SHARD_N]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

print("loading 32B across all visible GPUs (device_map=auto, bf16)...", flush=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL)
print("device map:", set(str(v) for v in model.hf_device_map.values()), flush=True)

LET = re.compile(r"\b([A-J])\b")
LETTER_SET = set(chr(ord('A')+k) for k in range(10))
def _lid(L):
    for v in (L, " "+L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: return e[0]
    return None
LID = {L:_lid(L) for L in LETTER_SET}
def pred(t):
    tail = t.split("</think>")[-1]; m = LET.search(tail) or LET.search(t)
    return (m.group(1), 1) if m else ("?", 0)
def opt_logprob(seq_ids, scores):
    ids = seq_ids.tolist(); seen=""; start=0
    for step,tok in enumerate(ids):
        seen += proc.tokenizer.decode([tok])
        if "</think>" in seen: start=step+1; break
    for step in range(start, min(len(ids), len(scores))):
        d = proc.tokenizer.decode([ids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d)==1 or d[1] in ").: "):
            lp = F.log_softmax(scores[step][0].float(), dim=-1)
            return {L: round(float(lp[LID[L]]),4) for L in sorted(LETTER_SET) if LID[L] is not None}
    return {}
def build(ex, cap):
    opts = parse_opts(ex["options"]); assert opts
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    img=[{"type":"image","image":im,"max_pixels":cap,"min_pixels":MIN_PX}
         for im in (ex.get("images") or [])]
    msgs=[{"role":"system","content":SYS_THINK},
          {"role":"user","content":img+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    return text, imgs

for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! unknown {name}", flush=True); continue
    sel = fixed_slice(DATASET_IDX[name]())
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{name}_think_norag{SHARD_TAG}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    print(f"\n--- {name}: {len(sel)} samples, {len(done)} done -> {ckpt} ---", flush=True)
    correct = tot = 0
    with open(ckpt, "a") as fh:
        for j,i in enumerate(sel):
            if i in done: continue
            ex=data[i]; o=None; inp=None; dt=0.0
            for cap in RETRY_CAPS:
                try:
                    text,imgs=build(ex,cap)
                    inp=proc(text=[text],images=imgs,return_tensors="pt").to(model.device)
                    t0=time.time()
                    with torch.no_grad():
                        o=model.generate(**inp,max_new_tokens=512,do_sample=False,
                                         output_scores=True,return_dict_in_generate=True)
                    dt=time.time()-t0; break
                except torch.cuda.OutOfMemoryError:
                    o=None
                    if inp is not None: del inp; inp=None
                    torch.cuda.empty_cache()
                    print(f"   [{j+1}] OOM at cap {cap//784} -> retry", flush=True)
            if o is None:
                fh.write(json.dumps({"idx":i,"gold":gold(ex),"pred":"?","ok":0,"parse_ok":0,
                    "opt_logprobs":{},"gen_tokens":0,"latency_s":0.0,"raw_output":"<OOM_SKIPPED>"})+"\n")
                fh.flush(); print(f"   [{j+1}] OOM-SKIP idx={i}", flush=True); continue
            seq=o.sequences[:,inp.input_ids.shape[1]:]
            gen=proc.batch_decode(seq,skip_special_tokens=True)[0]
            g=gold(ex); p,parse_ok=pred(gen); lp=opt_logprob(seq[0],o.scores)
            ok=int(g==p); correct+=ok; tot+=1
            del inp,o,seq; torch.cuda.empty_cache()
            fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":ok,"parse_ok":parse_ok,
                "opt_logprobs":lp,"gen_tokens":int(seq.shape[1]) if 'seq' in dir() else 0,
                "latency_s":round(dt,3),"raw_output":gen})+"\n"); fh.flush()
            if tot % 10 == 0 or j+1 == len(sel):
                print(f"   [{j+1}/{len(sel)}] acc={correct/max(tot,1):.3f} {dt:.1f}s", flush=True)
    print(f">> {name} 32B acc={correct/max(tot,1):.3f} (n={tot})", flush=True)
print("\nDONE", flush=True)
