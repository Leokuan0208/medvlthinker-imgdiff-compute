#!/usr/bin/env python3
"""
gate_router.py - unified compute-router gate for MedVLThinker-7B.
Per-sample OOM guard: retries at lower image resolution, skips (recorded) if still OOM,
so one oversized image can't kill the run.

Cells: (nothink,norag) (think,norag) (think,rag) [(nothink,rag) via --full_grid]
  --cells think_rag  : run only listed cell(s)
  --shard k/N        : row-stride split across VMs (shard-tagged checkpoints)

Record: idx, gold, pred, ok, parse_ok, opt_logprobs{A..J}, gen_tokens, latency_s, raw_output
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # bake in, don't rely on export
import argparse, re, json, random, time, torch
import torch.nn.functional as F
from collections import Counter
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL    = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT     = "/data/dan/dataset/MedVLThinker-Eval"
RETR_DIR = "/data/dan/retrieval_kb"        # retrieved_{name}_{corpus}_n{N}.jsonl
SYS_THINK   = ("You will solve a problem/request. You should provide your thoughts "
               "within <think> </think> tags before providing the answer.")
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28      # full res; OOM guard shrinks only offenders
RETRY_CAPS = [HIGH_PX, HIGH_PX//4, HIGH_PX//16, HIGH_PX//64]   # 1280 -> 640 -> 320 -> 160 img-tok
MAX_SNIP_CHARS  = 500

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--datasets", nargs="+",
                default=["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"])
ap.add_argument("--rag_corpus", default="Textbooks")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--ckpt_dir", default="ckpts/gate_7b_v2")
ap.add_argument("--full_grid", action="store_true")
ap.add_argument("--cells", nargs="+", default=None,
                help="run only these cells, e.g. --cells think_rag. Default = first three.")
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
CKPT_DIR = A.ckpt_dir
os.makedirs(CKPT_DIR, exist_ok=True)

MASTER = [("nothink","norag"), ("think","norag"), ("think","rag"), ("nothink","rag")]
if A.cells:
    want = set(A.cells)
    CELLS = [(r,g) for (r,g) in MASTER if f"{r}_{g}" in want]
    assert CELLS, f"no valid cells in {A.cells}; pick from {[f'{r}_{g}' for r,g in MASTER]}"
else:
    CELLS = [("nothink","norag"), ("think","norag"), ("think","rag")]
    if A.full_grid: CELLS.append(("nothink","rag"))

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]
print("SPLIT:", split, "| N:", len(data), "| shard:", A.shard,
      "| rag_corpus:", A.rag_corpus, "| cells:", [f"{r}_{g}" for r,g in CELLS], flush=True)

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
    "MedXpert-MM":            lambda: subset("medxpert"),
    "PMC-VQA":                lambda: subset("pmcvqa","pmc"),
    "SLAKE":                  lambda: subset("slake"),
    "VQA-RAD":                lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                lambda: subset("pathvqa","path"),
    "MMMU":                   lambda: subset("mmmu"),
}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]
    return s[SHARD_K::SHARD_N]
def load_retr(name):
    p = os.path.join(RETR_DIR, f"retrieved_{name}_{A.rag_corpus}_n{A.n}.jsonl")
    if not os.path.exists(p): return None
    r = {}
    for line in open(p):
        line=line.strip()
        if line:
            d=json.loads(line); r[d["idx"]]=d["snippets"]
    return r

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")
LETTER_SET = set(chr(ord('A')+k) for k in range(10))
def _lid(L):
    for v in (L, " "+L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: return e[0]
    return None
LID = {L:_lid(L) for L in LETTER_SET}

def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]
def pred(t):
    tail = t.split("</think>")[-1]
    m = LET.search(tail) or LET.search(t)
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

def build(ex, idx, rmode, gmode, retr, cap):
    opts = parse_opts(ex["options"]); assert opts, "no options parsed"
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    if gmode=="rag":
        snips = [s[:MAX_SNIP_CHARS] for s in (retr.get(idx, []) if retr else [])]
        ctx = "\n\n".join(f"[{n+1}] {s}" for n,s in enumerate(snips))
        q = ("Relevant medical references:\n"+ctx+
             "\n\nUsing the references above where helpful, answer the question:\n"+q)
    sys_prompt = SYS_NOTHINK if rmode=="nothink" else SYS_THINK
    img=[{"type":"image","image":im,"max_pixels":cap,"min_pixels":MIN_PX}
         for im in (ex.get("images") or [])]
    msgs=[{"role":"system","content":sys_prompt},
          {"role":"user","content":img+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    return text, imgs

def run_cell(name, sel, rmode, gmode, retr):
    cell = f"{rmode}_{gmode}" + (f"_{A.rag_corpus}" if gmode=="rag" else "")
    maxtok = 8 if rmode=="nothink" else 512
    ckpt = os.path.join(CKPT_DIR, f"ckpt_{name}_{cell}_s{SHARD_K}of{SHARD_N}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            line=line.strip()
            if line:
                try: r=json.loads(line); done[r["idx"]]=int(r["ok"])
                except Exception: pass
    correct = sum(done[i] for i in sel if i in done)
    tot     = sum(1       for i in sel if i in done)
    print(f"   [{cell}] {ckpt} | resuming {tot}/{len(sel)}", flush=True)
    with open(ckpt,"a") as fh:
        for j,i in enumerate(sel):
            if i in done: continue
            ex=data[i]; o=None; inp=None; dt=0.0
            for cap in RETRY_CAPS:                       # OOM guard: shrink image, retry
                try:
                    text,imgs=build(ex,i,rmode,gmode,retr,cap)
                    inp=proc(text=[text],images=imgs,return_tensors="pt").to(model.device)
                    t0=time.time()
                    with torch.no_grad():
                        o=model.generate(**inp,max_new_tokens=maxtok,do_sample=False,
                                         output_scores=True,return_dict_in_generate=True)
                    dt=time.time()-t0; break
                except torch.cuda.OutOfMemoryError:
                    o=None
                    if inp is not None: del inp; inp=None
                    torch.cuda.empty_cache()
                    print(f"   [{j+1}] OOM at {cap//784} img-tok -> retry smaller", flush=True)
            if o is None:                                # still OOM at smallest cap -> record + skip
                torch.cuda.empty_cache(); tot+=1
                fh.write(json.dumps({"idx":i,"gold":gold(ex),"pred":"?","ok":0,
                    "parse_ok":0,"opt_logprobs":{},"gen_tokens":0,
                    "latency_s":0.0,"raw_output":"<OOM_SKIPPED>"})+"\n"); fh.flush()
                print(f"   [{j+1}/{len(sel)}] {cell} OOM-SKIP", flush=True); continue
            seq=o.sequences[:,inp.input_ids.shape[1]:]
            gen=proc.batch_decode(seq,skip_special_tokens=True)[0]
            g=gold(ex); p,parse_ok=pred(gen)
            lp=opt_logprob(seq[0],o.scores); gen_tokens=int(seq.shape[1])
            ok=int(g==p); correct+=ok; tot+=1
            del inp,o,seq; torch.cuda.empty_cache()
            fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":ok,
                "parse_ok":parse_ok,"opt_logprobs":lp,"gen_tokens":gen_tokens,
                "latency_s":round(dt,3),"raw_output":gen})+"\n"); fh.flush()
            tl=gen.split("</think>")[-1].strip().replace("\n"," ")[:42]
            print(f"   [{j+1}/{len(sel)}] {cell} g={g} p={p} {'OK' if ok else 'XX'} "
                  f"acc={correct/tot:.3f} tok={gen_tokens} {dt:.1f}s | {tl!r}", flush=True)
    acc=correct/tot if tot else 0.0
    print(f">> {name} [{cell}] shard {A.shard} n={tot} ACC={acc:.3f}", flush=True)
    return acc

print("="*64, "\nROUTER GATE | MedVLThinker-7B | cells:", CELLS, "| shard:", A.shard, flush=True)
summary={}
for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! unknown dataset {name}", flush=True); continue
    idxs=DATASET_IDX[name]()
    if not idxs: print(f"!! empty subset {name}", flush=True); continue
    sel=fixed_slice(idxs); retr=load_retr(name)
    print(f"\n--- {name} (pool={len(idxs)} shard_n={len(sel)} retr={'yes' if retr else 'NONE'}) ---", flush=True)
    for rmode,gmode in CELLS:
        if gmode=="rag" and retr is None:
            print(f"   [skip] {rmode}_{gmode}: no retrieval cache for {name}/{A.rag_corpus}", flush=True); continue
        key=f"{name}/{rmode}_{gmode}"+(f"_{A.rag_corpus}" if gmode=="rag" else "")
        summary[key]=run_cell(name, sel, rmode, gmode, retr)
print("\n===== SUMMARY (shard "+A.shard+") =====", flush=True)
for k,v in summary.items(): print(f"  {k}: {v:.3f}", flush=True)
