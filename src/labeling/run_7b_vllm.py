#!/usr/bin/env python3
"""7B cheap-arm (nothink_norag) vLLM runner — full eval set. Same stack as the
32B run so cascade comparison is confound-free. Writes ckpts/gate_7b_vllm/."""
import argparse, re, json, random, time, os
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

MODEL = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CHUNK = 256

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--datasets", nargs="+",
                default=["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA","SLAKE","VQA-RAD","PathVQA"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--ckpt_dir", default="ckpts/gate_7b_vllm")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=16)
ap.add_argument("--max_images", type=int, default=8)
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"  # single shard -> no suffix
os.makedirs(A.ckpt_dir, exist_ok=True)
ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]

def subset(*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc=data[i].get("misc")
        try: qt=json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt=""
        if qt.lower()==t: out.append(i)
    return out
DATASET_IDX = {
    "MedXpert-Reasoning":               lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding":           lambda: mx_by_type("understanding"),
    "PMC-VQA":                          lambda: subset("pmcvqa","pmc"),
    "SLAKE":                            lambda: subset("slake"),
    "VQA-RAD":                          lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                          lambda: subset("pathvqa","path"),
    "MMMU":                             lambda: subset("mmmu"),
}
def fixed_slice(idxs):
    rng=random.Random(42); s=idxs[:]; rng.shuffle(s); s=s[:A.n]; return s[SHARD_K::SHARD_N]
def parse_opts(s):
    if isinstance(s,dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")
LETTER_SET = set(chr(ord('A')+k) for k in range(10))
def _lid(L):
    for v in (L," "+L):
        e=proc.tokenizer.encode(v,add_special_tokens=False)
        if e: return e[0]
    return None
LID={L:_lid(L) for L in LETTER_SET}; ID2LET={v:k for k,v in LID.items() if v is not None}
def pred_from_text(t):
    m=LET.search(t); return (m.group(1),1) if m else ("?",0)
def opt_lp(token_ids, logprobs_list):
    for step in range(len(token_ids)):
        d=proc.tokenizer.decode([token_ids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d)==1 or d[1] in ").: "):
            lp=logprobs_list[step] if step<len(logprobs_list) else None
            if not lp: return {}
            return {ID2LET[t]:round(float(o.logprob),4) for t,o in lp.items() if t in ID2LET}
    return {}
def build(ex):
    opts=parse_opts(ex["options"]); assert opts
    q=ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    im=[{"type":"image","image":x,"max_pixels":HIGH_PX,"min_pixels":MIN_PX} for x in (ex.get("images") or [])]
    msgs=[{"role":"system","content":SYS_NOTHINK},{"role":"user","content":im+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    r={"prompt":text}
    if imgs: r["multi_modal_data"]={"image":imgs}
    return r

print(f"loading 7B vLLM (TP={A.tp}, nothink, max_tokens={A.max_tokens})...", flush=True)
llm=LLM(model=MODEL,tensor_parallel_size=A.tp,dtype="bfloat16",gpu_memory_utilization=A.gpu_mem,
        max_model_len=A.max_model_len,limit_mm_per_prompt={"image":A.max_images},trust_remote_code=True)
sp=SamplingParams(temperature=0.0,max_tokens=A.max_tokens,logprobs=20)
t0=time.time(); tn=0
for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! {name}",flush=True); continue
    sel=fixed_slice(DATASET_IDX[name]())
    ckpt=os.path.join(A.ckpt_dir,f"ckpt_{name}_nothink_norag{SHARD_TAG}.jsonl")
    done=set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except: pass
    todo=[i for i in sel if i not in done]
    print(f"\n--- {name}: {len(sel)} total, {len(done)} done, {len(todo)} to run ---",flush=True)
    with open(ckpt,"a") as fh:
        for c0 in range(0,len(todo),CHUNK):
            ch=todo[c0:c0+CHUNK]; reqs=[build(data[i]) for i in ch]
            try: outs=llm.generate(reqs,sp)
            except Exception as e:
                print(f"   chunk failed ({e}); one-by-one",flush=True); outs=[]
                for r in reqs:
                    try: outs.append(llm.generate([r],sp)[0])
                    except Exception as e2: print(f"     skip: {e2}",flush=True); outs.append(None)
            nc=nd=0
            for i,o in zip(ch,outs):
                if o is None: continue
                gen=o.outputs[0].text; tk=list(o.outputs[0].token_ids); lps=o.outputs[0].logprobs or []
                g=gold(data[i]); p,pk=pred_from_text(gen); ok=int(g==p)
                fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":ok,"parse_ok":pk,
                    "opt_logprobs":opt_lp(tk,lps),"gen_tokens":len(tk),"latency_s":None,"raw_output":gen})+"\n")
                nc+=ok; nd+=1; tn+=1
            fh.flush(); el=time.time()-t0
            print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} | {tn/el:.2f} samp/s",flush=True)
    print(f">> {name} done",flush=True)
print(f"\nDONE 7B {tn} samples in {(time.time()-t0)/60:.1f} min",flush=True)
