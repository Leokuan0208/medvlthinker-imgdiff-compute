#!/usr/bin/env python3
"""
run_7b_prune_sweep.py - 7B-nothink vision-token-BUDGET sweep (resolution-based), vLLM.

Reduces the 7B's vision tokens by lowering max_pixels (native to vLLM; no model surgery).
Loads the 7B ONCE, then for each budget labels:
  - the 4 EVAL sets  -> ckpts/gate_7b_prune/cap<tok>/         (compare vs full-token 32B)
  - the PMC-TRAIN set -> ckpts/gate_7b_pmctrain_prune/cap<tok>/ (recalibrate the gate per budget)
Schema/prompt/opt_logprobs mirror run_7b_vllm.py (eval, A-J) and run_pmctrain_vllm.py
(train, A-D) EXACTLY, so the analysis scripts read these unchanged. Fully resumable
(skips done idx per file). 32B is NOT re-run: escalated questions keep FULL tokens.

NOTE: this is RESOLUTION reduction, not random patch-drop. It is the overnight-safe way
to get the accuracy/confidence-vs-vision-budget curve in vLLM; random-drop (HF-eager) is a
separate build. gen_tokens is recorded so prefill-inclusive cost can be recomputed per budget.
"""
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
EVAL_DATASETS = ["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA","SLAKE","VQA-RAD","PathVQA"]

ap = argparse.ArgumentParser()
ap.add_argument("--shard", default="0/1")                 # VM1: 0/2  VM2: 1/2
ap.add_argument("--n", type=int, default=4000)            # per-eval-dataset cap (matches run_7b_vllm)
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA","SLAKE","VQA-RAD","PathVQA"])
ap.add_argument("--train_manifest", default="/data/dan/dataset/pmc_vqa_train/train_sample_3000.jsonl")
ap.add_argument("--budgets", default="", help="comma list of max_pixels divisors; default 2,4,8,16")
ap.add_argument("--eval_dir",  default="ckpts/gate_7b_prune")
ap.add_argument("--train_dir", default="ckpts/gate_7b_pmctrain_prune")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_images", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=16)
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"  # single shard -> no suffix
DIVS = [int(x) for x in A.budgets.split(",")] if A.budgets else [2, 4, 8, 16]
BUDGETS = [HIGH_PX // d for d in DIVS]                     # max_pixels values to sweep

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
train_rows = [json.loads(l) for l in open(A.train_manifest) if l.strip()]

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
def gold(x): return str(x).strip().upper()[:1]

proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")
def make_ids(letters):
    def _lid(L):
        for v in (L," "+L):
            e=proc.tokenizer.encode(v,add_special_tokens=False)
            if e: return e[0]
        return None
    lid={L:_lid(L) for L in letters}; return {v:k for k,v in lid.items() if v is not None}
ID2LET_J = make_ids([chr(ord('A')+k) for k in range(10)])   # eval: A-J
ID2LET_D = make_ids(list("ABCD"))                            # train: A-D
def pred_from_text(t):
    m=LET.search(t); return (m.group(1),1) if m else ("?",0)
def opt_lp(tok, lps, id2let):
    for step in range(len(tok)):
        d=proc.tokenizer.decode([tok[step]]).strip()
        if d and d[0] in set("ABCDEFGHIJ") and (len(d)==1 or d[1] in ").: "):
            lp=lps[step] if step<len(lps) else None
            if not lp: return {}
            return {id2let[t]:round(float(o.logprob),4) for t,o in lp.items() if t in id2let}
    return {}
def build(images, q, max_pixels):
    im=[{"type":"image","image":x,"max_pixels":max_pixels,"min_pixels":MIN_PX} for x in (images or [])]
    msgs=[{"role":"system","content":SYS_NOTHINK},{"role":"user","content":im+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    r={"prompt":text}
    if imgs: r["multi_modal_data"]={"image":imgs}
    return r

print(f"loading 7B vLLM ONCE (TP={A.tp}); budgets(max_pixels)={BUDGETS} "
      f"(~token caps {[b//(28*28) for b in BUDGETS]}); shard {SHARD_K}/{SHARD_N}", flush=True)
llm=LLM(model=MODEL,tensor_parallel_size=A.tp,dtype="bfloat16",gpu_memory_utilization=A.gpu_mem,
        max_model_len=A.max_model_len,limit_mm_per_prompt={"image":A.max_images},trust_remote_code=True, disable_custom_all_reduce=True)
sp=SamplingParams(temperature=0.0,max_tokens=A.max_tokens,logprobs=20)

def run_file(ckpt, items, build_fn, gold_fn, id2let, label):
    done=set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    todo=[it for it in items if it[0] not in done]
    print(f"  [{label}] {len(items)} total, {len(done)} done, {len(todo)} to run -> {ckpt}", flush=True)
    t0=time.time(); tn=0
    with open(ckpt,"a") as fh:
        for c0 in range(0,len(todo),CHUNK):
            ch=todo[c0:c0+CHUNK]; reqs=[build_fn(it) for it in ch]
            try: outs=llm.generate(reqs,sp)
            except Exception as e:
                print(f"     chunk failed ({e}); one-by-one",flush=True); outs=[]
                for r in reqs:
                    try: outs.append(llm.generate([r],sp)[0])
                    except Exception as e2: print(f"       skip: {e2}",flush=True); outs.append(None)
            nc=nd=0
            for it,o in zip(ch,outs):
                if o is None: continue
                gen=o.outputs[0].text; tk=list(o.outputs[0].token_ids); lps=o.outputs[0].logprobs or []
                g=gold_fn(it); p,pk=pred_from_text(gen); ok=int(g==p)
                fh.write(json.dumps({"idx":it[0],"gold":g,"pred":p,"ok":ok,"parse_ok":pk,
                    "opt_logprobs":opt_lp(tk,lps,id2let),"gen_tokens":len(tk),"latency_s":None,"raw_output":gen})+"\n")
                nc+=ok; nd+=1; tn+=1
            fh.flush()
            print(f"     [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} | {tn/max(time.time()-t0,1e-9):.2f} samp/s",flush=True)
    print(f"  [{label}] done", flush=True)

for B in BUDGETS:
    tok=B//(28*28)
    print(f"\n========== BUDGET max_pixels={B} (~{tok} vision-token cap) ==========", flush=True)
    for name in A.datasets:
        if name not in DATASET_IDX: print(f"  !! unknown {name}",flush=True); continue
        sel=fixed_slice(DATASET_IDX[name]())
        d=os.path.join(A.eval_dir,f"cap{tok}"); os.makedirs(d,exist_ok=True)
        ckpt=os.path.join(d,f"ckpt_{name}_nothink_norag{SHARD_TAG}.jsonl")
        items=[(i,) for i in sel]
        bf=lambda it,_B=B: build(data[it[0]].get("images") or [],
                                 data[it[0]]["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in parse_opts(data[it[0]]["options"]).items()), _B)
        run_file(ckpt, items, bf, lambda it: gold(data[it[0]]["answer_label"]), ID2LET_J, f"{name} cap{tok}")
    tr=[r for j,r in enumerate(train_rows) if j % SHARD_N == SHARD_K]
    d=os.path.join(A.train_dir,f"cap{tok}"); os.makedirs(d,exist_ok=True)
    ckpt=os.path.join(d,f"ckpt_nothink{SHARD_TAG}.jsonl")
    items=[(r["idx"], r) for r in tr]
    bf=lambda it,_B=B: build([it[1]["image_path"]],
                             it[1]["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in it[1]["options"].items()), _B)
    run_file(ckpt, items, bf, lambda it: gold(it[1]["answer_label"]), ID2LET_D, f"pmctrain cap{tok}")

print("\nDONE all budgets.", flush=True)
