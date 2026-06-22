#!/usr/bin/env python3
"""
run_pmctrain_vllm.py - run a model on the PMC-VQA TRAIN sample JSONL (held-out
threshold-fitting data). Mirrors run_32b_vllm.py's prompt/margin/parsing EXACTLY
so labels are comparable to the eval-set labels. Reads images from disk paths.

--model_path + --arm select which model/mode:
  32B think   : --model_path .../MedVLThinker-32B-RL_m23k --arm think  --max_tokens 2048
  cheap-7B    : --model_path .../MedVLThinker-7B-RL_m23k  --arm nothink --max_tokens 16
Output: ckpts/pmctrain/ckpt_<arm>.jsonl  (idx,gold,pred,ok,parse_ok,opt_logprobs,...)
"""
import argparse, re, json, time, os
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

SYS_THINK   = ("You will solve a problem/request. You should provide your thoughts "
               "within <think> </think> tags before providing the answer.")
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
CHUNK = 128

ap = argparse.ArgumentParser()
ap.add_argument("--sample", default="/data/dan/dataset/pmc_vqa_train/train_sample_3000.jsonl")
ap.add_argument("--model_path", required=True)
ap.add_argument("--arm", choices=["think","nothink"], required=True)
ap.add_argument("--cap", choices=list(CAP_DIV), default="fullres")
ap.add_argument("--ckpt_dir", default="ckpts/pmctrain")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=2048)
ap.add_argument("--max_images", type=int, default=8)
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
SYS = SYS_THINK if A.arm=="think" else SYS_NOTHINK
MAXPX = HIGH_PX // CAP_DIV[A.cap]

rows = [json.loads(l) for l in open(A.sample) if l.strip()]
print(f"loaded {len(rows)} train-sample rows | arm={A.arm} max_tokens={A.max_tokens}", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path)
LET = re.compile(r"\b([A-D])\b")
LETTER_SET = set("ABCD")
def _lid(L):
    for v in (L," "+L):
        e=proc.tokenizer.encode(v,add_special_tokens=False)
        if e: return e[0]
    return None
LID={L:_lid(L) for L in LETTER_SET}; ID2LET={v:k for k,v in LID.items() if v is not None}

def pred_from_text(t):
    if A.arm=="think":
        tail=t.split("</think>")[-1]; m=LET.search(tail) or LET.search(t)
    else:
        m=LET.search(t)
    return (m.group(1),1) if m else ("?",0)
def opt_lp(tok,lps):
    seen=""; start=0
    if A.arm=="think":
        for step,tk in enumerate(tok):
            seen+=proc.tokenizer.decode([tk])
            if "</think>" in seen: start=step+1; break
    for step in range(start,len(tok)):
        d=proc.tokenizer.decode([tok[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d)==1 or d[1] in ").: "):
            lp=lps[step] if step<len(lps) else None
            if not lp: return {}
            return {ID2LET[t]:round(float(o.logprob),4) for t,o in lp.items() if t in ID2LET}
    return {}
def build(r):
    q=r["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in r["options"].items())
    img=[{"type":"image","image":r["image_path"],"max_pixels":MAXPX,"min_pixels":MIN_PX}]
    msgs=[{"role":"system","content":SYS},{"role":"user","content":img+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    req={"prompt":text}
    if imgs: req["multi_modal_data"]={"image":imgs}
    return req

print(f"loading model {A.model_path} (TP={A.tp})...", flush=True)
llm=LLM(model=A.model_path,tensor_parallel_size=A.tp,dtype="bfloat16",gpu_memory_utilization=A.gpu_mem,
        max_model_len=A.max_model_len,limit_mm_per_prompt={"image":A.max_images},trust_remote_code=True)
sp=SamplingParams(temperature=0.0,max_tokens=A.max_tokens,logprobs=20)

ckpt=os.path.join(A.ckpt_dir,f"ckpt_{A.arm}.jsonl")
done=set()
if os.path.exists(ckpt):
    for l in open(ckpt):
        if l.strip():
            try: done.add(json.loads(l)["idx"])
            except: pass
todo=[r for r in rows if r["idx"] not in done]
print(f"{len(done)} done, {len(todo)} to run -> {ckpt}", flush=True)

t0=time.time(); tn=0
with open(ckpt,"a") as fh:
    for c0 in range(0,len(todo),CHUNK):
        ch=todo[c0:c0+CHUNK]; reqs=[build(r) for r in ch]
        try: outs=llm.generate(reqs,sp)
        except Exception as e:
            print(f"   chunk failed ({e}); one-by-one",flush=True); outs=[]
            for rq in reqs:
                try: outs.append(llm.generate([rq],sp)[0])
                except Exception as e2: print(f"     skip: {e2}",flush=True); outs.append(None)
        nc=nd=0
        for r,o in zip(ch,outs):
            if o is None: continue
            gen=o.outputs[0].text; tk=list(o.outputs[0].token_ids); lps=o.outputs[0].logprobs or []
            g=r["answer_label"]; p,pk=pred_from_text(gen); ok=int(g==p)
            fh.write(json.dumps({"idx":r["idx"],"gold":g,"pred":p,"ok":ok,"parse_ok":pk,
                "opt_logprobs":opt_lp(tk,lps),"gen_tokens":len(tk),"latency_s":None,"raw_output":gen})+"\n")
            nc+=ok; nd+=1; tn+=1
        fh.flush(); el=time.time()-t0
        print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} | {tn/el:.2f} samp/s",flush=True)
print(f"\nDONE {A.arm} {tn} samples in {(time.time()-t0)/60:.1f} min",flush=True)
