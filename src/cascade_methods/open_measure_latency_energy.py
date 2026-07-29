#!/usr/bin/env python3
"""Batch-1 latency + energy (NVML) for the open-text cascade tiers, on REAL images.
modes:  gen    = one open-ended generation (the cheap/strong answer leg)
        verify = one verifier forward (read Yes/No logit; the best-of-N selection cost)
Measures n real (image,question) items from vqa_rad after `warmup` discarded iters; logs per-iter
latency_s + energy_j (integrated NVML power over the call across all visible GPUs). Prints mean/median.
Launch from repo root. Pin GPUs via CUDA_VISIBLE_DEVICES (tp must match #visible GPUs).
"""
import argparse, os, io, glob, time, json, threading, math
import numpy as np, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image
import pynvml
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--mode", choices=["gen","verify"], required=True)
ap.add_argument("--adapter", default=""); ap.add_argument("--n", type=int, default=25)
ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--max_new", type=int, default=32)
ap.add_argument("--out", default=""); ap.add_argument("--device_map", default="")
ap.add_argument("--think", action="store_true", help="gen mode only: use the native <think> reasoning system prompt (long reasoning generation)")
A = ap.parse_args()
MAXPX, MINPX = 1280*28*28//4, 4*28*28  # cap320 (matches run_openvqa default)
DEV = "cuda"
SYS_GEN = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts within "
             "<think> </think> tags before providing the answer. After </think>, give only the short final answer.")
SYS_VER = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
           "proposed answer is correct. Respond with only 'Yes' or 'No'.")
# ---- NVML power sampler (integrates W over the timed call) ----
pynvml.nvmlInit()
# NVML uses PHYSICAL indices regardless of CUDA_VISIBLE_DEVICES -> sample ONLY the visible GPU(s),
# else energy is contaminated by the other card's load (e.g. a concurrent job on GPU0).
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
_toks = [x.strip() for x in _vis.split(",") if x.strip() != ""]
if _toks and all(t.isdigit() for t in _toks):
    _idxs = [int(t) for t in _toks]                       # integer indices: NVML physical index == visible index (PCI order)
else:                                                     # UUID-form or unset -> can't map to an NVML index reliably; sample all GPUs
    _idxs = list(range(pynvml.nvmlDeviceGetCount()))
HS = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in _idxs]; NG = len(HS)
def watts(): return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in HS)/1000.0  # W across VISIBLE GPUs only
class E:
    def __init__(s): s.go=False; s.j=0.0
    def __enter__(s):
        s.j=0.0; s.go=True; s.t=time.time()
        def loop():
            last=time.time()
            while s.go:
                time.sleep(0.01); now=time.time(); s.j += watts()*(now-last); last=now
        s.th=threading.Thread(target=loop); s.th.start(); return s
    def __exit__(s,*a): s.go=False; s.th.join()
# ---- load a few real items ----
df_files = sorted(glob.glob("/data/dan/dataset/vqa_rad/data/test-*.parquet"))
import pandas as pd
df = pd.concat([pd.read_parquet(f) for f in df_files], ignore_index=True)
items=[]
for _,r in df.iterrows():
    q=r.get("question"); a=r.get("answer")
    if q is None and "conversations" in r:
        conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
    if str(a).strip().lower() in ("yes","no"): continue
    img=r["image"]
    if isinstance(img,dict) and "bytes" in img:
        items.append((str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"), str(a)))
    if len(items)>=A.n+A.warmup: break
print(f"{A.tag}: {len(items)} items, mode={A.mode}, visible GPUs={NG}", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
kw = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
if A.device_map: kw["device_map"]="auto"
else: kw["attn_implementation"]="flash_attention_2"
model = AutoModelForImageTextToText.from_pretrained(A.model_path, **kw)
if not A.device_map: model = model.to(DEV)
if A.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.expanduser(A.adapter))
model.eval()
YES = proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO = proc.tokenizer.encode("No",add_special_tokens=False)[0]
def build(q,img,ans=None):
    if A.mode=="verify":
        body=f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        msgs=[{"role":"system","content":SYS_VER},{"role":"user","content":[
            {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},{"type":"text","text":body}]}]
    else:
        msgs=[{"role":"system","content":(SYS_THINK if A.think else SYS_GEN)},{"role":"user","content":[
            {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    igs,vids=process_vision_info(msgs)
    enc=proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True)
    return {k:(v.to(DEV) if hasattr(v,'to') else v) for k,v in enc.items()}
lat=[]; en=[]; pt=[]; gt=[]
for i,(q,img,ans) in enumerate(items):
    enc=build(q,img,ans); ptok=enc["input_ids"].shape[1]
    torch.cuda.synchronize()
    with E() as e:
        t0=time.time()
        with torch.no_grad():
            if A.mode=="verify":
                _=model(**enc).logits[0,-1]; ntok=1
            else:
                out=model.generate(**enc, max_new_tokens=A.max_new, do_sample=False)
                ntok=out.shape[1]-ptok
        torch.cuda.synchronize(); dt=time.time()-t0
    if i>=A.warmup: lat.append(dt); en.append(e.j); pt.append(ptok); gt.append(ntok)
    if i%10==0: print(f"  [{i}] {dt*1000:.0f}ms {e.j:.1f}J ptok={ptok} gtok={ntok}", flush=True)
res={"tag":A.tag,"mode":A.mode,"n":len(lat),"lat_ms_mean":float(np.mean(lat)*1000),
     "lat_ms_median":float(np.median(lat)*1000),"energy_j_mean":float(np.mean(en)),
     "prefill_tok_mean":float(np.mean(pt)),"gen_tok_mean":float(np.mean(gt)),"gpus":NG}
print("RESULT "+json.dumps(res), flush=True)
if A.out:
    with open(A.out,"a") as fh: fh.write(json.dumps(res)+"\n")
