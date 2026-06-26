#!/usr/bin/env python3
"""Best-of-N test-time-scaling curve for the trained free-text verifier: does verifier-selected accuracy
keep improving as you sample more (K=1,2,4,8)? Loads the pooled-4 adapter, scores all 8 samples per
held-out question (seed-0 grouped split = the §5.10 test set), then computes verifier-best-of-K vs oracle@K
vs random@K for K in {1,2,4,8}. A rising verifier curve = the method benefits from test-time compute.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/verifier_scaling_curve.py
"""
import os, json, glob, io, math, random
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ADAPTER="ckpts/train/lora_verifier_pooled4"; MODEL="lingshu-medical-mllm/Lingshu-7B"; SEED=0
SC=os.environ.get("SC_TAG","sc8")           # sc8 (K<=8) or sc16 (K<=16)
KS=[1,2,4,8,16] if SC=="sc16" else [1,2,4,8]
OUTNAME="scaling_curve16.json" if SC=="sc16" else "scaling_curve.json"
DEV="cuda"; MAXPX,MINPX=1280*28*28,4*28*28
SYS=("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
     "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK=os.path.join(ROOT,"ckpts/openvqa/cheap_lingshu7b")
def loadj(p): return {r["idx"]:r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower()
def slake_imgs():
    m={}
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type")=="OPEN" and x.get("q_lang")=="en":
            ip=os.path.join("/data/dan/dataset/slake/imgs",x["img_name"])
            if os.path.exists(ip): m[x["qid"]]=(x["question"],ip)
    return m
def parquet_imgs(base):
    import pandas as pd; m={}
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))],ignore_index=True)
    for i,r in df.iterrows():
        q=r.get("question"); a=r.get("answer")
        if q is None and "conversations" in r:
            conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
        if str(a).strip().lower() in ("yes","no"): continue
        img=r["image"]
        if isinstance(img,dict) and "bytes" in img: m[int(i)]=(str(q),Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
def kvasir_imgs():
    m={}
    for r in json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")):
        if os.path.exists(r["img_path"]): m[r["idx"]]=(r["question"],r["img_path"])
    return m
IMG={"slake_open":slake_imgs(),"pathvqa_open":parquet_imgs("/data/dan/dataset/path_vqa/data"),
     "vqa_rad_open":parquet_imgs("/data/dan/dataset/vqa_rad/data"),"kvasir_open":kvasir_imgs()}
QREC={}
for ds in ["slake_open","pathvqa_open","vqa_rad_open","kvasir_open"]:
    sc=loadj(f"{CK}/ckpt_{ds}_lingshu7b_{SC}.jsonl"); exp=loadj(f"{CK}/ckpt_{ds}_lingshu7b_{SC}_scexploded.jsonl")
    jud={k:v["judge_ok"] for k,v in loadj(f"{CK}/ckpt_{ds}_lingshu7b_{SC}_scexploded.judge.jsonl").items()}
    aj=defaultdict(dict)
    for cid,r in exp.items():
        if cid in jud:
            oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])]=jud[cid]
    for i in sc:
        if i in IMG[ds] and i in aj:
            q,img=IMG[ds][i]; QREC[(ds,i)]={"q":q,"img":img,"preds":sc[i]["preds"],"slabels":aj[i]}
keys=list(QREC.keys()); rng=random.Random(SEED); rng.shuffle(keys)
test_keys=keys[int(0.7*len(keys)):]
print(f"test questions={len(test_keys)}",flush=True)
proc=AutoProcessor.from_pretrained(MODEL)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
print("loading base + pooled-4 adapter...",flush=True)
model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
model=PeftModel.from_pretrained(model,os.path.join(ROOT,ADAPTER)); model.eval()
def pyes(q,img,a):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[
        {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},
        {"type":"text","text":f"Question: {q}\nProposed answer: {a}\nIs the proposed answer correct? Answer Yes or No."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    igs,vids=process_vision_info(msgs); enc=proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True).to(DEV)
    with torch.no_grad():
        lg=model(**enc).logits[0,-1]; return math.exp(lg[YES].item())/(math.exp(lg[YES].item())+math.exp(lg[NO].item())+1e-9)
# score all 8 per test question, store (labels, scores)
rows=[]
for k in test_keys:
    r=QREC[k]; preds=r["preds"]; sl=[r["slabels"].get(norm(a)) for a in preds]
    if all(x is None for x in sl): continue
    sc=[pyes(r["q"],r["img"],a) for a in preds]
    rows.append((sl,sc))
print(f"scored {len(rows)} questions",flush=True)
Ks=KS; out={}
rngc=np.random.default_rng(0)
for K in Ks:
    ver=[];orc=[];rnd=[]
    for sl,sc in rows:
        idx=list(range(min(K,len(sl))))
        lv=[sl[i] for i in idx if sl[i] is not None]
        if not lv: continue
        # verifier: argmax score among first K (with a label)
        cand=[i for i in idx if sl[i] is not None]
        kv=max(cand,key=lambda i:sc[i]); ver.append(sl[kv])
        orc.append(max(lv)); rnd.append(sl[cand[int(rngc.integers(len(cand)))]])
    out[K]={"verifier":float(np.mean(ver)),"oracle":float(np.mean(orc)),"random":float(np.mean(rnd)),"n":len(ver)}
print("\n==================== VERIFIER BEST-of-K SCALING (pooled-4 free-text, held-out) ====================")
print(f"{'K':>3} {'random':>8} {'verifier':>9} {'oracle@K':>9}")
for K in Ks: print(f"{K:>3} {out[K]['random']:>8.3f} {out[K]['verifier']:>9.3f} {out[K]['oracle']:>9.3f}")
# save per-question (labels, scores) so any CI can be computed offline (never re-run the GPU pass)
json.dump([{"sl":sl,"sc":sc} for sl,sc in rows], open(os.path.join(ROOT,ADAPTER,f"perq_{SC}.json"),"w"))
# bootstrap CI on the verifier gain at max K (verifier@maxK - first-sample@1), 2000 resamples
maxK=max(Ks)
def sel(sl,sc,K):
    cand=[i for i in range(min(K,len(sl))) if sl[i] is not None]
    return sl[max(cand,key=lambda i:sc[i])] if cand else None
pairs=[(sel(sl,sc,maxK), (sl[0] if sl[0] is not None else None)) for sl,sc in rows]
pairs=[(v,g) for v,g in pairs if v is not None and g is not None]
v=np.array([p[0] for p in pairs]); g=np.array([p[1] for p in pairs]); n=len(v)
rb=np.random.default_rng(0); diffs=[]
for _ in range(2000):
    ix=rb.integers(0,n,n); diffs.append(v[ix].mean()-g[ix].mean())
lo,hi=np.percentile(diffs,[2.5,97.5])
print(f"\nbootstrap (n={n}, 2000 resamples): verifier@{maxK} {v.mean():.3f} vs first-sample {g.mean():.3f} | "
      f"gain {v.mean()-g.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
out["bootstrap_gain_vs_K1"]={"gain":float(v.mean()-g.mean()),"ci_lo":float(lo),"ci_hi":float(hi),"n":n,"K":maxK}
json.dump(out,open(os.path.join(ROOT,ADAPTER,OUTNAME),"w"),indent=1)
print("\nREAD: rising verifier curve with K = the trained verifier converts more test-time samples into accuracy.")
