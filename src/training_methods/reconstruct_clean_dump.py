#!/usr/bin/env python3
"""Best-of-N test-time-scaling curve for the trained free-text verifier: does verifier-selected accuracy
keep improving as you sample more (K=1,2,4,8)? Loads the pooled-4 adapter, scores all 8 samples per
held-out question (seed-0 grouped split = the §5.10 test set), then computes verifier-best-of-K vs oracle@K
vs random@K for K in {1,2,4,8}. A rising verifier curve = the method benefits from test-time compute.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/verifier_scaling_curve.py
"""
import os, json, glob, io, math, random
import numpy as np
from collections import defaultdict
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
# align saved verifier scores (perq_sc8.json) with reconstructed test_keys -> write clean_dump.json (no GPU)
perq=json.load(open(os.path.join(ROOT,ADAPTER,"perq_sc8.json")))
recs=[]; pi=0; mism=0
for k in test_keys:
    r=QREC[k]; preds=r["preds"]; sl=[r["slabels"].get(norm(a)) for a in preds]
    if all(x is None for x in sl): continue
    if pi>=len(perq): break
    if perq[pi]["sl"]!=sl: mism+=1
    recs.append({"ds":k[0],"idx":k[1],"preds":preds,"sl":[(-1 if x is None else int(x)) for x in sl],"sc":perq[pi]["sc"]})
    pi+=1
print(f"reconstructed {len(recs)} (perq {len(perq)}); sl-mismatches={mism}",flush=True)
if mism==0 and len(recs)==len(perq):
    json.dump(recs, open(os.path.join(ROOT,ADAPTER,"clean_dump.json"),"w"))
    print("VALIDATED -> wrote clean_dump.json (reconstruction matches saved scores exactly)")
else:
    print("MISMATCH -> NOT writing; will rely on the running GPU re-score instead")
