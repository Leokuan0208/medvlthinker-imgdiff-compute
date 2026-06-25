#!/usr/bin/env python3
"""Does the trained verifier USE THE IMAGE (vs a 'lazy verifier' judging text only, per Verification
Mirage)? Loads the seed-0 LoRA adapter, rebuilds the SAME held-out split, and scores best-of-8 selection
with REAL images vs BLANK (gray) images. If real >> blank, the verifier is image-grounded.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/training_methods/verifier_image_ablation.py \
    --adapter ckpts/train/lora_verifier_open --seed 0
"""
import argparse, os, json, glob, io, math, random
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_open"); ap.add_argument("--seed", type=int, default=0)
A=ap.parse_args(); DEV="cuda"; MAXPX,MINPX=1280*28*28,4*28*28
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
def pathvqa_imgs():
    import pandas as pd; m={}
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/data/dan/dataset/path_vqa/data/test-*.parquet"))],ignore_index=True)
    for i,r in df.iterrows():
        q=r.get("question"); a=r.get("answer")
        if q is None and "conversations" in r:
            conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
        if str(a).strip().lower() in ("yes","no"): continue
        img=r["image"]
        if isinstance(img,dict) and "bytes" in img: m[int(i)]=(str(q),Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
IMG={"slake_open":slake_imgs(),"pathvqa_open":pathvqa_imgs()}
QREC={}
for ds in ["slake_open","pathvqa_open"]:
    sc=loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8.jsonl"); exp=loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.jsonl")
    jud={k:v["judge_ok"] for k,v in loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.judge.jsonl").items()}
    aj=defaultdict(dict)
    for cid,r in exp.items():
        if cid in jud:
            oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])]=jud[cid]
    for i in sc:
        if i in IMG[ds] and i in aj:
            q,img=IMG[ds][i]; QREC[(ds,i)]={"q":q,"img":img,"preds":sc[i]["preds"],"slabels":aj[i]}
keys=list(QREC.keys()); rng=random.Random(A.seed); rng.shuffle(keys)
test_keys=set(keys[int(0.7*len(keys)):])
proc=AutoProcessor.from_pretrained(A.model_path)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
print("loading base + adapter...", flush=True)
model=AutoModelForImageTextToText.from_pretrained(A.model_path,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
model=PeftModel.from_pretrained(model, os.path.join(ROOT,A.adapter)); model.eval()
def encode(q,img,ans):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[
        {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},
        {"type":"text","text":f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,vids=process_vision_info(msgs)
    return proc(text=[text],images=imgs,videos=vids,return_tensors="pt",padding=True)
def pyes(q,img,ans):
    with torch.no_grad():
        enc=encode(q,img,ans).to(DEV); lg=model(**enc).logits[0,-1]
        py=math.exp(lg[YES].item()); pn=math.exp(lg[NO].item()); return py/(py+pn) if (py+pn)>0 else 0.5
BLANK=Image.new("RGB",(336,336),(127,127,127))
res=defaultdict(lambda: defaultdict(list))
for k in test_keys:
    ds=k[0]; r=QREC[k]; preds=r["preds"]; sl=[r["slabels"].get(norm(a)) for a in preds]
    if all(x is None for x in sl): continue
    for mode,img in [("real",r["img"]),("blank",BLANK)]:
        scores=[pyes(r["q"],img,a) for a in preds]; ksel=int(np.argmax(scores))
        res[ds][mode].append(sl[ksel] if sl[ksel] is not None else 0)
print("\n==================== IMAGE ABLATION (trained verifier selection) ====================")
allr=defaultdict(list)
for ds in res:
    rm,bm=np.mean(res[ds]["real"]),np.mean(res[ds]["blank"])
    print(f"  {ds:<14} n={len(res[ds]['real']):>4}  real-image={rm:.3f}  blank-image={bm:.3f}  Δ(image-grounding)={rm-bm:+.3f}")
    allr["real"]+=res[ds]["real"]; allr["blank"]+=res[ds]["blank"]
print(f"  {'POOLED':<14} n={len(allr['real']):>4}  real-image={np.mean(allr['real']):.3f}  blank-image={np.mean(allr['blank']):.3f}  Δ={np.mean(allr['real'])-np.mean(allr['blank']):+.3f}")
print("\nREAD: real >> blank => the verifier is IMAGE-GROUNDED (refutes the 'lazy verifier' critique).")
