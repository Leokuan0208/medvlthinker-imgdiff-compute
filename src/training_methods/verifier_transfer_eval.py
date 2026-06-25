#!/usr/bin/env python3
"""CROSS-DATASET transfer of the trained verifier: the verifier was trained on SLAKE+PathVQA only; here we
test it on FULLY HELD-OUT datasets (VQA-RAD, Kvasir-OOD GI-endoscopy) — does the learned correctness signal
generalize? Loads the seed-0 LoRA adapter, scores best-of-8 selection, compares greedy/SC/trained/oracle.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/verifier_transfer_eval.py \
    --adapter ckpts/train/lora_verifier_open --datasets vqa_rad_open kvasir_open
"""
import argparse, os, json, glob, io, math
import numpy as np, torch
from collections import defaultdict, Counter
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_open")
ap.add_argument("--datasets", nargs="+", default=["vqa_rad_open","kvasir_open"])
A=ap.parse_args(); DEV="cuda"; MAXPX,MINPX=1280*28*28,4*28*28
SYS=("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
     "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK=os.path.join(ROOT,"ckpts/openvqa/cheap_lingshu7b")
def loadj(p): return {r["idx"]:r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower()
def imgs_for(ds):
    m={}
    if ds in ("kvasir_open","radimagenet_open"):
        jp = "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json" if ds=="kvasir_open" else "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json"
        for r in json.load(open(jp)):
            if os.path.exists(r["img_path"]): m[r["idx"]]=(r["question"], r["img_path"])
    else:
        import pandas as pd
        base="/data/dan/dataset/vqa_rad/data" if ds=="vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
        df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base,"test-*.parquet")))],ignore_index=True)
        for i,r in df.iterrows():
            q=r.get("question"); a=r.get("answer")
            if q is None and "conversations" in r:
                conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
            if str(a).strip().lower() in ("yes","no"): continue
            img=r["image"]
            if isinstance(img,dict) and "bytes" in img: m[int(i)]=(str(q),Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
proc=AutoProcessor.from_pretrained(A.model_path)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
print("loading base + adapter...", flush=True)
model=AutoModelForImageTextToText.from_pretrained(A.model_path,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
model=PeftModel.from_pretrained(model, os.path.join(ROOT,A.adapter)); model.eval()
def pyes(q,img,ans):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[
        {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},
        {"type":"text","text":f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    igs,vids=process_vision_info(msgs); enc=proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True).to(DEV)
    with torch.no_grad():
        lg=model(**enc).logits[0,-1]; py=math.exp(lg[YES].item()); pn=math.exp(lg[NO].item())
    return py/(py+pn) if (py+pn)>0 else 0.5
out={}
for ds in A.datasets:
    sc=loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8.jsonl")
    exp=loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.jsonl")
    jud={k:v["judge_ok"] for k,v in loadj(f"{CK}/ckpt_{ds}_lingshu7b_sc8_scexploded.judge.jsonl").items()}
    if not jud: print(f"{ds}: NO judge file yet, skip", flush=True); continue
    aj=defaultdict(dict)
    for cid,r in exp.items():
        if cid in jud:
            oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])]=jud[cid]
    IMG=imgs_for(ds)
    g=[]; s=[]; t=[]; o=[]
    for i in sc:
        if i not in aj or i not in IMG: continue
        q,img=IMG[i]; preds=sc[i]["preds"]; sl=[aj[i].get(norm(a)) for a in preds]
        if all(x is None for x in sl): continue
        g.append(aj[i].get(norm(sc[i]["modal_pred"]),0))
        c=Counter(norm(a) for a in preds); top=c.most_common(1)[0][0]; s.append(aj[i].get(top,0))
        o.append(max([x for x in sl if x is not None]))
        scores=[pyes(q,img,a) for a in preds]; k=int(np.argmax(scores)); t.append(sl[k] if sl[k] is not None else 0)
    out[ds]={"n":len(t),"greedy":float(np.mean(g)),"sc":float(np.mean(s)),"trained":float(np.mean(t)),"oracle":float(np.mean(o))}
    print(f"  {ds:<14} n={len(t):>4}  greedy={np.mean(g):.3f}  SC={np.mean(s):.3f}  trained-verify={np.mean(t):.3f}  oracle@8={np.mean(o):.3f}", flush=True)
json.dump(out, open(os.path.join(ROOT,A.adapter,"transfer_result.json"),"w"), indent=1)
print("\nREAD: trained-verify > greedy/SC on HELD-OUT datasets => the trained verifier GENERALIZES cross-dataset.")
