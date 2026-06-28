#!/usr/bin/env python3
"""Cross-GENERATOR transfer: the verifier was trained on Lingshu-7B answers; here we apply it to score
MedVLThinker-7B's answers (a different generator) for best-of-N selection. Tests generator-agnosticism.
Datasets: SLAKE-open, VQA-RAD-open (we have MedVLThinker-7B sc8 + a fresh 32B judge for them)."""
import os, json, glob, io, math
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ADAPTER="ckpts/train/lora_verifier_pooled4"; MODEL="lingshu-medical-mllm/Lingshu-7B"
DEV="cuda"; MAXPX,MINPX=1280*28*28,4*28*28
SYS=("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
     "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK=os.path.join(ROOT,"ckpts/openvqa/cheap")  # MedVLThinker-7B
def loadj(p): return {r["idx"]:r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower().rstrip(".")
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
            conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip()
        if str(a).strip().lower() in ("yes","no"): continue
        img=r["image"]
        if isinstance(img,dict) and "bytes" in img: m[int(i)]=(str(q),Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
IMG={"slake_open":slake_imgs(),"vqa_rad_open":parquet_imgs("/data/dan/dataset/vqa_rad/data")}
QREC={}
for ds in ["slake_open","vqa_rad_open"]:
    sc=loadj(f"{CK}/ckpt_{ds}_7b_sc8.jsonl"); exp=loadj(f"{CK}/ckpt_{ds}_7b_sc8_scexploded.jsonl")
    jud={k:v["judge_ok"] for k,v in loadj(f"{CK}/ckpt_{ds}_7b_sc8_scexploded.judge.jsonl").items()}
    aj=defaultdict(dict)
    for cid,r in exp.items():
        if cid in jud:
            oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])]=jud[cid]
    for i in sc:
        if i in IMG[ds] and i in aj:
            q,img=IMG[ds][i]; QREC[(ds,i)]={"q":q,"img":img,"preds":sc[i]["preds"],"slabels":aj[i],"modal":sc[i].get("modal_pred")}
print(f"cross-gen questions={len(QREC)}",flush=True)
proc=AutoProcessor.from_pretrained(MODEL)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
model=AutoModelForImageTextToText.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
model=PeftModel.from_pretrained(model,os.path.join(ROOT,ADAPTER)); model.eval()
def pyes(q,img,a):
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MINPX},
        {"type":"text","text":f"Question: {q}\nProposed answer: {a}\nIs the proposed answer correct? Answer Yes or No."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    igs,vids=process_vision_info(msgs); enc=proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True).to(DEV)
    with torch.no_grad():
        lg=model(**enc).logits[0,-1]; return math.exp(lg[YES].item())/(math.exp(lg[YES].item())+math.exp(lg[NO].item())+1e-9)
res={}
for ds in ["slake_open","vqa_rad_open"]:
    g=[];sc_=[];orc=[];ver=[]
    for (d,i),r in QREC.items():
        if d!=ds: continue
        preds=r["preds"]; sl=[r["slabels"].get(norm(a)) for a in preds]; lab=[x for x in sl if x is not None]
        if not lab: continue
        scr=[pyes(r["q"],r["img"],a) for a in preds]
        cand=[k for k in range(len(sl)) if sl[k] is not None]
        g.append(sl[0] if sl[0] is not None else 0); orc.append(max(lab))
        sc_.append(r["slabels"].get(norm(r["modal"]),0) or 0)
        ver.append(sl[max(cand,key=lambda k:scr[k])])
    res[ds]=dict(n=len(ver),greedy=float(np.mean(g)),sc=float(np.mean(sc_)),verifier=float(np.mean(ver)),oracle=float(np.mean(orc)))
    print(f"{ds}: n={res[ds]['n']} greedy={res[ds]['greedy']:.3f} SC={res[ds]['sc']:.3f} verifier={res[ds]['verifier']:.3f} oracle={res[ds]['oracle']:.3f}",flush=True)
json.dump(res,open(os.path.join(ROOT,ADAPTER,"crossgen_result.json"),"w"),indent=1)
print("CROSSGEN_DONE")
