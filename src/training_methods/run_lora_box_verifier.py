#!/usr/bin/env python3
"""
run_lora_box_verifier.py - the STRUCTURED-OUTPUT analog of the answer-verifier (§5.10). Trains a box-verifier
to judge "does this drawn box correctly localize the {organ}?" on SLAKE grounding (Qwen2.5-VL's 8 sampled
boxes; IoU vs gold detection.json = label), then selects best-of-8 boxes. Tests whether TRAINING beats the
SC-medoid luck floor for grounding (§5.9 showed grounding selection is luck-floored for a competent grounder).
Box drawn as a red rectangle on the image (visual verification). Grouped split by image (no leakage).
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/run_lora_box_verifier.py \
    --ground ckpts/ground/slake_qwenvl7b.jsonl --thr 0.3 --epochs 1
"""
import argparse, os, json, re, math, random, time
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from PIL import Image, ImageDraw
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--ground", default="ckpts/ground/slake_qwenvl7b.jsonl")
ap.add_argument("--thr", type=float, default=0.3); ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=2); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--max_train", type=int, default=6000)
ap.add_argument("--out_dir", default="ckpts/train/lora_box_verifier"); ap.add_argument("--seed", type=int, default=0)
A=ap.parse_args(); os.makedirs(os.path.join(ROOT,A.out_dir),exist_ok=True)
DEV="cuda"; MAXPX,MINPX=1280*28*28,4*28*28
try:
    from qwen_vl_utils.vision_process import smart_resize
    _MAXP,_MINP=1280*28*28,4*28*28
except Exception:
    smart_resize=None
def parse_box(s,W,H):
    m=re.search(r'bbox_2d"?\s*:?\s*\[([^\]]+)\]', s)
    nums=re.findall(r"-?\d+\.?\d*", m.group(1) if m else s.replace(","," "))
    if len(nums)<4: return None
    v=[float(x) for x in nums[:4]]
    if max(v)<=1.5: v=[v[0]*W,v[1]*H,v[2]*W,v[3]*H]
    elif smart_resize is not None:   # Qwen abs coords in smart-resized space -> scale to original
        hb,wb=smart_resize(H,W,28,_MINP,_MAXP); v=[v[0]*W/wb,v[1]*H/hb,v[2]*W/wb,v[3]*H/hb]
    x1,y1,x2,y2=v; return [max(0,min(x1,x2)),max(0,min(y1,y2)),min(W,max(x1,x2)),min(H,max(y1,y2))]
def iou(a,b):
    if not a or not b: return 0.0
    ix1,iy1,ix2,iy2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0
# build per-target records: image path, label, gold, list of (box, ok)
rows=[json.loads(l) for l in open(os.path.join(ROOT,A.ground)) if l.strip()]
REC=[]
for r in rows:
    W,H=r["W"],r["H"]; gold=r["gold"]; boxes=[parse_box(x,W,H) for x in r["sample_raws"]]
    gb=parse_box(r["greedy_raw"],W,H)
    oks=[1 if iou(b,gold)>=A.thr else 0 for b in boxes]
    REC.append({"img":r["img"],"label":r["label"],"gold":gold,"W":W,"H":H,"boxes":boxes,"oks":oks,
                "greedy":gb,"greedy_ok":1 if iou(gb,gold)>=A.thr else 0})
rng=random.Random(A.seed); idx=list(range(len(REC))); rng.shuffle(idx)
ntr=int(0.7*len(idx)); tr_ix=set(idx[:ntr]); te_ix=set(idx[ntr:])
proc=AutoProcessor.from_pretrained(A.model_path)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
def draw(img_path, box):
    im=Image.open(img_path).convert("RGB"); d=ImageDraw.Draw(im); W,H=im.size
    if box:
        x0,x1=sorted([max(0,min(box[0],W)), max(0,min(box[2],W))])
        y0,y1=sorted([max(0,min(box[1],H)), max(0,min(box[3],H))])
        if x1>x0 and y1>y0: d.rectangle([x0,y0,x1,y1], outline=(255,0,0), width=max(2,W//100))
    return im
def encode(img_path, label, box, lab=None):
    im=draw(img_path, box)
    msgs=[{"role":"user","content":[{"type":"image","image":im,"max_pixels":MAXPX,"min_pixels":MINPX},
        {"type":"text","text":f"The red box is a proposed location for the {label}. Does the red box correctly localize the {label}? Answer Yes or No."}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    if lab is not None: text=text+("Yes" if lab==1 else "No")
    igs,vids=process_vision_info(msgs); return proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True)
# training examples
train_ex=[]
for i in tr_ix:
    r=REC[i]
    for b,ok in zip(r["boxes"],r["oks"]):
        if b is not None: train_ex.append((r["img"],r["label"],b,ok))
rng.shuffle(train_ex); train_ex=train_ex[:A.max_train]
print(f"targets={len(REC)} train_ex={len(train_ex)} (pos {np.mean([e[3] for e in train_ex]):.3f}) test={len(te_ix)} thr={A.thr}", flush=True)
print("loading Qwen2.5-VL + LoRA...", flush=True)
model=AutoModelForImageTextToText.from_pretrained(A.model_path,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
lcfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=get_peft_model(model,lcfg); model.print_trainable_parameters()
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=A.lr); model.train(); t0=time.time(); step=0
for ep in range(A.epochs):
    rng.shuffle(train_ex)
    for bi in range(0,len(train_ex),A.bs):
        for (ip,lab_name,box,ok) in train_ex[bi:bi+A.bs]:
            try:
                enc=encode(ip,lab_name,box,ok).to(DEV); labels=enc["input_ids"].clone(); labels[:,:-1]=-100
                (model(**enc,labels=labels).loss/(A.bs*A.accum)).backward()
            except Exception as e: print(f"  skip: {str(e)[:60]}",flush=True)
        step+=1
        if step%A.accum==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); opt.zero_grad()
        if step%50==0: print(f"  ep{ep} step{step}/{len(train_ex)//A.bs} {(time.time()-t0)/60:.1f}min",flush=True)
model.eval()
def pyes(ip,lab_name,box):
    if box is None: return 0.0
    with torch.no_grad():
        enc=encode(ip,lab_name,box,None).to(DEV); lg=model(**enc).logits[0,-1]
        py=math.exp(lg[YES].item()); pn=math.exp(lg[NO].item()); return py/(py+pn) if (py+pn)>0 else 0.5
print("scoring held-out...", flush=True)
g=[]; med=[]; tv=[]; orc=[]
for i in te_ix:
    r=REC[i]; oks=r["oks"]
    if all(b is None for b in r["boxes"]): continue
    g.append(r["greedy_ok"]); orc.append(max(oks))
    # SC-medoid (spatial agreement)
    n=len(r["boxes"]); pair=np.zeros((n,n))
    for a in range(n):
        for c in range(n):
            if a!=c: pair[a,c]=iou(r["boxes"][a],r["boxes"][c])
    med.append(oks[int(np.argmax(pair.sum(1)))])
    # trained box-verifier
    sc=[pyes(r["img"],r["label"],b) for b in r["boxes"]]; tv.append(oks[int(np.argmax(sc))])
out={"n":len(g),"greedy":float(np.mean(g)),"sc_medoid":float(np.mean(med)),"trained_box_verify":float(np.mean(tv)),"oracle":float(np.mean(orc)),"thr":A.thr}
print(f"\n==================== BOX-VERIFIER (SLAKE grounding, thr={A.thr}) ====================")
print(f"  n={out['n']}  greedy={out['greedy']:.3f}  SC-medoid={out['sc_medoid']:.3f}  trained-box-verify={out['trained_box_verify']:.3f}  oracle@8={out['oracle']:.3f}")
json.dump(out,open(os.path.join(ROOT,A.out_dir,"result.json"),"w"),indent=1)
print("READ: trained-box-verify > SC-medoid toward oracle => TRAINING breaks the grounding luck floor too (2nd positive).")
