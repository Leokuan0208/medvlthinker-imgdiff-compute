#!/usr/bin/env python3
"""run_lora_verifier_ranking.py - RANKING/contrastive verifier for open-ended best-of-N SELECTION.
Diagnostic (UNIFIED_METHOD_EXPERIMENTS.md): the pointwise YES/NO verifier is good per-answer (AUROC ~0.90) but
selection efficiency is only 74-82% -- it loses on within-question NEAR-TIES (a wrong answer outscores the correct
one). The task is RANKING within a question, not pointwise classification. This trains the verifier with a COMBINED
loss on the Yes-No logit margin s = logit(Yes)-logit(No):
  - pointwise BCE(s, label)         -> preserves absolute calibration (the verifier-confidence GATE needs it)
  - within-question Bradley-Terry -logsigmoid(s_correct - s_wrong)  -> directly targets selection efficiency
Trains on both-class questions (>=1 correct AND >=1 wrong of the 8). Honest grouped split by question (seed-0, same
split as run_lora_verifier_open.py). Reports held-out selection accuracy + per-answer AUROC.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/run_lora_verifier_ranking.py \
    --epochs 2 --lam 1.0 --out_dir ckpts/train/lora_verifier_rank
"""
import argparse, os, json, glob, io, math, random, time
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict, Counter
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from PIL import Image
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--accum", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lam", type=float, default=1.0, help="ranking-loss weight")
ap.add_argument("--pairs_per_q", type=int, default=2); ap.add_argument("--max_q", type=int, default=100000)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--out_dir", default="ckpts/train/lora_verifier_rank"); ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args(); os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
MAXPX, MIN_PX = 1280*28*28, 4*28*28; DEV = "cuda"
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK = os.path.join(ROOT, os.environ.get("VERIF_CK", "ckpts/openvqa/cheap_lingshu7b")); TAG = os.environ.get("VERIF_TAG", "lingshu7b")
def loadj(p): return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower()
# ---- image maps (same as run_lora_verifier_open.py) ----
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
DSETS = os.environ.get("VERIF_DSETS", "slake_open,pathvqa_open,vqa_rad_open,kvasir_open").split(",")
IMG={}
if "slake_open" in DSETS: IMG["slake_open"]=slake_imgs()
if "pathvqa_open" in DSETS: IMG["pathvqa_open"]=parquet_imgs("/data/dan/dataset/path_vqa/data")
if "vqa_rad_open" in DSETS: IMG["vqa_rad_open"]=parquet_imgs("/data/dan/dataset/vqa_rad/data")
if "kvasir_open" in DSETS: IMG["kvasir_open"]=kvasir_imgs()
# ---- per-question records ----
QREC={}
for ds in DSETS:
    sc=loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl"); exp=loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
    jud={k:v["judge_ok"] for k,v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
    aj=defaultdict(dict)
    for cid,r in exp.items():
        if cid in jud:
            oi=cid.split("#")[0]; oi=int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])]=jud[cid]
    for i in sc:
        if i not in IMG[ds] or i not in aj: continue
        q,img=IMG[ds][i]
        QREC[(ds,i)]={"q":q,"img":img,"preds":sc[i]["preds"],"modal":sc[i]["modal_pred"],"slabels":aj[i]}
print(f"{len(QREC)} questions with images+labels", flush=True)
keys=list(QREC.keys()); rng=random.Random(A.seed); rng.shuffle(keys)
ntr=int(0.7*len(keys)); train_keys,test_keys=set(keys[:ntr]),set(keys[ntr:])
# build training pairs from BOTH-CLASS train questions (have >=1 correct and >=1 wrong distinct answer)
train_q=[]
for k in train_keys:
    r=QREC[k]; corr=[a for a in r["preds"] if r["slabels"].get(norm(a))==1]; wro=[a for a in r["preds"] if r["slabels"].get(norm(a))==0]
    corr=list({norm(a):a for a in corr}.values()); wro=list({norm(a):a for a in wro}.values())
    if corr and wro: train_q.append((k[0],r["q"],r["img"],corr,wro))
train_q=train_q[:A.max_q]
# FULL pointwise pairs (all distinct answers of every train question) -> keeps calibration/coverage like v2
point_ex=[]
for k in train_keys:
    r=QREC[k]
    for na,lab in r["slabels"].items():
        surf=next((a for a in r["preds"] if norm(a)==na), na); point_ex.append((r["q"],r["img"],surf,lab))
print(f"both-class ranking questions={len(train_q)}; pointwise pairs={len(point_ex)}; test questions={len(test_keys)}", flush=True)

proc=AutoProcessor.from_pretrained(A.model_path)
YES=proc.tokenizer.encode("Yes",add_special_tokens=False)[0]; NO=proc.tokenizer.encode("No",add_special_tokens=False)[0]
def enc_one(q,img,ans):
    body=f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
    msgs=[{"role":"system","content":SYS},{"role":"user","content":[
        {"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},{"type":"text","text":body}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    igs,vids=process_vision_info(msgs); return proc(text=[text],images=igs,videos=vids,return_tensors="pt",padding=True).to(DEV)
def margin(q,img,ans):  # s = logit(Yes) - logit(No) at last position (differentiable)
    e=enc_one(q,img,ans); lg=model(**e).logits[0,-1]; return lg[YES]-lg[NO]
print("loading base + LoRA...", flush=True)
model=AutoModelForImageTextToText.from_pretrained(A.model_path,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2").to(DEV)
lcfg=LoraConfig(r=A.lora_r,lora_alpha=2*A.lora_r,lora_dropout=0.05,bias="none",
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model=get_peft_model(model,lcfg); model.print_trainable_parameters()
model.gradient_checkpointing_enable(); model.enable_input_require_grads()  # cut activation memory (3 fwds/step)
model.config.use_cache=False
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
one=torch.tensor(1.0,device=DEV); zero=torch.tensor(0.0,device=DEV)
labf=lambda lab: one if lab==1 else zero
model.train(); t0=time.time(); step=0
for ep in range(A.epochs):
    rng.shuffle(point_ex); rng.shuffle(train_q); ri=0
    for pi,(q,img,ans,lab) in enumerate(point_ex):   # primary: full pointwise BCE (calibration + coverage, like v2)
        try:
            # split backwards so peak holds ONE graph at a time (not 3) -> avoids OOM
            (F.binary_cross_entropy_with_logits(margin(q,img,ans), labf(lab))/A.accum).backward()
            if train_q and A.lam>0:                    # + within-question Bradley-Terry on a near-tie question
                _,qr,imgr,corr,wro=train_q[ri%len(train_q)]; ri+=1
                sc=margin(qr,imgr,rng.choice(corr)); sw=margin(qr,imgr,rng.choice(wro))
                (A.lam*(-F.logsigmoid(sc-sw))/A.accum).backward()
        except Exception as e:
            print(f"  skip: {str(e)[:70]}", flush=True); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
        step+=1
        if step%A.accum==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step(); opt.zero_grad()
        if step%200==0: print(f"  ep{ep} {pi}/{len(point_ex)} {(time.time()-t0)/60:.1f}min", flush=True)
# ---- eval: held-out selection accuracy + per-answer AUROC ----
from sklearn.metrics import roc_auc_score
model.eval()
def pyes(q,img,ans):
    with torch.no_grad():
        e=enc_one(q,img,ans); lg=model(**e).logits[0,-1]; py=math.exp(lg[YES].item()); pn=math.exp(lg[NO].item())
        return py/(py+pn) if (py+pn)>0 else 0.5
res=defaultdict(lambda:defaultdict(list)); allscore=[]; alllab=[]
for k in test_keys:
    ds=k[0]; r=QREC[k]; preds=r["preds"]; sl=[r["slabels"].get(norm(a)) for a in preds]
    if all(x is None for x in sl): continue
    scores=[pyes(r["q"],r["img"],a) for a in preds]; ksel=int(np.argmax(scores))
    res[ds]["trained"].append(sl[ksel] if sl[ksel] is not None else 0)
    res[ds]["greedy"].append(r["slabels"].get(norm(r["modal"]),0)); res[ds]["oracle"].append(max([x for x in sl if x is not None]))
    for a,s in zip(preds,scores):
        lb=r["slabels"].get(norm(a))
        if lb is not None: allscore.append(s); alllab.append(lb)
out={}; allm=defaultdict(list)
print("\n==================== RANKING VERIFIER (held-out selection) ====================")
for ds in res:
    row={m:float(np.mean(v)) for m,v in res[ds].items()}; row["n"]=len(res[ds]["trained"]); out[ds]=row
    for m,v in res[ds].items(): allm[m]+=v
    print(f"  {ds:<14} n={row['n']:>4} greedy={row['greedy']:.3f} trained-verify={row['trained']:.3f} oracle@8={row['oracle']:.3f}")
pera=roc_auc_score(alllab,allscore) if len(set(alllab))>1 else float('nan')
out["pooled"]={m:float(np.mean(v)) for m,v in allm.items()}; out["per_answer_auroc"]=float(pera)
print(f"  {'POOLED':<14} n={len(allm['trained']):>4} greedy={np.mean(allm['greedy']):.3f} trained-verify={np.mean(allm['trained']):.3f} oracle@8={np.mean(allm['oracle']):.3f} | per-answer AUROC={pera:.3f}")
json.dump(out, open(os.path.join(ROOT,A.out_dir,"result.json"),"w"), indent=1)
model.save_pretrained(os.path.join(ROOT,A.out_dir))
print(f"saved -> {A.out_dir}", flush=True)
