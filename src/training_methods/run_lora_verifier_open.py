#!/usr/bin/env python3
"""
run_lora_verifier_open.py - TRAINED verifier for open-ended best-of-N selection (needs peft).
LoRA-fine-tune Lingshu-7B to score P(correct | image, question, free-text answer) on the per-sample
LLM-JUDGE labels (the 8234 judged (idx,answer) pairs from the sc8 exploded judge files). Then use the
trained verifier to SELECT the best of the 8 sampled open-ended answers (argmax P(Yes)). Honest GROUPED
split by question idx (no question leaks between train/eval). Reports SELECTION accuracy vs
greedy / self-consistency / oracle@8 (and the zero-shot 32B verifier where available).
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/run_lora_verifier_open.py \
    --epochs 1 --out_dir ckpts/train/lora_verifier_open
"""
import argparse, os, json, glob, io, math, random, time
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from PIL import Image
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max_train", type=int, default=6000); ap.add_argument("--cap_div", type=int, default=1)
ap.add_argument("--out_dir", default="ckpts/train/lora_verifier_open"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lora_r", type=int, default=16, help="LoRA rank (alpha=2*r); test verifier capacity")
A = ap.parse_args(); os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; MAXPX = HIGH_PX // A.cap_div; DEV = "cuda"
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK = os.path.join(ROOT, os.environ.get("VERIF_CK", "ckpts/openvqa/cheap_lingshu7b"))
TAG = os.environ.get("VERIF_TAG", "lingshu7b")  # filename model tag (e.g. "7b" for MedVLThinker)
def loadj(p): return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower()

# ---- image maps per dataset ----
def slake_imgs():
    m = {}
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type")=="OPEN" and x.get("q_lang")=="en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip): m[x["qid"]] = (x["question"], ip)
    return m
def parquet_imgs(base):
    import pandas as pd
    m = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    for i, r in df.iterrows():
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
        if str(a).strip().lower() in ("yes","no"): continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
def kvasir_imgs():
    m = {}
    for r in json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")):
        if os.path.exists(r["img_path"]): m[r["idx"]] = (r["question"], r["img_path"])
    return m
DSETS = os.environ.get("VERIF_DSETS", "slake_open,pathvqa_open,vqa_rad_open,kvasir_open").split(",")
IMG = {}
if "slake_open" in DSETS: IMG["slake_open"] = slake_imgs()
if "pathvqa_open" in DSETS: IMG["pathvqa_open"] = parquet_imgs("/data/dan/dataset/path_vqa/data")
if "vqa_rad_open" in DSETS: IMG["vqa_rad_open"] = parquet_imgs("/data/dan/dataset/vqa_rad/data")
if "kvasir_open" in DSETS: IMG["kvasir_open"] = kvasir_imgs()

# ---- build per-question records + per-(idx,answer) labels from exploded judge ----
QREC = {}   # (ds, idx) -> {"q":..,"img":..,"preds":[8], "modal":.., "slabels":{normans:0/1}}
for ds in DSETS:
    sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
    jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]
    for i in sc:
        if i not in IMG[ds] or i not in aj: continue
        q, img = IMG[ds][i]
        QREC[(ds, i)] = {"q": q, "img": img, "preds": sc[i]["preds"], "modal": sc[i]["modal_pred"], "slabels": aj[i]}
print(f"{len(QREC)} questions with images+labels", flush=True)

# ---- grouped split by question ----
keys = list(QREC.keys()); rng = random.Random(A.seed); rng.shuffle(keys)
ntr = int(0.7*len(keys)); train_keys, test_keys = set(keys[:ntr]), set(keys[ntr:])
# training examples: unique (answer,label) per train question
train_ex = []
for k in train_keys:
    r = QREC[k]
    for na, lab in r["slabels"].items():
        # recover a surface form of the answer from preds
        surf = next((a for a in r["preds"] if norm(a)==na), na)
        train_ex.append((k[0], r["q"], r["img"], surf, lab))
rng.shuffle(train_ex); train_ex = train_ex[:A.max_train]
print(f"train examples={len(train_ex)} (pos rate {np.mean([e[4] for e in train_ex]):.3f}); test questions={len(test_keys)}", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]; NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
def build_msgs(q, img, proposed):
    body = f"Question: {q}\nProposed answer: {proposed}\nIs the proposed answer correct? Answer Yes or No."
    return [{"role":"system","content":SYS},
            {"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},
                                       {"type":"text","text":body}]}]
def encode(q, img, proposed, label=None):
    msgs = build_msgs(q, img, proposed)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if label is not None: text = text + ("Yes" if label==1 else "No")
    imgs, vids = process_vision_info(msgs)
    return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)

print("loading Lingshu-7B + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.model_path, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").to(DEV)
lcfg = LoraConfig(r=A.lora_r, lora_alpha=2*A.lora_r, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lcfg); model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
model.train(); t0=time.time(); step=0
for ep in range(A.epochs):
    rng.shuffle(train_ex)
    for bi in range(0, len(train_ex), A.bs):
        for (ds,q,img,ans,lab) in train_ex[bi:bi+A.bs]:
            try:
                enc = encode(q,img,ans,lab).to(DEV)
                labels = enc["input_ids"].clone(); labels[:, :-1] = -100
                out = model(**enc, labels=labels); (out.loss/(A.bs*A.accum)).backward()
            except Exception as e:
                print(f"  skip: {str(e)[:70]}", flush=True)
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if step % 50 == 0: print(f"  ep{ep} step{step}/{len(train_ex)//A.bs} {(time.time()-t0)/60:.1f}min", flush=True)

model.eval()
print("scoring held-out questions...", flush=True)
def pyes(q,img,ans):
    with torch.no_grad():
        enc = encode(q,img,ans,None).to(DEV); lg = model(**enc).logits[0,-1]
        py=math.exp(lg[YES].item()); pn=math.exp(lg[NO].item()); return py/(py+pn) if (py+pn)>0 else 0.5
res = defaultdict(lambda: defaultdict(list))   # ds -> metric -> list
for k in test_keys:
    ds = k[0]; r = QREC[k]; preds = r["preds"]; sl = [r["slabels"].get(norm(a)) for a in preds]
    if all(x is None for x in sl): continue
    # trained verifier select
    scores = [pyes(r["q"], r["img"], a) for a in preds]
    ksel = int(np.argmax(scores))
    res[ds]["trained"].append(sl[ksel] if sl[ksel] is not None else 0)
    res[ds]["greedy"].append(r["slabels"].get(norm(r["modal"]),0))
    res[ds]["oracle"].append(max([x for x in sl if x is not None]))
    from collections import Counter
    c=Counter(norm(a) for a in preds); top=c.most_common(1)[0][0]
    res[ds]["sc"].append(r["slabels"].get(top,0))
out = {}
print("\n==================== TRAINED VERIFIER (open-ended best-of-N) ====================")
for ds in res:
    row = {m: float(np.mean(v)) for m,v in res[ds].items()}; row["n"]=len(res[ds]["trained"]); out[ds]=row
    print(f"  {ds:<14} n={row['n']:>4}  greedy={row['greedy']:.3f}  SC={row['sc']:.3f}  trained-verify={row['trained']:.3f}  oracle@8={row['oracle']:.3f}")
# pooled
allm = defaultdict(list)
for ds in res:
    for m,v in res[ds].items(): allm[m]+=v
print(f"  {'POOLED':<14} n={len(allm['trained']):>4}  greedy={np.mean(allm['greedy']):.3f}  SC={np.mean(allm['sc']):.3f}  trained-verify={np.mean(allm['trained']):.3f}  oracle@8={np.mean(allm['oracle']):.3f}")
out["pooled"]={m:float(np.mean(v)) for m,v in allm.items() if m!='n'}
json.dump(out, open(os.path.join(ROOT,A.out_dir,"result.json"),"w"), indent=1)
model.save_pretrained(os.path.join(ROOT,A.out_dir))
print(f"saved -> {A.out_dir}; compare trained-verify vs zero-shot 32B verify (SLAKE 0.758) and oracle.", flush=True)
