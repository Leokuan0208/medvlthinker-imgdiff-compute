#!/usr/bin/env python3
"""
lora_stability_router.py - TRAINING-BASED gate (needs peft): LoRA-fine-tune the small 7B to predict its
own answer's STABILITY under stronger compute, directly from image+question, and test whether that beats
the training-free signal gates (margin) and the signal-trained CASP gate.

Target  S(x) = 1[ 7B-nothink@cap320 pred == 32B-think pred ]  (the learnable routing label from CASP).
Method  generative LoRA self-verifier: prompt the 7B with (image, question, options, its proposed answer)
        and "Will a stronger reasoning model give the SAME answer? Yes/No"; LoRA-train the LM head on the
        Yes/No token (LM loss masked to that token). Escalation score = 1 - P(Yes).
Eval    honest 50/50 split (seed 0): AUROC of the LoRA scorer vs S on the held-out half, compared to
        margin AUROC and a logistic-on-signals CASP AUROC (recomputed here). LoRA adapter saved to --out_dir.
Run from repo root (after the bulk labels exist):
  CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/lora_stability_router.py --n 4000 --epochs 2
"""
import argparse, os, re, json, glob, random, time, math
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="/data/dan/weights/MedVLThinker-7B-RL_m23k")
ap.add_argument("--pred_dir", default="ckpts/gate_7b_prune/cap320")      # 7B no-think preds + opt_logprobs
ap.add_argument("--strong_dir", default="ckpts/gate_32b")               # 32B-think preds
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
                                                  "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--cap_div", type=int, default=4)   # cap320 = HIGH_PX//4
ap.add_argument("--out_dir", default="ckpts/train/lora_stability")
ap.add_argument("--max_images", type=int, default=4)
A = ap.parse_args()
os.makedirs(A.out_dir, exist_ok=True)
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX // A.cap_div
DEV = "cuda"
SYS = ("You are a careful medical exam grader. Given a question, the options, and a proposed answer, decide "
       "whether a stronger reasoning model would give the SAME answer. Respond with only 'Yes' or 'No'.")

def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def load_preds(d, arm):
    m = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        mm = re.match(r"ckpt_(.+?)_", os.path.basename(f)); ds = mm.group(1) if mm else None
        for l in open(f):
            if l.strip(): r = json.loads(l); m[(ds, r["idx"])] = r
    return m

print("loading labels...", flush=True)
nt = load_preds(A.pred_dir, "nothink"); st = load_preds(A.strong_dir, "think")
ds_all = load_dataset(ROOT); split = "test" if "test" in ds_all else list(ds_all.keys())[0]; data = ds_all[split]
def subset(*keys): return [i for i, n in enumerate(data["dataset_name"]) if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
def mx_by_type(t):
    out = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: out.append(i)
    return out
DSI = {"MedXpert-Reasoning": lambda: mx_by_type("reasoning"), "MedXpert-Understanding": lambda: mx_by_type("understanding"),
       "PMC-VQA": lambda: subset("pmcvqa", "pmc"), "SLAKE": lambda: subset("slake"),
       "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"), "PathVQA": lambda: subset("pathvqa", "path"), "MMMU": lambda: subset("mmmu")}

# build items: (ds, idx, S, proposed, signals-vector)
SIGN = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]
import sys; sys.path.insert(0, "src/cascade_methods")
from harness import signals_from_logprobs
items = []
for name in A.datasets:
    if name not in DSI: continue
    idxs = DSI[name](); rng = random.Random(42); rng.shuffle(idxs); idxs = idxs[:A.n]
    for i in idxs:
        a, b = nt.get((name, i)), st.get((name, i))
        if not a or not b: continue
        S = int(a["pred"] == b["pred"])
        sig = signals_from_logprobs(a.get("opt_logprobs"))
        items.append((name, i, S, a["pred"], [sig[k] for k in SIGN]))
print(f"{len(items)} items; stability rate {np.mean([x[2] for x in items]):.3f}", flush=True)
rng = random.Random(0); order = list(range(len(items))); rng.shuffle(order)
half = len(order) // 2; tr_ix = set(order[:half]); te_ix = set(order[half:])
train = [items[i] for i in range(len(items)) if i in tr_ix]; test = [items[i] for i in range(len(items)) if i in te_ix]

# ---- baselines (signals): margin AUROC + logistic-CASP AUROC on the SAME split ----
Xtr = np.array([x[4] for x in train], float); ytr = np.array([x[2] for x in train])
Xte = np.array([x[4] for x in test], float); yte = np.array([x[2] for x in test])
margin_te = Xte[:, 0]  # margin is signal 0; higher margin -> more stable
auc_margin = roc_auc_score(yte, margin_te)
casp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.clip(Xtr, -8, 8), ytr)
auc_casp = roc_auc_score(yte, casp.predict_proba(np.clip(Xte, -8, 8))[:, 1])
print(f"[baseline] margin AUROC={auc_margin:.4f}  signal-CASP AUROC={auc_casp:.4f}", flush=True)

# ---- LoRA generative self-verifier ----
proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
def build_msgs(ds_i, opts, proposed):
    ex = data[ds_i]; body = (ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
                             + f"\n\nProposed answer: {proposed}\nWill a stronger reasoning model give the same answer? Answer Yes or No.")
    imgs = (ex.get("images") or [])[:A.max_images]
    im = [{"type": "image", "image": x, "max_pixels": MAXPX, "min_pixels": MIN_PX} for x in imgs]
    return [{"role": "system", "content": SYS}, {"role": "user", "content": im + [{"type": "text", "text": body}]}]
def encode(it, with_answer):
    name, i, S, proposed, _ = it
    msgs = build_msgs(i, parse_opts(data[i]["options"]), proposed)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if with_answer: text = text + ("Yes" if S == 1 else "No")
    imgs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)
    return enc

print("loading 7B + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.model_path, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").to(DEV)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg); model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
model.train(); t0 = time.time(); step = 0
for ep in range(A.epochs):
    rng.shuffle(train)
    for bi in range(0, len(train), A.bs):
        batch = train[bi:bi + A.bs]; loss_acc = 0.0
        for it in batch:
            try:
                enc = encode(it, with_answer=True).to(DEV)
                labels = enc["input_ids"].clone(); labels[:, :-1] = -100  # LM loss only on the final (Yes/No) token
                out = model(**enc, labels=labels); loss = out.loss / (A.bs * A.accum)
                loss.backward(); loss_acc += loss.item()
            except Exception as e:
                print(f"  skip train item: {str(e)[:80]}", flush=True)
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if step % 100 == 0:
            print(f"  ep{ep} step{step}/{len(train)//A.bs} loss~{loss_acc*A.bs*A.accum:.3f} {(time.time()-t0)/60:.1f}min", flush=True)
model.eval()
print("eval LoRA scorer on held-out half...", flush=True)
scores = []
with torch.no_grad():
    for it in test:
        try:
            enc = encode(it, with_answer=False).to(DEV)
            logits = model(**enc).logits[0, -1]  # next-token logits at the generation position
            py = math.exp(logits[YES].item()); pn = math.exp(logits[NO].item())
            scores.append(py / (py + pn) if (py + pn) > 0 else 0.5)
        except Exception as e:
            scores.append(0.5)
scores = np.array(scores)
auc_lora = roc_auc_score(yte, scores)
print("\n==================== LoRA STABILITY ROUTER ====================", flush=True)
print(f"  stability label rate (test): {yte.mean():.3f}   n_train={len(train)} n_test={len(test)}", flush=True)
print(f"  margin gate AUROC        : {auc_margin:.4f}", flush=True)
print(f"  signal-CASP AUROC        : {auc_casp:.4f}", flush=True)
print(f"  LoRA self-verifier AUROC : {auc_lora:.4f}  (Δ vs CASP {auc_lora-auc_casp:+.4f})", flush=True)
model.save_pretrained(A.out_dir)
json.dump({"auc_margin": auc_margin, "auc_casp": auc_casp, "auc_lora": auc_lora,
           "n_train": len(train), "n_test": len(test), "stab_rate": float(yte.mean()),
           "epochs": A.epochs}, open(os.path.join(A.out_dir, "result.json"), "w"), indent=1)
print(f"  saved adapter + result.json -> {A.out_dir}", flush=True)
