#!/usr/bin/env python3
"""
fld_distill.py - FastLeg-Distill (FLD): LoRA-distill the BIG no-think model's competence INTO the small
no-think cheap leg, to RAISE the cheap leg's accuracy -> fewer escalations -> cheaper cascade at parity.
This attacks the escalation RATE (the dominant cost), not the (capped) escalation decision.

Teacher = 32B-nt@cap320 (ckpts/gate_32b_modes/nothink_cap320).  Student = 7B, LoRA, no-think format.
Agreement-gate: train only on samples the teacher gets RIGHT (distill its correct perception answers);
target = teacher's answer letter (hard-label CE on the letter token).
DE-RISK design: eval-CV (50/50 seed-0 split) — distill on train-half, eval distilled 7B-nt on the held-out
half; compare to the ORIGINAL 7B-nt accuracy on the SAME held-out half. (Clean train-split version is the
follow-up if this lifts the cheap leg.) No test leakage (student never sees test-half during training).
Run: CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/fld_distill.py --n 4000 --epochs 2
"""
import argparse, os, re, json, glob, random, time
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
ROOT = "/data/dan/dataset/MedVLThinker-Eval"
ap = argparse.ArgumentParser()
ap.add_argument("--student", default="/data/dan/weights/MedVLThinker-7B-RL_m23k")
ap.add_argument("--teacher_dir", default="ckpts/gate_32b_modes/nothink_cap320")  # 32B-nt teacher answers
ap.add_argument("--orig7b_dir", default="ckpts/gate_7b_prune/cap320")            # original 7B-nt (baseline)
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
                                                  "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--n", type=int, default=4000); ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--bs", type=int, default=1); ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--cap_div", type=int, default=4)
ap.add_argument("--out_dir", default="ckpts/train/fld"); ap.add_argument("--max_images", type=int, default=4)
A = ap.parse_args(); os.makedirs(A.out_dir, exist_ok=True)
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28; MAXPX = HIGH_PX // A.cap_div; DEV = "cuda"
SYS = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
ALL5 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU"]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def load_preds(d):
    m = {}
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        mm = re.match(r"ckpt_(.+?)_", os.path.basename(f)); ds = mm.group(1) if mm else None
        for l in open(f):
            if l.strip(): r = json.loads(l); m[(ds, r["idx"])] = r
    return m
print("loading labels...", flush=True)
tea = load_preds(A.teacher_dir); ori = load_preds(A.orig7b_dir)
ds_all = load_dataset(ROOT); split = "test" if "test" in ds_all else list(ds_all.keys())[0]; data = ds_all[split]
def subset(*k): return [i for i, n in enumerate(data["dataset_name"]) if any(x in n.lower().replace("-", "").replace("_", "") for x in k)]
def mx(t):
    o = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: o.append(i)
    return o
DSI = {"MedXpert-Reasoning": lambda: mx("reasoning"), "MedXpert-Understanding": lambda: mx("understanding"),
       "PMC-VQA": lambda: subset("pmcvqa", "pmc"), "SLAKE": lambda: subset("slake"),
       "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"), "PathVQA": lambda: subset("pathvqa", "path"), "MMMU": lambda: subset("mmmu")}
items = []  # (ds, idx, teacher_ans, teacher_ok, gold, orig7b_ok)
for name in A.datasets:
    if name not in DSI: continue
    idxs = DSI[name](); rng = random.Random(42); rng.shuffle(idxs); idxs = idxs[:A.n]
    for i in idxs:
        t, o = tea.get((name, i)), ori.get((name, i))
        if not t or not o: continue
        items.append((name, i, t["pred"], t["ok"], str(data[i]["answer_label"]).strip().upper()[:1], o["ok"]))
rng = random.Random(0); order = list(range(len(items))); rng.shuffle(order)
tr = [items[k] for k in order[:len(order) // 2]]; te = [items[k] for k in order[len(order) // 2:]]
# distill ONLY the learnable delta (teacher right, student WRONG) + 1:1 replay of both-right (anti-forgetting)
delta = [x for x in tr if x[3] == 1 and x[5] == 0]
both = [x for x in tr if x[3] == 1 and x[5] == 1]; random.Random(1).shuffle(both)
tr_pos = delta + both[:len(delta)]
print(f"{len(items)} items | train {len(tr)} | delta(teacher✓,student✗)={len(delta)} + replay={min(len(delta),len(both))} = {len(tr_pos)} | test {len(te)}", flush=True)

proc = AutoProcessor.from_pretrained(A.student)
LET = re.compile(r"\b([A-J])\b")
def msgs_for(i):
    ex = data[i]; opts = parse_opts(ex["options"]); q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    im = [{"type": "image", "image": x, "max_pixels": MAXPX, "min_pixels": MIN_PX} for x in (ex.get("images") or [])[:A.max_images]]
    return [{"role": "system", "content": SYS}, {"role": "user", "content": im + [{"type": "text", "text": q}]}]
def encode(i, target=None):
    m = msgs_for(i); text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    if target is not None: text = text + target
    imgs, vids = process_vision_info(m)
    return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)

print("loading 7B student + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.student, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
model.train(); t0 = time.time(); step = 0
for ep in range(A.epochs):
    random.Random(ep).shuffle(tr_pos)
    for it in tr_pos:
        try:
            enc = encode(it[1], target=it[2]).to(DEV)  # target = teacher's (correct) answer letter
            labels = enc["input_ids"].clone(); labels[:, :-1] = -100
            loss = model(**enc, labels=labels).loss / A.accum; loss.backward()
        except Exception as e:
            print(f"  skip: {str(e)[:70]}", flush=True); continue
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); opt.zero_grad()
        if step % 200 == 0: print(f"  ep{ep} step{step}/{len(tr_pos)} {(time.time()-t0)/60:.1f}min loss~{loss.item()*A.accum:.3f}", flush=True)
model.eval()
print("eval base (adapter OFF) vs distilled (adapter ON) on held-out test half, same forward...", flush=True)
from collections import defaultdict
dist_ok = defaultdict(list); orig_ok = defaultdict(list)
LETTER_IDS = {}
for L in [chr(ord('A') + k) for k in range(10)]:
    for v in (L, " " + L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: LETTER_IDS.setdefault(e[-1], L)
ids = torch.tensor(list(LETTER_IDS.keys()), device=DEV); idL = list(LETTER_IDS.values())
def pred_letter(enc):
    logit = model(**enc).logits[0, -1]
    return idL[int(torch.argmax(logit[ids]).item())]
with torch.no_grad():
    for it in te:
        name, i, _, _, gold, _ = it
        try:
            enc = encode(i).to(DEV)
            with model.disable_adapter():
                orig_ok[name].append(int(pred_letter(enc) == gold))   # base 7B, same forward (fair baseline)
            dist_ok[name].append(int(pred_letter(enc) == gold))       # distilled
        except Exception:
            orig_ok[name].append(0); dist_ok[name].append(0)
def pool(d, names):
    v = [x for n in names for x in d.get(n, [])]; return float(np.mean(v)) if v else float("nan")
print("\n==================== FastLeg-Distill (FLD) — cheap-leg lift ====================", flush=True)
print(f"  {'benchmark':<22}{'orig 7B-nt':>12}{'distilled':>12}{'Δ':>9}", flush=True)
for n in A.datasets:
    if dist_ok.get(n): print(f"  {n:<22}{np.mean(orig_ok[n]):>12.3f}{np.mean(dist_ok[n]):>12.3f}{np.mean(dist_ok[n])-np.mean(orig_ok[n]):>+9.3f}", flush=True)
for lab, names in [("POOLED ALL-5", ALL5), ("POOLED ALL-6", A.datasets)]:
    print(f"  {lab:<22}{pool(orig_ok,names):>12.3f}{pool(dist_ok,names):>12.3f}{pool(dist_ok,names)-pool(orig_ok,names):>+9.3f}", flush=True)
res = {"orig": {n: float(np.mean(orig_ok[n])) for n in orig_ok}, "distilled": {n: float(np.mean(dist_ok[n])) for n in dist_ok},
       "all5_orig": pool(orig_ok, ALL5), "all5_distilled": pool(dist_ok, ALL5),
       "all6_orig": pool(orig_ok, A.datasets), "all6_distilled": pool(dist_ok, A.datasets),
       "n_train_pos": len(tr_pos), "n_test": len(te), "epochs": A.epochs}
json.dump(res, open(os.path.join(A.out_dir, "result.json"), "w"), indent=1)
model.save_pretrained(A.out_dir); print(f"  saved -> {A.out_dir}", flush=True)
