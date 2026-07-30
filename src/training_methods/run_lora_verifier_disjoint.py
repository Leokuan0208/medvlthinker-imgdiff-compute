#!/usr/bin/env python3
"""run_lora_verifier_disjoint.py -- retrain the open-text answer verifier on a STRICTLY DISJOINT split.

Identical architecture / objective / hyperparameters to src/training_methods/run_lora_verifier_open.py
(the trainer that produced the contaminated ckpts/train/lora_verifier_pooled4). THE ONLY CHANGE IS THE
DATA: instead of a grouped 70/30 split *of the evaluation sets*, the training items come from
src/training_methods/build_disjoint_verifier_split.py, which shares no question and no image with the
three reported eval sets (proven by decoded-pixel hash; see verifier_disjoint_split.json).

Unchanged from run_lora_verifier_open.py, verbatim:
  base model  lingshu-medical-mllm/Lingshu-7B, bf16, flash_attention_2
  LoRA        r=16, alpha=2*r=32, dropout=0.05, bias=none,
              target_modules = q,k,v,o,gate,up,down _proj    (matches pooled4/adapter_config.json)
  objective   next-token CE on the single "Yes"/"No" continuation token (labels[:, :-1] = -100)
  optimizer   AdamW lr=1e-4, grad-clip 1.0, bs=2, accum=8, epochs=1
  prompt      SYS + image(max_pixels=1280*28*28, min_pixels=4*28*28) + "Question/Proposed answer" body
  examples    one per unique NORMALIZED answer per question, label = the 32B judge's judge_ok
  seed        0

COMPOSITION-MATCHED (--match_composition, default on). The contaminated verifier's 10364 training
examples were 894 SLAKE / 4973 PathVQA / 522 VQA-RAD / 3975 Kvasir (reproduced exactly from its
recorded seed-0 grouped split; see the artifact). This trainer draws the SAME per-family example
counts from the disjoint pools, so the clean-vs-contaminated contrast isolates data PROVENANCE
(contaminated vs disjoint) rather than training-set size or domain mix. Any family that cannot fill
its quota from disjoint data is topped up from radimagenet_open and the shortfall is recorded.

--level L1 = image-disjoint (no eval image, no eval item)          <- fair "only the source changed"
--level L2 = L1 + no eval question TEXT at all                     <- conservative bound

Every eval item is held out BY CONSTRUCTION, so there is no internal test split: training uses the
disjoint pool and measurement is done afterwards by src/training_methods/verifier_transfer_eval.py
(the same pyes() scoring function) over the FULL eval sets.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/run_lora_verifier_disjoint.py --level L1 \
    --out_dir ckpts/train/lora_verifier_disjoint
"""
import argparse, os, json, glob, io, math, random, time
import numpy as np, torch
from collections import defaultdict
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
# per-FAMILY training-example quotas = the contaminated verifier's own composition, reproduced from its
# recorded seed-0 grouped 70/30 split over {slake,pathvqa,vqa_rad,kvasir}_open (total 10364 = the count
# in logs/verif_retrain_r16.log). radimagenet is the top-up pool, not a quota.
CONTAMINATED_QUOTA = {"slake": 894, "pathvqa": 4973, "vqa_rad": 522, "kvasir": 3975}
FAMILY = {"slake_open_train": "slake", "pathvqa_open_train": "pathvqa",
          "vqa_rad_open_train": "vqa_rad", "kvasir_open": "kvasir", "radimagenet_open": "radimagenet"}

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--max_train", type=int, default=10364,
                help="total training (answer,label) examples. Default = the contaminated verifier's count.")
ap.add_argument("--match_composition", type=int, default=1)
ap.add_argument("--level", choices=["L1", "L2"], default="L1")
ap.add_argument("--cap_div", type=int, default=1)
ap.add_argument("--out_dir", default="ckpts/train/lora_verifier_disjoint"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--split", default="results/cascade_methods/artifacts/verifier_disjoint_split.json")
ap.add_argument("--idx_dir", default="data/disjoint_split")
ap.add_argument("--deadline_s", type=float, default=0.0, help=">0: stop training cleanly at this wall time")
A = ap.parse_args(); os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; MAXPX = HIGH_PX // A.cap_div; DEV = "cuda"
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK = os.path.join(ROOT, os.environ.get("VERIF_CK", "ckpts/openvqa/cheap_lingshu7b"))
TAG = os.environ.get("VERIF_TAG", "lingshu7b")
T0 = time.time()
def loadj(p): return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}
def norm(s): return str(s).strip().lower()

# ---------------- the disjoint training pool ----------------
SPL = json.load(open(os.path.join(ROOT, A.split)))
assert SPL["disjointness_assertion"]["image_pixel_hash_intersection"] == 0, "split is not image-disjoint"
DSETS = list(SPL["train"].keys())
pref = "idx_" if A.level == "L1" else "strict_idx_"
ALLOW = {}
for ds in DSETS:
    p = os.path.join(ROOT, A.idx_dir, f"{pref}{ds}.json")
    assert os.path.exists(p), f"missing allowlist {p} -- run build_disjoint_verifier_split.py"
    ALLOW[ds] = set(json.load(open(p)))
print(f"[split] {A.split} level={A.level}; sources={DSETS}", flush=True)
print(f"[split] eval held out by construction: {SPL['eval']}", flush=True)
print(f"[split] train n eval image intersection = 0 (asserted in the split builder)", flush=True)

# ---------------- image maps ----------------
def slake_imgs(split):
    m = {}
    for x in json.load(open(f"/data/dan/dataset/slake/{split}.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip): m[x["qid"]] = (x["question"], ip)
    return m
def parquet_imgs(base, split, want):
    import pandas as pd
    m = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))], ignore_index=True)
    for i, r in df.iterrows():
        if int(i) not in want: continue
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"): continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m
def json_imgs(jp):
    return {r["idx"]: (r["question"], r["img_path"]) for r in json.load(open(jp)) if os.path.exists(r["img_path"])}

print("[data] loading training images ...", flush=True)
IMG = {}
for ds in DSETS:
    if ds == "slake_open_train":     IMG[ds] = slake_imgs("train")
    elif ds == "vqa_rad_open_train": IMG[ds] = parquet_imgs("/data/dan/dataset/vqa_rad/data", "train", ALLOW[ds])
    elif ds == "pathvqa_open_train": IMG[ds] = parquet_imgs("/data/dan/dataset/path_vqa/data", "train", ALLOW[ds])
    elif ds == "kvasir_open":        IMG[ds] = json_imgs("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")
    elif ds == "radimagenet_open":   IMG[ds] = json_imgs("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")
    else: raise SystemExit(f"unknown training source {ds}")
    IMG[ds] = {k: v for k, v in IMG[ds].items() if k in ALLOW[ds]}
    print(f"  {ds:20s} images={len(IMG[ds])}", flush=True)

# ---------------- per-question records + judge labels ----------------
QREC = {}
for ds in DSETS:
    sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
    jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
    if not sc or not jud:
        print(f"  !! {ds}: missing sc8({len(sc)}) or judge({len(jud)}) -- SKIPPED", flush=True); continue
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]
    n = 0
    for i in sc:
        if i not in ALLOW[ds] or i not in IMG[ds] or i not in aj: continue
        q, img = IMG[ds][i]
        QREC[(ds, i)] = {"q": q, "img": img, "preds": sc[i]["preds"], "slabels": aj[i]}
        n += 1
    print(f"  {ds:20s} usable questions={n}", flush=True)
print(f"{len(QREC)} disjoint training questions with images+labels", flush=True)
assert QREC, "no training data -- generate the *_train sc8 + judge files first"

# ---------------- examples, with composition matching ----------------
rng = random.Random(A.seed)
by_src = defaultdict(list)
for k in sorted(QREC, key=lambda t: (t[0], str(t[1]))):
    r = QREC[k]
    for na, lab in r["slabels"].items():
        surf = next((a for a in r["preds"] if norm(a) == na), na)
        by_src[k[0]].append((k[0], r["q"], r["img"], surf, lab))
avail = {ds: len(v) for ds, v in by_src.items()}
print(f"[examples] available per source: {avail}", flush=True)

train_ex, taken, short = [], {}, {}
if A.match_composition:
    for ds in DSETS:
        fam = FAMILY[ds]
        if fam == "radimagenet": continue
        want = CONTAMINATED_QUOTA[fam]
        pool = by_src.get(ds, []); rng.shuffle(pool)
        take = pool[:want]; train_ex += take; taken[ds] = len(take)
        if len(take) < want: short[ds] = want - len(take)
    deficit = A.max_train - len(train_ex)
    if deficit > 0:
        pool = by_src.get("radimagenet_open", []); rng.shuffle(pool)
        take = pool[:deficit]; train_ex += take; taken["radimagenet_open"] = len(take)
        print(f"[examples] topped up {len(take)} examples from radimagenet_open "
              f"(quota shortfalls: {short})", flush=True)
else:
    for ds in DSETS: train_ex += by_src.get(ds, [])
    rng.shuffle(train_ex); train_ex = train_ex[:A.max_train]
    for e in train_ex: taken[e[0]] = taken.get(e[0], 0) + 1
rng.shuffle(train_ex)
print(f"train examples={len(train_ex)} (target {A.max_train}); pos rate "
      f"{np.mean([e[4] for e in train_ex]):.3f}", flush=True)
print(f"[examples] per-source taken: {taken}", flush=True)
print(f"[examples] contaminated reference composition: {CONTAMINATED_QUOTA} (total 10364)", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]; NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
def build_msgs(q, img, proposed):
    body = f"Question: {q}\nProposed answer: {proposed}\nIs the proposed answer correct? Answer Yes or No."
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                         {"type": "text", "text": body}]}]
def encode(q, img, proposed, label=None):
    msgs = build_msgs(q, img, proposed)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if label is not None: text = text + ("Yes" if label == 1 else "No")
    imgs, vids = process_vision_info(msgs)
    return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)

print("loading Lingshu-7B + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.model_path, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").to(DEV)
lcfg = LoraConfig(r=A.lora_r, lora_alpha=2*A.lora_r, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg); model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
model.train(); t0 = time.time(); step = 0; nskip = 0; stop = False
for ep in range(A.epochs):
    rng.shuffle(train_ex)
    for bi in range(0, len(train_ex), A.bs):
        for (ds, q, img, ans, lab) in train_ex[bi:bi+A.bs]:
            try:
                enc = encode(q, img, ans, lab).to(DEV)
                labels = enc["input_ids"].clone(); labels[:, :-1] = -100
                out = model(**enc, labels=labels); (out.loss/(A.bs*A.accum)).backward()
            except Exception as e:
                nskip += 1; print(f"  skip: {str(e)[:70]}", flush=True)
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if step % 50 == 0: print(f"  ep{ep} step{step}/{len(train_ex)//A.bs} {(time.time()-t0)/60:.1f}min", flush=True)
        if A.deadline_s > 0 and (time.time()-T0) > A.deadline_s:
            print(f"  DEADLINE at step {step}; stopping cleanly", flush=True); stop = True; break
    if stop: break

model.save_pretrained(os.path.join(ROOT, A.out_dir))
json.dump({"split": A.split, "level": A.level, "train_sources": DSETS,
           "n_train_questions": len(QREC), "n_train_examples": len(train_ex),
           "available_examples_per_source": avail, "taken_examples_per_source": taken,
           "contaminated_reference_composition": CONTAMINATED_QUOTA,
           "quota_shortfalls_topped_up_from_radimagenet": short,
           "match_composition": bool(A.match_composition),
           "pos_rate": float(np.mean([e[4] for e in train_ex])), "skipped_examples": nskip,
           "steps": step, "epochs": A.epochs, "bs": A.bs, "accum": A.accum, "lr": A.lr,
           "lora_r": A.lora_r, "lora_alpha": 2*A.lora_r, "lora_dropout": 0.05,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
           "base_model": A.model_path, "cap_div": A.cap_div, "max_pixels": MAXPX, "seed": A.seed,
           "train_minutes": round((time.time()-t0)/60, 1), "early_stopped": stop},
          open(os.path.join(ROOT, A.out_dir, "train_config.json"), "w"), indent=1)
print(f"saved -> {A.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)
