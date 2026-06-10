#!/usr/bin/env python3
"""
Select a fixed-seed sample of PMC-VQA TRAIN rows for held-out threshold fitting,
VERIFIED disjoint from the eval set's PMC-VQA questions (no leak).
Writes a clean JSONL the train-label runner consumes:
  {idx, question, options{A-D}, answer_label, image_path}
"""
import pandas as pd, json, random, os, re
from datasets import load_dataset

N = 3000
CSV = "/data/dan/dataset/pmc_vqa_train/train.csv"
IMG_DIR = "/data/dan/dataset/pmc_vqa_train/images"
OUT = "/data/dan/dataset/pmc_vqa_train/train_sample_3000.jsonl"
random.seed(42)

df = pd.read_csv(CSV)
print(f"train.csv rows: {len(df)}")

# --- build the eval set's PMC-VQA question set, to guarantee disjointness ---
ev = load_dataset("/data/dan/dataset/MedVLThinker-Eval")
ev = ev["test" if "test" in ev else list(ev.keys())[0]]
eval_q = set()
for i,n in enumerate(ev["dataset_name"]):
    if "pmc" in n.lower():
        eval_q.add(ev[i]["question"].strip())
print(f"eval PMC-VQA questions to exclude: {len(eval_q)}")

def clean_choice(s):
    # options look like ' A:Diffuse uptake pattern ' or ' B: Red ' -> strip letter prefix + ws
    s = str(s).strip()
    return re.sub(r'^[A-D]\s*:\s*', '', s).strip()

# filter: image exists AND question not in eval set
rows = []
skipped_img = skipped_leak = 0
for _, r in df.iterrows():
    q = str(r["Question"]).strip()
    if q in eval_q: skipped_leak += 1; continue
    fp = str(r["Figure_path"]).strip()
    if not os.path.exists(os.path.join(IMG_DIR, fp)): skipped_img += 1; continue
    lab = str(r["Answer_label"]).strip().upper()[:1]
    if lab not in "ABCD": continue
    rows.append({
        "question": q,
        "options": {L: clean_choice(r[f"Choice {L}"]) for L in "ABCD"},
        "answer_label": lab,
        "image_path": os.path.join(IMG_DIR, fp),
    })
print(f"eligible rows: {len(rows)}  (skipped {skipped_leak} eval-leak, {skipped_img} missing-image)")

random.shuffle(rows)
sample = rows[:N]
with open(OUT, "w") as f:
    for k, r in enumerate(sample):
        r["idx"] = k
        f.write(json.dumps(r) + "\n")
print(f">> wrote {len(sample)} samples to {OUT}")
print("   leak check: 0 eval questions in sample by construction (filtered above)")
