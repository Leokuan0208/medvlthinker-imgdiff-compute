#!/usr/bin/env python3
"""
prep_pmctrain_sample.py - sample a clean training subset from PMC-VQA's train split.

m23k (the RL data for our 7B) is text-only [MedQA/MedMCQA/HeadQA], so PMC-VQA's
train split is UNSEEN by the model -> clean to train the router's gate on. This
samples N rows (seed 42), resolves image paths, and writes a subset manifest that
the 7B-nothink labeler will consume. NO model is run here; this is just selection.

It PRINTS the detected schema first -- if the column mapping below doesn't match,
the script stops and shows the real columns so we fix the mapping (no silent error).
"""
import os, glob, json, sys, pandas as pd

ROOT    = "/data/dan/dataset/pmc_vqa_train"
IMG_DIR = os.path.join(ROOT, "images")
OUT     = os.path.join(ROOT, "router_train_sample_4k_seed42.parquet")
N, SEED = 4000, 42

# --- column mapping (PMC-VQA standard); ADJUST if the printed schema differs ---
COLS = {
    "question":     "Question",
    "answer_label": "Answer_label",          # the gold letter, e.g. "A"
    "image":        "Figure_path",           # filename under images/
    "choices":      ["Choice A", "Choice B", "Choice C", "Choice D"],
}

files = sorted(glob.glob(os.path.join(ROOT, "data", "*.parquet"))) or \
        sorted(glob.glob(os.path.join(ROOT, "data", "*", "*.parquet")))
if not files:
    sys.exit(f"No parquet under {ROOT}/data — run `ls -R {ROOT}/data` and tell me the layout.")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

print("=" * 80)
print(f"loaded {len(df):,} rows from {len(files)} parquet file(s)")
print("COLUMNS:", df.columns.tolist())
print("FIRST ROW:")
for k, v in df.iloc[0].to_dict().items():
    s = str(v)
    print(f"   {k!r}: {s[:120]}{'...' if len(s) > 120 else ''}")
print("=" * 80)

# validate mapping loudly
need = [COLS["question"], COLS["answer_label"], COLS["image"], *COLS["choices"]]
missing = [c for c in need if c not in df.columns]
if missing:
    sys.exit(f"\nMAPPING MISMATCH — these columns are not in the data: {missing}\n"
             f"Edit COLS at the top of this script to match the COLUMNS printed above, then rerun.")

# sample, resolve images, write subset
sub = df.sample(n=min(N, len(df)), random_state=SEED).reset_index(drop=True)
sub["__img_path"] = sub[COLS["image"]].apply(lambda fn: os.path.join(IMG_DIR, str(fn)))
missing_img = (~sub["__img_path"].apply(os.path.exists)).sum()
print(f"sampled {len(sub):,} rows (seed {SEED}); images missing on disk: {missing_img}")
if missing_img > len(sub) * 0.02:
    print("WARNING: >2% images missing — check the image filename column / extension.")

sub.to_parquet(OUT, index=False)
print(f"wrote subset -> {OUT}")
print("eval-set 7B accuracy for the contamination check after labeling: 0.539")
