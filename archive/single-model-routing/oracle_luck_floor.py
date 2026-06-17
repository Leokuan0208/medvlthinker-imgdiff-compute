#!/usr/bin/env python3
"""Permutation control: is the oracle gap real complementarity or max-of-K luck?
Shuffles each cell's ok-vector independently (preserves marginal acc, kills
any real per-question complementarity) and recomputes the oracle B times."""
import json, glob, os, re, random
from collections import defaultdict

CKDIR = "archive/single-model-routing/gate_7b_rag_axes"
DATASETS = ["MedXpert-Reasoning", "MedXpert-Understanding", "PMC-VQA"]
CELLS = ["nothink_norag", "think_norag", "think_rag_StatPearls", "think_rag_Textbooks"]
B = 2000
random.seed(42)

pat = re.compile(r"ckpt_(.+?)_(" + "|".join(CELLS) + r")(?:_s\dof\d)?\.jsonl$")
rec = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(CKDIR, "*.jsonl"))):
    m = pat.search(os.path.basename(f))
    if not m: continue
    ds, cell = m.group(1), m.group(2)
    for l in open(f):
        if l.strip():
            r = json.loads(l); rec[(ds, cell)][r["idx"]] = r["ok"]

data = {ds: [[rec[(ds,c)][i] for i in sorted(rec[(ds,CELLS[0])])] for c in CELLS]
        for ds in DATASETS}

def hits(vecs):
    n = len(vecs[0])
    return sum(1 for i in range(n) if any(v[i] for v in vecs))

def shuffled(vecs):
    out = []
    for v in vecs:
        w = v[:]; random.shuffle(w); out.append(w)
    return out

print("="*72)
print(f"ORACLE LUCK-FLOOR CONTROL   (permutations={B}, seed=42)")
print("="*72)
for ds in DATASETS:
    vecs = data[ds]; n = len(vecs[0])
    real = hits(vecs)/n
    sims = [hits(shuffled(vecs))/n for _ in range(B)]
    mu = sum(sims)/B; sd = (sum((s-mu)**2 for s in sims)/B)**0.5
    z = (real-mu)/sd if sd else float('inf')
    print(f"\n### {ds}")
    print(f"    real oracle     = {real:.3f}")
    print(f"    luck floor      = {mu:.3f} +/- {sd:.3f}")
    print(f"    real - luck     = {real-mu:+.3f}   (z = {z:+.1f} sigma)")

tot = sum(len(data[ds][0]) for ds in DATASETS)
real = sum(hits(data[ds]) for ds in DATASETS)/tot
sims = [sum(hits(shuffled(data[ds])) for ds in DATASETS)/tot for _ in range(B)]
mu = sum(sims)/B; sd = (sum((s-mu)**2 for s in sims)/B)**0.5
z = (real-mu)/sd if sd else float('inf')
print("\n" + "="*72 + "\nPOOLED\n" + "="*72)
print(f"    real oracle = {real:.3f}    luck floor = {mu:.3f} +/- {sd:.3f}")
print(f"    real - luck = {real-mu:+.3f}   (z = {z:+.1f} sigma)")
print("\n  real ~ luck   -> heterogeneity is max-of-K artifact; direction dead")
print("  real >> luck  -> genuine complementarity; weak-but-real router signal stands")
