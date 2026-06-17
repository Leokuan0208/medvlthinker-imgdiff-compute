#!/usr/bin/env python3
"""
router_learned_6ds.py - train a LEARNED multi-feature router (gradient-boosted) and test
whether a real ML model beats the 1-D margin gate on transfer.

** This is a NEW file (router_learned.py is left unchanged). It extends the learned-router
   comparison to all 6 datasets: the 4 competent benchmarks + MMMU + MedXpert (its two
   native tracks, combined into MedX-M below). Pooled rows for competent-4 (continuity with
   the old number) and all-6. Reads the MMMU/MedXpert gate checkpoints that already exist in
   ckpts/gate_7b_vllm and ckpts/gate_32b. Saves to ckpts/router_learned_6ds.pkl. **

Features per question (all from 7B opt_logprobs, NO labels at test): margin, entropy,
and the sorted 10-dim option log-prob distribution = 12 dims. Fit a HistGradientBoosting
classifier on the CLEAN PMC-VQA-train labels, set tau by the err-rate budget (7B labels
only), freeze, apply UNCHANGED to all 6 eval datasets. Printed beside the 1-D margin
baseline (ckpts/router_margin.pkl), with always-7B / always-32B / oracle as anchors.

Honest hypothesis: the learned router fits PMC in-domain but does NOT transfer better
than the 1-D margin -- richer features carry PMC's confidence SCALE, which shifts across
datasets. If so, this is the ablation showing the simple gate is at the frontier.
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier

EVAL      = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
             "MedXpert-Reasoning", "MedXpert-Understanding"]
COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
MEDX      = ["MedXpert-Reasoning", "MedXpert-Understanding"]
PMCTRAIN_DIR = "ckpts/gate_7b_pmctrain"
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}(?:_s\dof\d)?\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def load_pmctrain(ckdir):
    rows = []
    for f in glob.glob(os.path.join(ckdir, "ckpt_nothink*.jsonl")):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def entropy(row):
    lp = row.get("opt_logprobs") or {}; v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return 0.0
    p = np.exp(v - v.max()); p /= p.sum(); return float(-(p*np.log(p+1e-12)).sum())
def dist_vec(row, K=10, pad=-20.0):
    lp = row.get("opt_logprobs") or {}; v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return np.full(K, pad)
    v = v - (v.max() + np.log(np.exp(v - v.max()).sum())); v = v[:K]
    if v.size < K: v = np.concatenate([v, np.full(K - v.size, pad)])
    return v
def feat(row): return np.concatenate([[margin(row), entropy(row)], dist_vec(row)]).astype(np.float32)
def feats(rows): return np.array([feat(r) for r in rows], dtype=np.float32)
def feats_idx(by, idx): return np.array([feat(by[i]) for i in idx], dtype=np.float32)

# ---- train learned router on clean PMC-train ----
train = load_pmctrain(PMCTRAIN_DIR); assert train, f"no labels in {PMCTRAIN_DIR}"
ytr = np.array([r["ok"] for r in train]).astype(float); Xtr = feats(train)
clf = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.08,
        l2_regularization=1.0, random_state=42, validation_fraction=0.15, early_stopping=True)
clf.fit(Xtr, ytr)
err = 1.0 - ytr.mean()
tau = float(np.quantile(clf.predict_proba(Xtr)[:, 1], err))

# ---- load 7B (nothink) + 32B (think) correctness + frozen 1-D margin gate ----
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
FR  = pickle.load(open("ckpts/router_margin.pkl", "rb")); fg, ftau = FR["gate"], FR["tau"]
def m1(by, idx): return np.array([[margin(by[i])] for i in idx], dtype=np.float32)

# ---- evaluate every dataset; keep raw per-question arrays for exact (count-weighted) pooling ----
store = {}
for name in EVAL:
    if name not in r7 or name not in r32:
        print(f"    [missing checkpoints] {name}"); continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    a7  = np.array([r7[name][i]["ok"]  for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    Pb  = fg.predict_proba(m1(r7[name], idx))[:, 1];        eb = Pb < ftau; rb = np.where(eb, a32, a7)
    Pl  = clf.predict_proba(feats_idx(r7[name], idx))[:, 1]; el = Pl < tau;  rl = np.where(el, a32, a7)
    store[name] = dict(a7=a7, a32=a32, eb=eb, rb=rb, el=el, rl=rl, orac=np.maximum(a7, a32))

def cat(keys, field): return np.concatenate([store[k][field] for k in keys if k in store])
def line(label, keys):
    keys = [k for k in keys if k in store]
    if not keys: return
    a7=cat(keys,'a7'); a32=cat(keys,'a32'); eb=cat(keys,'eb'); rb=cat(keys,'rb')
    el=cat(keys,'el'); rl=cat(keys,'rl'); orac=cat(keys,'orac'); n=len(a7)
    print(f"    {label:<20}{n:>6}{a7.mean():>8.3f}{a32.mean():>8.3f} |"
          f"{rb.mean():>8.3f}{eb.mean()*100:>5.0f}%{rb.mean()-a32.mean():>+8.3f} |"
          f"{rl.mean():>8.3f}{el.mean()*100:>5.0f}%{rl.mean()-a32.mean():>+8.3f} |{orac.mean():>8.3f}")

print("=" * 110)
print(f"LEARNED ROUTER (HistGBM, 12 feats)  train={PMCTRAIN_DIR}  n={len(train)}  "
      f"7B_acc={ytr.mean():.3f}  tau={tau:.3f}")
print("=" * 110)
print(f"    {'dataset':<20}{'n':>6}{'a7':>8}{'a32':>8} |{'  margin-gate (1-D)':<21} |{'  learned (HistGBM)':<21} |{'oracle':>8}")
print(f"    {'':<20}{'':>6}{'':>8}{'':>8} |{'     acc  esc    gain':<21} |{'     acc  esc    gain':<21} |{'':>8}")
print("    " + "-" * 106)
for ds in COMPETENT: line(ds, [ds])
line("MMMU", ["MMMU"])
line("MedX-M (comb)", MEDX)
print("    " + "-" * 106)
line("POOLED competent-4", COMPETENT)
line("POOLED all-6", COMPETENT + ["MMMU"] + MEDX)

with open("ckpts/router_learned_6ds.pkl", "wb") as f:
    pickle.dump({"clf": clf, "tau": tau, "features": "margin+entropy+dist10", "trained_on": "pmc_vqa_train"}, f)
print("\n    -> saved ckpts/router_learned_6ds.pkl")
print("=" * 110)
print("READ: 'gain' = routed acc minus always-32B (in accuracy points). Compare the margin-gate")
print("block with the HistGBM block: if HistGBM's gain is no better than the 1-D margin gate across")
print("the 6 datasets, the richer features overfit PMC's confidence scale -> the 1-D margin gate is")
print("the deployable frontier. MMMU + MedX-M are the harder (excluded-from-cascade) sets.")
