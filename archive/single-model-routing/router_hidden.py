#!/usr/bin/env python3
"""
router_hidden.py - Flavor-2 feasibility test: 4-arm correctness router on
layer-L hidden states from extract_features.py. Same eval protocol as
router_scalar.py (repeated stratified k-fold, shuffle control, oracle framing).

Features (3584-d) -> StandardScaler -> PCA(n) -> LogisticRegression per arm.
PCA is essential: 3584 dims on ~350 train rows/fold overfits wildly otherwise.
Compares h_last vs h_mean pooling. Routes argmax P(correct), cheap tie-break.
"""
import json, glob, os, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CKDIR    = "archive/single-model-routing/gate_7b_rag_axes"
FEATDIR  = "feats"
DATASETS = ["MedXpert-Reasoning", "MedXpert-Understanding", "PMC-VQA"]
ARMS     = ["nothink_norag", "think_norag", "think_rag_StatPearls", "think_rag_Textbooks"]
ARM_COST = np.array([0, 1, 2, 2])
N_SPLITS, N_REPEATS = 5, 10
PCA_DIMS = 64                      # sweepable; 64 is safe for ~350 rows/fold

# ---- labels from gate checkpoints ----
pat = re.compile(r"ckpt_(.+?)_(" + "|".join(ARMS) + r")(?:_s\dof\d)?\.jsonl$")
lab = defaultdict(lambda: defaultdict(dict))     # ds -> cell -> idx -> ok
for f in sorted(glob.glob(os.path.join(CKDIR, "*.jsonl"))):
    m = pat.search(os.path.basename(f))
    if not m: continue
    dsx, cell = m.group(1), m.group(2)
    for l in open(f):
        if l.strip():
            r = json.loads(l); lab[dsx][cell][r["idx"]] = r["ok"]

# ---- features from npz ----
def load_feats(ds):
    fs = glob.glob(os.path.join(FEATDIR, f"feat_{ds}_L*.npz"))
    if not fs: return None
    parts = [np.load(f) for f in fs]
    idx   = np.concatenate([p["idx"] for p in parts])
    hlast = np.concatenate([p["h_last"] for p in parts])
    hmean = np.concatenate([p["h_mean"] for p in parts])
    return idx, hlast, hmean

def build(ds, pooling):
    got = load_feats(ds)
    if got is None: return None
    idx, hlast, hmean = got
    feat = hlast if pooling == "last" else hmean
    fmap = {int(i): feat[k] for k,i in enumerate(idx)}
    common = set(fmap) & set(lab[ds][ARMS[0]])
    for a in ARMS[1:]: common &= set(lab[ds][a])
    common = sorted(common)
    X = np.stack([fmap[i] for i in common])
    Y = np.array([[lab[ds][a][i] for a in ARMS] for i in common], int)
    return X, Y

def route_eval(X, Y, shuffle_labels=False, seed0=0):
    res = defaultdict(list); arm_auc = defaultdict(list)
    strat = Y.max(axis=1)
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed0+rep)
        for tr, te in skf.split(X, strat):
            Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
            if shuffle_labels:
                Ytr = Ytr[np.random.RandomState(seed0+rep).permutation(len(tr))]
            P = np.zeros((len(te), len(ARMS)))
            for a in range(len(ARMS)):
                ya = Ytr[:, a]
                if ya.min() == ya.max():
                    P[:, a] = float(ya.mean()); continue
                pipe = make_pipeline(
                    StandardScaler(),
                    PCA(n_components=min(PCA_DIMS, Xtr.shape[0]-1, Xtr.shape[1])),
                    LogisticRegression(max_iter=2000, C=0.5))
                pipe.fit(Xtr, ya)
                P[:, a] = pipe.predict_proba(Xte)[:, 1]
                if not shuffle_labels and len(set(Yte[:, a])) > 1:
                    arm_auc[ARMS[a]].append(roc_auc_score(Yte[:, a], P[:, a]))
            choice = (P - 1e-6*ARM_COST).argmax(axis=1)
            res["routed"].append(Yte[np.arange(len(te)), choice].mean())
            res["baseline"].append(Yte[:, 0].mean())
            res["bestfixed"].append(Yte[:, Ytr.mean(axis=0).argmax()].mean())
            res["oracle"].append(Yte.max(axis=1).mean())
    return res, arm_auc

def ms(v): return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"

def report(name, X, Y, pooling):
    print("\n" + "="*70 + f"\n{name}  [{pooling}-pool]  (N={len(X)}, PCA={PCA_DIMS})\n" + "="*70)
    res, auc = route_eval(X, Y)
    sh, _   = route_eval(X, Y, shuffle_labels=True)
    print("  per-arm correctness AUROC:")
    for a in ARMS:
        print(f"     {a:24s} {np.mean(auc[a]):.3f}" if auc[a] else f"     {a:24s} (degenerate)")
    print(f"  best-fixed acc = {ms(res['bestfixed'])}")
    print(f"  ROUTED     acc = {ms(res['routed'])}")
    print(f"  oracle     acc = {ms(res['oracle'])}")
    diff = np.array(res['routed']) - np.array(res['bestfixed'])
    print(f"  >> routed - best_fixed = {np.mean(diff):+.3f} +/- {np.std(diff):.3f}")
    print(f"  >> shuffle-control routed = {ms(sh['routed'])}")
    gap = np.mean(res['oracle']) - np.mean(res['bestfixed'])
    if gap > 0:
        print(f"  >> oracle gap captured = {100*np.mean(diff)/gap:+.0f}%")
    return np.mean(diff)

summary = {}
for pooling in ["last", "mean"]:
    for ds in DATASETS:
        b = build(ds, pooling)
        if b is None: print(f"[missing feats] {ds}"); continue
        summary[(ds,pooling)] = report(ds, *b, pooling)
    # pooled
    parts = [build(ds, pooling) for ds in DATASETS]
    if all(p is not None for p in parts):
        X = np.concatenate([p[0] for p in parts]); Y = np.concatenate([p[1] for p in parts])
        summary[("POOLED",pooling)] = report("POOLED (3 datasets)", X, Y, pooling)

print("\n" + "="*70 + "\nSUMMARY: routed - best_fixed\n" + "="*70)
for (ds,pl),v in summary.items():
    print(f"  {ds:22s} [{pl}] {v:+.3f}")
print("\nREAD: any cell clearly > 0 beyond std with shuffle collapsing to best-fixed")
print("=> hidden state carries routing signal the cheap features lacked. Expect PMC-VQA")
print("best case, MedXpert-Reasoning worst (correctness direction weak for reasoning).")
