#!/usr/bin/env python3
"""
calm_fuse.py - CALM-Fuse: trained, model-agnostic ANSWER FUSION of the two no-think tiers.
Motivation: small-nt and big-nt are COMPLEMENTARY (union-oracle beats the best single by +0.07..+0.14 on ALL
5 families). Routing/distillation are capped/flat; FUSION can capture part of that headroom at the cost of
running BOTH no-think passes (~0.37s total << 11s think). Train a tiny head on [small_logprobs(A-J),
big_logprobs(A-J), margins, disagree] -> per-option score; predict argmax over valid options.
Compares: always-small-nt, always-big-nt (best single), CALM-Fuse (per-family), CALM-Fuse (LOFO transfer:
train on other 4 families, test held-out -> model-agnostic), union-oracle (ceiling). Light CPU training.
Run: python3 src/training_methods/calm_fuse.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs, ALL6, ALL5
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
LETTERS = [chr(ord('A') + k) for k in range(10)]
FAM = {"medvlthinker": ("ckpts/gate_7b_prune/cap320", "ckpts/gate_32b_modes/nothink_cap320"),
       "lingshu": ("ckpts/acc_gen/lingshu7b/cap320", "ckpts/acc_gen/lingshu32b/nothink_cap320"),
       "qoq": ("ckpts/acc_gen/qoq7b/cap320", "ckpts/acc_gen/qoq32b/nothink_cap320"),
       "chiron": ("ckpts/acc_gen/chiron2b/nt", "ckpts/acc_gen/chiron8b/nt"),
       "medgemma": ("ckpts/acc_gen/medgemma4b/nt", "ckpts/acc_gen/medgemma27b/nt")}
def loadarm(d):
    o = defaultdict(dict)
    for f in glob.glob(os.path.join(d, "ckpt_*.jsonl")):
        m = re.match(r"ckpt_(.+?)_", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); o[m.group(1)][r["idx"]] = r
    return o
def vec(lp): lp = lp or {}; return np.array([lp.get(L, -20.0) for L in LETTERS], float)
def build(fam):
    s = loadarm(FAM[fam][0]); b = loadarm(FAM[fam][1])
    X, Y, M, DS, SOK, BOK = [], [], [], [], [], []
    for ds in ALL6:
        for i in set(s.get(ds, {})) & set(b.get(ds, {})):
            sr, br = s[ds][i], b[ds][i]; sl, bl = sr.get("opt_logprobs") or {}, br.get("opt_logprobs") or {}
            gold = sr["gold"]; valid = sorted(set(sl) | set(bl) | {gold})
            if gold not in LETTERS: continue
            ssig, bsig = signals_from_logprobs(sl), signals_from_logprobs(bl)
            X.append(np.concatenate([vec(sl), vec(bl), [ssig["margin"], bsig["margin"], float(sr["pred"] != br["pred"])]]))
            Y.append(LETTERS.index(gold)); M.append(np.array([1.0 if L in valid else 0.0 for L in LETTERS]))
            DS.append(ds); SOK.append(sr["ok"]); BOK.append(br["ok"])
    return (np.array(X), np.array(Y), np.array(M), np.array(DS), np.array(SOK), np.array(BOK))
DATA = {fam: build(fam) for fam in FAM}
def mk(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, early_stopping=True, random_state=0))
def fuse_acc(clf, Xte, Yte, Mte):
    full = np.full((len(Xte), 10), -1e9); pr = clf.predict_proba(Xte)
    for j, c in enumerate(clf.classes_): full[:, c] = pr[:, j]
    full[Mte == 0] = -1e9
    return (full.argmax(1) == Yte).mean()
def pool_acc(ok, DS, names, te=None):
    m = np.isin(DS, names)
    if te is not None: m = m & te
    return ok[m].mean() if m.any() else float("nan")
print("############  CALM-Fuse: trained model-agnostic answer fusion (small-nt + big-nt), ALL-6  ############")
print(f"  {'family':<13}{'small-nt':>9}{'big-nt':>8}{'FUSE/fam':>10}{'FUSE/LOFO':>11}{'oracle':>9}{'capture%':>10}")
rows = {}
for fam in FAM:
    X, Y, M, DS, SOK, BOK = DATA[fam]; n = len(X)
    rng = np.random.default_rng(0); perm = rng.permutation(n); tr, te = perm[:n // 2], perm[n // 2:]
    small = SOK[te].mean(); big = BOK[te].mean(); oracle = np.maximum(SOK, BOK)[te].mean(); best = max(small, big)
    cf = mk().fit(X[tr], Y[tr]); fuse = fuse_acc(cf, X[te], Y[te], M[te])
    # LOFO: train on the other 4 families (all their data), test on this family's te split
    oX = np.vstack([DATA[f][0] for f in FAM if f != fam]); oY = np.concatenate([DATA[f][1] for f in FAM if f != fam])
    cl = mk().fit(oX, oY); lofo = fuse_acc(cl, X[te], Y[te], M[te])
    cap = (fuse - best) / (oracle - best) * 100 if oracle > best else float("nan")
    rows[fam] = (small, big, fuse, lofo, oracle, cap)
    print(f"  {fam:<13}{small:>9.3f}{big:>8.3f}{fuse:>10.3f}{lofo:>11.3f}{oracle:>9.3f}{cap:>9.0f}%")
print("\n  capture% = fraction of the small+big complementarity headroom (oracle - best-single) that the trained fuser captures")
print(f"  {'mean':<13}" + "".join(f"{np.nanmean([rows[f][k] for f in FAM]):>{[9,8,10,11,9,9][k-1]+(1 if k==6 else 0)}.3f}" for k in range(1,6)) + f"{np.nanmean([rows[f][5] for f in FAM]):>9.0f}%")
json.dump({f: {"small": rows[f][0], "big": rows[f][1], "fuse_perfam": rows[f][2], "fuse_lofo": rows[f][3],
               "oracle": rows[f][4], "capture_pct": rows[f][5]} for f in FAM},
          open("results/cascade_methods/calm_fuse.json", "w"), indent=1)
print("\n  -> results/cascade_methods/calm_fuse.json")
