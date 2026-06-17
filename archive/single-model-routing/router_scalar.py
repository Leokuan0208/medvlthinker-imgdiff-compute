#!/usr/bin/env python3
"""
router_scalar.py - Flavor-1 feasibility test (zero GPU).
Trains a 4-arm correctness-prediction router on CHEAP features that already
exist on disk: the baseline (nothink_norag) cell's option-logprob confidence
+ gen_tokens. Routes to argmax predicted P(correct), tie-break to cheapest arm.
Evaluates with repeated stratified k-fold CV vs baseline / best-fixed / oracle.

This is a cascade-style router: it observes the cheap baseline's confidence,
then decides whether/where to escalate. Features come ONLY from the baseline,
never from the expensive arms (which you wouldn't have run at decision time).
"""
import json, glob, os, re, math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CKDIR    = "archive/single-model-routing/gate_7b_rag_axes"
DATASETS = ["MedXpert-Reasoning", "MedXpert-Understanding", "PMC-VQA"]
ARMS     = ["nothink_norag", "think_norag", "think_rag_StatPearls", "think_rag_Textbooks"]
ARM_COST = np.array([0, 1, 2, 2])      # tie-break preference: cheaper arm wins ties
BASELINE = "nothink_norag"             # features come from this cell
N_SPLITS, N_REPEATS = 5, 10

pat = re.compile(r"ckpt_(.+?)_(" + "|".join(ARMS) + r")(?:_s\dof\d)?\.jsonl$")
rec = defaultdict(lambda: defaultdict(dict))   # ds -> cell -> idx -> row
for f in sorted(glob.glob(os.path.join(CKDIR, "*.jsonl"))):
    m = pat.search(os.path.basename(f))
    if not m: continue
    ds, cell = m.group(1), m.group(2)
    for l in open(f):
        if l.strip():
            r = json.loads(l); rec[ds][cell][r["idx"]] = r

def feats(row):
    """Confidence features from the baseline cell's option logprobs."""
    lp = row.get("opt_logprobs") or {}
    vals = sorted(lp.values(), reverse=True)
    gt, pk = row.get("gen_tokens", 0), row.get("parse_ok", 0)
    if len(vals) < 2:                          # no/low info -> low-confidence sentinel
        return [0.0, 0.0, 0.0, math.log(max(len(vals),2)), 1.0/max(len(vals),2), len(vals), gt, pk]
    top1, top2 = vals[0], vals[1]
    top3 = vals[2] if len(vals) > 2 else top2
    mx = vals[0]; exps = [math.exp(v - mx) for v in vals]; Z = sum(exps)
    probs = [e / Z for e in exps]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return [top1, top1 - top2, top1 - top3, ent, probs[0], len(vals), gt, pk]

def build(ds_list):
    X, Y = [], []
    for ds in ds_list:
        idxs = set(rec[ds][ARMS[0]])
        for a in ARMS[1:]: idxs &= set(rec[ds][a])
        for i in sorted(idxs):
            X.append(feats(rec[ds][BASELINE][i]))
            Y.append([rec[ds][a][i]["ok"] for a in ARMS])
    return np.array(X, float), np.array(Y, int)

def route_eval(X, Y, shuffle_labels=False, seed0=0):
    res = defaultdict(list); arm_auc = defaultdict(list)
    strat = Y.max(axis=1)                       # stratify on "any arm correct"
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed0 + rep)
        for tr, te in skf.split(X, strat):
            Xtr, Xte, Ytr, Yte = X[tr], X[te], Y[tr], Y[te]
            if shuffle_labels:                  # control: break feature<->label link
                Ytr = Ytr[np.random.RandomState(seed0 + rep).permutation(len(tr))]
            sc = StandardScaler().fit(Xtr)
            Xtr2, Xte2 = sc.transform(Xtr), sc.transform(Xte)
            P = np.zeros((len(te), len(ARMS)))
            for a in range(len(ARMS)):
                ya = Ytr[:, a]
                if ya.min() == ya.max():        # degenerate arm (all right/all wrong on train)
                    P[:, a] = float(ya.mean())
                else:
                    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr2, ya)
                    P[:, a] = clf.predict_proba(Xte2)[:, 1]
                    if not shuffle_labels and len(set(Yte[:, a])) > 1:
                        arm_auc[ARMS[a]].append(roc_auc_score(Yte[:, a], P[:, a]))
            choice = (P - 1e-6 * ARM_COST).argmax(axis=1)     # route, cheap tie-break
            res["routed"].append(Yte[np.arange(len(te)), choice].mean())
            res["baseline"].append(Yte[:, 0].mean())          # always arm 0 = nothink_norag
            res["bestfixed"].append(Yte[:, Ytr.mean(axis=0).argmax()].mean())  # best arm chosen on TRAIN
            res["oracle"].append(Yte.max(axis=1).mean())
    return res, arm_auc

def ms(v): return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"

def report(name, X, Y):
    print("\n" + "=" * 70 + f"\n{name}   (N={len(X)})\n" + "=" * 70)
    res, auc = route_eval(X, Y)
    sh, _   = route_eval(X, Y, shuffle_labels=True)
    print("  per-arm correctness-probe AUROC (how well cheap features predict each arm):")
    for a in ARMS:
        print(f"     {a:24s} {np.mean(auc[a]):.3f}" if auc[a] else f"     {a:24s} (degenerate)")
    print(f"\n  baseline (always nothink_norag) acc = {ms(res['baseline'])}")
    print(f"  best-fixed (best arm on train)  acc = {ms(res['bestfixed'])}")
    print(f"  ROUTED (our router)             acc = {ms(res['routed'])}")
    print(f"  oracle (best arm per question)  acc = {ms(res['oracle'])}")
    diff = np.array(res['routed']) - np.array(res['bestfixed'])
    print(f"\n  >> routed - best_fixed = {np.mean(diff):+.3f} +/- {np.std(diff):.3f}   (THE number)")
    print(f"  >> shuffle-control routed = {ms(sh['routed'])}  (should ~= best-fixed if router is real)")
    gap = np.mean(res['oracle']) - np.mean(res['bestfixed'])
    if gap > 0:
        print(f"  >> oracle gap captured = {100*np.mean(diff)/gap:+.0f}% of the {gap:.3f} available")

for ds in DATASETS:
    X, Y = build([ds]); report(ds, X, Y)
Xa, Ya = build(DATASETS); report("POOLED (all 3 datasets)", Xa, Ya)
print("\n" + "=" * 70)
print("READ: routed-best_fixed > 0 beyond its std, AND shuffle collapses to")
print("best-fixed => cheap features carry a real (if small) routing signal,")
print("=> extract hidden states for the strong test. Flat => cheap signal dead.")
