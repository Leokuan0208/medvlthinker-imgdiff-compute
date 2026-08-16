#!/usr/bin/env python3
"""PART B -- independent re-derivation of the PMC-VQA output-side (answer-prior) correction,
and the leakage / answer-key-exploitation tests.  Written from scratch against the raw dumps."""
from __future__ import annotations
import csv, json, os, re, sys
import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
CK = os.path.join(ROOT, "ckpts/output_bias")
NBOOT, SEED = 10000, 20260817
csv.field_size_limit(10**9)

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def load(stem):
    rows = []
    for s in ("_s0of2", "_s1of2", ""):
        p = os.path.join(CK, stem + s + ".jsonl")
        if os.path.exists(p): rows += jl(p)
    return rows

def boot(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"), n=int(len(d)))

def strip_marker(t):
    return str(t).strip().lstrip("Ġ▁ ").strip()

def letter_logits(row, K):
    lp = row.get("first_logprobs") or {}
    best = {}
    for t, v in lp.items():
        s = strip_marker(t)
        if len(s) == 1 and "A" <= s <= "Z": best[s] = max(float(v), best.get(s, -1e9))
    floor = min([float(v) for v in lp.values()], default=-30.0) if lp else -30.0
    return np.array([best.get(chr(65 + i), floor) for i in range(K)]), \
           sum(1 for i in range(K) if chr(65 + i) in best)

def fit_shift(logits, target, iters=800, lr=0.3):
    n, K = logits.shape; w = np.zeros(K)
    for _ in range(iters):
        cur = np.bincount((logits - w).argmax(1), minlength=K) / n
        w = w + lr * np.log(np.maximum(cur, 1e-4) / np.maximum(target, 1e-4)); w -= w.mean()
    return w

R = {}
# ------------------------------------------------------------------ 1. LEAKAGE: train/test overlap
tr = list(csv.DictReader(open("/data/dan/dataset/medevalkit/PMC-VQA/train_2.csv")))
te = list(csv.DictReader(open("/data/dan/dataset/medevalkit/PMC-VQA/test_2.csv")))
fk = "Figure_path"
tr_fig = {r[fk] for r in tr}; te_fig = {r[fk] for r in te}
tr_qq = {(r[fk], r.get("Question", "").strip()) for r in tr}
te_qq = {(r[fk], r.get("Question", "").strip()) for r in te}
R["B1_leakage"] = dict(
    n_train=len(tr), n_test=len(te),
    figure_path_intersection=len(tr_fig & te_fig),
    figure_question_intersection=len(tr_qq & te_qq),
    train_gold_marginal={k: float(v) for k, v in
        zip("ABCD", np.bincount([ord(r["Answer"].strip()[0]) - 65 for r in tr], minlength=4) / len(tr))},
    test_gold_marginal={k: float(v) for k, v in
        zip("ABCD", np.bincount([ord(r["Answer"].strip()[0]) - 65 for r in te], minlength=4) / len(te))})

# ------------------------------------------------------------------ 2. eval + train dumps
ev = load("gen_PMC_VQA_id"); ev.sort(key=lambda r: r["i"])
trd = load("gen_PMC_TRAIN_train"); trd.sort(key=lambda r: r["i"])
X = np.array([letter_logits(r, 4)[0] for r in ev])
cov = np.array([letter_logits(r, 4)[1] for r in ev])
gold = np.array([ord(str(r["answer"]).strip()[0]) - 65 for r in ev])
Xtr = np.array([letter_logits(r, 4)[0] for r in trd])
gold_tr = np.array([ord(str(r["answer"]).strip()[0]) - 65 for r in trd])
tgt_train = np.bincount(gold_tr, minlength=4) / len(gold_tr)

R["B2_setup"] = dict(n_eval=len(ev), n_train_dump=len(trd),
                     all_four_letters_in_top20_frac=float((cov == 4).mean()),
                     train_dump_gold_marginal=[float(x) for x in tgt_train],
                     eval_gold_marginal=[float(x) for x in np.bincount(gold, minlength=4) / len(gold)])

# ------------------------------------------------------------------ 3. arms
readout = X.argmax(1)
w = fit_shift(Xtr, tgt_train)
pm_train = (X - w).argmax(1)
# cross-fit transductive
f = np.arange(len(X)) % 5
pm_cv = np.zeros(len(X), int)
for k in range(5):
    m = f != k
    pm_cv[~m] = (X[~m] - fit_shift(X[m], tgt_train)).argmax(1)
pm_nocv = (X - fit_shift(X, tgt_train)).argmax(1)
# ILLEGITIMATE control: fit on the EVAL GOLD marginal (the leakage trap, run deliberately)
tgt_eval = np.bincount(gold, minlength=4) / len(gold)
pm_evalgold = (X - fit_shift(X, tgt_eval)).argmax(1)

ok = {k: (v == gold).astype(float) for k, v in
      dict(readout=readout, pm_train=pm_train, pm_transductive_cv=pm_cv,
           pm_transductive_NOCV=pm_nocv, pm_EVALGOLD_leaky_control=pm_evalgold).items()}

R["B3_arms_vs_readout"] = {k: boot(v, ok["readout"]) for k, v in ok.items() if k != "readout"}
R["B3_absolute"] = {k: float(v.mean()) for k, v in ok.items()}
R["B3_w_pm_train"] = [float(x) for x in w]
R["B3_cross_fit_gap"] = float(ok["pm_transductive_cv"].mean() - ok["pm_transductive_NOCV"].mean())

# ------------------------------------------------------------------ 4. BALANCED ANSWER KEY
def balanced(okv, okref, gold, seed=SEED, reps=200):
    cls = {k: np.where(gold == k)[0] for k in range(4)}
    m = min(len(v) for v in cls.values()); rng = np.random.default_rng(seed)
    ds = []; per = {k: [] for k in cls}
    for _ in range(reps):
        sub = []
        for k, v in cls.items():
            pick = v[rng.choice(len(v), m, replace=False)]
            sub.append(pick); per[k].append(float((okv[pick] - okref[pick]).mean()))
        sub = np.concatenate(sub); ds.append(float((okv[sub] - okref[sub]).mean()))
    ds = np.array(ds)
    return dict(delta=float(ds.mean()),
                ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                per_gold_letter={"ABCD"[k]: float(np.mean(v)) for k, v in per.items()},
                n_per_class=int(m), reps=reps)

R["B4_balanced_key"] = {k: balanced(v, ok["readout"], gold) for k, v in ok.items() if k != "readout"}

# ------------------------------------------------------------------ 5. predicted marginals
R["B5_predicted_marginal"] = {k: [float(x) for x in np.bincount(v, minlength=4) / len(v)]
                              for k, v in dict(readout=readout, pm_train=pm_train,
                                               pm_transductive_cv=pm_cv, gold=gold).items()}
json.dump(R, open(os.path.join(OUT, "partB.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
