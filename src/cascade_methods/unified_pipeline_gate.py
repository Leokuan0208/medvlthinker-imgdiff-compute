#!/usr/bin/env python3
"""unified_pipeline_gate.py -- ATTACK 2, amendment 3.

Arm A said the option-scoring verifier is a BAD ANSWERER on the option cells.  This asks the
different question: is it a better ESCALATION GATE than the deployed 7B margin?  (A gate routes to
the strong model; the pipeline still answers every item -- CRITICAL RULE 6 respected.)

Two targets:
  DETECTION       y = (7B greedy is wrong)
  RECOVERABILITY  y = (32B-direct right AND 7B greedy wrong)   <- what a cascade gate must rank

Four unfitted features, plus a logistic combiner under NESTED 5-fold CV with 10 fold-split seeds.

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_gate.py --tag zeroshot
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

NFOLD = 5
FOLD_SEEDS = list(range(10))


def logistic_fit(X, y, iters=400, lr=0.5, l2=1e-3):
    """Plain gradient-descent logistic regression on standardised features (no sklearn dependency,
    fully deterministic, no thread sensitivity)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0]
        w -= lr * g
    return (mu, sd, w)


def logistic_apply(m, X):
    mu, sd, w = m
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1.0 / (1.0 + np.exp(-Z @ w))


def run(tag):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    work = U.build_worklist()
    out = {"tag": tag, "nfold": NFOLD, "fold_seeds": FOLD_SEEDS, "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        keep = [j for j in range(len(rows))
                if sc.get(rows[j]["i"]) is not None
                and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
        if not keep:
            out["cells"][cell] = {"status": "not measured"}
            continue
        idx = [rows[j]["i"] for j in keep]
        ok7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        ok32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        mar = U.sevenb_margin(cell)[idx]
        picks7, _n5 = U.sevenb_pick(cell, [rows[j] for j in keep])
        top12, s_of_7b, disagree = [], [], []
        for j in keep:
            r = rows[j]
            v = np.array([sc[r["i"]][c] for c in range(len(r["cands"]))], float)
            sv = np.sort(v)[::-1]
            top12.append(float(sv[0] - sv[1]))
            p7 = picks7.get(r["i"])
            s_of_7b.append(float(v[p7]) if p7 is not None and 0 <= p7 < len(v) else float(v.mean()))
            disagree.append(0.0 if p7 is None else float(int(np.argmax(v)) != p7))
        F = {"neg_margin7b": -mar,
             "neg_verifier_top1_minus_top2": -np.array(top12),
             "verifier_disagrees_with_7b": np.array(disagree),
             "neg_verifier_score_of_7b_answer": -np.array(s_of_7b)}
        X = np.column_stack([F[k] for k in F])
        targets = {"detection_7b_wrong": (1 - ok7),
                   "recoverability_32b_fixes_it": ((ok32 == 1) & (ok7 == 0)).astype(float)}
        cellout = {"n": len(keep), "base_rate": {}, "unfitted_auroc": {}, "fitted_combiner_auroc": {}}
        for tname, y in targets.items():
            cellout["base_rate"][tname] = float(y.mean())
            if y.sum() == 0 or y.sum() == len(y):
                cellout["unfitted_auroc"][tname] = "degenerate target"
                continue
            cellout["unfitted_auroc"][tname] = {k: U.auroc(F[k], y.astype(int)) for k in F}
            aucs = []
            for fs in FOLD_SEEDS:
                rng = np.random.default_rng(20260812 + fs)
                fold = rng.permutation(len(y)) % NFOLD
                pred = np.empty(len(y))
                for f in range(NFOLD):
                    tr, te = fold != f, fold == f
                    if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                        pred[te] = 0.0
                        continue
                    pred[te] = logistic_apply(logistic_fit(X[tr], y[tr]), X[te])
                aucs.append(U.auroc(pred, y.astype(int)))
            aucs = np.array(aucs, float)
            cellout["fitted_combiner_auroc"][tname] = {
                "mean": float(aucs.mean()), "sd": float(aucs.std(ddof=1)),
                "min": float(aucs.min()), "max": float(aucs.max()),
                "beats_margin7b_by": float(aucs.mean()
                                           - cellout["unfitted_auroc"][tname]["neg_margin7b"])}
        out["cells"][cell] = cellout
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag)
    p = os.path.join(U.PARTS, f"gate_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
