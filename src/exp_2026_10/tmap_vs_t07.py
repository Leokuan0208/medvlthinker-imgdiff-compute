#!/usr/bin/env python3
"""tmap_vs_t07.py -- is the frozen per-benchmark temperature map a real improvement?

Both policies were scored with the SAME probes (final_method_eval FINAL_cap320, --policy T07 Tmap), so the
paired difference isolates the candidate pools. The map was chosen (temp_crossfit_2026-10-15.json) using
ONLY questions from image half 0 (md5("nd"+img)%2==0); image half 1 (fold 1 here) played no part in choosing
it, so fold 1 is the OUT-OF-SAMPLE test. Reported for fold 1 (the test), fold 0 (in-sample for the choice)
and both. Paired bootstrap, 10,000, clustered by image within cell; macro over the eight open cells.
"""
import json
import numpy as np

REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_{}_records.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/tmap_vs_t07_2026-10-16.json"
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
CURS = ["judge", "xjudge", "em", "f1", "llava_recall"]
a, b = json.load(open(REC.format("Tmap"))), json.load(open(REC.format("T07")))
rng = np.random.default_rng(20261020)
out = {"title": "Frozen per-benchmark temperature map vs fixed T=0.7, same probes", "no_fabricated_numbers": True,
       "out_of_sample_fold": 1, "cells": {}, "macro": {}}
parts = {}
for c in OPEN8:
    A = {r["idx"]: r for r in a[c]}; B = {r["idx"]: r for r in b[c]}
    qs = [q for q in A if q in B]
    out["cells"][c] = {}
    for scope, folds in (("fold1_out_of_sample", [1]), ("fold0_in_sample", [0]), ("both", [0, 1])):
        Q = [q for q in qs if A[q]["fold"] in folds]
        keys, inv = np.unique([A[q]["img"] for q in Q], return_inverse=True)
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=10000).astype(float)
        o = {"n": len(Q)}
        for cu in CURS:
            m = np.array([A[q]["pick"].get(cu) is not None and B[q]["pick"].get(cu) is not None for q in Q])
            d = np.array([(A[q]["pick"][cu] - B[q]["pick"][cu]) if mm else 0.0 for q, mm in zip(Q, m)])
            Rs = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / np.maximum(
                W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
            o[cu] = float(d[m].mean()) if m.any() else 0.0
            parts.setdefault((scope, cu), []).append((o[cu], Rs))
        out["cells"][c][scope] = o
for (scope, cu), lst in parts.items():
    pts = np.array([p for p, _ in lst]); Rm = np.stack([r for _, r in lst], 1).mean(1)
    lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    out["macro"].setdefault(scope, {})[cu] = {"macro": float(pts.mean()), "ci": [lo, hi],
                                              "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                                              "n_positive": int((pts > 0).sum()), "n_changed_cells": int((pts != 0).sum())}
json.dump(out, open(OUT, "w"), indent=1)
for scope in ("fold1_out_of_sample", "fold0_in_sample", "both"):
    print(scope)
    for cu in CURS:
        v = out["macro"][scope][cu]
        print(f"  {cu:13} {v['macro']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {v['verdict']:4} positive {v['n_positive']}/8")
for c in OPEN8:
    o = out["cells"][c]["fold1_out_of_sample"]
    print(f"  fold1 {c:17} n {o['n']:5} judge {o['judge']:+.4f} xjudge {o['xjudge']:+.4f} em {o['em']:+.4f}")
print("wrote", OUT)
