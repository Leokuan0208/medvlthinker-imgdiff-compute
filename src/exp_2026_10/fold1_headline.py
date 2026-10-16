#!/usr/bin/env python3
"""fold1_headline.py -- the final method's gain over greedy on image half 1 ONLY: the questions that played no
part in any choice made in this round (the temperature map was chosen on half-0 questions; the readout and
probe recipe were chosen on half-0 questions of the cap320 pools). Same records, same bootstrap convention."""
import json
import numpy as np

REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_{}_records.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/final_fold1_headline_2026-10-16.json"
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
CURS = ["judge", "xjudge", "em", "f1", "llava_recall"]
rng = np.random.default_rng(20261021)
out = {"title": "Final method, image half 1 only (out of sample for every choice made this round)",
       "no_fabricated_numbers": True, "policies": {}}
for pol in ("Tmap", "T07"):
    R = json.load(open(REC.format(pol)))
    parts = {cu: [] for cu in CURS}
    cells = {}
    for c in OPEN8:
        Q = [r for r in R[c] if r["fold"] == 1]
        keys, inv = np.unique([r["img"] for r in Q], return_inverse=True)
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=10000).astype(float)
        cells[c] = {"n": len(Q)}
        for cu in CURS:
            m = np.array([r["pick"].get(cu) is not None and r["greedy"].get(cu) is not None for r in Q])
            d = np.array([(r["pick"][cu] - r["greedy"][cu]) if mm else 0.0 for r, mm in zip(Q, m)])
            Rs = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / np.maximum(
                W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
            cells[c][cu] = float(d[m].mean())
            parts[cu].append((cells[c][cu], Rs))
    mac = {}
    for cu, lst in parts.items():
        pts = np.array([p for p, _ in lst]); Rm = np.stack([r for _, r in lst], 1).mean(1)
        lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
        mac[cu] = {"macro": float(pts.mean()), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                   "n_positive": int((pts > 0).sum())}
    out["policies"][pol] = {"macro": mac, "cells": cells}
    print(pol, "fold 1 only:")
    for cu, v in mac.items():
        print(f"  {cu:13} {v['macro']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {v['verdict']} {v['n_positive']}/8")
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
