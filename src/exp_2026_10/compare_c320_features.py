#!/usr/bin/env python3
"""compare_c320_features.py — paired comparison: probes refit on hidden states read at the generation cap (cap320, C320)
vs the shipped run (hidden states read at max_pixels 1,003,520), same pools, same picks rule, same folds.

Per question: Δ = pick(C320) − pick(shipped) in each metric. Macro = mean over the 8 open benchmarks of the per-benchmark
mean Δ. CI: paired bootstrap, 10,000 resamples of images within each benchmark (as final_method_eval.py).
Pre-specified adoption rule (readout_variants_2026-10-15): adopt only if it helps under both judges, significantly
under at least one, and does not significantly hurt lenient EM. Also reports how often the two runs pick differently.
"""
import json
from collections import defaultdict
import numpy as np

REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_{}_records.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/c320_features_vs_shipped_2026-10-19.json"
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
METRICS = ["judge", "xjudge", "em", "f1", "llava_recall"]
B, SEED = 10000, 0

base, new = json.load(open(REC.format("T07"))), json.load(open(REC.format("C320")))
rng = np.random.default_rng(SEED)
res = {"title": "C320 features (refit) vs shipped features, paired", "no_fabricated_numbers": True,
       "inputs": [REC.format("T07"), REC.format("C320")], "bootstrap": f"{B} image-clustered resamples within benchmark, seed {SEED}",
       "cells": {}, "macro": {}}
boot = {m: np.zeros(B) for m in METRICS}
for c in OPEN8:
    a = {(r["idx"]): r for r in base[c]}
    b = {(r["idx"]): r for r in new[c]}
    keys = sorted(set(a) & set(b))
    assert len(keys) == len(a) == len(b), c
    imgs = sorted({a[k]["img"] for k in keys})
    gi = {g: i for i, g in enumerate(imgs)}
    G = np.array([gi[a[k]["img"]] for k in keys])
    cell = {"n": len(keys), "picks_differ": int(sum(a[k]["pick_ans"] != b[k]["pick_ans"] for k in keys))}
    for m in METRICS:
        d = np.array([b[k]["pick"][m] - a[k]["pick"][m] for k in keys], float)
        s = np.bincount(G, d, len(imgs)); n = np.bincount(G, None, len(imgs))
        idx = rng.integers(0, len(imgs), (B, len(imgs)))
        bs = s[idx].sum(1) / n[idx].sum(1)
        boot[m] += bs / len(OPEN8)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        cell[m] = {"delta": float(d.mean()), "ci": [float(lo), float(hi)]}
    res["cells"][c] = cell
for m in METRICS:
    est = float(np.mean([res["cells"][c][m]["delta"] for c in OPEN8]))
    lo, hi = np.percentile(boot[m], [2.5, 97.5])
    res["macro"][m] = {"delta": est, "ci": [float(lo), float(hi)],
                       "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"}
j, x, e = (res["macro"][k] for k in ("judge", "xjudge", "em"))
adopt = j["delta"] > 0 and x["delta"] > 0 and (j["verdict"] == "WIN" or x["verdict"] == "WIN") and e["verdict"] != "LOSS"
res["rule"] = "adopt iff positive under both judges, significant under at least one, and lenient EM not significantly negative"
res["adopt"] = bool(adopt)
res["picks_differ_total"] = sum(res["cells"][c]["picks_differ"] for c in OPEN8)
res["n_total"] = sum(res["cells"][c]["n"] for c in OPEN8)
json.dump(res, open(OUT, "w"), indent=1)
for m in METRICS:
    r = res["macro"][m]
    print(f"{m:13s} {r['delta']:+.4f} [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] {r['verdict']}")
print("picks differ", res["picks_differ_total"], "/", res["n_total"], "| adopt:", res["adopt"])
for c in OPEN8:
    cc = res["cells"][c]
    print(f"  {c:17s} differ {cc['picks_differ']:5d}/{cc['n']:5d}  judge {cc['judge']['delta']:+.4f} [{cc['judge']['ci'][0]:+.4f},{cc['judge']['ci'][1]:+.4f}]  "
          f"xjudge {cc['xjudge']['delta']:+.4f}  em {cc['em']['delta']:+.4f}")
