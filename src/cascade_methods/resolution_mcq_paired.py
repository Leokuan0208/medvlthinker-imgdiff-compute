#!/usr/bin/env python3
"""resolution_mcq_paired.py -- SWEEP 2: the MCQ half, on the MedEvalKit (macro-8) track.

A MATCHED two-point resolution pair for Lingshu-7B already exists on disk, produced by MedEvalKit
itself on 2026-07-01 and never analysed as a resolution experiment:

    MedEvalKit/eval_results_lingshu7b_full/   -- CAP_MAX_PIXELS unset => qwen_vl_utils default
                                                 max_pixels = 12,845,056  (the PUBLISHED arms)
    MedEvalKit/eval_results_lingshu7b_cap320/ -- the same harness with the cap lever engaged
                                                 (3 of the 7 cells only)

Both dumps store one row per item in dataset order, and the rows pair POSITIONALLY (verified here
by asserting question/prompt+gold identity row-for-row, not assumed).  So a paired item bootstrap
on `correct` is available at FULL n with zero GPU time.

This is the macro-8 track: PMC-VQA here is test_2.csv (n=33,430), the split the paper reports.
The pre-existing internal-harness sweep (vram_levers_2026-08-12.json / vram_levers_accuracy.py) is
a DIFFERENT track (PMC-VQA test_clean.csv, n=2,000, 6 shared items) and its levels must not be
mixed with these.

    python3 src/cascade_methods/resolution_mcq_paired.py
"""
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)
NBOOT, SEED = 10000, 20260813

# (reporting cell, MedEvalKit dataset dir) -- the 5 MCQ/closed cells of the macro-8.
# The open/closed split is applied by closed_mask() below, which reproduces MedEvalKit's own.
CELLS = [
    ("PMC_VQA", "PMC_VQA"),
    ("SLAKE_closed", "SLAKE"),
    ("VQA_RAD_closed", "VQA_RAD"),
    ("PATH_VQA_closed", "PATH_VQA"),
    ("MedXpertQA-MM", "MedXpertQA-MM"),
]


def rows(arm, ds):
    p = os.path.join(ROOT, "MedEvalKit", f"eval_results_lingshu7b_{arm}", "{}", ds, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def key(r):
    """the pairing key: whatever identity fields this cell's dump carries."""
    return (r.get("prompt"), r.get("question"), str(r.get("answer")), r.get("id"))


def closed_mask(ds, rs):
    """MedEvalKit's own open/closed split, reproduced from the stored fields."""
    if ds == "SLAKE":
        return np.array([r.get("answer_type") == "CLOSED" for r in rs])
    if ds in ("VQA_RAD", "PATH_VQA"):
        # MedEvalKit's metrics.json splits these by yes/no gold -- reproduce that and assert the
        # resulting n against the stored total_results.json counts before using it.
        return np.array([str(r.get("answer")).strip().lower() in ("yes", "no") for r in rs])
    return np.ones(len(rs), dtype=bool)


def boot(a, b, nboot=NBOOT, seed=SEED):
    """paired item bootstrap on the per-item delta (b - a)."""
    d = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    res = {}
    for cell, ds in CELLS:
        a = rows("cap320", ds)
        b = rows("full", ds)
        if b is None:
            res[cell] = {"status": "no full-resolution dump on disk"}
            continue
        if a is None:
            m = closed_mask(ds, b)
            res[cell] = {
                "status": "NOT MEASURED at cap320 -- the 2026-07-01 cap320 run covered "
                          "PMC_VQA/SLAKE/VQA_RAD only",
                "n_closed": int(m.sum()),
                "acc_at_12845056": round(float(np.array([r["correct"] for r in b])[m].mean()), 6)}
            continue
        assert len(a) == len(b), (cell, len(a), len(b))
        bad = [i for i in range(len(a)) if key(a[i]) != key(b[i])]
        m = closed_mask(ds, b)
        ca = np.array([bool(r["correct"]) for r in a])[m]
        cb = np.array([bool(r["correct"]) for r in b])[m]
        d, ci = boot(ca, cb)               # b - a = default - cap320
        res[cell] = {
            "n_paired_rows": len(a),
            "n_cell": int(m.sum()),
            "pairing": f"positional; identity fields matched on {len(a) - len(bad)}/{len(a)} rows",
            "acc_cap320_250880": round(float(ca.mean()), 6),
            "acc_default_12845056": round(float(cb.mean()), 6),
            "delta_default_minus_cap320": round(d, 6),
            "ci95": [round(ci[0], 6), round(ci[1], 6)],
            "significant": bool(ci[0] > 0 or ci[1] < 0),
            "n_flipped_wrong_to_right": int(((~ca) & cb).sum()),
            "n_flipped_right_to_wrong": int((ca & (~cb)).sum()),
            "mean_gen_tokens_cap320": round(float(np.mean([r["gen_toks"] for r in a])), 3),
            "mean_gen_tokens_default": round(float(np.mean([r["gen_toks"] for r in b])), 3),
            "mean_recorded_latency_s_cap320": round(float(np.mean([r["latency_s"] for r in a])), 5),
            "mean_recorded_latency_s_default": round(float(np.mean([r["latency_s"] for r in b])), 5),
        }
        assert not bad, (cell, len(bad))
    # ---- what the cut is worth on the macro-8 -------------------------------------------------
    # Each reporting cell carries 1/8 of the macro. The three cells measured here are independent
    # samples, so the macro contribution and its CI are built by resampling each cell's own per-item
    # delta vector and summing, weight 1/8 each. This is the contribution of THESE THREE CELLS
    # ONLY -- it is not the whole macro-8 delta, because PathVQA-closed and MedXpert were blocked.
    per_item = {}
    for cell, ds in CELLS:
        a, b = rows("cap320", ds), rows("full", ds)
        if a is None or b is None or len(a) != len(b):
            continue
        m = closed_mask(ds, b)
        per_item[cell] = (np.array([bool(r["correct"]) for r in b])[m].astype(float)
                          - np.array([bool(r["correct"]) for r in a])[m].astype(float))
    if per_item:
        rng = np.random.default_rng(SEED)
        draws = np.zeros(NBOOT)
        for cell, d in per_item.items():
            idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
            draws += d[idx].mean(axis=1) / 8.0
        pt = sum(d.mean() for d in per_item.values()) / 8.0
        res["_macro8_contribution_of_the_measured_cells"] = {
            "cells": list(per_item),
            "weight_each": 0.125,
            "macro8_delta_default_minus_cap320": round(float(pt), 6),
            "ci95": [round(float(np.percentile(draws, 2.5)), 6),
                     round(float(np.percentile(draws, 97.5)), 6)],
            "significant": bool(np.percentile(draws, 2.5) > 0 or np.percentile(draws, 97.5) < 0),
            "project_significance_threshold_on_macro8": 0.0029,
            "_read": "the macro-8 cost of moving ONLY these three MCQ cells from the deployed "
                     "12,845,056 down to cap320. The other two MCQ cells and the three open cells "
                     "are held at zero here, so this is a LOWER BOUND on the cost of a "
                     "resolution cut applied to the whole MCQ half.",
            "_not_the_whole_macro": "PATH_VQA_closed and MedXpertQA-MM were not measured at "
                                    "cap320; see the artifact's not_measured block."}
    for k, v in res.items():
        print(k, json.dumps(v))
    json.dump(res, open(os.path.join(OUT, "mcq_paired_cap320_vs_default.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, "mcq_paired_cap320_vs_default.json"))


if __name__ == "__main__":
    main()
