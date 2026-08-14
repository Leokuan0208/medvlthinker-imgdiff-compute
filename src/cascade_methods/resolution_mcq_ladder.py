#!/usr/bin/env python3
"""resolution_mcq_ladder.py -- SWEEP 2: analyse THIS SESSION's MedEvalKit resolution ladder.

runners/run_resolution_mcq_ladder.sh writes MedEvalKit/eval_results_res7b_px<PX>/ for each cap,
including a control at the harness default (12,845,056).  Every arm is tp=1, seed 42, same vLLM,
same session, so any two of them pair item-for-item and the delta is clean.  The 2026-07-01 tp=2
dumps are NOT differenced against these (see resolution_mcq_paired.py, which pairs those two
against each other instead).

    python3 src/cascade_methods/resolution_mcq_ladder.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)
DEFAULT_PX = 12845056
NBOOT, SEED = 10000, 20260813
CELLS = [("SLAKE_closed", "SLAKE"), ("VQA_RAD_closed", "VQA_RAD"),
         ("PATH_VQA_closed", "PATH_VQA"), ("MedXpertQA-MM", "MedXpertQA-MM"),
         ("PMC_VQA", "PMC_VQA")]


def rows(px, ds):
    p = os.path.join(ROOT, "MedEvalKit", f"eval_results_res7b_px{px}", "{}", ds, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def mask(ds, rs):
    if ds == "SLAKE":
        return np.array([r.get("answer_type") == "CLOSED" for r in rs])
    if ds in ("VQA_RAD", "PATH_VQA"):
        return np.array([str(r.get("answer")).strip().lower() in ("yes", "no") for r in rs])
    return np.ones(len(rs), bool)


def ident(r):
    return (r.get("prompt"), r.get("question"), str(r.get("answer")), r.get("id"))


def boot(a, b):
    d = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), size=(NBOOT, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    caps = sorted({int(os.path.basename(d).split("px")[-1])
                   for d in glob.glob(os.path.join(ROOT, "MedEvalKit", "eval_results_res7b_px*"))})
    res = {"_control": DEFAULT_PX, "_caps_found": caps, "by_cap": {}, "macro": {}}
    for px in caps:
        blk = {"max_pixels": px, "vision_token_budget": px // (28 * 28), "per_cell": {}}
        for cell, ds in CELLS:
            a = rows(DEFAULT_PX, ds)
            b = rows(px, ds)
            if a is None or b is None:
                blk["per_cell"][cell] = "not measured at this cap"
                continue
            if len(a) != len(b):
                blk["per_cell"][cell] = f"length mismatch {len(a)} vs {len(b)}"
                continue
            bad = sum(1 for i in range(len(a)) if ident(a[i]) != ident(b[i]))
            m = mask(ds, a)
            ca = np.array([bool(r["correct"]) for r in a])[m]
            cb = np.array([bool(r["correct"]) for r in b])[m]
            row = {"n": int(m.sum()),
                   "identity_mismatched_rows": bad,
                   "acc_control_12845056": round(float(ca.mean()), 6),
                   "acc_at_cap": round(float(cb.mean()), 6),
                   "mean_gen_tokens_at_cap": round(float(np.mean([r["gen_toks"] for r in b])), 3),
                   "mean_recorded_latency_s_at_cap": round(
                       float(np.mean([r["latency_s"] for r in b])), 5)}
            if px != DEFAULT_PX:
                d, ci = boot(ca, cb)
                row.update(delta_vs_control=round(d, 6),
                           ci95=[round(ci[0], 6), round(ci[1], 6)],
                           significant=bool(ci[0] > 0 or ci[1] < 0))
            blk["per_cell"][cell] = row
        got = [c for c in blk["per_cell"].values() if isinstance(c, dict)]
        if got:
            blk["macro_over_measured_cells"] = {
                "n_cells": len(got),
                "cells": [k for k, v in blk["per_cell"].items() if isinstance(v, dict)],
                "mean_acc_at_cap": round(float(np.mean([g["acc_at_cap"] for g in got])), 6),
                "mean_acc_control": round(float(np.mean([g["acc_control_12845056"] for g in got])), 6),
                "mean_delta": round(float(np.mean([g.get("delta_vs_control", 0.0) for g in got])), 6),
                "n_cells_significantly_worse": int(sum(
                    1 for g in got if g.get("significant") and g.get("delta_vs_control", 0) < 0)),
                "n_cells_significantly_better": int(sum(
                    1 for g in got if g.get("significant") and g.get("delta_vs_control", 0) > 0))}
        res["by_cap"][px] = blk
    json.dump(res, open(os.path.join(OUT, "mcq_ladder.json"), "w"), indent=1)
    for px, b in res["by_cap"].items():
        mm = b.get("macro_over_measured_cells", {})
        print(f"px={px:>9d}  cells={mm.get('n_cells')}  mean_acc={mm.get('mean_acc_at_cap')}  "
              f"mean_delta={mm.get('mean_delta')}  worse={mm.get('n_cells_significantly_worse')}")
    print("wrote", os.path.join(OUT, "mcq_ladder.json"))


if __name__ == "__main__":
    main()
