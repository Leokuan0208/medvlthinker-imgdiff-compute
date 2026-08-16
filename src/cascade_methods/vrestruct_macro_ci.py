#!/usr/bin/env python3
"""vrestruct_macro_ci.py -- the 8-cell MACRO delta of head-only against always-7B, with its CI.

The headline of the head-only deliverable is a macro-8 number, so it needs an interval. Only the
3 open cells move; the 5 MCQ cells are byte-identical constants in both arms (greedy-7B), so the
macro delta is exactly (1/8) * sum over the 3 open cells of the per-cell accuracy delta, and the
bootstrap resamples WITHIN each open cell independently and averages with equal weight.

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_macro_ci.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import genframe_data as G          # noqa: E402
import vrestruct_lib as V          # noqa: E402
import vrestruct_freehead_eval as FH   # noqa: E402

CELL = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
        "pathvqa_open": "PATH_VQA_open"}


def macro_boot(P, got_a, got_b, nboot=V.NBOOT, seed=V.BOOT_SEED):
    rng = np.random.default_rng(seed)
    tot = np.zeros(nboot)
    per = {}
    for j, ds in enumerate(G.EVAL_DS):
        m = P["ds_index"] == j
        d = (np.asarray(got_a, float)[m] - np.asarray(got_b, float)[m])
        idx = rng.integers(0, len(d), size=(nboot, len(d)))
        bs = d[idx].mean(1)
        per[CELL[ds]] = dict(delta=float(d.mean()), lo=float(np.percentile(bs, 2.5)),
                             hi=float(np.percentile(bs, 97.5)))
        tot += bs / 8.0
    md = sum(per[CELL[ds]]["delta"] for ds in G.EVAL_DS) / 8.0
    lo, hi = float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))
    return dict(macro8_delta=float(md), lo=lo, hi=hi,
                significant=bool(lo > 0 or hi < 0), per_cell=per,
                open3_macro_delta=float(md * 8.0 / 3.0),
                _note="only the 3 open cells move; the 5 MCQ cells are identical constants in both "
                      "arms, so they contribute no resampling noise and a fixed 0 to the delta.")


def main():
    P = V.load_pool()
    g = P["greedy_ok"].astype(int)
    inc_rank = V.rank_rows(P["inc"])
    out = {}

    arms = {}
    Lref = V.head_logits(P)
    arms["head_only_deployed_cache_fullres_TF"] = V.head_rank_slots(P, Lref, range(8))
    for cap, which in (("cap320", "ar"), ("fullres", "ar")):
        try:
            Xa, sr, _ = FH.load_free(cap, which, P)
            arms[f"head_only_captured_{cap}_{which}"] = FH.head_rank_from(Xa, sr)
        except Exception as e:
            print(f"  [skip] {cap}/{which}: {e}")
    # the fused arm, for the banked comparison
    arms["fused_deployed_cache"] = None

    for name, HR in arms.items():
        if HR is None:
            S = inc_rank + V.rank_rows(V.head_rank_slots(P, Lref, range(8)))
        else:
            S = HR
        r = V.evaluate(P, V.picks_of(S), name)
        out[name] = dict(
            macro8_accuracy=float(np.mean(
                [0.542656, 0.825359, 0.780876, 0.840869, 0.2615]
                + [r["judge"]["per_ds"][d]["acc"] for d in G.EVAL_DS])),
            vs_always_7b_greedy_judge=macro_boot(P, r["judge"]["got"], g),
            per_cell_acc_judge={CELL[d]: r["judge"]["per_ds"][d]["acc"] for d in G.EVAL_DS},
            per_cell_acc_em={CELL[d]: r["em"]["per_ds"][d]["acc"] for d in G.EVAL_DS},
            acc_judge=r["judge"]["acc"], acc_em=r["em"]["acc"],
            sel_eff_judge=r["judge"]["sel_eff"])
        b = out[name]["vs_always_7b_greedy_judge"]
        out[name]["guardrail_clean_vs_greedy"] = bool(all(
            v["delta"] >= 0 for v in b["per_cell"].values()))
        print(f"  {name:38s} macro8 {out[name]['macro8_accuracy']:.6f}  "
              f"delta {b['macro8_delta']:+.6f} [{b['lo']:+.6f},{b['hi']:+.6f}] "
              f"sig={b['significant']} guardrail={out[name]['guardrail_clean_vs_greedy']}")

    # head-only vs fused, macro-8, both currencies
    Sh = arms.get("head_only_captured_cap320_ar")
    if Sh is not None:
        rh = V.evaluate(P, V.picks_of(Sh))
        rf = V.evaluate(P, V.picks_of(inc_rank + V.rank_rows(V.head_rank_slots(P, Lref, range(8)))))
        out["_head_only_captured_vs_fused"] = dict(
            judge=macro_boot(P, rh["judge"]["got"], rf["judge"]["got"]),
            em=macro_boot(P, rh["em"]["got"], rf["em"]["got"]))
        b = out["_head_only_captured_vs_fused"]["judge"]
        print(f"  head-only(captured) vs fused: macro8 {b['macro8_delta']:+.6f} "
              f"[{b['lo']:+.6f},{b['hi']:+.6f}] sig={b['significant']}")

    json.dump(dict(title="8-cell macro deltas with CIs, always-7B greedy as the baseline",
                   date="2026-08-16", nboot=V.NBOOT, seed=V.BOOT_SEED,
                   mcq_cells_held_at_greedy_7b={
                       "PMC_VQA": 0.542656, "SLAKE_closed": 0.825359,
                       "VQA_RAD_closed": 0.780876, "PATH_VQA_closed": 0.840869,
                       "MedXpertQA-MM": 0.2615},
                   baseline_macro8=0.597087, arms=out),
              open(os.path.join(V.PARTS, "macro_ci.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(V.PARTS, "macro_ci.json"))


if __name__ == "__main__":
    main()
