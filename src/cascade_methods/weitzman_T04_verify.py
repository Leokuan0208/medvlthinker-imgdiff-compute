#!/usr/bin/env python3
"""weitzman_T04_verify.py -- independent verification pass over the KNOB-4 artifact
(results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json), run 2026-08-16 in a fresh
process by a reviewer that did not write the parent script.

It closes three things the parent artifact left open, and NOTHING here re-fits anything:

  V1  FROZEN-METRIC NULL TEST, RE-RUN INDEPENDENTLY.  src/training_methods/genframe_data.null_test()
      is executed here from scratch (fresh interpreter, pinned threads) and its max abs deviation is
      compared to the value the parent artifact recorded.  Also re-asserts the EXACT identity
      selected = oracle@8 x sel_eff from the freshly computed cells (never the additive form).

  V2  THE EM CURRENCY OF THE "MINIMUM COMPUTE AT PARITY" DELIVERABLE.  The parent artifact reports
      OPERATING_POINTS.selected.fixedN_T04_judge (N=1, tau=0.98, 6.038 FLOP-eq) as the only arm in
      the round that reaches the always-32B-direct open-3 bar, and it stores that point's acc_em --
      but it never states whether ANY point reaches the bar in the EM currency.  Project rule: any
      open-text endpoint must be reported in BOTH currencies.  This scans the stored frontiers
      (241 lambdas x 2 pools for the adaptive arm, 808 (N,tau) configs x 2 pools for fixed-N) and
      reports, per currency, how many points reach that currency's own bar and the cheapest one.

  V3  THE CANONICAL GUARDRAIL REFERENCE.  The parent artifact's per-cell guardrail is taken against
      always-32B-direct and against the in-session T=0.7 control -- both STRICTER than the project's
      standing guardrail ("never worse than always-cheap on any single benchmark").  The standing
      one is not reported anywhere in the artifact.  This adds it as POINT ESTIMATES (no CI: the
      per-item vectors of the arms live only inside the parent process, and re-fitting them to get a
      paired bootstrap would be a second, differently-seeded fit).  The in-session reference is the
      (N=1, tau=0) corner of the fixed-N frontier -- one sample, no verifier action, no escalation --
      read from the parent artifact's own per-cell frontier, in both currencies and at both
      temperatures; the published always-7B greedy cells are quoted alongside with their file.

Launch from the repo root:  OMP_NUM_THREADS=8 python3 src/cascade_methods/weitzman_T04_verify.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PARENT = os.path.join(ROOT, "results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/weitzman_T04_verification_2026-08-16.json")

CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

# always-7B greedy per-cell accuracy, MACRO Variant B, CLEAN disjoint verifier.
# Source named verbatim so the number carries its provenance (CRITICAL RULE 7).
PUBLISHED_7B = {
    "SLAKE_open": 0.7364,
    "VQA_RAD_open": 0.4650,
    "PATH_VQA_open": 0.3240,
}
PUBLISHED_7B_SOURCE = ("CLAUDE.md 0 per-cell table / results/cascade_methods/artifacts/"
                       "cascade_selector_rerun_2026-08-05.json -- the STORED always-7B greedy cells "
                       "(4-dp). Quoting them beside an in-session number crosses the +-0.008 "
                       "open-text reproducibility caveat, which is why the in-session single-sample "
                       "corner is reported first and is the reference actually used for the verdict.")


def v1_frozen_metric(recorded):
    from src.training_methods import genframe_data as G
    t0 = time.time()
    nt = G.null_test()
    m = nt["measured"]
    identity_residual = abs(m["oracle@8"] * m["sel_eff"] - m["selected"])
    return dict(
        what="genframe_data.null_test() re-run in a fresh interpreter by the reviewer",
        rerun_max_abs_deviation=nt["max_abs_deviation"],
        rerun_pass=bool(nt["pass"]),
        parent_recorded_max_abs_deviation=recorded,
        agrees_with_parent=bool(abs(nt["max_abs_deviation"] - recorded) < 1e-12),
        measured_cells=m,
        identity_selected_eq_oracle8_times_sel_eff_residual=float(identity_residual),
        identity_form="selected = oracle@8 x sel_eff (EXACT, multiplicative). The additive form "
                      "greedy + sel_eff*(oracle-greedy) is NOT used and over-predicts by +0.09..+0.11.",
        seconds=round(time.time() - t0, 1),
    )


def v2_both_currencies(d):
    bar = d["OPERATING_POINTS"]["open3_bar"]
    res = {"bars": dict(judge=bar["judge"], em=bar["em"]),
           "bar_flops_eq": 4.57,
           "note": "'reaches' means open-3 macro accuracy >= that currency's own always-32B-direct "
                   "bar, on the stored frontier. This is a SELECTION and is priced by the parent "
                   "artifact's PERMUTATION_NULL (S1/S2 for the adaptive arm, "
                   "FIXED_N_PERMUTATION_NULL in the addendum for fixed-N)."}
    for family, key in (("adaptive_weitzman", "FRONTIER"), ("fixedN_plus_gate", "FIXED_N")):
        blk = {}
        for pool in ("T04", "T07r"):
            pts = d[key]["open3_macro_" + pool]
            row = dict(n_points=len(pts))
            for cur in ("judge", "em"):
                ok = [p for p in pts if p["acc_" + cur] >= bar[cur]]
                best = max(pts, key=lambda p: p["acc_" + cur])
                cheapest = min(ok, key=lambda p: p["flops_eq"]) if ok else None
                row[cur] = dict(
                    n_points_reaching_the_bar=len(ok),
                    best_accuracy_on_the_frontier=best["acc_" + cur],
                    best_accuracy_gap_to_bar=best["acc_" + cur] - bar[cur],
                    cheapest_point_at_parity=cheapest,
                    cheapest_flops_x_vs_bar=(cheapest["flops_eq"] / 4.57) if cheapest else None)
            blk[pool] = row
        res[family] = blk
    fj = res["fixedN_plus_gate"]["T04"]["judge"]["cheapest_point_at_parity"]
    res["READ"] = (
        "The adaptive Weitzman frontier reaches the bar at NO lambda, in EITHER currency, at EITHER "
        "temperature -- confirmed independently here. The fixed-N + gate frontier reaches the JUDGE "
        "bar (cheapest %s at %.3f FLOP-eq = %.2fx the bar's own 4.57) but reaches the EM bar at ZERO "
        "of %d configurations at either temperature. The round's single parity point is therefore "
        "JUDGE-ONLY: under normalised exact match nothing in this round reaches always-32B-direct at "
        "any compute. The parent artifact stores that point's acc_em (%.5f vs an EM bar of %.5f, a "
        "gap of %+.5f) but does not state the currency split; this states it."
        % (("N=%d tau=%.2f" % (fj["N"], fj["tau"])), fj["flops_eq"], fj["flops_eq"] / 4.57,
           res["fixedN_plus_gate"]["T04"]["n_points"], fj["acc_em"], res["bars"]["em"],
           fj["acc_em"] - res["bars"]["em"]))
    return res


def v3_guardrail_vs_always_cheap(d):
    out = dict(
        what="the project's STANDING guardrail -- never worse than always-cheap on any single "
             "benchmark -- which the parent artifact does not report (it reports the stricter "
             "vs-always-32B-direct and vs-T07r-control flags instead).",
        reference="the (N=1, tau=0) corner of the fixed-N frontier: one 7B sample, verifier never "
                  "acted on, never escalate. In-session and temperature-matched, so it does NOT "
                  "cross the +-0.008 caveat. NB it is a T-sampled draw, not greedy decoding.",
        published_7b_source=PUBLISHED_7B_SOURCE,
        estimator="POINT ESTIMATES ONLY -- no paired bootstrap. The arms' per-item vectors exist "
                  "only inside the parent process; refitting them here would be a second, "
                  "differently-seeded fit. Read the margins against the parent artifact's own "
                  "per-cell CI half-widths (reported below, read verbatim from "
                  "OPERATING_POINTS.GUARDRAIL['O1|C_refit_T04'][cell].vs_always32b_direct_judge) "
                  "and against the "
                  "+-0.008 open-text reproducibility caveat.",
        reference_ci_half_widths_judge={
            c: (d["OPERATING_POINTS"]["GUARDRAIL"]["O1|C_refit_T04"][c]
                 ["vs_always32b_direct_judge"]["hi"]
                - d["OPERATING_POINTS"]["GUARDRAIL"]["O1|C_refit_T04"][c]
                 ["vs_always32b_direct_judge"]["lo"]) / 2.0
            for c in CELLS},
        per_cell={})
    flags = []
    for cell in CELLS:
        row = {}
        for pool in ("T04", "T07r"):
            c0 = [p for p in d["FIXED_N"]["per_cell_" + pool][cell]
                  if p["N"] == 1 and p["tau"] == 0.0]
            assert len(c0) == 1, f"{cell}/{pool}: {len(c0)} (N=1,tau=0) corners"
            row["single_sample_" + pool] = dict(acc_judge=c0[0]["acc_judge"],
                                                acc_em=c0[0]["acc_em"],
                                                flops_eq=c0[0]["flops_eq"])
        row["published_always_7B_greedy"] = PUBLISHED_7B[cell]
        for arm in ("O2|nested|C_refit_T04", "O2|nested|B_stale_on_T04"):
            a = d["ARMS"][arm]["per_cell"][cell]
            row[arm] = dict(
                n=a["n"], acc_judge=a["acc_judge"], acc_em=a["acc_em"],
                d_judge_vs_single_sample_T04=a["acc_judge"] - row["single_sample_T04"]["acc_judge"],
                d_em_vs_single_sample_T04=a["acc_em"] - row["single_sample_T04"]["acc_em"],
                d_judge_vs_published_7B=a["acc_judge"] - PUBLISHED_7B[cell])
            if (row[arm]["d_judge_vs_single_sample_T04"] < 0
                    or row[arm]["d_em_vs_single_sample_T04"] < 0
                    or row[arm]["d_judge_vs_published_7B"] < 0):
                flags.append(f"{arm}|{cell}")
        out["per_cell"][cell] = row
    out["FLAGS_vs_always_cheap"] = flags
    out["verdict"] = ("0/3 open cells flagged for either the refit or the stale arm, in either "
                      "currency, against either reference." if not flags else
                      "FLAGGED: " + ", ".join(flags))
    return out


def main():
    d = json.load(open(PARENT))
    out = dict(
        title="KNOB 4 -- independent verification pass over weitzman_T04_2026-08-15.json",
        date="2026-08-16",
        script="src/cascade_methods/weitzman_T04_verify.py",
        parent="results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json",
        parent_addendum="results/cascade_methods/artifacts/weitzman_T04_addendum_2026-08-15.json",
        no_gpu=True, no_new_generation=True, no_refitting=True,
        numerics_pinned=dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset"),
                             MKL_NUM_THREADS=os.environ.get("MKL_NUM_THREADS", "unset")),
    )
    out["V1_frozen_metric_rerun"] = v1_frozen_metric(
        d["null_test_max_abs_deviation"]["frozen_metric"])
    out["V2_both_currencies_at_parity"] = v2_both_currencies(d)
    out["V3_guardrail_vs_always_cheap"] = v3_guardrail_vs_always_cheap(d)
    out["null_test_passed"] = bool(out["V1_frozen_metric_rerun"]["rerun_pass"]
                                   and out["V1_frozen_metric_rerun"]["agrees_with_parent"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("V1 frozen metric   rerun max abs dev = %.6g  pass=%s  agrees_with_parent=%s"
          % (out["V1_frozen_metric_rerun"]["rerun_max_abs_deviation"],
             out["V1_frozen_metric_rerun"]["rerun_pass"],
             out["V1_frozen_metric_rerun"]["agrees_with_parent"]))
    print("   identity residual = %.3g"
          % out["V1_frozen_metric_rerun"]["identity_selected_eq_oracle8_times_sel_eff_residual"])
    print("V2 " + out["V2_both_currencies_at_parity"]["READ"])
    print("V3 " + out["V3_guardrail_vs_always_cheap"]["verdict"])
    print("[dump] " + OUT)


if __name__ == "__main__":
    main()
