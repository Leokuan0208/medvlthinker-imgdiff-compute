#!/usr/bin/env python3
"""verifier_hparams_macro.py -- KNOB 3: re-run the WHOLE cascade end-to-end for each rung of the
verifier scoring-resolution ladder and re-report the canonical 8-cell MACRO.

WHY THE MACRO CANNOT BE READ OFF sel_eff.  Escalation to the 32B is driven by the SELECTOR'S OWN
CONFIDENCE -- max(scores) over the pool -- so changing the verifier's scoring resolution changes
BOTH which candidate is picked AND which questions escalate.  The arm has to be re-run.  This
driver changes exactly one thing: the directory the open-text per-candidate scores are read from.
Every mechanic (margin cascade, F1 slice router, F8 certified veto, Pandora Weitzman draw, F10
L2D rejector, 5-fold cross-fitting, cost constants, macro weighting, bootstrap) is the existing
code called unmodified, via cascade_selector_rerun.run_source.

THE CONTROL IS THE IN-SESSION 1,003,520 ARM, NOT THE PUBLISHED NUMBER.  A batch-1 re-score of
the stored pairs at the deployed resolution already deviates by max 6.03e-2 / mean 5.86e-3 per
candidate (ckpts/openvqa/verifier_hparams/null_test_rescore.json), so the published 0.6575 is
the ANCHOR and `disjoint` is re-run here alongside the ladder to price that nuisance at the
macro level.

    python3 src/cascade_methods/verifier_hparams_macro.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))

import cascade_selector_rerun as CSR   # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
CONTROL_PX = 1003520


def main():
    os.makedirs(PARTS, exist_ok=True)
    CSR.PARTS = PARTS                       # keep the published _selector_rerun_parts untouched
    os.makedirs(CSR.PARTS, exist_ok=True)

    rungs = sorted(int(os.path.basename(d).split("px")[1])
                   for d in glob.glob(os.path.join(ROOT, "ckpts/train/verifhp_px*")))
    # `disjoint` = the STORED deployed dumps = the published anchor arm.
    names = ["disjoint"] + [f"verifhp_px{px}" for px in rungs]
    for px in rungs:
        CSR.SOURCES[f"verifhp_px{px}"] = dict(
            dir=f"ckpts/train/verifhp_px{px}",
            label=f"clean disjoint LoRA verifier, scored IN-SESSION at max_pixels {px}")

    out = {}
    for n in names:
        print(f"\n==================== {n} ====================", flush=True)
        try:
            out[n] = CSR.run_source(n)
        except Exception as e:
            print(f"  ARM FAILED {n}: {type(e).__name__} {e}", flush=True)
            out[n] = {"_error": f"{type(e).__name__}: {e}"}

    # -------- cross-arm table, all against the IN-SESSION control ----------------------
    ctrl = f"verifhp_px{CONTROL_PX}"
    tab = {}
    for n, s in out.items():
        if "_error" in s:
            tab[n] = s
            continue
        row = {"macro_acc": s["macro_acc"],
               "per_cell_acc": s["per_cell_acc"],
               "open_only": s["open_only"],
               "deltas_vs_always_32b_direct": s["deltas"]["method_accuracy_max_veto"]["always_32b_direct"],
               "deltas_vs_always_32b_reasoning": s["deltas"]["method_accuracy_max_veto"]["always_32b_reasoning"],
               "cost_macro_as_charged": s["cost_macro"],
               "cost_macro_honest": s["cost_macro_honest"],
               "ratios_macro": s["ratios_macro"],
               "escalation": s["escalation"].get("macro", s["escalation"]),
               "open_cell_detail": s["open_cell_detail"]}
        tab[n] = row
    if ctrl in tab and "_error" not in tab[ctrl]:
        c = tab[ctrl]
        for n in tab:
            if n == ctrl or "_error" in tab[n]:
                continue
            for m in ["method_compute_lean", "method_accuracy_max_veto",
                      "method_accuracy_max_fusion"]:
                tab[n].setdefault("vs_in_session_control", {})[m] = {
                    "d_macro": tab[n]["macro_acc"][m] - c["macro_acc"][m]}
            tab[n]["vs_in_session_control"]["_per_cell_open"] = {
                k: {m: tab[n]["per_cell_acc"][k][m] - c["per_cell_acc"][k][m]
                    for m in ["method_compute_lean", "method_accuracy_max_veto",
                              "method_accuracy_max_fusion"]}
                for k in ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]}
    # -------- cross-arm PAIRED bootstrap CIs on the macro (shared resample stream) ------
    # CSR.combine() bootstraps every arm's macro on ONE multinomial resample per cell, so
    # arm-vs-arm deltas are paired.  It is the same code that produced the published
    # cascade_selector_rerun_2026-08-05.json CIs; only the arm list differs.
    try:
        S, arm_vs_arm, method_vs_base, changed_cells, ident, names_c = CSR.combine()
        cis = {"arm_vs_arm_macro": arm_vs_arm, "method_vs_baseline_macro": method_vs_base,
               "cells_that_differ_between_arms": changed_cells,
               "n_identical_per_sample_vectors": len(ident.get("identical", [])),
               "n_differing_per_sample_vectors": len(ident.get("differs", [])),
               "_note": "only the 3 open-text cells may differ between arms; every MCQ cell must "
                        "be byte-identical because only the open arm's per-candidate score source "
                        "moved. `cells_that_differ_between_arms` is the assertion, measured."}
        json.dump(cis, open(os.path.join(PARTS, "macro_cis.json"), "w"), indent=1, default=float)
        print(f"wrote {PARTS}/macro_cis.json  (arms: {names_c})")
    except Exception as e:
        print(f"  combine() failed: {type(e).__name__} {e}")

    json.dump(tab, open(os.path.join(PARTS, "macro_table.json"), "w"), indent=1, default=float)
    print(f"\nwrote {PARTS}/macro_table.json")
    for n, r in tab.items():
        if "_error" in r:
            print(f"  {n:>22}  FAILED")
            continue
        print(f"  {n:>22}  macro(am_veto) {r['macro_acc']['method_accuracy_max_veto']:.6f}  "
              f"cost {r['ratios_macro'].get('method_accuracy_max_veto', {})}")


if __name__ == "__main__":
    main()
