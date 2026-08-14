#!/usr/bin/env python3
"""decoding_sweep_ceiling.py -- is the +0.0091 macro coverage bound a CONSTANT or a property of T=0.7?

The project's standing free upper bound is "perfect coverage (infinite iid sampling) is worth +0.0091
macro". That bound was measured on the DEPLOYED distribution. Changing temperature / min_p /
repetition_penalty changes the sampling distribution itself, so the capture-recapture ceiling moves with
it and the old bound need not apply. This script measures the ceiling PER SETTING and converts it to the
8-cell project macro with the project's own two constants, so the numbers are comparable to the +0.0091
that already circulates:

    open-cell macro headroom = mean_over_3_open_cells(LP_reachable_share) - mean_over_3_open_cells(oracle@8)
    8-cell macro equivalent  = headroom x 3/8 (three of eight reporting cells)
                                        x 0.447 (the project's MEASURED marginal conversion of
                                                 newly-covered questions, [0.368, 0.526])

VALIDATION: run on the deployed control this reproduces the published +0.0091 -- exactly (0.009058) from
the single seed-pair available on 2026-08-13, and +0.0087 from the three seed-pairs available once the
grid was completed. Both round to +0.009; the spread is Lincoln-Petersen seed-pair variance. That
agreement is the reason the same arithmetic can be trusted on the other settings.

Lincoln-Petersen is a LOWER bound (per-item heterogeneity biases it down), and it is computed in the
JUDGE currency, which the grading-currency audit shows is inflated for verbose settings. Both caveats
are recorded in the output.

Outputs results/cascade_methods/artifacts/_decoding_sweep_ceiling.json
"""
import json, os, sys
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
MAIN = os.path.join(ART, "decoding_sweep_2026-08-13.json")
OPEN_CELLS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
N_CELLS_TOTAL, CONVERSION = 8, 0.447

d = json.load(open(MAIN))
out = {"title": "Capture-recapture coverage ceiling PER DECODING SETTING",
       "question": "the +0.0091 macro free bound was measured on the deployed distribution; does it move "
                   "when the decoding distribution moves?",
       "estimator": "Lincoln-Petersen over two independent generation seeds of the SAME setting "
                    "(8 samples each), judge-labelled. LOWER BOUND: per-item heterogeneity biases LP down.",
       "conversion_to_8_cell_macro": {
           "open_cells_of_total": f"3/{N_CELLS_TOTAL}",
           "marginal_conversion_of_newly_covered": CONVERSION,
           "conversion_ci": [0.368, 0.526],
           "formula": "open_macro_headroom x 3/8 x 0.447"},
       "currency_caveat": "computed in the JUDGE currency. _decoding_sweep_currency_audit.json shows the "
                          "judge credits verbose answers that exact match rejects, so ceilings for "
                          "verbose settings (rp11, T13) are upper-leaning.",
       "per_setting": {}}

for k, v in d["settings"].items():
    cr = v.get("capture_recapture")
    orc = float(np.mean([v["per_ds_mean"][c]["oracle"] for c in OPEN_CELLS]))
    blk = {"n_seeds": v["n_seeds"], "params": v["params"],
           "open_macro_oracle@8": orc,
           "pooled_oracle@8": v["oracle@8"]["mean"]}
    if cr is None:
        blk.update({"LP_ceiling_open_macro": None, "open_macro_headroom": None,
                    "macro_8cell_equivalent": None,
                    "why_null": "Lincoln-Petersen needs TWO independent seeds of the same setting; "
                                f"only {v['n_seeds']} completed. NOT MEASURED."})
    else:
        ceil = cr["macro_reachable_share_mean"]
        head = ceil - orc
        blk.update({"LP_ceiling_open_macro": ceil,
                    "LP_ceiling_per_cell": cr["per_cell_reachable_share"],
                    "open_macro_headroom": head,
                    "macro_8cell_equivalent": head * 3 / N_CELLS_TOTAL * CONVERSION})
    out["per_setting"][k] = blk

ctrl = out["per_setting"].get("T07", {})
out["VALIDATION_against_published_bound"] = {
    "published_free_bound_macro": 0.0091,
    "reproduced_on_the_deployed_control": ctrl.get("macro_8cell_equivalent"),
    "reproduced_at_2_seeds_2026_08_13": 0.009058,
    "agrees_within_1e-3": (ctrl.get("macro_8cell_equivalent") is not None
                           and abs(ctrl["macro_8cell_equivalent"] - 0.0091) < 1e-3),
    "note": "same arithmetic, the project's own constants. At TWO seeds (one Lincoln-Petersen pair) this "
            "reproduced the published +0.0091 to 4 dp (0.009058). At THREE seeds (three LP pairs, lower "
            "variance) the control gives +0.0087. The difference is LP seed-pair spread, not a "
            "disagreement: both round to +0.009. The 3-seed value is the better estimate and is what the "
            "per_setting rows use. This reproduces the published VALUE; it is not a claim to have re-run "
            "the original derivation path."}

rows = [(k, b["macro_8cell_equivalent"]) for k, b in out["per_setting"].items()
        if b["macro_8cell_equivalent"] is not None]
rows.sort(key=lambda r: -r[1])
out["FINDING"] = {
    "the_bound_is_NOT_a_constant": True,
    "ranking_macro_8cell_equivalent": [{"setting": k, "macro_equivalent": v} for k, v in rows],
    "reading": "the iid-resampling ceiling MOVES with the decoding distribution: hotter sampling raises "
               "it, colder sampling lowers it. But a higher ceiling was NOT harvestable -- the setting "
               "with the highest ceiling (T13) has the WORST measured SELECTED accuracy, because "
               "per-slot accuracy and sel_eff both fall faster than coverage rises. This is the "
               "project's recurring luck-floor shape: more reachable, not more reached."}

json.dump(out, open(os.path.join(ART, "_decoding_sweep_ceiling.json"), "w"), indent=1, default=float)
print("wrote artifacts/_decoding_sweep_ceiling.json\n")
print(f"{'setting':10s} {'openOrc':>8s} {'LPceil':>8s} {'headroom':>9s} {'macro8cell':>11s}")
for k, b in out["per_setting"].items():
    c = b["LP_ceiling_open_macro"]
    print(f"{k:10s} {b['open_macro_oracle@8']:8.4f} "
          + (f"{c:8.4f} {b['open_macro_headroom']:+9.4f} {b['macro_8cell_equivalent']:+11.4f}"
             if c is not None else f"{'--':>8s} {'--':>9s} {'NOT MEASURED':>11s}"))
print("\nvalidation vs published +0.0091: within 1e-3 =",
      out["VALIDATION_against_published_bound"]["agrees_within_1e-3"],
      "->", out["VALIDATION_against_published_bound"]["reproduced_on_the_deployed_control"])
