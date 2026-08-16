#!/usr/bin/env python3
"""verifier_hparams_recost.py -- KNOB 3: the macro cost column is BLIND to this knob, and here is
what it costs to fix that.

THE FINDING THIS SCRIPT PRICES.  The project's open-arm cost model charges one verifier forward
exactly as much as one 7B generation:

    src/cascade_methods/pandora_controller.py:49-52
        GEN7 = (347.0, 45.8, 1.0)
        VER7 = (175.0, 25.3, 1.0)
        C_CHEAP_F = GEN7[2] + VER7[2]     # 2.0 FLOP-eq per cheap draw (generate + verify)

so every open cell's FLOP-eq is  meanN * 2.0 + esc * 4.57  and moving the verifier's resolution
changes NOTHING in that column.  This round measured, on all 8,965 scored triples per rung, that
one verifier forward at the DEPLOYED 1,003,520 is not 1.0 but ~1.88 generator-forwards -- so the
deployed cheap draw is charged 2.0 and measures ~2.88, an under-charge of ~44%.

This script recomputes each arm's per-cell and macro FLOP-eq with that ONE constant replaced by
the measured ratio, and reports the as-charged number beside it.  Nothing else in the cost model
is touched: the 4.57 strong-leg constant, meanN, esc and the macro weighting are the existing
code's own outputs, read from the per-arm summaries.

    python3 src/cascade_methods/verifier_hparams_recost.py
"""
import json
import os
import sys

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
CONTROL_PX = 1003520
STRONG_F = 4.57          # pandora_controller.C_STRONG_F, unchanged
AS_CHARGED_CHEAP = 2.0   # pandora_controller.C_CHEAP_F, the constant being replaced
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]


def main():
    cost = json.load(open(os.path.join(PARTS, "cost.json")))
    gen_f = float(cost["_meta"]["generator_term_held_fixed"]["flops_per_candidate"])
    rungs = sorted(int(k) for k in cost["by_max_pixels"])

    out = {"_what": "macro FLOP-eq with the verifier charged at its MEASURED cost instead of the "
                    "model's flat 1.0 generator-equivalents.",
           "_the_constant_being_replaced": {
               "code": "src/cascade_methods/pandora_controller.py:50-52",
               "VER7_flop_eq_as_charged": 1.0,
               "C_CHEAP_F_as_charged": AS_CHARGED_CHEAP,
               "_meaning": "one verifier forward is charged the same as one 7B generation"},
           "_generator_forward_flops_cap320": gen_f,
           "by_max_pixels": {}}

    for px in rungs:
        vf = float(cost["by_max_pixels"][str(px)]["flops_per_verifier_forward"])
        ratio = vf / gen_f
        cheap_honest = 1.0 + ratio
        row = {"max_pixels": px,
               "verifier_forward_flops": vf,
               "verifier_forward_in_generator_equivalents": ratio,
               "C_CHEAP_F_honest": cheap_honest,
               "undercharge_factor_of_the_cheap_draw": cheap_honest / AS_CHARGED_CHEAP}
        name = "disjoint" if False else f"verifhp_px{px}"
        p = os.path.join(PARTS, f"summary_{name}.json")
        if os.path.exists(p):
            s = json.load(open(p))
            det = s["open_cell_detail"]
            per_cell = {}
            for m, esck in [("method_compute_lean", "esc"),
                            ("method_accuracy_max_veto", "am2_esc")]:
                cells = {}
                for k in OPEN:
                    v = det[k]
                    e = v[esck] if v.get(esck) is not None else v["esc"]
                    cells[k] = {
                        "meanN": v["meanN"], "esc_used": e,
                        "flops_as_charged": v["meanN"] * AS_CHARGED_CHEAP + e * STRONG_F,
                        "flops_honest_verifier": v["meanN"] * cheap_honest + e * STRONG_F}
                mcq_as = {k: s["cost_macro"] for k in []}   # MCQ cells are unaffected
                # macro = mean over 8 cells; the 5 MCQ cells are identical in both costings, so
                # recover their summed contribution from the arm's own as-charged macro.
                macro_as = s["cost_macro"][m]["flops"]
                open_as = sum(c["flops_as_charged"] for c in cells.values())
                open_hon = sum(c["flops_honest_verifier"] for c in cells.values())
                mcq_sum = macro_as * 8.0 - open_as
                per_cell[m] = {
                    "open_cells": cells,
                    "mcq_cells_summed_flops_unchanged": mcq_sum,
                    "macro_flops_as_charged": macro_as,
                    "macro_flops_honest_verifier": (mcq_sum + open_hon) / 8.0,
                    "macro_flops_ratio_honest_over_as_charged":
                        ((mcq_sum + open_hon) / 8.0) / macro_as,
                }
                del mcq_as
            row["macro"] = per_cell
        out["by_max_pixels"][str(px)] = row

    # ---- everything relative to the DEPLOYED rung -------------------------------------
    b = out["by_max_pixels"].get(str(CONTROL_PX))
    if b and "macro" in b:
        for k, r in out["by_max_pixels"].items():
            if "macro" not in r:
                continue
            r["vs_deployed"] = {
                "verifier_forward_flops_ratio":
                    r["verifier_forward_flops"] / b["verifier_forward_flops"],
                "macro_flops_honest_ratio": {
                    m: r["macro"][m]["macro_flops_honest_verifier"] /
                       b["macro"][m]["macro_flops_honest_verifier"] for m in r["macro"]},
                "macro_flops_as_charged_ratio": {
                    m: r["macro"][m]["macro_flops_as_charged"] /
                       b["macro"][m]["macro_flops_as_charged"] for m in r["macro"]},
                "_read": "the as-charged ratio moves ONLY because escalation moved; the honest "
                         "ratio also carries the verifier's own resolution saving.",
            }
    json.dump(out, open(os.path.join(PARTS, "recost.json"), "w"), indent=1, default=float)
    for k, r in out["by_max_pixels"].items():
        if "macro" not in r:
            print(f"  px{k:>9}  (no macro arm yet)  ver/gen {r['verifier_forward_in_generator_equivalents']:.3f}")
            continue
        m = r["macro"]["method_accuracy_max_veto"]
        vs = r.get("vs_deployed", {})
        print(f"  px{k:>9}  ver/gen {r['verifier_forward_in_generator_equivalents']:.3f}  "
              f"cheap_draw {r['C_CHEAP_F_honest']:.3f} (charged 2.000)  "
              f"macro FLOPeq as-charged {m['macro_flops_as_charged']:.3f} / honest "
              f"{m['macro_flops_honest_verifier']:.3f}  "
              f"honest-vs-deployed {vs.get('macro_flops_honest_ratio', {}).get('method_accuracy_max_veto', float('nan')):.4f}")
    print(f"\nwrote {PARTS}/recost.json")


if __name__ == "__main__":
    main()
