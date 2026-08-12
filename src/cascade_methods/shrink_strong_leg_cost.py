#!/usr/bin/env python3
"""
shrink_strong_leg_cost.py -- ATTACK 3 part 4: what does SHRINKING THE STRONG LEG do to the
PRIMARY objective, macro FLOP-eq as a ratio of one always-32B-direct pass?

Method.  artifacts/cost_floor_2026-08-10.json decomposes every published arm, per cell, into
(n_gen7, n_ver7, n_32b) -- the expected number of 7B generation passes, 7B verifier passes and
STRONG-LEG passes per query.  That decomposition was asserted there to reproduce every published
per-cell cost to 4.71e-4.  So the as-charged cost of a cell is

    cost_cell(R_S) = n_gen7 * 1.0 + n_ver7 * 1.0 + n_32b * R_S

and its ratio to always-32B-direct (which costs exactly R32) is cost_cell(R_S) / R32.  Swapping
the strong leg changes R_S and NOTHING ELSE in that expression -- the escalation rates, the
verifier and the cheap leg are untouched.  That makes the compute consequence of a smaller
strong leg exact arithmetic on published quantities, not a new estimate.

NULL TEST: with R_S = R32 the table must reproduce the published macro costs (accuracy-max
1.740x, compute-lean 1.46x).  Asserted, not assumed.

    python3 src/cascade_methods/shrink_strong_leg_cost.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(A, "_shrink_parts/strong_leg_cost.json")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

# Published macro costs to null-test against (artifacts/cascade_selector_rerun_2026-08-05.json
# and CLAUDE.md section 0's canonical table).
PUBLISHED = {"method_accuracy_max_veto": 1.740, "method_compute_lean": 1.46,
             "always_32b_direct": 1.000, "always_7b": 0.219}


def main():
    cf = json.load(open(os.path.join(A, "cost_floor_2026-08-10.json")))
    foot = json.load(open(os.path.join(A, "_shrink_parts/footprint.json")))
    table = cf["arm_decomposition"]["table"]

    R = {k: v["R_vs_lingshu_7b"] for k, v in foot["per_pass_flops"].items()}

    # The two conventions the round demands, side by side and never mixed.
    CONV = {
        "as_charged_R32_4p57": dict(
            R32=4.57,
            # AS-CHARGED is a convention of LITERALS: the project pins Lingshu-7B at 1.0 and
            # Lingshu-32B at 4.57.  A candidate with no published literal is charged by its
            # DERIVED ratio to the 7B, because the 7B is the convention's own unit.
            legs={"lingshu_32b": 4.57, "lingshu_32b_nf4": 4.57, "qwen25vl_32b_awq": 4.57,
                  "lingshu_i8b": R["lingshu_i8b"], "lingshu_7b": 1.0}),
        "derived_R32_3p816": dict(
            R32=R["lingshu_32b"],
            legs={"lingshu_32b": R["lingshu_32b"], "lingshu_32b_nf4": R["lingshu_32b"],
                  "qwen25vl_32b_awq": R["lingshu_32b"], "lingshu_i8b": R["lingshu_i8b"],
                  "lingshu_7b": 1.0}),
    }

    def macro_cost(arm, R_S, R32):
        per = {}
        for c in CELLS:
            d = table[c][arm]
            per[c] = (d["n_gen7"] * 1.0 + d["n_ver7"] * 1.0 + d["n_32b"] * R_S) / R32
        return sum(per.values()) / len(CELLS), per

    out = dict(
        title="ATTACK 3 part 4 -- macro FLOP-eq when the STRONG LEG is swapped, both R32 "
              "conventions.  The PRIMARY objective of the round.",
        date="2026-08-12",
        cpu_only=True,
        method=__doc__.strip(),
        decomposition_source="artifacts/cost_floor_2026-08-10.json:arm_decomposition "
                             "(max_abs_dev_vs_published_per_cell_flops = %s)"
                             % cf["arm_decomposition"]["max_abs_dev_vs_published_per_cell_flops"],
        strong_leg_ratios_used=CONV,
    )

    # ---- NULL TEST ---------------------------------------------------------------------
    null = {}
    for arm, want in PUBLISHED.items():
        got, _ = macro_cost(arm, CONV["as_charged_R32_4p57"]["legs"]["lingshu_32b"], 4.57)
        null[arm] = dict(rebuilt=round(got, 4), published=want,
                         abs_dev=round(abs(got - want), 4))
    mx = max(v["abs_dev"] for v in null.values())
    out["null_test_N5_reproduces_published_costs"] = dict(
        per_arm=null, max_abs_dev=mx,
        threshold=0.005,
        verdict="PASS" if mx <= 0.005 else "FAIL",
        note="the published values are quoted to 3-4 significant figures, so a deviation at "
             "the 1e-3 level is rounding, not disagreement.")

    # ---- the swap table ----------------------------------------------------------------
    arms = ["method_accuracy_max_veto", "method_compute_lean"]
    swap = {}
    for cname, conv in CONV.items():
        swap[cname] = {}
        for leg, R_S in conv["legs"].items():
            if leg == "lingshu_7b":
                continue
            row = {}
            for arm in arms:
                m, per = macro_cost(arm, R_S, conv["R32"])
                row[arm] = dict(macro_x_of_always_32b_direct=round(m, 4),
                                per_cell={k: round(v, 4) for k, v in per.items()})
            # the degenerate "no cascade at all" policy: run the candidate on everything
            row["always_this_leg_alone"] = dict(
                macro_x_of_always_32b_direct=round(R_S / conv["R32"], 4))
            swap[cname][leg] = row

    out["macro_cost_when_strong_leg_is_swapped"] = swap

    ac = swap["as_charged_R32_4p57"]
    de = swap["derived_R32_3p816"]
    out["headline"] = dict(
        quantised_32b=(
            "NO CHANGE, EXACTLY.  An NF4 or AWQ Lingshu-32B has the identical logical parameter "
            "count (33,452,718,336, read from both checkpoints' safetensors headers) and "
            "therefore the identical MAC count, so R_S is unchanged and accuracy-max stays at "
            "%.4fx (as-charged) / %.4fx (derived).  Quantisation moves the SECONDARY objective "
            "(footprint) and nothing else."
            % (ac["lingshu_32b_nf4"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
               de["lingshu_32b_nf4"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"])),
        i8b_strong_leg=(
            "accuracy-max with a Lingshu-I-8B strong leg would cost %.4fx (as-charged) / %.4fx "
            "(derived) instead of %.4fx / %.4fx -- BELOW the baseline on both conventions.  "
            "always-Lingshu-I-8B alone costs %.4fx / %.4fx.  These are COST figures only; "
            "whether either holds the accuracy constraint is a separate question and is "
            "answered in quant_acc_paired.json (it does not, on the evidence available)."
            % (ac["lingshu_i8b"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
               de["lingshu_i8b"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
               ac["lingshu_32b"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
               de["lingshu_32b"]["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
               ac["lingshu_i8b"]["always_this_leg_alone"]["macro_x_of_always_32b_direct"],
               de["lingshu_i8b"]["always_this_leg_alone"]["macro_x_of_always_32b_direct"])),
        caveat="Macro-weighted COST must never be paired with a sample-weighted accuracy, and "
               "these are macro costs.  They also inherit the arm_decomposition's own caveat: "
               "it is DERIVED from published per-cell costs, not re-measured from a run.",
    )

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(dict(null=out["null_test_N5_reproduces_published_costs"],
                          headline=out["headline"]), indent=1))
    for cname in CONV:
        print("\n--- %s ---" % cname)
        for leg, row in swap[cname].items():
            print("  %-20s acc-max %.4fx  lean %.4fx  alone %.4fx" % (
                leg, row["method_accuracy_max_veto"]["macro_x_of_always_32b_direct"],
                row["method_compute_lean"]["macro_x_of_always_32b_direct"],
                row["always_this_leg_alone"]["macro_x_of_always_32b_direct"]))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
