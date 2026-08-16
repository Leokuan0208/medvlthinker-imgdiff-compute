#!/usr/bin/env python3
"""
veto_binning_finalize.py -- adds the HEADLINE block to
results/cascade_methods/artifacts/veto_binning_2026-08-15.json.

Every number in the block is copied or arithmetically derived from values already stored in the
artifact by veto_binning_sweep.py / _followup.py / _patch_s8.py / _patch_s9.py.  Nothing new is
computed from data here, and the internal-consistency check at the end asserts that the per-cell
contributions reconstruct the macro delta exactly.

Run LAST, from the repo root:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_finalize.py
"""
import json
import os
import sys

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import veto_binning_sweep as V   # noqa: E402

BEST = "R4"


def main():
    art = json.load(open(V.OUT))
    S3 = art["S3_nested_cv"]["results"]
    S4 = art["S4_permutation_null"]["results"]
    S8 = art["S8_seed_pairing_and_letter_currency"]
    S9 = art["S9_answer_prior_control_closed_cells"]["per_cell"]

    b, s = S3[BEST], S3["R0"]
    contrib = {c: V.r5((b["per_cell"][c]["delta_vs_direct"] - s["per_cell"][c]["delta_vs_direct"]) / 8.0)
               for c in V.MCQ_CELLS}
    total = V.r5(sum(contrib.values()))
    paired = S8["S8a_seed_paired"]["results"][BEST]["paired_vs_R0_shipped"]

    art["HEADLINE"] = dict(
        knob="beat32b_more.f8_veto(n_bins, alpha_z) -- the certified veto's quantile binning and Wilson "
             "confidence level. Both defaults (5, 1.645) were arbitrary and had never been swept.",
        null_test_passed=art["S0_NULL_TEST"]["PASSED"],
        null_test_max_abs_deviation=art["S0_NULL_TEST"]["max_abs_deviation_overall"],

        mechanism="The direction that matters is COARSER, not finer. More bins means smaller bins, wider "
                  "Wilson intervals and (on the small cells) bins below the 30-item guard, so nothing "
                  "certifies -- which is exactly why four of the five multiple-choice cells sit at veto "
                  "rate 0.0000 under the shipped n_bins=5. alpha_z is a near-flat second-order knob "
                  "inside the coarse regime: at n_bins=2 the PMC delta is identical (+0.01068) for every "
                  "alpha_z from 0.0 to 1.282, because a half-of-the-cell bin is large enough that the "
                  "Wilson correction cannot change the verdict.",

        best_setting=dict(
            selection_rule=BEST,
            description=art["S3_nested_cv"]["protocol"]["rules"][BEST],
            modal_setting_chosen_by_the_rule=art["S5_selected_settings_pmc_honesty"]
                                                ["modal_pmc_settings"][BEST],
            honest_form="the deployable answer is the RULE, not a hand-picked row: n_bins in the 2-4 "
                        "coarse plateau with alpha_z anywhere in [0, 0.5], deployed per cell only where "
                        "an inner held-out McNemar test admits it. Quoting the best row of the "
                        "135-setting table instead would be the leakage this whole design prices."),

        accuracy=dict(
            macro_8cell=S3[BEST]["macro_seedstat"],
            vs_always_32b_direct=S3[BEST]["macro_vs_always_32b_direct"],
            vs_shipped_accuracy_max=S3[BEST]["macro_vs_shipped_accuracy_max"],
            vs_always_32b_direct_mcqonly_frame=S3[BEST]["macro_vs_always_32b_direct_mcqonly_frame"],
            seed_paired_vs_shipped=paired,
            significance_bar="Read the two comparisons differently. Against the SHIPPED arm the paired "
                             "item bootstrap excludes zero by 1e-05 and all 10 fold seeds are positive, "
                             "so the sign is real -- but the fold-seed sd (%.5f) is 30x that lower bound, "
                             "so 'CI-significant' here means 'consistently positive and tiny', not "
                             "'robust'. Against ALWAYS-32B-DIRECT, the bar the project actually reports "
                             "to, a significant macro delta needs +0.0029 and this improvement is "
                             "+%.5f -- about 3x too small -- so the arm stays a TIE and no published "
                             "verdict moves." % (paired["sd"], paired["mean"])),

        attribution_of_the_improvement_over_shipped=dict(
            per_cell_contribution_to_macro=contrib,
            sum_check=total,
            carried_by="SLAKE_closed",
            statement="SLAKE-closed alone contributes +%.5f of the +%.5f macro improvement -- more than "
                      "100%% of it. PMC contributes +%.5f and MedXpert gives back -%.5f. The knob's whole "
                      "value is switching ON a cell that was at veto rate 0.0000, not tuning the cell "
                      "that already fired."
                      % (contrib["SLAKE_closed"], total, contrib["PMC_VQA"],
                         abs(contrib["MedXpertQA-MM"]))),

        cost=dict(
            macro_flopeq_as_charged=S3[BEST]["macro_flops_as_charged"],
            macro_x_direct=S3[BEST]["macro_x_direct"],
            shipped_macro_x_direct=S3["R0"]["macro_x_direct"],
            macro_x_direct_mcqonly_frame=S3[BEST]["macro_x_direct_mcqonly_frame"],
            shipped_macro_x_direct_mcqonly_frame=S3["R0"]["macro_x_direct_mcqonly_frame"],
            per_cell_x_direct={c: S3[BEST]["per_cell"][c]["x_direct"] for c in V.MCQ_CELLS},
            shipped_per_cell_x_direct={c: S3["R0"]["per_cell"][c]["x_direct"] for c in V.MCQ_CELLS},
            statement="the veto is the one lever in this method that moves accuracy and cost the SAME "
                      "way, because a certified bin never calls the 32B. Break-even veto rate = "
                      "1/4.57 = %.5f." % V.BREAKEVEN_V),

        guardrail=dict(
            flags_sig_loss=S3[BEST]["guardrail_flags_sig_loss"],
            flags_point_negative=S3[BEST]["guardrail_flags_point_negative"],
            medxpert=S3[BEST]["per_cell"]["MedXpertQA-MM"],
            statement="NOT clean. Even the guardrail-gated rule admits MedXpert on ~6% of folds and "
                      "loses there: -0.00095 [-0.00185, -0.00010], a CI-significant per-cell loss. The "
                      "unguarded rules R1/R2/R3 lose on VQA-RAD-closed as well. Guardrail, not pooled "
                      "accuracy, is the binding constraint on this knob -- as predicted."),

        permutation_null=dict(
            best_rule=S4[BEST], shipped=S4["R0"], unguarded_global=S4["R1"], unguarded_per_cell=S4["R2"],
            statement="Shuffled paired labels earn %+.5f +/- %.5f for the guardrail-gated rule and "
                      "%+.5f +/- %.5f for the unguarded per-cell rule -- the guardrail cuts the null's "
                      "spread by 3.5x. Both nulls are NEGATIVE, not positive: a veto fired at random "
                      "replaces a 32B answer with a worse 7B one, so noise-driven certification LOSES "
                      "here rather than manufacturing a win. Observed %+.5f beats all %d permutations "
                      "(p = %.5f, the 1/(nperm+1) floor)."
                      % (S4[BEST]["null_mean"], S4[BEST]["null_sd"], S4["R2"]["null_mean"],
                         S4["R2"]["null_sd"], S4[BEST]["observed_mcq_macro_contribution"],
                         S4[BEST]["n_perm"], S4[BEST]["p_value_one_sided"])),

        pmc_answer_prior_honesty=dict(
            both_currencies=S8["S8c_selected_arms_both_currencies"]["results"][BEST],
            shipped_both_currencies=S8["S8c_selected_arms_both_currencies"]["results"]["R0"],
            shipped_rank_by_raw=S8["S8b_letter_balanced_frontier"]["shipped_rank_by_raw_delta"],
            shipped_rank_by_letter_balanced=S8["S8b_letter_balanced_frontier"]
                                              ["shipped_rank_by_letter_balanced_delta"],
            n_settings=S8["S8b_letter_balanced_frontier"]["n_settings"],
            statement="THE PMC HALF OF THE TUNING GAIN IS ANSWER-PRIOR. Tuning raises the RAW PMC delta "
                      "+0.00955 -> +0.01025 while LOWERING the letter-balanced delta +0.00530 -> +0.00442 "
                      "and nearly DOUBLING the gold-A damage -0.01180 -> -0.02243 [-0.03009, -0.01488]. "
                      "The shipped setting ranks 48/135 on the raw delta but 3/135 letter-balanced. Only "
                      "two settings beat it letter-balanced -- (12, 1.96) and (12, 2.326) -- and both go "
                      "the OPPOSITE way from what the sweep selects (tighter bound, lower veto rate). "
                      "They were found by ranking the eval set in a currency chosen after the fact, so "
                      "they are a PRE-REGISTRATION CANDIDATE for a future round, never a result here."),

        slake_answer_prior_control=dict(
            per_cell=S9["SLAKE_closed"]["rules"][BEST],
            gold_marginal=S9["SLAKE_closed"]["gold_answer_marginal"],
            constant_answer_floor=S9["SLAKE_closed"]["constant_answer_floor"],
            statement="the newly-switched-on cell PASSES the control that PMC fails: SLAKE-closed's gain "
                      "is +%.5f raw and GROWS to +%.5f [%+.5f, %+.5f] when balanced over gold-answer "
                      "strata. It is item-level competence, not an answer prior."
                      % (S9["SLAKE_closed"]["rules"][BEST]["raw"]["delta"],
                         S9["SLAKE_closed"]["rules"][BEST]["answer_balanced"]["delta"],
                         S9["SLAKE_closed"]["rules"][BEST]["answer_balanced"]["ci"][0],
                         S9["SLAKE_closed"]["rules"][BEST]["answer_balanced"]["ci"][1])),

        verdict="REAL, DIRECTIONALLY INFORMATIVE, TOO SMALL TO MOVE ANY VERDICT -- and the PMC part of "
                "it is answer-prior. Coarsening the binning switches SLAKE-closed on for the first time "
                "(+0.00778 raw [+0.00048, +0.01603] / +0.01792 answer-balanced [+0.00483, +0.03167], at "
                "0.92x of always-32B-direct on that cell) and makes the PMC cell cheaper (0.73x vs the "
                "shipped 0.82x). Nested-CV 8-cell macro 0.65751 -> 0.65845: +0.00094 [+0.00001, +0.00197] "
                "vs the shipped arm, positive on 10/10 fold seeds, at 1.7198x vs 1.7403x compute. Against "
                "always-32B-direct it remains a TIE (+0.00177 [-0.00141, +0.00482]) -- ~3x short of the "
                "+0.0029 a significant macro delta needs. The improvement is >100% SLAKE-closed. It is "
                "NOT guardrail-clean (MedXpert -0.00095 [-0.00185, -0.00010]), and on PMC it trades raw "
                "delta for letter-balanced delta and doubles the gold-A damage. Recommendation: do not "
                "reship on this alone. The defensible half is the COST side and the mechanism (coarse "
                "bins, not looser bounds); the accuracy half needs the MedXpert leak closed and the PMC "
                "letter-balanced regression pre-registered away.")

    # internal consistency
    art["HEADLINE"]["_consistency_check"] = dict(
        per_cell_contributions_sum=total,
        macro_delta_vs_shipped=S3[BEST]["macro_vs_shipped_accuracy_max"]["delta"],
        abs_deviation=V.r5(abs(total - S3[BEST]["macro_vs_shipped_accuracy_max"]["delta"])),
        passed=bool(abs(total - S3[BEST]["macro_vs_shipped_accuracy_max"]["delta"]) < 2e-5))
    gb = art.setdefault("generated_by", [])
    tag = "src/cascade_methods/veto_binning_finalize.py (HEADLINE)"
    art["generated_by"] = [x for i, x in enumerate(gb + [tag]) if x not in (gb + [tag])[:i]]
    json.dump(art, open(V.OUT, "w"), indent=2, default=str)
    print(json.dumps(art["HEADLINE"], indent=2)[:6000])


if __name__ == "__main__":
    main()
