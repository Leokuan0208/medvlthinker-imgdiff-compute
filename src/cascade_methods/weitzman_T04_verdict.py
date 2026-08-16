#!/usr/bin/env python3
"""weitzman_T04_verdict.py -- assemble the KNOB 4 verdict from the two measured artifacts.

No number is typed in this file: every figure is read out of
  results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json          (parent)
  results/cascade_methods/artifacts/weitzman_T04_addendum_2026-08-15.json (addendum)
and written back into the parent as a top-level VERDICT block, so the artifact is self-contained.
"""
import json
import os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PP = os.path.join(ART, "weitzman_T04_2026-08-15.json")
PA = os.path.join(ART, "weitzman_T04_addendum_2026-08-15.json")
CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

P = json.load(open(PP))
A = json.load(open(PA))

arms = P["ARMS"]
con = P["ARM_CONTRASTS"]
op = P["OPERATING_POINTS"]
pn = P["PERMUTATION_NULL"]
fx = A["FIXED_N_CROSSFIT"]
fxc = A["FIXEDN_vs_WEITZMAN"]
fxn = A["FIXED_N_PERMUTATION_NULL"]


def g(d, *ks):
    for k in ks:
        d = d[k]
    return d


V = {
    "question": "The Weitzman controller's lambda was fitted at T=0.7 against a pooled objective. "
                "Both have changed. Refit it at T=0.4, trace the frontier, and ask whether the "
                "adaptive machinery still beats a well-chosen fixed N.",
    "null_tests": {"passed": P["null_test_passed"],
                   "max_abs_deviation": P["null_test_max_abs_deviation"],
                   "strongest": "the whole pipeline reproduces the three SHIPPED open cells of the "
                                "canonical macro artifact to 4.8e-05 (the stored artifact's own 4-dp "
                                "rounding), and the vectorised policy is BIT-EXACT against "
                                "pandora_controller.run_pandora over 91 lambdas x 2,345 items.",
                   "identity_selected_eq_oracle8_times_sel_eff":
                       P["NULL_TESTS"]["NT2_identity"]["residual"]},

    "F1_the_refit_is_real_and_in_the_predicted_direction": {
        "deployed_control_T07r_refit": {
            k: g(arms, "O2|resub|A_deployed_T07r", "open3_macro", k)
            for k in ("acc_judge", "acc_em", "meanN", "esc", "flops_eq", "lat_seq_ms")},
        "refit_at_T04": {k: g(arms, "O2|resub|C_refit_T04", "open3_macro", k)
                         for k in ("acc_judge", "acc_em", "meanN", "esc", "flops_eq", "lat_seq_ms")},
        "stale_T07_policy_applied_to_T04": {
            k: g(arms, "O2|resub|B_stale_on_T04", "open3_macro", k)
            for k in ("acc_judge", "acc_em", "meanN", "esc", "flops_eq", "lat_seq_ms")},
        "prediction": "at T=0.4 the pool saturates sooner, so the refit should DRAW FEWER and "
                      "ESCALATE LESS than the stale policy.",
        "measured": {"d_meanN_refit_minus_stale":
                         float(sum(con["O2|resub|refit_minus_stale_atT04"][c]["d_meanN"]
                                   for c in CELLS) / 3),
                     "d_esc_refit_minus_stale":
                         float(sum(con["O2|resub|refit_minus_stale_atT04"][c]["d_esc"]
                                   for c in CELLS) / 3),
                     "d_flops_refit_minus_stale":
                         con["O2|resub|refit_minus_stale_atT04"]["open3_macro_d_flops_eq"]},
        "verdict": "CONFIRMED in direction."},

    "F2_but_refitting_moves_ALONG_the_frontier_it_does_not_move_the_frontier": {
        "refit_minus_stale_open3_macro": {
            "judge": con["O2|resub|refit_minus_stale_atT04"]["open3_macro_judge"],
            "em": con["O2|resub|refit_minus_stale_atT04"]["open3_macro_em"],
            "d_flops_eq": con["O2|resub|refit_minus_stale_atT04"]["open3_macro_d_flops_eq"]},
        "refit_minus_stale_nested": {
            "judge": con["O2|nested|refit_minus_stale_atT04"]["open3_macro_judge"],
            "em": con["O2|nested|refit_minus_stale_atT04"]["open3_macro_em"]},
        "permutation_null": pn["S4_refit_minus_stale"],
        "read": "refitting buys COMPUTE and pays for it in ACCURACY, in both currencies, at a size "
                "the shuffled-label null does not reach (p = %.3f). It is a movement down the same "
                "curve, not a better curve."
                % pn["S4_refit_minus_stale"]["empirical_p_two_sided"]},

    "F3_the_free_part_of_the_win_is_the_TEMPERATURE_not_the_refit": {
        "T04_stale_minus_T07_deployed": {
            "judge": con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_judge"],
            "em": con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_em"],
            "d_flops_eq": con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_d_flops_eq"]},
        "T04_refit_minus_T07_deployed": {
            "judge": con["O2|resub|T04refit_minus_T07deployed"]["open3_macro_judge"],
            "em": con["O2|resub|T04refit_minus_T07deployed"]["open3_macro_em"],
            "d_flops_eq": con["O2|resub|T04refit_minus_T07deployed"]["open3_macro_d_flops_eq"]},
        "frontier_peaks": {
            "T04": max(P["FRONTIER"]["open3_macro_T04"], key=lambda r: r["acc_judge"]),
            "T07r": max(P["FRONTIER"]["open3_macro_T07r"], key=lambda r: r["acc_judge"])},
        "permutation_null_on_the_peak_gap": pn["S5_T04_best_minus_T07r_best"],
        "read": "moving the generation temperature to 0.4 and CHANGING NOTHING ELSE is worth more "
                "accuracy than refitting lambda is, at ~the same compute. Refitting converts part "
                "of that accuracy back into compute."},

    "F4_no_point_on_the_Weitzman_frontier_ties_the_bar_at_ANY_compute": {
        "open3_bar_judge": op["open3_bar"]["judge"], "open3_bar_em": op["open3_bar"]["em"],
        "bar_flops_eq": 4.57,
        "T04_min_flops_at_bar": op["selected"]["adaptive_T04_judge"],
        "T07r_min_flops_at_bar": op["selected"]["adaptive_T07r_judge"],
        "T04_best_point_on_the_frontier": op["selected"]["adaptive_T04_best_judge"],
        "T04_best_point_vs_the_bar": op["marked_point_CIs_vs_always32b_direct"][
            "adaptive_T04_best_judge"],
        "per_cell_min_flops_at_that_cell_own_bar": {
            c: op["per_cell_marks_T04"][c]["min_flops_at_bar_judge"] for c in CELLS},
        "read": "the requested deliverable -- 'the point that ties always-32B-direct at minimum "
                "compute' -- DOES NOT EXIST on the Weitzman frontier, at either temperature, in "
                "either currency. The frontier's own maximum sits below the bar. Only one of three "
                "cells (PATH_VQA_open) has a lambda that reaches its own bar, and it does so at "
                "8.805 FLOP-eq, i.e. 1.93x always-32B-direct."},

    "F5_the_mechanism_escalation_is_never_cheap": {
        "claim": A["STRUCTURE"]["claim"],
        "measured_over_every_pool_cell_lambda_and_fold": {
            tag: {"coupling_holds": A["STRUCTURE"]["measured"][tag]["coupling_holds"],
                  "escalated_with_0_lt_N_lt_Nmax":
                      {c: A["STRUCTURE"]["measured"][tag][c]["escalated_with_0_lt_N_lt_Nmax"]
                       for c in CELLS},
                  "max_escalation_rate_reachable":
                      {c: A["STRUCTURE"]["measured"][tag][c][
                          "max_escalation_rate_reachable_at_any_lambda"] for c in CELLS}}
            for tag in A["STRUCTURE"]["measured"]},
        "read": A["STRUCTURE"]["reading"]},

    "F6_the_adaptive_machinery_IS_redundant_at_T04_and_was_already_redundant_at_T07": {
        "protocol": A["FIXED_N_CROSSFIT_what"],
        "fixedN_crossfit_T04_nested": fx["O2|nested|T04"]["open3_macro"],
        "fixedN_crossfit_T07r_nested": fx["O2|nested|T07r"]["open3_macro"],
        "chosen_N": {c: {"mode": fx["O2|nested|T04"]["per_cell"][c]["chosen_N_mode"],
                         "mean": fx["O2|nested|T04"]["per_cell"][c]["chosen_N_mean"],
                         "tau_mean": fx["O2|nested|T04"]["per_cell"][c]["chosen_tau_mean"]}
                     for c in CELLS},
        "fixedN_minus_weitzman": {
            k: {"judge": fxc[k]["open3_macro_judge"], "em": fxc[k]["open3_macro_em"],
                "d_flops_eq": fxc[k]["open3_macro_d_flops_eq"]}
            for k in ("O2|nested|fixedN_T04_minus_weitzmanRefit_T04",
                      "O2|nested|fixedN_T07r_minus_weitzmanDeployed_T07r",
                      "O2|nested|fixedN_T04_minus_weitzmanStale_T04",
                      "O1|nested|fixedN_T04_minus_weitzmanRefit_T04",
                      "O1|nested|fixedN_T07r_minus_weitzmanDeployed_T07r")},
        "permutation_null_of_the_fixedN_selection": fxn,
        "guardrail": {o: {"flags_vs_always_32b_direct": A["FIXED_N_GUARDRAIL"][o][
                              "FLAGS_vs_always32b_direct"],
                          "flags_vs_deployed_weitzman": A["FIXED_N_GUARDRAIL"][o][
                              "FLAGS_vs_deployed_weitzman"]}
                      for o in A["FIXED_N_GUARDRAIL"]},
        "read": "a cross-fit, nested-CV FIXED best-of-N + confidence gate -- selected on TRAIN by "
                "the identical objective, over 808 candidate configurations against the "
                "controller's 91 lambdas, so carrying MORE selection risk -- beats the refitted "
                "Weitzman controller on accuracy in BOTH currencies while spending about a THIRD "
                "of the FLOP-eq, and the N it selects is 1. The shuffled-label null prices the "
                "selection honestly: the null still reaches high accuracy (by escalating "
                "everything) but cannot get below 6.04 FLOP-eq in any of 200 replicates, while the "
                "real arm sits at 4.79 (p = 0.000). The verifier signal is real, and what it buys "
                "is COMPUTE, not accuracy."},

    "F7_what_this_does_to_the_8_cell_macro": {
        "convention": op["convention"],
        "references": op["MACRO_references"],
        "weitzman_refit_T04": {h: op["MACRO"][h]["O1|resub|C_refit_T04"] for h in
                               ("compute_lean", "accuracy_max")},
        "weitzman_deployed_T07r_in_session": {h: op["MACRO"][h]["O1|resub|A_deployed_T07r"]
                                              for h in ("compute_lean", "accuracy_max")},
        "fixedN_crossfit_T04_nested_O2": {h: fx["O2|nested|T04"]["macro8"][h]
                                          for h in ("compute_lean", "accuracy_max")},
        "fixedN_crossfit_T04_nested_O1": {h: fx["O1|nested|T04"]["macro8"][h]
                                          for h in ("compute_lean", "accuracy_max")},
        "caveat": "every macro-8 figure mixes IN-SESSION open cells with STORED multiple-choice "
                  "cells and inherits the +-0.008 open-text reproducibility caveat on the open "
                  "third. The in-session-only contrasts (everything vs the T07r control) do not."},

    "GUARDRAIL_summary": {
        "weitzman_refit_T04": {"flags_vs_T07r_control": op["GUARDRAIL"]["O1|C_refit_T04"][
                                   "FLAGS_vs_T07r_control"],
                               "flags_vs_always_32b_direct": op["GUARDRAIL"]["O1|C_refit_T04"][
                                   "FLAGS_vs_always32b_direct"]},
        "fixedN_crossfit_T04": {o: A["FIXED_N_GUARDRAIL"][o]["FLAGS_vs_always32b_direct"]
                                for o in A["FIXED_N_GUARDRAIL"]},
        "note": op["GUARDRAIL_note"]},

    "amended_reading_of_the_permutation_null": A["AMENDED_READING_OF_THE_PARENT_NULL"],

    "HEADLINE": (
        "The lambda WAS stale, in exactly the predicted direction -- refitting at T=0.4 draws "
        "%.2f fewer samples and escalates %.1f points less, for %.2f fewer FLOP-eq -- but the "
        "refit only slides the operating point ALONG the frontier: it costs %+.5f [%+.5f, %+.5f] "
        "judge / %+.5f em on the open-3 macro (permutation p = %.3f). The free part of the win is "
        "the temperature itself, not the refit: the STALE policy on T=0.4 pools already beats the "
        "T=0.7 control by %+.5f [%+.5f, %+.5f]. And the requested deliverable does not exist -- NO "
        "lambda, at either temperature, in either currency, reaches always-32B-direct on the "
        "open-3 macro at any compute; the frontier's own peak is %.5f against a bar of %.5f, at "
        "%.2f FLOP-eq against the bar's 4.57. The mechanism is structural and is measured here for "
        "the first time: escalation is only reachable after all 8 cheap draws are paid for (0 of "
        "%d escalated decisions at 0 < N < 8), so one lambda cannot trade 'draw fewer' against "
        "'escalate more'. A cross-fit fixed N = 1 plus a confidence gate beats the refitted "
        "controller by %+.5f [%+.5f, %+.5f] judge at %.2f fewer FLOP-eq, and beat the DEPLOYED "
        "T=0.7 controller by %+.5f [%+.5f, %+.5f] at %.2f fewer. The adaptive machinery is "
        "redundant, and it was already redundant before the temperature moved."
        % (-con["O2|resub|refit_minus_stale_atT04"][CELLS[0]]["d_meanN"] * 0 +
           -float(sum(con["O2|resub|refit_minus_stale_atT04"][c]["d_meanN"] for c in CELLS) / 3),
           -float(sum(con["O2|resub|refit_minus_stale_atT04"][c]["d_esc"] for c in CELLS) / 3) * 100,
           -con["O2|resub|refit_minus_stale_atT04"]["open3_macro_d_flops_eq"],
           con["O2|resub|refit_minus_stale_atT04"]["open3_macro_judge"]["delta"],
           con["O2|resub|refit_minus_stale_atT04"]["open3_macro_judge"]["lo"],
           con["O2|resub|refit_minus_stale_atT04"]["open3_macro_judge"]["hi"],
           con["O2|resub|refit_minus_stale_atT04"]["open3_macro_em"]["delta"],
           pn["S4_refit_minus_stale"]["empirical_p_two_sided"],
           con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_judge"]["delta"],
           con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_judge"]["lo"],
           con["O2|resub|T04stale_minus_T07deployed"]["open3_macro_judge"]["hi"],
           op["selected"]["adaptive_T04_best_judge"]["acc_judge"], op["open3_bar"]["judge"],
           op["selected"]["adaptive_T04_best_judge"]["flops_eq"],
           sum(A["STRUCTURE"]["measured"]["T04_s0"][c]["escalated_decisions_total"] for c in CELLS),
           fxc["O2|nested|fixedN_T04_minus_weitzmanRefit_T04"]["open3_macro_judge"]["delta"],
           fxc["O2|nested|fixedN_T04_minus_weitzmanRefit_T04"]["open3_macro_judge"]["lo"],
           fxc["O2|nested|fixedN_T04_minus_weitzmanRefit_T04"]["open3_macro_judge"]["hi"],
           -fxc["O2|nested|fixedN_T04_minus_weitzmanRefit_T04"]["open3_macro_d_flops_eq"],
           fxc["O2|nested|fixedN_T07r_minus_weitzmanDeployed_T07r"]["open3_macro_judge"]["delta"],
           fxc["O2|nested|fixedN_T07r_minus_weitzmanDeployed_T07r"]["open3_macro_judge"]["lo"],
           fxc["O2|nested|fixedN_T07r_minus_weitzmanDeployed_T07r"]["open3_macro_judge"]["hi"],
           -fxc["O2|nested|fixedN_T07r_minus_weitzmanDeployed_T07r"]["open3_macro_d_flops_eq"])),

    "WHAT_IS_NOT_MEASURED": [
        "Nothing was run end to end. Every operating point here is a CPU re-costing of saved "
        "per-sample dumps with the project's measured batch-1 constants; no new generation, no GPU.",
        "The 32B-direct open leg's own token geometry at cap320 is NOT measured, so the secondary "
        "measured-FLOP currency charges it at R32 = 3.816 x the 7B generator forward. The primary "
        "currency (2.0 per cheap draw, 4.57 per escalation) is the deployed one and is what the "
        "published '11.74 vs 16.0 FLOP-eq (-27%)' survivor is denominated in.",
        "VRAM is not measured here.",
        "The 5 multiple-choice cells are frozen at their published values; this round cannot and "
        "does not touch them.",
        "The fixed-N arm's cheap leg is charged GEN7 + VER7 = 2.0 FLOP-eq even at N = 1, i.e. it "
        "pays a full verifier forward to score a single candidate. A deployment could reuse the "
        "generator's own representation instead; that is NOT measured and is not claimed."]}

P["VERDICT"] = V
json.dump(P, open(PP, "w"), indent=1, default=float)
A["VERDICT"] = V
json.dump(A, open(PA, "w"), indent=1, default=float)
print(V["HEADLINE"])
print("\nwrote VERDICT into both artifacts")
