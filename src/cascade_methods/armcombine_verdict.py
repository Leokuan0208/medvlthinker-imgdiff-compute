#!/usr/bin/env python3
"""
Append the VERDICT / HEADLINE_TABLE / limitations block to armcombine_2026-08-11.json.

Reads ONLY that artifact's own numbers -- it computes no new statistic, it adjudicates the
pre-registered criteria against what was measured.  Kept separate from armcombine.py so the
expensive run is never re-executed to change wording.

Reproduce:  python3 src/cascade_methods/armcombine_verdict.py
"""
import os
import json

import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
P = os.path.join(ART, "armcombine_2026-08-11.json")
d = json.load(open(P))

GS = ["l32_bo8_s0", "l32_bo8_s1", "l32_bo8_s2"]

# --- repair one mislabelled field written by armcombine.py -------------------------------------
# `not_sig_worse` was emitted as `hi > 0 or lo > -1`, which is True for every row and therefore
# carries no information.  It is recomputed here from the row's own CI (not significantly worse
# means the upper bound of the delta is not below zero) and the original is dropped, so no reader
# can mistake the placeholder for a test.
_repaired = 0
for _fr in [d["frame_P_published_bar_old_menu"], d["frame_X_unmatched_DIAGNOSTIC"]] + \
           [d["frame_M_matched_headline"][t] for t in GS]:
    for _r in _fr.get("eps_frontier_crossfit", []):
        _r.pop("not_sig_worse", None)
        _r["not_significantly_worse_than_direct"] = bool(_r["hi"] >= 0)
        _repaired += 1
M = d["frame_M_matched_headline"]
S0 = M[GS[0]]
BAR = S0["bar_macro"]
EST_ORDER = ["E1_crossfit_argmax", "E2_nested_margin", "E3_nested_costaware",
             "E5_format_crossfit_argmax_POSTHOC", "E6_format_nested_margin_POSTHOC"]


def gen_avg(key, field):
    return float(np.mean([M[t]["headline_ci_foldseed_averaged"][key][field] for t in GS]))


def gen_list(key, field):
    return [round(M[t]["headline_ci_foldseed_averaged"][key][field], 5) for t in GS]


out = {}

# ------------------------------------------------------------------ 1. headline table
rows = [["always-32B-direct  (THE BAR, frame M / matched serving run)", round(BAR, 4),
         "0 (reference)", 1.0, "reference"]]
for k in EST_ORDER:
    dl = gen_avg(k, "delta")
    lo = gen_avg(k, "lo")
    hi = gen_avg(k, "hi")
    xd = float(np.mean([M[t]["fold_seed_summary"][k]["x_direct_as_charged_mean"] for t in GS]))
    rows.append([k, round(gen_avg(k, "macro_acc"), 4),
                 f"{dl:+.4f} [{lo:+.4f}, {hi:+.4f}]", round(xd, 3),
                 "TIE" if (lo < 0 < hi) else ("WIN" if lo > 0 else "LOSS")])
rows.append(["E0 naive best-arm-per-cell (EVAL-VISIBLE DIAGNOSTIC)",
             round(float(np.mean([M[t]["E0_naive_evalvisible_DIAGNOSTIC"]["macro"] for t in GS])), 4),
             f"{float(np.mean([M[t]['E0_naive_evalvisible_DIAGNOSTIC']['delta'] for t in GS])):+.4f} "
             f"(no CI -- eval-visible)", "n/a", "DIAGNOSTIC ONLY"])
out["HEADLINE_TABLE"] = dict(
    columns=["policy", "macro_acc", "delta vs always-32B-direct [95% CI]",
             "as-charged x_direct (R32=4.57, MACRO-weighted)", "verdict"],
    rows=rows,
    weighting_label="EVERY accuracy is MACRO (8 cells, 1/8 each, Variant B) and every cost column is "
                    "MACRO-weighted.  Never pair one of these accuracies with a sample-weighted cost.",
    generation_seed_note="delta / CI / cost are averaged over the 3 available 32B generation seeds; "
                         "each per-seed value is itself averaged over 12 fold seeds.")

# ------------------------------------------------------------------ 2. criteria
PN = S0["permutation_null"]
e2 = dict(delta=gen_avg("E2_nested_margin", "delta"), lo=gen_avg("E2_nested_margin", "lo"),
          hi=gen_avg("E2_nested_margin", "hi"))
guard_e2 = {t: M[t]["headline_ci_foldseed_averaged"]["E2_nested_margin"]["guardrail_flags"] for t in GS}
guard_e6 = {t: M[t]["headline_ci_foldseed_averaged"]["E6_format_nested_margin_POSTHOC"]["guardrail_flags"]
            for t in GS}

out["PRE_REGISTERED_CRITERIA"] = {
    "S1_honest_CI_excludes_zero": dict(
        estimator="E2_nested_margin (the pre-registered headline)",
        delta=round(e2["delta"], 5), ci=[round(e2["lo"], 5), round(e2["hi"], 5)],
        per_generation_seed_delta=gen_list("E2_nested_margin", "delta"),
        per_generation_seed_lo=gen_list("E2_nested_margin", "lo"),
        met=bool(e2["lo"] > 0)),
    "S2_permutation_p_below_0p05": dict(
        p_one_sided_naive_null=PN["E2_nested_margin"]["p_one_sided"],
        p_one_sided_dedup_null=json.load(open(os.path.join(
            ART, "_armcombine_dedup_null_2026-08-11.json")))["dedup_permutation_null"]
            ["E2_nested_margin"]["p_one_sided"],
        met=bool(PN["E2_nested_margin"]["p_one_sided"] < 0.05)),
    "S3_no_guardrail_flag": dict(flags_per_generation_seed=guard_e2,
                                 met=all(len(v) == 0 for v in guard_e2.values())),
    "K1_CI_covers_zero": dict(fired=bool(e2["lo"] <= 0 <= e2["hi"])),
    "K2_permutation_null_explains_the_gain": dict(
        null_p97p5=PN["E2_nested_margin"]["null_p97p5"],
        observed=PN["E2_nested_margin"]["observed_primary_fold_seed"],
        fired=bool(PN["E2_nested_margin"]["null_p97p5"] >= PN["E2_nested_margin"]["observed_primary_fold_seed"])),
    "K3_gain_only_in_the_unmatched_frame": dict(
        frame_M_matched_E2=round(e2["delta"], 5),
        frame_X_unmatched_E2=d["frame_X_unmatched_DIAGNOSTIC"]["headline_ci_foldseed_averaged"]
                              ["E2_nested_margin"]["delta"],
        drift_contribution=round(
            d["frame_X_unmatched_DIAGNOSTIC"]["headline_ci_foldseed_averaged"]["E2_nested_margin"]["delta"]
            - e2["delta"], 5),
        fired=False,
        note="K3 as worded asks whether the gain exists ONLY against the published (unmatched) bar. "
             "It does not -- the point estimate is positive in both frames.  What the two frames "
             "differ by is the size: the unmatched frame is inflated by the serving-config drift, "
             "which is why frame M is the headline."),
}

# ------------------------------------------------------------------ 2b. the dedup null
DN = json.load(open(os.path.join(ART, "_armcombine_dedup_null_2026-08-11.json")))
out["DEDUPLICATED_PERMUTATION_NULL"] = dict(
    source="results/cascade_methods/artifacts/_armcombine_dedup_null_2026-08-11.json "
           "(src/cascade_methods/armcombine_dedupnull.py)",
    why=DN["why"],
    effective_menu_size={c: dict(n_arms_on_menu=v["n_arms_on_menu"],
                                 n_DISTINCT_columns=v["n_DISTINCT_columns"])
                         for c, v in DN["effective_menu_size"].items()},
    duplicate_groups={c: v["duplicate_groups"] for c, v in DN["effective_menu_size"].items()
                      if v["duplicate_groups"]},
    table=DN["dedup_permutation_null"],
    finding="the correction is REAL but IMMATERIAL in size: the null sd for the pre-registered "
            f"headline estimator moves from {PN['E2_nested_margin']['null_sd']} (naive) to "
            f"{DN['dedup_permutation_null']['E2_nested_margin']['null_sd']} (dedup), and every "
            "one-sided p-value moves by less than 0.01.  Both nulls are reported; the dedup one is "
            "the primary because it matches the real menu's effective size.")

# ------------------------------------------------------------------ 3. the permutation-null story
out["PERMUTATION_NULL_IS_THE_RESULT"] = {
    "what_it_measures": "how much macro gain the SAME selection procedure produces when every arm is "
                        "made exchangeable (per-item random permutation of the arm labels), i.e. when "
                        "no arm is truly better than any other.  Anything at or below this is "
                        "selection bias, not method.",
    "table": {k: {kk: PN[k][kk] for kk in ("null_mean", "null_sd", "null_p97p5", "null_max",
                                           "observed_primary_fold_seed", "p_one_sided")}
              for k in PN if isinstance(PN[k], dict)},
    "the_single_most_important_number": (
        f"the EVAL-VISIBLE best-arm-per-cell rule gains +{PN['E0_naive_evalvisible']['null_mean']:.4f} "
        f"macro ON AVERAGE from shuffled labels alone, against an observed "
        f"+{PN['E0_naive_evalvisible']['observed_primary_fold_seed']:.4f}.  The naive number this "
        f"attack was asked to test is SMALLER than the average gain the same procedure produces on "
        f"pure noise (p = {PN['E0_naive_evalvisible']['p_one_sided']})."),
    "why_cross_fit_is_not_enough": (
        "cross-fit removes the BIAS (null mean falls to ~0) but not the VARIANCE: the null sd is "
        f"{PN['E1_crossfit_argmax']['null_sd']:.4f} for per-cell argmax and "
        f"{PN['E2_nested_margin']['null_sd']:.4f} for the nested-margin rule, so a +0.003-level "
        "observed gain sits well inside the noise a null menu produces.  With 8 cells and 6-9 arms "
        "there simply is not enough independent evidence per cell to pick reliably."),
    "the_fix_that_did_NOT_work": (
        "reducing the number of decisions from 8 (per cell) to 2 (per answer FORMAT) was expected to "
        "shrink the null.  It does not: the dedup null sd is "
        f"{DN['dedup_permutation_null']['E5_format_crossfit_argmax_POSTHOC']['null_sd']} for "
        "format-level argmax against "
        f"{DN['dedup_permutation_null']['E1_crossfit_argmax']['null_sd']} for per-cell argmax -- "
        "slightly LARGER, because one bad format-level pick contaminates 5 or 3 cells at once "
        "instead of 1.  Fewer decisions do not mean less noise when each decision is applied more "
        "widely.  E5's larger point estimate is bought with correspondingly larger variance and its "
        f"one-sided p is {DN['dedup_permutation_null']['E5_format_crossfit_argmax_POSTHOC']['p_one_sided']}."),
    "duplicate_column_correction": "reported separately in DEDUPLICATED_PERMUTATION_NULL -- real, "
                                   "immaterial (< 0.01 on every p-value).",
}

# ------------------------------------------------------------------ 3b. the POSTHOC estimators
out["POSTHOC_FORMAT_LEVEL_ESTIMATORS"] = dict(
    status="POST-HOC / EXPLORATORY.  Added AFTER seeing that per-cell selection is noise-dominated. "
           "NOT the pre-registered headline.  Reported with their own permutation null so a later "
           "reader cannot mistake them for a pre-registered result.",
    definition="ONE arm for the 5 MCQ cells and ONE arm for the 3 open cells -- 2 cross-fit "
               "decisions instead of 8.  This is the project's existing format-aware architecture "
               "applied at the arm-selection level.",
    E5_argmax=dict(per_generation_seed_delta=gen_list("E5_format_crossfit_argmax_POSTHOC", "delta"),
                   mean_delta=round(gen_avg("E5_format_crossfit_argmax_POSTHOC", "delta"), 5),
                   mean_ci=[round(gen_avg("E5_format_crossfit_argmax_POSTHOC", "lo"), 5),
                            round(gen_avg("E5_format_crossfit_argmax_POSTHOC", "hi"), 5)],
                   dedup_p_one_sided=DN["dedup_permutation_null"]
                                       ["E5_format_crossfit_argmax_POSTHOC"]["p_one_sided"],
                   guardrail_flags={t: M[t]["headline_ci_foldseed_averaged"]
                                       ["E5_format_crossfit_argmax_POSTHOC"]["guardrail_flags"]
                                    for t in GS},
                   x_direct_as_charged=round(float(np.mean(
                       [M[t]["fold_seed_summary"]["E5_format_crossfit_argmax_POSTHOC"]
                          ["x_direct_as_charged_mean"] for t in GS])), 3)),
    E6_nested_margin=dict(per_generation_seed_delta=gen_list("E6_format_nested_margin_POSTHOC", "delta"),
                          mean_delta=round(gen_avg("E6_format_nested_margin_POSTHOC", "delta"), 5),
                          mean_ci=[round(gen_avg("E6_format_nested_margin_POSTHOC", "lo"), 5),
                                   round(gen_avg("E6_format_nested_margin_POSTHOC", "hi"), 5)],
                          dedup_p_one_sided=DN["dedup_permutation_null"]
                                              ["E6_format_nested_margin_POSTHOC"]["p_one_sided"],
                          guardrail_flags=guard_e6),
    format_level_picks={t: M[t]["format_level_picks"] for t in GS},
    reading="E5's point estimate (the largest in the table) is also the least stable across the 3 "
            "generation seeds relative to its own size, and E6 -- the same rule with an honest "
            "shrinkage margin -- collapses from +0.0054 on generation seed s0 to -0.0000 and +0.0004 "
            "on s1 and s2.  A quantity that changes by more than its own point estimate when only "
            "the 32B's sampling seed changes is not a result.  Neither clears its permutation null.")

# ------------------------------------------------------------------ 4. cheapest honest tie
best = None
for t in GS:
    for r in M[t]["eps_frontier_crossfit"]:
        if r["lo"] >= -0.0029 and not r["guardrail_flags"]:
            if best is None or r["x_direct_as_charged_12seed"] < best["x_direct_as_charged_12seed"]:
                best = dict(r, gen_seed=t)
best_any = None
for t in GS:
    for r in M[t]["eps_frontier_crossfit"]:
        if r["lo"] >= -0.0029:
            if best_any is None or r["x_direct_as_charged_12seed"] < best_any["x_direct_as_charged_12seed"]:
                best_any = dict(r, gen_seed=t)
# robust across ALL generation seeds at the same eps
robust = []
for e in [r["eps"] for r in M[GS[0]]["eps_frontier_crossfit"]]:
    rr = [next(r for r in M[t]["eps_frontier_crossfit"] if r["eps"] == e) for t in GS]
    if all(x["lo"] >= -0.0029 for x in rr):
        robust.append(dict(eps=e,
                           x_direct=round(float(np.mean([x["x_direct_as_charged_12seed"] for x in rr])), 4),
                           delta=round(float(np.mean([x["delta"] for x in rr])), 5),
                           lo=round(float(np.mean([x["lo"] for x in rr])), 5),
                           guardrail_flags=sorted({g for x in rr for g in x["guardrail_flags"]})))
robust.sort(key=lambda r: r["x_direct"])
out["SECONDARY_ENDPOINT_cheapest_honest_tie"] = dict(
    definition="cheapest cross-fit operating point on the ENLARGED menu whose bootstrap lower bound "
               "is >= -0.0029 (the pre-registered tie), 12 fold seeds, macro-weighted as-charged cost",
    cheapest_tie_robust_across_all_3_generation_seeds=(robust[0] if robust else None),
    all_robust_rows=robust,
    cheapest_tie_any_single_generation_seed=best_any,
    cheapest_tie_any_seed_and_guardrail_clean=best,
    comparison=dict(
        shipped_accuracy_max="0.6575 macro at 1.740x always-32B-direct "
                             "(cascade_selector_rerun_2026-08-05.json)",
        round1_cost_floor_eps0_crossfit="0.6578 macro, +0.0011 [-0.0014,+0.0035], 1.1648x "
                                        "(cost_floor_2026-08-10.json:VERDICT.what_DID_clear_the_bar)",
        this_attack="the enlarged menu does NOT produce a cheaper tie than round 1's -- the arms it "
                    "adds (32B best-of-4/8, 22.3 / 44.6 FLOP-eq per item) are the most expensive on "
                    "the whole menu, so every point that uses them moves RIGHT on the cost axis."))

# ------------------------------------------------------------------ 5. limitations
out["LIMITATIONS_AND_HONESTY_CAVEATS"] = [
    "MENU-LEVEL EVAL VISIBILITY, uncorrectable.  The 32B best-of-N family is on the menu BECAUSE "
    "round 1 saw it win PathVQA-open on this same eval.  Cross-fit corrects the per-cell arm choice; "
    "it cannot correct the decision to include an arm family.  Mitigation applied: all three "
    "best-of-N arms were placed on all three open cells, not only on the winning one.  This is the "
    "single largest honesty caveat of this attack and it points the same direction as the verdict.",
    "ONLY 3 GENERATION SEEDS of the 32B 8-sample pool exist (round 1 produced s0/s1/s2; each costs "
    "GPU).  Methodology rule 4 asks for >=10 where sampling is involved.  The generation-seed spread "
    "of the headline estimator is reported and is NOT the binding uncertainty here -- the item-level "
    "bootstrap CI is roughly 3x wider than the generation-seed range -- but 3 is 3.",
    "LATENCY AND ENERGY ARE NOT MEASURED for any policy that deploys a 32B best-of-N arm.  "
    "openstrong_bestofn_2026-08-10.json:cost.provenance states this explicitly.  Only FLOP-eq is "
    "reported for those policies; no latency or energy number is quoted or modelled for them.",
    "CONVENTIONS B AND C (shared prefill) REMAIN UNCORROBORATED -- cost_floor_2026-08-10.json's code "
    "audit found vLLM V1 implements SamplingParams(n=N) as N child requests relying on prefix "
    "caching, not a post-prefill fork.  The primary cost convention here is A (as-charged).",
    "THE 7B-BASED OPEN ARMS CARRY OLD-CONFIG 32B ANSWERS on their escalated fraction.  Their "
    "accuracy is therefore measured with a 32B run from a different serving configuration than the "
    "frame-M bar.  This makes those arms look WORSE relative to the raised matched bar, i.e. it is "
    "conservative for any policy that would otherwise deploy them; it is not corrected because the "
    "per-item escalation mask is not exported by the published artifact.",
    "E5 / E6 (format-level selection) were ADDED AFTER seeing that per-cell selection is "
    "noise-dominated.  They are labelled POSTHOC in the artifact, carry their own permutation null, "
    "and are NOT the pre-registered headline.  If they look promising they are a round-3 "
    "pre-registration, not a result.",
]

# ------------------------------------------------------------------ 5b. transferable finding
DNT = DN["dedup_permutation_null"]
out["OPERATIONAL_RULE_FOR_FUTURE_ROUNDS"] = dict(
    finding="THIS SUITE HAS A MEASURED SELECTION-NOISE FLOOR, and it is larger than the significance "
            "bar the project has been aiming at.",
    numbers={
        "eval_visible_best_arm_per_cell": f"gains +{DNT['E0_naive_evalvisible']['null_mean']:.4f} macro "
                                          f"on average, and up to +{DNT['E0_naive_evalvisible']['null_max']:.4f}, "
                                          "from SHUFFLED LABELS alone (8 cells, 6-9 arms)",
        "cross_fit_per_cell_argmax": f"unbiased (null mean {DNT['E1_crossfit_argmax']['null_mean']:+.5f}) "
                                     f"but sd {DNT['E1_crossfit_argmax']['null_sd']:.4f}, "
                                     f"97.5th pct +{DNT['E1_crossfit_argmax']['null_p97p5']:.4f}",
        "cross_fit_nested_margin": f"null mean {DNT['E2_nested_margin']['null_mean']:+.5f}, sd "
                                   f"{DNT['E2_nested_margin']['null_sd']:.4f}, 97.5th pct "
                                   f"+{DNT['E2_nested_margin']['null_p97p5']:.4f}",
        "format_level_2_decisions": f"null sd {DNT['E5_format_crossfit_argmax_POSTHOC']['null_sd']:.4f} "
                                    "-- NOT smaller than per-cell selection",
        "the_significance_bar_being_chased": 0.0029,
    },
    rule=("Any macro gain obtained by SELECTING among arms/policies per cell on this 8-cell suite "
          "must be reported against this null.  An eval-visible per-cell-selected gain below about "
          f"+{DNT['E0_naive_evalvisible']['null_p97p5']:.4f} is inside the range shuffled labels "
          "produce; a cross-fit per-cell-selected gain below about "
          f"+{DNT['E1_crossfit_argmax']['null_p97p5']:.4f} is inside the range shuffled labels "
          "produce after cross-fitting.  Since the bar for a significant win is +0.0029, PER-CELL "
          "SELECTION CANNOT BE THE MECHANISM THAT DELIVERS IT on this suite -- the noise it "
          "introduces is roughly 4x the effect being sought.  Future attacks should look for a "
          "single mechanism that moves MANY cells in the same direction, not for a better way to "
          "choose per cell."),
    corollary=("this also retro-explains round 1's Attack 3 result: its +0.0011 [-0.0014,+0.0035] "
               "eps=0 cross-fit row is well inside this null and was correctly reported as a TIE."),
    what_is_actually_real_here=(
        "two per-cell effects survive their own CIs in every generation seed: PMC_VQA "
        "+0.0133 [+0.0099,+0.0167] from deploying the shipped fusion arm there, and PATH_VQA_open "
        "+0.023 to +0.028 from the 32B best-of-N arm.  At equal weight those are worth "
        "+0.0017 and +0.0029 macro respectively.  They are real; they are simply not enough, and "
        "the six remaining cells contribute nothing but the noise of being chosen over."),
)

MQ = json.load(open(os.path.join(ART, "armcombine_mcqonly_2026-08-11.json")))
out["FOLLOW_UP_E7_the_one_positive_result"] = dict(
    artifact="results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json",
    code="src/cascade_methods/armcombine_mcqonly.py (+ _verdict.py)",
    status="POST-HOC, written after this attack's per-cell SE decomposition showed the open cells "
           "supply 99.6% of the macro's variance",
    what_it_is="freeze the open-text half at always-32B-direct and change only the multiple-choice "
               "half -- the project's own format-aware architecture, with the open arm switched OFF",
    rows=MQ["VERDICT"]["rows"],
    honest_sentence=MQ["VERDICT"]["honest_sentence"],
    relation_to_this_attack=(
        "it is the OPPOSITE conclusion drawn from the same decomposition.  This attack asked whether "
        "ADDING the 32B best-of-N arms to the menu produces a win; it does not, and the permutation "
        "null says per-cell selection cannot.  E7 asks what happens if the open half is left alone, "
        "and the answer is a small CI-clean win that was already implicit in the 2026-08-05 "
        "artifact's MCQ-half number.  Both findings point the same way: on this suite the "
        "multiple-choice half is where the resolvable signal is, and the open-text half is where "
        "the variance is."),
)

out["HANDOFF_next_round"] = [
    "THE ONE POSITIVE RESULT OF THIS ROUND IS E7 (see FOLLOW_UP_E7_the_one_positive_result): "
    "MCQ = certified veto, OPEN = always-32B-direct, +0.0012 [+0.0009,+0.0015] macro at 0.977x "
    "FLOP-eq and -0.5% energy, guardrail-clean.  It is small, it is entirely PMC-VQA, and the "
    "underlying MCQ-half number was already published -- but on the stated axis it is a WIN with a "
    "CI excluding zero, and it is the first one this project has that is also not more expensive.",
    "THE MACRO DESIGN'S OWN RESOLUTION IS THE BINDING CONSTRAINT for anything that touches open "
    "text.  Measured here: the 3 open cells supply 99.6% of the 8-cell macro's standard error for a "
    "policy that changes them, and VQA_RAD_open (n=200) alone supplies 71%.  A policy touching the "
    "open cells has a macro SE of ~0.0033, so a +0.003 effect is BELOW the design's resolution by "
    "construction.  Either grow the open cells or stop trying to win the macro there.",
    "DO NOT re-run per-cell arm selection in any form.  It is now measured, not argued: the "
    "procedure's own noise exceeds the target effect on this suite.",
    "The two real per-cell effects are PMC_VQA (fusion arm, +0.0133, n=33430 -- the most robust "
    "single number in this attack) and PATH_VQA_open (32B best-of-N, +0.023..+0.028).  Any round-3 "
    "attack should be judged on whether it moves the OTHER six cells, because those two are already "
    "harvested and together they are worth only +0.0046 macro.",
    "MedXpertQA-MM is where every selection policy in this attack LOSES significantly.  It has "
    "chance-level keep-7B AUROC (0.4877), 89.6% escalation, and 0.3065 accuracy at the bar -- it is "
    "a cell where nothing on the current menu beats always-32B-direct and any deviation costs.  "
    "Treat it as fixed at always-32B-direct in any future policy, and say so explicitly.",
    "The serving-config drift is now quantified end-to-end: the same estimator reads "
    "+0.0060 against the published open-cell bar and +0.0028 against the matched one.  Half the "
    "apparent gain of any open-text comparison in this project can be serving configuration.  The "
    "matched-control requirement is not a formality.",
    "Cost is still the more tractable endpoint, and round 1 still owns the best point: 0.6578 macro "
    "at 1.165x (cost_floor_2026-08-10.json).  Nothing in this attack improves on it -- the arms it "
    "adds cost 22.3 and 44.6 FLOP-eq per item against the baseline's 4.57.",
]

# ------------------------------------------------------------------ 6. verdict sentence
sig = bool(e2["lo"] > 0)
out["VERDICT"] = dict(
    pre_registered_headline_estimator="E2_nested_margin, frame M (matched serving run), "
                                      "generation-seed-averaged over 3 seeds, 12 fold seeds each",
    macro_delta_vs_always_32b_direct=round(e2["delta"], 5),
    ci=[round(e2["lo"], 5), round(e2["hi"], 5)],
    verdict="WIN" if sig else "TIE -- SUCCESS CRITERION NOT MET",
    K1_fired=bool(not sig),
    K2_fired=bool(PN["E2_nested_margin"]["null_p97p5"] >= PN["E2_nested_margin"]["observed_primary_fold_seed"]),
    sentence=(
        "Combining every per-cell arm round 1 measured -- including the 32B best-of-N arms that "
        "Attack 3's menu was missing -- does NOT beat always-32B-direct on the 8-cell macro under an "
        f"honest estimator.  The pre-registered nested per-cell arm-selection policy is "
        f"{e2['delta']:+.4f} [{e2['lo']:+.4f}, {e2['hi']:+.4f}] against a matched always-32B-direct "
        f"bar of {BAR:.4f}, at "
        f"{float(np.mean([M[t]['fold_seed_summary']['E2_nested_margin']['x_direct_as_charged_mean'] for t in GS])):.2f}x "
        "that baseline's macro-weighted as-charged compute.  The naive arithmetic the attack was "
        "asked to test (+0.0269 on PathVQA-open / 8, plus PMC's veto win / 8) does not survive: a "
        "1,000-draw exchangeability permutation null shows that the eval-visible best-arm-per-cell "
        f"rule earns +{PN['E0_naive_evalvisible']['null_mean']:.4f} macro ON AVERAGE from shuffled "
        "labels alone, which is MORE than the +"
        f"{PN['E0_naive_evalvisible']['observed_primary_fold_seed']:.4f} it earns on the real menu.  "
        "The per-cell effects that are real -- PMC_VQA and PATH_VQA_open -- are, at equal weight, "
        "too few and too small to carry the macro past its own selection noise."),
)

out["field_repairs"] = dict(
    n_rows=_repaired,
    what="armcombine.py wrote `not_sig_worse` as `hi > 0 or lo > -1`, a placeholder that is True on "
         "every row.  It is removed and replaced by `not_significantly_worse_than_direct` = (hi >= 0), "
         "computed from the row's own bootstrap CI.  No other value in the artifact is touched.")

d["VERDICT_BLOCK"] = out
json.dump(d, open(P, "w"), indent=1, default=str)
print(json.dumps(out["HEADLINE_TABLE"], indent=1))
print(json.dumps(out["PRE_REGISTERED_CRITERIA"], indent=1))
print(json.dumps(out["VERDICT"], indent=1))
print(json.dumps(out["SECONDARY_ENDPOINT_cheapest_honest_tie"], indent=1))
