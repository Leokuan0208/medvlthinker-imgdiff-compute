#!/usr/bin/env python3
"""verifarch_features_merge.py -- fold the objective/model-class sweep into the main artifact and add the
explicit optimism correction + verdict. Numbers are copied verbatim from the two measured artifacts; this
script computes NOTHING new except differences between cells already present.

  python3 src/cascade_methods/verifarch_features_merge.py
"""
import json, os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
MAIN = J("results/cascade_methods/artifacts/verifarch_features_2026-08-04.json")
FUP = J("results/cascade_methods/artifacts/verifarch_features_followup_2026-08-04.json")
SEEDS = J("results/cascade_methods/artifacts/verifarch_features_seeds_2026-08-04.json")

d = json.load(open(MAIN))
f = json.load(open(FUP)) if os.path.exists(FUP) else None
sd = json.load(open(SEEDS))
d["seed_stability"] = {"artifact": "results/cascade_methods/artifacts/verifarch_features_seeds_2026-08-04.json",
                       "per_seed_sel_eff": [r["sel_eff"] for r in sd["per_seed"]],
                       "mean": sd["sel_eff_mean"], "sd": sd["sel_eff_sd"],
                       "min": sd["sel_eff_min"], "max": sd["sel_eff_max"],
                       "seed_ensemble": sd["seed_ensemble"],
                       "stratified_by_gold_answer_length": sd["stratified_by_gold_answer_length"],
                       "read": ("the ARM A table above is seed 0, which is the TOP of the 10-seed range. "
                                "The honest ARM A point estimate is the 10-seed mean %.4f (sd %.4f, range "
                                "%.4f-%.4f); the seed-averaged ensemble scores %.4f, delta vs incumbent "
                                "%+.4f [%.4f, %.4f] -- still a statistical tie, but the point estimate sits "
                                "BELOW the incumbent, not level with it."
                                % (sd["sel_eff_mean"], sd["sel_eff_sd"], sd["sel_eff_min"], sd["sel_eff_max"],
                                   sd["seed_ensemble"]["sel_eff"],
                                   sd["seed_ensemble"]["vs_incumbent"]["d_sel_eff"],
                                   *sd["seed_ensemble"]["vs_incumbent"]["ci_sel_eff"]))}
if f is not None:
    d["followup"] = {"artifact": "results/cascade_methods/artifacts/verifarch_features_followup_2026-08-04.json",
                     "objective_sweep": f["objective_sweep"],
                     "listwise_minus_pointwise_on_identical_features":
                         f["listwise_minus_pointwise_on_identical_features"],
                     "bt_minus_pointwise_on_identical_features":
                         f.get("bt_minus_pointwise_on_identical_features"),
                     "hgb_sweep": f["hgb_sweep"],
                     "complementarity_with_incumbent": f["complementarity_with_incumbent"]}

inc = d["controls"]["incumbent_trained_lora"]["sel_eff"]
A = d["arm_A_deployable"]["results"]["mlp_pointwise_bce"]["sel_eff"]
Bf = d["arm_B_crossfit_diagnostic"]["results"]["features_only"]["sel_eff"]
Bi = d["arm_B_crossfit_diagnostic"]["results"]["features_plus_incumbent"]["sel_eff"]
d["optimism_correction"] = {
    "why": ("ARM B is fitted on the eval set (cross-fitted by image), ARM A is frozen from disjoint train "
            "data. The SAME feature set measured both ways gives the size of the cross-fit optimism, and it "
            "must be subtracted before reading ARM B's fusion gain as a deployable number."),
    "features_only_frozen_arm_A": A,
    "features_only_crossfit_arm_B": Bf,
    "crossfit_optimism": Bf - A,
    "arm_B_fusion_gain_over_incumbent": Bi - inc,
    "fusion_gain_after_subtracting_crossfit_optimism": (Bi - inc) - (Bf - A),
    "detection_floor_preregistered": 0.021,
    "read": ("the fusion's raw +%.4f falls to ~+%.4f once the measured cross-fit optimism is removed, i.e. "
             "below the pre-registered 2-sigma detection floor of +0.021. The fusion is therefore PLAUSIBLE "
             "but NOT ESTABLISHED as a deployable gain; settling it requires running the incumbent adapter "
             "over the 16,621 disjoint train items (one pointwise pass, ~132K forwards) so the fusion can be "
             "fitted off-eval and frozen." % (Bi - inc, (Bi - inc) - (Bf - A)))}

E = sd["seed_ensemble"]
d["verdict"] = {
    "headline": ("A 39-feature discriminative selector that makes ZERO model calls reaches sel_eff %.4f "
                 "(10-seed mean; seed-averaged ensemble %.4f, delta vs the trained same-family LoRA verifier "
                 "%+.4f [%.4f, %.4f]) -- a STATISTICAL TIE with a trained 7B judge at no inference cost -- "
                 "and beats it on candidate-level AUROC (%.4f vs %.4f). It does NOT beat it."
                 % (sd["sel_eff_mean"], E["sel_eff"], E["vs_incumbent"]["d_sel_eff"],
                    *E["vs_incumbent"]["ci_sel_eff"], E["auroc"], d["null_test"]["reproduced"]["auroc"])),
    "beat_incumbent_deployable": False,
    "beat_incumbent_diagnostic_fusion": True,
    "guardrail_arm_A": ("NOT clean (seed-averaged ensemble): SLAKE +0.0282 [-0.0018,+0.0580], "
                        "VQA-RAD -0.0159 [-0.0957,+0.0620], PathVQA -0.0400 [-0.0730,-0.0075] "
                        "(significantly worse on PathVQA)"),
    "guardrail_arm_B_fusion": "clean: all three per-dataset deltas positive (+0.0159 / +0.0159 / +0.0271)",
    "why_it_matters": (
        "This is the first scorer in the programme that is simultaneously (a) as good as the incumbent and "
        "(b) substantially DE-CORRELATED from it: the two agree on only %.1f%% of picks, each is uniquely "
        "right on ~6%% of items, and their pair-oracle is %.4f (+%.4f headroom). Every previous de-correlated "
        "source (six cross-family judges) was far WORSE than the incumbent, which is what produced the "
        "project's 'selection quality tracks agreement with the generator' law. A non-generative scorer "
        "breaks that law's confound: agreement is low AND quality is equal."
        % (100 * d["pair_oracle"]["agreement_rate_on_pick"], d["pair_oracle"]["sel_eff"],
           d["pair_oracle"]["headroom_over_incumbent"])),
    "seed_caveat": d["seed_stability"]["read"],
    "objective_question_settled": (
        ("On IDENTICAL features, the listwise softmax objective LOSES to pointwise BCE "
         "(%+.4f [%.4f, %.4f]) even after selecting the epoch budget by held-out sel_eff inside train. "
         "Combined with N25 (within-question Bradley-Terry on the LoRA: AUROC +0.030, selection +0.000), "
         "the objective is not the binding constraint on this problem -- the information is."
         % (f["listwise_minus_pointwise_on_identical_features"]["d_sel_eff"],
            *f["listwise_minus_pointwise_on_identical_features"]["ci_sel_eff"]))
        if f is not None else
        ("At the fixed configuration of ARM A, on IDENTICAL features: pointwise BCE 0.7704 > "
         "within-question Bradley-Terry 0.7459 > listwise softmax 0.6907. Combined with N25 "
         "(within-question BT on the LoRA: AUROC +0.030, selection +0.000), the objective is not the "
         "binding constraint on this problem -- the information is.")),
    "what_carries_the_signal": (
        "Permutation on the selection endpoint: soft_vote (similarity-weighted within-pool agreement) "
        "+%.4f, the TRAIN-derived answer-string correctness prior +%.4f, raw duplicate count +%.4f. "
        "No single feature group reaches the incumbent alone (best group-alone %.4f); the model is a "
        "consensus-plus-prior machine, not a grounder."
        % (d["feature_importance"]["permutation"]["soft_vote"]["drop_sel_eff_mean"],
           d["feature_importance"]["permutation"]["train_str_poscorrect_rate"]["drop_sel_eff_mean"],
           d["feature_importance"]["permutation"]["dup_count"]["drop_sel_eff_mean"],
           max(v["sel_eff_group_alone"] for v in d["feature_importance"]["group_alone"].values()))),
    "limits": [
        "ARM A carries NO image information at all, so it cannot fix the documented failure mode "
        "(one-token visual contrasts such as left/right); it loses on PathVQA for exactly that reason.",
        "The train-derived answer-string prior is legitimate (L1 image- and item-disjoint) but is an "
        "in-distribution answer prior, not grounding; on a set with an unfamiliar answer vocabulary it "
        "would contribute nothing.",
        "ARM B is cross-fitted on eval and is a diagnostic upper bound, not a deployable number.",
        "Seed spread is 0.0050 sd on sel_eff, so any single-seed comparison of two feature models below "
        "+0.015 is noise; the reported ARM A table is seed 0 and the seed-mean is 0.0086 lower.",
        "The gold-answer-length strata are descriptive only -- n_recoverable is 80 (medium) and 17 (long), "
        "far too small for the paired bootstrap, and no CI is quoted for them.",
    ],
}
json.dump(d, open(MAIN, "w"), indent=1)
print("merged ->", MAIN)
for k in ("optimism_correction", "verdict"):
    print(json.dumps(d[k], indent=1))
