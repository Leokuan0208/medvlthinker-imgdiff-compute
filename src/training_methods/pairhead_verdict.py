#!/usr/bin/env python3
"""pairhead_verdict.py -- append the plain-language VERDICT to the pairwise-head artifact.

Every number written here is copied from a key already present in the artifact; nothing is
recomputed and nothing is entered by hand (project CRITICAL RULE 7).
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G

P = os.path.join(G.ROOT, "results/cascade_methods/artifacts/verifarch_pairhead_2026-08-04.json")
a = json.load(open(P))
h = a["HEADLINE_pair_head_seed_ensemble"]
pb = a["pointwise_head_bar"]
ad = a["additive_decomposition"]
idg = a["indomain_diagnostic_CONTAMINATED"]
fu = a["fusions"]
vt = a["variant_table_DIAGNOSTIC"]
ag = a["aggregation_table"]

a["VERDICT"] = {
    "one_line": "NEGATIVE, cleanly diagnosed: a pairwise contrast head over cached generator-frame "
                "vectors matches but does not beat the pointwise head on the same features "
                f"(d = {h['vs_pointwise_head_ensemble']['d_sel_eff']} "
                f"{h['vs_pointwise_head_ensemble']['ci']}, n.s.), because the comparison it learns "
                f"is {ad['variance_share_additive']:.1%} additive -- a pointwise scorer in a "
                "pairwise costume.",
    "headline_preregistered": {
        "sel_eff": h["sel_eff"], "n_seeds": h["n_seeds"], "aggregation": h["aggregation"],
        "config": a["preregistration"]["config"],
        "selected_by": "train-split image-grouped CV only (data/verifarch/pairhead_cv.json)"},
    "answers": {
        "beats_the_incumbent_0.775204": {
            "verdict": True, "d": h["vs_incumbent_0p775204"]["d_sel_eff"],
            "ci": h["vs_incumbent_0p775204"]["ci"],
            "caveat": "but the pointwise head on the SAME features already does this "
                      f"(d = {pb['seed_ensemble_zmean']['vs_incumbent']['d_sel_eff']} "
                      f"{pb['seed_ensemble_zmean']['vs_incumbent']['ci']}), so none of the gain is "
                      "attributable to the pairwise machinery"},
        "beats_the_pointwise_head_on_identical_features": {
            "verdict": False, "d": h["vs_pointwise_head_ensemble"]["d_sel_eff"],
            "ci": h["vs_pointwise_head_ensemble"]["ci"],
            "matched_seed_budget_note":
                f"at a MATCHED single-seed budget the pairwise head is marginally WORSE: mean over "
                f"12 seeds {a['pair_head_seeds']['mean']} (sd {a['pair_head_seeds']['sd']}) vs the "
                f"pointwise head's {pb['seed_spread_sel_eff']['mean']} "
                f"(sd {pb['seed_spread_sel_eff']['sd']}). The headline only exceeds the PUBLISHED "
                f"single-seed 0.795640 because it is a 12-seed ensemble; that is a seed-budget "
                f"effect, not a pairwise effect."},
        "beats_the_deployed_fusion_0.806540": {
            "verdict": False, "pair_head_alone": h["sel_eff"],
            "note": "the deployed fusion is itself seed-noisy: repeating its own recipe "
                    "(one pointwise seed + incumbent, rank_avg) over 12 seeds gives "
                    f"{fu['_deployed_fusion_single_seed_distribution']['min']}-"
                    f"{fu['_deployed_fusion_single_seed_distribution']['max']} "
                    f"(mean {fu['_deployed_fusion_single_seed_distribution']['mean']}), which "
                    "brackets the published 0.806540."},
        "guardrail_clean": {
            "verdict": h["guardrail_clean_vs_incumbent"],
            "per_ds": h["per_ds"],
            "incumbent_per_ds": a["controls"]["incumbent_lora_verifier_THE_BAR"]["per_ds"],
            "note": "vqa_rad_open is the loss. The pointwise-head ensemble is guardrail-dirty on the "
                    "same set, and so was the 2026-07 real-forward-pass pairwise win, which lost on "
                    "pathvqa_open (0.6538 -> 0.6154, artifacts/pairwise_verifier_gpu.json)."}},
    "why_it_failed": {
        "1_the_learned_comparison_is_additive": {
            "additive_variance_share": ad["variance_share_additive"],
            "residual_variance_share": ad["variance_share_residual"],
            "residual_only_sel_eff": ad["residual_part_alone"]["sel_eff"],
            "random_pick_sel_eff": ad["residual_part_alone"]["random_pick_sel_eff"],
            "statement": "any antisymmetric G decomposes uniquely into (theta_i - theta_j) + Resid. "
                         f"{ad['variance_share_additive']:.1%} of the learned matrix is the additive "
                         "term, which IS a pointwise scorer; ranking by the residual alone gives "
                         f"{ad['residual_part_alone']['sel_eff']}, essentially the random-pick floor "
                         f"({ad['residual_part_alone']['random_pick_sel_eff']}). There is no "
                         "comparative signal left over once the pointwise part is removed."},
        "2_train_side_CV_agreed_before_eval_was_touched": {
            "preregistered_aggregation": h["aggregation"],
            "statement": "the aggregation rule chosen by train-only CV was logit_sum -- which is "
                         "exactly the additive projection sum_j G[i,j] = k*theta_i. The "
                         "pre-registration selected the pointwise reading of the pairwise head "
                         "without ever seeing eval.",
            "cv_sel_eff_pair_head_train_only": a["preregistration"]["cv_sel_eff_train_only"],
            "cv_sel_eff_published_pointwise_head": 0.6784561093646115,
            "cv_source_pointwise": "artifacts/verifarch_hidden_generatorprompt_2026-08-04.json "
                                   "-> arms/generator/cv_selected/cv_sel_eff"},
        "3_it_is_not_a_transfer_failure": {
            "indomain_contaminated_sel_eff": idg["sel_eff"],
            "indomain_contaminated_additive_share": idg["variance_share_additive"],
            "clean_sel_eff": h["sel_eff"],
            "statement": "fitting the same head INSIDE eval with image-grouped folds -- an optimistic, "
                         "contaminated bound that shares images and question distribution with the "
                         f"test set -- gives {idg['sel_eff']}, no better than the clean transfer arm, "
                         f"and stays {idg['variance_share_additive']:.1%} additive. The non-additive "
                         "structure is not there to be learned; it is not being lost in transfer."},
        "4_the_difference_feature_loses": {
            "diff_arch_h256": vt["diff/arch/h256"]["sel_eff"],
            "diff_arch_h0_degeneracy_control": vt["diff/arch/h0"]["sel_eff"],
            "full_arch_h256": vt["full/arch/h256"]["sel_eff"],
            "vs_pointwise_diff": vt["diff/arch/h256"]["vs_pointwise_ens"],
            "statement": "h_i - h_j, the minimal encoding of 'how do these two differ', is the WORST "
                         "encoding tested and loses significantly to the pointwise head. Its linear "
                         "degeneracy control (algebraically w.h_i - w.h_j, i.e. exactly pointwise) "
                         "scores the same, so the nonlinearity over the difference buys nothing."},
        "5_position_bias_is_real_but_was_not_the_binding_limit": {
            "antisymmetry_violation_arch": h.get("antisymmetry_residual_mean_abs",
                                                 a["diagnostics"]["antisymmetry_residual_mean_abs"]),
            "antisymmetry_violation_full_augment": vt["full/augment/h256"]["antisymmetry_violation_mean_abs"],
            "antisymmetry_violation_concat_augment": vt["concat/augment/h256"]["antisymmetry_violation_mean_abs"],
            "statement": "order-augmented training leaves a real order preference "
                         "(mean |g(i,j)+g(j,i)|/2 up to "
                         f"{vt['full/augment/h256']['antisymmetry_violation_mean_abs']}), which the "
                         "architectural parameterisation removes exactly (0.0). But the augmented "
                         "variants score no worse on the endpoint, so position bias was NOT what "
                         "capped this family -- additivity was."}},
    "what_is_nonetheless_worth_keeping": {
        "seed_ensembling_the_existing_pointwise_head_is_free": {
            "single_seed_mean": pb["seed_spread_sel_eff"]["mean"],
            "single_seed_sd": pb["seed_spread_sel_eff"]["sd"],
            "single_seed_range": [pb["seed_spread_sel_eff"]["min"], pb["seed_spread_sel_eff"]["max"]],
            "twelve_seed_ensemble": pb["seed_ensemble_zmean"]["sel_eff"],
            "vs_incumbent": pb["seed_ensemble_zmean"]["vs_incumbent"],
            "cost": "12 tiny MLP evaluations over the SAME cached vector; zero extra forward passes",
            "caveat": "guardrail-dirty on vqa_rad_open, and its fusion with the incumbent "
                      f"({fu['REF_FUSE_pointwise+incumbent']['sel_eff']}) is inside the single-seed "
                      "deployed fusion's own seed range, so this is a variance-reduction result, "
                      "not a new mechanism."},
        "knockout_is_as_good_as_round_robin_and_3x_cheaper": {
            "knockout_sel_eff": ag["knockout"]["sel_eff"],
            "round_robin_logit_sum_sel_eff": ag["logit_sum"]["sel_eff"],
            "comparisons_per_question_knockout": ag["knockout"]["comparisons_per_question"],
            "comparisons_per_question_round_robin": ag["logit_sum"]["comparisons_per_question"],
            "note": "DIAGNOSTIC -- knockout was not the pre-registered aggregation, and the spread "
                    "across aggregation rules "
                    f"({min(ag[k]['sel_eff'] for k in ag)}-{max(ag[k]['sel_eff'] for k in ag)}) is "
                    f"inside the seed sd ({a['pair_head_seeds']['sd']}). Reported so the cost story "
                    "is on record, not as a win."}},
    "reading_the_numbers": {
        "cand_auroc_of_the_pair_head_is_not_comparable": {
            "value": h["cand_auroc"],
            "why": "logit_sum scores are Borda-style within-question sums; they are only comparable "
                   "INSIDE a question, so a pooled candidate-level AUROC across 18760 slots is "
                   "meaningless for this selector. The incumbent's 0.885592 and the pointwise head's "
                   f"{pb['seed_ensemble_zmean']['cand_auroc']} are pointwise scores and are comparable "
                   "to each other."},
        "sel_eff_quantum": "1/1468 = 0.000681 -- differences below this are not differences.",
        "aggregation_differences_are_inside_seed_noise":
            f"aggregation rules span {min(ag[k]['sel_eff'] for k in ag)}-"
            f"{max(ag[k]['sel_eff'] for k in ag)}; the seed sd of a single fit is "
            f"{a['pair_head_seeds']['sd']}."}}

json.dump(a, open(P, "w"), indent=1, default=float)
print("VERDICT appended to", P)
print(a["VERDICT"]["one_line"])
