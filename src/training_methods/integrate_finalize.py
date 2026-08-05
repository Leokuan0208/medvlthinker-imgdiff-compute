#!/usr/bin/env python3
"""integrate_finalize.py -- assemble the round's single artifact from the stage parts.

  python3 src/training_methods/integrate_finalize.py
writes results/cascade_methods/artifacts/verifarch_integrated_2026-08-04.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G          # noqa: E402
import integrate_lib as IL         # noqa: E402

OUT = os.path.join(G.ROOT, "results/cascade_methods/artifacts/verifarch_integrated_2026-08-04.json")


def load(name):
    return json.load(open(os.path.join(IL.PARTS, name)))


def main():
    ver, cpu, pre, pre2, ev = (load("verify.json"), load("cpuref.json"), load("prereg.json"),
                               load("prereg2.json"), load("eval.json"))
    items = G.load_items()
    nd = [len(set(G.norm(a) for a in it["preds"])) for it in items]
    nsur = [len(set(it["preds"])) for it in items]
    A = ev["arms"]
    hl, dep, incm = A["HEADLINE"], A["deployed_fusion_seed0"], A["incumbent"]
    vsdep = ev["HEADLINE_vs_deployed_fusion"]
    greedy, oracle = 0.4494669509594883, 0.6260127931769722

    art = {
        "what": "ONE deployable selector for open-text best-of-8, assembled from the components "
                "that survived verification, and measured against the CURRENTLY DEPLOYED fusion "
                "(0.806540) rather than only against the incumbent verifier (0.775204).",
        "date": "2026-08-05",
        "endpoint": {
            "definition": "selection efficiency at N=8 = P(pick correct | pool contains a correct "
                          "answer), on 2345 open-text eval questions (slake_open 645 / "
                          "vqa_rad_open 200 / pathvqa_open 1500), 1468 recoverable.",
            "greedy_accuracy": greedy, "oracle_at_8_accuracy": oracle,
            "incumbent_sel_eff": 0.775204, "deployed_fusion_sel_eff": 0.806540,
            "quantum": 1.0 / 1468,
            "quantum_note": "sel_eff moves in steps of 1/1468 = 0.000681; differences smaller "
                            "than that are not differences."},

        # ------------------------------------------------------------------ 1. VERIFICATION
        "verification": {
            "metric_null_test": {"pass": ver["null_test"]["pass"],
                                 "max_abs_deviation": ver["null_test"]["max_abs_deviation"],
                                 "measured": ver["null_test"]["measured"]},
            "trainer_null_test": {
                "what": "the published head and the published deployed fusion refit end-to-end "
                        "in THIS agent's code on CPU at seed 0",
                "head_measured": cpu["published_head"]["measured_sel_eff"],
                "head_published": 0.795640,
                "head_abs_dev": cpu["published_head"]["abs_dev"],
                "fusion_measured": cpu["published_deployed_fusion"]["measured_sel_eff"],
                "fusion_published": 0.806540,
                "fusion_abs_dev": cpu["published_deployed_fusion"]["abs_dev"],
                "pass": cpu["pass"],
                "gotcha_found": "the published recipe STANDARDIZES features with the train mu/sd "
                                "(fit_hidden_head.py:507). Omitting that step gives 0.788147 at the "
                                "same seed -- an 0.0075 error, larger than most effects in this "
                                "round. A first attempt here reproduced 0.788147 and was rejected "
                                "by this very test before anything was built on it.",
                "timing_note": "a CPU fit takes ~15 s on an idle box; the 825 s reported on "
                               "2026-08-04 was contention from four concurrent agents. CPU is "
                               "therefore affordable, and every arm in this round is fit on CPU "
                               "with the published trainer -- so the GPU/TF32 arithmetic gap that "
                               "worried the earlier rounds does not enter here at all."},
            "disjointness_reproved_here": {
                m: {k: v for k, v in ver["disjointness"][m].items()
                    if k in ("train_images", "eval_images", "image_pixel_md5_intersection",
                             "train_rows", "eval_rows", "train_rows_failed", "eval_rows_failed")}
                for m in ver["disjointness"]},
            "sibling_spot_checks": ver["sibling_spot_checks"],
            "real_pairwise_recheck": ver["real_pairwise_recheck"],
        },

        # ------------------------------------------------------------------ 2. PREREGISTRATION
        "preregistration": {
            "protocol": pre["criterion"],
            "head_config_frozen_from": pre["head_config_frozen_from"],
            "head_config": pre["head_config"],
            "Q1_Q2_readout_grid": pre["Q1_Q2_readout"],
            "Q3_combiner_form": pre["Q3_combiner"],
            "Q4_self_consistency_member": pre2["Q4_self_consistency_member"],
            "Q4_decision": pre2["Q4_DECISION"],
            "Q5_grader_frame_member": pre2["Q5_grader_frame_member"],
            "Q5_decision": pre2["Q5_DECISION"],
            "DECISION": pre["PREREGISTERED_DECISION"],
            "eval_visibility": "none. The eval labels were read only after this file was written."},

        # ------------------------------------------------------------------ 3. THE ANSWER
        "headline": {
            "definition": ev["preregistration"]["decision"]["headline_definition"],
            "sel_eff": hl["sel_eff"], "selected_accuracy": hl["acc"],
            "per_ds": hl["per_ds"], "guardrail_clean": hl["guardrail_clean"],
            "contested_sel_eff": hl["contested_sel_eff"], "contested_n": hl["contested_n"],
            "short_answer_sel_eff": hl["short_answer_le3w_sel_eff"],
            "short_answer_n": hl["short_answer_le3w_n"],
            "long_answer_sel_eff": hl["long_answer_gt3w_sel_eff"],
            "long_answer_n": hl["long_answer_gt3w_n"],
            "vs_incumbent_0.775204": {
                "d_sel_eff": hl["d_sel_eff"], "ci": hl["d_sel_eff_ci"],
                "d_acc": hl["d_acc"], "d_acc_ci": hl["d_acc_ci"],
                "d_contested": hl["d_contested"], "ci_contested": hl["d_contested_ci"],
                "d_short": hl["short_answer_le3w_d"], "ci_short": hl["short_answer_le3w_d_ci"],
                "d_long": hl["long_answer_gt3w_d"], "ci_long": hl["long_answer_gt3w_d_ci"]},
            "vs_deployed_fusion_0.806540": {
                "d_sel_eff": vsdep["d_sel_eff"], "ci": vsdep["d_sel_eff_ci"],
                "d_acc": vsdep["d_acc"], "d_acc_ci": vsdep["d_acc_ci"],
                "d_contested": vsdep["d_contested"], "ci_contested": vsdep["d_contested_ci"],
                "d_short": vsdep["d_short_answer_le3w"], "ci_short": vsdep["d_short_answer_le3w_ci"],
                "d_long": vsdep["d_long_answer_gt3w"], "ci_long": vsdep["d_long_answer_gt3w_ci"]},
        },
        "comparators_on_the_same_items": {
            "incumbent": incm, "deployed_fusion_seed0": dep,
            "head_ensemble_alone": A["head_ensemble_alone"],
            "head_ensemble_alone_vs_deployed": ev["head_ensemble_vs_deployed_fusion"]},
        "seed_spread": ev["seed_spread"],
        "ensemble_size_curve_disjoint_blocks": ev["ensemble_size_curve_disjoint_blocks"],
        "diagnostics": ev["diagnostics"],

        # ------------------------------------------------------------------ 4. COST
        "cost": {
            "unit": "full Lingshu-7B forward passes per QUESTION, over and above the 8 sampled "
                    "generations",
            "candidate_dedup": {
                "mean_distinct_normalized_answers": float(np.mean(nd)),
                "mean_distinct_surface_strings": float(np.mean(nsur)),
                "distinct_answer_histogram": {str(k): int(v) for k, v in
                                              sorted(Counter(nd).items())}},
            "incumbent_verifier": {"passes": float(np.mean(nsur)),
                                   "evidence": "the stored scores are constant within every "
                                               "distinct SURFACE string (0 of 8965 such groups "
                                               "carry two different scores), so it scores each "
                                               "distinct surface answer once"},
            "generator_frame_head": {"passes": float(np.mean(nd)),
                                     "note": "one teacher-forced pass per distinct normalized "
                                             "answer to read h_span at layer 21"},
            "deployed_fusion_total": float(np.mean(nsur) + np.mean(nd)),
            "HEADLINE_total": float(np.mean(nsur) + np.mean(nd)),
            "HEADLINE_marginal_cost_over_deployed": {
                "forward_passes": 0.0,
                "what_changes": "the same cached 3584-d vector is scored by k tiny MLPs instead "
                                "of 1. One head is 918,529 params ~ 1.8 MFLOP/candidate; k heads "
                                "over 3.81 candidates is ~10^8 FLOP/question against ~10^12 for "
                                "one 7B VLM pass -- i.e. below measurement noise.",
                "storage": "k x 3.5 MB of head weights"},
            "rejected_alternatives": {
                "real_A_vs_B_round_robin": "13.07 extra full VLM passes/question (6.54 unordered "
                                           "distinct pairs x 2 orders for position debias) -- and "
                                           "it does not help (see verification.real_pairwise_recheck)",
                "grader_frame_second_head": "+%.2f passes/question" % float(np.mean(nd))},
        },

        # ------------------------------------------------------------------ 5. HEADROOM
        "headroom": {
            "greedy_accuracy": greedy, "oracle_at_8_accuracy": oracle,
            "gap_available": oracle - greedy,
            "incumbent_accuracy": incm["acc"], "deployed_fusion_accuracy": dep["acc"],
            "headline_accuracy": hl["acc"],
            "fraction_of_greedy_to_oracle_accuracy_gap_closed": {
                "incumbent": (incm["acc"] - greedy) / (oracle - greedy),
                "deployed_fusion": (dep["acc"] - greedy) / (oracle - greedy),
                "headline": (hl["acc"] - greedy) / (oracle - greedy),
                "note": "this is NOT sel_eff. sel_eff conditions on recoverable pools and is the "
                        "published endpoint; this ratio additionally pays for the fact that the "
                        "greedy answer is better than a random pool slot."},
            "selection_wall_remaining_sel_eff": 1.0 - hl["sel_eff"],
            "coverage_wall": {
                "unrecoverable_fraction": 1.0 - oracle,
                "note": "37.4% of questions have NO correct answer anywhere in the 8-sample pool. "
                        "Closing the remaining selection gap entirely (sel_eff 1.0) would reach "
                        "0.626013; the coverage wall is 4.5x the selection wall and is a "
                        "GENERATOR problem, not a verifier problem."}},
    }

    # ---------------------------------------------------------------- open-text arm effect
    per_ds_acc = {}
    for d in G.EVAL_DS:
        sub = [i for i, it in enumerate(items) if it["ds"] == d]
        per_ds_acc[d] = {"n": len(sub)}
    art["open_text_arm_translation"] = {
        "selected_accuracy": {"incumbent": incm["acc"], "deployed_fusion": dep["acc"],
                              "headline": hl["acc"], "greedy": greedy},
        "d_accuracy_vs_incumbent": {"d": hl["d_acc"], "ci": hl["d_acc_ci"]},
        "d_accuracy_vs_deployed_fusion": {"d": vsdep["d_acc"], "ci": vsdep["d_acc_ci"]},
        "what_this_is_NOT": [
            "It is NOT the open-text arm's end-to-end accuracy. In the deployed cascade the open "
            "arm escalates to the 32B when verifier confidence is low (CLAUDE.md reports 15.81% "
            "SLAKE-open / 12.50% VQA-RAD-open / 35.67% PathVQA-open), and escalation is DRIVEN BY "
            "the selector's own confidence -- change the selector and the escalation set changes. "
            "Only a re-run of the cascade can give the arm's accuracy.",
            "It is NOT the macro headline. The macro number weights 8 cells at 1/8 each and is "
            "produced by src/cascade_methods/macro_average_headline.py from "
            "artifacts/macro_average_headline_2026-07-30.json. Translating requires RE-RUNNING "
            "that script -- this round does not do it and no macro number is quoted here.",
            "The pools are not the same either: this selection pool is 2345 items "
            "(645/200/1500), which is not the MedEvalKit open-cell pool the macro headline uses."],
        "what_can_be_said": "on THIS pool, and on the selection endpoint alone, the accuracy of "
                            "the answer the open arm would return before any escalation moves "
                            "from %.6f (incumbent) / %.6f (deployed fusion) to %.6f."
                            % (incm["acc"], dep["acc"], hl["acc"])}

    dist = ev["seed_spread"]["deployed_recipe_single_seed"]
    curve = ev["ensemble_size_curve_disjoint_blocks"]
    art["headline"]["guardrail_vs_deployed_fusion"] = {
        "clean": all(hl["per_ds"][d] >= dep["per_ds"][d] for d in G.EVAL_DS),
        "per_ds_delta": {d: hl["per_ds"][d] - dep["per_ds"][d] for d in G.EVAL_DS}}
    art["headline"]["what_the_gain_actually_is"] = {
        "deployed_recipe_is_a_lottery": {
            "definition": "rank_avg(incumbent, ONE-seed head) -- the published 0.806540 -- re-run "
                          "at 16 seeds",
            "mean": dist["mean"], "sd": dist["sd"], "range": dist["range"],
            "published_0.806540_is_at_percentile": dist["published_0.806540_percentile"]},
        "expected_value_gain_from_ensembling": {
            "fused_mean_at_k1": curve["k=1"]["fused_mean"],
            "fused_mean_at_k8": curve["k=8"]["fused_mean"],
            "fused_mean_at_k16": curve["k=16"]["fused_mean"],
            "reading": "seed-ensembling is worth ~+0.010 to the head ALONE (%.6f -> %.6f) but "
                       "only ~+0.001 to +0.002 AFTER fusion with the incumbent (%.6f -> %.6f), "
                       "because the fusion was already averaging away part of the head's seed "
                       "noise. Most of the headline's +0.004087 over the published cell is that "
                       "the published cell drew a below-median seed."
                       % (curve["k=1"]["head_ens_mean"], curve["k=8"]["head_ens_mean"],
                          curve["k=1"]["fused_mean"], curve["k=8"]["fused_mean"])},
        "what_is_unambiguously_bought": "determinism. A single-seed deployment lands anywhere in "
                                        "[%.6f, %.6f]; the pre-registered 8-seed ensemble is a "
                                        "fixed artifact at %.6f, guardrail-clean against BOTH the "
                                        "incumbent and the deployed fusion, at zero extra forward "
                                        "passes." % (dist["range"][0], dist["range"][1],
                                                     hl["sel_eff"])}

    art["what_did_not_earn_its_place"] = {
        "principle": "a component is in the deployable stack only if it improves the endpoint "
                     "OVER the components already there, at a cost worth paying.",
        "real_A_vs_B_pairwise_forward_passes": {
            "independently_rechecked_here": True,
            "result": ver["real_pairwise_recheck"],
            "verdict": "EXCLUDED. Re-aggregated by this agent from the stored teacher matrix: on "
                       "the 1345 items it covers, Borda/Copeland/knockout all sit BELOW the "
                       "incumbent on those same items, and fusing it in moves nothing. It costs "
                       "13.07 extra full VLM passes/question."},
        "pairwise_contrast_head_over_cached_vectors": {
            "independently_rechecked_here": False,
            "verdict": "EXCLUDED on the sibling's own evidence, which is internally consistent: "
                       "it ties the pointwise head on identical features (+0.002044 "
                       "[-0.012943, +0.017711]) and 97.93% of its learned comparison matrix is "
                       "an additive (i.e. pointwise) term."},
        "set_aware_listwise_architectures": {
            "independently_rechecked_here": False,
            "verdict": "EXCLUDED: point estimate NEGATIVE vs the pointwise head on identical "
                       "features, and train CV ranked every set-aware form below pointwise "
                       "BEFORE eval was touched."},
        "pool_relative_contrast_FEATURES": {
            "independently_rechecked_here": True,
            "verdict": "EXCLUDED from the headline: reproduced here bit-exact (0.810627, "
                       "dev 3.0e-07) and it is a genuine guardrail repair on vqa_rad_open, but "
                       "it does not separate from the same head on raw features (+0.004768 "
                       "[-0.010218, +0.020436]) and it needs an extra cross-fitted stage. Kept "
                       "as the named fallback if the vqa_rad_open guardrail becomes binding.",
            "rederived_here": ver["sibling_spot_checks"]["cheapcontrast_preregistered"]},
        "self_consistency_and_grader_frame_members": {
            "decided_on_train_CV_only": True,
            "Q4_self_consistency": pre2["Q4_DECISION"],
            "Q5_grader_frame": pre2["Q5_DECISION"]},
    }
    art["deployment_recipe"] = {
        "inputs": "the 8 sampled open-text answers already produced by the open arm",
        "steps": [
            "1. deduplicate the 8 answers by normalized string (mean 3.81 distinct).",
            "2. incumbent LoRA verifier (ckpts/train/lora_verifier_disjoint) scores each distinct "
            "SURFACE answer -- unchanged, this is already deployed.",
            "3. frozen base Lingshu-7B, NO adapter, generator prompt with the candidate as the "
            "assistant turn: read h_span at layer 21 for each distinct normalized answer "
            "(src/training_methods/extract_generator_hidden.py) -- unchanged, already deployed.",
            "4. score that 3584-d vector with the k pre-registered head replicas and average "
            "their within-pool ranks -- THIS IS THE ONLY CHANGE, and it adds no forward pass.",
            "5. rank_avg-fuse the incumbent's ranks with the ensemble's ranks; pick argmax "
            "(first-index tie-break)."],
        "artifacts_needed": ["ckpts/train/lora_verifier_disjoint (existing)",
                             "k head checkpoints (~3.5 MB each) fit by "
                             "src/training_methods/integrate_eval.py",
                             "the train mu/sd standardization vector"],
        "reproduce": ["python3 src/training_methods/integrate_verify.py",
                      "python3 src/training_methods/integrate_cpuref.py",
                      "python3 src/training_methods/integrate_prereg.py --device cpu",
                      "python3 src/training_methods/integrate_prereg2.py",
                      "python3 src/training_methods/integrate_eval.py --seeds 16",
                      "python3 src/training_methods/integrate_finalize.py"]}

    lc = ev["diagnostics"]["learned_combiner_crossfitted_on_eval"]
    sw = ev["diagnostics"]["eval_visible_weight_sweep"]
    art["verdict"] = {
        "one_sentence": "The deployable answer is the fusion that is already deployed, with one "
                        "free change: replace the single-seed generator-frame head with a "
                        "pre-registered 8-seed rank ensemble of the same head. That gives "
                        "sel_eff %.6f (selected accuracy %.6f), guardrail-clean against both the "
                        "incumbent and the deployed fusion, at ZERO extra forward passes -- but "
                        "the gain over the deployed fusion is +%.6f [%.6f, %.6f], which is NOT "
                        "significant, and it should be sold as variance elimination, not as a "
                        "new mechanism."
                        % (hl["sel_eff"], hl["acc"], vsdep["d_sel_eff"],
                           vsdep["d_sel_eff_ci"][0], vsdep["d_sel_eff_ci"][1]),
        "beats_incumbent": True,
        "beats_deployed_fusion": bool(vsdep["d_sel_eff_ci"][0] > 0),
        "no_comparative_component_survived": (
            "All four comparative directions were excluded, two of them re-tested here against "
            "this stack: real A-vs-B forward passes ADDED AS A MEMBER make the headline "
            "significantly WORSE on the items they cover (%.6f -> %.6f, d = %.6f [%.6f, %.6f]) "
            "while costing 13.07 extra VLM passes per question; and pool-relative contrast "
            "features reproduce exactly but do not separate from the same head on raw features."
            % (ev["diagnostics"]["add_real_pairwise_as_fourth_member_on_covered_items"]["headline_on_covered"],
               ev["diagnostics"]["add_real_pairwise_as_fourth_member_on_covered_items"]["headline_plus_pairwise_on_covered"],
               ev["diagnostics"]["add_real_pairwise_as_fourth_member_on_covered_items"]["d"],
               ev["diagnostics"]["add_real_pairwise_as_fourth_member_on_covered_items"]["ci"][0],
               ev["diagnostics"]["add_real_pairwise_as_fourth_member_on_covered_items"]["ci"][1])),
        "the_learned_combiner_question_is_settled_negatively": (
            "Train CV preferred a learned logistic combiner over parameter-free rank averaging "
            "(%.6f vs %.6f) when the two members were the generator- and grader-frame heads. On "
            "eval, with the actual members (incumbent + head ensemble), a combiner cross-fitted "
            "ON EVAL with image-disjoint folds -- i.e. given an advantage no deployable version "
            "could have -- scores %.6f, BELOW the parameter-free fusion (%.6f [%.6f, %.6f]) and "
            "guardrail-dirty. And the eval-visible weight sweep peaks at exactly w = 0.5, the "
            "parameter-free point. There is nothing for a combiner to learn here."
            % (pre["Q3_combiner"]["learned_logistic_combiner_fold_fit"]["cv_sel_eff"],
               pre["Q3_combiner"]["rank_avg_parameter_free"], lc["sel_eff"],
               lc["vs_headline"]["d_sel_eff"], lc["vs_headline"]["d_sel_eff_ci"][0],
               lc["vs_headline"]["d_sel_eff_ci"][1])),
        "weight_sweep_peak_w": sw["best_w"],
        "honest_limits": [
            "The headline instance (seeds 0-7) is 0.810627; the OTHER disjoint 8-seed block of "
            "the same recipe gives 0.807902. The deployable quantity therefore carries a "
            "block-to-block spread of about 0.003, and the +0.004087 over the published cell is "
            "of that size. Nothing here separates from the deployed fusion statistically.",
            "sel_eff moves in quanta of 1/1468 = 0.000681; the expected-value gain from "
            "ensembling after fusion (~+0.0017) is 2-3 quanta.",
            "This is a SELECTION endpoint on a 2345-item pool. It is not the open arm's "
            "end-to-end accuracy and it is not the macro headline; neither was computed here.",
            "The verifier-contamination caveat that gates the open-text arm is unchanged: this "
            "round uses the CLEAN disjoint-trained incumbent throughout, but the open-text "
            "accuracy claim in the paper still carries its own PROVISIONAL status."],
    }

    IL.jdump(art, OUT)
    print("wrote", OUT)
    print(json.dumps({"headline": art["headline"], "cost": art["cost"]["HEADLINE_total"]},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
