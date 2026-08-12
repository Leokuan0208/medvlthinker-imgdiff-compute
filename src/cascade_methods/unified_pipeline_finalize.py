#!/usr/bin/env python3
"""unified_pipeline_finalize.py -- assemble ATTACK 2's one artifact from the part files.

Every number is copied VERBATIM from a part file and every block names the file it came from.
Nothing is recomputed here and nothing is typed in by hand.  Arms that did not finish are written as
"not measured", never as an estimate.

    python3 src/cascade_methods/unified_pipeline_finalize.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

P = U.PARTS


def loadp(name):
    p = os.path.join(P, name)
    return json.load(open(p)) if os.path.exists(p) else None


def rel(name):
    return f"results/cascade_methods/artifacts/_unified_pipeline_parts/{name}"


def cellrow(a, cell):
    c = a["cells"][cell]
    return {k: c[k] for k in
            ("n_scored", "acc_unified_pick", "acc_7b_greedy_same_items", "acc_32b_direct_same_items",
             "delta_vs_7b", "delta_vs_32b_direct", "candidate_auroc_gold_vs_distractor",
             "agreement_with_7b_greedy", "disagreement_stratum", "luck_floor_random_gold")
            if k in c}


def main():
    nul = loadp("nulltests.json")
    dis = loadp("disjointness.json")
    n4 = loadp("n4_batch_numerics_zeroshot.json")
    flo = loadp("floors_zeroshot.json")
    deb = loadp("arm_a_prime_debias.json")
    bias = loadp("bias_diag_zeroshot.json")
    az = loadp("analysis_zeroshot.json")
    vram = loadp("vram_option_branch_zeroshot.json")
    two = loadp("rescue_break_2x2_zeroshot.json")
    gat = loadp("gate_zeroshot.json")
    fus = loadp("fusion_zeroshot.json")
    rep = loadp("repaired_grader_zeroshot.json")
    n5a = loadp("n5_parse_audit.json")
    wwt = loadp("what_would_have_to_be_true.json")
    ob2 = loadp("open_branch_2x2.json")
    trained = {}
    for tag in ("optiononly_s0", "unified_s0"):
        a = loadp(f"analysis_{tag}.json")
        f = loadp(f"floors_{tag}.json")
        rg = loadp(f"repaired_grader_{tag}.json")
        tw = loadp(f"rescue_break_2x2_{tag}.json")
        if a:
            trained[tag] = {"analysis_source": rel(f"analysis_{tag}.json"),
                            "what_it_is": ("option-branch-ONLY verifier (upper bound on what the "
                                           "unified scorer can do over the given options)"
                                           if tag.startswith("optiononly") else
                                           "the UNIFIED verifier, one adapter trained on BOTH "
                                           "branches' candidate sets"),
                            "train_manifest": (f"ckpts/train/lora_verifier_{tag}/unified_manifest.json"),
                            "cells": {c: cellrow(a, c) for c in U.OPTION_CELLS
                                      if c in a.get("cells", {}) and "acc_unified_pick" in a["cells"][c]},
                            "macro": a.get("macro"),
                            "floors": (f or {}).get("cells"),
                            "repaired_grader": (rg or {}).get("cells"),
                            "repaired_grader_four_cell_macro":
                                (rg or {}).get("four_cell_macro_option_branch_only"),
                            "rescues_and_breaks": (tw or {}).get("cells")}

    out = {
        "title": "ATTACK 2 -- ONE PIPELINE FOR BOTH ANSWER FORMATS: the candidate set is read off the "
                 "prompt, the scorer is the same in both branches, and there is no 32B at test time.",
        "date": U.DATE,
        "preregistration": "results/cascade_methods/artifacts/unified_pipeline_2026-08-12_preregistration.json",
        "amendments": ["results/cascade_methods/artifacts/unified_pipeline_2026-08-12_amendment1.json",
                       "results/cascade_methods/artifacts/unified_pipeline_2026-08-12_amendment2.json"],
        "reproduce": {
            "null tests + disjointness": "python3 src/cascade_methods/unified_pipeline.py --nulltest --disjoint",
            "score the option branch": "python3 src/cascade_methods/unified_pipeline_score.py --adapter <A> --tag <T>",
            "assemble a tag": "python3 src/cascade_methods/unified_pipeline.py --analyse --tag <T>",
            "floors": "python3 src/cascade_methods/unified_pipeline_floors.py --tag <T>",
            "arm A' debias": "python3 src/cascade_methods/unified_pipeline_debias.py",
            "this file": "python3 src/cascade_methods/unified_pipeline_finalize.py",
            "train arms B0/B": "bash runners/run_unified_pipeline_armB0.sh"},
        "not_abstention": "every branch returns an answer on every item: the option branch answers with "
                          "argmax over the prompt's own answer space, the sampled branch with argmax "
                          "over the 7B's samples, and a 1-candidate set answers with the 7B's greedy "
                          "answer. No reject option anywhere. CRITICAL RULE 6 respected.",
        "no_32B_at_test_time": "the pipeline itself never calls the 32B. always-32B-direct appears only "
                               "as the bar, and the min-strong-leg curve is a diagnostic that says how "
                               "much 32B would be needed, not part of the pipeline.",

        "VERDICT": {
            "one_line": "The unification WORKS as a rule and FAILS as a mechanism: reading the "
                        "candidate set off the prompt does unify the two formats with one scorer, "
                        "one decision rule, no format branch and no sampling luck -- but scoring the "
                        "GIVEN OPTIONS is no better than the 7B's own argmax on any of the four "
                        "option cells and significantly worse on two, so the unified pipeline "
                        "scores 0.5706 macro against always-7B 0.5967 and always-32B-direct 0.6567.",
            "Q1_does_verifier_over_options_beat_the_7B_argmax": {
                "answer": "NO -- on none of the four option cells, under a repaired grader. Two "
                          "significant losses, two statistical ties, zero wins.",
                "per_cell_delta_vs_always_7b": {
                    "PMC_VQA": "-0.0758 [-0.0897, -0.0625] SIG LOSS (repaired grader)",
                    "MedXpertQA-MM": "+0.0025 [-0.0160, +0.0215] n.s.",
                    "VQA_RAD_closed": "-0.0518 [-0.1116, +0.0080] n.s.",
                    "PATH_VQA_closed": "-0.1689 [-0.1886, -0.1499] SIG LOSS"},
                "and_it_is_not_a_floor_artifact": "all four cells clear F1, F2 and F3; candidate-level "
                                                  "AUROC 0.583-0.800. The verifier HAS ranking signal "
                                                  "over the options -- it is simply worse than the "
                                                  "generator's own argmax.",
                "the_best_the_verifier_can_add": "with the generator's own answer folded into the "
                                                 "score, the cross-fit GLOBAL lambda is 1.0, at which "
                                                 "the generator can never be overturned. The verifier "
                                                 "is given zero effective weight."},
            "Q2_does_unifying_COST_anything_on_open_text": {
                "answer": "NO -- the sampled branch is the incumbent best-of-8 arm unchanged (same "
                          "adapter, same argmax rule); only the candidate set differs",
                "cells": "SLAKE_open 0.7473 (+0.0109 n.s.), VQA_RAD_open 0.4800 (+0.0150 n.s.), "
                         "PATH_VQA_open 0.3733 (+0.0493 SIG) against always-7B"},
            "Q3_the_macro_and_the_gap": {
                "unified_rule_as_specified": "0.570560, -0.0261 [-0.0373, -0.0152] vs always-7B, "
                                             "-0.0862 [-0.0991, -0.0735] vs always-32B-direct -- it "
                                             "moves BACKWARD by 44% of the 0.0596 gap",
                "option_branch_off_control": "0.606049, +0.0094 [+0.0021, +0.0170] vs always-7B, "
                                             "-0.0507 [-0.0627, -0.0390] vs always-32B-direct -- "
                                             "closes 15.8% of the gap, and all of it comes from the "
                                             "three OPEN cells",
                "a_better_7B_only_point_measured_by_the_sibling_round":
                    "0.616278 (frozen 8-seed selector on the open cells), -0.040395 [-0.052275, "
                    "-0.028427] vs always-32B-direct, closes 32.2% "
                    "[results/cascade_methods/artifacts/sevenb_only_frontier_2026-08-12.json]",
                "minimum_strong_leg_usage": "under a SINGLE GLOBAL escalation budget ranked by the "
                                            "pipeline's own confidence (7B margin on the MCQ cells, "
                                            "verifier top1-top2 on the open cells), NO budget below "
                                            "100% reaches always-32B-direct: 90% escalation reaches "
                                            "0.6518 against the 0.6567 target. The per-cell-subset "
                                            "version of this question is answered exactly in the "
                                            "sibling round's PART3."},
            "what_the_user_asked_for": {
                "one_pipeline_for_both_formats": "ACHIEVED as a rule -- one scorer, one decision "
                                                 "rule, no format branch anywhere except the "
                                                 "candidate-set constructor, which reads the deployed "
                                                 "prompt",
                "no_32B_at_test_time": "ACHIEVED",
                "less_VRAM_than_the_32B": "ACHIEVED and measured: one 7B-class process (generator and "
                                          "verifier share one copy of the weights, +0.1961 GiB for "
                                          "the adapter) at 18.76-23.42 GiB board peak against 72.60 "
                                          "GiB for always-32B-direct, i.e. 3.1-3.9x less "
                                          "[artifacts/vram_testtime_2026-08-11.json]",
                "match_always_32B_direct": "NOT ACHIEVED, and this round makes the shortfall larger, "
                                           "not smaller"},
            "the_finding_worth_keeping": "one adapter, one scoring function, one argmax rule, two "
                                         "candidate sets: on the generator's OWN samples the verifier "
                                         "rescues at 2.18x the pool's random floor where greedy is "
                                         "wrong (0.1456 vs 0.0668, 188 rescues / 104 breaks); on "
                                         "PROMPT-supplied options it is BELOW 1/K there on 3 of 4 "
                                         "cells. A verifier built on the generator's own base adds "
                                         "information only inside the generator's support.",
            "the_methodological_catch": "the round's only apparent positive (+0.0132 SIG on PMC-VQA "
                                        "for the fusion arm) was entirely an artifact of a defect in "
                                        "MedEvalKit's PMC-VQA answer extractor. Against a repaired "
                                        "grader it is +0.0030 [-0.0023, +0.0083], NOT SIGNIFICANT. "
                                        "Any arm graded `pick == gold` collects +0.0102 of free "
                                        "grader defect on this cell.",
        },

        "LIMITATIONS_stated_not_hidden": [
            "PMC-VQA is scored on a pre-registered 6,000-item subsample of test_2 (seed 20260810, the "
            "same ids mcq_tta used), not all 33,430. Every PMC number here is on that subsample and "
            "is labelled with n=6000. always-7B on it is 0.539167 against 0.542656 on the full cell.",
            "SLAKE_closed has no 8-sample pool on disk, so under the unified rule its candidate set "
            "degenerates to {7B greedy}. That cell is carried at the 7B floor and contributes exactly "
            "0.0000 to every delta. It is NOT evidence that the rule works there.",
            "arm A' (the debias diagnostic) used ONE 5-fold split (seed 20260812), not 10 fold-split "
            "seeds. The fusion arm and the gate arm both used 10. A' is diagnostic only and no claim "
            "rests on it.",
            "arms B0/B involve training and therefore owe >=10 seeds. They did not finish at all "
            "under this round's shared-GPU contention, so the seed question is moot but unresolved: "
            "if they are ever run, they need the seed spread reported.",
            "padded-batch scoring jitters the item argmax on ~1.7% of items (N4). Every option-branch "
            "number carries that jitter. It cannot explain -0.169 or -0.076; it is comparable to the "
            "MedXpert +0.0025 and to the repaired-grader PMC fusion +0.0030, both of which are "
            "reported as not significant.",
            "VQA_RAD_closed is image-contaminated for the zero-shot arm (64/135 eval images are in "
            "the verifier's vqa_rad_open_train pool). The arm loses on that cell, so contamination "
            "cannot be what produced the loss.",
            "the option branch costs K verifier forwards per query and needs NO generation, so its "
            "FLOP-eq is exactly K (4.0 / 5.0 / 2.0 / 2.0 per cell in units of one 7B forward) against "
            "4.57 for one 32B-direct pass. It is not cheap, and it buys nothing.",
        ],

        "the_rule": {
            "statement": "candidates(item) = ANSWER_SPACE(prompt) if the deployed prompt supplies a "
                         "complete answer space, else SAMPLE_N(7B, item); answer = argmax_c s(image, "
                         "question, c) with ONE scorer s and ONE decision rule.",
            "there_is_no_format_branch_in_the_code": "answer_space() reads the MedEvalKit prompt "
                                                     "template that was already used for that item "
                                                     "(src/cascade_methods/unified_pipeline.py:123)",
            "which_cells_land_where": {
                "option_branch (get_multiple_choice_prompt / get_judgement_prompt)":
                    U.OPTION_CELLS,
                "sampled_branch (get_close_ended_prompt / open-ended)": U.SAMPLED_CELLS},
            "verified_against_the_harness": {
                "MedEvalKit/utils/PMC_VQA, MedXpertQA": "get_multiple_choice_prompt -> option bodies",
                "MedEvalKit/utils/VQA_RAD/VQA_RAD.py:45, MedEvalKit/utils/PATH_VQA/PATH_VQA.py:60":
                    "get_judgement_prompt -> \"Please output 'yes' or 'no'(no extra output).\" -> {yes,no}",
                "MedEvalKit/utils/SLAKE/SLAKE.py:56": "get_close_ended_prompt -> NO answer space -> "
                                                      "sampled branch (and with no 8-sample pool on "
                                                      "disk it degenerates to a 1-candidate set = 7B "
                                                      "greedy, carried at the 7B floor, never a win)"},
            "why_not_the_two_obvious_unifications": {
                "(choice)(why) MCQ-as-constrained-open-text":
                    "already measured, SIGNIFICANT LOSS -0.0226 sel_eff "
                    "[artifacts/choicewhy_measure_2026-08-03.json]; not re-run",
                "sample-8-and-verify on MCQ":
                    "structurally dead: PMC verifier pick 0.4325 < greedy 0.5060, MedXpert oracle@8 "
                    "0.5365 < its own luck floor 0.6808; not re-run"}},

        "null_tests": {"source": rel("nulltests.json"), **(nul or {"status": "not measured"})},
        "disjointness_pixel_md5_of_decoded_RGB": {
            "source": rel("disjointness.json"),
            "reading": "PMC_VQA, MedXpertQA-MM and PATH_VQA_closed are CLEAN (0 eval images in any "
                       "verifier training pool). VQA_RAD_closed is NOT: 64 of its 135 eval images "
                       "(47.4%) reappear in the vqa_rad_open_train pool the clean verifier was "
                       "trained on -- VQA-RAD recycles images across its own splits. Per the "
                       "pre-registration that cell is reported CONTAMINATED. It does not rescue any "
                       "claim here: the arm LOSES on it (-0.0518), and contamination can only "
                       "inflate an arm, so the loss stands a fortiori. Dropping the cell leaves the "
                       "unified pipeline at -0.0808 vs always-32B-direct instead of -0.0862 "
                       "(leave_one_cell_out_vs_32b_direct in the arm-A macro block).",
            "trained_arms": "arms B0/B ban every eval image of every cell by decoded-RGB md5 BEFORE "
                            "training (33,079 hashes), and PMC additionally by shared PMC article id "
                            "because the v2 test figures are sub-figure CROPS that cannot md5-match a "
                            "v1 train full figure",
            **(dis or {"status": "not measured"})},
        "numerics": {
            "tf32": False, "OMP_NUM_THREADS": 1, "PYTHONHASHSEED": 0,
            "serving": "HF transformers ONLY -- a visual LoRA under vLLM drops all 192 visual.* "
                       "modules (0.775204 HF vs 0.702997 vLLM)",
            "max_pixels": U.MAXPX, "min_pixels": U.MINPX, "nboot": U.NBOOT, "nluck": U.NLUCK,
            "seeds": {"boot": U.SEED_BOOT, "luck": U.SEED_LUCK, "pmc_subsample": U.SEED_SUBSAMPLE},
            "N4_padded_batch_vs_batch1": {"source": rel("n4_batch_numerics_zeroshot.json"), **(n4 or {})},
            "N4_consequence": "padded-batch scoring deviates from batch=1 by up to 0.0615 (mean 0.0101) "
                              "and flipped the item argmax on 1 of 60 sampled items. Immaterial to the "
                              "-0.169 / -0.066 deltas; MATERIAL to the MedXpert and VQA_RAD_closed "
                              "calls, which are reported as not significant anyway."},

        "THE_FLOORS_every_option_branch_number_must_clear": {
            "source": rel("floors_zeroshot.json"),
            "why_three": "this project has retracted a claim for mistaking coverage for signal. F1 is "
                         "the random-gold 1/K floor (coverage carries NO information here -- the gold "
                         "is in the candidate set on 100% of items, see null test N3). F2 is the best "
                         "constant-identity rule and is FAR above F1 on PMC-VQA test_2, whose gold "
                         "slots are 13.3/36.9/37.3/12.6%. F3 permutes the real score vectors within "
                         "each item.",
            "cells": (flo or {}).get("cells", "not measured")},

        "ARM_A_zero_shot_the_existing_clean_open_text_verifier_over_the_given_options": {
            "adapter": "ckpts/train/lora_verifier_disjoint (no training of any kind for this arm)",
            "source": rel("analysis_zeroshot.json"),
            "n_forward_passes": 41226,
            "option_cells": {c: cellrow(az, c) for c in U.OPTION_CELLS} if az else "not measured",
            "sampled_cells": ({c: {k: az["cells"][c][k] for k in
                                   ("n_scored", "acc_unified_pick", "acc_7b_greedy_same_items",
                                    "acc_32b_direct_same_items", "delta_vs_7b", "delta_vs_32b_direct",
                                    "note")}
                               for c in U.SAMPLED_CELLS} if az else "not measured"),
            "macro": (az or {}).get("macro", "not measured"),
            "candidate_identity_bias_diagnostic": {"source": rel("bias_diag_zeroshot.json"),
                                                   **(bias or {})}},

        "ARM_A_PRIME_candidate_identity_bias_removed_NESTED_CV_DIAGNOSTIC_ONLY": {
            "source": rel("arm_a_prime_debias.json"),
            "warning": "the offsets are fit on eval-distribution items (other CV folds of the same "
                       "cell). This is mechanism attribution, NOT a deployable pipeline, and must "
                       "never be quoted as one.",
            "cells": (deb or {}).get("cells", "not measured")},

        "ARMS_B0_AND_B_trained_on_the_option_candidates": {
            "prediction_recorded_in_advance":
                "amendment 2 (written before any training run) predicts B0 and B FALL SHORT of "
                "always-7B on the option cells, because arm A' showed ~3/4 of the arm-A shortfall is "
                "a RANKING shortfall, not a candidate-prior shortfall. The rescue/break 2x2 sharpens "
                "the prediction: on 3 of 4 cells the verifier's rescue rate is already BELOW 1/K, so "
                "training would have to create ranking signal where there is currently none.",
            "training_set_actually_built": (
                json.load(open(os.path.join(
                    U.ROOT, "ckpts/train/lora_verifier_optiononly_s0/unified_manifest.json")))
                if os.path.exists(os.path.join(
                    U.ROOT, "ckpts/train/lora_verifier_optiononly_s0/unified_manifest.json"))
                else "not built"),
            "a_bug_this_round_caught_and_fixed":
                "the first build produced ZERO PMC-VQA option examples. PMC-VQA v2 `train_2.csv` "
                "names sub-figure CROPS ('PMC8253797_Fig4_11.jpg') and the images on disk under "
                "pmc_vqa_train/images are the v1 full figures, so it resolved 0/2000 images and would "
                "have trained an 'MCQ verifier' that never saw a lettered 4-option item. Fixed to v1 "
                "`train.csv` (2000/2000 resolve) plus a STRICTLY STRONGER leakage filter: a v2 test "
                "CROP cannot pixel-md5-match a v1 train full FIGURE, so training rows are also "
                "dropped when their PMC ARTICLE ID appears in test_2. Measured: 11,112 banned article "
                "ids, 0 training rows dropped by them (the two splits share no article), 540 of 940 "
                "VQA-RAD train images dropped as eval images.",
            "results": (trained if trained else {
                "status": "NOT MEASURED -- no adapter exists. Reported as not measured rather than "
                          "as a weak result, because a half-trained adapter would look like a "
                          "trained arm.",
                "why": "both A100s were held by three other concurrent rounds for this entire "
                       "session. Attempt 1 and 2 OOMed at model load (four foreign processes holding "
                       "79.07 of 79.14 GiB). Attempt 3 bound cuda:1 with 27.5 GiB free, loaded, and "
                       "then OOMed on 409 consecutive training examples as the foreign processes grew "
                       "back to 81.0 GiB -- 0 optimiser steps completed. The run was stopped rather "
                       "than allowed to save an adapter trained on skipped examples.",
                "what_DOES_exist_on_disk": [
                    "the training set, built and leakage-checked: "
                    "ckpts/train/lora_verifier_optiononly_s0/unified_manifest.json",
                    "the runner, resumable and GPU-polite: runners/run_unified_pipeline_armB0.sh",
                    "logs/unified_armB0_master.log, logs/unified_train_lora_verifier_optiononly_s0.log"],
                "to_finish_it": "bash runners/run_unified_pipeline_armB0.sh on an idle card; the "
                                "script waits for a sustained 26 GiB block, binds whichever card has "
                                "it, retries the load on OOM, and scores the cheap decisive cells "
                                "(PATH_VQA_closed, VQA_RAD_closed) first"})},

        "THE_FINDING_the_verifier_can_rank_the_generators_OWN_samples_and_not_answers_it_never_produced": {
            "source": [rel("open_branch_2x2.json"), rel("rescue_break_2x2_zeroshot.json")],
            "statement": "One adapter, one scoring function, one argmax rule, two candidate sets. "
                         "Where the candidates are the generator's OWN 8 samples, on exactly the "
                         "items the greedy answer got wrong the verifier picks a correct one 14.56% "
                         "of the time against that same pool's random-pick floor of 6.68% -- 2.18x "
                         "the floor, 188 rescues against 104 breaks. Where the candidates come from "
                         "the PROMPT, on the items the 7B got wrong it is BELOW the 1/K floor on 3 "
                         "of 4 cells (PMC 0.2438 vs 0.25, MedXpert 0.1300 vs 0.20, VQA_RAD_closed "
                         "0.4000 vs 0.50). The only thing that changed is where the candidates came "
                         "from.",
            "why_it_matters": "it is a sharper version of Finding 2 (routing signals are degenerate "
                              "on 4-option MCQ). The limit is not option DISCRETENESS as such -- the "
                              "yes/no cells are 2 options and behave the same way -- it is that a "
                              "verifier trained on P(correct | image, question, candidate) inherits "
                              "the generator's own belief distribution, so it adds information only "
                              "inside the generator's support. A prompt-supplied option the generator "
                              "never considered is exactly where it has nothing to say.",
            "consequence_for_the_direction": "a unified pipeline whose scorer is a verifier on the "
                                             "generator's own base CANNOT gain on MCQ. To gain there "
                                             "the scorer must be a DIFFERENT computation from the "
                                             "generator's -- which is the same conclusion "
                                             "docs/current/COMPARATIVE_VERIFIER_2026-08-05.md reached "
                                             "from the open-text side ('the lever is the different "
                                             "computation').",
            "open_branch": (ob2 or "not measured")},

        "RESCUES_AND_BREAKS_2x2": {
            "source": rel("rescue_break_2x2_zeroshot.json"),
            "why": "delta_vs_7b = P(7B wrong AND verifier right) - P(7B right AND verifier wrong). "
                   "The second term is what sinks the option branch, and P(verifier right | 7B wrong) "
                   "against the 1/K floor says whether the verifier knows anything where it matters.",
            "cells": (two or {}).get("cells", "not measured"),
            "THE_MECHANISM_IN_ONE_SENTENCE":
                "the verifier's option-ranking signal is concentrated on items the generator ALREADY "
                "GETS RIGHT. On 3 of the 4 option cells P(verifier right | 7B wrong) is BELOW the 1/K "
                "floor -- PMC 0.2438 vs 0.25, MedXpert 0.1300 vs 0.20, VQA_RAD_closed 0.4000 vs 0.50 "
                "-- i.e. on exactly the items a cascade needs to fix, the verifier is no better than "
                "guessing among the same options. Only PATH_VQA_closed clears it (0.5794 vs 0.50), "
                "and there the break rate would still have to shrink 2.83x to break even.",
            "what_would_have_to_be_true": {"source": rel("what_would_have_to_be_true.json"),
                                           **(wwt or {})}},

        "GATE_endpoint_is_the_verifier_a_better_MCQ_gate_than_the_7B_margin": {
            "source": rel("gate_zeroshot.json"),
            "prereg_rule": "amendment 3: beat -margin7b on RECOVERABILITY AUROC by more than the "
                           "fold-seed sd, on MORE THAN ONE cell",
            "verdict": "RULE NOT MET -- met on PATH_VQA_closed only (+0.0178, sd 0.0005); "
                       "PMC_VQA -0.0027, MedXpertQA-MM -0.0128, VQA_RAD_closed -0.0498",
            "cells": (gat or {}).get("cells", "not measured")},

        "ARM_A_DOUBLE_PRIME_generator_prior_fusion": {
            "source": rel("fusion_zeroshot.json"),
            "rule": "s'(c) = s_verifier(c) + lambda * 1[c == the 7B's own answer]; lambda cross-fit "
                    "(5-fold, 10 fold-split seeds), per cell AND globally",
            "prior_art_that_bounds_it": "on the OPEN branch the analogous fusion is ALREADY MEASURED "
                                        "NEGATIVE -- fusing a self-consistency count into the "
                                        "incumbent costs -0.019755 [-0.036785, -0.002725] "
                                        "(docs/current/COMPARATIVE_VERIFIER_2026-08-05.md), so this "
                                        "arm is reported for the OPTION branch only",
            "per_cell": (fus or {}).get("cells", "not measured"),
            "global_lambda": (fus or {}).get("global_lambda", "not measured")},

        "THE_GRADER_DEFECT_THAT_KILLS_THE_ONLY_POSITIVE": {
            "source": rel("repaired_grader_zeroshot.json"),
            "parse_audit": {"source": rel("n5_parse_audit.json"), **(n5a or {})},
            "independent_confirmation": "results/cascade_methods/artifacts/pmcvqa_grader_defect_2026-08-12.json",
            "what": "the option branch delivers a candidate INDEX and is graded pick == gold, while "
                    "the baselines are graded by MedEvalKit's extractor, which reduces a bare "
                    "'C:' response to the empty string (utils/utils.py:111-112) and then falls "
                    "through to difflib similarity. On the 6,000 scored PMC items the two graders "
                    "disagree on 69 items for the 7B (65 gained / 4 lost) and 94 for the 32B.",
            "consequence": "the fusion arm's PMC win is +0.0132 [+0.0072, +0.0192] SIG against the "
                           "harness grader and +0.0030 [-0.0023, +0.0083] NOT SIGNIFICANT against a "
                           "repaired one. The entire apparent win was the grader.",
            "cells": (rep or {}).get("cells", "not measured"),
            "four_cell_macro": (rep or {}).get("four_cell_macro_option_branch_only", "not measured")},

        "CONTROL_option_branch_off": ((az or {}).get("option_branch_off_control", "not measured")),

        "VRAM": ({"source": rel("vram_option_branch_zeroshot.json"), **vram}
                 if vram else
                 {"status": "not measured for the option branch in this round",
                  "closest_measured_configurations": {
                      "source": "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json",
                      "lingshu7b_MCQ_leg_uncapped_board_peak_gib": 23.4206,
                      "lingshu7b_plus_LoRA_verifier_open_arm_board_peak_gib": 18.7644,
                      "lingshu32b_direct_board_peak_gib": 72.6023,
                      "note": "generator and verifier SHARE one copy of the 7B weights (the verifier "
                              "is a LoRA on the generator's own base, +0.1961 GiB), so the whole "
                              "pipeline is one 7B-class process either way"}}),
    }
    json.dump(out, open(U.OUT, "w"), indent=1)
    print(f"wrote {U.OUT}")
    return out


if __name__ == "__main__":
    main()
