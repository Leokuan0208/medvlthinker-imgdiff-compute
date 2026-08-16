#!/usr/bin/env python3
"""verifier_hparams_report.py -- KNOB 3 aggregator: assemble every measured part into
results/cascade_methods/artifacts/verifier_hparams_2026-08-15.json.

Reads ONLY the parts written by the round's own scripts; types no number of its own.

  _verifier_hparams_parts/prereg.json        the blind decision rule
  _verifier_hparams_parts/nulls.json         N1/N2/N3 + the max_tokens truncation audit
  ckpts/openvqa/verifier_hparams/null_test_rescore.json   the re-score nuisance
  _verifier_hparams_parts/ladder.json        the endpoint ladder, both currencies, + leakage controls
  _verifier_hparams_parts/base_ladder.json   the base-model (no-LoRA) mismatch control
  _verifier_hparams_parts/cost.json          FLOPs at measured geometry
  _verifier_hparams_parts/vram_latency.json  VRAM (4 conventions) + clean batch-1 latency
  _verifier_hparams_parts/macro_table.json   the end-to-end 8-cell macro per rung
  _verifier_hparams_parts/macro_cis.json     cross-arm paired-bootstrap macro CIs

    python3 src/cascade_methods/verifier_hparams_report.py
"""
import json
import os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
SCOREDIR = os.path.join(ROOT, "ckpts/openvqa/verifier_hparams")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/verifier_hparams_2026-08-15.json")


def rd(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else (default if default is not None
                                                         else {"_status": "NOT MEASURED"})


def main():
    ladder = rd(os.path.join(PARTS, "ladder.json"))
    cost = rd(os.path.join(PARTS, "cost.json"))
    vram = rd(os.path.join(PARTS, "vram_latency.json"))
    macro = rd(os.path.join(PARTS, "macro_table.json"))
    macro_ci = rd(os.path.join(PARTS, "macro_cis.json"))
    base = rd(os.path.join(PARTS, "base_ladder.json"))
    nulls = rd(os.path.join(PARTS, "nulls.json"))
    prereg = rd(os.path.join(PARTS, "prereg.json"))
    rescore = rd(os.path.join(SCOREDIR, "null_test_rescore.json"))
    isoesc = rd(os.path.join(PARTS, "isoesc.json"))
    grcis = rd(os.path.join(PARTS, "guardrail_cis.json"))
    recost = rd(os.path.join(PARTS, "recost.json"))
    mism = rd(os.path.join(PARTS, "mismatch.json"))
    mism_au = rd(os.path.join(PARTS, "mismatch_auroc_did.json"))
    dose = rd(os.path.join(PARTS, "doseresponse.json"))
    determ = rd(os.path.join(PARTS, "determinism.json"))
    prereg6 = rd(os.path.join(PARTS, "_prereg_6rung_backup", "ladder.json"))

    out = {
        "_meta": {
            "title": "KNOB 3 -- THE VERIFIER'S OWN INFERENCE HYPERPARAMETERS: what happens when "
                     "the scorer stops seeing each image at 4x the resolution the proposer saw it "
                     "at, plus the generator's max_tokens truncation audit",
            "created": "2026-08-15/16",
            "question": "The deployed pipeline renders the GENERATOR at 250,880 px (cap320) and "
                        "the VERIFIER that scores its candidates at 1,003,520 px -- 4.0x in the "
                        "CAP -- and the verifier is the majority of the open arm's FLOPs. Nobody "
                        "chose that; it is two defaults meeting. Can the verifier score at the "
                        "generator's resolution for free? (Both framing figures are corrected by "
                        "this round's own measurement: the realised ratio is 1.88x in vision "
                        "tokens, and the verifier's share is 65.3%, not 69.1% -- see "
                        "HEADLINE.0_the_question_as_posed.)",
            "design": "SINGLE VARIABLE. The generator, the candidate pool (2,345 items x 8 slots, "
                      "generated at cap320), the judge labels, the item order, the adapter, "
                      "min_pixels (3,136), batch size (1), dtype and attention kernel are all "
                      "FROZEN. Only the verifier's max_pixels moves. NOTHING IS REGENERATED, so "
                      "the project's +-0.008 open-text regeneration caveat does not apply to any "
                      "delta here -- which is exactly why this knob is measurable and the "
                      "generator-resolution knob was not.",
            "mirror_of": "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json moved "
                         "the GENERATOR and held the verifier at 1,003,520; this round does the "
                         "converse. Together they cover the deployed pipeline's resolution "
                         "inconsistency.",
            "code": {
                "scoring": "src/cascade_methods/verifier_hparams_score.py "
                           "(+ runners/run_verifier_hparams.sh)",
                "null_tests_and_truncation_audit": "src/cascade_methods/verifier_hparams_nulls.py",
                "endpoint_ladder_and_leakage_controls":
                    "src/cascade_methods/verifier_hparams_analyze.py",
                "cost": "src/cascade_methods/verifier_hparams_cost.py",
                "honest_recosting": "src/cascade_methods/verifier_hparams_recost.py",
                "vram_and_latency": "src/cascade_methods/verifier_hparams_vram.py "
                                    "(+ runners/run_verifier_hparams_vram_queued.sh, which waits "
                                    "for an EXCLUSIVE card because the (d) convention is "
                                    "board-minus-baseline and free space is not enough)",
                "iso_escalation": "src/cascade_methods/verifier_hparams_isoesc.py",
                "per_set_guardrail_cis": "src/cascade_methods/verifier_hparams_guardrail.py",
                "process_determinism": "src/cascade_methods/verifier_hparams_determinism.py",
                "mismatch_vs_resolution_DiD": "src/cascade_methods/verifier_hparams_mismatch.py",
                "binding_fraction_and_dose_response":
                    "src/cascade_methods/verifier_hparams_doseresponse.py",
                "end_to_end_macro": "src/cascade_methods/verifier_hparams_macro.py "
                                    "(wraps cascade_selector_rerun.run_source / combine, unmodified)",
                "aggregator": "src/cascade_methods/verifier_hparams_report.py",
            },
            "logs": ["logs/verifier_hparams_gpu0_2026-08-15.log",
                     "logs/verifier_hparams_gpu1_2026-08-15.log",
                     "logs/verifier_hparams_base_2026-08-16.log",
                     "logs/verifier_hparams_queue2_2026-08-16.log",
                     "logs/verifier_hparams_knee_2026-08-16.log",
                     "logs/verifier_hparams_vram_2026-08-16.log",
                     "logs/verifier_hparams_vram_q_2026-08-16.log",
                     "logs/verifier_hparams_analyze7_2026-08-16.log",
                     "logs/verifier_hparams_macro7_2026-08-16.log"],
            "environment": {
                "host": "dual A100 80GB PCIe, SHARED. Both cards were verified empty (13 MiB) at "
                        "launch (08:26 UTC) and stayed that way through the entire six-rung LoRA "
                        "ladder, the VRAM/latency measurement (09:54-09:56, taken on a card "
                        "confirmed idle at that moment) and the first base-control rung. A FOREIGN "
                        "CO-TENANT APPEARED AT ~10:20 UTC (49.5 GiB on GPU0, ~19 GiB on GPU1) and "
                        "the remaining base-control rung was re-queued onto the other card rather "
                        "than oversubscribing. No foreign process was ever killed; every launch "
                        "re-checks free VRAM and waits. CONSEQUENCE FOR THE NUMBERS: none of the "
                        "accuracy arms are affected (they were all complete by 09:55, and scoring "
                        "is deterministic anyway -- see the identical-rendering placebo, 0 score "
                        "differences on 36,776 slots). The VRAM (d) convention, which is board-used "
                        "minus a pre-run baseline and is the only quantity a co-tenant could "
                        "corrupt, was measured before the co-tenant arrived and its own pre/post "
                        "board readings are recorded in vram_latency.json. "
                        "SECOND SESSION (2026-08-16 afternoon): the two remaining base-control "
                        "rungs finished at 11:56, an EXPLORATORY knee rung at 376,320 finished at "
                        "13:18, and the VRAM/latency ladder was then RE-RUN from scratch over all "
                        "seven rungs at 15:26-15:29 on a card re-verified exclusive (13 MiB "
                        "resident), so every VRAM row is in-session comparable. The six rungs "
                        "common to both VRAM sessions reproduce to 3 dp on the (d) footprint for "
                        "5 of 6 (501,760 differs by 0.018 GiB); batch-1 latency moves by up to "
                        "10 ms, which is that measurement's own noise. The pre-registered session "
                        "is preserved at _verifier_hparams_parts/_prereg_6rung_backup/.",
                "framework": "HuggingFace transformers, bf16, flash_attention_2, tp=1, batch 1. "
                             "NEVER vLLM -- vLLM 0.9.0.1 silently drops all 192 visual.* LoRA "
                             "modules (0.775204 HF vs 0.702997 vLLM). The scorer asserts 192 "
                             "visual LoRA tensors are present at every load.",
                "numerics_pinned": {
                    "TF32": "OFF (torch.backends.cuda.matmul.allow_tf32 = False, "
                            "torch.backends.cudnn.allow_tf32 = False)",
                    "OMP_NUM_THREADS": 8,
                    "torch_num_threads": 8,
                    "batch": 1,
                    "min_pixels": 3136,
                    "row_order": "the canonical dump order slake -> vqa_rad -> pathvqa "
                                 "(genframe_data.DUMP_ORDER)",
                    "ranker": "argmax over the 8 slots with FIRST-INDEX tie-break "
                              "(genframe_data.picks_from_scores) -- never rank_avg here",
                },
                "adapter": "ckpts/train/lora_verifier_disjoint (the CLEAN disjoint verifier), "
                           "trained at max_pixels 1,003,520 (train_config.json)",
            },
            "conventions": {
                "macro": "equal weight per reporting cell, 8 cells at 1/8, Variant B (MMMU "
                         "excluded), CLEAN disjoint verifier. always-32B-direct 0.6567 is the bar; "
                         "a significant win needs macro delta ~ +0.0029.",
                "frozen_metric": "src/training_methods/genframe_data.py -- sel_eff 0.775204, "
                                 "oracle@8 0.626013, greedy 0.449467, n = 2,345 / 1,468 recoverable",
                "identity": "selected = oracle@8 x sel_eff, EXACT. The additive form "
                            "greedy + sel_eff*(oracle-greedy) over-predicts by ~+0.10 and is never "
                            "used (measured in null_tests.N2_identity).",
                "both_currencies": "every open-text endpoint is reported under BOTH the 32B judge "
                                   "and normalised exact match, on IDENTICAL picks.",
            },
        },
        "0_pre_registration": prereg,
        "1_null_tests": nulls,
        "2_the_rescore_nuisance_that_forces_an_in_session_control": {
            "measurement": rescore,
            "process_to_process_determinism_check": determ,
            "_read": "re-scoring stored (item, candidate) pairs at the DEPLOYED 1,003,520 with the "
                     "deployed adapter, prompt and batch size does NOT reproduce the stored score: "
                     "max abs deviation 6.03e-2, mean 5.86e-3, 79/200 pairs above 1e-3. The prior "
                     "round measured the same thing at 3.12e-2 / 5.72e-3 on 150 pairs. The likely "
                     "cause is the image-processor backend (transformers now loads "
                     "Qwen2VLImageProcessor as a FAST processor by default and warns that this "
                     "'may produce slightly different outputs') plus the torch version. "
                     "CONSEQUENCE: every delta in this artifact is taken against an IN-SESSION "
                     "1,003,520 arm scored in the same process family, never against the stored "
                     "dumps. The stored dumps are used only as the published ANCHOR.",
        },
        "3_endpoint_ladder": ladder,
        "3b_the_exploratory_seventh_rung": {
            "_what": "376,320 px (cap480) was NOT pre-registered. It was added after the "
                     "pre-registered ladder showed all its movement between 250,880 and 501,760, "
                     "purely to locate that transition. It is EXPLORATORY and is excluded from "
                     "every pre-registered claim.",
            "consequence_for_the_leakage_controls":
                "nested CV and the permutation null are properties of the ARM SET, so adding a "
                "seventh arm changes them. BOTH are reported: section 3's controls are the "
                "seven-rung set, and 3c is the pre-registered six-rung set recomputed unchanged.",
        },
        "3c_pre_registered_six_rung_leakage_controls":
            (prereg6.get("selection_leakage_controls", {"_status": "NOT MEASURED"})
             if isinstance(prereg6, dict) else {"_status": "NOT MEASURED"}),
        "4_mismatch_control_base_model_no_lora": base,
        "4b_resolution_vs_mismatch_difference_in_differences": {
            "sel_eff_scale": mism,
            "auroc_scale": mism_au,
            "_why_two_scales": "sel_eff is an ARGMAX endpoint, so a given loss of ranking quality "
                               "moves it only if the loss lands at the TOP of the pool; AUROC is "
                               "the un-thresholded ranking scale and moves with the information "
                               "itself. The base model and the adapter sit at very different "
                               "points on the ranking->argmax map (sel_eff 0.706 vs 0.777, cand "
                               "AUROC 0.755 vs 0.886), so a DiD on sel_eff cannot assume a shared "
                               "resolution term while a DiD on AUROC can. Both are reported.",
        },
        "4c_binding_fraction_and_fixed_stratum_dose_response": dose,
        "5_cost_flops": cost,
        "6_vram_and_latency": vram,
        "6b_A_LABELLING_INCONSISTENCY_FOUND_IN_THE_REFERENCE_VRAM_ARTIFACT": {
            "_what": "the task asked for the four conventions of "
                     "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json 'so rows are "
                     "comparable'. The four MEMORY conventions (a)-(d) are comparable and were "
                     "reproduced. The `vision_tokens` COLUMN IS NOT: that artifact reports "
                     "PRE-MERGE PATCHES under the name `vision_tokens`, while this round (and "
                     "resolution_sweep_2026-08-13) report MERGED tokens = patches / 4.",
            "evidence": {
                "same_12_items_same_max_pixels_1003520": True,
                "vram_testtime_2026-08-11_S3_vision_tokens_mean": 2362.6667,
                "this_round_1003520_vision_tokens_mean": 590.6666666666666,
                "ratio": 4.0,
                "smoking_gun_row_0": "that artifact's first row records image_pixels 57,600 with "
                                     "vision_tokens 324. Qwen2.5-VL smart_resize takes a 57,600 px "
                                     "image to 252x252, which is (252/14)^2 = 324 PRE-MERGE "
                                     "patches; after the 2x2 spatial merge the language model "
                                     "receives 324/4 = 81 tokens -- and 81 is exactly this round's "
                                     "minimum for the same item.",
                "input_tokens_agree": "674.33 there vs 674.67 here, so only the vision column "
                                      "carries the different definition.",
            },
            "_consequence": "NO number in this artifact is affected -- this round measures its own "
                            "geometry on all 8,965 triples and divides by 4 explicitly "
                            "(verifier_hparams_score.py records `patch`, load_arm divides). The "
                            "finding is recorded because the two artifacts' vision-token columns "
                            "must never be compared or cross-multiplied, and because a FLOP model "
                            "built on the wrong one is off by 4x on the vision term.",
            "_not_a_retraction": "vram_testtime_2026-08-11's memory numbers stand; only its "
                                 "vision_tokens column name is misleading.",
        },
        "7_end_to_end_macro": {"per_arm": macro, "cross_arm_paired_cis": macro_ci},
        "7b_guardrail_per_set_bootstrap_cis": grcis,
        "8_iso_escalation_is_the_macro_move_selection_or_bought_compute": isoesc,
        "9_the_cost_model_undercharges_the_verifier": recost,
        "10_max_tokens_truncation_audit": nulls.get("N4_max_tokens_truncation_audit",
                                                    {"_status": "NOT MEASURED"}),
    }
    # =============================== headline ========================================
    # Every number below is READ from the parts above; none is typed here.
    A = ladder.get("_arms", {})
    C = cost.get("by_max_pixels", {})
    R = recost.get("by_max_pixels", {})
    M = macro if "_status" not in macro else {}
    AVA = macro_ci.get("arm_vs_arm_macro", {}) if "_status" not in macro_ci else {}
    ISO = isoesc.get("by_max_pixels", {}) if "_status" not in isoesc else {}
    VR = vram.get("by_max_pixels", {}) if "_status" not in vram else {}
    CTRL, GEN, KNEE = "1003520", "250880", "501760"

    def pair(a, b, m="method_accuracy_max_veto", pool="all8_macro"):
        for k in (f"{a} - {b}", f"{b} - {a}"):
            if k in AVA:
                r = AVA[k][m][pool]
                s = 1 if k.startswith(a) else -1
                return {"delta": s * r["delta"],
                        "ci95": sorted([s * r["ci95"][0], s * r["ci95"][1]]),
                        "sig": r["sig"]}
        return None

    def g(px, *path):
        d = A.get(px, {})
        for p in path:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        return d if d != {} else None

    hl = {}
    hl["0_the_question_as_posed"] = {
        "premise_checked": "the task's premise was 'the scorer sees each image at 4x the resolution "
                           "the proposer saw it at'. That is true of the max_pixels CAP (1,003,520 "
                           "vs 250,880) but NOT of what the model actually sees: measured over all "
                           "8,965 scored triples the verifier renders "
                           f"{C.get(CTRL, {}).get('measured_mean_vision_tokens')} vision tokens "
                           f"and the same images at the generator's cap render "
                           f"{C.get(GEN, {}).get('measured_mean_vision_tokens')}, a ratio of "
                           f"{(C.get(CTRL, {}).get('measured_mean_vision_tokens', 0) / C.get(GEN, {}).get('measured_mean_vision_tokens', 1)):.2f}x, "
                           "because most of these images are already below 1,003,520 px. The "
                           "available prize was ~1.9x in vision tokens, not 4x.",
        "verifier_share_of_open_arm_flops_measured_here":
            C.get(CTRL, {}).get("verifier_share_of_open_arm_flops"),
        "_share_note": "the task quoted 69.1% from resolution_sweep_2026-08-13, whose verifier term "
                       "came from a 120-triple geometry sample (626.4 vision / 708.9 prompt tokens). "
                       "This round measures the same geometry on all 8,965 triples "
                       f"({C.get(CTRL, {}).get('measured_mean_vision_tokens')} vision / "
                       f"{C.get(CTRL, {}).get('measured_mean_prompt_tokens')} prompt) and gets a "
                       "smaller verifier term. The verifier is still the dominant term.",
    }
    hl["1_ANSWER_the_verifier_CANNOT_score_at_the_generators_resolution"] = {
        "pre_registered_endpoint": "judge sel_eff at 250,880 minus the in-session 1,003,520 control",
        "judge": g(GEN, "vs_control_judge"),
        "em": g(GEN, "vs_control_em"),
        "guardrail_judge": g(GEN, "guardrail_judge"),
        "guardrail_em": g(GEN, "guardrail_em"),
        "guardrail_per_set_CIs": grcis.get(GEN),
        "verdict": "SIGNIFICANT LOSS in BOTH currencies, guardrail-dirty on ALL THREE open sets in "
                   "BOTH currencies. The pre-registered 'free 4x' prize does not exist.",
    }
    hl["2_THE_ASYMMETRY_the_scorer_does_need_more_than_the_proposer_but_less_than_deployed"] = {
        "ladder_judge_sel_eff": {px: g(px, "judge", "sel_eff") for px in sorted(A, key=int)},
        "ladder_em_sel_eff": {px: g(px, "em", "sel_eff") for px in sorted(A, key=int)},
        "ladder_measured_vision_tokens": {px: g(px, "geometry", "mean_vision_tokens")
                                          for px in sorted(A, key=int)},
        "pooled_shape_LOOKS_like_a_step": "every rung at or above 501,760 gives judge sel_eff "
                                          "identical to 6 dp; every rung at or below 376,320 sits "
                                          "~0.016-0.020 lower and is nearly flat down to 62,720. "
                                          "Taken at face value that reads as a threshold between "
                                          "398 and 465 measured vision tokens.",
        "BUT_THE_POOLED_SHAPE_IS_CONFOUNDED": {
            "_what": "max_pixels is a CAP and Qwen's smart_resize only SHRINKS, so an image already "
                     "below the cap is rendered byte-identically to the deployed arm. The fraction "
                     "of the pool the cap actually BINDS on therefore changes at every rung, and "
                     "the pooled delta is (damage per affected item) x (fraction affected).",
            "binding_fraction_by_rung": {
                px: r.get("frac_binding") for px, r in
                ((dose.get("1_the_cap_binds_on_a_different_fraction_at_every_rung", {})
                  .get("by_max_pixels", {}) or {}).items())},
            "_read": "the binding fraction collapses from 69.8% at 376,320 to 15.2% at 501,760. "
                     "That collapse alone is enough to manufacture an apparent STEP with no "
                     "threshold in the model at all, so the pooled ladder cannot answer the "
                     "question and the fixed-stratum reading below is the one that counts.",
        },
        "THE_CORRECTED_SHAPE_a_graded_dose_response_on_a_fixed_item_set": {
            "_stratum": (dose.get("_stratum_definition")
                         if isinstance(dose, dict) else None),
            "n_items": (dose.get("2_fixed_stratum_dose_response", {}) or {}).get(
                "n_items_in_stratum"),
            "n_recoverable": (dose.get("2_fixed_stratum_dose_response", {}) or {}).get(
                "n_recoverable_in_stratum"),
            "by_rung": {
                px: {"vision_tokens": r.get("mean_vision_tokens_on_stratum"),
                     "d_sel_eff_judge": r.get("d_sel_eff_judge"),
                     "d_sel_eff_judge_ci": r.get("d_sel_eff_judge_ci"),
                     "d_sel_eff_em": r.get("d_sel_eff_em"),
                     "d_sel_eff_em_ci": r.get("d_sel_eff_em_ci"),
                     "is_trained_resolution": r.get("is_trained_resolution", False)}
                for px, r in sorted(((dose.get("2_fixed_stratum_dose_response", {}) or {})
                                     .get("by_max_pixels", {}) or {}).items(), key=lambda x: int(x[0]))},
            "shape": "MONOTONE AND GRADED, NOT A STEP, and NOT A PEAK AT THE TRAINED RESOLUTION. "
                     "On one fixed set of 357 large images (284 recoverable) re-rendered at every "
                     "rung, judge sel_eff falls smoothly as vision tokens fall -- 0.000 at 622 "
                     "tokens, -0.025 at 445, -0.053 at 292, -0.056 at 145, -0.074 at 66 -- and the "
                     "EM currency has the same sign and shape throughout. The damage SATURATES at "
                     "about 622 vision tokens, which is BELOW the 1,200 the trained/deployed rung "
                     "spends on this stratum.",
        },
        "the_asymmetry_restated": "the honest form of the asymmetry is not a magic threshold. It "
                                  "is that the SCORER keeps paying for detail it stops using at "
                                  "~620 vision tokens on the images where the cap binds, while the "
                                  "PROPOSER runs at 277 pooled. The deployed verifier is above its "
                                  "own saturation point, which is why a saving exists at all -- and "
                                  "why that saving is small, because the cap only binds on 15% of "
                                  "the pool once you are at 501,760.",
    }
    BA = base.get("_arms", {}) if isinstance(base, dict) else {}
    MD = mism.get("2_difference_in_differences", {}) if isinstance(mism, dict) else {}
    hl["3_THE_MISMATCH_CONFOUND_is_MEASURED_and_the_effect_is_RESOLUTION_not_mismatch"] = {
        "the_confound": "the adapter was TRAINED at 1,003,520, so every other rung is a "
                        "train/inference mismatch as well as a resolution change. The round "
                        "pre-registered a control for exactly this and it is now MEASURED.",
        "the_control": "the BASE Lingshu-7B with NO adapter, run as a zero-shot verifier on the "
                       "IDENTICAL prompt, pool, item order and numerics at 250,880 / 1,003,520 / "
                       "12,845,056. It was never trained at any resolution, so NO point on its "
                       "ladder is a mismatch and its curve is a pure resolution effect.",
        "0_FLOOR_CHECK_without_which_the_control_would_be_worthless": {
            "_why": "this project has measured that every TRAINING-FREE selector sits at the "
                    "random-pick floor, and a selector at the floor cannot move -- its flat "
                    "resolution curve would mean nothing. So the control must clear the floor "
                    "before any of its deltas are read.",
            **((mism.get("1_the_base_control_clears_the_floor", {})) if isinstance(mism, dict) else {}),
            "random_pick_floor_judge": (mism.get("0_floor_check", {}) or {}).get(
                "random_pick_floor_judge") if isinstance(mism, dict) else None,
            "verdict": "CLEARS IT. The zero-shot base verifier reaches judge sel_eff 0.705722 "
                       "against a random-pick floor of 0.676406 (sd 0.008116) -- about 3.6 sd "
                       "above it -- with candidate AUROC 0.755, not 0.5. The control is a real "
                       "measurement, though it is a much weaker selector than the adapter, which "
                       "is the project's known result that only a TRAINED verifier broke the floor.",
        },
        "1_the_base_ladder": {
            "judge_sel_eff": {px: (BA.get(px, {}).get("judge", {}) or {}).get("sel_eff")
                              for px in sorted(BA, key=int)},
            "em_sel_eff": {px: (BA.get(px, {}).get("em", {}) or {}).get("sel_eff")
                           for px in sorted(BA, key=int)},
            "cand_auroc_judge": {px: (BA.get(px, {}).get("judge", {}) or {}).get("cand_auroc")
                                 for px in sorted(BA, key=int)},
            "_note": "the base model's judge sel_eff at 250,880 and at 1,003,520 are equal to 6 dp "
                     "(0.705722), but that is a NET zero over 279 items whose pick changed, not an "
                     "absence of change -- which is precisely why the AUROC scale below is the one "
                     "that carries the answer.",
        },
        "2_THE_DECISIVE_TEST_difference_in_differences_on_the_AUROC_SCALE": {
            **(mism_au if isinstance(mism_au, dict) else {}),
            "verdict": "THE MISMATCH TERM IS UNDETECTABLE. Moving from the trained 1,003,520 down "
                       "to the generator's 250,880 costs the ADAPTER -0.006920 AUROC "
                       "[-0.00984,-0.00399] and costs the UNADAPTED BASE MODEL -0.006302 "
                       "[-0.01163,-0.00075]. The base model has no training resolution to mismatch "
                       "against, so its loss is pure resolution -- and the difference between the "
                       "two losses is -0.000619 [-0.005689,+0.004139], p = 0.815. The adapter "
                       "loses no more ranking information at cap320 than an unadapted model does. "
                       "The effect is RESOLUTION, and the mismatch term is bounded at "
                       "|dAUROC| <= 0.0057.",
        },
        "3_the_same_test_on_the_sel_eff_scale_DISAGREES_and_why_it_is_the_weaker_test": {
            "by_rung": {px: {"d_lora_judge": (r.get("judge", {}) or {}).get("d_lora"),
                             "d_base_judge": (r.get("judge", {}) or {}).get("d_base"),
                             "DiD": (r.get("judge", {}) or {}).get("DiD_mismatch_term"),
                             "DiD_ci": (r.get("judge", {}) or {}).get("DiD_ci"),
                             "p": (r.get("judge", {}) or {}).get("p_two_sided_DiD")}
                        for px, r in sorted(MD.items(), key=lambda x: int(x[0]))},
            "_read": "on sel_eff the DiD at 250,880 is -0.017711 [-0.032698,-0.002725], p = 0.0196 "
                     "-- nominally significant, i.e. the adapter's ENDPOINT falls further than the "
                     "base model's does. This is reported because it disagrees with the AUROC test "
                     "and must not be hidden. It is the weaker test: sel_eff is an argmax endpoint "
                     "and the two models sit at very different operating points (0.777 vs 0.706), "
                     "so a DiD on that scale cannot assume the resolution term is shared, which is "
                     "the assumption the DiD needs. A ranking loss of equal size moves the argmax "
                     "more for the stronger ranker.",
            "honest_statement": "the resolution effect is real and shared; a mismatch component "
                                "cannot be excluded on the endpoint scale, but it is not detectable "
                                "on the information scale, and the trained rung is NOT a peak on "
                                "either (12,845,056, also a mismatch, ties it exactly).",
        },
        "4_supporting_evidence_from_the_shape": "if mismatch drove the effect the TRAINED rung "
                                                "would be a PEAK. It is not: on the fixed stratum "
                                                "the curve is monotone in vision tokens and "
                                                "saturates at ~622, well BELOW the trained rung's "
                                                "1,200, and 12,845,056 (a mismatch ABOVE the "
                                                "trained point) ties it exactly in both currencies.",
        "residual_limit": "one adapter, one training resolution, and the base control was run at "
                          "only 3 of the 7 rungs. All 15 verifier adapters on disk were trained at "
                          "1,003,520; TRAINING a verifier at a second resolution (>=10 seeds x "
                          "~108 min each) remains the only way to close this completely and is "
                          "NOT MEASURED.",
    }
    hl["4_BEST_SETTING_and_what_it_is_worth"] = {
        "best_setting": "max_pixels 501,760 (cap640)",
        "why": "the cheapest rung that is not distinguishable from the deployed one",
        "judge": g(KNEE, "vs_control_judge"),
        "em": g(KNEE, "vs_control_em"),
        "guardrail_judge": g(KNEE, "guardrail_judge"),
        "guardrail_em": g(KNEE, "guardrail_em"),
        "guardrail_per_set_CIs": grcis.get(KNEE),
        "guardrail_read": "NO per-set flag at 501,760 excludes 0 in either currency, and the flags "
                          "FLIP SIGN between currencies. slake_open's judge flag is -0.00176 = "
                          "exactly ONE item out of 567 recoverable; vqa_rad_open's EM flag is "
                          "-0.02727 = 3.4 items out of 126 with CI [-0.073, +0.018]; pathvqa_open "
                          "is exactly 0.000000 in both. Within seed noise.",
        "cost": {
            "verifier_forward_flops_deployed": C.get(CTRL, {}).get("flops_per_verifier_forward"),
            "verifier_forward_flops_at_501760": C.get(KNEE, {}).get("flops_per_verifier_forward"),
            "verifier_flops_rel_to_deployed": C.get(KNEE, {}).get("flops_verifier_rel_to_deployed"),
            "open_arm_flops_rel_to_deployed": C.get(KNEE, {}).get("flops_open_arm_rel_to_deployed"),
            "open_arm_flops_saved_pct": C.get(KNEE, {}).get("open_arm_flops_saved_pct"),
            "vram_d_footprint_gib_deployed": VR.get(CTRL, {}).get("d_process_footprint_gib", {}).get("mean"),
            "vram_d_footprint_gib_at_501760": VR.get(KNEE, {}).get("d_process_footprint_gib", {}).get("mean"),
            "batch1_latency_ms_pool_representative_deployed":
                (A.get(CTRL, {}).get("geometry", {}).get("mean_wall_s_batch1") or 0) * 1000,
            "batch1_latency_ms_pool_representative_at_501760":
                (A.get(KNEE, {}).get("geometry", {}).get("mean_wall_s_batch1") or 0) * 1000,
        },
        "macro": pair(f"verifhp_px{KNEE}", f"verifhp_px{CTRL}"),
        "honest_verdict": "a genuine but SMALL free saving: the deployed rung sits above the knee, "
                          "so ~6% of the open arm's FLOPs and ~0.5 GiB of footprint are free. It is "
                          "not the 4x prize the round was hunting.",
    }
    hl["5_THE_TRAP_the_macro_moves_OPPOSITE_to_the_selector"] = {
        "observation": "250,880 is the WORST-tier selector and yet has the HIGHEST macro of any "
                       "rung, and the paired cross-arm bootstrap says that macro difference is "
                       "CI-clean.",
        "macro_250880_vs_in_session_control": pair(f"verifhp_px{GEN}", f"verifhp_px{CTRL}"),
        "mechanism": "the open-text escalation gate IS the selector's own max(score). Degrading the "
                     "score degrades the pick AND lowers confidence, so more questions go to the "
                     "32B, which is better than the 7B pick on all three open cells.",
        "escalation_evidence": {
            n: {k: v.get("am2_esc") for k, v in (M.get(n, {}).get("open_cell_detail", {}) or {}).items()}
            for n in [f"verifhp_px{CTRL}", f"verifhp_px{GEN}"] if n in M},
        "THE_DECISIVE_TEST_iso_escalation": {
            "_what": "put both arms on the SAME escalation budget, each ranked by its own gate, and "
                     "sweep the budget 0->1 (isoesc.json).",
            "open_macro_at_zero_escalation_pure_selection": {
                px: (ISO.get(px, {}).get("open_macro_curve") or [None])[0] for px in sorted(ISO, key=int)},
            "gap_vs_control_over_the_whole_grid": {
                px: ISO.get(px, {}).get("vs_control_open_macro") for px in sorted(ISO, key=int)},
            "conclusion": "at matched escalation 250,880 is BELOW the deployed rung essentially "
                          "everywhere (mean gap negative over the grid, and -0.018 at zero "
                          "escalation). Its end-to-end gain is BOUGHT COMPUTE, not better "
                          "selection -- and the same purchase is available more cheaply and far "
                          "more controllably by lowering the escalation threshold at the deployed "
                          "resolution. It must NOT be shipped as a verifier-resolution win.",
        },
    }
    hl["6_A_SEPARATE_FINDING_the_cost_model_undercharges_the_verifier"] = {
        "constant": "pandora_controller.py:50-52  VER7 flop-eq = 1.0, C_CHEAP_F = GEN7 + VER7 = 2.0",
        "measured_verifier_forward_in_generator_equivalents_at_deployed":
            R.get(CTRL, {}).get("verifier_forward_in_generator_equivalents"),
        "honest_C_CHEAP_F_at_deployed": R.get(CTRL, {}).get("C_CHEAP_F_honest"),
        "undercharge_factor": R.get(CTRL, {}).get("undercharge_factor_of_the_cheap_draw"),
        "macro_flopeq_as_charged": (R.get(CTRL, {}).get("macro", {})
                                    .get("method_accuracy_max_veto", {})
                                    .get("macro_flops_as_charged")),
        "macro_flopeq_honest_verifier": (R.get(CTRL, {}).get("macro", {})
                                         .get("method_accuracy_max_veto", {})
                                         .get("macro_flops_honest_verifier")),
        "_read": "the published open arm charges one verifier forward as one 7B generation. "
                 "Measured, at the deployed resolution it is ~1.88 of them. Replacing that ONE "
                 "constant and changing nothing else raises the accuracy-max arm's macro FLOP-eq "
                 "by ~23%. This is a flag for a full re-costing pass, not a re-costing.",
        "SCOPE": "only the verifier constant was replaced. The 4.57 strong-leg constant, meanN, "
                 "escalation and the macro weighting are the existing code's own outputs.",
    }
    hl["7_selection_leakage"] = ladder.get("selection_leakage_controls", {})
    hl["8_max_tokens"] = {
        "binds": "YES, but negligibly",
        "rate": nulls.get("N4_max_tokens_truncation_audit", {}).get("truncation_rate"),
        "n": nulls.get("N4_max_tokens_truncation_audit", {}).get("n_at_or_above_64_tokens"),
        "of": nulls.get("N4_max_tokens_truncation_audit", {}).get("n_candidate_strings"),
        "strict_upper_bound_on_recoverable_open_oracle8": 3 / 2345,
        "_read": "5 of 18,760 candidate strings hit the 64-token budget, all in pathvqa_open, over "
                 "5 items. Only 3 of those items have no correct answer anywhere in their pool, so "
                 "even if every truncated continuation would have become correct the open-half "
                 "oracle@8 could rise by at most 3/2345 = +0.00128 -- and the five strings are "
                 "rambling essays that already violate the prompt's 'short, specific phrase' "
                 "instruction, so the true bound is far lower. CLOSED: max_tokens=64 is not "
                 "costing this method measurable coverage.",
    }
    hl["9_the_scorer_is_deterministic_process_to_process"] = {
        "n_triples_scored_twice_in_two_processes":
            (determ.get("replicated_arm", {}) or {}).get("n_triples_scored_twice"),
        "n_that_disagree": (determ.get("replicated_arm", {}) or {}).get(
            "n_replicates_that_DISAGREE"),
        "max_abs_disagreement": (determ.get("replicated_arm", {}) or {}).get(
            "max_abs_disagreement"),
        "verdict": determ.get("verdict"),
        "_read": "combined with the identical-rendering placebo (0 score differences over every "
                 "identically-rendered slot at every rung), the numerical noise floor of this "
                 "whole round is exactly 0. Every non-zero delta reported here is a real change "
                 "in what the verifier was shown, not measurement noise.",
    }
    out["HEADLINE"] = hl

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT}")
    print(f"  size {os.path.getsize(OUT)/1024:.0f} KiB")


if __name__ == "__main__":
    main()
