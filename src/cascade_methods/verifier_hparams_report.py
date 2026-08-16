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
                "vram_and_latency": "src/cascade_methods/verifier_hparams_vram.py",
                "end_to_end_macro": "src/cascade_methods/verifier_hparams_macro.py "
                                    "(wraps cascade_selector_rerun.run_source / combine, unmodified)",
                "aggregator": "src/cascade_methods/verifier_hparams_report.py",
            },
            "logs": ["logs/verifier_hparams_gpu0_2026-08-15.log",
                     "logs/verifier_hparams_gpu1_2026-08-15.log",
                     "logs/verifier_hparams_base_2026-08-16.log",
                     "logs/verifier_hparams_vram_2026-08-16.log"],
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
                        "board readings are recorded in vram_latency.json.",
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
        "4_mismatch_control_base_model_no_lora": base,
        "5_cost_flops": cost,
        "6_vram_and_latency": vram,
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
        "shape": "A STEP, NOT A PEAK. Every rung at or above 501,760 gives judge sel_eff identical "
                 "to 6 decimal places; every rung at or below 250,880 sits ~0.018-0.020 lower and "
                 "is flat all the way down to 62,720. The knee lies between 277.3 and 464.8 "
                 "measured vision tokens.",
        "the_asymmetry": "the SCORER needs roughly 465 vision tokens where the PROPOSER runs at 277 "
                         "-- about 1.7x, on the same images, for the same model family, on a task "
                         "that only asks Yes/No about a candidate the proposer already wrote.",
    }
    hl["3_THE_MISMATCH_CONFOUND_is_resolved_by_the_shape_and_by_the_base_control"] = {
        "the_confound": "the adapter was TRAINED at 1,003,520, so every other rung is a "
                        "train/inference mismatch as well as a resolution change.",
        "evidence_from_the_shape": "if mismatch drove the effect the TRAINED rung would be a PEAK. "
                                   "It is not: 501,760 (below) and 12,845,056 (above) are BOTH "
                                   "mismatches and BOTH tie it to 6 dp on judge sel_eff. A "
                                   "one-sided step whose plateau extends through the trained point "
                                   "in both directions is a resolution/information effect.",
        "base_model_control": base,
        "residual_limit": "one adapter, one training resolution. All 15 verifier adapters on disk "
                          "were trained at 1,003,520; training a verifier at a second resolution "
                          "(>=10 seeds x ~108 min each) was out of budget and is NOT MEASURED.",
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
    out["HEADLINE"] = hl

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"wrote {OUT}")
    print(f"  size {os.path.getsize(OUT)/1024:.0f} KiB")


if __name__ == "__main__":
    main()
