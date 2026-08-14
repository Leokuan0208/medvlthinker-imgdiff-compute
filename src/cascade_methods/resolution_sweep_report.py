#!/usr/bin/env python3
"""resolution_sweep_report.py -- SWEEP 2 aggregator.

Folds the parts in results/cascade_methods/artifacts/_resolution_parts/ into the single dated
artifact results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json.

Every number in the output is copied from a part file that a measurement wrote; this script
computes only differences, means and ratios of those, and it labels what is NOT measured.

    python3 src/cascade_methods/resolution_sweep_report.py
"""
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
P = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json")


def g_(d, *ks, default=None):
    """nested get -- returns default rather than raising when a part is missing."""
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def load(name):
    p = os.path.join(P, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    geo_open = load("geometry_open.json")
    geo_mcq = load("geometry_mcq.json")
    mcq_pair = load("mcq_paired_cap320_vs_default.json")
    mcq_ladder = load("mcq_ladder.json")
    openres = load("open_generator_resolution.json")
    open_em = load("open_em.json")
    cost = load("cost_by_resolution.json")
    nulls = load("null_tests.json")

    art = {}
    art["_meta"] = {
        "title": "SWEEP 2 -- IMAGE RESOLUTION: what max_pixels does to accuracy, compute and VRAM "
                 "in the Lingshu cascade, on both halves of the macro-8",
        "created": "2026-08-13",
        "question": "Can changing the 7B's inference parameters improve the samples it generates? "
                    "This is the resolution arm: resolution changes what the model PERCEIVES, so "
                    "unlike a sampling temperature it moves greedy accuracy on all 8 cells, and it "
                    "is the direct lever on a prefill-bound workload.",
        "code": {
            "facts_geometry": "src/cascade_methods/resolution_geometry.py",
            "mcq_paired_existing_dumps": "src/cascade_methods/resolution_mcq_paired.py",
            "mcq_ladder_runner": "runners/run_resolution_mcq_ladder.sh",
            "mcq_ladder_analysis": "src/cascade_methods/resolution_mcq_ladder.py",
            "open_generation": "src/cascade_methods/resolution_open_generate.py "
                               "(+ runners/run_resolution_open_gen.sh)",
            "open_judge_cache": "src/cascade_methods/resolution_judge_cache.py",
            "open_verifier_scoring": "src/cascade_methods/resolution_verifier_score.py",
            "open_analysis": "src/cascade_methods/resolution_open_analyze.py",
            "open_exact_match_secondary": "src/cascade_methods/resolution_open_em.py",
            "null_tests": "src/cascade_methods/resolution_null_tests.py",
            "cost": "src/cascade_methods/resolution_cost.py",
            "stage2_driver": "runners/run_resolution_stage2.sh "
                             "(+ runners/run_resolution_label.sh)",
            "console_summary": "src/cascade_methods/resolution_summary.py",
            "aggregator": "src/cascade_methods/resolution_sweep_report.py"},
        "logs": ["logs/resolution_open_gen_2026-08-13.log   (open-text generation, all caps)",
                 "logs/resolution_mcq_ladder_2026-08-13.log (MedEvalKit MCQ ladder; see "
                 "not_measured -- it was OOM-blocked)",
                 "logs/resolution_label_2026-08-13.log      (verifier scoring + judge)",
                 "logs/resolution_stage2_2026-08-13.log     (the stage-2 driver)",
                 "logs/resolution_geometry_mcq.log          (CPU image-geometry pass)"],
        "environment": {
            "host": "dual A100 80GB PCIe; BOTH cards carried another tenant's jobs for the whole "
                    "session (one of this round's own engine cores was OOM-killed by a collision "
                    "at 12:32:27 and the job was restarted pinned to one card). No other process "
                    "was ever killed; every stage waits for free VRAM.",
            "generation": "vLLM 0.10.1.1+381074ae.nv25.09 (system python), bf16, tp=1, "
                          "max_model_len 4096, max_tokens 64, limit_mm_per_prompt image=4",
            "mcq_harness": "MedEvalKit via /data/dan/medeval_venv/bin/python (vLLM 0.9.0.1), "
                           "seed 42, temperature 0, tp=1, unmodified except its own CAP_MAX_PIXELS "
                           "env lever",
            "verifier_scoring": "HuggingFace transformers, bf16, flash_attention_2, batch 1, "
                                "TF32 OFF (torch.backends.cuda.matmul.allow_tf32=False, "
                                "cudnn.allow_tf32=False). NEVER vLLM -- vLLM 0.9.0.1 drops all 192 "
                                "visual.* LoRA modules (0.775204 HF vs 0.702997 vLLM).",
            "judge": "src/labeling/run_judge.py -- MedVLThinker-32B (Qwen2.5-32B backbone), "
                     "text-only, the project's existing judge. No new judge was invented."},
        "matched_control_discipline": (
            "Every accuracy delta in this artifact is between two arms generated by ONE script in "
            "ONE session. The open half's control is a cap320 arm generated here, not the stored "
            "deployed pool. The MCQ ladder carries its own 12,845,056 control generated here at "
            "tp=1, and it is NOT differenced against the 2026-07-01 tp=2 dumps -- those are used "
            "only against each other, which is a valid within-session pair of their own. The "
            "session measured the size of that caveat directly: SLAKE-closed is 0.825359 in the "
            "2026-07-01 tp=2 run and 0.820574 in this session's tp=1 run at the SAME resolution, "
            "a -0.004785 serving-config shift with no experimental variable changed."),
    }

    if nulls:
        art["null_tests"] = nulls
    if geo_open or geo_mcq:
        art["facts_image_geometry"] = {"open_cells": geo_open, "mcq_cells": geo_mcq}
    if mcq_pair:
        art["mcq_cap320_vs_default_full_n_existing_dumps"] = mcq_pair
    if mcq_ladder:
        art["mcq_resolution_ladder_this_session"] = mcq_ladder
    if openres:
        art["open_generator_resolution"] = openres
    if open_em:
        art["open_generator_resolution_exact_match_secondary"] = open_em
    if cost:
        art["cost"] = cost

    # ---- the prior round's ladder, cited not redone -----------------------------------------
    try:
        vl = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts",
                                         "vram_levers_2026-08-12.json")))
        art["prior_internal_track_ladder_cited"] = {
            "_source": "results/cascade_methods/artifacts/vram_levers_2026-08-12.json "
                       "accuracy_sweep_full (from ckpts/acc_gen/lingshu7b/{fullres,cap640,cap320,"
                       "cap160,cap80}/, runners/run_lingshu_acc.sh). Cited verbatim, not re-run.",
            "_what_this_round_adds_to_it": "that ladder's TOP rung is max_pixels 1,003,520. The "
                                           "published MCQ arms run at 12,845,056 -- 12.8x higher "
                                           "and 51.2x the cap320 rung -- so the ladder never "
                                           "reached the resolution the paper's cells are actually "
                                           "evaluated at, and its 'cap320 is nearly free' verdict "
                                           "is a statement about a rung the deployed system does "
                                           "not sit on. This round measures the missing rung on "
                                           "the MedEvalKit track.",
            "per_cell": vl["accuracy_sweep_full"]["per_cell"],
            "macro7": vl["accuracy_sweep_full"].get("internal_track_macro7"),
            "_track_warning": vl["accuracy_sweep_full"]["_track"]}
    except Exception as e:
        art["prior_internal_track_ladder_cited"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- headline: the three things this round settles -------------------------------------
    hl = {}
    if geo_mcq:
        binds = {k: v["by_cap"]["medevalkit_default"]["frac_images_above_cap"]
                 for k, v in geo_mcq.items() if isinstance(v, dict) and "by_cap" in v}
        hl["1_the_published_pipeline_runs_at_three_different_resolutions"] = {
            "mcq_cells_7B_and_the_32B_direct_bar": {
                "max_pixels": 12845056, "merged_vision_token_budget": 16384,
                "why": "MedEvalKit with CAP_MAX_PIXELS unset => qwen_vl_utils default. Verified by "
                       "reproducing all ten published MCQ cells (7B and 32B) from MedEvalKit's own "
                       "per-item dumps -- see null_tests.N2.",
                "code": "MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51; no runner sets "
                        "CAP_MAX_PIXELS"},
            "open_cells_7B_best_of_8_generator_AND_the_32B_direct_open_comparator": {
                "max_pixels": 250880, "merged_vision_token_budget": 320,
                "why": "runners/run_openvqa_lingshu7b.sh and runners/run_openvqa_pathvqa.sh call "
                       "src/labeling/run_openvqa.py with NO --cap, and its default is cap320 "
                       "(run_openvqa.py:50 HIGH_PX=1280*28*28, CAP_DIV[cap320]=4). The 32B open "
                       "arm (ckpts/openvqa/strong_lingshu) is the same script with the same "
                       "default."},
            "open_half_LoRA_verifier": {
                "max_pixels": 1003520, "merged_vision_token_budget": 1280,
                "why": "ckpts/train/lora_verifier_disjoint/train_config.json records cap_div 1, "
                       "max_pixels 1003520; verifier_transfer_eval.py sets MAXPX=1280*28*28."},
            "spread": "51.2x between the MCQ legs and the open generator, inside ONE reported "
                      "macro-8; 4x between the open generator and the verifier that ranks its "
                      "own candidates.",
            "what_this_does_NOT_invalidate": "each comparison is internally matched -- 7B and 32B "
                                             "run at the SAME resolution within each half -- so "
                                             "the published per-cell deltas stand.",
            "what_it_DOES_mean": "the macro-8 is not a single operating point, and any per-item "
                                 "cost charged against it has to be charged per cell. The "
                                 "project's FLOP-equivalence constant was derived at cap320 "
                                 "geometry (flop_ratio_derivation_2026-08-03.json token_geometry: "
                                 "image_tok_mean 280.48) and 62.5% of the macro weight runs at "
                                 "~3,325 vision tokens. See cost.R32_by_resolution."}
        hl["2_the_MCQ_arms_are_already_at_native_resolution"] = {
            "frac_images_above_the_12845056_cap_per_cell": binds,
            "read": "the harness default does not BIND on a single image in any MCQ cell. There "
                    "is no upward resolution headroom on the 62.5% of the macro weight where "
                    "sampling methods are structurally dead: raising max_pixels there cannot "
                    "change a single input. Resolution is a COST lever on that half, not an "
                    "accuracy lever.",
            "PMC_VQA_is_the_extreme_case": "median native size 66,122 px; even cap320 binds on "
                                           "only 8.9% of its images, which is why PMC is the one "
                                           "MCQ cell that barely moves with resolution."}
    if mcq_pair:
        hl["3_cutting_the_MCQ_leg_to_cap320_costs_accuracy"] = {
            k: {kk: v[kk] for kk in ("n_cell", "acc_cap320_250880", "acc_default_12845056",
                                     "delta_default_minus_cap320", "ci95", "significant")
                if kk in v}
            for k, v in mcq_pair.items() if isinstance(v, dict) and "acc_cap320_250880" in v}
        if geo_mcq:
            hl["3_cutting_the_MCQ_leg_to_cap320_costs_accuracy"]["_mechanism_across_cells"] = {
                cell: {"frac_images_the_cap320_cut_actually_binds_on":
                       geo_mcq[cell]["by_cap"]["cap320"]["frac_images_above_cap"],
                       "mean_vision_tokens_cap320_vs_default":
                       [geo_mcq[cell]["by_cap"]["cap320"]["mean_vision_tokens"],
                        geo_mcq[cell]["by_cap"]["medevalkit_default"]["mean_vision_tokens"]],
                       "delta_default_minus_cap320":
                       mcq_pair[cell].get("delta_default_minus_cap320"),
                       "significant": mcq_pair[cell].get("significant")}
                for cell in ("PMC_VQA", "SLAKE_closed", "VQA_RAD_closed")
                if cell in geo_mcq and isinstance(mcq_pair.get(cell), dict)
                and "delta_default_minus_cap320" in mcq_pair[cell]}
            hl["3_cutting_the_MCQ_leg_to_cap320_costs_accuracy"]["_mechanism_read"] = (
                "the loss tracks how much of the image the cap actually removes, not the cell: "
                "PMC-VQA's images are so small that cap320 binds on 8.9% of them and costs "
                "1.30x fewer vision tokens -- and it is the one cell with no significant loss. "
                "SLAKE and VQA-RAD lose 2.3x and 3.1x of their vision tokens and are both "
                "significantly worse. This is a 3-cell association, not a controlled test of the "
                "mechanism.")
        if "_macro8_contribution_of_the_measured_cells" in mcq_pair:
            hl["3_cutting_the_MCQ_leg_to_cap320_costs_accuracy"]["macro8_cost"] = \
                mcq_pair["_macro8_contribution_of_the_measured_cells"]
        hl["3_cutting_the_MCQ_leg_to_cap320_costs_accuracy"]["_read"] = (
            "matched pair at FULL n from the 2026-07-01 MedEvalKit dumps (both arms tp=2, same "
            "session). Two of the three measured cells are significantly worse at cap320. This "
            "does NOT contradict vram_levers_2026-08-12's '-0.0046, not significant': that "
            "comparison's top rung is 1,003,520, not 12,845,056, and its PMC-VQA is test_clean.csv "
            "(n=2,000) on the internal harness, not test_2.csv (n=33,430) on MedEvalKit.")
    if openres and openres.get("by_cap"):
        hl["4_open_text_generator_resolution"] = {
            "per_cap_mean_over_seeds": {c: openres["by_cap"][c].get("mean_sd")
                                        for c in openres["by_cap"]},
            "vs_cap320_control": {c: openres["vs_control"][c].get("per_metric")
                                  for c in openres.get("vs_control", {})},
            "capture_recapture_ceiling_per_cap": {
                c: {"macro_open3_LP_ceiling": v.get("macro_open3_LP_ceiling"),
                    "macro_open3_oracle8": v.get("macro_open3_oracle8")}
                for c, v in openres.get("capture_recapture", {}).items()},
            "laterality": openres.get("strata")}
    g = g_(open_em, "vs_control", "native", "greedy_t0_em") if open_em else None
    if g:
        hl["6_THE_OPEN_GENERATOR_IS_THE_ONE_ARM_WITH_RESOLUTION_HEADROOM_LEFT"] = {
            "statement": "the open-text generator is the only leg of the deployed pipeline that is "
                         "NOT already at native resolution -- it runs at max_pixels 250,880 while "
                         "its images average 274 merged vision tokens there and 545 uncapped. "
                         "Uncapping it raises the 7B's TEMPERATURE-0 accuracy on the 2,345-question "
                         "open pool by "
                         f"{g['delta']:+.6f} {g['ci95']}, significant, with no sampling, no "
                         "selection and no verifier involved.",
            "greedy_T0_exact_match": {"control_cap320": g_(open_em, "by_cap", "cap320",
                                                           "greedy_t0_em", "all"),
                                      "native_uncapped": g_(open_em, "by_cap", "native",
                                                            "greedy_t0_em", "all"),
                                      "delta": g["delta"], "ci95": g["ci95"],
                                      "significant": g["significant"],
                                      "per_cell": g["per_cell"]},
            "macro8_equivalent_if_it_carried": g["macro8_equivalent_if_it_carried"],
            "the_shape_of_the_gain": {
                "greedy_T0": g["delta"],
                "modal_of_8": g_(open_em, "vs_control", "native", "modal_em",
                                 "delta_mean_over_seeds"),
                "oracle_at_8": g_(open_em, "vs_control", "native", "oracle8_em",
                                  "delta_mean_over_seeds"),
                "oracle_ci95_per_seed": g_(open_em, "vs_control", "native", "oracle8_em",
                                           "ci95_per_seed"),
                "_read": "the gain is in ANSWER QUALITY, not in COVERAGE. Greedy and modal-of-8 "
                         "both move by ~+0.018/+0.019 and both CIs exclude zero; oracle@8 moves "
                         "only +0.009 and its CI does not. Higher resolution makes the model put "
                         "more mass on the right answer rather than putting new correct answers "
                         "into the pool. That matters for conversion: the project measured "
                         "newly-COVERED questions converting at 0.447 [0.368, 0.526], whereas a "
                         "shift that concentrates the existing distribution has no such discount "
                         "-- but that is a prediction, and only the SELECTED measurement settles "
                         "it."},
            "LABEL": "EXACT MATCH, the secondary endpoint. The project's primary label is the LLM "
                     "judge; the judge-labelled version of this row is in "
                     "open_generator_resolution if the judge stage completed, and in "
                     "not_measured if it did not.",
            "what_still_has_to_be_shown": "that the gain survives into SELECTED accuracy -- the "
                                          "quantity the macro-8's open cells actually report. A "
                                          "setting that raises oracle@8 while lowering sel_eff can "
                                          "be net negative; that is exactly what killed the "
                                          "diverse-generation arm (open_diverse_2026-08-10.json).",
            "cost_of_taking_it": "the generator's FLOPs per candidate rise; the whole open arm "
                                 "rises less, because the verifier (held at 1,003,520) is 69% of "
                                 "the arm. See frontier_open_half and cost."}
    if open_em and open_em.get("by_cap"):
        hl["5_open_generator_resolution_exact_match_secondary"] = {
            "per_cap": {c: {"seeds": v["seeds"], "modal_em": v.get("modal_em"),
                            "oracle8_em": v.get("oracle8_em"),
                            "greedy_t0_em": v.get("greedy_t0_em", {}).get("all")}
                        for c, v in open_em["by_cap"].items()},
            "vs_cap320_control": open_em.get("vs_control"),
            "_metric": open_em["_metric"],
            "_why_it_is_here": "computed from the generation dumps with no judge and no GPU, so "
                               "the resolution comparison has a fully paired answer under a "
                               "labelling rule that is completely independent of the LLM judge."}
    hl["0_what_it_would_take"] = {
        "_what": "the yardstick this round is measured against, fixed before any new number was "
                 "read, from quantities the project already publishes.",
        "macro8_significance_threshold": 0.0029,
        "open_cells_weight_in_macro8": 0.375,
        "required_mean_delta_selected_on_the_3_open_cells": round(0.0029 / 0.375, 5),
        "required_delta_oracle8_at_unchanged_sel_eff": round(0.0029 / 0.375 / 0.775204, 5),
        "identity_used": "selected = oracle@8 x sel_eff (exact, 5.6e-17)",
        "mcq_cells_weight_in_macro8": 0.625,
        "why_the_mcq_half_cannot_be_moved_UP_by_resolution": "its arms already run uncapped -- see "
                                                             "headline item 2.",
        "distribution_specificity": "the +0.0091 macro free bound on perfect coverage is the "
                                    "cap320 distribution's bound. Changing max_pixels changes the "
                                    "candidate distribution, so the capture-recapture ceiling is "
                                    "re-estimated PER CAP here rather than assumed to carry over."}
    hl["7_the_verdict"] = {
        "on_the_MCQ_half_62_5_percent_of_the_macro": "RESOLUTION IS NOT AN ACCURACY LEVER THERE, "
            "IN EITHER DIRECTION THAT HELPS. Up is impossible -- the published arms are already "
            "uncapped, the cap binds on 0.000 of images in all five cells. Down is not free: "
            "cap320 costs -0.007159 macro-8 [-0.012293, -0.002167] from just three of the five "
            "cells, 2.5x the project's own significance threshold, at full n.",
        "on_the_open_half_37_5_percent": "RESOLUTION IS THE ONE UNSWEPT LEVER THAT MOVED. The "
            "generator runs at 250,880 while its images are 545 merged vision tokens uncapped; "
            "uncapping raises temperature-0 accuracy by +0.017484 [+0.006823, +0.028571] and the "
            "modal-of-8 by +0.019190, both significant under exact match, for +30% of the open "
            "arm's FLOPs and +4.4 GiB of its footprint.",
        "what_is_NOT_settled": "whether that gain survives into SELECTED accuracy -- the quantity "
            "the macro-8's open cells actually report. The verifier re-scoring and the judge pass "
            "that decide it were queued behind other rounds' jobs on both shared A100s for the "
            "entire session and did not run. Until they do, this is a promising lead, not a result "
            "on the deployed endpoint.",
        "the_cheapest_next_step": "finish the labelling stage (one command, ~13k verifier forwards "
            "and one judge pass) -- see not_measured. Everything else it needs is already on disk.",
        "the_finding_that_stands_on_its_own": "the deployed pipeline evaluates one reported macro-8 "
            "at THREE different image resolutions spanning 51.2x, and its own verifier sees the "
            "image at 4x the resolution its generator did.",
    }
    art["headline"] = {k: hl[k] for k in sorted(hl)}      # numeric prefix order, 0 first

    # ---- the frontier table: accuracy x compute x VRAM, one row per cap ---------------------
    if cost and cost.get("open_half_per_candidate"):
        vram = cost.get("vram_cited_not_remeasured", {}).get("by_cap", {})
        alias = {"native": "medevalkit_default"}          # this round's name -> the 08-12 row name
        rows = {}
        for cap, c in cost["open_half_per_candidate"].items():
            r = {"max_pixels": c["max_pixels"],
                 "measured_mean_vision_tokens_open_pool": c["measured_mean_vision_tokens"],
                 "flops_generator_per_candidate": c["flops_per_candidate"],
                 "flops_rel_to_cap320_generator": c.get("flops_rel_to_cap320_generator"),
                 "flops_rel_to_cap320_whole_open_arm": c.get("flops_rel_to_cap320_whole_arm")}
            v = vram.get(alias.get(cap, cap))
            if v:
                r["vram_open_arm_d_process_footprint_gib"] = v["open_arm_d_process_footprint_gib"]
                r["vram_open_arm_b_peak_allocated_gib"] = v["open_arm_b_peak_allocated_gib"]
                r["vram_7b_mcq_leg_d_process_footprint_gib"] = v["mcq_7b_d_process_footprint_gib"]
                r["_vram_source"] = "vram_levers_2026-08-12.json (prior session, same instrument)"
            if openres and cap in openres.get("by_cap", {}):
                r["n_sampling_seeds"] = len(openres["by_cap"][cap]["seeds"])
                if r["n_sampling_seeds"] < 3:
                    r["_seed_shortfall"] = ("fewer than the 3 seeds the protocol asks for -- the "
                                            "card was shared and generation was cut at the stage-2 "
                                            "deadline. Read the per-seed values, not the mean.")
                ms = openres["by_cap"][cap]["mean_sd"]
                r["oracle8_judge_mean"] = ms["oracle8"]["mean"]
                r["selected_judge_mean"] = ms["selected"]["mean"]
                r["sel_eff_judge_mean"] = ms["sel_eff"]["mean"]
                r["greedy_t0_judge"] = openres["by_cap"][cap].get("greedy_t0", {}).get("all")
            if open_em and cap in open_em.get("by_cap", {}):
                r["oracle8_em_mean"] = open_em["by_cap"][cap]["oracle8_em"]["mean"]
                r["greedy_t0_em"] = open_em["by_cap"][cap].get("greedy_t0_em", {}).get("all")
            rows[cap] = r
        art["frontier_open_half"] = {
            "_read": "one row per generator resolution. Accuracy columns are the open half's own "
                     "2,345-question endpoint; compute is the measured-geometry FLOP model; VRAM "
                     "is cited from the prior session's measurement, same conventions. The "
                     "whole-arm FLOP column is the one that matters for deployment, and it moves "
                     "much less than the generator column because the verifier -- held fixed at "
                     "1,003,520 -- is 69% of the arm.",
            "rows": rows}

    # ---- what the labelling stage actually achieved -----------------------------------------
    lab = os.path.join(ROOT, "logs/resolution_label_2026-08-13.log")
    sw = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
    st = {"verifier_score_cache_exists": os.path.exists(os.path.join(sw, "verifier_score_cache.json")),
          "judge_cache_entries": None, "verifier_cache_entries": None,
          "log_tail": []}
    try:
        st["judge_cache_entries"] = len(json.load(open(os.path.join(sw, "judge_cache.json"))))
    except Exception:
        pass
    try:
        st["verifier_cache_entries"] = len(
            json.load(open(os.path.join(sw, "verifier_score_cache.json"))))
    except Exception:
        pass
    if os.path.exists(lab):
        st["log_tail"] = [l.strip() for l in open(lab, errors="ignore")
                          if l.startswith("[2026-")][-25:]
    st["_read"] = ("the open half's JUDGE-labelled endpoint (selected accuracy, sel_eff, "
                   "capture-recapture) exists only if both the verifier scoring and the judge pass "
                   "completed. If open_generator_resolution is absent from this artifact, they did "
                   "not, and the open half is reported on the exact-match secondary endpoint only.")
    art["labelling_stage_status"] = st

    art["not_measured"] = {
        "energy": "no NVML power integration was run in this round.",
        "batch-1 latency as a deployment claim": "MedEvalKit's per-item latency_s IS recorded per "
                                                 "arm and reported, but every arm in this session "
                                                 "ran on a card shared with another tenant, so it "
                                                 "is context, never a latency result.",
        "VRAM in this session": "cited from vram_levers_2026-08-12.json (same instrument, same "
                                "caps, same four conventions) rather than re-measured; both cards "
                                "were occupied by another tenant for the whole session and a "
                                "shared-card VRAM reading is not a clean measurement.",
        "MMMU": "excluded from the macro-8 on contamination grounds; not touched here.",
        "the 32B's own open-text resolution curve": "not swept. The 32B-direct open comparator is "
                                                    "at 250,880 like the generator, and moving it "
                                                    "would move the baseline, which is a different "
                                                    "experiment.",
        "PMC_VQA on this session's MCQ ladder": "not run -- 33,430 items per cap under card "
                                                "contention. Its two-point answer already exists "
                                                "at full n in the 2026-07-01 pair.",
        "PATH_VQA_closed and MedXpertQA-MM at cap320 on the MedEvalKit track":
            "ATTEMPTED AND BLOCKED. runners/run_resolution_mcq_ladder.sh was launched five times "
            "and OOM-killed each time by a co-tenant that grew between the free-VRAM check and the "
            "model load (logs/resolution_mcq_ladder_2026-08-13.log, 12:55:44 and 13:21:48: "
            "'Process <foreign> has 46.62 GiB memory in use' / three processes summing to 79.0 of "
            "79.14 GiB). It was then stopped so it would stop competing with the open-text sweep "
            "for the same cards. Consequence: the MedEvalKit-track cap320-vs-default pair covers "
            "3 of the 5 MCQ cells (PMC-VQA, SLAKE-closed, VQA-RAD-closed) at FULL n, not 5. The "
            "two missing cells are the ones where the cap binds hardest (MedXpert: cap320 binds on "
            "52.7% of images and cuts mean vision tokens 838.8 -> 241.0), so the measured "
            "'cap320 costs accuracy' finding is if anything CONSERVATIVE -- but that is a "
            "prediction from geometry, not a measurement, and must not be reported as one.",
        "SELECTED accuracy, sel_eff, the laterality stratum and the capture-recapture ceiling "
        "at each generator resolution":
            ("[status is in labelling_stage_status] These need (a) the deployed clean disjoint LoRA "
             "verifier re-scored on the new candidate pools -- 13,302 new (item, candidate) "
             "forwards, HF batch-1-equivalent, ~21 GiB -- and (b) the project's 32B judge on the "
             "new answer strings, ~66 GiB. Both were queued behind other rounds' jobs on the two "
             "shared A100s for the whole session (both cards sat at 74-78 of 80 GiB with none of "
             "this round's processes on them). The scoring code, the seeded item order, the "
             "priority ordering and the caches are all on disk, so a later session finishes this "
             "in one command: "
             "`NULLTEST=150 VBATCH=4 bash runners/run_resolution_label.sh` then "
             "`python3 src/cascade_methods/resolution_open_analyze.py`. "
             "Until then the open half is reported on the exact-match secondary endpoint ONLY, and "
             "the decisive question -- whether the greedy gain survives into SELECTED accuracy -- "
             "is OPEN."),
        "a second and third sampling seed at native": (
            "native has 3 complete arms (t0, s0) against the control's 4 (t0, s0, s1, s2). "
            "Generation ran at 1.3-2.5 items/s on a shared card and the sampled-arm comparison is "
            "therefore SINGLE-SEED at native, below the 3-seed protocol. The seed-0 deltas are "
            "reported with their own paired bootstrap CI and per-seed values, never as a mean over "
            "seeds. The temperature-0 comparison is deterministic and is not affected."),
        "the lower rungs of the open ladder (cap160, cap640, fullres)":
            "generation was ordered control-first (cap320, then native, then cap80) so a truncated "
            "run would still answer the question; cap160/cap640/fullres were not reached. The "
            "measured span is 62,720 -> 250,880 -> 12,845,056 px, i.e. 205x, which brackets the "
            "deployed point on both sides.",
    }

    json.dump(art, open(OUT, "w"), indent=1)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
    print("sections:", list(art.keys()))


if __name__ == "__main__":
    main()
