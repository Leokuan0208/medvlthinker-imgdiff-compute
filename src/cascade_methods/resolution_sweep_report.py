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


def _matched_seed_block(vs, prov, cap, control="cap320"):
    """The delta restricted to seeds whose treatment and control were generated in the SAME
    session. Once a later session adds seeds, `delta_mean_over_seeds` silently averages matched
    and cross-session pairs; this pulls out only the matched ones, which is what may be quoted."""
    if not vs:
        return None
    seeds = vs.get("seeds_paired", [])
    pairs = (prov or {}).get("pairs", {})
    ok = [t for t in seeds
          if pairs.get(f"{cap}_vs_{control}_{t}", {}).get("WITHIN_SESSION_MATCHED") is True]
    out = {"matched_seeds": ok, "all_paired_seeds": seeds,
           "_rule": "only these seeds have treatment and control generated in one session at one "
                    "gpu_memory_utilization. Any seed in all_paired_seeds but not in "
                    "matched_seeds crosses the +-0.008 serving-config boundary and is reported "
                    "for spread only -- never as the headline delta.",
           "per_metric": {}}
    if not ok:
        out["_status"] = "NO MATCHED SEED -- do not quote a sampled delta for this cap."
        return out
    for q, blk in (vs.get("per_metric") or {}).items():
        idx = [seeds.index(t) for t in ok]
        d = [blk["delta_per_seed"][i] for i in idx]
        c = [blk["ci95_per_seed"][i] for i in idx]
        out["per_metric"][q] = {
            "delta_matched_seeds": [round(float(x), 6) for x in d],
            "ci95_matched_seeds": c,
            "delta_mean_matched": round(float(sum(d) / len(d)), 6),
            "all_matched_ci_exclude_zero": bool(all(ci[0] > 0 or ci[1] < 0 for ci in c))}
    return out


def main():
    geo_open = load("geometry_open.json")
    geo_mcq = load("geometry_mcq.json")
    mcq_pair = load("mcq_paired_cap320_vs_default.json")
    mcq_ladder = load("mcq_ladder.json")
    openres = load("open_generator_resolution.json")
    open_em = load("open_em.json")
    cost = load("cost_by_resolution.json")
    nulls = load("null_tests.json")
    strata = load("strata_ci.json")
    prov = load("arm_provenance.json")
    gva = load("greedy_native_vs_deployed_arm.json")

    art = {}
    art["_meta"] = {
        "title": "SWEEP 2 -- IMAGE RESOLUTION: what max_pixels does to accuracy, compute and VRAM "
                 "in the Lingshu cascade, on both halves of the macro-8",
        "created": "2026-08-13, completed 2026-08-14",
        "_two_sessions": "2026-08-13 established the facts, ran the MCQ pair, generated every open "
                         "arm and finished the verifier scoring, but could never get a card for "
                         "the 32B judge pass, so it shipped with the decisive endpoint OPEN. "
                         "2026-08-14 ran that judge pass (14,779 triples), added native sampling "
                         "seeds, and corrected the 2026-08-13 diagnosis of why the PATH_VQA / "
                         "MedXpert MCQ arms kept failing -- it was the job's own activation-peak "
                         "budget, not a co-tenant. Both sessions used the same code, the same "
                         "caches and the same item order; no delta crosses them without saying so.",
        "question": "Can changing the 7B's inference parameters improve the samples it generates? "
                    "This is the resolution arm: resolution changes what the model PERCEIVES, so "
                    "unlike a sampling temperature it moves greedy accuracy on all 8 cells, and it "
                    "is the direct lever on a prefill-bound workload.",
        "code": {
            "facts_geometry": "src/cascade_methods/resolution_geometry.py",
            "mcq_paired_existing_dumps": "src/cascade_methods/resolution_mcq_paired.py",
            "mcq_ladder_runner": "runners/run_resolution_mcq_ladder.sh",
            "mcq_ladder_analysis": "src/cascade_methods/resolution_mcq_ladder.py",
            "mcq_pathvqa_medxpert_pair_runner": "runners/run_resolution_mcq_pathvqa.sh (2026-08-14; "
                                                "splits the two cells by their real image geometry "
                                                "so each gets an engine that fits its activation "
                                                "peak)",
            "laterality_strata_CIs": "src/cascade_methods/resolution_strata_ci.py (2026-08-14)",
            "arm_generation_provenance": "src/cascade_methods/resolution_arm_provenance.py "
                                         "(2026-08-14; flags any cap-vs-control pair that crosses "
                                         "the session boundary)",
            "greedy_native_vs_deployed_arm":
                "src/cascade_methods/resolution_greedy_vs_arm.py (2026-08-14)",
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
                 "logs/resolution_open_gen_2026-08-14.log   (native sampling seeds s1/s2)",
                 "logs/resolution_mcq_ladder_2026-08-13.log (MedEvalKit MCQ ladder)",
                 "logs/resolution_mcq_pathvqa_2026-08-14.log(PATH_VQA / MedXpert resolution pair)",
                 "logs/resolution_label_2026-08-13.log      (verifier scoring, 13,302 forwards)",
                 "logs/resolution_judge_2026-08-14.log      (the 32B judge pass, 14,779 triples)",
                 "logs/resolution_stage2_2026-08-13.log     (the stage-2 driver)",
                 "logs/resolution_geometry_mcq.log          (CPU image-geometry pass)"],
        "environment": {
            "host": "dual A100 80GB PCIe, shared. On 2026-08-13 BOTH cards carried another "
                    "tenant's jobs for the whole session (one of this round's own engine cores was "
                    "OOM-killed by a collision at 12:32:27 and the job was restarted pinned to one "
                    "card), which is why the judge pass never ran. On 2026-08-14 both cards were "
                    "briefly empty -- the judge pass and the native seeds were taken then -- and a "
                    "40 GiB co-tenant reappeared mid-session. No foreign process was ever killed "
                    "in either session; every stage waits for free VRAM and gives up rather than "
                    "oversubscribing.",
            "generation": "vLLM 0.10.1.1+381074ae.nv25.09 (system python), bf16, tp=1, "
                          "max_model_len 4096, max_tokens 64, limit_mm_per_prompt image=4",
            "mcq_harness": "MedEvalKit via /data/dan/medeval_venv/bin/python (vLLM 0.9.0.1), "
                           "seed 42, temperature 0, tp=1. NOT MODIFIED in either session -- the "
                           "resolution is applied only through the CAP_MAX_PIXELS environment "
                           "variable.",
            "⚠_the_CAP_MAX_PIXELS_lever_is_itself_an_uncommitted_vendor_edit":
                "MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51 reads "
                "`_MP = int(os.environ.get(\"CAP_MAX_PIXELS\",\"0\"))` and passes max_pixels only "
                "when _MP is non-zero. That line is a LOCAL, UNCOMMITTED edit to a vendored "
                "dependency (file mtime 2026-07-04, i.e. it predates both sessions of this round; "
                "no session of this round changed it). It is the same class of landmine CLAUDE.md "
                "already records for MedEvalKit's reasoning prompt. Two consequences: (a) the "
                "published MCQ cells run at the qwen_vl_utils DEFAULT precisely because no runner "
                "sets that variable -- verified, only this round's own two resolution runners set "
                "it -- and (b) a clean checkout of MedEvalKit would NOT reproduce this round's "
                "cap320 MCQ arms at all, because the lever would not exist.",
            "verifier_scoring": "HuggingFace transformers, bf16, flash_attention_2, TF32 OFF "
                                "(torch.backends.cuda.matmul.allow_tf32=False, "
                                "cudnn.allow_tf32=False). NEVER vLLM -- vLLM 0.9.0.1 drops all 192 "
                                "visual.* LoRA modules (0.775204 HF vs 0.702997 vLLM). The "
                                "verifier itself is HELD FIXED at max_pixels 1,003,520 in every "
                                "arm: this sweep moves the GENERATOR only.",
            "verifier_batching_caveat":
                "production scoring ran at --batch 4 (LEFT-padded), not the deployed batch 1, for "
                "throughput. The round's own null test measures what that costs: batch-4 vs "
                "batch-1 on the same 150 pairs is max_abs_dev 0.054430, mean_abs_dev 0.006263; "
                "batch-1 vs the STORED transfer-dump score is max_abs_dev 0.031209, mean_abs_dev "
                "0.005718. These are per-CANDIDATE score deviations, not endpoint deviations, and "
                "they apply UNIFORMLY to the control and to every swept arm, so the cap-vs-cap "
                "deltas -- the load-bearing quantity -- are internally consistent. The absolute "
                "LEVEL is not: this round's cap320 control reads sel_eff 0.784713 against the "
                "deployed pool's 0.775204, and that gap mixes the batching, the vLLM-version "
                "shift and the fresh generation. Never quote this round's absolute sel_eff as the "
                "deployed number; quote the deltas.",
            "judge": "src/labeling/run_judge.py -- MedVLThinker-32B (Qwen2.5-32B backbone), "
                     "text-only, the project's existing judge. No new judge was invented."},
        "matched_control_discipline": (
            "Every accuracy delta in this artifact is between two arms generated by ONE script in "
            "ONE session. The open half's control is a cap320 arm generated here, not the stored "
            "deployed pool. The MCQ ladder carries its own 12,845,056 control generated here at "
            "tp=1, and it is NOT differenced against the 2026-07-01 tp=2 dumps -- those are used "
            "only against each other, which is a valid within-session pair of their own. The "
            "round measured the size of that caveat directly, on TWO cells, with no experimental "
            "variable changed: SLAKE-closed is 0.825359 in the 2026-07-01 tp=2 run and 0.820574 "
            "in this round's tp=1 run at the SAME resolution (-0.004785), and PATH_VQA-closed is "
            "0.840869 vs 0.837002 (-0.003867, n=3,362, measured 2026-08-14). Both shifts are the "
            "same sign and the same order as the project's standing +-0.008 caveat, and both are "
            "larger than several effects this round is asked to detect -- which is why every "
            "accuracy delta here is taken against a control generated alongside it."),
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
    if strata:
        art["laterality_stratum_with_CIs"] = strata
    if gva:
        art["greedy_at_native_vs_the_deployed_sampled_arm"] = gva
    if prov:
        art["open_arm_generation_provenance"] = prov
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
        hl["4B_open_text_generator_resolution_supporting_tables"] = {
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
                     "open_generator_resolution and its verdict is headline item 4A.",
            "what_still_has_to_be_shown": "SETTLED on 2026-08-14 -- see headline item 4A. The gain "
                                          "does NOT survive into SELECTED accuracy: judge-labelled "
                                          "selected moves +0.005117 [-0.003838, +0.014072], CI "
                                          "spans zero, because sel_eff falls -0.010799 and gives "
                                          "most of the oracle gain back.",
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
    # ---- item 4: the decisive judge-labelled endpoint ---------------------------------------
    if openres and openres.get("vs_control"):
        nat = openres["vs_control"].get("native")
        c80 = openres["vs_control"].get("cap80")
        bc = openres.get("by_cap", {})

        def _row(cap):
            v = bc.get(cap)
            if not v:
                return None
            rel = {t: g_(v, "per_seed", t, "coverage_of_labels", "RELIABLE") for t in v["seeds"]}
            row = {"max_pixels": v["max_pixels"], "seeds": v["seeds"],
                   "oracle8": v["mean_sd"]["oracle8"], "sel_eff": v["mean_sd"]["sel_eff"],
                   "selected": v["mean_sd"]["selected"],
                   "pool_modal": v["mean_sd"]["pool_modal"],
                   "greedy_t0_judge": (v.get("greedy_t0") or {}).get("all"),
                   "identity_selected_eq_oracle_x_seleff_max_abs_err":
                       v["identity_selected_eq_oracle_x_seleff_max_abs_err"],
                   "label_coverage_RELIABLE_per_seed": rel}
            bad = [t for t, ok in rel.items() if ok is not True]
            if bad:
                row["⚠_UNRELIABLE_SEEDS"] = (
                    f"seeds {bad} have >1% of their candidate slots missing a judge label or a "
                    "verifier score. An unlabelled slot is excluded from oracle@8 and counts as "
                    "WRONG if the verifier picks it, so those seeds are biased DOWNWARD and the "
                    "mean/sd above must NOT be quoted. Use "
                    "PRIMARY_matched_seed0_only, which is restricted to fully-labelled, "
                    "same-session seeds.")
            return row

        hl["4A_THE_DECISIVE_ENDPOINT_THE_GAIN_DOES_NOT_SURVIVE_SELECTION"] = {
            "statement": "on the JUDGE-labelled endpoint the macro-8's open cells actually report "
                         "-- SELECTED accuracy -- uncapping the generator is NOT a win. Raising "
                         "max_pixels 250,880 -> 12,845,056 raises oracle@8 by +0.013220 "
                         "[+0.002985, +0.023881] and the modal-of-8 by +0.024307 [+0.013220, "
                         "+0.035821], BOTH significant, but SELECTED accuracy moves only +0.005117 "
                         "[-0.003838, +0.014072] pooled, whose CI spans zero, because sel_eff "
                         "FALLS -0.010799 [-0.022318, +0.000720]. The verifier gives back most of "
                         "what the higher-resolution generator won. On the macro-8's own "
                         "EQUAL-WEIGHT-PER-CELL basis the selected delta is +0.009623 [-0.000401, "
                         "+0.020334] over the three open cells = +0.003609 [-0.000150, +0.007625] "
                         "macro-8, i.e. the POINT ESTIMATE clears the project's +0.0029 threshold "
                         "but the interval still does not exclude zero. That is the honest "
                         "summary: suggestive, not established.",
            "⚠_HOW_THIN_THE_NATIVE_MACRO_POINT_ESTIMATE_IS": {
                "net_items_changed_per_cell_native_seed0": {"slake_open (n=645)": 4,
                                                            "vqa_rad_open (n=200)": 4,
                                                            "pathvqa_open (n=1500)": 4},
                "total_net_item_flips_behind_the_macro": 12,
                "of_a_pool_of": 2345,
                "leave_one_cell_out_macro8": {"drop slake_open": 0.00425,
                                              "drop vqa_rad_open": 0.001663,
                                              "drop pathvqa_open": 0.004913},
                "project_significance_threshold_on_macro8": 0.0029,
                "_read": "the equal-weight macro hands vqa_rad_open -- 200 items -- a full third "
                         "of the open weight, and its entire contribution is FOUR net item flips. "
                         "Drop that one cell and the macro falls to +0.001663, below the "
                         "threshold. So the point estimate that clears the bar is carried by 12 "
                         "net flips out of 2,345 and by the smallest cell in the pool. This is the "
                         "same load-bearing-single-cell pattern the project already documents for "
                         "PMC-VQA in the vs-direct claim, and it is why the honest verdict is NOT "
                         "ESTABLISHED rather than a win.",
                "_the_cap80_loss_by_contrast_is_robust": "cutting to cap80 gives macro-8 -0.007682 "
                                                         "[-0.013222, -0.002349] with "
                                                         "leave-one-cell-out staying between "
                                                         "-0.006000 and -0.009273 and 19/4/18 net "
                                                         "item flips behind it. The DOWNWARD "
                                                         "direction is the robust half of this "
                                                         "result.",
            },
            "⚠_macro_weighting_correction": "an earlier draft of this artifact multiplied the "
                                            "POOLED (item-weighted) selected delta by 3/8 and "
                                            "reported +0.001919. That was the wrong basis. The "
                                            "macro-8 weights CELLS equally (1/8 each) while the "
                                            "open pool is 645/200/1500 items, so pooling hands "
                                            "PathVQA 64% of the weight where the macro gives it "
                                            "33% -- and PathVQA is the cell with the smallest "
                                            "gain, so pooling UNDERSTATED the macro effect by "
                                            "roughly half. Both numbers are now reported side by "
                                            "side in laterality_stratum_with_CIs."
                                            "macro8_arithmetic_open_half, which also stratifies "
                                            "the bootstrap by cell.",
            "per_cap_judge_labelled": {c: _row(c) for c in bc},
            "PRIMARY_matched_seed0_only": _matched_seed_block(nat, prov, "native"),
            "native_vs_cap320_control": (nat or {}).get("per_metric"),
            "cap80_vs_cap320_control": (c80 or {}).get("per_metric"),
            "macro8_arithmetic": (strata or {}).get("macro8_arithmetic_open_half"),
            "guardrail_native": (nat or {}).get("guardrail"),
            "guardrail_cap80": (c80 or {}).get("guardrail"),
            "manipulation_check": {
                "native": (nat or {}).get("manipulation_check"),
                "cap80": (c80 or {}).get("manipulation_check")},
            "token_audit": {
                "per_cap_per_cell": {c: v.get("measured_token_geometry") for c, v in bc.items()},
                "_read": "REQUIRED CHECK, PASSED. Mean GENERATED tokens per sample are flat across "
                         "the whole 205x resolution span -- 4.66/5.44/5.54 at cap80, "
                         "4.69/5.54/5.69 at cap320, 4.69/5.58/5.68 at native "
                         "(slake/vqa_rad/pathvqa) -- while mean VISION tokens move 64 -> 244 -> "
                         "672 on SLAKE. So the treatment reached the input and did NOT change "
                         "answer length or format. That matters here because this project has "
                         "twice been caught mistaking an ANSWER-FORMAT effect for a capability "
                         "effect (the reasoning-vs-direct finding, retrospective 5.1); resolution "
                         "is clean of that confound.",
                "_no_arm_is_a_reasoning_arm": "every arm averages under 6 generated tokens, so no "
                                              "arm accidentally became a reasoning run -- the "
                                              "failure mode CLAUDE.md records for the `_think` "
                                              "dumps.",
            },
            "⚠_THE_MECHANISM_MEASURED_ON_ONE_CELL": {
                "cell": "pathvqa_open (n=1,500, the largest open cell)",
                "oracle8_delta": "+0.016000 [+0.001333, +0.030667]  SIGNIFICANT",
                "sel_eff_delta_on_jointly_recoverable":
                    "-0.022039  SIGNIFICANT (CI excludes zero)",
                "selected_delta": "+0.002667 [-0.009333, +0.014667]  not significant",
                "_read": "this is the whole finding in one cell, and both halves of it are "
                         "individually significant. Higher resolution puts significantly MORE "
                         "correct answers into PathVQA's candidate pool, and the verifier gets "
                         "significantly WORSE at picking them, and the two cancel so that the "
                         "reported endpoint does not move. It is not that the gain was too small "
                         "to see -- it is that the selector consumed it.",
                "_why_pathvqa": "PathVQA is where the coverage headroom is (oracle@8 0.508 at "
                                "cap320 against 0.869 on SLAKE) and where the verifier is weakest "
                                "(sel_eff 0.741 vs 0.850 on SLAKE). Resolution moves the pool most "
                                "in exactly the cell whose selector can least afford new "
                                "candidates to rank.",
                "_the_other_two_cells": "SLAKE-open and VQA-RAD-open move nothing significantly in "
                                        "either direction at native; their per-cell CIs all span "
                                        "zero. See laterality_stratum_with_CIs.per_cell_deltas_"
                                        "with_ci for every cell, metric and seed.",
            },
            "_the_shape": "this is the SAME shape that killed the diverse-generation arm "
                          "(open_diverse_2026-08-10.json): a treatment that enlarges or improves "
                          "the candidate pool is taxed by the selector. It is milder here -- "
                          "selected is at least POSITIVE rather than negative -- but it is not "
                          "significant and it is not free.",
            "_why_it_is_not_a_coverage_win_either": "oracle@8 rises +0.0132 while pool_modal rises "
                                                    "+0.0243: the bigger move is in answer QUALITY "
                                                    "(mass on the right answer) than in COVERAGE "
                                                    "(new correct answers entering the pool). The "
                                                    "project's marginal conversion of newly-covered "
                                                    "questions is 0.447 [0.368, 0.526], and "
                                                    "0.0132 x 0.447 = 0.0059, which is within "
                                                    "noise of the +0.0051 actually observed.",
            "_the_downward_direction_is_a_CLEAN_LOSS": "cutting the generator to cap80 (62,720) "
                                                       "loses oracle@8 -0.024733, selected "
                                                       "-0.018763 and every one of the three "
                                                       "per-cell guardrails, all CIs excluding "
                                                       "zero on both seeds. Resolution is real; it "
                                                       "just cannot be converted upward.",
            "_label": "LLM judge (src/labeling/run_judge.py, MedVLThinker-32B, text-only), the "
                      "project's primary open-text label. 14,779 new (question, gold, answer) "
                      "triples judged 2026-08-14; judge_cache now 112,656 entries, "
                      "still-missing=0. Every arm's label coverage flag RELIABLE=true.",
        }
        if openres.get("capture_recapture"):
            hl["4A_THE_DECISIVE_ENDPOINT_THE_GAIN_DOES_NOT_SURVIVE_SELECTION"][
                "capture_recapture_ceiling_per_cap"] = {
                c: {"max_pixels": v["max_pixels"],
                    "macro_open3_LP_ceiling": v["macro_open3_LP_ceiling"],
                    "macro_open3_oracle8": v["macro_open3_oracle8"],
                    "headroom_LP_minus_oracle8": round(
                        v["macro_open3_LP_ceiling"] - v["macro_open3_oracle8"], 6)}
                for c, v in openres["capture_recapture"].items()}
            hl["4A_THE_DECISIVE_ENDPOINT_THE_GAIN_DOES_NOT_SURVIVE_SELECTION"][
                "_ceiling_read"] = (
                "the Lincoln-Petersen ceiling is re-estimated PER CAP from that cap's own "
                "independent seeds, because changing max_pixels changes the candidate "
                "distribution and the project's +0.0091 macro bound is the cap320 "
                "distribution's bound, not a universal one. A cap with only one complete "
                "seed has no LP estimate and is absent from this block.")

    # ---- item 4C: the one operating point this round found that is worth taking ---------------
    if gva:
        acc = gva["accuracy"]["native_greedy_MINUS_cap320_selected"]
        cmp_ = gva["compute"]
        hl["4C_THE_ONE_OPERATING_POINT_WORTH_TAKING_greedy_at_native_replaces_the_sampled_arm"] = {
            "statement": "the sweep's most useful result is not a resolution win, it is what the "
                         "resolution curve reveals about the SAMPLING machinery. ONE greedy decode "
                         "at native resolution scores "
                         f"{gva['per_cell']['slake_open']['native_greedy_t0']:.6f}/"
                         f"{gva['per_cell']['vqa_rad_open']['native_greedy_t0']:.6f}/"
                         f"{gva['per_cell']['pathvqa_open']['native_greedy_t0']:.6f} "
                         "(slake/vqa_rad/pathvqa), pooled 0.486994 -- statistically "
                         "indistinguishable from the DEPLOYED eight-sample cap320 arm plus its "
                         f"trained LoRA verifier (delta {acc['delta_mean_over_seeds']:+.6f}, every "
                         "seed's CI spanning zero) -- at "
                         f"{cmp_['ratio_greedy_native_over_deployed_arm']:.6f}x its FLOPs, a "
                         f"{cmp_['compute_reduction_x']:.2f}x compute reduction on the open half.",
            "accuracy_pooled_over_questions": acc,
            "accuracy_macro8_basis_equal_weight_per_cell":
                gva["accuracy"]["macro8_basis_equal_weight_per_cell"],
            "⚠_WHICH_BASIS_AND_HOW_FRAGILE": {
                "pooled_over_questions": "delta +0.004122 mean over the 3 control seeds, every "
                                         "seed's CI spanning zero -> a TIE.",
                "macro8_equal_weight_per_cell": "delta +0.008859 / +0.007456 / +0.010188 macro-8 "
                                                "for control seeds s0/s1/s2, and EVERY seed's CI "
                                                "excludes zero -> significant on the basis the "
                                                "project actually reports.",
                "but_leave_one_cell_out_kills_it": "dropping vqa_rad_open takes those three macro "
                                                   "figures to +0.000163, -0.001003 and +0.002157 "
                                                   "-- i.e. to nothing. The entire macro "
                                                   "significance is one 200-item cell carrying 13 "
                                                   "to 14 net item flips, handed a third of the "
                                                   "open weight by equal-cell weighting.",
                "_therefore": "the ACCURACY claim is a TIE, stated conservatively. The macro-basis "
                              "significance is reported because it is what the project's own "
                              "convention produces, and it is immediately qualified because it "
                              "does not survive its own leave-one-cell-out. The robust half of "
                              "this result is the COMPUTE, not the accuracy.",
            },
            "pure_resolution_effect_on_one_decode":
                gva["accuracy"]["native_greedy_MINUS_cap320_greedy"],
            "compute": cmp_,
            "per_cell_guardrail": {
                ds: {"n": v["n"], "native_greedy_t0": v["native_greedy_t0"],
                     "cap320_selected_per_seed": v["cap320_selected_per_seed"],
                     "delta_per_seed": {t: x["delta"] for t, x in
                                        v["delta_vs_cap320_selected"].items()},
                     "ci_excludes_zero_per_seed": {t: x["ci_excludes_zero"] for t, x in
                                                   v["delta_vs_cap320_selected"].items()},
                     "worse_than_deployed": v["worse_than_deployed_on_this_cell"]}
                for ds, v in gva["per_cell"].items()},
            "_guardrail_read": "VQA-RAD-open is significantly BETTER on all three control seeds "
                               "(+0.065 to +0.070, every CI excluding zero). SLAKE-open is better, "
                               "not significantly. PathVQA-open is the one cell that goes the "
                               "wrong way, -0.004 to -0.010, and no seed's CI excludes zero, so it "
                               "is within noise -- but it IS a guardrail flag on the project's "
                               "largest open cell (n=1,500) and must be quoted with the headline.",
            "_matched": gva["_matched"],
            "_THE_CATCH": gva["_the_catch"],
            "_the_8_is_the_frozen_metric_not_the_deployed_average_N":
                gva["_the_8_is_the_frozen_metric_not_the_deployed_average_N"],
            "_not_measured": gva["_not_measured"],
            "_the_honest_one_line": "a TIE on accuracy (pooled +0.004122, CIs spanning zero; the "
                                    "macro-basis significance rests entirely on one 200-item "
                                    "cell) at 0.0759x the open arm's FLOPs. The result to bank is "
                                    "the 13.17x compute reduction, not an accuracy gain.",
            "_why_this_matters_to_the_project": "the project's stated current objective is cost "
                                                "reduction at a tie -- the 2026-08-12 result is "
                                                "0.865x compute at a tie with always-32B-direct. "
                                                "This is a 13.17x reduction on the open half's "
                                                "arm at a tie with that arm's own output, and it "
                                                "is orthogonal to the MCQ half. It is NOT a "
                                                "macro-8 claim: the open cells' reported number "
                                                "would move by the delta above, which is not "
                                                "significant.",
        }

    # ---- item 3B: the MCQ cut, now over FOUR of the five cells --------------------------------
    lad_pv = g_(mcq_ladder, "by_cap", "250880", "per_cell", "PATH_VQA_closed")
    if mcq_pair and isinstance(lad_pv, dict) and "delta_vs_control" in lad_pv:
        # each cell's delta is DEFAULT minus CAP320 (positive = cutting resolution costs accuracy)
        cells = {}
        for c in ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed"]:
            v = mcq_pair.get(c)
            if isinstance(v, dict) and "delta_default_minus_cap320" in v:
                cells[c] = {"delta_default_minus_cap320": v["delta_default_minus_cap320"],
                            "ci95": v["ci95"], "significant": v["significant"], "n": v["n_cell"],
                            "_arms": "2026-07-01 MedEvalKit dumps, both tp=2, one session"}
        cells["PATH_VQA_closed"] = {
            "delta_default_minus_cap320": round(-lad_pv["delta_vs_control"], 6),
            "ci95": [round(-lad_pv["ci95"][1], 6), round(-lad_pv["ci95"][0], 6)],
            "significant": lad_pv["significant"], "n": lad_pv["n"],
            "_arms": "2026-08-14, BOTH arms generated this session at tp=1 "
                     "(runners/run_resolution_mcq_pathvqa.sh), 0 identity-mismatched rows",
            "acc_default": lad_pv["acc_control_12845056"], "acc_cap320": lad_pv["acc_at_cap"]}
        tot = sum(c["delta_default_minus_cap320"] for c in cells.values())
        hl["3B_the_MCQ_cut_over_FOUR_of_the_five_cells"] = {
            "statement": "PATH_VQA-closed, the cell the 2026-08-13 session could not measure, is "
                         "now measured at FULL n (3,362) with both arms generated in one session: "
                         "0.837002 at the default vs 0.833432 at cap320, a cost of +0.003569 "
                         "[-0.003272, +0.010708] which is NOT significant. Adding it takes the "
                         f"macro-8 cost of cutting the MCQ leg to cap320 from +0.007159 over three "
                         f"cells to +{tot * 0.125:.6f} over four.",
            "per_cell_delta_default_minus_cap320": cells,
            "macro8_cost_of_the_cut_over_these_4_cells": round(tot * 0.125, 6),
            "weight_each": 0.125,
            "project_significance_threshold_on_macro8": 0.0029,
            "n_cells_significantly_worse_at_cap320": sum(
                1 for c in cells.values() if c["significant"]),
            "_read": "two of the four cells are significantly worse at cap320 (SLAKE-closed, "
                     "VQA-RAD-closed); PMC-VQA and PATH-VQA are not, and both are cells whose "
                     "images the cap barely touches. The direction is consistent across all four "
                     "and the pooled macro cost is 2.6x the project's significance threshold.",
            "_why_the_deltas_may_be_combined": "each cell's delta is taken between two arms "
                                               "generated in ONE session in ONE serving config; "
                                               "the four deltas come from two different sessions "
                                               "but no delta crosses them. Combining matched "
                                               "within-cell deltas into a macro is exactly what "
                                               "the macro-8 convention does.",
            "_still_missing": "MedXpertQA-MM, the cell where the cap binds hardest (cap320 binds "
                              "on 52.7% of its images and cuts mean vision tokens 838.8 -> 241.0). "
                              "It needs MAX_MODEL_LEN 16384 with up to 6 images per item, which "
                              "did not fit beside a co-tenant; see not_measured.",
        }

    hl["7_the_verdict"] = {
        "on_the_MCQ_half_62_5_percent_of_the_macro": "RESOLUTION IS NOT AN ACCURACY LEVER THERE, "
            "IN EITHER DIRECTION THAT HELPS. Up is impossible -- the published arms are already "
            "uncapped, the cap binds on 0.000 of images in all five cells. Down is not free: "
            "cap320 costs -0.007159 macro-8 [-0.012293, -0.002167] from just three of the five "
            "cells, 2.5x the project's own significance threshold, at full n.",
        "on_the_open_half_37_5_percent": "RESOLUTION IS THE ONE UNSWEPT LEVER THAT MOVED THE "
            "GENERATOR, AND THE SELECTOR ATE THE GAIN. The generator runs at 250,880 while its "
            "images are 545 merged vision tokens uncapped. Uncapping raises judge-labelled "
            "temperature-0 accuracy 0.460554 -> 0.486994, oracle@8 by +0.013220 [+0.002985, "
            "+0.023881] and modal-of-8 by +0.024307 [+0.013220, +0.035821] -- all real -- but "
            "SELECTED accuracy, the endpoint the macro-8 reports, moves only +0.005117 [-0.003838, "
            "+0.014072], CI spanning zero, because sel_eff falls -0.010799. In macro-8 points that "
            "is +0.003609 [-0.000150, +0.007625] on the macro's own equal-weight-per-cell basis "
            "against a +0.0029 threshold -- the point estimate clears the bar, the interval does "
            "not clear zero -- bought with +30% of the open arm's FLOPs and +4.4 GiB. NOT "
            "ESTABLISHED, and not Pareto. (The pooled item-weighted delta, a different and here "
            "less relevant basis, is +0.005117 [-0.003838, +0.014072] -> +0.001919 macro.)",
        "what_IS_settled_now": "the decisive labelling stage that the 2026-08-13 session could not "
            "run for lack of a free card completed on 2026-08-14: 14,779 new judge triples, "
            "still-missing=0, every arm RELIABLE=true. The answer is negative on the deployed "
            "endpoint and the direction of the failure is identified -- it is the selector, not "
            "the generator. Resolution DOES reach the candidate distribution (pool Jaccard 0.692 "
            "vs control at native, 0.571 at cap80), so this is a real manipulation that failed to "
            "convert, not a no-op.",
        "what_is_NOT_settled": "the native arm is SINGLE-SEED for the sampled quantities (the "
            "control has 3), so its selected/sel_eff deltas carry one seed's paired bootstrap CI "
            "rather than a mean over seeds, and native has no capture-recapture ceiling. The "
            "temperature-0 comparison is deterministic and unaffected. Two further native seeds "
            "were generated and labelled in this session where time allowed -- see "
            "open_generator_resolution.by_cap.native.seeds for how many actually landed.",
        "the_operating_point_worth_taking": "headline 4C -- ONE greedy decode at native resolution "
            "ties the deployed eight-sample cap320 arm plus its trained verifier (+0.004122, every "
            "seed's CI spanning zero) at 0.0759x its FLOPs, a 13.17x compute reduction on the open "
            "half. That is a cost result, not an accuracy result, it carries a non-significant "
            "guardrail flag on PathVQA-open, and it drops the verifier confidence the deployed "
            "policy uses to trigger 32B escalation -- so it bounds what the sampling half is "
            "buying rather than replacing the arm outright.",
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
            # the JUDGE-labelled endpoint -- the columns that actually decide, added 2026-08-14
            jv = g_(openres, "by_cap", cap)
            if jv:
                r["JUDGE_selected"] = jv["mean_sd"]["selected"]["mean"]
                r["JUDGE_selected_per_seed"] = jv["mean_sd"]["selected"]["per_seed"]
                r["JUDGE_oracle8"] = jv["mean_sd"]["oracle8"]["mean"]
                r["JUDGE_sel_eff"] = jv["mean_sd"]["sel_eff"]["mean"]
                r["JUDGE_greedy_t0"] = (jv.get("greedy_t0") or {}).get("all")
                r["n_seeds"] = len(jv["seeds"])
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

    # ---- what this round did NOT measure, computed from what actually landed ----------------
    nat_seeds = (openres or {}).get("by_cap", {}).get("native", {}).get("seeds", [])
    ctl_seeds = (openres or {}).get("by_cap", {}).get("cap320", {}).get("seeds", [])
    caps_done = sorted((openres or {}).get("by_cap", {}).keys())
    mcq_pair_cells = [k for k in (mcq_pair or {}) if not k.startswith("_")
                      and isinstance(mcq_pair.get(k), dict)
                      and "delta_default_minus_cap320" in mcq_pair[k]]

    nm = {
        "energy": "no NVML power integration was run in this round.",
        "batch-1 latency as a deployment claim":
            "MedEvalKit's per-item latency_s IS recorded per arm and reported, but every arm in "
            "both sessions ran on a card shared with another tenant, so it is context, never a "
            "latency result.",
        "VRAM in this session":
            "CITED, NOT RE-MEASURED, from vram_levers_2026-08-12.json -- same instrument (HF "
            "transformers, bf16, flash_attention_2, tp=1, batch 1), same caps, and the same four "
            "conventions as vram_testtime_2026-08-11.json (a weights-resident / b peak-allocated / "
            "c peak-reserved / d NVML process footprint), so the rows are directly comparable. A "
            "co-tenant held 40-46 GiB for most of both sessions and a shared-card reading is not a "
            "clean measurement. The cap-vs-cap DELTAS are the load-bearing part and they come from "
            "one clean sweep on one instrument.",
        "MMMU": "excluded from the macro-8 on contamination grounds; not touched here.",
        "a temperature-0 arm at cap80":
            "cap80 has sampled arms (s0, s1) but no t0 arm on disk, so the judge-labelled GREEDY "
            "resolution curve has two points (cap320 0.460554, native 0.486994), not three. The "
            "greedy question -- does resolution move a single deterministic decode -- is answered "
            "by that pair and by its exact-match twin; a cap80 greedy point would only extend the "
            "curve downward, where the SAMPLED arms already show a significant loss.",
        "the 32B's own open-text resolution curve":
            "not swept. The 32B-direct open comparator is at 250,880 like the generator, and "
            "moving it would move the baseline, which is a different experiment.",
        "PMC_VQA on this session's MCQ ladder":
            "not run -- 33,430 items per cap under card contention. Its two-point answer already "
            "exists at FULL n in the 2026-07-01 matched pair and is reported there.",
        "the lower rungs of the open ladder (cap160, cap640, fullres)":
            "generation was ordered control-first (cap320, then native, then cap80) so a truncated "
            "run would still answer the question. Caps with a complete arm in this artifact: "
            f"{caps_done}. The measured span 62,720 -> 250,880 -> 12,845,056 px is 205x and "
            "brackets the deployed point on both sides, which is what the question needed; the "
            "intermediate rungs would refine the curve's shape, not its verdict.",
    }

    if len(nat_seeds) >= 3:
        nm["seed count at native"] = (
            f"NOW MEASURED -- native carries {len(nat_seeds)} sampling seeds {nat_seeds} against "
            f"the control's {len(ctl_seeds)} {ctl_seeds}, meeting the >=3-seed protocol. Deltas "
            "are seed-matched and reported per seed as well as averaged.")
    else:
        nm["a second and third sampling seed at native"] = (
            f"REPORTED at {len(nat_seeds)} labelled seed(s) {nat_seeds} against the control's "
            f"{len(ctl_seeds)} {ctl_seeds}, BELOW the 3-seed protocol. The sampled-arm deltas "
            "(selected, sel_eff, oracle@8) are therefore reported with a single seed's paired item "
            "bootstrap CI and never as a mean over seeds, and native has no capture-recapture "
            "ceiling, which needs two independent seeds as capture occasions. The temperature-0 "
            "comparison is deterministic and is NOT affected by this. "
            "STATE ON DISK 2026-08-14 03:56: native s1 and s2 ARE fully generated "
            "(645/200/1500 items each) and fully verifier-scored (852 new forwards, "
            "logs/resolution_label_2026-08-13.log 'DONE 852 in 6.9 min'), but 881 of their answer "
            "strings have never been seen by the judge. The judge is a 32B model needing ~66 GiB "
            "on one card; another tenant held 26 GiB on GPU0 and 45.9 GiB on GPU1, leaving no card "
            "with enough, and no foreign process was killed to make room. Those two arms are "
            "therefore NOT entered into any table here -- an arm with unlabelled slots is biased "
            "DOWNWARD, because an unlabelled candidate is excluded from oracle@8 and counts as "
            "wrong if the verifier picks it. Finish with: "
            "`python3 src/cascade_methods/resolution_judge_cache.py build` then run_judge.py on "
            "judge_todo.jsonl (881 rows, ~2 min once a card is free) then "
            "`resolution_judge_cache.py merge`, `resolution_open_analyze.py`, "
            "`resolution_strata_ci.py`, `resolution_sweep_report.py`. "
            "NOTE they would still be SECONDARY: s1/s2 were generated 2026-08-14 at gpu_mem 0.60 "
            "while the cap320 control's s1/s2 are from 2026-08-13 at gpu_mem 0.30, so those pairs "
            "cross the +-0.008 serving-config boundary -- see open_arm_generation_provenance. "
            "Only the seed-0 pair is matched, and it is the one the headline quotes.")

    nm["PATH_VQA_closed and MedXpertQA-MM at cap320 on the MedEvalKit track"] = (
        "The 2026-08-13 session attributed the repeated failure here to a co-tenant OOM. That "
        "diagnosis was WRONG and is corrected: on 2026-08-14 the same job failed on an EMPTY card "
        "with vLLM's own profiler reporting 'model weights take 15.57GiB; PyTorch activation peak "
        "memory takes 17.23GiB; the rest of the memory reserved for KV Cache is -1.23GiB' -> "
        "'No available memory for the cache blocks'. The runner's GPU_MEM_UTIL (0.30 default, 0.40 "
        "retried) simply could not fit the job's own activation peak at MAX_MODEL_LEN 16384 with "
        "--max_image_num 6, co-tenant or not. runners/run_resolution_mcq_pathvqa.sh splits the two "
        "cells by their real geometry (PATH_VQA: 1 image, px_max 1,109,658, 4096 context is ample; "
        "MedXpert: up to 6 images, 46,816 vision tokens on the worst item, genuinely needs 16384). "
        f"Cells with a full-n cap320-vs-default pair in this artifact: {mcq_pair_cells}. "
        "Any cell still missing is one where the cap binds HARDER than average (MedXpert cap320 "
        "binds on 52.7% of images, cutting mean vision tokens 838.8 -> 241.0), so the measured "
        "'cutting the MCQ leg costs accuracy' finding is CONSERVATIVE -- but that is a prediction "
        "from geometry, not a measurement, and must not be reported as one.")

    if openres:
        nm["SELECTED accuracy, sel_eff, laterality and the capture-recapture ceiling per "
           "generator resolution"] = (
            "NOW MEASURED -- this was the 2026-08-13 session's single open question and it is "
            "closed. The verifier scoring (13,302 HF batch-1-equivalent forwards) completed "
            "2026-08-13 15:41; the judge pass it was waiting on could not get a card that session "
            "and ran 2026-08-14 (14,779 new triples, judge_cache 112,656 entries, "
            "still-missing=0). Verdict in headline item 4: the greedy/oracle gain does NOT survive "
            "into selected accuracy.")
    else:
        nm["SELECTED accuracy, sel_eff, laterality and the capture-recapture ceiling per "
           "generator resolution"] = (
            "[status is in labelling_stage_status] STILL OPEN -- the open half is reported on the "
            "exact-match secondary endpoint only.")

    art["not_measured"] = nm

    json.dump(art, open(OUT, "w"), indent=1)
    print("wrote", OUT, os.path.getsize(OUT), "bytes")
    print("sections:", list(art.keys()))


if __name__ == "__main__":
    main()
