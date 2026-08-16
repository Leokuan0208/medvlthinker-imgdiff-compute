#!/usr/bin/env python3
"""vrestruct_finalize2.py -- assemble artifacts/verifier_restructure_2026-08-16.json.

THE DELIVERABLE CHANGED MID-ROUND (2026-08-16): the method is HEAD-ONLY.  The LoRA adapter is
dropped, the generator-frame head reads a layer-21 state captured DURING generation, and
verification cost therefore goes to ~0.  The open arm's cost IS the generation cost, so the whole
question becomes: what does drawing 8 samples actually cost, and can the last unshared term --
the vision tower -- be shared?

This supersedes vrestruct_finalize.py, which was written for the fused (head + adapter) method.
The adapter work is retained here as a BANKED section, not deleted: it contains the one result
that constrains the head-only decision (head-only is a significant exact-match LOSS against the
adapter it replaces).

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_finalize2.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import vrestruct_lib as V     # noqa: E402

PARTS = V.PARTS
OUT = os.path.join(V.ART, "verifier_restructure_2026-08-16.json")
MCQ = {"PMC_VQA": 0.542656, "SLAKE_closed": 0.825359, "VQA_RAD_closed": 0.780876,
       "PATH_VQA_closed": 0.840869, "MedXpertQA-MM": 0.2615}
MCQ_SRC = ("artifacts/sevenb_only_frontier_2026-08-12.json PART1_7B_only_frontier."
           "menu_per_cell_accuracy_EVAL_VISIBLE.*.greedy_7b -- the 5 MCQ cells stay at greedy-7B "
           "because no 7B-only MCQ mechanism has ever measured positive on this pool (that file's "
           "menu_note)")
SHARE = dict(vision=0.2537, lm_prefill=0.7348, decode_and_head=0.0115)


def load(n):
    p = os.path.join(PARTS, n)
    return json.load(open(p)) if os.path.exists(p) else None


def macro8(open_cells):
    cells = dict(MCQ)
    cells.update(open_cells)
    return float(np.mean(list(cells.values()))), cells


def main():
    ho = load("headonly.json")
    vs = load("vision_sharing.json")
    pref = load("prefill_analysis.json")
    st = load("structures.json")
    fh = load("freehead.json")
    wzf = load("weitzman_frozen.json")
    rf = load("resolution_fused.json")
    probe = load("vision_probe.json")
    mci = load("macro_ci.json")
    c = V.cost_constants()
    if ho is None:
        raise SystemExit("headonly.json missing")

    # ------------------------------------------------------------------ generation cost (Q1)
    gen = {}
    if pref:
        for k, row in pref["verdict"].items():
            pts = {n: row[f"N{n}"]["flopeq_rel_to_N1"] for n in (1, 2, 4, 8)
                   if row.get(f"N{n}") and row[f"N{n}"].get("flopeq_rel_to_N1") is not None}
            if pts:
                gen[k] = dict(flopeq_rel_to_N1=pts,
                              lm_prefill_sharing_ratio={n: row[f"N{n}"]["lm_prefill_sharing_ratio"]
                                                        for n in pts},
                              vision_sharing_ratio={n: row[f"N{n}"]["vision_sharing_ratio"]
                                                    for n in pts},
                              wall_rel_to_N1={n: row[f"N{n}"]["wall_rel_to_N1"] for n in pts})
    G8_default = gen.get("count|default", {}).get("flopeq_rel_to_N1", {}).get(8)
    G8_off = gen.get("count|off", {}).get("flopeq_rel_to_N1", {}).get(8)

    # best measured vision ratio across the sharing arms, and the implied generation cost
    best_arm = best_vis = None
    if vs:
        for arm, a in vs["arms"].items():
            r8 = a.get("N8", {}).get("vision_sharing_ratio")
            if r8 and (best_vis is None or r8["mean"] < best_vis):
                best_arm, best_vis = arm, r8["mean"]
    lm8 = (vs["arms"]["A_batch16_default"]["N8"]["lm_prefill_sharing_ratio"]["mean"]
           if vs else None)
    # the STOCK vision ratio (arm A) -- what the project's generations actually paid.
    vis_stock = (vs["arms"]["A_batch16_default"]["N8"]["vision_sharing_ratio"]["mean"]
                 if vs else None)
    # THE MEASURED fixed-vision cost: arm I (pre-computed image embeddings) -- vision ratio 1.000
    # with generation intact (6.3 generated tokens per sequence against the stock arm's 6.0).
    # Arm J (prompt padding) also reaches 1.000 but its generation COLLAPSED to 1.0 token per
    # sequence, because the padding lands inside the assistant turn, so J confirms the mechanism
    # and is NOT a deployable cost.
    armI = (vs or {}).get("arms", {}).get("I_image_embeds_b16", {})
    G8_if_vision_shared = (armI.get("implied_open_arm_flopeq_headonly", {}) or {}).get("value")
    if G8_if_vision_shared is None and lm8:
        G8_if_vision_shared = (lm8 * SHARE["lm_prefill"] + 1.0 * SHARE["vision"]
                               + 8 * SHARE["decode_and_head"])
    G8_best_measured = (lm8 * SHARE["lm_prefill"] + best_vis * SHARE["vision"]
                        + 8 * SHARE["decode_and_head"]) if (lm8 and best_vis) else None

    # ------------------------------------------------------------------ head-only accuracy
    dep = ho["arms"]["deployed_cache_fullres_TF"]
    cap = ho["arms"].get("captured_cap320_ar", dep)
    m_dep, cells_dep = macro8(dep["per_cell_judge"])
    m_cap, cells_cap = macro8(cap["per_cell_judge"])
    base_macro = float(np.mean(list(MCQ.values()) + [0.736434, 0.465, 0.324]))

    head_cost = ho["cost"]

    def crow(name, gen_flopeq, head_flopeq, acc_macro8, note, provenance):
        tot = gen_flopeq + head_flopeq
        return dict(structure=name, gen_flopeq=gen_flopeq, verification_flopeq=head_flopeq,
                    open_question_flopeq=tot, macro8_flopeq=(5 * 1.0 + 3 * tot) / 8.0,
                    macro8_accuracy=acc_macro8, note=note, provenance=provenance)

    free = head_cost["total_head_flopeq_per_question"]
    tf = head_cost["removed_total_flopeq"]
    table = [
        crow("baseline_always_7b", 1.0, 0.0, base_macro,
             "one greedy Lingshu-7B answer per question -- the baseline the claim is judged against",
             "MEASURED (the project's own always-7B cells)"),
        crow("head_only__head_recomputed__generation_as_charged", 8.0, tf, m_dep,
             "the project's own cost convention (1.0 FLOP-eq per sample) with the head still read "
             "by a separate teacher-forced pass at max_pixels 1,003,520",
             "cost: convention A + MODELLED head pass; accuracy: MEASURED"),
        crow("head_only__head_recomputed__generation_measured", G8_default, tf, m_dep,
             "same pipeline, generation charged at what it MEASURABLY costs (Q1)",
             "cost: MEASURED generation + MODELLED head pass; accuracy: MEASURED"),
        crow("head_only__head_CAPTURED__generation_measured__DEPLOYABLE_TODAY",
             G8_default, free, m_cap,
             "the head's layer-21 state captured DURING generation at the generator's own cap320. "
             "Verification is 2.07e-05 FLOP-eq/question -- five orders of magnitude below "
             "everything else here. THE OPEN ARM'S COST IS NOW THE GENERATION COST.",
             "cost: MEASURED generation + MEASURED head arithmetic; accuracy: MEASURED on "
             "feats_free/free_cap320_L21.h_span_ar.npy"),
    ]
    if G8_if_vision_shared:
        table.append(crow("head_only__head_CAPTURED__vision_shared_via_image_embeds__THE_PRIZE",
                          G8_if_vision_shared, free, m_cap,
                          "as above, with each image encoded ONCE and handed to vLLM as "
                          "pre-computed embeddings instead of being re-encoded ~4.77 times. "
                          "MEASURED: vision_sharing_ratio 1.000 exactly (vit_patches == "
                          "patches_ref) with generation intact -- 6.3 generated tokens per "
                          "sequence against the stock arm's 6.0.",
                          "cost: MEASURED end to end (vision_sharing.jsonl arm "
                          "I_image_embeds_b16); accuracy: carried over from the deployable "
                          "head-only arm -- moving where the image embedding is computed does not "
                          "change the embedding, and this round did NOT re-score the pool through "
                          "the embeds path, so the accuracy is asserted-unchanged, not re-measured"))
    if G8_off:
        table.append(crow("control__prefix_caching_OFF", G8_off, free, m_cap,
                          "what the pipeline would cost if automatic prefix caching were disabled "
                          "-- the clean control that validates the instrument",
                          "MEASURED"))

    # ------------------------------------------------------------------ Q1: generation sharing
    q1 = dict(
        question="What does drawing 8 samples actually cost? Does SamplingParams(n=N) share work?",
        answer=(
            "The LANGUAGE-MODEL prefill IS shared and the VISION TOWER IS NOT. At N=8 with prefix "
            f"caching on (vLLM 0.9.0.1 V1's default, so this is what every generation in this "
            f"project got) the LM prefill runs {lm8:.3f}x -- one prefill plus a block-granularity "
            f"remainder, not eight -- while the vision tower runs {vis_stock:.3f}x. "
            f"Net measured cost of 8 samples = {G8_default:.3f}x one greedy answer, against 8.0 "
            f"as-charged by the project and {V.G_of_N(8, c):.4f} under cost_floor convention B. "
            "BOTH EXISTING CONVENTIONS ARE WRONG: the project overcharges generation by "
            f"{8.0 / G8_default:.2f}x and convention B undercharges it by "
            f"{G8_default / V.G_of_N(8, c):.2f}x." if (lm8 and G8_default) else "NOT MEASURED"),
        instruments=(pref or {}).get("instruments"),
        design=(pref or {}).get("design"),
        controlled_AB=dict(
            cache_off_scales_as_N=dict(
                lm_prefill_sharing_ratio_N8=gen.get("count|off", {})
                .get("lm_prefill_sharing_ratio", {}).get(8),
                vision_sharing_ratio_N8=gen.get("count|off", {})
                .get("vision_sharing_ratio", {}).get(8),
                flopeq_rel_to_N1_N8=G8_off,
                _read="with caching off every term scales as exactly N (7.976x, 8.000x). That is "
                      "what validates the instrument: it reproduces the project's own as-charged "
                      "8.0 convention to within 0.3% when nothing is shared."),
            cache_on=dict(
                lm_prefill_sharing_ratio_N8=gen.get("count|on", {})
                .get("lm_prefill_sharing_ratio", {}).get(8),
                vision_sharing_ratio_N8=gen.get("count|on", {})
                .get("vision_sharing_ratio", {}).get(8)),
            default_equals_on="vLLM 0.9.0.1 V1 turns automatic prefix caching ON by default; the "
                              "'default' arm's effective_enable_prefix_caching is True."),
        per_config=gen)

    # ------------------------------------------------------------------ Q1b: the vision tower
    q1b = dict(question="Why is the vision tower only ~4.77/8 shared, and can it be fixed?")
    if vs:
        q1b["mechanism_read_from_the_installed_source"] = vs["verdict"]["mechanism_from_source"]
        q1b["arms"] = {k: dict(meaning=a["meaning"], submission_batch=a["submission_batch"],
                               max_num_seqs=a["max_num_seqs"],
                               effective_max_num_seqs=a.get("effective_max_num_seqs"),
                               mm_input_cache_gib=a["mm_input_cache_gib"],
                               prime=a.get("prime"), embeds=a.get("embeds"),
                               lm_prefill_sharing_ratio_N8=a.get("N8", {})
                               .get("lm_prefill_sharing_ratio"),
                               vision_sharing_ratio_N8=a.get("N8", {}).get("vision_sharing_ratio"),
                               wall_s_N8=a.get("N8", {}).get("wall_s"),
                               implied_open_arm_flopeq=a.get("implied_open_arm_flopeq_headonly"))
                       for k, a in vs["arms"].items()}
        q1b["control_reproduces"] = vs["verdict"]["control_reproduces"]
        q1b["what_does_NOT_work"] = (
            "Every scheduling and cache-size remedy failed, and they failed IDENTICALLY -- to three "
            "decimal places on the same item slices. Submission batch 16 / 4 / 1: 4.766 in all "
            "three. max_num_seqs 256 / 8 / 1: 4.766 in all three, and max_num_seqs=1 demonstrably "
            "took effect (wall clock 33.0 s against 3.1 s, a 10x serialisation). "
            "VLLM_MM_INPUT_CACHE_GIB 4 -> 32: 4.766, exactly as the source predicts, because that "
            "variable sizes the PREPROCESSED-INPUT cache (vllm/v1/engine/mm_input_cache.py) and "
            "not the encoder-output cache. Priming each question with a 1-token n=1 request first "
            "made it WORSE (5.304): the priming pass is itself an encode and it bought no skips. "
            "The invariance is the finding: this is not a cache-thrashing or concurrency problem, "
            "so it cannot be tuned away.")
        q1b["why"] = (
            "TWO FACTS, THE SECOND MEASURED INSIDE THE SCHEDULER. (1) vLLM 0.9.0.1 keys the "
            "encoder-OUTPUT cache by REQUEST ID (gpu_model_runner.py:147, dict[req_id][input_id]) "
            "and SamplingParams(n=N) becomes N child requests with N distinct ids "
            "(v1/engine/parallel_sampling.py), so siblings can never share an encoder-cache entry "
            "by design. The single escape hatch in "
            "v1/core/sched/scheduler.py:_try_schedule_encoder_inputs is "
            "`if start_pos + num_encoder_tokens <= num_computed_tokens: continue`. "
            "(2) THAT ESCAPE HATCH MISSES BY A FRACTION OF ONE KV BLOCK. A probe on the live "
            "scheduler recorded, for a typical child: start_pos 31, num_encoder_tokens 294 -- so "
            "the image span ends at token 325 -- against num_computed_tokens 320, the "
            "block-aligned prefix-cache hit for a 336-token prompt (KV block size 16, and the hit "
            "is floor((T-1)/16)*16 because a request must always compute at least one token). "
            "325 > 320, so vLLM RE-ENCODES ALL 294 IMAGE TOKENS TO RECOVER FIVE. The prefix cache "
            "IS hitting -- that is why the LM prefill shares at 1.14x -- it just stops a few "
            "tokens short of the end of the image. Whether a given item is covered depends on "
            "where its image span happens to fall relative to a 16-token boundary, which is why "
            "the ratio sits at an item-determined ~0.6*N and why it is invariant to every "
            "scheduling and cache knob: those knobs change residency and concurrency, and the "
            "binding constraint is arithmetic alignment.")
        if probe:
            q1b["scheduler_probe"] = {k: v for k, v in probe.items() if k != "example_events"}
    emb = (vs or {}).get("arms", {}).get("I_image_embeds_b16") if vs else None
    pad = (vs or {}).get("arms", {}).get("J_pad_tail_b16") if vs else None
    q1b["THE_FIX"] = dict(
        route="pass PRE-COMPUTED image embeddings to vLLM (arm I) -- MEASURED, WORKS",
        source_basis="vllm/model_executor/models/qwen2_5_vl.py:972 -- "
                     "`if image_input['type'] == 'image_embeds': image_embeds = "
                     "image_input['image_embeds']` else `self.visual(pixel_values, ...)`. "
                     "Supplying multi_modal_data={'image': {'image_embeds': T, "
                     "'image_grid_thw': G}} bypasses the vision tower entirely.",
        measured_vision_sharing_ratio_N8=(emb or {}).get("N8", {}).get("vision_sharing_ratio"),
        measured_lm_prefill_sharing_ratio_N8=(emb or {}).get("N8", {})
        .get("lm_prefill_sharing_ratio"),
        implied_open_arm_flopeq=(emb or {}).get("implied_open_arm_flopeq_headonly"),
        exactness="vit_patches == patches_ref EXACTLY in all four cells, so the ratio is 1.000, "
                  "not approximately 1.",
        generation_intact="6.3 generated tokens per sequence at N=8 against the stock arm's 6.0 -- "
                          "the arm is doing the real workload, not a degenerate one.",
        honesty_note="the one encode per image is performed with the ENGINE'S OWN vision tower "
                     "through the same counting hook, so this arm is not handed a free image.",
        gotcha="vLLM's multimodal serialisation rejects bf16 ('Got unsupported ScalarType "
               "BFloat16'); cast the embeddings to fp16 first. qwen2_5_vl.py:973 casts back with "
               ".type(self.visual.dtype), so fp16 round-trips.")
    q1b["MECHANISM_CONFIRMATION_not_a_recommendation"] = dict(
        route="lengthen the text AFTER the image so the block-aligned prefix hit covers the whole "
              "image span (arm J)",
        measured_vision_sharing_ratio_N8=(pad or {}).get("N8", {}).get("vision_sharing_ratio"),
        verdict="This is the decisive test of the diagnosis: adding ~25 tokens of tail text takes "
                "the vision ratio from 4.766 to 1.000 EXACTLY, with no other change. Nothing else "
                "tried moved it at all.",
        WHY_IT_IS_NOT_A_RECOMMENDATION="as implemented the padding is appended to the templated "
                                       "prompt, which puts it INSIDE the assistant turn, and the "
                                       "model then stops almost immediately: 1.0 generated token "
                                       "per sequence against the stock arm's 6.0. The arm proves "
                                       "the mechanism; its cost and accuracy are not deployable. A "
                                       "real version would have to lengthen the USER text, which "
                                       "changes the prompt and therefore the answers, and that "
                                       "accuracy effect is NOT measured here.")
    q1b["the_prize"] = dict(
        measured_today=G8_default,
        measured_with_the_fix=G8_if_vision_shared,
        further_reduction_factor=(G8_default / G8_if_vision_shared
                                  if (G8_default and G8_if_vision_shared) else None),
        decomposition="lm_prefill_ratio*0.7348 + vision_ratio*0.2537 + 8*0.0115, shares from "
                      "flop_ratio_derivation_2026-08-03 component_shares_pct.lingshu_7b")

    # ------------------------------------------------------------------ Q2: head-only accuracy
    q2 = dict(
        question="HEAD-ONLY on its own terms: what does it buy over always-7B, and at what N?",
        deployable_arm="captured_cap320_ar -- the layer-21 state captured during generation at the "
                       "generator's own max_pixels 250,880, which is what a deployment gets",
        arms={k: {kk: vv for kk, vv in a.items() if kk != "fixed_N"} for k, a in ho["arms"].items()},
        fixed_N_curve={k: {str(n): dict(
            acc_judge=a["fixed_N"]["by_N"][str(n)]["judge"]["acc"],
            acc_em=a["fixed_N"]["by_N"][str(n)]["em"]["acc"],
            macro3_judge=a["fixed_N"]["by_N"][str(n)]["judge"]["macro3"],
            vs_N8_judge=a["fixed_N"]["by_N"][str(n)]["judge"]["vs_N8"],
            vs_greedy7b_judge=a["fixed_N"]["by_N"][str(n)]["judge"]["vs_greedy7b"])
            for n in range(1, 9)} for k, a in ho["arms"].items()},
        macro8=dict(baseline_always_7b=base_macro,
                    head_only_deployed_cache=m_dep, head_only_captured_cap320=m_cap,
                    delta_captured_vs_baseline=m_cap - base_macro,
                    cells_captured=cells_cap, mcq_source=MCQ_SRC),
        THE_HEADLINE=(
            "HEAD-ONLY, CAPTURED DURING GENERATION, IS +0.009086 [+0.000949,+0.017187] ON THE "
            "8-CELL MACRO AGAINST ALWAYS-7B, AT 1.514x THE 7B'S OWN COMPUTE -- 1.076x once the "
            "vision tower is shared, which is now MEASURED (Q1b). It is significant, and the "
            "lower bound is 0.0009 -- a hair "
            "above zero. The same pool with the adapter still in the fusion gives +0.019191 "
            "[+0.011830,+0.026685], and head-only LOSES to it by -0.010105 [-0.016188,-0.004273], "
            "significant. So the head-only simplification costs a measured, significant HALF of "
            "the total gain, and it is bought back as compute, not accuracy."
            if mci else "NOT COMPUTED"),
        macro8_with_CIs=(mci or {}).get("arms"),
        THE_GUARDRAIL_PROBLEM=(
            "head-only is NOT guardrail-clean against always-7B greedy. On the deployable arm the "
            "pooled open-text gain is +0.045203 [+0.029851,+0.060981] and PATH_VQA_open "
            "(+0.061333) and SLAKE_open (+0.026357) carry it, but VQA_RAD_open is NEGATIVE at "
            "-0.015000 [-0.070000,+0.040000]. The fused selector this replaces WAS guardrail-clean "
            "(VQA_RAD_open +0.045000). Dropping the adapter breaks the guardrail on one of three "
            "cells. The interval is wide (n=200) so this is not a significant loss -- but "
            "'never worse than always-cheap on any single benchmark' is the project's own "
            "standing criterion and head-only does not meet it."),
        T04_status=ho["T04_status"])

    # ------------------------------------------------------------------ Q3: Weitzman
    q3 = dict(question="Refit the adaptive-N controller with verification cost ~0.")
    wa = dep.get("weitzman_box_raw_logit")
    if wa:
        q3["answer"] = (
            "THE CONTROLLER IS NOW WORTHLESS AND SHOULD BE DELETED. With inspection free, the "
            "optimal policy draws essentially the whole pool: meanN lands at 7.94-8.00 of 8 across "
            "every cost setting (c_cheap = 1.0 generation-only, 0.011544 shared-prefill marginal, "
            "and 3.2e-07 head-inspection), and its accuracy at the optimum EQUALS fixed N=8's to "
            "six decimals. Weitzman's reservation value is a function of the inspection cost; "
            "drive that to zero and there is no reason to stop early, and the fixed-N curve says "
            "stopping early is exactly what you must not do. The brief's expectation that a "
            "near-zero inspection cost would REDUCE N is refuted in both directions.")
        q3["scenarios_deployed_cache"] = {k: dict(c_cheap=v["c_cheap"], best=v["best"],
                                                  cheapest_tying=v["cheapest_tying"])
                                          for k, v in wa["scenarios"].items()}
        cw = cap.get("weitzman_box_raw_logit")
        if cw:
            q3["scenarios_captured_cap320"] = {k: dict(c_cheap=v["c_cheap"], best=v["best"],
                                                       cheapest_tying=v["cheapest_tying"])
                                               for k, v in cw["scenarios"].items()}
        q3["the_box_value_must_be_cardinal"] = (
            "A head-only method has no adapter score to stop on, so the box value has to come from "
            "the head. The head's RAW mean logit is cardinal and cross-question comparable and "
            "works; the selector's rank_avg score is a WITHIN-QUESTION rank and does not -- fed to "
            "the controller it collapses to meanN 1.63 at accuracy 0.4730-0.4738, far below fixed "
            "N=8. That degenerate control is reported alongside every scenario.")
        q3["degenerate_control"] = {k: dict(best=v["best"]) for k, v in
                                    dep["weitzman_box_rank_DEGENERATE_CONTROL"]["scenarios"].items()}
        q3["head_inspection_cost_flopeq"] = wa["head_inspection_cost_flopeq"]

    # ------------------------------------------------------------------ banked
    banked = dict(
        _why="the user dropped the adapter from this deliverable on 2026-08-16. These results are "
             "measured, null-tested and reusable; they are shelved, not retracted. One of them "
             "constrains the head-only decision and is repeated in Q2's guardrail note.",
        head_only_vs_adapter_only_vs_fused=(st or {}).get("comparisons"),
        THE_CURRENCY_REVERSAL=(
            "head-only beats adapter-only by +0.016205 [+0.002559,+0.029851] under the 32B judge "
            "and LOSES by -0.014499 [-0.028571,-0.000426] under normalised exact match, on "
            "IDENTICAL picks. Both are significant and they point opposite ways. The fusion beats "
            "head-only by +0.017484 [+0.007249,+0.028145] in exact match, guardrail-clean, and by "
            "+0.007136 [+0.001980,+0.012480] on the 8-cell macro. So the adapter was doing real "
            "work in the exact-match currency, and head-only buys its simplicity and its zero cost "
            "at a measured price."),
        free_head_evaluation=(fh or {}).get("verdict"),
        free_head_arms=(fh or {}).get("arms"),
        prefix_shared_verifier="artifacts/shared_prefix_verifier_2026-08-16.json -- a real "
                               "implementation, measured TIE, verifier FLOP-eq/question "
                               "7.4654 -> 2.1862. Banked.",
        verifier_resolution_through_the_fusion=(rf or {}).get("rungs"),
        fused_fixed_N_curve=(wzf or {}).get("fixed_N_curve"))

    lim = [
        "THE VISION FIX IS MEASURED AS A COST, NOT AS AN ACCURACY. Arm I drives "
        "vision_sharing_ratio to exactly 1.000 with generation intact, so the 1.203 FLOP-eq figure "
        "is a real measurement. But this round did NOT re-score the 2,345-question pool through "
        "the embeds path: the accuracy of that row is carried over on the argument that moving "
        "where an image embedding is computed does not change the embedding. That argument is "
        "sound in exact arithmetic and approximate in bf16/fp16, so it is an ASSERTION, not a "
        "measurement. Re-scoring the pool through the embeds path is the obvious next job.",
        "HEAD-ONLY IS NOT GUARDRAIL-CLEAN. VQA_RAD_open goes negative (-0.015000) against "
        "always-7B greedy on the deployable arm. n=200 there, so the interval is wide and this is "
        "not a significant loss, but the project's own guardrail criterion is not met.",
        "HEAD-ONLY IS A SIGNIFICANT EXACT-MATCH LOSS AGAINST THE ADAPTER IT REPLACES "
        "(-0.014499 [-0.028571,-0.000426]). That is banked, not retracted.",
        "T=0.4 IS NOT MEASURED FOR HEAD-ONLY. The generator-frame feature cache and the free-head "
        "capture exist only for the deployed T=0.7 pool; evaluating the head at T=0.4 needs a new "
        "~90-minute capture per generation seed. The adapter-box Weitzman refit at T=0.4 IS "
        "measured (weitzman.json) but it is not the head-only method.",
        "NOTHING WAS RUN END TO END. As with every operating point in this project, the accuracy "
        "numbers are re-scorings of saved per-candidate dumps and the cost numbers are a "
        "measured-token re-costing; the recommended pipeline has never been executed as one "
        "program. (CLAUDE.md standing caveat.)",
        "THE GENERATION MEASUREMENT IS 16 QUESTIONS PER CELL x 3 REPLICATES on real SLAKE/PathVQA "
        "images at cap320, not the full 2,345-question pool. Sharing ratios are ratios of counted "
        "tokens and patches, which are exact for the items measured; their variation across item "
        "slices is visible in the per-arm sd fields.",
        "THE +0.0115 DECODE SHARE SCALES WITH N in the constructed rows, which slightly favours "
        "the shared-vision row; at N=8 that term is 0.092 FLOP-eq, about 8% of the constructed "
        "total, so it cannot change the conclusion.",
        "EM sel_eff against the always-7B greedy baseline is not available: the greedy arm's "
        "exact-match labels are not in the frozen transfer dumps. Selector-vs-selector comparisons "
        "are in both currencies throughout; only vs-greedy is judge-only.",
    ]

    out = dict(
        title="HEAD-ONLY BEST-OF-8 ON A 7B MEDICAL VLM: what it buys, what it costs, and why the "
              "vision tower is the last unshared term",
        date="2026-08-16",
        objective="baseline = ALWAYS-7B (macro 0.5971, 1.0 FLOP-eq/question). Claim shape: a small "
                  "verifier improves a 7B medical VLM by +X on N of 8 cells at Y x the 7B's own "
                  "compute. With head-only + captured states, verification is ~0 and Y IS the "
                  "generation cost.",
        scripts=["src/cascade_methods/vrestruct_lib.py",
                 "src/cascade_methods/vrestruct_prefill.py",
                 "src/cascade_methods/vrestruct_prefill_analyze.py",
                 "src/cascade_methods/vrestruct_vision_sharing.py",
                 "src/cascade_methods/vrestruct_vision_analyze.py",
                 "src/cascade_methods/vrestruct_vision_probe.py",
                 "src/cascade_methods/vrestruct_headonly.py",
                 "src/cascade_methods/vrestruct_freehead_eval.py",
                 "src/cascade_methods/vrestruct_structures.py",
                 "src/cascade_methods/vrestruct_weitzman.py",
                 "src/cascade_methods/vrestruct_weitzman_frozen.py",
                 "src/cascade_methods/vrestruct_resolution_fused.py",
                 "src/cascade_methods/vrestruct_finalize2.py"],
        numerics_pinned=dict(OMP_NUM_THREADS="4 analysis / 8 controller refit", PYTHONHASHSEED="0",
                             nboot=V.NBOOT, bootstrap_seed=V.BOOT_SEED,
                             rank_convention="rank_avg (rank_argsort gives 0.798365, NOT this)",
                             row_order="concat", tf32="off in the capture (meta tf32=false); the "
                                                      "analysis scripts are CPU numpy",
                             vllm="0.9.0.1, V1 engine, VLLM_ENABLE_V1_MULTIPROCESSING=0, "
                                  "enforce_eager=True for the counting phase"),
        null_tests=dict(
            frozen_metric=(st or {}).get("null_tests", {}).get("NT1_frozen_metric"),
            head_only_reproduces=ho.get("null_test"),
            free_head_harness=(fh or {}).get("null_tests"),
            generation_instrument="cache-off arm scales as exactly N (LM 7.976x, vision 8.000x), "
                                  "reproducing the as-charged convention when nothing is shared",
            vision_control="arm A reproduces vrestruct_prefill.py's count|default vision ratio on "
                           "pinned identical item slices",
            _summary="frozen metric max abs deviation 3.5967e-07 (n=2345/1468, sel_eff 0.775204, "
                     "oracle@8 0.626013, greedy 0.449467); the free-head teacher-forced path "
                     "reproduces the deployed feature cache with 0 picks changed and 0.0 "
                     "deviation in sel_eff; selected = oracle@8 x sel_eff holds to 5.55e-17."),
        cost_constants=c,
        THE_COST_TABLE=table,
        Q1_generation_cost_and_prefill_sharing=q1,
        Q1b_the_vision_tower=q1b,
        Q2_head_only_accuracy=q2,
        Q3_adaptive_N_with_free_verification=q3,
        BANKED_adapter_work=banked,
        limitations_and_what_could_not_be_settled=lim,
        no_fabricated_numbers=True,
        not_abstention="every structure here returns an answer; no reject option anywhere")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    print(f"\n{'structure':66s} {'gen':>7s} {'verif':>9s} {'open':>7s} {'macro8$':>8s} {'macro8acc':>9s}")
    for r in table:
        print(f"{r['structure']:66s} {r['gen_flopeq']:7.3f} {r['verification_flopeq']:9.5f} "
              f"{r['open_question_flopeq']:7.3f} {r['macro8_flopeq']:8.3f} "
              f"{r['macro8_accuracy']:9.6f}")


if __name__ == "__main__":
    main()
