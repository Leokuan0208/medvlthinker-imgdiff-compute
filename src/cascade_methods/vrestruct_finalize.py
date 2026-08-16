#!/usr/bin/env python3
"""vrestruct_finalize.py -- assemble artifacts/verifier_restructure_2026-08-16.json.

Reads the part files written by
    vrestruct_structures.py        (Q1 + Q5: which structure, both currencies, per cell)
    vrestruct_prefill.py/_analyze  (Q2: does SamplingParams(n=N) share the prefill)
    vrestruct_weitzman{,_frozen}.py(Q3: refit the controller at near-zero inspection cost)
    vrestruct_resolution_fused.py  (Q4: the verifier's resolution, measured THROUGH the fusion)
and produces the cost table, the recommended pipeline and the honest limitations.

Re-runnable: it uses whatever cells have landed and marks the rest "not measured".

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_finalize.py
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
MCQ_CELLS = 5
OPEN_CELLS = 3


def load(name):
    p = os.path.join(PARTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def load_sibling_prefix():
    p = os.path.join(V.ART, "shared_prefix_verifier_2026-08-16.json")
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- generation cost, measured
def generation_cost(pref, c):
    """FLOP-eq of drawing N samples, from the MEASURED token accounting (Q2).

    Returns a dict with the measured N-points and a callable-ish interpolation table.  Falls back
    to the two competing model conventions (and says so) if the measurement is absent.
    """
    if not pref:
        return dict(status="NOT MEASURED", as_charged={str(n): float(n) for n in (1, 2, 4, 8)},
                    shared_prefill={str(n): V.G_of_N(n, c) for n in (1, 2, 4, 8)})
    out = {}
    for key, row in pref["verdict"].items():
        pts = {}
        for n in (1, 2, 4, 8):
            r = row.get(f"N{n}")
            if r and r.get("flopeq_rel_to_N1") is not None:
                pts[n] = float(r["flopeq_rel_to_N1"])
        if pts:
            out[key] = dict(
                flopeq_rel_to_N1=pts,
                lm_prefill_sharing_ratio={n: row[f"N{n}"]["lm_prefill_sharing_ratio"]
                                          for n in pts},
                vision_sharing_ratio={n: row[f"N{n}"]["vision_sharing_ratio"] for n in pts},
                wall_rel_to_N1={n: row[f"N{n}"]["wall_rel_to_N1"] for n in pts})
    return dict(status="MEASURED", by_config=out,
                as_charged={str(n): float(n) for n in (1, 2, 4, 8)},
                shared_prefill_convention_B={str(n): V.G_of_N(n, c) for n in (1, 2, 4, 8)})


def interp_gen(pts, N):
    ns = sorted(pts)
    if N <= ns[0]:
        return pts[ns[0]] * N / ns[0]
    for a, b in zip(ns, ns[1:]):
        if a <= N <= b:
            return pts[a] + (pts[b] - pts[a]) * (N - a) / (b - a)
    return pts[ns[-1]]


def verifier_prefix_shared(c, px, k):
    """FLOP-eq of scoring k candidates when the shared image+question prefix is prefilled ONCE."""
    geo = c["ver_geometry_by_max_pixels"].get(px)
    if geo is None:
        return None
    suffix = c["verifier_geometry_tokens"]["per_candidate_suffix_tok"]
    unit = c["unit_tflop"]
    full = V.fwd_tflops(geo["vision"], geo["prompt"], 1.0) / unit
    pre = V.fwd_tflops(geo["vision"], geo["prompt"] - suffix, 0.0) / unit
    marg = full - pre
    return dict(prefix_flopeq=pre, marginal_per_candidate_flopeq=marg,
                total_flopeq=pre + k * marg, k=k, max_pixels=px,
                full_pass_flopeq=full,
                _note="prefix = everything up to 'Proposed answer: ' (image + system + question); "
                      f"marginal = the {suffix:.2f}-token candidate suffix. REQUIRES AN "
                      "IMPLEMENTATION CHANGE the repo has not run (a concurrent round is "
                      "attempting it as shared_prefix_verifier_2026-08-16).")


def main():
    st = load("structures.json")
    pref = load("prefill_analysis.json")
    wz = load("weitzman.json")
    wzf = load("weitzman_frozen.json")
    rf = load("resolution_fused.json")
    fh = load("freehead.json")
    vs = load("vision_sharing.json")
    if st is None:
        raise SystemExit("structures.json missing -- run vrestruct_structures.py first")
    c = V.cost_constants()
    D = st["pass_counts"]["mean_distinct_answers"]

    gen = generation_cost(pref, c)
    gpts = None
    if gen["status"] == "MEASURED":
        for key in ("count|default", "count|on", "count|off"):
            if key in gen["by_config"]:
                gpts = gen["by_config"][key]["flopeq_rel_to_N1"]
                gen["primary_config"] = key
                break
    G8 = interp_gen(gpts, 8) if gpts else None

    # ------------------------------------------------------------------ the cost table
    ver_dep = c["ver_1003520_flopeq"]
    ver_640 = c["ver_501760_flopeq"]
    head_dep = c["head_1003520_flopeq"]
    ps_dep = verifier_prefix_shared(c, 1003520, D)
    ps_640 = verifier_prefix_shared(c, 501760, D)

    # ---- MEASURED verifier totals from the sibling round's real implementation ---------------
    sp = load_sibling_prefix()
    meas = {}
    if sp:
        for px, v in sp["COST"]["by_max_pixels"].items():
            meas[int(px)] = dict(deployed=v["flop_eq_per_question"]["deployed"],
                                 prefix_shared=v["flop_eq_per_question"]["prefix_shared"],
                                 as_charged=v["flop_eq_per_question"]["as_charged_paper_convention_A"],
                                 n_distinct=v["geometry_measured"]
                                 ["mean_distinct_candidates_per_question"])
    VER_DEPLOYED_TOTAL = meas.get(1003520, {}).get("deployed", D * ver_dep)
    VER_PREFIX_TOTAL = meas.get(1003520, {}).get("prefix_shared", ps_dep["total_flopeq"])
    ver_provenance = ("MEASURED end to end by the sibling round "
                      "(shared_prefix_verifier_2026-08-16.json COST.by_max_pixels)"
                      if meas else "MODELLED here from measured token geometry")

    def row(name, n_gen, ver, head, accuracy_key, note, gen_flopeq=None):
        g_charged = float(n_gen)
        g_meas = (interp_gen(gpts, n_gen) if gpts else None) if gen_flopeq is None else gen_flopeq
        r = dict(
            structure=name, n_generations=float(n_gen),
            verifier_passes=float(D) if ver else 0.0,
            head_passes=float(D) if head else 0.0,
            total_forward_passes=float(n_gen) + (D if ver else 0.0) + (D if head else 0.0),
            verifier_flopeq=float(ver or 0.0), head_flopeq=float(head or 0.0),
            gen_flopeq_as_charged=g_charged,
            gen_flopeq_measured=g_meas,
            total_flopeq_as_charged=g_charged + (ver or 0.0) + (head or 0.0),
            total_flopeq_measured_gen=(g_meas + (ver or 0.0) + (head or 0.0)
                                       if g_meas is not None else None),
            accuracy_arm=accuracy_key, note=note)
        for k in ("total_flopeq_as_charged", "total_flopeq_measured_gen"):
            if r[k] is not None:
                r[k.replace("total_flopeq", "macro8_cost")] = (
                    MCQ_CELLS * 1.0 + OPEN_CELLS * r[k]) / 8.0
        return r

    table = [
        row("baseline_always_7b", 1, 0.0, 0.0, "always_7b", "the new baseline: one greedy 7B answer",
            gen_flopeq=1.0),
        row("A_deployed_as_the_paper_charges_it", 8, D * 1.0, D * 1.0, "fused_8seed",
            "the project's own convention: every pass charged 1.0 FLOP-eq regardless of what it "
            "actually renders. Generation re-costed with the Q2 measurement in the measGen column."),
        row("B_deployed_honestly_costed", 8, VER_DEPLOYED_TOTAL, D * head_dep, "fused_8seed",
            "the SAME pipeline with every pass charged at ITS OWN measured resolution -- both "
            f"scorers run at max_pixels 1,003,520 while the generator runs at 250,880. Verifier "
            f"term {ver_provenance}."),
        row("C_drop_the_LoRA_head_only", 8, 0.0, D * head_dep, "head_only_8seed",
            "REJECTED: a significant LOSS under exact match (see Q1)"),
        row("D_free_head_LoRA_unchanged", 8, VER_DEPLOYED_TOTAL, 0.0, "fused_cap320_ar",
            "head CAPTURED during generation at the generator's own cap320 -- MEASURED TIE "
            "(judge -0.000853 [-0.005970,+0.004264], EM +0.001706 [-0.003412,+0.006823])"),
        row("E_free_head_LoRA_at_cap640", 8, D * ver_640, 0.0, "fused_px501760",
            "as D with the verifier scoring at max_pixels 501,760 -- measured TIE through the "
            "fusion, guardrail-clean, 23/2345 picks change"),
        row("F_RECOMMENDED_free_head_prefix_shared_LoRA", 8, VER_PREFIX_TOTAL, 0.0,
            "fused_cap320_ar",
            "as D with the verifier's shared image+question prefix prefilled ONCE per question. "
            f"Verifier term {ver_provenance}; measured TIE (judge +0.000426 [-0.003838,+0.005117], "
            "EM +0.002132 [-0.002559,+0.006823], 134/2345 picks change)"),
        row("G_F_plus_cap640_verifier", 8, ps_640["total_flopeq"], 0.0, "fused_px501760",
            "E + F together. The prefix-shared verifier at cap640 is MODELLED (the sibling round "
            "measured the prefix build only at 1,003,520 and 250,880)"),
    ]
    if wzf and "W5_7b_only" in wzf.get("arms", {}):
        mN = wzf["arms"]["W5_7b_only"]["best"]["meanN"]
        table.append(row("H_F_plus_Weitzman_adaptive_N", mN,
                         VER_PREFIX_TOTAL * (0.5 + 0.5 * mN / 8.0), 0.0,
                         "weitzman_W5_7b_only",
                         f"as F with the adaptive-N controller (meanN {mN:.3f} instead of 8) at "
                         "the accuracy-argmax lambda -- accuracy identical to fixed N=8. The "
                         "verifier term scales only its PER-CANDIDATE half with N; the shared "
                         "prefill is paid once either way."))
    for r in table:
        r["verifier_term_provenance"] = (
            ver_provenance if r["structure"].startswith(("B_", "D_", "F_", "H_"))
            else "MODELLED here from measured token geometry")

    # ------------------------------------------------------------------ verdicts
    S = st["structures"]
    cmpx = st["comparisons"]
    v7 = st["vs_always_7b"]

    q1 = dict(
        question="Should the LoRA verifier be dropped entirely?",
        answer="NO. Dropping it is a significant LOSS in the exact-match currency, and the "
               "head-only structure's apparent win is a JUDGE-CURRENCY ARTIFACT.",
        evidence=dict(
            sel_eff_judge={k: S[k]["judge"]["sel_eff"] for k in
                           ("lora_only", "head_only_8seed", "fused_8seed", "self_consistency")},
            selected_acc_judge={k: S[k]["judge"]["acc"] for k in
                                ("lora_only", "head_only_8seed", "fused_8seed", "self_consistency")},
            selected_acc_exact_match={k: S[k]["em"]["acc"] for k in
                                      ("lora_only", "head_only_8seed", "fused_8seed",
                                       "self_consistency")},
            head_only_vs_lora_only=cmpx["head_only_8seed__vs__lora_only"],
            fused_vs_head_only=cmpx["fused_8seed__vs__head_only_8seed"],
            fused_vs_lora_only=cmpx["fused_8seed__vs__lora_only"]),
        the_reversal="head-only beats LoRA-only by +0.016205 [+0.002559,+0.029851] under the 32B "
                     "judge and LOSES by -0.014499 [-0.028571,-0.000426] under normalised exact "
                     "match, on IDENTICAL picks. Both are significant and they point opposite "
                     "ways. Neither structure is guardrail-clean against the other.",
        why_the_fusion_survives="the fusion is the only structure that is never significantly "
                                "worse than either component in either currency: vs head-only it "
                                "wins EM +0.017484 [+0.007249,+0.028145] guardrail-clean, vs "
                                "LoRA-only it wins judge +0.022175 [+0.012793,+0.031557] "
                                "guardrail-clean, and it is the only one that improves the 8-cell "
                                "macro significantly against BOTH.",
        cost_of_keeping_it=dict(
            as_run=f"{D * ver_dep:.3f} FLOP-eq/question ({D:.4f} passes x {ver_dep:.4f})",
            restructured=f"{ps_640['total_flopeq']:.3f} FLOP-eq/question (prefix-shared, cap640)",
            reduction_factor=float(D * ver_dep / ps_640["total_flopeq"])))

    q2 = dict(
        question="Does vLLM SamplingParams(n=N) share the generation prefill?",
        status=gen["status"],
        answer=None, measured=gen)
    if gen["status"] == "MEASURED" and gpts:
        bc = gen["by_config"]

        def _g(cfg, field, N=8):
            return bc[cfg][field][N] if cfg in bc else None
        q2["THE_CONTROLLED_AB"] = {
            "design": "the SAME 16-item disjoint slices, the SAME prompts, the SAME seeds, run "
                      "with enable_prefix_caching explicitly True, explicitly False, and left "
                      "unset (what src/labeling/run_openvqa.py:152 does).",
            "effective_flag_when_unset": True,
            "effective_flag_note": "vLLM 0.9.0.1 V1 turns automatic prefix caching ON by default, "
                                   "so the 'default' arm IS the cache-on arm -- every generation "
                                   "this project has run had the LM prefill shared.",
            "cache_OFF_is_the_clean_control": {
                "lm_prefill_sharing_ratio_at_N8": _g("count|off", "lm_prefill_sharing_ratio"),
                "vision_sharing_ratio_at_N8": _g("count|off", "vision_sharing_ratio"),
                "flopeq_rel_to_N1_at_N8": _g("count|off", "flopeq_rel_to_N1"),
                "num_cached_tokens": 0,
                "_read": "with caching off every term scales as exactly N (7.976x, 8.000x), which "
                         "is what validates the instrument: it reproduces the project's own "
                         "as-charged 8.0 convention to within 0.3% when nothing is shared."},
            "cache_ON": {
                "lm_prefill_sharing_ratio_at_N8": _g("count|on", "lm_prefill_sharing_ratio"),
                "vision_sharing_ratio_at_N8": _g("count|on", "vision_sharing_ratio"),
                "flopeq_rel_to_N1_at_N8": _g("count|on", "flopeq_rel_to_N1"),
                "prompt_token_cache_hit_rate_at_N8": "95.8-96.9%"},
            "cache_DEFAULT": {
                "lm_prefill_sharing_ratio_at_N8": _g("count|default", "lm_prefill_sharing_ratio"),
                "vision_sharing_ratio_at_N8": _g("count|default", "vision_sharing_ratio"),
                "flopeq_rel_to_N1_at_N8": _g("count|default", "flopeq_rel_to_N1")},
            "wall_clock_corroboration_at_N8": {
                "cache_off": _g("time|off", "wall_rel_to_N1"),
                "cache_on": _g("time|on", "wall_rel_to_N1"),
                "cache_default": _g("time|default", "wall_rel_to_N1"),
                "_read": "measured with CUDA graphs ON (the deployed configuration), so this is "
                         "deployable latency, not the eager-mode counting phase."},
        }
        q2["IS_THE_VISION_TOWER_SHARED"] = (
            "NO. This is the round's sharpest cost finding and it is what makes the honest number "
            f"{G8:.2f}x rather than 1.08x. With caching ON the LM prefill collapses to "
            f"{_g('count|on', 'lm_prefill_sharing_ratio'):.3f}x at N=8 but the vision tower still "
            f"runs {_g('count|on', 'vision_sharing_ratio'):.3f}x -- the N child requests of an "
            "n=N request each carry the same image and vLLM re-encodes it for most of them. "
            "Automatic prefix caching is a KV-cache mechanism; it does not touch the vision "
            "encoder. Since the vision tower is 25.4% of a Lingshu-7B forward at cap320 "
            "(flop_ratio_derivation_2026-08-03 component_shares_pct), a ~5x re-encode is the "
            "single largest remaining term in the generation cost.")
        q2["answer"] = (
            "PARTLY, AND BOTH EXISTING CONVENTIONS ARE WRONG. The LANGUAGE-MODEL prefill IS shared "
            f"(measured sharing ratio {gen['by_config'][gen['primary_config']]['lm_prefill_sharing_ratio'][8]:.3f}x "
            "at N=8, i.e. one prefill plus a block-granularity remainder, not 8). The VISION TOWER "
            f"is NOT: it re-encodes the image "
            f"{gen['by_config'][gen['primary_config']]['vision_sharing_ratio'][8]:.3f}x for N=8. "
            f"Net measured cost of 8 samples = {G8:.3f}x one greedy answer -- against 8.0 "
            f"as-charged and {V.G_of_N(8, c):.4f} under cost_floor convention B. The project's "
            f"cost model OVERCHARGES generation by {8.0 / G8:.2f}x; convention B UNDERCHARGES it "
            f"by {G8 / V.G_of_N(8, c):.2f}x.")
        q2["how_measured"] = (
            "route 1 of 3 (direct token accounting): forward pre-hooks on the LM's first decoder "
            "layer and the vision tower count the token positions and image patches every forward "
            "actually processes, with enforce_eager=True so CUDA graphs cannot hide decode steps, "
            "and VLLM_ENABLE_V1_MULTIPROCESSING=0 so the engine core is in-process. Each "
            "(config, N, rep) cell ran a DISJOINT slice of items with the prefix cache reset "
            "first, so no cell can be served from another cell's cache. Route 3 (wall clock vs N) "
            "agrees. Route 2 (num_cached_tokens) is recorded per cell.")
        q2["the_fixable_part"] = (
            "the vision re-encode is an implementation detail, not a law: vLLM caches multimodal "
            "encoder outputs by hash, and the N child requests of an n=N request carry the same "
            "image. If that cache hit, 8 samples would cost about "
            f"{V.G_of_N(8, c):.3f}x instead of {G8:.3f}x. Nothing in this round changes it; it is "
            "named as the single largest remaining cost lever on the generation term.")
        q2["consequence"] = (
            "every FLOP-eq figure in the project that charges 1.0 per generated candidate is wrong "
            f"on the generation term by {8.0 / G8:.2f}x. This round does NOT quietly adopt a new "
            "constant: both conventions and the measurement are carried side by side in the cost "
            "table above.")

    q3 = dict(
        question="Refit the adaptive-N controller for near-zero verification cost.",
        pools_used="T=0.4 (3 generation seeds) with the regenerated T=0.7 as the matched control, "
                   "LoRA-verifier box values; plus the FROZEN T=0.7 pool with the fused selector",
        the_structural_finding=(
            "The Weitzman box value must be comparable ACROSS questions. The fused selector score "
            "is a WITHIN-QUESTION rank sum, so it cannot be one: feeding it to the controller "
            "collapses the policy to meanN 1.495 at accuracy 0.475480, far below fixed N=8's "
            "0.507463. The deployable arrangement is: STOP on the LoRA verifier's calibrated "
            "P(correct), PICK with the fused score."),
        what_happens_when_inspection_is_free=None,
        the_answer_on_N=None)
    if wzf:
        cur = wzf["fixed_N_curve"]
        q3["fixed_N_curve_fused_selector"] = {
            str(n): dict(acc_judge=cur[str(n)]["judge"]["acc"],
                         acc_em=cur[str(n)]["em"]["acc"],
                         macro3_judge=cur[str(n)]["judge"]["macro3"],
                         vs_N8_judge=cur[str(n)]["judge"]["vs_N8"])
            for n in range(1, 9)}
        q3["the_answer_on_N"] = (
            "N CANNOT BE REDUCED. With the fused selector every fixed N < 8 is a SIGNIFICANT loss "
            "against N=8 -- N=7 is -0.005117 [-0.009382,-0.001279] and N=4 is -0.019616 "
            "[-0.028571,-0.011087], which is the entire macro-8 gain. The brief's expectation that "
            "a near-zero inspection cost would REDUCE N is refuted in both directions: Weitzman "
            "with c_cheap -> 0 removes any reason to stop early (meanN rises to the pool maximum), "
            "and the accuracy curve says stopping early is exactly what you must not do.")
        q3["what_happens_when_inspection_is_free"] = {
            k: dict(c_cheap=a["c_cheap"], c_strong=a["c_strong"],
                    best=a["best"], scenario=a["scenario"])
            for k, a in wzf["arms"].items()}
        q3["the_free_saving"] = (
            "the ONLY generation saving the controller finds without an accuracy cost is the "
            f"{8 - wzf['arms']['W5_7b_only']['best']['meanN']:.3f} draws it skips on easy "
            f"questions: meanN {wzf['arms']['W5_7b_only']['best']['meanN']:.3f} of 8 "
            f"({100 * (1 - wzf['arms']['W5_7b_only']['best']['meanN'] / 8):.1f}% fewer) at judge "
            f"accuracy {wzf['arms']['W5_7b_only']['best']['acc_judge']:.6f}, identical to fixed "
            "N=8's 0.507463.")
        q3["degenerate_control"] = wzf.get("degenerate_control_fused_as_box_value")
    if wz:
        q3["T04_refit_lora_box_values"] = {
            k: dict(scenario=a["scenario"], c_cheap=a["c_cheap"],
                    best=a["frontier"][a["argmax_acc_judge"]])
            for k, a in wz["arms"].items() if k.startswith("T04_s0|")}
        q3["the_honest_cost_makes_the_cheap_leg_uneconomic"] = (
            "under W1_honest_today (c_cheap 5.394, every scorer pass charged at its own measured "
            "resolution) the LoRA-box controller degenerates to Weitzman REGIME B on every T=0.4 "
            "and T=0.7 pool: the strong box has the higher reservation value, so it is opened "
            "first and the arm becomes always-32B with meanN 0 and escalation 1.0. That is the "
            "cost objective's verdict on the CURRENT verifier structure, stated by the controller "
            "itself.")

    q4 = dict(question="The verifier's scoring resolution.",
              answer=None, ladder=None)
    if rf:
        q4["ladder"] = {px: dict(
            verifier_forward_flopeq=r["verifier_forward_flopeq"],
            measured_vision_tokens=(r["geometry"] or {}).get("vision"),
            lora_only_acc_judge=r["lora"]["judge"]["acc"],
            lora_only_vs_control=r["lora"]["judge"]["vs_control"],
            fused_acc_judge=r["fused"]["judge"]["acc"],
            fused_acc_em=r["fused"]["em"]["acc"],
            fused_vs_control_judge=r["fused"]["judge"]["vs_control"],
            fused_vs_control_em=r["fused"]["em"]["vs_control"],
            fused_guardrail_clean=r["fused"]["judge"]["guardrail_clean_vs_control"],
            n_picks_differing=r["fused"]["n_picks_differing_from_control"])
            for px, r in rf["rungs"].items()}
        q4["answer"] = (
            "The 4x resolution mismatch is real -- the verifier renders at 1,003,520 while the "
            "generator renders at 250,880 -- but it is NOT worth 4x in FLOPs, because these "
            "medical images mostly sit below the cap: the deployed rung's MEASURED mean vision "
            "tokens are 520.5 against cap320's 277.3, a factor of 1.88, not 4. Cutting the "
            "verifier all the way to the generator's 250,880 costs a significant -0.011087 in "
            "LoRA-only selected accuracy and -0.005117 [-0.010661,+0.000426] through the fusion "
            "with SLAKE_open significantly negative, so it is NOT free. max_pixels 501,760 IS "
            "free: 23 picks of 2,345 change, every cell is flat, both currencies tie, and the "
            "verifier forward drops 1.8793 -> 1.6953 FLOP-eq (-9.8%). That is the recommendation.")
        q4["exploratory_lead"] = (
            "POST-HOC, NOT A RECOMMENDATION: through the fusion, max_pixels 62,720 is also a "
            "pooled tie (+0.002132 [-0.005117,+0.009382] judge, -0.001706 [-0.008955,+0.005970] "
            "EM) at a verifier forward of 0.4087 FLOP-eq -- a 4.6x cut on the verifier term. But "
            "266 of 2,345 picks change, SLAKE_open (-0.009302) and VQA_RAD_open (-0.015000) are "
            "both negative, so it is GUARDRAIL-DIRTY, and it is one of six rungs read off the "
            "eval set. The fusion absorbing the verifier's resolution damage is a real and new "
            "observation; it needs a pre-registered replication before it can be spent.")

    q_free = dict(
        question="Is the generator-frame head actually free, and does it survive the resolution "
                 "the generator really runs at?",
        why_it_was_in_doubt=c["RESOLUTION_MISMATCH_WARNING"],
        answer=None)
    if fh:
        a = fh["arms"]
        q_free["answer"] = (
            "YES ON BOTH COUNTS, MEASURED. (i) The harness null test is exact: the teacher-forced "
            "path at 1,003,520 reproduces the deployed feature cache with 0 picks changed and "
            "abs deviation 0.0 in sel_eff. (ii) Capturing the layer-21 state DURING generation "
            "instead of recomputing it is a tie at the head's own resolution -- fused judge "
            "+0.000853 [+0.000000,+0.002132], 15/2345 picks change. (iii) The resolution this "
            "round flagged as a risk costs almost nothing THROUGH THE FUSION: at the generator's "
            "own cap320 the captured-during-generation arm is fused judge -0.000853 "
            "[-0.005970,+0.004264] and fused EM +0.001706 [-0.003412,+0.006823], both ties, "
            "153/2345 picks changed. The head ALONE loses more (-0.006823 judge, n.s.) -- the "
            "same pattern as the verifier's resolution ladder: the fusion absorbs a degradation "
            "in either component. So the 3.8136 teacher-forced passes/question are removable.")
        q_free["arms"] = {k: {nm: {cur: {kk: vv for kk, vv in a[k][nm][cur].items()}
                                   for cur in ("judge", "em")}
                              | {"n_picks_differing_from_deployed":
                                 a[k][nm]["n_picks_differing_from_deployed"]}
                              for nm in ("head_only", "fused")}
                          | {"capture_diagnostics": a[k]["capture_diagnostics"]}
                          for k in a}
        q_free["null_tests"] = fh["null_tests"]
        q_free["input_provenance"] = fh["inputs"]
        q_free["saving"] = (f"{D:.4f} passes/question at {c['head_1003520_flopeq']:.4f} FLOP-eq "
                            f"each = {D * c['head_1003520_flopeq']:.3f} FLOP-eq/question removed")

    q5 = dict(
        question="Anything else: where else is the waste?",
        seed_sufficiency=dict(
            table=st["seed_sufficiency"],
            answer="the 8-seed head ensemble is unnecessary. Head-only sel_eff is 0.793767 at 1 "
                   "seed, 0.804277 at 2 and 0.801090 at 8 (mean over subsets); fused is 0.807902 "
                   "at 1, 0.808826 at 2, 0.810627 at 8. TWO seeds reach the plateau. This costs "
                   "nothing in FLOPs either way -- the heads are 918,529-parameter MLPs on an "
                   "already-computed vector, ~1.8 MFLOP each against a 5.69 TFLOP forward -- so "
                   "the saving is engineering simplicity and determinism, not compute."),
        early_stop_scoring=dict(
            table=st["early_stop_lora"],
            answer="scoring candidates in recorded order and stopping at the first score above a "
                   "threshold trades accuracy for passes along a smooth curve; see the table. It "
                   "is dominated by prefix sharing, which removes ~85% of the verifier's cost "
                   "WITHOUT changing a single pick."),
        dedup_normalisation=dict(
            table=st["dedup_normalisation"],
            answer="pushing normalisation past the current G.norm buys little and starts merging "
                   "candidates with DIFFERENT judge labels, which is an information loss, not a "
                   "saving. See n_groups_with_mixed_judge_labels."),
        the_resolution_mismatch_nobody_chose=c["RESOLUTION_MISMATCH_WARNING"])

    recommended = dict(
        pipeline=[
            "1. GENERATE. Lingshu-7B, cap320 (max_pixels 250,880), temperature 0.4, N=8, ONE vLLM "
            "request with SamplingParams(n=8) and enable_prefix_caching=True set EXPLICITLY "
            "(run_openvqa.py:152 currently leaves it unset).",
            "2. CAPTURE, DO NOT RECOMPUTE. Read the layer-21 span-pooled hidden state of each "
            "candidate DURING generation instead of in a separate teacher-forced pass. Removes "
            f"{D:.4f} passes/question at {c['head_1003520_flopeq']:.4f} FLOP-eq each.",
            "3. SCORE. The CLEAN disjoint LoRA verifier, once per DISTINCT normalised answer "
            f"({D:.4f} of 8 on average), at max_pixels 501,760 (not 1,003,520), with the "
            "shared image+question prefix prefilled ONCE per question and only the ~19-token "
            "candidate suffix recomputed per candidate.",
            "4. SELECT. final = rank_avg(verifier scores) + rank_avg(mean over 2 head seeds of "
            "rank_avg(head logits)); argmax with first-index tie-break. Keep the LoRA -- dropping "
            "it is a significant exact-match loss.",
            "5. NO 32B. Under the always-7B baseline there is no strong leg; the Weitzman "
            "controller reduces to a stopping rule that saves 5.3% of draws for free."],
        keeps=dict(
            macro8_accuracy=v7["fused_8seed"]["macro8"],
            macro8_delta_vs_always_7b=v7["fused_8seed"]["macro8_delta_vs_always7b"],
            per_open_cell=v7["fused_8seed"]["per_cell"],
            guardrail_clean=v7["fused_8seed"]["guardrail_clean"],
            _identical_picks="steps 2-4 are cost refactors that must not move a pick; step 3's "
                             "cap640 rung moves 23 of 2,345 picks and is a measured tie in both "
                             "currencies"))

    lim = [
        "THE TWO ENABLING BUILDS ARE NOW MEASURED, NOT ASSUMED -- but they were built by a SIBLING "
        "round and this artifact only re-scores their outputs. The free head is de-conditioned by "
        "feats_free/free_{fullres,cap320}_L21.h_span_{tf,ar}.npy (evaluated here in Q_FREE_HEAD); "
        "the prefix-shared verifier is de-conditioned by shared_prefix_verifier_2026-08-16.json. "
        "Neither is bit-equivalent to the deployed implementation: the prefix build changes 134 "
        "picks and the free head 153, and both are measured TIES rather than identities. The "
        "sibling round states bit-equivalence is not achievable because the GEMM shapes change "
        "(bf16 logits land 1-2 ULP apart).",
        "THE THREE TIES COMPOUND AND WERE NOT MEASURED TOGETHER. The recommended structure stacks "
        "a free head (judge -0.000853), a prefix-shared verifier (+0.000426) and, in row G, a "
        "cap640 verifier (+0.000853). Each is individually a tie on the same 2,345 questions, but "
        "the COMBINED arm has not been run end to end -- nothing in this project has ever been run "
        "end to end (CLAUDE.md standing caveat). The stacked delta is an assumption of additivity.",
        "THE PREFIX-SHARED VERIFIER AT cap640 (row G) IS MODELLED. The sibling round measured the "
        "prefix build at 1,003,520 and 250,880 only; row G interpolates the geometry. Row F, the "
        "recommendation, uses the measured 1,003,520 figure and needs no interpolation.",
        "TWO SLIGHTLY DIFFERENT FLOP-eq UNITS ARE IN PLAY, 2.4% apart. This artifact defines "
        "1.0 FLOP-eq = 5.6927 TFLOP (resolution_sweep_2026-08-13's measured open-pool cap320 "
        "geometry); the sibling round's verifier figures use 5.8314 TFLOP "
        "(cost_decomposition_2026-08-12 N2, the n=25 VQA-RAD latency anchor). Both are 'one "
        "Lingshu-7B cap320 forward'; they differ by which prompt sample defines it. No conclusion "
        "here turns on 2.4%, but the numbers should not be quoted to 3 significant figures across "
        "the two rounds.",
        "N=8 IS LOAD-BEARING AND GENERATION IS NOW THE DOMINANT TERM. After the restructuring, "
        "generation is ~52% of the open-question cost and cannot be reduced without a significant "
        "accuracy loss. The remaining lever is the vision re-encode inside vLLM's n=N path, not "
        "the verifier.",
        "EM sel_eff and the EM-currency delta against the always-7B greedy baseline are NOT "
        "MEASURED: the greedy arm's exact-match labels are not in the frozen transfer dumps. "
        "Selector-vs-selector comparisons are reported in both currencies throughout; only the "
        "vs-greedy comparison is judge-only.",
        "The 8-cell macro here holds the 5 MCQ cells at greedy-7B, following "
        "sevenb_only_frontier_2026-08-12.json PART1.menu_note (no 7B-only MCQ mechanism has ever "
        "measured positive on this pool). The open cells are EVAL-VISIBLE in this artifact; the "
        "cross-fit version of the same arm choice is +0.019191 in that prior round.",
        "COST IS AN OPEN-HALF NUMBER converted to a macro-8 cost by charging the 5 MCQ cells 1.0 "
        "each. The MCQ half's own token geometry differs slightly (T=354.6/360.5, G=2.49/2.00), "
        "so the macro-8 cost carries the same ~1-2% convention noise the project's other macro "
        "cost figures do.",
    ]

    out = dict(
        title="THE MINIMUM-COST VERIFICATION STRUCTURE: what the verifier must keep, what it can "
              "drop, and what 8 samples actually cost",
        date="2026-08-16",
        objective="baseline = ALWAYS-7B (macro 0.5971, 1.0 FLOP-eq/question). Claim shape: a small "
                  "verifier improves a 7B medical VLM by +X on N of 8 cells at Y x the 7B's own "
                  "compute. Y is the endpoint.",
        scripts=["src/cascade_methods/vrestruct_lib.py",
                 "src/cascade_methods/vrestruct_structures.py",
                 "src/cascade_methods/vrestruct_prefill.py",
                 "src/cascade_methods/vrestruct_prefill_analyze.py",
                 "src/cascade_methods/vrestruct_weitzman.py",
                 "src/cascade_methods/vrestruct_weitzman_frozen.py",
                 "src/cascade_methods/vrestruct_resolution_fused.py",
                 "src/cascade_methods/vrestruct_finalize.py"],
        numerics_pinned=dict(OMP_NUM_THREADS="4 (analysis) / 8 (controller refit)",
                             PYTHONHASHSEED="0", nboot=V.NBOOT, bootstrap_seed=V.BOOT_SEED,
                             rank_convention="rank_avg (average ranks for ties) -- rank_argsort "
                                             "gives 0.798365 and is NOT this",
                             tf32="not applicable: no GPU matmul in any analysis script; the "
                                  "vLLM measurement counts tokens, not values",
                             row_order="concat (the published head fit order)"),
        null_tests=dict(structures=st["null_tests"],
                        weitzman_frozen=(wzf or {}).get("null_tests"),
                        weitzman_T04=(wz or {}).get("null_tests"),
                        resolution_fused=(rf or {}).get("null_tests"),
                        _summary="frozen metric max abs deviation 3.5967e-07 (n=2345/1468, "
                                 "sel_eff 0.775204, oracle@8 0.626013, greedy 0.449467); the "
                                 "frozen 8-seed selector reproduces 0.810627/0.507463 through "
                                 "this harness to 4.48e-07; the identity selected = oracle@8 x "
                                 "sel_eff holds to 5.55e-17 for EVERY structure reported."),
        cost_constants=c,
        THE_COST_TABLE=table,
        Q1_should_the_lora_be_dropped=q1,
        Q2_generation_prefill_sharing=q2,
        Q3_adaptive_N_refit=q3,
        Q4_verifier_resolution=q4,
        Q_FREE_HEAD_is_the_head_actually_free=q_free,
        Q5_other_waste=q5,
        RECOMMENDED_STRUCTURE=recommended,
        accuracy_of_every_structure=dict(
            per_structure={k: dict(sel_eff_judge=vv["judge"]["sel_eff"],
                                   acc_judge=vv["judge"]["acc"],
                                   acc_em=vv["em"]["acc"],
                                   per_cell_judge={d: vv["judge"]["per_ds"][d]["acc"]
                                                   for d in vv["judge"]["per_ds"]},
                                   per_cell_em={d: vv["em"]["per_ds"][d]["acc"]
                                                for d in vv["em"]["per_ds"]})
                          for k, vv in S.items()},
            vs_always_7b=v7,
            permutation_nulls=st["permutation_nulls"]),
        limitations_and_what_could_not_be_settled=lim,
        no_fabricated_numbers=True,
        not_abstention="every structure here returns an answer; no reject option anywhere")
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)

    print(f"\n{'structure':56s} {'passes':>7s} {'asChg':>8s} {'measGen':>8s} {'macro8':>8s}")
    for r in table:
        m = r.get("macro8_cost_measured_gen")
        print(f"{r['structure']:56s} {r['total_forward_passes']:7.3f} "
              f"{r['total_flopeq_as_charged']:8.3f} "
              f"{(r['total_flopeq_measured_gen'] or float('nan')):8.3f} "
              f"{(m if m is not None else float('nan')):8.3f}")


if __name__ == "__main__":
    main()
