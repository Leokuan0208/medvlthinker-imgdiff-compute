#!/usr/bin/env python3
"""vrestruct_vision_analyze.py -- aggregate the vision-sharing arms into the Q2b verdict.

Reads _vrestruct_parts/vision_sharing.jsonl and reports, per arm, the LM-prefill and vision
sharing ratios at N=8 against the identical 16-item slices, plus what each arm implies about the
open-arm cost under HEAD-ONLY (where verification is ~0 and generation is the whole cost).

    OMP_NUM_THREADS=2 python3 src/cascade_methods/vrestruct_vision_analyze.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import vrestruct_lib as V     # noqa: E402

PARTS = V.PARTS
SINK = os.path.join(PARTS, "vision_sharing.jsonl")

ARM_MEANING = {
    "A_batch16_default": "CONTROL. 16 questions submitted together, stock config -- must reproduce "
                         "vrestruct_prefill.py's count|default numbers.",
    "B_batch1_default": "one question per generate() call: only 8 child requests are ever in "
                        "flight, so if the cause were cache thrashing from concurrency this arm "
                        "would recover.",
    "C_batch16_maxseqs8": "max_num_seqs=8 -- caps concurrent sequences at one question's worth.",
    "D_batch16_maxseqs1": "max_num_seqs=1 -- forces strictly sequential execution, so every child "
                          "after the first MUST see its sibling's completed prefill.",
    "E_batch16_mmcache32": "VLLM_MM_INPUT_CACHE_GIB raised 4 -> 32.",
    "F_batch4_default": "4 questions per call -- an intermediate concurrency point.",
    "G_prime_then_fanout_b16": "THE CANDIDATE FIX. A 1-token n=1 request per question FIRST (its "
                               "prompt blocks become computed and resident), then the n=8 request. "
                               "The priming pass is real work and is counted.",
    "H_prime_then_fanout_b1": "as G, one question at a time.",
}


def main():
    if not os.path.exists(SINK):
        raise SystemExit("no vision_sharing.jsonl yet")
    recs = [json.loads(l) for l in open(SINK) if l.strip()]
    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], r["N"])].append(r)

    def agg(rows, k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return dict(mean=float(np.mean(xs)), sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                    n=len(xs), values=[float(x) for x in xs]) if xs else None

    c = V.cost_constants()
    # workload decomposition of ONE Lingshu-7B cap320 forward (flop_ratio_derivation_2026-08-03)
    SHARE = dict(vision=0.2537, lm_prefill=0.7348, decode_and_head=0.0115)
    arms = {}
    for (arm, N), rows in sorted(by.items()):
        a = arms.setdefault(arm, dict(meaning=ARM_MEANING.get(arm, ""),
                                      submission_batch=rows[0]["submission_batch"],
                                      max_num_seqs=rows[0]["max_num_seqs"],
                                      effective_max_num_seqs=rows[0].get("effective_max_num_seqs"),
                                      mm_input_cache_gib=rows[0]["mm_input_cache_gib"],
                                      prime=rows[0].get("prime", False),
                                      n_distinct_images=rows[0].get("n_distinct_images")))
        a[f"N{N}"] = dict(lm_prefill_sharing_ratio=agg(rows, "lm_prefill_sharing_ratio"),
                          vision_sharing_ratio=agg(rows, "vision_sharing_ratio"),
                          wall_s=agg(rows, "wall_s"),
                          gen_tok_total=agg(rows, "gen_tok_total"))
    # implied open-arm cost under HEAD-ONLY: generation is the whole cost
    for arm, a in arms.items():
        r8 = a.get("N8")
        if not r8 or not r8["vision_sharing_ratio"]:
            continue
        lm = r8["lm_prefill_sharing_ratio"]["mean"]
        vi = r8["vision_sharing_ratio"]["mean"]
        a["implied_open_arm_flopeq_headonly"] = dict(
            formula="lm_prefill_ratio*0.7348 + vision_ratio*0.2537 + 8*0.0115 "
                    "(decode scales with N; shares from flop_ratio_derivation_2026-08-03 "
                    "component_shares_pct for Lingshu-7B at cap320)",
            value=lm * SHARE["lm_prefill"] + vi * SHARE["vision"] + 8 * SHARE["decode_and_head"],
            lm_term=lm * SHARE["lm_prefill"], vision_term=vi * SHARE["vision"],
            decode_term=8 * SHARE["decode_and_head"])

    ctrl = arms.get("A_batch16_default", {})
    best = None
    for arm, a in arms.items():
        v = a.get("N8", {}).get("vision_sharing_ratio")
        if v and (best is None or v["mean"] < best[1]):
            best = (arm, v["mean"])

    verdict = dict(
        question="Why is the vision tower only ~4.7/8 shared, and can it be made fully shared?",
        mechanism_from_source=[
            "vllm/v1/worker/gpu_model_runner.py:147 -- self.encoder_cache is dict[req_id][input_id]"
            " -> torch.Tensor. The ENCODER-OUTPUT cache is keyed by REQUEST ID.",
            "vllm/v1/engine/parallel_sampling.py -- SamplingParams(n=N) becomes N CHILD requests "
            "with N distinct request ids, so two children of the same question can NEVER share an "
            "encoder-cache entry.",
            "vllm/v1/core/sched/scheduler.py:_try_schedule_encoder_inputs -- a request skips the "
            "vision tower by exactly ONE route: `if start_pos + num_encoder_tokens <= "
            "num_computed_tokens: continue`, i.e. its PREFIX-CACHE hit must already cover the whole "
            "image-token span. Vision sharing is therefore a scheduling race, not a cache lookup.",
            "vllm/v1/engine/mm_input_cache.py, sized by VLLM_MM_INPUT_CACHE_GIB (default 4), caches "
            "PREPROCESSED INPUTS (pixel tensors) for transfer between the frontend and the engine "
            "core. It does not hold encoder outputs and cannot stop the ViT from running.",
        ],
        control_reproduces=dict(
            arm="A_batch16_default",
            vision_sharing_ratio_N8=ctrl.get("N8", {}).get("vision_sharing_ratio", {}).get("mean"),
            original_measurement=4.739,
            _read="the original 4.739 came from vrestruct_prefill.py's count|default cells; this "
                  "round pins the IDENTICAL item slices so every arm is paired against it."),
        cheapest_arm=best,
    )
    json.dump(dict(title="Q2b -- the vision tower is the last unshared term in generation",
                   date="2026-08-16", arms=arms, verdict=verdict,
                   workload_shares=SHARE,
                   workload_shares_source="artifacts/flop_ratio_derivation_2026-08-03.json "
                                          "flop_model.component_shares_pct.lingshu_7b "
                                          "(vision 24.32+0.62+0.43, lm_prefill 73.11+0.37, "
                                          "decode+head 1.04+0.01+0.11)"),
              open(os.path.join(PARTS, "vision_sharing.json"), "w"), indent=1, default=float)

    print(f"{'arm':26s} {'batch':>5s} {'maxseq':>6s} {'mmGiB':>5s} {'prime':>5s} "
          f"{'LM@8':>6s} {'VIS@8':>6s} {'implied':>8s} {'wall@8':>7s}")
    for arm, a in sorted(arms.items()):
        r8 = a.get("N8")
        if not r8 or not r8["vision_sharing_ratio"]:
            continue
        imp = a.get("implied_open_arm_flopeq_headonly", {}).get("value", float("nan"))
        print(f"{arm:26s} {a['submission_batch']:5d} {str(a['max_num_seqs']):>6s} "
              f"{str(a['mm_input_cache_gib']):>5s} {str(a['prime']):>5s} "
              f"{r8['lm_prefill_sharing_ratio']['mean']:6.3f} "
              f"{r8['vision_sharing_ratio']['mean']:6.3f} {imp:8.3f} "
              f"{r8['wall_s']['mean']:7.1f}")
    print("\nwrote", os.path.join(PARTS, "vision_sharing.json"))


if __name__ == "__main__":
    main()
