#!/usr/bin/env python3
"""vrestruct_vision_probe.py -- WHY does the vision tower still run for ~60% of the children?

Six configurations all give the SAME vision_sharing_ratio at N=8 (4.75-4.78): submission batch
16/4/1, max_num_seqs 256/8/1 (verified effective -- 10x slower), VLLM_MM_INPUT_CACHE_GIB 4->32,
and prime-then-fan-out (which makes it WORSE, 5.28-5.32).  So it is not a scheduling race and not
a cache-size problem.  Meanwhile the LM prefill IS shared (1.14x), which means the children DO hit
the KV prefix cache.  Those two facts are in tension, and the scheduler is the only place that can
resolve them.

This script monkeypatches vllm.v1.core.sched.scheduler.Scheduler._try_schedule_encoder_inputs to
record, for every call, the exact quantities its skip conditions test:
    num_computed_tokens, num_new_tokens, start_pos, num_encoder_tokens,
    and which branch fired (covered_by_kv / has_cache / scheduled / budget-limited)
then aggregates them by request.  No inference behaviour is changed -- the patch calls through.

    HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      /data/dan/medeval_venv/bin/python src/cascade_methods/vrestruct_vision_probe.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_vrestruct_parts")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import vrestruct_prefill as PF        # noqa: E402

EVENTS = []


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=8)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--reserve_mib", type=int, default=22000)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler

    orig = Scheduler._try_schedule_encoder_inputs

    def patched(self, request, num_computed_tokens, num_new_tokens, encoder_budget):
        pre = []
        for i, pos in enumerate(request.mm_positions):
            sp_, ln = pos.offset, pos.length
            pre.append(dict(
                i=i, start_pos=int(sp_), n_enc=int(ln),
                num_computed_tokens=int(num_computed_tokens),
                num_new_tokens=int(num_new_tokens),
                covered_by_kv=bool(sp_ + ln <= num_computed_tokens),
                not_needed_yet=bool(sp_ >= num_computed_tokens + num_new_tokens),
                has_encoder_cache=bool(self.encoder_cache_manager.has_cache(request, i)),
                can_allocate=bool(self.encoder_cache_manager.can_allocate(request, i)),
                encoder_budget_in=int(encoder_budget)))
        out = orig(self, request, num_computed_tokens, num_new_tokens, encoder_budget)
        scheduled_ids = set(out[0])
        for p in pre:
            p["SCHEDULED_ENCODER"] = p["i"] in scheduled_ids
            p["req_id"] = str(request.request_id)
            p["num_tokens_total"] = int(request.num_tokens)
            EVENTS.append(p)
        return out

    Scheduler._try_schedule_encoder_inputs = patched

    proc = AutoProcessor.from_pretrained(PF.MODEL, trust_remote_code=True)
    ORIG_PER_CELL, ORIG_REPS = 16, 3
    orig_cells = [(ph, apc, N, rep) for ph in ("count", "time")
                  for apc in ("default", "on", "off") for N in (1, 2, 4, 8)
                  for rep in range(1, ORIG_REPS + 1)]
    pool = PF.load_items(ORIG_PER_CELL * len(orig_cells) + 8)
    reqs_all = PF.build_reqs(pool, proc)
    body = reqs_all[8:]
    idx = orig_cells.index(("count", "default", 8, 1))
    sl = body[idx * ORIG_PER_CELL:(idx + 1) * ORIG_PER_CELL][:A.n_items]

    llm = LLM(model=PF.MODEL, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=PF._gpu_mem_util(A.reserve_mib), max_model_len=8192,
              limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
              enable_prefix_caching=True, enforce_eager=True)
    path, model = PF._get_model(llm)
    cnt = PF.Counter()
    cnt.attach(model)
    llm.generate(reqs_all[:4], SamplingParams(temperature=0.7, max_tokens=4, n=1))
    EVENTS.clear()
    PF._reset_prefix_cache(llm)
    cnt.reset()
    outs = llm.generate(sl, SamplingParams(temperature=0.7, max_tokens=64, n=A.N,
                                           seed=20260817))

    # ---- aggregate ------------------------------------------------------------------------
    per_req = collections.defaultdict(list)
    for e in EVENTS:
        per_req[e["req_id"]].append(e)
    n_sched_reqs = sum(1 for r, evs in per_req.items() if any(e["SCHEDULED_ENCODER"] for e in evs))
    first = {}
    for r, evs in per_req.items():
        first[r] = evs[0]
    reasons = collections.Counter()
    for r, evs in per_req.items():
        sched = any(e["SCHEDULED_ENCODER"] for e in evs)
        f = evs[0]
        if sched:
            reasons["ENCODED"] += 1
        elif f["covered_by_kv"]:
            reasons["skipped_covered_by_kv_prefix"] += 1
        elif f["has_encoder_cache"]:
            reasons["skipped_encoder_cache_hit"] += 1
        else:
            reasons["skipped_other"] += 1
    ncomp = [e["num_computed_tokens"] for e in EVENTS]
    span_end = [e["start_pos"] + e["n_enc"] for e in EVENTS]
    covered = [e["covered_by_kv"] for e in EVENTS]

    rep = dict(
        title="scheduler probe: which branch decides whether a child re-runs the vision tower",
        n_items=len(sl), N=A.N, n_requests_seen=len(per_req),
        n_requests_that_encoded=n_sched_reqs,
        vision_encodes_per_question=n_sched_reqs / len(sl),
        measured_vit_patches=cnt.vit_patches,
        reasons_by_request=dict(reasons),
        first_call_stats=dict(
            num_computed_tokens_min=min(ncomp), num_computed_tokens_max=max(ncomp),
            num_computed_tokens_mean=sum(ncomp) / len(ncomp),
            image_span_end_min=min(span_end), image_span_end_max=max(span_end),
            image_span_end_mean=sum(span_end) / len(span_end),
            frac_calls_covered_by_kv=sum(covered) / len(covered),
            n_calls=len(EVENTS)),
        example_events=EVENTS[:24],
        _read="`covered_by_kv` is the ONLY branch that lets a child skip the vision tower without "
              "an encoder-cache hit, and the encoder cache is keyed by request id so a sibling can "
              "never provide one. If frac_calls_covered_by_kv is low while the LM prefill is "
              "shared, the prefix hit is being computed AFTER this decision, not before it.")
    json.dump(rep, open(os.path.join(OUT, "vision_probe.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "example_events"},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
