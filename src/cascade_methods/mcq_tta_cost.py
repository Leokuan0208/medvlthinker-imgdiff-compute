#!/usr/bin/env python3
"""
mcq_tta_cost.py -- ATTACK 2 cost endpoint: MEASURED batch-1 wall-clock and NVML energy for K=1 vs
K=4 MCQ views on Lingshu-32B, so the cost claim is measured rather than modelled.

The pre-registered plan carried a PRIOR ("4 permutations ~ 1.2x one 32B pass, because the ~596
image tokens are byte-identical across permutations and the LM prefill is 91.6% of the 32B's FLOPs
-- flop_ratio_derivation_2026-08-03.json"). That prior is NOT a measurement and this script exists
to replace it with one. KILL criterion (iii) fires if measured K=4 cost exceeds 2.0x K=1.

NVML sampler policy is verbatim from the repo's canonical harness
(src/cascade_methods/open_measure_latency_energy.py, via bestofn_measure_batch8.py): 10 ms
sampling, hold-last-good on NVMLError, sum over visible GPUs.  GPU model and enforced power limit
are recorded, because energy numbers from a different card are not comparable.

    python3 src/cascade_methods/mcq_tta_cost.py --n 20 --reps 2
"""
import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import pynvml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mcq_tta as M  # noqa: E402
import mcq_tta_generate as G  # noqa: E402

OUT = os.path.join(M.ART, "mcq_tta_cost_2026-08-10.json")

pynvml.nvmlInit()
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
_toks = [x.strip() for x in _vis.split(",") if x.strip() != ""]
_idxs = [int(t) for t in _toks] if (_toks and all(t.isdigit() for t in _toks)) \
        else list(range(pynvml.nvmlDeviceGetCount()))
HS = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in _idxs]
GPU_NAMES = [(g.decode() if isinstance(g, bytes) else g)
             for g in [pynvml.nvmlDeviceGetName(h) for h in HS]]
GPU_LIMIT_W = [pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0 for h in HS]
_W_LAST = [0.0]


def watts():
    try:
        w = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in HS) / 1000.0
        _W_LAST[0] = w
        return w
    except pynvml.NVMLError:
        return _W_LAST[0]


class E:
    def __init__(s): s.go = False; s.j = 0.0
    def __enter__(s):
        s.j = 0.0; s.go = True
        def loop():
            last = time.time()
            while s.go:
                time.sleep(0.01); now = time.time(); s.j += watts() * (now - last); last = now
        s.th = threading.Thread(target=loop); s.th.start(); return s
    def __exit__(s, *a):
        s.go = False; s.th.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--cell", default="PMC_VQA")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.62)
    ap.add_argument("--max_model_len", type=int, default=16384)
    a = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    items = M.build_items()[a.cell]
    sub = sorted(set(M.pmc_subsample_ids()))[:a.n + a.warmup] if a.cell == "PMC_VQA" else None
    pick = [r for r in items if (sub is None or r["i"] in set(sub))][:a.n + a.warmup]

    llm = LLM(model=G.MODEL, tensor_parallel_size=a.tp, enforce_eager=True, trust_remote_code=True,
              limit_mm_per_prompt={"image": 6}, seed=42, gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len)
    proc = AutoProcessor.from_pretrained(G.MODEL)
    sp = SamplingParams(temperature=0, top_p=0.0001, repetition_penalty=1, max_tokens=2048,
                        stop_token_ids=[], logprobs=20)

    def run_one(item, ks):
        lis = []
        for k in ks:
            view = M.VIEWS[a.cell][k]
            p, _ = M.view_prompt(item, view)
            lis.append(G.build_llm_input(proc, item, p, M.resolve_cap(item, view)))
        t0 = time.time()
        with E() as e:
            outs = llm.generate(lis, sampling_params=sp)
        return time.time() - t0, e.j, sum(len(o.outputs[0].token_ids) for o in outs), \
            sum(len(o.prompt_token_ids) for o in outs)

    for r in pick[:a.warmup]:
        run_one(r, [0]); run_one(r, [0, 1, 2, 3])

    res = dict(title="ATTACK 2 measured batch-1 cost, K=1 vs K=4", date=M.DATE, cell=a.cell,
               n_items=a.n, reps=a.reps, provenance="measured",
               gpu=dict(names=GPU_NAMES, enforced_power_limit_w=GPU_LIMIT_W, tp=a.tp,
                        engine="vLLM 0.9.0.1, enforce_eager=True (matches the deployed baseline path); "
                               "vLLM's automatic prefix caching is at its DEFAULT for this build and "
                               "the measured ratio below already includes whatever sharing it gives"),
               arms={})
    for lab, ks in [("K1", [0]), ("K4", [0, 1, 2, 3])]:
        lat, en, gt, pt = [], [], [], []
        for rep in range(a.reps):
            for r in pick[a.warmup:]:
                L, J, G_, P = run_one(r, ks)
                lat.append(L * 1000.0); en.append(J); gt.append(G_); pt.append(P)
        res["arms"][lab] = dict(
            n_calls=len(lat),
            latency_ms_mean=float(np.mean(lat)), latency_ms_median=float(np.median(lat)),
            latency_ms_sd=float(np.std(lat, ddof=1)),
            energy_j_mean=float(np.mean(en)), energy_j_median=float(np.median(en)),
            energy_j_sd=float(np.std(en, ddof=1)),
            gen_tokens_mean=float(np.mean(gt)), prompt_tokens_mean=float(np.mean(pt)))
    k1, k4 = res["arms"]["K1"], res["arms"]["K4"]
    res["ratios_measured"] = dict(
        latency=k4["latency_ms_mean"] / k1["latency_ms_mean"],
        energy=k4["energy_j_mean"] / k1["energy_j_mean"],
        prompt_tokens=k4["prompt_tokens_mean"] / k1["prompt_tokens_mean"],
        note="a ratio below 4.0 is the measured prefix/pipeline saving; the naive as-charged model "
             "would say 4.0x. Report BOTH.")
    R32 = 4.57          # FLOP-eq of one Lingshu-32B direct pass, as charged everywhere in this repo
    R32_HONEST = 3.816  # the derived alternative (flop_ratio_derivation_2026-08-03.json)
    res["flop_eq"] = dict(
        provenance="as-charged (K x R32) is a MODEL, not a measurement; the measured columns above "
                   "are the wall-clock and NVML numbers. Both are reported, never mixed.",
        R32_as_charged=R32, R32_derived_honest=R32_HONEST,
        K1_as_charged=R32, K4_as_charged=4 * R32,
        K4_over_K1_as_charged=4.0,
        K4_over_K1_measured_energy=res["ratios_measured"]["energy"],
        K4_over_K1_measured_latency=res["ratios_measured"]["latency"],
        note="the as-charged model assumes no sharing across views and is the CONSERVATIVE (worse "
             "for us) number; the measured energy ratio is what the serving path actually costs. "
             "Quote both, each labelled.")
    res["kill_iii"] = dict(
        threshold=2.0, measured_energy_ratio=res["ratios_measured"]["energy"],
        fires=bool(res["ratios_measured"]["energy"] > 2.0),
        meaning="if K=4 costs > 2.0x a single 32B pass, the attack is uninteresting even if it wins "
                "on accuracy: 'accuracy is buyable on MCQ but only at >=2x the strong model'.")
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
