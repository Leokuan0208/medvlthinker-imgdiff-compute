#!/usr/bin/env python3
"""vrestruct_vision_sharing.py -- WHY is the vision tower only half-shared, and can it be fixed?

vrestruct_prefill.py measured, at N=8 with automatic prefix caching on (vLLM 0.9.0.1 V1's default,
and therefore what every generation in this project got):
    LM prefill sharing  1.16x   (essentially ONE prefill for eight samples)
    VISION sharing      4.74x   (about five image encodes for eight samples)
which is why 8 samples cost 2.28 FLOP-eq and not the ~1.08 a fully-shared convention predicts.

THE MECHANISM, read out of the installed source (not assumed):

  * vllm/v1/worker/gpu_model_runner.py:147
        self.encoder_cache: dict[str, dict[int, torch.Tensor]]     # req_id -> input_id -> output
    The ENCODER-OUTPUT cache is keyed by REQUEST ID.  SamplingParams(n=N) is split into N CHILD
    requests with N distinct request ids (vllm/v1/engine/parallel_sampling.py, ParentRequest), so
    two children of the same question can NEVER share an encoder-cache entry.

  * vllm/v1/core/sched/scheduler.py:_try_schedule_encoder_inputs
        if start_pos + num_encoder_tokens <= num_computed_tokens:
            # The encoder input is already computed and stored in the decoder's KV cache.
            continue
    So a child skips the vision tower by EXACTLY ONE ROUTE: its prefix-cache hit must already cover
    the whole image-token span.  That makes vision sharing a SCHEDULING RACE -- a child that is
    scheduled before a sibling's prefill has landed finds nothing in the prefix cache and re-encodes.

  * VLLM_MM_INPUT_CACHE_GIB (default 4) sizes vllm/v1/engine/mm_input_cache.py, which caches
    PREPROCESSED INPUTS (pixel tensors) to avoid re-preprocessing and re-sending them.  It does not
    cache encoder OUTPUTS and cannot stop the ViT from running.

PREDICTIONS THIS SCRIPT TESTS
  P1  Submitting fewer questions concurrently -> the first child's prefill lands before its siblings
      are scheduled -> vision sharing falls toward 1.0.       (fix = scheduling)
  P2  Raising VLLM_MM_INPUT_CACHE_GIB changes NOTHING.        (it is the wrong cache)
  P3  max_num_seqs is the deployable knob that implements P1 without changing the call pattern.

    HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      /data/dan/medeval_venv/bin/python src/cascade_methods/vrestruct_vision_sharing.py --arm A
"""
from __future__ import annotations

import argparse
import json
import os
import time

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_vrestruct_parts")
SINK = os.path.join(OUT, "vision_sharing.jsonl")

import sys                                                     # noqa: E402
import torch                                                   # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import vrestruct_prefill as PF                                 # noqa: E402

# arm -> (submission batch size, max_num_seqs or None for default, MM_INPUT_CACHE_GIB or None)
ARMS = {
    "A_batch16_default":      dict(batch=16, max_num_seqs=None, mm_gib=None),
    "B_batch1_default":       dict(batch=1,  max_num_seqs=None, mm_gib=None),
    "C_batch16_maxseqs8":     dict(batch=16, max_num_seqs=8,    mm_gib=None),
    "D_batch16_maxseqs1":     dict(batch=16, max_num_seqs=1,    mm_gib=None),
    "E_batch16_mmcache32":    dict(batch=16, max_num_seqs=None, mm_gib=32),
    "F_batch4_default":       dict(batch=4,  max_num_seqs=None, mm_gib=None),
    # THE CANDIDATE FIX.  Issue a 1-token n=1 request per question FIRST so its prompt blocks are
    # computed and resident, then issue the n=N request: every child now finds the image span
    # already in the prefix cache and the scheduler's
    #     if start_pos + num_encoder_tokens <= num_computed_tokens: continue
    # branch skips the vision tower for all of them.  The priming pass is REAL WORK and is counted.
    "G_prime_then_fanout_b16": dict(batch=16, max_num_seqs=None, mm_gib=None, prime=True),
    "H_prime_then_fanout_b1":  dict(batch=1,  max_num_seqs=None, mm_gib=None, prime=True),
    # THE OTHER ROUTE.  vllm/model_executor/models/qwen2_5_vl.py:972 --
    #     if image_input["type"] == "image_embeds": image_embeds = image_input["image_embeds"]
    #     else:                                     image_embeds = self.visual(pixel_values, ...)
    # so handing vLLM PRE-COMPUTED image embeddings bypasses the vision tower entirely.  We encode
    # each image ONCE with the engine's own vision tower (that call is hooked and counted, so it
    # appears in the ratio as the one honest encode) and then issue the n=N request on embeddings.
    "I_image_embeds_b16":     dict(batch=16, max_num_seqs=None, mm_gib=None, embeds=True),
    # MECHANISM CONFIRMATION (not a deployment recommendation): lengthen the text AFTER the image
    # so the image span ends below the block-aligned prefix-cache hit. If the probe's diagnosis is
    # right this alone collapses the vision ratio toward 1.0.
    "J_pad_tail_b16":         dict(batch=16, max_num_seqs=None, mm_gib=None, pad=True),
}


def _to_embed_reqs(chunk, proc, model):
    """Encode each image ONCE with the engine's own vision tower, return image_embeds requests.

    The manual visual() call goes through the same forward pre-hook as vLLM's internal one, so the
    single honest encode per image IS counted in vit_patches -- the arm is not given a free image.
    """
    import torch
    p = next(model.visual.parameters())
    dev, dt = p.device, p.dtype
    out = []
    for r in chunk:
        imgs = r.get("multi_modal_data", {}).get("image")
        o = proc(text=[r["prompt"]], images=imgs, return_tensors="pt")
        pv = o["pixel_values"].to(dev, dt)
        gt = o["image_grid_thw"].to(dev)
        with torch.no_grad():
            emb = model.visual(pv, grid_thw=gt)
        # vLLM's multimodal serialisation cannot carry bf16 ("Got unsupported ScalarType BFloat16");
        # qwen2_5_vl.py:973 casts back with .type(self.visual.dtype), so fp16 round-trips exactly.
        out.append({"prompt": r["prompt"],
                    "multi_modal_data": {"image": {
                        "image_embeds": emb.detach().to(torch.float16).cpu(),
                        "image_grid_thw": gt.detach().cpu()}}})
    return out


#: A tail long enough to push the image span's end below the block-aligned prefix-cache hit.
#: The probe measured image_span_end 325 against num_computed_tokens 320 -- the vision tower is
#: re-run for the sake of FIVE tokens, because the KV block size is 16 and the hit is
#: floor((T-1)/16)*16.  Lengthening the text AFTER the image moves the hit past the span.
PAD_TAIL = (" Consider the anatomy, the imaging modality, any visible abnormality, and the most "
            "likely clinical interpretation before answering.")


def _to_padded_reqs(chunk):
    return [{k: (v + PAD_TAIL if k == "prompt" else v) for k, v in r.items()} for r in chunk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--n_items", type=int, default=16)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--Ns", default="1,8")
    ap.add_argument("--reserve_mib", type=int, default=22000)
    A = ap.parse_args()
    cfg = ARMS[A.arm]
    if cfg["mm_gib"] is not None:
        os.environ["VLLM_MM_INPUT_CACHE_GIB"] = str(cfg["mm_gib"])

    os.makedirs(OUT, exist_ok=True)
    done = set()
    if os.path.exists(SINK):
        for l in open(SINK):
            if l.strip():
                r = json.loads(l)
                done.add((r["arm"], r["N"], r["rep"]))

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import vllm

    Ns = [int(x) for x in A.Ns.split(",")]
    proc = AutoProcessor.from_pretrained(PF.MODEL, trust_remote_code=True)

    # ---- IDENTICAL ITEM SLICES TO THE ORIGINAL MEASUREMENT --------------------------------
    # Arm A must REPRODUCE vrestruct_prefill.py's count|default numbers (LM 1.162 / vision 4.739
    # at N=8) or the comparison across arms is meaningless.  We therefore rebuild that script's
    # exact pool and cell layout and pull the exact same slices.  Every arm runs in its OWN
    # process with a fresh engine, so re-using the same slices across arms cannot contaminate
    # them and makes the arms directly paired.
    ORIG_PER_CELL, ORIG_REPS = 16, 3
    orig_cells = [(ph, apc, N, rep) for ph in ("count", "time")
                  for apc in ("default", "on", "off") for N in (1, 2, 4, 8)
                  for rep in range(1, ORIG_REPS + 1)]
    pool = PF.load_items(ORIG_PER_CELL * len(orig_cells) + 8)      # == 1160, as in the original
    reqs_all = PF.build_reqs(pool, proc)
    warm, body = reqs_all[:8], reqs_all[8:]
    orig_slice = {c: body[i * ORIG_PER_CELL:(i + 1) * ORIG_PER_CELL]
                  for i, c in enumerate(orig_cells)}
    cells, slices = [], {}
    for N in Ns:
        for rep in range(1, A.reps + 1):
            c = (A.arm, N, rep)
            cells.append(c)
            slices[c] = orig_slice[("count", "default", N, rep)]

    # exact per-cell reference geometry (patches + prompt tokens + DISTINCT images), CPU
    import hashlib
    geo = {}
    for c, sl in slices.items():
        pt = pp = 0
        seen = set()
        for r in sl:
            o = proc(text=[r["prompt"]], images=r.get("multi_modal_data", {}).get("image"),
                     return_tensors="pt")
            pt += int(o["input_ids"].shape[1])
            pp += int(o["pixel_values"].shape[0])
            seen.add(hashlib.md5(o["pixel_values"].numpy().tobytes()).hexdigest())
        geo[c] = (pt, pp, len(seen))

    kw = dict(model=PF.MODEL, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=PF._gpu_mem_util(A.reserve_mib), max_model_len=8192,
              limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
              enable_prefix_caching=True, enforce_eager=True)
    if cfg["max_num_seqs"] is not None:
        kw["max_num_seqs"] = cfg["max_num_seqs"]
    llm = LLM(**kw)
    path, model = PF._get_model(llm)
    cnt = PF.Counter()
    cnt.attach(model)
    llm.generate(warm, SamplingParams(temperature=0.7, max_tokens=8, n=2))

    for c in cells:
        if c in done:
            continue
        _, N, rep = c
        PF._reset_prefix_cache(llm)
        sp = SamplingParams(temperature=0.7, max_tokens=64, n=N, seed=20260816 + rep)
        cnt.reset()
        t0 = time.time()
        outs = []
        B = cfg["batch"]
        sl = slices[c]
        prime_sp = SamplingParams(temperature=0.0, max_tokens=1, n=1)
        prime_tok = 0
        embed_fail = None
        for i in range(0, len(sl), B):
            chunk = sl[i:i + B]
            if cfg.get("prime"):
                po = llm.generate(chunk, prime_sp)
                prime_tok += sum(len(o_.token_ids) for o in po for o_ in o.outputs)
            if cfg.get("pad"):
                chunk = _to_padded_reqs(chunk)
            if cfg.get("embeds"):
                try:
                    chunk = _to_embed_reqs(chunk, proc, model)
                except Exception as e:                      # report, never fabricate
                    embed_fail = f"{type(e).__name__}: {e}"
                    print(f"    [embeds] FAILED: {embed_fail}", flush=True)
            try:
                outs.extend(llm.generate(chunk, sp))
            except Exception as e:
                embed_fail = (embed_fail or "") + f" | generate: {type(e).__name__}: {e}"
                print(f"    [embeds] generate FAILED: {e}", flush=True)
                break
        dt = time.time() - t0
        gen_tok = sum(len(o_.token_ids) for o in outs for o_ in o.outputs) + prime_tok
        pt_ref, pp_ref, n_distinct_images = geo[c]
        rec = dict(arm=A.arm, N=N, rep=rep, n_items=len(sl), submission_batch=B,
                   max_num_seqs=cfg["max_num_seqs"], mm_input_cache_gib=cfg["mm_gib"],
                   effective_max_num_seqs=getattr(
                       getattr(getattr(llm, "llm_engine", None), "vllm_config", None),
                       "scheduler_config", None) and
                   llm.llm_engine.vllm_config.scheduler_config.max_num_seqs,
                   prime=bool(cfg.get("prime")), prime_gen_tok=prime_tok,
                   embeds=bool(cfg.get("embeds")), embed_fail=embed_fail,
                   pad=bool(cfg.get("pad")),
                   vllm=vllm.__version__, wall_s=dt,
                   n_distinct_images=n_distinct_images,
                   prompt_tok_ref=pt_ref, patches_ref=pp_ref,
                   lm_positions=cnt.lm_positions, vit_patches=cnt.vit_patches,
                   gen_tok_total=gen_tok,
                   lm_prefill_sharing_ratio=(cnt.lm_positions - gen_tok) / pt_ref,
                   vision_sharing_ratio=cnt.vit_patches / pp_ref,
                   num_cached_tokens_total=sum(getattr(o, "num_cached_tokens", 0) or 0
                                               for o in outs),
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(SINK, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  {A.arm} N={N} rep={rep}: LMprefill x{rec['lm_prefill_sharing_ratio']:.3f}  "
              f"VISION x{rec['vision_sharing_ratio']:.3f}  wall {dt:.1f}s", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
