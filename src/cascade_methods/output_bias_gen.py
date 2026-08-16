#!/usr/bin/env python3
"""output_bias_gen.py -- ATTACK 1 GPU half.  Lingshu-7B, vLLM, one resumable JSONL row per
(cell, arm, item) carrying the FULL first-token top-20 logprob distribution.

Why this run exists: the deployed MedEvalKit dump stores only `conf` and `margin` (top-5 scalars,
MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:32), NOT which option each mass belongs to, so the
option posterior needed for an output-side prior correction cannot be recovered from it.

Every knob is byte-matched to the deployed 7B path (runners/run_full_matrix_medeval.sh runjob):
    temperature 0, top_p 0.0001, repetition_penalty 1, max_tokens 2048, stop_token_ids [],
    seed 42, enforce_eager, trust_remote_code, limit_mm_per_prompt {"image": 6}, tp=1,
    NO max_pixels (CAP_MAX_PIXELS unset == full resolution).
The ONLY deliberate change is logprobs 5 -> 20.  Null test N3 measures whether that moved the
greedy argmax.

MedEvalKit is READ ONLY.  Every prompt here is rebuilt in output_bias_lib (byte-equality to the
harness's own stored prompt strings is null test N2).

  HF_HOME=/data/dan/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/output_bias_gen.py --arms id --cells PMC_VQA --shard 0 --nshard 2

Resumable per item, per-chunk error guard, engine death is FATAL (exit 17) so a supervising runner
restarts and RESUMES.  Launch from the repo root.  nohup, never tmux.
"""
import argparse
import json
import os
import re
import sys
import time

PAD_RUN = re.compile(r"(?:<\|image_pad\|>)+")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import output_bias_lib as L                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["id"])
ap.add_argument("--cells", nargs="+", default=L.GEN_CELLS)
ap.add_argument("--model", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--gpu_mem", type=float, default=0.85)
# 0 == do not pass max_model_len at all, which is exactly what MedEvalKit does (it only sets it
# from the optional MAX_MODEL_LEN env lever), so the model's own config value is used.
#
# ⚠️ DATA-INTEGRITY BUG FOUND 2026-08-17, and the reason this default is 0.  The first attempt of
# this round ran at max_model_len 8192.  Full-resolution PMC_VQA and multi-image MedXpertQA prompts
# exceed that, and vLLM raises the ValueError from inside LLM.generate's request-VALIDATION loop --
# AFTER the earlier requests of that batch have already been added to the engine.  Those orphaned
# requests are never consumed by the failed call; the NEXT generate() runs them too and returns
# them FIRST (outputs are sorted by request id), so a naive zip(chunk, outputs) pairs items with
# other items' answers.  It was silent: SLAKE_closed came back answering "B." / "E." to yes/no
# questions with MedXpert-sized prompt-token counts (233 of 418 rows carried a token count that is
# arithmetically impossible for their own 512x512 image), and PMC_VQA's response byte-agreement
# with the deployed dump fell to 0.9036.  ALL of that first pass was deleted.
# Three guards now: (1) this default, so the trigger is gone; (2) PROMPT_IDENTITY_CHECK below, which
# compares each returned RequestOutput.prompt to the string we submitted and aborts on any
# mismatch; (3) any generate() exception exits the process so the engine is never reused dirty.
ap.add_argument("--max_model_len", type=int, default=0)
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
#: cf_blank collapses every item of a cell to the SAME content-free prompt (question "N/A",
#: option bodies "N/A", gray image), so it is a GLOBAL prior and needs only a handful of items --
#: more than one only because MedXpertQA items carry a varying number of images, which changes the
#: content-free prompt's image count.  Running all 33,430 would be pure waste.
ap.add_argument("--blank_limit", type=int, default=200)
A = ap.parse_args()

os.makedirs(L.CKPT, exist_ok=True)


def shard_of(cell, arm):
    """Which items this worker owns.  Interleaved (i %% nshard) so either shard alone is a uniform
    subsample of the cell -- a half-finished run is still an unbiased sample."""
    its = ITEMS[cell]
    if arm == "cf_blank":
        its = its[:A.blank_limit]
    if A.nshard <= 1:
        return list(its)
    return [r for r in its if r["i"] % A.nshard == A.shard]


print("[items] loading ...", flush=True)
ITEMS = L.build_items([c for c in A.cells])
for c in A.cells:
    print(f"[items] {c}: {len(ITEMS[c])}", flush=True)

WORK = []
for arm in A.arms:
    for cell in A.cells:
        if cell not in L.ARMS[arm]["cells"]:
            continue
        have = L.load_gen(cell, arm)
        todo = [r for r in shard_of(cell, arm) if r["i"] not in have]
        if todo:
            WORK.append((arm, cell, len(todo)))
        else:
            print(f"[skip] {arm} {cell}: shard complete", flush=True)
if not WORK:
    print("OUTPUT_BIAS_GEN_DONE (nothing to do)", flush=True)
    sys.exit(0)
print(f"[plan] " + ", ".join(f"{a}/{c}:{n}" for a, c, n in WORK), flush=True)

import torch                                                            # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from PIL import Image                                                   # noqa: E402
from transformers import AutoProcessor                                  # noqa: E402
from qwen_vl_utils import process_vision_info                           # noqa: E402
from vllm import LLM, SamplingParams                                    # noqa: E402

proc = AutoProcessor.from_pretrained(A.model, trust_remote_code=True)
GRAY = Image.new("RGB", L.GRAY_SIZE, (128, 128, 128))


def load_images(item, kind):
    """Materialise images exactly the way the deployed loader did (`real`), or the fixed neutral
    gray content-free probe image (`gray`), one per real image so the prompt shape is unchanged."""
    out = []
    for p in item["images"]:
        if kind == "gray":
            out.append(GRAY.copy())
        elif item["img_kind"] == "path":
            out.append(p)                              # qwen_vl_utils opens the path itself
        elif item["img_kind"] == "raw":
            out.append(Image.open(p))                  # VQA_RAD: HF-decoded PIL, no .convert
        elif item["img_kind"] == "rawrgb":
            out.append(Image.open(p).convert("RGB"))   # PATH_VQA: loader converts
        else:
            raise ValueError(item["img_kind"])
    return out


def build_req(cell, item, arm):
    prompt = L.arm_prompt(cell, item, arm)
    imgs = load_images(item, L.ARMS[arm]["image"])
    imds = [{"type": "image", "image": im} for im in imgs]      # no max_pixels == full resolution
    if len(imds) == 1:
        content = [imds[0], {"type": "text", "text": prompt}]
    else:                                              # multi-image shape, verbatim from the wrapper
        content = []
        for k, d in enumerate(imds):
            content += [{"type": "text", "text": f"<image_{k+1}>: "}, d]
        content.append({"type": "text", "text": prompt})
    msgs = [{"role": "user", "content": content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    vis, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if vis:
        req["multi_modal_data"] = {"image": vis}
    return req, prompt, text


_kw = dict(model=A.model, tensor_parallel_size=1, enforce_eager=True, trust_remote_code=True,
           limit_mm_per_prompt={"image": 6}, seed=42, gpu_memory_utilization=A.gpu_mem)
if A.max_model_len:
    _kw["max_model_len"] = A.max_model_len
llm = LLM(**_kw)
SP = SamplingParams(temperature=0, top_p=0.0001, repetition_penalty=1, max_tokens=2048,
                    stop_token_ids=[], logprobs=20)

for arm in A.arms:
    for cell in A.cells:
        if cell not in L.ARMS[arm]["cells"]:
            continue
        have = L.load_gen(cell, arm)
        todo = [r for r in shard_of(cell, arm) if r["i"] not in have]
        if not todo:
            print(f"[skip] {arm} {cell}", flush=True)
            continue
        ck = L.gen_path(cell, arm, A.shard, A.nshard)
        print(f"[run ] {arm} {cell}: {len(todo)} -> {os.path.basename(ck)}", flush=True)
        t0 = time.time()
        with open(ck, "a", encoding="utf-8") as fh:
            for c0 in range(0, len(todo), A.chunk):
                ch = todo[c0:c0 + A.chunk]
                try:
                    built = [build_req(cell, it, arm) for it in ch]
                    outs = llm.generate([b[0] for b in built], SP)
                except Exception as e:
                    # NEVER continue on a dirty engine: a raise from inside generate() can leave
                    # already-added requests queued, and they come back attached to the next call.
                    # Exit so the supervising runner restarts with a fresh engine and RESUMES.
                    print(f"   !!! generate() raised at chunk {c0}: {type(e).__name__}: {e}"
                          f" -- exiting with a clean engine (resumable)", flush=True)
                    fh.flush()
                    sys.exit(17)
                # ---- PROMPT_IDENTITY_CHECK: the returned output must belong to the request we
                # submitted.  vLLM returns RequestOutput.prompt; if it is present it must be
                # byte-equal to the chat-templated string we sent for this item.
                if len(outs) != len(ch):
                    print(f"   !!! {len(outs)} outputs for {len(ch)} requests at chunk {c0}"
                          f" -- ALIGNMENT VIOLATION, exiting", flush=True)
                    fh.flush()
                    sys.exit(19)
                # vLLM returns the EXPANDED prompt (one <|image_pad|> per visual token), so
                # collapse each run back to a single pad before comparing.  What survives is the
                # full text, the number of images and their positions -- enough to prove the output
                # belongs to this item.
                bad = [k for k, (b, o) in enumerate(zip(built, outs))
                       if getattr(o, "prompt", None) is not None
                       and PAD_RUN.sub("<|image_pad|>", o.prompt) != b[2]]
                if bad:
                    print(f"   !!! PROMPT IDENTITY MISMATCH on {len(bad)} of {len(ch)} outputs at "
                          f"chunk {c0} -- ALIGNMENT VIOLATION, exiting", flush=True)
                    fh.flush()
                    sys.exit(19)
                for it, (_req, prompt, _text), o in zip(ch, built, outs):
                    try:
                        g = o.outputs[0]
                        lp = {}
                        try:
                            for _tid, obj in (g.logprobs[0] or {}).items():
                                tok = getattr(obj, "decoded_token", None)
                                if tok is None:
                                    continue
                                lp[str(tok)] = max(float(obj.logprob), lp.get(str(tok), -1e9))
                        except Exception:
                            pass
                        fh.write(json.dumps(dict(
                            cell=cell, arm=arm, i=it["i"], src=it["src"], prompt=prompt,
                            answer=it["answer"],
                            n_choices=(len(it["choices"]) if it.get("choices") else 0),
                            response=g.text, gen_toks=len(g.token_ids),
                            n_prompt_toks=len(o.prompt_token_ids or []),
                            cum_logprob=getattr(g, "cumulative_logprob", None),
                            first_logprobs=lp), ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"   !! item {it['i']} failed: {type(e).__name__}: {e}", flush=True)
                fh.flush()
                n = min(c0 + A.chunk, len(todo))
                print(f"   [{n}/{len(todo)}] {n/(time.time()-t0):.1f} it/s", flush=True)
        written = L.load_gen(cell, arm)          # hoisted: calling this inside the comprehension
        left = [r for r in shard_of(cell, arm) if r["i"] not in written]
        if left:
            print(f"   !!! INCOMPLETE {arm} {cell}: {len(left)} items missing -- exiting to resume",
                  flush=True)
            sys.exit(18)
        print(f"[done] {arm} {cell} in {(time.time()-t0)/60:.2f} min", flush=True)
print("OUTPUT_BIAS_GEN_DONE", flush=True)
