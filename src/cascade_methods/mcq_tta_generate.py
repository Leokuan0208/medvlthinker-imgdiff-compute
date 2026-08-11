#!/usr/bin/env python3
"""
mcq_tta_generate.py -- ATTACK 2 GPU half.  Runs Lingshu-32B (vLLM, tp=2) over K test-time-augmented
views of every item on the five MCQ/closed reporting cells and writes ONE resumable JSONL row per
(cell, item, view), each carrying THE FULL PROMPT STRING.

Every LLM/SamplingParams knob is byte-matched to the deployed baseline path
(MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py, read-only, never modified):
    temperature 0, top_p 0.0001, repetition_penalty 1, max_tokens 2048, stop_token_ids [],
    enforce_eager=True, trust_remote_code=True, limit_mm_per_prompt={"image":6}, tp=2, seed 42.
The ONLY deliberate change is logprobs 5 -> 20, which is required to read the per-option first-token
posterior.  Whether that changed the greedy argmax is exactly what null test N3 measures.

    python3 src/cascade_methods/mcq_tta_generate.py --stage A --cells PMC_VQA,...
    python3 src/cascade_methods/mcq_tta_generate.py --stage B --cells ...     # temperature control

Resumable: on restart the already-written (item, view) keys are skipped.  Per-batch error guard.
Launched from the repo root (relative checkpoint paths).  nohup, never tmux.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mcq_tta as M  # noqa: E402

MODEL = os.environ.get("LINGSHU32B", "lingshu-medical-mllm/Lingshu-32B")
CKPT = M.CKPT


def load_image(item, cap):
    """Materialise the image reference the way the deployed loader did, then wrap it the way
    Qwen2_5_VL_vllm.process_messages does (max_pixels only when a cap is requested)."""
    from PIL import Image
    out = []
    for p in item["images"]:
        if item["img_kind"] == "path":
            img = p                                  # qwen_vl_utils opens the path itself
        elif item["img_kind"] == "raw":
            img = Image.open(p)                      # VQA_RAD: HF-decoded PIL, no .convert
        elif item["img_kind"] == "rawrgb":
            img = Image.open(p).convert("RGB")       # PATH_VQA: loader converts
        else:
            raise ValueError(item["img_kind"])
        d = {"type": "image", "image": img}
        if cap:
            d["max_pixels"] = int(cap)
            d["min_pixels"] = M.MIN_PX
        out.append(d)
    return out


def build_llm_input(processor, item, prompt, cap):
    from qwen_vl_utils import process_vision_info
    imds = load_image(item, cap)
    if len(imds) == 1:
        content = [imds[0], {"type": "text", "text": prompt}]
    else:                                            # MedXpertQA multi-image branch, verbatim shape
        content = []
        for i, d in enumerate(imds):
            content.append({"type": "text", "text": f"<image_{i+1}>: "})
            content.append(d)
        content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    li = {"prompt": text}
    if image_inputs is not None:
        li["multi_modal_data"] = {"image": image_inputs}
    return li


def done_keys(path):
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    seen.add((r["i"], r["v"]))
                except Exception:
                    pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B"])
    ap.add_argument("--cells", default=",".join(M.CELLS))
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.7)     # stage B only
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.0)
    ap.add_argument("--max_model_len", type=int, default=0)
    args = ap.parse_args()

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    os.makedirs(CKPT, exist_ok=True)
    items = M.build_items()
    sub = set(M.pmc_subsample_ids())

    # ---- work list, built BEFORE the model loads so a bad plan fails cheap --------------------
    work = {}
    for cell in args.cells.split(","):
        its = items[cell]
        if cell == "PMC_VQA":
            its = [r for r in its if r["i"] in sub]
        # Item-major, but the ITEM ORDER is shuffled with the pre-registered seed so that ANY
        # PREFIX of the work list is a uniformly random subsample of the cell.  The GPU slot on this
        # box is shared and a run can be cut short; without this, a partial run would be the
        # low-index prefix of test_2.csv (which is grouped by source article) and would not be a
        # random sample.  All K views of an item stay adjacent, so a prefix is always complete items.
        import numpy as _np
        order = _np.random.default_rng(M.SEED_SUBSAMPLE).permutation(len(its))
        rows = []
        for j in order:
            r = its[int(j)]
            nv = 1 if (r["malformed"] and cell in M.PERMUTABLE) else M.K
            for v in range(nv):
                rows.append((r, v))
        work[cell] = (its, rows)
        print(f"[plan] {cell}: {len(its)} items -> {len(rows)} forwards", flush=True)

    kw = dict(model=MODEL, tensor_parallel_size=args.tp, enforce_eager=True,
              trust_remote_code=True, limit_mm_per_prompt={"image": 6}, seed=42)
    if args.gpu_mem:
        kw["gpu_memory_utilization"] = args.gpu_mem
    if args.max_model_len:
        kw["max_model_len"] = args.max_model_len
    llm = LLM(**kw)
    processor = AutoProcessor.from_pretrained(MODEL)
    if args.stage == "A":
        sp = SamplingParams(temperature=0, top_p=0.0001, repetition_penalty=1, max_tokens=2048,
                            stop_token_ids=[], logprobs=20)
    else:
        sp = SamplingParams(temperature=args.temp, top_p=1.0, repetition_penalty=1, max_tokens=2048,
                            stop_token_ids=[], logprobs=20, seed=42)

    for cell in args.cells.split(","):
        its, rows = work[cell]
        out = os.path.join(CKPT, f"{cell}_stage{args.stage}.jsonl")
        seen = done_keys(out)
        todo = [(r, v) for (r, v) in rows if (r["i"], v) not in seen]
        print(f"[{cell}] {len(rows)} total, {len(seen)} done, {len(todo)} to run", flush=True)
        f = open(out, "a")
        for b0 in range(0, len(todo), args.batch):
            chunk = todo[b0:b0 + args.batch]
            try:
                lis, meta = [], []
                for (r, v) in chunk:
                    view = M.VIEWS[cell][v]
                    if args.stage == "B":                       # temperature control: identity view
                        view = M.VIEWS[cell][0]
                    prompt, oos = M.view_prompt(r, view)
                    cap = M.resolve_cap(r, view)
                    lis.append(build_llm_input(processor, r, prompt, cap))
                    meta.append((r, v, prompt, oos, view, cap))
                t0 = time.time()
                outs = llm.generate(lis, sampling_params=sp)
                per = (time.time() - t0) / max(1, len(outs))
                for (r, v, prompt, oos, view, cap), o in zip(meta, outs):
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
                    f.write(json.dumps(dict(
                        cell=cell, i=r["i"], src=r["src"], v=v, view=view, resolved_max_pixels=cap,
                        prompt=prompt, orig_of_slot=oos, answer=r["answer"],
                        n_choices=(len(r["choices"]) if r.get("choices") else 0),
                        response=g.text, gen_toks=len(g.token_ids),
                        cum_logprob=getattr(g, "cumulative_logprob", None),
                        first_logprobs=lp, latency_s=per,
                        malformed=r["malformed"]), ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{cell}] {b0 + len(chunk)}/{len(todo)}  {per*1000:.0f} ms/sample", flush=True)
            except Exception as e:                              # per-batch error guard
                print(f"[{cell}] BATCH ERROR at {b0}: {type(e).__name__}: {e}", flush=True)
        f.close()
    print("MCQ_TTA_GENERATE_DONE", flush=True)


if __name__ == "__main__":
    main()
