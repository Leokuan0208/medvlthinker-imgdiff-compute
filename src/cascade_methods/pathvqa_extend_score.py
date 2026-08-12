#!/usr/bin/env python3
"""pathvqa_extend_score.py -- ATTACK D / PART 2.

Scores, with the FROZEN CLEAN disjoint-trained LoRA verifier, every distinct candidate answer
that any PathVQA-open EXTENSION arm can draw on, so that all arms are compared on ONE score
scale.  The candidate universe for the 1500 PathVQA-open eval items is

    sc8     ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b_sc8.jsonl   (the incumbent pool)
    greedy  ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b.jsonl       (the T=0 N=1 pass)
    sc16    ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b_sc16.jsonl  (an independent N=16 pool)

= 15,861 distinct (idx, normalized answer) pairs.

WHY RESCORE sc8 WHEN THE INCUMBENT DUMP ALREADY HAS ITS SCORES.  Mixing scores produced by two
different runs would put the sc8 slots and the sc16 slots on two different numeric scales, and
an argmax over a union of two scales is not a selection experiment.  Rescoring everything in one
process removes that.  The rescored sc8 scores are then a NULL TEST against the incumbent dump
(`transfer_dump_pathvqa_open_lingshu7b.json:scores`), reported as max abs deviation.

NUMERICS.  The scoring path is copied verbatim from src/training_methods/verifier_transfer_eval.py,
which produced the incumbent dump -- INCLUDING its torch backend flags, which is to say it sets
none, so matmul-TF32 keeps the torch default and cudnn-TF32 keeps the torch default.  The observed
values of both are recorded in the header row of the cache.  (openstrong_score.py pins BOTH to
False; that is a different pin, and pinning differently is exactly what would make the null test
fail for an uninteresting reason.)

HF TRANSFORMERS ONLY -- vLLM 0.9.0.1 silently drops all 192 visual.* LoRA modules.

Resumable: one JSON line per (idx, na) scored; re-running skips what is cached.  Per-candidate
error guard: a failure records score 0.5 and is counted, never aborts the run.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 \
  python3 src/cascade_methods/pathvqa_extend_score.py
"""
import argparse
import glob
import io
import json
import math
import os

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
ap.add_argument("--ck", default="ckpts/openvqa/cheap_lingshu7b")
ap.add_argument("--outdir", default="ckpts/openvqa/cheap_lingshu7b/verif_partD")
ap.add_argument("--only", choices=["all", "sc8greedy", "sc16"], default="all",
                help="restrict the candidate universe so two processes can split the work; the "
                     "cache is shared and keyed by (idx, normalized answer), so the union of two "
                     "restricted runs is byte-identical to one unrestricted run.")
A = ap.parse_args()
DEV = "cuda"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28           # verbatim: verifier scores at FULLRES
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
       "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")
CK = os.path.join(ROOT, A.ck)
OUT = os.path.join(ROOT, A.outdir)
os.makedirs(OUT, exist_ok=True)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def norm(s):
    return str(s).strip().lower()


# ------------------------------------------------------------------ candidate universe
inc = json.load(open(os.path.join(ROOT, A.adapter,
                                  "transfer_dump_pathvqa_open_lingshu7b.json")))
ids = [r["idx"] for r in inc]
p8 = {r["idx"]: r for r in jl(f"{CK}/ckpt_pathvqa_open_lingshu7b_sc8.jsonl")}
p16 = {r["idx"]: r for r in jl(f"{CK}/ckpt_pathvqa_open_lingshu7b_sc16.jsonl")}
gr = {r["idx"]: r for r in jl(f"{CK}/ckpt_pathvqa_open_lingshu7b.jsonl")}

need = []                                    # [(idx, na, raw_answer)] in a deterministic order
SRC_ON = {"all": (1, 1, 1), "sc8greedy": (1, 1, 0), "sc16": (0, 0, 1)}[A.only]
for i in ids:
    seen = {}
    for on, src in zip(SRC_ON, (p8.get(i, {}).get("preds", []),
                                gr.get(i, {}).get("preds", []),
                                p16.get(i, {}).get("preds", []))):
        if not on:
            continue
        for a in src:
            na = norm(a)
            if na not in seen:
                seen[na] = a
    for na in sorted(seen):
        need.append((i, na, seen[na]))
print(f"candidate universe: {len(need)} distinct (idx, answer) over {len(ids)} items", flush=True)

# ------------------------------------------------------------------ images (verbatim loader)
df = None
IMG = {}
import pandas as pd
df = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob("/data/dan/dataset/path_vqa/data/test-*.parquet"))],
               ignore_index=True)
for i, r in df.iterrows():
    q = r.get("question")
    a = r.get("answer")
    if q is None and "conversations" in r:
        conv = r["conversations"]
        q = conv[0]["value"].replace("<image>", "").strip()
        a = conv[1]["value"]
    if str(a).strip().lower() in ("yes", "no"):
        continue
    img = r["image"]
    if isinstance(img, dict) and "bytes" in img:
        IMG[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
print(f"images: {len(IMG)}", flush=True)

cache_p = os.path.join(OUT, "scorecache_pathvqa_open.jsonl")
cache = {}
for r in jl(cache_p):
    if "na" in r:
        cache[(r["idx"], r["na"])] = float(r["score"])
todo = [t for t in need if (t[0], t[1]) not in cache]
print(f"cached={len(cache)}  todo={len(todo)}", flush=True)

if todo:
    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    print("loading base + adapter (HF transformers, never vLLM) ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to(DEV)
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    flags = {"matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
             "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
             "torch": torch.__version__}
    print("BACKEND FLAGS (left at the defaults verifier_transfer_eval.py runs with):",
          json.dumps(flags), flush=True)

    def pyes(q, img, ans):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                    {"type": "text",
                     "text": f"Question: {q}\nProposed answer: {ans}\n"
                             f"Is the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids,
                   return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            py = math.exp(lg[YES].item())
            pn = math.exp(lg[NO].item())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    n_fail = 0
    with open(cache_p, "a") as cf:
        cf.write(json.dumps({"header_backend_flags": flags}) + "\n")
        for k, (i, na, raw) in enumerate(todo):
            if i not in IMG:
                continue
            q, img = IMG[i]
            try:
                v = float(pyes(q, img, raw))
            except Exception as e:                        # per-candidate error guard
                print(f"  SCORE FAIL idx={i}: {type(e).__name__}: {e}", flush=True)
                v = 0.5
                n_fail += 1
            v = round(v, 5)
            cache[(i, na)] = v
            cf.write(json.dumps({"idx": i, "na": na, "score": v}) + "\n")
            if (k + 1) % 500 == 0:
                cf.flush()
                print(f"   scored {k+1}/{len(todo)}  (fails={n_fail})", flush=True)
    print(f"DONE scoring: {len(todo)} new, {n_fail} failures", flush=True)

# ------------------------------------------------------------------ null test vs the incumbent
dev = []
for r in inc:
    for a, s in zip(r["preds"], r["scores"]):
        v = cache.get((r["idx"], norm(a)))
        if v is not None:
            dev.append(abs(v - float(s)))
print(json.dumps({"null_test_rescore_vs_incumbent_dump": {
    "n_compared": len(dev),
    "max_abs_deviation": (max(dev) if dev else None),
    "mean_abs_deviation": (float(np.mean(dev)) if dev else None)}}, indent=1), flush=True)
print("PATHVQA_EXTEND_SCORE_DONE", flush=True)
