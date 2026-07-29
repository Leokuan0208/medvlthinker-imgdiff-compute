#!/usr/bin/env python3
"""
measure_batch1_latency_unified.py -- [GPU EXPERIMENT — QUEUE AFTER OmniMed; do NOT run while GPUs are busy]

Replaces the amortized-batch latency_s proxy used in unified_router.py with REAL batch-1 wall-clock latency
for every leg of the unified cascade, so the efficiency claim is stated on measured latency (the axis the user
made first-class).  Measures, batch size = 1, warmed up, median of K reps, on a held-out sample of each format:

  cheap-MCQ-letter     : 7B, options shown, single letter answer        (MCQ margin cheap leg)
  cheap-open-1         : 7B, free-text, 1 sample                        (open cheap, 1 draw)
  cheap-open-bestofN   : 7B, free-text, N samples (vLLM n=N, parallel)   (open cheap leg — the "talk more" cost)
  verifier-1          : 7B+LoRA, Yes/No grade of 1 answer              (per-candidate verifier pass)
  strong-direct       : 32B/38B, direct answer                          (strong leg, no reasoning)
  strong-think        : 32B/38B, <think> reasoning then answer          (the expensive tier we argue to AVOID)

Prints a table {leg, median_s, p90_s, gen_toks} and the DERIVED per-query latency of:
  (a) always-strong-direct, (b) always-strong-think, (c) unified router (from the offline escalation rates),
  (d) open best-of-N parallel.  This is what turns the offline FLOPs numbers into a measured latency Pareto.

Run TWICE (cheap model on GPU0, strong model on GPU1) or sequentially; pass --role {cheap,strong}.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/measure_batch1_latency_unified.py \
     --role cheap  --model_path lingshu-medical-mllm/Lingshu-7B --verifier_lora ckpts/train/lora_verifier_pooled4
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/cascade_methods/measure_batch1_latency_unified.py \
     --role strong --model_path lingshu-medical-mllm/Lingshu-32B
Writes results/cascade_methods/artifacts/latency_unified_<role>.jsonl.
"""
import argparse, json, os, time, glob, io
import numpy as np
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SYS_GEN = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
SYS_MCQ = "You are an expert medical image analyst. Answer the multiple-choice question."
SYS_VERIFY = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
              "proposed answer is correct. Respond with only 'Yes' or 'No'.")
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts within <think> </think> tags "
             "before providing the answer. After </think>, give only the short final answer.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; MAXPX = HIGH_PX // 4    # cap320

ap = argparse.ArgumentParser()
ap.add_argument("--role", required=True, choices=["cheap", "strong"])
ap.add_argument("--model_path", required=True)
ap.add_argument("--verifier_lora", default=None)
ap.add_argument("--reps", type=int, default=30); ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--bestof", type=int, default=8); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88)
A = ap.parse_args()

# a few real medical VQA prompts+images (SLAKE open) for a representative image-prefill cost
def sample_items(k=40):
    d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"; out = []
    for x in d:
        if x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if os.path.exists(ip): out.append((x["question"], Image.open(ip).convert("RGB")))
        if len(out) >= k: break
    return out
ITEMS = sample_items()
proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

def req(sys, body, img):
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                         {"type": "text", "text": body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); r = {"prompt": text}
    if imgs: r["multi_modal_data"] = {"image": imgs}
    return r

lora = None
if A.verifier_lora:
    from vllm.lora.request import LoRARequest
    lora = LoRARequest("verifier", 1, os.path.join(REPO, A.verifier_lora))
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=8192, limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
          enable_lora=bool(lora), max_lora_rank=32)

def time_leg(name, build_req, sp, lora_req=None):
    lat = []; gt = []
    for i in range(A.warmup + A.reps):
        q, img = ITEMS[i % len(ITEMS)]
        r = build_req(q, img)
        t = time.time()
        o = llm.generate([r], sp, lora_request=lora_req) if lora_req else llm.generate([r], sp)
        dt = time.time() - t
        if i >= A.warmup:
            lat.append(dt); gt.append(len(o[0].outputs[0].token_ids))
    return dict(leg=name, median_s=round(float(np.median(lat)), 4), p90_s=round(float(np.percentile(lat, 90)), 4),
                mean_gen=round(float(np.mean(gt)), 1))

rows = []
if A.role == "cheap":
    rows.append(time_leg("cheap-MCQ-letter", lambda q, im: req(SYS_MCQ, f"Question: {q}\nAnswer with the option's letter.", im),
                         SamplingParams(temperature=0.0, max_tokens=8, logprobs=5)))
    rows.append(time_leg("cheap-open-1", lambda q, im: req(SYS_GEN, f"Question: {q}", im),
                         SamplingParams(temperature=0.7, max_tokens=64, n=1)))
    rows.append(time_leg(f"cheap-open-bestof{A.bestof}", lambda q, im: req(SYS_GEN, f"Question: {q}", im),
                         SamplingParams(temperature=0.7, max_tokens=64, n=A.bestof)))
    if lora:
        rows.append(time_leg("verifier-1", lambda q, im: req(SYS_VERIFY, f"Question: {q}\nProposed answer: X\nIs the proposed answer correct? Answer Yes or No.", im),
                             SamplingParams(temperature=0.0, max_tokens=2, logprobs=20), lora_req=lora))
else:
    rows.append(time_leg("strong-direct", lambda q, im: req(SYS_GEN, f"Question: {q}", im),
                         SamplingParams(temperature=0.0, max_tokens=64)))
    rows.append(time_leg("strong-think", lambda q, im: req(SYS_THINK, f"Question: {q}", im),
                         SamplingParams(temperature=0.0, max_tokens=1024)))

out = os.path.join(REPO, f"results/cascade_methods/artifacts/latency_unified_{A.role}.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    for r in rows: fh.write(json.dumps(r) + "\n"); print(r, flush=True)
print(f"-> {out}", flush=True)
