#!/usr/bin/env python3
"""
sd_test.py - feasibility + speedup of LOSSLESS speculative decoding for the 32B-think tier:
32B target + 7B draft (both Qwen2.5-VL, shared vocab), measure batch-1 think-mode latency/token vs
the measured 32B-think baseline (0.0716 s/tok). Lossless => same accuracy as 32B-think, only faster;
structurally immune to the recoverability wall. TP=2 (both GPUs). Run from repo root with
HF_HOME=/data/dan/hf_cache (weights are local).
"""
import sys, os, re, json, random, time
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
M32 = "/data/dan/weights/MedVLThinker-32B-RL_m23k"; M7 = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts within "
             "<think> </think> tags before providing the answer.")
MAXPX = 1280 * 28 * 28 // 4   # cap320-ish
N_ITEMS = 20; BASE_S_PER_TOK = 0.0716

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
sel = [i for i, n in enumerate(data["dataset_name"]) if "pmc" in n.lower() or "slake" in n.lower()]
random.Random(0).shuffle(sel); sel = sel[:N_ITEMS]
proc = AutoProcessor.from_pretrained(M32)
def build(ex):
    opts = ex["options"]; opts = json.loads(opts) if isinstance(opts, str) else opts
    q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    im = [{"type": "image", "image": x, "max_pixels": MAXPX} for x in (ex.get("images") or [])]
    msgs = [{"role": "system", "content": SYS_THINK}, {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req
reqs = [build(data[i]) for i in sel]

print("loading 32B target + 7B draft (speculative, TP=2)...", flush=True)
try:
    llm = LLM(model=M32, tensor_parallel_size=2, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=4096, limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
              speculative_config={"model": M7, "num_speculative_tokens": 5})
    mode = "SPECULATIVE (32B+7B draft)"
except Exception as e:
    print(f"!! speculative_config failed: {str(e)[:300]}\n   falling back to 32B-only baseline timing", flush=True)
    llm = LLM(model=M32, tensor_parallel_size=2, dtype="bfloat16", gpu_memory_utilization=0.92,
              max_model_len=4096, limit_mm_per_prompt={"image": 4}, trust_remote_code=True)
    mode = "32B-ONLY (no SD)"
sp = SamplingParams(temperature=0.0, max_tokens=512)
tot_t = tot_tok = 0
print(f"mode={mode}; timing {len(reqs)} items batch-1...", flush=True)
for k, r in enumerate(reqs):
    t0 = time.time(); o = llm.generate([r], sp, use_tqdm=False)[0]; dt = time.time() - t0
    g = len(o.outputs[0].token_ids); tot_t += dt; tot_tok += g
    if k < 3 or k % 5 == 0: print(f"  item {k}: {dt:.2f}s, {g} tok, {dt/max(g,1)*1000:.1f} ms/tok", flush=True)
spt = tot_t / max(tot_tok, 1)
print(f"\n=== {mode} ===")
print(f"  total {tot_tok} tok in {tot_t:.1f}s  ->  {spt*1000:.1f} ms/tok ({spt:.4f} s/tok)")
print(f"  baseline 32B-think = {BASE_S_PER_TOK:.4f} s/tok  ->  SPEEDUP = {BASE_S_PER_TOK/spt:.2f}x")
print(f"  GO if speedup > 1.3x (lossless => 32B-think accuracy at {BASE_S_PER_TOK/spt:.2f}x less latency)")
