#!/usr/bin/env python3
"""
bench_int4_strong_leg.py -- REAL batch-1 vLLM latency of an AWQ-INT4 vs FP16 32B (Qwen2.5-VL-32B
architecture, the SAME backbone as Lingshu-32B) to empirically anchor the INT4 strong-leg projection
for the integrated cascade (G3).

Why this is a fair anchor (NOT a quantization rabbit hole): Lingshu-32B is a medical fine-tune of
Qwen2.5-VL-32B-Instruct. Batch-1 prefill/decode *latency* is a function of the ARCHITECTURE + kernel
(weight-value-independent), so the official Qwen/Qwen2.5-VL-32B-Instruct-AWQ gives the true INT4
speedup on this exact backbone WITHOUT us quantizing Lingshu ourselves (no calib, no lib install).
(Medical ACCURACY of INT4 is NOT measured here -- it is projected from literature; see the JSON.)

Measures two regimes at batch-1 on a real medical image + MCQ prompt (cap320):
  * max_tokens=4    -> PREFILL-BOUND (the method's 32B-NO-THINK strong leg emits ~1-2 tokens)
  * max_tokens=256  -> DECODE-HEAVY  (the 32B-THINK baseline emits ~318 tokens)
Two-point decomposition:  decode_ms_per_tok = (lat_256 - lat_4)/(256-4);  prefill_ms = lat_4 - 4*decode_ms.

Run (pin ONE gpu; INT4 ~20GB / FP16 ~63GB both fit tp=1 on 1x80GB):
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/bench_int4_strong_leg.py --tag int4 \
      --model /data/dan/hf_cache/hub/models--Qwen--Qwen2.5-VL-32B-Instruct-AWQ/snapshots/<snap> --quant awq_marlin
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/bench_int4_strong_leg.py --tag fp16 \
      --model /data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/<snap>
"""
import argparse, json, os, time, statistics as st
from PIL import Image
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True, help="int4 | fp16")
ap.add_argument("--quant", default="", help="vLLM quantization arg, e.g. awq_marlin; empty for fp16")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--max_pixels", type=int, default=1280 * 28 * 28 // 4)  # cap320
ap.add_argument("--out", default="results/cascade_methods/artifacts/int4_bench_raw.jsonl")
A = ap.parse_args()

SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."

# ---- build ONE real medical image + MCQ prompt (cap320) --------------------------------------------
ds = load_dataset("/data/dan/dataset/MedVLThinker-Eval")
data = ds["test" if "test" in ds else list(ds.keys())[0]]
ex = next(data[i] for i in range(len(data)) if "pmc" in data[i]["dataset_name"].lower()
          and (data[i].get("images")))
img = ex["images"][0].convert("RGB")

# resize so total pixels <= max_pixels (deterministic prefill token count), sides multiple of 28
import math
w, h = img.size
scale = min(1.0, math.sqrt(A.max_pixels / (w * h)))
nw = max(28, int(round(w * scale / 28)) * 28); nh = max(28, int(round(h * scale / 28)) * 28)
img = img.resize((nw, nh), Image.BICUBIC)

opts = ex["options"] if isinstance(ex["options"], dict) else json.loads(ex["options"])
qtext = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
messages = [{"role": "system", "content": SYS_NOTHINK},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qtext}]}]

# ---- load vLLM ------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
kw = dict(model=A.model, tensor_parallel_size=1, gpu_memory_utilization=A.gpu_mem,
          max_model_len=4096, limit_mm_per_prompt={"image": 1}, dtype="float16",
          enforce_eager=False, disable_log_stats=True, trust_remote_code=True)
if A.quant:
    kw["quantization"] = A.quant
print(f"[load] {A.tag}  model={A.model}  quant={A.quant or 'none'}", flush=True)
t0 = time.time()
llm = LLM(**kw)
print(f"[load] done in {time.time()-t0:.1f}s  img={img.size}", flush=True)

proc = llm.get_tokenizer()
# build the text prompt via the model's chat template, image passed as multi_modal_data
from transformers import AutoProcessor
hf_proc = AutoProcessor.from_pretrained(A.model, trust_remote_code=True)
prompt_text = hf_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
req = {"prompt": prompt_text, "multi_modal_data": {"image": img}}


def run(max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens)
    lats, ptoks, gtoks = [], [], []
    for _ in range(A.reps):
        t = time.time()
        out = llm.generate([req], sp, use_tqdm=False)
        lats.append(time.time() - t)
        o = out[0]
        ptoks.append(len(o.prompt_token_ids))
        gtoks.append(len(o.outputs[0].token_ids))
    return lats, ptoks, gtoks


# warmup
warm = SamplingParams(temperature=0.0, max_tokens=8, min_tokens=8)
for _ in range(A.warmup):
    llm.generate([req], warm, use_tqdm=False)

res = {}
for mt in (4, 256):
    lats, ptoks, gtoks = run(mt)
    res[mt] = dict(lat_ms_med=st.median(lats) * 1000, lat_ms_min=min(lats) * 1000,
                   prefill_tok=st.median(ptoks), gen_tok=st.median(gtoks),
                   lats_ms=[round(x * 1000, 1) for x in lats])
    print(f"[{A.tag}] max_tokens={mt:3d}  prefill_tok={st.median(ptoks):.0f} gen_tok={st.median(gtoks):.0f}"
          f"  lat_med={st.median(lats)*1000:8.1f}ms  lat_min={min(lats)*1000:8.1f}ms", flush=True)

lat4 = res[4]["lat_ms_med"]; lat256 = res[256]["lat_ms_med"]
decode_ms_tok = (lat256 - lat4) / (256 - 4)
prefill_ms = lat4 - 4 * decode_ms_tok
rec = dict(tag=A.tag, model=A.model, quant=A.quant or "fp16", max_pixels=A.max_pixels,
           reps=A.reps, prefill_ms=round(prefill_ms, 1), decode_ms_per_tok=round(decode_ms_tok, 3),
           regimes=res)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "a") as f:
    f.write(json.dumps(rec) + "\n")
print(f"\n[{A.tag}] DERIVED  prefill_ms={prefill_ms:.1f}  decode_ms_per_tok={decode_ms_tok:.3f}", flush=True)
print(f"wrote {A.out}", flush=True)
