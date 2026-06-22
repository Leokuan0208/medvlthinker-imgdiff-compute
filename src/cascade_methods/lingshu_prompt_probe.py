#!/usr/bin/env python3
"""Probe whether Lingshu genuinely answers DIRECTLY (no reasoning) for MCQ med-VQA, or whether our prompt
format suppressed its reasoning. Compares 3 prompt variants on a few samples; prints gen length + raw output.
If ALL variants give ~3-token direct answers -> Lingshu natively doesn't reason here (data is right).
Run: CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/lingshu_prompt_probe.py"""
import re, json, random
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
M = "lingshu-medical-mllm/Lingshu-7B"; ROOT = "/data/dan/dataset/MedVLThinker-Eval"
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
proc = AutoProcessor.from_pretrained(M)
data = load_dataset(ROOT)["test"]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
idx = [i for i, n in enumerate(data["dataset_name"]) if "rad" in n.lower() or "slake" in n.lower() or "mmmu" in n.lower()]
random.Random(0).shuffle(idx); idx = idx[:6]
BOX = 'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
def variant(ex, kind):
    opts = parse_opts(ex["options"]); ol = "\n".join(f"{k}) {v}" for k, v in opts.items())
    if kind == "mine":      txt = f'{ex["question"]}\n{ol}\n{BOX}'
    elif kind == "medeval": txt = f'Question: {ex["question"]}\nOptions:\n{ol}\n{BOX}'
    elif kind == "explicit":txt = f'{ex["question"]}\n{ol}\nFirst reason step by step, then {BOX[0].lower()}{BOX[1:]}'
    im = [{"type": "image", "image": x, "max_pixels": HIGH_PX, "min_pixels": MIN_PX} for x in (ex.get("images") or [])]
    return [{"role": "user", "content": im + [{"type": "text", "text": txt}]}]
llm = LLM(model=M, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.6,
          max_model_len=8192, limit_mm_per_prompt={"image": 4}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=1024)
for kind in ["mine", "medeval", "explicit"]:
    print(f"\n===== variant: {kind} =====", flush=True)
    for i in idx:
        ex = data[i]; msgs = variant(ex, kind)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs); req = {"prompt": text}
        if imgs: req["multi_modal_data"] = {"image": imgs}
        o = llm.generate([req], sp, use_tqdm=False)[0]
        g = len(o.outputs[0].token_ids); raw = o.outputs[0].text
        print(f"  gen={g:4d}  raw={raw[:120]!r}", flush=True)
