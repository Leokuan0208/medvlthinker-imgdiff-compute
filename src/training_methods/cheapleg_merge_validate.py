#!/usr/bin/env python3
"""cheapleg_merge_validate.py -- prove the MERGED adapted checkpoint is the adapter, and that serving
it with vLLM does not silently drop the visual LoRA.

Standing rule 10 of this research loop: NEVER score a visual LoRA under vLLM, because vLLM 0.9.0.1
silently drops all 192 `visual.*` modules of a PEFT adapter (0.775204 HF vs 0.702997 vLLM on the same
verifier).  Attack B sidesteps that by MERGING the adapter into full weights and serving an ordinary
checkpoint -- but "sidesteps" has to be demonstrated, not asserted.  Three greedy arms on the SAME
items:

    A  HF transformers, base weights + PEFT adapter attached   (the reference: the adapter as trained)
    B  HF transformers, merged checkpoint                      (is the merge arithmetic faithful?)
    C  vLLM,            merged checkpoint                      (does vLLM serve it the same?)

A vs B isolates the merge; B vs C isolates the serving engine.  If the visual modules were being
dropped anywhere, A and C would disagree the way the verifier did.  Reported as exact-string
agreement and normalized-answer agreement on the open-text (cap320) frame.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/cheapleg_merge_validate.py --adapter ckpts/train/lora_cheapleg_s0 \
      --merged ckpts/train/merged_cheapleg_s0 --n 60
"""
import argparse, json, os, re, string

import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
BASE = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
        "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
SYS = ("You are an expert medical image analyst. Answer the question with a short, specific phrase. "
       "Do not explain.")
MAXPX, MINPX = (1280 * 28 * 28) // 4, 4 * 28 * 28          # run_openvqa.py cap320

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--merged", required=True)
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--gpu_mem", type=float, default=0.35)
ap.add_argument("--out", default="results/cascade_methods/artifacts/_cheapleg_merge_validate.json")
A = ap.parse_args()


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


items = []
for x in json.load(open("/data/dan/dataset/slake/test.json")):
    if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
        p = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
        if os.path.exists(p):
            items.append((x["qid"], x["question"], p))
items = items[:A.n]
print(f"{len(items)} SLAKE-open items", flush=True)

proc = AutoProcessor.from_pretrained(BASE)


def msgs_for(q, img):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "image", "image": img,
                                          "max_pixels": MAXPX, "min_pixels": MINPX},
                                         {"type": "text", "text": q}]}]


def hf_gen(model_path, adapter=None):
    m = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    if adapter:
        m = PeftModel.from_pretrained(m, os.path.join(ROOT, adapter))
    m.eval()
    outs = []
    for (_, q, p) in items:
        ms = msgs_for(q, p)
        text = proc.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(ms)
        enc = proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            o = m.generate(**enc, max_new_tokens=64, do_sample=False)
        outs.append(proc.tokenizer.decode(o[0][enc["input_ids"].shape[1]:],
                                          skip_special_tokens=True).strip())
    del m
    torch.cuda.empty_cache()
    return outs


print("[A] HF base + adapter ...", flush=True)
A_out = hf_gen(BASE, A.adapter)
print("[B] HF merged ...", flush=True)
B_out = hf_gen(os.path.join(ROOT, A.merged))

print("[C] vLLM merged ...", flush=True)
from vllm import LLM, SamplingParams
llm = LLM(model=os.path.join(ROOT, A.merged), tensor_parallel_size=1, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=4096,
          limit_mm_per_prompt={"image": 4}, trust_remote_code=True)
reqs = []
for (_, q, p) in items:
    ms = msgs_for(q, p)
    text = proc.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(ms)
    r = {"prompt": text}
    if imgs:
        r["multi_modal_data"] = {"image": imgs}
    reqs.append(r)
C_out = [o.outputs[0].text.strip() for o in llm.generate(reqs, SamplingParams(temperature=0, max_tokens=64))]


def agree(x, y):
    return dict(exact=sum(a == b for a, b in zip(x, y)) / len(x),
                normalized=sum(norm(a) == norm(b) for a, b in zip(x, y)) / len(x))


res = dict(n=len(items),
           A_hf_base_plus_adapter_vs_B_hf_merged=agree(A_out, B_out),
           B_hf_merged_vs_C_vllm_merged=agree(B_out, C_out),
           A_hf_base_plus_adapter_vs_C_vllm_merged=agree(A_out, C_out),
           adapter=A.adapter, merged=A.merged,
           examples=[dict(q=items[i][1], A=A_out[i], B=B_out[i], C=C_out[i]) for i in range(min(8, len(items)))],
           interpretation="A-vs-B tests the merge arithmetic; B-vs-C tests the serving engine. The "
                          "verifier's known vLLM-drops-visual-LoRA failure showed up as a large "
                          "accuracy gap; here it would show up as A and C disagreeing.")
json.dump(res, open(os.path.join(ROOT, A.out), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k.startswith(("A_", "B_", "n"))}, indent=1))
print(f"wrote {A.out}")
