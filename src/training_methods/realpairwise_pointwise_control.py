#!/usr/bin/env python3
"""realpairwise_pointwise_control.py -- ENGINE-MATCHED pointwise control for the clean
real-pairwise run.

WHY THIS EXISTS. The incumbent bar (sel_eff 0.775204) was scored with HuggingFace +
PeftModel, which applies the LoRA to EVERY target module including the vision tower. The
pairwise arm runs under vLLM 0.9.0.1, which prints

    "Regarding multimodal models, vLLM currently only supports adding LoRA to language
     model, visual.* will be ignored"

so the pairwise arm effectively carries a LANGUAGE-MODEL-ONLY adapter. Comparing the vLLM
pairwise arm against the HF pointwise arm therefore mixes the prompt frame with the engine's
LoRA coverage. This script scores the SAME clean adapter POINTWISE under the SAME vLLM
engine, same max_pixels, same verbatim pointwise prompt -- so the pairwise-vs-pointwise
delta can also be read within one engine.

One forward pass per DISTINCT candidate (8943 over the 2345 questions).

  PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    /data/dan/medeval_venv/bin/python src/training_methods/realpairwise_pointwise_control.py \
      --dataset pathvqa_open
"""
import argparse, os, sys, json, math, time

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.training_methods.realpairwise_clean_gpu import (  # noqa: E402
    POINT_SYS, HIGH_PX, MIN_PX, imgs_for, distinct_cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=G.EVAL_DS)
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--tag", default="disjoint")
    ap.add_argument("--out_dir", default="ckpts/pairwise_clean")
    ap.add_argument("--max_pixels", type=int, default=HIGH_PX)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    A = ap.parse_args()
    if os.environ.get("PAIRWISE_GPU_OK") != "1":
        sys.exit("[REFUSED] set PAIRWISE_GPU_OK=1")

    items = [it for it in G.load_items() if it["ds"] == A.dataset]
    img_map = imgs_for(A.dataset)
    os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
    outp = os.path.join(ROOT, A.out_dir, f"pointwise_{A.dataset}_{A.tag}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                r = json.loads(l)
                done.add((str(r["idx"]), r["na"]))
    print(f"[pointwise-control] {A.dataset}: {len(items)} items, {len(done)} rows resumed", flush=True)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]

    def prep_img(img):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img,
                                              "max_pixels": A.max_pixels, "min_pixels": MIN_PX}]}]
        return process_vision_info(msgs)[0][0]

    def build(q, pil, ans):
        msgs = [{"role": "system", "content": POINT_SYS},
                {"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text":
                                              f"Question: {q}\nProposed answer: {ans}\n"
                                              f"Is the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": text, "multi_modal_data": {"image": pil}}

    llm = LLM(model=A.model_path, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32)
    lora_req = LoRARequest("verifier", 1, os.path.join(ROOT, A.adapter))
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    fh = open(outp, "a"); reqs, meta = [], []; t0 = time.time(); n = 0

    def flush():
        nonlocal reqs, meta, n
        if not reqs:
            return
        try:
            outs = llm.generate(reqs, sp, lora_request=lora_req)
        except Exception as e:
            print(f"  !! chunk failed: {e}", flush=True)
            reqs, meta = [], []
            return
        for o, m in zip(outs, meta):
            lps = (o.outputs[0].logprobs or [{}])[0]
            py = math.exp(lps[YES].logprob) if YES in lps else 0.0
            pn = math.exp(lps[NO].logprob) if NO in lps else 0.0
            fh.write(json.dumps({**m, "pyes": round(py / (py + pn), 6) if (py + pn) > 0 else 0.5}) + "\n")
            n += 1
        fh.flush(); reqs, meta = [], []

    for it in items:
        na, slots, text = distinct_cands(it["preds"])
        need = [a for a in na if (str(it["idx"]), a) not in done]
        if not need:
            continue
        pil = prep_img(img_map[it["idx"]][1])
        q = img_map[it["idx"]][0]
        for a in need:
            reqs.append(build(q, pil, text[a]))
            meta.append({"ds": A.dataset, "idx": str(it["idx"]), "na": a})
            if len(reqs) >= A.chunk:
                flush()
                print(f"   [{n}] {(time.time()-t0)/60:.1f} min", flush=True)
    flush(); fh.close()
    print(f"[pointwise-control] DONE {A.dataset}: {n} rows -> {outp}", flush=True)


if __name__ == "__main__":
    main()
