#!/usr/bin/env python3
"""predict_modality.py -- the generator names each image's modality, so temperature can depend on modality
WITHOUT the dataset label.

For the "per-modality T" item of the 2026-10-19 deck's next steps (temp_labelfree.py scores the policy). Two of
the eight benchmarks (VQA-Med, VQA-RAD) carry no modality metadata, and using metadata where it exists while
falling back to the benchmark elsewhere would leak the dataset label back in. So every image's modality comes
from the same model the method already runs: Lingshu-7B, one greedy answer to a fixed question, at the
generation cap (cap320), with run_openvqa.py's system prompt and image handling. Its cost is one extra short
greedy decode per image (the image prefill is the bulk of it), reported beside the result.

The item loader below is run_openvqa.py's own block (lines quoted verbatim at build time, see LOADER), so
idx, question and image are exactly what generation saw. Images are de-duplicated before inference.
The free-text answer is mapped to a modality class by the keyword rules in CLASSES (first match wins); the
mapping's agreement with the metadata is reported wherever metadata exists (SLAKE, RadImageNet, OmniMedVQA,
and the single-modality benchmarks PathVQA = pathology, Kvasir = endoscopy, GEMeX = chest X-ray).

  CUDA_VISIBLE_DEVICES=0 python3 src/exp_2026_10/predict_modality.py   (vLLM, ~20 min)
"""
import argparse, glob, hashlib, io, json, os, re, string, sys, time
import pandas as pd   # used by the verbatim loader block (parquet benchmarks)
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
MP = "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/"
SYS = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
QUESTION = "What is the imaging modality of this image?"
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX // 4                                 # cap320, the pools' generation cap
OUT = "/data/dan/exp_2026_10/modality_pred_lingshu7b.jsonl"
ART = ("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/"
       "artifacts/modality_pred_2026-10-22.json")
CLASSES = [("oct", r"\boct\b|optical coherence"), ("pet_nuclear", r"\bpet\b|spect|nuclear|scintigra"),
           ("ct", r"\bct\b|computed tomograph|\bcat scan"), ("mri", r"\bmri?\b|magnetic resonance|\bflair\b|\bt[12]\b"),
           ("ultrasound", r"ultrasound|sonogra|\bus\b|echocardio|doppler"),
           ("xray", r"x-?ray|radiograph|\bcxr\b|mammogra|fluoroscop"), ("angiography", r"angiogra"),
           ("pathology", r"patholog|histolog|microscop|h\s*&\s*e|hematoxylin|cytolog|stain|biopsy|slide"),
           ("endoscopy", r"endoscop|colonoscop|gastroscop|capsule|polyp"),
           ("fundus", r"fundus|retina|ophthalmoscop"), ("dermoscopy", r"dermoscop|dermatoscop|\bskin\b|dermat")]


def modality_class(text):
    t = str(text).lower()
    return next((c for c, rx in CLASSES if re.search(rx, t)), "other")


META_MAP = {"ct": "ct", "computed tomography": "ct", "ct(computed tomography)": "ct", "mri": "mri",
            "mr (mag-netic resonance imaging)": "mri", "x-ray": "xray", "x-ray": "xray", "microscopy images": "pathology",
            "fundus photography": "fundus", "dermoscopy": "dermoscopy", "oct (optical coherence tomography": "oct",
            "ultrasound": "ultrasound", "endoscopy": "endoscopy"}
SINGLE = {"pathvqa_open": "pathology", "kvasir_x1_open": "endoscopy", "gemex_open": "xray"}


def metadata_modality(cell):
    """idx -> metadata class, where the dataset records one (None otherwise)."""
    if cell in SINGLE:
        return defaultdict(lambda: SINGLE[cell])
    src = {"slake_open": ("/data/dan/dataset/slake/test.json", "qid"),
           "radimagenet_open": ("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json", "idx"),
           "omnimed_open": ("/data/dan/dataset/omnimed_opentext/omnimed_open.json", "idx")}.get(cell)
    if not src:
        return {}
    d = json.load(open(src[0]))
    return {r[src[1]]: META_MAP.get(str(r.get("modality", "")).strip().lower(), modality_class(r.get("modality", "")))
            for r in d}


def load_items(dataset):
    """run_openvqa.py's item loader, verbatim (SPLIT='test'); returns (idx, question, gold, image path or PIL)."""
    class _A: pass
    A = _A(); A.dataset = dataset
    ns = dict(globals()); ns["A"] = A                  # ONE namespace: the block rebinds `items`
    exec(compile(LOADER, "run_openvqa_loader", "exec"), ns)   # the verbatim block from run_openvqa.py
    return ns["items"]


def main():
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MP, trust_remote_code=True)
    llm = LLM(model=MP, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    done = set()
    if os.path.exists(OUT):
        done = {(json.loads(l)["cell"], json.loads(l)["idx"]) for l in open(OUT) if l.strip()}
    with open(OUT, "a") as fh:
        for cell in OPEN8:
            items = [it for it in load_items(cell) if (cell, it[0]) not in done]
            if not items:
                continue
            key = lambda im: (im if isinstance(im, str) else hashlib.md5(im.tobytes()).hexdigest())
            uniq = {}
            for it in items:
                uniq.setdefault(key(it[3]), it[3])
            ks = list(uniq); ans = {}
            for c0 in range(0, len(ks), 256):
                reqs = []
                for k in ks[c0:c0 + 256]:
                    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [
                        {"type": "image", "image": uniq[k], "max_pixels": MAXPX, "min_pixels": MIN_PX},
                        {"type": "text", "text": QUESTION}]}]
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    imgs, _ = process_vision_info(msgs)
                    reqs.append({"prompt": text, "multi_modal_data": {"image": imgs}})
                for k, o in zip(ks[c0:c0 + 256], llm.generate(reqs, sp)):
                    ans[k] = o.outputs[0].text.strip()
            for it in items:
                a = ans[key(it[3])]
                fh.write(json.dumps({"cell": cell, "idx": it[0], "pred_raw": a, "modality": modality_class(a)}) + "\n")
            fh.flush()
            print(f"[{time.time() - t0:5.0f}s] {cell}: {len(items)} questions, {len(uniq)} unique images", flush=True)
    write_agreement(time.time() - t0)


def write_agreement(seconds):
    """agreement with metadata where it exists (run alone: python3 predict_modality.py --agreement_only)"""
    rows = [json.loads(l) for l in open(OUT) if l.strip()]
    agree = {}
    for cell in OPEN8:
        meta = metadata_modality(cell); R = [r for r in rows if r["cell"] == cell]
        m = [(r["modality"], meta[r["idx"]]) for r in R if (cell in SINGLE or r["idx"] in meta)]   # [] not .get: a defaultdict's .get ignores the default
        agree[cell] = {"questions": len(R), "with_metadata": len(m),
                       "agreement": (float(np.mean([a == b for a, b in m])) if m else None),
                       "predicted_classes": dict(Counter(r["modality"] for r in R).most_common(8))}
        print(f"  {cell:17} agreement with metadata {agree[cell]['agreement']}  classes {agree[cell]['predicted_classes']}")
    json.dump({"date": "2026-10-22", "no_fabricated_numbers": True, "question": QUESTION, "cap": "cap320",
               "classes": [c for c, _ in CLASSES] + ["other"], "agreement_with_metadata": agree,
               "seconds": round(seconds, 1), "COMPLETE": True},
              open(ART, "w"), indent=1)


LOADER = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling", "run_openvqa.py")).read()
_s = LOADER.index("\nitems = []"); _e = LOADER.index("\nif A.idx_file:")
LOADER = LOADER[_s:_e].strip("\n")                                   # "items = []" ... up to the idx_file filter
if __name__ == "__main__":
    write_agreement(json.load(open(ART))["seconds"]) if "--agreement_only" in sys.argv else main()
