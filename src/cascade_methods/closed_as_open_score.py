#!/usr/bin/env python3
"""closed_as_open_score.py -- BUILD 3 GPU half B: score every candidate with the FROZEN verifier.

Adapter: ckpts/train/lora_verifier_disjoint (the clean disjoint-trained incumbent, sel_eff 0.775204).
NOTHING is trained here.  pyes() is copied VERBATIM from src/training_methods/verifier_transfer_eval.py
so the score is on the same scale as every published cell.

HF transformers ONLY -- vLLM 0.9/0.10 silently drops all 192 visual.* LoRA modules (0.775204 HF vs
0.702997 vLLM).  Candidates are DEDUPLICATED by normalised answer, so a question with 8 identical
samples costs ONE forward, not eight.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 \
      src/cascade_methods/closed_as_open_score.py --arms openPRJ_s8 --cells SLAKE_closed

Resumable per item.  Launch from the repo root.  nohup, never tmux.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["openPRJ_s8", "openMEK_s8", "closedD_s8"])
ap.add_argument("--cells", nargs="+", default=L.CELLS)
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
A = ap.parse_args()

import torch                                                              # noqa: E402
torch.backends.cuda.matmul.allow_tf32 = False                             # numerics pinned
torch.backends.cudnn.allow_tf32 = False

from PIL import Image                                                     # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText       # noqa: E402
from qwen_vl_utils import process_vision_info                             # noqa: E402
from peft import PeftModel                                                # noqa: E402

DEV = "cuda"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
       "the proposed answer is correct. Respond with only 'Yes' or 'No'.")

print("[items] loading ...", flush=True)
ITEMS = {c: {it["i"]: it for it in v} for c, v in L.build_items().items()}

proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
print("[model] loading base + FROZEN adapter", A.adapter, flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
model = PeftModel.from_pretrained(model, os.path.join(L.ROOT, A.adapter))
model.eval()


def load_image(item):
    p = item["images"][0]
    if item["img_kind"] == "path":
        return p
    if item["img_kind"] == "raw":
        return Image.open(p)
    if item["img_kind"] == "rawrgb":
        return Image.open(p).convert("RGB")
    raise ValueError(item["img_kind"])


def pyes(q, img, ans):
    """VERBATIM from src/training_methods/verifier_transfer_eval.py."""
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [
        {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
        {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                 f"Is the proposed answer correct? Answer Yes or No."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item())
        pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


for arm in A.arms:
    for cell in A.cells:
        gen = L.load_gen(cell, arm)
        if not gen:
            print(f"[skip] {arm} {cell}: no generations yet", flush=True)
            continue
        out = L.scores_path(cell, arm)
        done = set(L.load_scores(cell, arm))
        todo = [i for i in sorted(gen) if i not in done]
        if not todo:
            print(f"[skip] {arm} {cell}: complete ({len(done)})", flush=True)
            continue
        print(f"[run ] {arm} {cell}: {len(todo)} items to score -> {os.path.basename(out)}", flush=True)
        t0, nfwd = time.time(), 0
        with open(out, "a", encoding="utf-8") as fh:
            for k, i in enumerate(todo):
                try:
                    r = gen[i]
                    item = ITEMS[cell][i]
                    img = load_image(item)
                    q = r["question"]
                    cache = {}
                    sc = []
                    for a in r["preds"]:
                        na = L.norm_text(a)
                        if na not in cache:
                            cache[na] = pyes(q, img, a)
                            nfwd += 1
                        sc.append(round(float(cache[na]), 6))
                    fh.write(json.dumps({"i": i, "cell": cell, "arm": arm, "scores": sc,
                                         "n_forward": len(cache)}) + "\n")
                except Exception as e:                              # per-item error guard
                    print(f"   !! item {i} failed: {type(e).__name__}: {e}", flush=True)
                if (k + 1) % 200 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"   [{k+1}/{len(todo)}] {(k+1)/el:.2f} it/s  fwd/item {nfwd/(k+1):.2f}",
                          flush=True)
            fh.flush()
        got = len(L.load_scores(cell, arm))
        print(f"[done] {arm} {cell} {got}/{len(gen)} items, {nfwd} forwards, "
              f"{(time.time()-t0)/60:.2f} min", flush=True)
print("CLOSED_AS_OPEN_SCORE_DONE", flush=True)
