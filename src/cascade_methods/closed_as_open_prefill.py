#!/usr/bin/env python3
"""closed_as_open_prefill.py -- BUILD 3: MEASURE, do not assert, that the reformatted open arm is
cheaper than the deployed operating point.

The reformat result (PathVQA +0.038 over the deployed fullres point) is only interesting if the open
arm is genuinely no more expensive.  Two things differ between openMEK_g/openPRJ_g and closedD_g_full:
the prompt text, and the pixel budget (cap320 = HIGH_PX/4 against fullres = HIGH_PX).  Both land in
the PREFILL, which is where this workload's cost lives (82.1% LM prefill per
artifacts/cost_decomposition_2026-08-12.json).

So this counts REAL prefill tokens -- len(input_ids) from the processor on the byte-identical request
the generator built -- rather than reasoning about pixel budgets.  Decode is 3 tokens on every arm
(measured, see the artifact's descriptives), so prefill IS the cost.

CPU only, no GPU, no model weights -- processor and image resize only.  A pre-specified seeded sample
per cell (the full PathVQA cell is 3,362 images and the point does not need all of them); n and seed
are reported with every number.

  python3 src/cascade_methods/closed_as_open_prefill.py [--n 200]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

SEED = 20260816
MODEL = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
         "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ARMS = ["closedD_g", "closedD_g_full", "openMEK_g", "openPRJ_g"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model_path", default=MODEL)
    A = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
    ITEMS = L.build_items()

    def load_image(item):
        out = []
        for p in item["images"]:
            if item["img_kind"] == "path":
                out.append(p)
            elif item["img_kind"] == "raw":
                out.append(Image.open(p))
            elif item["img_kind"] == "rawrgb":
                out.append(Image.open(p).convert("RGB"))
            else:
                raise ValueError(item["img_kind"])
        return out

    def prefill_tokens(item, arm, cell):
        cfg = L.ARMS[arm]
        prompt = L.build_prompt(cfg["prompt"], cell, item["question"], item["lang"])
        maxpx = L.HIGH_PX // L.CAP_DIV[cfg["cap"]]
        imds = [{"type": "image", "image": im, "max_pixels": maxpx, "min_pixels": L.MIN_PX}
                for im in load_image(item)]
        if len(imds) == 1:
            content = [imds[0], {"type": "text", "text": prompt}]
        else:
            content = []
            for k, d in enumerate(imds):
                content += [{"type": "text", "text": f"<image_{k+1}>: "}, d]
            content.append({"type": "text", "text": prompt})
        msgs = []
        if cfg["sys"]:
            msgs.append({"role": "system", "content": cfg["sys"]})
        msgs.append({"role": "user", "content": content})
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        enc = proc(text=[text], images=imgs, return_tensors="pt")
        return int(enc["input_ids"].shape[1])

    out = {"title": "BUILD 3 -- MEASURED prefill length per arm; is the reformatted open arm actually "
                    "cheaper than the deployed operating point?",
           "date": L.DATE, "currency": "prefill tokens = len(input_ids) from the Lingshu-7B processor "
                                       "on the byte-identical request the generator built",
           "decode_is_not_the_cost": "every arm decodes 3.0 generated tokens (artifact descriptives), "
                                     "so prefill is the whole per-question cost difference",
           "sample": {"n_per_cell": A.n, "seed": SEED, "rule": "seeded uniform draw without "
                                                               "replacement over the cell's items"},
           "cells": {}}
    for cell in L.CELLS:
        rows = ITEMS[cell]
        rng = np.random.default_rng(SEED)
        pick = sorted(rng.choice(len(rows), size=min(A.n, len(rows)), replace=False).tolist())
        sub = [rows[j] for j in pick]
        cd = {"n_sampled": len(sub), "n_cell": len(rows)}
        for arm in ARMS:
            t = [prefill_tokens(it, arm, cell) for it in sub]
            cd[arm] = {"mean_prefill_tokens": round(float(np.mean(t)), 2),
                       "median": int(np.median(t)), "min": int(np.min(t)), "max": int(np.max(t)),
                       "cap": L.ARMS[arm]["cap"]}
            print(f"  {cell:16s} {arm:15s} cap={L.ARMS[arm]['cap']:8s} "
                  f"mean prefill = {np.mean(t):8.1f} tokens", flush=True)
        ref = cd["closedD_g_full"]["mean_prefill_tokens"]
        cd["ratio_vs_deployed_operating_point"] = {
            a: round(cd[a]["mean_prefill_tokens"] / ref, 4) for a in ARMS}
        print(f"  {cell:16s} ratio vs closedD_g_full: "
              + "  ".join(f"{a}={cd['ratio_vs_deployed_operating_point'][a]:.4f}" for a in ARMS),
              flush=True)
        out["cells"][cell] = cd
    os.makedirs(L.PARTS, exist_ok=True)
    p = os.path.join(L.PARTS, "prefill_cost.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
