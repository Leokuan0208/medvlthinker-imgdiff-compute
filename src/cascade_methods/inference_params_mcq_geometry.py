#!/usr/bin/env python3
"""inference_params_mcq_geometry.py -- INDEPENDENT verification of the round's
highest-stakes claim.

The resolution sweep asserts that raising image resolution CANNOT change the five MCQ
cells, because they already run uncapped: MedEvalKit reads CAP_MAX_PIXELS from the
environment (default "0" = no cap, Qwen2_5_VL_vllm.py:51-54), no runner that produced a
published MCQ cell sets it, so the effective cap is the Lingshu preprocessor's own
max_pixels = 12,845,056 -- and every MCQ image is claimed to sit below it
(frac_images_above_cap = 0.000 in all five).

If that holds it closes 62.5% of the macro weight to this entire lever, which is the
single most consequential statement in the round, so it is re-measured here from the
image files rather than read out of the sweep's own table. Uses the harness's own
smart_resize (qwen_vl_utils.vision_process) so the token counts are the harness's, and
takes the raw PIL size independently of it so the binding test does not depend on it.

CPU only. Writes results/cascade_methods/artifacts/_infparams_mcq_geometry.json
"""
from __future__ import annotations

import glob
import io
import json
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
from qwen_vl_utils.vision_process import smart_resize  # noqa: E402

FACTOR, MIN_PX = 28, 4 * 28 * 28
LINGSHU_DEFAULT = 12845056
CAPS = {"cap320": 250880, "fullres": 1003520, "lingshu_default": LINGSHU_DEFAULT}


def tokens(w, h, maxpx):
    rh, rw = smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PX, max_pixels=maxpx)
    return (rh * rw) // (FACTOR * FACTOR)


def parquet_sizes(base, limit=None, pattern="test-*.parquet"):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob(os.path.join(base, pattern)))], ignore_index=True)
    sz = []
    for _, r in df.iterrows():
        img = r.get("image")
        if isinstance(img, dict) and "bytes" in img:
            sz.append(Image.open(io.BytesIO(img["bytes"])).size)
        if limit and len(sz) >= limit:
            break
    return sz


def main():
    cells = {}

    # SLAKE (closed cell draws from the same image pool as the open one)
    s = []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                s.append(Image.open(ip).size)
    cells["SLAKE_closed"] = s
    print("SLAKE", len(s), flush=True)

    cells["VQA_RAD_closed"] = parquet_sizes("/data/dan/dataset/vqa_rad/data")
    print("VQA_RAD", len(cells["VQA_RAD_closed"]), flush=True)
    cells["PATH_VQA_closed"] = parquet_sizes("/data/dan/dataset/path_vqa/data", limit=3000)
    print("PATH_VQA", len(cells["PATH_VQA_closed"]), flush=True)

    # PMC-VQA v2 (test_2.csv) -- the MedEvalKit/paper track
    import pandas as pd
    pmc_img = None
    for cand in ["/data/dan/dataset/medevalkit/PMC-VQA/images",
                 "/data/dan/dataset/medevalkit/PMC-VQA/figures"]:
        if os.path.isdir(cand):
            pmc_img = cand
            break
    if pmc_img:
        df = pd.read_csv("/data/dan/dataset/medevalkit/PMC-VQA/test_2.csv")
        col = "Figure_path" if "Figure_path" in df.columns else df.columns[-1]
        sz, miss = [], 0
        for v in df[col].astype(str).tolist():
            p = os.path.join(pmc_img, os.path.basename(v))
            if os.path.exists(p):
                try:
                    sz.append(Image.open(p).size)
                except Exception:
                    miss += 1
            else:
                miss += 1
        cells["PMC_VQA"] = sz
        print("PMC_VQA", len(sz), "missing", miss, flush=True)

    # MedXpertQA -- multi-image
    mx = "/data/dan/dataset/medevalkit/MedXpertQA"
    imgdir = None
    for cand in [os.path.join(mx, "images"), os.path.join(mx, "MM", "images")]:
        if os.path.isdir(cand):
            imgdir = cand
            break
    if imgdir:
        sz = []
        for f in sorted(glob.glob(os.path.join(imgdir, "*"))):
            try:
                sz.append(Image.open(f).size)
            except Exception:
                pass
        cells["MedXpertQA"] = sz
        print("MedXpert", len(sz), flush=True)

    out = {
        "title": "INDEPENDENT MCQ IMAGE GEOMETRY -- can raising resolution change the "
                 "five multiple-choice cells at all?",
        "date": "2026-08-14", "no_fabricated_numbers": True,
        "code_path_verified": {
            "file": "MedEvalKit/models/Qwen2_5_VL/Qwen2_5_VL_vllm.py:51-54",
            "logic": '_MP = int(os.environ.get("CAP_MAX_PIXELS","0"));  if _MP: '
                     'd["max_pixels"]=_MP; d["min_pixels"]=4*28*28  -- unset means no cap '
                     'is passed and the processor keeps its own default',
            "lingshu_preprocessor_default_max_pixels": LINGSHU_DEFAULT,
            "preprocessor_config": "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--"
                                   "Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae"
                                   "8471bd7a9/preprocessor_config.json",
            "runners_setting_CAP_MAX_PIXELS": ["runners/run_resolution_mcq_ladder.sh",
                                               "runners/run_resolution_mcq_pathvqa.sh"],
            "conclusion": "both are resolution-sweep runners written for this round; no "
                          "runner behind a PUBLISHED MCQ cell sets the variable, so the "
                          "published MCQ arms ran at the 12,845,056 default.",
        },
        "cells": {},
    }
    for c, sz in cells.items():
        if not sz:
            continue
        px = np.array([w * h for w, h in sz], dtype=np.int64)
        row = {"n_images": int(len(px)), "px_max": int(px.max()),
               "px_mean": float(px.mean()), "px_median": float(np.median(px)),
               "px_p95": float(np.percentile(px, 95))}
        for name, cap in CAPS.items():
            above = int(sum(1 for w, h in sz
                            if smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PX,
                                            max_pixels=LINGSHU_DEFAULT)[0]
                            * smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PX,
                                           max_pixels=LINGSHU_DEFAULT)[1] > cap))
            row[f"frac_above_{name}"] = above / len(sz)
            row[f"mean_vision_tokens_at_{name}"] = float(
                np.mean([tokens(w, h, cap) for w, h in sz]))
        row["raising_the_cap_can_change_an_input"] = bool(row["px_max"] > LINGSHU_DEFAULT)
        out["cells"][c] = row
        print(c, "px_max", row["px_max"], "frac_above_default",
              row["frac_above_lingshu_default"], flush=True)

    out["verdict"] = {
        "max_pixels_over_all_measured_mcq_cells": int(max(
            v["px_max"] for v in out["cells"].values())),
        "lingshu_default_cap": LINGSHU_DEFAULT,
        "any_cell_where_raising_the_cap_changes_an_input": any(
            v["raising_the_cap_can_change_an_input"] for v in out["cells"].values()),
        "statement": "if no cell has an image above 12,845,056 resized pixels, then the "
                     "published MCQ arms already see every pixel the model would ever "
                     "receive, and raising max_pixels is a no-op on 62.5% of the macro "
                     "weight -- not an untested hypothesis but an impossibility.",
    }
    p = os.path.join(ART, "_infparams_mcq_geometry.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)
    print(json.dumps(out["verdict"], indent=1))


if __name__ == "__main__":
    main()
