#!/usr/bin/env python3
"""image_geometry_all8.py -- backlog item 0.9 (OPEN_EXPERIMENTS_2026-09-17.md).

On how many images does the cap320 pixel ceiling bind, and what do cap320 / fullres / native cost in
vision tokens? Read image sizes for the eight open benchmarks, with the SAME item filters as
src/labeling/run_openvqa.py, and apply Qwen2.5-VL's smart_resize (28-px patches, 2x2 merge -> one
vision token per 28x28 block) at each max_pixels setting.

  cap320  = HIGH_PX // 4 = 250,880 px   (every pool generated so far)
  fullres = HIGH_PX      = 1,003,520 px (what extract_generator_hidden.py extracts at)
  native  = qwen_vl_utils default MAX_PIXELS = 12,845,056 px

CPU only; reads image headers, no model. Output: results/cascade_methods/artifacts/image_geometry_all8_2026-10-15.json
"""
import glob, io, json, os, statistics, sys
import pandas as pd
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize, MAX_PIXELS

HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAPS = {"cap320": HIGH_PX // 4, "fullres": HIGH_PX, "native": MAX_PIXELS}
OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/image_geometry_all8_2026-10-15.json"


def items(ds):
    """(idx, size) per open-ended item, filters copied from run_openvqa.py."""
    if ds == "slake_open":
        root = "/data/dan/dataset/slake/imgs"
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en":
                continue
            ip = os.path.join(root, x["img_name"])
            if os.path.exists(ip):
                yield x["qid"], ip
        return
    paths = {"radimagenet_open": "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json",
             "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json",
             "vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
             "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
             "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json"}
    if ds in paths:
        for r in json.load(open(paths[ds])):
            if ds == "radimagenet_open" and not os.path.exists(r["img_path"]):
                continue
            yield r["idx"], r["img_path"]
        return
    base = "/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                   ignore_index=True)
    for i, r in df.iterrows():
        a = r.get("answer")
        if a is None and "conversations" in r:
            a = r["conversations"][1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            yield int(i), io.BytesIO(img["bytes"])


def tokens(w, h, maxpx):
    rh, rw = smart_resize(h, w, factor=28, min_pixels=MIN_PX, max_pixels=maxpx)
    return (rh // 28) * (rw // 28)


res = {"date": "2026-10-15", "no_fabricated_numbers": True,
       "caps_px": CAPS, "method": "PIL header size -> qwen_vl_utils smart_resize(factor 28) -> tokens = (h/28)*(w/28)",
       "benchmarks": {}}
for ds in ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open",
           "omnimed_open", "vqamed_open", "gemex_open"]:
    px, tok = [], {k: [] for k in CAPS}
    cache = {}
    for idx, src in items(ds):
        key = src if isinstance(src, str) else None
        if key and key in cache:
            w, h = cache[key]
        else:
            with Image.open(src) as im:
                w, h = im.size
            if key:
                cache[key] = (w, h)
        px.append(w * h)
        for k, m in CAPS.items():
            tok[k].append(tokens(w, h, m))
    n = len(px)
    row = {"n_questions": n, "n_distinct_image_paths": len(cache) or None,
           "median_px": int(statistics.median(px)),
           "frac_above_cap320": round(sum(p > CAPS["cap320"] for p in px) / n, 4),
           "frac_above_fullres": round(sum(p > CAPS["fullres"] for p in px) / n, 4),
           "mean_tokens": {k: round(sum(v) / n, 1) for k, v in tok.items()}}
    row["fullres_over_cap320_tokens"] = round(row["mean_tokens"]["fullres"] / row["mean_tokens"]["cap320"], 3)
    res["benchmarks"][ds] = row
    print(ds, json.dumps(row), flush=True)

tot = sum(r["n_questions"] for r in res["benchmarks"].values())
res["pooled_question_weighted"] = {
    k: round(sum(r["mean_tokens"][k] * r["n_questions"] for r in res["benchmarks"].values()) / tot, 1)
    for k in CAPS}
res["n_total"] = tot
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(res, open(OUT, "w"), indent=1)
print("pooled mean tokens", res["pooled_question_weighted"], "n", tot)
print("wrote", OUT)
