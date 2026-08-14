#!/usr/bin/env python3
"""resolution_geometry.py -- SWEEP 2 step 1: THE FACTS.

Establishes, per dataset, what the max_pixels cap actually DOES: the native image pixel
distribution, and the MERGED VISION TOKEN count Qwen2.5-VL would produce at each cap.
This is the precondition for reading any resolution accuracy sweep, because a cap that
does not BIND on a dataset's images changes nothing at all.

CPU only, no model load.  smart_resize is imported from qwen_vl_utils so the token counts
are the harness's own, not a re-derivation.

    python3 src/cascade_methods/resolution_geometry.py
"""
import glob
import io
import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)

from qwen_vl_utils.vision_process import smart_resize  # noqa: E402

FACTOR, MIN_PX = 28, 4 * 28 * 28
CAPS = {
    "cap80": 62720, "cap160": 125440, "cap320": 250880, "cap640": 501760,
    "fullres": 1003520, "cap2560": 2007040, "medevalkit_default": 12845056,
}


def tokens(w, h, maxpx):
    """merged vision tokens for one image at this cap (Qwen2.5-VL: 28x28 merged patch)."""
    rh, rw = smart_resize(h, w, factor=FACTOR, min_pixels=MIN_PX, max_pixels=maxpx)
    return (rh * rw) // (FACTOR * FACTOR)


def open_sets():
    """the 3 OPEN cells, loaded exactly as run_openvqa.py / verifier_transfer_eval.py do."""
    out = {}
    s = []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                s.append(Image.open(ip).size)
    out["slake_open"] = s
    import pandas as pd
    for ds, base in [("vqa_rad_open", "/data/dan/dataset/vqa_rad/data"),
                     ("pathvqa_open", "/data/dan/dataset/path_vqa/data")]:
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
        sz = []
        for i, r in df.iterrows():
            a = r.get("answer")
            if a is None and "conversations" in r:
                a = r["conversations"][1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                sz.append(Image.open(io.BytesIO(img["bytes"])).size)
            if ds == "pathvqa_open" and len(sz) >= 1500:
                break
        out[ds] = sz
    return out


def mcq_sets():
    """the 5 MCQ cells as MedEvalKit serves them (datasets_path=hf under /data/dan/dataset/medevalkit)."""
    import pandas as pd
    out = {}
    # SLAKE closed / VQA-RAD closed / PathVQA closed share the loaders above but keep yes/no
    for ds, base, lim in [("VQA_RAD_closed", "/data/dan/dataset/vqa_rad/data", None),
                          ("PATH_VQA_closed", "/data/dan/dataset/path_vqa/data", 3000)]:
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
        sz = []
        for i, r in df.iterrows():
            a = r.get("answer")
            if a is None and "conversations" in r:
                a = r["conversations"][1]["value"]
            if str(a).strip().lower() not in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                sz.append(Image.open(io.BytesIO(img["bytes"])).size)
            if lim and len(sz) >= lim:
                break
        out[ds] = sz
    s = []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type") == "CLOSED" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                s.append(Image.open(ip).size)
    out["SLAKE_closed"] = s
    return out


def summarise(sizes):
    px = np.array([w * h for w, h in sizes], dtype=np.float64)
    row = dict(n=len(sizes),
               px_mean=float(px.mean()), px_median=float(np.median(px)),
               px_p95=float(np.percentile(px, 95)), px_max=float(px.max()), px_min=float(px.min()))
    row["by_cap"] = {}
    base = None
    for c in sorted(CAPS, key=lambda k: CAPS[k]):
        t = np.array([tokens(w, h, CAPS[c]) for w, h in sizes], dtype=np.float64)
        binds = float(np.mean([w * h > CAPS[c] for w, h in sizes]))
        row["by_cap"][c] = dict(max_pixels=CAPS[c], mean_vision_tokens=float(t.mean()),
                                max_vision_tokens=float(t.max()),
                                frac_images_above_cap=round(binds, 4))
        if c == "medevalkit_default":
            base = t.mean()
    for c in row["by_cap"]:
        row["by_cap"][c]["mean_tokens_rel_to_medevalkit_default"] = round(
            row["by_cap"][c]["mean_vision_tokens"] / base, 4)
    return row


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "open"
    sets = open_sets() if which == "open" else mcq_sets()
    res = {k: summarise(v) for k, v in sets.items()}
    for k, v in res.items():
        print(k, "n=%d" % v["n"], "px_median=%.0f" % v["px_median"], "px_max=%.0f" % v["px_max"])
        for c, d in v["by_cap"].items():
            print("   %-20s maxpx=%9d  mean_tok=%8.1f  max_tok=%7.0f  binds_on=%.3f"
                  % (c, d["max_pixels"], d["mean_vision_tokens"], d["max_vision_tokens"],
                     d["frac_images_above_cap"]))
    json.dump(res, open(os.path.join(OUT, f"geometry_{which}.json"), "w"), indent=1)
    print("wrote", os.path.join(OUT, f"geometry_{which}.json"))
