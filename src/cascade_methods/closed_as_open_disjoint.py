#!/usr/bin/env python3
"""closed_as_open_disjoint.py -- BUILD 3: PROVE, not assume, that the frozen verifier never saw an
image of the three NEW eval cells.

The published disjointness assertion (artifacts/verifier_disjoint_split.json) covers only the three
OPEN eval sets.  SLAKE_closed / VQA_RAD_closed / PATH_VQA_closed are new eval surface, so the
intersection has to be recomputed against them.

Currency: md5 of DECODED RGB pixels with a "WxH|" prefix -- pixhash() verbatim from
src/training_methods/build_disjoint_verifier_split.py, so the numbers are on the published scale.

CPU only.  python3 src/cascade_methods/closed_as_open_disjoint.py
Writes results/cascade_methods/artifacts/_closed_as_open_parts/disjointness.json
"""
import glob
import hashlib
import io
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

OUT = os.path.join(L.PARTS, "disjointness.json")
IDX = os.path.join(L.ROOT, "data/disjoint_split")


def pixhash(img):
    """VERBATIM from src/training_methods/build_disjoint_verifier_split.py:66."""
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def train_hashes():
    """Every image in the verifier's L1 training pool (a superset of what it actually sampled)."""
    import pandas as pd
    out = {}

    allow = set(json.load(open(os.path.join(IDX, "idx_slake_open_train.json"))))
    h = set()
    for x in json.load(open("/data/dan/dataset/slake/train.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en" and x["qid"] in allow:
            p = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(p):
                h.add(pixhash(p))
    out["slake_open_train"] = h

    for name, base in [("vqa_rad_open_train", "/data/dan/dataset/vqa_rad/data"),
                       ("pathvqa_open_train", "/data/dan/dataset/path_vqa/data")]:
        allow = set(json.load(open(os.path.join(IDX, f"idx_{name}.json"))))
        h = set()
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/train-*.parquet"))],
                       ignore_index=True)
        for i, r in df.iterrows():
            if int(i) not in allow:
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                h.add(pixhash(Image.open(io.BytesIO(img["bytes"]))))
        out[name] = h

    for name, jp in [("kvasir_open", "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
                     ("radimagenet_open", "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")]:
        allow = set(json.load(open(os.path.join(IDX, f"idx_{name}.json"))))
        h = set()
        for r in json.load(open(jp)):
            if r["idx"] in allow and os.path.exists(r["img_path"]):
                h.add(pixhash(r["img_path"]))
        out[name] = h
    return out


def eval_hashes():
    items = L.build_items()
    out = {}
    for cell in L.CELLS:
        h = {}
        for it in items[cell]:
            p = it["images"][0]
            img = p if it["img_kind"] == "path" else Image.open(p)
            h.setdefault(pixhash(img), []).append(it["i"])
        out[cell] = h
    return out


TRCACHE = os.path.join(L.PARTS, "train_pixhashes.json")


def main():
    os.makedirs(L.PARTS, exist_ok=True)
    if os.path.exists(TRCACHE):
        print(f"[train] reusing {TRCACHE}", flush=True)
        tr = {k: set(v) for k, v in json.load(open(TRCACHE)).items()}
    else:
        print("[train] hashing the verifier's L1 training pool ...", flush=True)
        tr = train_hashes()
        json.dump({k: sorted(v) for k, v in tr.items()}, open(TRCACHE, "w"))
    allt = set().union(*tr.values())
    print(f"[train] {len(allt)} distinct training images "
          + ", ".join(f"{k}={len(v)}" for k, v in tr.items()), flush=True)
    print("[eval ] hashing the three NEW eval cells ...", flush=True)
    ev = eval_hashes()

    doc = {"title": "BUILD 3 image disjointness: frozen verifier training images vs the three NEW "
                    "closed eval cells",
           "date": L.DATE,
           "currency": "md5 of DECODED RGB pixels with a 'WxH|' prefix -- pixhash() verbatim from "
                       "src/training_methods/build_disjoint_verifier_split.py:66",
           "adapter": "ckpts/train/lora_verifier_disjoint (frozen)",
           "train_pool": {k: len(v) for k, v in tr.items()},
           "train_pool_distinct_images": len(allt),
           "per_cell": {}}
    worst = 0
    flags = {}
    for cell in L.CELLS:
        inter = set(ev[cell]) & allt
        bad = sorted(i for h in inter for i in ev[cell][h])
        worst = max(worst, len(bad))
        flags[cell] = bad
        doc["per_cell"][cell] = {
            "n_items": L.EXPECT_N[cell], "n_distinct_images": len(ev[cell]),
            "image_pixel_md5_intersection_with_training_pool": len(inter),
            "n_eval_items_on_a_shared_image": len(bad),
            "frac_items_on_a_shared_image": round(len(bad) / L.EXPECT_N[cell], 5),
            "n_items_image_disjoint": L.EXPECT_N[cell] - len(bad),
            "per_source": {k: len(set(ev[cell]) & v) for k, v in tr.items()},
        }
        print(f"  {cell:<16} images={len(ev[cell]):>5}  intersection={len(inter)}  "
              f"items_affected={len(bad)}", flush=True)
    doc["clean"] = bool(worst == 0)
    doc["note"] = ("the L1 pool is a SUPERSET of the examples the verifier actually sampled "
                   "(10,364 composition-matched draws), so a zero here is conservative. Where the "
                   "intersection is NON-zero the analysis reports BOTH the full cell and the "
                   "IMAGE-DISJOINT subset, and the image-disjoint subset is the honest number.")
    doc["why_the_published_split_did_not_catch_this"] = (
        "artifacts/verifier_disjoint_split.json proved disjointness against the three OPEN eval "
        "sets only; SLAKE_closed / VQA_RAD_closed were not eval surface then. VQA-RAD and SLAKE "
        "split train/test by QUESTION, not by image, so the same picture carries both an open "
        "training question and a closed test question.")
    json.dump(doc, open(OUT, "w"), indent=1)
    json.dump(flags, open(os.path.join(L.PARTS, "contaminated_items.json"), "w"))
    print(f"wrote {OUT}  clean={doc['clean']}")


if __name__ == "__main__":
    main()
