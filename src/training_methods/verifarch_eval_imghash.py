#!/usr/bin/env python3
"""verifarch_eval_imghash.py -- cache the image pixel hash of every one of the 2345 open-text EVAL
items, so cross-fitting folds can be made IMAGE-DISJOINT (items share images: 2345 items / 528 images).

Reuses build_disjoint_verifier_split.py's loaders and its md5-of-decoded-RGB-pixels convention verbatim.
  -> data/verifarch/eval_imghash.json   {ds: {str(idx): pixhash}}
"""
import glob, hashlib, io, json, os
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SLAKE = "/data/dan/dataset/slake"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"


def pixhash(img):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def load_slake(split):
    out = []
    for x in json.load(open(f"{SLAKE}/{split}.json")):
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en":
            continue
        ip = os.path.join(f"{SLAKE}/imgs", x["img_name"])
        if os.path.exists(ip):
            out.append((x["qid"], ip))
    return out


def load_parquet(base, split):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))],
                   ignore_index=True)
    out = []
    for i, r in df.iterrows():
        a = r.get("answer")
        if a is None and "conversations" in r:
            a = r["conversations"][1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        out.append((int(i), Image.open(io.BytesIO(img["bytes"]))))
    return out


def sc8_idx(ds):
    p = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl"
    return [json.loads(l)["idx"] for l in open(p) if l.strip()]


raw = {"slake_open": load_slake("test"),
       "vqa_rad_open": load_parquet(VQARAD, "test"),
       "pathvqa_open": load_parquet(PATHVQA, "test")}

out = {}
for ds, items in raw.items():
    want = set(sc8_idx(ds))
    d = {}
    for idx, img in items:
        if idx in want:
            d[str(idx)] = pixhash(img)
    out[ds] = d
    print(ds, "items", len(d), "distinct images", len(set(d.values())), flush=True)

os.makedirs(f"{ROOT}/data/verifarch", exist_ok=True)
json.dump(out, open(f"{ROOT}/data/verifarch/eval_imghash.json", "w"))
print("total items", sum(len(v) for v in out.values()),
      "total distinct images", len(set(h for v in out.values() for h in v.values())))
