#!/usr/bin/env python3
"""Decoded-RGB-pixel md5 (the project's img_md5 currency, extract_generator_hidden.img_md5) and a 64-bit
dHash + 16x16 grayscale thumbnail signature for every unique image of every open dataset.
Read-only on datasets. 4 worker processes. Output out/pixhash.json {ds: {img_key: [pixmd5, dhash, w, h]}}"""
import json, os, glob, hashlib, io, sys
from multiprocessing import Pool
from PIL import Image
import numpy as np
OUT = "/data/dan/audit_2026-09-18/tmp/data-integrity/out"

def sig(im):
    im = im.convert("RGB")
    md5 = hashlib.md5(im.tobytes()).hexdigest()
    g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    dh = "%016x" % int("".join("1" if b else "0" for b in bits), 2)
    return md5, dh, im.size[0], im.size[1]

def work_path(p):
    try:
        return p, sig(Image.open(p))
    except Exception as e:
        return p, ("ERR:" + str(e)[:80], "", 0, 0)

def work_bytes(a):
    k, b = a
    try:
        return k, sig(Image.open(io.BytesIO(b)))
    except Exception as e:
        return k, ("ERR:" + str(e)[:80], "", 0, 0)

if __name__ == "__main__":
    ref = json.load(open(f"{OUT}/reference.json"))
    res = {}
    with Pool(4) as pool:
        for ds, items in ref.items():
            keys = sorted({x["img_key"] for x in items})
            if keys and keys[0].startswith("bytesmd5:"):
                continue
            res[ds] = dict(pool.imap_unordered(work_path, keys, chunksize=32))
            print(ds, len(res[ds]), "images;", len({v[0] for v in res[ds].values()}), "unique pixmd5", flush=True)
        import pandas as pd
        for nm, base, split in (("vqa_rad_open", "/data/dan/dataset/vqa_rad/data", "test"),
                                ("vqa_rad_open_train", "/data/dan/dataset/vqa_rad/data", "train"),
                                ("pathvqa_open", "/data/dan/dataset/path_vqa/data", "test"),
                                ("pathvqa_open_train", "/data/dan/dataset/path_vqa/data", "train")):
            want = {x["img_key"] for x in ref[nm]}
            seen, jobs = set(), []
            for f in sorted(glob.glob(os.path.join(base, f"{split}-*.parquet"))):
                df = pd.read_parquet(f, columns=["image"])
                for img in df["image"]:
                    b = img["bytes"]; k = "bytesmd5:" + hashlib.md5(b).hexdigest()
                    if k in want and k not in seen:
                        seen.add(k); jobs.append((k, b))
            res[nm] = dict(pool.imap_unordered(work_bytes, jobs, chunksize=16))
            print(nm, len(res[nm]), "images;", len({v[0] for v in res[nm].values()}), "unique pixmd5", flush=True)
    json.dump(res, open(f"{OUT}/pixhash.json", "w"))
    print("DONE")
