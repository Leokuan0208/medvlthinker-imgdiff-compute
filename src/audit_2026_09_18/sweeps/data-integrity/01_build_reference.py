#!/usr/bin/env python3
"""Rebuild, read-only, the item list run_openvqa.py would build for every open dataset (WITHOUT the
items[:A.n] truncation), so dumps can be compared against the dataset itself.
Writes out/reference.json: {ds: [{idx, question, gold, img_key}]}. img_key = path (JSON cells, slake)
or 'bytesmd5:<md5 of stored image bytes>' (parquet)."""
import json, os, glob, hashlib, io, sys
OUT = "/data/dan/audit_2026-09-18/tmp/data-integrity/out"
ref = {}
def slake(split):
    d = json.load(open(f"/data/dan/dataset/slake/{split}.json")); root = "/data/dan/dataset/slake/imgs"
    items, missing = [], 0
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if not os.path.exists(ip): missing += 1; continue
        items.append({"idx": x["qid"], "question": x["question"], "gold": str(x["answer"]), "img_key": ip})
    return items, missing
for s, nm in (("test", "slake_open"), ("train", "slake_open_train")):
    ref[nm], miss = slake(s); print(nm, len(ref[nm]), "missing imgs", miss)
JSON_CELLS = {"kvasir_open": "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json",
              "radimagenet_open": "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json",
              "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json",
              "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
              "vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
              "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json",
              "quilt_open": "/data/dan/dataset/quilt_cell/quilt_open.json"}
import time
for nm, p in JSON_CELLS.items():
    d = json.load(open(p)); miss = sum(1 for r in d if not os.path.exists(r["img_path"]))
    ref[nm] = [{"idx": r["idx"], "question": r["question"], "gold": r["answer"], "img_key": r["img_path"]} for r in d
               if (nm not in ("kvasir_open", "radimagenet_open")) or os.path.exists(r["img_path"])]
    print(nm, len(d), "rows in json;", len(ref[nm]), "kept; missing imgs", miss, "| json mtime",
          time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p))))
import pandas as pd
for nm, base, split in (("vqa_rad_open", "/data/dan/dataset/vqa_rad/data", "test"),
                        ("vqa_rad_open_train", "/data/dan/dataset/vqa_rad/data", "train"),
                        ("pathvqa_open", "/data/dan/dataset/path_vqa/data", "test"),
                        ("pathvqa_open_train", "/data/dan/dataset/path_vqa/data", "train")):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, f"{split}-*.parquet")))], ignore_index=True)
    items = []
    for i, r in df.iterrows():
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        a = str(a).strip()
        if a.lower() in ("yes", "no"): continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img): continue
        items.append({"idx": int(i), "question": str(q), "gold": a,
                      "img_key": "bytesmd5:" + hashlib.md5(img["bytes"]).hexdigest()})
    ref[nm] = items; print(nm, "parquet rows", len(df), "-> open items", len(items))
json.dump(ref, open(f"{OUT}/reference.json", "w"))
for k, v in ref.items():
    print(f"REF {k:22} n={len(v):6} unique_idx={len({str(x['idx']) for x in v}):6} unique_img={len({x['img_key'] for x in v}):6}")
