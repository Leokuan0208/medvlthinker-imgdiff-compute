#!/usr/bin/env python3
"""Prep RadImageNet-VQA open-ended eval slice into the Kvasir-style {images/, json} format the open-VQA
pipeline already supports. Extracts the 2000 open questions (anatomy+pathology), saves PNGs + a json list
[{idx, question, answer, img_path, content_type}]. Run once.
  HF_HOME=/data/dan/hf_cache python3 src/data_prep/prep_radimagenet.py
"""
import os, io, json
import pandas as pd
from PIL import Image
SRC = "/data/dan/dataset/radimagenet_vqa/benchmark/test-00000-of-00001.parquet"
OUT = "/data/dan/dataset/radimagenet_vqa"
IMGDIR = os.path.join(OUT, "images"); os.makedirs(IMGDIR, exist_ok=True)
df = pd.read_parquet(SRC)
op = df[df["question_type"] == "open"].reset_index(drop=True)
recs = []
for i, r in op.iterrows():
    img = r["image"]
    if not (isinstance(img, dict) and "bytes" in img): continue
    ip = os.path.join(IMGDIR, f"radimg_{i}.png")
    if not os.path.exists(ip):
        Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(ip)
    md = r["metadata"] if isinstance(r["metadata"], dict) else {}
    recs.append({"idx": int(i), "question": str(r["question"]), "answer": str(r["answer"]),
                 "img_path": ip, "content_type": md.get("content_type"), "modality": md.get("modality")})
json.dump(recs, open(os.path.join(OUT, "radimagenet_open_2000.json"), "w"))
print(f"wrote {len(recs)} open-ended items -> {OUT}/radimagenet_open_2000.json ; images in {IMGDIR}")
print("content_type:", {c: sum(1 for r in recs if r['content_type']==c) for c in set(r['content_type'] for r in recs)})
