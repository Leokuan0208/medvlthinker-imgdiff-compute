#!/usr/bin/env python3
"""build_comparability_cells.py -- the cells the published protocols need and our open-only cells lack.

The comparable papers (survey 2026-10-15; LLaVA-Med 2306.00890, Lingshu 2506.07044, MedRegA 2410.18387,
Med-Gemini, MedGemma) report the FULL official test sets, closed + open, and VQA-Med 2019 on the
official 500-question test. Our eight cells are open-only (yes/no dropped) and our vqamed cell is
mostly the 2019 training split. This writes, under /data/dan/dataset/comparability_cells/:

  vqa_rad_closed.json   VQA-RAD HF test split, rows whose answer IS yes/no          (open cell = the rest)
  pathvqa_closed.json   PathVQA HF test split, rows whose answer IS yes/no          (open cell = the rest)
  slake_closed.json     SLAKE test.json, q_lang en, answer_type CLOSED              (open cell = OPEN)
  vqamed2019_test.json  ImageCLEF VQA-Med 2019 official test, all 500 (4 categories), reference
                        answers verbatim ('#' alternatives, '()' optional parts) + category

idx conventions match the open cells exactly (parquet row index; SLAKE qid), so an open cell and its
closed cell together are the full official test split. Embedded parquet images are written as PNG.
Each record: idx, question, answer, img_path (+ category / answer_type).
"""
import glob, io, json, os
import pandas as pd
from PIL import Image

OUT = "/data/dan/dataset/comparability_cells"
os.makedirs(OUT, exist_ok=True)


def parquet_closed(name, base):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    imgdir = f"{OUT}/{name}_images"
    os.makedirs(imgdir, exist_ok=True)
    recs, n_open = [], 0
    for i, r in df.iterrows():
        q, a = r.get("question"), r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        a = str(a).strip()
        if a.lower() not in ("yes", "no"):
            n_open += 1
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        p = f"{imgdir}/{int(i)}.png"
        if not os.path.exists(p):
            Image.open(io.BytesIO(img["bytes"])).convert("RGB").save(p)
        recs.append({"idx": int(i), "question": str(q), "answer": a, "img_path": p, "answer_type": "CLOSED"})
    json.dump(recs, open(f"{OUT}/{name}.json", "w"), indent=0)
    print(f"{name}: {len(recs)} closed (yes/no); {n_open} open rows not written; total {len(df)}")


def slake_closed():
    recs = []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("q_lang") != "en" or x.get("answer_type") != "CLOSED":
            continue
        p = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
        if os.path.exists(p):
            recs.append({"idx": x["qid"], "question": x["question"], "answer": str(x["answer"]),
                         "img_path": p, "answer_type": "CLOSED"})
    json.dump(recs, open(f"{OUT}/slake_closed.json", "w"), indent=0)
    print(f"slake_closed: {len(recs)}")


def vqamed2019_test():
    D = "/data/dan/dataset/vqamed2019/x/VQA-Med-2019/VQAMed2019Test"
    recs = []
    for i, line in enumerate(open(f"{D}/VQAMed2019_Test_Questions_w_Ref_Answers.txt")):
        if not line.strip():
            continue
        img_id, cat, q, a = line.rstrip("\n").split("|")
        p = f"{D}/VQAMed2019_Test_Images/{img_id}.jpg"
        assert os.path.exists(p), p
        # 'answer' is what generation stores as the gold and the LLM judge sees: '#' alternatives are
        # spelled out as "a or b" (parentheses kept). 'answer_ref' is the official reference verbatim,
        # used for ImageCLEF-style strict accuracy.
        recs.append({"idx": i, "question": q, "answer": " or ".join(a.split("#")), "answer_ref": a,
                     "img_path": p, "category": cat, "image_id": img_id})
    json.dump(recs, open(f"{OUT}/vqamed2019_test.json", "w"), indent=0)
    print(f"vqamed2019_test: {len(recs)}")


parquet_closed("vqa_rad_closed", "/data/dan/dataset/vqa_rad/data")
parquet_closed("pathvqa_closed", "/data/dan/dataset/path_vqa/data")
slake_closed()
vqamed2019_test()
