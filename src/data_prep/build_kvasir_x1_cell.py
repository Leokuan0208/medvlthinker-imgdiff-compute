#!/usr/bin/env python3
"""build_kvasir_x1_cell.py -- carve a CLEAN open-text eval cell out of Kvasir-VQA-x1's test split.

THE CONTAMINATION THIS EXISTS TO AVOID.  Our kvasir_open_1200 pool was drawn from this very test
split: 1,052 of its 4,058 images are in /data/dan/dataset/kvasir_vqa_x1/images, and the frozen head
was trained on 5,562 candidate rows from them.  Every one of those images is excluded here, by
image id, before anything else happens.  What survives is 10,703 questions over 3,006 images.

  python3 src/data_prep/build_kvasir_x1_cell.py --max_words 0     # all 10,703
  python3 src/data_prep/build_kvasir_x1_cell.py --max_words 6     # the tight ~3,504-item subset
  python3 src/data_prep/build_kvasir_x1_cell.py --n 2000 --seed 0 # a fixed-size random draw

Writes /data/dan/dataset/kvasir_x1_cell/{images/*.jpg, kvasir_x1_open.json} in the project's
standard {idx, question, answer, img_path} shape, so run_openvqa.py and
extract_generator_hidden.py can take it with a one-line branch each.
"""
import argparse, io, json, os, hashlib
import numpy as np

SRC = "/data/dan/dataset/kvasir_vqa_x1_official/data/test-00000-of-00001.parquet"
BURNED_DIR = "/data/dan/dataset/kvasir_vqa_x1/images"          # what the head already trained on
OUT = "/data/dan/dataset/kvasir_x1_cell"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max_words", type=int, default=0,
                    help="keep only answers with <= this many words (0 = no filter)")
    ap.add_argument("--single_class", action="store_true",
                    help="drop multi-label question_class rows, which carry the long answers")
    ap.add_argument("--n", type=int, default=0, help="random subsample size (0 = keep all)")
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    burned = {os.path.splitext(x)[0] for x in os.listdir(BURNED_DIR)}
    print(f"[excl] {len(burned)} contaminated image ids from {BURNED_DIR}", flush=True)

    f = pq.ParquetFile(A.src)
    df = f.read().to_pandas()
    n0 = len(df)
    df["img_id"] = df["img_id"].astype(str)
    df = df[~df["img_id"].isin(burned)].copy()
    print(f"[excl] {n0} -> {len(df)} questions after removing contaminated images "
          f"({df['img_id'].nunique()} images)", flush=True)

    df["w"] = [len(str(a).split()) for a in df["answer"]]
    if A.single_class:
        df = df[df["question_class"].astype(str).str.count(",") == 0]
        print(f"[filt] single-class only -> {len(df)}", flush=True)
    if A.max_words:
        df = df[df["w"] <= A.max_words]
        print(f"[filt] answers <= {A.max_words} words -> {len(df)} "
              f"({df['img_id'].nunique()} images)", flush=True)
    if A.n and A.n < len(df):
        # sample by IMAGE so a question and its image never straddle the draw
        rng = np.random.default_rng(A.seed)
        imgs = sorted(df["img_id"].unique())
        rng.shuffle(imgs)
        keep, taken = set(), 0
        for im in imgs:
            k = int((df["img_id"] == im).sum())
            if taken + k > A.n:
                continue
            keep.add(im); taken += k
            if taken >= A.n:
                break
        df = df[df["img_id"].isin(keep)]
        print(f"[samp] image-grouped draw -> {len(df)} over {df['img_id'].nunique()} images",
              flush=True)

    imgdir = os.path.join(A.out, "images")
    os.makedirs(imgdir, exist_ok=True)
    written, rows = {}, []
    for i, (_, r) in enumerate(df.iterrows()):
        iid = r["img_id"]
        p = os.path.join(imgdir, f"{iid}.jpg")
        if iid not in written:
            b = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
            Image.open(io.BytesIO(b)).convert("RGB").save(p, quality=95)
            written[iid] = p
        rows.append({"idx": i, "question": str(r["question"]), "answer": str(r["answer"]),
                     "img_path": p, "img_id": iid,
                     "question_class": str(r["question_class"]),
                     "complexity": (int(r["complexity"]) if str(r["complexity"]).isdigit()
                                    else str(r["complexity"]))})
    jp = os.path.join(A.out, "kvasir_x1_open.json")
    json.dump(rows, open(jp, "w"))
    ws = [len(r["answer"].split()) for r in rows]
    print(f"\nwrote {jp}\n  {len(rows)} questions / {len(written)} images"
          f"\n  answer words mean {np.mean(ws):.2f} median {np.median(ws):.1f} "
          f"p90 {np.percentile(ws,90):.1f}")
    print(f"  DISJOINTNESS: intersection with the head's kvasir train images = "
          f"{len(set(written) & burned)}  (must be 0)")


if __name__ == "__main__":
    main()
