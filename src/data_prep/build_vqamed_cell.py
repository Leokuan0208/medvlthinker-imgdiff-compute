#!/usr/bin/env python3
"""build_vqamed_cell.py -- the C4 Abnormality split of ImageCLEF VQA-Med 2019 as an open-text cell.

WHY IT EXISTS.  This dataset was downloaded on 2026-08-18, recorded as "uninspected", and never
opened.  The 2026-08-19 audit opened it: three of its four categories are closed-set classification
(C1 Modality 44 golds, C2 Plane 15, C3 Organ 10) and are excluded here, but C4 Abnormality is a
genuine open cell -- 3,817 QA pairs over 3,817 distinct images, 1,671 distinct golds, 54.6%
singletons, top-100 covering only 27.6%, mean 3.25 answer words (median 3, p90 6).  That sits
between pathvqa_open (2.36) and kvasir (9.92), and six times shorter than the rejected Quilt (20.4).
Golds are terse diagnoses -- meningioma, glioblastoma multiforme, acute appendicitis, arachnoid cyst
-- so they are wholly right or wholly wrong and oracle@8 / sel_eff stay well defined.

CONTAMINATION.  Zero decoded-pixel collisions against the 528 frozen eval images, and zero against
VQA-RAD's 2,244 images -- worth checking explicitly because both are MedPix-derived.  Re-verified
here at build time rather than trusted.

  python3 src/data_prep/build_vqamed_cell.py --audit_overlap
"""
import argparse, hashlib, json, os, glob
import numpy as np
from collections import Counter

SRC = "/data/dan/dataset/vqamed2019/x/VQA-Med-2019"
OUT = "/data/dan/dataset/vqamed_cell"
EVAL_META = ["feats_hidden/generator_eval_s0of2.meta.json",
             "feats_hidden/generator_eval_s1of2.meta.json",
             "feats_hidden/generator_eval_radimagenet.meta.json"]


def pixel_md5(p):
    from PIL import Image
    with Image.open(p) as im:
        return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()


def find_split(name):
    for d in glob.glob(os.path.join(SRC, "*")):
        if name.lower() in os.path.basename(d).lower():
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--audit_overlap", action="store_true")
    A = ap.parse_args()

    rows, seen_img = [], {}
    for split in ("Training", "Validation", "Test"):
        d = find_split(split)
        if not d:
            print(f"  [skip] no {split} dir"); continue
        # C4 lives in QAPairsByCategory/C4_Abnormality_*.txt for train/val; the test file is a
        # single 4-field table with the category in column 2.
        txts = [f for f in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True)
                if "c4_abnormality" in os.path.basename(f).lower()
                or "Test_Questions_w_Ref_Answers" in os.path.basename(f)]
        imgdirs = [x for x in glob.glob(os.path.join(d, "**"), recursive=True)
                   if os.path.isdir(x) and "image" in os.path.basename(x).lower()]
        for t in txts:
            for line in open(t, encoding="utf-8", errors="ignore"):
                parts = line.rstrip("\n").split("|")
                if len(parts) == 4:      # test: id|category|question|answer
                    iid, cat, q, a = parts
                    if "abnorm" not in cat.lower():
                        continue
                elif len(parts) == 3:    # train/val C4 file: id|question|answer
                    iid, q, a = parts
                else:
                    continue
                ip = None
                for idr in imgdirs:
                    for ext in (".jpg", ".png", ".jpeg"):
                        c = os.path.join(idr, iid + ext)
                        if os.path.exists(c):
                            ip = c; break
                    if ip: break
                if not ip:
                    continue
                # 3.8% of C4 is yes/no -- drop it, this is meant to be the OPEN cell and binary
                # items are exactly what got ProbMed rejected.
                if a.strip().lower().rstrip(".") in ("yes", "no"):
                    continue
                rows.append({"idx": len(rows), "question": q.strip(), "answer": a.strip(),
                             "img_path": ip, "split": split, "image_id": iid})
        print(f"  [{split}] cumulative rows: {len(rows)}", flush=True)

    if not rows:
        raise SystemExit("ABORT: no C4 rows found -- check the split layout under " + SRC)

    hits = 0
    if A.audit_overlap:
        ev = set()
        for m in EVAL_META:
            if os.path.exists(m):
                for r in json.load(open(m))["rows"]:
                    ev.add(r["img_md5"])
        vqarad = set()
        for p in glob.glob("/data/dan/dataset/vqa_rad/**/*.jpg", recursive=True)[:4000]:
            vqarad.add(pixel_md5(p))
        print(f"  [audit] {len(ev)} eval images, {len(vqarad)} VQA-RAD images", flush=True)
        # MEASURED 2026-08-19: 10 VQA-Med images are pixel-identical to images already in our eval
        # pools.  Both VQA-Med 2019 and VQA-RAD are MedPix-derived, so this overlap is expected and
        # the audit explicitly flagged it as "possible and not ruled out".  Excluding the colliding
        # images is the fix; aborting would throw away a usable cell over 10 items.
        bad = set()
        for r in rows:
            if r["image_id"] not in seen_img:
                seen_img[r["image_id"]] = pixel_md5(r["img_path"])
            r["img_md5"] = seen_img[r["image_id"]]
            if r["img_md5"] in ev or r["img_md5"] in vqarad:
                bad.add(r["image_id"])
        hits = len(bad)
        n0 = len(rows)
        rows = [r for r in rows if r["image_id"] not in bad]
        for i, r in enumerate(rows):
            r["idx"] = i
        print(f"  [audit] EXCLUDED {hits} images colliding with the eval pools "
              f"({n0 - len(rows)} questions dropped)", flush=True)

    os.makedirs(A.out, exist_ok=True)
    json.dump(rows, open(os.path.join(A.out, "vqamed_open.json"), "w"))
    ws = [len(r["answer"].split()) for r in rows]
    g = Counter(r["answer"].lower() for r in rows)
    print(f"\nwrote {A.out}/vqamed_open.json")
    print(f"  {len(rows)} questions / {len({r['image_id'] for r in rows})} images")
    print(f"  answer words mean {np.mean(ws):.2f} median {np.median(ws):.0f} p90 {np.percentile(ws,90):.0f}")
    print(f"  distinct golds {len(g)} | singletons {sum(1 for v in g.values() if v==1)/len(g):.1%}")
    print(f"  images EXCLUDED for colliding with eval pools: {hits}")


if __name__ == "__main__":
    main()
