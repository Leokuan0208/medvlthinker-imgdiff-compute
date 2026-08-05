#!/usr/bin/env python3
"""realpairwise_assert_disjoint.py -- INDEPENDENT re-assertion (protocol rule 2).

Does NOT trust verifier_disjoint_split.json, data/verifarch/eval_imghash.json, or
genframe_data.assert_disjoint(). Recomputes md5 of DECODED RGB PIXELS from the raw sources
for (a) every image behind the 2345 eval questions this round reports on, and (b) every
image in the L1 allowlists that the CLEAN adapter ckpts/train/lora_verifier_disjoint could
have drawn its 10,364 training examples from (a SUPERSET of what it actually saw, so a
clean intersection here is conservative), and intersects them.

  python3 src/training_methods/realpairwise_assert_disjoint.py
"""
import os, sys, json, glob, io, hashlib
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.training_methods.realpairwise_clean_gpu import imgs_for  # noqa: E402


def pix_md5(img):
    im = Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
    return hashlib.md5(im.tobytes()).hexdigest()


def parquet_pool(base, split, want):
    import pandas as pd
    out = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))],
                   ignore_index=True)
    for i, r in df.iterrows():
        if int(i) not in want:
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            out[int(i)] = pix_md5(Image.open(io.BytesIO(img["bytes"])))
    return out


def main():
    items = G.load_items()
    # ---- eval side: only the images actually behind the 2345 reported questions
    eval_h, per_ds = set(), {}
    for ds in G.EVAL_DS:
        m = imgs_for(ds)
        want = [it["idx"] for it in items if it["ds"] == ds]
        hs = set()
        for k in want:
            q, img = m[k]
            hs.add(pix_md5(img))
        per_ds[ds] = {"n_items": len(want), "n_images": len(hs)}
        eval_h |= hs
        print(f"[eval] {ds:14s} items={len(want):5d} distinct images={len(hs)}", flush=True)

    # ---- train side: the L1 allowlists of every source the clean adapter drew from
    ALLOW = {p.split("idx_")[-1][:-5]: set(json.load(open(p)))
             for p in sorted(glob.glob(os.path.join(ROOT, "data/disjoint_split/idx_*.json")))}
    train_h, per_src = set(), {}
    for src, ids in ALLOW.items():
        if src == "slake_open_train":
            hs = set()
            for x in json.load(open("/data/dan/dataset/slake/train.json")):
                if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en" and x["qid"] in ids:
                    ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                    if os.path.exists(ip):
                        hs.add(pix_md5(ip))
        elif src in ("vqa_rad_open_train", "pathvqa_open_train"):
            base = ("/data/dan/dataset/vqa_rad/data" if src.startswith("vqa_rad")
                    else "/data/dan/dataset/path_vqa/data")
            hs = set(parquet_pool(base, "train", ids).values())
        else:
            jp = ("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json" if src == "kvasir_open"
                  else "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")
            hs = set(pix_md5(r["img_path"]) for r in json.load(open(jp))
                     if r["idx"] in ids and os.path.exists(r["img_path"]))
        per_src[src] = len(hs)
        train_h |= hs
        print(f"[train] {src:22s} allow={len(ids):6d} distinct images={len(hs)}", flush=True)

    inter = eval_h & train_h
    out = {"method": "md5 of decoded RGB pixel bytes, recomputed in this script from the raw sources",
           "adapter_under_test": "ckpts/train/lora_verifier_disjoint",
           "eval_per_ds": per_ds, "eval_distinct_images": len(eval_h),
           "train_per_source_distinct_images": per_src, "train_distinct_images": len(train_h),
           "train_pool_is_superset_of_what_the_adapter_saw": True,
           "image_pixel_md5_intersection": len(inter),
           "pass": len(inter) == 0}
    print(json.dumps({k: v for k, v in out.items() if k != "eval_per_ds"}, indent=1))
    p = os.path.join(ROOT, "results/cascade_methods/artifacts/realpairwise_disjointness_2026-08-05.json")
    json.dump(out, open(p, "w"), indent=1)
    print("->", p)


if __name__ == "__main__":
    main()
