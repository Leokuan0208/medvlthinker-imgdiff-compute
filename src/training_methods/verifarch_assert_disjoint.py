#!/usr/bin/env python3
"""verifarch_assert_disjoint.py -- RE-PROVE, from scratch, that the training pools used by the
feature-based discriminative selector share no IMAGE and no ITEM with the three eval sets, and emit the
per-eval-item image pixel hash so cross-fitted folds can be grouped by image.

This re-derives (rather than trusts) the disjointness of
  data/disjoint_split/idx_{slake_open_train,vqa_rad_open_train,pathvqa_open_train,kvasir_open,radimagenet_open}.json
against the eval sets {slake_open, vqa_rad_open, pathvqa_open} exactly as
src/training_methods/build_disjoint_verifier_split.py does: md5 of DECODED RGB pixels (WxH + raw bytes),
which catches re-encoded copies.

  python3 src/training_methods/verifarch_assert_disjoint.py
  -> results/cascade_methods/artifacts/verifarch_disjoint_assert.json   (summary, committed)
  -> data/verifarch/eval_imghash.json                                   (eval idx -> pixel hash, for GroupKFold)
"""
import glob, hashlib, io, json, os, re, string
from collections import defaultdict

from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
SLAKE = "/data/dan/dataset/slake"
PATHVQA = "/data/dan/dataset/path_vqa/data"
VQARAD = "/data/dan/dataset/vqa_rad/data"
CK = J("ckpts/openvqa/cheap_lingshu7b")


def qnorm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


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
            out.append((x["qid"], x["question"], str(x["answer"]), ip))
    return out


def load_parquet(base, split):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))],
                   ignore_index=True)
    out = []
    for i, r in df.iterrows():
        q, a = r.get("question"), r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]
            q = conv[0]["value"].replace("<image>", "").strip()
            a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img):
            continue
        out.append((int(i), str(q), str(a).strip(), Image.open(io.BytesIO(img["bytes"]))))
    return out


def sc8_idx(ds, tag="lingshu7b"):
    p = f"{CK}/ckpt_{ds}_{tag}_sc8.jsonl"
    return [json.loads(l)["idx"] for l in open(p) if l.strip()]


print("[eval] hashing the three eval sets ...", flush=True)
raw_eval = {"slake_open": load_slake("test"),
            "vqa_rad_open": load_parquet(VQARAD, "test"),
            "pathvqa_open": load_parquet(PATHVQA, "test")}
EVAL_IMG, EVAL_ITEM = set(), set()
eval_hash = {}      # ds -> {idx: hash}
summary = {"eval": {}, "train": {}}
for ds, pool in raw_eval.items():
    want = set(sc8_idx(ds))
    got = [it for it in pool if it[0] in want]
    assert len(got) == len(want), f"{ds}: sc8 has {len(want)} idx but {len(got)} resolve"
    hs = {it[0]: pixhash(it[3]) for it in got}
    eval_hash[ds] = hs
    EVAL_IMG |= set(hs.values())
    fam = ds.replace("_open", "")
    EVAL_ITEM |= set((fam, qnorm(it[1]), hs[it[0]]) for it in got)
    summary["eval"][ds] = {"n_items": len(got), "n_images": len(set(hs.values()))}
    print(f"  {ds:14s} items={len(got):5d} images={len(set(hs.values())):5d}", flush=True)
print(f"  EVAL total images={len(EVAL_IMG)} items={len(EVAL_ITEM)}", flush=True)

print("\n[train] hashing the L1 training pools ...", flush=True)
train_pools = {"slake_open_train": load_slake("train"),
               "vqa_rad_open_train": load_parquet(VQARAD, "train"),
               "pathvqa_open_train": load_parquet(PATHVQA, "train")}
inter_img_total, inter_item_total = 0, 0
for name, pool in train_pools.items():
    allow = set(json.load(open(J(f"data/disjoint_split/idx_{name}.json"))))
    got = [it for it in pool if it[0] in allow]
    hs = {it[0]: pixhash(it[3]) for it in got}
    fam = name.replace("_open_train", "")
    items = set((fam, qnorm(it[1]), hs[it[0]]) for it in got)
    ii = set(hs.values()) & EVAL_IMG
    it_ = items & EVAL_ITEM
    inter_img_total += len(ii)
    inter_item_total += len(it_)
    summary["train"][name] = {"n_items": len(got), "n_images": len(set(hs.values())),
                              "image_pixel_hash_intersection_with_eval": len(ii),
                              "item_intersection_with_eval": len(it_)}
    print(f"  {name:20s} items={len(got):6d} images={len(set(hs.values())):5d} "
          f"img_inter={len(ii)} item_inter={len(it_)}", flush=True)

for name, jp in [("kvasir_open", "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
                 ("radimagenet_open", "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")]:
    allow = set(json.load(open(J(f"data/disjoint_split/idx_{name}.json"))))
    rows = [r for r in json.load(open(jp)) if r["idx"] in allow and os.path.exists(r["img_path"])]
    hs = {r["idx"]: pixhash(r["img_path"]) for r in rows}
    items = set((name, qnorm(r["question"]), hs[r["idx"]]) for r in rows)
    ii = set(hs.values()) & EVAL_IMG
    it_ = items & EVAL_ITEM
    inter_img_total += len(ii)
    inter_item_total += len(it_)
    summary["train"][name] = {"n_items": len(rows), "n_images": len(set(hs.values())),
                              "image_pixel_hash_intersection_with_eval": len(ii),
                              "item_intersection_with_eval": len(it_)}
    print(f"  {name:20s} items={len(rows):6d} images={len(set(hs.values())):5d} "
          f"img_inter={len(ii)} item_inter={len(it_)}", flush=True)

assert inter_img_total == 0, f"IMAGE LEAK: {inter_img_total} training images are eval images"
assert inter_item_total == 0, f"ITEM LEAK: {inter_item_total} training items are eval items"
summary["asserted"] = ["assert inter_img_total == 0", "assert inter_item_total == 0"]
summary["total_image_pixel_hash_intersection"] = inter_img_total
summary["total_item_intersection"] = inter_item_total
summary["method"] = ("md5 of decoded RGB pixels (WxH + raw bytes); item identity = "
                     "(dataset family, punctuation/whitespace-normalized question text, image pixel hash)")
os.makedirs(J("data/verifarch"), exist_ok=True)
json.dump({ds: {str(k): v for k, v in h.items()} for ds, h in eval_hash.items()},
          open(J("data/verifarch/eval_imghash.json"), "w"))
json.dump(summary, open(J("results/cascade_methods/artifacts/verifarch_disjoint_assert.json"), "w"), indent=1)
print("\nOK -- 0 shared images, 0 shared items. wrote data/verifarch/eval_imghash.json", flush=True)
