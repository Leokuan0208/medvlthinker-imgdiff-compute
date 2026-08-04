#!/usr/bin/env python3
"""align_build_manifest.py -- build the (image, question, 8 candidates, judge labels) manifest for the
CONTRASTIVE IMAGE-TEXT ALIGNMENT verifier experiment, for both the 3 EVAL sets (the 2345 items the
incumbent verifier is reported on) and the 3 image-DISJOINT TRAIN pools.

Images are decoded once, saved as PNG under data/align_cache/img/<md5>.png, and keyed by the md5 of the
DECODED RGB PIXELS (the project's disjointness protocol).  The manifest therefore also *proves* the
train/eval image disjointness by construction -- align_fit.py asserts the intersection is empty.

  python3 src/verifier_arch/align_build_manifest.py
  -> data/align_cache/manifest.json
  -> data/align_cache/img/<md5>.png
Run from the repo root.
"""
import os, io, json, glob, hashlib
from collections import defaultdict
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
CK = J("ckpts/openvqa/cheap_lingshu7b")
TAG = "lingshu7b"
OUT = J("data/align_cache")
IMGDIR = os.path.join(OUT, "img")
os.makedirs(IMGDIR, exist_ok=True)

EVAL = ["slake_open", "vqa_rad_open", "pathvqa_open"]
TRAIN = ["slake_open_train", "vqa_rad_open_train", "pathvqa_open_train"]


def loadj(p):
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


def norm(s):
    return str(s).strip().lower()


def pixhash(im):
    im = im.convert("RGB")
    return hashlib.md5(im.tobytes() + repr(im.size).encode()).hexdigest()


def save_img(im):
    h = pixhash(im)
    p = os.path.join(IMGDIR, h + ".png")
    if not os.path.exists(p):
        im.convert("RGB").save(p)
    return h


# ---------------- image loaders (verbatim logic from verifier_transfer_eval.py /
#                  run_lora_verifier_disjoint.py so the item sets match exactly) ----------------
def slake_imgs(split):
    m = {}
    for x in json.load(open(f"/data/dan/dataset/slake/{split}.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                m[x["qid"]] = (x["question"], ip)
    return m


def parquet_imgs(base, split, want):
    """Only `want` indices are decoded -- decoding every row of the PathVQA train parquet exhausts RAM."""
    import pandas as pd
    m = {}
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/{split}-*.parquet"))], ignore_index=True)
    for i, r in df.iterrows():
        if int(i) not in want:
            continue
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    del df
    return m


def imgs_for(ds, want):
    if ds == "slake_open":       return slake_imgs("test")
    if ds == "slake_open_train": return slake_imgs("train")
    if ds == "vqa_rad_open":       return parquet_imgs("/data/dan/dataset/vqa_rad/data", "test", want)
    if ds == "vqa_rad_open_train": return parquet_imgs("/data/dan/dataset/vqa_rad/data", "train", want)
    if ds == "pathvqa_open":       return parquet_imgs("/data/dan/dataset/path_vqa/data", "test", want)
    if ds == "pathvqa_open_train": return parquet_imgs("/data/dan/dataset/path_vqa/data", "train", want)
    raise SystemExit(ds)


ALLOW = {}
for ds in TRAIN:
    p = J(f"data/disjoint_split/idx_{ds}.json")   # L1 image-disjoint allowlist
    ALLOW[ds] = set(json.load(open(p)))

# eval items must be EXACTLY the incumbent's 2345 -> take them from its own dump
EVAL_DUMP = {}
for short, ds in zip(["slake", "vqa_rad", "pathvqa"], EVAL):
    EVAL_DUMP[ds] = {r["idx"]: r for r in json.load(open(J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{short}_open_lingshu7b.json")))}

man = {"eval": [], "train": []}
for split, dss in (("eval", EVAL), ("train", TRAIN)):
    for ds in dss:
        sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
        want = set(EVAL_DUMP[ds].keys()) if split == "eval" else (set(sc.keys()) & ALLOW[ds])
        IMG = imgs_for(ds, want)
        exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
        jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
        aj = defaultdict(dict)
        for cid, r in exp.items():
            if cid in jud:
                oi = cid.split("#")[0]; oi = int(oi) if oi.lstrip("-").isdigit() else oi
                aj[oi][norm(r["modal_pred"])] = jud[cid]
        n = 0
        for i in sc:
            if i not in IMG:
                continue
            if split == "eval" and i not in EVAL_DUMP[ds]:
                continue
            if split == "train" and (i not in ALLOW[ds] or i not in aj):
                continue
            q, img = IMG[i]
            if isinstance(img, str):
                img = Image.open(img)
            h = save_img(img)
            preds = sc[i]["preds"]
            if split == "eval":
                sl = EVAL_DUMP[ds][i]["sl"]          # the incumbent's own labels, verbatim
                inc = EVAL_DUMP[ds][i]["scores"]
                greedy_ok = EVAL_DUMP[ds][i]["greedy_ok"]
            else:
                sl = [aj[i].get(norm(a), -1) for a in preds]
                if all(x < 0 for x in sl):
                    continue
                inc = None
                greedy_ok = int(aj[i].get(norm(sc[i]["modal_pred"]), 0))
            man[split].append({"ds": ds, "idx": i, "q": q, "img": h, "preds": preds, "sl": sl,
                               "incumbent": inc, "greedy_ok": greedy_ok,
                               "modal_pred": sc[i]["modal_pred"],
                               "self_consistency": sc[i].get("self_consistency"),
                               "n_distinct": sc[i].get("n_distinct")})
            n += 1
        print(f"  {split:5s} {ds:22s} items={n} (images on disk {len(IMG)})", flush=True)

ev_h = {r["img"] for r in man["eval"]}
tr_h = {r["img"] for r in man["train"]}
inter = ev_h & tr_h
print(f"eval items={len(man['eval'])}  train items={len(man['train'])}")
print(f"distinct eval images={len(ev_h)}  train images={len(tr_h)}  PIXEL-HASH INTERSECTION={len(inter)}")
assert len(inter) == 0, f"CONTAMINATION: {len(inter)} shared images"
json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"))
print("wrote", os.path.join(OUT, "manifest.json"))
