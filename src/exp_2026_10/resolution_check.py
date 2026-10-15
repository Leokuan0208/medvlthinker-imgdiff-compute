#!/usr/bin/env python3
"""resolution_check.py -- does full resolution help the GENERATOR, before any probe is involved?

Per cell, judge currency (MedVLThinker-32B, run_judge.py default) unless noted:
  greedy  : fullres (lingshu7bFR) vs cap320 generated in the SAME session (lingshu7bC320) -- the matched
            control; also the stored cap320 greedy (lingshu7b) as a serving-drift check
  oracle@8: fullres N=8 pool vs the stored cap320 N=8 pool (no same-session N=8 control exists; the
            greedy drift check sizes how much serving drift alone can explain)
  distinct: mean distinct candidates per question
Paired bootstrap, 10,000 resamples clustered by image (questions sharing an image move together).
  python3 resolution_check.py <cell> [<cell> ...]     (writes artifacts/resolution_check_2026-10-15.json, merged)
"""
import glob, io, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
CK = ROOT + "/ckpts/openvqa/cheap_lingshu7b"
OUT = ROOT + "/results/cascade_methods/artifacts/resolution_check_2026-10-15.json"
JSONS = {"radimagenet_open": "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json",
         "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json",
         "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
         "vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
         "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json"}


def loadj(p):
    return {d["idx"]: d for d in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


def image_key(cell):
    """question idx -> an image identifier for clustering."""
    if cell in JSONS:
        return {r["idx"]: r["img_path"] for r in json.load(open(JSONS[cell]))}
    if cell == "slake_open":
        return {x["qid"]: x["img_name"] for x in json.load(open("/data/dan/dataset/slake/test.json"))}
    import pandas as pd, hashlib
    base = "/data/dan/dataset/vqa_rad/data" if cell == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/test-*.parquet"))], ignore_index=True)
    return {int(i): hashlib.md5(r["image"]["bytes"]).hexdigest() for i, r in df.iterrows()
            if isinstance(r["image"], dict)}


def judge_of_pool(cell, tag):
    exp = loadj(f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.jsonl")
    jud = loadj(f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl")
    per = defaultdict(list)
    for cid, r in exp.items():
        if cid in jud:
            q = cid.split("#")[0]; q = int(q) if q.lstrip("-").isdigit() else q
            per[q].append(int(jud[cid]["judge_ok"]))
    return {q: (max(v), len(v)) for q, v in per.items()}


def boot(diff, clusters, seed=20261015, n=10000):
    keys, inv = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)
    W = np.random.default_rng(seed).multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=n).astype(float)
    s = np.bincount(inv, weights=diff, minlength=len(keys)); c = np.bincount(inv, minlength=len(keys)).astype(float)
    R = (W @ s) / (W @ c)
    lo, hi = float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))
    return {"delta": float(diff.mean()), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}


res = json.load(open(OUT)) if os.path.exists(OUT) else {"title": "Does full resolution help the generator?",
                                                        "no_fabricated_numbers": True, "cells": {}}
for cell in sys.argv[1:]:
    img = image_key(cell)
    g = {t: {k: int(v["judge_ok"]) for k, v in loadj(f"{CK}/ckpt_{cell}_{t}.judge.jsonl").items()}
         for t in ("lingshu7bFR", "lingshu7bC320", "lingshu7b")}
    o = {}
    q = sorted(set(g["lingshu7bFR"]) & set(g["lingshu7bC320"]) & set(img), key=str)
    if q:
        d = np.array([g["lingshu7bFR"][i] - g["lingshu7bC320"][i] for i in q], float)
        o["greedy_fullres"] = float(np.mean([g["lingshu7bFR"][i] for i in q]))
        o["greedy_cap320_same_session"] = float(np.mean([g["lingshu7bC320"][i] for i in q]))
        o["greedy_fullres_minus_cap320"] = {**boot(d, [img[i] for i in q]), "n": len(q)}
        q2 = [i for i in q if i in g["lingshu7b"]]
        d2 = np.array([g["lingshu7bC320"][i] - g["lingshu7b"][i] for i in q2], float)
        o["drift_cap320_today_minus_stored"] = {**boot(d2, [img[i] for i in q2]), "n": len(q2)}
    pf, pc = judge_of_pool(cell, "lingshu7bFR"), judge_of_pool(cell, "lingshu7b")
    q3 = sorted(set(pf) & set(pc) & set(img), key=str)
    if q3:
        d3 = np.array([pf[i][0] - pc[i][0] for i in q3], float)
        o["oracle8_fullres"] = float(np.mean([pf[i][0] for i in q3]))
        o["oracle8_cap320_stored"] = float(np.mean([pc[i][0] for i in q3]))
        o["oracle8_fullres_minus_cap320"] = {**boot(d3, [img[i] for i in q3]), "n": len(q3)}
        o["distinct_fullres"] = float(np.mean([pf[i][1] for i in q3]))
        o["distinct_cap320"] = float(np.mean([pc[i][1] for i in q3]))
    res["cells"][cell] = o
    print(cell, json.dumps(o, indent=1))
json.dump(res, open(OUT, "w"), indent=1)
