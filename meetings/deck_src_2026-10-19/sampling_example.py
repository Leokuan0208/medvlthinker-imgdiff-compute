#!/usr/bin/env python3
"""sampling_example.py — real samples for the temperature-sampling slide: SLAKE question 12231 (the deck's worked example)
at T = 0.2 / 0.4 / 0.7 / 1.0 (8 samples each, cap320 pools) plus the greedy answer and judge labels.
Writes sampling_example_12231.json next to this script."""
import json, os
from collections import Counter
CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
IDX = 12231
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampling_example_12231.json")


def find(path):
    for l in open(path):
        if l.strip():
            d = json.loads(l)
            if int(d.get("idx", -1)) == IDX:
                return d
    return None


def judged(path):
    """distinct-answer judge labels from the exploded judge file, keyed by lower-cased answer."""
    lab = {}
    if not os.path.exists(path):
        return lab
    for l in open(path):
        if l.strip():
            d = json.loads(l)
            if str(d.get("idx", "")).split("|")[0] == str(IDX):
                lab[str(d.get("pred", d.get("modal_pred", ""))).strip().lower()] = int(d["judge_ok"])
    return lab


res = {"cell": "slake_open", "idx": IDX, "source_dir": CK, "pools": {}}
g = find(f"{CK}/ckpt_slake_open_lingshu7b.jsonl")
res["greedy"] = {k: g.get(k) for k in ("pred", "gold", "question", "ok") if k in g} if g else None
for T, tag in (("0.2", "lingshu7bT02_sc8"), ("0.4", "lingshu7bT04_sc8"), ("0.7", "lingshu7b_sc8"), ("1.0", "lingshu7bT10_sc8")):
    p = f"{CK}/ckpt_slake_open_{tag}.jsonl"
    if not os.path.exists(p):
        res["pools"][T] = "missing: " + p
        continue
    d = find(p)
    preds = d.get("preds") or d.get("samples") or d.get("pred") if d else None
    res["pools"][T] = {"file": os.path.basename(p), "keys": sorted(d.keys()) if d else None, "samples": preds,
                       "counts": Counter(str(x).strip().lower() for x in preds).most_common() if isinstance(preds, list) else None,
                       "judge_by_answer": judged(p.replace(".jsonl", "_scexploded.judge.jsonl"))}
json.dump(res, open(OUT, "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)[:3000])
