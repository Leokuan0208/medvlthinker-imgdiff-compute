#!/usr/bin/env python3
"""Numbers the deck needs that no artifact holds yet, computed from the shipped method's per-question records
(/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json, judge currency unless noted):
  1. fixed / harmed vs greedy per benchmark (Hu 2026 accounting), and the 'unavoidable harm' (greedy right, no
     correct candidate in the pool)
  2. probe − majority vote, macro, with a paired bootstrap CI (10,000 resamples of images within each benchmark)
Writes extra_numbers.json; every value is reproducible from the records file."""
import json, random
from collections import defaultdict

REC = "/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/extra_numbers.json"
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
R = json.load(open(REC))
out = {"source": REC, "cells": {}}
for c in OPEN8:
    rs = R[c]
    n = len(rs)
    fixed = sum(1 for r in rs if r["greedy"]["judge"] == 0 and r["pick"]["judge"] == 1)
    harmed = sum(1 for r in rs if r["greedy"]["judge"] == 1 and r["pick"]["judge"] == 0)
    unavoid = sum(1 for r in rs if r["greedy"]["judge"] == 1 and r["oracle_judge"] == 0)
    recoverable = sum(1 for r in rs if r["greedy"]["judge"] == 0 and r["oracle_judge"] == 1)
    g_right = sum(1 for r in rs if r["greedy"]["judge"] == 1)
    out["cells"][c] = {"n": n, "fixed": fixed, "harmed": harmed, "net": fixed - harmed,
                       "harmed_unavoidable_no_correct_candidate": unavoid, "recoverable": recoverable,
                       "greedy_right": g_right, "fix_rate_of_recoverable": fixed / recoverable if recoverable else None,
                       "harm_rate_of_greedy_right": harmed / g_right if g_right else None}
tot = {k: sum(out["cells"][c][k] for c in OPEN8) for k in ["n", "fixed", "harmed", "net", "recoverable", "greedy_right",
                                                          "harmed_unavoidable_no_correct_candidate"]}
out["total"] = tot


def boot(cur):
    rng = random.Random(20261019)
    per = {}
    for c in OPEN8:
        by = defaultdict(list)
        for r in R[c]:
            by[r["img"]].append(r["pick"][cur] - r["sc"][cur])
        per[c] = list(by.values())
    point = sum(sum(sum(v) for v in per[c]) / sum(len(v) for v in per[c]) for c in OPEN8) / 8
    sims = []
    for _ in range(10000):
        m = 0
        for c in OPEN8:
            imgs = per[c]
            s = k = 0
            for _ in range(len(imgs)):
                v = imgs[rng.randrange(len(imgs))]
                s += sum(v); k += len(v)
            m += s / k
        sims.append(m / 8)
    sims.sort()
    return {"macro": point, "ci": [sims[249], sims[9749]]}


for cur in ["judge", "em"]:
    out[f"probe_minus_vote_{cur}"] = boot(cur)
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
