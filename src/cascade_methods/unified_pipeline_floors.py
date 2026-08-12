#!/usr/bin/env python3
"""unified_pipeline_floors.py -- ATTACK 2: the FLOORS that any option-branch number must clear.

This project has retracted a claim for mistaking coverage for signal, so the option branch is given
THREE floors, weakest to strongest, and every arm is reported against all three:

  F1  random-gold permutation      gold re-drawn uniformly among the item's own candidates with the
                                   scores held fixed  ->  exactly 1/K.  This is the floor that proves
                                   candidate COVERAGE carries no information (it cannot, the gold is
                                   in the set with probability 1).
  F2  best constant-identity rule  always answer the SAME candidate identity (the most frequent gold
                                   slot for the lettered cells, the most frequent gold string for the
                                   yes/no cells).  This is the floor a scorer that has learned only
                                   the dataset's ANSWER PRIOR would reach.  It is STRICTLY ABOVE F1
                                   wherever the gold is not uniform over identities -- and on PMC-VQA
                                   test_2 it is far above (gold slots are 13.3/36.9/37.3/12.6%).
  F3  label-permutation null       the scorer's per-item candidate scores are randomly permuted
                                   WITHIN the item, then argmax; equals F1 in expectation but is
                                   computed on the real score vectors so ties are handled the same
                                   way the deployed pick handles them.

Nothing here needs a GPU.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_floors.py --tag zeroshot
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

NPERM = 1000


def run(tag):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    work = U.build_worklist()
    out = {"tag": tag, "nperm": NPERM, "seed": U.SEED_LUCK, "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        keep = [j for j in range(len(rows))
                if sc.get(rows[j]["i"]) is not None
                and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
        if not keep:
            out["cells"][cell] = {"status": "not measured"}
            continue
        n = len(keep)
        S = [np.array([sc[rows[j]["i"]][c] for c in range(len(rows[j]["cands"]))], float) for j in keep]
        gold = np.array([rows[j]["gold"] for j in keep], int)
        ks = np.array([len(rows[j]["cands"]) for j in keep])
        two = ks[0] == 2
        ident = ([[str(c).strip().lower() for c in rows[j]["cands"]] for j in keep] if two
                 else [[f"slot{c}" for c in range(len(rows[j]["cands"]))] for j in keep])
        gid = [ident[a][gold[a]] for a in range(n)]
        cnt = Counter(gid)
        best_id, best_hits = cnt.most_common(1)[0]
        pick = np.array([int(np.argmax(S[a])) for a in range(n)])
        ok = (pick == gold).astype(float)
        idx = [rows[j]["i"] for j in keep]
        ok7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        ok32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        # F3: permute the score vector within each item
        rng = np.random.default_rng(U.SEED_LUCK)
        a3 = np.empty(NPERM)
        for b in range(NPERM):
            hits = 0
            for a in range(n):
                p = rng.permutation(len(S[a]))
                hits += int(int(np.argmax(S[a][p])) == gold[a])
            a3[b] = hits / n
        out["cells"][cell] = {
            "n": n,
            "acc_verifier_over_options": float(ok.mean()),
            "acc_7b_greedy": float(ok7.mean()),
            "acc_32b_direct": float(ok32.mean()),
            "F1_random_gold_1_over_K": float(np.mean(1.0 / ks)),
            "F2_best_constant_identity": {
                "identity": best_id, "acc": best_hits / n,
                "gold_identity_distribution": {k: v / n for k, v in sorted(cnt.items())}},
            "F3_within_item_score_permutation": {
                "mean": float(a3.mean()), "sd": float(a3.std(ddof=1)),
                "p95": float(np.percentile(a3, 95))},
            "margin_above_F2": float(ok.mean() - best_hits / n),
            "margin_of_7b_above_F2": float(ok7.mean() - best_hits / n),
            "clears_F2": bool(ok.mean() > best_hits / n),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag)
    p = os.path.join(U.PARTS, f"floors_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
