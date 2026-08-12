#!/usr/bin/env python3
"""unified_pipeline_textleak.py -- ATTACK 2: the QUESTION-TEXT leakage control for the TRAINED arms.

WHY THIS EXISTS.  The pre-registered disjointness criterion is pixel-md5 of DECODED RGB between every
verifier-training image and every scored eval image, and the trained arms satisfy it (33,079-hash ban
list applied before training; measured drops in the manifest).  But an image-disjoint split does NOT
make a yes/no cell text-disjoint: PathVQA and VQA-RAD reuse question WORDING across their own splits,
so a scorer trained on the train split can learn "this exact question text -> yes" and carry it to
eval without ever seeing the eval image.  This project has already caught a
"question-text -> answer prior" once (docs/current/VERIFIER_ARCHITECTURES_2026-08-04.md).

MEASURED HERE, per option cell:
  * how much of the eval cell has a question text that also occurs in the arm's training POOL, and
  * the arm's accuracy and its paired deltas SPLIT INTO the TEXT-SEEN and TEXT-UNSEEN strata.

The TEXT-UNSEEN stratum uses the WHOLE training pool as the "possibly seen" set, which is a strict
SUPERSET of the rows the arm actually drew.  So TEXT-UNSEEN is conservatively clean: an item there
cannot have had its question text in training under any draw.  TEXT-SEEN is correspondingly an upper
bound on the contaminated fraction.

The comparison is exactly paired on items and uses U.paired_boot (nboot=10000, seed 20260812).

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_textleak.py --tag optiononly_s0
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

QN = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())  # noqa: E731


def train_question_texts():
    """The question texts of the yes/no and lettered TRAIN pools the option arms draw from.
    Returns {cell: set(text)}; a cell absent from the dict has no in-domain option training pool."""
    import csv
    import pandas as pd
    out = {}

    # PathVQA train, yes/no rows -- the pool build_option_examples() draws pathvqa_closed_train from
    q = set()
    for f in sorted(glob.glob("/data/dan/dataset/path_vqa/data/train-*.parquet")):
        df = pd.read_parquet(f, columns=["question", "answer"])
        for _, r in df.iterrows():
            if str(r["answer"]).strip().lower() in ("yes", "no"):
                q.add(QN(r["question"]))
    out["PATH_VQA_closed"] = q

    # VQA-RAD train, yes/no rows
    from datasets import load_dataset
    q = set()
    for s in load_dataset("flaviagiammarino/vqa-rad", split="train"):
        if str(s["answer"]).strip().lower() in ("yes", "no"):
            q.add(QN(s["question"]))
    out["VQA_RAD_closed"] = q

    # PMC-VQA v1 train.csv -- the 4-option pool (v2 train_2.csv resolves 0 images, see the manifest)
    q = set()
    p = "/data/dan/dataset/pmc_vqa_train/train.csv"
    if os.path.exists(p):
        for r in list(csv.reader(open(p, encoding="utf-8")))[1:]:
            if len(r) >= 8:
                q.add(QN(r[1]))
    out["PMC_VQA"] = q

    # MedXpertQA: NO in-domain option training pool exists in this project -> empty set on purpose
    out["MedXpertQA-MM"] = set()
    return out


def run(tag, compare_tag="zeroshot"):
    work = U.build_worklist()
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    trq = train_question_texts()
    res = {"tag": tag, "date": U.DATE, "nboot": U.NBOOT, "seed": U.SEED_BOOT,
           "why": "an image-disjoint split is NOT a text-disjoint split on a yes/no cell; "
                  "TEXT-UNSEEN uses the WHOLE training pool as the possibly-seen set and is "
                  "therefore conservatively clean under any draw",
           "train_pool_distinct_question_texts": {c: len(v) for c, v in trq.items()},
           "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        if not sc:
            res["cells"][cell] = {"status": "not measured -- no scores on disk"}
            continue
        ok, _, cov = U.pick_vector(rows, sc)
        keep = [j for j in range(len(rows)) if cov[j]]
        if not keep:
            res["cells"][cell] = {"status": "not measured -- no covered items"}
            continue
        if len(keep) < len(rows):
            res["cells"][cell] = {"status": f"PARTIAL -- only {len(keep)} of {len(rows)} items have "
                                            "every candidate scored; not reported rather than "
                                            "reported on a biased subset",
                                  "n_covered": len(keep), "n_items": len(rows)}
            continue
        idx = [rows[j]["i"] for j in keep]
        a = np.array([ok[j] for j in keep], float)
        a7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        a32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        pool = trq.get(cell, set())
        seen = np.array([QN(rows[j]["question"]) in pool for j in keep])
        blk = {"n_scored": len(keep),
               "n_text_seen": int(seen.sum()),
               "frac_text_seen": float(seen.mean()),
               "train_pool_distinct_question_texts": len(pool),
               "strata": {}}
        # the comparison arm (zero-shot), on the same items, when it is on disk
        scc = U.load_scores(cell, compare_tag)
        ac = None
        if scc:
            okc, _, covc = U.pick_vector(rows, scc)
            if all(covc[j] for j in keep):
                ac = np.array([okc[j] for j in keep], float)
        for name, m in (("ALL", np.ones(len(keep), bool)),
                        ("TEXT_SEEN_upper_bound_on_contamination", seen),
                        ("TEXT_UNSEEN_conservatively_clean", ~seen)):
            if m.sum() == 0:
                blk["strata"][name] = {"n": 0, "status": "empty stratum"}
                continue
            d = {"n": int(m.sum()), "acc": float(a[m].mean()),
                 "always_7b": float(a7[m].mean()), "always_32b_direct": float(a32[m].mean()),
                 "vs_always_7b": U.paired_boot(a[m], a7[m]),
                 "vs_always_32b_direct": U.paired_boot(a[m], a32[m])}
            if ac is not None:
                d[f"acc_{compare_tag}_same_items"] = float(ac[m].mean())
                d[f"vs_{compare_tag}"] = U.paired_boot(a[m], ac[m])
            blk["strata"][name] = d
        res["cells"][cell] = blk
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="optiononly_s0")
    ap.add_argument("--compare_tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag, a.compare_tag)
    os.makedirs(U.PARTS, exist_ok=True)
    p = os.path.join(U.PARTS, f"textleak_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
