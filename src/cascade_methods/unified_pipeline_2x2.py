#!/usr/bin/env python3
"""unified_pipeline_2x2.py -- ATTACK 2: decompose the option-branch delta into rescues and breaks.

delta_vs_7b = P(7B wrong AND verifier right)  -  P(7B right AND verifier wrong)
              \_____ RESCUES _____/             \_____ BREAKS _____/

and, because a cascade only cares about errors the strong model would actually fix, the same 2x2 is
also reported restricted to the RECOVERABLE items (32B-direct right AND 7B wrong).

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_2x2.py --tag zeroshot
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402


def run(tag):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    work = U.build_worklist()
    out = {"tag": tag, "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        keep = [j for j in range(len(rows))
                if sc.get(rows[j]["i"]) is not None
                and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
        if not keep:
            out["cells"][cell] = {"status": "not measured"}
            continue
        idx = [rows[j]["i"] for j in keep]
        okv = np.array([int(int(np.argmax([sc[rows[j]["i"]][c]
                                           for c in range(len(rows[j]["cands"]))])) == rows[j]["gold"])
                        for j in keep], float)
        ok7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        ok32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        n = len(keep)
        rescue = float(((ok7 == 0) & (okv == 1)).sum())
        brk = float(((ok7 == 1) & (okv == 0)).sum())
        rec = (ok32 == 1) & (ok7 == 0)
        out["cells"][cell] = {
            "n": n,
            "acc_verifier_over_options": float(okv.mean()),
            "acc_7b_greedy": float(ok7.mean()),
            "rescues_n": int(rescue), "rescues_rate": rescue / n,
            "breaks_n": int(brk), "breaks_rate": brk / n,
            "delta_vs_7b_check": (rescue - brk) / n,
            "rescue_per_break": (rescue / brk) if brk else None,
            "P_verifier_right_given_7b_wrong": float(okv[ok7 == 0].mean()) if (ok7 == 0).any() else None,
            "P_verifier_wrong_given_7b_right": float(1 - okv[ok7 == 1].mean()) if (ok7 == 1).any() else None,
            "recoverable_stratum": {
                "definition": "32B-direct right AND 7B greedy wrong -- the only errors a cascade can win",
                "n": int(rec.sum()),
                "share_of_cell": float(rec.mean()),
                "P_verifier_right_there": float(okv[rec].mean()) if rec.any() else None,
                "note": "compare with 1/K: on a recoverable item the verifier is choosing among the "
                        "same complete option set, so anything above 1/K is signal and anything below "
                        "means the verifier is ANTI-correlated with recoverability"},
            "one_over_K": float(np.mean([1.0 / len(rows[j]["cands"]) for j in keep])),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag)
    p = os.path.join(U.PARTS, f"rescue_break_2x2_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
