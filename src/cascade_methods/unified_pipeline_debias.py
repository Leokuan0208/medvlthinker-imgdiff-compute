#!/usr/bin/env python3
"""unified_pipeline_debias.py -- ATTACK 2 arm A' (DIAGNOSTIC ONLY, see
results/cascade_methods/artifacts/unified_pipeline_2026-08-12_amendment1.json).

Arm A showed the zero-shot open-text verifier ranks the GIVEN OPTIONS well above its own random-gold
luck floor (AUROC 0.583-0.800) yet still loses to the 7B's own greedy argmax, because it carries a
CANDIDATE-IDENTITY bias (it says the string 'no' is correct far more readily than 'yes'; it scores
all four PMC options below 0.5 on 79.9% of items).

This arm removes that bias with NESTED 5-fold CV -- the offset for an identity is estimated only on
the folds that do not contain the item being scored -- and asks how much of the arm-A shortfall it
recovers.  It is NOT a deployable pipeline (the offsets are fit on eval-distribution items) and must
never be quoted as one.  Its job is mechanism attribution: candidate-prior shortfall (arm B is the
right fix) vs ranking shortfall (arm B will fail too).

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_debias.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

NFOLD = 5
SEED_CV = 20260812


def identity_of(cell, rows, j, c):
    """Candidate identity used for the offset: the STRING for 2-option yes/no cells, the SLOT index
    for the lettered cells (whose option bodies are all distinct)."""
    r = rows[j]
    if len(r["cands"]) == 2:
        return str(r["cands"][c]).strip().lower()
    return f"slot{c}"


def run(tag="zeroshot"):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    work = U.build_worklist()
    out = {"tag": tag, "nfold": NFOLD, "seed_cv": SEED_CV, "diagnostic_only": True, "cells": {}}
    for cell in U.OPTION_CELLS:
        rows = work[cell]
        sc = U.load_scores(cell, tag)
        keep = [j for j in range(len(rows))
                if sc.get(rows[j]["i"]) is not None and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
        if not keep:
            out["cells"][cell] = {"status": "not measured"}
            continue
        n = len(keep)
        rng = np.random.default_rng(SEED_CV)
        fold = rng.permutation(n) % NFOLD
        S = [np.array([sc[rows[j]["i"]][c] for c in range(len(rows[j]["cands"]))], float) for j in keep]
        ID = [[identity_of(cell, rows, j, c) for c in range(len(rows[j]["cands"]))] for j in keep]
        gold = np.array([rows[j]["gold"] for j in keep], int)
        ok_raw = np.array([int(np.argmax(S[a]) == gold[a]) for a in range(n)], float)
        ok_deb = np.empty(n)
        pick_deb = np.empty(n, int)
        for f in range(NFOLD):
            tr = np.where(fold != f)[0]
            acc, cnt = {}, {}
            for a in tr:
                for c, idn in enumerate(ID[a]):
                    acc[idn] = acc.get(idn, 0.0) + S[a][c]
                    cnt[idn] = cnt.get(idn, 0) + 1
            off = {k: acc[k] / cnt[k] for k in acc}
            gm = float(np.mean([S[a].mean() for a in tr]))
            for a in np.where(fold == f)[0]:
                v = np.array([S[a][c] - off.get(ID[a][c], gm) for c in range(len(S[a]))], float)
                p = int(np.argmax(v))
                pick_deb[a] = p
                ok_deb[a] = int(p == gold[a])
        idx = [rows[j]["i"] for j in keep]
        ok7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
        ok32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
        # luck floor on the DEBIASED picks
        ks = np.array([len(rows[j]["cands"]) for j in keep])
        rngl = np.random.default_rng(U.SEED_LUCK)
        accs = np.empty(U.NLUCK)
        for b in range(U.NLUCK):
            g = (rngl.random(n) * ks).astype(int)
            accs[b] = float((g == pick_deb).mean())
        out["cells"][cell] = {
            "n": n,
            "acc_raw_argmax": float(ok_raw.mean()),
            "acc_debiased_argmax": float(ok_deb.mean()),
            "acc_7b_greedy": float(ok7.mean()),
            "acc_32b_direct": float(ok32.mean()),
            "delta_debiased_vs_raw": U.paired_boot(ok_deb, ok_raw),
            "delta_debiased_vs_7b": U.paired_boot(ok_deb, ok7),
            "delta_debiased_vs_32b_direct": U.paired_boot(ok_deb, ok32),
            "luck_floor_random_gold_debiased": {
                "analytic_1_over_K": float(np.mean(1.0 / ks)),
                "permutation_mean": float(accs.mean()),
                "permutation_p95": float(np.percentile(accs, 95))},
            "recovered_fraction_of_arm_A_shortfall_vs_7b": (
                float((ok_deb.mean() - ok_raw.mean()) / (ok7.mean() - ok_raw.mean()))
                if ok7.mean() > ok_raw.mean() else None),
        }
    return out


if __name__ == "__main__":
    r = run()
    p = os.path.join(U.PARTS, "arm_a_prime_debias.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
