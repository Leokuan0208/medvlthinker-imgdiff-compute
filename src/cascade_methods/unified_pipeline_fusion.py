#!/usr/bin/env python3
"""unified_pipeline_fusion.py -- ATTACK 2 arm A'' (amendment 4).

ONE scorer, ONE decision rule, the generator's own preference folded INTO the score:

    s'(c) = s_verifier(image, question, c) + lambda * 1[c == the 7B's own greedy answer]
    answer = argmax_c s'(c)

lambda = 0 is arm A (pure verifier over the given options); lambda -> infinity is always-7B.
lambda is chosen by NESTED 5-fold CV over 10 fold-split seeds, per cell AND globally, so a
fold-luck win cannot be mistaken for a real one and the macro is reported with ONE knob.

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline_fusion.py --tag zeroshot
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
NFOLD = 5
FOLD_SEEDS = list(range(10))


def cell_data(cell, tag):
    z = np.load(U.VEC_NPZ, allow_pickle=True)
    rows = U.build_worklist()[cell]
    sc = U.load_scores(cell, tag)
    keep = [j for j in range(len(rows))
            if sc.get(rows[j]["i"]) is not None and len(sc[rows[j]["i"]]) == len(rows[j]["cands"])]
    picks7, n5 = U.sevenb_pick(cell, [rows[j] for j in keep])
    S, G, P7 = [], [], []
    for j in keep:
        r = rows[j]
        S.append(np.array([sc[r["i"]][c] for c in range(len(r["cands"]))], float))
        G.append(int(r["gold"]))
        P7.append(picks7.get(r["i"]))
    idx = [rows[j]["i"] for j in keep]
    ok7 = np.array([z[f"{cell}|always_7b"][i] for i in idx], float)
    ok32 = np.array([z[f"{cell}|always_32b_direct"][i] for i in idx], float)
    return S, np.array(G), P7, ok7, ok32, n5


def ok_at(S, G, P7, lam):
    """Delivered correctness of argmax_c [ s(c) + lam*1[c == 7B greedy] ].  If the 7B's answer could
    not be parsed the bonus is simply absent -- the pipeline still answers (the verifier's argmax)."""
    out = np.empty(len(S))
    for a in range(len(S)):
        v = S[a].copy()
        if P7[a] is not None and 0 <= P7[a] < len(v):
            v[P7[a]] += lam
        out[a] = float(int(np.argmax(v)) == G[a])
    return out


def run(tag):
    out = {"tag": tag, "grid": GRID, "nfold": NFOLD, "fold_seeds": FOLD_SEEDS, "cells": {},
           "global_lambda": {}}
    data = {}
    for cell in U.OPTION_CELLS:
        S, G, P7, ok7, ok32, n5 = cell_data(cell, tag)
        if not S:
            out["cells"][cell] = {"status": "not measured"}
            continue
        data[cell] = (S, G, P7, ok7, ok32)
        OK = {lam: ok_at(S, G, P7, lam) for lam in GRID}
        # ---- per-cell cross-fit lambda -------------------------------------------------------
        seed_acc, seed_lams = [], []
        for fs in FOLD_SEEDS:
            rng = np.random.default_rng(20260812 + fs)
            fold = rng.permutation(len(S)) % NFOLD
            pred = np.empty(len(S))
            lams = []
            for f in range(NFOLD):
                tr, te = fold != f, fold == f
                best = max(GRID, key=lambda l: OK[l][tr].mean())
                lams.append(best)
                pred[te] = OK[best][te]
            seed_acc.append(float(pred.mean()))
            seed_lams.append(lams)
        seed_acc = np.array(seed_acc)
        out["cells"][cell] = {
            "n": len(S),
            "acc_by_lambda_EVAL_VISIBLE_not_a_result": {str(l): float(OK[l].mean()) for l in GRID},
            "acc_7b_greedy": float(ok7.mean()),
            "acc_32b_direct": float(ok32.mean()),
            "crossfit_percell": {
                "acc_seed_mean": float(seed_acc.mean()), "acc_seed_sd": float(seed_acc.std(ddof=1)),
                "acc_seed_min": float(seed_acc.min()), "acc_seed_max": float(seed_acc.max()),
                "lambda_chosen_counts": {str(l): int(sum(x.count(l) for x in seed_lams))
                                         for l in GRID if sum(x.count(l) for x in seed_lams)},
                "delta_vs_7b_at_seed_mean_lambda": None},
            "n5_dump_consistency": n5,
        }
        # paired CI at the modal cross-fit lambda (the deployable single value)
        cnt = {l: sum(x.count(l) for x in seed_lams) for l in GRID}
        modal = max(cnt, key=lambda l: cnt[l])
        out["cells"][cell]["crossfit_percell"]["modal_lambda"] = modal
        out["cells"][cell]["crossfit_percell"]["acc_at_modal_lambda"] = float(OK[modal].mean())
        out["cells"][cell]["crossfit_percell"]["delta_vs_7b_at_modal_lambda"] = U.paired_boot(
            OK[modal], ok7)
        out["cells"][cell]["crossfit_percell"]["delta_vs_32b_direct_at_modal_lambda"] = U.paired_boot(
            OK[modal], ok32)
        out["cells"][cell]["OK_cache_note"] = ("acc_by_lambda is EVAL-VISIBLE and is printed only to "
                                               "show the shape of the curve; the reported number is "
                                               "the cross-fit one")
        data[cell] = (S, G, P7, ok7, ok32, OK)

    # ---- ONE GLOBAL lambda, cross-fit jointly over the four cells -----------------------------
    cells = [c for c in U.OPTION_CELLS if c in data]
    if cells:
        seed_macro, seed_lams = [], []
        for fs in FOLD_SEEDS:
            folds = {}
            for c in cells:
                rng = np.random.default_rng(20260812 + fs + hash(c) % 1000)
                folds[c] = rng.permutation(len(data[c][0])) % NFOLD
            per_cell_acc = {c: np.empty(len(data[c][0])) for c in cells}
            lams = []
            for f in range(NFOLD):
                # choose ONE lambda on the training folds of ALL cells (equal weight per cell)
                def score(l):
                    return float(np.mean([data[c][5][l][folds[c] != f].mean() for c in cells]))
                best = max(GRID, key=score)
                lams.append(best)
                for c in cells:
                    te = folds[c] == f
                    per_cell_acc[c][te] = data[c][5][best][te]
            seed_macro.append(float(np.mean([per_cell_acc[c].mean() for c in cells])))
            seed_lams += lams
        sm = np.array(seed_macro)
        cnt = {l: seed_lams.count(l) for l in GRID if seed_lams.count(l)}
        modal = max(cnt, key=lambda l: cnt[l])
        out["global_lambda"] = {
            "cells": cells,
            "four_cell_macro_seed_mean": float(sm.mean()), "sd": float(sm.std(ddof=1)),
            "min": float(sm.min()), "max": float(sm.max()),
            "lambda_chosen_counts": cnt, "modal_lambda": modal,
            "four_cell_macro_always_7b": float(np.mean([data[c][3].mean() for c in cells])),
            "four_cell_macro_always_32b_direct": float(np.mean([data[c][4].mean() for c in cells])),
            "four_cell_macro_at_modal_lambda": float(np.mean([data[c][5][modal].mean() for c in cells])),
            "per_cell_at_modal_lambda": {c: float(data[c][5][modal].mean()) for c in cells},
            "per_cell_delta_vs_7b_at_modal_lambda": {
                c: U.paired_boot(data[c][5][modal], data[c][3]) for c in cells},
            "macro_vs_7b_at_modal_lambda": U.macro_boot(
                {c: data[c][5][modal] for c in cells}, {c: data[c][3] for c in cells}),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    r = run(a.tag)
    p = os.path.join(U.PARTS, f"fusion_{a.tag}.json")
    json.dump(r, open(p, "w"), indent=1)
    print(json.dumps(r, indent=1))
    print("wrote", p)
