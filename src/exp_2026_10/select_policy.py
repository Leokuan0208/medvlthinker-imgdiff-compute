#!/usr/bin/env python3
"""select_policy.py -- choose, PER BENCHMARK, which generation arm the final method uses (resolution x
temperature), honestly: cross-fitted over the two image folds of final_method_eval.py.

Each arm is a final_method_eval records file (per question: fold, pick/greedy currencies). The folds are
the image halves, and each fold's probe never saw that fold's images. For questions in fold f, a cell
uses the arm with the highest mean PICK judge accuracy on fold 1-f (ties -> the first arm listed, which
should be the incumbent). No question is scored with a choice made on itself.

Reported against a FIXED reference greedy (the first arm's greedy: pass the cap320 arm first, so the
reference is the standard cap320 greedy decode) and, separately, every fixed arm on its own.
Currencies: whichever are present in every arm (judge, em, f1, llava_recall, xjudge, ...).
Paired bootstrap, 10,000 resamples clustered by image within cell; macro = mean over cells.

  python3 select_policy.py --out <name> --arm cap320=<records.json> --arm fullres=<records.json> [...]
"""
import argparse, json, os
import numpy as np

ART = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
NBOOT, SEED = 10000, 20261018


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=path; first arm = incumbent + reference greedy")
    A = ap.parse_args()
    arms = [a.split("=", 1) for a in A.arm]
    R = {n: json.load(open(p)) for n, p in arms}
    names = [n for n, _ in arms]
    cells = [c for c in R[names[0]] if all(c in R[n] for n in names)]
    curs = [k for k in R[names[0]][cells[0]][0]["pick"]
            if all(k in R[n][cells[0]][0]["pick"] for n in names)]
    out = {"title": "Per-benchmark arm selection, cross-fitted over image folds", "no_fabricated_numbers": True,
           "arms": dict(arms), "reference_greedy": f"{names[0]} greedy", "selection_currency": "judge (pick accuracy)",
           "cells": {}, "macro": {}}
    parts = {}
    rng_ss = np.random.SeedSequence(SEED)
    for c in cells:
        by = {n: {(r["idx"] if not isinstance(r["idx"], list) else tuple(r["idx"])): r for r in R[n][c]} for n in names}
        qs = [q for q in by[names[0]] if all(q in by[n] for n in names)]
        choice = {}
        for f in (0, 1):
            other = [q for q in qs if by[names[0]][q]["fold"] == 1 - f]
            acc = {n: np.mean([by[n][q]["pick"]["judge"] for q in other]) if other else -1 for n in names}
            choice[f] = max(names, key=lambda n: (acc[n], -names.index(n)))
        ref = by[names[0]]
        sel = {q: by[choice[ref[q]["fold"]]][q] for q in qs}
        imgs = [ref[q]["img"] for q in qs]
        keys, inv = np.unique(imgs, return_inverse=True)
        W = np.random.default_rng(rng_ss.spawn(1)[0]).multinomial(
            len(keys), np.full(len(keys), 1 / len(keys)), size=NBOOT).astype(float)
        o = {"n": len(qs), "arm_chosen_for_fold": {str(k): v for k, v in choice.items()}}
        for cu in curs:
            g = np.array([ref[q]["greedy"][cu] if ref[q]["greedy"][cu] is not None else np.nan for q in qs], float)
            rows = {"selected": np.array([sel[q]["pick"][cu] if sel[q]["pick"][cu] is not None else np.nan for q in qs], float)}
            for n in names:
                rows[n] = np.array([by[n][q]["pick"][cu] if by[n][q]["pick"][cu] is not None else np.nan for q in qs], float)
            for k, v in rows.items():
                m = ~np.isnan(v) & ~np.isnan(g)
                d = np.where(m, v - g, 0.0)
                Rs = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / np.maximum(
                    W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
                key = f"{k}_minus_ref_greedy_{cu}"
                o[key] = float(d[m].mean()) if m.any() else None
                o["ci_" + key] = [float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))]
                parts.setdefault(key, []).append((o[key], Rs))
            # selected minus incumbent, paired
            v, b = rows["selected"], rows[names[0]]
            m = ~np.isnan(v) & ~np.isnan(b)
            d = np.where(m, v - b, 0.0)
            Rs = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / np.maximum(
                W @ np.bincount(inv, weights=m.astype(float), minlength=len(keys)), 1e-12)
            key = f"selected_minus_{names[0]}_{cu}"
            o[key] = float(d[m].mean()) if m.any() else None
            parts.setdefault(key, []).append((o[key], Rs))
        out["cells"][c] = o
        print(f"  {c:17} chosen {o['arm_chosen_for_fold']}  sel-ref judge {o['selected_minus_ref_greedy_judge']:+.4f}  "
              + "  ".join(f"{n} {o[f'{n}_minus_ref_greedy_judge']:+.4f}" for n in names), flush=True)
    for key, lst in parts.items():
        if len(lst) != len(cells) or any(p is None for p, _ in lst):
            continue
        pts = np.array([p for p, _ in lst]); Rm = np.stack([r for _, r in lst], 1).mean(1)
        lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
        out["macro"][key] = {"macro": float(pts.mean()), "ci": [lo, hi],
                             "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"), "n_positive": int((pts > 0).sum())}
    path = f"{ART}/select_policy_{A.out}.json"
    json.dump(out, open(path, "w"), indent=1)
    for k in sorted(out["macro"]):
        v = out["macro"][k]
        print(f"  {k:44} {v['macro']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {v['verdict']} {v['n_positive']}/{len(cells)}")
    print("wrote", path)


if __name__ == "__main__":
    main()
