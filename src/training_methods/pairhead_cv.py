#!/usr/bin/env python3
"""pairhead_cv.py -- STAGE 1: pre-register the pairwise-contrast-head configuration by
image-grouped cross-validation INSIDE THE DISJOINT TRAIN POOL.  Eval is never touched here
(protocol rule 3): this script does not import a single eval label.

  python3 src/training_methods/pairhead_cv.py --out data/verifarch/pairhead_cv.json
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G
import pairhead_lib as P

ROOT = G.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/verifarch/pairhead_cv.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv_seeds", type=int, default=3)
    A = ap.parse_args()
    t0 = time.time()

    dis = P.independent_disjointness()
    print("DISJOINT:", json.dumps(dis), flush=True)

    tr = G.load_candidates("train", mode="generator", layers=[7, 14, 21, 28],
                           pooling=("last", "span"), order="concat")
    pos, neg, pf = P.train_pairs(tr)
    print(f"train rows {len(tr.rows)}  questions {len(tr.questions)}  pairs {len(pos)}", flush=True)
    print("pairs per fold:", np.bincount(pf).tolist(), flush=True)

    cache = {}

    def X(layers, pooling):
        k = (tuple(layers), pooling)
        if k not in cache:
            cache.clear()          # only ever keep one materialized representation (RAM)
            cache[k] = P.base_matrix(tr, layers, pooling)
        return cache[k]

    log = []

    def run(cfg, tag, aggs=("copeland_borda",)):
        Xnp = X(cfg["layers"], cfg["pooling"])
        accs = {a: [] for a in aggs}
        for s in range(A.cv_seeds):
            r = P.cv_sel_eff(tr, Xnp, cfg, pf, A.folds, seed=s, aggs=aggs)
            for a in aggs:
                accs[a].append(r[a])
        rec = {**cfg, "tag": tag, "d": int(Xnp.shape[1]),
               "cv_sel_eff": {a: float(np.mean(v)) for a, v in accs.items()},
               "cv_sel_eff_per_seed": {a: [float(x) for x in v] for a, v in accs.items()},
               "cv_seeds": A.cv_seeds}
        rec["cv_best"] = float(max(rec["cv_sel_eff"].values()))
        log.append(rec)
        print(f"  CV[{tag}] {cfg['layers']}/{cfg['pooling']}/{cfg['inp']}/{cfg['antisym']}/"
              f"h{cfg['hidden']}/wd{cfg['wd']}/ep{cfg['epochs']} -> "
              + " ".join(f"{a}={rec['cv_sel_eff'][a]:.4f}" for a in aggs)
              + f"   [{time.time()-t0:.0f}s]", flush=True)
        return rec

    BASE = {"inp": "full", "antisym": "arch", "hidden": 256, "wd": 1e-2, "epochs": 30,
            "lr": 1e-3, "bs": 256}

    # ---- stage A: representation (layer x pooling) at a fixed reference architecture
    print("\n== stage A: representation ==", flush=True)
    sA = []
    for L in [7, 14, 21, 28]:
        for pool in ["last", "span", "both"]:
            sA.append(run({**BASE, "layers": [L], "pooling": pool}, "A"))
    bestA = max(sA, key=lambda r: r["cv_best"])
    # stage A2: multi-layer stacks at the winning pooling
    for LS in [[14, 21], [21, 28], [7, 14, 21, 28]]:
        sA.append(run({**BASE, "layers": LS, "pooling": bestA["pooling"]}, "A2"))
    bestA = max(sA, key=lambda r: r["cv_best"])
    REP = {"layers": bestA["layers"], "pooling": bestA["pooling"]}
    print("stage A winner:", REP, bestA["cv_best"], flush=True)

    # ---- stage B: pair encoding x antisymmetry  (the load-bearing design question)
    print("\n== stage B: pair encoding x antisymmetry ==", flush=True)
    sB = []
    for inp in ["concat", "diff", "full"]:
        for anti in ["arch", "augment"]:
            if inp == BASE["inp"] and anti == BASE["antisym"]:
                sB.append({**bestA, "tag": "B(=A winner)"})
                continue
            sB.append(run({**BASE, **REP, "inp": inp, "antisym": anti}, "B"))
    # degeneracy control: linear-on-difference IS a pointwise scorer (w.h_i - w.h_j)
    run({**BASE, **REP, "inp": "diff", "antisym": "arch", "hidden": 0}, "B_degeneracy_linear_diff")
    bestB = max(sB, key=lambda r: r["cv_best"])
    ENC = {"inp": bestB["inp"], "antisym": bestB["antisym"]}
    print("stage B winner:", ENC, bestB["cv_best"], flush=True)

    # ---- stage C: capacity x regularisation x epochs
    print("\n== stage C: capacity x regularisation x epochs ==", flush=True)
    sC = [bestB]
    for h in [256, 512]:
        for wd in [1e-2, 1e-1]:
            for ep in [30, 60]:
                if (h, wd, ep) == (BASE["hidden"], BASE["wd"], BASE["epochs"]):
                    continue
                sC.append(run({**BASE, **REP, **ENC, "hidden": h, "wd": wd, "epochs": ep}, "C"))
    bestC = max(sC, key=lambda r: r["cv_best"])
    CFG = {k: bestC[k] for k in ["layers", "pooling", "inp", "antisym", "hidden", "wd",
                                 "epochs", "lr", "bs"]}
    print("stage C winner:", CFG, bestC["cv_best"], flush=True)

    # ---- stage D: aggregation rule, at the pre-registered config
    print("\n== stage D: aggregation ==", flush=True)
    rD = run(CFG, "D_aggregation", aggs=tuple(P.AGGS))
    AGG = max(P.AGGS, key=lambda a: rD["cv_sel_eff"][a])
    print("stage D winner:", AGG, rD["cv_sel_eff"], flush=True)

    art = {"what": "PRE-REGISTRATION of the pairwise contrast head. Image-grouped 5-fold CV inside "
                   "the disjoint TRAIN pool ONLY; no eval label is read by this script.",
           "date": "2026-08-05", "disjointness_independent": dis,
           "train": {"rows": len(tr.rows), "questions": len(tr.questions), "pairs": int(len(pos)),
                     "pairs_per_fold": np.bincount(pf).tolist(),
                     "questions_with_both_labels": int(len(set(
                         (q.ds, q.idx) for q in tr.questions
                         if any(c.y == 1 for c in q.cands) and any(c.y == 0 for c in q.cands))))},
           "cv_protocol": f"{A.folds} folds by md5(decoded-RGB image md5) % {A.folds}; a pair never "
                          f"straddles a fold; criterion = within-question selection efficiency on the "
                          f"held-out fold, averaged over {A.cv_seeds} training seeds. Stage A picks the "
                          f"representation, B the pair encoding x antisymmetry, C capacity/reg/epochs, "
                          f"D the aggregation rule.",
           "grid": log,
           "PREREGISTERED": {"config": CFG, "aggregation": AGG,
                             "cv_sel_eff": rD["cv_sel_eff"][AGG]},
           "minutes": round((time.time() - t0) / 60, 1)}
    op = os.path.join(ROOT, A.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(art, open(op, "w"), indent=1)
    print(f"\nwrote {op}   ({art['minutes']} min)", flush=True)


if __name__ == "__main__":
    main()
