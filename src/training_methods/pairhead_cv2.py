#!/usr/bin/env python3
"""pairhead_cv2.py -- supplementary PRE-REGISTRATION pass, still TRAIN-ONLY.

pairhead_cv.py ranked stages A-C under a single aggregation rule (copeland_borda) and only swept
the aggregation at the end. Aggregation is free once a fold's model is fitted, so this pass takes
the top-K configurations of that grid and re-scores them under ALL aggregation rules, selecting
the (config, aggregation) pair jointly. No eval label is read here either -- the joint winner is
what fit_pair_head.py deploys as the headline.

  python3 src/training_methods/pairhead_cv2.py --topk 4
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G
import pairhead_lib as P

KEYS = ["layers", "pooling", "inp", "antisym", "hidden", "wd", "epochs", "lr", "bs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="data/verifarch/pairhead_cv.json")
    ap.add_argument("--out", default="data/verifarch/pairhead_cv.json")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cv_seeds", type=int, default=3)
    A = ap.parse_args()
    t0 = time.time()
    pre = json.load(open(os.path.join(G.ROOT, A.cv)))
    grid = [g for g in pre["grid"] if not g["tag"].startswith("B_degeneracy")]
    grid.sort(key=lambda r: -r["cv_best"])
    seen, top = set(), []
    for g in grid:
        k = tuple(str(g[x]) for x in KEYS)
        if k in seen:
            continue
        seen.add(k)
        top.append(g)
        if len(top) >= A.topk:
            break
    print("top-k configs:", [{x: t[x] for x in ["layers", "pooling", "inp", "antisym", "hidden", "wd", "epochs"]}
                             for t in top], flush=True)

    tr = G.load_candidates("train", mode="generator", layers=[7, 14, 21, 28],
                           pooling=("last", "span"), order="concat")
    pos, neg, pf = P.train_pairs(tr)
    rows = []
    cache = {}
    for t in top:
        cfg = {k: t[k] for k in KEYS}
        key = (tuple(cfg["layers"]), cfg["pooling"])
        if key not in cache:
            cache.clear()
            cache[key] = P.base_matrix(tr, cfg["layers"], cfg["pooling"])
        Xnp = cache[key]
        accs = {a: [] for a in P.AGGS}
        for s in range(A.cv_seeds):
            r = P.cv_sel_eff(tr, Xnp, cfg, pf, 5, seed=s, aggs=tuple(P.AGGS))
            for a in P.AGGS:
                accs[a].append(r[a])
        rec = {**cfg, "tag": "E_joint_agg",
               "cv_sel_eff": {a: float(np.mean(v)) for a, v in accs.items()},
               "cv_sel_eff_per_seed": {a: [float(x) for x in v] for a, v in accs.items()},
               "cv_seeds": A.cv_seeds}
        rec["cv_best"] = float(max(rec["cv_sel_eff"].values()))
        rec["cv_best_agg"] = max(P.AGGS, key=lambda a: rec["cv_sel_eff"][a])
        rows.append(rec)
        print(f"  CV[E] {cfg['layers']}/{cfg['pooling']}/{cfg['inp']}/{cfg['antisym']}/h{cfg['hidden']}"
              f"/wd{cfg['wd']}/ep{cfg['epochs']} -> "
              + " ".join(f"{a}={rec['cv_sel_eff'][a]:.4f}" for a in P.AGGS)
              + f"  [{time.time()-t0:.0f}s]", flush=True)

    best = max(rows, key=lambda r: r["cv_best"])
    CFG = {k: best[k] for k in KEYS}
    AGG = best["cv_best_agg"]
    pre["grid"] = pre["grid"] + rows
    pre["PREREGISTERED_stageD_only"] = pre["PREREGISTERED"]
    pre["PREREGISTERED"] = {"config": CFG, "aggregation": AGG, "cv_sel_eff": best["cv_best"],
                            "selected_by": "joint (config x aggregation) argmax of train-split CV "
                                           "selection efficiency over the top-%d configs of the "
                                           "staged grid, %d training seeds each" % (A.topk, A.cv_seeds)}
    pre["cv_protocol"] += (" A supplementary pass (pairhead_cv2.py) then re-scored the top-%d "
                           "configurations under ALL aggregation rules and pre-registered the joint "
                           "argmax, because stages A-C had ranked configurations under a single "
                           "aggregation rule. Still train-only." % A.topk)
    pre["minutes_cv2"] = round((time.time() - t0) / 60, 1)
    op = os.path.join(G.ROOT, A.out)
    json.dump(pre, open(op, "w"), indent=1)
    print("PREREGISTERED:", json.dumps(pre["PREREGISTERED"]), flush=True)
    print("wrote", op, flush=True)


if __name__ == "__main__":
    main()
