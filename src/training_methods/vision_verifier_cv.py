#!/usr/bin/env python3
"""vision_verifier_cv.py -- the PRE-REGISTERED arm selection, entirely inside the TRAIN pool.

5-fold IMAGE-GROUPED cross-validation (md5(img_md5) % 5), endpoint = within-question argmax hit
rate on the held-out fold, protocol identical to fit_hidden_head.cv().  Eval is never touched.
The arm with the best cv_sel_eff (averaged over --seeds) is the ONE arm carried to the primary
eval comparison; everything else is secondary/exploratory.

  OMP_NUM_THREADS=8 python3 -u src/training_methods/vision_verifier_cv.py --seeds 0 1 2
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402
import vision_verifier_fit as F  # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts/train_cv.json")


def xattn_cv(Htr, itr, bank, ytr, q, folds, seeds):
    from collections import defaultdict
    effs, aucs = [], []
    for f in sorted(set(folds.tolist())):
        tr, va = folds != f, folds == f
        Hs, mu, sd = V.zstd(Htr[tr]); Hv = (Htr[va] - mu) / sd
        b = bank.reshape(-1, bank.shape[-1])
        vmu, vsd = b.mean(0), b.std(0) + 1e-6
        bk = (bank - vmu) / vsd
        for s in seeds:
            m = F.fit_xattn(Hs, itr[tr], bk, ytr[tr], [q[i] for i in np.where(tr)[0]], seed=s)
            sv, _ = F.predict_xattn(m, Hv, itr[va], bk)
            aucs.append(G.auroc(ytr[va], sv))
            vidx = np.where(va)[0]
            loc = {i: j for j, i in enumerate(vidx)}
            byq = defaultdict(list)
            for i in vidx:
                byq[q[i]].append(i)
            hit = tot = 0
            for _, ii in byq.items():
                if ytr[ii].sum() == 0:
                    continue
                bi = ii[int(np.argmax([sv[loc[i]] for i in ii]))]
                hit += int(ytr[bi] == 1); tot += 1
            effs.append(hit / max(tot, 1))
    return {"cv_sel_eff": float(np.mean(effs)), "cv_auroc": float(np.mean(aucs)),
            "per_fold": [round(e, 6) for e in effs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["L", "Vmean", "L_Vmean", "L_prod", "L_simgrid", "L_maxsim",
                             "L_prod_sim", "xattn"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--grid_layer", type=int, default=21)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()
    torch.set_num_threads(A.threads)

    tr = G.load_candidates("train", layers=[A.layer], pooling=("span",))
    vtr = V.load_vision("train")
    Vm, Vg, itr = V.align(tr, vtr, A.layer, A.grid_layer)
    H = tr.matrix("span", A.layer)
    y = np.array([r["y"] for r in tr.rows], dtype=np.float32)
    q = V.qid(tr.rows)
    folds = V.img_folds(tr.rows, A.folds)
    print(f"[cv] train rows {H.shape}, {len(set(q))} questions, folds "
          f"{np.bincount(folds).tolist()}", flush=True)

    res = {}
    for arm in A.arms:
        t0 = time.time()
        if arm == "xattn":
            bank = vtr["v_grid"][:, vtr["grid_layers"].index(A.grid_layer)].astype(np.float32)
            r = xattn_cv(H, itr, bank, y, q, folds, A.seeds)
        else:
            X = V.build_features(arm, H, Vm, Vg)
            rs = [V.train_cv_sel_eff(X, y, q, folds, seed=s) for s in A.seeds]
            r = {"cv_sel_eff": float(np.mean([x["cv_sel_eff"] for x in rs])),
                 "cv_auroc": float(np.mean([x["cv_auroc"] for x in rs])),
                 "per_seed": [round(x["cv_sel_eff"], 6) for x in rs],
                 "d": int(X.shape[1])}
        r["minutes"] = round((time.time() - t0) / 60, 2)
        res[arm] = r
        print(f"  [{arm}] cv_sel_eff={r['cv_sel_eff']:.6f} cv_auroc={r['cv_auroc']:.6f} "
              f"({r['minutes']:.1f} min)", flush=True)

    best = max(res, key=lambda a: res[a]["cv_sel_eff"])
    best_vis = max((a for a in res if a not in ("L", "Vmean")),
                   key=lambda a: res[a]["cv_sel_eff"], default=None)
    out = {"what": "PRE-REGISTERED arm selection, TRAIN-ONLY 5-fold image-grouped CV. Eval never touched.",
           "date": "2026-08-12", "folds": A.folds, "seeds": A.seeds, "layer": A.layer,
           "grid_layer": A.grid_layer, "threads": A.threads,
           "endpoint": "within-question argmax hit rate on the held-out fold "
                       "(identical to fit_hidden_head.cv)",
           "table": res, "cv_selected_overall": best, "cv_selected_vision_arm": best_vis,
           "note": "cv_sel_eff is NOT comparable in level to eval sel_eff: the train pool has no "
                   "8-slot sampling structure and a different candidate-count distribution. Only "
                   "the RANKING of arms is used."}
    json.dump(out, open(A.out, "w"), indent=1)
    print(json.dumps({k: round(v["cv_sel_eff"], 6) for k, v in res.items()}, indent=1))
    print("CV-SELECTED overall:", best, " vision arm:", best_vis)


if __name__ == "__main__":
    main()
