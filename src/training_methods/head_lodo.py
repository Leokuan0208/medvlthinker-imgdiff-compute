#!/usr/bin/env python3
"""head_lodo.py -- LEAVE-ONE-DATASET-OUT: how much of the head's skill is domain-specific?

WHY.  The RadImageNet cell (radimagenet_cell_2026-08-18.json) showed the frozen head adds
+0.0065 [-0.0090,+0.0220] -- a TIE -- on a dataset it was never trained on, with selection
efficiency 0.6396 against a 0.5823 random-pick floor.  One cell is not enough to conclude the head
does not transfer.  The training pool already contains FOUR datasets, so the same question can be
asked four more times at zero GPU cost:

    IN-DOMAIN   train on all four (D's other image-folds included), test on D's held-out fold
    OUT-DOMAIN  train on the other three ONLY, test on all of D

Both are scored on the same items with the same rule, so the gap between them is the transfer
penalty, isolated from everything else.  A per-dataset random-pick floor is computed alongside, so
"above chance" is a measurement rather than an assumption.

  python3 src/training_methods/head_lodo.py --threads 8
"""
import argparse, json, os, sys, hashlib, time
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS

OUT = os.path.join(HS.OUTDIR, "head_lodo_2026-08-18.json")

CONFIGS = {
    "deployed_bt_h256": dict(HS.BASE),
    "sweep_bce_h1024": {**HS.BASE, "objective": "bce", "hidden": 1024},
    "sweep_bce_h256": {**HS.BASE, "objective": "bce", "hidden": 256},
}


def pick_stats(scores, y, qid, mask):
    """sel_eff and accuracy of argmax-per-question, over the rows selected by `mask`."""
    idx = np.where(mask)[0]
    byq = defaultdict(list)
    for i in idx:
        byq[qid[i]].append(i)
    got, rec = [], []
    for q, ii in byq.items():
        ii = np.array(ii)
        got.append(int(y[ii][int(np.argmax(scores[ii]))]))
        rec.append(int(y[ii].max()))
    got, rec = np.array(got), np.array(rec)
    return {"n_q": len(got), "acc": float(got.mean()),
            "sel_eff": float(got[rec == 1].mean()) if rec.sum() else float("nan"),
            "oracle": float(rec.mean())}


def random_floor(y, qid, mask):
    idx = np.where(mask)[0]
    byq = defaultdict(list)
    for i in idx:
        byq[qid[i]].append(i)
    num = den = 0
    for q, ii in byq.items():
        yy = y[np.array(ii)]
        if yy.max() == 1:
            num += yy.mean(); den += 1
    return float(num / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--bigtrain", action="store_true",
                    help="use the full 60,384-row judged pool instead of the inherited draw")
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    if A.bigtrain:
        sys.path.insert(0, D)
        from head_eval_bce import load_train_big
        H, y, qid, img, ds = load_train_big()
    else:
        H, y, qid, img, ds = HS.load_train()
    sets = sorted(set(ds))
    print(f"rows={len(y)} datasets={ {d: int((ds==d).sum()) for d in sets} }", flush=True)

    art = {"title": "Leave-one-dataset-out: is the head's skill domain-specific?",
           "date": "2026-08-18", "no_fabricated_numbers": True,
           "protocol": "IN-DOMAIN = 5-fold md5(image)%5 CV over the FULL pool, scored on the held-out "
                       "rows of D only. OUT-DOMAIN = fit on every row NOT in D, scored on all of D. "
                       "Identical head, identical standardiser recipe (train mu/sd), identical "
                       "argmax-per-question rule. Seeds averaged.",
           "threads": A.threads, "seeds": A.seeds,
           "pool": {d: int((ds == d).sum()) for d in sets},
           "results": {}}

    fold_of = {h: int(hashlib.md5(str(h).encode()).hexdigest(), 16) % HS.FOLDS for h in set(img)}
    fo = np.array([fold_of[h] for h in img])

    for cname, cfg in CONFIGS.items():
        X = HS.assemble(H, cfg)
        art["results"][cname] = {}
        for d in sets:
            isD = ds == d
            floor = random_floor(y, qid, isD)
            # ---- OUT-OF-DOMAIN: never sees a single row of D -------------------------------
            od = []
            for s in range(A.seeds):
                tr = ~isD
                mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
                m = HS.fit((X[tr] - mu) / sg, y[tr], qid[tr], None,
                           objective=cfg["objective"], hidden=cfg["hidden"],
                           depth=cfg.get("depth", 1), drop=cfg.get("drop", 0.0),
                           ln=cfg.get("ln", False), wd=cfg["wd"], lr=cfg.get("lr", 1e-3),
                           epochs=cfg["epochs"], seed=s)
                sc = HS.predict(m, (X - mu) / sg)
                od.append(pick_stats(sc, y, qid, isD))
            # ---- IN-DOMAIN: full pool, D's own held-out folds only -------------------------
            idr = []
            for s in range(A.seeds):
                per_fold = []
                for f in range(HS.FOLDS):
                    tr = fo != f
                    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
                    m = HS.fit((X[tr] - mu) / sg, y[tr], qid[tr], None,
                               objective=cfg["objective"], hidden=cfg["hidden"],
                               depth=cfg.get("depth", 1), drop=cfg.get("drop", 0.0),
                               ln=cfg.get("ln", False), wd=cfg["wd"], lr=cfg.get("lr", 1e-3),
                               epochs=cfg["epochs"], seed=s + 100 * f)
                    sc = HS.predict(m, (X - mu) / sg)
                    per_fold.append(pick_stats(sc, y, qid, isD & (fo == f)))
                w = np.array([p["n_q"] for p in per_fold], float)
                idr.append({"n_q": int(w.sum()),
                            "acc": float(np.average([p["acc"] for p in per_fold], weights=w)),
                            "sel_eff": float(np.average([p["sel_eff"] for p in per_fold], weights=w)),
                            "oracle": float(np.average([p["oracle"] for p in per_fold], weights=w))})

            def agg(rs, k):
                v = [r[k] for r in rs]
                return {"mean": float(np.mean(v)), "sd": float(np.std(v))}

            gap = agg(idr, "sel_eff")["mean"] - agg(od, "sel_eff")["mean"]
            art["results"][cname][d] = {
                "n_rows": int(isD.sum()), "n_questions": od[0]["n_q"],
                "random_pick_floor_sel_eff": floor,
                "in_domain_sel_eff": agg(idr, "sel_eff"), "in_domain_acc": agg(idr, "acc"),
                "out_domain_sel_eff": agg(od, "sel_eff"), "out_domain_acc": agg(od, "acc"),
                "oracle": od[0]["oracle"],
                "TRANSFER_PENALTY_sel_eff": float(gap),
                "headroom_above_floor_in": float((agg(idr, "sel_eff")["mean"] - floor) / (1 - floor)),
                "headroom_above_floor_out": float((agg(od, "sel_eff")["mean"] - floor) / (1 - floor)),
            }
            r = art["results"][cname][d]
            print(f"  [{cname}] {d:22} floor {floor:.4f} | in {r['in_domain_sel_eff']['mean']:.4f} "
                  f"| out {r['out_domain_sel_eff']['mean']:.4f} | penalty {gap:+.4f}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
