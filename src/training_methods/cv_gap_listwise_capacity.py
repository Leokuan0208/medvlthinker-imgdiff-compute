#!/usr/bin/env python3
"""cv_gap_listwise_capacity.py -- close a protocol gap in fit_hidden_head.py.

The staged CV grid there selected (layer, pooling) with a linear pointwise head, then swept
objective at hidden=0, then swept capacity at the winning objective. It therefore NEVER cross-
validated listwise x hidden=256 or bt x hidden=256 -- yet those are the two arms that did best on
eval. This script fills the missing cells with the IDENTICAL fold protocol, so we can say whether
the CV criterion would have chosen them given the chance (protocol gap) or would still have
preferred pointwise (a genuine CV->eval transfer failure).

No eval data is touched. Output is CV-only.

  python3 src/training_methods/cv_gap_listwise_capacity.py
"""
import os, sys, json, hashlib
import numpy as np
from collections import defaultdict
import importlib.util

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
spec = importlib.util.spec_from_file_location("fh", os.path.join(ROOT, "src/training_methods/fit_hidden_head.py"))
_a = sys.argv; sys.argv = ["fit_hidden_head"]
fh = importlib.util.module_from_spec(spec); spec.loader.exec_module(fh); sys.argv = _a

FOLDS = 5
ztr, mtr = fh.load_cache("feats_hidden", "grader", "train")
layers = list(ztr["layers"])


def cv(cfg):
    li = layers.index(cfg["layer"])
    X, y, keys, grp, _ = fh.build_matrix(ztr, mtr, li, cfg["pooling"], cfg.get("setrel", 0))
    qid = [f"{a}|{b}" for (a, b, c) in keys]
    fo = np.array([int(hashlib.md5(str(g).encode()).hexdigest(), 16) % FOLDS for g in grp])
    aucs, effs = [], []
    for f in range(FOLDS):
        tr, va = fo != f, fo == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = fh.fit_head((X[tr] - mu) / sd, y[tr], [qid[i] for i in np.where(tr)[0]],
                        objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                        epochs=cfg["epochs"], seed=f)
        sv = fh.predict(m, (X[va] - mu) / sd)
        aucs.append(fh.auroc(y[va], sv))
        vidx = np.where(va)[0]; loc = {i: j for j, i in enumerate(vidx)}
        byq = defaultdict(list)
        for i in vidx:
            byq[qid[i]].append(i)
        hit = tot = 0
        for q, ii in byq.items():
            if y[ii].sum() == 0:
                continue
            b = ii[int(np.argmax([sv[loc[i]] for i in ii]))]
            hit += int(y[b] == 1); tot += 1
        effs.append(hit / max(tot, 1))
    r = {**cfg, "cv_auroc": float(np.mean(aucs)), "cv_sel_eff": float(np.mean(effs))}
    print(f"  cv {cfg['objective']}/h{cfg['hidden']}/wd{cfg['wd']}: auroc={r['cv_auroc']:.4f} "
          f"sel_eff={r['cv_sel_eff']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    base = {"layer": 21, "pooling": "last", "wd": 0.01, "epochs": 30, "setrel": 0}
    out = [cv({**base, "objective": o, "hidden": h})
           for o in ["listwise", "bt"] for h in [256]]
    op = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_cvgap_2026-08-04.json")
    json.dump({"what": "missing CV cells (listwise/bt x hidden=256) at the stage-1 representation; "
                       "same fold protocol as fit_hidden_head.py; CV-only, eval untouched",
               "reference_cv_cells_from_main_run": {
                   "bce/h256/wd0.01": 0.68976815559549, "bce/h0/wd0.01": 0.6688,
                   "listwise/h0/wd0.01": 0.6583, "bt/h0/wd0.01": 0.6641},
               "new_cells": out}, open(op, "w"), indent=1)
    print(f"wrote {op}", flush=True)
