#!/usr/bin/env python3
"""hole17_null.py -- the PERMUTATION NULL for the macro-objective threshold refit.

Resumable, one JSON line per replicate, per-replicate error guard.  Each replicate destroys the
gate<->outcome association inside every cell (a within-cell row permutation that leaves every
marginal -- a7, a32, the score distribution, the pool composition -- untouched) and then runs the
IDENTICAL refit machinery.  What it earns is what "refit 8 per-cell thresholds against an aggregate
objective" is worth from noise alone.

  nohup env OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_null.py \
        --mode nested --n 200 > logs/hole17/null_nested.log 2>&1 &
"""
import os, sys, json, time, argparse, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hole17_run as R

ANCHORS = [("macro_iso_acc", "macro", "iso_acc"), ("macro_iso_cost", "macro", "iso_cost"),
           ("pooled_iso_acc", "pooled", "iso_acc"), ("pooled_iso_cost", "pooled", "iso_cost")]
OUTDIR = os.path.join(R.ART, "_hole17_null")
os.makedirs(OUTDIR, exist_ok=True)


def one_replicate(D0, seed, mode):
    D = D0 if seed == 0 else D0.permuted(1_000_000 + seed)
    fa = {k: R.folds(D.n[k], R.K_OUTER) for k in R.ORDER_B}
    rec = dict(seed=seed, mode=mode)
    if mode == "nested":
        res = R.heldout_pass(D, fa)
        inc = R.summarise(res["incumbent"], D.n)
        nes = R.nested_pass(D, fa, ANCHORS)
        arms = {lab: R.summarise(nes[lab], D.n) for lab, _, _ in ANCHORS}
        rec["mu"] = {lab: nes[lab]["_mu_per_outer_fold"] for lab, _, _ in ANCHORS}
    else:
        dia, res = R.diagnostic_pass(D, fa, ANCHORS)
        inc = R.summarise(res["incumbent"], D.n)
        arms = {lab: R.summarise(dia[lab], D.n) for lab, _, _ in ANCHORS}
        rec["mu"] = {lab: [dia[lab]["_mu"]] for lab, _, _ in ANCHORS}
    rec["incumbent"] = {q: inc[q] for q in ("macro_acc", "macro_cost", "pooled_acc", "pooled_cost")}
    rec["arms"] = {lab: {q: arms[lab][q] for q in ("macro_acc", "macro_cost", "pooled_acc", "pooled_cost")}
                   for lab, _, _ in ANCHORS}
    rec["d_macro_acc"] = {lab: arms[lab]["macro_acc"] - inc["macro_acc"] for lab, _, _ in ANCHORS}
    rec["d_macro_cost"] = {lab: arms[lab]["macro_cost"] - inc["macro_cost"] for lab, _, _ in ANCHORS}
    rec["d_pooled_acc"] = {lab: arms[lab]["pooled_acc"] - inc["pooled_acc"] for lab, _, _ in ANCHORS}
    rec["d_pooled_cost"] = {lab: arms[lab]["pooled_cost"] - inc["pooled_cost"] for lab, _, _ in ANCHORS}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["nested", "diagnostic"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--tag", default="")
    A = ap.parse_args()
    path = os.path.join(OUTDIR, f"null_{A.mode}{A.tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path):
            try: done.add(json.loads(l)["seed"])
            except Exception: pass
    D0 = R.Data()
    t0 = time.time()
    with open(path, "a") as fh:
        for s in range(A.lo, A.n + 1):                   # seed 0 == the REAL data (unpermuted)
            if s in done: continue
            try:
                rec = one_replicate(D0, s, A.mode)
            except Exception:
                rec = dict(seed=s, mode=A.mode, error=traceback.format_exc()[-800:])
            fh.write(json.dumps(rec, default=float) + "\n"); fh.flush()
            print("seed %d done  (%.1fs elapsed)" % (s, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
