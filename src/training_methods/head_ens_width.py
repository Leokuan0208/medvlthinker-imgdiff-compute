#!/usr/bin/env python3
"""head_ens_width.py -- how many layers should the ensemble average over?

THE SHIPPED ARTIFACT rank-ensembles layers 18/20/22.  That set was chosen before 19 and 21 had been
extracted, so it was never a choice so much as what happened to be on disk.  With all five now
available for four benchmarks, single-layer end-to-end says:

    L18 +0.0275   L19 +0.0331   L20 +0.0235   L21 +0.0201   L22 +0.0277

The ensemble is therefore missing the BEST single layer (19) and includes the worst (nothing, but
21 is excluded and is worst).  The whole spread is ~0.013, which is the regime where averaging more
correlated-but-not-identical signals usually pays -- and where picking one is mostly noise-fitting.

Compared here, all fitted on the POOLED training set and scored on the held-out image halves, i.e.
the configuration that would actually ship:
    each single layer 18..22
    {18,20,22}        the shipped set
    {18,19,20,21,22}  every layer we have
    {19,20,21}        a narrow band around the best single layer

  python3 src/training_methods/head_ens_width.py --threads 4 --seeds 5
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
# 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
# generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
# the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
# generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
SHARED = {"slake_open", "vqa_rad_open"}
SETS = {"L18": [18], "L19": [19], "L20": [20], "L21": [21], "L22": [22],
        "shipped_18_20_22": [18, 20, 22], "all5_18_22": [18, 19, 20, 21, 22],
        "narrow_19_20_21": [19, 20, 21]}


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, want, dsf=None):
    """{layer: X} for the layers in `want` that this cache actually holds, plus its rows."""
    zp, mp = f"{HS.FEATS}/{stem}.npz", f"{HS.FEATS}/{stem}.meta.json"
    if not os.path.exists(zp):
        return None, None
    z = np.load(zp); m = json.load(open(mp))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    if not keep:
        return None, None
    X = {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in want if L in lay}
    return X, [m["rows"][i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_ens_width_2026-08-25.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg
    ALL = [18, 19, 20, 21, 22]

    # training features: even layers from finelayer, odd from oddlayer, merged by row identity
    Xe, rows_e = load("generator_train_finelayer", ALL, TRAIN_DOMAINS)
    Xo, rows_o = load("generator_train_oddlayer", ALL, TRAIN_DOMAINS)
    key_e = [(r["ds"], r["idx"], r["na"]) for r in rows_e]
    pos_o = {(r["ds"], r["idx"], r["na"]): i for i, r in enumerate(rows_o)}
    sel = [pos_o.get(k, -1) for k in key_e]
    ok = np.array([i >= 0 for i in sel])
    idx_o = np.array([i for i in sel if i >= 0])
    Xtr = {L: (Xe[L][ok] if L in Xe else Xo[L][idx_o]) for L in ALL}
    rows = [r for r, k in zip(rows_e, ok) if k]
    print(f"train rows aligned across even+odd caches: {len(rows):,} "
          f"(of {len(rows_e):,} even / {len(rows_o):,} odd)", flush=True)

    # add each benchmark's training half
    add = {L: [Xtr[L]] for L in ALL}
    src = [r["ds"] for r in rows]
    ev = {}
    for cell in BENCH:
        dsf = {cell} if cell in SHARED else None
        Xa, ra = load("generator_eval_finelayer" if cell in SHARED
                      else f"generator_eval_finelayer_{cell}", ALL, dsf)
        Xb, rb = load("generator_eval_oddlayer" if cell in SHARED
                      else f"generator_eval_oddlayer_{cell}", ALL, dsf)
        if Xa is None or Xb is None:
            print(f"  [skip] {cell}: missing an even or odd cache", flush=True); continue
        pb = {(r["idx"], r["na"]): i for i, r in enumerate(rb)}
        s2 = [pb.get((r["idx"], r["na"]), -1) for r in ra]
        m2 = np.array([i >= 0 for i in s2]); i2 = np.array([i for i in s2 if i >= 0])
        Xc = {L: (Xa[L][m2] if L in Xa else Xb[L][i2]) for L in ALL}
        rc = [r for r, k in zip(ra, m2) if k]
        istr = np.array([half(r["img_md5"]) == 1 for r in rc])
        for L in ALL:
            add[L].append(Xc[L][istr])
        rows += [rc[i] for i in np.where(istr)[0]]; src += [cell] * int(istr.sum())
        gok = {}
        for l in open(f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = {"X": {L: Xc[L][~istr] for L in ALL},
                    "rows": [rc[i] for i in np.where(~istr)[0]], "gok": gok}
    X = {L: np.concatenate(add[L]) for L in ALL}

    evimgs = set()
    for d in ev.values():
        evimgs |= {r["img_md5"] for r in d["rows"]}
    drop = np.array([r["img_md5"] in evimgs for r in rows])
    if drop.any():
        for L in ALL:
            X[L] = X[L][~drop]
        rows = [r for r, d in zip(rows, drop) if not d]
        src = [s for s, d in zip(src, drop) if not d]
        print(f"dropped {int(drop.sum())} leaking training rows", flush=True)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src, rows)])
    print(f"pooled training set: {len(y):,} rows over {len(ev)} benchmarks\n", flush=True)

    M = {}
    for L in ALL:
        mu, sg = X[L].mean(0), X[L].std(0) + 1e-6
        M[L] = (mu, sg, [HS.fit((X[L] - mu) / sg, y, qid, None, objective="bce", hidden=256,
                                wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)])
        print(f"  fitted layer {L}", flush=True)

    art = {"title": "How wide should the layer ensemble be?", "date": "2026-08-25",
           "no_fabricated_numbers": True, "seeds": A.seeds, "sets": {k: v for k, v in SETS.items()},
           "pooled_rows": int(len(y)), "cells": {}}
    for cell, d in ev.items():
        rr, gok = d["rows"], d["gok"]
        ye = np.array([r["y"] for r in rr], dtype=int)
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = [q for q in byq if q in gok]
        if len(qs) < 40:
            continue
        S = {}
        for L in ALL:
            mu, sg, ms = M[L]
            S[L] = np.stack([HS.predict(mm, (d["X"][L] - mu) / sg) for mm in ms])
        out = {"n_questions": len(qs), "greedy": float(np.mean([gok[q] for q in qs]))}
        for nm, layers in SETS.items():
            acc = []
            for q in qs:
                ii = np.array(byq[q])
                hr = np.mean([rank_avg(S[L][k][ii]) for L in layers for k in range(S[L].shape[0])],
                             axis=0)
                acc.append(int(ye[ii][int(np.argmax(hr))]))
            out[nm] = float(np.mean(acc)) - out["greedy"]
        art["cells"][cell] = out
        print(f"  {cell:17} n{len(qs):6} " +
              "  ".join(f"{k[:9]} {out[k]:+.4f}" for k in SETS), flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    cs = art["cells"]
    if cs:
        art["macro"] = {k: float(np.mean([v[k] for v in cs.values()])) for k in SETS}
        best = max(art["macro"], key=art["macro"].get)
        ship = art["macro"]["shipped_18_20_22"]
        art["VERDICT"] = (f"best is {best} at {art['macro'][best]:+.4f}; the shipped 18/20/22 set is "
                          f"{ship:+.4f} ({art['macro'][best]-ship:+.4f}). " +
                          ("Widen the ensemble." if art["macro"][best] - ship > 0.005 else
                           "The shipped set is within noise of the best -- ensemble width is not a "
                           "lever worth re-freezing the artifact for."))
        for k in SETS:
            print(f"  MACRO {k:20} {art['macro'][k]:+.4f}")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
