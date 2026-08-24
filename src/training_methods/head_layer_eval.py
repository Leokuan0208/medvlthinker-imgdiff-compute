#!/usr/bin/env python3
"""head_layer_eval.py -- take the layer chosen on TRANSFER end to end, on every cell.

WHAT THE SWEEP FOUND (head_finelayer_shard*.json, 8 layers, 2026-08-24):

    layer   10       12       16       18       20       22       24       26
    CV     0.64635  0.66646  0.67463  0.68760  0.69497  0.68636  0.67992  0.67139
    transf +0.01533 +0.01569 +0.03969 +0.07137 +0.05935 +0.06526 +0.04315 +0.04536

CV and transfer DISAGREE: in-domain CV picks layer 20, out-of-domain transfer picks 18, and 18 is
worth +0.012 over 20 on transfer while costing 0.007 of CV.  The deployed head reads 21, chosen
years-equivalent ago from a four-point grid (7/14/21/28) on CV -- the metric the 2026-08-19 audit
showed is blind to the failure that matters.

This trains a head at a given layer on the frozen train pool and scores EVERY cell that has a
fine-layer cache, against the deployed layer-21 head measured the same way, so the comparison is
end-to-end rather than a proxy.  The string prior is reported beside it because on template-heavy
cells it eats most of an apparent gain.

  python3 src/training_methods/head_layer_eval.py --layer 18 --threads 4
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
sys.path.insert(0, os.path.join(os.path.dirname(D), "cascade_methods"))
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
# Each cell may have its features in more than one cache, because the layer grids were extracted
# in two passes: the even grid 10..26 as "finelayer", and the odd layers 19/21 as "oddlayer" so the
# DEPLOYED layer 21 can be compared under the same training recipe rather than against the frozen
# ensemble. Try each candidate stem and take the first that actually holds the requested layer --
# asking for a layer that is not on disk is what made the layer-21 job fail in ~30 consecutive
# waves before this.
FL_EVAL = {c: [f"generator_eval_finelayer_{c}", f"generator_eval_oddlayer_{c}"]
           for c in ("kvasir_x1_open", "vqamed_open", "omnimed_open", "gemex_open",
                     "radimagenet_open")}
for c in ("pathvqa_open", "slake_open", "vqa_rad_open"):
    FL_EVAL[c] = ["generator_eval_finelayer", "generator_eval_oddlayer"]
TRAIN_STEMS = ["generator_train_finelayer", "generator_train_oddlayer"]


def stem_with_layer(cands, layer, feats):
    """First candidate stem whose npz actually contains `layer`."""
    for st in cands:
        p = os.path.join(feats, st + ".npz")
        if os.path.exists(p):
            try:
                if layer in [int(x) for x in np.load(p)["layers"]]:
                    return st
            except Exception:
                pass
    return None


def norm(s):
    return str(s).strip().lower().rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    if A.out is None:
        A.out = os.path.join(HS.OUTDIR, f"head_layer{A.layer}_eval_2026-08-24.json")
    from genframe_data import rank_avg

    # ---- train an ensemble at the requested layer, on the frozen train pool ----------------
    tst = stem_with_layer(TRAIN_STEMS, A.layer, HS.FEATS)
    if tst is None:
        raise SystemExit(f"layer {A.layer} is in no training cache; extract it first "
                         f"(candidates: {TRAIN_STEMS})")
    base = f"{HS.FEATS}/{tst}"
    z = np.load(base + ".npz")
    rows = json.load(open(base + ".meta.json"))["rows"]
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    layers = [int(x) for x in z["layers"]]
    if A.layer not in layers:
        raise SystemExit(f"layer {A.layer} not extracted; on disk: {layers}")
    li = layers.index(A.layer)
    X = z["h_span"][keep, li].astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    mu, sg = X.mean(0), X.std(0) + 1e-6
    print(f"training layer-{A.layer} head on {len(y)} rows, {A.seeds} seeds", flush=True)
    models = [HS.fit((X - mu) / sg, y, qid, None, objective="bce", hidden=256, wd=1e-2,
                     epochs=30, seed=s) for s in range(A.seeds)]

    # string prior from the same training rows
    pos, tot = defaultdict(int), defaultdict(int)
    for r in rows:
        a = norm(r["na"]); tot[a] += 1; pos[a] += int(r["y"])
    gp = float(y.mean())

    art = {"title": f"Layer-{A.layer} head, end to end on every cell", "date": "2026-08-24",
           "no_fabricated_numbers": True, "layer": A.layer, "seeds": A.seeds,
           "why": "in-domain CV picks layer 20 and out-of-domain transfer picks 18; the deployed "
                  "head reads 21, chosen on CV from a four-point grid",
           "cells": {}}
    for cell, cands in FL_EVAL.items():
        stem = stem_with_layer(cands, A.layer, HS.FEATS)
        if stem is None:
            continue
        p = f"{HS.FEATS}/{stem}"
        jp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(p + ".npz") and os.path.exists(jp)):
            continue
        m = json.load(open(p + ".meta.json"))
        dsf = cell if stem in ("generator_eval_finelayer",
                               "generator_eval_oddlayer") else None
        ki = [i for i, r in enumerate(m["rows"])
              if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") == dsf)]
        if not ki:
            continue
        rr = [m["rows"][i] for i in ki]
        ml = [int(x) for x in np.load(p + ".npz")["layers"]]
        if A.layer not in ml:
            continue
        Xe = np.load(p + ".npz")["h_span"][ki, ml.index(A.layer)].astype(np.float32)
        ye = np.array([r["y"] for r in rr], dtype=int)
        nae = np.array([norm(r["na"]) for r in rr])
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        gok = {}
        for l in open(jp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        S = np.stack([HS.predict(mm, (Xe - mu) / sg) for mm in models])   # (seeds, n)
        spv = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in nae])
        hd, pr, gr, orc = [], [], [], []
        for q in byq:
            if q not in gok:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], axis=0)
            hd.append(int(ye[ii][int(np.argmax(hr))]))
            pr.append(int(ye[ii][int(np.argmax(spv[ii]))]))
            gr.append(gok[q]); orc.append(int(ye[ii].max()))
        if len(hd) < 50:
            continue
        hd, pr, gr, orc = map(np.array, (hd, pr, gr, orc))
        art["cells"][cell] = {
            "n_questions": int(len(hd)), "greedy": float(gr.mean()),
            "head": float(hd.mean()), "string_prior": float(pr.mean()),
            "oracle_at_8": float(orc.mean()),
            "head_minus_greedy": float(hd.mean() - gr.mean()),
            "head_minus_string_prior": float(hd.mean() - pr.mean())}
        c = art["cells"][cell]
        print(f"  {cell:17} head {c['head']:.4f} greedy {c['greedy']:.4f} "
              f"h-g {c['head_minus_greedy']:+.4f}  h-prior {c['head_minus_string_prior']:+.4f}",
              flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    if art["cells"]:
        art["macro_head_minus_greedy"] = float(
            np.mean([v["head_minus_greedy"] for v in art["cells"].values()]))
        art["cells_beating_greedy"] = (
            f"{sum(1 for v in art['cells'].values() if v['head_minus_greedy'] > 0)}"
            f"/{len(art['cells'])}")
        print(f"\n  MACRO head-minus-greedy {art['macro_head_minus_greedy']:+.4f} "
              f"({art['cells_beating_greedy']} cells beat greedy)")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
