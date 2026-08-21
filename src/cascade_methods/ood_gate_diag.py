#!/usr/bin/env python3
"""ood_gate_diag.py -- is a cheap OOD score predictive of WHICH selector to trust?

THE OBSERVATION IT STARTS FROM (free_signal_bakeoff_2026-08-21.json).  Self-consistency -- the
sampling frequency of a candidate among the raw 8, which needs no training and carries no answer
vocabulary -- beats the trained head on exactly the cells the head was not trained on, and loses to
it on exactly the cells it was:

    pathvqa_open     IN    head 0.3900  SC 0.3260   SC-head -0.0640 LOSS
    slake_open       IN    head 0.7690  SC 0.7395   SC-head -0.0295 LOSS
    kvasir_x1_open   IN    head 0.3629  SC 0.2699   SC-head -0.0930 LOSS
    omnimed_open     OUT   head 0.3396  SC 0.3846   SC-head +0.0449 WIN
    vqamed_open      OUT   head 0.0688  SC 0.0863   SC-head +0.0175 WIN

The sign flips at the domain boundary.  If the boundary is DETECTABLE at test time from the features
alone, the flip is exploitable: trust the head where it has support and the free signal where it does
not.  Note also that out of domain NEITHER beats greedy (omnimed greedy 0.3885, vqamed 0.0947), so
the most valuable decision may be whether to draw the 8 samples at all.

THE SCORE.  Mahalanobis distance of a candidate's layer-21 h_span to the head's own training
distribution, computed in a 64-dimensional PCA basis fitted on those same training rows (the full
3584-d covariance is neither invertible at 31,498 rows nor necessary -- the regularisation sweep
found the usable signal is low-dimensional).  Nothing here is fitted on any eval cell.

THIS SCRIPT ONLY DIAGNOSES.  It reports, per cell, the mean OOD score beside the measured
head-minus-SC gap, and within each cell the per-question rank correlation between OOD and which
selector was right.  A gate is only worth building if the cell-level ordering is monotone.

  python3 src/cascade_methods/ood_gate_diag.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/ood_gate_diag_2026-08-21.json")
LAYERS = [7, 14, 21, 28]
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from free_signal_bakeoff import CELLS, RAW, norm, load_cell, boot


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    # --- the head's own training distribution, in a 64-d PCA basis --------------------------
    TR = []
    for sh in (0, 1):
        z = np.load(os.path.join(FEATS, f"generator_train_s{sh}of2.npz"))
        m = json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))
        keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
        TR.append(z["h_span"][keep, LAYERS.index(21)].astype(np.float32))
    TR = np.concatenate(TR)
    mu = TR.mean(0)
    Xc = TR - mu
    rng = np.random.default_rng(0)
    g = rng.standard_normal((Xc.shape[1], 80)).astype(np.float32)
    Q, _ = np.linalg.qr(Xc @ g)
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    P = Vt[:64].T.astype(np.float32)
    Z = Xc @ P
    sd = Z.std(0) + 1e-6
    print(f"PCA basis on {len(TR)} training rows; explained sd range "
          f"{sd.min():.3f}..{sd.max():.3f}", flush=True)

    def ood(H):
        """RMS whitened deviation from the training mean, per candidate."""
        return np.sqrt(((((H - mu) @ P) / sd) ** 2).mean(1))

    art = {"title": "Is a cheap OOD score predictive of which selector to trust?",
           "date": "2026-08-21", "no_fabricated_numbers": True,
           "score": "Mahalanobis distance of layer-21 h_span to the head's TRAINING distribution, "
                    "in a 64-d PCA basis fitted on those training rows only",
           "cells": {}}
    TRAINED = {"pathvqa_open", "slake_open", "vqa_rad_open", "kvasir_x1_open"}

    for cell in CELLS:
        got = load_cell(cell)
        if got is None:
            continue
        H, rows = got
        stem = RAW.get(cell, cell)
        rawp = os.path.join(CK, f"ckpt_{stem}_lingshu7b_sc8.jsonl")
        jp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.judge.jsonl")
        if not (os.path.exists(rawp) and os.path.exists(jp)):
            continue
        y = np.array([r["y"] for r in rows], dtype=int)
        na = np.array([norm(r["na"]) for r in rows])
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        qids = sorted(byq, key=lambda k: (len(str(k)), str(k)))
        raw = {}
        for l in open(rawp):
            if l.strip():
                d = json.loads(l); raw[d["idx"]] = d
        sc = np.zeros(len(rows), dtype=np.float32)
        for i, r in enumerate(rows):
            d = raw.get(r["idx"])
            if d:
                c = Counter(norm(p) for p in d["preds"])
                sc[i] = c.get(na[i], 0) / max(len(d["preds"]), 1)
        od = ood(H)
        L = sel.head_logits(H)
        g = {}
        for l in open(jp):
            if l.strip():
                d = json.loads(l); g[d["idx"]] = d

        hq, sq, gq, oq = [], [], [], []
        for q in qids:
            if q not in g:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            hq.append(int(y[ii][int(np.argmax(hr))]))
            sq.append(int(y[ii][int(np.argmax(rank_avg(sc[ii])))]))
            gq.append(int(g[q]["judge_ok"]))
            oq.append(float(od[ii].mean()))
        hq, sq, gq, oq = map(np.array, (hq, sq, gq, oq))

        # within-cell: does OOD separate the questions where the head beat SC from the reverse?
        diff = hq - sq
        hi = oq >= np.median(oq)
        art["cells"][cell] = {
            "domain": "IN" if cell in TRAINED else "OUT",
            "n_questions": int(len(hq)),
            "mean_ood": float(oq.mean()), "median_ood": float(np.median(oq)),
            "head": float(hq.mean()), "sc": float(sq.mean()), "greedy": float(gq.mean()),
            "head_minus_sc": float(hq.mean() - sq.mean()),
            "head_minus_greedy": float(hq.mean() - gq.mean()),
            "within_cell_head_minus_sc_lowOOD": float(diff[~hi].mean()),
            "within_cell_head_minus_sc_highOOD": float(diff[hi].mean())}
        c = art["cells"][cell]
        print(f"  {cell:17} {c['domain']:4} meanOOD {c['mean_ood']:7.4f}  "
              f"head-SC {c['head_minus_sc']:+.4f}  head-greedy {c['head_minus_greedy']:+.4f}  "
              f"| within-cell lowOOD {c['within_cell_head_minus_sc_lowOOD']:+.4f} "
              f"highOOD {c['within_cell_head_minus_sc_highOOD']:+.4f}", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    # the whole question: is the cell-level ordering monotone?
    cs = art["cells"]
    o = np.array([v["mean_ood"] for v in cs.values()])
    d = np.array([v["head_minus_sc"] for v in cs.values()])
    if len(o) > 2:
        r = float(np.corrcoef(o, d)[0, 1])
        ro = np.argsort(np.argsort(o)); rd = np.argsort(np.argsort(d))
        sr = float(np.corrcoef(ro, rd)[0, 1])
        art["cell_level_pearson_ood_vs_head_minus_sc"] = r
        art["cell_level_spearman_ood_vs_head_minus_sc"] = sr
        art["VERDICT"] = ("OOD score is predictive -- a gate is worth building"
                          if sr < -0.6 else
                          "OOD score does NOT order the cells -- a gate on this score is not "
                          "supported by the evidence")
        print(f"\ncell-level pearson {r:+.3f}  spearman {sr:+.3f}\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
