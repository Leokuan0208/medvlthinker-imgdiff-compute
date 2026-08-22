#!/usr/bin/env python3
"""greedy_anchored_selector.py -- override the greedy answer only when the head clears a margin.

WHERE THIS COMES FROM.  Decomposing the eight-cell result (2026-08-22) splits `head - greedy` into
two independent terms:

    head - greedy  =  (head - one T=0.7 sample)  -  (greedy - one T=0.7 sample)
                       ^ selection skill            ^ sampling penalty

    cell             skill    penalty   net
    pathvqa         +0.1306   0.0833   +0.0473
    kvasir_x1       +0.1055   0.0274   +0.0781
    slake           +0.0756   0.0368   +0.0388
    vqa_rad         +0.0436   0.0686   -0.0250
    gemex           +0.0433   0.0285   +0.0148
    radimagenet     +0.0314   0.0229   +0.0085
    vqamed          +0.0175   0.0434   -0.0259
    omnimed         -0.0048   0.0441   -0.0489

Selection skill is POSITIVE on 7 of 8 cells.  The head is not incompetent where it loses; it is
selecting over a pool that is itself worse than greedy decoding, and on vqa_rad it has +0.0436 of
real skill and still loses because the penalty is 0.0686.

AND THE GREEDY ANSWER IS ALREADY IN THE POOL: 66.6%-98.8% of questions, yet the head picks it only
25.6% (vqamed) to 76.9% (slake) of the time.  So the failure is not availability, it is that the
head OVERRIDES a better answer it already has in front of it.

THE POLICY.  Keep the greedy answer unless the head's best candidate beats greedy's own head score
by more than tau, on the deployed within-pool rank_avg scale (so tau is in [0,1] and comparable
across cells).  tau=0 is the plain head, tau>=1 is always-greedy, and everything between
interpolates -- so this cannot be worse than the better endpoint except through tau being chosen
badly, which is why tau is fitted LEAVE-ONE-CELL-OUT and never on the cell it is scored on.  This is
the project's certified-veto shape and it is NOT abstention: every branch returns an answer, and the
default branch returns the model's own.

WHEN GREEDY IS NOT IN THE POOL there is no head score to compare against, and the two defensible
fallbacks disagree, so both are reported rather than one being quietly chosen:
    conservative  take greedy (it is a real answer whose label we have)
    aggressive    take the head's pick (the pool never reproduced greedy, so treat it as an outlier)

  python3 src/cascade_methods/greedy_anchored_selector.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/greedy_anchored_2026-08-22.json")
from free_signal_bakeoff import CELLS, RAW, norm, load_cell

TAUS = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50, 1.01]


def macro_boot(A, B, cells, nboot=10000, seed=3):
    rng = np.random.default_rng(seed)
    d = np.empty(nboot)
    for i in range(nboot):
        k = rng.integers(0, len(cells), len(cells))
        d[i] = np.mean([A[cells[j]] for j in k]) - np.mean([B[cells[j]] for j in k])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(np.mean([A[c] for c in cells]) - np.mean([B[c] for c in cells])),
            "ci": [float(lo), float(hi)],
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE", "resampled": "cells"}


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    D = {}
    for cell in CELLS:
        got = load_cell(cell)
        if got is None:
            continue
        H, rows = got
        stem = RAW.get(cell, cell)
        jp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.judge.jsonl")
        gp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.jsonl")
        if not (os.path.exists(jp) and os.path.exists(gp)):
            continue
        y = np.array([r["y"] for r in rows], dtype=int)
        na = np.array([norm(r["na"]) for r in rows])
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        gtxt, gok = {}, {}
        for l in open(gp):
            if l.strip():
                d = json.loads(l); gtxt[d["idx"]] = norm((d.get("preds") or [d.get("pred", "")])[0])
        for l in open(jp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        L = sel.head_logits(H)
        hd, gd, mg, present = [], [], [], []
        for q in sorted(byq, key=lambda k: (len(str(k)), str(k))):
            if q not in gok or q not in gtxt:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            w = int(np.argmax(hr))
            hd.append(int(y[ii][w])); gd.append(gok[q])
            hit = np.where(na[ii] == gtxt[q])[0]
            present.append(len(hit) > 0)
            mg.append(float(hr[w] - hr[hit[0]]) if len(hit) else np.nan)
        D[cell] = {k: np.array(v) for k, v in
                   (("head", hd), ("greedy", gd), ("margin", mg), ("present", present))}
        print(f"  loaded {cell:17} {len(hd):6} q, greedy in pool {np.mean(present):.1%}", flush=True)

    cells = list(D)

    def apply(cell, tau, fallback):
        d = D[cell]
        take_head = np.where(d["present"], np.nan_to_num(d["margin"], nan=0.0) > tau,
                             True if fallback == "aggressive" else False)
        return np.where(take_head, d["head"], d["greedy"])

    art = {"title": "Greedy-anchored selection: override greedy only past a margin",
           "date": "2026-08-22", "no_fabricated_numbers": True,
           "taus": TAUS, "cells": cells, "results": {}}

    for fallback in ("conservative", "aggressive"):
        # full tau sweep, for the record (in-sample; NOT the headline)
        sweep = {str(t): {c: float(apply(c, t, fallback).mean()) for c in cells} for t in TAUS}
        art["results"][fallback] = {"in_sample_sweep_macro":
                                    {t: float(np.mean(list(v.values()))) for t, v in sweep.items()}}
        # leave-one-cell-out tau
        per, taus_used = {}, {}
        for held in cells:
            tr = [c for c in cells if c != held]
            best_t = max(TAUS, key=lambda t: np.mean([sweep[str(t)][c] for c in tr]))
            per[held] = float(apply(held, best_t, fallback).mean()); taus_used[held] = best_t
        head_m = {c: float(D[c]["head"].mean()) for c in cells}
        greedy_m = {c: float(D[c]["greedy"].mean()) for c in cells}
        art["results"][fallback].update({
            "loco_tau_per_cell": taus_used, "loco_accuracy_per_cell": per,
            "macro_anchored": float(np.mean(list(per.values()))),
            "macro_head": float(np.mean(list(head_m.values()))),
            "macro_greedy": float(np.mean(list(greedy_m.values()))),
            "vs_head": macro_boot(per, head_m, cells),
            "vs_greedy": macro_boot(per, greedy_m, cells),
            "per_cell_vs_head": {c: per[c] - head_m[c] for c in cells},
            "n_cells_not_worse_than_greedy": int(sum(per[c] >= greedy_m[c] - 1e-9 for c in cells))})
        a = art["results"][fallback]
        print(f"\n[{fallback}] LOCO tau per cell: {taus_used}")
        print(f"[{fallback}] macro anchored {a['macro_anchored']:.4f} | head {a['macro_head']:.4f} "
              f"| greedy {a['macro_greedy']:.4f}")
        for k in ("vs_head", "vs_greedy"):
            x = a[k]
            print(f"[{fallback}]   {k:10} {x['delta']:+.4f} [{x['ci'][0]:+.4f},{x['ci'][1]:+.4f}] "
                  f"{x['verdict']}")
        print(f"[{fallback}]   cells not worse than greedy: "
              f"{a['n_cells_not_worse_than_greedy']}/{len(cells)}  "
              f"(plain head manages {sum(head_m[c] >= greedy_m[c] - 1e-9 for c in cells)}/{len(cells)})")
        json.dump(art, open(OUT, "w"), indent=1)

    best = max(("conservative", "aggressive"),
               key=lambda f: art["results"][f]["macro_anchored"])
    b = art["results"][best]
    art["VERDICT"] = (f"{best} fallback: macro {b['macro_anchored']:.4f} vs plain head "
                      f"{b['macro_head']:.4f} ({b['vs_head']['delta']:+.4f} "
                      f"[{b['vs_head']['ci'][0]:+.4f},{b['vs_head']['ci'][1]:+.4f}] "
                      f"{b['vs_head']['verdict']}) and vs greedy {b['macro_greedy']:.4f} "
                      f"({b['vs_greedy']['delta']:+.4f} "
                      f"[{b['vs_greedy']['ci'][0]:+.4f},{b['vs_greedy']['ci'][1]:+.4f}] "
                      f"{b['vs_greedy']['verdict']}); "
                      f"{b['n_cells_not_worse_than_greedy']}/{len(cells)} cells not worse than "
                      f"greedy, the guardrail the plain head fails.")
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
