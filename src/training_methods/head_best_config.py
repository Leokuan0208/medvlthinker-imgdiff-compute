#!/usr/bin/env python3
"""head_best_config.py -- do the two gains combine, and do they survive END TO END?

TWO THINGS WORKED, on the same leave-one-training-domain-out endpoint:

    layers_18_20_22_ens   +0.01577 over h_span   (head_representation_2026-08-24.json, layer 20 base)
    h_span + self-consistency as an INPUT FEATURE
                          +0.00888 over h_span   (head_input_augmentation_2026-08-24.json, layer 21 base)

They are plausibly independent -- one changes which hidden states are read, the other adds a signal
that is not a hidden state at all -- so this fits the 2x2 and checks.

WHY THIS SCRIPT DOES NOT TRUST THE ENDPOINT THAT FOUND THEM.  Both gains were selected on
leave-one-training-domain-out sel_eff minus the answer-prior null, over the four training domains.
That proxy has already been caught disagreeing with deployment: the layer sweep's transfer metric
picked layer 18, in-domain CV picked 20, and the end-to-end winner over eight benchmarks was 22.
So every configuration here is re-measured the way it would actually be used -- verifier pick vs
greedy decoding, per benchmark, on all eight -- and the proxy numbers are carried alongside only so
the disagreement is visible.

CONFIGURATIONS
  A  layer 21, h_span                    the deployed recipe
  B  rank-ensemble over layers 18/20/22  the representation winner
  C  layer 21, h_span + self-consistency the input-augmentation winner
  D  B + C                               both

  python3 src/training_methods/head_best_config.py --threads 4 --seeds 5
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict, Counter

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS
from head_domain_scaling import norm

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
ENS_LAYERS = [18, 20, 22]
SHARED = {"pathvqa_open", "slake_open", "vqa_rad_open"}   # live in the combined finelayer cache


def sc_for(rows, ds_of_row, tag="lingshu7b"):
    raw = {}
    for ds in sorted({ds_of_row(r) for r in rows}):
        p = f"{CK}/ckpt_{ds}_{tag}_sc8.jsonl"
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    d = json.loads(l); raw[(ds, d["idx"])] = d
    out = np.zeros(len(rows), np.float32)
    for i, r in enumerate(rows):
        d = raw.get((ds_of_row(r), r["idx"]))
        if d:
            c = Counter(norm(p) for p in d["preds"])
            out[i] = c.get(norm(r["na"]), 0) / max(len(d["preds"]), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_best_config_2026-08-24.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    # ---- training features: layer 21 from the standard cache, 18/20/22 from finelayer ----------
    Xs21, rows21 = [], []
    for sh in (0, 1):
        p = f"{HS.FEATS}/generator_train_s{sh}of2"
        z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
        lay = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and r["ds"] in TRAIN_DOMAINS]
        Xs21.append(z["h_span"][keep, lay.index(21)].astype(np.float32))
        rows21 += [m["rows"][i] for i in keep]
    X21 = np.concatenate(Xs21)
    pf = f"{HS.FEATS}/generator_train_finelayer"
    zf = np.load(pf + ".npz"); mf = json.load(open(pf + ".meta.json"))
    layf = [int(x) for x in zf["layers"]]
    keepf = [i for i, r in enumerate(mf["rows"])
             if r.get("n_tok", -1) > 0 and r["ds"] in TRAIN_DOMAINS]
    rowsf = [mf["rows"][i] for i in keepf]
    Xtri = {L: zf["h_span"][keepf, layf.index(L)].astype(np.float32) for L in ENS_LAYERS}
    print(f"train: {len(rows21)} rows (layer 21 cache) / {len(rowsf)} rows (finelayer cache)",
          flush=True)

    def prep(rows):
        return (np.array([r["y"] for r in rows], dtype=np.float32),
                np.array([f"{r['ds']}|{r['idx']}" for r in rows]))
    y21, q21 = prep(rows21); yf, qf = prep(rowsf)
    sc21 = sc_for(rows21, lambda r: r["ds"])
    scf = sc_for(rowsf, lambda r: r["ds"])

    def fit_all(X, y, q, seeds):
        mu, sg = X.mean(0), X.std(0) + 1e-6
        return mu, sg, [HS.fit((X - mu) / sg, y, q, None, objective="bce", hidden=256,
                               wd=1e-2, epochs=30, seed=s) for s in range(seeds)]

    print("fitting A (layer 21) ...", flush=True)
    A_ = fit_all(X21, y21, q21, A.seeds)
    print("fitting C (layer 21 + sc) ...", flush=True)
    C_ = fit_all(np.concatenate([X21, sc21[:, None]], 1), y21, q21, A.seeds)
    print("fitting B (layers 18/20/22) ...", flush=True)
    B_ = {L: fit_all(Xtri[L], yf, qf, A.seeds) for L in ENS_LAYERS}
    print("fitting D (layers 18/20/22 + sc) ...", flush=True)
    D_m = {L: fit_all(np.concatenate([Xtri[L], scf[:, None]], 1), yf, qf, A.seeds)
           for L in ENS_LAYERS}

    def score(models, X):
        mu, sg, ms = models
        return np.stack([HS.predict(mm, (X - mu) / sg) for mm in ms])

    art = {"title": "Do the representation and input-augmentation gains combine, end to end?",
           "date": "2026-08-24", "no_fabricated_numbers": True, "seeds": A.seeds,
           "configs": {"A": "layer 21, h_span (deployed recipe)",
                       "B": "rank-ensemble over layers 18/20/22",
                       "C": "layer 21, h_span + self-consistency feature",
                       "D": "B + C"},
           "endpoint": "verifier pick vs greedy decoding, judge currency, per benchmark",
           "cells": {}}
    for cell in BENCH:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        # layer-21 view
        p1 = f"{HS.FEATS}/generator_eval_{cell}"
        # finelayer view
        p2 = (f"{HS.FEATS}/generator_eval_finelayer" if cell in SHARED
              else f"{HS.FEATS}/generator_eval_finelayer_{cell}")
        if not (os.path.exists(p1 + ".npz") and os.path.exists(p2 + ".npz")):
            continue
        z1 = np.load(p1 + ".npz"); m1 = json.load(open(p1 + ".meta.json"))
        l1 = [int(x) for x in z1["layers"]]
        k1 = [i for i, r in enumerate(m1["rows"]) if r.get("n_tok", -1) > 0]
        r1 = [m1["rows"][i] for i in k1]
        E21 = z1["h_span"][k1, l1.index(21)].astype(np.float32)
        z2 = np.load(p2 + ".npz"); m2 = json.load(open(p2 + ".meta.json"))
        l2 = [int(x) for x in z2["layers"]]
        dsf = cell if cell in SHARED else None
        k2 = [i for i, r in enumerate(m2["rows"])
              if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") == dsf)]
        r2 = [m2["rows"][i] for i in k2]
        Etri = {L: z2["h_span"][k2, l2.index(L)].astype(np.float32) for L in ENS_LAYERS}
        sc1 = sc_for(r1, lambda r: cell)
        sc2 = sc_for(r2, lambda r: cell)

        def picks(rows, getscores):
            y = np.array([r["y"] for r in rows], dtype=int)
            byq = defaultdict(list)
            for i, r in enumerate(rows):
                byq[r["idx"]].append(i)
            S = getscores()
            out, gr = [], []
            for q in byq:
                if q not in gok:
                    continue
                ii = np.array(byq[q])
                if isinstance(S, list):
                    hr = np.mean([rank_avg(s[k][ii]) for s in S for k in range(s.shape[0])], axis=0)
                else:
                    hr = np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], axis=0)
                out.append(int(y[ii][int(np.argmax(hr))])); gr.append(gok[q])
            return np.array(out), np.array(gr)

        res = {}
        aA, gr = picks(r1, lambda: score(A_, E21))
        aC, _ = picks(r1, lambda: score(C_, np.concatenate([E21, sc1[:, None]], 1)))
        aB, gr2 = picks(r2, lambda: [score(B_[L], Etri[L]) for L in ENS_LAYERS])
        aD, _ = picks(r2, lambda: [score(D_m[L], np.concatenate([Etri[L], sc2[:, None]], 1))
                                   for L in ENS_LAYERS])
        res = {"n_layer21": int(len(aA)), "n_finelayer": int(len(aB)),
               "greedy": float(gr.mean()),
               "A_layer21": float(aA.mean()), "B_ens": float(aB.mean()),
               "C_sc": float(aC.mean()), "D_ens_sc": float(aD.mean()),
               "A_minus_greedy": float(aA.mean() - gr.mean()),
               "B_minus_greedy": float(aB.mean() - gr2.mean()),
               "C_minus_greedy": float(aC.mean() - gr.mean()),
               "D_minus_greedy": float(aD.mean() - gr2.mean())}
        art["cells"][cell] = res
        print(f"  {cell:17} greedy {res['greedy']:.4f} | A {res['A_minus_greedy']:+.4f} "
              f"B {res['B_minus_greedy']:+.4f} C {res['C_minus_greedy']:+.4f} "
              f"D {res['D_minus_greedy']:+.4f}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mac = {k: float(np.mean([v[f"{k}_minus_greedy"] for v in cs.values()]))
               for k in ("A", "B", "C", "D")}
        art["macro_minus_greedy"] = mac
        art["cells_beating_greedy"] = {k: f"{sum(1 for v in cs.values() if v[f'{k}_minus_greedy']>0)}"
                                          f"/{len(cs)}" for k in ("A", "B", "C", "D")}
        best = max(mac, key=mac.get)
        art["VERDICT"] = (
            f"end to end over {len(cs)} benchmarks: A {mac['A']:+.4f}, B {mac['B']:+.4f}, "
            f"C {mac['C']:+.4f}, D {mac['D']:+.4f} -- best is {best}. " +
            (f"The two gains COMBINE: D beats both B ({mac['D']-mac['B']:+.4f}) and C "
             f"({mac['D']-mac['C']:+.4f})." if best == "D" else
             f"They do NOT combine; {best} alone is as good as using both."))
        for k in ("A", "B", "C", "D"):
            print(f"  MACRO {k} {mac[k]:+.4f}  ({art['cells_beating_greedy'][k]} beat greedy)")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
