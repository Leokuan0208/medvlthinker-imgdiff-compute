#!/usr/bin/env python3
"""integrate_prereg.py -- STAGE 2: pre-register this round's ONLY free choices on the
TRAIN split, with the eval labels never read.

The head architecture is NOT re-tuned: it is frozen at the configuration the 2026-08-04
round selected by train-only CV (L21 / h_span / Bradley-Terry / hidden 256 / wd 1e-2 /
30 epochs). What this round adds is a READOUT and a COMBINER, and those are the two things
that must be pre-registered here:

  Q1  seed-ensembling convention:  mean within-pool rank  vs  mean raw score  vs  mean z
  Q2  how many seeds k:            1, 2, 4, 8, 16
  Q3  combiner form:               parameter-free rank_avg  vs  weight-swept rank blend
                                   vs  learned logistic combiner
      -- On train there is no incumbent verifier (it has no scores there and was trained on
         part of it), so Q3 is pre-registered on the two scorers that DO exist on both
         splits: the generator-frame head and the grader-frame head. Only the FORM of the
         combiner transfers to eval; any fitted weight would need eval visibility and is
         therefore reported as a diagnostic, never as the headline.

Protocol: 5 folds, image-grouped (md5(decoded-RGB image md5) % 5), so a fold boundary is an
image boundary. Criterion: within-question selection efficiency on the held-out fold,
restricted to contested + recoverable train questions.

  python3 src/training_methods/integrate_prereg.py --device cuda:0
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G          # noqa: E402
import integrate_lib as IL         # noqa: E402
import cheapcontrast as CC         # noqa: E402  (standardize + device-portable port)
from fit_hidden_head import fit_head, predict  # noqa: E402  (the published CPU trainer)

CFG = IL.BASE_CFG
GRADER_CFG = dict(layer=21, pooling="last", objective="bce", hidden=256, wd=1e-2,
                  epochs=30, lr=1e-3, bs=256)


def cv_scores(X, y, gq, gimg, cfg, seeds, device, folds=5, tag=""):
    """Cross-fitted per-row scores: (n_seeds, n_rows), each row scored by a model that
    never saw that row's IMAGE."""
    fold = IL.fold_of_group(gimg, folds)
    S = np.zeros((len(seeds), len(y)), dtype=np.float64)
    for f in range(folds):
        tr = np.where(fold != f)[0]
        te = np.where(fold == f)[0]
        # standardization is part of the published recipe and must be FOLD-HONEST:
        # mu/sd come from the training folds only.
        Xa, Xb = CC.standardize(X[tr], X[te])
        ytr = y[tr]
        gtr = [gq[i] for i in tr]
        for si, sd in enumerate(seeds):
            t = time.time()
            if device == "cpu":
                m = fit_head(Xa, ytr, gtr, objective=cfg["objective"], hidden=cfg["hidden"],
                             wd=cfg["wd"], lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"],
                             seed=sd)
                S[si, te] = predict(m, Xb)
            else:
                m = CC.fit_head_dev(Xa, ytr, gtr, objective=cfg["objective"], hidden=cfg["hidden"],
                                    wd=cfg["wd"], lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"],
                                    seed=sd, device=device)
                S[si, te] = CC.predict_dev(m, Xb, device=device)
            print(f"  [{tag}] fold {f} seed {sd} ({len(tr)} rows) {time.time()-t:.1f}s", flush=True)
    return S, fold


def pool_rank(S_rows, qidx):
    """Per-row within-question rank_avg in [0,1] (rows outside any scored question stay 0)."""
    out = np.zeros_like(S_rows, dtype=np.float64)
    for rows, _ in qidx:
        out[rows] = G.rank_avg(S_rows[rows])
    return out


def ens_train(S, qidx, conv):
    """Seed ensemble on the train split, same three conventions as on eval."""
    if conv == "score":
        return S.mean(0)
    if conv == "z":
        Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
        return Z.mean(0)
    return np.mean([pool_rank(s, qidx) for s in S], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--grader_seeds", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    A = ap.parse_args()
    out = {"what": "train-only pre-registration of the readout and combiner",
           "date": "2026-08-05", "device": A.device, "folds": A.folds,
           "head_config_frozen_from": "verifarch_hidden_generatorprompt_2026-08-04.json "
                                      "-> arms.generator.cv_selected (train-only CV)",
           "head_config": CFG, "grader_config": GRADER_CFG,
           "criterion": "within-question selection efficiency on the held-out fold, over "
                        "contested (>=2 distinct candidates) recoverable TRAIN questions",
           "eval_visibility": "none -- no eval label is read by this script"}

    t0 = time.time()
    trg = G.load_candidates("train", "generator", layers=[CFG["layer"]], pooling=(CFG["pooling"],))
    Xg = trg.matrix(CFG["pooling"], CFG["layer"])
    y = np.array([r["y"] for r in trg.rows], np.float32)
    gq = G.group_ids(trg)
    gimg = [r["img_md5"] for r in trg.rows]
    qidx = IL.train_question_index(trg)
    out["train_pool"] = {"rows": int(len(y)), "questions": len(trg.questions),
                         "scorable_questions": len(qidx),
                         "images": len(set(gimg)), "pos_rate": float(y.mean())}
    print(f"train loaded {Xg.shape} in {time.time()-t0:.0f}s, {len(qidx)} scorable questions",
          flush=True)

    seeds = list(range(A.seeds))
    Sg, fold = cv_scores(Xg, y, gq, gimg, CFG, seeds, A.device, A.folds, "gen")
    np.save(os.path.join(IL.PARTS, "prereg_cv_gen.npy"), Sg)
    del Xg

    # ---------------------------------------------------------------- Q1/Q2 readout
    q12 = {}
    for conv in ("rank", "score", "z"):
        for k in (1, 2, 4, 8, 16):
            if k > len(seeds):
                continue
            v = [IL.train_sel_eff(ens_train(Sg[i:i + k], qidx, conv), qidx)
                 for i in range(0, len(seeds) - k + 1, k)]     # disjoint seed blocks
            q12[f"{conv}|k={k}"] = {"cv_sel_eff_mean": float(np.mean(v)),
                                    "cv_sel_eff_blocks": [float(x) for x in v],
                                    "n_blocks": len(v)}
    best = max(q12, key=lambda k: q12[k]["cv_sel_eff_mean"])
    out["Q1_Q2_readout"] = {"grid": q12, "selected": best,
                            "selected_cv_sel_eff": q12[best]["cv_sel_eff_mean"],
                            "single_seed_reference": q12["rank|k=1"]["cv_sel_eff_mean"]}
    print("READOUT selected:", best, q12[best]["cv_sel_eff_mean"], flush=True)

    # ---------------------------------------------------------------- Q3 combiner form
    trr = G.load_candidates("train", "grader", layers=[GRADER_CFG["layer"]],
                            pooling=(GRADER_CFG["pooling"],))
    assert [ (r["ds"], r["idx"], r["na"]) for r in trr.rows ] == \
           [ (r["ds"], r["idx"], r["na"]) for r in trg.rows ], "grader/generator row order differs"
    Xr = trr.matrix(GRADER_CFG["pooling"], GRADER_CFG["layer"])
    Sr, _ = cv_scores(Xr, y, gq, gimg, GRADER_CFG, list(range(A.grader_seeds)), A.device,
                      A.folds, "grader")
    np.save(os.path.join(IL.PARTS, "prereg_cv_grader.npy"), Sr)
    del Xr

    conv_sel = best.split("|")[0]
    k_sel = int(best.split("=")[1])
    a = ens_train(Sg[:k_sel], qidx, conv_sel)
    b = ens_train(Sr[:A.grader_seeds], qidx, conv_sel)
    ra, rb = pool_rank(a, qidx), pool_rank(b, qidx)

    def sel_of(v):
        return IL.train_sel_eff(v, qidx)

    # fold-honest weight/combiner fitting: choose on the other folds, apply to this fold
    qfold = np.array([fold[rows[0]] for rows, _ in qidx])
    ws = np.round(np.arange(0, 1.001, 0.05), 3)

    def fold_fit_weight():
        got = []
        for f in range(A.folds):
            insub = [qidx[i] for i in np.where(qfold != f)[0]]
            outsub = [qidx[i] for i in np.where(qfold == f)[0]]
            sc = [IL.train_sel_eff(w * ra + (1 - w) * rb, insub) for w in ws]
            w = float(ws[int(np.argmax(sc))])
            got.append((IL.train_sel_eff(w * ra + (1 - w) * rb, outsub), len(outsub), w))
        n = sum(g[1] for g in got)
        return float(sum(g[0] * g[1] for g in got) / n), [g[2] for g in got]

    def fold_fit_logistic():
        import torch
        got = []
        for f in range(A.folds):
            inq = [qidx[i] for i in np.where(qfold != f)[0]]
            outq = [qidx[i] for i in np.where(qfold == f)[0]]
            ri = np.concatenate([r for r, _ in inq]) if inq else np.array([], int)
            F = np.stack([ra[ri], rb[ri], ra[ri] * rb[ri]], 1).astype(np.float32)
            yy = y[ri].astype(np.float32)
            w = torch.zeros(3, requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b0], lr=0.05)
            Ft, yt = torch.tensor(F), torch.tensor(yy)
            for _ in range(400):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(Ft @ w + b0, yt)
                loss.backward(); opt.step()
            W = w.detach().numpy(); B = float(b0.detach().numpy()[0])
            v = np.zeros_like(ra)
            allo = np.concatenate([r for r, _ in outq]) if outq else np.array([], int)
            v[allo] = (np.stack([ra[allo], rb[allo], ra[allo] * rb[allo]], 1) @ W) + B
            got.append((IL.train_sel_eff(v, outq), len(outq), W.tolist()))
        n = sum(g[1] for g in got)
        return float(sum(g[0] * g[1] for g in got) / n), [g[2] for g in got]

    wsel, wfolds = fold_fit_weight()
    lsel, lw = fold_fit_logistic()
    q3 = {
        "generator_head_alone": sel_of(a),
        "grader_head_alone": sel_of(b),
        "rank_avg_parameter_free": sel_of(ra + rb),
        "weighted_rank_blend_fold_fit": {"cv_sel_eff": wsel, "weights_per_fold": wfolds},
        "learned_logistic_combiner_fold_fit": {"cv_sel_eff": lsel, "coefs_per_fold": lw},
    }
    cands = {"rank_avg_parameter_free": q3["rank_avg_parameter_free"],
             "weighted_rank_blend": wsel, "learned_logistic_combiner": lsel}
    q3["selected_form"] = max(cands, key=cands.get)
    q3["margin_over_parameter_free"] = float(max(cands.values()) - q3["rank_avg_parameter_free"])
    q3["note"] = ("members here are generator-head + grader-head, because the incumbent LoRA "
                  "verifier has no scores on the train split. Only the FORM is pre-registered; "
                  "a fitted weight cannot transfer to a different pair of members.")
    out["Q3_combiner"] = q3
    print("COMBINER:", json.dumps(cands, default=float), "->", q3["selected_form"], flush=True)

    out["PREREGISTERED_DECISION"] = {
        "ensembling_convention": conv_sel, "n_seeds": k_sel,
        "combiner": q3["selected_form"],
        "headline_definition": "rank_avg( incumbent , {k}-seed {conv} ensemble of the frozen "
                               "generator-frame head )".format(k=k_sel, conv=conv_sel),
    }
    IL.jdump(out, os.path.join(IL.PARTS, "prereg.json"))
    print(json.dumps(out["PREREGISTERED_DECISION"], indent=1), flush=True)


if __name__ == "__main__":
    main()
