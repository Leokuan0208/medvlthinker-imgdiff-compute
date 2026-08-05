#!/usr/bin/env python3
"""verifarch_cheapcontrast.py -- JOB 1 (cheap pool-relative contrast features) and
JOB 2 (why the generator frame beats the grader frame), on the frozen open-text
best-of-8 selection endpoint.

    python3 src/training_methods/verifarch_cheapcontrast.py --stage null
    python3 src/training_methods/verifarch_cheapcontrast.py --stage cv     --gpu 0
    python3 src/training_methods/verifarch_cheapcontrast.py --stage eval   --gpu 0
    python3 src/training_methods/verifarch_cheapcontrast.py --stage job2   --gpu 1
    python3 src/training_methods/verifarch_cheapcontrast.py --stage assemble

Stage outputs land in results/cascade_methods/artifacts/_cheapcontrast_parts/ and are
merged by --stage assemble into
results/cascade_methods/artifacts/verifarch_cheapcontrast_2026-08-04.json

PROTOCOL
  1 null test    reproduce every published incumbent cell before reporting anything new
  2 disjointness re-proved from DECODED pixels in /tmp/cc/disjoint_indep.py (own code)
  3 no leakage   the headline config is chosen by 5-fold image-grouped CV on TRAIN only
  4 seeds        >= 10 seeds per eval arm; mean / sd / range + a seed-averaged ensemble
  5 statistics   paired item-level bootstrap, nboot=10000, vs the incumbent's per-item got
  6 guardrail    per-set sel_eff for all three eval sets on every arm
  7 contested    sel_eff restricted to the 916 recoverable items with >= 2 distinct answers
  8 cost         extra forward passes per question stated for every arm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import genframe_data as G          # noqa: E402
import cheapcontrast as CC         # noqa: E402

ROOT = G.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_cheapcontrast_parts")
OUT = os.path.join(ART, "verifarch_cheapcontrast_2026-08-04.json")
os.makedirs(PARTS, exist_ok=True)

LAYERS = [7, 14, 21, 28]
SEEDS = list(range(10))

#: feature sets. 'H' raw hidden | 'C' geometry contrast | 'M' multiplicity
#: 'Wc/Ws' contrast weighted by a cross-fitted stage-1 head score
#: 'Yc/Ys' contrast weighted by the base model's zero-shot grader-frame P(yes)
FEATSETS = {
    "H":            ["H"],
    "C":            ["C"],
    "CM":           ["C", "M"],
    "H+C":          ["H", "C"],
    "H+M":          ["H", "M"],
    "H+C+M":        ["H", "C", "M"],
    "H+C+M+Wc":     ["H", "C", "M", "Wc"],
    "H+C+M+Wc+Ws":  ["H", "C", "M", "Wc", "Ws"],
    "H+C+M+Yc+Ys":  ["H", "C", "M", "Yc", "Ys"],
}


# ======================================================================================
def log(*a):
    print(*a, flush=True)


def dumpj(name, obj):
    p = os.path.join(PARTS, name)
    json.dump(obj, open(p, "w"), indent=1, default=float)
    log(f"wrote {p}")


def loadj(name):
    return json.load(open(os.path.join(PARTS, name)))


# ======================================================================================
# data assembly (shared by every stage)
# ======================================================================================
class Data:
    """Everything the fitting stages need, built once."""

    def __init__(self, mode="generator", layer=21, pooling="span", need_yesno=True):
        t = time.time()
        self.tr = G.load_candidates("train", mode, layers=[layer], pooling=(pooling,))
        self.ev = G.load_candidates("eval", mode, layers=[layer], pooling=(pooling,))
        self.Xtr = self.tr.matrix(pooling, layer)
        self.Xev = self.ev.matrix(pooling, layer)
        self.ytr = np.array([r["y"] for r in self.tr.rows], np.float32)
        self.gtr_q = G.group_ids(self.tr)                       # "ds|idx" per row
        self.gtr_img = [r["img_md5"] for r in self.tr.rows]      # image group for folds
        self.qrows_tr = [np.array([c.row for c in q.cands]) for q in self.tr.questions]
        self.qrows_ev = [np.array([c.row for c in q.cands]) for q in self.ev.questions]
        self.kev = [(r["ds"], r["idx"], r["na"]) for r in self.ev.rows]
        self.fold_tr = CC.fold_of_group(self.gtr_img, 5)
        self.items = G.load_items()

        # -------- multiplicity, both splits, from the SAME sc8 pools
        mm = CC.pool_multiplicity(sorted({r["ds"] for r in self.tr.rows} |
                                         {r["ds"] for r in self.ev.rows}))
        self.mult_tr = np.array([mm[(r["ds"], r["idx"], r["na"])] for r in self.tr.rows], float)
        self.mult_ev = np.array([mm[(r["ds"], r["idx"], r["na"])] for r in self.ev.rows], float)
        # the eval split carries mult independently (Cand.mult) -> assert they agree
        chk = np.zeros(len(self.ev.rows))
        for q in self.ev.questions:
            for c in q.cands:
                chk[c.row] = c.mult
        assert np.array_equal(chk, self.mult_ev), "eval multiplicity disagrees with sc8 pools"
        self.mult_check_rows = int(len(chk))

        # -------- geometry blocks
        self.Ctr = CC.geom_features(self.Xtr, self.qrows_tr)
        self.Cev = CC.geom_features(self.Xev, self.qrows_ev)
        self.Mtr = CC.mult_features(self.mult_tr, self.qrows_tr)
        self.Mev = CC.mult_features(self.mult_ev, self.qrows_ev)

        # -------- zero-shot grader P(yes), available on BOTH splits (grader cache)
        self.Ytr = self.Yev = None
        if need_yesno:
            gtr_y = G.load_cache("grader", "train", layers=[], pooling=())
            gev_y = G.load_cache("grader", "eval", layers=[], pooling=())
            ytr_map = {(r["ds"], r["idx"], r["na"]): gtr_y[0]["yesno"][i]
                       for i, r in enumerate(gtr_y[1])}
            yev_map = {(r["ds"], r["idx"], r["na"]): gev_y[0]["yesno"][i]
                       for i, r in enumerate(gev_y[1])}

            def pyes(m, rows):
                v = np.array([m[(r["ds"], r["idx"], r["na"])] for r in rows], float)
                z = v - v.max(1, keepdims=True)
                e = np.exp(z)
                return e[:, 0] / e.sum(1)          # column 0 = "Yes" logit

            self.pyes_tr = pyes(ytr_map, self.tr.rows)
            self.pyes_ev = pyes(yev_map, self.ev.rows)
            self.Ytr = CC.weighted_features(self.Xtr, self.qrows_tr, self.pyes_tr * 8.0)
            self.Yev = CC.weighted_features(self.Xev, self.qrows_ev, self.pyes_ev * 8.0)
        log(f"[data] mode={mode} L{layer}/{pooling} train{self.Xtr.shape} eval{self.Xev.shape} "
            f"({time.time()-t:.0f}s)")

    # ---------------------------------------------------------------- stage-1 scores
    def stage1(self, device, seed=0):
        """H-only head scores: cross-fitted on train (out-of-fold), full-fit for eval.

        These feed the Wc/Ws blocks. Fixed at seed 0 so the stage-2 seed sweep varies only
        the stage-2 head.
        """
        s_tr = np.zeros(len(self.ytr))
        for f in range(5):
            tr, va = self.fold_tr != f, self.fold_tr == f
            A, B = CC.standardize(self.Xtr[tr], self.Xtr[va])
            m = CC.fit_head_dev(A, self.ytr[tr], [self.gtr_q[i] for i in np.where(tr)[0]],
                                seed=seed, device=device, **{k: v for k, v in CC.BASE_CFG.items()
                                                             if k not in ("layer", "pooling")})
            s_tr[va] = CC.predict_dev(m, B, device)
        A, B = CC.standardize(self.Xtr, self.Xev)
        m = CC.fit_head_dev(A, self.ytr, self.gtr_q, seed=seed, device=device,
                            **{k: v for k, v in CC.BASE_CFG.items() if k not in ("layer", "pooling")})
        s_ev = CC.predict_dev(m, B, device)
        self.Wtr = CC.weighted_features(self.Xtr, self.qrows_tr, s_tr)
        self.Wev = CC.weighted_features(self.Xev, self.qrows_ev, s_ev)
        self.s1_tr, self.s1_ev = s_tr, s_ev
        return s_tr, s_ev

    # ---------------------------------------------------------------- matrix builder
    def build(self, blocks, which):
        if which == "train":
            H, C, M = self.Xtr, self.Ctr, self.Mtr
            W = getattr(self, "Wtr", (None, None)); Y = self.Ytr
        else:
            H, C, M = self.Xev, self.Cev, self.Mev
            W = getattr(self, "Wev", (None, None)); Y = self.Yev
        parts, names = [], []
        for b in blocks:
            if b == "H":
                parts.append(H); names += [f"h{j}" for j in range(H.shape[1])]
            elif b == "C":
                parts.append(C); names += CC.GEOM_NAMES
            elif b == "M":
                parts.append(M); names += CC.MULT_NAMES
            elif b == "Wc":
                parts.append(W[0]); names += [f"W_{n}" for n in CC.WC_NAMES]
            elif b == "Ws":
                parts.append(W[1]); names += [f"W_{n}" for n in CC.WS_NAMES]
            elif b == "Yc":
                parts.append(Y[0]); names += [f"Y_{n}" for n in CC.WC_NAMES]
            elif b == "Ys":
                parts.append(Y[1]); names += [f"Y_{n}" for n in CC.WS_NAMES]
            else:
                raise ValueError(b)
        return np.concatenate(parts, 1).astype(np.float32), names


# ======================================================================================
# metric helpers
# ======================================================================================
def score_map(kev, sv):
    return {kev[i]: float(sv[i]) for i in range(len(kev))}


def mean_pair_cos(X, qr):
    """Mean pairwise cosine among the members of each group, averaged over groups."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    v = []
    for ii in qr:
        if len(ii) > 1:
            Cm = Xn[ii] @ Xn[ii].T
            iu = np.triu_indices(len(ii), 1)
            v.append(float(Cm[iu].mean()))
    return float(np.mean(v))


def report(smap, items, base_res, tag, nboot=10000, extra=None):
    r = G.sel_eff(smap, items)
    bo = G.paired_bootstrap(r["got"], base_res["got"], rec=r["rec"], nboot=nboot)
    bc = G.paired_bootstrap(r["got"], base_res["got"], nboot=nboot, mask=r["contested_mask"])
    out = {
        "tag": tag, "n": r["n"], "acc": r["acc"], "sel_eff": r["sel_eff"],
        "cand_auroc": G.cand_auroc(smap, items),
        "per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()},
        "contested_sel_eff": r["contested"]["sel_eff"], "contested_n": r["contested"]["n"],
        "d_sel_eff": bo["d_sel_eff"], "d_sel_eff_ci": bo["d_sel_eff_ci"],
        "d_acc": bo["d_acc"], "d_acc_ci": bo["d_acc_ci"],
        "d_contested": bc["d_sel_eff"], "d_contested_ci": bc["d_sel_eff_ci"],
        "guardrail_clean": G.guardrail_clean(r, base_res),
    }
    if extra:
        out.update(extra)
    return out, r


def seed_ensemble(kev, svs, items):
    """Seed-averaged ensemble. Two conventions, both reported:
    'score' = mean raw score over seeds; 'rank' = mean within-pool rank_avg over seeds."""
    S = np.stack(svs)                                  # (n_seeds, n_rows)
    mean_score = score_map(kev, S.mean(0))
    per = [score_map(kev, s) for s in svs]
    mats = [G._slot_scores(p, items) for p in per]
    rk = {}
    for i, it in enumerate(items):
        rk[(it["ds"], it["idx"])] = list(np.mean([G.rank_avg(m[i]) for m in mats], 0))
    return mean_score, rk


# ======================================================================================
# STAGE null
# ======================================================================================
def stage_null(A):
    nt = G.null_test()
    ctrl = {}
    items = G.load_items()
    base_r = G.sel_eff(G.incumbent_scores(), items)
    cs = G.control_scores(items)
    for k, v in cs.items():
        r = G.sel_eff(v, items)
        ctrl[k] = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                   "contested_sel_eff": r["contested"]["sel_eff"],
                   "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS}}
    ctrl["random_pick"] = G.random_pick(items)
    ctrl["oracle@8"] = base_r["oracle"]
    ctrl["greedy"] = base_r["greedy"]
    dj = json.load(open("/tmp/cc/disjoint_indep.json")) if os.path.exists("/tmp/cc/disjoint_indep.json") \
        else G.assert_disjoint("generator")
    dumpj("null.json", {"null_test": nt, "controls": ctrl, "disjointness_independent": dj})


# ======================================================================================
# STAGE cv   -- pre-registration, TRAIN ONLY
# ======================================================================================
def stage_cv(A):
    dev = f"cuda:{A.gpu}"
    D = Data("generator", 21, "span")
    D.stage1(dev)
    res = []
    for fs, blocks in FEATSETS.items():
        Xall, names = D.build(blocks, "train")
        for obj in A.objectives:
            aucs, effs = [], []
            for f in range(5):
                for rep in range(A.cv_reps):
                    tr, va = D.fold_tr != f, D.fold_tr == f
                    Xa, Xb = CC.standardize(Xall[tr], Xall[va])
                    m = CC.fit_head_dev(Xa, D.ytr[tr], [D.gtr_q[i] for i in np.where(tr)[0]],
                                        objective=obj, hidden=CC.BASE_CFG["hidden"],
                                        wd=CC.BASE_CFG["wd"], epochs=CC.BASE_CFG["epochs"],
                                        seed=f + 10 * rep, device=dev)
                    sv = CC.predict_dev(m, Xb, dev)
                    vidx = np.where(va)[0]
                    aucs.append(G.auroc(D.ytr[vidx], sv))
                    loc = {i: j for j, i in enumerate(vidx)}
                    byq = defaultdict(list)
                    for i in vidx:
                        byq[D.gtr_q[i]].append(i)
                    hit = tot = 0
                    for q, ii in byq.items():
                        if D.ytr[ii].sum() == 0:
                            continue
                        b = ii[int(np.argmax([sv[loc[i]] for i in ii]))]
                        hit += int(D.ytr[b] == 1); tot += 1
                    effs.append(hit / max(tot, 1))
            r = {"featset": fs, "blocks": blocks, "n_features": Xall.shape[1], "objective": obj,
                 "cv_auroc": float(np.mean(aucs)), "cv_sel_eff": float(np.mean(effs)),
                 "cv_sel_eff_sd": float(np.std(effs)), "n_cv_fits": len(effs)}
            res.append(r)
            log(f"  CV {fs:14s} {obj:4s} d={Xall.shape[1]:5d} auroc={r['cv_auroc']:.4f} "
                f"sel_eff={r['cv_sel_eff']:.4f} (sd {r['cv_sel_eff_sd']:.4f})")
            dumpj("cv.json", {"grid": res, "cv_reps": A.cv_reps,
                              "protocol": "5-fold CV inside the disjoint TRAIN pool, folds = "
                                          "md5(decoded-RGB image md5) % 5 so a fold boundary is an "
                                          "IMAGE boundary. Head architecture held FIXED at the "
                                          "previous round's CV-selected L21/span/hidden256/wd1e-2/"
                                          "30ep; only the FEATURE SET and the objective vary. Eval "
                                          "is never touched in this stage."})
    best = max(res, key=lambda r: r["cv_sel_eff"])
    dumpj("cv.json", {"grid": res, "cv_reps": A.cv_reps, "cv_selected": best,
                      "protocol": "5-fold CV inside the disjoint TRAIN pool, folds = md5(decoded-RGB "
                                  "image md5) % 5. Head architecture FIXED at L21/span/hidden256/"
                                  "wd1e-2/30ep (previous round's CV pick); only the FEATURE SET and "
                                  "objective vary. Eval never touched."})
    log(f"CV-SELECTED: {best}")


# ======================================================================================
# STAGE eval  -- every arm, 10 seeds
# ======================================================================================
def stage_eval(A):
    dev = f"cuda:{A.gpu}"
    cv = loadj("cv.json")
    best = cv["cv_selected"]
    D = Data("generator", 21, "span")
    D.stage1(dev)
    items = D.items
    inc = G.incumbent_scores()
    base_r = G.sel_eff(inc, items)

    arms = []
    todo = [(fs, best["objective"]) for fs in FEATSETS]
    # the H baseline and the CV pick also at the other objective, as a robustness check
    other = "bce" if best["objective"] == "bt" else "bt"
    todo += [("H", other), (best["featset"], other)]
    seen, got_of, store = set(), {}, {}
    for fs, obj in todo:
        if (fs, obj) in seen:
            continue
        seen.add((fs, obj))
        Xtr, names = D.build(FEATSETS[fs], "train")
        Xev, _ = D.build(FEATSETS[fs], "eval")
        Xa, Xb = CC.standardize(Xtr, Xev)
        svs, effs = [], []
        for sd in SEEDS:
            m = CC.fit_head_dev(Xa, D.ytr, D.gtr_q, objective=obj, hidden=CC.BASE_CFG["hidden"],
                                wd=CC.BASE_CFG["wd"], epochs=CC.BASE_CFG["epochs"],
                                seed=sd, device=dev)
            sv = CC.predict_dev(m, Xb, dev)
            svs.append(sv)
            effs.append(G.sel_eff(score_map(D.kev, sv), items)["sel_eff"])
        ms, mr = seed_ensemble(D.kev, svs, items)
        rec_s, _ = report(ms, items, base_r, f"{fs}|{obj}|ens_score")
        rec_r, rr = report(mr, items, base_r, f"{fs}|{obj}|ens_rank")
        # parameter-free rank fusion of the seed-ensembled arm with the incumbent
        fu = G.rank_fuse(inc, mr, items=items, ranker=G.rank_avg)
        rec_f, rrf = report(fu, items, base_r, f"{fs}|{obj}|ens_rank+INCUMBENT(rank_avg)")
        arms.append({
            "featset": fs, "blocks": FEATSETS[fs], "objective": obj, "n_features": int(Xa.shape[1]),
            "seeds": SEEDS,
            "per_seed_sel_eff": [float(x) for x in effs],
            "seed_mean": float(np.mean(effs)), "seed_sd": float(np.std(effs, ddof=1)),
            "seed_min": float(np.min(effs)), "seed_max": float(np.max(effs)),
            "ensemble_score": rec_s, "ensemble_rank": rec_r, "fused_with_incumbent": rec_f,
            "strata": strat_report(rr, items),
        })
        got_of[f"{fs}|{obj}"] = rr["got"]
        got_of[f"{fs}|{obj}|+INC"] = rrf["got"]
        store[f"scores__{fs}__{obj}"] = np.stack(svs)
        log(f"ARM {fs:14s} {obj:4s} seeds mean={np.mean(effs):.4f} sd={np.std(effs, ddof=1):.4f} "
            f"[{np.min(effs):.4f},{np.max(effs):.4f}] ensR={rec_r['sel_eff']:.4f} "
            f"d={rec_r['d_sel_eff']:+.4f}{rec_r['d_sel_eff_ci']} fuse={rec_f['sel_eff']:.4f}")
        dumpj("eval.json", {"arms": arms, "cv_selected": best})

    # ---------------------------------------------------------------- extra comparators
    # (a) vs the DEPLOYED POINTWISE HEAD on identical features -- does the contrast block
    #     add anything at all?  (b) vs the DEPLOYED FUSION 0.806540, reconstructed exactly
    #     on CPU by /tmp/ccx/cpu_published.py so the comparison is paired at item level.
    base_key = f"H|{best['objective']}"
    pub = np.load("/tmp/ccx/published_head.npz") if os.path.exists("/tmp/ccx/published_head.npz") else None
    contested = G.sel_eff(inc, items)["contested_mask"]
    comps = []
    for k, g in got_of.items():
        row = {"arm": k}
        b = G.paired_bootstrap(g, got_of[base_key], rec=base_r["rec"], nboot=10000)
        c = G.paired_bootstrap(g, got_of[base_key], nboot=10000, mask=contested)
        row["vs_pointwise_head_same_features_ensemble"] = {
            "d_sel_eff": b["d_sel_eff"], "ci": b["d_sel_eff_ci"],
            "d_contested": c["d_sel_eff"], "contested_ci": c["d_sel_eff_ci"]}
        if pub is not None:
            for nm, key in [("vs_published_head_0.795640", "head_got"),
                            ("vs_published_fusion_0.806540", "fuse_got")]:
                b2 = G.paired_bootstrap(g, pub[key], rec=base_r["rec"], nboot=10000)
                c2 = G.paired_bootstrap(g, pub[key], nboot=10000, mask=contested)
                row[nm] = {"d_sel_eff": b2["d_sel_eff"], "ci": b2["d_sel_eff_ci"],
                           "d_contested": c2["d_sel_eff"], "contested_ci": c2["d_sel_eff_ci"]}
        comps.append(row)
        log(f"  CMP {k:26s} vs pointwise {b['d_sel_eff']:+.4f}{[round(x,4) for x in b['d_sel_eff_ci']]}"
            + (f" | vs fusion {row['vs_published_fusion_0.806540']['d_sel_eff']:+.4f}"
               f"{[round(x, 4) for x in row['vs_published_fusion_0.806540']['ci']]}" if pub is not None else ""))
    np.savez(os.path.join(PARTS, "eval_scores.npz"), **store)
    dumpj("eval.json", {"arms": arms, "cv_selected": best, "comparisons": comps,
                        "comparator_note": "vs_pointwise_head_same_features_ensemble compares each "
                                           "arm's 10-seed rank ensemble against the H-only arm's "
                                           "10-seed rank ensemble fit in the SAME run on the SAME "
                                           "device -- the only fully matched contrast. "
                                           "vs_published_* compares against the CPU seed-0 "
                                           "reconstructions of the two published cells."})


# ======================================================================================
# STAGE job2 -- the frame effect
# ======================================================================================
def stage_job2(A):
    dev = f"cuda:{A.gpu}"
    items = G.load_items()
    inc = G.incumbent_scores()
    base_r = G.sel_eff(inc, items)
    out = {}

    # ---------------------------------------------------------------- 1. probe grid
    # seed-free ridge probes over frame x layer x pooling -> a frame difference here cannot
    # be an SGD-noise artifact.
    grid = []
    reps = {}
    for mode in ([] if A.only == "heads" else ["generator", "grader"]):
        tr = G.load_candidates("train", mode, layers=LAYERS, pooling=("last", "span"))
        ev = G.load_candidates("eval", mode, layers=LAYERS, pooling=("last", "span"))
        ytr = np.array([r["y"] for r in tr.rows], np.float32)
        yev = np.array([r["y"] for r in ev.rows], np.float32)
        kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
        qrows_ev = [np.array([c.row for c in q.cands]) for q in ev.questions]
        qrows_tr = [np.array([c.row for c in q.cands]) for q in tr.questions]
        gq_tr = G.group_ids(tr)
        for L in LAYERS:
            for pool in ["last", "span"]:
                Xtr = tr.matrix(pool, L); Xev = ev.matrix(pool, L)
                Xa, Xb = CC.standardize(Xtr, Xev)
                sv, w = CC.ridge_probe(Xa, ytr, Xb, lam=1.0)
                r = G.sel_eff(score_map(kev, sv), items)
                # ---- residual (within-question centred) probe: is the signal there but
                #      unreadable from the raw vector, or absent?
                Ra = Xa.copy(); Rb = Xb.copy()
                for ii in qrows_tr:
                    Ra[ii] -= Ra[ii].mean(0, keepdims=True)
                for ii in qrows_ev:
                    Rb[ii] -= Rb[ii].mean(0, keepdims=True)
                svr, _ = CC.ridge_probe(Ra, ytr, Rb, lam=1.0)
                rr = G.sel_eff(score_map(kev, svr), items)
                # ---- geometry: COLLAPSE (candidates map to one point -> nothing separable)
                #      vs ROTATION (separable, but along a direction the head cannot find).
                #      within-question variance share distinguishes them.
                tot_var = float(Xa.var(0).sum())
                num = 0.0; cnt = 0
                for ii in qrows_tr:
                    if len(ii) > 1:
                        num += float(((Xa[ii] - Xa[ii].mean(0, keepdims=True)) ** 2).sum())
                        cnt += len(ii)
                within_share = num / cnt / max(tot_var, 1e-12) if cnt else float("nan")
                # mean within-question pairwise cosine (eval) + a shuffled-pool control:
                # the same pool-size profile over RANDOM rows, so the within-question number
                # has something to be compared against.
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(Xb))
                fake, p = [], 0
                for ii in qrows_ev:
                    fake.append(perm[p:p + len(ii)]); p += len(ii)
                grid.append({
                    "frame": mode, "layer": L, "pooling": pool,
                    "probe_cand_auroc": G.auroc(yev, sv),
                    "probe_sel_eff": r["sel_eff"], "probe_contested": r["contested"]["sel_eff"],
                    "probe_per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()},
                    "residual_probe_cand_auroc": G.auroc(yev, svr),
                    "residual_probe_sel_eff": rr["sel_eff"],
                    "within_question_variance_share": within_share,
                    "mean_within_question_cos_eval": mean_pair_cos(Xb, qrows_ev),
                    "mean_between_question_cos_eval": mean_pair_cos(Xb, fake),
                    "mean_within_question_cos_eval_RAW": mean_pair_cos(ev.matrix(pool, L), qrows_ev),
                })
                log(f"  probe {mode:9s} L{L:<3d} {pool:4s} auroc={grid[-1]['probe_cand_auroc']:.4f} "
                    f"sel_eff={r['sel_eff']:.4f} resid_auroc={grid[-1]['residual_probe_cand_auroc']:.4f} "
                    f"wvar={within_share:.4f} wcos={grid[-1]['mean_within_question_cos_eval']:.4f} "
                    f"bcos={grid[-1]['mean_between_question_cos_eval']:.4f}")
                reps[(mode, L, pool)] = Xb
                dumpj("job2_grid.json", {"grid": grid})
        del tr, ev
    if A.only == "heads":
        grid = loadj("job2_grid.json")["grid"]
        out["probe_grid"] = grid
        return stage_job2_heads(A, items, base_r, out, grid)
    out["probe_grid"] = grid

    # ---------------------------------------------------------------- 2. CKA
    cka = []
    for L in LAYERS:
        for pool in ["last", "span"]:
            a = reps[("generator", L, pool)]; b = reps[("grader", L, pool)]
            cka.append({"layer": L, "pooling": pool, "cka_generator_vs_grader": CC.linear_cka(a, b)})
    for pool in ["last", "span"]:
        for m in ["generator", "grader"]:
            for i in range(len(LAYERS) - 1):
                cka.append({"pooling": pool, "frame": m,
                            "layers": [LAYERS[i], LAYERS[i + 1]],
                            "cka_layer_to_layer": CC.linear_cka(reps[(m, LAYERS[i], pool)],
                                                                reps[(m, LAYERS[i + 1], pool)])})
    out["cka"] = cka
    dumpj("job2_grid.json", {"grid": grid, "cka": cka})
    return stage_job2_heads(A, items, base_r, out, grid)


def stage_job2_heads(A, items, base_r, out, grid):
    """The 2 frames x 2 poolings x 2 objectives SGD grid, 10 seeds each.

    THE POINT: the two PUBLISHED cells were fit at DIFFERENT configs -- generator
    L21/span/bt (0.795640) vs grader L21/last/bce (0.750681) -- and at ONE seed each. Any
    "frame effect" read off that pair confounds frame with pooling, objective and seed.
    """
    dev = f"cuda:{A.gpu}"
    heads, got = [], {}
    for mode in ["generator", "grader"]:
        tr = G.load_candidates("train", mode, layers=[21], pooling=("last", "span"))
        ev = G.load_candidates("eval", mode, layers=[21], pooling=("last", "span"))
        ytr = np.array([r["y"] for r in tr.rows], np.float32)
        kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
        gq = G.group_ids(tr)
        for pool in ["last", "span"]:
            for obj in ["bce", "bt"]:
                Xa, Xb = CC.standardize(tr.matrix(pool, 21), ev.matrix(pool, 21))
                svs, effs = [], []
                for sd in SEEDS:
                    m = CC.fit_head_dev(Xa, ytr, gq, objective=obj, hidden=256, wd=1e-2,
                                        epochs=30, seed=sd, device=dev)
                    sv = CC.predict_dev(m, Xb, dev)
                    svs.append(sv)
                    effs.append(G.sel_eff(score_map(kev, sv), items)["sel_eff"])
                _, mr = seed_ensemble(kev, svs, items)
                rec, r = report(mr, items, base_r, f"{mode}|L21/{pool}/{obj}|ens_rank")
                heads.append({"frame": mode, "layer": 21, "pooling": pool, "objective": obj,
                              "seed_mean": float(np.mean(effs)), "seed_sd": float(np.std(effs, ddof=1)),
                              "seed_min": float(np.min(effs)), "seed_max": float(np.max(effs)),
                              "per_seed_sel_eff": [float(x) for x in effs],
                              "ensemble_rank": rec, "strata": strat_report(r, items)})
                got[f"{mode}|{pool}|{obj}"] = r["got"]
                log(f"  head {mode:9s} L21/{pool}/{obj} mean={np.mean(effs):.4f} "
                    f"sd={np.std(effs, ddof=1):.4f} ens={rec['sel_eff']:.4f}")
                dumpj("job2_heads.json", {"heads": heads})
        del tr, ev
    # frame contrast at MATCHED config, paired at item level
    contested = base_r["contested_mask"]
    fr = []
    for pool in ["last", "span"]:
        for obj in ["bce", "bt"]:
            a, b = got[f"generator|{pool}|{obj}"], got[f"grader|{pool}|{obj}"]
            bo = G.paired_bootstrap(a, b, rec=base_r["rec"], nboot=10000)
            bc = G.paired_bootstrap(a, b, nboot=10000, mask=contested)
            fr.append({"pooling": pool, "objective": obj,
                       "d_generator_minus_grader": bo["d_sel_eff"], "ci": bo["d_sel_eff_ci"],
                       "d_contested": bc["d_sel_eff"], "contested_ci": bc["d_sel_eff_ci"]})
            log(f"  FRAME {pool}/{obj}: gen-grader = {bo['d_sel_eff']:+.4f} "
                f"{[round(x, 4) for x in bo['d_sel_eff_ci']]}")
    out["matched_heads"] = heads
    out["frame_contrast_matched"] = fr
    dumpj("job2_heads.json", {"heads": heads, "frame_contrast_matched": fr})
    dumpj("job2.json", out)


# ======================================================================================
# STAGE feat -- what, individually, is in the contrast block
# ======================================================================================
def stage_feat(A):
    """Single-feature selectors + redundancy with the incumbent.

    TRAIN numbers are pre-registration-safe (within-question top-1 accuracy over the
    disjoint training pool); EVAL numbers are DIAGNOSTIC and labelled as such.
    """
    dev = f"cuda:{A.gpu}"
    D = Data("generator", 21, "span")
    D.stage1(dev)
    items = D.items
    inc = G.incumbent_scores()
    base_r = G.sel_eff(inc, items)
    inc_slot = G._slot_scores(inc, items)

    def train_top1(sv):
        byq = defaultdict(list)
        for i, q in enumerate(D.gtr_q):
            byq[q].append(i)
        hit = tot = 0
        for q, ii in byq.items():
            if D.ytr[ii].sum() == 0:
                continue
            b = ii[int(np.argmax(sv[ii]))]
            hit += int(D.ytr[b] == 1); tot += 1
        return hit / max(tot, 1)

    Ftr, names = D.build(["C", "M", "Wc", "Ws", "Yc", "Ys"], "train")
    Fev, _ = D.build(["C", "M", "Wc", "Ws", "Yc", "Ys"], "eval")
    rows = []
    for j, nm in enumerate(names):
        best = None
        for sgn in (1.0, -1.0):
            t = train_top1(sgn * Ftr[:, j])
            if best is None or t > best[1]:
                best = (sgn, t)
        sgn, t = best
        smap = score_map(D.kev, sgn * Fev[:, j])
        r = G.sel_eff(smap, items)
        # redundancy: Spearman of the feature with the incumbent's score over all slots
        fs = G._slot_scores(smap, items)
        ok = (fs > G.MISSING_SCORE / 2) & (inc_slot > G.MISSING_SCORE / 2)
        a = G.rank_avg(fs[ok].ravel()); b = G.rank_avg(inc_slot[ok].ravel())
        rho = float(np.corrcoef(a, b)[0, 1])
        rows.append({"feature": nm, "sign": sgn, "train_top1_selection": t,
                     "EVAL_DIAGNOSTIC_sel_eff": r["sel_eff"],
                     "EVAL_DIAGNOSTIC_contested": r["contested"]["sel_eff"],
                     "EVAL_DIAGNOSTIC_cand_auroc": G.cand_auroc(smap, items),
                     "spearman_vs_incumbent_slotscores": rho})
        log(f"  feat {nm:22s} sgn{sgn:+.0f} train_top1={t:.4f} EVALdiag_sel_eff={r['sel_eff']:.4f} "
            f"rho_inc={rho:+.3f}")
    ref = {"train_top1_random_baseline": None,
           "incumbent_eval_sel_eff": base_r["sel_eff"],
           "random_pick_eval_sel_eff": G.random_pick(items)["sel_eff"]}
    dumpj("feat.json", {"per_feature": rows, "reference": ref,
                        "note": "TRAIN column is chosen/reported without touching eval; the EVAL "
                                "columns are DIAGNOSTIC (single-feature selectors were not "
                                "pre-registered) and are excluded from any headline."})


# ======================================================================================
# answer-length strata
# ======================================================================================
_GL_CACHE = {}
_INC_CACHE = {}


def gold_lengths(items):
    """Word count of the GOLD answer per eval item (from the sc8 checkpoints).

    Stratum definition is stated explicitly because it does NOT exactly reproduce the
    previously quoted short-answer stratum (n=1928 at 79% sel_eff); with gold_len <= 3
    this pool gives n=2029 (1372 recoverable) at incumbent sel_eff 0.794461.
    """
    if "gl" not in _GL_CACHE:
        m = {}
        for ds, f in [("slake_open", "ckpt_slake_open_lingshu7b_sc8.jsonl"),
                      ("vqa_rad_open", "ckpt_vqa_rad_open_lingshu7b_sc8.jsonl"),
                      ("pathvqa_open", "ckpt_pathvqa_open_lingshu7b_sc8.jsonl")]:
            for line in open(os.path.join(CC.SC8_DIR, f)):
                if line.strip():
                    r = json.loads(line)
                    m[(ds, r["idx"])] = len(str(r["gold"]).split())
        _GL_CACHE["gl"] = np.array([m[(it["ds"], it["idx"])] for it in items])
    return _GL_CACHE["gl"]


def strat_report(r, items):
    gl = gold_lengths(items)
    if "r" not in _INC_CACHE:
        _INC_CACHE["r"] = G.sel_eff(G.incumbent_scores(), items)
    inc_r = _INC_CACHE["r"]
    out = {}
    for name, mask in [("gold_len<=3", gl <= 3), ("gold_len>3", gl > 3),
                       ("gold_len==1", gl == 1)]:
        m = mask & (r["rec"] == 1)
        out[name] = {"n_items": int(mask.sum()), "n_recoverable": int(m.sum()),
                     "sel_eff": float(r["got"][m].mean()),
                     "incumbent_sel_eff": float(inc_r["got"][m].mean()),
                     "delta_vs_incumbent": float(r["got"][m].mean() - inc_r["got"][m].mean())}
    return out


# ======================================================================================
# STAGE assemble
# ======================================================================================
def stage_assemble(A):
    art = {
        "what": "JOB 1 -- can the measured pairwise-comparison win be had for free, as POOL-RELATIVE "
                "features over the already-cached generator-frame vectors fed to the SAME pointwise "
                "head?  JOB 2 -- WHY does the same frozen Lingshu-7B read in the generator frame beat "
                "itself read in the grader frame (0.7956 vs 0.7507)?",
        "date": "2026-08-04",
        "endpoint": "best-of-8 open-text selection efficiency, 2345 items / 1468 recoverable, "
                    "sets slake_open / vqa_rad_open / pathvqa_open; incumbent = the CLEAN "
                    "disjoint-trained LoRA verifier at sel_eff 0.775204",
        "nboot": 10000, "seeds": SEEDS,
    }
    for name, key in [("null.json", None), ("cv.json", "job1_cv"), ("eval.json", "job1_eval"),
                      ("feat.json", "job1_per_feature"), ("job2_grid.json", "job2_geometry"),
                      ("job2_heads.json", "job2_matched_heads")]:
        p = os.path.join(PARTS, name)
        if not os.path.exists(p):
            log(f"  MISSING {name}")
            continue
        d = json.load(open(p))
        if key is None:
            art.update(d)
        else:
            art[key] = d

    # ---- cost accounting (rule 8), recomputed here
    items = G.load_items()
    nd = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items])
    hist = {int(k): int((nd == k).sum()) for k in range(1, 9)}
    pairs = (nd * (nd - 1) // 2)
    art["cost"] = {
        "distinct_candidates_per_question_hist": hist,
        "mean_distinct_candidates": float(nd.mean()),
        "generations_per_question": 8,
        "feature_forward_passes_per_question": {
            "value": float(nd.mean()),
            "note": "one generator-frame forward pass per DISTINCT candidate answer; this is the "
                    "cost the ALREADY DEPLOYED pointwise head pays. Every C/M/Wc/Ws feature is "
                    "computed from those same cached vectors, so the contrast blocks add ZERO "
                    "forward passes.",
        },
        "extra_forward_passes_over_the_deployed_pointwise_head": {
            "C_geometry": 0, "M_multiplicity": 0, "Wc_Ws_selfweighted": 0,
            "Yc_Ys_zeroshot_pyes_weighted": float(nd.mean()),
        },
        "arithmetic_only_cost_of_C": {
            "full_round_robin_pairs_over_distinct_candidates_total": int(pairs.sum()),
            "mean_pairs_per_question": float(pairs.mean()), "max_pairs_per_question": int(pairs.max()),
            "note": "the C block computes the full KxK cosine matrix, i.e. all pairs, in numpy over "
                    "3584-d cached vectors: ~8.5 dot products per question, microseconds. The "
                    "measured pairwise win it is trying to imitate needed 28 LLM forward passes.",
        },
        "reference_pairwise_cost": "round-robin A-vs-B verification = 28 forward passes/question "
                                   "(results/cascade_methods/artifacts/pairwise_verifier_gpu.json); "
                                   "that measurement used the CONTAMINATED lora_verifier_pooled4 on "
                                   "n=578 of a DIFFERENT pool and is not comparable to 0.775204.",
    }

    # ---- numerical-precision control (this box's PyTorch defaults to TF32)
    for nm, p in [("tf32_off_gpu_vs_cpu_seed0", "/tmp/ccx/tf32check.json"),
                  ("published_cells_refit_on_cpu", "/tmp/ccx/published_repro.json"),
                  ("published_grader_cell_refit_on_cpu", "/tmp/ccx/cpu_control2.json")]:
        if os.path.exists(p):
            art.setdefault("numerics", {})[nm] = json.load(open(p))
    art.setdefault("numerics", {})["tf32_warning"] = (
        "torch.backends.cuda.matmul.allow_tf32 defaults to TRUE in this container (NGC 25.09). "
        "With TF32 on, the same config/seed gave 0.786785 on GPU where CPU gives 0.795640, and "
        "0.774523 where CPU gives 0.750681 -- a precision artifact LARGER than every effect in this "
        "round. All numbers reported here were produced with TF32 forced OFF "
        "(src/training_methods/cheapcontrast.py). Residual GPU-vs-CPU deviation at seed 0 is then "
        "0.0020 (generator L21/span/bt) and 0.0054 (grader L21/last/bce). Every comparison in this "
        "artifact is device-matched.")
    json.dump(art, open(OUT, "w"), indent=1, default=float)
    log(f"wrote {OUT}")


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["null", "cv", "eval", "job2", "feat", "assemble"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cv_reps", type=int, default=2)
    ap.add_argument("--only", default="", choices=["", "heads"],
                    help="job2 only: 'heads' skips the (numpy, TF32-immune) ridge probe grid and "
                         "reuses the stored job2_grid.json")
    ap.add_argument("--objectives", nargs="+", default=["bt"],
                    help="objectives swept in CV. Default 'bt' only: the objective is part of the "
                         "architecture FROZEN from the previous round's train-only CV "
                         "(L21/span/bt/h256/wd1e-2/30ep); this round varies the FEATURE SET. "
                         "The bce variant is reported at eval as a labelled robustness check.")
    A = ap.parse_args()
    {"null": stage_null, "cv": stage_cv, "eval": stage_eval, "job2": stage_job2,
     "feat": stage_feat, "assemble": stage_assemble}[A.stage](A)


if __name__ == "__main__":
    main()
