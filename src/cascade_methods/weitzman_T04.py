#!/usr/bin/env python3
"""weitzman_T04.py -- KNOB 4: the Weitzman/Pandora open-text controller REFIT at the new
operating point (generation temperature 0.4 instead of the deployed 0.7).

THE QUESTION.  src/cascade_methods/pandora_controller.py implements Weitzman optimal stopping: one
lambda yields BOTH the draw-another-sample reservation value and the escalate reservation value.
That lambda was fitted on T=0.7 pools.  At T=0.4 the candidate pool is materially different
(distinct-of-8 3.66 -> 2.58, oracle@8 0.6277 -> 0.5864, sel_eff 0.766 -> 0.836), so the value of
drawing one more sample is LOWER while the value of the sample already in hand is HIGHER.  The
fitted lambda is stale in a specific, predictable direction.

WHAT IS MEASURED (CPU only, no GPU, no new generation -- every pool already exists on disk):
  1. NULL TESTS -- the frozen metric; the selected = oracle@8 x sel_eff identity; a bit-exact
     re-derivation of the deployed controller (this file's vectorised policy vs
     pandora_controller.run_pandora item by item, and this file's whole pipeline vs the SHIPPED
     open cells of the canonical macro artifact).
  2. THREE ARMS: refit-at-T0.7 (the matched in-session control) / the T0.7 policy frozen and
     applied unchanged to T0.4 (the cost of NOT refitting) / fully refit at T0.4.
     10 CV-partition seeds x 3 generation seeds, mean/sd/range reported.
  3. THE LAMBDA FRONTIER at T=0.4 and T=0.7 -- cross-fit, NOTHING selected, the whole curve, in
     BOTH grading currencies, with the minimum-compute point that ties always-32B-direct marked.
  4. THE FIXED-N CONTROL: is the adaptive machinery still beating a well-chosen fixed N at T=0.4,
     or has the temperature change made it redundant?
  5. A PERMUTATION NULL of every selection step (shuffled labels), because this project has
     already measured that per-cell "pick the best" earns +0.0109 macro from shuffled labels alone.
  6. Per-cell guardrail flags and the 8-cell macro consequence.

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 src/cascade_methods/weitzman_T04.py
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))

import weitzman_lib as W                                       # noqa: E402
import pandora_controller as PC                                # noqa: E402
import integrated_method as IM                                 # noqa: E402
import integrated_pandora as IP                                # noqa: E402
from src.training_methods import genframe_data as G            # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (        # noqa: E402
    load_judge, load_vscores, DS)

# ------------------------------------------------------------------ pinned config
NBOOT = 10000
BSEED = 20260815
NFOLD = 5
ISO_TOL = 3e-3                       # the deployed iso-accuracy band (integrated_pandora.ISO_TOL)
CV_SEEDS = list(range(10))           # >=10 seeds, protocol rule 4
GEN_SEEDS = ["s0", "s1", "s2"]
LAMS_DEPLOYED = PC.LAMS                       # the deployed grid: [0] + geomspace(1e-4, 1, 90)
LAMS_DENSE = np.concatenate([[0.0], np.geomspace(1e-5, 3.0, 240)])
TAUS = np.linspace(0.0, 1.0, 101)
NFIX = [1, 2, 3, 4, 5, 6, 7, 8]
ARM_NAMES = ["A_deployed_T07r", "B_stale_on_T04", "B2_lambdaStale_recalibrated_T04", "C_refit_T04"]
OBJS = ["O1", "O2"]
SELS = ["resub", "nested"]

CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

# 32B-direct (THE BAR) per open cell -- measured from ckpts/openvqa/strong_lingshu/ (the judge
# column equals the published always_32b_direct cells in _selector_rerun_parts/summary_disjoint.json)
BAR_JUDGE = {"SLAKE_open": 0.8186046511627907, "VQA_RAD_open": 0.6, "PATH_VQA_open": 0.376}
BAR_EM = {"SLAKE_open": 0.8465116279069768, "VQA_RAD_open": 0.545, "PATH_VQA_open": 0.344}

# frozen MCQ half of the 8-cell macro (artifacts/_selector_rerun_parts/summary_disjoint.json)
MCQ_CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
MCQ_ACC = {
    "compute_lean": {"PMC_VQA": 0.5508, "SLAKE_closed": 0.8517, "VQA_RAD_closed": 0.8327,
                     "PATH_VQA_closed": 0.8882, "MedXpertQA-MM": 0.3005},
    "accuracy_max": {"PMC_VQA": 0.5613, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
                     "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}}
MCQ_FLOPS = {
    "compute_lean": {"PMC_VQA": 1.386, "SLAKE_closed": 1.935, "VQA_RAD_closed": 3.604,
                     "PATH_VQA_closed": 3.089, "MedXpertQA-MM": 5.095},
    "accuracy_max": {"PMC_VQA": 3.741, "SLAKE_closed": 4.57, "VQA_RAD_closed": 4.57,
                     "PATH_VQA_closed": 4.57, "MedXpertQA-MM": 4.57}}
BAR_MACRO_ACC = 0.6567
BAR_MACRO_FLOPS = 4.57
SHIPPED_MACRO = {"compute_lean": dict(acc=0.6443, flops=6.674),
                 "accuracy_max": dict(acc=0.6575, flops=7.951)}
SHIPPED_OPEN = {"slake_open": dict(meanN=5.547286821705426, esc=0.4341, acc=0.8078),
                "vqa_rad_open": dict(meanN=6.63, esc=0.54, acc=0.5400),
                "pathvqa_open": dict(meanN=4.371333333333333, esc=0.16, acc=0.3827)}


# ================================================================== per-fold fit
class FoldFit:
    """Everything a fold's TRAIN split determines: the isotonic map, the calibrated cheap-score
    pool, q_strong, and the reservation values for every lambda on the requested grids."""

    def __init__(self, view, tr_idx, lam_grids):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(view.raw[tr_idx].ravel(), view.labs["judge"][tr_idx].ravel())
        pool = self.iso.predict(view.raw[tr_idx].ravel())
        self.q = float(view.strongs["judge"][tr_idx].mean())
        self.cal = view.calibrated(self.iso)
        self.zc = {k: W.zeta_cheap_many(pool, g) for k, g in lam_grids.items()}
        self.zs = {k: np.array([W.zeta_strong(self.q, float(l)) for l in g])
                   for k, g in lam_grids.items()}


def fold_fits(view, folds, lam_grids):
    return [FoldFit(view, np.where(folds != f)[0], lam_grids) for f in range(NFOLD)]


def lam_tables(view, fits, grid):
    """Per fold: (nlam, n) float32 arrays of N / esc / ok_judge / ok_em under that fold's TRAIN
    calibration.  Held-out reading = item i from fold folds[i]'s table; TRAIN reading for fold f =
    the non-f columns of fold f's own table.  Exactly what the deployed loop does, precomputed."""
    out = []
    for fit in fits:
        nl = len(fit.zc[grid])
        N = np.empty((nl, view.n), np.float32); E = np.empty((nl, view.n), np.float32)
        OJ = np.empty((nl, view.n), np.float32); OE = np.empty((nl, view.n), np.float32)
        for li in range(nl):
            r = view.run(fit.cal, float(fit.zc[grid][li]), float(fit.zs[grid][li]))
            N[li] = r["N"]; E[li] = r["esc"]
            OJ[li] = r["ok"]["judge"]; OE[li] = r["ok"]["em"]
        out.append(dict(N=N, esc=E, okj=OJ, oke=OE))
    return out


def gather_heldout(tab, folds, li_per_fold):
    n = len(folds)
    o = {k: np.zeros(n) for k in ("N", "esc", "okj", "oke")}
    for f in range(NFOLD):
        te = folds == f
        for k in o:
            o[k][te] = tab[f][k][li_per_fold[f]][te]
    return o


def subset_idx(view, idx):
    return W.PoolView(view.raw[idx], {k: v[idx] for k, v in view.labs.items()},
                      {k: v[idx] for k, v in view.strongs.items()}, view.ds_index[idx],
                      [view.item_keys[i] for i in idx],
                      None if view.gen_tokens is None else view.gen_tokens[idx])


# ================================================================== lambda selection
def select_lambda(tab_f, tr_mask, target, tol):
    """The DEPLOYED objective, verbatim: on TRAIN, the min-FLOP-eq lambda whose TRAIN accuracy
    reaches target - tol; fall back to the max-TRAIN-accuracy lambda when none does."""
    acc = tab_f["okj"][:, tr_mask].mean(1)
    fl = tab_f["N"][:, tr_mask].mean(1) * W.C_CHEAP_F + tab_f["esc"][:, tr_mask].mean(1) * W.C_STRONG_F
    ok = acc >= target - tol
    if ok.any():
        cand = np.where(ok)[0]
        return int(cand[np.argmin(fl[cand])])
    return int(np.argmax(acc))


def select_lambda_nested(view, folds, f_out, target, tol, cvseed):
    """NESTED selection: an inner 5-fold INSIDE the outer-train split, so lambda is chosen on
    inner-HELD-OUT estimates instead of the resubstitution the deployed code uses."""
    tr_idx = np.where(folds != f_out)[0]
    sub = subset_idx(view, tr_idx)
    inner = W.image_folds_for_keys(cvseed * 1000 + 7, sub.item_keys, k=NFOLD)
    fits = fold_fits(sub, inner, {"dep": LAMS_DEPLOYED})
    tab = lam_tables(sub, fits, "dep")
    nl = len(LAMS_DEPLOYED)
    acc = np.zeros(nl); mN = np.zeros(nl); esc = np.zeros(nl); cnt = 0
    for f in range(NFOLD):
        te = inner == f
        if te.sum() == 0:
            continue
        acc += tab[f]["okj"][:, te].sum(1)
        mN += tab[f]["N"][:, te].sum(1)
        esc += tab[f]["esc"][:, te].sum(1)
        cnt += int(te.sum())
    acc /= cnt; mN /= cnt; esc /= cnt
    fl = mN * W.C_CHEAP_F + esc * W.C_STRONG_F
    ok = acc >= target - tol
    if ok.any():
        cand = np.where(ok)[0]
        return int(cand[np.argmin(fl[cand])])
    return int(np.argmax(acc))


# ================================================================== phase 1: the arms
def phase_arms(VIEWS, targets):
    """For each (cell, generation seed, CV seed): build the deployed-grid tables for BOTH
    temperatures once, then evaluate every (objective, selection mode, arm) off them."""
    store = {(o, s, a): defaultdict(list) for o in OBJS for s in SELS for a in ARM_NAMES}
    lam_log = defaultdict(list)
    for ci, cell in enumerate(CELLS):
        for gs in GEN_SEEDS:
            SUB = {T: subset_idx(VIEWS[f"{T}_{gs}"], np.where(VIEWS[f"{T}_{gs}"].ds_index == ci)[0])
                   for T in ("T04", "T07r")}
            for cv in CV_SEEDS:
                folds = W.image_folds_for_keys(cv, SUB["T04"].item_keys, k=NFOLD)
                FIT = {T: fold_fits(SUB[T], folds, {"dep": LAMS_DEPLOYED}) for T in ("T04", "T07r")}
                TAB = {T: lam_tables(SUB[T], FIT[T], "dep") for T in ("T04", "T07r")}
                for obj in OBJS:
                    tol = ISO_TOL if obj == "O1" else 0.0
                    tgt = {T: (targets[(obj, cell, T, gs)] if obj == "O1" else BAR_JUDGE[cell])
                           for T in ("T04", "T07r")}
                    for sel in SELS:
                        LI = {}
                        for T in ("T04", "T07r"):
                            if sel == "resub":
                                LI[T] = [select_lambda(TAB[T][f], folds != f, tgt[T], tol)
                                         for f in range(NFOLD)]
                            else:
                                LI[T] = [select_lambda_nested(SUB[T], folds, f, tgt[T], tol, cv)
                                         for f in range(NFOLD)]
                            lam_log[(obj, sel, T, cell)].extend(
                                [float(LAMS_DEPLOYED[x]) for x in LI[T]])
                        store[(obj, sel, "A_deployed_T07r")][cell].append(
                            gather_heldout(TAB["T07r"], folds, LI["T07r"]))
                        store[(obj, sel, "C_refit_T04")][cell].append(
                            gather_heldout(TAB["T04"], folds, LI["T04"]))
                        for arm, recal in (("B_stale_on_T04", False),
                                           ("B2_lambdaStale_recalibrated_T04", True)):
                            n = SUB["T04"].n
                            o = {k: np.zeros(n) for k in ("N", "esc", "okj", "oke")}
                            for f in range(NFOLD):
                                te = folds == f
                                li = LI["T07r"][f]
                                if recal:     # only lambda is stale; calibration + q refit at T=0.4
                                    fit = FIT["T04"][f]
                                    cal, zc, zs = fit.cal, fit.zc["dep"][li], fit.zs["dep"][li]
                                else:         # EVERYTHING frozen from the T=0.7 fit
                                    fit = FIT["T07r"][f]
                                    cal = SUB["T04"].calibrated(fit.iso)
                                    zc, zs = fit.zc["dep"][li], fit.zs["dep"][li]
                                r = SUB["T04"].run(cal, float(zc), float(zs))
                                o["N"][te] = r["N"][te]; o["esc"][te] = r["esc"][te]
                                o["okj"][te] = r["ok"]["judge"][te]; o["oke"][te] = r["ok"]["em"][te]
                            store[(obj, sel, arm)][cell].append(o)
                del TAB, FIT
        print(f"    arms: {cell} done", flush=True)
    return store, lam_log


# ================================================================== phase 2: the frontier
def phase_frontier(VIEWS):
    """The full cross-fit lambda curve, per cell, per temperature.  NOTHING is selected."""
    nl = len(LAMS_DENSE)
    curves, items = {}, {}
    for ci, cell in enumerate(CELLS):
        for T in ("T04", "T07r"):
            n = int((VIEWS[f"{T}_{GEN_SEEDS[0]}"].ds_index == ci).sum())
            AJ = np.zeros((nl, n)); AE = np.zeros((nl, n))
            NN = np.zeros((nl, n)); EE = np.zeros((nl, n))
            cnt = 0
            for gs in GEN_SEEDS:
                sub = subset_idx(VIEWS[f"{T}_{gs}"],
                                 np.where(VIEWS[f"{T}_{gs}"].ds_index == ci)[0])
                for cv in CV_SEEDS:
                    folds = W.image_folds_for_keys(cv, sub.item_keys, k=NFOLD)
                    fits = fold_fits(sub, folds, {"dense": LAMS_DENSE})
                    tab = lam_tables(sub, fits, "dense")
                    for f in range(NFOLD):
                        te = folds == f
                        AJ[:, te] += tab[f]["okj"][:, te]
                        AE[:, te] += tab[f]["oke"][:, te]
                        NN[:, te] += tab[f]["N"][:, te]
                        EE[:, te] += tab[f]["esc"][:, te]
                    cnt += 1
                    del tab, fits
            AJ /= cnt; AE /= cnt; NN /= cnt; EE /= cnt
            items[(cell, T)] = (AJ, AE)
            rows = []
            for li in range(nl):
                c = W.cost_of(float(NN[li].mean()), float(EE[li].mean()))
                rows.append(dict(lam=float(LAMS_DENSE[li]), acc_judge=float(AJ[li].mean()),
                                 acc_em=float(AE[li].mean()), meanN=float(NN[li].mean()),
                                 esc=float(EE[li].mean()), flops_eq=c["flops"],
                                 lat_seq_ms=c["lat_seq"], lat_par_ms=c["lat_bat"],
                                 energy_j=c["energy"]))
            curves[(cell, T)] = rows
        print(f"    frontier: {cell} done", flush=True)
    return curves, items


def macro_curve(curves, T):
    out = []
    for li in range(len(LAMS_DENSE)):
        r = [curves[(c, T)][li] for c in CELLS]
        mN = float(np.mean([x["meanN"] for x in r])); e = float(np.mean([x["esc"] for x in r]))
        c = W.cost_of(mN, e)
        out.append(dict(lam=float(LAMS_DENSE[li]),
                        acc_judge=float(np.mean([x["acc_judge"] for x in r])),
                        acc_em=float(np.mean([x["acc_em"] for x in r])),
                        meanN=mN, esc=e, flops_eq=c["flops"], lat_seq_ms=c["lat_seq"],
                        lat_par_ms=c["lat_bat"], energy_j=c["energy"]))
    return out


# ================================================================== phase 3: fixed-N control
def phase_fixedN(VIEWS, T="T04"):
    """FIXED best-of-N + escalate-if-max-calibrated-verifier-score-below-tau, same folds, same
    cross-fit calibration, tau swept as a free knob -> frontier vs frontier, not point vs point."""
    grid = {}
    for ci, cell in enumerate(CELLS):
        AJ = np.zeros((len(NFIX), len(TAUS))); AE = np.zeros_like(AJ); ES = np.zeros_like(AJ)
        cnt = 0
        for gs in GEN_SEEDS:
            sub = subset_idx(VIEWS[f"{T}_{gs}"], np.where(VIEWS[f"{T}_{gs}"].ds_index == ci)[0])
            rows = np.arange(sub.n)
            for cv in CV_SEEDS:
                folds = W.image_folds_for_keys(cv, sub.item_keys, k=NFOLD)
                fits = fold_fits(sub, folds, {})
                aj = np.zeros_like(AJ); ae = np.zeros_like(AJ); es = np.zeros_like(AJ)
                for f in range(NFOLD):
                    te = folds == f
                    pm = np.maximum.accumulate(fits[f].cal, axis=1)
                    for ni, Nf in enumerate(NFIX):
                        d = Nf - 1
                        escm = pm[:, d][:, None] < TAUS[None, :]
                        pj = sub.prefix_lab["judge"][rows, d][:, None]
                        pe = sub.prefix_lab["em"][rows, d][:, None]
                        okj = np.where(escm, sub.strongs["judge"][:, None], pj)
                        oke = np.where(escm, sub.strongs["em"][:, None], pe)
                        aj[ni] += okj[te].sum(0); ae[ni] += oke[te].sum(0); es[ni] += escm[te].sum(0)
                AJ += aj / sub.n; AE += ae / sub.n; ES += es / sub.n
                cnt += 1
        grid[cell] = (AJ / cnt, AE / cnt, ES / cnt)
        print(f"    fixed-N[{T}]: {cell} done", flush=True)
    rows_cell = {}
    for cell in CELLS:
        rr = []
        for ni, Nf in enumerate(NFIX):
            for ti, tau in enumerate(TAUS):
                c = W.cost_of(float(Nf), float(grid[cell][2][ni, ti]))
                rr.append(dict(N=Nf, tau=float(tau), acc_judge=float(grid[cell][0][ni, ti]),
                               acc_em=float(grid[cell][1][ni, ti]),
                               esc=float(grid[cell][2][ni, ti]), flops_eq=c["flops"],
                               lat_seq_ms=c["lat_seq"], lat_par_ms=c["lat_bat"]))
        rows_cell[cell] = rr
    macro = []
    for ni, Nf in enumerate(NFIX):
        for ti, tau in enumerate(TAUS):
            e = float(np.mean([grid[c][2][ni, ti] for c in CELLS]))
            cst = W.cost_of(float(Nf), e)
            macro.append(dict(N=Nf, tau=float(tau),
                              acc_judge=float(np.mean([grid[c][0][ni, ti] for c in CELLS])),
                              acc_em=float(np.mean([grid[c][1][ni, ti] for c in CELLS])),
                              esc=e, flops_eq=cst["flops"], lat_seq_ms=cst["lat_seq"],
                              lat_par_ms=cst["lat_bat"]))
    return rows_cell, macro


# ================================================================== aggregation helpers
def agg(runs):
    n = len(runs[0]["okj"])
    o = {k: np.zeros(n) for k in ("N", "esc", "okj", "oke")}
    for r in runs:
        for k in o:
            o[k] += r[k]
    for k in o:
        o[k] /= len(runs)
    return o


def seed_spread(runs, n_cv):
    """mean/sd/min/max of the cell-level judge accuracy across the CV-partition seeds (generation
    seeds pooled inside each) and across the generation seeds (CV seeds pooled inside each).
    Runs are appended in the order gen-seed-outer, cv-seed-inner."""
    by_cv, by_gs = defaultdict(list), defaultdict(list)
    for i, r in enumerate(runs):
        by_cv[i % n_cv].append(float(r["okj"].mean()))
        by_gs[i // n_cv].append(float(r["okj"].mean()))

    def s(d):
        v = np.array([np.mean(x) for x in d.values()], float)
        return dict(n=int(len(v)), mean=float(v.mean()),
                    sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    min=float(v.min()), max=float(v.max()))
    return {"across_cv_seeds": s(by_cv), "across_generation_seeds": s(by_gs)}


def pareto(rows, key="acc_judge"):
    pts = sorted(rows, key=lambda p: (p["flops_eq"], -p[key]))
    keep, best = [], -1.0
    for p in pts:
        if p[key] > best + 1e-12:
            keep.append(p); best = p[key]
    return keep


def min_flops_at(rows, target, key="acc_judge"):
    ok = [p for p in rows if p[key] >= target]
    return min(ok, key=lambda p: p["flops_eq"]) if ok else None


def macro8(open_acc, open_flops, half):
    accs = [MCQ_ACC[half][c] for c in MCQ_CELLS] + [open_acc[c] for c in CELLS]
    fl = [MCQ_FLOPS[half][c] for c in MCQ_CELLS] + [open_flops[c] for c in CELLS]
    return float(np.mean(accs)), float(np.mean(fl))


# ================================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--out", default="results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json")
    A = ap.parse_args()
    t0 = time.time()

    O = {"title": "KNOB 4 -- the Weitzman/Pandora adaptive-N open-text controller refit at the new "
                  "operating point (generation temperature 0.4 instead of the deployed 0.7)",
         "date": "2026-08-15",
         "scripts": ["src/cascade_methods/weitzman_T04.py", "src/cascade_methods/weitzman_lib.py"],
         "no_gpu": True, "no_new_generation": True,
         "inputs": {"pools": "ckpts/openvqa/decoding_sweep/ckpt_{ds}_{T04,T07r}_s{0,1,2}.jsonl "
                             "(generated 2026-08-14 by src/cascade_methods/decoding_sweep_gen.py)",
                    "verifier_scores": "ckpts/openvqa/decoding_sweep/vscore_cache_shard*.jsonl -- the "
                                       "CLEAN disjoint LoRA verifier (ckpts/train/"
                                       "lora_verifier_disjoint) scored under HF transformers, batch 1 "
                                       "(never vLLM: it drops all 192 visual.* modules)",
                    "judge_labels": "ckpts/openvqa/decoding_sweep/judgecache_preload_{ds}.jsonl + "
                                    "judgein_{ds}.judge.jsonl",
                    "em_labels": "the oks_em field recorded at generation time by run_openvqa.score()",
                    "strong_leg": "ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b{,.judge}.jsonl",
                    "shipped_reference": "results/cascade_methods/artifacts/_selector_rerun_parts/"
                                         "summary_disjoint.json"},
         "numerics_pinned": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                             "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                             "numpy": np.__version__,
                             "note": "no SGD anywhere in this round -- the only fitted objects are a "
                                     "deterministic sklearn IsotonicRegression and a scalar lambda, "
                                     "so the TF32 / thread-count / rank_avg-vs-argsort landmines do "
                                     "not bind. Row order is the FROZEN canonical item order "
                                     "(genframe_data.DUMP_ORDER)."},
         "nboot": A.nboot, "bootstrap_seed": BSEED, "cv_seeds": CV_SEEDS,
         "generation_seeds": GEN_SEEDS, "n_folds": NFOLD,
         "fold_protocol": "IMAGE-DISJOINT 5-fold, md5(seed|decoded-RGB-image-hash) % 5, so all items "
                          "sharing an image land in one fold (2,345 items share only 528 images). "
                          "The DEPLOYED code uses i%5 modulo folds; NT5 reproduces the shipped cells "
                          "under that folding, and the arms use the image-disjoint family."}

    # ---------------------------------------------------------------- NULL TESTS
    print("NULL TESTS ...", flush=True)
    nt1 = G.null_test()
    r0 = G.sel_eff(G.incumbent_scores())
    ident = abs(r0["acc"] - r0["oracle"] * r0["sel_eff"])
    print(f"  NT1 frozen metric pass={nt1['pass']} maxdev={nt1['max_abs_deviation']:.3e}", flush=True)
    print(f"  NT2 identity residual: {ident:.3e}", flush=True)

    rng = np.random.default_rng(0)
    zdev = 0.0
    for _ in range(30):
        v = rng.random(int(rng.integers(50, 5000))) ** rng.uniform(0.3, 4)
        zdev = max(zdev, float(np.max(np.abs(
            W.zeta_cheap_many(v, LAMS_DEPLOYED)
            - np.array([PC.zeta_cheap(v, float(l)) for l in LAMS_DEPLOYED])))))
    print(f"  NT3 zeta_cheap closed form vs bisection: {zdev:.3e}", flush=True)

    vdep = W.build_view_from_frozen_dumps()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(vdep.raw.ravel(), vdep.labs["judge"].ravel())
    cal = vdep.calibrated(iso)
    zcs = W.zeta_cheap_many(iso.predict(vdep.raw.ravel()), LAMS_DEPLOYED)
    q = float(vdep.strongs["judge"].mean())
    dN = dE = dO = 0.0
    for l, zc in zip(LAMS_DEPLOYED, zcs):
        zs = W.zeta_strong(q, float(l))
        r = vdep.run(cal, float(zc), float(zs))
        for i in range(vdep.n):
            Nk, e, ok = PC.run_pandora(list(vdep.raw[i]), list(cal[i]),
                                       list(vdep.labs["judge"][i].astype(int)),
                                       vdep.strongs["judge"][i], float(zc), float(zs))
            dN = max(dN, abs(r["N"][i] - Nk)); dE = max(dE, abs(r["esc"][i] - e))
            dO = max(dO, abs(r["ok"]["judge"][i] - ok))
    print(f"  NT4 vectorised policy vs run_pandora: N={dN:g} esc={dE:g} ok={dO:g}", flush=True)

    IM.OPEN_VERIFIER_DIR = "ckpts/train/lora_verifier_disjoint"
    IP.ADAPTER = "ckpts/train/lora_verifier_disjoint"
    nt5 = {}
    for j, ds in enumerate(DS):
        sub = subset_idx(vdep, np.where(vdep.ds_index == j)[0])
        folds = W.modulo_folds(sub.n, NFOLD)
        fits = fold_fits(sub, folds, {"dep": LAMS_DEPLOYED})
        tab = lam_tables(sub, fits, "dep")
        dfix = IM.open_bestof8(ds)
        target, _ = IM.heldout(dfix["ok7"], dfix["ok32"], dfix["gate"])
        li = [select_lambda(tab[f], folds != f, target, ISO_TOL) for f in range(NFOLD)]
        h = gather_heldout(tab, folds, li)
        got = dict(meanN=float(h["N"].mean()), esc=float(h["esc"].mean()), acc=float(h["okj"].mean()))
        nt5[ds] = dict(target_iso_accuracy=target, measured=got, shipped=SHIPPED_OPEN[ds],
                       abs_dev={k: abs(got[k] - SHIPPED_OPEN[ds][k]) for k in SHIPPED_OPEN[ds]},
                       lambdas=[float(LAMS_DEPLOYED[x]) for x in li])
        print(f"  NT5 {ds:14s} meanN {got['meanN']:.6f}/{SHIPPED_OPEN[ds]['meanN']:.6f}  "
              f"esc {got['esc']:.6f}/{SHIPPED_OPEN[ds]['esc']:.4f}  "
              f"acc {got['acc']:.6f}/{SHIPPED_OPEN[ds]['acc']:.4f}", flush=True)
    nt5max = max(max(v["abs_dev"].values()) for v in nt5.values())

    O["NULL_TESTS"] = {
        "NT1_frozen_metric": {"what": "src/training_methods/genframe_data.py reproduces every "
                                      "published incumbent cell", "pass": bool(nt1["pass"]),
                              "max_abs_deviation": nt1["max_abs_deviation"],
                              "measured": nt1["measured"]},
        "NT2_identity": {"what": "selected = oracle@8 x sel_eff (the EXACT identity, never the "
                                 "additive form)", "residual": ident, "oracle@8": r0["oracle"],
                         "sel_eff": r0["sel_eff"], "selected": r0["acc"]},
        "NT3_zeta_closed_form": {"what": "this file's exact zeta_cheap vs pandora_controller's "
                                         "80-step bisection, 30 random pools x 91 lambdas",
                                 "max_abs_deviation": zdev},
        "NT4_policy_bit_exact": {"what": "this file's vectorised Weitzman policy vs "
                                         "pandora_controller.run_pandora, item by item",
                                 "lambdas": int(len(LAMS_DEPLOYED)), "items": int(vdep.n),
                                 "max_abs_deviation_N": dN, "max_abs_deviation_esc": dE,
                                 "max_abs_deviation_ok": dO},
        "NT5_reproduces_the_shipped_open_cells": {
            "what": "the whole pipeline (modulo folds, deployed lambda grid, deployed iso-accuracy "
                    "objective, CLEAN disjoint verifier) reproduces the three open cells of the "
                    "canonical macro artifact _selector_rerun_parts/summary_disjoint.json",
            "per_cell": nt5, "max_abs_deviation": nt5max,
            "note": "residuals are the 4-dp rounding of the stored artifact, not disagreement."}}
    O["null_test_passed"] = bool(nt1["pass"] and ident < 1e-12 and zdev < 1e-9
                                 and dN == 0 and dE == 0 and dO == 0 and nt5max < 1e-3)
    O["null_test_max_abs_deviation"] = {"frozen_metric": nt1["max_abs_deviation"],
                                        "identity": ident, "zeta": zdev,
                                        "policy": max(dN, dE, dO), "shipped_open_cells": nt5max}

    # ---------------------------------------------------------------- pools
    print("\nLOADING POOLS ...", flush=True)
    lab, vsc, ref = load_judge(), load_vscores(), G.load_items()
    strongs = W._strong_labels()
    VIEWS = {}
    for T in ("T04", "T07r"):
        for s in GEN_SEEDS:
            tag = f"{T}_{s}"
            VIEWS[tag] = W.build_view(tag, lab, vsc, ref, strongs)
            print(f"  {tag}: n={VIEWS[tag].n} tok={VIEWS[tag].gen_tokens.mean():.3f}", flush=True)
    O["pools"] = {}
    for tag, v in VIEWS.items():
        nd = np.array([len(set(np.round(x, 12))) for x in v.raw], float)
        O["pools"][tag] = dict(
            n=int(v.n), Nmax=int(v.Nmax),
            mean_gen_tokens_all_slots=float(v.gen_tokens.mean()),
            distinct_verifier_scores_of_8=float(nd.mean()),
            oracle8_judge=float((v.labs["judge"].max(1) > 0).mean()),
            oracle8_em=float((v.labs["em"].max(1) > 0).mean()),
            random_slot_judge=float(v.labs["judge"].mean()),
            random_slot_em=float(v.labs["em"].mean()),
            bo8_selected_judge=float(v.prefix_lab["judge"][:, 7].mean()),
            bo8_selected_em=float(v.prefix_lab["em"][:, 7].mean()))
    O["control"] = ("T07r = the DEPLOYED temperature 0.7 REGENERATED in the 2026-08-14 session "
                    "(artifacts/decoding_ladder_cold_2026-08-14.json). Every T=0.4 delta here is "
                    "against T07r and never against a stored number -- the +-0.008 open-text "
                    "reproducibility caveat. That artifact's REPRODUCIBILITY block measures the "
                    "stored-vs-in-session nuisance for this exact pool pair.")
    O["strong_leg"] = {"what": "the 32B-direct answers do NOT depend on the 7B's sampling "
                               "temperature; they are the deployed dumps, unchanged in every arm",
                       "judge": BAR_JUDGE, "em": BAR_EM}
    O["cost_model"] = {"as_charged_deployed": {
                           "cheap_draw_flops_eq": W.C_CHEAP_F, "escalation_flops_eq": W.C_STRONG_F,
                           "GEN7_ms_J_flop": W.GEN7, "VER7_ms_J_flop": W.VER7,
                           "GEN32N_ms_J_flop": W.GEN32N,
                           "source": "paper_baselines.py / pandora_controller.py measured batch-1 "
                                     "constants -- the currency the published '11.74 vs 16.0 FLOP-eq "
                                     "(-27%)' survivor is denominated in, so the frontier is traced "
                                     "in it.",
                           "latency_honesty": "adaptive draws are SEQUENTIAL (draw -> check -> draw) "
                                              "so lat_seq = meanN*522ms + esc*665ms is the honest "
                                              "number; a FIXED N can batch its draws, so lat_par = "
                                              "522ms + esc*665ms. Both are reported for both arms."},
                       "measured_cap320_secondary": W.measured_cost_model()}

    # ---------------------------------------------------------------- targets
    targets = {}
    for ci, cell in enumerate(CELLS):
        for T in ("T04", "T07r"):
            for gs in GEN_SEEDS:
                v = VIEWS[f"{T}_{gs}"]
                sub = subset_idx(v, np.where(v.ds_index == ci)[0])
                a_fix, _ = IM.heldout(sub.prefix_lab["judge"][:, 7], sub.strongs["judge"],
                                      np.max(sub.raw, axis=1))
                targets[("O1", cell, T, gs)] = float(a_fix)
    O["objectives"] = {
        "O1_deployed": {"what": "the deployed iso-accuracy objective, verbatim: the min-FLOP-eq "
                                "lambda whose TRAIN accuracy reaches (fixed best-of-8 + "
                                "verifier-confidence gate, held-out) - 0.003 "
                                "(integrated_pandora.pandora_open_arm).",
                        "targets": {f"{c}|{T}|{g}": targets[("O1", c, T, g)]
                                    for c in CELLS for T in ("T04", "T07r") for g in GEN_SEEDS}},
        "O2_parity_with_the_bar": {"what": "the min-FLOP-eq lambda whose TRAIN accuracy reaches "
                                           "always-32B-direct on that cell (tol 0). The project's "
                                           "stated objective is minimum compute at parity.",
                                   "targets": BAR_JUDGE}}

    # ---------------------------------------------------------------- phases
    print("\nPHASE 1: arms ...", flush=True)
    STORE, LAMLOG = phase_arms(VIEWS, targets)
    print("PHASE 2: frontier ...", flush=True)
    CURVES, FITEMS = phase_frontier(VIEWS)
    print("PHASE 3: fixed-N control ...", flush=True)
    FX_CELL, FX_MACRO = phase_fixedN(VIEWS, "T04")
    FX7_CELL, FX7_MACRO = phase_fixedN(VIEWS, "T07r")

    # ---- arms report -------------------------------------------------------------------------
    ARMS = {}
    for key, per in STORE.items():
        obj, sel, arm = key
        src_T = "T04" if arm == "C_refit_T04" else "T07r"
        blk = {"per_cell": {}, "lambda_fitted_on": src_T,
               "note": {"A_deployed_T07r": "lambda refit on the T=0.7 pool and evaluated there -- "
                                           "the MATCHED IN-SESSION CONTROL for every T=0.4 number",
                        "B_stale_on_T04": "the ENTIRE T=0.7 fit (lambda, isotonic map, zeta_cheap, "
                                          "zeta_strong) frozen and applied unchanged to the T=0.4 "
                                          "pool -- this is what NOT refitting costs",
                        "B2_lambdaStale_recalibrated_T04": "only lambda is stale; the isotonic map, "
                                                           "q_strong and both reservation values are "
                                                           "recomputed on T=0.4 -- separates a stale "
                                                           "LAMBDA from a stale CALIBRATION",
                        "C_refit_T04": "everything refit on the T=0.4 pool"}[arm]}
        for cell in CELLS:
            a = agg(per[cell])
            c = W.cost_of(float(a["N"].mean()), float(a["esc"].mean()))
            lams = np.array(LAMLOG[(obj, sel, src_T, cell)], float)
            blk["per_cell"][cell] = dict(
                n=int(len(a["okj"])), acc_judge=float(a["okj"].mean()), acc_em=float(a["oke"].mean()),
                meanN=float(a["N"].mean()), esc=float(a["esc"].mean()), flops_eq=c["flops"],
                lat_seq_ms=c["lat_seq"], lat_par_ms=c["lat_bat"], energy_j=c["energy"],
                lambda_source_pool=src_T, lambda_mean=float(lams.mean()),
                lambda_median=float(np.median(lams)), lambda_min=float(lams.min()),
                lambda_max=float(lams.max()),
                lambda_frac_zero=float((lams == 0).mean()),
                seed_spread_judge=seed_spread(per[cell], len(CV_SEEDS)),
                bar_judge=BAR_JUDGE[cell], bar_em=BAR_EM[cell])
        blk["open3_macro"] = {k: float(np.mean([blk["per_cell"][c][k] for c in CELLS]))
                              for k in ("acc_judge", "acc_em", "meanN", "esc", "flops_eq",
                                        "lat_seq_ms", "lat_par_ms", "energy_j")}
        ARMS[f"{obj}|{sel}|{arm}"] = blk

    CONTR = {}
    for obj in OBJS:
        for sel in SELS:
            for a, b, label in (("C_refit_T04", "B_stale_on_T04", "refit_minus_stale_atT04"),
                                ("C_refit_T04", "A_deployed_T07r", "T04refit_minus_T07deployed"),
                                ("B_stale_on_T04", "A_deployed_T07r", "T04stale_minus_T07deployed"),
                                ("B2_lambdaStale_recalibrated_T04", "B_stale_on_T04",
                                 "recalibration_only_atT04")):
                blk = {}
                va_all, vb_all = [], []
                for cell in CELLS:
                    va = agg(STORE[(obj, sel, a)][cell]); vb = agg(STORE[(obj, sel, b)][cell])
                    va_all.append(va); vb_all.append(vb)
                    blk[cell] = {
                        "judge": W.boot(va["okj"], vb["okj"], nboot=A.nboot, seed=BSEED),
                        "em": W.boot(va["oke"], vb["oke"], nboot=A.nboot, seed=BSEED),
                        "d_flops_eq": (W.cost_of(va["N"].mean(), va["esc"].mean())["flops"]
                                       - W.cost_of(vb["N"].mean(), vb["esc"].mean())["flops"]),
                        "d_meanN": float(va["N"].mean() - vb["N"].mean()),
                        "d_esc": float(va["esc"].mean() - vb["esc"].mean())}
                for cur, k in (("judge", "okj"), ("em", "oke")):
                    pt, dist = W.boot_macro([x[k] for x in va_all], [x[k] for x in vb_all],
                                            nboot=A.nboot, seed=BSEED)
                    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
                    blk[f"open3_macro_{cur}"] = dict(
                        delta=pt, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0),
                        verdict="WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                        macro8_scaled_delta=pt * 3.0 / 8.0)
                blk["open3_macro_d_flops_eq"] = float(np.mean([blk[c]["d_flops_eq"] for c in CELLS]))
                CONTR[f"{obj}|{sel}|{label}"] = blk
    O["ARMS"] = ARMS
    O["ARM_CONTRASTS"] = CONTR

    # ---- frontier ----------------------------------------------------------------------------
    M04 = macro_curve(CURVES, "T04"); M07 = macro_curve(CURVES, "T07r")
    O["FRONTIER"] = {
        "what": "the full accuracy-vs-compute curve traced by sweeping lambda. NOTHING is selected "
                "here: calibration is cross-fit per fold, the same lambda is used in every fold, and "
                "the whole curve is reported. The project's objective is minimum compute at parity, "
                "so the frontier IS the deliverable.",
        "lambda_grid": "concat([0], geomspace(1e-5, 3.0, 240))",
        "averaging": "per-item held-out outcome averaged over 3 generation seeds x 10 CV-partition "
                     "seeds, then aggregated (the temperature ladder's protocol).",
        "open3_macro_T04": M04, "open3_macro_T07r": M07,
        "per_cell_T04": {c: CURVES[(c, "T04")] for c in CELLS},
        "per_cell_T07r": {c: CURVES[(c, "T07r")] for c in CELLS},
        "pareto_open3_macro_T04": pareto(M04), "pareto_open3_macro_T07r": pareto(M07)}
    O["FIXED_N"] = {
        "what": "the control the adaptive machinery must beat: FIXED best-of-N plus a fixed "
                "escalate-if-max-calibrated-verifier-score-below-tau gate. Same folds, same "
                "cross-fit calibration, same cost model, tau swept as a free knob so this is a "
                "frontier-vs-frontier comparison. A fixed N can BATCH its draws (lat_par); the "
                "adaptive controller cannot (lat_seq).",
        "N_grid": NFIX, "tau_grid": "linspace(0,1,101) on the CALIBRATED score",
        "open3_macro_T04": FX_MACRO, "open3_macro_T07r": FX7_MACRO,
        "per_cell_T04": FX_CELL, "per_cell_T07r": FX7_CELL,
        "pareto_open3_macro_T04": pareto(FX_MACRO), "pareto_open3_macro_T07r": pareto(FX7_MACRO)}

    # ---- operating points / macro / guardrail ------------------------------------------------
    O["OPERATING_POINTS"] = operating_points(M04, M07, FX_MACRO, FX7_MACRO, CURVES, FX_CELL,
                                             FITEMS, STORE, VIEWS, A.nboot)

    # ---- permutation null --------------------------------------------------------------------
    print("\nPERMUTATION NULL ...", flush=True)
    O["PERMUTATION_NULL"] = permutation_null(VIEWS, A.nperm, O)

    O["runtime_s"] = time.time() - t0
    p = os.path.join(ROOT, A.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(O, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {A.out}  ({time.time()-t0:.0f}s)")
    console(O)
    return O


# ================================================================== operating points / macro
def operating_points(M04, M07, FX_MACRO, FX7_MACRO, CURVES, FX_CELL, FITEMS, STORE, VIEWS, nboot):
    out = {"convention": "MACRO, equal weight per reporting cell, 8 cells at 1/8, Variant B (MMMU "
                         "excluded), CLEAN disjoint verifier. The 5 multiple-choice cells are FROZEN "
                         "at their published values because nothing in this round can touch them; "
                         "only the 3 open cells move. Every macro-8 number therefore mixes "
                         "IN-SESSION open cells with STORED multiple-choice cells -- stated, not "
                         "hidden -- and the in-session-only contrast (T=0.4 vs the T07r control) is "
                         "reported beside it. Open-3 macro deltas scale to macro-8 by 3/8."}
    bar3j = float(np.mean([BAR_JUDGE[c] for c in CELLS]))
    bar3e = float(np.mean([BAR_EM[c] for c in CELLS]))
    out["open3_bar"] = {"judge": bar3j, "em": bar3e}
    out["per_cell_marks_T04"] = {
        c: dict(min_flops_at_bar_judge=min_flops_at(CURVES[(c, "T04")], BAR_JUDGE[c], "acc_judge"),
                min_flops_at_bar_em=min_flops_at(CURVES[(c, "T04")], BAR_EM[c], "acc_em"),
                best_judge=max(CURVES[(c, "T04")], key=lambda p: p["acc_judge"]),
                bar_judge=BAR_JUDGE[c], bar_em=BAR_EM[c]) for c in CELLS}
    out["per_cell_marks_T07r"] = {
        c: dict(min_flops_at_bar_judge=min_flops_at(CURVES[(c, "T07r")], BAR_JUDGE[c], "acc_judge"),
                min_flops_at_bar_em=min_flops_at(CURVES[(c, "T07r")], BAR_EM[c], "acc_em"),
                best_judge=max(CURVES[(c, "T07r")], key=lambda p: p["acc_judge"])) for c in CELLS}

    sel = {"adaptive_T04_judge": min_flops_at(M04, bar3j, "acc_judge"),
           "adaptive_T04_em": min_flops_at(M04, bar3e, "acc_em"),
           "adaptive_T07r_judge": min_flops_at(M07, bar3j, "acc_judge"),
           "adaptive_T07r_em": min_flops_at(M07, bar3e, "acc_em"),
           "fixedN_T04_judge": min_flops_at(FX_MACRO, bar3j, "acc_judge"),
           "fixedN_T07r_judge": min_flops_at(FX7_MACRO, bar3j, "acc_judge"),
           "adaptive_T04_best_judge": max(M04, key=lambda p: p["acc_judge"]),
           "adaptive_T07r_best_judge": max(M07, key=lambda p: p["acc_judge"]),
           "fixedN_T04_best_judge": max(FX_MACRO, key=lambda p: p["acc_judge"])}
    out["selected"] = sel
    out["selected_note"] = ("these ARE selections -- a point picked off a swept curve by an accuracy "
                            "criterion. PERMUTATION_NULL runs the identical selection on shuffled "
                            "labels and reports what it earns from noise alone.")

    fixp = pareto(FX_MACRO)
    dom = []
    for p in pareto(M04):
        cheaper = [q for q in fixp if q["flops_eq"] <= p["flops_eq"] + 1e-9]
        bf = max(cheaper, key=lambda q: q["acc_judge"]) if cheaper else None
        dom.append(dict(lam=p["lam"], flops_eq=p["flops_eq"], adaptive_acc_judge=p["acc_judge"],
                        adaptive_lat_seq_ms=p["lat_seq_ms"],
                        best_fixed_acc_judge=(bf["acc_judge"] if bf else None),
                        best_fixed_config=(f"N={bf['N']} tau={bf['tau']:.2f}" if bf else None),
                        best_fixed_lat_par_ms=(bf["lat_par_ms"] if bf else None),
                        adaptive_advantage=(p["acc_judge"] - bf["acc_judge"]) if bf else None))
    adv = [d["adaptive_advantage"] for d in dom if d["adaptive_advantage"] is not None]
    out["adaptive_vs_fixedN_T04"] = dict(
        points=dom, n_points=len(adv),
        n_where_adaptive_wins=int(sum(1 for a in adv if a > 0)),
        max_advantage=float(max(adv)) if adv else None,
        min_advantage=float(min(adv)) if adv else None,
        median_advantage=float(np.median(adv)) if adv else None,
        read="positive = the adaptive controller delivers MORE accuracy than the best FIXED-N + gate "
             "configuration available at the same or lower FLOP-eq. Both curves are swept, both "
             "cross-fit, neither knob chosen on held-out data, so they are equally optimistic. The "
             "latency columns are the honest ones: a fixed N batches, adaptive N cannot.")

    macro = {}
    for half in ("compute_lean", "accuracy_max"):
        blk = {}
        for armkey, per in STORE.items():
            obj, s, arm = armkey
            oa, of = {}, {}
            for cell in CELLS:
                a = agg(per[cell])
                oa[cell] = float(a["okj"].mean())
                of[cell] = W.cost_of(float(a["N"].mean()), float(a["esc"].mean()))["flops"]
            acc, fl = macro8(oa, of, half)
            blk[f"{obj}|{s}|{arm}"] = dict(
                macro_acc=acc, macro_flops_eq=fl,
                vs_always32b_direct_acc=acc - BAR_MACRO_ACC,
                compute_x_vs_always32b_direct=fl / BAR_MACRO_FLOPS,
                vs_shipped_acc=acc - SHIPPED_MACRO[half]["acc"],
                compute_x_vs_shipped=fl / SHIPPED_MACRO[half]["flops"],
                open_cells_acc=oa, open_cells_flops=of)
        for label, mark, src, kind in (
                ("frontier_T04_minflops_at_open3_bar", sel["adaptive_T04_judge"], CURVES, "lam04"),
                ("frontier_T07r_minflops_at_open3_bar", sel["adaptive_T07r_judge"], CURVES, "lam07"),
                ("frontier_T04_best_judge", sel["adaptive_T04_best_judge"], CURVES, "lam04"),
                ("fixedN_T04_minflops_at_open3_bar", sel["fixedN_T04_judge"], FX_CELL, "fix")):
            if mark is None:
                blk[label] = None
                continue
            oa, of = {}, {}
            for cell in CELLS:
                if kind == "fix":
                    row = [r for r in src[cell] if r["N"] == mark["N"]
                           and abs(r["tau"] - mark["tau"]) < 1e-12][0]
                else:
                    key = (cell, "T04" if kind == "lam04" else "T07r")
                    row = [r for r in src[key] if abs(r["lam"] - mark["lam"]) < 1e-15][0]
                oa[cell] = row["acc_judge"]; of[cell] = row["flops_eq"]
            acc, fl = macro8(oa, of, half)
            blk[label] = dict(knob=mark, macro_acc=acc, macro_flops_eq=fl,
                              vs_always32b_direct_acc=acc - BAR_MACRO_ACC,
                              compute_x_vs_always32b_direct=fl / BAR_MACRO_FLOPS,
                              vs_shipped_acc=acc - SHIPPED_MACRO[half]["acc"],
                              compute_x_vs_shipped=fl / SHIPPED_MACRO[half]["flops"],
                              open_cells_acc=oa, open_cells_flops=of)
        macro[half] = blk
    out["MACRO"] = macro
    out["MACRO_references"] = dict(
        always_32b_direct=dict(acc=BAR_MACRO_ACC, flops=BAR_MACRO_FLOPS),
        shipped=SHIPPED_MACRO,
        significance_bar="a significant macro-8 win needs |delta| ~ 0.0029 (the published CI "
                         "half-width) = summed per-cell gain ~ 0.0235",
        caveat="the shipped accuracy-max open cells use pandora-N + the F10-L2D rejector, NOT the "
               "plain Weitzman escalation this round refits; the shipped compute-lean open cells DO "
               "use the plain Weitzman policy. Compare like with like: the plain-Weitzman arm's "
               "natural shipped reference is compute-lean (0.6443 @ 6.674 FLOP-eq).")

    ci = {}
    for label, mark, T in (("adaptive_T04_judge", sel["adaptive_T04_judge"], "T04"),
                           ("adaptive_T07r_judge", sel["adaptive_T07r_judge"], "T07r"),
                           ("adaptive_T04_best_judge", sel["adaptive_T04_best_judge"], "T04")):
        if mark is None:
            ci[label] = None
            continue
        li = int(np.argmin(np.abs(LAMS_DENSE - mark["lam"])))
        v0 = VIEWS[f"{T}_{GEN_SEEDS[0]}"]
        va = [FITEMS[(c, T)][0][li] for c in CELLS]
        vb = [v0.strongs["judge"][v0.ds_index == CELLS.index(c)] for c in CELLS]
        pt, dist = W.boot_macro(va, vb, nboot=nboot, seed=BSEED)
        lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
        ci[label] = dict(lam=mark["lam"], open3_macro_delta_vs_bar=pt, lo=lo, hi=hi,
                         verdict="WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                         macro8_scaled_delta=pt * 3.0 / 8.0,
                         per_cell={c: W.boot(va[i], vb[i], nboot=nboot, seed=BSEED)
                                   for i, c in enumerate(CELLS)})
    out["marked_point_CIs_vs_always32b_direct"] = ci

    gr = {}
    for obj in OBJS:
        for arm in ("C_refit_T04", "B_stale_on_T04", "B2_lambdaStale_recalibrated_T04"):
            blk = {}
            for cell in CELLS:
                va = agg(STORE[(obj, "resub", arm)][cell])
                vb = agg(STORE[(obj, "resub", "A_deployed_T07r")][cell])
                v0 = VIEWS[f"T04_{GEN_SEEDS[0]}"]
                m = v0.ds_index == CELLS.index(cell)
                blk[cell] = dict(
                    n=int(m.sum()),
                    vs_T07r_control_judge=W.boot(va["okj"], vb["okj"], nboot=nboot, seed=BSEED),
                    vs_T07r_control_em=W.boot(va["oke"], vb["oke"], nboot=nboot, seed=BSEED),
                    vs_always32b_direct_judge=W.boot(va["okj"], v0.strongs["judge"][m],
                                                     nboot=nboot, seed=BSEED),
                    vs_always32b_direct_em=W.boot(va["oke"], v0.strongs["em"][m],
                                                  nboot=nboot, seed=BSEED))
            blk["FLAGS_vs_T07r_control"] = [c for c in CELLS
                                            if blk[c]["vs_T07r_control_judge"]["verdict"] == "LOSS"
                                            or blk[c]["vs_T07r_control_em"]["verdict"] == "LOSS"]
            blk["FLAGS_vs_always32b_direct"] = [
                c for c in CELLS if blk[c]["vs_always32b_direct_judge"]["verdict"] == "LOSS"
                or blk[c]["vs_always32b_direct_em"]["verdict"] == "LOSS"]
            gr[f"{obj}|{arm}"] = blk
    out["GUARDRAIL"] = gr
    out["GUARDRAIL_note"] = ("cell sizes are SLAKE_open 645 / VQA_RAD_open 200 / PATH_VQA_open 1500. "
                             "Read a VQA_RAD_open flag together with ARMS[*].per_cell.VQA_RAD_open."
                             "seed_spread_judge -- at n=200 a single-cell verdict sits inside the "
                             "CV-seed spread.")
    return out


# ================================================================== permutation null
def permutation_null(VIEWS, R, RES):
    """SHUFFLED-LABEL null of every selection step in this round.

    Within each cell the LABEL BUNDLE (the 8 cheap-slot judge labels, the 8 cheap-slot EM labels,
    the 32B-direct judge label and the 32B-direct EM label) is permuted ACROSS ITEMS while the
    verifier scores stay put. Every marginal is preserved -- each slot position's cheap accuracy,
    the strong leg's accuracy, the pool geometry, the cost model -- and only the score<->label link
    is destroyed. The IDENTICAL pipeline then runs: cross-fit isotonic calibration, the lambda
    sweep, the 'minimum FLOP-eq point that reaches the always-32B-direct bar' selection, and the
    refit-versus-stale contrast. Anything it still earns is manufactured by the selection.

    The SAME permutation is applied to the T=0.4 and the T=0.7 pool of a replicate, so the
    refit-minus-stale and T04-minus-T07r contrasts are drawn under one null.
    """
    rng = np.random.default_rng(BSEED)
    bar3 = float(np.mean([BAR_JUDGE[c] for c in CELLS]))
    nl = len(LAMS_DENSE)
    gs = GEN_SEEDS[0]
    cvs = CV_SEEDS[:3]
    S = defaultdict(list)

    IDX = {c: np.where(VIEWS[f"T04_{gs}"].ds_index == i)[0] for i, c in enumerate(CELLS)}
    SUB = {(c, T): subset_idx(VIEWS[f"{T}_{gs}"], IDX[c]) for c in CELLS for T in ("T04", "T07r")}
    FOL = {(c, cv): W.image_folds_for_keys(cv, SUB[(c, "T04")].item_keys, k=NFOLD)
           for c in CELLS for cv in cvs}

    for rep in range(R):
        curve = {T: dict(a=np.zeros(nl), n=np.zeros(nl), e=np.zeros(nl)) for T in ("T04", "T07r")}
        rms = []
        for cell in CELLS:
            perm = rng.permutation(len(IDX[cell]))
            V = {T: perm_view(SUB[(cell, T)], perm) for T in ("T04", "T07r")}
            nS = V["T04"].n
            for T in ("T04", "T07r"):
                a = np.zeros(nl); n = np.zeros(nl); e = np.zeros(nl)
                for cv in cvs:
                    folds = FOL[(cell, cv)]
                    fits = fold_fits(V[T], folds, {"dense": LAMS_DENSE})
                    tab = lam_tables(V[T], fits, "dense")
                    for f in range(NFOLD):
                        te = folds == f
                        a += tab[f]["okj"][:, te].sum(1)
                        n += tab[f]["N"][:, te].sum(1)
                        e += tab[f]["esc"][:, te].sum(1)
                    del tab, fits
                k = len(cvs) * nS
                curve[T]["a"] += a / k / len(CELLS)
                curve[T]["n"] += n / k / len(CELLS)
                curve[T]["e"] += e / k / len(CELLS)
            per_cv = []
            for cv in cvs:
                folds = FOL[(cell, cv)]
                f4 = fold_fits(V["T04"], folds, {"dep": LAMS_DEPLOYED})
                f7 = fold_fits(V["T07r"], folds, {"dep": LAMS_DEPLOYED})
                t4 = lam_tables(V["T04"], f4, "dep")
                t7 = lam_tables(V["T07r"], f7, "dep")
                li4 = [select_lambda(t4[f], folds != f, BAR_JUDGE[cell], 0.0) for f in range(NFOLD)]
                li7 = [select_lambda(t7[f], folds != f, BAR_JUDGE[cell], 0.0) for f in range(NFOLD)]
                refit = gather_heldout(t4, folds, li4)["okj"]
                stale = np.zeros(nS)
                for f in range(NFOLD):
                    te = folds == f
                    cal4 = V["T04"].calibrated(f7[f].iso)
                    r = V["T04"].run(cal4, float(f7[f].zc["dep"][li7[f]]),
                                     float(f7[f].zs["dep"][li7[f]]))
                    stale[te] = r["ok"]["judge"][te]
                per_cv.append(float(refit.mean() - stale.mean()))
            rms.append(float(np.mean(per_cv)))
        for T in ("T04", "T07r"):
            Aq = curve[T]["a"]
            F = curve[T]["n"] * W.C_CHEAP_F + curve[T]["e"] * W.C_STRONG_F
            hit = Aq >= bar3
            S[f"reached_bar_{T}"].append(bool(hit.any()))
            S[f"min_flops_at_bar_{T}"].append(float(F[hit].min()) if hit.any() else float("nan"))
            S[f"best_acc_{T}"].append(float(Aq.max()))
        S["refit_minus_stale"].append(float(np.mean(rms)))
        S["best_T04_minus_best_T07r"].append(S["best_acc_T04"][-1] - S["best_acc_T07r"][-1])
        if (rep + 1) % 20 == 0:
            print(f"    perm {rep+1}/{R}", flush=True)

    def summ(key):
        v = np.array([x for x in S[key] if np.isfinite(x)], float)
        if len(v) == 0:
            return {"n": 0, "note": "the null NEVER reached the bar at any lambda"}
        return dict(n=int(len(v)), mean=float(v.mean()),
                    sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    p2_5=float(np.percentile(v, 2.5)), p50=float(np.percentile(v, 50)),
                    p97_5=float(np.percentile(v, 97.5)), min=float(v.min()), max=float(v.max()))

    op = RES["OPERATING_POINTS"]
    real_sel = op["selected"]["adaptive_T04_judge"]
    real_best = max(RES["FRONTIER"]["open3_macro_T04"], key=lambda p: p["acc_judge"])["acc_judge"]
    real_best7 = max(RES["FRONTIER"]["open3_macro_T07r"], key=lambda p: p["acc_judge"])["acc_judge"]
    real_rms = RES["ARM_CONTRASTS"]["O2|resub|refit_minus_stale_atT04"]["open3_macro_judge"]["delta"]
    mf = [x for x in S["min_flops_at_bar_T04"] if np.isfinite(x)]
    return {
        "design": permutation_null.__doc__.strip(),
        "n_permutations": R, "generation_seed_used": gs, "cv_seeds_used": cvs,
        "S1_did_the_null_ever_reach_the_bar": {
            "bar_open3_macro_judge": bar3,
            "null_rate_T04": float(np.mean(S["reached_bar_T04"])),
            "null_rate_T07r": float(np.mean(S["reached_bar_T07r"])),
            "real_T04_reached_the_bar": real_sel is not None,
            "read": "the fraction of shuffled-label replicates in which SOME lambda on the cross-fit "
                    "frontier still reaches the always-32B-direct open-3 bar -- the manufacturing "
                    "rate of the 'minimum compute at parity' claim."},
        "S2_min_flops_at_the_bar": {
            "null_T04": summ("min_flops_at_bar_T04"), "null_T07r": summ("min_flops_at_bar_T07r"),
            "real_T04": (real_sel["flops_eq"] if real_sel else None),
            "empirical_p_null_is_at_least_as_cheap": (
                float(np.mean([x <= real_sel["flops_eq"] for x in mf])) if (real_sel and mf) else None)},
        "S3_best_accuracy_on_the_frontier": {
            "null_T04": summ("best_acc_T04"), "real_T04": real_best,
            "empirical_p": float(np.mean([x >= real_best for x in S["best_acc_T04"]]))},
        "S4_refit_minus_stale": {
            "what": "THE headline contrast of this round (open-3 macro, judge currency), drawn "
                    "under shuffled labels with the identical selection",
            "null": summ("refit_minus_stale"), "real": float(real_rms),
            "empirical_p_two_sided": float(np.mean(
                [abs(x) >= abs(real_rms) for x in S["refit_minus_stale"]]))},
        "S5_T04_best_minus_T07r_best": {
            "null": summ("best_T04_minus_best_T07r"), "real": float(real_best - real_best7),
            "empirical_p_two_sided": float(np.mean(
                [abs(x) >= abs(real_best - real_best7) for x in S["best_T04_minus_best_T07r"]]))}}


def perm_view(v, perm):
    return W.PoolView(v.raw, {k: x[perm] for k, x in v.labs.items()},
                      {k: x[perm] for k, x in v.strongs.items()}, v.ds_index, v.item_keys,
                      v.gen_tokens)


# ================================================================== console
def console(O):
    print("\n" + "=" * 122)
    print("KNOB 4 -- WEITZMAN CONTROLLER AT T=0.4")
    print("=" * 122)
    print(f"NULL TESTS passed: {O['null_test_passed']}  {O['null_test_max_abs_deviation']}")
    print("\nARMS (open-3 macro, equal weight per open cell)")
    print(f"  {'arm':46s} {'acc_j':>8s} {'acc_em':>8s} {'meanN':>7s} {'esc%':>7s} {'FLOPeq':>7s} "
          f"{'latseq':>8s} {'latpar':>8s}")
    for k in sorted(O["ARMS"]):
        b = O["ARMS"][k]["open3_macro"]
        print(f"  {k:46s} {b['acc_judge']:8.5f} {b['acc_em']:8.5f} {b['meanN']:7.3f} "
              f"{b['esc']*100:7.2f} {b['flops_eq']:7.3f} {b['lat_seq_ms']:8.0f} {b['lat_par_ms']:8.0f}")
    print("\nCONTRASTS (open-3 macro, paired item bootstrap)")
    for k in sorted(O["ARM_CONTRASTS"]):
        j = O["ARM_CONTRASTS"][k]["open3_macro_judge"]; e = O["ARM_CONTRASTS"][k]["open3_macro_em"]
        print(f"  {k:54s} judge {j['delta']:+.5f} [{j['lo']:+.5f},{j['hi']:+.5f}] {j['verdict']:5s}"
              f" | em {e['delta']:+.5f} [{e['lo']:+.5f},{e['hi']:+.5f}] {e['verdict']:5s}"
              f" | dFLOP {O['ARM_CONTRASTS'][k]['open3_macro_d_flops_eq']:+.3f}")
    op = O["OPERATING_POINTS"]
    print("\nMIN-COMPUTE POINT AT THE always-32B-direct BAR (open-3 macro)")
    for k, v in op["selected"].items():
        print(f"  {k:28s} {v}")
    print("\nADAPTIVE vs FIXED-N at equal or lower cost (T=0.4)")
    s = op["adaptive_vs_fixedN_T04"]
    print(f"  points={s['n_points']}  adaptive wins at {s['n_where_adaptive_wins']}  "
          f"median {s['median_advantage']}  max {s['max_advantage']}  min {s['min_advantage']}")
    print("\nPERMUTATION NULL")
    pn = O["PERMUTATION_NULL"]
    print(f"  reached the bar under shuffled labels: "
          f"T04 {pn['S1_did_the_null_ever_reach_the_bar']['null_rate_T04']:.3f}  "
          f"T07r {pn['S1_did_the_null_ever_reach_the_bar']['null_rate_T07r']:.3f}")
    print(f"  refit-minus-stale: real {pn['S4_refit_minus_stale']['real']:+.5f}  "
          f"p={pn['S4_refit_minus_stale']['empirical_p_two_sided']:.3f}  "
          f"null {pn['S4_refit_minus_stale']['null']}")


if __name__ == "__main__":
    main()
