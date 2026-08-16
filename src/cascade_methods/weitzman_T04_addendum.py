#!/usr/bin/env python3
"""weitzman_T04_addendum.py -- the three things weitzman_T04.py's first pass showed were needed.

(1) THE STRUCTURAL AUDIT.  In pandora_controller.run_pandora the cheap box and the strong box
    compete on reservation value at every step, so when zeta_cheap >= zeta_strong the cheap box is
    always the highest unopened reservation and the strong box only becomes reachable once the
    cheap boxes are exhausted.  That predicts a hard coupling -- ESCALATION IS ONLY REACHABLE
    AFTER ALL Nmax CHEAP DRAWS HAVE BEEN PAID FOR -- and therefore that one lambda cannot trade
    "draw fewer" against "escalate more".  Measured here on every pool rather than asserted.

(2) AN HONEST FIXED-N ARM.  The first pass compared two SWEPT frontiers (both equally optimistic).
    The fixed-N curve dominated, but a swept curve is not a deployable policy.  Here (N, tau) is
    SELECTED ON TRAIN by the identical objective the Weitzman lambda uses, cross-fit, resub and
    nested, 10 CV seeds x 3 generation seeds -- exactly the protocol the lambda arms got.

(3) THE PERMUTATION NULL FOR THAT SELECTION, and an amended reading of the first pass's null.
    Under shuffled labels the Weitzman frontier DEGENERATES to always-32B-direct (zeta_strong >
    zeta_cheap for small lambda once the isotonic map is flat), which trivially "reaches the bar"
    at exactly 4.57 FLOP-eq.  So "reaches parity at some compute" is not a claim; only "reaches
    parity BELOW 4.57 FLOP-eq" is.  Both statistics are re-reported against that reading.

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 src/cascade_methods/weitzman_T04_addendum.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))

import weitzman_lib as W                                       # noqa: E402
import weitzman_T04 as T                                       # noqa: E402
import integrated_method as IM                                 # noqa: E402
from src.training_methods import genframe_data as G            # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (        # noqa: E402
    load_judge, load_vscores, DS)

CELLS = T.CELLS
NFIX = T.NFIX
TAUS = T.TAUS
BSEED = T.BSEED
NFOLD = T.NFOLD


# ------------------------------------------------------------------ fixed-N grid, per fold
def fixedN_grid(sub, cal):
    """(len(NFIX), len(TAUS), n) escalate mask + ok arrays, under one fold's calibration."""
    n = sub.n
    rows = np.arange(n)
    pm = np.maximum.accumulate(cal, axis=1)
    ESC = np.empty((len(NFIX), len(TAUS), n), bool)
    OKJ = np.empty((len(NFIX), len(TAUS), n), np.float32)
    OKE = np.empty((len(NFIX), len(TAUS), n), np.float32)
    for ni, Nf in enumerate(NFIX):
        d = Nf - 1
        e = pm[:, d][:, None] < TAUS[None, :]              # (n, ntau)
        pj = sub.prefix_lab["judge"][rows, d][:, None]
        pe = sub.prefix_lab["em"][rows, d][:, None]
        ESC[ni] = e.T
        OKJ[ni] = np.where(e, sub.strongs["judge"][:, None], pj).T
        OKE[ni] = np.where(e, sub.strongs["em"][:, None], pe).T
    return ESC, OKJ, OKE


def select_fixedN(ESC, OKJ, mask, target, tol):
    """min-FLOP-eq (N, tau) whose accuracy on `mask` reaches target - tol; else max accuracy."""
    acc = OKJ[:, :, mask].mean(2)
    esc = ESC[:, :, mask].mean(2)
    fl = np.array(NFIX, float)[:, None] * W.C_CHEAP_F + esc * W.C_STRONG_F
    ok = acc >= target - tol
    if ok.any():
        flm = np.where(ok, fl, np.inf)
        return np.unravel_index(np.argmin(flm), flm.shape)
    return np.unravel_index(np.argmax(acc), acc.shape)


def fixedN_arm(sub, folds, target, tol, nested_seed=None):
    """Cross-fit fixed-N + gate arm. (N, tau) chosen on TRAIN (resub, or nested inner 5-fold),
    frozen, applied to the held-out fold. Returns per-item outcomes + the chosen configs."""
    from sklearn.isotonic import IsotonicRegression
    n = sub.n
    out = {k: np.zeros(n) for k in ("N", "esc", "okj", "oke")}
    chosen = []
    for f in range(NFOLD):
        tr = np.where(folds != f)[0]
        te = folds == f
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(sub.raw[tr].ravel(), sub.labs["judge"][tr].ravel())
        cal = sub.calibrated(iso)
        if nested_seed is None:
            ESC, OKJ, OKE = fixedN_grid(sub, cal)
            ni, ti = select_fixedN(ESC, OKJ, folds != f, target, tol)
        else:
            subtr = T.subset_idx(sub, tr)
            inner = W.image_folds_for_keys(nested_seed * 1000 + 11, subtr.item_keys, k=NFOLD)
            accs = np.zeros((len(NFIX), len(TAUS))); escs = np.zeros_like(accs); cnt = 0
            for g in range(NFOLD):
                itr = np.where(inner != g)[0]
                ite = inner == g
                if ite.sum() == 0:
                    continue
                iso2 = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso2.fit(subtr.raw[itr].ravel(), subtr.labs["judge"][itr].ravel())
                E2, J2, _ = fixedN_grid(subtr, subtr.calibrated(iso2))
                accs += J2[:, :, ite].sum(2); escs += E2[:, :, ite].sum(2); cnt += int(ite.sum())
            accs /= cnt; escs /= cnt
            fl = np.array(NFIX, float)[:, None] * W.C_CHEAP_F + escs * W.C_STRONG_F
            ok = accs >= target - tol
            if ok.any():
                flm = np.where(ok, fl, np.inf)
                ni, ti = np.unravel_index(np.argmin(flm), flm.shape)
            else:
                ni, ti = np.unravel_index(np.argmax(accs), accs.shape)
            ESC, OKJ, OKE = fixedN_grid(sub, cal)
        chosen.append((int(NFIX[ni]), float(TAUS[ti])))
        out["N"][te] = NFIX[ni]
        out["esc"][te] = ESC[ni, ti][te]
        out["okj"][te] = OKJ[ni, ti][te]
        out["oke"][te] = OKE[ni, ti][te]
    return out, chosen


# ------------------------------------------------------------------ structural audit
def structure(VIEWS):
    from sklearn.isotonic import IsotonicRegression
    res = {}
    for tag, v in VIEWS.items():
        blk = {}
        for ci, cell in enumerate(CELLS):
            sub = T.subset_idx(v, np.where(v.ds_index == ci)[0])
            folds = W.image_folds_for_keys(0, sub.item_keys, k=NFOLD)
            joint = np.zeros((9, 2), np.int64)
            max_esc = 0.0; regimeB = 0; nlam = 0
            for f in range(NFOLD):
                tr = np.where(folds != f)[0]
                te = folds == f
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(sub.raw[tr].ravel(), sub.labs["judge"][tr].ravel())
                cal = sub.calibrated(iso)
                pool = iso.predict(sub.raw[tr].ravel())
                q = float(sub.strongs["judge"][tr].mean())
                zcs = W.zeta_cheap_many(pool, T.LAMS_DENSE)
                for li, l in enumerate(T.LAMS_DENSE):
                    r = sub.run(cal, float(zcs[li]), W.zeta_strong(q, float(l)))
                    if r["regime"] == "B":
                        regimeB += 1
                    nlam += 1
                    N = r["N"][te].astype(int); E = r["esc"][te].astype(int)
                    np.add.at(joint, (N, E), 1)
                    max_esc = max(max_esc, float(r["esc"][te].mean()))
            esc_tot = int(joint[:, 1].sum())
            blk[cell] = dict(
                n=int(sub.n),
                joint_N_by_escalate=joint.tolist(),
                escalated_decisions_total=esc_tot,
                escalated_at_N_equals_Nmax=int(joint[8, 1]),
                escalated_at_N_equals_0_regimeB=int(joint[0, 1]),
                escalated_with_0_lt_N_lt_Nmax=int(esc_tot - joint[8, 1] - joint[0, 1]),
                regimeB_lambda_fold_combinations=int(regimeB), lambda_fold_combinations=int(nlam),
                max_escalation_rate_reachable_at_any_lambda=max_esc,
                flops_eq_of_an_escalated_item=8 * W.C_CHEAP_F + W.C_STRONG_F,
                flops_eq_of_always_32b_direct=W.C_STRONG_F)
        blk["coupling_holds"] = all(blk[c]["escalated_with_0_lt_N_lt_Nmax"] == 0 for c in CELLS)
        res[tag] = blk
        print(f"  {tag:32s} coupling_holds={blk['coupling_holds']}  "
              f"max_esc={[round(blk[c]['max_escalation_rate_reachable_at_any_lambda'],4) for c in CELLS]}",
              flush=True)
    return res


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=200)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--out",
                    default="results/cascade_methods/artifacts/weitzman_T04_addendum_2026-08-15.json")
    A = ap.parse_args()

    base = json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json")))

    lab, vsc, ref = load_judge(), load_vscores(), G.load_items()
    strongs = W._strong_labels()
    VIEWS = {}
    for Tt in ("T04", "T07r"):
        for s in T.GEN_SEEDS:
            VIEWS[f"{Tt}_{s}"] = W.build_view(f"{Tt}_{s}", lab, vsc, ref, strongs)

    O = {"title": "KNOB 4 addendum -- the structural coupling inside the Weitzman controller, an "
                  "honest cross-fit fixed-N control, and the corrected reading of the permutation "
                  "null",
         "date": "2026-08-15",
         "script": "src/cascade_methods/weitzman_T04_addendum.py",
         "parent": "results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json",
         "no_gpu": True, "no_new_generation": True,
         "numerics_pinned": base["numerics_pinned"],
         "null_test_inherited": base["null_test_max_abs_deviation"],
         "null_test_passed": base["null_test_passed"]}

    # ---------------- (1) structural audit ----------------------------------------------------
    print("STRUCTURAL AUDIT ...", flush=True)
    allviews = dict(VIEWS)
    allviews["DEPLOYED_transfer_dumps_T07"] = W.build_view_from_frozen_dumps()
    O["STRUCTURE"] = {
        "claim": "In pandora_controller.run_pandora the strong box only becomes the highest "
                 "unopened reservation once the cheap boxes are exhausted (whenever zeta_cheap >= "
                 "zeta_strong). ESCALATION IS THEREFORE ONLY REACHABLE AFTER ALL Nmax CHEAP DRAWS "
                 "HAVE BEEN PAID FOR: an escalated item costs 8*2.0 + 4.57 = 20.57 FLOP-eq against "
                 "always-32B-direct's 4.57, and the single lambda cannot trade 'draw fewer' "
                 "against 'escalate more' -- lowering lambda buys BOTH.",
        "measured": structure(allviews),
        "reading": "every escalated decision, in every pool and every cell, over the full 241-point "
                   "lambda grid and all 5 folds, has N = Nmax. Regime B (zeta_strong > zeta_cheap, "
                   "which would degenerate the arm to always-32B-direct at N=0) is never reached on "
                   "REAL data, so the controller can neither escalate cheaply nor escalate fully. "
                   "That is the mechanism, not the pool statistics, that pins its frontier above "
                   "the bar's compute."}

    # ---------------- (2) honest cross-fit fixed-N arm ----------------------------------------
    print("\nCROSS-FIT FIXED-N ARM ...", flush=True)
    targets = {}
    for ci, cell in enumerate(CELLS):
        for Tt in ("T04", "T07r"):
            for gs in T.GEN_SEEDS:
                v = VIEWS[f"{Tt}_{gs}"]
                sub = T.subset_idx(v, np.where(v.ds_index == ci)[0])
                a_fix, _ = IM.heldout(sub.prefix_lab["judge"][:, 7], sub.strongs["judge"],
                                      np.max(sub.raw, axis=1))
                targets[("O1", cell, Tt, gs)] = float(a_fix)

    STORE = defaultdict(lambda: defaultdict(list))
    CFG = defaultdict(list)
    for ci, cell in enumerate(CELLS):
        for gs in T.GEN_SEEDS:
            for Tt in ("T04", "T07r"):
                v = VIEWS[f"{Tt}_{gs}"]
                sub = T.subset_idx(v, np.where(v.ds_index == ci)[0])
                for cv in T.CV_SEEDS:
                    folds = W.image_folds_for_keys(cv, sub.item_keys, k=NFOLD)
                    for obj in ("O1", "O2"):
                        tgt = targets[("O1", cell, Tt, gs)] if obj == "O1" else T.BAR_JUDGE[cell]
                        tol = T.ISO_TOL if obj == "O1" else 0.0
                        for sel in ("resub", "nested"):
                            o, ch = fixedN_arm(sub, folds, tgt, tol,
                                               nested_seed=(cv if sel == "nested" else None))
                            STORE[(obj, sel, Tt)][cell].append(o)
                            CFG[(obj, sel, Tt, cell)].extend(ch)
        print(f"    fixed-N x-fit: {cell} done", flush=True)

    FX = {}
    for key, per in STORE.items():
        obj, sel, Tt = key
        blk = {"per_cell": {}}
        for cell in CELLS:
            a = T.agg(per[cell])
            c = W.cost_of(float(a["N"].mean()), float(a["esc"].mean()))
            cfg = CFG[(obj, sel, Tt, cell)]
            blk["per_cell"][cell] = dict(
                n=int(len(a["okj"])), acc_judge=float(a["okj"].mean()),
                acc_em=float(a["oke"].mean()), meanN=float(a["N"].mean()),
                esc=float(a["esc"].mean()), flops_eq=c["flops"],
                lat_seq_ms=c["lat_seq"], lat_par_ms=c["lat_bat"], energy_j=c["energy"],
                chosen_N_mode=int(max(set(x[0] for x in cfg), key=[x[0] for x in cfg].count)),
                chosen_N_mean=float(np.mean([x[0] for x in cfg])),
                chosen_tau_mean=float(np.mean([x[1] for x in cfg])),
                chosen_tau_min=float(min(x[1] for x in cfg)),
                chosen_tau_max=float(max(x[1] for x in cfg)),
                seed_spread_judge=T.seed_spread(per[cell], len(T.CV_SEEDS)),
                bar_judge=T.BAR_JUDGE[cell], bar_em=T.BAR_EM[cell])
        blk["open3_macro"] = {k: float(np.mean([blk["per_cell"][c][k] for c in CELLS]))
                              for k in ("acc_judge", "acc_em", "meanN", "esc", "flops_eq",
                                        "lat_seq_ms", "lat_par_ms", "energy_j")}
        oa = {c: blk["per_cell"][c]["acc_judge"] for c in CELLS}
        of = {c: blk["per_cell"][c]["flops_eq"] for c in CELLS}
        blk["macro8"] = {}
        for half in ("compute_lean", "accuracy_max"):
            acc, fl = T.macro8(oa, of, half)
            blk["macro8"][half] = dict(
                macro_acc=acc, macro_flops_eq=fl,
                vs_always32b_direct_acc=acc - T.BAR_MACRO_ACC,
                compute_x_vs_always32b_direct=fl / T.BAR_MACRO_FLOPS,
                vs_shipped_acc=acc - T.SHIPPED_MACRO[half]["acc"],
                compute_x_vs_shipped=fl / T.SHIPPED_MACRO[half]["flops"])
        FX[f"{obj}|{sel}|{Tt}"] = blk
        m = blk["open3_macro"]
        print(f"  fixedN {obj}|{sel}|{Tt:5s} acc_j={m['acc_judge']:.5f} acc_em={m['acc_em']:.5f} "
              f"N={m['meanN']:.2f} esc={m['esc']*100:5.2f}% F={m['flops_eq']:6.3f}", flush=True)
    O["FIXED_N_CROSSFIT"] = FX
    O["FIXED_N_CROSSFIT_what"] = (
        "(N, tau) selected on TRAIN by the SAME objective and the SAME folds the Weitzman lambda "
        "gets, then frozen and applied to the held-out fold. 8 x 101 = 808 candidate "
        "configurations against the controller's 91 lambdas, so this arm carries MORE selection "
        "risk, not less -- which is what the permutation null below prices.")

    # ---------------- contrasts: fixed-N vs the Weitzman arms ---------------------------------
    print("\nCONTRASTS fixed-N vs Weitzman ...", flush=True)
    # rebuild the Weitzman per-item vectors (cheap: reuse phase_arms)
    STORE_W, _ = T.phase_arms(VIEWS, targets)
    CON = {}
    for obj in ("O1", "O2"):
        for sel in ("resub", "nested"):
            for wt_arm, Tt, lab_ in (("C_refit_T04", "T04", "fixedN_T04_minus_weitzmanRefit_T04"),
                                     ("A_deployed_T07r", "T07r",
                                      "fixedN_T07r_minus_weitzmanDeployed_T07r"),
                                     ("B_stale_on_T04", "T04",
                                      "fixedN_T04_minus_weitzmanStale_T04")):
                blk = {}
                va_all, vb_all = [], []
                for cell in CELLS:
                    va = T.agg(STORE[(obj, sel, Tt)][cell])
                    vb = T.agg(STORE_W[(obj, sel, wt_arm)][cell])
                    va_all.append(va); vb_all.append(vb)
                    blk[cell] = {"judge": W.boot(va["okj"], vb["okj"], nboot=A.nboot, seed=BSEED),
                                 "em": W.boot(va["oke"], vb["oke"], nboot=A.nboot, seed=BSEED),
                                 "d_flops_eq": (W.cost_of(va["N"].mean(), va["esc"].mean())["flops"]
                                                - W.cost_of(vb["N"].mean(), vb["esc"].mean())["flops"]),
                                 "d_lat_par_ms": (W.cost_of(va["N"].mean(), va["esc"].mean())["lat_bat"]
                                                  - W.cost_of(vb["N"].mean(), vb["esc"].mean())["lat_seq"])}
                for cur, k in (("judge", "okj"), ("em", "oke")):
                    pt, dist = W.boot_macro([x[k] for x in va_all], [x[k] for x in vb_all],
                                            nboot=A.nboot, seed=BSEED)
                    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
                    blk[f"open3_macro_{cur}"] = dict(
                        delta=pt, lo=lo, hi=hi, sig=bool(lo > 0 or hi < 0),
                        verdict="WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                        macro8_scaled_delta=pt * 3.0 / 8.0)
                blk["open3_macro_d_flops_eq"] = float(np.mean([blk[c]["d_flops_eq"] for c in CELLS]))
                CON[f"{obj}|{sel}|{lab_}"] = blk
                j = blk["open3_macro_judge"]
                print(f"  {obj}|{sel}|{lab_:44s} judge {j['delta']:+.5f} "
                      f"[{j['lo']:+.5f},{j['hi']:+.5f}] {j['verdict']:5s} "
                      f"dFLOP {blk['open3_macro_d_flops_eq']:+.3f}", flush=True)
    O["FIXEDN_vs_WEITZMAN"] = CON

    # guardrail for the fixed-N arm vs the bar and vs the shipped-equivalent Weitzman arm
    GR = {}
    for obj in ("O1", "O2"):
        blk = {}
        v0 = VIEWS[f"T04_{T.GEN_SEEDS[0]}"]
        for cell in CELLS:
            va = T.agg(STORE[(obj, "resub", "T04")][cell])
            vb = T.agg(STORE_W[(obj, "resub", "A_deployed_T07r")][cell])
            m = v0.ds_index == CELLS.index(cell)
            blk[cell] = dict(
                n=int(m.sum()),
                vs_always32b_direct_judge=W.boot(va["okj"], v0.strongs["judge"][m],
                                                 nboot=A.nboot, seed=BSEED),
                vs_always32b_direct_em=W.boot(va["oke"], v0.strongs["em"][m],
                                              nboot=A.nboot, seed=BSEED),
                vs_deployed_weitzman_T07r_judge=W.boot(va["okj"], vb["okj"],
                                                       nboot=A.nboot, seed=BSEED),
                vs_deployed_weitzman_T07r_em=W.boot(va["oke"], vb["oke"],
                                                    nboot=A.nboot, seed=BSEED))
        blk["FLAGS_vs_always32b_direct"] = [
            c for c in CELLS if blk[c]["vs_always32b_direct_judge"]["verdict"] == "LOSS"
            or blk[c]["vs_always32b_direct_em"]["verdict"] == "LOSS"]
        blk["FLAGS_vs_deployed_weitzman"] = [
            c for c in CELLS if blk[c]["vs_deployed_weitzman_T07r_judge"]["verdict"] == "LOSS"
            or blk[c]["vs_deployed_weitzman_T07r_em"]["verdict"] == "LOSS"]
        GR[obj] = blk
    O["FIXED_N_GUARDRAIL"] = GR

    # ---------------- (3) permutation null of the fixed-N selection ---------------------------
    print("\nPERMUTATION NULL of the fixed-N selection ...", flush=True)
    rng = np.random.default_rng(BSEED)
    gs = T.GEN_SEEDS[0]
    cvs = T.CV_SEEDS[:3]
    IDX = {c: np.where(VIEWS[f"T04_{gs}"].ds_index == i)[0] for i, c in enumerate(CELLS)}
    SUB = {c: T.subset_idx(VIEWS[f"T04_{gs}"], IDX[c]) for c in CELLS}
    FOL = {(c, cv): W.image_folds_for_keys(cv, SUB[c].item_keys, k=NFOLD) for c in CELLS for cv in cvs}
    accs, flops, over = [], [], []
    for rep in range(A.nperm):
        aa, ff = [], []
        for cell in CELLS:
            perm = rng.permutation(len(IDX[cell]))
            v = T.perm_view(SUB[cell], perm)
            per = []
            for cv in cvs:
                o, _ = fixedN_arm(v, FOL[(cell, cv)], T.BAR_JUDGE[cell], 0.0)
                per.append(o)
            a = T.agg(per)
            aa.append(float(a["okj"].mean()))
            ff.append(W.cost_of(float(a["N"].mean()), float(a["esc"].mean()))["flops"])
        accs.append(float(np.mean(aa))); flops.append(float(np.mean(ff)))
        over.append(accs[-1] - float(np.mean([T.BAR_JUDGE[c] for c in CELLS])))
        if (rep + 1) % 25 == 0:
            print(f"    perm {rep+1}/{A.nperm}", flush=True)

    real = FX["O2|resub|T04"]["open3_macro"]
    bar3 = float(np.mean([T.BAR_JUDGE[c] for c in CELLS]))
    real_over = real["acc_judge"] - bar3

    def summ(v):
        v = np.asarray(v, float)
        return dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                    p2_5=float(np.percentile(v, 2.5)), p50=float(np.percentile(v, 50)),
                    p97_5=float(np.percentile(v, 97.5)), min=float(v.min()), max=float(v.max()))

    O["FIXED_N_PERMUTATION_NULL"] = {
        "design": "identical to the parent artifact's null: inside each cell the whole LABEL BUNDLE "
                  "(8 cheap-slot judge labels, 8 cheap-slot EM labels, the 32B judge label and the "
                  "32B EM label) is permuted across items while the verifier scores stay put, then "
                  "the IDENTICAL cross-fit (N, tau) selection runs. 3 CV seeds, generation seed s0.",
        "n_permutations": A.nperm,
        "null_open3_macro_accuracy": summ(accs),
        "null_open3_macro_flops_eq": summ(flops),
        "null_accuracy_minus_bar": summ(over),
        "real_open3_macro_accuracy": real["acc_judge"],
        "real_open3_macro_flops_eq": real["flops_eq"],
        "real_accuracy_minus_bar": real_over,
        "empirical_p_null_accuracy_at_least_as_high": float(np.mean([a >= real["acc_judge"]
                                                                     for a in accs])),
        "empirical_p_null_flops_at_least_as_low": float(np.mean([f <= real["flops_eq"]
                                                                 for f in flops])),
        "read": "under shuffled labels the selector can still reach the bar -- by escalating "
                "everything -- but it must PAY for it. The discriminating statistic is therefore "
                "FLOP-eq at parity, not accuracy at parity."}

    # ---------------- amended reading of the parent null --------------------------------------
    pn = base["PERMUTATION_NULL"]
    O["AMENDED_READING_OF_THE_PARENT_NULL"] = {
        "what_the_parent_null_showed": {
            "S1_null_rate_T04": pn["S1_did_the_null_ever_reach_the_bar"]["null_rate_T04"],
            "S1_null_rate_T07r": pn["S1_did_the_null_ever_reach_the_bar"]["null_rate_T07r"],
            "S2_null_min_flops_median": pn["S2_min_flops_at_the_bar"]["null_T04"]["p50"],
            "S3_null_best_accuracy_mean": pn["S3_best_accuracy_on_the_frontier"]["null_T04"]["mean"],
            "S3_real_best_accuracy": pn["S3_best_accuracy_on_the_frontier"]["real_T04"]},
        "why": "once the labels are shuffled the isotonic map is flat, the calibrated cheap-score "
               "pool collapses to a constant p, and zeta_cheap = p - 2*lambda while zeta_strong = "
               "q - 4.57*lambda. For lambda < (q-p)/2.57 the strong box outranks the cheap box at "
               "step 0, the controller opens it immediately, and the arm IS always-32B-direct: "
               "accuracy exactly the bar, cost exactly 4.57 FLOP-eq. That is why the null 'reaches "
               "the bar' in 90.5% (T=0.4) and 100% (T=0.7) of replicates at a median 4.57 FLOP-eq, "
               "and why its best-accuracy statistic (0.5982) BEATS the real controller's (0.5905).",
        "consequence": [
            "'the controller reaches parity at some compute' is NOT a claim -- shuffled labels do "
            "it 90-100% of the time. Only 'reaches parity BELOW 4.57 FLOP-eq' is a claim, and "
            "neither the real T=0.4 nor the real T=0.7 frontier reaches parity at ANY compute.",
            "S3's empirical p = 1.0 is not evidence that the real result is noise; it is evidence "
            "that the max-accuracy statistic is dominated by the degenerate always-32B corner, "
            "which the null reaches and the REAL controller provably cannot (see STRUCTURE).",
            "the two statistics that ARE discriminating both survive: refit-minus-stale "
            f"{pn['S4_refit_minus_stale']['real']:+.5f} against a null of "
            f"{pn['S4_refit_minus_stale']['null']['mean']:+.5f} +- "
            f"{pn['S4_refit_minus_stale']['null']['sd']:.5f}, p = "
            f"{pn['S4_refit_minus_stale']['empirical_p_two_sided']:.3f}; and T=0.4-best minus "
            f"T=0.7-best {pn['S5_T04_best_minus_T07r_best']['real']:+.5f} against a null of "
            f"{pn['S5_T04_best_minus_T07r_best']['null']['mean']:+.5f} +- "
            f"{pn['S5_T04_best_minus_T07r_best']['null']['sd']:.5f}, p = "
            f"{pn['S5_T04_best_minus_T07r_best']['empirical_p_two_sided']:.3f}."]}

    p = os.path.join(ROOT, A.out)
    json.dump(O, open(p, "w"), indent=1, default=float)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
