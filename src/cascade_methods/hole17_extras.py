#!/usr/bin/env python3
"""hole17_extras.py -- two side findings from the hole-17 refit.

(A) A COSTING INCONSISTENCY IN THE SHIPPED ACCURACY-MAX ARM.
    method_final_mmmu_corrected.add_v2_vectors:155-162 builds the open-text accuracy-max vector from
    beat32b_more.f10_l2d / open_features, whose 7B answer is  pick = argmax(scores[:8])  and whose 7
    gate features (smax, srange, smean, sstd, n_uniq_pred, sc_frac, seqlogprob) are ALL computed over
    all 8 candidates.  That vector therefore requires 8 generations + 8 verifier forwards = 16.0
    FLOP-eq.  The same function then charges the cell  cost_pandora(c["meanN"], esc_F10)  where
    meanN is the COMPUTE-LEAN Pandora adaptive draw count (4.37-6.63, not 8).  Accuracy from 8 draws,
    cost from fewer than 8.  Both numbers are real; the pairing is not.

(B) DECOUPLING THE WEITZMAN LAMBDA.  One lambda sets BOTH reservations
    (z_cheap = draw-another, z_strong = escalate).  This sweeps them independently.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_extras.py
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hole17_run as R
import hole17_engine as EN
import hole17_data as HD
import hole17_currency as CU
from pandora_controller import zeta_strong

FIXED_BO8_FLOPS = 8 * HD.C_CHEAP      # 16.0: 8 generations + 8 verifier forwards


# ---------------------------------------------------------------------------------------------
# (A) the accuracy-max costing inconsistency, computed from the published artifact only
# ---------------------------------------------------------------------------------------------
def amax_cost_audit():
    S = json.load(open(os.path.join(R.ART, "_selector_rerun_parts", "summary_disjoint.json")))
    det = S["open_cell_detail"]
    per = {}
    for k in R.OPEN_B:
        meanN = det[k]["meanN"]; esc10 = det[k]["am2_esc"]
        charged = det[k]["cost_am2"]["flops"]
        consistent = FIXED_BO8_FLOPS + esc10 * HD.C_STRONG
        per[k] = dict(meanN_charged=meanN, esc_F10=esc10, flops_as_charged=charged,
                      flops_consistent_with_the_accuracy_vector=consistent,
                      undercharge=consistent - charged,
                      check_charged_formula=meanN * HD.C_CHEAP + esc10 * HD.C_STRONG)
    d_macro = sum(per[k]["undercharge"] for k in R.OPEN_B) / 8.0
    pub = S["cost_macro"]["method_accuracy_max_veto"]["flops"]
    pub_cl = S["cost_macro"]["method_compute_lean"]["flops"]
    return dict(per_cell=per,
                published_macro_flops_accuracy_max=pub,
                consistent_macro_flops_accuracy_max=pub + d_macro,
                delta_macro_flops=d_macro,
                published_x_direct=pub / R.DIRECT_FLOPS,
                consistent_x_direct=(pub + d_macro) / R.DIRECT_FLOPS,
                compute_lean_macro_flops_unaffected=pub_cl,
                note="COMPUTE-LEAN is NOT affected: its open cells' accuracy vector IS the Pandora "
                     "adaptive-N outcome, so meanN is the right charge there.  Only the accuracy-max "
                     "arm pairs a fixed-best-of-8 accuracy vector with an adaptive-N cost.",
                source_lines=["src/cascade_methods/method_final_mmmu_corrected.py:155-162",
                              "src/cascade_methods/beat32b_more.py:237-254 (pick = argmax(scores[:8]))"])


# ---------------------------------------------------------------------------------------------
# (B) decoupled (z_cheap, z_strong)
# ---------------------------------------------------------------------------------------------
def decoupled_curves(cal_tr_pool, q_tr, cal, raw, sl, strong, lams=EN.LAMS):
    """acc/cost/esc/meanN on a (lam_c, lam_s) grid.  For a fixed z_c the draw trajectory, the picked
    slot and the exhausted set are FIXED, so the whole z_s axis is one sort + searchsorted."""
    n, Nmax = cal.shape
    A = np.zeros((len(lams), len(lams))); C = np.zeros_like(A)
    E = np.zeros_like(A); MN = np.zeros_like(A)
    zs_all = np.array([zeta_strong(q_tr, l) for l in lams])
    for a, lam_c in enumerate(lams):
        z_c = EN.zeta_cheap_exact(cal_tr_pool, lam_c)
        N, esc0, pick = CU.pandora_pick(cal, raw, z_c, -np.inf)   # z_s=-inf: never escalate
        base = sl[np.arange(n), pick]
        exhausted = (N == Nmax)
        bc = np.where(np.arange(Nmax)[None, :] < N[:, None], cal, -np.inf).max(axis=1)
        gain = strong - base
        m = exhausted
        order = np.argsort(bc[m], kind="mergesort")
        bcs = bc[m][order]; gs = np.concatenate([[0.0], np.cumsum(gain[m][order])])
        cnt = np.arange(len(bcs) + 1)
        j = np.searchsorted(bcs, zs_all, side="left")             # #escalated = # of bc < z_s
        # a z_s above z_c makes the strong box preempt at k=0: escalate everything, N=0
        preempt = zs_all > z_c
        acc = base.mean() + gs[j] / n
        escr = cnt[j] / n
        meanN = N.mean()
        A[a] = np.where(preempt, strong.mean(), acc)
        E[a] = np.where(preempt, 1.0, escr)
        MN[a] = np.where(preempt, 0.0, meanN)
        C[a] = MN[a] * HD.C_CHEAP + E[a] * HD.C_STRONG
    return A, C, E, MN


def decoupled_eval(D, fa, mus=R.MUS):
    """5-fold cross-fit; per fold the (lam_c, lam_s) pair is the Lagrangian argmax on TRAIN."""
    out = {}
    for k in R.OPEN_B:
        c = D.open[k]
        acc = np.zeros(len(mus)); cost = np.zeros(len(mus)); escr = np.zeros(len(mus))
        coup_a = np.zeros(len(mus)); coup_c = np.zeros(len(mus))
        for f in range(R.K_OUTER):
            te = fa[k] == f; tr = ~te
            iso = EN.open_fit_calibrator(c["raw"][tr], c["sl"][tr])
            pool = iso.predict(c["raw"][tr].ravel())
            q = float(c["strong"][tr].mean())
            cal_tr = iso.predict(c["raw"][tr].ravel()).reshape(c["raw"][tr].shape)
            cal_te = iso.predict(c["raw"][te].ravel()).reshape(c["raw"][te].shape)
            Atr, Ctr, _, _ = decoupled_curves(pool, q, cal_tr, c["raw"][tr], c["sl"][tr], c["strong"][tr])
            Ate, Cte, Ete, _ = decoupled_curves(pool, q, cal_te, c["raw"][te], c["sl"][te], c["strong"][te])
            nte = int(te.sum())
            for i, mu in enumerate(mus):
                J = Atr - mu * Ctr
                a, b = np.unravel_index(int(np.argmax(J)), J.shape)
                acc[i] += Ate[a, b] * nte; cost[i] += Cte[a, b] * nte; escr[i] += Ete[a, b] * nte
                d = np.diag(Atr) - mu * np.diag(Ctr)               # the COUPLED family lam_c == lam_s
                g = int(np.argmax(d))
                coup_a[i] += Ate[g, g] * nte; coup_c[i] += Cte[g, g] * nte
        out[k] = dict(acc=acc / D.n[k], cost=cost / D.n[k], esc=escr / D.n[k],
                      coupled_acc=coup_a / D.n[k], coupled_cost=coup_c / D.n[k])
    return out


if __name__ == "__main__":
    t0 = time.time()
    audit = amax_cost_audit()
    print(json.dumps(audit, indent=1, default=float))
    D = R.Data()
    fa = {k: R.folds(D.n[k], R.K_OUTER) for k in R.ORDER_B}
    dec = decoupled_eval(D, fa)
    # macro over the 3 OPEN cells only (the 5 MCQ cells are untouched by this knob)
    A = sum(dec[k]["acc"] for k in R.OPEN_B) / 3; C = sum(dec[k]["cost"] for k in R.OPEN_B) / 3
    Ac = sum(dec[k]["coupled_acc"] for k in R.OPEN_B) / 3; Cc = sum(dec[k]["coupled_cost"] for k in R.OPEN_B) / 3
    j = int(np.argmax(A)); jc = int(np.argmax(Ac))
    print("\nOPEN-ONLY macro (3 cells, eval-visible best over mu):")
    print("  coupled  (shipped family) best acc %.6f at cost %.4f" % (Ac[jc], Cc[jc]))
    print("  decoupled (z_c, z_s free) best acc %.6f at cost %.4f" % (A[j], C[j]))
    print("  decoupling gain on the 3 open cells: %+.6f  ->  on the 8-cell macro: %+.6f"
          % (A[j] - Ac[jc], (A[j] - Ac[jc]) * 3 / 8))
    json.dump(dict(amax_cost_audit=audit,
                   decoupled=dict(mu=R.MUS.tolist(), open_macro_acc=A.tolist(), open_macro_cost=C.tolist(),
                                  coupled_open_macro_acc=Ac.tolist(), coupled_open_macro_cost=Cc.tolist(),
                                  per_cell={k: {q: dec[k][q].tolist() for q in dec[k]} for k in R.OPEN_B})),
              open(os.path.join(R.ART, "_hole17_extras.json"), "w"), indent=1, default=float)
    print("\nwrote _hole17_extras.json (%.1fs)" % (time.time() - t0))
