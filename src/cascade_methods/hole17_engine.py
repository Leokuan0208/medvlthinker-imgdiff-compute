#!/usr/bin/env python3
"""hole17_engine.py -- the policy families, their exact accuracy/cost curves, and the fitting rules.

TWO OBJECTIVES ARE COMPARED, ON IDENTICAL POLICY FAMILIES AND IDENTICAL FOLDS.

  INCUMBENT (what ships).  Per cell, independently:
      MCQ  tau  = min-escalation tau s.t. TRAIN cascade acc >= TRAIN 32B acc      (IM.pick_tau_isocost)
      OPEN lam  = min-FLOP lam s.t. TRAIN Pandora acc >= target_k - 3e-3          (IP / paper_baselines)
    i.e. a PER-CELL accuracy FLOOR, minimised on that cell's own compute.

  REFIT (the macro objective).  A COMMON exchange rate mu (accuracy per FLOP-eq) across all 8 cells:
      theta_k(mu) = argmax_theta [ acc_k(theta) - mu * cost_k(theta) ]  on TRAIN
    then mu is chosen to hit an aggregate anchor (macro parity in accuracy, or macro parity in cost).

THE ALGEBRA THAT MAKES THIS THE RIGHT TEST.  The per-cell thresholds are independent, so for a fixed
mu the argmax above does NOT depend on the reporting weights w_k -- w_k multiplies a term that
contains only theta_k.  Pooled-optimal and macro-optimal threshold vectors at the same mu are
therefore IDENTICAL, and the weighting can only enter through which mu an aggregate anchor selects.
This file computes both anchors so that statement is measured, not asserted.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_engine.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np
from sklearn.isotonic import IsotonicRegression

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

import pandora_controller as PC
import hole17_data as HD

LAMS = PC.LAMS                      # the incumbent's own 91-point lambda grid, verbatim
ISO_TOL = 3e-3                      # the incumbent's iso band, verbatim
C_CHEAP, C_STRONG = HD.C_CHEAP, HD.C_STRONG
F_MCQ_CHEAP, F_MCQ_STRONG = 1.0, 4.57


# ===================================================================================================
# MCQ policy family:  escalate iff margin < tau.  Exact curves over every candidate tau, O(n log n).
# ===================================================================================================
def mcq_curves(ok7, ok32, margin):
    """Return (taus, acc, esc) over the same candidate grid pick_tau_isocost uses.
    Escalating means taking ok32; tau sweeps low->high so the escalated set grows by margin order."""
    o = np.argsort(margin, kind="mergesort")
    m, a7, a32 = margin[o], ok7[o], ok32[o]
    n = len(m)
    c32 = np.concatenate([[0.0], np.cumsum(a32)])           # sum of ok32 over the first i (escalated)
    c7 = np.concatenate([[0.0], np.cumsum(a7)])
    tot7 = c7[-1]
    uniq = np.unique(m)
    # tau candidates: below-all, each unique margin, above-all  (escalate iff margin < tau)
    taus = np.concatenate([[uniq[0] - 1e-9], uniq, [uniq[-1] + 1e-9]])
    k = np.searchsorted(m, taus, side="left")               # number escalated at each tau
    acc = (c32[k] + (tot7 - c7[k])) / n
    esc = k / n
    return taus, acc, esc


def mcq_apply(ok7, ok32, margin, tau):
    e = margin < tau
    return np.where(e, ok32, ok7), e.astype(float)


def mcq_cost(esc):
    return F_MCQ_CHEAP + F_MCQ_STRONG * esc


# ===================================================================================================
# OPEN policy family: Weitzman.  lam -> (z_cheap, z_strong).  Curves over the incumbent's lam grid.
# ===================================================================================================
def zeta_cheap_exact(v, lam, c=C_CHEAP):
    """Closed-form inverse of  mean((v-z)^+) = lam*c.  Asserted against PC.zeta_cheap (bisection)."""
    t = lam * c
    if t <= 0.0: return float(np.max(v))
    vs = np.sort(np.asarray(v, float))[::-1]
    n = len(vs)
    if t >= vs.mean() - vs[-1]: return float(vs.mean() - t)
    S = np.cumsum(vs)
    # for z in [vs[i], vs[i-1]] the top-i values exceed z:  g(z) = (S[i-1] - i*z)/n
    for i in range(1, n + 1):
        z = (S[i - 1] - n * t) / i
        lo = vs[i] if i < n else -np.inf
        if z >= lo - 1e-15 and z <= vs[i - 1] + 1e-15:
            return float(z)
    return float(vs.mean() - t)


def open_fit_calibrator(raw_tr, sl_tr):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw_tr.ravel(), sl_tr.ravel())
    return iso


def open_curves(cal_tr, q_tr, cal, raw, sl, strong, lams=LAMS):
    """(lams, acc, cost, meanN, esc) on the items whose (cal, raw, sl, strong) are given.
    z_cheap comes from the TRAIN calibrated pool, z_strong from the TRAIN strong accuracy."""
    accs = np.empty(len(lams)); cost = np.empty(len(lams))
    mN = np.empty(len(lams)); esc = np.empty(len(lams))
    for j, lam in enumerate(lams):
        z_c = zeta_cheap_exact(cal_tr, lam); z_s = PC.zeta_strong(q_tr, lam)
        N, E, O = HD.pandora_vec(cal, raw, sl, strong, z_c, z_s)
        accs[j] = O.mean(); mN[j] = N.mean(); esc[j] = E.mean()
        cost[j] = mN[j] * C_CHEAP + esc[j] * C_STRONG
    return lams, accs, cost, mN, esc


def open_apply(cal_tr, q_tr, lam, cal, raw, sl, strong):
    z_c = zeta_cheap_exact(cal_tr, lam); z_s = PC.zeta_strong(q_tr, lam)
    return HD.pandora_vec(cal, raw, sl, strong, z_c, z_s)


# ===================================================================================================
# the two FITTING RULES
# ===================================================================================================
def fit_incumbent_mcq(ok7, ok32, margin):
    """min-escalation tau s.t. train acc >= train 32B acc  (fallback max-acc).  == IM.pick_tau_isocost."""
    taus, acc, esc = mcq_curves(ok7, ok32, margin)
    target = ok32.mean()
    ok = acc >= target - 1e-12
    if ok.any():
        idx = np.where(ok)[0]
        return float(taus[idx[np.argmin(esc[idx])]])
    return float(taus[int(np.argmax(acc))])


def fit_lagrange_mcq(ok7, ok32, margin, mu):
    taus, acc, esc = mcq_curves(ok7, ok32, margin)
    return float(taus[int(np.argmax(acc - mu * mcq_cost(esc)))])


def fit_incumbent_open(cal_tr, q_tr, cal, raw, sl, strong, target):
    """min-FLOP lam s.t. train acc >= target - ISO_TOL (fallback max-train-acc).  == paper_baselines."""
    lams, acc, cost, _, _ = open_curves(cal_tr, q_tr, cal, raw, sl, strong)
    ok = acc >= target - ISO_TOL
    if ok.any():
        idx = np.where(ok)[0]
        return float(lams[idx[np.argmin(cost[idx])]])
    return float(lams[int(np.argmax(acc))])


def fit_lagrange_open(cal_tr, q_tr, cal, raw, sl, strong, mu):
    lams, acc, cost, _, _ = open_curves(cal_tr, q_tr, cal, raw, sl, strong)
    return float(lams[int(np.argmax(acc - mu * cost))])


if __name__ == "__main__":
    import json, integrated_pandora as IP
    IP.ADAPTER = "ckpts/train/lora_verifier_disjoint"
    openc = HD.load_open()
    # zeta_cheap_exact must equal PC.zeta_cheap (80-step bisection) over the whole lambda grid
    worst = 0.0
    for name, d in openc.items():
        iso = open_fit_calibrator(d["raw"], d["sl"])
        pool = iso.predict(d["raw"].ravel())
        for lam in LAMS:
            a = zeta_cheap_exact(pool, lam); b = PC.zeta_cheap(pool, lam)
            worst = max(worst, abs(a - b))
    print("max |zeta_cheap_exact - PC.zeta_cheap| over 3 cells x %d lambdas = %.3e" % (len(LAMS), worst))
    assert worst < 1e-9, worst
    # MCQ curve engine must match a brute-force evaluation
    mcq = HD.load_mcq()
    rng = np.random.default_rng(0); bad = 0.0
    for name, d in mcq.items():
        taus, acc, esc = mcq_curves(d["ok7"], d["ok32"], d["margin"])
        for t in rng.choice(taus, size=25, replace=False):
            ok, e = mcq_apply(d["ok7"], d["ok32"], d["margin"], t)
            j = int(np.where(taus == t)[0][0])
            bad = max(bad, abs(ok.mean() - acc[j]), abs(e.mean() - esc[j]))
    print("max |mcq_curves - brute force| over 5 cells x 25 taus = %.3e" % bad)
    assert bad < 1e-12, bad
    print("ENGINE CURVES VERIFIED.")
