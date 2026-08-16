#!/usr/bin/env python3
"""hole17_amax_consistency.py -- quantify the documented-but-never-quantified accuracy-max
cost/accuracy mismatch on the 3 open cells.

The caveat is already on record (docs/current/METHOD_FINAL_2026-07.md:355-356 and
RESEARCH_RESULTS_2026-07.md:978-979): "F10 open-arm cost is billed at Pandora's adaptive meanN (<8)
draws while the kept-leg accuracy is scored on the best-of-8 verifier pick -- mildly optimistic".
It has never been given a number.  There are exactly two self-consistent repairs:

  (i)  CHARGE 8 DRAWS.  Keep the published accuracy vector, bill the 16.0 FLOP-eq the best-of-8 pick
       and the 8-candidate F10 features actually require.
  (ii) SCORE best-of-meanN.  Keep the published cost, re-pick each item from the first N candidates
       Pandora actually drew (N from the incumbent controller, floored at 1 since a kept item must
       have at least one answer), holding the F10 routing decision fixed.

Repair (ii) with the routing held fixed is an UPPER bound on the accuracy of a fully consistent
best-of-meanN arm, because F10's rejector was trained on 8-candidate features.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_amax_consistency.py
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hole17_run as R
import hole17_data as HD
import hole17_currency as CU
import beat32b_more as BB
import method_final_mmmu_corrected as MFC

FIXED_BO8_FLOPS = 8 * HD.C_CHEAP


def main():
    D = R.Data()
    BB.OPEN_VERIFIER_DIR = D.verifier
    em = CU.load_em(D.verifier)
    fa = {k: R.folds(D.n[k], R.K_OUTER) for k in R.ORDER_B}
    res = R.heldout_pass(D, fa, mus=np.array([0.0]))          # incumbent Pandora N per item
    out = {}
    for k in R.OPEN_B:
        ds = HD.OPEN_KEY[k]
        c = D.open[k]; n = c["n"]
        team, ok32_f10, took7 = MFC.f10_persample(ds)          # the SHIPPED accuracy-max open vector
        d = BB.open_features(ds)
        keep7 = np.array([team[i] == d["ok7"][i] and (d["ok7"][i] != d["ok32"][i] or True)
                          for i in range(n)])                  # placeholder, replaced below
        # recover the routing decision exactly: rerun f10 and keep the took7 mask
        keep7 = _f10_keep_mask(ds)
        ii = np.arange(n)
        pick8 = c["raw"].argmax(axis=1)
        N = np.maximum(res["incumbent"][k]["N"], 1).astype(int)
        maskN = np.arange(c["Nmax"])[None, :] < N[:, None]
        pickN = np.where(maskN, c["raw"], -np.inf).argmax(axis=1)
        ok7_8, ok7_N = c["sl"][ii, pick8], c["sl"][ii, pickN]
        em7_8, em7_N = em[k]["sl_em"][ii, pick8], em[k]["sl_em"][ii, pickN]
        team8 = np.where(keep7, ok7_8, c["strong"])
        teamN = np.where(keep7, ok7_N, c["strong"])
        team8_em = np.where(keep7, em7_8, em[k]["strong_em"])
        teamN_em = np.where(keep7, em7_N, em[k]["strong_em"])
        assert np.array_equal(team8, np.asarray(team, float)), (k, "F10 reconstruction mismatch")
        out[k] = dict(n=n, keep7_rate=float(keep7.mean()), esc_F10=float(1 - keep7.mean()),
                      meanN_incumbent=float(res["incumbent"][k]["N"].mean()),
                      meanN_floored=float(N.mean()),
                      acc_published_bo8_judge=float(team8.mean()),
                      acc_consistent_boN_judge=float(teamN.mean()),
                      acc_published_bo8_em=float(team8_em.mean()),
                      acc_consistent_boN_em=float(teamN_em.mean()),
                      d_judge=float(teamN.mean() - team8.mean()),
                      d_em=float(teamN_em.mean() - team8_em.mean()),
                      flops_as_charged=float(res["incumbent"][k]["N"].mean() * HD.C_CHEAP +
                                             (1 - keep7.mean()) * HD.C_STRONG),
                      flops_if_8_draws_charged=float(FIXED_BO8_FLOPS + (1 - keep7.mean()) * HD.C_STRONG))
    d8_judge = sum(out[k]["d_judge"] for k in R.OPEN_B) / 8.0
    d8_em = sum(out[k]["d_em"] for k in R.OPEN_B) / 8.0
    dflops = sum(out[k]["flops_if_8_draws_charged"] - out[k]["flops_as_charged"] for k in R.OPEN_B) / 8.0
    rep = dict(per_cell=out,
               repair_i_charge_8_draws=dict(delta_macro_flops=dflops,
                                            accuracy_unchanged=True),
               repair_ii_score_best_of_meanN=dict(delta_macro_acc_judge=d8_judge,
                                                  delta_macro_acc_em=d8_em,
                                                  cost_unchanged=True,
                                                  is_an_upper_bound="yes -- F10's rejector was trained "
                                                  "on 8-candidate features and its routing is held fixed"),
               reading="The published pair (macro accuracy 0.6575, macro compute 1.740x direct) is not "
                       "self-consistent on the 3 open cells. Making it consistent costs either "
                       "+%.3f FLOP-eq on the macro (1.740x -> %.3fx) or %+.4f macro accuracy under the "
                       "judge / %+.4f under exact match." % (dflops, (7.951 + dflops) / R.DIRECT_FLOPS,
                                                             d8_judge, d8_em),
               already_on_record=["results/cascade_methods/docs/current/METHOD_FINAL_2026-07.md:355-356",
                                  "results/cascade_methods/docs/current/RESEARCH_RESULTS_2026-07.md:978-979"],
               never_quantified_until_now=True)
    json.dump(rep, open(os.path.join(R.ART, "_hole17_amax_consistency.json"), "w"), indent=2, default=float)
    print(json.dumps(rep, indent=1, default=float))
    return rep


def _f10_keep_mask(dskey, K=BB.K):
    """The F10 routing mask, from method_final_mmmu_corrected.f10_persample, exposed per item."""
    from sklearn.linear_model import LogisticRegression
    d = BB.open_features(dskey)
    ok7, ok32, X = d["ok7"], d["ok32"], d["X"]
    n = len(ok7); ii = np.arange(n); keep = np.zeros(n, bool)
    for f in range(K):
        te = ii % K == f; tr = ~te
        mdl = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], ok7[tr])
        g_tr = mdl.predict_proba(X[tr])[:, 1]; g_te = mdl.predict_proba(X[te])[:, 1]
        grid = np.unique(np.quantile(g_tr, np.linspace(0, 1, 41)))
        best_t, best_a = 1.0, -1.0
        for t in grid:
            a = np.where(g_tr >= t, ok7[tr], ok32[tr]).mean()
            if a > best_a: best_a, best_t = a, t
        keep[te] = g_te >= best_t
    return keep


if __name__ == "__main__":
    main()
