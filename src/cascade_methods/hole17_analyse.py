#!/usr/bin/env python3
"""hole17_analyse.py -- HOLE 17 headline analysis: is the shipped threshold vector on a common
exchange-rate frontier at all, what does a macro-objective refit buy, and does the permutation null
already explain it?

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_analyse.py
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

ORDER_B, MCQ_B, OPEN_B = R.ORDER_B, R.MCQ_B, R.OPEN_B
ANCHORS = [("macro_iso_acc", "macro", "iso_acc"), ("macro_iso_cost", "macro", "iso_cost"),
           ("pooled_iso_acc", "pooled", "iso_acc"), ("pooled_iso_cost", "pooled", "iso_cost")]


# ---------------------------------------------------------------------------------------------
# 1.  Is the SHIPPED per-cell threshold vector a common-mu (equal marginal exchange rate) point?
# ---------------------------------------------------------------------------------------------
def implied_mu(D, fa, mus=R.MUS):
    """For every cell and every outer fold, the interval of mu on which the INCUMBENT's chosen
    threshold is the Lagrangian argmax on that fold's TRAIN split.  If the 8 intervals do not
    intersect, the deployed method runs its cells at DIFFERENT marginal accuracy-per-FLOP rates,
    which is a misallocation independent of any reporting convention."""
    out = {}
    for k in ORDER_B:
        c = D.mcq[k] if k in MCQ_B else D.open[k]
        lo_all, hi_all, exact = [], [], []
        for f in range(R.K_OUTER):
            te = fa[k] == f; tr = ~te
            if k in MCQ_B:
                sp = R.mcq_split(c, tr, te); j0 = R.mcq_pick_incumbent(sp)
            else:
                sp = R.open_split(c, tr, te); j0 = R.open_pick_incumbent(sp, D.open_target[k])
            js = np.array([R.pick_lagrange(sp["acc_tr"], sp["cost_tr"], m) for m in mus])
            hit = np.where(js == j0)[0]
            if len(hit):
                lo_all.append(float(mus[hit[0]])); hi_all.append(float(mus[hit[-1]])); exact.append(True)
            else:
                # not Lagrangian-optimal at any exchange rate -> report the nearest-cost mu
                ct = sp["cost_tr"][js]; j = int(np.argmin(np.abs(ct - sp["cost_tr"][j0])))
                lo_all.append(float(mus[j])); hi_all.append(float(mus[j])); exact.append(False)
        out[k] = dict(mu_lo=float(np.median(lo_all)), mu_hi=float(np.median(hi_all)),
                      per_fold_lo=lo_all, per_fold_hi=hi_all,
                      lagrangian_optimal_on_all_folds=bool(all(exact)))
    lo = max(v["mu_lo"] for v in out.values()); hi = min(v["mu_hi"] for v in out.values())
    out["_intersection"] = dict(lo=lo, hi=hi, nonempty=bool(lo <= hi),
                                spread_ratio=float(max(v["mu_hi"] for v in out.values() if isinstance(v, dict) and "mu_hi" in v) /
                                                   max(1e-12, min(v["mu_lo"] for v in out.values() if isinstance(v, dict) and "mu_lo" in v))))
    return out


# ---------------------------------------------------------------------------------------------
# 2.  the common-mu FRONTIER (eval-visible; the picture, not the headline)
# ---------------------------------------------------------------------------------------------
def frontier(res, n, mus=R.MUS):
    wM = {k: 1.0 / 8 for k in ORDER_B}
    N = sum(n[k] for k in ORDER_B); wP = {k: n[k] / N for k in ORDER_B}
    A_m = sum(res["mu"][k]["acc"] * wM[k] for k in ORDER_B)
    C_m = sum(res["mu"][k]["cost"] * wM[k] for k in ORDER_B)
    A_p = sum(res["mu"][k]["acc"] * wP[k] for k in ORDER_B)
    C_p = sum(res["mu"][k]["cost"] * wP[k] for k in ORDER_B)
    return dict(mu=mus.tolist(), macro_acc=A_m.tolist(), macro_cost=C_m.tolist(),
                pooled_acc=A_p.tolist(), pooled_cost=C_p.tolist())


# ---------------------------------------------------------------------------------------------
# 3.  guardrail + baselines
# ---------------------------------------------------------------------------------------------
def baselines(D):
    b = {}
    for k in MCQ_B:
        b[k] = dict(a7=D.mcq[k]["ok7"], a32=D.mcq[k]["ok32"])
    for k in OPEN_B:
        b[k] = dict(a7=D.open[k]["greedy"], a32=D.open[k]["strong"])
    return b


def guardrail(arm_vec, base, D):
    out = {}
    for k in ORDER_B:
        a = float(arm_vec[k].mean())
        out[k] = dict(n=int(D.n[k]), acc=a,
                      vs_always_7b=a - float(base[k]["a7"].mean()),
                      vs_always_32b_direct=a - float(base[k]["a32"].mean()))
        out[k]["guardrail_ok_vs_7b"] = bool(out[k]["vs_always_7b"] >= 0)
    out["_n_cells_below_always_7b"] = sum(1 for k in ORDER_B if not out[k]["guardrail_ok_vs_7b"])
    return out


def main():
    t0 = time.time()
    D = R.Data()
    fa = {k: R.folds(D.n[k], R.K_OUTER) for k in ORDER_B}
    base = baselines(D)

    res = R.heldout_pass(D, fa, want_items=True)
    inc_vec = {k: res["incumbent"][k]["ok"] for k in ORDER_B}
    inc = R.summarise(res["incumbent"], D.n)

    imu = implied_mu(D, fa)
    fr = frontier(res, D.n)

    nes = R.nested_pass(D, fa, ANCHORS)
    dia, _ = R.diagnostic_pass(D, fa, ANCHORS, res=res)

    arms = {}
    for lab, _, _ in ANCHORS:
        arms["nested_" + lab] = dict(vec={k: nes[lab][k]["ok"] for k in ORDER_B},
                                     s=R.summarise(nes[lab], D.n),
                                     mu=nes[lab]["_mu_per_outer_fold"])
        arms["diag_" + lab] = dict(vec={k: dia[lab][k]["ok"] for k in ORDER_B},
                                   s=R.summarise(dia[lab], D.n), mu=[dia[lab]["_mu"]])

    a32 = {k: base[k]["a32"] for k in ORDER_B}
    a7 = {k: base[k]["a7"] for k in ORDER_B}

    out = dict(
        incumbent=dict(summary=inc, per_cell_esc=inc["per_cell_esc"],
                       per_fold_thresholds={k: res["incumbent"][k]["pol"] for k in ORDER_B}),
        baselines=dict(always_7b_macro=float(np.mean([a7[k].mean() for k in ORDER_B])),
                       always_32b_direct_macro=float(np.mean([a32[k].mean() for k in ORDER_B])),
                       always_32b_direct_macro_cost=R.DIRECT_FLOPS,
                       per_cell_a7={k: float(a7[k].mean()) for k in ORDER_B},
                       per_cell_a32={k: float(a32[k].mean()) for k in ORDER_B}),
        implied_mu=imu, frontier=fr, arms={}, guardrail={},
        deltas_vs_incumbent={}, deltas_vs_32b_direct={})

    for lab, A in arms.items():
        out["arms"][lab] = dict(summary={q: A["s"][q] for q in
                                         ("macro_acc", "macro_cost", "macro_x_direct",
                                          "pooled_acc", "pooled_cost", "pooled_x_direct")},
                                per_cell_acc=A["s"]["per_cell_acc"],
                                per_cell_esc=A["s"]["per_cell_esc"],
                                per_cell_cost=A["s"]["per_cell_cost"],
                                mu_per_outer_fold=A["mu"])
        out["deltas_vs_incumbent"][lab] = R.boot_macro_delta(A["vec"], inc_vec, D.n)
        out["deltas_vs_incumbent"][lab]["d_macro_cost"] = A["s"]["macro_cost"] - inc["macro_cost"]
        out["deltas_vs_32b_direct"][lab] = R.boot_macro_delta(A["vec"], a32, D.n)
        out["guardrail"][lab] = guardrail(A["vec"], base, D)
    out["deltas_vs_incumbent"]["_incumbent_vs_32b_direct"] = R.boot_macro_delta(inc_vec, a32, D.n)
    out["guardrail"]["incumbent"] = guardrail(inc_vec, base, D)
    out["_elapsed_s"] = time.time() - t0
    json.dump(out, open(os.path.join(R.ART, "_hole17_main.json"), "w"), indent=1, default=float)
    print("wrote _hole17_main.json  (%.1fs)" % out["_elapsed_s"])
    return out


if __name__ == "__main__":
    o = main()
    print("\nINCUMBENT   macro %.6f  cost %.4f (%.3fx)  pooled %.6f cost %.4f" %
          (o["incumbent"]["summary"]["macro_acc"], o["incumbent"]["summary"]["macro_cost"],
           o["incumbent"]["summary"]["macro_x_direct"], o["incumbent"]["summary"]["pooled_acc"],
           o["incumbent"]["summary"]["pooled_cost"]))
    for lab in o["arms"]:
        s = o["arms"][lab]["summary"]; d = o["deltas_vs_incumbent"][lab]
        print("%-26s macro %.6f (%+.4f [%+.4f,%+.4f] %s)  cost %.4f (%.3fx, %+.4f)" %
              (lab, s["macro_acc"], d["delta"], d["ci95"][0], d["ci95"][1], d["verdict"],
               s["macro_cost"], s["macro_x_direct"], d["d_macro_cost"]))
    print("\nIMPLIED mu per cell (the shipped operating point's marginal exchange rate):")
    for k in ORDER_B:
        v = o["implied_mu"][k]
        print("  %-16s mu in [%.5f, %.5f]  lagrangian_optimal=%s" %
              (k, v["mu_lo"], v["mu_hi"], v["lagrangian_optimal_on_all_folds"]))
    print("  intersection:", o["implied_mu"]["_intersection"])
