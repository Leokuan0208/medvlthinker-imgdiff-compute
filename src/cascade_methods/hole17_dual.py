#!/usr/bin/env python3
"""hole17_dual.py -- the open-text half of the refit in BOTH currencies, on IDENTICAL picks.

The policies are fit EXACTLY as shipped (judge labels, judge-trained verifier, judge-calibrated
isotonic).  Only the label applied to the delivered slot changes.  Same folds, same lambda per fold,
same escalation set, same picked slot -- two graders.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_dual.py
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
import hole17_engine as EN
import hole17_data as HD
import hole17_currency as CU
from pandora_controller import zeta_strong

ANCHORS = [("macro_iso_acc", "macro", "iso_acc"), ("macro_iso_cost", "macro", "iso_cost"),
           ("pooled_iso_acc", "pooled", "iso_acc"), ("pooled_iso_cost", "pooled", "iso_cost")]


def open_policy_vectors(D, fa, mu_per_fold=None, incumbent=False):
    """Per open cell: held-out (esc, pick, N) under either the incumbent rule or a per-fold mu."""
    out = {}
    for k in R.OPEN_B:
        c = D.open[k]; n = c["n"]
        ESC = np.zeros(n, bool); PICK = np.zeros(n, int); NN = np.zeros(n)
        for f in range(R.K_OUTER):
            te = fa[k] == f; tr = ~te
            sp = R.open_split(c, tr, te)
            if incumbent:
                j = R.open_pick_incumbent(sp, D.open_target[k])
            else:
                j = R.pick_lagrange(sp["acc_tr"], sp["cost_tr"], mu_per_fold[f])
            iso = EN.open_fit_calibrator(c["raw"][tr], c["sl"][tr])
            pool = iso.predict(c["raw"][tr].ravel()); q = float(c["strong"][tr].mean())
            cal_te = iso.predict(c["raw"][te].ravel()).reshape(c["raw"][te].shape)
            z_c = EN.zeta_cheap_exact(pool, EN.LAMS[j]); z_s = zeta_strong(q, EN.LAMS[j])
            N, e, p = CU.pandora_pick(cal_te, c["raw"][te], z_c, z_s)
            ESC[te] = e; PICK[te] = p; NN[te] = N
        out[k] = dict(esc=ESC, pick=PICK, N=NN)
    return out


def score(D, em, pol):
    """Deliver the same answers under both graders."""
    res = {}
    for k in R.OPEN_B:
        c = D.open[k]; p = pol[k]; ii = np.arange(c["n"])
        judge = np.where(p["esc"], c["strong"], c["sl"][ii, p["pick"]])
        exact = np.where(p["esc"], em[k]["strong_em"], em[k]["sl_em"][ii, p["pick"]])
        res[k] = dict(judge=judge, em=exact, esc_rate=float(p["esc"].mean()),
                      meanN=float(p["N"].mean()),
                      acc_judge=float(judge.mean()), acc_em=float(exact.mean()))
    return res


def main():
    D = R.Data()
    em = CU.load_em(D.verifier)
    fa = {k: R.folds(D.n[k], R.K_OUTER) for k in R.ORDER_B}
    main_art = json.load(open(os.path.join(R.ART, "_hole17_main.json")))

    arms = {"incumbent": open_policy_vectors(D, fa, incumbent=True)}
    for lab, _, _ in ANCHORS:
        mus = main_art["arms"]["nested_" + lab]["mu_per_outer_fold"]
        arms["nested_" + lab] = open_policy_vectors(D, fa, mu_per_fold=mus)

    S = {a: score(D, em, arms[a]) for a in arms}
    base = {k: dict(judge_a32=float(D.open[k]["strong"].mean()),
                    em_a32=float(em[k]["strong_em"].mean()),
                    judge_a7=float(D.open[k]["greedy"].mean()),
                    em_a7=float(em[k]["greedy_em"].mean())) for k in R.OPEN_B}

    out = dict(
        what="open-text arm under BOTH graders on IDENTICAL picks; policies fit under the judge, as shipped",
        em_scorer="src/labeling/run_openvqa.py norm()+score() lifted verbatim (normalised EM + short-answer contains)",
        baselines_open=base, arms={})
    for a in S:
        oj = float(np.mean([S[a][k]["acc_judge"] for k in R.OPEN_B]))
        oe = float(np.mean([S[a][k]["acc_em"] for k in R.OPEN_B]))
        out["arms"][a] = dict(open_macro_judge=oj, open_macro_em=oe,
                              per_cell={k: {q: S[a][k][q] for q in
                                            ("acc_judge", "acc_em", "esc_rate", "meanN")}
                                        for k in R.OPEN_B})
    inc = out["arms"]["incumbent"]
    for a in out["arms"]:
        out["arms"][a]["d_open_macro_judge_vs_incumbent"] = out["arms"][a]["open_macro_judge"] - inc["open_macro_judge"]
        out["arms"][a]["d_open_macro_em_vs_incumbent"] = out["arms"][a]["open_macro_em"] - inc["open_macro_em"]
        # translate the open-only delta onto the 8-cell macro (3 of 8 cells; the 5 MCQ cells are EM already)
        out["arms"][a]["d_8cell_macro_judge"] = out["arms"][a]["d_open_macro_judge"] * 3 / 8 if False else \
            out["arms"][a]["d_open_macro_judge_vs_incumbent"] * 3 / 8
        out["arms"][a]["d_8cell_macro_em"] = out["arms"][a]["d_open_macro_em_vs_incumbent"] * 3 / 8
    json.dump(out, open(os.path.join(R.ART, "_hole17_dual_currency.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    o = main()
    print("open-text baselines (3 cells):")
    for k, v in o["baselines_open"].items():
        print("  %-14s 7B judge %.4f / EM %.4f   32B judge %.4f / EM %.4f" %
              (k, v["judge_a7"], v["em_a7"], v["judge_a32"], v["em_a32"]))
    print("\n%-24s %-10s %-10s %-12s %-12s" % ("arm", "openJUDGE", "openEM", "d8macroJUDGE", "d8macroEM"))
    for a, v in o["arms"].items():
        print("%-24s %-10.6f %-10.6f %+-12.5f %+-12.5f" %
              (a, v["open_macro_judge"], v["open_macro_em"], v["d_8cell_macro_judge"], v["d_8cell_macro_em"]))
