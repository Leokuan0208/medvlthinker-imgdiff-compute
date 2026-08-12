#!/usr/bin/env python3
"""
ATTACK A, follow-up E7 -- the NARROWEST honest estimator on the menu.

WHY THIS EXISTS.  The per-cell SE decomposition in armcombine_2026-08-11.json shows that essentially
ALL of the 8-cell macro's variance comes from the three open cells (99.6% for the format-level
policy), and 71% of it from VQA_RAD_open alone -- a 200-item cell that carries 1/8 of the weight.
Any policy that touches the open cells therefore inherits a macro standard error of ~0.0033, which
makes a +0.003-scale effect unresolvable by construction.  E7 asks the opposite question: what does
the honest estimator say if the open half is left AT THE BASELINE and only the multiple-choice half
is allowed to change?  One cross-fit decision, over 6 arms, backed by 39,879 items.

STATUS: POST-HOC.  It was written after seeing the variance decomposition.  It is reported with its
own permutation null and is a candidate for a round-3 PRE-REGISTRATION, not a pre-registered result.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/armcombine_mcqonly.py
Artifact:   results/cascade_methods/artifacts/armcombine_mcqonly_2026-08-11.json
"""
import os
import json
import time

import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SRC = os.path.join(REPO, "src/cascade_methods/armcombine.py")
_body = open(SRC).read().split('if __name__ == "__main__":')[0]
exec(compile(_body, SRC, "exec"))          # noqa: S102 -- reuse the exact loaders and estimators

OUTP = os.path.join(ART, "armcombine_mcqonly_2026-08-11.json")
NPERM7 = 2000


def e7_mcq_only(T, OKm, FD, want_vec=True):
    """ONE arm for the 5 MCQ cells, chosen by cross-fit on the group's own macro.  The 3 open cells
    are FIXED at always-32B-direct -- the baseline -- and are never selected over."""
    vec = {c: (np.empty(N[c]) if want_vec else None) for c in CELLS}
    rows = {c: [] for c in CELLS}
    accs = {c: 0.0 for c in CELLS}
    picks = []
    menu = _group_menu_static(T.menu, "MCQ")
    for f in range(KFOLD):
        trm, tem = T.of != f, T.of == f
        a = {x: float(np.mean([T.acc(c, x, trm) for c in MCQ_CELLS])) for x in menu}
        best = max(a.values())
        cands = [x for x in menu if a[x] >= best - 1e-12]
        a_star = min(cands, key=lambda x: (np.mean([cost_A(ARMC[c][x]) for c in MCQ_CELLS]), x))
        picks.append(a_star)
        for c in CELLS:
            arm = a_star if c in MCQ_CELLS else BASE
            if want_vec:
                sel = FD[c] == f
                vec[c][sel] = OKm[c][arm][sel]
            accs[c] += T.corr(c, arm, tem) / N[c]
            rows[c].append((T.n(c, tem), arm))
    return accs, rows, vec, picks


def fixed_policy(OKm, mcq_arm):
    return {c: OKm[c][mcq_arm if c in MCQ_CELLS else BASE] for c in CELLS}


def main():
    t0 = time.time()
    res = dict(
        title="ATTACK A follow-up E7 -- multiple-choice half only, open half frozen at the baseline",
        date="2026-08-11", status="POST-HOC / EXPLORATORY -- see the header of "
                                  "src/cascade_methods/armcombine_mcqonly.py",
        reproduce="OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/armcombine_mcqonly.py",
        parent="results/cascade_methods/artifacts/armcombine_2026-08-11.json",
        no_gpu=True, no_fabricated_numbers=True, seed=SEED, nboot=NBOOT, n_permutations=NPERM7,
        motivation=("the parent artifact's per-cell SE decomposition: the 3 open cells supply 99.6% "
                    "of the 8-cell macro's standard error and VQA_RAD_open (n=200) alone supplies "
                    "71%.  Freezing the open half at always-32B-direct removes that variance "
                    "entirely -- and also removes all serving-config exposure, because the open "
                    "cells then cancel exactly between policy and baseline."))

    OKm, menu = build_frame("P", GENSEEDS[0])          # frame-independent: open cells are frozen
    bar = {c: OKm[c][BASE] for c in CELLS}
    bar_macro = macro_of_vec(bar)
    res["bar_macro_always_32b_direct"] = round(bar_macro, 6)
    res["frame_independence"] = (
        "this policy never changes an open cell, so its macro delta is IDENTICAL in frame M "
        "(matched serving run) and frame P (published bar): the open-cell terms cancel exactly "
        "between policy and baseline.  The +/-0.008 serving-config caveat does not touch it.")

    # ---- cross-fit, 12 fold seeds ----
    deltas, costs, picks = [], [], Counter()
    avg = {c: np.zeros(N[c]) for c in CELLS}
    for si in range(NFOLDSEEDS):
        FD = folds(SEED + 100 * si)
        FDI = folds(SEED + 100 * si + 7, KINNER)
        T = Tab(OKm, menu, FD, FDI)
        a7, r7, v7, p7 = e7_mcq_only(T, OKm, FD)
        deltas.append(macro_of_acc(a7) - bar_macro)
        costs.append(policy_cost(r7, cost_A, R32_CHARGED)[0])
        for x in p7:
            picks[x] += 1
        for c in CELLS:
            avg[c] += v7[c] / NFOLDSEEDS
        if si == 0:
            rows0 = r7
    res["E7_crossfit"] = dict(
        n_fold_seeds=NFOLDSEEDS, delta_mean=round(float(np.mean(deltas)), 6),
        delta_sd=round(float(np.std(deltas, ddof=1)), 6),
        delta_range=[round(float(np.min(deltas)), 6), round(float(np.max(deltas)), 6)],
        arm_picks_over_60_fold_decisions=dict(picks),
        x_direct_as_charged=round(float(np.mean(costs)) / R32_CHARGED, 4))

    # ---- bootstrap ----
    fixed = {a: fixed_policy(OKm, a) for a in _group_menu_static(menu, "MCQ")}
    sig = {c: [OKm[c][a] for a in menu[c]] + [avg[c]] for c in CELLS}
    Bt = Boot(sig)
    bar_b = Bt.macro(bar)

    def block(vv, label):
        out = ci(Bt.macro(vv) - bar_b, macro_of_vec(vv) - bar_macro)
        out["macro_acc"] = round(macro_of_vec(vv), 6)
        out["label"] = label
        out["per_cell"] = {}
        gr = []
        for c in CELLS:
            cc = ci(Bt.cell(c, vv[c]) - Bt.cell(c, bar[c]), float(vv[c].mean() - bar[c].mean()))
            cc["acc"] = round(float(vv[c].mean()), 5)
            out["per_cell"][c] = cc
            if cc["hi"] < 0:
                gr.append(c)
        out["guardrail_flags"] = gr
        loo = {c: round(float(np.mean([out["per_cell"][j]["delta"] for j in CELLS if j != c])), 6)
               for c in CELLS}
        out["macro_leave_one_out"] = dict(per_dropped_cell=loo,
                                          range=[min(loo.values()), max(loo.values())],
                                          cell_carrying_the_claim=min(loo, key=lambda z: loo[z]))
        return out

    res["E7_crossfit_CI_foldseed_averaged"] = block(avg, "cross-fit choice of the MCQ arm")
    res["fixed_mcq_arm_policies"] = {a: block(v, f"MCQ={a}, OPEN=always_32b_direct (FIXED)")
                                     for a, v in fixed.items()}
    for a in fixed:
        cst = float(np.mean([cost_A(ARMC[c][a if c in MCQ_CELLS else BASE]) for c in CELLS]))
        res["fixed_mcq_arm_policies"][a]["x_direct_as_charged"] = round(cst / R32_CHARGED, 4)
        res["fixed_mcq_arm_policies"][a]["macro_flopeq_as_charged"] = round(cst, 4)
        cstd = float(np.mean([cost_A(ARMC[c][a if c in MCQ_CELLS else BASE], R32_DERIVED)
                              for c in CELLS]))
        res["fixed_mcq_arm_policies"][a]["x_direct_R32_3.816"] = round(cstd / R32_DERIVED, 4)

    res["cost"] = {}
    for cn, fn in CONV.items():
        for R, lab in ((R32_CHARGED, "R32_4.57"), (R32_DERIVED, "R32_3.816")):
            m, per = policy_cost(rows0, fn, R)
            res["cost"][f"{cn}|{lab}"] = dict(macro_flopeq=round(m, 4), x_direct=round(m / R, 4))
    res["cost"]["_note"] = ("conventions B and C are UNCORROBORATED (cost_floor_2026-08-10.json:"
                            "VERDICT.kill_criteria.i).  A is the primary.  This policy runs one 7B "
                            "forward plus one 32B forward on PMC_VQA and a single 32B forward "
                            "everywhere else, so B and C barely move it.")

    # ---- permutation null for THIS estimator ----
    GRPS = {}
    for c in CELLS:
        reps, of = [], {}
        for a in menu[c]:
            hit = next((j for j, r in enumerate(reps) if np.array_equal(OKm[c][a], OKm[c][r])), None)
            if hit is None:
                reps.append(a)
                hit = len(reps) - 1
            of[a] = hit
        GRPS[c] = (reps, of)
    res["effective_menu_size_MCQ"] = {c: dict(n_arms=len(menu[c]), n_distinct=len(GRPS[c][0]))
                                      for c in MCQ_CELLS}

    FD, FDI = folds(SEED), folds(SEED + 7, KINNER)
    T0 = Tab(OKm, menu, FD, FDI)
    obs_cf = macro_of_acc(e7_mcq_only(T0, OKm, FD, want_vec=False)[0]) - bar_macro
    obs_ev = max(macro_of_vec(v) for v in fixed.values()) - bar_macro
    rng = np.random.default_rng(SEED + 777)
    ncf, nev = [], []
    for p in range(NPERM7):
        OKp = {}
        for c in CELLS:
            reps, of = GRPS[c]
            Mx = np.column_stack([OKm[c][r] for r in reps])
            order = np.argsort(rng.random(Mx.shape), axis=1)
            Mp = np.take_along_axis(Mx, order, axis=1)
            OKp[c] = {a: np.ascontiguousarray(Mp[:, of[a]]) for a in menu[c]}
        barp = float(np.mean([OKp[c][BASE].mean() for c in CELLS]))
        Tp = Tab(OKp, menu, FD, FDI)
        ncf.append(macro_of_acc(e7_mcq_only(Tp, OKp, FD, want_vec=False)[0]) - barp)
        nev.append(max(macro_of_vec(fixed_policy(OKp, a))
                       for a in _group_menu_static(menu, "MCQ")) - barp)
    res["permutation_null_dedup"] = {}
    for k, arr, obs in (("E7_crossfit", ncf, obs_cf), ("best_fixed_MCQ_arm_evalvisible", nev, obs_ev)):
        a = np.asarray(arr)
        res["permutation_null_dedup"][k] = dict(
            n_perm=NPERM7, null_mean=round(float(a.mean()), 6), null_sd=round(float(a.std(ddof=1)), 6),
            null_p2p5=round(float(np.percentile(a, 2.5)), 6),
            null_p97p5=round(float(np.percentile(a, 97.5)), 6),
            null_max=round(float(a.max()), 6), observed=round(float(obs), 6),
            p_one_sided=round(float((1 + (a >= obs).sum()) / (1 + len(a))), 5))

    # ---- cross-check against the ALREADY-PUBLISHED MCQ-half number ----
    pub_mcq = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json"))) \
        ["per_arm"]["disjoint"]["deltas_mcq"]["method_accuracy_max_fusion"]["always_32b_direct"]
    fus = res["fixed_mcq_arm_policies"]["method_accuracy_max_fusion"]
    res["cross_check_vs_published"] = dict(
        published_mcq_half_delta=pub_mcq,
        published_is="cascade_selector_rerun_2026-08-05.json:per_arm.disjoint.deltas_mcq."
                     "method_accuracy_max_fusion.always_32b_direct -- a 5-cell MCQ-macro delta that "
                     "the project ALREADY reports as a significant WIN",
        implied_8cell_delta_by_scaling=round(pub_mcq["delta"] * 5 / 8, 6),
        implied_8cell_ci_by_scaling=[round(pub_mcq["lo"] * 5 / 8, 6), round(pub_mcq["hi"] * 5 / 8, 6)],
        measured_here_8cell_delta=fus["delta"], measured_here_8cell_ci=[fus["lo"], fus["hi"]],
        max_abs_dev=round(abs(pub_mcq["delta"] * 5 / 8 - fus["delta"]), 6),
        reading="the 8-cell macro number is the published MCQ-half number times 5/8, as it must be, "
                "because this policy changes nothing outside the MCQ half.  E7 is therefore NOT a "
                "new measurement -- it is an already-published significant result re-expressed on "
                "the 8-cell macro, which is the axis the project's target is stated on.")

    json.dump(res, open(OUTP, "w"), indent=1, default=str)
    print("wrote", OUTP, round(time.time() - t0, 1), "s")
    print(json.dumps(res["E7_crossfit"], indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("delta", "lo", "hi", "verdict", "macro_acc",
                                               "guardrail_flags", "x_direct_as_charged")}
                      for k, v in res["fixed_mcq_arm_policies"].items()}, indent=1))
    print(json.dumps(res["E7_crossfit_CI_foldseed_averaged"]["per_cell"], indent=1))
    print(json.dumps(res["permutation_null_dedup"], indent=1))
    print(json.dumps(res["cross_check_vs_published"], indent=1))


if __name__ == "__main__":
    main()
