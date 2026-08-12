#!/usr/bin/env python3
"""
A SECOND permutation null for Attack A that fixes a real flaw in the first one.

THE FLAW.  armcombine.py's null permutes the arm labels independently at every item.  That preserves
the per-item multiset of outcomes -- but it does NOT preserve the fact that, on 4 of the 5 MCQ cells,
several arms are the SAME VECTOR (the certified veto never fires there, so
method_accuracy_max_veto == method_accuracy_max_fusion == always_32b_direct item-for-item).  After a
per-item shuffle those duplicate columns come apart, so the permuted menu offers MORE independent
choices than the real one and the null OVERSTATES how much selection bias is available.  An
overstated null is biased towards killing the attack, which is the wrong direction to be sloppy in.

THE FIX.  Deduplicate the columns of each cell by exact vector equality first, permute the labels of
the DISTINCT columns per item, then map every arm back to its representative's permuted value.
Duplicate arms stay duplicates, so the permuted menu has exactly the real menu's effective size.

Both nulls are reported.  The dedup null is the primary; the naive one is the conservative bound.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/armcombine_dedupnull.py
Artifact:   results/cascade_methods/artifacts/_armcombine_dedup_null_2026-08-11.json
"""
import os
import json
import time

import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SRC = os.path.join(REPO, "src/cascade_methods/armcombine.py")
_body = open(SRC).read().split('if __name__ == "__main__":')[0]
exec(compile(_body, SRC, "exec"))          # noqa: S102 -- reuse the exact loaders and estimators

OUTP = os.path.join(ART, "_armcombine_dedup_null_2026-08-11.json")
NPERM_D = 1000


def column_groups(OKm, menu, c):
    """-> (list of representative arm names, arm -> representative index)."""
    reps, of = [], {}
    for a in menu[c]:
        hit = None
        for j, r in enumerate(reps):
            if np.array_equal(OKm[c][a], OKm[c][r]):
                hit = j
                break
        if hit is None:
            reps.append(a)
            hit = len(reps) - 1
        of[a] = hit
    return reps, of


def permute_dedup(OKm, menu, GRPS, rng):
    """Permute the DISTINCT columns per item; duplicates follow their representative."""
    out = {}
    for c in CELLS:
        reps, of = GRPS[c]
        M = np.column_stack([OKm[c][r] for r in reps])
        order = np.argsort(rng.random(M.shape), axis=1)
        Mp = np.take_along_axis(M, order, axis=1)
        out[c] = {a: np.ascontiguousarray(Mp[:, of[a]]) for a in menu[c]}
    return out


def main():
    t0 = time.time()
    res = dict(
        title="ATTACK A -- the DEDUPLICATED permutation null (primary) beside the naive one",
        date="2026-08-11",
        reproduce="OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 "
                  "src/cascade_methods/armcombine_dedupnull.py",
        parent_artifact="results/cascade_methods/artifacts/armcombine_2026-08-11.json",
        why=("the naive per-item label permutation breaks the real menu's duplicate-column structure "
             "(on 4 of 5 MCQ cells the certified veto never fires, so veto == fusion == "
             "always-32B-direct item-for-item).  That gives the permuted menu more independent "
             "choices than the real one and OVERSTATES the selection bias.  This null dedupes first."),
        n_perm=NPERM_D, seed=SEED + 999, frame="M (matched)", gen_seed=GENSEEDS[0],
        fold_seed="the primary fold seed only")

    OKm, menu = build_frame("M", GENSEEDS[0])
    bar_macro = macro_of_vec({c: OKm[c][BASE] for c in CELLS})
    GRPS = {c: column_groups(OKm, menu, c) for c in CELLS}
    res["effective_menu_size"] = {
        c: dict(n_arms_on_menu=len(menu[c]), n_DISTINCT_columns=len(GRPS[c][0]),
                duplicate_groups={r: [a for a in menu[c] if GRPS[c][1][a] == j]
                                  for j, r in enumerate(GRPS[c][0])
                                  if sum(1 for a in menu[c] if GRPS[c][1][a] == j) > 1})
        for c in CELLS}
    res["effective_menu_note"] = (
        "n_DISTINCT_columns is what the dedup null permutes.  Where it is smaller than n_arms_on_menu "
        "the naive null was inflating the available selection bias on that cell.")

    FD, FDI = folds(SEED), folds(SEED + 7, KINNER)
    T0 = Tab(OKm, menu, FD, FDI)
    v0, _ = e0_evalvisible(OKm, menu)

    def best_fixed(OKx):
        b = -9.9
        for ma in _group_menu_static(menu, "MCQ"):
            for oa in _group_menu_static(menu, "OPEN"):
                b = max(b, float(np.mean([OKx[c][ma].mean() for c in MCQ_CELLS] +
                                         [OKx[c][oa].mean() for c in OPEN])))
        return b

    obs = {
        "E0_naive_evalvisible": macro_of_vec(v0) - bar_macro,
        "E1_crossfit_argmax": macro_of_acc(e1_crossfit_argmax(T0, OKm, FD, want_vec=False)[0]) - bar_macro,
        "E2_nested_margin": macro_of_acc(e2_nested_margin(T0, OKm, FD, want_vec=False)[0]) - bar_macro,
        "E5_format_crossfit_argmax_POSTHOC":
            macro_of_acc(e5_format_crossfit(T0, OKm, FD, None, want_vec=False)[0]) - bar_macro,
        "E6_format_nested_margin_POSTHOC":
            macro_of_acc(e5_format_crossfit(T0, OKm, FD, "nested", want_vec=False)[0]) - bar_macro,
        "BEST_FIXED_format_policy_evalvisible": best_fixed(OKm) - bar_macro,
    }

    rng = np.random.default_rng(SEED + 999)
    acc = {k: [] for k in obs}
    for p in range(NPERM_D):
        OKp = permute_dedup(OKm, menu, GRPS, rng)
        barp = float(np.mean([OKp[c][BASE].mean() for c in CELLS]))
        Tp = Tab(OKp, menu, FD, FDI)
        acc["E0_naive_evalvisible"].append(macro_of_vec(e0_evalvisible(OKp, menu)[0]) - barp)
        acc["E1_crossfit_argmax"].append(
            macro_of_acc(e1_crossfit_argmax(Tp, OKp, FD, want_vec=False)[0]) - barp)
        acc["E2_nested_margin"].append(
            macro_of_acc(e2_nested_margin(Tp, OKp, FD, want_vec=False)[0]) - barp)
        acc["E5_format_crossfit_argmax_POSTHOC"].append(
            macro_of_acc(e5_format_crossfit(Tp, OKp, FD, None, want_vec=False)[0]) - barp)
        acc["E6_format_nested_margin_POSTHOC"].append(
            macro_of_acc(e5_format_crossfit(Tp, OKp, FD, "nested", want_vec=False)[0]) - barp)
        acc["BEST_FIXED_format_policy_evalvisible"].append(best_fixed(OKp) - barp)
        if p == 9:
            print(f"  dedup-null eta ~{(time.time()-t0)/10*NPERM_D/60:.1f} min", flush=True)

    res["dedup_permutation_null"] = {}
    for k, arr in acc.items():
        a = np.asarray(arr)
        res["dedup_permutation_null"][k] = dict(
            null_mean=round(float(a.mean()), 6), null_sd=round(float(a.std(ddof=1)), 6),
            null_p2p5=round(float(np.percentile(a, 2.5)), 6),
            null_p50=round(float(np.percentile(a, 50)), 6),
            null_p97p5=round(float(np.percentile(a, 97.5)), 6),
            null_max=round(float(a.max()), 6),
            observed_primary_fold_seed=round(float(obs[k]), 6),
            p_one_sided=round(float((1 + (a >= obs[k]).sum()) / (1 + len(a))), 5))
    json.dump(res, open(OUTP, "w"), indent=1, default=str)
    print("wrote", OUTP, round(time.time() - t0, 1), "s")
    print(json.dumps(res["dedup_permutation_null"], indent=1))


if __name__ == "__main__":
    main()
