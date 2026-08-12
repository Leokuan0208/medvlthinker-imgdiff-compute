#!/usr/bin/env python3
"""
pmcvqa_answer_bias_reweight.py -- adds E5 to pmcvqa_answer_bias_extend_2026-08-11.json (additive only).

E5 asks the most decision-relevant version of the question. The letter-BALANCED delta (uniform gold
marginal) is the clean statistical control, but no real split is uniform. The operationally meaningful
counterfactual is: **what would the PMC gain be if test_2.csv carried the answer-position profile of
the HUMAN-VERIFIED v1 split (test_clean.csv) instead of the training split's profile it actually
carries?**  That is a post-stratification reweight of the same 33,430 items onto test_clean's measured
gold-letter marginal (A .2185 / B .3190 / C .3050 / D .1575), with a stratified paired bootstrap.

Reported alongside, for completeness: reweighting onto the v1 FULL test split (test.csv) marginal.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_reweight.py
"""
import os, sys, csv, json, copy, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as B          # noqa: E402
import beat32b_more as M            # noqa: E402

EXT = os.path.join(ROOT, "results/cascade_methods/artifacts/pmcvqa_answer_bias_extend_2026-08-11.json")
PMC_DIR = "/data/dan/dataset/medevalkit/PMC-VQA"
LETTERS = ["A", "B", "C", "D"]
NBOOT = 10000
SEED = 20260811


def r5(x):
    return round(float(x), 5)


def marginal_from_csv(fname, col):
    rd = csv.reader(open(os.path.join(PMC_DIR, fname), encoding="utf-8"))
    next(rd)
    c = collections.Counter()
    for row in rd:
        g = row[col].strip()
        if g in LETTERS:
            c[g] += 1
    tot = sum(c.values())
    return {L: c[L] / tot for L in LETTERS}, tot


def reweighted_boot(diff, gold, w, nboot=NBOOT, seed=SEED, scale=1.0):
    """Post-stratified mean onto target marginal w, with a stratified paired bootstrap."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(gold == L) for L in LETTERS]
    point = float(sum(w[L] * diff[g].mean() for L, g in zip(LETTERS, groups))) * scale
    boots = np.empty(nboot)
    for i in range(nboot):
        boots[i] = sum(w[L] * diff[g[rng.integers(0, len(g), len(g))]].mean()
                       for L, g in zip(LETTERS, groups)) * scale
    return dict(delta=r5(point), ci=[r5(np.percentile(boots, 2.5)), r5(np.percentile(boots, 97.5))],
                sig=bool(np.percentile(boots, 2.5) > 0 or np.percentile(boots, 97.5) < 0))


def main():
    art = json.load(open(EXT))
    before = copy.deepcopy(art)

    d = B.mcq("PMC_VQA")
    r7 = B.load_raw("lingshu7b_full", "PMC_VQA")
    n = len(d["ok7"])
    gold = np.array([str(r["answer"]).strip() for r in r7[:n]])
    ok32 = d["ok32"]
    ok_veto, _ = M.f8_veto(d)
    ok_fuse = B.confadv_fuse(d)

    w_clean, n_clean = marginal_from_csv("test_clean.csv", 7)
    w_v1, n_v1 = marginal_from_csv("test.csv", 7)
    w_self, n_self = marginal_from_csv("test_2.csv", 8)
    w_unif = {L: 0.25 for L in LETTERS}

    targets = {
        "test_2.csv own marginal (= the published number; identity check)": (w_self, n_self),
        "test_clean.csv v1 HUMAN-VERIFIED marginal": (w_clean, n_clean),
        "test.csv v1 full-test marginal": (w_v1, n_v1),
        "uniform (the letter-balanced control)": (w_unif, None),
    }

    e5 = dict(
        what="Post-stratify the SAME 33,430 items onto a different gold-letter marginal and re-read "
             "the PMC cell delta, and the MCQ-only macro delta (= cell delta / 8, since the policy is "
             "byte-identical to always-32B-direct on the other 7 cells).",
        target_marginals={k: {L: round(v[0][L], 4) for L in LETTERS} for k, v in targets.items()},
        source_row_counts={k: v[1] for k, v in targets.items()},
        arms={},
    )
    for arm_name, ok_arm in [("certified veto (F8)", ok_veto), ("F3 confidence-advantage fusion", ok_fuse)]:
        diff = ok_arm - ok32
        rows = {}
        for label, (w, _) in targets.items():
            rows[label] = dict(
                pmc_cell_delta=reweighted_boot(diff, gold, w),
                mcq_only_macro_delta=reweighted_boot(diff, gold, w, scale=1.0 / 8),
            )
        e5["arms"][arm_name] = rows
    e5["reading"] = (
        "The first row must reproduce the published cell delta (identity check on the post-stratifier: "
        "reweighting onto a split's own marginal is the identity). The second row is the operationally "
        "meaningful counterfactual: what the gain would be on a split carrying the HUMAN-VERIFIED v1 "
        "answer-position profile rather than the training-split profile test_2.csv actually carries.")

    art["E5_post_stratified_onto_other_split_marginals"] = e5
    for k in before:
        assert json.dumps(before[k], sort_keys=True) == json.dumps(art[k], sort_keys=True), k
    json.dump(art, open(EXT, "w"), indent=1)
    print("UPDATED (additive only)", EXT)
    print(json.dumps(e5["target_marginals"], indent=1))
    for a, rows in e5["arms"].items():
        print("==", a)
        for lab, r in rows.items():
            print(f"   {lab:<58} cell {r['pmc_cell_delta']['delta']:+.5f} "
                  f"{r['pmc_cell_delta']['ci']}  macro {r['mcq_only_macro_delta']['delta']:+.5f} "
                  f"{r['mcq_only_macro_delta']['ci']}")


if __name__ == "__main__":
    main()
