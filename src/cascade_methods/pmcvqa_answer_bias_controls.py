#!/usr/bin/env python3
"""
pmcvqa_answer_bias_controls.py -- null tests of the CONTROL used in pmcvqa_answer_bias_audit.py, plus
row-order robustness of its finding.

The audit's central control is: **macro-average the per-item delta over the four GOLD letters
(1/4 each)**, on the claim that a pure answer-letter prior confers exactly zero advantage under that
weighting.  That claim is analytically true (a model that predicts letter L with probability q(L)
independent of the item is correct with probability q(L) given gold==L, so its macro accuracy is
(1/4)sum_L q(L) = 0.25 for ANY q).  This file verifies it EMPIRICALLY before the audit leans on it,
and then checks the finding survives the one numeric knob the standing caveat says matters here.

N1  NULL TEST OF THE CONTROL.  Build two purely prior-driven pseudo-arms on the real PMC items:
    arm P draws its answer letter iid from the 7B's measured letter marginal, arm Q from the 32B's.
    Both have ZERO item-level skill by construction, and they differ ONLY in their letter marginal.
    -> the sample-weighted delta must be positive (~ the measured marginal_only_gap, +0.00627),
       and the LETTER-BALANCED delta must be ~0.  20 seeds.

N2  ROW-ORDER ROBUSTNESS.  f8_veto and confadv_fuse both cross-fit with folds `arange(n) % 5`, so
    row order is load-bearing (standing caveat: row order is worth up to +-0.0041).  Re-run the veto
    under 10 random row permutations and recheck: the overall delta, the letter-balanced delta, the
    gold-A delta, and the frequent-vs-rare split.  A finding that only holds in file order is not a
    finding.

N3  SIGN CHECK ON THE MODELS' OWN POSITION BIAS across 10 permutations is not applicable (greedy
    decoding on stored dumps is order-invariant); what varies is only the cross-fit partition.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_controls.py
"""
import os, sys, json, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as B          # noqa: E402
import beat32b_more as M            # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/pmcvqa_answer_bias_controls_2026-08-11.json")
LETTERS = ["A", "B", "C", "D"]


def r5(x):
    return round(float(x), 5)


def macro_letter(diff, gold):
    return float(np.mean([diff[gold == L].mean() for L in LETTERS]))


def main():
    d = B.mcq("PMC_VQA")
    r7 = B.load_raw("lingshu7b_full", "PMC_VQA")
    r32 = B.load_raw("lingshu32b_full", "PMC_VQA")
    n = len(d["ok7"])
    gold = np.array([str(r["answer"]).strip() for r in r7[:n]])

    def as_letter(s):
        for ch in str(s).strip().upper():
            if ch in "ABCD":
                return ch
        return "other"
    p7L = np.array([as_letter(r["response"]) for r in r7[:n]])
    p32L = np.array([as_letter(r["response"]) for r in r32[:n]])

    out = dict(
        title="Null tests of the letter-balanced control, and row-order robustness of the PMC "
              "answer-letter finding.",
        date="2026-08-11", no_gpu=True, no_fabricated_numbers=True,
        parent="results/cascade_methods/artifacts/pmcvqa_answer_bias_audit_2026-08-11.json",
        reproduce="OMP_NUM_THREADS=1 python3 src/cascade_methods/pmcvqa_answer_bias_controls.py",
        numerics=dict(numpy=np.__version__, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset")),
    )

    # ---------------- N1: null test of the control ----------------
    c7 = collections.Counter(p7L)
    c32 = collections.Counter(p32L)
    q7 = np.array([c7.get(L, 0) / n for L in LETTERS])
    q32 = np.array([c32.get(L, 0) / n for L in LETTERS])
    q7 = q7 / q7.sum()
    q32 = q32 / q32.sum()
    goldi = np.array([LETTERS.index(g) for g in gold])
    sw, lb = [], []
    for seed in range(20):
        rng = np.random.default_rng(1000 + seed)
        pickP = rng.choice(4, size=n, p=q7)
        pickQ = rng.choice(4, size=n, p=q32)
        diff = (pickP == goldi).astype(float) - (pickQ == goldi).astype(float)
        sw.append(diff.mean())
        lb.append(macro_letter(diff, gold))
    out["N1_null_test_of_the_control"] = dict(
        what="Two pseudo-arms with ZERO item-level skill whose ONLY difference is the letter marginal "
             "(arm P ~ the 7B's measured marginal, arm Q ~ the 32B's). 20 seeds.",
        marginal_7B={L: r5(v) for L, v in zip(LETTERS, q7)},
        marginal_32B={L: r5(v) for L, v in zip(LETTERS, q32)},
        analytic_expectation=dict(
            sample_weighted="sum_L (q7(L)-q32(L)) * p_gold(L) = the measured marginal_only_gap",
            letter_balanced="(1/4) sum_L (q7(L)-q32(L)) = 0 exactly, since both marginals sum to 1"),
        sample_weighted_delta=dict(mean=r5(np.mean(sw)), sd=r5(np.std(sw)),
                                   lo=r5(np.min(sw)), hi=r5(np.max(sw))),
        letter_balanced_delta=dict(mean=r5(np.mean(lb)), sd=r5(np.std(lb)),
                                   lo=r5(np.min(lb)), hi=r5(np.max(lb))),
        max_abs_letter_balanced=r5(np.max(np.abs(lb))),
        PASSED=bool(np.mean(sw) > 0 and abs(np.mean(lb)) < 3 * np.std(lb) + 1e-6
                    and abs(np.mean(lb)) < 0.001),
        conclusion="A pure letter-prior difference shows up at full size under sample weighting and "
                   "vanishes under letter balancing => the control does what it claims.",
    )

    # ---------------- N2: row-order robustness ----------------
    rows = []
    base = None
    for seed in range(10):
        if seed == 0:
            perm = np.arange(n)          # file order == the published configuration
        else:
            perm = np.random.default_rng(2000 + seed).permutation(n)
        dp = dict(d)
        for k in ("ok7", "ok32", "c7", "c32"):
            dp[k] = d[k][perm]
        dp["p7"] = [d["p7"][i] for i in perm]
        dp["p32"] = [d["p32"][i] for i in perm]
        if d.get("okT") is not None:
            dp["okT"] = d["okT"][perm]
        if d.get("cT") is not None:
            dp["cT"] = d["cT"][perm]
        okv, veto = M.f8_veto(dp)
        g = gold[perm]
        diff = okv - dp["ok32"]
        r = dict(seed=int(seed), row_order="file order (PUBLISHED)" if seed == 0 else f"permutation {seed}",
                 veto_rate=r5(veto.mean()),
                 overall_delta=r5(diff.mean()),
                 letter_balanced_delta=r5(macro_letter(diff, g)),
                 delta_gold_A=r5(diff[g == "A"].mean()),
                 delta_gold_B=r5(diff[g == "B"].mean()),
                 delta_gold_C=r5(diff[g == "C"].mean()),
                 delta_gold_D=r5(diff[g == "D"].mean()),
                 delta_frequent_BC=r5(diff[(g == "B") | (g == "C")].mean()),
                 delta_rare_AD=r5(diff[(g == "A") | (g == "D")].mean()))
        rows.append(r)
        if seed == 0:
            base = r
    arr = lambda k: np.array([r[k] for r in rows])
    out["N2_row_order_robustness"] = dict(
        what="f8_veto cross-fits with folds arange(n) %% 5, so row order is load-bearing (standing "
             "caveat: row order is worth up to +-0.0041). 10 orderings; seed 0 IS the published file order.",
        per_seed=rows,
        summary={k: dict(published_file_order=base[k], mean=r5(arr(k).mean()), sd=r5(arr(k).std()),
                         lo=r5(arr(k).min()), hi=r5(arr(k).max()),
                         n_seeds_with_same_sign_as_file_order=int(
                             (np.sign(arr(k)) == np.sign(base[k])).sum()))
                 for k in ("overall_delta", "letter_balanced_delta", "delta_gold_A",
                           "delta_frequent_BC", "delta_rare_AD")},
        FINDING_STABLE=dict(
            letter_balanced_delta_positive_in_all_10=bool((arr("letter_balanced_delta") > 0).all()),
            gold_A_delta_negative_in_all_10=bool((arr("delta_gold_A") < 0).all()),
            rare_AD_delta_negative_in_all_10=bool((arr("delta_rare_AD") < 0).all()),
            frequent_BC_delta_positive_in_all_10=bool((arr("delta_frequent_BC") > 0).all()),
        ),
    )

    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT)
    print(json.dumps(out["N1_null_test_of_the_control"], indent=1))
    print(json.dumps(out["N2_row_order_robustness"]["summary"], indent=1))
    print(json.dumps(out["N2_row_order_robustness"]["FINDING_STABLE"], indent=1))


if __name__ == "__main__":
    main()
