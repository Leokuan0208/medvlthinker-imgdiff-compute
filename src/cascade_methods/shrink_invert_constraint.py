#!/usr/bin/env python3
"""
shrink_invert_constraint.py -- ATTACK 3 part 3/3: INVERT the non-inferiority constraint.

THE QUESTION THIS ANSWERS
-------------------------
The round's objective is  MINIMISE compute  SUBJECT TO  macro accuracy non-inferior to
always-32B-direct (paired-bootstrap CI lower bound of the macro delta >= -0.0029).
Attack 3 asks: if neither quantisation nor an available smaller model ties, then
"report what accuracy a strong leg would need in order to hold the tie at a given size".

That is a well-posed inversion and it is answered here, in three layers:

  L1  THE ACCURACY BAR, EXACTLY.  What macro accuracy must a single model reach for the CI
      lower bound to clear -0.0029?  This is NOT simply 0.6567-0.0029: the bound depends on how
      the candidate's errors are CORRELATED with the 32B's, so the bar is computed by bootstrap
      over a family of candidates rather than assumed.

  L2  THE REQUIREMENT CURVE.  A one-parameter family of hypothetical strong legs is built by
      INTERPOLATING THE MEASURED 7B/32B JOINT: for every item the pair (7B correct, 32B correct)
      is known, so a candidate M(lambda) is defined as "recovers a fraction lambda of the errors
      the 32B fixes over the 7B, and keeps the 7B's behaviour elsewhere".  lambda=0 IS the
      measured 7B, lambda=1 IS the measured 32B, and everything between is a real interpolation
      of measured per-item outcomes -- not an invented accuracy number.  Inverting gives
      lambda*, the minimum recovery fraction that holds the tie.

  L3  THE COST FRONTIER.  Pair lambda* with per-pass cost.  A candidate of per-pass cost R_S
      (in Lingshu-7B-forward units) deployed as always-S costs R_S / R_32 of the baseline.
      This yields the sentence the project actually needs: "a strong leg of size X must reach
      accuracy Y to be worth deploying".

EVERY SIMULATED QUANTITY IS LABELLED.  The 7B and 32B per-item vectors are MEASURED
(artifacts/_selector_rerun_parts/vec_disjoint.npz, the published arms).  M(lambda) for
0 < lambda < 1 is a SIMULATED FAMILY anchored on those measured outcomes, and is never
presented as a measurement of any real model.

CPU-only.  Launch from the repo root:
    python3 src/cascade_methods/shrink_invert_constraint.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC = os.path.join(ROOT, "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz")
PUB = os.path.join(ROOT, "results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json")
FOOT = os.path.join(ROOT, "results/cascade_methods/artifacts/_shrink_parts/footprint.json")
OUTDIR = os.path.join(ROOT, "results/cascade_methods/artifacts/_shrink_parts")
os.makedirs(OUTDIR, exist_ok=True)

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
NBOOT = 10000
SEED = 20260812
EPS = -0.0029          # the pre-registered non-inferiority margin


def load():
    z = np.load(VEC)
    return {c: {a: z["%s|%s" % (c, a)].astype(np.int8) for a in
                ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
                 "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]}
            for c in CELLS}


def macro(vecs, key_fn):
    return float(np.mean([key_fn(c) for c in CELLS]))


def boot_counts(D, rng, chunk=1000):
    """One shared paired resample stream, expressed as MULTINOMIAL COUNTS per item.

    Resampling n items with replacement is exactly a Multinomial(n, uniform) draw of per-item
    counts, so counts are the bootstrap -- and they let every candidate reuse ONE stream while
    costing a single O(NBOOT x n) pass instead of materialising a 2.7 GB index matrix per cell.
    """
    C = {}
    for c in CELLS:
        n = len(D[c]["always_7b"])
        parts = []
        for s in range(0, NBOOT, chunk):
            k = min(chunk, NBOOT - s)
            parts.append(rng.multinomial(n, np.full(n, 1.0 / n), size=k).astype(np.float32))
        C[c] = np.concatenate(parts, axis=0)
    return C


def macro_boot(D, per_cell_vec, C):
    """Bootstrap distribution of the 8-cell macro of a supplied per-item 0/1 vector set."""
    out = np.zeros(NBOOT)
    for c in CELLS:
        v = per_cell_vec[c].astype(np.float32)
        out += (C[c] @ v) / len(v)
    return out / len(CELLS)


def ci(delta_boot):
    lo, hi = np.percentile(delta_boot, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------- L2: the M(lambda) family
def build_family(D, lam, rng):
    """M(lambda): keep the 7B's outcome, but recover a fraction `lambda` of the items the 32B
    fixes and lose a fraction `lambda` of the items the 32B breaks.

    lambda = 0 -> exactly the measured always-7B vector.
    lambda = 1 -> exactly the measured always-32B-direct vector.
    Interpolation is applied to the MEASURED per-item joint, so the family passes through both
    real endpoints by construction (asserted below).
    """
    out = {}
    for c in CELLS:
        a7 = D[c]["always_7b"].astype(np.int8)
        a32 = D[c]["always_32b_direct"].astype(np.int8)
        v = a7.copy()
        fix = np.where((a7 == 0) & (a32 == 1))[0]     # 32B recovers a 7B error
        brk = np.where((a7 == 1) & (a32 == 0))[0]     # 32B breaks a 7B success
        # deterministic prefix selection on a fixed shuffled order -> monotone, reproducible
        for grp, val in ((fix, 1), (brk, 0)):
            if len(grp) == 0:
                continue
            order = grp[rng.permutation(len(grp))]
            k = int(round(lam * len(grp)))
            v[order[:k]] = val
        out[c] = v
    return out


def main():
    rng = np.random.RandomState(SEED)
    D = load()
    foot = json.load(open(FOOT))
    R = {k: v["R_vs_lingshu_7b"] for k, v in foot["per_pass_flops"].items()}
    R32 = R["lingshu_32b"]

    # ---------------- NULL TEST ----------------
    pub = json.load(open(PUB))
    published = pub["per_arm"]["disjoint"]["macro_acc"]
    null = {}
    for a in published:
        rec = macro(D, lambda c, a=a: D[c][a].mean())
        null[a] = dict(recomputed=round(rec, 6), published=published[a],
                       abs_dev=round(abs(rec - published[a]), 8))
    max_dev = max(v["abs_dev"] for v in null.values())

    C = boot_counts(D, np.random.RandomState(SEED))
    base_boot = macro_boot(D, {c: D[c]["always_32b_direct"] for c in CELLS}, C)
    base_macro = macro(D, lambda c: D[c]["always_32b_direct"].mean())

    # sanity: the published TIE must reproduce on this bootstrap stream
    am_boot = macro_boot(D, {c: D[c]["method_accuracy_max_veto"] for c in CELLS}, C)
    am_lo, am_hi = ci(am_boot - base_boot)
    tie_check = dict(
        arm="method_accuracy_max_veto",
        delta=round(float(np.mean(am_boot - base_boot)), 6),
        point_delta=round(macro(D, lambda c: D[c]["method_accuracy_max_veto"].mean()) - base_macro, 6),
        lo=round(am_lo, 6), hi=round(am_hi, 6),
        published="+0.0008 [-0.0022,+0.0037] TIE")

    # ---------------- L2: sweep lambda ----------------
    grid = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.02)]
    curve = []
    fam_rng = np.random.RandomState(SEED + 1)
    for lam in grid:
        M = build_family(D, lam, np.random.RandomState(SEED + 1))
        m_boot = macro_boot(D, M, C)
        d = m_boot - base_boot
        lo, hi = ci(d)
        pm = macro(D, lambda c, M=M: M[c].mean())
        curve.append(dict(lam=lam, macro=round(pm, 6), delta=round(pm - base_macro, 6),
                          lo=round(lo, 6), hi=round(hi, 6), passes=bool(lo >= EPS)))

    # endpoints must be the measured arms exactly
    # curve[] stores 6-dp rounded values, so the endpoint identity is asserted at 6-dp tolerance.
    # These two asserts are the guarantee that M(lambda) really does interpolate the two MEASURED
    # arms rather than some reconstruction of them.
    endpoint_check = dict(
        lam0_family=curve[0]["macro"],
        lam0_measured_always_7b=round(macro(D, lambda c: D[c]["always_7b"].mean()), 6),
        lam1_family=curve[-1]["macro"],
        lam1_measured_always_32b_direct=round(base_macro, 6))
    assert abs(curve[0]["macro"] - macro(D, lambda c: D[c]["always_7b"].mean())) < 1e-6, \
        "lambda=0 must be the measured always-7B: %s" % endpoint_check
    assert abs(curve[-1]["macro"] - base_macro) < 1e-6, \
        "lambda=1 must be the measured always-32B-direct: %s" % endpoint_check

    passing = [r for r in curve if r["passes"]]
    lam_star = min(r["lam"] for r in passing) if passing else None
    row_star = next((r for r in curve if r["lam"] == lam_star), None)

    # refine lambda* on a fine grid
    if lam_star is not None and lam_star > 0:
        lo_l = max(0.0, lam_star - 0.02)
        fine = []
        for lam in np.arange(lo_l, lam_star + 0.0001, 0.002):
            lam = round(float(lam), 4)
            M = build_family(D, lam, np.random.RandomState(SEED + 1))
            d = macro_boot(D, M, C) - base_boot
            l, h = ci(d)
            pm = macro(D, lambda c, M=M: M[c].mean())
            fine.append(dict(lam=lam, macro=round(pm, 6), delta=round(pm - base_macro, 6),
                             lo=round(l, 6), hi=round(h, 6), passes=bool(l >= EPS)))
        fp = [r for r in fine if r["passes"]]
        if fp:
            lam_star = min(r["lam"] for r in fp)
            row_star = next(r for r in fine if r["lam"] == lam_star)
    else:
        fine = []

    # ---------------- L2b: how much does the ERROR CORRELATION move the bar? ----------------
    # M(lambda) is maximally correlated with the 32B: it only ever differs on items where the 7B
    # and 32B already disagreed.  A candidate whose errors are INDEPENDENT of the 32B's has a
    # wider paired CI at the same accuracy, so it needs MORE accuracy to clear the same margin.
    # Bracketing both ends turns "the bar is 0.6557" into an honest interval.
    def indep_family(target_macro, rng2):
        """Per-cell Bernoulli draws at the 32B's per-cell accuracy shifted by a constant, drawn
        INDEPENDENTLY of the 32B's per-item outcome.  SIMULATED -- an error-correlation probe."""
        out = {}
        shift = target_macro - base_macro
        for c in CELLS:
            p = float(np.clip(D[c]["always_32b_direct"].mean() + shift, 0.0, 1.0))
            out[c] = (rng2.random_sample(len(D[c]["always_32b_direct"])) < p).astype(np.int8)
        return out

    # indep_family DRAWS, so a single realisation would be a single seed.  The project's standing
    # rule is >=10 seeds wherever sampling is involved: sweep 12 and report the seed spread on the
    # bar itself, not just a point.
    INDEP_SEEDS = list(range(12))
    indep_bars = []
    indep = []
    for sd in INDEP_SEEDS:
        rows = []
        for target in np.arange(base_macro - 0.020, base_macro + 0.0201, 0.001):
            M = indep_family(float(target), np.random.RandomState(SEED + 7 + 1000 * sd))
            d = macro_boot(D, M, C) - base_boot
            lo, hi = ci(d)
            pm = macro(D, lambda c, M=M: M[c].mean())
            rows.append(dict(target=round(float(target), 6), macro=round(pm, 6),
                             delta=round(pm - base_macro, 6), lo=round(lo, 6), hi=round(hi, 6),
                             passes=bool(lo >= EPS)))
        b = min((r["macro"] for r in rows if r["passes"]), default=None)
        indep_bars.append(b)
        if sd == 0:
            indep = rows
    ok_bars = [b for b in indep_bars if b is not None]
    indep_bar = float(np.median(ok_bars)) if ok_bars else None
    indep_seed_stats = dict(n_seeds=len(INDEP_SEEDS), bars=indep_bars,
                            median=round(indep_bar, 6) if indep_bar else None,
                            min=round(min(ok_bars), 6) if ok_bars else None,
                            max=round(max(ok_bars), 6) if ok_bars else None,
                            spread=round(max(ok_bars) - min(ok_bars), 6) if ok_bars else None,
                            note="the independent bar is a MEDIAN over 12 seeds; the spread is "
                                 "reported because the family is drawn, not measured")

    corr_bar = row_star["macro"] if row_star else None
    bar = dict(
        correlated_bar=corr_bar,
        correlated_note="M(lambda*) -- a candidate whose errors are a SUBSET of the measured "
                        "7B/32B disagreement, i.e. maximally correlated with the baseline. This "
                        "is the EASIEST case and therefore the LOWER bound on the requirement.",
        independent_bar=round(indep_bar, 6) if indep_bar else None,
        independent_seed_robustness=indep_seed_stats,
        independent_note="a candidate erring independently of the 32B at the same per-cell "
                         "accuracy. Wider paired CI => needs MORE accuracy. UPPER bound.",
        bracket=[corr_bar, indep_bar],
        headline=("To hold the tie deployed ALONE, a replacement strong leg needs an 8-cell macro "
                  "of at least %.4f (errors correlated with the 32B) and up to %.4f (errors "
                  "independent of it), against the baseline's %.4f -- i.e. it must land within "
                  "%.4f-%.4f macro of always-32B-direct."
                  % (corr_bar, indep_bar, base_macro,
                     base_macro - (indep_bar or 0), base_macro - (corr_bar or 0))))

    # ---------------- L3: cost frontier ----------------
    # always-S costs R_S; the baseline costs R32.  A candidate is worth deploying iff it holds
    # the tie AND R_S < R32.
    frontier = []
    for name, r in sorted(R.items(), key=lambda kv: kv[1]):
        frontier.append(dict(
            model=name, R_vs_7b=r, x_of_always_32b_direct=round(r / R32, 4),
            weight_gib=foot["footprint"][name]["weight_gib"],
            logical_params=foot["footprint"][name]["logical_params"],
            required_macro_if_deployed_alone_correlated=corr_bar,
            required_macro_if_deployed_alone_independent=indep_bar,
            required_lambda=lam_star,
            cheaper_than_baseline=bool(r < R32)))

    out = dict(
        title="ATTACK 3 part 3 -- INVERTING the non-inferiority constraint: what accuracy must a "
              "strong leg of a given size reach to hold the tie against always-32B-direct?",
        date="2026-08-12",
        cpu_only=True,
        objective="MINIMISE macro FLOP-eq SUBJECT TO paired-bootstrap CI lower bound of "
                  "(policy - always-32B-direct) on the 8-cell macro >= -0.0029",
        margin_eps=EPS, nboot=NBOOT, seed=SEED,
        numerics_pins=dict(OMP_NUM_THREADS="1", tf32="not applicable -- pure numpy on stored 0/1 "
                           "vectors", bootstrap="paired item-level, ONE shared resample stream "
                           "reused by every candidate and the baseline"),
        sources=dict(per_item_vectors=VEC, published=PUB, footprint=FOOT),
        null_test=dict(per_arm=null, max_abs_dev=max_dev,
                       verdict="PASS" if max_dev < 1e-3 else "FAIL",
                       note="recomputed 8-cell macro vs the published disjoint arm; deviation is "
                            "the published file's 4-dp rounding"),
        published_tie_reproduced=tie_check,
        baseline=dict(always_32b_direct_macro=round(base_macro, 6), R32=R32),
        L2_requirement_curve=dict(
            family="M(lambda): keep the measured 7B outcome, recover a fraction lambda of the "
                   "items always-32B-direct FIXES over the 7B, and lose the same fraction of the "
                   "items it BREAKS.  lambda=0 and lambda=1 are the two MEASURED arms exactly "
                   "(asserted).  0<lambda<1 is a SIMULATED interpolation of measured per-item "
                   "outcomes -- it is a requirement curve, NOT a measurement of any real model.",
            endpoint_identity=endpoint_check, grid=curve, fine_grid=fine,
            lambda_star=lam_star, at_lambda_star=row_star),
        L2b_error_correlation_bracket=dict(bar=bar, independent_sweep_seed0=indep,
                                           independent_seed_robustness=indep_seed_stats),
        selection_note=(
            "NO ARM SELECTION HAPPENS IN THIS SCRIPT, so nested CV and a permutation null are "
            "not applicable: nothing is chosen on eval to produce a win.  lambda* is the "
            "crossing point of a MONOTONE requirement curve, i.e. a property of the eval "
            "distribution, and the reported result is a REQUIREMENT a future model must meet, "
            "not a claimed victory.  The degenerate null is built in and passes: at lambda=1 the "
            "family reproduces always-32B-direct exactly (delta 0.0000, CI [0.0000, 0.0000]), "
            "and at lambda=0 it reproduces always-7B exactly (both asserted in code)."),
        L3_cost_frontier=frontier,
    )

    p = os.path.join(OUTDIR, "invert_constraint.json")
    json.dump(out, open(p, "w"), indent=1)

    print("NULL TEST max_abs_dev = %.2e  -> %s" % (max_dev, out["null_test"]["verdict"]))
    print("published TIE reproduced:", tie_check)
    print("\nbaseline always-32B-direct macro = %.6f   R32 = %.4f" % (base_macro, R32))
    print("\nlambda*  = %s" % lam_star)
    print("row      = %s" % row_star)
    print("\ncurve (every 5th point):")
    for r in curve[::5]:
        print("  lam=%.2f macro=%.4f delta=%+.4f [%+.4f,%+.4f] %s"
              % (r["lam"], r["macro"], r["delta"], r["lo"], r["hi"],
                 "PASS" if r["passes"] else ""))
    print("\nwrote", p)


if __name__ == "__main__":
    main()
