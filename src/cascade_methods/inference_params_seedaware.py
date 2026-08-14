#!/usr/bin/env python3
"""inference_params_seedaware.py -- THE ADVERSARIAL TEST THE SWEEP DID NOT RUN.

The 2026-08-13 decoding sweep reports every delta with a PAIRED ITEM bootstrap, which
holds the generation seed FIXED and resamples only the 2345 questions. But the claim
being made is about two sampling DISTRIBUTIONS ("T=0.5 is better than T=0.7"), and each
generation seed is one draw from its distribution. Seed-to-seed variance is therefore a
real component of the uncertainty in the claim, and the item bootstrap does not contain
it: for T=0.7 the seed sd of SELECTED is 0.00396, which is comparable to the entire
effect being claimed (0.00569).

This script recomputes every headline delta THREE ways:

  (A) ITEM-ONLY   -- seeds stacked, resample items. What the sweep reported.
  (B) SEED-AWARE  -- resample items WITH replacement and, independently for each arm,
                     resample the 3 generation seeds WITH replacement. This is the
                     interval that supports a claim about the SETTING rather than about
                     three particular sampled pools.
  (C) SEED-LEVEL t -- Welch t on the 3 per-seed SELECTED values. Crude, 3 df, but it is
                     free of any bootstrap assumption and it is what a referee would do.

It also applies the FAMILY-WISE correction the sweep omitted: seven settings were each
tested against one control and the winner was picked by looking, so the per-comparison
95% interval is not the operating error rate.

Writes results/cascade_methods/artifacts/_infparams_seedaware.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
EVAL_DS = G.EVAL_DS
SETTINGS = ["T03", "T05", "T07", "T10", "T13", "minp01", "rp105", "rp11"]
CTRL = "T07"
NBOOT = 10000
SEED = 20260814


def load():
    z = np.load(os.path.join(ART, "_infparams_verify_got.npz"))
    got = {s: np.stack([z[f"{s}_s{d}_got"] for d in (0, 1, 2)]) for s in SETTINGS}
    rec = {s: np.stack([z[f"{s}_s{d}_rec"] for d in (0, 1, 2)]) for s in SETTINGS}
    return got, rec, z["ds_index"]


def boot(a, b, nboot=NBOOT, seed=SEED, seed_aware=True, mask=None):
    """a, b: (3, n) 0/1 matrices (seeds x items). Returns point estimate + CI.

    seed_aware=True resamples the 3 seeds of each arm independently WITH replacement,
    then resamples items; False averages the 3 seeds and resamples items only.
    """
    rng = np.random.default_rng(seed)
    ns, n = a.shape
    idx_all = np.arange(n) if mask is None else np.where(mask)[0]
    m = len(idx_all)
    point = a[:, idx_all].mean() - b[:, idx_all].mean()
    d = np.empty(nboot)
    for k in range(nboot):
        it = idx_all[rng.integers(0, m, m)]
        if seed_aware:
            sa = rng.integers(0, ns, ns)
            sb = rng.integers(0, ns, ns)
            d[k] = a[np.ix_(sa, it)].mean() - b[np.ix_(sb, it)].mean()
        else:
            d[k] = a[:, it].mean() - b[:, it].mean()
    return {"point": float(point),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "ci_fwer7": [float(np.percentile(d, 0.357)), float(np.percentile(d, 99.643))],
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
            "sd_boot": float(d.std())}


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        return {"diff": float(x.mean() - y.mean()), "se": 0.0, "t": None, "df": None,
                "ci95": [None, None]}
    t = (x.mean() - y.mean()) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    # two-sided 95% t critical value without scipy
    from math import erf, sqrt

    def tcrit(df_):
        # Cornish-Fisher style refinement of the normal quantile for small df
        z = 1.959963985
        g1 = (z ** 3 + z) / 4.0
        g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
        g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
        return z + g1 / df_ + g2 / df_ ** 2 + g3 / df_ ** 3
    tc = tcrit(df)
    return {"diff": float(x.mean() - y.mean()), "se": float(se), "t": float(t),
            "df": float(df), "t_crit_95": float(tc),
            "ci95": [float(x.mean() - y.mean() - tc * se),
                     float(x.mean() - y.mean() + tc * se)],
            "significant_95": bool(abs(t) > tc)}


def main():
    got, rec, dsidx = load()
    out = {
        "title": "SEED-AWARE RE-INFERENCE of the 2026-08-13 decoding sweep",
        "date": "2026-08-14",
        "no_fabricated_numbers": True,
        "why": "the sweep's paired ITEM bootstrap holds the generation seed fixed; the "
               "claim is about a sampling DISTRIBUTION, so generation-seed variance "
               "belongs in the interval. T=0.7's seed sd of SELECTED is 0.003962, "
               "comparable to the +0.005686 effect claimed for T=0.5.",
        "nboot": NBOOT, "bootstrap_seed": SEED,
        "n_settings_tested_against_one_control": len(SETTINGS) - 1,
        "fwer_note": "ci_fwer7 is the Bonferroni-corrected 95% family-wise interval "
                     "(per-comparison 99.286%) for the 7 settings tested against the "
                     "single control. The sweep's pre-registration predicted NO setting "
                     "would win, so the winner was identified by looking at the eval "
                     "endpoint -- the family-wise rate is the operating one.",
    }
    rows = {}
    for s in SETTINGS:
        if s == CTRL:
            continue
        a, b = got[s], got[CTRL]
        r = {
            "item_only_boot": boot(a, b, seed_aware=False),
            "seed_aware_boot": boot(a, b, seed_aware=True),
            "seed_level_welch": welch(a.mean(axis=1), b.mean(axis=1)),
            "per_seed_selected": {"setting": [float(v) for v in a.mean(axis=1)],
                                  "control": [float(v) for v in b.mean(axis=1)]},
            "seed_sd": {"setting": float(a.mean(axis=1).std(ddof=1)),
                        "control": float(b.mean(axis=1).std(ddof=1))},
        }
        pc = {}
        for j, d in enumerate(EVAL_DS):
            m = dsidx == j
            pc[d] = {"item_only": boot(a, b, seed_aware=False, mask=m),
                     "seed_aware": boot(a, b, seed_aware=True, mask=m)}
        r["per_cell"] = pc
        rows[s] = r
        io, sa, wl = r["item_only_boot"], r["seed_aware_boot"], r["seed_level_welch"]
        print(f"{s:7s} d={io['point']:+.6f}  item[{io['ci95'][0]:+.5f},{io['ci95'][1]:+.5f}]"
              f"  seedaware[{sa['ci95'][0]:+.5f},{sa['ci95'][1]:+.5f}]"
              f"  fwer[{sa['ci_fwer7'][0]:+.5f},{sa['ci_fwer7'][1]:+.5f}]"
              f"  welch t={wl['t']:.2f} df={wl['df']:.1f} "
              f"[{wl['ci95'][0]:+.5f},{wl['ci95'][1]:+.5f}]", flush=True)
    out["settings"] = rows

    # ---- how many of the 3x3 cross-seed pairings are positive, for T05 (robustness)
    a, b = got["T05"], got[CTRL]
    cross = [[float(a[i].mean() - b[j].mean()) for j in range(3)] for i in range(3)]
    out["T05_all_9_cross_seed_pairings"] = {
        "matrix_setting_seed_x_control_seed": cross,
        "n_positive": int(sum(1 for r_ in cross for v in r_ if v > 0)),
        "n_negative": int(sum(1 for r_ in cross for v in r_ if v < 0)),
        "min": float(min(v for r_ in cross for v in r_)),
        "max": float(max(v for r_ in cross for v in r_)),
        "note": "the sweep's per-seed deltas pair seed k with seed k, which is arbitrary "
                "-- the seeds are independent draws, so all 9 pairings are equally valid.",
    }
    print("T05 cross-seed pairings:", out["T05_all_9_cross_seed_pairings"]["n_positive"],
          "positive of 9, range",
          out["T05_all_9_cross_seed_pairings"]["min"],
          out["T05_all_9_cross_seed_pairings"]["max"])

    p = os.path.join(ART, "_infparams_seedaware.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
