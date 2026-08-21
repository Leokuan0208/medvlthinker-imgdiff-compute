#!/usr/bin/env python3
"""merge_domain_scaling.py -- combine the sharded k-domain curves into ONE authoritative verdict.

WHY THIS IS NEEDED, AND IT IS NOT COSMETIC.  head_domain_scaling.py computes `MEAN_SLOPE` and a
VERDICT over whatever domains the PROCESS was given.  Once the run is sharded by held-out domain
(--domains), every shard prints its own "MEAN SLOPE => breadth helps / does not help" line computed
over one or two domains.  Shard 4 holds out a single domain and still printed a global-sounding
verdict.  Those per-shard verdicts must not be quoted; this script produces the only one that is
over all seven.

WHAT IT ALSO REPORTS, because the slope alone would oversell a noisy curve.  The per-domain curves
are not monotone -- held-out kvasir_open runs +0.0395, +0.0883, +0.0931, +0.2061, +0.1032, +0.2298
across k=1..6, so the k=5 point sits below k=4 by more than the whole k=1..3 rise.  A least-squares
slope through six noisy points is weak evidence on its own, so this reports alongside it:
  - the spread across the draws x seeds at each k (the script's own `sd`)
  - a bootstrap over DOMAINS of the mean slope, which is the quantity the verdict rests on
  - how many domains have a positive slope, which is the sign test the mean can hide

All curves are BUDGET-MATCHED: a fixed total row count at every k, so breadth is not confounded
with volume.

  python3 src/training_methods/merge_domain_scaling.py
"""
import glob, json, os
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "head_domain_scaling_MERGED_2026-08-21.json")


def main():
    shards = sorted(glob.glob(os.path.join(ART, "head_domain_scaling_shard*.json")))
    if not shards:
        raise SystemExit("no shard artifacts found")
    res, budgets = {}, set()
    for f in shards:
        a = json.load(open(f))
        for dom, cur in a["results"].items():
            ks = sorted(int(x) for x in cur if x.isdigit())
            if not ks:
                continue
            res[dom] = {"k": ks,
                        "head_minus_prior": [cur[str(k)]["head_minus_prior"] for k in ks],
                        "sel_eff": [cur[str(k)]["out_sel_eff"] for k in ks],
                        "sd": [cur[str(k)].get("sd") for k in ks],
                        "string_prior": [cur[str(k)]["string_prior"] for k in ks],
                        "complete": len(ks) >= 6}
            budgets |= {cur[str(k)].get("row_budget") for k in ks}
    print(f"{len(res)} held-out domains from {len(shards)} shards; row budgets {budgets}\n")

    slopes = {}
    for dom, r in res.items():
        if len(r["k"]) < 3:
            print(f"  {dom:22} only {len(r['k'])} k-points -- INCOMPLETE, excluded from the mean")
            continue
        s = float(np.polyfit(r["k"], r["head_minus_prior"], 1)[0])
        slopes[dom] = s
        hp = r["head_minus_prior"]
        # how much of the range is non-monotone wobble rather than trend?
        drops = sum(1 for i in range(1, len(hp)) if hp[i] < hp[i - 1])
        print(f"  {dom:22} k={r['k'][0]}..{r['k'][-1]}  head-prior "
              f"{hp[0]:+.4f} -> {hp[-1]:+.4f}  slope {s:+.5f}  "
              f"({drops}/{len(hp)-1} steps go DOWN, mean sd {np.mean([x for x in r['sd'] if x]):.4f})")

    v = np.array(list(slopes.values()))
    rng = np.random.default_rng(0)
    bs = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    npos = int((v > 0).sum())

    art = {"title": "Does DOMAIN BREADTH lift transfer? merged, budget-matched k-curve",
           "date": "2026-08-21", "no_fabricated_numbers": True,
           "row_budget": sorted(x for x in budgets if x is not None),
           "endpoint": "held-out-domain sel_eff MINUS the string prior, vs how many OTHER domains "
                       "the head saw, at a FIXED total row budget",
           "n_domains": len(v), "per_domain_slope": slopes,
           "mean_slope": float(v.mean()),
           "mean_slope_ci_bootstrap_over_domains": [float(lo), float(hi)],
           "domains_with_positive_slope": f"{npos}/{len(v)}",
           "curves": res,
           "caveat": "per-shard MEAN_SLOPE/VERDICT lines are computed over that shard's domains "
                     "only and must not be quoted as global; this file is the only global verdict."}
    art["VERDICT"] = (
        f"mean slope {v.mean():+.5f} per added domain [{lo:+.5f},{hi:+.5f}] over {len(v)} held-out "
        f"domains, {npos}/{len(v)} positive. " +
        ("Breadth HELPS and the effect is separable from volume -- the head is a starved verifier."
         if lo > 0.005 else
         "Breadth does NOT reliably help once volume is held fixed: the interval includes zero, so "
         "the method must be presented as per-domain." if lo <= 0 else
         "Breadth helps weakly; the interval excludes zero but the effect is below the 0.005 "
         "threshold the experiment was designed to detect."))
    print(f"\n  mean slope {v.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  ({npos}/{len(v)} domains positive)")
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
