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
    # SENSITIVITY THAT DECIDES THIS RESULT.  kvasir_open and kvasir_x1_open are both Kvasir GI
    # endoscopy -- the same underlying image source, split into two cells.  They are NOT two
    # independent domains, and they happen to carry the two largest slopes AND the two largest
    # per-k spreads (sd 0.0553 and 0.0545).  Counting them as 2 of 7 independent points inflates
    # both the mean and the confidence, so the collapsed figure is reported beside the raw one and
    # the verdict is required to hold under BOTH.
    KVASIR = [d for d in slopes if d.startswith("kvasir")]
    if len(KVASIR) > 1:
        coll = {d: sl for d, sl in slopes.items() if not d.startswith("kvasir")}
        coll["kvasir_family"] = float(np.mean([slopes[d] for d in KVASIR]))
        cv = np.array(list(coll.values()))
        cbs = np.array([np.mean(rng.choice(cv, len(cv), replace=True)) for _ in range(10000)])
        clo, chi = np.percentile(cbs, [2.5, 97.5])
        art["collapsed_kvasir_family"] = {
            "why": "kvasir_open and kvasir_x1_open are the same GI-endoscopy source; they carry the "
                   "two largest slopes and the two largest per-k spreads, so counting them twice "
                   "inflates the mean and the confidence",
            "per_domain_slope": coll, "n_domains": len(cv),
            "mean_slope": float(cv.mean()), "ci": [float(clo), float(chi)],
            "domains_with_positive_slope": f"{int((cv > 0).sum())}/{len(cv)}"}
        print(f"  collapsing the Kvasir pair: mean slope {cv.mean():+.5f} [{clo:+.5f},{chi:+.5f}] "
              f"over {len(cv)} sources ({int((cv > 0).sum())}/{len(cv)} positive)")
    else:
        clo = lo

    both_positive = lo > 0 and clo > 0
    art["VERDICT"] = (
        f"mean slope {v.mean():+.5f} per added domain [{lo:+.5f},{hi:+.5f}] over {len(v)} held-out "
        f"domains, {npos}/{len(v)} positive; collapsing the Kvasir pair gives "
        f"[{clo:+.5f},...]. " +
        ("Breadth HELPS and the effect survives both the volume control and the Kvasir collapse -- "
         "the head is a starved verifier, not only a per-domain tool."
         if both_positive and lo > 0.005 else
         "Breadth helps only WEAKLY: the raw interval excludes zero but its lower bound is below "
         "the 0.005-per-domain effect this design was built to detect, and the mean is carried by "
         "two correlated Kvasir curves that are also the two noisiest. Do not present breadth as "
         "the fix." if both_positive else
         "Breadth does NOT reliably help once volume is held fixed and the duplicated Kvasir "
         "source is collapsed: the method must be presented as per-domain."))
    print(f"\n  mean slope {v.mean():+.5f} [{lo:+.5f},{hi:+.5f}]  ({npos}/{len(v)} domains positive)")
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
