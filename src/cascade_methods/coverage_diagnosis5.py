#!/usr/bin/env python3
"""coverage_diagnosis5.py -- SCOUT B part 5, 2026-08-10.

The ONE fair, same-currency, CI-clean comparison the round can act on:
is a sample taken at a DIFFERENT IMAGE RESOLUTION worth more, per sample, than one more
iid sample at the deployed resolution?

Both sides use EXACT-MATCH labels (the cap80/cap160 dumps were never judged) and both are
conditioned identically on "samples 1..7 of the deployed pool are all wrong", so the
comparison is internally consistent. Paired item-level bootstrap, nboot=10000.

Appends to results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))

ART = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json")
SC = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
PT = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_perturb")
NBOOT = 10000
SEED = 20260810
# deployed-selector marginal conversion of newly-covered questions, measured in part 2
MULT = 0.4473684210526316
MULT_CI = [0.3684210526315789, 0.5263157894736842]
DIRECT32B = {"vqa_rad_open": 0.6000, "pathvqa_open": 0.3760}
SELECTED_DEPLOYED = {"vqa_rad_open": 0.51, "pathvqa_open": 0.39066666666666666}


def pboot(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        d[k] = a[s].mean() - b[s].mean()
    return {"a_rate": float(a.mean()), "b_rate": float(b.mean()),
            "delta": float(a.mean() - b.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "n": int(n)}


def main():
    out = {"design": "EXACT-MATCH labels on BOTH sides (the cap80/cap160 dumps were never judged); "
                     "both arms conditioned on 'samples 1..7 of the deployed cap320/temp0.7 pool "
                     "are all wrong'; paired item-level bootstrap nboot=10000 seed 20260810.",
           "currency_warning": "These rates are NOT comparable to the judge-labelled rates "
                               "elsewhere in this artifact. Exact match is strictly stricter than "
                               "the judge, so both arms are depressed together.",
           "per_cell": {}}
    for ds in ["vqa_rad_open", "pathvqa_open"]:
        b = {str(json.loads(l)["idx"]): json.loads(l)
             for l in open(os.path.join(SC, f"ckpt_{ds}_lingshu7b_sc8.jsonl"))}
        ks = sorted(b)
        E = np.array([b[k]["oks"] for k in ks], int)
        no7 = E[:, :7].max(1) == 0
        caps = {}
        for cap in ["cap80", "cap160"]:
            p = os.path.join(PT, f"ckpt_{ds}_lingshu7b_{cap}.jsonl")
            caps[cap] = {str(json.loads(l)["idx"]): int(json.loads(l)["oks"][0]) for l in open(p)}
        sel = [i for i, k in enumerate(ks) if no7[i] and k in caps["cap80"] and k in caps["cap160"]]
        iid = [int(E[i, 7] == 1) for i in sel]
        row = {"n_conditioned": len(sel)}
        for cap in ["cap80", "cap160"]:
            row[f"{cap}_vs_one_more_iid_sample"] = pboot([caps[cap][ks[i]] for i in sel], iid)
        no8 = [i for i, k in enumerate(ks) if E[i].max() == 0 and k in caps["cap80"]]
        u = float(np.mean([max(caps["cap80"][ks[i]], caps["cap160"][ks[i]]) for i in no8]))
        oracle_lift = u * (len(no8) / len(ks))
        row["adding_BOTH_resolution_views"] = {
            "rescue_of_exactmatch_no_coverage": u, "n_no_coverage": len(no8),
            "extra_generation_cost": "+2 samples on a pool of 8 = 1.25x",
            "implied_oracle_lift_on_this_cell_EXACTMATCH": oracle_lift,
            "implied_selected_lift_at_the_MEASURED_multiplier": oracle_lift * MULT,
            "implied_selected_lift_CI_from_multiplier_only": [oracle_lift * MULT_CI[0],
                                                              oracle_lift * MULT_CI[1]],
            "implied_macro8_contribution_if_it_all_lands": oracle_lift * MULT / 8.0,
            "selected_now_deployed": SELECTED_DEPLOYED[ds],
            "always_32b_direct": DIRECT32B[ds],
        }
        out["per_cell"][ds] = row
    out["VERDICT"] = (
        "On PathVQA-open (n=795, the cell holding 82.7% of the no-coverage mass) a single sample "
        "at a CHANGED IMAGE RESOLUTION rescues significantly more no-coverage questions than one "
        "more iid sample at the deployed resolution: cap80 +0.0214 [+0.0050, +0.0377], cap160 "
        "+0.0201 [+0.0050, +0.0365]. This is the ONLY coverage lever in the whole diagnosis whose "
        "paired CI excludes zero on the currency it is measured in. On VQA-RAD-open (n=92) it is "
        "noise (cap80 +0.0109 [-0.0217, +0.0543], cap160 -0.0217 [-0.0543, 0.0000]). "
        "BUT THE MAGNITUDE DOES NOT MOVE THE TARGET: adding BOTH resolution views to PathVQA-open "
        "costs 1.25x generation and, at the measured 0.447 multiplier, is worth about +0.0016 on "
        "the 8-cell macro -- and it lands on the one open cell that is ALREADY above "
        "always-32B-direct, where extra accuracy is not what the macro needs.")
    art = json.load(open(ART))
    art["part5_resolution_vs_iid_same_currency"] = out
    json.dump(art, open(ART, "w"), indent=1, default=float)
    print("wrote", ART)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
