#!/usr/bin/env python3
"""refit_summary.py -- the run-to-run spread of the final method's headline under pure probe refitting.

Replicate 0 = the shipped FINAL run (seeds 0-7); replicates 100-400 = refit_variance.py (seeds shifted, everything
else identical, final_method_eval.main() unchanged). For each policy (T07 = the deck's headline, Tmap =
FINAL_METHOD's) and currency: the macro gain over greedy per replicate, mean, sample sd and range, plus the share of
questions on which each replicate's pick differs from the shipped pick. The sd is the yardstick every claimed
difference in the probe must be read against.

  python3 src/exp_2026_10/refit_summary.py
"""
import json, os
import numpy as np
A = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results/cascade_methods/artifacts/refit_variance_2026-10-22.json")
REPS = [0, 100, 200, 300, 400]
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
name = lambda r, p: f"final_eval_FINAL_cap320_{p}" if r == 0 else f"final_eval_FINAL_refit{r}_{p}"
out = {"title": "Refit variance of the final method (probe seeds only)", "date": "2026-10-22", "no_fabricated_numbers": True,
       "script": os.path.abspath(__file__), "replicates": REPS, "policies": {}}
for p in ("T07", "Tmap"):
    have = [r for r in REPS if os.path.exists(f"{A}/{name(r, p)}.json")]
    e = {"replicates_present": have}
    for cu in ("judge", "xjudge", "em"):
        v = [json.load(open(f"{A}/{name(r, p)}.json"))["macro_over_open8"]["all"][f"pick_minus_greedy_{cu}"]["macro"] for r in have]
        e[cu] = {"per_replicate": dict(zip(map(str, have), v)), "mean": float(np.mean(v)),
                 "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None, "range": [float(min(v)), float(max(v))]}
    ref = json.load(open(f"/data/dan/exp_2026_10/{name(0, p)}_records.json"))
    pk0 = {(c, r["idx"]): r["pick_ans"] for c in OPEN8 for r in ref[c]}
    e["pick_change_vs_shipped"] = {}
    for r in have[1:]:
        rec = json.load(open(f"/data/dan/exp_2026_10/{name(r, p)}_records.json"))
        pk = {(c, x["idx"]): x["pick_ans"] for c in OPEN8 for x in rec[c]}
        sh = [k for k in pk if k in pk0]
        e["pick_change_vs_shipped"][str(r)] = float(np.mean([pk[k] != pk0[k] for k in sh]))
    out["policies"][p] = e
    print(p, {cu: (round(e[cu]["mean"], 4), None if e[cu]["sd"] is None else round(e[cu]["sd"], 4)) for cu in ("judge", "xjudge", "em")},
          "replicates", have)
json.dump(out, open(OUT, "w"), indent=1); print("wrote", os.path.abspath(OUT))
