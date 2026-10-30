#!/usr/bin/env python3
"""temp_modality.py -- per-modality sampling temperature, with the modality named by the model itself.

The "per-modality T" item of the 2026-10-19 deck's next steps. A deployed system knows (or can cheaply find out)
an image's modality but not which benchmark it came from. Inputs: temp_labelfree.py's per-question records
(the shipped probe's pick at each of T = 0.2 / 0.4 / 0.7 / 1.0, both folds, all 8 benchmarks) and
predict_modality.py's labels (Lingshu-7B's own answer, mapped to a class). Policy: for each predicted modality,
pooled ACROSS benchmarks, the temperature with the best training-judge gain on image-fold A is applied to that
modality's questions in fold B, and vice versa (cross-fitted, as temp_crossfit.py chose its per-benchmark map).
A modality with fewer than MIN_Q questions in the choosing fold keeps T = 0.7. Compared, paired and
image-clustered, with fixed T = 0.7 (label-free) and the frozen per-benchmark map (needs the label), in judge and
lenient EM (and MedGemma where every pick is labelled).

  python3 src/exp_2026_10/temp_modality.py
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np

REC = "/data/dan/exp_2026_10/temp_labelfree_records.json"
MOD = "/data/dan/exp_2026_10/modality_pred_lingshu7b.jsonl"
TMAP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                   "results/cascade_methods/artifacts/frozen_temperature_map_2026-10-15.json")))["frozen_T"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results/cascade_methods/artifacts/temp_modality_2026-10-22.json")
TS, MIN_Q, NB, SEED = ["0.2", "0.4", "0.7", "1.0"], 50, 10000, 20261022
R = json.load(open(REC))
mod = {(d["cell"], str(d["idx"])): d["modality"] for d in (json.loads(l) for l in open(MOD) if l.strip())}
Q = []
for c, recs in R.items():
    for r in recs:
        m = mod.get((c, str(r["idx"])))
        if m is None:
            continue
        Q.append({"c": c, "f": r["fold"], "img": r["img"], "m": m, "g": r["greedy"], "p": r["pick_by_T"]})
print(f"{len(Q)} questions with a predicted modality; classes {dict(Counter(x['m'] for x in Q).most_common())}")
gain = lambda x, T, cu: (x["p"][T][cu] if x["p"][T][cu] is not None else np.nan) - x["g"][cu]

choice = {}
for f in (0, 1):
    for m in sorted({x["m"] for x in Q}):
        sub = [x for x in Q if x["f"] == f and x["m"] == m]
        if len(sub) < MIN_Q:
            choice[(f, m)] = "0.7"; continue
        choice[(f, m)] = max(TS, key=lambda T: np.nanmean([gain(x, T, "judge") for x in sub]))
pol = {"per_modality_xfit": lambda x: choice[(1 - x["f"], x["m"])], "fixed_T07": lambda x: "0.7",
       "tmap_needs_label": lambda x: str(TMAP.get(x["c"], 0.7))}
cells = sorted({x["c"] for x in Q}); rng = np.random.default_rng(SEED); W = {}
for c in cells:
    sel = [i for i, x in enumerate(Q) if x["c"] == c]; keys, inv = np.unique([Q[i]["img"] for i in sel], return_inverse=True)
    W[c] = (np.array(sel), inv, rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NB).astype(float), len(keys))


def paired(a, b):
    reps, pts = [], []
    for c in cells:
        sel, inv, w, nk = W[c]; d = a[sel] - b[sel]; mk = ~np.isnan(d); d = np.where(mk, d, 0.0)
        reps.append((w @ np.bincount(inv, weights=d, minlength=nk)) / np.maximum(w @ np.bincount(inv, weights=mk.astype(float), minlength=nk), 1e-12))
        pts.append(d[mk].mean())
    Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}


res = {}
for name, fn in pol.items():
    e = {}
    for cu in ("judge", "em"):
        v = np.array([gain(x, fn(x), cu) for x in Q])
        e[cu] = {"gain_over_greedy": paired(v, np.zeros(len(v))),
                 "vs_fixed_T07": paired(v, np.array([gain(x, "0.7", cu) for x in Q])),
                 "vs_tmap": paired(v, np.array([gain(x, str(TMAP.get(x["c"], 0.7)), cu) for x in Q]))}
    res[name] = e
    print(f"  {name:20} judge {e['judge']['gain_over_greedy']['macro']:+.4f}  vs T07 {e['judge']['vs_fixed_T07']['macro']:+.4f} "
          f"{e['judge']['vs_fixed_T07']['verdict']}  vs Tmap {e['judge']['vs_tmap']['macro']:+.4f} {e['judge']['vs_tmap']['verdict']}"
          f"  | EM vs T07 {e['em']['vs_fixed_T07']['macro']:+.4f} {e['em']['vs_fixed_T07']['verdict']}")
json.dump({"title": "Per-modality temperature, modality predicted by Lingshu-7B (cross-fitted)", "date": "2026-10-22",
           "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "n_questions": len(Q), "min_q": MIN_Q,
           "choices": {f"fold{f}|{m}": T for (f, m), T in choice.items()}, "policies": res,
           "ci_caveat": "the per-modality choice is not re-made inside each bootstrap resample", "COMPLETE": True},
          open(OUT, "w"), indent=1)
print("wrote", os.path.abspath(OUT))
