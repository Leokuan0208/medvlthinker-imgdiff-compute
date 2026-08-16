#!/usr/bin/env python3
"""closed_as_open_permnull.py -- BUILD 3: the permutation null for the MULTIPLICITY of arms.

The pre-registered primary endpoint is a single fixed comparison (openPRJ_s8 vs openPRJ_g), so it
needs no multiplicity correction.  But the artifact reports THREE sampled arms x THREE cells, and the
best-looking of those nine is closedD_s8 on VQA_RAD_closed at +0.0199.  This project has already
learned the hard way that a pick-the-best rule earns +0.0109 macro from SHUFFLED LABELS
(artifacts/unified_pipeline_2026-08-12.json), so the honest guard is: how large a "best of nine"
would a verifier with NO SKILL AT ALL produce here?

THE NULL.  Replace the verifier's pick with a uniformly random slot -- exactly the random-pick floor
whose expectation the main artifact already reports -- redraw it `nperm` times, and record, per draw,
the maximum over the nine (cell, arm) combinations of (SELECTED_null - greedy).  The observed maximum
is then read against that distribution.

This is NOT abstention-related and changes no deployed policy: every arm still answers every item.

CPU only.  python3 src/cascade_methods/closed_as_open_permnull.py [--nperm 2000]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402
from closed_as_open_analyse import (GREEDY_OF, SAMPLED_ARMS, contaminated,  # noqa: E402
                                    picks_from)

SEED = 20260816


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nperm", type=int, default=2000)
    A = ap.parse_args()

    out = {"title": "BUILD 3 permutation null for arm x cell multiplicity",
           "date": L.DATE, "nperm": A.nperm, "seed": SEED,
           "null": "the verifier's pick is replaced by a uniformly random slot; the statistic is "
                   "max over the nine (cell, arm) combinations of (SELECTED - greedy)",
           "note": "the PRE-REGISTERED primary endpoint is a single fixed comparison and needs no "
                   "correction; this guards the reader against the best-looking of the nine arms "
                   "reported alongside it",
           "per_currency": {}}

    for cur in ("judge", "em_repaired"):
        combos, observed = [], {}
        for cell in L.CELLS:
            bad = contaminated(cell) or set()
            jm = L.judge_map(cell)
            for arm in SAMPLED_ARMS:
                gname = GREEDY_OF[arm]
                gen, gg, sc = L.load_gen(cell, arm), L.load_gen(cell, gname), L.load_scores(cell, arm)
                idxs = sorted(i for i in gen if i in gg and i in sc and i not in bad)
                if not idxs:
                    continue

                def lab(i, p):
                    if cur == "judge":
                        v = jm.get((i, L.norm_text(p)))
                        return 0 if v is None else int(v)
                    return L.em_repaired(cell, gen[i]["gold"], p)[0]
                O = np.array([[lab(i, p) for p in gen[i]["preds"]] for i in idxs], dtype=float)
                gv = np.array([lab(i, gg[i]["preds"][0]) for i in idxs], dtype=float)
                sel = O[np.arange(len(idxs)), [picks_from(sc[i]) for i in idxs]]
                combos.append((f"{cell}/{arm}", O, gv))
                observed[f"{cell}/{arm}"] = float(sel.mean() - gv.mean())

        rng = np.random.default_rng(SEED)
        nullmax = np.empty(A.nperm)
        for k in range(A.nperm):
            best = -np.inf
            for _, O, gv in combos:
                j = rng.integers(0, O.shape[1], O.shape[0])
                best = max(best, float(O[np.arange(O.shape[0]), j].mean() - gv.mean()))
            nullmax[k] = best
        obs_max_key = max(observed, key=observed.get)
        obs_max = observed[obs_max_key]
        p = float((nullmax >= obs_max).mean())
        out["per_currency"][cur] = {
            "n_combinations": len(combos),
            "observed_delta_per_combination": {k: round(v, 6) for k, v in observed.items()},
            "observed_best_combination": obs_max_key,
            "observed_best_delta": round(obs_max, 6),
            "null_max_mean": round(float(nullmax.mean()), 6),
            "null_max_p95": round(float(np.percentile(nullmax, 95)), 6),
            "p_value_best_of_nine": round(p, 6),
            "significant_at_0.05": bool(p < 0.05)}
        r = out["per_currency"][cur]
        print(f"[{cur}] best = {obs_max_key} {obs_max:+.4f} ; null max mean "
              f"{r['null_max_mean']:+.4f} p95 {r['null_max_p95']:+.4f} ; "
              f"p(best of {len(combos)}) = {p:.4f}")

    os.makedirs(L.PARTS, exist_ok=True)
    q = os.path.join(L.PARTS, "perm_null_multiplicity.json")
    json.dump(out, open(q, "w"), indent=1)
    print("wrote", q)


if __name__ == "__main__":
    main()
