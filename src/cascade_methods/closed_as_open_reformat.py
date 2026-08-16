#!/usr/bin/env python3
"""closed_as_open_reformat.py -- BUILD 3 step 2: does the REFORMAT ALONE cost or gain anything, and
if it gains, WHY?

The main analysis compares every open arm against `closedD_g`, the in-session cap320 control, which
isolates the prompt from the serving config.  This module adds the comparison a deployer actually
cares about: open-form cap320 against **closedD_g_full**, the DEPLOYED operating point (MedEvalKit's
own prompt at fullres) regenerated in this same engine.  The open arms run at LOWER resolution, so a
positive delta is CONSERVATIVE -- it is cheaper AND better.

It also decomposes the effect instead of asserting it.  The deployed judgement prompt ends
"Please output 'yes' or 'no'(no extra output)."  The pre-committed grading-artifact diagnostics
(length, unparsed rate, harness-vs-repaired gap) already rule out a grader effect, so the remaining
candidate mechanism is a POLARITY PRIOR: does that instruction bias the 7B toward "yes"?  Reported as
predicted-yes rate vs the gold base rate, plus sensitivity and specificity, per arm.

⚠️ THE MECHANISM DECOMPOSITION IS POST-HOC.  The reformat-vs-closedD_g comparison is pre-registered
(closed_as_open_2026-08-16_preregistration.json, arm_rationale.closedD_g); the yes-bias split and the
vs-fullres contrast were added after the deltas were seen and are labelled exploratory.

CPU only.  python3 src/cascade_methods/closed_as_open_reformat.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402
from closed_as_open_analyse import paired_boot                            # noqa: E402

#: openMEK_g_full / openPRJ_g_full are POST-HOC (added after the primary endpoint was read).  They
#: isolate the PROMPT from the RESOLUTION: without them, "just change the prompt" is confounded with
#: "and drop to cap320", because every pre-registered open arm ran at cap320.
GREEDY_ARMS = ["closedD_g", "closedD_g_full", "openMEK_g", "openPRJ_g",
               "openMEK_g_full", "openPRJ_g_full"]
REF = "closedD_g_full"


def main():
    out = {
        "question": "does the REFORMAT alone -- a prompt change, zero extra compute -- beat the "
                    "DEPLOYED operating point?",
        "deployed_operating_point": f"{REF} = MedEvalKit's own prompt for the cell at FULLRES, "
                                    "regenerated in THIS session's engine (the matched control the "
                                    "+-0.008 reproducibility caveat requires)",
        "why_a_positive_delta_is_conservative": "the open arms run at cap320, i.e. 1/4 the pixel "
                                               "budget of the deployed fullres point, so they are "
                                               "CHEAPER than the thing they are being compared to",
        "status": "the open-vs-closedD_g contrast is PRE-REGISTERED; the vs-fullres contrast and the "
                  "yes-bias decomposition below are POST-HOC and exploratory",
        "cells": {},
    }
    for cell in L.CELLS:
        jm = L.judge_map(cell)
        gen = {a: L.load_gen(cell, a) for a in GREEDY_ARMS}
        missing = [a for a in GREEDY_ARMS if len(gen[a]) != L.EXPECT_N[cell]]
        for a in missing:
            print(f"  [skip] {cell}/{a}: {len(gen[a])}/{L.EXPECT_N[cell]} rows -- NOT MEASURED")
            gen.pop(a)
        arms = [a for a in GREEDY_ARMS if a in gen]
        idxs = sorted(gen["closedD_g"])
        row = {"n": len(idxs),
               "published_always_7b_deployed_harness": L.PUBLISHED_ALWAYS_7B[cell],
               "arms_measured": arms,
               "arms_not_measured": missing,
               "post_hoc_arms": [a for a in arms if a in L.POST_HOC_ARMS]}
        print(f"\n===== {cell} (n={len(idxs)}) published={L.PUBLISHED_ALWAYS_7B[cell]} =====")
        for cur in ("judge", "em_repaired"):
            def lab(i, a):
                p = gen[a][i]["preds"][0]
                if cur == "judge":
                    v = jm.get((i, L.norm_text(p)))
                    return 0 if v is None else int(v)
                return L.em_repaired(cell, gen[a][i]["gold"], p)[0]
            V = {a: np.array([lab(i, a) for i in idxs], dtype=float) for a in arms}
            r = {a: round(float(V[a].mean()), 6) for a in arms}
            for a in [x for x in arms if x != REF]:
                pb = paired_boot(V[a], V[REF])
                r[f"{a}_minus_{REF}"] = pb
                print(f"  [{cur:11s}] {a:11s}={V[a].mean():.4f} vs {REF}={V[REF].mean():.4f} : "
                      f"{pb['delta']:+.4f} [{pb['ci'][0]:+.4f},{pb['ci'][1]:+.4f}] {pb['sign']}")
            row[cur] = r

        # ---- POST-HOC mechanism: is the deployed prompt inducing a yes-bias? -------------------
        golds = [L.norm_text(gen["closedD_g"][i]["gold"]) for i in idxs]
        yn = [k for k, _ in enumerate(idxs) if golds[k] in ("yes", "no")]
        if yn:
            base = float(np.mean([golds[k] == "yes" for k in yn]))
            mech = {"status": "POST-HOC", "n_yesno_items": len(yn),
                    "gold_yes_base_rate": round(base, 6)}
            print(f"  MECHANISM [POST-HOC] yes/no items n={len(yn)} gold yes base rate={base:.4f}")
            for a in arms:
                pol = [L.polarity(gen[a][idxs[k]]["preds"][0]) for k in yn]
                yr = float(np.mean([p == "yes" for p in pol]))
                sens = float(np.mean([pol[j] == "yes" for j, k in enumerate(yn) if golds[k] == "yes"]))
                spec = float(np.mean([pol[j] == "no" for j, k in enumerate(yn) if golds[k] == "no"]))
                mech[a] = {"predicted_yes_rate": round(yr, 6),
                           "yes_rate_minus_gold_base_rate": round(yr - base, 6),
                           "sensitivity_gold_yes": round(sens, 6),
                           "specificity_gold_no": round(spec, 6)}
                print(f"     {a:15s} predYes={yr:.4f} (bias {yr-base:+.4f}) "
                      f"sens={sens:.4f} spec={spec:.4f}")
            row["mechanism_yes_bias"] = mech
        out["cells"][cell] = row
    os.makedirs(L.PARTS, exist_ok=True)
    p = os.path.join(L.PARTS, "reformat_vs_deployed.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\nwrote", p)


if __name__ == "__main__":
    main()
