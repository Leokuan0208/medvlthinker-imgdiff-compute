#!/usr/bin/env python3
"""verifier_hparams_isoesc.py -- KNOB 3: is cap320's macro gain a BETTER SELECTOR or just MORE
ESCALATION?

THE OBSERVATION THIS SCRIPT EXPLAINS.  Scoring the verifier at 250,880 instead of the deployed
1,003,520 makes the SELECTOR significantly worse (judge sel_eff -0.0177 [-0.0300,-0.0055], EM
-0.0142 [-0.0263,-0.0021], guardrail-dirty on all three sets in both currencies) and yet makes the
end-to-end 8-cell MACRO significantly HIGHER (+0.0013 [+0.0001,+0.0026] paired vs the in-session
control). Those two facts are only consistent if the degraded verifier is ESCALATING MORE: the
open-text gate IS the selector's own max(score), so damaging the score damages the pick and, at
the same time, pushes confidence down and hands more questions to the 32B -- which is better than
the 7B pick on all three open cells.

THE TEST.  Put both arms on the SAME escalation budget and re-read the accuracy. For each open
cell and each arm, rank items by that arm's own gate (max verifier score over the pool), escalate
the lowest-gate fraction e, and take the 32B answer there and the arm's own pick elsewhere. Sweep
e from 0 to 1. If cap320's advantage is a better selector, its curve is ABOVE the control's at
matched e. If it is bought escalation, its curve is AT OR BELOW the control's everywhere and the
two only differ because the deployed policy lands them at different points on their own curves.

No new inference: the per-candidate scores come from this round's own arms, sl/greedy from the
frozen transfer dumps, and the 32B leg from the project's existing judged dumps.

    python3 src/cascade_methods/verifier_hparams_isoesc.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from src.training_methods import genframe_data as G   # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
STRONG = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu")
CONTROL_PX = 1003520
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
NBOOT = 10000
BSEED = 20260816


def load_judge(ds):
    """The 32B-direct judged per-item outcome, from the project's existing dump."""
    p = os.path.join(STRONG, f"ckpt_{ds}_lingshu32b.judge.jsonl")
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            v = r.get("judge_ok", r.get("ok"))
            if v is not None:
                out[int(r["idx"])] = int(v)
    return out


def arm_vectors(px):
    """Per open cell: ok7 (arm's own pick correct, judge), gate (arm's own max score), ok32."""
    d = os.path.join(ROOT, f"ckpts/train/verifhp_px{px}")
    out = {}
    for ds in DS:
        rows = json.load(open(os.path.join(d, f"transfer_dump_{ds}_lingshu7b.json")))
        sj = load_judge(ds)
        ok7, gate, ok32 = [], [], []
        for r in rows:
            if r["idx"] not in sj:
                continue
            sc = np.asarray(r["scores"][:8], float)
            sl = [0 if x in (None, -1) else int(x) for x in r["sl"][:8]]
            k = int(np.argmax(sc))
            ok7.append(sl[k]); gate.append(float(sc.max())); ok32.append(sj[r["idx"]])
        out[ds] = (np.array(ok7, float), np.array(gate, float), np.array(ok32, float))
    return out


def curve(ok7, gate, ok32, grid):
    """Accuracy when the lowest-gate fraction e is escalated (ties broken by gate order)."""
    n = len(ok7)
    order = np.argsort(gate, kind="mergesort")      # ascending: least confident first
    acc = []
    for e in grid:
        k = int(round(e * n))
        esc = np.zeros(n, bool)
        esc[order[:k]] = True
        acc.append(float(np.where(esc, ok32, ok7).mean()))
    return np.array(acc)


def main():
    grid = np.linspace(0, 1, 101)
    pxs = sorted(int(os.path.basename(p).split("px")[1])
                 for p in glob.glob(os.path.join(ROOT, "ckpts/train/verifhp_px*")))
    V = {px: arm_vectors(px) for px in pxs}
    ctrl = V[CONTROL_PX]

    rep = {"_what": "iso-escalation curves: accuracy vs escalation budget, each arm ranked by its "
                    "OWN gate (max verifier score). Answers whether a rung's end-to-end gain is a "
                    "better selector or simply more 32B calls.",
           "_grid": "escalation fraction 0..1 in 101 steps",
           "_open_cells_only": DS,
           "_32b_leg": "ckpts/openvqa/strong_lingshu/ckpt_{ds}_lingshu32b.judge.jsonl (existing)",
           "by_max_pixels": {}}

    # macro over the 3 open cells, equal weight (their weight inside the 8-cell macro is equal)
    for px in pxs:
        rows = {}
        macro = np.zeros_like(grid)
        for ds in DS:
            ok7, gate, ok32 = V[px][ds]
            c = curve(ok7, gate, ok32, grid)
            cc = curve(*ctrl[ds], grid)
            rows[ds] = {"n": int(len(ok7)),
                        "acc_at_esc0_own_pick": float(c[0]),
                        "acc_at_esc1_always_32b": float(c[-1]),
                        "curve": [round(float(x), 6) for x in c],
                        "max_gap_vs_control_over_grid": float(np.max(c - cc)),
                        "mean_gap_vs_control_over_grid": float(np.mean(c - cc)),
                        "n_grid_points_where_arm_beats_control": int((c > cc + 1e-12).sum()),
                        "n_grid_points_where_control_beats_arm": int((cc > c + 1e-12).sum())}
            macro += c / len(DS)
        rep["by_max_pixels"][str(px)] = {
            "max_pixels": px, "per_cell": rows,
            "open_macro_curve": [round(float(x), 6) for x in macro]}

    cm = np.array(rep["by_max_pixels"][str(CONTROL_PX)]["open_macro_curve"])
    for px in pxs:
        m = np.array(rep["by_max_pixels"][str(px)]["open_macro_curve"])
        rep["by_max_pixels"][str(px)]["vs_control_open_macro"] = {
            "max_gap": float(np.max(m - cm)), "mean_gap": float(np.mean(m - cm)),
            "min_gap": float(np.min(m - cm)),
            "n_grid_points_arm_ahead": int((m > cm + 1e-12).sum()),
            "n_grid_points_control_ahead": int((cm > m + 1e-12).sum()),
            "area_under_gap": float(np.trapz(m - cm, grid))}
    json.dump(rep, open(os.path.join(PARTS, "isoesc.json"), "w"), indent=1, default=float)

    print("iso-escalation, OPEN-CELL MACRO (equal weight over the 3 open cells)")
    print("  esc:      " + "".join(f"{e:>9.0%}" for e in [0, .1, .2, .3, .4, .5, .6, .8, 1.0]))
    idx = [0, 10, 20, 30, 40, 50, 60, 80, 100]
    for px in pxs:
        m = rep["by_max_pixels"][str(px)]["open_macro_curve"]
        tag = " <== DEPLOYED" if px == CONTROL_PX else ""
        print(f"  px{px:>9} " + "".join(f"{m[i]:>9.4f}" for i in idx) + tag)
    print("\n  gap vs control over the whole grid (positive = the rung is genuinely better):")
    for px in pxs:
        g = rep["by_max_pixels"][str(px)]["vs_control_open_macro"]
        print(f"  px{px:>9}  max {g['max_gap']:+.4f}  mean {g['mean_gap']:+.4f}  "
              f"min {g['min_gap']:+.4f}  ahead at {g['n_grid_points_arm_ahead']}/101 points, "
              f"behind at {g['n_grid_points_control_ahead']}")
    print(f"\nwrote {PARTS}/isoesc.json")


if __name__ == "__main__":
    main()
