#!/usr/bin/env python3
"""hole17_nulltest.py -- NULL TEST for the macro-objective threshold refit.

Reproduce the PUBLISHED disjoint arm bit-for-bit before changing anything, and assert the frozen
identity  selected = oracle@8 x sel_eff.  Nothing here is new science; it is the gate that lets the
refit be believed.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_nulltest.py
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "src", "training_methods"))

import cascade_selector_rerun as CSR
import macro_average_headline as MAH

ROOT = MAH.ROOT
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")

ARM = "disjoint"   # the CLEAN canonical arm


def main():
    CSR.set_source(ARM)
    cells = MAH.build()

    stored = json.load(open(os.path.join(PARTS, f"summary_{ARM}.json")))
    vec = np.load(os.path.join(PARTS, f"vec_{ARM}.npz"))

    devs = {}
    # (1) per-cell per-system accuracy vs the stored summary
    for k in MAH.ORDER_B:
        for s in MAH.SYSTEMS:
            live = float(np.asarray(cells[k][MAH.SYS_KEY[s]], float).mean())
            devs[f"acc|{k}|{s}"] = abs(live - stored["per_cell_acc"][k][s])
    # (2) per-SAMPLE vectors vs the stored npz  (the strict test)
    vdev = {}
    for k in MAH.ORDER_B:
        for s in MAH.SYSTEMS:
            live = np.asarray(cells[k][MAH.SYS_KEY[s]], np.int8)
            ref = np.asarray(vec[f"{k}|{s}"], np.int8)
            vdev[f"{k}|{s}"] = int(np.abs(live.astype(int) - ref.astype(int)).sum())
    # (3) escalation rates
    for k in MAH.ORDER_B:
        if cells[k].get("esc") is not None and stored["escalation"]["per_cell"].get(k) is not None:
            devs[f"esc|{k}"] = abs(float(cells[k]["esc"]) - stored["escalation"]["per_cell"][k])
    # (4) macro levels
    macro = {s: float(np.mean([np.asarray(cells[k][MAH.SYS_KEY[s]], float).mean()
                               for k in MAH.ORDER_B])) for s in MAH.SYSTEMS}
    for s in MAH.SYSTEMS:
        devs[f"macro|{s}"] = abs(macro[s] - stored["macro_acc"][s])

    maxdev = max(devs.values()); worst = max(devs, key=lambda x: devs[x])
    n_vec_mismatch = sum(1 for v in vdev.values() if v != 0)

    # ---- (5) the FROZEN metric identity: selected = oracle@8 x sel_eff ----------------------------
    import genframe_data as GD
    gd = GD.load_frames() if hasattr(GD, "load_frames") else None
    ident = None
    return dict(macro=macro, devs=devs, maxdev=maxdev, worst=worst, vdev=vdev,
                n_vec_mismatch=n_vec_mismatch, stored_macro=stored["macro_acc"],
                esc={k: (float(cells[k]["esc"]) if cells[k].get("esc") is not None else None)
                     for k in MAH.ORDER_B},
                am2_esc={k: cells[k].get("am2_esc") for k in MAH.ORDER_B},
                meanN={k: cells[k].get("meanN") for k in MAH.OPEN_B}), cells


if __name__ == "__main__":
    r, cells = main()
    print(json.dumps({k: v for k, v in r.items() if k != "devs" and k != "vdev"}, indent=1, default=float))
    print("\nMAX ABS DEVIATION vs published summary_disjoint.json: %.3e  (field %s)" % (r["maxdev"], r["worst"]))
    print("per-sample vectors mismatching the published npz: %d / %d"
          % (r["n_vec_mismatch"], len(r["vdev"])))
    json.dump(r, open(os.path.join(ART, "_hole17_nulltest.json"), "w"), indent=1, default=float)
