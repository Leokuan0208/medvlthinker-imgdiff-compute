#!/usr/bin/env python3
"""ATTACK 3 follow-up: does the VRAM escape hatch survive its own accuracy measurement?

PART4 of sevenb_only_frontier_2026-08-12.json established the bf16 VRAM cliff (any escalation
forces a second card) and noted that a 4-bit 32B removes it (19.53 GiB resident, MEASURED) --
but flagged the quantised model's ACCURACY as OPEN, because that run had crashed.

It has since completed.  This script asks the only question that matters for the direction:

  take PART3's minimum-escalation-that-ties policy, and swap the bf16 strong leg for the NF4
  strong leg on the cells where NF4 accuracy is MEASURED.  Does it still tie?

Only 3 of the 8 cells have a measured NF4 accuracy (SLAKE_closed, VQA_RAD_closed,
PATH_VQA_closed).  The other 5 are left on bf16 and that is stated, not hidden -- so the
answer is an UPPER BOUND on the NF4 policy's accuracy (the unmeasured cells cannot help it,
they are held at their bf16 values).

No GPU, no new inference, MedEvalKit untouched.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import json

import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
ART = ROOT + "/results/cascade_methods/artifacts"
PARTS = ART + "/_frontier_verify_parts"
MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
CELLS = MCQ + OPEN
SEED = 20260812
NBOOT = 10000
TOL = -0.0029

# the 3 cells with a measured NF4 accuracy, and the dataset dump each comes from
MEASURED = {"SLAKE_closed": "SLAKE", "VQA_RAD_closed": "VQA_RAD", "PATH_VQA_closed": "PATH_VQA"}


def cell_of(ds, row):
    """MedEvalKit's own closed/open split (shrink_quant_acc_analyze.py:66, unchanged)."""
    if ds == "SLAKE":
        return "SLAKE_%s" % ("open" if row.get("answer_type") == "OPEN" else "closed")
    if ds in ("PATH_VQA", "VQA_RAD"):
        return "%s_%s" % (ds, "closed" if str(row.get("answer", "")).strip().lower()
                          in ("yes", "no") else "open")
    return ds


def quant_cell_vec(arm, ds, cell):
    rows = json.load(open(f"{ROOT}/ckpts/shrink_quant/{arm}/{ds}/results.json"))
    return np.array([float(bool(r["correct"])) for r in rows if cell_of(ds, r) == cell])


def main():
    V = np.load(f"{ART}/_selector_rerun_parts/vec_disjoint.npz", allow_pickle=True)
    vec = {}
    for k in V.files:
        c, a = k.split("|")
        vec.setdefault(c, {})[a] = V[k].astype(np.float64)
    NS = {c: len(vec[c]["always_7b"]) for c in CELLS}
    ok7 = {c: vec[c]["always_7b"] for c in CELLS}
    ok32 = {c: vec[c]["always_32b_direct"] for c in CELLS}

    cache = np.load(f"{ART}/_sevenb_frontier_parts/loaded.npz", allow_pickle=True)
    best7b = {c: ok7[c] for c in MCQ}
    for c in OPEN:
        best7b[c] = cache[f"ens8|{c}"].astype(np.float64)

    out = {
        "title": "ATTACK 3 follow-up -- does the 4-bit VRAM escape hatch survive its own "
                 "accuracy measurement?",
        "date": "2026-08-12",
        "reproduce": "python3 src/cascade_methods/frontier_nf4_tie.py",
        "no_gpu": True, "no_new_inference": True, "no_fabricated_numbers": True,
        "numerics": {"OMP_NUM_THREADS": 1, "nboot": NBOOT, "seed": SEED,
                     "tf32": "not applicable -- numpy on stored 0/1 correctness vectors"},
        "sources": {
            "nf4_and_bf16_per_item": "ckpts/shrink_quant/{nf4,bf16}/<DS>/results.json "
                                     "(MedEvalKit's own unmodified cal_metrics 'correct')",
            "published_cells": "_selector_rerun_parts/vec_disjoint.npz",
            "policy": "sevenb_only_frontier_2026-08-12.json PART3 minimum_escalation_that_ties",
        },
    }

    # ---------- NULL TEST: the bf16 control reproduces the published cells ----------
    nulls, mx = {}, 0.0
    nf4v, bf16v = {}, {}
    for cell, ds in MEASURED.items():
        nf4v[cell] = quant_cell_vec("nf4", ds, cell)
        bf16v[cell] = quant_cell_vec("bf16", ds, cell)
        dev = abs(float(bf16v[cell].mean()) - float(ok32[cell].mean()))
        mx = max(mx, dev)
        nulls[cell] = {
            "n_nf4": int(len(nf4v[cell])), "n_bf16": int(len(bf16v[cell])),
            "n_published": NS[cell],
            "length_match": bool(len(nf4v[cell]) == len(bf16v[cell]) == NS[cell]),
            "bf16_control_acc": round(float(bf16v[cell].mean()), 6),
            "published_always_32b_direct_acc": round(float(ok32[cell].mean()), 6),
            "serving_stack_abs_deviation": round(dev, 6),
        }
    out["NULL_TEST_bf16_control_vs_published"] = {
        "what": "the HF bf16 32B arm re-scored here vs the published vLLM always-32B-direct "
                "cells.  Identical weights, items, prompts, greedy decoding -- any difference "
                "is the serving stack, and it BOUNDS how much of the swap below is artifact.",
        "per_cell": nulls,
        "max_abs_deviation": round(mx, 6),
        "PASSED": bool(all(v["length_match"] for v in nulls.values()) and mx < 0.01),
        "note": "the deviation does NOT contaminate the NF4-minus-bf16 delta, where it cancels; "
                "it is reported because the SWAP below mixes an HF-served NF4 cell into a "
                "vLLM-served macro.",
    }

    # ---------- the policy ----------
    esc = A3_esc = json.load(open(f"{ART}/sevenb_only_frontier_2026-08-12.json"))[
        "PART3_minimum_32B_frontier"]["a_exact_cell_subset_enumeration"][
        "minimum_escalation_that_ties"]["cells_to_32B"]

    rng = np.random.default_rng(SEED)
    idx = {c: rng.integers(0, NS[c], size=(NBOOT, NS[c])) for c in CELLS}

    def boot(a, b):
        d = np.zeros(NBOOT)
        for c in CELLS:
            d += (a[c][idx[c]].mean(axis=1) - b[c][idx[c]].mean(axis=1)) / len(CELLS)
        pt = float(np.mean([a[c].mean() for c in CELLS]) - np.mean([b[c].mean() for c in CELLS]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": round(pt, 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
                "ties_at_tol": bool(lo >= TOL)}

    # arm 1: the shipped policy, bf16 strong leg (the PART3 tie)
    pol_bf16 = {c: (ok32[c] if c in esc else best7b[c]) for c in CELLS}
    # arm 2: same policy, NF4 strong leg on the 3 cells where NF4 is MEASURED
    pol_nf4 = {}
    swapped = []
    for c in CELLS:
        if c in esc and c in MEASURED:
            pol_nf4[c] = nf4v[c]
            swapped.append(c)
        else:
            pol_nf4[c] = pol_bf16[c]

    out["policy"] = {
        "cells_escalated_to_the_strong_leg": esc,
        "cells_where_the_strong_leg_is_NF4_in_arm_2": swapped,
        "cells_left_on_bf16_because_NF4_IS_UNMEASURED_THERE": [
            c for c in esc if c not in MEASURED],
        "cells_on_the_7B_only_arm": [c for c in CELLS if c not in esc],
        "this_makes_arm_2_an_UPPER_BOUND": (
            "the 5 cells without a measured NF4 accuracy are held at their bf16 values, so a "
            "fully-NF4 strong leg can only do WORSE than arm 2, not better -- unless NF4 "
            "happens to beat bf16 on those cells, which is unmeasured either way."),
    }

    out["RESULT"] = {
        "arm1_bf16_strong_leg": {
            "macro_acc": round(float(np.mean([pol_bf16[c].mean() for c in CELLS])), 6),
            "vs_always_32b_direct": boot(pol_bf16, ok32),
        },
        "arm2_NF4_strong_leg_on_the_3_measured_cells": {
            "macro_acc": round(float(np.mean([pol_nf4[c].mean() for c in CELLS])), 6),
            "vs_always_32b_direct": boot(pol_nf4, ok32),
        },
        "paired_delta_arm2_minus_arm1": boot(pol_nf4, pol_bf16),
    }

    a1 = out["RESULT"]["arm1_bf16_strong_leg"]["vs_always_32b_direct"]
    a2 = out["RESULT"]["arm2_NF4_strong_leg_on_the_3_measured_cells"]["vs_always_32b_direct"]
    out["VERDICT"] = {
        "arm1_ties": a1["ties_at_tol"],
        "arm2_ties": a2["ties_at_tol"],
        "tie_definition": "95%% CI lower bound of (policy - always-32B-direct) on the 8-cell "
                          "macro >= %.4f (the round's pre-registered tolerance)" % TOL,
        "one_line": (
            "the bf16 policy ties (lo=%.6f >= %.4f); swapping in the NF4 strong leg on the 3 "
            "cells where NF4 accuracy is MEASURED moves the macro by %.6f and takes the CI "
            "lower bound to %.6f, which %s the pre-registered tie tolerance."
            % (a1["lo"], TOL, out["RESULT"]["paired_delta_arm2_minus_arm1"]["delta"], a2["lo"],
               "STILL CLEARS" if a2["ties_at_tol"] else "BREAKS")),
        "what_this_does_NOT_settle": (
            "5 of 8 cells have no measured NF4 accuracy -- PMC_VQA, MedXpertQA-MM and all "
            "three OPEN cells, which together carry 5/8 of the macro weight.  The open cells "
            "are the ones the whole direction rests on and they were scored without the LLM "
            "judge, so no open-cell NF4 accuracy exists at any quality."),
    }

    os.makedirs(PARTS, exist_ok=True)
    json.dump(out, open(f"{PARTS}/nf4_tie.json", "w"), indent=1)
    print(json.dumps(out["NULL_TEST_bf16_control_vs_published"], indent=1))
    print(json.dumps(out["RESULT"], indent=1))
    print(json.dumps(out["VERDICT"], indent=1))
    print("wrote", f"{PARTS}/nf4_tie.json")


if __name__ == "__main__":
    main()
