#!/usr/bin/env python3
"""
i8b_cheapleg_analyze.py -- ATTACK A analysis: Lingshu-I-8B vs a MATCHED Lingshu-7B control,
plus the driver's null test against the stored MedEvalKit vLLM run.

Three things, in this order:

  1. NULL TEST.  The new HF driver's Lingshu-7B arm vs MedEvalKit/eval_results_cheapleg_base7b
     (same weights, same items, same prompts, vLLM tp=1).  Reports max abs per-cell deviation.
     This quantifies how much of any I-8B-vs-7B delta could be a serving-stack artifact.
     PASS threshold declared in the pre-registration: <= 0.02 on the closed/MCQ cells.

  2. MATCHED CONTRAST.  I-8B minus base7b, BOTH from the new driver, per reporting cell,
     paired item bootstrap nboot=10000.  This is the number that decides KILL vs GO.

  3. ROUTER CEILING.  p10 = P(cheap correct AND 32B-direct wrong) per cell, against the stored
     always-32B-direct arm (eval_results_lingshu32b_full).  A stronger cheap leg should raise it.
     ALSO p01 = P(cheap wrong AND 32B right) = the escalation headroom the gate has to find.

    python3 src/cascade_methods/i8b_cheapleg_analyze.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEK = os.path.join(ROOT, "MedEvalKit")
DRIVER = os.path.join(ROOT, "ckpts/i8b_cheapleg")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
NBOOT = 10000
SEED = 20260812

# reporting cell -> (MedEvalKit dataset, row filter)
CELLS = {
    "PMC_VQA":         ("PMC_VQA", None),
    "SLAKE_closed":    ("SLAKE", "SLAKE_CLOSED"),
    "VQA_RAD_closed":  ("VQA_RAD", "YESNO"),
    "PATH_VQA_closed": ("PATH_VQA", "YESNO"),
    "MedXpertQA-MM":   ("MedXpertQA-MM", None),
}


def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))


def load_driver(arm, ds):
    p = os.path.join(DRIVER, arm, ds, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def load_mek(tag, ds):
    p = os.path.join(MEK, f"eval_results_{tag}", "{}", ds, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def keep_idx(rows, filt):
    if filt is None:
        return list(range(len(rows)))
    if filt == "SLAKE_CLOSED":
        return [i for i, r in enumerate(rows) if r.get("answer_type") == "CLOSED"]
    if filt == "YESNO":
        return [i for i, r in enumerate(rows)
                if str(r.get("answer", "")).strip().lower() in ("yes", "no")]
    raise ValueError(filt)


def boot_delta(a, b, rng, nboot=NBOOT):
    """paired item bootstrap of mean(a) - mean(b) on the SAME items"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    d = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(nboot, n))
    ds = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return d, float(lo), float(hi), bool(lo > 0 or hi < 0)


def mirror_to_mek(arm, tag):
    """Expose a driver arm in MedEvalKit's eval_results_<tag>/{}/<DS>/ layout, by SYMLINK.

    cheapleg_macro.py swaps the cheap leg by pointing integrated_method.MEK at a shadow directory
    whose eval_results_lingshu7b_full is the arm's own output.  It needs the arm to look like a
    normal MedEvalKit run.  Nothing is copied and no MedEvalKit CODE is touched -- only a new
    eval_results_* output directory is created, exactly as every runner in runners/ does.
    """
    dst_root = os.path.join(MEK, f"eval_results_{tag}", "{}")
    os.makedirs(dst_root, exist_ok=True)
    made = []
    for ds in ("PATH_VQA", "SLAKE", "VQA_RAD", "MedXpertQA-MM", "PMC_VQA"):
        src = os.path.join(DRIVER, arm, ds)
        dst = os.path.join(dst_root, ds)
        if not os.path.exists(os.path.join(src, "results.json")):
            continue
        if os.path.islink(dst):
            os.unlink(dst)
        if not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
            made.append(ds)
    return dst_root, made


def main():
    rng = np.random.default_rng(SEED)
    if "--mirror" in sys.argv:
        for arm, tag in (("i8b", "i8bhf"), ("base7b", "base7bhf")):
            root, made = mirror_to_mek(arm, tag)
            print(f"mirrored {arm} -> {root}  new: {made}")
        return
    out = {
        "title": "ATTACK A -- Lingshu-I-8B as the cascade's cheap leg",
        "preregistration": "results/cascade_methods/artifacts/lingshu_i8b_cheapleg_2026-08-11_preregistration.json",
        "driver": "src/cascade_methods/i8b_cheapleg_eval.py",
        "why_not_vllm": ("vLLM 0.9.0.1 registers only InternVLChatModel; Lingshu-I-8B is the HF-native "
                         "InternVLForConditionalGeneration port and raises `limit_mm_per_prompt` is only "
                         "supported for multimodal models (logs/i8b_vllm_try.log). MedEvalKit unmodified; "
                         "its dataset classes, prompts and cal_metrics are imported verbatim."),
        "model_verified": {
            "path": ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-I-8B/"
                     "snapshots/b004bfc0554d90bd44baedf4de08c361e71ef017"),
            "architecture": "InternVLForConditionalGeneration",
            "n_params_B": 7.94437376,
            "image_tokens_per_image": 256,
            "crop_to_patches": False,
        },
        "null_test": {}, "cells": {}, "macro": {}, "router_ceiling": {},
    }

    # ---------------- 1. NULL TEST: new HF driver vs stored MedEvalKit vLLM, same weights ------
    nt = {}
    for cell, (ds, filt) in CELLS.items():
        drv = load_driver("base7b", ds)
        mek = load_mek("cheapleg_base7b", ds)
        if drv is None or mek is None:
            continue
        n = min(len(drv), len(mek))
        di = keep_idx(drv[:n], filt); mi = keep_idx(mek[:n], filt)
        if di != mi:
            nt[cell] = {"error": "row filter disagrees between arms", "n_driver": len(di), "n_mek": len(mi)}
            continue
        a = np.array([as_ok(drv[i]) for i in di], float)
        b = np.array([as_ok(mek[i]) for i in di], float)
        nt[cell] = {"n": len(di), "driver_hf": float(a.mean()), "stored_mek_vllm": float(b.mean()),
                    "abs_dev": abs(float(a.mean() - b.mean())),
                    "item_agreement": float((a == b).mean())}
    if nt:
        mx = max(v["abs_dev"] for v in nt.values() if "abs_dev" in v)
        out["null_test"] = {
            "what": ("new HF driver running Lingshu-7B vs MedEvalKit/eval_results_cheapleg_base7b "
                     "(identical weights, items, prompts; vLLM tp=1)"),
            "per_cell": nt, "max_abs_deviation": mx,
            "pass_threshold_preregistered": 0.02,
            "verdict": ("PASS" if mx <= 0.02 else ("CAVEAT" if mx <= 0.05 else "FAIL")),
            "note": ("This is a SERVING-STACK deviation, not a bug: HF eager attention vs vLLM paged "
                     "attention on the same weights. It bounds how much of any cross-harness statement "
                     "is artifact. The I-8B vs base7b contrast below is WITHIN one driver and is "
                     "unaffected by it."),
        }

    # ---------------- 2. MATCHED CONTRAST: I-8B vs base7b, both in the new driver --------------
    macro_i, macro_b = [], []
    for cell, (ds, filt) in CELLS.items():
        A = load_driver("i8b", ds); B = load_driver("base7b", ds)
        if A is None or B is None:
            out["cells"][cell] = {"status": "not run"}
            continue
        n = min(len(A), len(B))
        ai = keep_idx(A[:n], filt); bi = keep_idx(B[:n], filt)
        assert ai == bi, f"{cell}: item filters disagree ({len(ai)} vs {len(bi)})"
        a = np.array([as_ok(A[i]) for i in ai], float)
        b = np.array([as_ok(B[i]) for i in ai], float)
        d, lo, hi, sig = boot_delta(a, b, rng)
        out["cells"][cell] = {
            "n": len(ai), "i8b": float(a.mean()), "base7b_matched": float(b.mean()),
            "delta": d, "ci95": [lo, hi], "significant": sig,
            "guardrail_flag": bool(hi < 0),
            "mean_gen_tokens_i8b": float(np.mean([A[i].get("gen_toks") or 0 for i in ai])),
            "mean_gen_tokens_base7b": float(np.mean([B[i].get("gen_toks") or 0 for i in ai])),
        }
        macro_i.append(float(a.mean())); macro_b.append(float(b.mean()))

    if macro_i:
        out["macro"] = {
            "note": ("equal weight over the CLOSED/MCQ reporting cells that were run. This is NOT the "
                     "8-cell Variant-B macro until the 3 open cells are added; it is labelled for what "
                     "it is."),
            "n_cells": len(macro_i),
            "i8b": float(np.mean(macro_i)), "base7b_matched": float(np.mean(macro_b)),
            "delta": float(np.mean(macro_i) - np.mean(macro_b)),
        }

    # ---------------- 3. ROUTER CEILING vs the stored always-32B-direct arm --------------------
    for cell, (ds, filt) in CELLS.items():
        s32 = load_mek("lingshu32b_full", ds)
        if s32 is None:
            continue
        row = {}
        for arm in ("i8b", "base7b"):
            A = load_driver(arm, ds)
            if A is None:
                continue
            n = min(len(A), len(s32))
            ai = keep_idx(A[:n], filt)
            a = np.array([as_ok(A[i]) for i in ai], float)
            s = np.array([as_ok(s32[i]) for i in ai], float)
            row[arm] = {"n": len(ai), "cheap_acc": float(a.mean()), "acc_32b_direct": float(s.mean()),
                        "p10_cheap_right_32b_wrong": float(((a == 1) & (s == 0)).mean()),
                        "p01_cheap_wrong_32b_right": float(((a == 0) & (s == 1)).mean()),
                        "oracle_route": float(np.maximum(a, s).mean())}
        if row:
            row["_caveat"] = ("the 32B arm is the STORED vLLM run; the cheap arms are HF. p10/p01 "
                              "therefore carry the null-test deviation. Compare p10 BETWEEN the two "
                              "cheap arms (both HF), not against published p10.")
            out["router_ceiling"][cell] = row

    dst = os.path.join(ART, "lingshu_i8b_cheapleg_2026-08-11.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("WROTE", dst)


if __name__ == "__main__":
    main()
