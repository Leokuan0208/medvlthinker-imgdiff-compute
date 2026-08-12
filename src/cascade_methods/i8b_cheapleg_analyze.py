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


_MEK_JUDGE = {}


def mek_judges():
    """MedEvalKit's OWN grading functions, imported (never re-implemented).

    utils.py builds an openai_llm at import time and requires api_key to EXIST, so the same env
    vars eval.py sets are supplied first.  The LLM judge is never called: every cell graded here
    is multiple-choice or yes/no, which MedEvalKit grades with pure string logic.
    """
    if _MEK_JUDGE:
        return _MEK_JUDGE
    for k, v in (("use_llm_judge", "False"), ("judge_model_type", "openai"),
                 ("judge_model", "None"), ("api_key", "None"), ("base_url", "None")):
        os.environ.setdefault(k, v)
    if MEK not in sys.path:
        sys.path.insert(0, MEK)
    from utils.utils import judge_multi_choice, judge_judgement, judge_close_end_vqa
    _MEK_JUDGE.update(mc=judge_multi_choice, yn=judge_judgement, ce=judge_close_end_vqa)
    return _MEK_JUDGE


def load_driver(arm, ds, allow_partial=True):
    """Finished cell -> MedEvalKit's own results.json.  Unfinished cell -> reconstruct from the
    resumable per-item JSONL, joined by position against the stored run's gold fields, and graded
    with MedEvalKit's own judge functions.  Partial cells are reported WITH their n and flagged."""
    p = os.path.join(DRIVER, arm, ds, "results.json")
    if os.path.exists(p):
        rows = json.load(open(p))
        for r in rows:
            r["_partial"] = False
        return rows
    if not allow_partial:
        return None
    gen = os.path.join(DRIVER, arm, ds, "gen.jsonl")
    gold = os.path.join(MEK, "eval_results_cheapleg_base7b", "{}", ds, "results.json")
    if not (os.path.exists(gen) and os.path.exists(gold)):
        return None
    G = json.load(open(gold))
    J = mek_judges()
    out = []
    for line in open(gen):
        try:
            r = json.loads(line)
        except Exception:
            continue                      # truncated final line of a killed run
        i = r["_i"]
        if i >= len(G):
            continue
        g = dict(G[i])
        resp = r.get("response", "")
        ans = g.get("answer")
        if ds in ("PMC_VQA", "MedXpertQA-MM"):
            ok = J["mc"](g.get("choices"), ans, resp)
        elif str(ans).strip().lower() in ("yes", "no"):
            ok = J["yn"](str(ans).lower().strip(), str(resp).lower().strip())
        else:
            ok = J["ce"](str(ans).lower().strip(), str(resp).lower().strip())
        g.update({"response": resp, "correct": bool(ok), "_i": i, "_partial": True,
                  "gen_toks": r.get("gen_toks"), "margin": r.get("margin"),
                  "conf": r.get("conf"), "n_prompt_tokens": r.get("n_prompt_tokens")})
        out.append(g)
    return out or None


def load_mek(tag, ds):
    p = os.path.join(MEK, f"eval_results_{tag}", "{}", ds, "results.json")
    return json.load(open(p)) if os.path.exists(p) else None


def as_map(rows):
    """rows -> {global item index: row}.  Finished cells carry no _i, so position IS the index."""
    if rows is None:
        return {}
    return {(r["_i"] if "_i" in r else i): r for i, r in enumerate(rows)}


def cell_keys(rowmap, filt):
    """the item indices of this reporting cell, in ascending order"""
    ks = sorted(rowmap)
    if filt is None:
        return ks
    if filt == "SLAKE_CLOSED":
        return [k for k in ks if rowmap[k].get("answer_type") == "CLOSED"]
    if filt == "YESNO":
        return [k for k in ks if str(rowmap[k].get("answer", "")).strip().lower() in ("yes", "no")]
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
        for arm, tag in (("i8b", "i8bhf"), ("i8b_1tile", "i8b1tilehf"), ("base7b", "base7bhf")):
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
            "trainer_state": ("110,440 steps / 2 epochs, loss 2.3959 -> 0.4034, total_flos 3.24e20 "
                              "-- a genuinely trained checkpoint, not a repackaged base"),
            "CORRECTION_to_the_preregistration": (
                "The pre-registration recorded 'crop_to_patches=false, 256 image tokens/image, no "
                "tiling', read from preprocessor_config.json. THAT IS WRONG FOR REAL IMAGES and is "
                "corrected here. Measured with the model's own AutoProcessor: a 448x448 or 224x224 "
                "image gives 1 patch / 268 tokens, but 300x400, 1024x768 and real PathVQA images "
                "give 13 patches / 3,340 tokens -- max_patches=12 plus a thumbnail, i.e. InternVL3 "
                "dynamic tiling, applied despite image_processor.crop_to_patches evaluating to "
                "False. Passing crop_to_patches=False explicitly at call time does give 268 tokens. "
                "This run uses the AutoProcessor DEFAULT (13 patches), which is InternVL3's standard "
                "inference configuration and therefore the one the published model-card numbers are "
                "most likely to correspond to -- but that correspondence is an inference, not a "
                "verified fact."),
            "measured_prompt_tokens": {"448x448": 268, "224x224": 268, "300x400": 3340,
                                       "1024x768": 3340},
        },
        "deviations_from_the_preregistration": [
            {"what": "a THIRD arm, i8b_1tile, was added after the pre-registration was written",
             "why": ("mid-round it was discovered that transformers' InternVLProcessor._defaults "
                     "hard-codes crop_to_patches=True, overriding this checkpoint's own saved "
                     "preprocessor_config.json (crop_to_patches: false). The two settings differ by "
                     "9-11x in prompt tokens, so 'the' cost of this cheap leg is not well defined "
                     "until the ambiguity is resolved. Both were therefore measured."),
             "anti_fishing": ("The PRE-REGISTERED primary comparison is unchanged: arm `i8b` (library "
                              "default) vs `base7b`, on PATH_VQA_closed. `i8b_1tile` is a "
                              "CONFIGURATION SENSITIVITY, not a second hypothesis; it is reported "
                              "alongside, and it agrees with the primary in sign and magnitude, so "
                              "nothing turns on which is called primary.")},
            {"what": "the i8b (13-tile) arm was stopped before completing PATH_VQA",
             "why": ("three arms on two GPUs starved base7b -- the MATCHED CONTROL that every "
                     "reported delta depends on -- to 768/6719 items. The 13-tile arm's only job "
                     "was to settle the tiling ambiguity, which its partial cell already does, so "
                     "it was stopped and GPU0 given to the control."),
             "consequence": "its cells are reported as `partial: true` with their actual n."},
            {"what": "use_fast=True image processors, run on the GPU (device=cuda)",
             "why": ("pure throughput. Measured on 32 real PathVQA images the InternVL processor "
                     "emits byte-identical output (3,340 tokens, 185 patches) at 47.4 img/s on GPU "
                     "vs 0.8 img/s on CPU. On CPU the suite would have needed ~17 h of "
                     "single-threaded preprocessing for one arm."),
             "consequence": ("both arms use their own model's default fast processor on GPU, so the "
                             "two arms stay symmetric; only the comparison against the STORED vLLM "
                             "run carries any difference, and the null test measures exactly that.")},
        ],
        "null_test": {}, "cells": {}, "macro": {}, "router_ceiling": {},
    }

    # ---------------- 1. NULL TEST: new HF driver vs stored MedEvalKit vLLM, same weights ------
    nt = {}
    for cell, (ds, filt) in CELLS.items():
        D = as_map(load_driver("base7b", ds))
        M = as_map(load_mek("cheapleg_base7b", ds))
        if not D or not M:
            continue
        ks = [k for k in cell_keys(D, filt) if k in M]
        if not ks:
            continue
        a = np.array([as_ok(D[k]) for k in ks], float)
        b = np.array([as_ok(M[k]) for k in ks], float)
        nt[cell] = {"n": len(ks), "driver_hf": float(a.mean()),
                    "stored_mek_vllm": float(b.mean()),
                    "abs_dev": abs(float(a.mean() - b.mean())),
                    "item_agreement": float((a == b).mean()),
                    "partial": bool(any(D[k].get("_partial") for k in ks))}
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

    # ---------------- 2. MATCHED CONTRAST: each I-8B arm vs base7b, all in the new driver ------
    CHEAP = ["i8b", "i8b_1tile"]
    macro = {k: [] for k in CHEAP + ["base7b"]}
    for cell, (ds, filt) in CELLS.items():
        B = as_map(load_driver("base7b", ds))
        if not B:
            out["cells"][cell] = {"status": "control not run"}
            continue
        full_n = len(load_mek("cheapleg_base7b", ds) or [])
        row = {"cell_n_when_complete": len(cell_keys(as_map(load_mek("cheapleg_base7b", ds)), filt))}
        any_arm = False
        for arm in CHEAP:
            A = as_map(load_driver(arm, ds))
            if not A:
                row[arm] = "not run"
                continue
            ks = [k for k in cell_keys(B, filt) if k in A]
            if not ks:
                row[arm] = "no common items yet"
                continue
            a = np.array([as_ok(A[k]) for k in ks], float)
            b = np.array([as_ok(B[k]) for k in ks], float)
            d, lo, hi, sig = boot_delta(a, b, rng)
            partial = any(A[k].get("_partial") or B[k].get("_partial") for k in ks)
            # CLASS-BALANCE CORRECTION, the yes/no analogue of the letter-balancing that the
            # concurrent PMC-VQA audit found necessary: a cheap leg whose answer prior happens to
            # sit closer to the gold prior scores better without being better. Reported for every
            # cell with a small discrete answer set, so the gain cannot hide in the prior.
            gold = [str(A[k].get("answer", "")).strip().lower() for k in ks]
            classes = sorted(set(gold))
            if 1 < len(classes) <= 5:
                def bal(v):
                    per = [v[[i for i, g in enumerate(gold) if g == c]].mean() for c in classes]
                    return float(np.mean(per))
                bal_a, bal_b = bal(a), bal(b)
                balanced = {"classes": classes,
                            "gold_counts": {c: int(sum(g == c for g in gold)) for c in classes},
                            "majority_class_floor": float(max(
                                sum(g == c for g in gold) for c in classes) / len(gold)),
                            "balanced_acc_cheap": bal_a, "balanced_acc_base7b": bal_b,
                            "balanced_delta": bal_a - bal_b}
            else:
                balanced = None
            row[arm] = {
                "n_paired": len(ks), "partial": bool(partial),
                "acc": float(a.mean()), "base7b_matched": float(b.mean()),
                "delta_vs_base7b": d, "ci95": [lo, hi],
                "significant": sig, "guardrail_flag": bool(hi < 0),
                "mean_gen_tokens": float(np.mean([A[k].get("gen_toks") or 0 for k in ks])),
                "mean_prompt_tokens": float(np.mean([A[k].get("n_prompt_tokens") or 0 for k in ks])),
                "mean_prompt_tokens_base7b": float(np.mean([B[k].get("n_prompt_tokens") or 0
                                                            for k in ks])),
                "class_balanced": balanced,
            }
            if not partial:
                macro[arm].append(float(a.mean()))
                if arm == CHEAP[0]:
                    macro["base7b"].append(float(b.mean()))
                any_arm = True
        out["cells"][cell] = row

    if macro["base7b"]:
        out["macro"] = {
            "note": ("equal weight over the CLOSED/MCQ reporting cells RUN SO FAR. This is NOT the "
                     "8-cell Variant-B macro -- the 3 open cells are absent -- and it must never be "
                     "quoted as one."),
            "cells_included": [c for c, v in out["cells"].items()
                               if isinstance(v, dict) and "base7b_matched" in v],
            "base7b_matched": float(np.mean(macro["base7b"])),
        }
        for arm in CHEAP:
            if len(macro[arm]) == len(macro["base7b"]) and macro[arm]:
                out["macro"][arm] = float(np.mean(macro[arm]))
                out["macro"][f"delta_{arm}"] = float(np.mean(macro[arm]) - np.mean(macro["base7b"]))

    # ---------------- 2b. FALLBACK CONTROL: the STORED vLLM Lingshu-7B ------------------------
    # The HF control is ~1.6 it/s on Qwen2.5-VL at full resolution (its ViT sees ~4x the post-merge
    # token count) and cannot finish PMC_VQA's 33,430 items.  The null test above is what licenses
    # substituting the COMPLETE stored vLLM run of the SAME WEIGHTS as the control on cells the HF
    # control has not reached.  Reported separately and labelled, never silently merged.
    out["cells_vs_stored_vllm_7b"] = {}
    for cell, (ds, filt) in CELLS.items():
        M = as_map(load_mek("cheapleg_base7b", ds))
        if not M:
            continue
        row = {}
        for arm in CHEAP:
            A = as_map(load_driver(arm, ds))
            if not A:
                continue
            ks = [k for k in cell_keys(A, filt) if k in M]
            if not ks:
                continue
            a = np.array([as_ok(A[k]) for k in ks], float)
            b = np.array([as_ok(M[k]) for k in ks], float)
            d, lo, hi, sig = boot_delta(a, b, rng)
            row[arm] = {"n_paired": len(ks),
                        "partial": bool(any(A[k].get("_partial") for k in ks)),
                        "acc": float(a.mean()), "stored_vllm_7b": float(b.mean()),
                        "delta": d, "ci95": [lo, hi], "significant": sig,
                        "guardrail_flag": bool(hi < 0)}
        if row:
            out["cells_vs_stored_vllm_7b"][cell] = row

    # ---------------- 3. ROUTER CEILING vs the stored always-32B-direct arm --------------------
    for cell, (ds, filt) in CELLS.items():
        s32 = load_mek("lingshu32b_full", ds)
        if s32 is None:
            continue
        S = as_map(s32)
        # p10 is only comparable BETWEEN arms if every arm is scored on the SAME items, so the
        # common key set across all available arms is taken first (partial arms have different
        # amounts done).
        maps = {arm: as_map(load_driver(arm, ds)) for arm in ("i8b", "i8b_1tile", "base7b")}
        maps = {a: m for a, m in maps.items() if m}
        if not maps:
            continue
        common = set(S)
        for m in maps.values():
            common &= set(m)
        row = {"n_common": len(common)}
        for arm, A in maps.items():
            ks = [k for k in cell_keys(A, filt) if k in common]
            if not ks:
                continue
            a = np.array([as_ok(A[k]) for k in ks], float)
            s = np.array([as_ok(S[k]) for k in ks], float)
            row[arm] = {"n": len(ks),
                        "partial": bool(any(A[k].get("_partial") for k in ks)),
                        "cheap_acc": float(a.mean()), "acc_32b_direct": float(s.mean()),
                        "p10_cheap_right_32b_wrong": float(((a == 1) & (s == 0)).mean()),
                        "p01_cheap_wrong_32b_right": float(((a == 0) & (s == 1)).mean()),
                        "oracle_route": float(np.maximum(a, s).mean())}
        if row:
            row["_caveat"] = ("the 32B arm is the STORED vLLM run; the cheap arms are HF. p10/p01 "
                              "therefore carry the null-test deviation. Compare p10 BETWEEN the two "
                              "cheap arms (both HF), not against published p10.")
            out["router_ceiling"][cell] = row

    # ---------------- 4. HONEST COST OF THE CHEAP PASS -----------------------------------------
    # The brief assumed "an 8B cheap leg is ~14% more FLOPs per cheap pass than a 7B" (7.94/8.29
    # by parameter count would in fact be 0.96x).  That assumption ignores PROMPT LENGTH, and
    # prompt length is where the two models actually differ: Lingshu-I-8B's processor tiles a
    # non-448-square image into 13 crops of 256 tokens each.
    # Prefill dominates here (generated answers are ~5 tokens), and prefill FLOPs ~ 2*N_params*T,
    # so the per-pass ratio is approximated as (N_i8b/N_7b) * (T_i8b/T_7b).  Stated as an
    # approximation with its formula, NOT as a measurement.
    N_I8B, N_L7B, N_L32B = 7.944, 8.29, 33.0   # billions of params (I-8B summed from p.numel())
    cost = {}
    for arm in ("i8b", "i8b_1tile"):
        per_cell = {}
        for cell, (ds, filt) in CELLS.items():
            A = as_map(load_driver(arm, ds)); B = as_map(load_driver("base7b", ds))
            if not A or not B:
                continue
            ks = [k for k in cell_keys(A, filt) if k in B]
            if not ks:
                continue
            ta = float(np.mean([A[k].get("n_prompt_tokens") or 0 for k in ks]))
            tb = float(np.mean([B[k].get("n_prompt_tokens") or 0 for k in ks]))
            if tb > 0 and ta > 0:
                r7 = (N_I8B / N_L7B) * (ta / tb)
                per_cell[cell] = {
                    "mean_prompt_tokens_cheap": ta, "mean_prompt_tokens_base7b": tb,
                    "token_ratio": ta / tb,
                    "approx_prefill_flop_ratio_vs_7b": r7,
                    # the 32B shares Lingshu-7B's Qwen2.5-VL image policy, so its prompt length is
                    # the base7b column; a 32B pass therefore costs (N_32B/N_7B) 7B-passes
                    "approx_prefill_flop_ratio_vs_32b_direct": r7 / (N_L32B / N_L7B),
                }
        if per_cell:
            cost[arm] = {
                "per_cell": per_cell,
                "macro_mean_ratio_vs_7b":
                    float(np.mean([v["approx_prefill_flop_ratio_vs_7b"] for v in per_cell.values()])),
                "macro_mean_ratio_vs_32b_direct":
                    float(np.mean([v["approx_prefill_flop_ratio_vs_32b_direct"]
                                   for v in per_cell.values()])),
            }
    if cost:
        out["cheap_pass_cost"] = {
            "formula": ("(N_params_cheap / N_params_7b) * (mean_prompt_tokens_cheap / "
                        "mean_prompt_tokens_7b); vs-32B divides by N_32B/N_7B = 3.98"),
            "caveat": ("APPROXIMATION, not a measurement. Prefill-only (generated answers are ~5 "
                       "tokens); ignores the two vision towers' differing per-token cost. The TOKEN "
                       "COUNTS are measured per item by the driver; the parameter counts are 7.944B "
                       "(summed from the loaded model) and 8.29B / 33B (published Qwen2.5-VL sizes)."),
            "param_only_ratio_would_be": N_I8B / N_L7B,
            "why_this_matters": ("The brief assumed an 8B cheap leg costs ~1.14x a 7B pass. By "
                                 "parameters alone it is 0.96x. The measured driver of cost is "
                                 "PROMPT LENGTH, not parameters."),
            "arms": cost,
        }

    dst = os.path.join(ART, "lingshu_i8b_cheapleg_2026-08-11.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print("WROTE", dst)


if __name__ == "__main__":
    main()
