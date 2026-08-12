#!/usr/bin/env python3
"""
shrink_quant_acc_analyze.py -- ATTACK 3, accuracy half of lever 1 (+ the matched-stack half of
lever 2).

Everything here is a PAIRED item-level comparison on identical MedEvalKit items, scored by
MedEvalKit's own unmodified cal_metrics (which writes out_sample["correct"] per item).

Four analyses, in the order they must be read:

  N4  SERVING-STACK NULL TEST.  My bf16 Lingshu-32B arm runs under HF transformers through
      src/cascade_methods/i8b_cheapleg_eval.py's driver.  Every PUBLISHED cell in this project
      came from vLLM 0.9.0.1 (MedEvalKit/eval_results_lingshu32b_full).  Same weights, same
      items, same prompts, same greedy decoding -- so any difference is the serving stack, and
      it BOUNDS every cross-stack statement below.  This is not optional: the concurrent round's
      HF Lingshu-7B arm lands 0.7662 on PATH_VQA_closed where the stored vLLM Lingshu-7B lands
      0.8370, so the driver is known to deviate on at least one cell and the size of that
      deviation for the 32B has to be measured, not assumed.

  A   PRIMARY: NF4 minus bf16, paired, per cell, with a paired item bootstrap.  Both arms are
      the SAME driver, SAME items, SAME batch size, SAME greedy decoding -- only the weight
      representation differs, so this delta is attributable to quantisation alone and is
      IMMUNE to the N4 deviation (it cancels).

  B   Lingshu-I-8B minus bf16 Lingshu-32B, paired, per cell.  The concurrent round (ATTACK A)
      measured I-8B under the SAME HF driver, so this is the matched-stack version of "can an
      8B replace the 32B as the strong leg".  It carries a processor/batch-size caveat, stated.

  C   MACRO PROPAGATION: what the measured per-cell deltas do to the 8-cell macro, with the
      unmeasured cells reported as unmeasured rather than filled in.

    python3 src/cascade_methods/shrink_quant_acc_analyze.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_shrink_parts/quant_acc_paired.json")
STORED_32B = os.path.join(ROOT, "MedEvalKit/eval_results_lingshu32b_full/{}")
I8B_ROOT = os.path.join(ROOT, "ckpts/i8b_cheapleg/i8b_1tile")
QUANT_ROOT = os.path.join(ROOT, "ckpts/shrink_quant")
# The ONLY model this project has run on BOTH serving stacks: Lingshu-7B.  HF (the driver the
# I-8B arm uses) vs vLLM (the stack every published cell uses).  Same weights, same items.
HF_7B = os.path.join(ROOT, "ckpts/i8b_cheapleg/base7b")
STORED_7B = os.path.join(ROOT, "MedEvalKit/eval_results_cheapleg_base7b/{}")

NBOOT = 10000
SEED = 20260812
NOT_MEASURED = "not measured"

# dataset -> the reporting cells it contains, and how MedEvalKit splits them.
# SLAKE splits on answer_type; PATH_VQA and VQA_RAD split on whether the gold answer is yes/no
# (MedEvalKit/utils/PATH_VQA/PATH_VQA.py:113, VQA_RAD.py:99).  MedXpertQA-MM and PMC_VQA are
# single MCQ cells with no split.
DATASETS = {
    "VQA_RAD": ["VQA_RAD_closed", "VQA_RAD_open"],
    "SLAKE": ["SLAKE_closed", "SLAKE_open"],
    "PATH_VQA": ["PATH_VQA_closed", "PATH_VQA_open"],
    "MedXpertQA-MM": ["MedXpertQA-MM"],
    "PMC_VQA": ["PMC_VQA"],
}


def cell_of(ds, row):
    if ds == "SLAKE":
        return "SLAKE_%s" % ("open" if row.get("answer_type") == "OPEN" else "closed")
    if ds in ("PATH_VQA", "VQA_RAD"):
        return "%s_%s" % (ds, "closed" if str(row.get("answer", "")).strip().lower()
                          in ("yes", "no") else "open")
    return ds


def load_results(path):
    """MedEvalKit's own per-item judged rows.  Returns list of dicts in dataset order."""
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def to_cells(ds, rows):
    """{cell: (correct[np.uint8], key[list])} preserving dataset order."""
    out = {}
    for r in rows:
        c = cell_of(ds, r)
        out.setdefault(c, {"correct": [], "key": []})
        out[c]["correct"].append(1 if r.get("correct") else 0)
        # a cheap alignment key that does not depend on the arm: the gold answer + question text
        out[c]["key"].append(str(r.get("question", ""))[:80] + "||" + str(r.get("answer", ""))[:40])
    for c in out:
        out[c]["correct"] = np.asarray(out[c]["correct"], dtype=np.uint8)
    return out


def paired_ci(a, b, rng):
    """CI of mean(a) - mean(b) under a PAIRED item bootstrap (same resample for both arms)."""
    n = len(a)
    if n == 0:
        return None
    d = a.astype(np.float64) - b.astype(np.float64)
    idx = rng.integers(0, n, size=(NBOOT, n))
    boot = d[idx].mean(axis=1)
    return dict(delta=float(d.mean()),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                n=int(n), n_discordant=int((a != b).sum()),
                a_acc=float(a.mean()), b_acc=float(b.mean()))


def compare(name_a, cells_a, name_b, cells_b, rng, note=""):
    res = {}
    for c in sorted(set(cells_a) & set(cells_b)):
        A, B = cells_a[c], cells_b[c]
        if len(A["correct"]) != len(B["correct"]):
            res[c] = dict(status="LENGTH MISMATCH -- not compared",
                          n_a=len(A["correct"]), n_b=len(B["correct"]))
            continue
        mismatch = sum(1 for x, y in zip(A["key"], B["key"]) if x != y)
        r = paired_ci(A["correct"], B["correct"], rng)
        r["alignment_key_mismatches"] = mismatch
        r["alignment_verdict"] = "PASS (item order identical)" if mismatch == 0 else \
            "FAIL -- %d/%d rows disagree on (question, gold); delta NOT trustworthy" % (
                mismatch, len(A["key"]))
        r["arm_a"], r["arm_b"] = name_a, name_b
        res[c] = r
    if note:
        res["_note"] = note
    return res


def main():
    rng = np.random.default_rng(SEED)
    arms = {}

    # -- my two quantisation arms + the stored vLLM control + the concurrent round's I-8B ------
    for arm, root in (("bf16", os.path.join(QUANT_ROOT, "bf16")),
                      ("nf4", os.path.join(QUANT_ROOT, "nf4")),
                      ("stored_vllm_32b", STORED_32B),
                      ("i8b_1tile_hf", I8B_ROOT),
                      ("hf_7b", HF_7B),
                      ("stored_vllm_7b", STORED_7B)):
        got = {}
        for ds in DATASETS:
            rows = load_results(os.path.join(root, ds, "results.json"))
            if rows:
                got.update(to_cells(ds, rows))
        arms[arm] = got

    have = {k: sorted(v) for k, v in arms.items()}

    out = dict(
        title="ATTACK 3 -- paired accuracy analysis: quantised strong leg, and Lingshu-I-8B "
              "against a MATCHED-STACK Lingshu-32B",
        date="2026-08-12",
        nboot=NBOOT, seed=SEED,
        scoring="MedEvalKit's own unmodified cal_metrics; per-item out_sample['correct'].  "
                "The three OPEN cells are scored WITHOUT the LLM judge (use_llm_judge=False), "
                "so their 'correct' is exact-match and is NOT comparable to the project's "
                "published open-cell accuracies, which used the Claude judge.  Open-cell rows "
                "are emitted for completeness and must not be read as accuracy.",
        cells_available_per_arm=have,
        numerics_pins=dict(OMP_NUM_THREADS="1", bootstrap="paired item-level, one shared RNG "
                           "stream drawn in a fixed arm order, nboot=%d, seed=%d" % (NBOOT, SEED),
                           tf32="not applicable -- numpy on stored 0/1 correctness vectors"),
    )

    # N4: serving-stack fidelity of the HF driver, measured on the 32B itself
    out["N4_serving_stack_null_test"] = dict(
        what="my bf16 Lingshu-32B (HF driver) vs the STORED vLLM Lingshu-32B that every "
             "published cell came from (MedEvalKit/eval_results_lingshu32b_full).  Identical "
             "weights, items, prompts, greedy decoding.  Any difference is the serving stack.",
        why_it_matters="it bounds how much of the I-8B-vs-32B comparison (B) is artifact.  It "
                       "does NOT affect (A), where the deviation cancels.",
        per_cell=compare("bf16_hf", arms["bf16"], "stored_vllm", arms["stored_vllm_32b"], rng)
        if arms["bf16"] else NOT_MEASURED,
    )

    # A: the primary paired quantisation delta
    out["A_quantisation_delta_PRIMARY"] = dict(
        what="NF4 minus bf16, same driver, same items, same batch size, same greedy decoding.  "
             "Attributable to weight quantisation alone.",
        per_cell=compare("nf4", arms["nf4"], "bf16", arms["bf16"], rng)
        if (arms["nf4"] and arms["bf16"]) else NOT_MEASURED,
    )

    # N4b: the serving-stack deviation, measured on the ONE model this project ran on BOTH
    # stacks.  This is the available substitute for N4 (which needs the bf16 32B HF arm).
    out["N4b_serving_stack_measured_on_lingshu_7b"] = dict(
        what="Lingshu-7B under the HF driver (ckpts/i8b_cheapleg/base7b) minus Lingshu-7B under "
             "vLLM 0.9.0.1 (MedEvalKit/eval_results_cheapleg_base7b).  IDENTICAL weights, "
             "items, prompts and greedy decoding -- the ONLY difference is the serving stack.",
        why="the Lingshu-I-8B arm is HF and always-32B-direct is vLLM, so the I-8B-vs-32B "
            "comparison is cross-stack.  This quantifies the confound on the same cells.  It is "
            "measured on the 7B, not the 32B, so it TRANSFERS ONLY AS AN INDICATION.",
        direction_note="a NEGATIVE value means the HF driver scores LOWER than vLLM on the same "
                       "weights.  Where that is so, Lingshu-I-8B (measured on HF) is being "
                       "compared against a 32B measured on the MORE GENEROUS stack, so its "
                       "reported advantage is if anything UNDERSTATED.",
        per_cell=compare("hf_7b", arms["hf_7b"], "stored_vllm_7b", arms["stored_vllm_7b"], rng)
        if (arms["hf_7b"] and arms["stored_vllm_7b"]) else NOT_MEASURED,
    )

    # B0: I-8B against the STORED vLLM 32B -- available now, cross-stack, and the cell that
    # decides "can an 8B replace the 32B".
    out["B0_i8b_vs_stored_vllm_32b"] = dict(
        what="Lingshu-I-8B (HF, ATTACK A arm i8b_1tile) minus always-32B-direct as PUBLISHED "
             "(stored vLLM, MedEvalKit/eval_results_lingshu32b_full).",
        caveat="CROSS-STACK: the two arms differ in serving stack as well as in model.  The "
               "N4 null test above measures how large that confound is; read this row only "
               "together with N4.",
        per_cell=compare("i8b_1tile_hf", arms["i8b_1tile_hf"], "stored_vllm_32b",
                         arms["stored_vllm_32b"], rng) if arms["i8b_1tile_hf"] else NOT_MEASURED,
    )

    # VOID CELLS: cells whose responses are empty, i.e. the arm never generated anything.
    # Scoring an all-blank cell does NOT produce an accuracy, it produces whatever MedEvalKit's
    # extraction fallback happens to match.  Detect and quarantine them rather than reporting
    # them as low accuracy.
    void = {}
    for arm, root in (("bf16", os.path.join(QUANT_ROOT, "bf16")),
                      ("nf4", os.path.join(QUANT_ROOT, "nf4")),
                      ("i8b_1tile_hf", I8B_ROOT), ("stored_vllm_32b", STORED_32B),
                      ("hf_7b", HF_7B), ("stored_vllm_7b", STORED_7B)):
        for ds in DATASETS:
            rows = load_results(os.path.join(root, ds, "results.json"))
            if not rows:
                continue
            n_blank = sum(1 for r in rows if not str(r.get("response", "")).strip())
            if n_blank > 0.5 * len(rows):
                void["%s/%s" % (arm, ds)] = dict(
                    n=len(rows), n_blank_responses=n_blank,
                    mean_gen_tokens=float(np.mean([r.get("gen_toks") or 0 for r in rows])),
                    verdict="VOID -- the arm generated nothing on this dataset.  Its reported "
                            "accuracy is an artifact of MedEvalKit scoring blank strings, NOT a "
                            "measurement.  Excluded from every macro below.")
    out["VOID_CELLS"] = void or "none detected"

    # B: I-8B against the matched-stack 32B
    out["B_i8b_vs_matched_32b"] = dict(
        what="Lingshu-I-8B (concurrent round ATTACK A, arm i8b_1tile) minus my bf16 "
             "Lingshu-32B, both under the SAME HF driver.",
        caveat="NOT fully matched: the I-8B arm used AutoProcessor(use_fast=True) at "
               "batch_size=32, my 32B arm the checkpoint's default slow processor at "
               "batch_size=4.  Both are HF transformers, bf16, greedy, and both go through "
               "MedEvalKit's own items/prompts/metrics.  Treat as a strong indication, not a "
               "certified contrast.",
        per_cell=compare("i8b_1tile", arms["i8b_1tile_hf"], "bf16_32b", arms["bf16"], rng)
        if (arms["i8b_1tile_hf"] and arms["bf16"]) else NOT_MEASURED,
    )

    # ---- C: partial macro + the requirement test -------------------------------------------
    # The pre-registered constraint lives on the 8-CELL macro.  Only some cells are measurable
    # here (the three OPEN cells need the LLM judge, which is off; PMC_VQA is VOID under this
    # driver).  So compute an equal-weight PARTIAL macro over the usable cells, say exactly
    # which they are, and then state what the full-macro verdict would be under an explicitly
    # labelled assumption -- never by filling the missing cells in.
    USABLE = ["SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
    inv = json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/_shrink_parts/invert_constraint.json")))
    bar = inv["L2b_error_correlation_bracket"]["bar"]
    base_macro = inv["baseline"]["always_32b_direct_macro"]

    def partial_macro(cells_a, cells_b, rng):
        use = [c for c in USABLE if c in cells_a and c in cells_b
               and len(cells_a[c]["correct"]) == len(cells_b[c]["correct"])]
        if not use:
            return None
        per, boots = {}, []
        for c in use:
            A = cells_a[c]["correct"].astype(np.float64)
            B = cells_b[c]["correct"].astype(np.float64)
            n = len(A)
            idx = rng.integers(0, n, size=(NBOOT, n))
            boots.append((A - B)[idx].mean(axis=1))
            per[c] = dict(a=float(A.mean()), b=float(B.mean()), delta=float((A - B).mean()))
        bm = np.mean(boots, axis=0)
        return dict(cells_used=use, n_cells=len(use), per_cell=per,
                    macro_a=float(np.mean([per[c]["a"] for c in use])),
                    macro_b=float(np.mean([per[c]["b"] for c in use])),
                    delta=float(np.mean([per[c]["delta"] for c in use])),
                    lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)))

    pm = partial_macro(arms["i8b_1tile_hf"], arms["stored_vllm_32b"], rng) \
        if arms["i8b_1tile_hf"] else None
    out["C_partial_macro_i8b_vs_32b"] = dict(
        what="equal-weight macro over the %d of 8 reporting cells that are usable here." % len(USABLE),
        excluded=dict(
            PMC_VQA="VOID under the HF driver (all 33,430 responses blank, mean gen_tokens 0.0) "
                    "-- see VOID_CELLS.  This also voids the PMC_VQA row of "
                    "artifacts/lingshu_i8b_cheapleg_2026-08-11.json.",
            SLAKE_open="scored without the LLM judge => exact-match => 0.0000 for BOTH arms; "
                       "carries no information.",
            VQA_RAD_open="same", PATH_VQA_open="same"),
        result=pm or NOT_MEASURED,
        requirement=dict(
            eight_cell_bar_correlated=bar["correlated_bar"],
            eight_cell_bar_independent=bar["independent_bar"],
            always_32b_direct_macro=base_macro,
            projection_if_i8b_MATCHED_the_32b_on_the_4_unmeasured_cells=(
                round(base_macro + 0.5 * pm["delta"], 4) if pm else NOT_MEASURED),
            projection_label="EXPLICITLY LABELLED ASSUMPTION, NOT A MEASUREMENT: it assumes "
                             "I-8B exactly ties always-32B-direct on PMC_VQA and on the three "
                             "OPEN cells.  Those four cells are 4/8 of the macro and three of "
                             "them are open-text, where the 32B's margin over the 7B is "
                             "largest, so this assumption is GENEROUS to I-8B.",
            which_bar_applies=("the INDEPENDENT bar.  Lingshu-I-8B is a different architecture "
                               "(InternViT-300M + Qwen2 vs Qwen2.5-VL) trained by a different "
                               "pipeline, so its errors cannot be assumed correlated with "
                               "Lingshu-32B's.  Under the paired test an independently-erring "
                               "candidate needs macro >= %.4f, i.e. it must BEAT "
                               "always-32B-direct by +%.4f."
                               % (bar["independent_bar"], bar["independent_bar"] - base_macro)),
            verdict=("FAILS the pre-registered constraint on the generous projection: %s < %.4f"
                     % (round(base_macro + 0.5 * pm["delta"], 4), bar["independent_bar"])
                     if pm and (base_macro + 0.5 * pm["delta"]) < bar["independent_bar"]
                     else NOT_MEASURED)),
    )

    # ---- D: which way does the cross-stack confound push? ----------------------------------
    n4b = out["N4b_serving_stack_measured_on_lingshu_7b"]["per_cell"]
    sens = {}
    if isinstance(n4b, dict) and pm:
        for c in pm["cells_used"]:
            row = n4b.get(c)
            if not isinstance(row, dict) or "delta" not in row:
                sens[c] = "no HF-vs-vLLM pair exists for this cell (the HF Lingshu-7B arm did " \
                          "not run it), so the confound is unquantified here"
                continue
            sens[c] = dict(
                i8b_minus_32b_as_measured=round(pm["per_cell"][c]["delta"], 4),
                hf_minus_vllm_on_lingshu_7b=round(row["delta"], 4),
                hf_penalty_is_significant=bool(row["hi"] < 0 or row["lo"] > 0),
                i8b_delta_if_the_same_stack_penalty_applied_to_i8b=round(
                    pm["per_cell"][c]["delta"] - row["delta"], 4))
    out["D_cross_stack_sensitivity"] = dict(
        what="how the HF-vs-vLLM deviation, measured on Lingshu-7B, would move the "
             "I-8B-minus-32B deltas if the same stack penalty applied to I-8B.",
        status="SENSITIVITY, NOT A CORRECTION.  Nothing in the reported numbers is adjusted by "
               "it.  It transfers a deviation measured on one model (Lingshu-7B, Qwen2.5-VL) to "
               "another (Lingshu-I-8B, InternVL), which is exactly the kind of cross-"
               "multiplication this repo forbids in a headline.  It is reported only to "
               "establish the SIGN of the confound.",
        per_cell=sens,
        conclusion=(
            "THE CONFOUND RUNS AGAINST LINGSHU-I-8B, NOT FOR IT.  On the one cell where the "
            "deviation is significant, PATH_VQA_closed, the HF driver scores 0.0708 LOWER than "
            "vLLM on IDENTICAL Lingshu-7B weights.  Lingshu-I-8B is measured on HF; "
            "always-32B-direct is measured on vLLM.  So I-8B is being scored on the harsher "
            "stack and its opponent on the kinder one.  Either the penalty transfers -- in "
            "which case I-8B's advantage is understated -- or it is Qwen2.5-VL-specific and "
            "does not apply to an InternVL model, in which case no adjustment is warranted.  "
            "Neither branch makes the measured +0.0104 partial-macro tie an overstatement on "
            "the SERVING-STACK axis.  It remains uncertified on the axis that actually matters: "
            "4 of the 8 cells are not measured at all."),
    )

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k.startswith(("N4", "A_", "B_")) or k == "cells_available_per_arm"},
                     indent=1)[:6000])
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
