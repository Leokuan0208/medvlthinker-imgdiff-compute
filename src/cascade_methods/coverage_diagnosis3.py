#!/usr/bin/env python3
"""coverage_diagnosis3.py -- SCOUT B part 3, 2026-08-10.

(1) Quantifies the GOLD-QUALITY confound inside the no-coverage subset (retrospective
    hole: PathVQA-open golds are caption fragments and the local judge penalizes
    substantive-but-non-matching answers; the independent Claude-judge cross-validation
    covers SLAKE + VQA-RAD only, NOT PathVQA -- retrospective L1852-1861).
(2) Redoes the "what would it take" projection with the MEASURED marginal multiplier
    instead of the brief's assumed 0.81, against the LP sampling ceiling.
(3) Merges parts 1-3 into the canonical artifact
    results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
from training_methods import genframe_data as G  # noqa: E402

A1 = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json")
A2 = os.path.join(ROOT, "results/cascade_methods/artifacts/coverage_diagnosis2_2026-08-10.json")
SCDIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
DEPLOYED = os.path.join(ROOT, "ckpts/train/selector_ens8_scaled")
EVAL_DS = G.EVAL_DS
DIRECT32B = {"slake_open": 0.8186, "vqa_rad_open": 0.6000, "pathvqa_open": 0.3760}
# 5 MCQ cells at the accuracy-max-veto setting, verbatim from
# artifacts/cascade_selector_rerun_2026-08-05.json per_arm.disjoint.per_cell_acc
MCQ_AM = {"PMC_VQA": 0.5613, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
          "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}
MCQ_D32 = {"PMC_VQA": 0.5518, "SLAKE_closed": 0.8589, "VQA_RAD_closed": 0.8526,
           "PATH_VQA_closed": 0.8891, "MedXpertQA-MM": 0.3065}


def norm(s):
    return str(s).strip().lower()


def words(s):
    return re.findall(r"[a-z0-9]+", norm(s))


def gold_quality(items):
    sc = {}
    for ds in EVAL_DS:
        sc[ds] = {}
        for l in open(os.path.join(SCDIR, f"ckpt_{ds}_lingshu7b_sc8.jsonl")):
            r = json.loads(l)
            sc[ds][str(r["idx"])] = r
    out = {"why": "The local LLM judge grades against dataset gold strings. On PathVQA-open those "
                  "golds are CAPTION FRAGMENTS and the judge is known to penalise substantive but "
                  "non-matching answers (PROJECT_RETROSPECTIVE_2026-07-29.md L1852-1861 + "
                  "artifacts/pathvqa_judge_audit.json). The independent Claude-judge "
                  "cross-validation covers SLAKE and VQA-RAD only -- NOT PathVQA. So part of the "
                  "'no coverage' mass is UNMEASURABLE GOLD, not model capability. Either way it is "
                  "not reachable by more sampling; but the label matters.",
           "per_cell": {}}
    for ds in EVAL_DS:
        R = [it for it in items if it["ds"] == ds]
        rec = np.array([1 if 1 in it["sl"] else 0 for it in R])
        gl = np.array([len(words(sc[ds][str(it["idx"])]["gold"])) for it in R])
        out["per_cell"][ds] = {
            "n": len(R),
            "gold_words_mean_recoverable": float(gl[rec == 1].mean()),
            "gold_words_mean_no_coverage": float(gl[rec == 0].mean()),
            "share_gold>=5_words_recoverable": float((gl[rec == 1] >= 5).mean()),
            "share_gold>=5_words_no_coverage": float((gl[rec == 0] >= 5).mean()),
            "oracle@8_by_gold_length": {
                "1": float(rec[gl == 1].mean()) if (gl == 1).sum() else None,
                "2": float(rec[gl == 2].mean()) if (gl == 2).sum() else None,
                "3-4": float(rec[(gl >= 3) & (gl <= 4)].mean()) if ((gl >= 3) & (gl <= 4)).sum() else None,
                "5+": float(rec[gl >= 5].mean()) if (gl >= 5).sum() else None,
            },
            "n_by_gold_length": {"1": int((gl == 1).sum()), "2": int((gl == 2).sum()),
                                 "3-4": int(((gl >= 3) & (gl <= 4)).sum()), "5+": int((gl >= 5).sum())},
            "no_coverage_share_with_gold>=5_words": float(((rec == 0) & (gl >= 5)).sum() /
                                                          max((rec == 0).sum(), 1)),
        }
    R = items
    rec = np.array([1 if 1 in it["sl"] else 0 for it in R])
    gl = np.array([len(words(sc[it["ds"]][str(it["idx"])]["gold"])) for it in R])
    out["pooled"] = {
        "n_no_coverage": int((rec == 0).sum()),
        "of_which_gold>=5_words": int(((rec == 0) & (gl >= 5)).sum()),
        "share": float(((rec == 0) & (gl >= 5)).sum() / (rec == 0).sum()),
        "oracle@8_on_gold>=5_items": float(rec[gl >= 5].mean()),
        "oracle@8_on_gold<=2_items": float(rec[gl <= 2].mean()),
        "n_gold>=5": int((gl >= 5).sum()), "n_gold<=2": int((gl <= 2).sum()),
    }
    return out


def projection(items, p1, p2):
    dep = {}
    for d in ["slake", "vqa_rad", "pathvqa"]:
        for it in json.load(open(os.path.join(DEPLOYED, f"transfer_dump_{d}_open_lingshu7b.json"))):
            dep[(it["ds"], it["idx"])] = list(it["scores"])
    r = G.sel_eff(dep, items)
    sel_dep = {d: r["per_ds"][d]["acc"] for d in EVAL_DS}
    orc = {d: r["per_ds"][d]["oracle"] for d in EVAL_DS}
    ceil = p2["F_ceiling"]["per_cell"]

    m_pt = p2["A_multiplier"]["deployed_ens8_scaled"]["conversion_of_added_coverage"][
        "ADDED_by_samples_5..8"]
    m = m_pt["rate"]
    m_lo, m_hi = m_pt["ci95"]

    rows = {}
    for d in EVAL_DS:
        gap = DIRECT32B[d] - sel_dep[d]
        rows[d] = {
            "selected_now_deployed_selector": sel_dep[d],
            "always_32b_direct": DIRECT32B[d],
            "accuracy_gap_to_close": gap,
            "oracle@8_now": orc[d],
            "oracle_needed__brief_assumption_fixed_sel_eff":
                DIRECT32B[d] / r["per_ds"][d]["sel_eff"],
            "oracle_needed__MEASURED_marginal_multiplier": orc[d] + gap / m if gap > 0 else orc[d],
            "oracle_needed__measured_multiplier_CI": [
                orc[d] + gap / m_hi if gap > 0 else orc[d],
                orc[d] + gap / m_lo if gap > 0 else orc[d]],
            "LP_sampling_ceiling_oracle_at_N_infinity": ceil[d]["LP_estimated_reachable_share"],
            "VERDICT": None,
        }
        need = rows[d]["oracle_needed__MEASURED_marginal_multiplier"]
        cap = rows[d]["LP_sampling_ceiling_oracle_at_N_infinity"]
        rows[d]["VERDICT"] = ("ALREADY THERE" if gap <= 0 else
                              ("REACHABLE by sampling" if need <= cap else
                               "UNREACHABLE by iid sampling at any N (need %.3f > ceiling %.3f)"
                               % (need, cap)))
        rows[d]["headroom_at_the_ceiling"] = (cap - orc[d]) * m - gap

    macro_open_now = float(np.mean([sel_dep[d] for d in EVAL_DS]))
    macro_open_at_ceiling = float(np.mean([
        min(1.0, sel_dep[d] + (ceil[d]["LP_estimated_reachable_share"] - orc[d]) * m)
        for d in EVAL_DS]))
    macro8_now = (sum(MCQ_AM.values()) + sum(
        max(sel_dep[d], DIRECT32B[d]) for d in EVAL_DS)) / 8.0
    macro8_ceiling = (sum(MCQ_AM.values()) + sum(
        max(min(1.0, sel_dep[d] + (ceil[d]["LP_estimated_reachable_share"] - orc[d]) * m),
            DIRECT32B[d]) for d in EVAL_DS)) / 8.0
    macro8_oracle8 = (sum(MCQ_AM.values()) + sum(max(orc[d], DIRECT32B[d]) for d in EVAL_DS)) / 8.0

    return {
        "multiplier_used": {"value": m, "ci95": [m_lo, m_hi],
                            "what": "MEASURED fraction of newly-covered questions (covered by "
                                    "samples 5..8 but not 1..4) that the DEPLOYED selector converts",
                            "brief_assumed": 0.810627,
                            "brief_is_optimistic_by_factor": 0.810627 / m},
        "per_cell": rows,
        "macro_open_now": macro_open_now,
        "macro_open_always_32b_direct": 0.5982,
        "macro_open_at_the_iid_sampling_CEILING": macro_open_at_ceiling,
        "macro8_scenarios": {
            "_note": "5 MCQ cells held at the accuracy-max-veto values; open cells taken as "
                     "max(7B-arm, 32B-direct) because the cascade always has the escalation option. "
                     "This is an UPPER BOUND on what the coverage track can deliver: it charges no "
                     "extra generation cost and assumes escalation is free and perfectly targeted.",
            "always_32b_direct": 0.6567,
            "method_accuracy_max_veto_measured": 0.6575,
            "macro8_if_open_arm_reached_its_iid_sampling_CEILING": macro8_ceiling,
            "macro8_if_open_arm_were_a_PERFECT_selector_over_the_CURRENT_8_pool": macro8_oracle8,
            "delta_ceiling_vs_32b_direct": macro8_ceiling - 0.6567,
            "delta_perfect_selector_at_N8_vs_32b_direct": macro8_oracle8 - 0.6567,
        },
    }


def main():
    items = G.load_items()
    p1 = json.load(open(A1))
    p2 = json.load(open(A2))
    gq = gold_quality(items)
    pr = projection(items, p1, p2)

    merged = {
        "title": "SCOUT B -- COVERAGE DIAGNOSIS (open-text best-of-N, Lingshu-7B, n=2,345). "
                 "Does raising oracle@N have a multiplier, and is the 8-sample coverage ceiling a "
                 "DIVERSITY failure or a CAPABILITY failure?",
        "date": "2026-08-10", "no_gpu": True, "no_fabricated_numbers": True,
        "reproduce": ["python3 src/cascade_methods/coverage_diagnosis.py",
                      "python3 src/cascade_methods/coverage_diagnosis2.py",
                      "python3 src/cascade_methods/coverage_diagnosis3.py"],
        "nboot": 10000, "seed": 20260810,
        "numerics": {"OMP_NUM_THREADS": 1, "PYTHONHASHSEED": 0,
                     "note": "pure-numpy counting + bootstrap; no GEMM, so the TF32 landmine does "
                             "not apply. No model was run."},
        "HEADLINE": {
            "1_the_briefs_arithmetic_is_WRONG":
                "selected = oracle x sel_eff is an EXACT identity (max |err| 5.6e-17 over 4 cells). "
                "The brief's additive form 'selected ~= greedy + sel_eff*(oracle-greedy)' "
                "over-predicts selected accuracy by +0.090 to +0.111 on every cell. It plugs the "
                "CONDITIONAL-MEAN sel_eff (0.775/0.811) into the DIFFERENCE-form formula; the "
                "difference-form sel_eff is 0.2029, not 0.81.",
            "2_the_multiplier_is_0.45_not_0.81":
                "MEASURED conversion of NEWLY-covered questions by the deployed selector falls "
                "monotonically with N: 0.935 (covered at N=1) -> 0.671 (added by sample 2) -> 0.531 "
                "(added by 3..4) -> 0.4474 [0.3684, 0.5263] (added by 5..8). The realized "
                "d(selected)/d(oracle) is 0.303 over N=4->8 and 0.400 at N=7->8. The brief's 0.81 "
                "is optimistic by 1.8x and its CI excludes 0.81.",
            "3_it_is_a_CAPABILITY_failure_not_a_DIVERSITY_failure":
                "The no-coverage subset is the HIGH-diversity subset. mean distinct answers 5.17 vs "
                "3.00 on recoverable; mean modal share 0.432 vs 0.701; mean normalized entropy 0.830 "
                "vs 0.522. 50.2% of no-coverage questions already produce 6-8 distinct answers out "
                "of 8; only 7.8% produce one answer eight times. oracle@8 by exploration stratum: "
                "0.890 at nd=1, 0.364 at nd=6-8. An INDEPENDENT 16-sample redraw (3x the total "
                "budget) rescues only 21.2% of them.",
            "4_they_are_not_near_misses_so_IMAGE-SIDE_is_NOT_indicated":
                "On the no-coverage subset the best token-F1 of any pooled answer against gold is "
                "0.113 (vs 0.889 on recoverable); 71.5% of no-coverage questions have ZERO gold "
                "tokens anywhere in the 8-answer pool; only 10.1% are near misses (F1>=0.5). The "
                "brief's conditional -- 'if coverage failures are perception failures on short "
                "factual answers, image-side interventions are indicated' -- has a FALSE antecedent. "
                "Gold is LONGER on no-coverage items (3.06 words vs 1.70) and only 74.8% are <=3 "
                "words vs 93.4% on recoverable. Laterality is a real but minor enrichment "
                "(14.5% of no-coverage vs 10.3% of recoverable; laterality oracle@8 0.543 vs 0.637).",
            "5_the_iid_sampling_ceiling_is_below_what_is_needed_on_2_of_3_cells":
                "Capture-recapture (endpoint-8 vs independent-16, judge-labelled) puts the "
                "reachable-by-sampling share at slake 0.917 / vqa_rad 0.692 / pathvqa 0.626 "
                "(macro 0.745). At the MEASURED multiplier the oracle needed to merely TIE "
                "always-32B-direct is 0.958 / 0.868 / 0.518. SLAKE-open and VQA-RAD-open are "
                "UNREACHABLE by iid sampling at any N; PathVQA-open is already there.",
            "6_the_one_real_lever_found_is_heterogeneous-config_union_and_it_is_too_small":
                "JUDGE-labelled, vqa_rad_open n=200: union of the endpoint 8-pool with a "
                "DIFFERENT-config 8-pool gives +0.065 [+0.035, +0.100] oracle (temp 1.0) or "
                "+0.060 [+0.030, +0.095] (think mode), versus +0.035 [+0.015, +0.060] for a second "
                "iid draw at the SAME config -- so config heterogeneity roughly DOUBLES the coverage "
                "gain per sample. At the measured 0.45 multiplier that is ~+0.029 selected accuracy "
                "for 2x the generation cost, against a 0.120 gap to 32B-direct on that cell.",
            "VERDICT":
                "The coverage track does NOT have the multiplier the round's premise assumes, and "
                "the ceiling it can reach is below the bar on 2 of the 3 open cells. Coverage is "
                "worth ~0.45 on the margin, not 0.81; it saturates at ~+0.035 oracle per doubling "
                "of N; and the questions it cannot reach are ones the model gets wrong "
                "confidently-diversely, not ones it failed to explore. RECOMMENDATION: do NOT spend "
                "GPU on raising oracle@N by sampling. The only coverage lever with a measured, "
                "significant effect is heterogeneous-CONFIG generation, and even it is ~4x too "
                "small on the binding cell.",
        },
        "sources": p1["sources"],
        "null_tests": p1["null_tests"],
        "part1_arithmetic": p1["arithmetic"],
        "part1_oracle_at_N": p1["oracle_at_N"],
        "part1_diversity_vs_capability": p1["diversity_vs_capability"],
        "part1_signature": p1["signature"],
        "part2_multiplier": p2["A_multiplier"],
        "part2_oracle_to_32": p2["B_oracle_to_32"],
        "part2_text_side_alternatives": p2["C_text_side_alternatives"],
        "part2_image_side_resolution": p2["D_image_side_resolution"],
        "part2_laterality": p2["E_laterality"],
        "part2_ceiling": p2["F_ceiling"],
        "part3_gold_quality_confound": gq,
        "part3_projection_with_MEASURED_multiplier": pr,
        "caveats": [
            "The endpoint labels are the project's local LLM judge. On PathVQA-open (725 of the 877 "
            "no-coverage items, 82.7%) the golds are caption fragments and the judge is known to "
            "penalise substantive-but-non-matching answers; the independent Claude-judge "
            "cross-validation covers SLAKE and VQA-RAD only, NOT PathVQA (retrospective L1852-1861). "
            "Part of the coverage wall on that cell is therefore UNMEASURABLE GOLD rather than model "
            "capability. This does not change the verdict -- neither is fixable by sampling -- but "
            "it does mean 'capability failure' is an upper bound on the model's share of the blame.",
            "The sc16 dump is an INDEPENDENT sampling run, not a superset of the endpoint pool "
            "(first-8 preds agree on only 340/645, 32/200, 140/1500). It is used as a second draw, "
            "which is exactly what the capture-recapture and rescue analyses require.",
            "The cap80/cap160 resolution dumps carry EXACT-MATCH labels only and are single GREEDY "
            "answers. Every image-side rescue number is a strict LOWER BOUND, and no HIGHER-than-"
            "cap320 open-text dump exists, so the interesting image-side direction (more pixels, "
            "multi-crop, multi-scale) is genuinely UNTESTED -- see 'what_was_never_tried'.",
            "oracle@N to 32 is measured on vqa_rad_open only (the only cell with an sc32 dump).",
            "The macro8 scenario table holds the 5 MCQ cells fixed at their measured accuracy-max-"
            "veto values and takes each open cell as max(7B arm, 32B-direct). It charges no extra "
            "generation cost. It is an UPPER BOUND, not a prediction.",
            "vqa_rad_open is n=200; per-cell flags there move with a handful of items (the project's "
            "own guardrail-resolution caveat). The text-side union result (+0.065) rests on that "
            "cell alone and has NOT been replicated on slake_open or pathvqa_open.",
        ],
        "what_was_never_tried": {
            "prior_diverse_generation_experiment":
                "src/cascade_methods/diversity_generate_gpu.py + artifacts/diverse_generation_gpu.json "
                "varied ONLY the SYSTEM PROMPT (5-way portfolio: base/anatomy/modality/differential/"
                "concise) x TEMPERATURE (0.7/1.0/1.3), M=15. Same image, same cap320 resolution, same "
                "crop, same model. Result: oracle lift +0.0271 [+0.0099, +0.0431] at matched N=8 "
                "(DPP-selected), +0.0635 [+0.0468, +0.0801] at M=15 (1.88x gen cost) -- but sel_eff "
                "FELL 0.7321 -> 0.7229 -> 0.6989 and the confident-distractor rate ROSE 0.268 -> "
                "0.301, so selected accuracy moved only +0.0142 [-0.0025, +0.0308] at matched N "
                "(NOT significant). Those numbers are on EXACT-MATCH labels with the CONTAMINATED "
                "pooled4 verifier and are not comparable to the judge-labelled endpoint.",
            "never_varied": [
                "image RESOLUTION upward (cap640 / fullres) as a generation-portfolio axis -- only "
                "cap80/cap160 (DOWNWARD) exist for open text, greedy, exact-match-labelled",
                "multi-crop / zoom / region-of-interest views",
                "multi-scale ensembling of the same image",
                "a cross-MODEL candidate pool for the open-text cells (the retrospective notes a "
                "cross-model pool raises oracle-of-N by +0.11 to +0.15 -- the largest coverage effect "
                "recorded anywhere in this project, and it is a CAPABILITY change, not a sampling one)",
                "the heterogeneous-config union measured here (endpoint-config + think-config, or "
                "+ temp-1.0-config) on slake_open and pathvqa_open -- measured on vqa_rad_open only",
            ],
            "why_image_side_is_STILL_not_recommended":
                "71.5% of no-coverage questions share ZERO tokens with gold anywhere in the pool. A "
                "resolution or crop change moves the answer (cap80 produces an answer outside the "
                "8-pool on 26-30% of questions) but converts almost nothing: on the judge-defined "
                "no-coverage subset a cap80 greedy answer is exact-match correct on 5.4% (vqa_rad) / "
                "7.7% (pathvqa) of items -- and those are LOWER bounds on a metric that also has "
                "single-sample-vs-eight-samples working in its favour on cost, not on yield.",
        },
    }
    json.dump(merged, open(A1, "w"), indent=1, default=float)
    print("wrote", A1)
    return merged


if __name__ == "__main__":
    m = main()
    print(json.dumps(m["part3_projection_with_MEASURED_multiplier"], indent=1, default=float))
    print(json.dumps(m["part3_gold_quality_confound"]["pooled"], indent=1, default=float))
