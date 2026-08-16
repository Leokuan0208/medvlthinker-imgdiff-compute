#!/usr/bin/env python3
"""closed_as_open_prereg.py -- freeze the BUILD 3 protocol BEFORE any candidate is generated.

Writes results/cascade_methods/artifacts/closed_as_open_2026-08-16_preregistration.json.
CPU only.  Run from the repo root:  python3 src/cascade_methods/closed_as_open_prereg.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

sys.path.insert(0, os.path.join(L.ROOT, "src"))
from training_methods import genframe_data as G                           # noqa: E402

OUT = os.path.join(L.ART, f"closed_as_open_{L.DATE}_preregistration.json")


def main():
    nt = G.null_test()
    m = nt["measured"]
    identity = abs(m["selected"] - m["oracle@8"] * m["sel_eff"])

    doc = {
        "title": "BUILD 3 pre-registration -- re-ask SLAKE_closed / VQA_RAD_closed / PATH_VQA_closed "
                 "as OPEN TEXT and test whether the frozen verifier's pick beats greedy on them.",
        "date": L.DATE,
        "written_before": "any candidate generation for these three cells; no accuracy from any new "
                          "arm had been computed when this file was written.",
        "no_fabricated_numbers": True,
        "not_abstention": "every arm returns an answer on every item; an UNPARSED answer is graded "
                          "WRONG, never withheld.",

        # -------------------------------------------------------------------------------------
        "null_tests": {
            "N1_frozen_open_text_metric": {
                "source_of_truth": "src/training_methods/genframe_data.py PUBLISHED",
                "pass": nt["pass"],
                "max_abs_deviation": nt["max_abs_deviation"],
                "measured": {k: m[k] for k in ["n", "n_recoverable", "oracle@8", "selected",
                                               "greedy", "sel_eff", "cand_auroc"]},
                "per_ds": m["per_ds"],
                "identity_selected_eq_oracle_times_sel_eff_abs_error": identity,
            },
            "G1_grader": L.grader_null_test(),
        },

        # -------------------------------------------------------------------------------------
        "what_the_harness_actually_does": {
            "VQA_RAD_closed": "MedEvalKit/utils/VQA_RAD/VQA_RAD.py:45 get_judgement_prompt -> "
                              "\"Please output 'yes' or 'no'(no extra output).\" -- ANSWER SPACE GIVEN (K=2)",
            "PATH_VQA_closed": "MedEvalKit/utils/PATH_VQA/PATH_VQA.py:60 get_judgement_prompt -- same",
            "SLAKE_closed": "MedEvalKit/utils/SLAKE/SLAKE.py:56 get_close_ended_prompt -> "
                            "\"Answer the question using a single word or phrase.\" -- NO ANSWER SPACE. "
                            "SLAKE_closed is ALREADY asked open-ended; only its GRADER is closed "
                            "(judge_close_end_vqa = strict string equality). The premise 'the harness "
                            "presents them as closed' is therefore true for 2 of the 3 cells, not 3, "
                            "and this is recorded BEFORE the experiment, not after.",
            "SLAKE_closed_composition": "836 items, 420 zh / 416 en, and the gold set is NOT yes/no "
                                        "only (no 180, yes 175, 不是 86, 是的 73, 包含 65, 不包含 61, "
                                        "有 21, 没有 19, 肺 14, lung 12, ...). Counted from "
                                        "/data/dan/dataset/medevalkit/SLAKE/test.json.",
        },

        # -------------------------------------------------------------------------------------
        "arms": {k: {kk: (vv if kk != "sys" else ("SYS_OPEN" if vv else None))
                     for kk, vv in v.items()} for k, v in L.ARMS.items()},
        "arm_rationale": {
            "closedD_g": "in-session greedy control under the DEPLOYED prompt -- isolates the reformat "
                         "from the serving config (the published cell is a different harness AND "
                         "fullres).",
            "closedD_g_full": "same at fullres -- quantifies the resolution half of the gap to the "
                              "published cell.",
            "closedD_s8": "MECHANISM CONTROL: candidates are SAMPLED but the answer space is still "
                          "given. Separates 'sampled provenance' from 'answer-space removal'.",
            "openMEK_*": "MedEvalKit's own open-ended prompt substituted for the deployed one -- one "
                         "variable changed, the trailing instruction.",
            "openPRJ_*": "the DEPLOYED OPEN ARM's generation recipe verbatim (run_openvqa.py SYS_OPEN "
                         "+ bare question + cap320 + max_tokens 64) -- the regime the frozen verifier "
                         "was built for.",
        },
        "PRIMARY_ENDPOINT": {
            "comparison": "SELECTED (frozen verifier argmax over the 8 openPRJ_s8 candidates) minus "
                          "GREEDY (openPRJ_g), per cell, in BOTH currencies.",
            "arms": L.PRIMARY,
            "why_openPRJ_and_not_openMEK": "its generation recipe is byte-identical to the deployed "
                                           "open arm, so a win transfers to the shipped method without "
                                           "a prompt change; openMEK is reported as the second open "
                                           "formulation and the 2-variant multiplicity is stated.",
            "decision_rule": "a cell COUNTS toward the claim only if SELECTED - GREEDY is positive "
                             "with a 95% paired-bootstrap CI excluding 0 in BOTH currencies.",
        },

        # -------------------------------------------------------------------------------------
        "grading_rules_frozen_here": {
            "EM_harness": "MedEvalKit's own function for the cell, re-implemented verbatim in "
                          "closed_as_open_lib.py and null-tested at G1 above (0 row disagreements "
                          "over 4,449 deployed rows).",
            "EM_repaired": {
                "yes_no_golds": "POLARITY, first token wins: scan the normalised response tokens "
                                "left to right; first token in AFF -> yes, first in NEG -> no; "
                                "neither -> UNPARSED, scored 0.",
                "AFF": sorted(L.AFF), "NEG": sorted(L.NEG),
                "other_golds": "STRICT normalised equality (lowercase, strip surrounding quotes and "
                               "whitespace, drop trailing ASCII/full-width period, collapse internal "
                               "whitespace). NO substring containment.",
                "why_no_containment": "containment is the mechanism by which a longer answer scores "
                                      "higher for free; the open arms produce longer answers by "
                                      "construction, so containment would manufacture the effect "
                                      "being tested.",
            },
            "judge": "MedVLThinker-32B text judge, src/labeling/run_judge.py, unchanged. Called ONCE "
                     "per (item, distinct normalised answer) across ALL arms of a cell, so identical "
                     "strings share a label and no arm can be advantaged by judge noise.",
            "both_currencies_required": "a judge-only verifier result is not interpretable "
                                        "(paraphrase drift, CLAUDE.md s0).",
            "grading_artifact_clause": "if the open-form arm scores differently from the closed-form "
                                       "arm mainly because of the grader, that is reported AS a "
                                       "grading artifact. The pre-committed diagnostics are: mean "
                                       "answer length per arm, UNPARSED rate per arm, and the "
                                       "EM_harness-vs-EM_repaired gap per arm.",
        },

        # -------------------------------------------------------------------------------------
        "statistics": {
            "bootstrap": "paired item-level, nboot=10000, seed 20260816, resampled within cell",
            "guardrail": "per cell vs the same cell's in-session greedy; a pooled win that loses a "
                         "cell is guardrail-dirty and is reported as such",
            "floors_reported_per_cell": ["random-pick floor = mean over items of (#correct slots)/8",
                                         "distinct-candidate count distribution",
                                         "oracle@8 (coverage)",
                                         "self-consistency / majority vote (training-free reference)"],
            "identity_asserted": "SELECTED == oracle@8 * sel_eff",
            "numerics_pinned": {"OMP_NUM_THREADS": 1, "PYTHONHASHSEED": 0, "TF32": "off",
                                "verifier_backend": "HF transformers only -- vLLM drops all 192 "
                                                    "visual.* LoRA modules"},
        },
        "verifier": {
            "adapter": "ckpts/train/lora_verifier_disjoint (FROZEN, clean/disjoint-trained; the "
                       "0.775204 incumbent)",
            "scoring_fn": "pyes() verbatim from src/training_methods/verifier_transfer_eval.py",
            "nothing_is_retrained": "no verifier is trained in this build, so the +0.006-0.009 "
                                    "paraphrase-drift free lunch cannot apply",
            "disjointness_to_be_proven_not_assumed": "pixel-md5 of decoded RGB between every "
                                                     "verifier-training image and every item of the "
                                                     "three NEW eval cells, reported per cell",
        },
        "secondary_self_consistency": {
            "cells": ["PMC_VQA (pre-registered 6,000 subsample, seed 20260810, reused from "
                      "unified_pipeline)", "MedXpertQA-MM (all 2,000)"],
            "test": "majority vote over 8 T=0.4 samples vs an in-session greedy control, under the "
                    "DEPLOYED MCQ prompt; training-free, generation cost only",
            "expectation_stated_in_advance": "these two cells are intrinsically multiple-choice (the "
                                             "question often cannot be answered without the options) "
                                             "and are EXPECTED to stay outside the claim; a null here "
                                             "scopes the claim to 6 cells and is reported as such.",
        },
        "prespecified_failure_modes": [
            "the open arms score lower purely because the strict grader punishes longer answers "
            "-> detected by EM_repaired (length-neutral) disagreeing with EM_harness",
            "the judge inflates the open arms because it is lenient about phrasing -> detected by "
            "the judge-vs-EM_repaired gap",
            "SLAKE_closed's reformat is a near no-op (no answer space either way) -> already "
            "recorded above, so a null there is not a surprise",
            "yes/no pools collapse to 1 distinct candidate -> oracle@8 == greedy and there is "
            "nothing for any selector to do; reported as the distinct-candidate distribution",
        ],
        "artifact": f"results/cascade_methods/artifacts/closed_as_open_{L.DATE}.json",
    }
    os.makedirs(L.ART, exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {OUT}")
    print("N1 pass", nt["pass"], "max_abs_dev", nt["max_abs_deviation"], "identity", identity)
    print("G1 pass", doc["null_tests"]["G1_grader"]["pass"],
          "max_abs_dev", doc["null_tests"]["G1_grader"]["max_abs_deviation"])


if __name__ == "__main__":
    main()
