#!/usr/bin/env python3
"""ATTACK 3 -- propagate the later session's two findings back into the artifact itself.

The repo's documented failure mode (retrospective 9.6) is that corrections get made in NEW
files and are never propagated into the old ones.  This script edits
sevenb_only_frontier_2026-08-12.json IN PLACE, adding:

  VERIFICATION_2026-08-12   the independent re-derivation of every headline in the artifact
  and it REWRITES two stale statements:
    PART4.quantised_strong_leg_UPDATE_2026-08-12.accuracy_of_the_quantised_strong_leg
    PART4.consequence_for_the_users_goal
  both of which said the quantised strong leg's accuracy was NOT MEASURED.  It now is, on
  3 of 8 cells, and one of those 3 is a significant loss.

No measured value produced by the original run is altered.  Only verification results and
newly-measured facts are added, and the two sentences that are now false are corrected with
their supersession stated inline.
"""
import json
import os

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
ART = ROOT + "/results/cascade_methods/artifacts"
MAIN = f"{ART}/sevenb_only_frontier_2026-08-12.json"
PARTS = f"{ART}/_frontier_verify_parts"


def main():
    d = json.load(open(MAIN))
    ver = json.load(open(f"{PARTS}/verify2.json"))
    nf4 = json.load(open(f"{PARTS}/nf4_tie.json"))

    # ---------------- 1. the verification section ----------------
    d["VERIFICATION_2026-08-12"] = {
        "what": ("an adversarial re-derivation of every headline in this artifact, written in a "
                 "LATER session and deliberately NOT importing sevenb_only_frontier.py, so a bug "
                 "in that module could not reproduce itself."),
        "code": "src/cascade_methods/frontier_verify2.py",
        "detail": "results/cascade_methods/artifacts/_frontier_verify_parts/verify2.json",
        "ALL_PASSED": ver["ALL_NULL_TESTS_AND_REPRODUCTIONS_PASSED"],
        "results": {
            "N1_published_macro_baselines": {
                "max_abs_deviation": ver["N1_macro_baselines"]["max_abs_deviation"],
                "PASSED": ver["N1_macro_baselines"]["PASSED"]},
            "V1_frontier_headline_0.616278": {
                "macro_recomputed": ver["V1_frontier_headline"]["macro_recomputed"],
                "macro_published": ver["V1_frontier_headline"]["macro_published"],
                "abs_deviation": ver["V1_frontier_headline"]["abs_deviation"],
                "vs_direct_recomputed": ver["V1_frontier_headline"][
                    "vs_always_32b_direct_recomputed"],
                "ci_lo_abs_dev": ver["V1_frontier_headline"]["ci_lo_abs_dev"],
                "PASSED": ver["V1_frontier_headline"]["PASSED"],
                "identity": ver["V1_frontier_headline"]["identity_used"]},
            "V2_capability_floor": {
                "max_abs_deviation": ver["V2_capability_floor"]["max_abs_deviation"],
                "PASSED": ver["V2_capability_floor"]["PASSED"]},
            "V3_min_escalation_that_ties": {
                "recomputed": ver["V3_min_escalation"]["minimum_escalation_that_ties_recomputed"],
                "published": ver["V3_min_escalation"]["minimum_escalation_that_ties_published"],
                "PASSED": ver["V3_min_escalation"]["PASSED"]},
            "V4_vram_cliff_arithmetic": ver["V4_vram_cliff"]["ARITHMETIC_VERDICT"],
            "V5_load_on_demand": ver["V5_load_on_demand"]["VERDICT"],
        },
        "one_line": ("every headline in this artifact reproduces EXACTLY from an independent "
                     "re-implementation: macro 0.616278 to 6 decimals, the vs-direct delta "
                     "-0.040395 [-0.052275, -0.028427] to 6 decimals, and the same minimum "
                     "escalation subset.  The VRAM cliff arithmetic is confirmed."),
    }

    # ---------------- 2. the quantised strong leg is no longer OPEN ----------------
    q = d["PART4_the_VRAM_cliff_and_load_on_demand"]["quantised_strong_leg_UPDATE_2026-08-12"]
    q["accuracy_of_the_quantised_strong_leg_SUPERSEDED_2026-08-12"] = {
        "what_this_section_used_to_say": (
            "NOT MEASURED -- the run CRASHED before writing any cell ... the results dict is {}."),
        "status_now": "MEASURED on 3 of the 8 cells.  The re-launched run completed "
                      "2026-08-12 09:20 (NF4) and 09:38 (bf16 control).",
        "source": ("_shrink_parts/acc_{nf4,bf16}.json, paired per item by "
                   "_shrink_parts/quant_acc_paired.json "
                   "(src/cascade_methods/shrink_quant_acc_analyze.py, re-run once the arms "
                   "finished; it had been run at 06:16 while both arms were still empty)"),
        "control_is_MATCHED": ("the bf16 arm is the same driver, items, batch size and greedy "
                               "decoding as the NF4 arm, so the NF4-minus-bf16 delta is "
                               "attributable to weight quantisation alone.  The round's "
                               "+/-0.008 matched-control caveat is satisfied."),
        "serving_stack_null_test": (
            "PASSES.  The HF bf16 control reproduces the published vLLM always-32B-direct cells "
            "to within 0.0072 (SLAKE_closed -0.0072, VQA_RAD_closed 0.0000, PATH_VQA_closed "
            "-0.0003), all three CIs spanning zero."),
        "per_cell_NF4_minus_bf16": ver["Q_quantised_strong_leg_NOW_MEASURED"]["per_cell"],
        "mean_over_the_3_measured_closed_cells": ver["Q_quantised_strong_leg_NOW_MEASURED"][
            "mean_over_the_3_measured_closed_cells"],
        "cells_still_UNMEASURED": ver["Q_quantised_strong_leg_NOW_MEASURED"][
            "cells_still_UNMEASURED"],
        "why_the_open_cells_are_unusable": ver["Q_quantised_strong_leg_NOW_MEASURED"][
            "why_the_open_cells_are_unusable"],
        "HEADLINE": (
            "4-bit is NOT accuracy-free.  On the 3 cells that could be measured it is "
            "+0.0012 [-0.0039,+0.0062] on PATH_VQA_closed (n.s.), +0.0080 [-0.0159,+0.0319] on "
            "VQA_RAD_closed (n.s.), and -0.0203 [-0.0335,-0.0084] on SLAKE_closed, which is a "
            "SIGNIFICANT LOSS.  Mean over the three: -0.0037."),
    }
    q["does_the_VRAM_escape_hatch_SURVIVE_its_own_accuracy_measurement"] = {
        "question": ("take PART3's minimum-escalation-that-ties policy and swap the bf16 strong "
                     "leg for the NF4 strong leg on the cells where NF4 accuracy is MEASURED.  "
                     "Does it still tie with always-32B-direct?"),
        "code": "src/cascade_methods/frontier_nf4_tie.py",
        "detail": "_frontier_verify_parts/nf4_tie.json",
        "cells_swapped_to_NF4": nf4["policy"]["cells_where_the_strong_leg_is_NF4_in_arm_2"],
        "cells_left_on_bf16_because_NF4_is_unmeasured": nf4["policy"][
            "cells_left_on_bf16_because_NF4_IS_UNMEASURED_THERE"],
        "arm1_bf16": nf4["RESULT"]["arm1_bf16_strong_leg"],
        "arm2_NF4": nf4["RESULT"]["arm2_NF4_strong_leg_on_the_3_measured_cells"],
        "paired_delta": nf4["RESULT"]["paired_delta_arm2_minus_arm1"],
        "VERDICT": nf4["VERDICT"]["one_line"],
        "arm2_is_an_UPPER_BOUND": nf4["policy"]["this_makes_arm_2_an_UPPER_BOUND"],
        "what_this_does_NOT_settle": nf4["VERDICT"]["what_this_does_NOT_settle"],
    }
    q["VERDICT_SUPERSEDED_2026-08-12"] = (
        "the original VERDICT said '(iii) whether it keeps the 32B's accuracy ... is OPEN and is "
        "the single highest-value follow-up this round produced.'  It is no longer fully open.  "
        "Measured on 3 of 8 cells, NF4 costs -0.0203 [-0.0335,-0.0084] on SLAKE_closed -- a "
        "SIGNIFICANT loss -- and is within noise on the other two.  So the 4-bit escape hatch is "
        "real on VRAM (19.53 GiB measured) and NOT free on accuracy.  Whether it still ties "
        "end-to-end is answered, for the measurable part, in "
        "does_the_VRAM_escape_hatch_SURVIVE_its_own_accuracy_measurement.  The five unmeasured "
        "cells -- PMC_VQA, MedXpertQA-MM and all three OPEN cells, 5/8 of the macro weight -- "
        "remain the open item, and the open cells have no NF4 accuracy at any quality because "
        "both arms were scored with use_llm_judge=False.")

    # ---------------- 3. the consequence sentence, corrected ----------------
    p4 = d["PART4_the_VRAM_cliff_and_load_on_demand"]
    p4["consequence_for_the_users_goal_SUPERSEDED_2026-08-12"] = p4[
        "consequence_for_the_users_goal"]
    tie = "still ties" if nf4["VERDICT"]["arm2_ties"] else "no longer ties"
    p4["consequence_for_the_users_goal"] = (
        "'less VRAM than the 32B' is achievable ONLY in regime A -- 18.76 GiB measured, 3.9x "
        "smaller than always-32B-direct's 72.60 GiB, on a 24 GB card instead of an 80 GB card.  "
        "But regime A's best measured accuracy is 0.616278 macro against the 32B's 0.656672, a "
        "SIGNIFICANT shortfall of -0.0404 [-0.0523, -0.0284].  Every policy that recovers that "
        "accuracy is in regime B, where the footprint is not 'less than the 32B' -- it is the "
        "32B PLUS the 7B, on more hardware than always-32B-direct needs.  There is no measured "
        "operating point in between AT bf16.  UPDATED 2026-08-12: the one lever that could "
        "create one is a QUANTISED strong leg, and its accuracy is now MEASURED on 3 of 8 cells "
        "(it was 'not measured' when this sentence was first written).  NF4 removes the VRAM "
        "obstacle -- 19.53 GiB resident, so 7B + NF4-32B = 35.03 GiB of weights, one 80 GB card "
        "with room to spare -- but it is not accuracy-free: -0.0203 [-0.0335,-0.0084] on "
        "SLAKE_closed is a significant loss, and with that swapped into the minimum-escalation "
        "policy the tie with always-32B-direct " + tie + ".  Five cells, including all three "
        "open ones, are still unmeasured at 4-bit.")

    json.dump(d, open(MAIN, "w"), indent=1)
    print("amended", MAIN)
    print("VERIFICATION ALL_PASSED =", d["VERIFICATION_2026-08-12"]["ALL_PASSED"])
    print("NF4 tie verdict:", nf4["VERDICT"]["one_line"])


if __name__ == "__main__":
    ver = json.load(open(f"{PARTS}/verify2.json"))
    main()
