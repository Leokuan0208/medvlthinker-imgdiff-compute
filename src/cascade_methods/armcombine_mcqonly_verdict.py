#!/usr/bin/env python3
"""
Append the cost (FLOP-eq / latency / energy, all MEASURED constants) and the VERDICT to
armcombine_mcqonly_2026-08-11.json.  Computes no new accuracy; every constant is read from
_selector_rerun_parts/macro_disjoint.json:cost.per_cell_as_charged, which is the file the published
per-cell costs come from.

Reproduce:  python3 src/cascade_methods/armcombine_mcqonly_verdict.py
"""
import os
import json

import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
P = os.path.join(ART, "armcombine_mcqonly_2026-08-11.json")
d = json.load(open(P))
d.pop("VERDICT", None)
d.pop("cost_measured_all_axes", None)

PCC = json.load(open(os.path.join(ART, "_selector_rerun_parts/macro_disjoint.json"))) \
    ["cost"]["per_cell_as_charged"]
CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = CELLS[:5]
BASE = "always_32b_direct"
AX = ["flops", "lat_par_ms", "lat_seq_ms", "energy_j"]


def macro_cost(mcq_arm):
    return {a: float(np.mean([PCC[c][mcq_arm if c in MCQ else BASE][a] for c in CELLS])) for a in AX}


base = macro_cost(BASE)
rows = {}
for arm in ["method_accuracy_max_veto", "method_accuracy_max_fusion", "method_compute_lean"]:
    m = macro_cost(arm)
    rows[f"MCQ={arm} / OPEN=always_32b_direct"] = dict(
        macro={k: round(v, 3) for k, v in m.items()},
        vs_always_32b_direct={k: round(m[k] / base[k], 4) for k in AX},
        pct_change={k: round(100 * (m[k] / base[k] - 1), 2) for k in AX},
        per_cell_PMC_VQA={k: PCC["PMC_VQA"][arm][k] for k in AX})

d["cost_measured_all_axes"] = dict(
    weighting="MACRO (8 cells, 1/8 each).  Pairs ONLY with the MACRO accuracies in this file.",
    provenance="every constant read verbatim from results/cascade_methods/artifacts/"
               "_selector_rerun_parts/macro_disjoint.json:cost.per_cell_as_charged -- the same file "
               "the published per-cell costs come from.  No constant is modelled here, and no "
               "best-of-N arm is involved, so nothing in this block is 'not measured'.",
    baseline_always_32b_direct={k: round(v, 3) for k, v in base.items()},
    rows=rows,
    R32_note="the FLOP-eq column is as-charged at R32 = 4.57.  At the derived R32 = 3.816 the ratios "
             "are in fixed_mcq_arm_policies[*]['x_direct_R32_3.816'].")

fus = d["fixed_mcq_arm_policies"]["method_accuracy_max_fusion"]
vet = d["fixed_mcq_arm_policies"]["method_accuracy_max_veto"]
pn = d["permutation_null_dedup"]

d["VERDICT"] = dict(
    what_is_positive=(
        "TWO FORMAT-AWARE FIXED POLICIES BEAT always-32B-direct ON THE 8-CELL MACRO WITH A CI THAT "
        "EXCLUDES ZERO, and one of them is cheaper on FLOP-eq and energy as well.  Both are "
        "'apply the shipped arm to the multiple-choice half, leave the open-text half at "
        "always-32B-direct'."),
    rows=[
        dict(policy="MCQ = certified veto (accuracy-max), OPEN = always-32B-direct",
             macro=vet["macro_acc"], delta=vet["delta"], ci=[vet["lo"], vet["hi"]],
             verdict=vet["verdict"], guardrail_flags=vet["guardrail_flags"],
             flopeq_x=rows["MCQ=method_accuracy_max_veto / OPEN=always_32b_direct"]
                          ["vs_always_32b_direct"]["flops"],
             lat_par_pct=rows["MCQ=method_accuracy_max_veto / OPEN=always_32b_direct"]
                             ["pct_change"]["lat_par_ms"],
             energy_pct=rows["MCQ=method_accuracy_max_veto / OPEN=always_32b_direct"]
                            ["pct_change"]["energy_j"]),
        dict(policy="MCQ = fusion (accuracy-max+), OPEN = always-32B-direct",
             macro=fus["macro_acc"], delta=fus["delta"], ci=[fus["lo"], fus["hi"]],
             verdict=fus["verdict"], guardrail_flags=fus["guardrail_flags"],
             flopeq_x=rows["MCQ=method_accuracy_max_fusion / OPEN=always_32b_direct"]
                          ["vs_always_32b_direct"]["flops"],
             lat_par_pct=rows["MCQ=method_accuracy_max_fusion / OPEN=always_32b_direct"]
                             ["pct_change"]["lat_par_ms"],
             energy_pct=rows["MCQ=method_accuracy_max_fusion / OPEN=always_32b_direct"]
                            ["pct_change"]["energy_j"]),
    ],
    THE_CAVEATS_THAT_MUST_TRAVEL_WITH_IT=[
        "IT IS 100% ONE CELL.  The policy is byte-identical to always-32B-direct on 7 of the 8 "
        "cells, by construction.  Leave-one-out: dropping PMC_VQA takes the delta to exactly 0.000. "
        "This is the most concentrated claim the project has ever made -- more concentrated than "
        "hole 2 ever was.",
        "PMC_VQA HERE IS test_2.csv -- 33,430 items, 79.2% of the Variant-B pool, and the split with "
        "ZERO published verification (CLAUDE.md two-split landmine; "
        "docs/current/PMCVQA_PROVENANCE_2026-07-30.md).  A claim carried entirely by that cell "
        "inherits that cell's provenance risk entirely.",
        "IT IS NOT A NEW MEASUREMENT.  cascade_selector_rerun_2026-08-05.json already reports the "
        "MCQ-half delta of the fusion arm as +0.0027 [+0.0020, +0.0034], a significant WIN.  The "
        "8-cell macro number is that number times 5/8 and reproduces it to 2e-06.  What is new is "
        "only the OBSERVATION that restricting the shipped arm to the multiple-choice half turns a "
        "macro LOSS (the full fusion arm is 0.6503, -0.0064 vs direct) into a macro WIN.",
        "THE EFFECT IS +0.0017, not +0.0029.  It clears zero, it does not clear the scale the "
        "project has been aiming at.  Calling it 'beating always-32B-direct' is literally true on "
        "the stated axis and would be misleading without the size attached.",
        "IF THE MCQ ARM IS *SELECTED* ON THIS EVAL RATHER THAN DEPLOYED FROM THE PUBLISHED RESULT, "
        f"the gain does not survive: cross-fit selection of the MCQ arm gives p = "
        f"{pn['E7_crossfit']['p_one_sided']} against its own exchangeability null (null sd "
        f"{pn['E7_crossfit']['null_sd']}), and eval-visible selection gives p = "
        f"{pn['best_fixed_MCQ_arm_evalvisible']['p_one_sided']}.  The claim rests on the policy "
        "being PRE-SPECIFIED from the 2026-08-05 artifact, not on it being discovered here.",
        "THE PERMUTATION NULL IS CONSERVATIVE HERE AND SHOULD NOT BE OVER-READ.  The exchangeability "
        "null forces always_7b -- an arm that is genuinely ~6 macro points worse -- to be "
        "exchangeable with always-32B-direct, so a permuted 'selection' can land on what is "
        "effectively a random column and the null's variance balloons.  That null is the right test "
        "for 'is any arm better than the others'; it is NOT the right test for a pre-specified "
        "policy, for which the paired bootstrap CI is.  Both are reported so the reader can judge.",
        "The open-text half is UNTOUCHED, so this claim carries no serving-config exposure at all -- "
        "the open cells cancel exactly between policy and baseline in every frame.",
        "The veto row is cheaper on FLOP-eq (0.977x) and energy (-0.5%) but SLIGHTLY WORSE on "
        "parallel latency (+1.5%), because the 7B pass runs before the 32B one.  It is therefore a "
        "Pareto improvement on accuracy+FLOPs+energy, NOT on latency.  Do not write 'Pareto "
        "dominates' -- that phrasing is already retired in this project (retrospective C26).",
    ],
    honest_sentence=(
        f"Restricting the shipped multiple-choice policy to the multiple-choice half and leaving the "
        f"open-text half at always-32B-direct beats always-32B-direct on the 8-cell macro by "
        f"{vet['delta']:+.4f} [{vet['lo']:+.4f}, {vet['hi']:+.4f}] at "
        f"{rows['MCQ=method_accuracy_max_veto / OPEN=always_32b_direct']['vs_always_32b_direct']['flops']:.3f}x "
        f"its FLOP-eq and {rows['MCQ=method_accuracy_max_veto / OPEN=always_32b_direct']['pct_change']['energy_j']:+.1f}% "
        f"its energy (certified-veto variant), or by {fus['delta']:+.4f} "
        f"[{fus['lo']:+.4f}, {fus['hi']:+.4f}] at "
        f"{rows['MCQ=method_accuracy_max_fusion / OPEN=always_32b_direct']['vs_always_32b_direct']['flops']:.3f}x "
        "its FLOP-eq (fusion variant) -- but the entire effect is PMC-VQA, the policy is identical "
        "to the baseline on the other seven cells by construction, and the underlying MCQ-half "
        "result was already published on 2026-08-05.  It is a real, CI-clean, guardrail-clean win "
        "of +0.0012 to +0.0017 macro; it is not the +0.0029-scale win the round was chasing, and it "
        "comes entirely from the one cell whose split has no published verification."),
    what_it_does_NOT_show=[
        "it does NOT show the open-text machinery helps -- the policy wins BY SWITCHING THE OPEN "
        "MACHINERY OFF.  Everything the shipped method does on the three open cells is what turned "
        "this into a tie in the 2026-08-05 headline.",
        "it does NOT rehabilitate the full accuracy-max arm (0.6575, +0.0008 [-0.0022,+0.0037], TIE) "
        "or the full fusion arm (0.6503, a LOSS).",
        "it does NOT beat always-32B-direct on latency.",
    ],
)

json.dump(d, open(P, "w"), indent=1, default=str)
print(json.dumps(d["cost_measured_all_axes"], indent=1))
print(json.dumps(d["VERDICT"]["rows"], indent=1))
print(d["VERDICT"]["honest_sentence"])
