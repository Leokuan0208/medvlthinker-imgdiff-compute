#!/usr/bin/env python3
"""Consolidate the five independent verification passes into one dated artifact."""
import json, os, subprocess
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
P = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
out = {
 "title": "CHEAP INTERVENTIONS 2026-08-17 -- INDEPENDENT ADVERSARIAL VERIFICATION of the three-attack "
          "round, plus the combined 8-cell policy and its permutation null.",
 "date": "2026-08-17",
 "baseline": "always-7B, macro 0.5971 as published (see BASELINE_DEFECT: its 3 open cells are "
             "self-consistency@8, so it is NOT a 1.0 FLOP-eq baseline)",
 "no_fabricated_numbers": True,
 "not_abstention": "every arm returns an answer for every item.",
 "MedEvalKit_untouched": True,
 "independence": "nothing in this artifact is imported from output_bias_correct.py, selfcons_suite.py "
                 "or free_head_lib.py; every per-item ok vector is rebuilt from the raw dumps by "
                 "src/analysis/cheap_interventions_verify.py, cheap_verify_partB/C/D/E/F.py.",
 "numerics_pinned": {"OMP_NUM_THREADS": 1, "nboot": 10000, "nperm": 1000, "seed": 20260817,
                     "TF32": "not used (CPU numpy only)", "no_GPU_job_ran": True},
}
for k, f in (("A_prompt_side_and_balanced_key", "partA.json"),
             ("B_PMC_output_side_and_leakage", "partB.json"),
             ("C_open_cell_baseline_defect", "partC.json"),
             ("D_self_consistency_and_slot_exchangeability", "partD.json"),
             ("E_combined_policy_and_permutation_null", "partE.json"),
             ("F_answer_key_skew_diagnostic", "partF.json"),
             ("G_prior_matching_fit_is_not_converged", "fit_stability.json")):
    out[k] = json.load(open(os.path.join(P, f)))
dst = os.path.join(ROOT, "results/cascade_methods/artifacts/cheap_interventions_verify_2026-08-17.json")
json.dump(out, open(dst, "w"), indent=1)
print(dst, os.path.getsize(dst))
