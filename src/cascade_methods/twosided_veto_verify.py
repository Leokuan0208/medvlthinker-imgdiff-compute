#!/usr/bin/env python3
"""ATTACK C -- post-hoc VERIFICATION of twosided_veto_2026-08-11.json.

Written 2026-08-12 after the 2026-08-11 session was killed by a session limit.
The main artifact survived and is COMPLETE; this script does three things it does not do:

  (1) INDEPENDENT re-derivation of the artifact's INPUTS from the canonical per-arm
      vectors (vec_disjoint.npz), rather than trusting the artifact's self-report:
      per-cell always-7B / always-32B-direct / shipped accuracy-max, the macro
      baselines, and the p10 = P(7B right AND 32B wrong) arbitration ceiling.
  (2) The OVERFITTING-GAP diagnosis: macro oracle vs macro in-sample upper bound
      (fit WITH full eval visibility) vs honest nested vs leave-one-cell-out.
  (3) VERDICT RECONCILIATION.  The artifact records BOTH
      HEADLINE.beats_always_32B_direct = true AND HEADLINE.kill_criterion_met = true.
      Those two flags are in tension and the session died before reconciling them.

NO new inference, NO GPU, NO new measurement: every number is either read from the
canonical vectors or is arithmetic on numbers already inside the 2026-08-11 artifact.
The 2026-08-11 artifact is READ ONLY and is NOT modified.

Run from the repo root:  python3 src/cascade_methods/twosided_veto_verify.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
MAIN = os.path.join(ART, "twosided_veto_2026-08-11.json")
VEC = os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz")
OUT = os.path.join(ART, "twosided_veto_2026-08-11_verification.json")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed",
         "MedXpertQA-MM", "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

# Published bar, from cascade_selector_rerun_2026-08-05.json (arm "disjoint").
BRIEF = {"always_7b": 0.5971, "always_32b_direct": 0.6567,
         "accuracy_max": 0.6575, "shipped_delta": 0.0008,
         "sigma_p10": 0.529, "macro_ceiling": 0.0661}


def main():
    d = json.load(open(MAIN))
    z = np.load(VEC, allow_pickle=True)
    pc = d["part3_honest_realised_gain"]["per_cell"]
    loco = d["part3b_leave_one_cell_out"]["per_cell"]
    p2 = d["part2_identifiability"]

    # ---- (1) independent re-derivation of the inputs -------------------------
    rows, m7, m32, mam, p10s, devs = {}, [], [], [], [], []
    for c in CELLS:
        a7 = z[c + "|always_7b"].astype(int)
        a32 = z[c + "|always_32b_direct"].astype(int)
        am = z[c + "|method_accuracy_max_veto"].astype(int)
        p10 = float(((a7 == 1) & (a32 == 0)).mean())
        art_p10 = pc[c]["oracle_gain"]
        devs.append(abs(p10 - art_p10))
        m7.append(float(a7.mean())); m32.append(float(a32.mean()))
        mam.append(float(am.mean())); p10s.append(p10)
        rows[c] = {"n": int(len(a7)),
                   "always_7b": round(float(a7.mean()), 4),
                   "always_32b_direct": round(float(a32.mean()), 4),
                   "shipped_accuracy_max": round(float(am.mean()), 4),
                   "p10_rederived": round(p10, 6),
                   "p10_in_artifact": art_p10,
                   "abs_dev": round(abs(p10 - art_p10), 8)}

    macro = {"always_7b": round(float(np.mean(m7)), 4),
             "always_32b_direct": round(float(np.mean(m32)), 4),
             "shipped_accuracy_max": round(float(np.mean(mam)), 4),
             "shipped_delta_vs_direct": round(float(np.mean(mam) - np.mean(m32)), 5),
             "sigma_p10": round(float(np.sum(p10s)), 4),
             "macro_oracle_ceiling": round(float(np.sum(p10s)) / 8, 4)}
    null = {"what": "this script's re-derivation from vec_disjoint.npz vs the "
                    "2026-08-11 artifact and the briefed published bar",
            "max_abs_deviation_p10_percell": round(float(max(devs)), 8),
            "macro_baselines_match_brief": {
                "always_7b": macro["always_7b"] == BRIEF["always_7b"],
                "always_32b_direct": macro["always_32b_direct"] == BRIEF["always_32b_direct"],
                "shipped_accuracy_max": macro["shipped_accuracy_max"] == BRIEF["accuracy_max"],
                "macro_oracle_ceiling": macro["macro_oracle_ceiling"] == BRIEF["macro_ceiling"]},
            "shipped_delta_rederived": macro["shipped_delta_vs_direct"],
            "shipped_delta_in_brief": BRIEF["shipped_delta"],
            "passed": bool(max(devs) < 1e-4
                           and macro["always_32b_direct"] == BRIEF["always_32b_direct"]
                           and macro["always_7b"] == BRIEF["always_7b"])}

    # ---- (2) the overfitting-gap diagnosis -----------------------------------
    O, U, N, L, tab = [], [], [], [], {}
    for c in CELLS:
        o = pc[c]["oracle_gain"]
        u = pc[c]["insample_upper_bound"]["gain"]
        ci = pc[c]["ci_on_seed_averaged_per_item_gain"]
        lc = loco[c]["gain"]
        O.append(o); U.append(u); N.append(ci["mean"]); L.append(lc)
        tab[c] = {"n_decisive": pc[c]["n_decisive"],
                  "base_rate_pi": p2[c].get("base_rate_pi"),
                  "best_auroc_one_sided_7B_only": p2[c]["best_one_sided_7B_only"]["auroc"],
                  "best_auroc_using_the_32B_side": p2[c]["best_signal_using_the_32B_side"]["auroc"],
                  "two_sided_auroc_advantage": p2[c]["two_sided_advantage_auroc"],
                  "oracle_gain": o,
                  "insample_upper_bound_DIAGNOSTIC": u,
                  "honest_nested_gain": ci["mean"],
                  "honest_nested_ci": [ci["lo"], ci["hi"]],
                  "sig": ci["sig"],
                  "leave_one_cell_out_gain": lc}

    mo, mu, mn, ml = float(np.mean(O)), float(np.mean(U)), float(np.mean(N)), float(np.mean(L))
    sig_pos = [c for c in CELLS
               if pc[c]["ci_on_seed_averaged_per_item_gain"]["sig"]
               and pc[c]["ci_on_seed_averaged_per_item_gain"]["mean"] > 0]
    sig_neg = [c for c in CELLS
               if pc[c]["ci_on_seed_averaged_per_item_gain"]["sig"]
               and pc[c]["ci_on_seed_averaged_per_item_gain"]["mean"] < 0]

    diag = {
        "macro_oracle_ceiling": round(mo, 4),
        "macro_insample_upper_bound_DIAGNOSTIC": round(mu, 4),
        "macro_honest_nested": round(mn, 4),
        "macro_leave_one_cell_out": round(ml, 4),
        "conversion_honest_over_oracle_pct": round(100 * mn / mo, 1),
        "conversion_honest_over_insample_pct": round(100 * mn / mu, 1),
        "overfitting_gap_insample_over_honest": round(mu / mn, 1),
        "cells_sig_positive": sig_pos,
        "cells_sig_negative": sig_neg,
        "macro_contribution_of_the_2_sig_positive_cells":
            round(sum(pc[c]["ci_on_seed_averaged_per_item_gain"]["mean"] for c in sig_pos) / 8, 5),
        "macro_contribution_of_the_other_6_cells":
            round(sum(pc[c]["ci_on_seed_averaged_per_item_gain"]["mean"]
                      for c in CELLS if c not in sig_pos) / 8, 5),
        "mean_two_sided_auroc_advantage": round(
            float(np.mean([p2[c]["two_sided_advantage_auroc"] for c in CELLS])), 4),
        "cells_where_two_sided_beats_one_sided_auroc": sum(
            1 for c in CELLS if p2[c]["two_sided_advantage_auroc"] > 0),
        "reading": (
            "The PREMISE of Attack C is CONFIRMED at the signal level: adding the 32B's own "
            "confidence profile raises the best available AUROC on 8/8 cells (mean +0.094). "
            "It does NOT convert to accuracy. Even fitting the arbiter WITH FULL EVAL "
            "VISIBILITY (in-sample, the most optimistic quantity that exists) reaches only "
            "+0.0339 macro, 51% of the +0.0661 oracle; cross-fitting honestly collapses that "
            "to +0.0026, a 13.1x overfitting gap, and removing within-cell fitting entirely "
            "(leave-one-cell-out) makes it NEGATIVE at -0.0066. The discrimination is measured "
            "on decisive-disagreement sets of 36-554 items on six of the eight cells, and it "
            "does not survive cross-fitting there."),
    }

    # ---- (3) verdict reconciliation ------------------------------------------
    H = d["HEADLINE"]
    verdict = {
        "the_tension": (
            "twosided_veto_2026-08-11.json records HEADLINE.beats_always_32B_direct = true "
            "AND HEADLINE.kill_criterion_met = true. The 2026-08-11 session was killed before "
            "reconciling them. This block reconciles them; it does not change any measured number."),
        "success_criterion_as_briefed": "macro delta vs always-32B-direct with CI excluding zero at a stated cost",
        "success_criterion_literally_met": True,
        "kill_criterion_as_briefed": "cross-fit arbiter gain CI covers zero on >=2 of the cells where it fires",
        "kill_criterion_met": H["kill_criterion_met"],
        "cells_where_it_fires_and_its_CI_covers_zero": H["cells_where_it_fires_and_its_CI_covers_zero"],
        "why_the_kill_criterion_wins": [
            "PRE-REGISTERED KILL. The kill criterion was stated in the brief BEFORE the run and it "
            "fired on 4 cells, twice the threshold of 2. A pre-registered kill that is argued past "
            "after seeing the result is not a pre-registration.",
            "GUARDRAIL FAILS. Methodology rule 6 is 'never worse on any single cell'. Two cells are "
            "SIGNIFICANTLY worse: MedXpertQA-MM -0.0036 [-0.0056,-0.0016] and SLAKE_open "
            "-0.0019 [-0.0033,-0.0006]. These are not the small-n vqa_rad cells the standing caveat "
            "excuses; MedXpert is n=2000 and SLAKE_open n=645.",
            "SEED FRAGILITY. Methodology rule 4: a single seed is not a result. Only 3/10 fold seeds "
            "have their own 95% lower bound above zero, and 1/10 is negative (-0.00039). The headline "
            "CI bootstraps the SEED-AVERAGED per-item gain, which removes fit-to-fit variance a "
            "deployed system actually carries; the artifact flags this itself.",
            "BELOW THE STATED BAR. The briefed significant-win threshold is macro +0.0029; the "
            "measured +0.0026 is under it, with a lower bound of +0.0001.",
            "DOES NOT TRANSFER. Leave-one-cell-out, the only arm with zero within-cell fitting, is "
            "-0.0066 [-0.0132,-0.0002], a SIGNIFICANT LOSS.",
            "REDISCOVERY. PRIOR_ART in the main artifact judges Attack C 'substantially a "
            "rediscovery' of beat32b_fusion.py F3/F5, whose published outcome was already "
            "'wins on PMC-VQA only'. This round reproduces that shape with a richer feature set.",
        ],
        "VERDICT": "NEGATIVE -- KILLED BY ITS OWN PRE-REGISTERED CRITERION",
        "beats_always_32B_direct_defensible": False,
        "what_may_be_quoted": (
            "Post-hoc two-sided arbitration converts on exactly 2 of 8 cells -- PMC_VQA "
            "+0.0126 [+0.0095,+0.0158] and PATH_VQA_open +0.0263 [+0.0147,+0.0383], both "
            "seed-stable and both surviving their sensitivity check -- and the other 6 cells "
            "give it back (-0.00228 macro), two of them significantly. The macro is +0.0026 "
            "[+0.0001,+0.0051] at 1.219x always-32B-direct FLOP-eq (macro weighting), which "
            "clears zero but fails the guardrail, fails the pre-registered kill criterion, is "
            "individually significant in only 3/10 seeds, and reverses to -0.0066 without "
            "within-cell fitting. It must NOT be reported as beating always-32B-direct."),
        "what_must_NOT_be_quoted": [
            "'two-sided arbitration beats always-32B-direct' -- the guardrail and the "
            "pre-registered kill criterion both say no",
            "the in-sample upper bounds (+0.0339 macro) as anything but a DIAGNOSTIC ceiling",
            "any macro accuracy paired with a sample-weighted cost",
        ],
    }

    out = {
        "title": "ATTACK C verification, overfitting-gap diagnosis, and verdict reconciliation",
        "date": "2026-08-12",
        "reproduce": "python3 src/cascade_methods/twosided_veto_verify.py",
        "verifies": "results/cascade_methods/artifacts/twosided_veto_2026-08-11.json",
        "main_artifact_is_modified": False,
        "no_gpu": True,
        "no_new_inference": True,
        "sources": {
            "canonical_vectors": "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
            "published_bar": "results/cascade_methods/artifacts/cascade_selector_rerun_2026-08-05.json (arm disjoint)",
            "attack_c": "results/cascade_methods/artifacts/twosided_veto_2026-08-11.json",
        },
        "convention": "MACRO, equal weight per reporting cell, Variant B (MMMU excluded), "
                      "8 cells, 1/8 each, n=42,224; CLEAN disjoint open-text verifier",
        "part1_independent_input_verification": {
            "per_cell": rows, "macro": macro, "null_test": null,
            "note": "the 4 null tests INSIDE the 2026-08-11 artifact (cascade_rerun, "
                    "rebuilt_vectors 84,448 items, open_text_bar sel_eff, prior_art_F5) all "
                    "record max_abs_deviation 0.0 / 3.6e-07 and passed:true; this block is an "
                    "INDEPENDENT check of the same inputs from the canonical vectors.",
        },
        "part2_overfitting_gap_diagnosis": {"per_cell": tab, "macro": diag},
        "part3_verdict_reconciliation": verdict,
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print("null test passed:", null["passed"], "max_abs_dev", null["max_abs_deviation_p10_percell"])
    print("macro oracle %.4f | insample %.4f | honest %.4f | LOCO %.4f"
          % (mo, mu, mn, ml))
    print("overfitting gap %.1fx" % (mu / mn))
    print("VERDICT:", verdict["VERDICT"])


if __name__ == "__main__":
    main()
