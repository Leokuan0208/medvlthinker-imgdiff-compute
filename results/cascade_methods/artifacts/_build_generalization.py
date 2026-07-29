#!/usr/bin/env python3
"""Build generalization.json for cross-family external validity of Findings 1/2/3.
OFFLINE. Finding-1 numbers recomputed directly from master_data.csv (source of truth).
Findings 2/3 evidence copied verbatim from the cited archive/current docs (no new compute).
No fabricated numbers.
"""
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "master_data.csv")

PERC = ["PMC", "SLAKE", "VQARAD", "PathV"]      # perception regime
REAS = ["MMMU", "MX-R", "MX-U"]                 # reasoning regime
FAMS = ["medvlthinker", "lingshu", "qoq", "chiron", "medgemma"]

# ---- load ALL-6 big-nt and big-think rows per family ----
rows = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["pool"] != "ALL-6":
            continue
        rows.setdefault(r["family"], {})[r["method"]] = r

def f6(v):
    return round(float(v), 4)

finding1 = {}
perc_le0_strict = 0
perc_le0_noise = 0   # |d|<=0.02 counted as "not a perception win"
perc_total = 0
for fam in FAMS:
    bignt = rows[fam]["always-big-nt"]
    bigth = rows[fam]["always-big-think [PARITY]"]
    lat_ratio = round(float(bigth["latency_s"]) / float(bignt["latency_s"]), 1)
    per_bench = {}
    for b in PERC + REAS:
        d = round(float(bigth[b]) - float(bignt[b]), 4)
        regime = "perception" if b in PERC else "reasoning"
        per_bench[b] = {"regime": regime, "nothink": f6(bignt[b]),
                        "think": f6(bigth[b]), "dThink_minus_nt": d}
        if regime == "perception":
            perc_total += 1
            if d <= 0:
                perc_le0_strict += 1
            if d <= 0.02:
                perc_le0_noise += 1
    finding1[fam] = {
        "think_over_nt_latency_ratio": lat_ratio,
        "bignt_latency_s": round(float(bignt["latency_s"]), 3),
        "bigthink_latency_s": round(float(bigth["latency_s"]), 3),
        "per_benchmark_dThink_minus_nt": per_bench,
    }

# perception wins (think > nt strictly, beyond noise band)
perc_wins = []
for fam in FAMS:
    for b in PERC:
        d = finding1[fam]["per_benchmark_dThink_minus_nt"][b]["dThink_minus_nt"]
        if d > 0.02:
            perc_wins.append(f"{fam}:{b} (+{d})")

out = {
  "_meta": {
    "title": "Cross-family external validity of the paper's key findings (OFFLINE, no GPU)",
    "date": "2026-07-08",
    "sources": {
      "finding1": "results/cascade_methods/artifacts/master_data.csv (ALL-6, NGC harness, 5 medical VLM families); "
                  "recomputed here as dThink = (always-big-think per-benchmark) - (always-big-nt per-benchmark). "
                  "Supplementary 2 non-medical architectures from artifacts/overthink_generalize.txt. "
                  "Faithful reasoning-regime deltas from reframe_vs_bigthink.json -> MedEvalKit eval_results_*.",
      "finding2": "docs/archive_mcq/OPENENDED_CASCADE.md §2c; docs/current/RESEARCH_RESULTS_2026-07.md §1.5 gate bake-off; "
                  "docs/current/VERIFIED_FACTS.md.",
      "finding3": "docs/archive_mcq/TRAINED_VERIFIER_RESULT.md; docs/current/VERIFIED_FACTS.md §F/H/I; "
                  "docs/current/RESEARCH_RESULTS_2026-07.md §1.3."
    },
    "no_fabrication": "Finding-1 deltas recomputed from master_data.csv; Findings 2/3 quoted verbatim from cited docs. No new GPU runs."
  },

  "finding1_reasoning_vs_perception": {
    "claim": "Reasoning (think) HURTS or is flat on perception VQA (SLAKE/VQA-RAD/PathVQA/PMC) and HELPS on reasoning (MMMU/MedXpert). External validity = does the sign pattern hold across model families, not just Lingshu.",
    "per_family": finding1,
    "consistency_perception": {
      "definition": "perception cell = one family x one perception benchmark; dThink_minus_nt <= 0 means think does NOT help (premise holds).",
      "n_perception_cells": perc_total,
      "n_le_0_strict": perc_le0_strict,
      "n_le_0_within_noise(<=+0.02)": perc_le0_noise,
      "perception_cells_where_think_WINS(>+0.02)": perc_wins,
      "read": f"{perc_le0_strict}/{perc_total} perception cells have think<=no-think strictly; "
              f"{perc_le0_noise}/{perc_total} within a +/-0.02 noise band. "
              "VQA-RAD is negative for ALL 5 families; SLAKE negative for 4/5 (MedGemma +0.005 ~ 0). "
              "The only genuine perception win is MedGemma:PathVQA (+0.040)."
    },
    "consistency_reasoning": {
      "read": "Think helps reasoning for the GENUINE-think families: MedVLThinker (MMMU +0.065, MX-R +0.047, MX-U +0.092) and QoQ (MMMU +0.071). "
              "Muted/absent where the model has no promptable think mode (Lingshu: MMMU -0.012, latency ratio 1.2x -> it answers directly) "
              "or inverse-scales (Chiron: MMMU 0.0). Corroborated on a faithful harness below."
    },
    "faithful_reasoning_regime_MedEvalKit": {
      "source": "reframe_vs_bigthink.json -> MedEvalKit eval_results_{lingshu32b,mvt32b,iv3_38b}_reason",
      "MMMU_150": {"lingshu32b": 0.027, "mvt32b": 0.10, "iv3_38b": 0.12},
      "MedXpert_2000": {"lingshu32b": -0.003, "mvt32b": 0.045, "iv3_38b": 0.031},
      "read": "think>no-think on reasoning for MVT and InternVL3-38B (3rd family); Lingshu tiny/none (no real think mode)."
    },
    "supplementary_non_medical_architectures": {
      "source": "artifacts/overthink_generalize.txt (POOLED over 4 perception benchmarks; separate run, signs agree, magnitudes differ from NGC harness)",
      "InternVL2.5-8B": {"POOLED_perc_dThink": -0.008, "note": "3/4 perception benchmarks negative"},
      "Phi-3.5-V": {"POOLED_perc_dThink": -0.019, "note": "4/4 perception benchmarks negative"},
      "read": "Two additional distinct architectures (InternVL2.5, Phi-3.5-V, general-domain) both POOLED-negative on perception -> premise is not Qwen/medical-specific."
    },
    "verdict": "STRONGLY CROSS-FAMILY on the perception half (5 medical families + 2 extra architectures; 15/20 cells <=0 strict, 19/20 within noise; "
               "VQA-RAD negative in all 5, SLAKE 4/5). The reasoning half (think helps) is cross-family for genuine-think models "
               "(MedVLThinker, InternVL3, QoQ-on-MMMU) but muted where no real think mode exists (Lingshu) or the model inverse-scales (Chiron)."
  },

  "finding2_format_signal_gap": {
    "claim": "The routing-signal ceiling (~0.6 AUROC on MCQ) is a benchmark artifact; open-ended free-text unlocks a much stronger 'when to trust the cheap model' signal (~0.87 AUROC).",
    "mcq_ceiling_evidence": {
      "source": "RESEARCH_RESULTS_2026-07.md §1.5 gate bake-off, MedVLThinker competent-4 MCQ (cheap 0.622 / strong 0.645)",
      "AUROC_detect_range": "0.643 (7B self-verify) .. 0.693 (learned-RICH)",
      "AUROC_recover_range": "0.506 .. 0.614",
      "read": "Every gate (margin/maxprob/entropy/self-verify/learned) saturates near the ~0.6 wall on MCQ. "
              "Cross-family corroboration: in the 5-family ACC bake-off all 10 gate methods cluster within ~0.003 accuracy per family "
              "(2SIZE_VALIDATION.md), i.e. no gate separates -> MCQ signal is weak across families (indirect)."
    },
    "opentext_signal_evidence": {
      "source": "OPENENDED_CASCADE.md §2c (AUROC for 'cheap 7B is WRONG', pooled SLAKE-open+VQA-RAD-open)",
      "MedVLThinker-7B_cheap": {"confidence_AUROC": 0.735, "self_consistency_AUROC": 0.781},
      "Lingshu-7B_cheap": {"confidence_AUROC": 0.866, "self_consistency_AUROC": 0.845},
      "verifier_discrimination_AUROC": {"value": 0.924, "n": 8512, "source": "VERIFIED_FACTS.md line 31"},
      "read": "Open-text routing AUROC is 0.735-0.781 for MedVLThinker-7B (MCQ-RL-tuned, miscalibrated) and 0.845-0.866 for Lingshu-7B "
              "(natively calibrated) - BOTH well above the ~0.6 MCQ ceiling. Answers are median 1-2 tokens, so the driver is 4-option "
              "discreteness, NOT answer length."
    },
    "verdict": "CROSS-FAMILY IN DIRECTION (open-text signal >> MCQ ~0.6 for BOTH MedVLThinker-7B [0.735-0.781] and Lingshu-7B [0.845-0.866]). "
               "The PEAK ~0.87 magnitude is Lingshu-specific (calibration-dependent): MedVLThinker's miscalibration caps it lower, but still "
               "clears the MCQ wall. The MCQ ~0.6 saturation itself is primarily established on MedVLThinker (the deployed family), with "
               "indirect cross-family support from the all-gates-cluster result."
  },

  "finding3_trained_verifier_bestofN": {
    "claim": "A trained best-of-N outcome verifier beats training-free selection (self-consistency / zero-shot P(True)) and matches/beats the 32B on open-ended VQA.",
    "headline_lingshu": {
      "source": "TRAINED_VERIFIER_RESULT.md + VERIFIED_FACTS.md §F/I",
      "base": "LoRA-finetuned Lingshu-7B verifier, 4 datasets, n=1064 grouped held-out",
      "greedy": 0.413, "self_consistency": 0.411, "trained_verifier": 0.501, "oracle@8": 0.592,
      "gap_captured_pct": 49,
      "vs_32B": "verifier 0.501 vs 32B same-split 0.462 (+0.039 [+0.010,+0.066] seed0); TIE on seed1 -> HONEST: matches/competitive, not 'beats' unconditionally"
    },
    "cross_family_medvlthinker": {
      "source": "VERIFIED_FACTS.md §H (MedVLThinker-7B verifier, from scratch, SLAKE+VQA-RAD)",
      "SLAKE": {"greedy": 0.564, "trained": 0.622, "gap_captured_pct": 42, "note": "works"},
      "VQARAD": {"greedy": 0.500, "trained": 0.470, "note": "FAILS (n=54 noisy)"},
      "POOLED": {"greedy": 0.547, "trained": 0.583, "gap_captured_pct": 25, "note": "positive but weaker than Lingshu (49%)"},
      "read": "The method transfers to the ORIGINAL MedVLThinker family (positive pooled) but non-uniformly; base-model quality matters. "
              "The Lingshu-trained verifier even transfers ONTO MedVLThinker outputs better (49-61%) than a from-scratch MVT verifier (25%)."
    },
    "bestofN_signal_on_medvlthinker": {
      "source": "OPENENDED_CASCADE.md §2c",
      "note": "self-consistency / answer-diversity best-of-N error-detection was demonstrated on MedVLThinker-7B (SC AUROC 0.781 > confidence 0.735) - the original family."
    },
    "generator_portfolio_multifamily": {
      "source": "RESEARCH_RESULTS_2026-07.md §1.3",
      "generators": ["Lingshu-7B", "MedVLThinker-7B", "InternVL3-8B"],
      "note": "The best-of-N cheap-generator pool spans 3 families."
    },
    "verdict": "CROSS-FAMILY BUT BASE-QUALITY-DEPENDENT. Trained verifier demonstrated on TWO base families - Lingshu-7B (headline, 49% of oracle gap, "
               "4 datasets, 2 seeds) and MedVLThinker-7B (from-scratch, 25% pooled, partial). The robust cross-family claim is 'training beats "
               "training-free selection and zero-shot self-verify'; the 'beats the 32B' claim is seed-dependent even on Lingshu (downgrade to matches/competitive). "
               "Best-of-N selection signal + 3-family generator portfolio confirm the mechanism is not Lingshu-only."
  }
}

with open(os.path.join(HERE, "generalization.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---- print human-readable table ----
print("=== FINDING 1: dThink = think - no-think (ALL-6, master_data.csv) ===")
hdr = "family        " + "".join(f"{b:>9}" for b in PERC+REAS) + "   lat(th:nt)"
print(hdr); print("-"*len(hdr))
for fam in FAMS:
    d = finding1[fam]["per_benchmark_dThink_minus_nt"]
    line = f"{fam:<13}" + "".join(f"{d[b]['dThink_minus_nt']:+9.3f}" for b in PERC+REAS)
    line += f"     {finding1[fam]['think_over_nt_latency_ratio']:>5.1f}x"
    print(line)
print("-"*len(hdr))
print(f"Perception cells think<=no-think: {perc_le0_strict}/{perc_total} strict; {perc_le0_noise}/{perc_total} within +/-0.02.")
print(f"Perception WINS (think>+0.02): {perc_wins}")
print("\nWrote generalization.json")
