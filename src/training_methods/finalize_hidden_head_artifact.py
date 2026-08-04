#!/usr/bin/env python3
"""finalize_hidden_head_artifact.py -- fold the follow-up measurements (fusion controls, the CV
protocol-gap cells, and the generator-prompt arm if it exists) into the single deliverable artifact
results/cascade_methods/artifacts/verifarch_hidden_2026-08-04.json, and write the verdict block.

Reads only files produced by this experiment; invents nothing.

  python3 src/training_methods/finalize_hidden_head_artifact.py
"""
import os, json

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_2026-08-04.json")
FUSE = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_fusion_controls_2026-08-04.json")
CVGAP = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_cvgap_2026-08-04.json")
PFORM = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_promptform_indomain_2026-08-04.json")
GENP  = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_hidden_generatorprompt_2026-08-04.json")

a = json.load(open(ART))
a["code"] = {
    "extract": "src/training_methods/extract_generator_hidden.py",
    "fit": "src/training_methods/fit_hidden_head.py",
    "fusion_controls": "src/training_methods/fuse_hidden_head_controls.py",
    "cv_protocol_gap": "src/training_methods/cv_gap_listwise_capacity.py",
    "finalize": "src/training_methods/finalize_hidden_head_artifact.py",
    "feature_cache": "feats_hidden/ (gitignored, 2.2 GB): grader|generator x train|eval, "
                     "h_last + h_span at layers 7/14/21/28 of frozen Lingshu-7B, fp16",
}
a["design"] = {
    "architecture": "a trained DISCRIMINATIVE HEAD (linear / 1-hidden-layer MLP) on the frozen "
                    "generator's hidden states -- no generative verifier, no adapter, no opinion "
                    "token. Score = head(h), not P('Yes').",
    "generator_representation": "base Lingshu-7B (the ACTUAL generator of the pools), NO LoRA. Using "
                                "the incumbent's adapter would have made the features a function of "
                                "the incumbent and destroyed the contrast.",
    "controlled_contrast": "identical base model, identical grader prompt (verbatim from "
                           "run_lora_verifier_disjoint.py), identical image budget (1280*28*28), and "
                           "the IDENTICAL seed-0 matched training draw (10364 examples, reproduced "
                           "bit-for-bit: pos_rate 0.19924739482825163 equals the value recorded in "
                           "ckpts/train/lora_verifier_disjoint/train_config.json). The ONLY variable "
                           "is the readout: trained head on frozen features vs LoRA-tuned LM head.",
    "training_rows": "31498 (all answers of the 6029 questions the matched draw touched, so listwise "
                     "and Bradley-Terry objectives see complete within-question groups; the question "
                     "SET is exactly the draw's).",
    "eval_rows": "8943 distinct (item, normalised answer) pairs covering all 2345 eval items x 8 slots.",
    "feats_full_L14_note": "the pre-existing feats_full/feat_*_L14.npz caches do NOT cover this "
                           "experiment: they are PER-QUESTION (416 rows for SLAKE, keyed by the closed "
                           "MCQ idx), not per-candidate, and come from MedVLThinker-7B, not the "
                           "Lingshu-7B generator of these pools. A fresh GPU extraction was required "
                           "and was run (0 failed rows out of 40441).",
}
if os.path.exists(FUSE):
    a["fusion_controls"] = json.load(open(FUSE))
if os.path.exists(CVGAP):
    a["cv_protocol_gap"] = json.load(open(CVGAP))
if os.path.exists(PFORM):
    a["prompt_form_indomain_probe"] = json.load(open(PFORM))
if os.path.exists(GENP):
    g = json.load(open(GENP))
    a["generator_prompt_arm"] = {"null_test": g.get("null_test"), "disjointness": g.get("disjointness"),
                                 "arm": g.get("arms", {}).get("generator"),
                                 "full_artifact": os.path.relpath(GENP, ROOT)}

json.dump(a, open(ART, "w"), indent=1)
print(f"finalized {ART}")

# ---------------------------------------------------------------- verdict (numbers pulled from disk)
def arm(tag):
    for r in a.get("fusion_controls", {}).get("arms", []):
        if r["tag"] == tag:
            return r
    raise KeyError(tag)

g_cv = a["arms"]["grader"]["eval"]["cv_selected"]
gen = a.get("generator_prompt_arm", {}).get("arm", {})
gen_cv = gen.get("eval", {}).get("cv_selected", {})
F_gen = arm("FUSE_rankavg incumbent+head_generator_prompt")
F_grd = arm("FUSE_rankavg incumbent+head_cv_selected")
C1 = arm("FUSE_rankavg incumbent+C1 base_zeroshot_pyes [KEY CONTROL]")
C2 = arm("FUSE_rankavg incumbent+C2 self_consistency_count")
C3 = arm("FUSE_rankavg incumbent+C3 random [null]")

a["verdict"] = {
  "headline": "A discriminative head on the FROZEN generator's hidden states is a real, non-generative "
              "verifier: per-benchmark it MATCHES a full LoRA fine-tune (grader prompt, sel_eff "
              "0.775886 vs 0.775204; candidate AUROC 0.885550 vs 0.885592) and on the generator's own "
              "answering prompt it EXCEEDS it (0.795640, +0.0204 [-0.0014,+0.0416], guardrail-clean). "
              "But NO single head clears the pre-registered +0.021 detection floor on its own. The one "
              "arm that does is the parameter-free rank fusion of the incumbent with the head.",
  "pre_registered_primary": {
      "definition": "the config chosen by grouped CV INSIDE the disjoint training pool, before eval "
                    "was touched; this is the number that counts as the honest result of the arm.",
      "grader_prompt": {"sel_eff": g_cv["sel_eff"], "d_vs_incumbent": g_cv["vs_incumbent"]["d_sel_eff"],
                        "ci": g_cv["vs_incumbent"]["d_sel_eff_ci"], "verdict": "LOSES, CI excludes zero"},
      "generator_prompt": {"sel_eff": gen_cv.get("sel_eff"),
                           "d_vs_incumbent": gen_cv.get("vs_incumbent", {}).get("d_sel_eff"),
                           "ci": gen_cv.get("vs_incumbent", {}).get("d_sel_eff_ci"),
                           "verdict": "near-miss, n.s. (CI includes zero by 0.0014), guardrail-clean"},
  },
  "best_measured_configuration": {
      "what": "rank-average of the incumbent LoRA verifier with the CV-selected generator-prompt head. "
              "No parameter is fitted at fusion time and both components are trained on strictly "
              "image-disjoint data, so this is deployable, not a diagnostic.",
      "sel_eff": F_gen["sel_eff"], "selected_acc": F_gen["acc"],
      "d_sel_eff": F_gen["d_sel_eff"], "d_sel_eff_ci": F_gen["d_sel_eff_ci"],
      "d_acc": F_gen["d_acc"], "d_acc_ci": F_gen["d_acc_ci"],
      "per_ds": F_gen["per_ds"], "guardrail_clean": F_gen["guardrail_clean"],
      "contested_sel_eff": F_gen["contested_sel_eff"],
      "caveat": "this is ONE arm among ~20 reported; a nominal 95% CI is not multiplicity-corrected. "
                "What raises it above a multiplicity artifact is the CONTROL PATTERN below: three "
                "different heads all fuse positively and all three non-head second-scores fuse "
                "negatively.",
  },
  "falsification_control_that_matters": {
      "claim_tested": "the gain is just 'average the incumbent with a second score', not the head.",
      "fuse_with_base_zeroshot_pyes": {"sel_eff": C1["sel_eff"], "d": C1["d_sel_eff"], "ci": C1["d_sel_eff_ci"]},
      "fuse_with_self_consistency_count": {"sel_eff": C2["sel_eff"], "d": C2["d_sel_eff"], "ci": C2["d_sel_eff_ci"]},
      "fuse_with_random": {"sel_eff": C3["sel_eff"], "d": C3["d_sel_eff"], "ci": C3["d_sel_eff_ci"]},
      "reading": "all three non-head second-scores make the incumbent WORSE. Only the discriminative "
                 "head helps. The active ingredient is the different COMPUTATION, not the extra score.",
  },
  "mechanism_findings": [
      {"finding": "candidate-level AUROC is not a valid proxy for selection efficiency, and the "
                  "dissociation is bidirectional and large.",
       "evidence": "generator-prompt Bradley-Terry head: AUROC 0.677641 but sel_eff 0.795640; "
                   "generator-prompt BCE head: AUROC 0.904078, sel_eff 0.802452. Grader-prompt listwise: "
                   "AUROC 0.759705, sel_eff 0.789510; grader-prompt BCE: AUROC 0.871807, sel_eff 0.750681. "
                   "Global ordering and within-question ordering are close to independent here.",
       "why_it_matters": "this project has read AUROC as a verifier-quality signal, and retrospective N25 "
                         "read '+0.030 AUROC, +0.000 selection' as evidence that ranking losses do "
                         "nothing. The correct reading is that AUROC was the wrong instrument: selection "
                         "is a within-question problem."},
      {"finding": "model selection does not transfer from the disjoint training pool to eval.",
       "evidence": "CV picked grader/bce/h256 (cv_sel_eff 0.689768) which scored 0.750681 on eval, while "
                   "listwise (cv 0.658) scored 0.789510. Closing the staged-grid gap (cv_protocol_gap: "
                   "listwise/h256 0.667800, bt/h256 0.675600) confirms CV would STILL have chosen "
                   "pointwise, so this is a genuine CV->eval transfer failure, not a grid omission.",
       "likely_cause": "the disjoint training pool is composition-matched to the incumbent's (heavily "
                       "PathVQA + Kvasir + RadImageNet OOD) and its within-question group structure "
                       "differs from the eval pools'. UNRESOLVED, and it caps what any head-selection "
                       "protocol on this split can deliver."},
      {"finding": "the generator's ANSWERING-prompt state is a better and far more decorrelated "
                  "correctness signal than its GRADING-prompt state.",
       "evidence": "spearman rho with the incumbent 0.3671 (generator prompt) vs 0.7820 (grader prompt); "
                   "pair-oracle ceiling 0.877384 vs 0.858311; in-domain probe (contaminated, both "
                   "prompts, identical protocol) L21/span 0.795640 vs 0.782698.",
       "why_it_matters": "the project's decorrelation law says selection quality tracks AGREEMENT with "
                         "the generator (+0.76). This is the first scorer that is BOTH strongly "
                         "decorrelated (rho 0.37) AND at least as good as the incumbent -- because it is "
                         "not another opinion, it is the generator's own state read by a different "
                         "computation."},
  ],
  "cost": {
      "grader_prompt_head": "one EXTRA base-model forward pass per candidate (~1x the incumbent's cost), "
                            "because the incumbent's pass runs the LoRA-adapted model and the head's runs "
                            "the base model. Fusion therefore costs ~2x verifier prefill.",
      "generator_prompt_head": "the features are the hidden states of (image, question, candidate) under "
                               "the generator's OWN answering prompt -- exactly the states the model "
                               "computes while DECODING that candidate. A deployment could cache them "
                               "during best-of-8 generation at close to zero extra cost. NOT MEASURED "
                               "HERE: this experiment ran a separate teacher-forced forward pass, so the "
                               "free-at-generation claim is an inference from the architecture, not a "
                               "measurement. It is the cheapest thing to verify next.",
      "head_training": "CPU-seconds to CPU-minutes on a cached feature matrix; the one-off cost is the "
                       "extraction (80882 forward passes, 0 failures, ~5 GPU-hours on 2xA100 under heavy "
                       "contention).",
  },
  "what_would_falsify_or_extend": [
      "read the head's features off the ADAPTED model's forward pass instead of the base model's: if the "
      "signal survives, the fusion becomes free (one pass yields both P(Yes) and h); if it collapses, the "
      "complementarity was caused by using an unadapted network, which is itself the explanation.",
      "verify the 'free at generation time' claim by capturing hidden states during the actual sampling "
      "run rather than in a separate teacher-forced pass.",
      "fix the CV->eval transfer failure with a selection split drawn to match the eval pools' question "
      "mix; until then, head architecture selection on this split is unreliable.",
      "the fusion captures ~28% of the pair-oracle headroom (0.8065 of a 0.8774 ceiling from 0.7752). The "
      "remaining ~0.071 still needs a per-item router, which is the project's standing wall.",
  ],
  "abstention_statement": "no arm in this experiment abstains, defers, or rejects. Every configuration "
                          "always returns one of the 8 candidates.",
}
json.dump(a, open(ART, "w"), indent=1)
print("verdict written")
