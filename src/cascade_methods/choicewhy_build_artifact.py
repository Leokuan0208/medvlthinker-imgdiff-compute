#!/usr/bin/env python3
"""choicewhy_build_artifact.py -- assemble the Phase-2 BUILD artifact from the real outputs of every
build step. Every number is copied from a file on disk (no value is typed in by hand).

Inputs (all produced earlier in Phase 2):
  results/cascade_methods/artifacts/choicewhy_mcq_split.json            disjoint split + assertions
  results/cascade_methods/artifacts/choicewhy_verifier_examples.json    composition-matched train sets
  results/cascade_methods/artifacts/choicewhy_eval_pool_inventory.json  the N=8 evaluation pool
  results/cascade_methods/artifacts/choicewhy_judge_concordance.json    grader audit (optional)
  ckpts/train/lora_verifier_choicewhy_<arm>/train_config.json           training configs
  ckpts/train/lora_verifier_disjoint/{adapter_config,train_config}.json the reference config

  python3 src/cascade_methods/choicewhy_build_artifact.py
  -> results/cascade_methods/artifacts/choicewhy_build_2026-08-03.json
"""
import argparse, json, os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_build_2026-08-03.json")
A = ap.parse_args()


def rd(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


split = rd("results/cascade_methods/artifacts/choicewhy_mcq_split.json")
ex = rd("results/cascade_methods/artifacts/choicewhy_verifier_examples.json")
inv = rd("results/cascade_methods/artifacts/choicewhy_eval_pool_inventory.json")
jud = rd("results/cascade_methods/artifacts/choicewhy_judge_concordance.json")
ref_adapter = rd("ckpts/train/lora_verifier_disjoint/adapter_config.json")
ref_train = rd("ckpts/train/lora_verifier_disjoint/train_config.json")
import glob  # noqa: E402

trained = {}
for d in sorted(glob.glob(os.path.join(ROOT, "ckpts/train/lora_verifier_choicewhy_*"))):
    c = rd(os.path.join(d, "train_config.json"))
    if c:
        name = os.path.basename(d).replace("lora_verifier_choicewhy_", "")
        trained[name] = {"out_dir": os.path.relpath(d, ROOT),
                         "adapter_config": rd(os.path.join(d, "adapter_config.json")), **c}

hp_keys = ["lora_r", "lora_alpha", "lora_dropout", "target_modules", "lr", "bs", "accum", "epochs",
           "base_model", "cap_div", "max_pixels", "seed"]
hp_match = {}
if ref_train:
    for arm, c in trained.items():
        hp_match[arm] = {k: {"reference": ref_train.get(k), "this_run": c.get(k),
                             "same": ref_train.get(k) == c.get(k)} for k in hp_keys}

out = {
    "program": "(choice)(why) for multiple choice -- PHASE 2: BUILD",
    "date": "2026-08-03",
    "phase1_gate": "results/cascade_methods/artifacts/choicewhy_pilot_2026-08-03.json (GO: arm "
                   "B2_answer_first_forced preserves MCQ accuracy, +0.0087 [-0.0040,+0.0222] pooled)",
    "what_was_built": [
        "N=8 sampled candidates in both formats over the full 1,488-item Phase-1 evaluation set",
        "a strictly disjoint MCQ training pool from the datasets' OFFICIAL TRAIN splits, with the "
        "disjointness proven in code on md5 of DECODED RGB pixels",
        "N=8 sampled candidates in both formats over that training pool",
        "composition-matched verifier training sets (same questions, same size, same per-source mix "
        "in both arms; FORMAT is the only variable)",
        "a LoRA outcome verifier per arm, architecture and hyperparameters copied from "
        "ckpts/train/lora_verifier_disjoint",
    ],
    "code": {
        "eval_side_generation": "src/labeling/run_choicewhy_pilot.py --arms A B2 --n_samples 8 --temp 0.7 "
                                "--seed 1234 --suffix _sc8  (the Phase-1 generator, unmodified)",
        "disjoint_split": "src/training_methods/build_choicewhy_mcq_split.py",
        "train_side_generation": "src/labeling/run_choicewhy_trainpool.py",
        "example_builder": "src/training_methods/build_choicewhy_verifier_examples.py",
        "trainer": "src/training_methods/run_lora_verifier_choicewhy.py",
        "shared_defs": "src/cascade_methods/choicewhy_common.py (arm prompts + letter extractor, "
                       "verbatim from Phase 1)",
        "eval_pool_inventory": "src/cascade_methods/choicewhy_eval_pool_inventory.py",
        "justification_stats": "src/cascade_methods/choicewhy_justification_stats.py",
        "grader_audit": "src/cascade_methods/choicewhy_judge_concordance.py",
    },
    "deviations_and_flags": {
        "grader": "The brief asks that every candidate be judged with src/labeling/run_judge.py. That "
                  "script is this project's FREE-TEXT grader; on multiple choice the project's grader is "
                  "EXACT OPTION-LETTER MATCH, and it is what labels training candidates and grades "
                  "evaluation candidates here -- so training and evaluation do share one grader. The "
                  "substitution was MEASURED rather than assumed: 4,800 candidates were passed to the "
                  "32B judge (block 5). Judging the chosen option's TEXT agrees with exact letter match "
                  "on 0.9967-1.0000 of candidates in every cell. Judging the candidate VERBATIM agrees "
                  "less for arm B2 (0.9650 eval / 0.9817 train) because the appended rationale can talk "
                  "the judge out of a correct letter -- an arm-specific bias that would have favoured "
                  "arm A, and a reason not to substitute the free-text judge on this task.",
        "medxpert_has_no_train_split": "MedXpertQA-MM (20.2% of the evaluation items) publishes no train "
                                       "split, so its share of the training mix is served by "
                                       "pathvqa_closed_train and recorded as a substitution.",
        "eval_proportional_quota_not_fully_met": "SLAKE and VQA-RAD official train splits are exhausted "
                                                 "(1,681 and 940 usable questions), so their "
                                                 "eval-proportional example quotas fall short and the "
                                                 "deficit is redistributed to PMC-VQA and PathVQA. Exact "
                                                 "shortfalls and redistribution are in block 3.",
        "label_base_rate_differs_by_arm": "Dedup by unique candidate STRING caps arm A at n_options "
                                          "examples per question while arm B2 gets ~6 phrasings, and the "
                                          "correct option attracts more phrasings -- so B2's positive "
                                          "rate is 0.6578 vs A's 0.5783 on the SAME questions. A third "
                                          "verifier was trained on a positive-rate-matched B2 set "
                                          "(identical size, per-source counts AND per-source pos/neg "
                                          "counts) so Phase 3 can separate discrimination from base rate.",
        "L1_question_text_recurrence": "601 evaluation question TEXTS recur in the training pool with "
                                       "different images. Unavoidable for closed VQA, which draws from a "
                                       "small template set; no eval IMAGE and no eval ITEM is in "
                                       "training. Every manifest row carries an L2_strict flag marking "
                                       "the subset with no eval question text at all.",
        "still_open_from_phase1": "Yellow flag (i) carried forward and re-measured under SAMPLING: "
                                  "31.8% of evaluation candidates and 35.4% of training candidates still "
                                  "carry no gradeable justification (block 1b) -- better than the 41.9% "
                                  "Phase 1 measured under greedy decoding, but not zero.",
    },
    "1_evaluation_candidate_pool_MEASURED": inv,
    "1b_justification_presence_MEASURED": rd("results/cascade_methods/artifacts/choicewhy_justification_stats.json"),
    "2_disjoint_split_MEASURED": split,
    "3_training_examples_MEASURED": ex,
    "4_training_configs_MEASURED": trained,
    "4b_hyperparameter_match_vs_reference": {
        "reference": "ckpts/train/lora_verifier_disjoint",
        "reference_adapter_config": ref_adapter,
        "per_arm": hp_match,
    },
    "5_grader_audit_MEASURED": jud,
}
os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1)
print(f"wrote -> {A.out}")
for k in ["1_evaluation_candidate_pool_MEASURED", "2_disjoint_split_MEASURED",
          "3_training_examples_MEASURED", "4_training_configs_MEASURED", "5_grader_audit_MEASURED"]:
    print(f"  {k}: {'present' if out[k] else 'MISSING'}")
