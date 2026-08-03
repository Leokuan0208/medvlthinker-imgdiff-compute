#!/usr/bin/env python3
"""choicewhy_judge_concordance.py -- audit the MCQ label rule against the project's LLM judge.

WHY THIS EXISTS.  The Phase-2 brief requires that training labels and evaluation labels come from the
SAME grader.  On multiple choice this project's grader is EXACT OPTION-LETTER MATCH -- that is what
scores every MCQ number in the repo (src/labeling/run_vlm_eval.py, ckpts/gate_lingshu7b_mcq, and the
Phase-1 pilot), and it is applied identically to the training candidates and the evaluation candidates,
so the same-grader requirement is met by construction.  src/labeling/run_judge.py is the grader for
FREE-TEXT answers, where no exact match is possible; it is not the grader of any MCQ number here.

Rather than assert that substitution is harmless, this script MEASURES it: it feeds a sample of MCQ
candidates to the very same judge (MedVLThinker-32B, JUDGE_SYS verbatim, temperature 0) and reports how
often the judge's verdict differs from exact letter match.

  --prep   writes judge-input jsonls {idx, question, gold, modal_pred} for two readings:
             optiontext : model answer = the TEXT of the option the candidate chose
                          -> isolates the GRADER (letter match vs judged text equivalence)
             full       : model answer = the candidate string verbatim, justification included
                          -> tests whether the (choice)(why) rationale can talk the judge out of a
                             correct letter (an arm-specific risk that arm A cannot have)
  --score  reads the judge outputs and reports agreement, per arm and per split

  python3 src/cascade_methods/choicewhy_judge_concordance.py --prep
  HF_HOME=/data/dan/hf_cache python3 src/labeling/run_judge.py --tp 2 --preds \
      ckpts/choicewhy_judge_audit/*.jsonl
  python3 src/cascade_methods/choicewhy_judge_concordance.py --score
"""
import argparse, glob, json, os, random, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from choicewhy_common import ARM_NAME, extract, parse_opts  # noqa: E402

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--prep", action="store_true")
ap.add_argument("--score", action="store_true")
ap.add_argument("--eval_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--train_dir", default="ckpts/choicewhy_train")
ap.add_argument("--out_dir", default="ckpts/choicewhy_judge_audit")
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--n_per_cell", type=int, default=600, help="candidates sampled per (split, arm, reading)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_judge_concordance.json")
A = ap.parse_args()
os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)

EVAL_BENCH = ["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"]


def eval_records(arm):
    """Phase-1 evaluation candidates. The option block lives in MedVLThinker-Eval, so it is loaded here."""
    from datasets import load_dataset
    data = load_dataset("/data/dan/dataset/MedVLThinker-Eval")["test"]
    out = []
    for b in EVAL_BENCH:
        p = os.path.join(ROOT, A.eval_dir, f"ckpt_{b}_{ARM_NAME[arm]}_sc8.jsonl")
        if not os.path.exists(p):
            continue
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            ex = data[r["idx"]]
            opts = parse_opts(ex["options"])
            for si, s in enumerate(r["raw_outputs"]):
                out.append({"key": f"{b}#{r['idx']}#{si}", "bench": b, "question": ex["question"],
                            "options": opts, "gold": r["gold"], "cand": s.strip()})
    return out


def train_records(arm):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, A.train_dir, f"ckpt_*_{ARM_NAME[arm]}_sc8.jsonl"))):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            for si, s in enumerate(r["raw_outputs"]):
                out.append({"key": f"{r['src']}#{r['idx']}#{si}", "bench": r["src"], "question": r["question"],
                            "options": r["options"], "gold": r["gold"], "cand": s.strip()})
    return out


def fname(split, arm, reading):
    return os.path.join(ROOT, A.out_dir, f"judge_{split}_{ARM_NAME[arm]}_{reading}.jsonl")


if A.prep:
    for split, fn in [("eval", eval_records), ("train", train_records)]:
        for arm in A.arms:
            recs = fn(arm)
            if not recs:
                print(f"  (no {split} records for arm {arm})", flush=True)
                continue
            rng = random.Random(A.seed)
            rng.shuffle(recs)
            sel = recs[:A.n_per_cell]
            for reading in ("optiontext", "full"):
                with open(fname(split, arm, reading), "w") as fh:
                    for r in sel:
                        letter, ok, _ = extract(r["cand"], ARM_NAME[arm])
                        chosen = r["options"].get(letter, "") if ok else ""
                        pred = chosen if reading == "optiontext" else r["cand"]
                        qq = r["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in r["options"].items())
                        fh.write(json.dumps({
                            "idx": r["key"], "question": qq,
                            "gold": r["options"].get(r["gold"], r["gold"]),
                            "modal_pred": pred,
                            "_letter": letter, "_parse_ok": ok, "_bench": r["bench"],
                            "_exact": int(ok and letter == r["gold"])}) + "\n")
                print(f"wrote {fname(split, arm, reading)} ({len(sel)} candidates)", flush=True)

if A.score:
    res = {}
    for f in sorted(glob.glob(os.path.join(ROOT, A.out_dir, "judge_*.jsonl"))):
        if f.endswith(".judge.jsonl"):
            continue
        jf = f.replace(".jsonl", ".judge.jsonl")
        if not os.path.exists(jf):
            print(f"  !! no judge output for {os.path.basename(f)}", flush=True)
            continue
        rows = {json.loads(l)["idx"]: json.loads(l) for l in open(f) if l.strip()}
        jud = {json.loads(l)["idx"]: json.loads(l)["judge_ok"] for l in open(jf) if l.strip()}
        both = [k for k in rows if k in jud]
        agree = sum(1 for k in both if rows[k]["_exact"] == jud[k])
        ex_pos = sum(rows[k]["_exact"] for k in both)
        j_pos = sum(jud[k] for k in both)
        cell = os.path.basename(f).replace("judge_", "").replace(".jsonl", "")
        res[cell] = {"n": len(both), "agreement": round(agree / max(1, len(both)), 4),
                     "exact_match_pos_rate": round(ex_pos / max(1, len(both)), 4),
                     "judge_pos_rate": round(j_pos / max(1, len(both)), 4),
                     "judge_yes_exact_no": sum(1 for k in both if jud[k] == 1 and rows[k]["_exact"] == 0),
                     "judge_no_exact_yes": sum(1 for k in both if jud[k] == 0 and rows[k]["_exact"] == 1),
                     "per_bench_agreement": {b: round(
                         sum(1 for k in both if rows[k]["_bench"] == b and rows[k]["_exact"] == jud[k]) /
                         max(1, sum(1 for k in both if rows[k]["_bench"] == b)), 4)
                         for b in sorted({rows[k]["_bench"] for k in both})}}
        r = res[cell]
        print(f"{cell:52s} n={r['n']:4d} agree={r['agreement']:.4f} "
              f"exact_pos={r['exact_match_pos_rate']:.4f} judge_pos={r['judge_pos_rate']:.4f} "
              f"(J+/E- {r['judge_yes_exact_no']}, J-/E+ {r['judge_no_exact_yes']})", flush=True)
    out = {
        "purpose": "does substituting the free-text LLM judge for exact option-letter match change MCQ labels?",
        "date": "2026-08-03",
        "judge": "src/labeling/run_judge.py, MedVLThinker-32B-RL_m23k, JUDGE_SYS verbatim, temperature 0",
        "readings": {"optiontext": "model answer = the TEXT of the chosen option (isolates the grader)",
                     "full": "model answer = the candidate verbatim, justification included"},
        "cells": res,
    }
    os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1)
    print(f"wrote -> {A.out}")
