#!/usr/bin/env python3
"""choicewhy_justification_stats.py -- how much GRADEABLE TEXT do the candidates actually carry?

Phase 1's yellow flag (i) was that ~42% of greedy arm-B2 answers restate the option and stop, leaving a
verifier nothing to grade.  That flag has to be carried into Phase 2 with real numbers for the SAMPLED
(n=8, temperature 0.7) pools that the verifier is actually trained and measured on -- greedy decoding is
not the same distribution.

Definition, matching the Phase-1 supplement: strip the answer token (choicewhy_common.strip_letter),
remove the words of the option the candidate chose, and count what remains.  Fewer than 3 remaining
words = "no gradeable justification".

  python3 src/cascade_methods/choicewhy_justification_stats.py
"""
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from choicewhy_common import ARM_NAME, extract, norm, strip_letter  # noqa: E402

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--eval_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--train_dir", default="ckpts/choicewhy_train")
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_justification_stats.json")
A = ap.parse_args()
EVAL_BENCH = ["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"]


def residual_words(cand, options, an):
    letter, ok, _ = extract(cand, an)
    body = strip_letter(cand, an)
    chosen = norm(options.get(letter, "")) if ok else ""
    stop = set(chosen.split())
    return [w for w in norm(body).split() if w not in stop]


def summarize(recs, an):
    n = nbare = 0
    wsum = 0
    uniq = set()
    for question, options, cands in recs:
        for c in cands:
            c = c.strip()
            if not c:
                continue
            r = residual_words(c, options, an)
            n += 1
            wsum += len(r)
            nbare += int(len(r) < 3)
            uniq.add(norm(c))
    return {"n_candidates": n,
            "mean_residual_words": round(wsum / max(1, n), 2),
            "frac_no_gradeable_justification": round(nbare / max(1, n), 4),
            "distinct_candidate_strings": len(uniq)}


out = {"purpose": "fraction of SAMPLED candidates that carry text a verifier can grade",
       "date": "2026-08-03",
       "definition": "strip the answer token, remove the chosen option's own words, count the rest; "
                     "<3 remaining words = no gradeable justification (Phase-1 supplement's rule)",
       "eval": {}, "train": {}}

from datasets import load_dataset  # noqa: E402
from choicewhy_common import parse_opts  # noqa: E402

data = load_dataset("/data/dan/dataset/MedVLThinker-Eval")["test"]
for arm in A.arms:
    an = ARM_NAME[arm]
    per = {}
    allrec = []
    for b in EVAL_BENCH:
        p = os.path.join(ROOT, A.eval_dir, f"ckpt_{b}_{an}_sc8.jsonl")
        if not os.path.exists(p):
            continue
        recs = []
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            recs.append((r.get("question", ""), parse_opts(data[r["idx"]]["options"]), r["raw_outputs"]))
        per[b] = summarize(recs, an)
        allrec += recs
    per["POOLED"] = summarize(allrec, an)
    out["eval"][an] = per
    print(f"\n=== EVAL arm {an} ===")
    for b, c in per.items():
        print(f"  {b:24s} n={c['n_candidates']:6d} residual_words={c['mean_residual_words']:6.2f} "
              f"no_justification={c['frac_no_gradeable_justification']:.4f} "
              f"distinct_strings={c['distinct_candidate_strings']}")

for arm in A.arms:
    an = ARM_NAME[arm]
    per, allrec = {}, []
    for p in sorted(glob.glob(os.path.join(ROOT, A.train_dir, f"ckpt_*_{an}_sc8.jsonl"))):
        src = os.path.basename(p).replace("ckpt_", "").replace(f"_{an}_sc8.jsonl", "")
        recs = []
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            recs.append((r["question"], r["options"], r["raw_outputs"]))
        per[src] = summarize(recs, an)
        allrec += recs
    per["POOLED"] = summarize(allrec, an)
    out["train"][an] = per
    print(f"\n=== TRAIN arm {an} ===")
    for b, c in per.items():
        print(f"  {b:24s} n={c['n_candidates']:6d} residual_words={c['mean_residual_words']:6.2f} "
              f"no_justification={c['frac_no_gradeable_justification']:.4f} "
              f"distinct_strings={c['distinct_candidate_strings']}")

os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, A.out), "w"), indent=1)
print(f"\nwrote -> {A.out}")
