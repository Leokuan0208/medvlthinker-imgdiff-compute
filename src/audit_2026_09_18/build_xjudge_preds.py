#!/usr/bin/env python3
"""Build ONE preds file for a cross-family re-judge (MedGemma-27B-it) of the Lingshu-7B held-out candidates.

Rows = every DISTINCT (benchmark, idx, raw answer string) among the held-out halves' candidates (from the
em-rescore per-question cache, which is exactly what the frozen probes were scored on) plus each question's
greedy answer. Schema is run_judge.py's: {idx, question, gold, modal_pred}; idx is a composite string so
the judge file can be joined back. Question text and gold are taken from the sc8 dump (same strings the
original judge saw)."""
import json, os, sys
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
CK = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"
CACHE = "/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json"
OUT = sys.argv[1]
R = json.load(open(CACHE))["records"]
n_rows, n_q, miss = 0, 0, 0
with open(OUT, "w") as fh:
    for cell, recs in R.items():
        q = {}
        for l in open(f"{CK}/ckpt_{cell}_lingshu7b_sc8.jsonl"):
            if l.strip():
                d = json.loads(l); q[d["idx"]] = d
        for r in recs:
            d = q.get(r["idx"])
            if d is None or "question" not in d:
                miss += 1; continue
            n_q += 1
            seen = {}
            answers = [(f"c{k}", c["ans"]) for k, c in enumerate(r["cands"])] + [("g", r["greedy"]["ans"])]
            for tag, a in answers:
                if a in seen:
                    continue            # identical raw string already queued for this question
                seen[a] = tag
                fh.write(json.dumps({"idx": f"{cell}|{r['idx']}|{tag}", "question": d["question"],
                                     "gold": d["gold"], "modal_pred": a}) + "\n")
                n_rows += 1
print(f"questions {n_q}  rows {n_rows}  questions missing from dump {miss}  -> {OUT}")
