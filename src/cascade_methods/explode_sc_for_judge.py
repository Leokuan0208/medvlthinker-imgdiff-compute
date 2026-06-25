#!/usr/bin/env python3
"""Explode an sc8 open-ended ckpt into one judge-input file with composite idx 'origidx#k' so the
existing run_judge.py can score ALL 8 samples per question in a single load. Output goes next to the
sc8 file as ckpt_<ds>_<tag>_scexploded.jsonl with fields {idx, question, gold, modal_pred=sample_k}.
Dedups identical (idx,answer) pairs to save judge calls (judge is text-only, so identical strings tie).
Usage: python3 src/cascade_methods/explode_sc_for_judge.py <sc8_jsonl_path>"""
import json, sys, os
src = sys.argv[1]
out = src.replace(".jsonl", "_scexploded.jsonl")
n_in = n_out = 0
with open(out, "w") as fh:
    for l in open(src):
        if not l.strip(): continue
        r = json.loads(l); n_in += 1
        seen = {}
        for k, ans in enumerate(r["preds"]):
            key = str(ans).strip().lower()
            if key in seen: continue          # same string -> judge once
            seen[key] = k
            fh.write(json.dumps({"idx": f'{r["idx"]}#{k}', "question": r["question"],
                                 "gold": r["gold"], "modal_pred": ans}) + "\n")
            n_out += 1
print(f"{os.path.basename(src)}: {n_in} questions -> {n_out} unique (idx,answer) judge rows -> {os.path.basename(out)}")
