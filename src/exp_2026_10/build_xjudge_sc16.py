#!/usr/bin/env python3
"""build_xjudge_sc16.py -- MedGemma-27B work list for the N=16 cap320 pools (T=0.7 and frozen-temperature),
reusing every label already in xjudge_c320/merged (same cell, question, answer string -> same label).
  build : writes /data/dan/exp_2026_10/xjudge_sc16/todo_part{0,1}.jsonl
  merge : writes merged.jsonl / merged.judge.jsonl = previous merged + the new labels
"""
import json, os, sys

CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
PREV = "/data/dan/exp_2026_10/xjudge_c320"
OUT = "/data/dan/exp_2026_10/xjudge_sc16"
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open"]
TAGS = {"gemex_open": "lingshu7bT10", "omnimed_open": "lingshu7bT10", "radimagenet_open": "lingshu7bT10",
        "vqa_rad_open": "lingshu7bT02", "vqamed_open": "lingshu7bT04"}


def loadj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def build():
    os.makedirs(OUT, exist_ok=True)
    have = set()
    for r in loadj(f"{PREV}/merged.jsonl"):
        c, q, _ = r["idx"].split("|")
        have.add((c, q, r["modal_pred"]))
    todo = []
    for c in CELLS:
        n0 = len(todo)
        for tag in ["lingshu7b"] + ([TAGS[c]] if c in TAGS else []):
            for d in loadj(f"{CK}/ckpt_{c}_{tag}_sc16.jsonl"):
                seen = set()
                for a in d["preds"]:
                    k = (c, str(d["idx"]), a)
                    if a in seen or k in have:
                        continue
                    seen.add(a); have.add(k)
                    todo.append({"idx": f"{c}|{d['idx']}|s{len(todo)}", "question": d["question"],
                                 "gold": d["gold"], "modal_pred": a})
        print(f"{c:17} new rows {len(todo) - n0}")
    for k in (0, 1):
        with open(f"{OUT}/todo_part{k}.jsonl", "w") as fh:
            for r in todo[k::2]:
                fh.write(json.dumps(r) + "\n")
    print("total to judge", len(todo))


def merge():
    with open(f"{OUT}/merged.jsonl", "w") as fa, open(f"{OUT}/merged.judge.jsonl", "w") as fb:
        for src, stems in ((PREV, ["merged"]), (OUT, ["todo_part0", "todo_part1"])):
            for s in stems:
                for l in open(f"{src}/{s}.jsonl"):
                    fa.write(l)
                for l in open(f"{src}/{s}.judge.jsonl"):
                    fb.write(l)
    print("merged ->", OUT)


if __name__ == "__main__":
    {"build": build, "merge": merge}[sys.argv[1]]()
