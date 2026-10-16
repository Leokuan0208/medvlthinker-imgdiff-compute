#!/usr/bin/env python3
"""build_xjudge_c320.py -- the cross-family judge's (MedGemma-27B-it) work list for the FINAL method's pools.

The final method keeps the cap320 pools (select_policy_cap320_vs_fullres_T07.json), and is scored by
two-fold image cross-fitting, so it needs MedGemma-27B labels on BOTH image halves; the 2026-09-20 audit
labelled only the held-out half of the T=0.7 pools. Rows = every distinct raw answer per (cell, question)
across the cell's cap320 pools that the final evaluation can use -- T=0.7 (lingshu7b_sc8) and the frozen
temperature map's pool where one exists (lingshu7bT02/T04/T10_sc8) -- plus the greedy answer (lingshu7b),
for all questions of the eight open cells and the four comparability cells.
Rows already labelled by the audit (same cell, question, answer string) are NOT re-judged: their labels are
copied into the merged files. Writes, under /data/dan/exp_2026_10/xjudge_c320/:
  todo_part{0,1}.jsonl  rows to judge (two halves, one per GPU)
  reused.jsonl / reused.judge.jsonl   audit rows and labels, re-keyed
and merge() joins everything into merged.jsonl + merged.judge.jsonl for final_method_eval --xjudge.
  python3 build_xjudge_c320.py build | merge
"""
import glob, json, os, sys

CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
AUD = "/data/dan/audit_2026-09-18/tmp/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"
OUT = "/data/dan/exp_2026_10/xjudge_c320"
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open", "vqa_rad_closed", "pathvqa_closed", "slake_closed", "vqamed2019_test"]
TAGS = {"gemex_open": ["lingshu7bT10"], "omnimed_open": ["lingshu7bT10"], "radimagenet_open": ["lingshu7bT10"],
        "vqa_rad_open": ["lingshu7bT02"], "vqamed_open": ["lingshu7bT04"],
        "vqa_rad_closed": ["lingshu7bT02"], "vqamed2019_test": ["lingshu7bT04"]}


def loadj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def build():
    os.makedirs(OUT, exist_ok=True)
    aud_rows = {r["idx"]: r for r in loadj(AUD)}
    aud_lab = {d["idx"]: d["judge_ok"] for d in loadj(AUD.replace(".jsonl", ".judge.jsonl"))}
    have = {}
    for i, r in aud_rows.items():
        if i in aud_lab:
            c, q, _ = i.split("|")
            have[(c, q, r["modal_pred"])] = aud_lab[i]
    todo, reused = [], []
    for c in CELLS:
        per_q = {}
        for tag in ["lingshu7b"] + TAGS.get(c, []):
            for d in loadj(f"{CK}/ckpt_{c}_{tag}_sc8.jsonl"):
                s = per_q.setdefault(d["idx"], {"q": d["question"], "gold": d["gold"], "ans": []})
                s["ans"] += d["preds"]
        for d in loadj(f"{CK}/ckpt_{c}_lingshu7b.jsonl"):
            s = per_q.setdefault(d["idx"], {"q": d["question"], "gold": d["gold"], "ans": []})
            s["ans"].append(d["preds"][0])
        n_t, n_r = len(todo), len(reused)
        for q, s in per_q.items():
            seen = set()
            for a in s["ans"]:
                if a in seen:
                    continue
                seen.add(a)
                row = {"idx": f"{c}|{q}|x{len(seen)}", "question": s["q"], "gold": s["gold"], "modal_pred": a}
                k = (c, str(q), a)
                if k in have:
                    reused.append((row, have[k]))
                else:
                    todo.append(row)
        print(f"{c:17} questions {len(per_q):6} to judge {len(todo) - n_t:7} reused {len(reused) - n_r:7}")
    for k in (0, 1):
        with open(f"{OUT}/todo_part{k}.jsonl", "w") as fh:
            for r in todo[k::2]:
                fh.write(json.dumps(r) + "\n")
    with open(f"{OUT}/reused.jsonl", "w") as fa, open(f"{OUT}/reused.judge.jsonl", "w") as fb:
        for r, lab in reused:
            fa.write(json.dumps(r) + "\n"); fb.write(json.dumps({"idx": r["idx"], "judge_ok": lab}) + "\n")
    print(f"total to judge {len(todo)}, reused {len(reused)} -> {OUT}")


def merge():
    with open(f"{OUT}/merged.jsonl", "w") as fa, open(f"{OUT}/merged.judge.jsonl", "w") as fb:
        for stem in ("reused", "todo_part0", "todo_part1"):
            for l in open(f"{OUT}/{stem}.jsonl"):
                fa.write(l)
            for l in open(f"{OUT}/{stem}.judge.jsonl"):
                fb.write(l)
    print("merged ->", OUT)


if __name__ == "__main__":
    {"build": build, "merge": merge}[sys.argv[1]]()
