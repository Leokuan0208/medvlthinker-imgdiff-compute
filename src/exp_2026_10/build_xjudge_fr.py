#!/usr/bin/env python3
"""build_xjudge_fr.py -- the cross-family judge's work list for the full-resolution pools.

Rows = every DISTINCT raw answer string per (cell, question) across the cell's full-resolution N=8 pools
(T=0.7 and, where it exists, the frozen-temperature pool) plus the full-resolution greedy answer, for
ALL questions (both image halves: the final evaluation cross-fits over images). Schema is
run_judge.py's {idx, question, gold, modal_pred}; idx = "cell|question_idx|n" so labels join back on
(cell, question, answer string) exactly as final_method_eval.py reads them. Written as two halves
(…_part0/1.jsonl) so both GPUs can judge in parallel.
"""
import glob, json, os

CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
OUT = "/data/dan/exp_2026_10/xjudge"
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
         "vqamed_open", "gemex_open", "vqa_rad_closed", "pathvqa_closed", "slake_closed", "vqamed2019_test"]
os.makedirs(OUT, exist_ok=True)


def loadj(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


rows = []
for c in CELLS:
    pools = sorted(glob.glob(f"{CK}/ckpt_{c}_lingshu7bFR*_sc8.jsonl"))
    greedy = {d["idx"]: d for d in loadj(f"{CK}/ckpt_{c}_lingshu7bFR.jsonl")}
    per_q = {}
    for p in pools:
        for d in loadj(p):
            s = per_q.setdefault(d["idx"], {"q": d["question"], "gold": d["gold"], "ans": []})
            s["ans"] += d["preds"]
    for q, g in greedy.items():
        s = per_q.setdefault(q, {"q": g["question"], "gold": g["gold"], "ans": []})
        s["ans"].append(g["preds"][0])
    n0 = len(rows)
    for q, s in per_q.items():
        seen = set()
        for a in s["ans"]:
            if a in seen:
                continue
            seen.add(a)
            rows.append({"idx": f"{c}|{q}|{len(seen)}", "question": s["q"], "gold": s["gold"], "modal_pred": a})
    print(f"{c:17} pools {len(pools)} questions {len(per_q)} rows {len(rows) - n0}")
for k in (0, 1):
    with open(f"{OUT}/fr_medgemma27b_part{k}.jsonl", "w") as fh:
        for r in rows[k::2]:
            fh.write(json.dumps(r) + "\n")
with open(f"{OUT}/fr_medgemma27b.jsonl", "w") as fh:          # the joined list final_method_eval reads
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print("total rows", len(rows), "->", OUT)
