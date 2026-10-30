#!/usr/bin/env python3
"""sft_template_split.py -- stress test of the generator-SFT baseline (2026-10-23): is its gain over the probe
learned answer vocabulary for question templates it trained on, or does it carry to templates it never saw?

Folds are split by IMAGE (crossfit_data), so an evaluation question's template (normalised question string, as in
template_clustered_ci.py) can still appear in that fold's SFT training manifest with another image. Each held-out
question is tagged: template unseen / template seen / (template, gold answer) pair seen, and separately by whether
its normalised GOLD ANSWER string occurs anywhere among the fold's training answers (the label-vocabulary test). SFT, the probe (T07
records) and greedy are then compared within each tag, in judge-of-record and lenient EM, macro over cells with
>= 20 questions in the tag, paired image-clustered bootstrap (10,000).

  python3 src/exp_2026_10/sft_template_split.py  -> results/cascade_methods/artifacts/sft_template_split_2026-10-23.json
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_method_eval as FME                      # noqa: E402

M = FME.MAIN; ER = FME.ER; OPEN8 = FME.OPEN8
D = f"{M}/data/sft_crossfit_2026-10-22"; SFT = f"{M}/ckpts/openvqa/sft_crossfit_1022"
OUT = os.path.join(FME.WT, "results/cascade_methods/artifacts/sft_template_split_2026-10-23.json")
NB, SEED = 10000, 20261023
norm = lambda q: re.sub(r"\s+", " ", q.lower()).strip().rstrip("?.!").strip()      # template_clustered_ci.py's
loadjl = lambda p: [json.loads(l) for l in open(p) if l.strip()]

REC = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"))
base = {(c, r["idx"]): r for c in OPEN8 for r in REC[c]}
J = {d["idx"]: int(d["judge_ok"]) for d in loadjl("/data/dan/exp_2026_10/xjudge_1022/new_all.judge.jsonl")}
seen_t = {f: set() for f in (0, 1)}; seen_ta = {f: set() for f in (0, 1)}; seen_a = {f: set() for f in (0, 1)}
for f in (0, 1):
    for e in json.load(open(f"{D}/fold{f}.json"))["open_generic"]:
        seen_t[f].add(norm(e["question"])); seen_ta[f].add((norm(e["question"]), ER.row_norm(e["answer"])))
        seen_a[f].add(ER.row_norm(e["answer"]))      # any training question of the fold, any benchmark
rows = []
for c in OPEN8:
    for f in (0, 1):
        for d in loadjl(f"{SFT}/ckpt_{c}_sft1022f{f}.jsonl"):
            k = (c, d["idx"]); r = base[k]; t = norm(d["question"])
            tag = ("pair_seen" if (t, ER.row_norm(d["gold"])) in seen_ta[f] else "template_seen") if t in seen_t[f] else "unseen"
            atag = "answer_seen" if ER.row_norm(d["gold"]) in seen_a[f] else "answer_unseen"
            rows.append({"c": c, "img": r["img"], "tag": tag, "atag": atag,
                         "sft": {"judge": J[f"{c}|{d['idx']}|sft{f}"], "em": ER.em_score(d["modal_pred"], d["gold"])},
                         "probe": r["pick"], "greedy": r["greedy"]})


def paired(R, a, b, cu):
    rng = np.random.default_rng(SEED); reps, pts, cells = [], [], []
    for c in OPEN8:
        rc = [r for r in R if r["c"] == c]
        if len(rc) < 20: continue
        d = np.array([float(r[a][cu]) - float(r[b][cu]) for r in rc]); keys, inv = np.unique([r["img"] for r in rc], return_inverse=True)
        w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NB).astype(float)
        reps.append((w @ np.bincount(inv, weights=d, minlength=len(keys))) / (w @ np.bincount(inv, minlength=len(keys)).astype(float)))
        pts.append(d.mean()); cells.append(c)
    if not pts: return None
    Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"), "cells": cells}


out = {"title": "Generator-SFT gain split by whether the question template was in its training fold", "date": "2026-10-23",
       "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "n_questions": len(rows),
       "tag_counts": {}, "per_cell_tag_share": {}, "by_tag": {}}
for c in OPEN8:
    rc = [r for r in rows if r["c"] == c]
    out["per_cell_tag_share"][c] = {t: round(float(np.mean([r["tag"] == t for r in rc])), 4) for t in ("unseen", "template_seen", "pair_seen")}
    out["per_cell_tag_share"][c]["answer_seen"] = round(float(np.mean([r["atag"] == "answer_seen" for r in rc])), 4)
for tag in ("unseen", "template_seen", "pair_seen", "answer_unseen", "answer_seen", "all"):
    R = rows if tag == "all" else [r for r in rows if tag in (r["tag"], r["atag"])]
    out["tag_counts"][tag] = len(R)
    out["by_tag"][tag] = {cu: {"sft_minus_greedy": paired(R, "sft", "greedy", cu), "probe_minus_greedy": paired(R, "probe", "greedy", cu),
                               "probe_minus_sft": paired(R, "probe", "sft", cu)} for cu in ("judge", "em")}
    j = out["by_tag"][tag]["judge"]
    print(f"{tag:14} n={len(R):6}  judge: sft-greedy {j['sft_minus_greedy']['macro']:+.4f}  probe-greedy {j['probe_minus_greedy']['macro']:+.4f}  "
          f"probe-sft {j['probe_minus_sft']['macro']:+.4f} {j['probe_minus_sft']['verdict']}  cells {len(j['probe_minus_sft']['cells'])}", flush=True)
print(json.dumps(out["per_cell_tag_share"], indent=0))
out["COMPLETE"] = True
json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT)
