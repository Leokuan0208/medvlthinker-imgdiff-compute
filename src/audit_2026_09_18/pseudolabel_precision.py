#!/usr/bin/env python3
"""PILOT 2026-09-20 -- how clean would PROBE-SELECTED self-training data be?

Setting: unlabeled questions from a benchmark the probe has been onboarded on (= the held-out halves: the
frozen probes never saw these images). For each question a pseudo-label is chosen and a confidence attached;
keep the most-confident fraction c of questions as training data. This is DATA FILTERING FOR TRAINING, not
abstention at inference -- the deployed model still answers every question.

  probe      pseudo-label = frozen-ensemble pick (argmax mean rank);  confidence = p_mean of the pick
  vote       pseudo-label = majority answer of the 8 samples;          confidence = its vote share   (the
             self-consistency filter used by TTRL / ScPO-style self-training -- the literature's default)
  probe+vote pseudo-label = probe pick;                                confidence = p_mean * vote share of pick

Precision is reported in three currencies on identical selections: judge A (MedVLThinker-32B, labels of
record), lenient EM, and -- if the cross-family file exists -- judge B (MedGemma-27B-it).
"""
import json, os, sys
import numpy as np

T = "/data/dan/audit_2026-09-18/tmp"
SC = json.load(open(f"{T}/me/probe_scores_heldout_lingshu.json"))
R = json.load(open(f"{T}/em-rescore/em_rescore_per_question_cache.json"))["records"]
PRED, JUD = f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl", f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.judge.jsonl"
labB = None
if os.path.exists(JUD):
    lab = {}
    for l in open(JUD):
        if l.strip():
            d = json.loads(l); lab.setdefault(d["idx"], int(d["judge_ok"]))
    labB = {}
    for l in open(PRED):
        if l.strip():
            d = json.loads(l); c, i, _ = d["idx"].split("|")
            if d["idx"] in lab:
                labB[(c, i, d["modal_pred"])] = lab[d["idx"]]
COV = [0.10, 0.25, 0.50, 0.75, 1.00]
out = {"title": "Pseudo-label precision vs coverage: probe-filtered vs self-consistency-filtered",
       "date": "2026-09-20", "no_fabricated_numbers": True, "judge_B_available": labB is not None,
       "coverages": COV, "cells": {}}
for cell, recs in R.items():
    s = {(r["idx"], r["ans"].strip().lower()): r for r in SC[cell]}
    rows = []
    for r in recs:
        cs = r["cands"]
        sc = [s.get((r["idx"], c["ans"].strip().lower())) for c in cs]
        if any(x is None for x in sc):
            continue
        pk = int(np.argmax([x["rank_mean"] for x in sc]))
        mv = int(np.argmax([c["votes"] for c in cs]))
        nv = sum(c["votes"] for c in cs)

        def cur(c):
            b = labB.get((cell, str(r["idx"]), c["ans"])) if labB else None
            return (c["y"], c["em"], b)
        rows.append({"probe": (sc[pk]["p_mean"],) + cur(cs[pk]),
                     "vote": (cs[mv]["votes"] / nv,) + cur(cs[mv]),
                     "probe+vote": (sc[pk]["p_mean"] * cs[pk]["votes"] / nv,) + cur(cs[pk]),
                     "greedy": (None, r["greedy"]["y"], r["greedy"]["em"],
                                labB.get((cell, str(r["idx"]), r["greedy"]["ans"])) if labB else None)})
    c = {"n": len(rows), "greedy_acc": {"A": float(np.mean([x["greedy"][1] for x in rows])),
                                        "EM": float(np.mean([x["greedy"][2] for x in rows]))}}
    for m in ("probe", "vote", "probe+vote"):
        conf = np.array([x[m][0] for x in rows]); order = np.argsort(-conf, kind="mergesort")
        c[m] = {}
        for cv in COV:
            k = max(1, int(round(cv * len(rows)))); ii = order[:k]
            d = {"k": k, "A": float(np.mean([rows[i][m][1] for i in ii])), "EM": float(np.mean([rows[i][m][2] for i in ii]))}
            if labB:
                bb = [rows[i][m][3] for i in ii if rows[i][m][3] is not None]
                d["B"] = float(np.mean(bb)) if bb else None
            # greedy's accuracy on the SAME kept questions (is the filter just finding easy questions?)
            d["greedy_A_same_q"] = float(np.mean([rows[i]["greedy"][1] for i in ii]))
            c[m][f"{cv:.2f}"] = d
    out["cells"][cell] = c
    print(f"{cell:17} n{c['n']:5} greedy A {c['greedy_acc']['A']:.3f} | " + " | ".join(
        f"{m}: " + " ".join(f"{cv:.2f}->{c[m][f'{cv:.2f}']['A']:.3f}/{c[m][f'{cv:.2f}']['EM']:.3f}"
                            + (f"/{c[m][f'{cv:.2f}']['B']:.3f}" if labB else "") for cv in (0.25, 0.50, 1.00))
        for m in ("probe", "vote")), flush=True)
out["macro"] = {m: {f"{cv:.2f}": {k: float(np.mean([v[m][f"{cv:.2f}"][k] for v in out["cells"].values()
                                                     if v[m][f"{cv:.2f}"].get(k) is not None]))
                                  for k in (("A", "EM", "B", "greedy_A_same_q") if labB else ("A", "EM", "greedy_A_same_q"))}
                    for cv in COV} for m in ("probe", "vote", "probe+vote")}
print(json.dumps(out["macro"], indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
