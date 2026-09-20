#!/usr/bin/env python3
"""AUDIT/PILOT 2026-09-20 -- is the probe's gain riding on question templates / an answer prior?

The by-image split keeps IMAGES disjoint, but 55-100% of held-out questions on five benchmarks reuse a
question string that also occurs in the training half (on a different image). Stratify the frozen probe's
held-out gain (verifier - greedy, identical picks) by
   question:  REPEAT (exact normalised question string seen in the train half of the same benchmark) vs NOVEL
   gold:      SEEN   (normalised gold answer occurs as a gold in the train half)                       vs UNSEEN
in three currencies: judge A (MedVLThinker-32B), judge B (MedGemma-27B-it), lenient EM.
Image-clustered bootstrap, 10,000 resamples."""
import json, sys
import numpy as np

T = "/data/dan/audit_2026-09-18/tmp"
R = json.load(open(f"{T}/em-rescore/em_rescore_per_question_cache.json"))["records"]
rng = np.random.default_rng(20260920); NB = 10000
nq = lambda s: " ".join(str(s).lower().split()).rstrip("?.! ")
trainQ, trainG = {}, {}
for l in open(f"{T}/me/xjudge/train_lingshu7b_medgemma27b.jsonl"):
    d = json.loads(l); ds = d["idx"].split("|")[0]
    trainQ.setdefault(ds, set()).add(nq(d["question"])); trainG.setdefault(ds, set()).add(nq(d["gold"]))
lab = {}
for l in open(f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.judge.jsonl"):
    d = json.loads(l); lab.setdefault(d["idx"], int(d["judge_ok"]))
hq, a2i = {}, {}
for l in open(f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"):
    d = json.loads(l); c, i, _ = d["idx"].split("|"); hq[(c, i)] = d["question"]; a2i[(c, i, d["modal_pred"])] = d["idx"]


def boot(imgs, v):
    ks = {}
    for im, x in zip(imgs, v):
        ks.setdefault(im, []).append(x)
    s = np.array([sum(x) for x in ks.values()], float); n = np.array([len(x) for x in ks.values()], float)
    ix = rng.integers(0, len(s), size=(NB, len(s))); b = s[ix].sum(1) / n[ix].sum(1)
    return [float(s.sum() / n.sum()), float(np.quantile(b, .025)), float(np.quantile(b, .975))]


out = {"title": "Probe gain stratified by question-template repeat and gold-answer seen-in-train", "date": "2026-09-20",
       "no_fabricated_numbers": True, "picks": "frozen genframe_head_pooled_ens_v2 (em-rescore cache)", "cells": {}}
pool = {k: [] for k in ("repeat", "novel", "gold_seen", "gold_unseen")}
for cell, recs in R.items():
    # train-domain siblings share templates with their benchmark (e.g. pathvqa_open_train <-> pathvqa_open)
    sib = [d for d in trainQ if d == cell or d.replace("_train", "") == cell or (cell == "kvasir_x1_open" and d == "kvasir_open")]
    TQ = set().union(*[trainQ[d] for d in sib]); TG = set().union(*[trainG[d] for d in sib])
    rows = []
    for r in recs:
        pk, g = r["cands"][r["pick"]], r["greedy"]
        pb, gb = lab.get(a2i.get((cell, str(r["idx"]), pk["ans"]))), lab.get(a2i.get((cell, str(r["idx"]), g["ans"])))
        rows.append({"img": r["img"], "rep": nq(hq[(cell, str(r["idx"]))]) in TQ, "gs": nq(r["gold"]) in TG,
                     "dA": pk["y"] - g["y"], "dB": pb - gb, "dE": pk["em"] - g["em"], "gA": g["y"]})
    c = {"n": len(rows), "frac_question_repeat": float(np.mean([x["rep"] for x in rows])),
         "frac_gold_seen": float(np.mean([x["gs"] for x in rows])), "train_sources": sib}
    for nm, f in (("repeat", lambda x: x["rep"]), ("novel", lambda x: not x["rep"]),
                  ("gold_seen", lambda x: x["gs"]), ("gold_unseen", lambda x: not x["gs"])):
        sub = [x for x in rows if f(x)]
        c[nm] = {"n": len(sub)}
        if len(sub) >= 30:
            im = [x["img"] for x in sub]
            c[nm].update({"greedy_A": float(np.mean([x["gA"] for x in sub])), "dA": boot(im, [x["dA"] for x in sub]),
                          "dB": boot(im, [x["dB"] for x in sub]), "dEM": boot(im, [x["dE"] for x in sub])})
        pool[nm] += [(cell + x["img"], x["dA"], x["dB"], x["dE"]) for x in sub]
    out["cells"][cell] = c
    f = lambda k: (f"n{c[k]['n']:5} dA {c[k]['dA'][0]:+.4f} dB {c[k]['dB'][0]:+.4f} dEM {c[k]['dEM'][0]:+.4f}" if "dA" in c[k] else f"n{c[k]['n']:5} (too few)")
    print(f"{cell:17} rep {c['frac_question_repeat']:.2f} goldseen {c['frac_gold_seen']:.2f} | REPEAT {f('repeat')} | NOVEL {f('novel')} | GOLD-UNSEEN {f('gold_unseen')}", flush=True)
out["pooled_micro"] = {k: {"n": len(v), "dA": boot([x[0] for x in v], [x[1] for x in v]), "dB": boot([x[0] for x in v], [x[2] for x in v]),
                           "dEM": boot([x[0] for x in v], [x[3] for x in v])} for k, v in pool.items()}
print(json.dumps(out["pooled_micro"], indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
