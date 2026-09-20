#!/usr/bin/env python3
"""AUDIT 2026-09-20 -- re-score the frozen pooled probe's held-out picks under a CROSS-FAMILY judge.

Labels of record: MedVLThinker-32B (Qwen2.5-VL-32B lineage; run_judge.py default) -- field 'y' in the cache.
New labels:       MedGemma-27B-it (Gemma 3 lineage), SAME script / prompt / decoding, only the model differs.
Picks are the frozen 24-probe ensemble's picks stored in the em-rescore per-question cache -- nothing is refit,
so this asks: do picks made by a probe trained on judge-A labels still beat greedy when judge B grades them?
"""
import json, sys
import numpy as np

CACHE = "/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json"
PRED = "/data/dan/audit_2026-09-18/tmp/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"
JUD = PRED.replace(".jsonl", ".judge.jsonl")
OUT = sys.argv[1]
B = 10000
rng = np.random.default_rng(20260920)

lab = {}
for l in open(JUD):
    if l.strip():
        d = json.loads(l); lab.setdefault(d["idx"], int(d["judge_ok"]))
ans2id = {}
for l in open(PRED):
    if l.strip():
        d = json.loads(l); c, i, _ = d["idx"].split("|"); ans2id[(c, i, d["modal_pred"])] = d["idx"]
R = json.load(open(CACHE))["records"]


def boot(imgs, vals):
    ks = {}
    for im, v in zip(imgs, vals):
        ks.setdefault(im, []).append(v)
    s = np.array([sum(v) for v in ks.values()], float); n = np.array([len(v) for v in ks.values()], float)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    b = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


out = {"title": "Frozen pooled probe, held-out picks, re-scored under a cross-family judge (MedGemma-27B-it)",
       "date": "2026-09-20", "no_fabricated_numbers": True,
       "judge_of_record": "MedVLThinker-32B-RL_m23k (run_judge.py default; field y)",
       "cross_family_judge": "google/medgemma-27b-it, same run_judge.py prompt + decoding, tp=1, bf16",
       "picks": "frozen genframe_head_pooled_ens_v2 (24 probes), from the em-rescore cache; nothing refit",
       "bootstrap": f"{B} resamples clustered by img_md5 within benchmark", "cells": {}}
n_missing = 0
for cell, recs in R.items():
    imgs, gA, gB, pA, pB, oB, rB = [], [], [], [], [], [], []
    A_all, B_all, EM_all = [], [], []
    up_f1zero_A, up_f1zero_B = 0, 0
    upA = upA_confirmed = dnA = dnA_confirmed = 0
    for r in recs:
        def jb(a):
            k = ans2id.get((cell, str(r["idx"]), a))
            return lab.get(k) if k else None
        cb = [jb(c["ans"]) for c in r["cands"]]
        g_b = jb(r["greedy"]["ans"])
        if g_b is None or any(x is None for x in cb):
            n_missing += 1; continue
        imgs.append(r["img"]); pk = r["pick"]
        gA.append(r["greedy"]["y"]); gB.append(g_b)
        pA.append(r["cands"][pk]["y"]); pB.append(cb[pk])
        oB.append(max(cb)); rB.append(float(np.mean(cb)))
        for c, b_ in zip(r["cands"], cb):
            A_all.append(c["y"]); B_all.append(b_); EM_all.append(c["em"])
        if r["cands"][pk]["y"] == 1 and r["greedy"]["y"] == 0:
            upA += 1; upA_confirmed += int(cb[pk] == 1 and g_b == 0)
            if r["cands"][pk]["f1"] == 0:
                up_f1zero_A += 1; up_f1zero_B += int(cb[pk] == 1)
        if r["cands"][pk]["y"] == 0 and r["greedy"]["y"] == 1:
            dnA += 1; dnA_confirmed += int(cb[pk] == 0 and g_b == 1)
    gA, gB, pA, pB = map(lambda v: np.array(v, float), (gA, gB, pA, pB))
    c = {"n": len(imgs), "greedy_A": float(gA.mean()), "verifier_A": float(pA.mean()),
         "greedy_B": float(gB.mean()), "verifier_B": float(pB.mean()),
         "random_pick_B": float(np.mean(rB)), "oracle_B": float(np.mean(oB))}
    c["delta_A"], c["delta_A_ci"] = boot(imgs, pA - gA)
    c["delta_B"], c["delta_B_ci"] = boot(imgs, pB - gB)
    c["verifier_minus_random_B"], c["verifier_minus_random_B_ci"] = boot(imgs, pB - np.array(rB))
    c["candidate_level"] = {"n": len(A_all), "pos_rate_A": float(np.mean(A_all)), "pos_rate_B": float(np.mean(B_all)),
                            "pos_rate_EM": float(np.mean(EM_all)), "agreement_AB": float(np.mean(np.array(A_all) == np.array(B_all))),
                            "kappa_AB": kappa(A_all, B_all), "kappa_A_EM": kappa(A_all, EM_all), "kappa_B_EM": kappa(B_all, EM_all),
                            "A_yes_B_no": int(np.sum((np.array(A_all) == 1) & (np.array(B_all) == 0))),
                            "A_no_B_yes": int(np.sum((np.array(A_all) == 0) & (np.array(B_all) == 1)))}
    c["flips"] = {"up_under_A": upA, "up_confirmed_by_B": upA_confirmed, "down_under_A": dnA, "down_confirmed_by_B": dnA_confirmed,
                  "up_with_zero_token_overlap": up_f1zero_A, "of_which_B_says_pick_correct": up_f1zero_B}
    out["cells"][cell] = c
    print(f"{cell:17} n{c['n']:5} | A: g {c['greedy_A']:.4f} v {c['verifier_A']:.4f} d {c['delta_A']:+.4f} | "
          f"B: g {c['greedy_B']:.4f} v {c['verifier_B']:.4f} d {c['delta_B']:+.4f} {np.round(c['delta_B_ci'], 4).tolist()} | "
          f"kappa AB {c['candidate_level']['kappa_AB']:.3f} | up {upA} confirmed {upA_confirmed} | f1=0 ups {up_f1zero_A} -> B yes {up_f1zero_B}", flush=True)
cs = list(out["cells"].values())
out["macro"] = {k: float(np.mean([v[k] for v in cs])) for k in ("delta_A", "delta_B", "greedy_A", "greedy_B", "verifier_A", "verifier_B",
                                                                "verifier_minus_random_B", "oracle_B")}
dB = np.array([v["delta_B"] for v in cs]); idx = rng.integers(0, len(dB), size=(B, len(dB)))
out["macro"]["delta_B_benchmark_level_ci_n8"] = [float(np.quantile(dB[idx].mean(1), .025)), float(np.quantile(dB[idx].mean(1), .975))]
out["macro"]["benchmarks_positive_B"] = int((dB > 0).sum())
out["macro"]["benchmarks_significant_B"] = int(sum(1 for v in cs if v["delta_B_ci"][0] > 0))
out["questions_skipped_missing_label"] = n_missing
print(json.dumps(out["macro"], indent=1)); print("skipped", n_missing)
json.dump(out, open(OUT, "w"), indent=1)
