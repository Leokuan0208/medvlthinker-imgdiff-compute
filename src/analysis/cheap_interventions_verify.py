#!/usr/bin/env python3
"""cheap_interventions_verify.py -- INDEPENDENT adversarial re-derivation of the 2026-08-17
cheap-intervention round.  Nothing is imported from output_bias_correct.py or selfcons_suite.py;
every number is recomputed from the raw per-item dumps with locally written code.

Traps under test:
  (a) a bias correction fitted on the EVAL marginal (leakage, not a method)
  (b) a prompt change that moves the GRADER not the model (token + parse audit)
  (c) a per-cell "best correction" selection with no permutation null
  (d) a judge-only gain whose EM diagnostic points the other way
"""
from __future__ import annotations
import hashlib, json, os, re, sys
import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
NBOOT, SEED = 10000, 20260817

# ---------------------------------------------------------------- local graders (written here)
AFF = {"yes", "yeah", "yep", "true", "correct", "y"}
NEG = {"no", "nope", "false", "incorrect", "n", "not"}

def norm_text(s):
    s = str(s).strip().lower().strip("\"'“”‘’ \t\n").rstrip("。.")
    return re.sub(r"\s+", " ", s).strip()

def toks(s):
    return [t for t in re.split(r"[^a-z0-9]+", norm_text(s)) if t]

def polarity(r):
    for t in toks(r):
        if t in AFF: return "yes"
        if t in NEG: return "no"
    return None

def em_binary(gold, resp):
    """polarity-based exact match on a yes/no cell; length-neutral by construction."""
    return int(polarity(resp) == polarity(gold))

def first_letter(resp, K):
    """MCQ EM: first standalone capital letter within range."""
    m = re.search(r"\b([A-Z])\b", str(resp))
    if m and ord(m.group(1)) - 65 < K: return ord(m.group(1)) - 65
    m = re.match(r"\s*([A-Z])", str(resp))
    if m and ord(m.group(1)) - 65 < K: return ord(m.group(1)) - 65
    return -1

def boot(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b; rng = np.random.default_rng(seed); n = len(d)
    bs = d[rng.integers(0, n, size=(nboot, n))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"), n=int(n))

def jl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def load_shards(d, stem):
    rows = []
    for s in ("", "_s0of2", "_s1of2"):
        p = os.path.join(d, stem + s + ".jsonl")
        if os.path.exists(p): rows += jl(p)
    return rows

R = {}

# =============================================================================================
# PART A -- PROMPT-SIDE on the three binary cells, judge + EM, matched fullres
# =============================================================================================
CAO = os.path.join(ROOT, "ckpts/closed_as_open")
partA = {}
for cell in ("PATH_VQA_closed", "SLAKE_closed", "VQA_RAD_closed"):
    ctl = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_closedD_g_full.jsonl")}
    itv = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_openMEK_g_full.jsonl")}
    # judge map: (i, normalised answer) -> judge_ok
    meta = {r["idx"]: r for r in jl(f"{CAO}/judge_{cell}.jsonl")}
    lab = {r["idx"]: r["judge_ok"] for r in jl(f"{CAO}/judge_{cell}.judge.jsonl")}
    jmap = {(meta[k]["i"], meta[k]["na"]): v for k, v in lab.items() if k in meta}
    ids = sorted(set(ctl) & set(itv))
    jc, ji, ec, ei, tc, ti, miss = [], [], [], [], [], [], 0
    for i in ids:
        gc, gi = ctl[i]["preds"][0], itv[i]["preds"][0]
        gold = ctl[i]["gold"]
        assert itv[i]["gold"] == gold
        kc, ki = (i, norm_text(gc)), (i, norm_text(gi))
        if kc not in jmap or ki not in jmap: miss += 1; continue
        jc.append(jmap[kc]); ji.append(jmap[ki])
        ec.append(em_binary(gold, gc)); ei.append(em_binary(gold, gi))
        tc.append(ctl[i]["gen_tokens"][0]); ti.append(itv[i]["gen_tokens"][0])
    partA[cell] = dict(
        n=len(jc), n_missing_judge=miss,
        ctl_judge=float(np.mean(jc)), itv_judge=float(np.mean(ji)),
        ctl_em=float(np.mean(ec)), itv_em=float(np.mean(ei)),
        judge=boot(ji, jc), em=boot(ei, ec),
        judge_minus_em=float(np.mean(ji) - np.mean(jc) - (np.mean(ei) - np.mean(ec))),
        mean_gen_tokens=dict(control=float(np.mean(tc)), intervention=float(np.mean(ti))),
        yes_rate=dict(gold=float(np.mean([polarity(ctl[i]["gold"]) == "yes" for i in ids])),
                      control=float(np.mean([polarity(ctl[i]["preds"][0]) == "yes" for i in ids])),
                      intervention=float(np.mean([polarity(itv[i]["preds"][0]) == "yes" for i in ids]))))
R["A_prompt_side"] = partA

# ---- A2: balanced-key rescore of the prompt-side arm (equal items per gold class) -----------
def balanced(ids, gold_of, a, b, seed=SEED, reps=200):
    """mean delta over seeded subsamples with equal n per gold class."""
    cls = {}
    for i in ids: cls.setdefault(gold_of[i], []).append(i)
    m = min(len(v) for v in cls.values())
    rng = np.random.default_rng(seed); ds = []
    per_class = {k: [] for k in cls}
    for _ in range(reps):
        sub = []
        for k, v in cls.items():
            pick = rng.choice(len(v), m, replace=False)
            sub += [v[j] for j in pick]
            per_class[k].append(np.mean([a[i] - b[i] for i in [v[j] for j in pick]]))
        ds.append(np.mean([a[i] - b[i] for i in sub]))
    ds = np.array(ds)
    return dict(delta=float(ds.mean()), ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                per_class={k: float(np.mean(v)) for k, v in per_class.items()},
                n_per_class=int(m), reps=reps)

bal = {}
for cell in ("PATH_VQA_closed",):
    ctl = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_closedD_g_full.jsonl")}
    itv = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_openMEK_g_full.jsonl")}
    ids = sorted(set(ctl) & set(itv))
    gold_of = {i: polarity(ctl[i]["gold"]) for i in ids}
    A = {i: em_binary(ctl[i]["gold"], itv[i]["preds"][0]) for i in ids}
    B = {i: em_binary(ctl[i]["gold"], ctl[i]["preds"][0]) for i in ids}
    bal[cell] = dict(natural_em=boot([A[i] for i in ids], [B[i] for i in ids]),
                     balanced_em=balanced(ids, gold_of, A, B))
R["A2_balanced_key_prompt_side"] = bal

json.dump(R, open(os.path.join(OUT, "partA.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
