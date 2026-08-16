#!/usr/bin/env python3
"""PART F -- the DIAGNOSTIC that separates a real bias correction from answer-key exploitation:
answer-key skew vs the gain a marginal-matching correction earns, natural key and balanced key."""
from __future__ import annotations
import csv, json, os, re, sys
import numpy as np
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
SEED = 20260817
csv.field_size_limit(10**9)

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def load_ob(stem):
    rows = []
    for s in ("_s0of2", "_s1of2", ""):
        p = os.path.join(ROOT, "ckpts/output_bias", stem + s + ".jsonl")
        if os.path.exists(p): rows += jl(p)
    return rows
def strip_marker(t):
    t = str(t); return t[1:] if (t.startswith("Ġ") or t.startswith(" ")) else t
def letter_logits(row, K):
    lp = row.get("first_logprobs") or {}; best = {}
    for t, v in lp.items():
        s = strip_marker(t)
        if len(s) == 1 and "A" <= s <= "Z": best[s] = max(float(v), best.get(s, -1e9))
    floor = min([float(v) for v in lp.values()], default=-30.0) if lp else -30.0
    return np.array([best.get(chr(65 + i), floor) for i in range(K)])
def fit_shift(logits, target, iters=800, lr=0.3):
    n, K = logits.shape; w = np.zeros(K)
    for _ in range(iters):
        cur = np.bincount((logits - w).argmax(1), minlength=K) / n
        w = w + lr * np.log(np.maximum(cur, 1e-4) / np.maximum(target, 1e-4)); w -= w.mean()
    return w
def balanced(okv, okref, cls, seed=SEED, reps=200):
    grp = {c: np.where(cls == c)[0] for c in np.unique(cls)}
    m = min(len(v) for v in grp.values()); rng = np.random.default_rng(seed); ds = []
    for _ in range(reps):
        sub = np.concatenate([v[rng.choice(len(v), m, replace=False)] for v in grp.values()])
        ds.append(float((okv[sub] - okref[sub]).mean()))
    ds = np.array(ds)
    return dict(delta=float(ds.mean()),
                ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                n_per_class=int(m))

R = {}
# ---------------- the two MCQ cells ------------------------------------------------------------
for cell, stem, K in (("PMC_VQA", "gen_PMC_VQA_id", 4), ("MedXpertQA-MM", "gen_MedXpertQA-MM_id", 5)):
    rows = load_ob(stem); rows.sort(key=lambda r: r["i"])
    X = np.zeros((len(rows), K)); g = np.zeros(len(rows), int)
    for k, r in enumerate(rows):
        n = int(r.get("n_choices") or K)
        v = letter_logits(r, K); v[n:] = -1e9; X[k] = v
        g[k] = ord(str(r["answer"]).strip()[0]) - 65
    gm = np.bincount(g, minlength=K) / len(g)
    ro = (X.argmax(1) == g).astype(float)
    # cross-fit marginal matching to the cell's OWN key shape (the most favourable legitimate form)
    f = np.arange(len(X)) % 5; pr = np.zeros(len(X), int)
    for k in range(5):
        m = f != k
        pr[~m] = (X[~m] - fit_shift(X[m], gm)).argmax(1)
    pm = (pr == g).astype(float)
    R[cell] = dict(
        n=len(rows), n_options=K,
        gold_marginal=[float(x) for x in gm],
        answer_key_skew_L1_from_uniform=float(np.abs(gm - 1.0 / K).sum()),
        natural_key_delta=float((pm - ro).mean()),
        balanced_key_delta=balanced(pm, ro, g),
        predicted_marginal_readout=[float(x) for x in np.bincount(X.argmax(1), minlength=K) / len(X)])

# ---------------- the three binary cells: prompt-side, on the yes/no subset --------------------
CAO = os.path.join(ROOT, "ckpts/closed_as_open")
AFF = {"yes","yeah","yep","true","correct","y"}; NEG = {"no","nope","false","incorrect","n","not"}
def norm_text(s):
    s = str(s).strip().lower().strip("\"'“”‘’ \t\n").rstrip("。.")
    return re.sub(r"\s+", " ", s).strip()
def pol(r):
    for t in [x for x in re.split(r"[^a-z0-9]+", norm_text(r)) if x]:
        if t in AFF: return "yes"
        if t in NEG: return "no"
    return None
for cell in ("SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed"):
    ctl = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_closedD_g_full.jsonl")}
    itv = {r["i"]: r for r in jl(f"{CAO}/gen_{cell}_openMEK_g_full.jsonl")}
    ids = [i for i in sorted(set(ctl) & set(itv)) if pol(ctl[i]["gold"]) is not None]
    gy = np.array([pol(ctl[i]["gold"]) == "yes" for i in ids], float)
    cy = np.array([pol(ctl[i]["preds"][0]) == "yes" for i in ids], float)
    iy = np.array([pol(itv[i]["preds"][0]) == "yes" for i in ids], float)
    okc = np.array([pol(ctl[i]["preds"][0]) == pol(ctl[i]["gold"]) for i in ids], float)
    oki = np.array([pol(itv[i]["preds"][0]) == pol(ctl[i]["gold"]) for i in ids], float)
    R[cell] = dict(
        n_yesno_subset=len(ids),
        deployed_prompt_names_the_answer_space=bool("output 'yes' or 'no'" in ctl[ids[0]]["prompt"]),
        intervention_prompt_names_the_answer_space=bool("output 'yes' or 'no'" in itv[ids[0]]["prompt"]),
        gold_yes_rate=float(gy.mean()),
        yes_bias_deployed=float(cy.mean() - gy.mean()),
        yes_bias_after=float(iy.mean() - gy.mean()),
        bias_removed=float(abs(cy.mean() - gy.mean()) - abs(iy.mean() - gy.mean())),
        natural_key_delta=float((oki - okc).mean()),
        balanced_key_delta=balanced(oki, okc, gy.astype(int)))
json.dump(R, open(os.path.join(OUT, "partF.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
