#!/usr/bin/env python3
"""PART D -- independent re-derivation of PMC-VQA self-consistency and the SLOT-EXCHANGEABILITY
defect.  Exact subset means over all C(8,N) subsets, not Monte Carlo."""
from __future__ import annotations
import itertools, json, os, re, sys
from collections import Counter
import numpy as np
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
NBOOT, SEED = 10000, 20260817

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def boot(a, b, nboot=NBOOT, seed=SEED):
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)],
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"), n=int(len(d)))
def letter(s):
    m = re.match(r"\s*([A-Z])", str(s))
    return m.group(1) if m else "?"

g = {r["i"]: r for r in jl(f"{ROOT}/ckpts/closed_as_open_mcq/gen_PMC_VQA_mcq_g.jsonl")}
s8 = {r["i"]: r for r in jl(f"{ROOT}/ckpts/closed_as_open_mcq/gen_PMC_VQA_mcq_s8.jsonl")}
ids = sorted(set(g) & set(s8))
gold = np.array([g[i]["gold"].strip()[0] for i in ids])
L = np.array([[letter(p) for p in s8[i]["preds"]] for i in ids])          # (n, 8)
gr = np.array([letter(g[i]["preds"][0]) for i in ids])
R = {}
R["D0"] = dict(n=len(ids), in_session_greedy_acc=float((gr == gold).mean()),
               published_always_7b=0.5427)

# ---- exact mean over all C(8,N) subsets, index tie-break (smallest slot index wins) ----------
def vote_ok(sub):
    out = np.zeros(len(ids))
    for r in range(len(ids)):
        c = Counter(); first = {}
        for j in sub:
            v = L[r, j]; c[v] += 1
            if v not in first: first[v] = j
        best = max(c.items(), key=lambda kv: (kv[1], -first[kv[0]]))[0]
        out[r] = float(best == gold[r])
    return out

curve = {}
for N in (1, 2, 4, 8):
    subs = list(itertools.combinations(range(8), N))
    acc = np.mean([vote_ok(s) for s in subs], axis=0)
    curve[N] = dict(n_subsets=len(subs), vote_acc=float(acc.mean()),
                    delta_vs_in_session_greedy=boot(acc, (gr == gold).astype(float)))
    # prefix variant (first N slots only) -- identifiability check
    acc_p = vote_ok(tuple(range(N)))
    curve[N]["delta_prefix_first_N_slots"] = boot(acc_p, (gr == gold).astype(float))
R["D1_self_consistency_curve"] = curve

# ---- SLOT EXCHANGEABILITY ---------------------------------------------------------------------
slot_acc = np.array([float((L[:, j] == gold).mean()) for j in range(8)])
obs_spread = float(slot_acc.max() - slot_acc.min())
rng = np.random.default_rng(SEED)
null = []
for _ in range(2000):
    P = np.take_along_axis(L, rng.permuted(np.tile(np.arange(8), (len(ids), 1)), axis=1), axis=1)
    a = np.array([float((P[:, j] == gold).mean()) for j in range(8)])
    null.append(a.max() - a.min())
null = np.array(null)
onmarg = []
for j in range(8):
    m = np.bincount([ord(c) - 65 for c in L[:, j] if c != "?"], minlength=4) / len(ids)
    onmarg.append(float(m[1] + m[2]))                      # B+C mass, the gold-heavy classes
R["D2_slot_exchangeability"] = dict(
    per_slot_acc=[float(x) for x in slot_acc], observed_spread=obs_spread,
    null_p95=float(np.percentile(null, 95)), null_mean=float(null.mean()),
    p_value=float((null >= obs_spread).mean()), nperm=2000,
    per_slot_letterA_rate=[float((L[:, j] == "A").mean()) for j in range(8)],
    gold_letterA_rate=float((gold == "A").mean()),
    per_slot_BC_mass=onmarg,
    corr_BCmass_vs_acc=float(np.corrcoef(onmarg, slot_acc)[0, 1]),
    mean_gen_tokens_per_slot=[float(np.mean([s8[i]["gen_tokens"][j] for i in ids])) for j in range(8)])

# ---- MCQ luck floor (mandatory) --------------------------------------------------------------
marg = np.bincount([ord(c) - 65 for c in gold], minlength=4) / len(gold)
rng2 = np.random.default_rng(SEED)
v8 = vote_ok(tuple(range(8)))
fake = rng2.choice(4, size=(200, len(ids)), p=marg)
floor_vote = float(np.mean([ (np.array([max(Counter(L[r]).items(), key=lambda kv: kv[1])[0] for r in range(len(ids))]) ==
                              np.array([chr(65+x) for x in f])).mean() for f in fake]))
R["D3_luck_floor"] = dict(gold_marginal=[float(x) for x in marg],
                          vote8_measured=float(v8.mean()),
                          vote8_under_random_gold_from_same_marginal=floor_vote,
                          clears_floor_by=float(v8.mean() - floor_vote))
json.dump(R, open(os.path.join(OUT, "partD.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
