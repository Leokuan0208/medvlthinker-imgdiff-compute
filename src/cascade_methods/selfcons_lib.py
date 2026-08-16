#!/usr/bin/env python3
"""selfcons_lib.py -- shared machinery for ATTACK 2 (self-consistency across all eight cells).

Pre-registration: results/cascade_methods/artifacts/self_consistency_suite_2026-08-17_preregistration.json

THE VOTE, EXACTLY (pre-specified in the prereg, implemented once here so eight cells cannot drift
into eight rules):

  * A pool is M slots in generation order.  Each slot carries (class_id, label) where class_id is the
    per-cell equivalence class of its answer and label is the grader's verdict on that slot's REAL
    generated string.
  * The vote over a subset S of the slots: the class with the most members of S wins; ties are broken
    by the SMALLEST slot index in S (the frozen `modal_pred` first-occurrence convention).  The
    RETURNED string is that earliest slot -- never a synthetic string.
  * At N < M the arm's value is the EXACT mean over all C(M,N) subsets, not a Monte-Carlo draw.
    C(8,1)=8, C(8,2)=28, C(8,4)=70, C(8,8)=1.  N=1 therefore reduces to a single random draw from the
    pool, i.e. the random-pick floor.

WHY EXACT MATTERS.  Protocol rule "3 seeds for anything sampled" exists because a single sampled draw
is noise.  Averaging over every subset is the limit of that -- it is the exact expectation, with zero
seed variance in the subsetting.  The only remaining seed dependence is the GENERATION of the 8-slot
pool itself, which is why the open cells report three independent generation seeds and the MCQ /
closed cells are explicitly flagged as single-pool at N=8.

CPU only.  No GPU, no model, no file writes on import.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selfcons_parts")
DATE = "2026-08-17"

NBOOT, BSEED = 10000, 20260817
NPERM, PSEED = 2000, 20260817
NS = [1, 2, 4, 8]

#: the eight macro cells and the published always-7B greedy value of each, with the grader that
#: DEFINES that value.  artifacts/cascade_selector_rerun_2026-08-05.json (per_cell_acc).
PUBLISHED_ALWAYS_7B = {
    "PMC_VQA":         (0.5427, "MedEvalKit judge_multi_choice"),
    "SLAKE_closed":    (0.8254, "MedEvalKit judge_close_end_vqa"),
    "VQA_RAD_closed":  (0.7809, "MedEvalKit judge_judgement"),
    "PATH_VQA_closed": (0.8409, "MedEvalKit judge_judgement"),
    "MedXpertQA-MM":   (0.2615, "MedEvalKit judge_multi_choice"),
    "SLAKE_open":      (0.7364, "32B LLM judge"),
    "VQA_RAD_open":    (0.4650, "32B LLM judge"),
    "PATH_VQA_open":   (0.3240, "32B LLM judge"),
}
CELL_ORDER = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
              "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MACRO_BASELINE = 0.5971      # mean of the eight values above, the round's stated baseline

#: MEASURED generation cost of N samples relative to one greedy answer, vLLM default (automatic
#: prefix caching ON -- what every generation in this project actually got).
#: artifacts/verifier_restructure_2026-08-16.json :
#:   Q1_generation_cost_and_prefill_sharing.per_config["count|default"].flopeq_rel_to_N1
FLOPEQ_BY_N = {1: 1.0, 2: 1.2600675815972704, 4: 1.6645496721798492, 8: 2.369969011752281}
FLOPEQ_SOURCE = ("artifacts/verifier_restructure_2026-08-16.json : Q1_generation_cost_and_prefill_"
                 "sharing.per_config['count|default'].flopeq_rel_to_N1 (MEASURED)")
#: the same measurement with the flag forced ON, and the cache-OFF control that validates the
#: instrument by reproducing the project's as-charged 8.0 convention.
FLOPEQ_BY_N_CACHE_ON = {1: 1.0, 2: 1.1393664485758466, 4: 1.5194129853535785,
                        8: 2.2605180144021815}
FLOPEQ_BY_N_CACHE_OFF = {1: 1.0, 2: 1.979661589593582, 4: 4.028884012567696,
                         8: 8.24449873332767}
#: N=8 with pre-computed image embeddings (the vision tower stops being re-run per sample)
FLOPEQ_N8_VISION_EMBEDS = 1.2026395573083093
FLOPEQ_N8_VISION_EMBEDS_SOURCE = ("artifacts/central_table_2026-08-16.json : COST_TABLE "
                                  "'+ vision-embeds fix (N=8)'.generation (MEASURED cost)")


# =============================================================================================
# 1. THE VOTE
# =============================================================================================
def vote_winner_slot(classes, subset):
    """Index (into the full pool) of the slot whose answer the vote returns for this subset.

    Winner = most-voted class; ties -> smallest slot index present in the subset.  The returned slot
    is the earliest member of the winning class inside the subset.
    """
    first, cnt = {}, Counter()
    for k in subset:
        c = classes[k]
        cnt[c] += 1
        if c not in first:
            first[c] = k
    best = min(cnt.items(), key=lambda kv: (-kv[1], first[kv[0]]))[0]
    return first[best]


def vote_exact(classes, labels, N):
    """EXACT E[label of the vote's returned answer] over all C(M, min(N,M)) subsets."""
    M = len(classes)
    Ne = min(N, M)
    tot, cnt = 0.0, 0
    for sub in combinations(range(M), Ne):
        tot += labels[vote_winner_slot(classes, sub)]
        cnt += 1
    assert cnt == comb(M, Ne), (cnt, M, Ne)
    return tot / cnt


def vote_exact_orderfree(classes, labels, N):
    """EXACT E[label] over all C(M,N) subsets AND a uniformly random SLOT ORDER.

    WHY THIS ARM EXISTS.  selfcons_slots.py measured that an n=8 vLLM pool with a fixed seed is NOT
    slot-exchangeable on 6 of 14 pools (PMC_VQA: per-slot accuracy spread 0.136 against a null p95 of
    0.016, p = 0.0).  A tie-break by slot INDEX therefore carries a position bias.  Under a random
    slot order the winner among the classes tied at the maximum vote count is the class holding the
    first slot, i.e. class x wins with probability count(x) / sum(counts of tied classes), and the
    returned string is a uniformly random slot of that class.  Both are exact, so no Monte Carlo.
    """
    M = len(classes)
    Ne = min(N, M)
    tot, cnt = 0.0, 0
    for sub in combinations(range(M), Ne):
        c = Counter(classes[k] for k in sub)
        mx = max(c.values())
        tied = [x for x, v in c.items() if v == mx]
        ntied = float(sum(c[x] for x in tied))
        val = 0.0
        for x in tied:
            sl = [k for k in sub if classes[k] == x]
            val += (c[x] / ntied) * (sum(labels[k] for k in sl) / len(sl))
        tot += val
        cnt += 1
    return tot / cnt


def vote_prefix(classes, labels, N):
    """The vote over the FIRST N slots of the pool, index tie-break.

    This is the arm that answers 'what would a smaller n= call have returned?' under the assumption
    that the fan is nested.  Its gap to vote_exact is a direct MEASUREMENT of how much the answer
    depends on WHICH slots of the fan you get, which the fixed-seed quantile structure makes a real
    question rather than a pedantic one."""
    M = len(classes)
    Ne = min(N, M)
    sub = tuple(range(Ne))
    return float(labels[vote_winner_slot(classes, sub)])


def vote_class_dist(classes, N):
    """EXACT distribution over WINNING CLASS ids at budget N.  Used by the random-gold luck floor,
    which needs P(the vote outputs class c) independently of whether c is the gold."""
    M = len(classes)
    Ne = min(N, M)
    d = Counter()
    for sub in combinations(range(M), Ne):
        d[classes[vote_winner_slot(classes, sub)]] += 1
    den = comb(M, Ne)
    return {c: v / den for c, v in d.items()}


def oracle_exact(labels, N):
    """EXACT P(at least one correct slot in a uniformly random N-subset) -- hypergeometric."""
    M = len(labels)
    Ne = min(N, M)
    k = int(sum(1 for x in labels if x))
    if M - k < Ne:
        return 1.0
    return 1.0 - comb(M - k, Ne) / comb(M, Ne)


def distinct_class_prob(classes, N):
    """EXACT P(class c appears in a uniformly random N-subset), per class.  For the oracle luck
    floor: an oracle over an N-subset is right iff the gold's class is present."""
    M = len(classes)
    Ne = min(N, M)
    out = {}
    for c in set(classes):
        m = sum(1 for x in classes if x == c)
        out[c] = 1.0 - (comb(M - m, Ne) / comb(M, Ne) if M - m >= Ne else 0.0)
    return out


# =============================================================================================
# 2. statistics
# =============================================================================================
def paired_boot(a, b, nboot=NBOOT, seed=BSEED):
    """Paired ITEM bootstrap of mean(a) - mean(b).  a and b are per-item values in [0,1]; for an
    exact-N arm they are per-item EXPECTATIONS, which is the same estimand the arm reports."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert len(a) == len(b)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        d[k] = a[s].mean() - b[s].mean()
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"delta": float(a.mean() - b.mean()), "ci": [lo, hi],
            "significant": bool(lo > 0 or hi < 0),
            "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
            "n": n, "nboot": nboot, "seed": seed}


def macro_boot(cell_pairs, nboot=NBOOT, seed=BSEED):
    """Bootstrap the MACRO (equal-weight mean of per-cell deltas) by resampling ITEMS INSIDE EACH
    CELL independently, which is the sampling model the per-cell CIs already use."""
    rng = np.random.default_rng(seed)
    names = list(cell_pairs)
    d = np.zeros(nboot)
    arrs = {c: (np.asarray(cell_pairs[c][0], float), np.asarray(cell_pairs[c][1], float))
            for c in names}
    for k in range(nboot):
        tot = 0.0
        for c in names:
            a, b = arrs[c]
            s = rng.integers(0, len(a), len(a))
            tot += a[s].mean() - b[s].mean()
        d[k] = tot / len(names)
    obs = float(np.mean([arrs[c][0].mean() - arrs[c][1].mean() for c in names]))
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"macro_delta": obs, "ci": [lo, hi], "significant": bool(lo > 0 or hi < 0),
            "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
            "n_cells": len(names), "nboot": nboot, "seed": seed}


def R(x, k=6):
    return round(float(x), k)


def rj(d):
    """round every float in a nested structure, for compact artifacts"""
    if isinstance(d, dict):
        return {k: rj(v) for k, v in d.items()}
    if isinstance(d, list):
        return [rj(v) for v in d]
    if isinstance(d, float):
        return round(d, 6)
    return d


def dump_part(name, obj):
    os.makedirs(PARTS, exist_ok=True)
    p = os.path.join(PARTS, name)
    json.dump(obj, open(p, "w"), indent=1, ensure_ascii=False)
    return p
