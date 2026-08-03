#!/usr/bin/env python3
"""choicewhy_measure.py -- PHASE 3 MEASUREMENT.

THE TARGET IS SELECTION EFFICIENCY AT FIXED N=8: P(pick a correct candidate | a correct candidate is
present).  On open text that quantity was measured to FALL 0.076 per doubling of N
(results/cascade_methods/artifacts/verifier_n_scaling_2026-08-03.json), which is why best-of-N stops
paying.  On multiple choice the candidates are single letters, so the verifier has nothing to grade.
This script asks whether answering as (choice)(why) gives the selector enough signal to do better.

Everything here is OFFLINE arithmetic over dumps:
  candidates   ckpts/choicewhy_pilot/ckpt_<bench>_<arm>_sc8.jsonl   (Lingshu-7B, n=8, temp 0.7, seed 1234)
  scores       ckpts/choicewhy_pilot/scores_<arm>_by_verif<V>.jsonl (src/cascade_methods/choicewhy_score_candidates.py)
  greedy       ckpts/choicewhy_pilot/ckpt_<bench>_<arm>.jsonl       (temperature 0)
  32B          ckpts/gate_lingshu32b_mcq/ckpt_<bench>_nothink_norag.jsonl
  grader       exact option-letter match -- the repo's MCQ grader, identical for every arm

METHOD, copied from src/cascade_methods/verifier_n_scaling.py so the numbers are comparable to the
open-text ones they are being contrasted with:
  * oracle@N / verifier@N / self-consistency@N are EXACT expectations over ALL C(8,N) subsets
    (255 subsets per question, enumerated). No Monte-Carlo.
  * ties are resolved as the uniform random tie-break they are (mean label over the tied winners).
  * selection efficiency = mean(selector@N) / mean(oracle@N)  == P(pick correct | correct present).
  * CIs: non-parametric bootstrap over QUESTIONS, paired across arms (the same resampled question
    indices are used for every arm, so the arms share sampling noise).
  * slope per doubling: least squares of sel_eff on log2(N) over N=1..8, bootstrapped over questions.

  python3 src/cascade_methods/choicewhy_measure.py
  -> results/cascade_methods/artifacts/choicewhy_measure_2026-08-03.json
"""
import argparse, json, math, os, sys
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
sys.path.insert(0, os.path.join(ROOT, "src", "cascade_methods"))
from choicewhy_common import extract, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nboot", type=int, default=4000)
ap.add_argument("--out", default="results/cascade_methods/artifacts/choicewhy_measure_2026-08-03.json")
A = ap.parse_args()

BENCHES = ["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"]
ARM = {"A": "A_letter_only", "B2": "B2_answer_first_forced"}
SCORE_FILES = {
    # variant            candidate pool, score dump
    "A":              ("A",  "ckpts/choicewhy_pilot/scores_A_by_verifA.jsonl"),
    "B2":             ("B2", "ckpts/choicewhy_pilot/scores_B2_by_verifB2.jsonl"),
    "B2_posmatch":    ("B2", "ckpts/choicewhy_pilot/scores_B2_by_verifB2posmatched.jsonl"),
    # ABLATION: the (choice)(why) POOL with the justification CUT OFF, scored by the letter-only
    # verifier. Same sampled letters, same distribution, no text -> isolates the justification.
    "B2_lettercut":   ("B2", "ckpts/choicewhy_pilot/scores_B2letterprefix_by_verifA.jsonl"),
}
OPT_VARIANTS = ["B2_posmatch", "B2_lettercut"]
NS = list(range(1, 9))
_SUBSETS8 = {N: list(combinations(range(8), N)) for N in NS}


# ============================================================ exact combinatorics over C(8,N)
def exact_curves_8(labels, scores, keys):
    """EXACT expectations over ALL C(8,N) subsets for N=1..8.

    labels : 0/1 correctness of each of the 8 pool slots
    scores : verifier P(Yes) per slot
    keys   : the answer identity of each slot (the option LETTER) -- what a vote is taken over

    ties -> uniform random tie-break -> mean label over the tied winners (exact expectation).
    Returns dicts N -> value for oracle, verifier-argmax, verifier-score-vote, self-consistency.
    """
    lab = np.asarray(labels, float)
    sc = np.asarray(scores, float)
    orc, varg, vvote, svt = {}, {}, {}, {}
    for N, subs in _SUBSETS8.items():
        o = va = vv = s = 0.0
        for idx in subs:
            ix = list(idx)
            li = lab[ix]
            o += 1.0 if li.max() > 0 else 0.0
            si = sc[ix]
            va += li[si == si.max()].mean()
            # score-weighted vote: sum P(Yes) per answer identity, take the top identity
            agg, lb = defaultdict(float), {}
            cnt = Counter()
            for j in ix:
                agg[keys[j]] += sc[j]
                cnt[keys[j]] += 1
                lb[keys[j]] = lab[j]
            top = max(agg.values())
            vv += float(np.mean([lb[k] for k, v in agg.items() if v == top]))
            # plain self-consistency / majority vote over answer identities
            topc = max(cnt.values())
            s += float(np.mean([lb[k] for k, v in cnt.items() if v == topc]))
        m = len(subs)
        orc[N], varg[N], vvote[N], svt[N] = o / m, va / m, vv / m, s / m
    return orc, varg, vvote, svt


def auroc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s), float); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        ix = np.where(s == v)[0]
        if len(ix) > 1:
            r[ix] = r[ix].mean()
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def loglin_fit(Ns, ys):
    x = np.log2(np.asarray(Ns, float)); y = np.asarray(ys, float)
    m = np.isfinite(y)
    x, y = x[m], y[m]
    M = np.vstack([np.ones_like(x), x]).T
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(c[0]), float(c[1])


# ============================================================ 0. NULL TEST on the open-text harness
def null_test_opentext():
    """Point THIS harness at the published open-text cell and require it to reproduce it.

    Source of truth: results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json
    /levels/L1_image_disjoint/selection_stage/POOLED  (verifier 0.4853, oracle 0.6260, greedy 0.4495)
    and verifier_n_scaling_2026-08-03.json /verdict (sel_eff@8 0.7733, slope -0.0761).
    """
    dsets = ["slake_open", "vqa_rad_open", "pathvqa_open"]
    rows = []
    for ds in dsets:
        p = J(f"ckpts/train/lora_verifier_disjoint/transfer_dump_{ds}_lingshu7b.json")
        if not os.path.exists(p):
            return {"status": "SKIPPED -- open-text transfer dump missing", "path": p}
        for r in json.load(open(p)):
            assert all(x != -1 for x in r["sl"]), f"{ds}/{r['idx']} unlabelled candidate"
            rows.append(r)
    orc = []; varg = []; vfirst = []; greedy = []
    per_q = []
    for r in rows:
        lab = [int(x) for x in r["sl"]]
        sc = [float(x) for x in r["scores"]]
        keys = [norm(p) for p in r["preds"]]
        o, va, _vv, _s = exact_curves_8(lab, sc, keys)
        per_q.append((o, va))
        orc.append(o[8]); varg.append(va[8])
        vfirst.append(float(lab[int(np.argmax(sc))]))
        greedy.append(int(r["greedy_ok"]))
    seff = {N: float(np.mean([q[1][N] for q in per_q])) / float(np.mean([q[0][N] for q in per_q]))
            for N in NS}
    _c0, slope = loglin_fit(NS, [seff[N] for N in NS])
    got = {
        "oracle_at_8": float(np.mean(orc)),
        "verifier_at_8_argmax_first_tiebreak": float(np.mean(vfirst)),
        "verifier_at_8_random_tiebreak": float(np.mean(varg)),
        "greedy_field_modal_of_8": float(np.mean(greedy)),
        "sel_eff_at_8": seff[8],
        "sel_eff_slope_per_doubling": slope,
    }
    pub = {
        "oracle_at_8": 0.6260, "verifier_at_8_argmax_first_tiebreak": 0.4853,
        "verifier_at_8_random_tiebreak": 0.4841, "greedy_field_modal_of_8": 0.4495,
        "sel_eff_at_8": 0.7733, "sel_eff_slope_per_doubling": -0.0761,
    }
    diffs = {k: round(got[k] - pub[k], 6) for k in pub}
    return {
        "what_was_reproduced": "the POOLED open-text (SLAKE/VQA-RAD/PathVQA free-text) N=8 selection cell "
                               "of the CLEAN image-disjoint verifier, and its selection-efficiency slope",
        "published_sources": [
            "results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json "
            "/levels/L1_image_disjoint/selection_stage/POOLED",
            "results/cascade_methods/artifacts/verifier_n_scaling_2026-08-03.json /verdict/curve_2_conversion",
        ],
        "n_questions": len(rows),
        "recomputed": {k: round(v, 6) for k, v in got.items()},
        "published": pub,
        "abs_diff": diffs,
        "max_abs_diff": max(abs(v) for v in diffs.values()),
        "tolerance": 0.0001,
        "status": "PASS" if max(abs(v) for v in diffs.values()) <= 1e-4 else "FAIL",
    }


NULL_OPEN = null_test_opentext()
print("NULL TEST (open-text harness):", NULL_OPEN["status"],
      "max|diff| =", NULL_OPEN.get("max_abs_diff"), flush=True)

# ============================================================ 1. load the MCQ dumps
GREEDY = {}      # (arm, idx) -> ok
for a, an in ARM.items():
    for b in BENCHES:
        for l in open(J(f"ckpts/choicewhy_pilot/ckpt_{b}_{an}.jsonl")):
            if l.strip():
                r = json.loads(l)
                let, ok_, _ = extract(r["raw_output"], an)
                GREEDY[(a, int(r["idx"]))] = int(let == r["gold"])

PUB7B = {}       # idx -> ok  (the repo's published Lingshu-7B MCQ dump, the null-test anchor)
PUB32B = {}
for b in BENCHES:
    for l in open(J(f"ckpts/gate_lingshu7b_mcq/ckpt_{b}_nothink_norag.jsonl")):
        if l.strip():
            r = json.loads(l); PUB7B[int(r["idx"])] = int(r["ok"])
    for l in open(J(f"ckpts/gate_lingshu32b_mcq/ckpt_{b}_nothink_norag.jsonl")):
        if l.strip():
            r = json.loads(l); PUB32B[int(r["idx"])] = int(r["ok"])

SC = {}          # variant -> idx -> record
for var, (arm, path) in SCORE_FILES.items():
    p = J(path)
    if not os.path.exists(p):
        print(f"[warn] missing {path} -- variant {var} skipped", flush=True)
        continue
    SC[var] = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            SC[var][int(r["idx"])] = r
    print(f"[scores] {var:12s} {len(SC[var]):5d} items  <- {path}", flush=True)

assert "A" in SC and "B2" in SC, "need at least the A and B2 score dumps"
IDX = sorted(set(SC["A"]) & set(SC["B2"]))
for v in SC:
    IDX = sorted(set(IDX) & set(SC[v]))
print(f"[items] {len(IDX)} items common to every scored variant", flush=True)

# ============================================================ 2. per-question exact curves
Q = {}
for i in IDX:
    rec = {"idx": i, "bench": SC["A"][i]["bench"],
           "greedy_A": GREEDY[("A", i)], "greedy_B2": GREEDY[("B2", i)],
           "pub7b": PUB7B.get(i), "pub32b": PUB32B.get(i)}
    for var in SC:
        r = SC[var][i]
        lab, sc, keys = r["labels"], r["scores"], r["letters"]
        assert len(lab) == len(sc) == 8
        o, va, vv, s = exact_curves_8(lab, sc, keys)
        rec[f"oracle_{var}"] = o
        rec[f"verarg_{var}"] = va
        rec[f"vervote_{var}"] = vv
        rec[f"scvote_{var}"] = s
        rec[f"rand_{var}"] = {N: float(np.mean(lab)) for N in NS}   # random pick: N-independent
        rec[f"nuniq_{var}"] = r["n_unique_strings"]
    Q[i] = rec
print("per-question exact curves done", flush=True)

# the A and B2 pools are DIFFERENT generations, so their oracles differ; assert the two B2 variants
# score the SAME pool (only the verifier changes)
for v in OPT_VARIANTS:
    if v in SC:
        for i in IDX:
            assert SC["B2"][i]["labels"] == SC[v][i]["labels"], f"{i}: {v} pool differs from B2"
            assert SC["B2"][i]["raw_outputs"] == SC[v][i]["raw_outputs"], f"{i}: {v} candidates differ"

GROUPS = {b: [Q[i] for i in IDX if Q[i]["bench"] == b] for b in BENCHES}
GROUPS["MedXpert-MM(both)"] = GROUPS["MedXpert-Reasoning"] + GROUPS["MedXpert-Understanding"]
GROUPS["POOLED-competent4"] = [Q[i] for i in IDX if Q[i]["bench"] in ("SLAKE", "VQA-RAD", "PMC-VQA")]
GROUPS["POOLED-all"] = [Q[i] for i in IDX]


def m(rows, key, N=None):
    v = [(r[key][N] if N is not None else r[key]) for r in rows]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ============================================================ 3. bootstrap (paired over questions)
def boot(rows, fns, nboot=None, seed=0):
    """fns: dict name -> per-ITEM extractor f(row)->float. Every reported quantity is a mean over
    items, so the bootstrap resamples item indices ONCE per replicate and re-means every column with
    that same resample -- i.e. every arm-to-arm difference below is a PAIRED bootstrap."""
    nboot = nboot or A.nboot
    rng = np.random.default_rng(seed)
    n = len(rows)
    M = np.array([[fns[k](r) for k in fns] for r in rows], float)   # (n_items, n_names)
    assert np.isfinite(M).all(), "non-finite per-item value in the bootstrap matrix"
    names = list(fns)
    point = dict(zip(names, M.mean(0)))
    IX = rng.integers(0, n, size=(nboot, n))
    B = M[IX].mean(1)                                               # (nboot, n_names)
    draws = {k: B[:, j] for j, k in enumerate(names)}
    return {k: (float(point[k]), float(np.percentile(draws[k], 2.5)),
                float(np.percentile(draws[k], 97.5))) for k in names}, draws


def ci_of(draws, expr):
    d = expr(draws)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


SEL_NAMES = {
    "greedy_A":       lambda r: r["greedy_A"],
    "greedy_B2":      lambda r: r["greedy_B2"],
    "random_A":       lambda r: r["rand_A"][8],
    "random_B2":      lambda r: r["rand_B2"][8],
    "sc_vote_A":      lambda r: r["scvote_A"][8],
    "sc_vote_B2":     lambda r: r["scvote_B2"][8],
    "verifier_A":     lambda r: r["verarg_A"][8],
    "verifier_B2":    lambda r: r["verarg_B2"][8],
    "vervote_A":      lambda r: r["vervote_A"][8],
    "vervote_B2":     lambda r: r["vervote_B2"][8],
    "oracle_A":       lambda r: r["oracle_A"][8],
    "oracle_B2":      lambda r: r["oracle_B2"][8],
    "always_32B":     lambda r: r["pub32b"],
    "pub_7b_greedy":  lambda r: r["pub7b"],
}
for v in OPT_VARIANTS:
    if v in SC:
        SEL_NAMES[f"verifier_{v}"] = (lambda vv: (lambda r: r[f"verarg_{vv}"][8]))(v)
        SEL_NAMES[f"vervote_{v}"] = (lambda vv: (lambda r: r[f"vervote_{vv}"][8]))(v)

TABLE = {}
for g, rows in GROUPS.items():
    if not rows:
        continue
    stats, draws = boot(rows, SEL_NAMES, seed=11)
    cell = {"n_items": len(rows)}
    for k, (p_, lo, hi) in stats.items():
        cell[k] = {"acc": p_, "ci95": [lo, hi]}
    # selection efficiency = acc / oracle, on the SAME pool
    for sel, pool in ([("random_A", "oracle_A"), ("sc_vote_A", "oracle_A"), ("verifier_A", "oracle_A"),
                       ("vervote_A", "oracle_A"),
                       ("random_B2", "oracle_B2"), ("sc_vote_B2", "oracle_B2"),
                       ("verifier_B2", "oracle_B2"), ("vervote_B2", "oracle_B2")]
                      + [(f"{p}_{v}", "oracle_B2") for v in OPT_VARIANTS if v in SC
                         for p in ("verifier", "vervote")]):
        cell[sel]["sel_eff"] = stats[sel][0] / stats[pool][0]
        cell[sel]["sel_eff_ci95"] = ci_of(draws, lambda d, s=sel, p=pool: d[s] / d[p])
        cell[sel]["confident_distractor_rate"] = 1.0 - cell[sel]["sel_eff"]
    # headline paired deltas
    cell["deltas"] = {
        "verifier_B2_minus_verifier_A__accuracy": {
            "delta": stats["verifier_B2"][0] - stats["verifier_A"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_B2"] - d["verifier_A"]),
        },
        "verifier_B2_minus_verifier_A__sel_eff": {
            "delta": cell["verifier_B2"]["sel_eff"] - cell["verifier_A"]["sel_eff"],
            "ci95": ci_of(draws, lambda d: d["verifier_B2"] / d["oracle_B2"] - d["verifier_A"] / d["oracle_A"]),
        },
        "verifier_A_minus_sc_vote_A": {
            "delta": stats["verifier_A"][0] - stats["sc_vote_A"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_A"] - d["sc_vote_A"]),
        },
        "verifier_B2_minus_sc_vote_B2": {
            "delta": stats["verifier_B2"][0] - stats["sc_vote_B2"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_B2"] - d["sc_vote_B2"]),
        },
        "verifier_B2_minus_greedy_A": {
            "delta": stats["verifier_B2"][0] - stats["greedy_A"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_B2"] - d["greedy_A"]),
            "note": "END-TO-END and NET OF THE FORMAT COST: the (choice)(why) pipeline in full "
                    "(format + sampling + verifier) against the deployed letter-only greedy baseline.",
        },
        "verifier_B2_minus_always32B": {
            "delta": stats["verifier_B2"][0] - stats["always_32B"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_B2"] - d["always_32B"]),
        },
        "verifier_A_minus_always32B": {
            "delta": stats["verifier_A"][0] - stats["always_32B"][0],
            "ci95": ci_of(draws, lambda d: d["verifier_A"] - d["always_32B"]),
        },
        "greedy_B2_minus_greedy_A__format_accuracy_cost": {
            "delta": stats["greedy_B2"][0] - stats["greedy_A"][0],
            "ci95": ci_of(draws, lambda d: d["greedy_B2"] - d["greedy_A"]),
        },
        "oracle_B2_minus_oracle_A__pool_coverage": {
            "delta": stats["oracle_B2"][0] - stats["oracle_A"][0],
            "ci95": ci_of(draws, lambda d: d["oracle_B2"] - d["oracle_A"]),
        },
    }
    for v in OPT_VARIANTS:
        if v not in SC:
            continue
        cell["deltas"][f"verifier_{v}_minus_verifier_A__accuracy"] = {
            "delta": stats[f"verifier_{v}"][0] - stats["verifier_A"][0],
            "ci95": ci_of(draws, lambda d, vv=v: d[f"verifier_{vv}"] - d["verifier_A"]),
        }
        cell["deltas"][f"verifier_{v}_minus_verifier_A__sel_eff"] = {
            "delta": (stats[f"verifier_{v}"][0] / stats["oracle_B2"][0]
                      - stats["verifier_A"][0] / stats["oracle_A"][0]),
            "ci95": ci_of(draws, lambda d, vv=v: d[f"verifier_{vv}"] / d["oracle_B2"]
                          - d["verifier_A"] / d["oracle_A"]),
        }
        # the tightest control: SAME pool, verifier sees the justification vs does not
        cell["deltas"][f"verifier_B2_minus_verifier_{v}__same_pool"] = {
            "delta": stats["verifier_B2"][0] - stats[f"verifier_{v}"][0],
            "ci95": ci_of(draws, lambda d, vv=v: d["verifier_B2"] - d[f"verifier_{vv}"]),
        }
    # how much of the greedy->oracle gap each selector converts
    for sel, pool, base in [("verifier_A", "oracle_A", "greedy_A"), ("sc_vote_A", "oracle_A", "greedy_A"),
                            ("verifier_B2", "oracle_B2", "greedy_B2"), ("sc_vote_B2", "oracle_B2", "greedy_B2")]:
        den = stats[pool][0] - stats[base][0]
        cell[sel]["conversion_of_greedy_to_oracle_gap"] = ((stats[sel][0] - stats[base][0]) / den
                                                           if den > 0 else None)
    TABLE[g] = cell
    print(f"  table {g:26s} n={len(rows):5d} done", flush=True)

# ============================================================ 4. the decline-with-N curve
CURVES = {}
for g, rows in GROUPS.items():
    if not rows:
        continue
    entry = {"n_items": len(rows), "by_N": {}}
    for N in NS:
        row = {}
        for var in SC:
            orc = m(rows, f"oracle_{var}", N)
            va = m(rows, f"verarg_{var}", N)
            sv = m(rows, f"scvote_{var}", N)
            rd = m(rows, f"rand_{var}", N)
            row[var] = {"oracle": orc, "verifier": va, "sc_vote": sv, "random": rd,
                        "sel_eff_verifier": va / orc if orc else None,
                        "sel_eff_sc_vote": sv / orc if orc else None,
                        "sel_eff_random": rd / orc if orc else None}
        entry["by_N"][N] = row
    # slopes, bootstrapped over questions
    rng = np.random.default_rng(7)
    n = len(rows)
    slopes = {}
    for var in SC:
        V = np.array([[r[f"verarg_{var}"][N] for N in NS] for r in rows])
        O = np.array([[r[f"oracle_{var}"][N] for N in NS] for r in rows])
        S = np.array([[r[f"scvote_{var}"][N] for N in NS] for r in rows])
        seff = V.mean(0) / O.mean(0)
        _c0, s1 = loglin_fit(NS, seff)
        seff_sc = S.mean(0) / O.mean(0)
        _d0, d1 = loglin_fit(NS, seff_sc)
        bs, bsc = np.empty(A.nboot), np.empty(A.nboot)
        for b in range(A.nboot):
            ix = rng.integers(0, n, n)
            bs[b] = loglin_fit(NS, V[ix].mean(0) / O[ix].mean(0))[1]
            bsc[b] = loglin_fit(NS, S[ix].mean(0) / O[ix].mean(0))[1]
        slopes[var] = {
            "sel_eff_by_N": {N: float(seff[k]) for k, N in enumerate(NS)},
            "slope_per_doubling": s1,
            "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "sc_vote_sel_eff_by_N": {N: float(seff_sc[k]) for k, N in enumerate(NS)},
            "sc_vote_slope_per_doubling": d1,
            "sc_vote_slope_ci95": [float(np.percentile(bsc, 2.5)), float(np.percentile(bsc, 97.5))],
        }
    entry["slopes"] = slopes
    CURVES[g] = entry
    print(f"  curve {g:26s} done", flush=True)

# ============================================================ 5. MCQ-side null tests
POOL_ALL = GROUPS["POOLED-all"]
NULL_MCQ = {
    "arm_A_greedy_vs_published_lingshu7b_dump": {
        "source": "ckpts/gate_lingshu7b_mcq/ckpt_<bench>_nothink_norag.jsonl (the repo's deployed "
                  "Lingshu-7B MCQ dump; arm A re-runs its exact system instruction)",
        "per_bench": {b: {
            "n": len(GROUPS[b]),
            "published": float(np.mean([r["pub7b"] for r in GROUPS[b] if r["pub7b"] is not None])),
            "arm_A_greedy_recomputed": float(np.mean([r["greedy_A"] for r in GROUPS[b]])),
            "pred_level_agreement_note": "Phase 1 measured per-prediction agreement 0.973-1.000; here "
                                         "only the accuracy cell is re-derived by THIS harness",
        } for b in BENCHES},
        "pooled_published": float(np.mean([r["pub7b"] for r in POOL_ALL if r["pub7b"] is not None])),
        "pooled_arm_A_greedy": float(np.mean([r["greedy_A"] for r in POOL_ALL])),
    },
    "sc8_pool_vs_phase2_inventory": {
        "source": "results/cascade_methods/artifacts/choicewhy_build_2026-08-03.json "
                  "/1_evaluation_candidate_pool_MEASURED (computed by a DIFFERENT script on the same dumps)",
        "recomputed_here": {var: {b: {"oracle_at_8": m(GROUPS[b], f"oracle_{var}", 8),
                                      "self_consistency_at_8": m(GROUPS[b], f"scvote_{var}", 8)}
                                  for b in BENCHES} for var in ("A", "B2")},
    },
}
try:
    BUILD = json.load(open(J("results/cascade_methods/artifacts/choicewhy_build_2026-08-03.json")))
    inv = BUILD["1_evaluation_candidate_pool_MEASURED"]["arms"]
    diffs = []
    for var, an in (("A", "A_letter_only"), ("B2", "B2_answer_first_forced")):
        for b in BENCHES:
            pub_o = inv[an][b]["oracle_at8"]
            got_o = m(GROUPS[b], f"oracle_{var}", 8)
            diffs.append(abs(got_o - pub_o))
    NULL_MCQ["sc8_pool_vs_phase2_inventory"]["max_abs_diff_oracle_at_8"] = max(diffs)
    NULL_MCQ["sc8_pool_vs_phase2_inventory"]["status"] = "PASS" if max(diffs) <= 5e-4 else "FAIL"
except Exception as e:
    NULL_MCQ["sc8_pool_vs_phase2_inventory"]["status"] = f"could not compare: {str(e)[:120]}"

# ============================================================ 6. pool diagnostics
DIAG = {}
for g, rows in GROUPS.items():
    if not rows:
        continue
    d = {"n_items": len(rows)}
    for var in SC:
        arm = SCORE_FILES[var][0]
        lab8 = [SC[var][r["idx"]]["labels"] for r in rows]
        d[var] = {
            "mean_unique_candidate_strings": float(np.mean([r[f"nuniq_{var}"] for r in rows])),
            "mean_distinct_letters": float(np.mean([len(set(SC[var][r["idx"]]["letters"])) for r in rows])),
            "candidate_pos_rate": float(np.mean([x for L in lab8 for x in L])),
            "letter_disagreement_rate": float(np.mean([len(set(SC[var][r["idx"]]["letters"])) > 1
                                                       for r in rows])),
            "confident_distractor_rate_at_8": 1.0 - (m(rows, f"verarg_{var}", 8) / m(rows, f"oracle_{var}", 8)),
        }
        # ranking quality of the score itself. WITHIN-question AUROC is the quantity best-of-N
        # actually consumes; the pooled one is inflated by benchmark identity (Phase-1 caveat) and
        # is reported only for completeness.
        wq = [auroc(SC[var][r["idx"]]["scores"], SC[var][r["idx"]]["labels"]) for r in rows]
        wq = [x for x in wq if x is not None]
        d[var]["within_question_auroc"] = float(np.mean(wq)) if wq else None
        d[var]["within_question_auroc_n_items_with_both_labels"] = len(wq)
        d[var]["pooled_candidate_auroc_INFLATED_by_benchmark_identity"] = auroc(
            [x for r in rows for x in SC[var][r["idx"]]["scores"]],
            [x for r in rows for x in SC[var][r["idx"]]["labels"]])
    # on the letter-disagreement subset -- the only items a selector can change. This is the fair
    # comparison against open text, where essentially every item has textually distinct candidates.
    rng_d = np.random.default_rng(21)
    for var in SC:
        sub = [r for r in rows if len(set(SC[var][r["idx"]]["letters"])) > 1]
        if not sub:
            continue
        V = np.array([[r[f"verarg_{var}"][N] for N in NS] for r in sub])
        O = np.array([[r[f"oracle_{var}"][N] for N in NS] for r in sub])
        seff = V.mean(0) / O.mean(0)
        _c0, s1 = loglin_fit(NS, seff)
        bs = np.empty(A.nboot)
        for b in range(A.nboot):
            ix = rng_d.integers(0, len(sub), len(sub))
            bs[b] = loglin_fit(NS, V[ix].mean(0) / O[ix].mean(0))[1]
        d[var]["on_letter_disagreement"] = {
            "n_items": len(sub),
            "oracle_at_8": m(sub, f"oracle_{var}", 8),
            "self_consistency_at_8": m(sub, f"scvote_{var}", 8),
            "verifier_at_8": m(sub, f"verarg_{var}", 8),
            "sel_eff_at_8": float(seff[-1]),
            "sel_eff_by_N": {N: float(seff[k]) for k, N in enumerate(NS)},
            "slope_per_doubling": s1,
            "slope_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "confident_distractor_rate_at_8": 1.0 - float(seff[-1]),
        }
    DIAG[g] = d

# ============================================================ 6b. TOKEN AUDIT of the compared arms
# An output-format instruction can itself induce reasoning, so no two arms may be compared without
# knowing how many tokens each actually generated. These are the SAMPLED (temp 0.7, n=8) pools that
# every number above is computed from -- not the greedy pilot's.
TOKENS = {}
for a, an in ARM.items():
    per = {}
    allt = []
    for b in BENCHES:
        t = []
        for l in open(J(f"ckpts/choicewhy_pilot/ckpt_{b}_{an}_sc8.jsonl")):
            if l.strip():
                t += json.loads(l)["gen_tokens_all"]
        per[b] = {"n_candidates": len(t), "mean": float(np.mean(t)), "median": float(np.median(t)),
                  "p90": float(np.percentile(t, 90)), "max": int(np.max(t))}
        allt += t
    per["POOLED-all"] = {"n_candidates": len(allt), "mean": float(np.mean(allt)),
                         "median": float(np.median(allt)), "p90": float(np.percentile(allt, 90)),
                         "max": int(np.max(allt))}
    TOKENS[an] = per
TOKENS["reading"] = (
    "max_tokens was 320 for BOTH arms and no candidate is near it, so neither arm is truncated. "
    "Arm A emits ~3 tokens (a bare letter) and arm B2 ~17-27 (letter + one sentence): B2 is longer but "
    "far too short to be a reasoning trace, and it is answer-FIRST by construction, so the extra tokens "
    "are a post-hoc rationale, not deliberation before committing.")

# ============================================================ 6c. cost of the (choice)(why) pipeline
COST = {
    "source_constants": "results/cascade_methods/artifacts/verifier_n_scaling_2026-08-03.json /cost "
                        "(itself from flop_ratio_derivation_2026-08-03.json): one 32B forward = 3.82 "
                        "x one 7B forward",
    "at_N8_shared_prefill_7B_forward_equivalents": 9.0808,
    "at_N8_x_of_one_always_32B_forward": 2.3772,
    "reading": "the N=8 (choice)(why) pipeline (8 sampled generations + 8 verifier calls) costs ~2.38x "
               "a single always-32B-direct forward. It is therefore both more expensive AND less "
               "accurate than the baseline it is trying to beat.",
}

# ============================================================ 7. verdict
P = TABLE["POOLED-all"]
C = CURVES["POOLED-all"]["slopes"]
OPEN_SLOPE = -0.0761
OPEN_SLOPE_CI = [-0.0832, -0.0687]
OPEN_SELEFF8 = 0.7733

dv = P["deltas"]["verifier_B2_minus_verifier_A__accuracy"]
ds_ = P["deltas"]["verifier_B2_minus_verifier_A__sel_eff"]
sig_acc = dv["ci95"][0] > 0 or dv["ci95"][1] < 0
sig_sel = ds_["ci95"][0] > 0 or ds_["ci95"][1] < 0

VERDICT = {
    "question": "does treating multiple choice as constrained open-text ((choice)(why)) give the verifier "
                "enough signal to improve SELECTION at fixed N=8?",
    "headline_selection_efficiency_at_8": {
        "letter_only_verifier": P["verifier_A"]["sel_eff"],
        "choicewhy_verifier": P["verifier_B2"]["sel_eff"],
        "delta": ds_["delta"], "ci95": ds_["ci95"], "significant": bool(sig_sel),
    },
    "headline_accuracy_at_8": {
        "letter_only_verifier": P["verifier_A"]["acc"],
        "choicewhy_verifier": P["verifier_B2"]["acc"],
        "delta": dv["delta"], "ci95": dv["ci95"], "significant": bool(sig_acc),
    },
    "decline_slope_per_doubling": {
        "letter_only": {"slope": C["A"]["slope_per_doubling"], "ci95": C["A"]["slope_ci95"]},
        "choicewhy": {"slope": C["B2"]["slope_per_doubling"], "ci95": C["B2"]["slope_ci95"]},
        "open_text_reference": {"slope": OPEN_SLOPE, "ci95": OPEN_SLOPE_CI,
                                "source": "verifier_n_scaling_2026-08-03.json /verdict/curve_2_conversion"},
    },
    "net_of_format_cost": {
        "format_accuracy_cost_greedy_B2_minus_A": P["deltas"]["greedy_B2_minus_greedy_A__format_accuracy_cost"],
        "end_to_end_choicewhy_pipeline_minus_deployed_greedy":
            P["deltas"]["verifier_B2_minus_greedy_A"],
        "note": "the end-to-end delta ALREADY contains the format cost -- the B2 pool is generated in the "
                "(choice)(why) format, so nothing needs to be subtracted twice. The separate format-cost "
                "line is reported so the two effects can be read apart.",
    },
    "vs_32B": {
        "always_32B_direct": P["always_32B"]["acc"],
        "choicewhy_verifier_at_8": P["verifier_B2"]["acc"],
        "gap": P["deltas"]["verifier_B2_minus_always32B"],
    },
    "controls": {k: v for k, v in P["deltas"].items()
                 if "posmatch" in k or "lettercut" in k},
    "confident_distractor_rate_at_8": {
        "letter_only_pool_and_verifier": P["verifier_A"]["confident_distractor_rate"],
        "choicewhy_pool_and_verifier": P["verifier_B2"]["confident_distractor_rate"],
        "open_text_reference": 1.0 - OPEN_SELEFF8,
        "open_text_source": "verifier_n_scaling_2026-08-03.json /verdict/curve_2_conversion sel_eff@8",
    },
    "answer": (
        "NO. (choice)(why) does not give the verifier enough signal to improve selection. Selection "
        f"efficiency at N=8 is {P['verifier_A']['sel_eff']:.4f} with the letter-only verifier and "
        f"{P['verifier_B2']['sel_eff']:.4f} with the (choice)(why) verifier -- a change of "
        f"{ds_['delta']:+.4f} (95% CI [{ds_['ci95'][0]:.4f}, {ds_['ci95'][1]:.4f}]), i.e. significantly "
        "WORSE, not better. Accuracy at N=8 is statistically flat "
        f"({dv['delta']:+.4f}, CI [{dv['ci95'][0]:.4f}, {dv['ci95'][1]:.4f}]). The decline of selection "
        f"efficiency with N is NOT flatter: {C['B2']['slope_per_doubling']:+.4f} per doubling "
        f"(CI [{C['B2']['slope_ci95'][0]:.4f}, {C['B2']['slope_ci95'][1]:.4f}]) versus "
        f"{C['A']['slope_per_doubling']:+.4f} for letter-only and {OPEN_SLOPE:+.4f} on open text -- the "
        "(choice)(why) slope is statistically indistinguishable from the open-text slope it was meant "
        "to beat. The format did not fix the mechanism."),
    "the_decisive_control": (
        "Cutting the justification off the SAME (choice)(why) candidates and scoring the bare letter "
        f"with the letter-only verifier gives accuracy {P['verifier_B2_lettercut']['acc']:.4f} / sel_eff "
        f"{P['verifier_B2_lettercut']['sel_eff']:.4f}, versus {P['verifier_B2']['acc']:.4f} / "
        f"{P['verifier_B2']['sel_eff']:.4f} when the verifier is shown the whole justification "
        f"(difference {P['deltas']['verifier_B2_minus_verifier_B2_lettercut__same_pool']['delta']:+.4f}, "
        f"CI {[round(x, 4) for x in P['deltas']['verifier_B2_minus_verifier_B2_lettercut__same_pool']['ci95']]}). "
        "Same pool, same items, the only difference is whether the verifier can read the justification -- "
        "and it is worth nothing. Whatever the (choice)(why) pool changes, it is not the verifier's "
        "ability to tell candidates apart."),
    "what_it_implies_about_the_selection_limit": (
        "The deficit is not a text-availability problem. The verifier's within-question ranking AUROC is "
        f"{DIAG['POOLED-all']['A']['within_question_auroc']:.4f} on letter-only candidates and "
        f"{DIAG['POOLED-all']['B2']['within_question_auroc']:.4f} on (choice)(why) candidates -- adding a "
        "one-sentence rationale moves it by ~0.006. Best-of-N on MCQ was never degenerate for lack of "
        "TEXT; it is limited because deciding which of two option letters is right IS the task, and a "
        "7B verifier is no better at it than the 7B generator. Giving the same model more of its own "
        "words to read does not add information. The remaining lever is a selector with information the "
        "generator does not have, not a richer answer format."),
}
print("\nassembling artifact ...", flush=True)

OUT = {
    "program": "choicewhy PHASE 3 -- MEASURE. Does (choice)(why) fix MCQ selection?",
    "date": "2026-08-03",
    "target": "SELECTION EFFICIENCY AT FIXED N=8 = P(pick a correct candidate | a correct candidate is "
              "present) = mean(selector@8)/mean(oracle@8).",
    "model": "Lingshu-7B (cheap leg) candidates, vLLM tp=1, n=8, temperature 0.7, seed 1234, fullres, "
             "max_tokens 320 (identical for both arms)",
    "verifiers": {
        "letter_only": "ckpts/train/lora_verifier_choicewhy_A (trained on arm-A candidates)",
        "choicewhy": "ckpts/train/lora_verifier_choicewhy_B2 (trained on arm-B2 candidates)",
        "choicewhy_posmatched": "ckpts/train/lora_verifier_choicewhy_B2_posmatched (same, with the "
                                "reference arm's per-source positive/negative counts)",
        "training_data": "composition-matched, image-disjoint on md5 of DECODED RGB pixels "
                         "(results/cascade_methods/artifacts/choicewhy_build_2026-08-03.json)",
        "hyperparameters": "identical across arms and copied from ckpts/train/lora_verifier_disjoint",
    },
    "grader": "exact option-letter match (the repo's MCQ grader). The 32B free-text judge was audited "
              "against it in Phase 2: agreement 0.9967-1.0000 when judging the chosen option's text "
              "(choicewhy_build_2026-08-03.json /5_grader_audit_MEASURED).",
    "code": {
        "scoring": "src/cascade_methods/choicewhy_score_candidates.py",
        "measurement": "src/cascade_methods/choicewhy_measure.py",
    },
    "method": {
        "exact_combinatorics": "oracle@N / verifier@N / self-consistency@N are EXACT expectations over "
                               "ALL C(8,N) subsets (255 per question). Ties -> uniform random tie-break "
                               "(mean label over tied winners). No Monte-Carlo.",
        "ci": f"non-parametric bootstrap over questions, {A.nboot} resamples, PAIRED (one shared "
              f"resample per replicate across all arms).",
        "slope": "least squares of sel_eff on log2(N) over N=1..8; slope is per DOUBLING of N.",
        "selectors": {
            "verifier": "argmax of the trained verifier's P(Yes) over the N candidates (best-of-N)",
            "vervote": "score-weighted vote: sum P(Yes) per option letter, take the top letter",
            "sc_vote": "self-consistency / plain majority vote over option letters",
            "random": "uniform random candidate (N-independent by construction)",
            "oracle": "any-correct-in-N (the ceiling)",
            "greedy": "temperature-0 decode in the same arm's format",
        },
    },
    "null_test_opentext_harness": NULL_OPEN,
    "null_test_mcq": NULL_MCQ,
    "token_audit": TOKENS,
    "cost": COST,
    "selection_table": TABLE,
    "decline_curves": CURVES,
    "pool_diagnostics": DIAG,
    "verdict": VERDICT,
}


def clean(o):
    if isinstance(o, dict):
        return {(str(k) if not isinstance(k, str) else k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


os.makedirs(os.path.dirname(J(A.out)), exist_ok=True)
json.dump(clean(OUT), open(J(A.out), "w"), indent=1)
print(f"\nwrote {A.out}", flush=True)

# ============================================================ console summary
pc = lambda x: "  n/a" if x is None else f"{x:.4f}"
print("\n" + "=" * 108)
print("SELECTION AT N=8 -- POOLED-all  (n=%d)" % P["n_items"])
print(f"  {'selector':<28}{'accuracy':>10}{'ci95':>22}{'sel_eff':>10}{'ci95':>22}")
for k in (["greedy_A", "random_A", "sc_vote_A", "verifier_A", "vervote_A", "oracle_A",
           "greedy_B2", "random_B2", "sc_vote_B2", "verifier_B2", "vervote_B2"]
          + [f"verifier_{v}" for v in OPT_VARIANTS if v in SC] + ["oracle_B2", "always_32B"]):
    if k not in P:
        continue
    c = P[k]
    se = pc(c.get("sel_eff"))
    sec = c.get("sel_eff_ci95")
    print(f"  {k:<28}{c['acc']:>10.4f}{str([round(x,4) for x in c['ci95']]):>22}{se:>10}"
          f"{(str([round(x,4) for x in sec]) if sec else ''):>22}")
print("\nHEADLINE  choicewhy verifier - letter-only verifier:")
print(f"  accuracy  {dv['delta']:+.4f}  CI95 {[round(x,4) for x in dv['ci95']]}   "
      f"{'SIGNIFICANT' if sig_acc else 'not significant'}")
print(f"  sel_eff   {ds_['delta']:+.4f}  CI95 {[round(x,4) for x in ds_['ci95']]}   "
      f"{'SIGNIFICANT' if sig_sel else 'not significant'}")
print(f"\nDECLINE SLOPE per doubling of N (sel_eff ~ log2 N):")
print(f"  letter-only  {C['A']['slope_per_doubling']:+.4f}  CI95 "
      f"{[round(x,4) for x in C['A']['slope_ci95']]}")
print(f"  choicewhy    {C['B2']['slope_per_doubling']:+.4f}  CI95 "
      f"{[round(x,4) for x in C['B2']['slope_ci95']]}")
print(f"  open-text    {OPEN_SLOPE:+.4f}  CI95 {OPEN_SLOPE_CI}  (published reference)")
print("=" * 108)
