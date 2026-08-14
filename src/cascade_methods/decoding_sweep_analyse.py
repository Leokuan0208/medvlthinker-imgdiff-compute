#!/usr/bin/env python3
"""decoding_sweep_analyse.py -- SWEEP 1 endpoints, per decoding setting.

Every number is computed through the FROZEN metric (src/training_methods/genframe_data.py) on items
built in the FROZEN canonical order, so the paired bootstrap is item-aligned with the published bar and
with the matched control generated in the same session.

Endpoints per setting: oracle@8, sel_eff (frozen incumbent LoRA verifier), SELECTED accuracy (headline),
distinct-candidate count, Lincoln-Petersen capture-recapture ceiling, oracle-vs-N curve, token audit.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
import numpy as np
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G      # noqa: E402

SWEEP = os.path.join(ROOT, "ckpts/openvqa/decoding_sweep")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
NBOOT, SEED = 10000, 20260813


# ---------------------------------------------------------------- inputs
def load_judge():
    """{(ds, idx, na) -> judge_ok}. Reuses the project's own judge, from two sources:

      1. judgecache_preload_{ds}.jsonl -- labels the SAME judge already produced in earlier project
         dumps for the same (question, gold, candidate). Validated to reproduce the frozen transfer-dump
         labels 18733/18733 with 0 disagreements (decoding_sweep_judgepreload.py).
      2. judgein_{ds}.judge.jsonl -- labels judged in THIS round for strings never seen before.
    """
    lab = {}
    for ds in DS:
        pf = os.path.join(SWEEP, f"judgecache_preload_{ds}.jsonl")
        if os.path.exists(pf):
            for l in open(pf):
                if l.strip():
                    r = json.loads(l); lab[(ds, int(r["idx"]), r["na"])] = int(r["judge_ok"])
    for ds in DS:
        m = json.load(open(os.path.join(SWEEP, f"judgemap_{ds}.json")))
        jid2key = {v: k for k, v in m.items()}
        jf = os.path.join(SWEEP, f"judgein_{ds}.judge.jsonl")
        if not os.path.exists(jf):
            continue
        for l in open(jf):
            if not l.strip():
                continue
            r = json.loads(l)
            k = jid2key.get(r["idx"])
            if k is None:
                continue
            idx, na = k.split("|", 1)
            lab[(ds, int(idx), na)] = int(r["judge_ok"])
    return lab


def load_vscores():
    v = {}
    for f in glob.glob(os.path.join(SWEEP, "vscore_cache_shard*.jsonl")):
        for l in open(f):
            if l.strip():
                try:
                    r = json.loads(l); v[(r["ds"], r["idx"], r["ans"])] = float(r["pyes"])
                except Exception:
                    pass
    return v


EXPECT = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}


def load_pool(tag, strict=True):
    """{(ds, idx) -> record} for one generated pool.

    STRICT COMPLETENESS: a vLLM engine death once let a run print [done] on a file holding 1024/1500
    rows (2026-08-13). A short pool must never reach the metric, so it is refused here as well as at
    generation time.
    """
    out = {}
    for ds in DS:
        f = os.path.join(SWEEP, f"ckpt_{ds}_{tag}.jsonl")
        if not os.path.exists(f):
            return None
        n = 0
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[(ds, r["idx"])] = r; n += 1
        if n != EXPECT[ds]:
            msg = f"INCOMPLETE POOL {tag} {ds}: {n}/{EXPECT[ds]} rows"
            if strict:
                raise ValueError(msg)
            print("  !! " + msg); return None
    return out


# ---------------------------------------------------------------- item construction
def build_items(pool, lab, vsc, ref=None):
    """Items in the FROZEN canonical order, frozen schema. Returns (items, n_missing_judge)."""
    items, miss = [], 0
    for refit in (ref if ref is not None else G.load_items()):
        ds, idx = refit["ds"], refit["idx"]
        r = pool[(ds, idx)]
        preds = r["preds"]
        sl, scores = [], []
        for a in preds:
            y = lab.get((ds, idx, G.norm(a)))
            if y is None:
                miss += 1; y = 0
            sl.append(int(y))
            scores.append(vsc.get((ds, idx, a), G.MISSING_SCORE))
        modal_na = G.norm(r["modal_pred"])
        items.append({"ds": ds, "idx": idx, "sl": sl, "scores": scores, "preds": preds,
                      "gold": r.get("gold", ""),
                      "greedy_ok": int(lab.get((ds, idx, modal_na), 0)),
                      "gen_tokens_all": r.get("gen_tokens_all", [])})
    return items, miss


def oracle_curve_within8(items):
    """E[oracle@n] over random n-subsets of the 8 drawn samples -- closed form (hypergeometric)."""
    from math import comb
    out = []
    for n in range(1, 9):
        v = []
        for it in items:
            k = int(sum(it["sl"]))
            v.append(1.0 - (comb(8 - k, n) / comb(8, n) if 8 - k >= n else 0.0))
        out.append(float(np.mean(v)))
    return out


def oracle_curve_pooled(items_by_seed):
    """E[oracle@n] for n=1..8*S using all seeds' samples as one iid pool of the SAME setting."""
    from math import comb
    seeds = sorted(items_by_seed)
    M = 8 * len(seeds)
    per_item = []
    for i in range(len(items_by_seed[seeds[0]])):
        k = sum(int(sum(items_by_seed[s][i]["sl"])) for s in seeds)
        per_item.append(k)
    out = []
    for n in range(1, M + 1):
        v = [1.0 - (comb(M - k, n) / comb(M, n) if M - k >= n else 0.0) for k in per_item]
        out.append(float(np.mean(v)))
    return out


def lp_ceiling(items_by_seed):
    """Lincoln-Petersen two-sample capture-recapture, run A = one seed's 8-pool, run B = another's.

    Same estimator as coverage_diagnosis2.ceiling(). Heterogeneous per-item detection probability biases
    LP DOWNWARD, so this is a LOWER BOUND on the share of questions reachable by iid sampling at any N.
    """
    seeds = sorted(items_by_seed)
    pairs = [(a, b) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    per_cell, macro = defaultdict(list), []
    for a, b in pairs:
        IA, IB = items_by_seed[a], items_by_seed[b]
        shares = []
        for ds in DS:
            A = np.array([1 if 1 in it["sl"] else 0 for it in IA if it["ds"] == ds])
            B = np.array([1 if 1 in it["sl"] else 0 for it in IB if it["ds"] == ds])
            nA, nB, nAB = int(A.sum()), int(B.sum()), int((A & B).sum())
            share = (nA * nB / nAB) / len(A) if nAB else float("nan")
            per_cell[ds].append(share); shares.append(share)
        macro.append(float(np.mean(shares)))
    return {"pairs": [f"{a}x{b}" for a, b in pairs],
            "per_cell_reachable_share": {d: float(np.mean(per_cell[d])) for d in DS},
            "per_cell_sd": {d: float(np.std(per_cell[d], ddof=1)) if len(per_cell[d]) > 1 else 0.0 for d in DS},
            "macro_reachable_share_mean": float(np.mean(macro)),
            "macro_reachable_share_sd": float(np.std(macro, ddof=1)) if len(macro) > 1 else 0.0,
            "method": "Lincoln-Petersen, run A/B = two independent generation seeds of the SAME setting, "
                      "8 samples each, judge-labelled; LOWER BOUND (heterogeneity biases LP down)."}


def laterality_mask(items):
    """Item mask for the LATERALITY stratum, using the project's own regex
    (src/training_methods/visverif_lib.py LATERAL) applied to the gold answer.

    This is the mechanism endpoint for the repetition-penalty axis: vLLM's repetition_penalty
    penalises tokens present in the PROMPT as well as the output
    (vllm/model_executor/layers/utils.py apply_penalties -> apply_repetition_penalties(logits,
    prompt_mask, output_mask, ...)), and on laterality questions the correct one-word answer
    ('left'/'right'/'upper'/...) is very often a word of the question. It is also the incumbent
    verifier's weakest stratum (sel_eff 0.613043 vs 0.817186 on short non-laterality items).
    """
    sys.path.insert(0, ROOT)
    from src.training_methods.visverif_lib import LATERAL
    return np.array([bool(LATERAL.search(str(it["gold"]))) for it in items], dtype=bool)


def measure(items, with_selection=True):
    """Coverage endpoints always; SELECTION endpoints only when every slot has a verifier score.

    with_selection=False is the JUDGE-ONLY tier: oracle@8, distinct-candidate count, the
    capture-recapture ceiling and the token audit need judge labels only, so they are available for
    every generated setting. sel_eff and SELECTED need the frozen verifier and are reported only where
    it was run.
    """
    r = G.sel_eff({(it["ds"], it["idx"]): it["scores"] for it in items}, items=items)
    nd = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items], dtype=float)
    tok = [t for it in items for t in it["gen_tokens_all"]]
    out = {"n": r["n"], "oracle@8": r["oracle"], "greedy_modal": r["greedy"],
           "n_recoverable": r["n_recoverable"], "mean_distinct": float(nd.mean()),
           "per_ds": {d: {k: r["per_ds"][d][k] for k in ("n", "oracle")} for d in DS},
           "mean_gen_tokens": float(np.mean(tok)) if tok else None,
           "has_selection": bool(with_selection),
           "_arrays": {"rec": r["rec"], "con": r["contested_mask"], "ds_index": r["ds_index"]}}
    if with_selection:
        out.update({"sel_eff": r["sel_eff"], "selected": r["acc"],
                    "contested_n": r["contested"]["n"], "contested_sel_eff": r["contested"]["sel_eff"],
                    "identity_residual": abs(r["acc"] - r["oracle"] * r["sel_eff"])})
        for d in DS:
            out["per_ds"][d].update({k: r["per_ds"][d][k] for k in ("acc", "sel_eff")})
        out["_arrays"]["got"] = r["got"]
    return out


def boot(a, b, mask=None, nboot=NBOOT, seed=SEED):
    """Paired item bootstrap on a - b (equal length, item-aligned)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if mask is not None:
        a, b = a[mask], b[mask]
    rng = np.random.default_rng(seed)
    n = len(a); d = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(nboot, n))
    ds_ = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(ds_, [2.5, 97.5])
    return {"delta": d, "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0),
            "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"), "n": n}


# ======================================================================================
# EXACT-MATCH CURRENCY (secondary, clearly separated from the frozen judge currency)
# ======================================================================================
def em_items(tag, ref):
    """Per-item EM correctness of the 8 slots, in the FROZEN canonical item order.

    Currency = run_openvqa.py's `score()` (normalised exact match + short-answer 'contains'),
    recorded at generation time as `oks_em`. This is NOT the judge currency the published cells use
    (0.626013 oracle@8 is judge-labelled); EM runs ~0.026 lower on the same deployed pool. It is
    reported here because it needs no GPU, and the project has precedent for EM-currency coverage
    comparisons (coverage_diagnosis2: 'PathVQA-open, n=795, EXACT-MATCH currency on both sides').
    Only compare EM to EM.
    """
    pool = load_pool(tag)
    if pool is None:
        return None
    out = []
    for it in ref:
        r = pool[(it["ds"], it["idx"])]
        out.append({"ds": it["ds"], "idx": it["idx"], "sl": list(r["oks_em"]),
                    "preds": r["preds"], "gold": r.get("gold", ""),
                    "gen_tokens_all": r.get("gen_tokens_all", [])})
    return out


def em_measure(items):
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items], dtype=int)
    nd = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items], dtype=float)
    tok = [t for it in items for t in it["gen_tokens_all"]]
    ds_index = np.array([DS.index(it["ds"]) for it in items], dtype=int)
    return {"n": len(items), "em_oracle@8": float(rec.mean()),
            "mean_distinct": float(nd.mean()),
            "mean_gen_tokens": float(np.mean(tok)) if tok else None,
            "per_ds": {d: float(rec[ds_index == j].mean()) for j, d in enumerate(DS)},
            "_rec": rec, "_ds_index": ds_index}
