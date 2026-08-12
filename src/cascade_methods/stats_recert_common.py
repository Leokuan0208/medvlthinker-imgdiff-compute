#!/usr/bin/env python3
"""
stats_recert_common.py -- shared loaders + bootstrap machinery for ATTACK C
(re-certify our statistics against the four 2026 corrections flagged by
results/cascade_methods/docs/current/LITERATURE_UPDATE_2026-08-11.md).

NO GPU, NO NEW INFERENCE.  Everything is read from artifacts already on disk:
  * per-arm per-item eval vectors  results/cascade_methods/artifacts/_selector_rerun_parts/vec_*.npz
  * per-item cluster / question metadata  results/cascade_methods/artifacts/_stats_recert/meta_*.json
      (built by src/cascade_methods/stats_recert_meta.py, order-asserted against the same
       dataset files MedEvalKit reads)
  * the open-text 8-sample pools  ckpts/train/lora_verifier_disjoint/transfer_dump_*.json
  * the i.i.d. re-generation pools ckpts/openvqa/cheap_lingshu7b/ckpt_*_sc{8,16}.jsonl
    and ckpts/openvqa/cheap_lingshu7b_scale/ckpt_vqa_rad_open_lingshu7b_sc32.jsonl

Launch from the repo root.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.special import gammaln

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_selector_rerun_parts")
META = os.path.join(ART, "_stats_recert")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
CKSCALE = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b_scale")
DISJOINT = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")

# Variant-B reporting order (equal weight 1/8 per cell) -- macro_average_headline.py's order.
CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
CELLS_OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
CELLS_MCQ = [c for c in CELLS if c not in CELLS_OPEN]
OPEN_DS = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}

ARMS = ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
        "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

# FLOP-eq cost constants, verbatim from
# results/cascade_methods/artifacts/_selector_rerun_parts/summary_disjoint.json:cost_macro
# (one 7B direct forward = 1.0; one 32B direct forward = 4.57 as charged).
FLOP_7B = 1.0
FLOP_32B = 4.57
FLOP_32B_HONEST = 3.816   # the "honest re-cost" figure the brief pins; makes ratios WORSE for us

NBOOT = 10000
SEED = 20260811

norm = lambda s: str(s).strip().lower()


# ---------------------------------------------------------------------------------- loaders
def load_vec(source="disjoint"):
    """-> {cell: {arm: int8 vector}}  from the frozen per-arm eval vectors."""
    z = np.load(os.path.join(PARTS, f"vec_{source}.npz"))
    out = {}
    for c in CELLS:
        out[c] = {a: z[f"{c}|{a}"].astype(np.float64) for a in ARMS}
    return out


def load_meta(cell):
    return json.load(open(os.path.join(META, f"meta_{cell}.json")))


def cluster_ids(cell):
    """-> int array of cluster ids, in the per-item order of the eval vectors."""
    m = load_meta(cell)
    u = {}
    out = np.empty(len(m["cluster"]), np.int64)
    for i, c in enumerate(m["cluster"]):
        if c not in u:
            u[c] = len(u)
        out[i] = u[c]
    return out


def loadjl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def judge_map(p):
    return {r["idx"]: int(r["judge_ok"]) for r in loadjl(p)}


def pool_labels(ds, sc_tag):
    """{idx: [judge label per pool slot]} for an i.i.d. sc{8,16,32} pool.

    VERBATIM the reconstruction used by src/cascade_methods/verifier_n_scaling.py:pool_labels
    (which asserts equality against the deployed transfer dump).  The exploded judge file is
    DEDUPLICATED per normalised prediction string, so a slot's label is looked up by norm(pred).
    """
    base = os.path.join(CKSCALE if sc_tag == "sc32" else CK, f"ckpt_{ds}_lingshu7b_{sc_tag}")
    sc = {r["idx"]: r for r in loadjl(base + ".jsonl")}
    exp = {r["idx"]: r for r in loadjl(base + "_scexploded.jsonl")}
    jud = judge_map(base + "_scexploded.judge.jsonl")
    ans = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]
            oi = int(oi) if oi.lstrip("-").isdigit() else oi
            ans[oi][norm(r["modal_pred"])] = jud[cid]
    out = {}
    for i, r in sc.items():
        if i not in ans:
            continue
        lab = [ans[i].get(norm(a)) for a in r["preds"]]
        if any(x is None for x in lab):
            continue
        out[i] = lab
    return out, {i: r["preds"] for i, r in sc.items()}, ans


def load_transfer(ds):
    return json.load(open(os.path.join(DISJOINT, f"transfer_dump_{ds}_lingshu7b.json")))


# ------------------------------------------------------------------- exact subset combinatorics
def _logC(n, k):
    if k < 0 or k > n or n < 0:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def oracle_at_N(labels, N):
    """P(at least one correct among a uniform N-subset of this pool) -- EXACT, not simulated."""
    M = len(labels)
    k = int(sum(labels))
    if N > M:
        return None
    lp = _logC(M - k, N) - _logC(M, N)
    return float(1.0 - (math.exp(lp) if np.isfinite(lp) else 0.0))


# ------------------------------------------------------------------------------- bootstraps
def _multinomial_chunks(rng, n_draw, p, nboot, chunk):
    """Yield (B_i, len(p)) multinomial count matrices summing to n_draw -- memory-bounded.

    `p` is the per-category draw probability: for the ITEM scheme the categories are unique
    outcome patterns and p must be their empirical frequency; for the CLUSTER scheme the
    categories are clusters and p is uniform (each cluster is one exchangeable unit).
    """
    done = 0
    while done < nboot:
        b = min(chunk, nboot - done)
        yield rng.multinomial(n_draw, p, size=b)
        done += b


def cell_boot(mat, groups, nboot, rng, scheme, chunk=400):
    """Bootstrap the column means of `mat` (n_items x n_arms) for ONE cell.

    scheme='item'   : resample items with replacement (what the published CIs do).
    scheme='cluster': resample CLUSTERS with replacement (n_clusters draws), take every item in
                      each drawn cluster -- the RouteGuard (arXiv:2608.07583) re-certification.
    Returns (nboot, n_arms).  The SAME draw is used for every arm => paired across arms.
    """
    n, A = mat.shape
    if scheme == "item":
        # collapse to unique outcome patterns first (exactly MAH.cell_boot_means' trick)
        pats, inv = np.unique(mat, axis=0, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(pats)).astype(np.float64)
        out = np.empty((nboot, A))
        i = 0
        for m in _multinomial_chunks(rng, n, cnt / n, nboot, chunk):
            out[i:i + len(m)] = (m @ pats) / n
            i += len(m)
        return out
    if scheme == "cluster":
        K = int(groups.max()) + 1
        S = np.zeros((K, A))
        np.add.at(S, groups, mat)                       # per-cluster sums of each arm
        C = np.bincount(groups, minlength=K).astype(np.float64)   # per-cluster item counts
        out = np.empty((nboot, A))
        i = 0
        for m in _multinomial_chunks(rng, K, np.full(K, 1.0 / K), nboot, chunk):
            num = m @ S
            den = (m @ C)[:, None]
            out[i:i + len(m)] = num / den
            i += len(m)
        return out
    raise ValueError(scheme)


def ci(dist, point, alpha=0.05):
    lo, hi = np.percentile(dist, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(delta=round(float(point), 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                width=round(float(hi - lo), 4),
                sig=bool(lo > 0 or hi < 0),
                verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


def jdump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)
    print(f"[write] {path}")
