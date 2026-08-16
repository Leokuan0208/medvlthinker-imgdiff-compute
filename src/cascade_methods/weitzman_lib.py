#!/usr/bin/env python3
"""weitzman_lib.py -- vectorised, EXACT re-implementation of the deployed Weitzman/Pandora
open-text controller, plus the pool loaders the T=0.4 refit needs.

WHY A RE-IMPLEMENTATION.  pandora_controller.run_pandora is a per-item Python loop; the refit here
needs ~10^8 policy evaluations (91 lambdas x 5 outer x 5 inner folds x 10 CV seeds x 3 generation
seeds x 3 cells, plus a permutation null).  This module reproduces run_pandora's decision EXACTLY
(asserted, not assumed -- see weitzman_T04.null_tests) in closed form.

THE CLOSED FORM.  run_pandora offers, at every step, the unopened boxes {cheap (if k<Nmax), strong}
and stops when best-so-far >= max remaining reservation value.

  * REGIME B, z_strong > z_cheap: at k=0 best_cal = -inf < z_strong, so the STRONG box is opened
    first and the arm degenerates to always-32B with N = 0.  (This is the degenerate corner
    cascade_selector_rerun.py documents for the raw-rank selector.)
  * REGIME A, z_cheap >= z_strong: cheap boxes are always the highest-reservation unopened box, so
    the controller draws until the first slot whose CALIBRATED score >= z_cheap.
        j* = min{ j : cal[j] >= z_cheap }
        if j* exists   -> N = j*+1, no escalation, answer = argmax RAW over slots 0..j*
        else           -> N = Nmax, and escalate iff max(cal) < z_strong
    (the pick uses RAW scores with a strict > update, i.e. FIRST-INDEX argmax, exactly as in
    run_pandora; isotonic calibration is monotone so the two orderings agree up to ties.)

zeta_cheap solves  lambda*c = mean_v (v-z)^+  exactly (piecewise-linear closed form) instead of by
bisection; asserted against pandora_controller.zeta_cheap.

Nothing here touches a GPU or writes a file.  Import is side-effect free.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from math import comb

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
_CM = os.path.join(ROOT, "src/cascade_methods")
if _CM not in sys.path:
    sys.path.insert(0, _CM)

from src.training_methods import genframe_data as G            # noqa: E402
from src.cascade_methods.decoding_sweep_analyse import (        # noqa: E402
    load_judge, load_vscores, load_pool, DS)

MISSING = G.MISSING_SCORE

# ---------------------------------------------------------------- cost model (deployed constants)
# (latency_ms, energy_J, FLOP-eq in 7B-forward-equivalents) -- paper_baselines.py / pandora_controller.py
GEN7 = (347.0, 45.8, 1.0)
VER7 = (175.0, 25.3, 1.0)
GEN32N = (665.0, 127.0, 4.57)
C_CHEAP_F = GEN7[2] + VER7[2]      # 2.0 per cheap draw (generate + verify)
C_STRONG_F = GEN32N[2]             # 4.57 per escalation


def cost_of(meanN, esc):
    """Deployed cost model, verbatim pandora_controller.cost_of."""
    return dict(flops=meanN * C_CHEAP_F + esc * C_STRONG_F,
                energy=meanN * (GEN7[1] + VER7[1]) + esc * GEN32N[1],
                lat_seq=meanN * (GEN7[0] + VER7[0]) + esc * GEN32N[0],
                lat_bat=(GEN7[0] + VER7[0] if meanN > 0 else 0.0) + esc * GEN32N[0])


# ---- measured cap320 FLOP model (secondary currency), read verbatim from the resolution sweep ----
def measured_cost_model():
    p = os.path.join(ROOT, "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json")
    if not os.path.exists(p):
        return None
    c = json.load(open(p))["cost"]["open_half_per_candidate"]["cap320"]
    dec = c["parts"]["lm_decode_dense"] + c["parts"]["lm_decode_attn"] + c["parts"]["lm_head"]
    r32 = json.load(open(p))["cost"]["R32_by_resolution"]["project_value"]
    return dict(
        source="artifacts/resolution_sweep_2026-08-13.json cost.open_half_per_candidate.cap320",
        gen_fixed_per_candidate=c["flops_per_candidate"] - dec,
        gen_decode_per_token=dec / c["measured_mean_gen_tokens"],
        verifier_per_candidate=c["flops_verifier_per_candidate_at_1003520"],
        gen_per_candidate_at_measured_tokens=c["flops_per_candidate"],
        strong_per_item=r32 * c["flops_per_candidate"],
        R32_used=r32,
        strong_assumption="the 32B-direct open leg is charged R32 x the cap320 7B generator forward. "
                          "R32 = 3.816 is the project value (flop_ratio_derivation_2026-08-03); the "
                          "32B open leg's OWN token geometry was NOT measured -- flagged, not hidden.")


# ---------------------------------------------------------------- zeta (reservation values)
def zeta_cheap_exact(v_sorted, suffix_sum, n, lam, c=C_CHEAP_F):
    """Exact solution of  lam*c = (1/n) sum_j (v_j - z)^+  .

    v_sorted ascending, suffix_sum[k] = sum(v_sorted[k:]), suffix_sum has length n+1.
    g(z) is continuous, piecewise linear, strictly decreasing on [min v, max v].
    Matches pandora_controller.zeta_cheap (which bisects 80 times) to <1e-9.
    """
    t = lam * c
    if t <= 0.0:
        return float(v_sorted[-1])
    mean_v = suffix_sum[0] / n
    if t >= mean_v - v_sorted[0]:                 # beyond the linear tail -> closed form
        return float(mean_v - t)
    # g at the breakpoints:  g(v_sorted[k]) = (suffix_sum[k] - (n-k)*v_sorted[k]) / n , non-increasing.
    # On the interval [v[k-1], v[k]] the exact form is g(z) = (suffix_sum[k] - (n-k)z)/n, so the
    # solution lies in the bracket indexed by the SMALLEST k with g(v[k]) <= t.
    k = np.arange(n)
    gk = (suffix_sum[:n] - (n - k) * v_sorted) / n
    kk = int(np.searchsorted(-gk, -t, side="right"))       # count of breakpoints with g >= t
    kk = max(1, min(kk, n - 1))
    m = n - kk
    if m <= 0:
        return float(v_sorted[-1])
    return float((suffix_sum[kk] - n * t) / m)


def zeta_cheap_many(v, lams, c=C_CHEAP_F):
    """Vectorised over lambdas. v = the calibrated TRAIN cheap-score pool."""
    v = np.sort(np.asarray(v, float))
    n = len(v)
    ss = np.concatenate([np.cumsum(v[::-1])[::-1], [0.0]])
    return np.array([zeta_cheap_exact(v, ss, n, float(l), c) for l in lams], float)


def zeta_strong(q, lam, c=C_STRONG_F):
    return q - lam * c


# ---------------------------------------------------------------- the policy, vectorised
class PoolView:
    """Per-item arrays needed to evaluate ANY (z_cheap, z_strong) in O(n) numpy.

    raw   (n, 8) verifier scores in POOL ORDER (recorded generation order)
    lab   dict currency -> (n, 8) 0/1 slot labels
    strong dict currency -> (n,) 0/1 label of the 32B-direct answer
    """

    def __init__(self, raw, labs, strongs, ds_index, item_keys, gen_tokens=None):
        self.raw = np.asarray(raw, float)
        self.n, self.Nmax = self.raw.shape
        self.labs = {k: np.asarray(v, float) for k, v in labs.items()}
        self.strongs = {k: np.asarray(v, float) for k, v in strongs.items()}
        self.ds_index = np.asarray(ds_index, int)
        self.item_keys = item_keys
        self.gen_tokens = None if gen_tokens is None else np.asarray(gen_tokens, float)
        # prefix first-index argmax of RAW  ->  (n, 8) index array
        pa = np.empty((self.n, self.Nmax), int)
        cur = np.zeros(self.n, int)
        best = self.raw[:, 0].copy()
        pa[:, 0] = 0
        for j in range(1, self.Nmax):
            better = self.raw[:, j] > best
            cur = np.where(better, j, cur)
            best = np.where(better, self.raw[:, j], best)
            pa[:, j] = cur
        self.prefix_argmax = pa
        rows = np.arange(self.n)[:, None]
        self.prefix_lab = {k: v[rows, pa] for k, v in self.labs.items()}   # (n,8) label of the pick@depth

    def calibrated(self, iso):
        return iso.predict(self.raw.ravel()).reshape(self.raw.shape)

    def run(self, cal, z_cheap, z_strong):
        """Vectorised Weitzman.  Returns dict(N, esc, ok{currency}).  cal = calibrated (n,8)."""
        n, M = self.raw.shape
        if z_strong > z_cheap:                       # REGIME B: strong box opened first, always
            N = np.zeros(n)
            esc = np.ones(n)
            ok = {k: self.strongs[k].copy() for k in self.strongs}
            return dict(N=N, esc=esc, ok=ok, regime="B")
        hit = cal >= z_cheap
        any_hit = hit.any(1)
        j = np.argmax(hit, axis=1)                   # first True; 0 when none (masked below)
        depth = np.where(any_hit, j, M - 1)          # index of the last drawn slot
        N = depth + 1.0
        prefmax = np.maximum.accumulate(cal, axis=1)
        esc = (~any_hit) & (prefmax[:, M - 1] < z_strong)
        rows = np.arange(n)
        ok = {}
        for k in self.strongs:
            pick_ok = self.prefix_lab[k][rows, depth]
            ok[k] = np.where(esc, self.strongs[k], pick_ok)
        return dict(N=N.astype(float), esc=esc.astype(float), ok=ok, regime="A")

    def fixedN_gate(self, cal, Nfix, tau):
        """Fixed best-of-N + escalate-if-max-verifier-confidence-below-tau (the baseline the
        controller claims to replace).  tau is applied to the CALIBRATED max over the N slots."""
        n, M = self.raw.shape
        Nf = int(min(Nfix, M))
        d = Nf - 1
        prefmax = np.maximum.accumulate(cal, axis=1)[:, d]
        esc = prefmax < tau
        rows = np.arange(n)
        ok = {}
        for k in self.strongs:
            pick_ok = self.prefix_lab[k][rows, np.full(n, d)]
            ok[k] = np.where(esc, self.strongs[k], pick_ok)
        return dict(N=np.full(n, float(Nf)), esc=esc.astype(float), ok=ok)


# ---------------------------------------------------------------- pool construction
def _strong_labels():
    """32B-direct per-item labels, BOTH currencies, from the deployed dumps."""
    sd = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu")
    out = {}
    for ds in DS:
        gen = {}
        for l in open(os.path.join(sd, f"ckpt_{ds}_lingshu32b.jsonl")):
            if l.strip():
                r = json.loads(l)
                gen[r["idx"]] = int(r["modal_ok"])
        jud = {}
        for l in open(os.path.join(sd, f"ckpt_{ds}_lingshu32b.judge.jsonl")):
            if l.strip():
                r = json.loads(l)
                jud[r["idx"]] = int(r["judge_ok"])
        out[ds] = dict(em=gen, judge=jud)
    return out


def build_view(tag, lab, vsc, ref, strong):
    """PoolView for one generated pool tag (e.g. 'T04_s0'), in the FROZEN canonical item order."""
    pool = load_pool(tag, strict=False)
    if pool is None:
        return None
    n = len(ref)
    raw = np.full((n, 8), MISSING, float)
    lj = np.zeros((n, 8)); le = np.zeros((n, 8)); tok = np.zeros((n, 8))
    sj = np.zeros(n); se = np.zeros(n)
    dsi = np.zeros(n, int)
    miss_j = miss_v = 0
    for i, it in enumerate(ref):
        ds, idx = it["ds"], it["idx"]
        r = pool[(ds, idx)]
        preds = r["preds"]
        assert len(preds) == 8, f"{tag} {ds}/{idx}: pool of {len(preds)}"
        for k, a in enumerate(preds):
            y = lab.get((ds, idx, G.norm(a)))
            if y is None:
                miss_j += 1; y = 0
            lj[i, k] = int(y)
            if (ds, idx, a) not in vsc:
                miss_v += 1
            raw[i, k] = vsc.get((ds, idx, a), MISSING)
        le[i] = np.asarray(r["oks_em"], float)
        tok[i] = np.asarray(r.get("gen_tokens_all", [0] * 8), float)
        sj[i] = strong[ds]["judge"][idx]
        se[i] = strong[ds]["em"][idx]
        dsi[i] = DS.index(ds)
    if miss_j or miss_v:
        raise ValueError(f"{tag}: {miss_j} unjudged / {miss_v} unscored slots -- refusing")
    return PoolView(raw, {"judge": lj, "em": le}, {"judge": sj, "em": se}, dsi,
                    [(it["ds"], it["idx"]) for it in ref], tok)


def build_view_from_frozen_dumps(verifier_dir="ckpts/train/lora_verifier_disjoint"):
    """PoolView built from the DEPLOYED transfer dumps (the stored T=0.7 pool the shipped macro used).

    Judge currency only for the cheap slots (the dumps carry judge labels in `sl`); the EM slot
    labels do not exist in those files, so 'em' is filled with NaN and must not be reported.
    """
    d = os.path.join(ROOT, verifier_dir)
    items = []
    for short, ds in zip(G.DUMP_ORDER, DS):
        items.extend(json.load(open(os.path.join(d, f"transfer_dump_{short}_open_lingshu7b.json"))))
    strong = _strong_labels()
    n = len(items)
    raw = np.full((n, 8), MISSING, float)
    lj = np.zeros((n, 8)); le = np.full((n, 8), np.nan)
    sj = np.zeros(n); se = np.zeros(n); dsi = np.zeros(n, int)
    greedy = np.zeros(n)
    for i, it in enumerate(items):
        ds, idx = it["ds"], it["idx"]
        sc = list(it["scores"]); sl = [0 if x in (None, -1) else int(x) for x in it["sl"]]
        raw[i, :len(sc)] = sc
        lj[i, :len(sl)] = sl
        sj[i] = strong[ds]["judge"][idx]
        se[i] = strong[ds]["em"][idx]
        dsi[i] = DS.index(ds)
        greedy[i] = int(it["greedy_ok"])
    v = PoolView(raw, {"judge": lj, "em": le}, {"judge": sj, "em": se}, dsi,
                 [(it["ds"], it["idx"]) for it in items])
    v.greedy = greedy
    return v


# ---------------------------------------------------------------- folds
_EH = None


def _imghash(ds, idx):
    global _EH
    if _EH is None:
        _EH = json.load(open(G.EVAL_IMGHASH))
    return _EH[ds][str(idx)]


def image_folds_for_keys(seed, item_keys, k=5):
    """Image-disjoint CV fold id for a list of (ds, idx) item keys, re-partitioned per seed.

    G.eval_folds is md5(image_hash) % k with no seed; the >=10-seed protocol needs a FAMILY of
    partitions, so the seed is mixed into the hash.  Image disjointness is preserved exactly --
    every item sharing an image gets the same id for every seed -- which is what the 2,345 items /
    528 images ratio requires (fit_hidden_head.py's protocol).
    """
    return np.array([int(hashlib.md5(f"{seed}|{_imghash(ds, idx)}".encode()).hexdigest(), 16) % k
                     for ds, idx in item_keys], int)


def image_folds(seed, k=5):
    return image_folds_for_keys(seed, [(it["ds"], it["idx"]) for it in G.load_items()], k)


def modulo_folds(n, k=5):
    """The DEPLOYED folding (paper_baselines.pandora_persample: i % K == f)."""
    return np.array([i % k for i in range(n)], int)


# ---------------------------------------------------------------- bootstrap
def boot(a, b, mask=None, nboot=10000, seed=20260815):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if mask is not None:
        a, b = a[mask], b[mask]
    rng = np.random.default_rng(seed)
    n = len(a)
    d = float(a.mean() - b.mean())
    idx = rng.integers(0, n, size=(nboot, n))
    ds_ = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(ds_, [2.5, 97.5])
    return {"delta": d, "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0),
            "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"), "n": n}


def boot_macro(vecs_a, vecs_b, nboot=10000, seed=20260815, fixed_delta=0.0):
    """Macro (equal-weight-per-cell) paired bootstrap: resample WITHIN each cell independently,
    average the per-cell deltas with equal weight, then add `fixed_delta` (the frozen MCQ half's
    contribution, which carries no resampling noise because those vectors are identical in both
    arms)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(nboot)
    for a, b in zip(vecs_a, vecs_b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        n = len(a)
        idx = rng.integers(0, n, size=(nboot, n))
        acc += (a[idx].mean(1) - b[idx].mean(1))
    acc = acc / len(vecs_a)
    point = float(np.mean([np.mean(a) - np.mean(b) for a, b in zip(vecs_a, vecs_b)]))
    return point, acc
