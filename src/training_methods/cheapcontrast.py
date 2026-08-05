#!/usr/bin/env python3
"""cheapcontrast.py -- POOL-RELATIVE ("cheap contrast") features for the generator-frame
best-of-8 selection head, plus the frame/layer geometry probes.

WHAT THIS IS FOR
----------------
The only two things that have ever beaten the incumbent LoRA verifier on this endpoint are
(a) reading the GENERATOR's own frame and (b) comparing candidates head-to-head with real
A-vs-B forward passes (28/question, shelved for cost).  This module asks whether the
comparative half can be had for free: every quantity here is computed from the ALREADY
CACHED per-candidate generator-frame vectors, so the architecture stays pointwise and the
inference cost stays at the 8 generations + 8 (shared) feature forward passes.

Blocks (each ablatable):
  H   raw hidden state h_i                                  3584 dims  -- the deployed head
  C   geometry contrast: h_i vs the pool centroid, nearest-neighbour similarity,
      within-pool duplication counts, norm/centrality ranks                18 dims
  M   multiplicity (self-consistency count of the distinct answer)          5 dims
  Wc  score-weighted geometry: cosine to a score-weighted centroid           3 dims
  Ws  stage-1 score stacking: pool-relative transforms of the stage-1 score  4 dims

NOTHING HERE TOUCHES EVAL DURING SELECTION.  Configs are chosen by 5-fold image-grouped CV
inside the disjoint TRAIN pool only (protocol rule 3).

The head itself is imported verbatim from fit_hidden_head (Head / fit_head / predict) so the
H-only arm reproduces the published 0.795640 bit-exact at seed 0.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

import genframe_data as G
from fit_hidden_head import fit_head, predict  # noqa: F401  (verbatim head code)

ROOT = G.ROOT
SC8_DIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")

#: the pre-registered head configuration, CV-selected on the TRAIN split in the previous
#: round (verifarch_hidden_generatorprompt_2026-08-04.json -> arms.generator.cv_selected).
#: Only the FEATURES vary in this round; the architecture is held fixed on purpose.
BASE_CFG = dict(layer=21, pooling="span", objective="bt", hidden=256, wd=1e-2, epochs=30)


# ======================================================================================
# multiplicity (self-consistency count) for BOTH splits
# ======================================================================================
_SC8_FILE = {
    "slake_open": "ckpt_slake_open_lingshu7b_sc8.jsonl",
    "vqa_rad_open": "ckpt_vqa_rad_open_lingshu7b_sc8.jsonl",
    "pathvqa_open": "ckpt_pathvqa_open_lingshu7b_sc8.jsonl",
    "slake_open_train": "ckpt_slake_open_train_lingshu7b_sc8.jsonl",
    "vqa_rad_open_train": "ckpt_vqa_rad_open_train_lingshu7b_sc8.jsonl",
    "pathvqa_open_train": "ckpt_pathvqa_open_train_lingshu7b_sc8.jsonl",
    "kvasir_open": "ckpt_kvasir_open_lingshu7b_sc8.jsonl",
    "radimagenet_open": "ckpt_radimagenet_open_lingshu7b_sc8.jsonl",
}


def pool_multiplicity(ds_set):
    """{(ds, idx, normalized answer) -> how many of the 8 samples produced it}.

    Read from the SAME sc8 checkpoints the feature rows were built from, so this is the
    identical quantity Cand.mult carries on the eval split (asserted in the driver).
    """
    out = {}
    for ds in ds_set:
        p = os.path.join(SC8_DIR, _SC8_FILE[ds])
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            cnt = defaultdict(int)
            for a in r["preds"]:
                cnt[G.norm(a)] += 1
            for na, c in cnt.items():
                out[(ds, r["idx"], na)] = c
    return out


# ======================================================================================
# feature blocks
# ======================================================================================
GEOM_NAMES = [
    "K", "inv_K", "is_singleton",
    "cos_centroid", "cos_centroid_loo", "cos_centroid_z", "cos_centroid_rank",
    "resid_norm_rel", "log_norm", "norm_z", "norm_rank",
    "max_cos_other", "mean_cos_other", "min_cos_other", "mean_cos_rank",
    "dup95", "dup99", "pool_mean_cos",
]
MULT_NAMES = ["mult", "mult_over8", "mult_frac", "is_modal", "mult_rank"]
WC_NAMES = ["cos_wcentroid", "cos_wcentroid_loo", "wmean_cos_other"]
WS_NAMES = ["s_z", "s_rank", "s_minus_maxother", "p_softmax"]


def _rank01(v):
    """rank in [0,1] within a pool; average ranks for ties; 0.5 if the pool has one member."""
    v = np.asarray(v, float)
    k = len(v)
    if k == 1:
        return np.array([0.5])
    return G.rank_avg(v)


def _z(v):
    v = np.asarray(v, float)
    if len(v) == 1:
        return np.zeros(1)
    s = v.std()
    return (v - v.mean()) / (s + 1e-8)


def geom_features(H, qrows):
    """Pool-relative geometry of each candidate vs its own question's pool.

    H      (n_rows, d) float32 -- the RAW hidden states (not standardized)
    qrows  list of int arrays, one per question, holding that question's row indices
    """
    n = H.shape[0]
    F = np.zeros((n, len(GEOM_NAMES)), np.float32)
    for ii in qrows:
        Hq = H[ii]
        k = len(ii)
        nrm = np.linalg.norm(Hq, axis=1) + 1e-8
        Hn = Hq / nrm[:, None]
        Cm = Hn @ Hn.T
        mu = Hq.mean(0)
        cos_mu = (Hq @ mu) / (nrm * (np.linalg.norm(mu) + 1e-8))
        resid = np.linalg.norm(Hq - mu[None], axis=1) / nrm
        if k > 1:
            tot = Hq.sum(0)
            loo = (tot[None] - Hq) / (k - 1)
            cos_loo = (Hq * loo).sum(1) / (nrm * (np.linalg.norm(loo, axis=1) + 1e-8))
            off = Cm.copy()
            np.fill_diagonal(off, np.nan)
            mx = np.nanmax(off, 1); mn = np.nanmin(off, 1); mo = np.nanmean(off, 1)
            d95 = (np.nan_to_num(off, nan=-9) > 0.95).sum(1) / (k - 1)
            d99 = (np.nan_to_num(off, nan=-9) > 0.99).sum(1) / (k - 1)
            pool_mc = float(np.nanmean(off))
        else:
            cos_loo = np.zeros(1); mx = mn = mo = np.zeros(1)
            d95 = d99 = np.zeros(1); pool_mc = 0.0
        F[ii] = np.stack([
            np.full(k, k, float), np.full(k, 1.0 / k), np.full(k, float(k == 1)),
            cos_mu, cos_loo, _z(cos_mu), _rank01(cos_mu),
            resid, np.log(nrm), _z(nrm), _rank01(nrm),
            mx, mo, mn, _rank01(mo),
            d95, d99, np.full(k, pool_mc),
        ], 1).astype(np.float32)
    return F


def mult_features(mult, qrows):
    n = len(mult)
    F = np.zeros((n, len(MULT_NAMES)), np.float32)
    for ii in qrows:
        m = np.asarray(mult)[ii].astype(float)
        F[ii] = np.stack([
            m, m / 8.0, m / max(m.sum(), 1.0), (m == m.max()).astype(float), _rank01(m),
        ], 1).astype(np.float32)
    return F


def weighted_features(H, qrows, s):
    """Score-weighted contrast.  s = a per-row score (stage-1 head, cross-fitted on train).

    Returns (Wc, Ws): Wc is representation contrast weighted by the score, Ws is the pure
    pool-relative transform of the score itself (i.e. stacking), kept separable so the two
    can be ablated apart.
    """
    n = H.shape[0]
    Fc = np.zeros((n, len(WC_NAMES)), np.float32)
    Fs = np.zeros((n, len(WS_NAMES)), np.float32)
    s = np.asarray(s, float)
    for ii in qrows:
        Hq = H[ii]; k = len(ii)
        sv = s[ii]
        e = np.exp(sv - sv.max()); p = e / e.sum()
        nrm = np.linalg.norm(Hq, axis=1) + 1e-8
        Hn = Hq / nrm[:, None]
        muw = (p[:, None] * Hq).sum(0)
        cos_w = (Hq @ muw) / (nrm * (np.linalg.norm(muw) + 1e-8))
        if k > 1:
            pl = p[None, :].repeat(k, 0)
            np.fill_diagonal(pl, 0.0)
            pl = pl / np.clip(pl.sum(1, keepdims=True), 1e-8, None)
            muw_loo = pl @ Hq
            cos_wl = (Hq * muw_loo).sum(1) / (nrm * (np.linalg.norm(muw_loo, axis=1) + 1e-8))
            Cm = Hn @ Hn.T
            np.fill_diagonal(Cm, 0.0)
            wmo = (pl * Cm).sum(1)
            so = np.tile(sv, (k, 1)).astype(float)
            np.fill_diagonal(so, -np.inf)
            smax_other = so.max(1)
        else:
            cos_wl = np.zeros(1); wmo = np.zeros(1); smax_other = sv.copy()
        Fc[ii] = np.stack([cos_w, cos_wl, wmo], 1).astype(np.float32)
        Fs[ii] = np.stack([_z(sv), _rank01(sv), sv - smax_other, p], 1).astype(np.float32)
    return Fc, Fs


# ======================================================================================
# assembly
# ======================================================================================
BLOCKS = ("H", "C", "M", "Wc", "Ws")


def assemble(blocks, H, C=None, M=None, Wc=None, Ws=None):
    parts, names = [], []
    for b in blocks:
        if b == "H":
            parts.append(H); names += [f"h{j}" for j in range(H.shape[1])]
        elif b == "C":
            parts.append(C); names += GEOM_NAMES
        elif b == "M":
            parts.append(M); names += MULT_NAMES
        elif b == "Wc":
            parts.append(Wc); names += WC_NAMES
        elif b == "Ws":
            parts.append(Ws); names += WS_NAMES
        else:
            raise ValueError(b)
    return np.concatenate(parts, 1).astype(np.float32), names


def standardize(Xtr, Xev):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (Xtr - mu) / sd, (Xev - mu) / sd


def fold_of_group(groups, k=5):
    """md5(image hash) % k -- the project's image-disjoint fold rule (verbatim)."""
    fo = {h: (int(hashlib.md5(str(h).encode()).hexdigest(), 16) % k) for h in set(groups)}
    return np.array([fo[h] for h in groups])


# ======================================================================================
# device-portable head -- a FAITHFUL port of fit_hidden_head.fit_head
# ======================================================================================
# The CPU original takes ~7 min per fit on this box, which makes a 10-seed x 20-arm design
# impossible.  This port keeps (a) the identical module and init (constructed on CPU under
# torch.manual_seed(seed), then moved), (b) the identical CPU-drawn batch permutations, and
# (c) the identical losses/optimizer.  Only the float arithmetic runs on the GPU, so results
# differ from the CPU original at ~1e-3 in sel_eff.  THAT GAP IS MEASURED AND REPORTED, and
# every comparison in this round is device-matched (baseline and arms both on GPU).
import torch
import torch.nn as nn
from fit_hidden_head import Head, _pad_groups

# !! THIS BOX'S PyTorch (NGC 25.09) DEFAULTS TO TF32 MATMUL (10-bit mantissa).  With TF32 on,
# the GPU port is NOT a float32 port: at seed 0 it gave 0.786785 where CPU gives 0.795640
# (generator L21/span/bt) and 0.774523 where CPU gives 0.750681 (grader L21/last/bce) -- i.e.
# a precision artifact as large as every effect in this round.  Forced off.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
try:
    torch.set_float32_matmul_precision("highest")
except Exception:
    pass


def fit_head_dev(Xtr, ytr, gtr, objective="bce", hidden=0, wd=1e-2, lr=1e-3, epochs=30,
                 bs=256, seed=0, drop=0.0, device="cuda"):
    torch.manual_seed(seed)
    m = Head(Xtr.shape[1], hidden, drop)          # init drawn on CPU == original
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    X = torch.tensor(Xtr, device=device)
    y = torch.tensor(ytr, device=device)
    if objective == "bce":
        pos = float(np.mean(ytr))
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=device))
        n = len(ytr)
        for ep in range(epochs):
            perm = torch.randperm(n).to(device)   # drawn on CPU == original
            for i in range(0, n, bs):
                j = perm[i:i + bs]
                opt.zero_grad(); lossf(m(X[j]), y[j]).backward(); opt.step()
        m.eval(); return m
    packed = _pad_groups(gtr, ytr, objective)
    if packed is None:
        m.eval(); return m
    idx, msk = packed
    idx = idx.to(device); msk = msk.to(device)
    NG, gb = idx.shape[0], 64
    for ep in range(epochs):
        perm = torch.randperm(NG).to(device)
        for i in range(0, NG, gb):
            j = perm[i:i + gb]
            gi, gm = idx[j], msk[j]
            s = m(X[gi.reshape(-1)]).reshape(gi.shape)
            yy = y[gi.reshape(-1)].reshape(gi.shape) * gm
            s = s.masked_fill(gm == 0, -1e9)
            if objective == "listwise":
                logp = torch.log_softmax(s, 1)
                l = -((logp * yy).sum(1) / yy.sum(1).clamp(min=1)).mean()
            else:
                pm = yy.unsqueeze(2); nm = ((1 - yy) * gm).unsqueeze(1)
                d = s.unsqueeze(2) - s.unsqueeze(1)
                w = pm * nm
                l = ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
            opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def predict_dev(m, X, device="cuda", bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.tensor(X[i:i + bs], device=device)).cpu().numpy())
    return np.concatenate(out)


# ======================================================================================
# ridge linear probe -- closed form, deterministic, SEED-FREE
# ======================================================================================
def ridge_probe(Xtr, ytr, Xev, lam=1.0):
    """Least-squares probe with an intercept. Used for the frame x layer x pooling geometry
    grid: no SGD, no seed, so a frame difference cannot be a training-noise artifact."""
    d = Xtr.shape[1]
    X64 = np.asarray(Xtr, np.float64)          # float32 gram loses ~5 digits at n=31k
    A = X64.T @ X64
    A[np.diag_indices(d)] += lam * len(Xtr) / d
    b = X64.T @ (np.asarray(ytr, np.float64) - float(np.mean(ytr)))
    w = np.linalg.solve(A, b)
    return np.asarray(Xev, np.float64) @ w, w


def linear_cka(X, Y):
    """Linear CKA between two representations of the SAME rows (column-centred)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    C = X.T @ Y
    num = float((C * C).sum())
    xx = X.T @ X; yy = Y.T @ Y
    den = float(np.sqrt((xx * xx).sum()) * np.sqrt((yy * yy).sum()))
    return num / max(den, 1e-30)
