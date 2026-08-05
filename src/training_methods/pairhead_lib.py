#!/usr/bin/env python3
"""pairhead_lib.py -- a PAIRWISE CONTRAST HEAD over cached generator-frame hidden states.

The bet: the two things that have worked on this endpoint are (a) reading the GENERATOR's own
frame and (b) comparing candidates HEAD-TO-HEAD.  The measured pairwise win in this project
(pointwise 0.783 -> round-robin 0.859 sel_eff, artifacts/pairwise_verifier_gpu.json) cost 28
real forward passes per question and was shelved.  If the comparative signal is computable from
the CACHED per-candidate generator-frame vectors, the same comparison costs 28 tiny MLP
evaluations over features the deployed pointwise head already computes -- i.e. zero extra GPU.

    g(h_i, h_j) -> logit that candidate i beats candidate j

Nothing in this module touches eval.  Configuration is selected by image-grouped CV inside the
disjoint TRAIN pool only (protocol rule 3).

Conventions inherited from genframe_data.py (do not change):
  * row order 'concat' (shard0 then shard1) -- the published bar reproduces bit-exact only there
  * pick rule = np.argmax over the 8 slots, first-index tie-break
  * rank fusion = rank_avg (average ranks), never rank_argsort
"""
from __future__ import annotations

import hashlib
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G  # noqa: E402

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# independent disjointness proof (protocol rule 2 -- my own code, not a previous claim)
# ======================================================================================
def independent_disjointness(featdir: str = "feats_hidden", mode: str = "generator") -> dict:
    """Read the cache meta files DIRECTLY (not through the shared loader) and intersect the
    md5-of-decoded-RGB-pixels recorded per row at extraction time."""
    import glob
    import json
    d = os.path.join(G.ROOT, featdir)
    out = {}
    for split in ("train", "eval"):
        hs, items, fails, n = set(), set(), 0, 0
        for p in sorted(glob.glob(os.path.join(d, f"{mode}_{split}_s*of*.meta.json"))):
            m = json.load(open(p))
            for r in m["rows"]:
                n += 1
                if r.get("img_md5"):
                    hs.add(r["img_md5"])
                items.add((r["ds"], r["idx"]))
                if r.get("n_tok", -1) <= 0:
                    fails += 1
        out[split] = {"rows": n, "images": len(hs), "questions": len(items), "failed": fails,
                      "_h": hs}
    inter = out["train"]["_h"] & out["eval"]["_h"]
    for s in ("train", "eval"):
        out[s].pop("_h")
    assert len(inter) == 0, f"CONTAMINATION: {len(inter)} shared decoded-RGB pixel md5s"
    assert out["train"]["failed"] == 0 and out["eval"]["failed"] == 0, "extraction failures"
    return {"mode": mode, "method": "md5 of decoded RGB pixels, recorded per row at extraction; "
                                    "meta files read directly in this module, not via the shared loader",
            "train": out["train"], "eval": out["eval"], "image_pixel_md5_intersection": 0,
            "verdict": "DISJOINT"}


# ======================================================================================
# feature assembly
# ======================================================================================
def base_matrix(sp: G.Split, layers: Sequence[int], pooling: str) -> np.ndarray:
    """(n_rows, d) float32 base vector per candidate.  pooling in {last, span, both};
    `layers` may name several layers, which are concatenated."""
    parts = []
    for L in layers:
        if pooling in ("last", "both"):
            parts.append(sp.matrix("last", L))
        if pooling in ("span", "both"):
            parts.append(sp.matrix("span", L))
    return parts[0] if len(parts) == 1 else np.concatenate(parts, 1)


def row_folds(sp: G.Split, n_folds: int = 5) -> np.ndarray:
    return G.train_folds(sp, n_folds)


def train_pairs(sp: G.Split) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All within-question (correct, incorrect) row-index pairs of the TRAIN split.

    Returns (pos_rows, neg_rows, fold) with fold taken from the question's image hash, so a
    pair never straddles a fold boundary and a fold boundary is an IMAGE boundary."""
    P, N, F = [], [], []
    for q in sp.questions:
        pos = [c.row for c in q.cands if c.y == 1]
        neg = [c.row for c in q.cands if c.y == 0]
        if not pos or not neg:
            continue
        f = int(hashlib.md5(str(q.img_md5).encode()).hexdigest(), 16) % 5
        for a in pos:
            for b in neg:
                P.append(a); N.append(b); F.append(f)
    return np.array(P), np.array(N), np.array(F)


# ======================================================================================
# the head
# ======================================================================================
class PairHead(nn.Module):
    """g(h_i, h_j).  `inp` fixes the pair encoding, `antisym` how order invariance is obtained.

      inp='concat'  phi = [h_i, h_j]
         ='diff'    phi = [h_i - h_j]           <- the minimal encoding of "how do these differ"
         ='full'    phi = [h_i, h_j, h_i-h_j, h_i*h_j]
      antisym='arch'    g(i,j) = f(phi(i,j)) - f(phi(j,i))   antisymmetric BY CONSTRUCTION
             ='augment' g(i,j) = f(phi(i,j)), order-augmented training data only

    NOTE (mechanistic, must be reported): with inp='diff' and hidden=0 the head is LINEAR in
    h_i - h_j, so g(i,j) = w.h_i - w.h_j -- an exactly pointwise scorer wearing a pairwise
    costume.  Only a nonlinearity makes a difference-encoding genuinely comparative.  That
    config is kept in the grid as the degeneracy control.
    """

    def __init__(self, d: int, inp: str = "full", hidden: int = 256, drop: float = 0.0):
        super().__init__()
        self.inp = inp
        k = {"concat": 2, "diff": 1, "full": 4}[inp]
        din = k * d
        self.f = (nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(hidden, 1))
                  if hidden else nn.Linear(din, 1))

    def phi(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.inp == "concat":
            return torch.cat([a, b], 1)
        if self.inp == "diff":
            return a - b
        return torch.cat([a, b, a - b, a * b], 1)

    def raw(self, a, b):
        return self.f(self.phi(a, b)).squeeze(-1)

    def score(self, a, b, antisym: str):
        return self.raw(a, b) - self.raw(b, a) if antisym == "arch" else self.raw(a, b)


def fit_pair_head(X: torch.Tensor, pos: np.ndarray, neg: np.ndarray, cfg: dict,
                  seed: int = 0) -> PairHead:
    """X is the FULL standardized row matrix on device; pos/neg index into it."""
    torch.manual_seed(seed)
    m = PairHead(X.shape[1], cfg["inp"], cfg["hidden"], cfg.get("drop", 0.0)).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg["wd"])
    lossf = nn.BCEWithLogitsLoss()
    P = torch.tensor(pos, device=DEV, dtype=torch.long)
    N = torch.tensor(neg, device=DEV, dtype=torch.long)
    n, bs, anti = len(pos), cfg.get("bs", 256), cfg["antisym"]
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(n, generator=gen).to(DEV)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            a, b = X[P[j]], X[N[j]]
            if anti == "arch":
                s = m.score(a, b, "arch")
                l = lossf(s, torch.ones_like(s))
            else:  # order-augmented: (pos,neg)->1 and (neg,pos)->0, no architectural constraint
                s = torch.cat([m.raw(a, b), m.raw(b, a)])
                t = torch.cat([torch.ones(len(j), device=DEV), torch.zeros(len(j), device=DEV)])
                l = lossf(s, t)
            opt.zero_grad(); l.backward(); opt.step()
    m.eval()
    return m


# ======================================================================================
# pairwise scoring of a set of questions
# ======================================================================================
def build_query_pairs(groups: List[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """groups[q] = the row indices of question q's DISTINCT candidates.
    Returns (rowA, rowB, qid, (ia, ib)) over all ORDERED i!=j pairs."""
    A, B, Q, IA, IB = [], [], [], [], []
    for qi, rows in enumerate(groups):
        k = len(rows)
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                A.append(rows[i]); B.append(rows[j]); Q.append(qi); IA.append(i); IB.append(j)
    return (np.array(A, dtype=np.int64), np.array(B, dtype=np.int64), np.array(Q, dtype=np.int64),
            np.array(IA, dtype=np.int64), np.array(IB, dtype=np.int64))


@torch.no_grad()
def pair_logits(m: PairHead, X: torch.Tensor, A, B, antisym: str, bs: int = 8192) -> np.ndarray:
    out = np.empty(len(A), dtype=np.float64)
    At = torch.tensor(A, device=DEV); Bt = torch.tensor(B, device=DEV)
    for i in range(0, len(A), bs):
        a, b = X[At[i:i + bs]], X[Bt[i:i + bs]]
        out[i:i + bs] = m.score(a, b, antisym).double().cpu().numpy()
    return out


def logits_to_matrices(gl: np.ndarray, Q, IA, IB, sizes: List[int]) -> List[np.ndarray]:
    """Per question: (k,k) logit matrix, diagonal 0."""
    mats = [np.zeros((k, k)) for k in sizes]
    for t in range(len(gl)):
        mats[Q[t]][IA[t], IB[t]] = gl[t]
    return mats


# ======================================================================================
# aggregation of a pairwise matrix into per-candidate scores
# ======================================================================================
def _bt_mle(P: np.ndarray, iters: int = 100) -> np.ndarray:
    """Zermelo / MM iterations for Bradley-Terry strengths from soft pairwise win probabilities.
    N_ij = 1 comparison per ordered pair; W_i = sum_j P_ij."""
    k = P.shape[0]
    off = ~np.eye(k, dtype=bool)
    W = (P * off).sum(1)
    pi = np.ones(k)
    for _ in range(iters):
        den = np.zeros(k)
        for i in range(k):
            for j in range(k):
                if i != j:
                    den[i] += 2.0 / (pi[i] + pi[j])   # N_ij + N_ji = 2 comparisons per unordered pair
        new = np.where(den > 0, (W + 1e-9) / np.maximum(den, 1e-12), pi)
        new = new / new.sum() * k
        if np.max(np.abs(new - pi)) < 1e-10:
            pi = new
            break
        pi = new
    return np.log(np.maximum(pi, 1e-12))


def aggregate(Gm: np.ndarray, kind: str) -> np.ndarray:
    """Gm = (k,k) logits, Gm[i,j] = logit(i beats j).  Returns per-candidate scores."""
    k = Gm.shape[0]
    if k == 1:
        return np.zeros(1)
    off = ~np.eye(k, dtype=bool)
    P = 1.0 / (1.0 + np.exp(-Gm))
    np.fill_diagonal(P, 0.0)
    borda = (P * off).sum(1)
    if kind == "borda":
        return borda
    if kind == "logit_sum":
        return (Gm * off).sum(1)
    if kind == "copeland":
        W = ((P > 0.5) & off).astype(float) + 0.5 * ((P == 0.5) & off)
        return W.sum(1)
    if kind == "copeland_borda":
        W = ((P > 0.5) & off).astype(float) + 0.5 * ((P == 0.5) & off)
        return W.sum(1) + 1e-6 * borda / max(k - 1, 1)
    if kind == "bt_mle":
        return _bt_mle(P)
    if kind == "knockout":
        # single elimination over the candidates in POOL ORDER (first-occurrence slot order);
        # k-1 comparisons instead of k(k-1)/2.  Score = round in which a candidate was
        # eliminated, so argmax is the bracket winner.
        alive = list(range(k))
        sc = np.zeros(k)
        rnd = 1
        while len(alive) > 1:
            nxt = []
            for i in range(0, len(alive) - 1, 2):
                a, b = alive[i], alive[i + 1]
                if Gm[a, b] >= 0:
                    nxt.append(a); sc[b] = max(sc[b], rnd)
                else:
                    nxt.append(b); sc[a] = max(sc[a], rnd)
            if len(alive) % 2 == 1:            # bye
                nxt.append(alive[-1])
            alive = nxt
            rnd += 1
        sc[alive[0]] = rnd + 1
        return sc
    raise ValueError(kind)


def n_knockout_comparisons(k: int) -> int:
    return max(k - 1, 0)


AGGS = ["copeland_borda", "copeland", "borda", "logit_sum", "bt_mle", "knockout"]


# ======================================================================================
# end-to-end: fit on a row subset, score a set of questions
# ======================================================================================
def standardize(Xtr_np: np.ndarray, fit_rows: np.ndarray):
    mu = Xtr_np[fit_rows].mean(0)
    sd = Xtr_np[fit_rows].std(0) + 1e-6
    return mu, sd


def to_dev(Xnp: np.ndarray, mu, sd) -> torch.Tensor:
    return torch.tensor((Xnp - mu) / sd, dtype=torch.float32, device=DEV)


def score_questions(m: PairHead, Xdev: torch.Tensor, groups: List[List[int]], antisym: str,
                    aggs: Sequence[str] = ("copeland_borda",)) -> Dict[str, List[np.ndarray]]:
    A, B, Q, IA, IB = build_query_pairs(groups)
    sizes = [len(g) for g in groups]
    if len(A) == 0:
        return {a: [np.zeros(k) for k in sizes] for a in aggs}, [np.zeros((k, k)) for k in sizes]
    gl = pair_logits(m, Xdev, A, B, antisym)
    mats = logits_to_matrices(gl, Q, IA, IB, sizes)
    return {a: [aggregate(Gm, a) for Gm in mats] for a in aggs}, mats


def cv_sel_eff(sp: G.Split, Xnp: np.ndarray, cfg: dict, folds: np.ndarray,
               n_folds: int = 5, seed: int = 0,
               aggs: Sequence[str] = ("copeland_borda",)) -> Dict[str, float]:
    """Image-grouped CV inside the TRAIN pool.  Criterion = within-question selection
    efficiency on the held-out fold (questions with >=1 correct candidate), identical in spirit
    to fit_hidden_head.cv so the pointwise and pairwise numbers are chosen the same way."""
    pos, neg, pf = train_pairs(sp)
    qfold = np.array([int(hashlib.md5(str(q.img_md5).encode()).hexdigest(), 16) % n_folds
                      for q in sp.questions])
    hits = {a: 0 for a in aggs}
    tot = 0
    for f in range(n_folds):
        tr = pf != f
        if tr.sum() == 0:
            continue
        fit_rows = np.unique(np.concatenate([pos[tr], neg[tr]]))
        mu, sd = standardize(Xnp, fit_rows)
        Xd = to_dev(Xnp, mu, sd)
        m = fit_pair_head(Xd, pos[tr], neg[tr], cfg, seed=seed)
        qi = [i for i in range(len(sp.questions)) if qfold[i] == f
              and any(c.y == 1 for c in sp.questions[i].cands)]
        groups = [[c.row for c in sp.questions[i].cands] for i in qi]
        out, _ = score_questions(m, Xd, groups, cfg["antisym"], aggs)
        for a in aggs:
            for t, i in enumerate(qi):
                cand = sp.questions[i].cands
                b = int(np.argmax(out[a][t]))
                hits[a] += int(cand[b].y == 1)
        tot += len(qi)
        del Xd, m
        torch.cuda.empty_cache()
    return {a: hits[a] / max(tot, 1) for a in aggs} | {"n_q": tot}
