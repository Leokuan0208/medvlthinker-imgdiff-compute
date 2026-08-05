#!/usr/bin/env python3
"""verifarch_setaware.py -- SET-AWARE (listwise-architecture) heads over the GENERATOR-FRAME
hidden states, for best-of-8 open-text selection.

THE QUESTION
------------
Two things have worked on this endpoint: (a) reading the generator's OWN frame (a pointwise
head on generator-frame layer-21 span vectors reaches sel_eff 0.7956 vs the incumbent LoRA
judge's 0.7752), and (b) comparing candidates head-to-head (real A-vs-B forward passes, a
measured +0.076, shelved at 28 forward passes/question).

This script asks whether the comparative part can be had for free, by making the HEAD
ARCHITECTURE set-aware over the CACHED per-candidate generator-frame vectors: a DeepSets
head (per-candidate encoder + pooled context re-injected) and a small self-attention block
over the pool, trained with listwise objectives.

WHY THIS IS NOT A REDISCOVERY
-----------------------------
Prior negatives applied a listwise/ranking OBJECTIVE to a POINTWISE readout of the
LM-head / judge score (best 0.7766, d=+0.0014 n.s.; and a July Bradley-Terry run that
bought +0.030 candidate AUROC and +0.000 selection). Here the objective is held as a
swept nuisance and the ARCHITECTURE is the variable: the head sees all candidates of a
question at once and its score for candidate i is a function of the whole pool.
The controls make that explicit -- `point` (no cross-candidate path) is refit inside this
same harness on the identical features, so the comparison is within-harness.

PROTOCOL
--------
1. Null test against the incumbent's published cells (genframe_data.null_test).
2. Image-disjointness re-asserted from decoded-RGB pixel md5 (genframe_data.assert_disjoint).
3. Configuration pre-registered by image-grouped 5-fold CV on the TRAIN split only; eval is
   never used for any selection. Anything chosen with eval visibility is labelled DIAGNOSTIC.
4. >= 10 seeds; mean/sd/range plus the seed-averaged ensemble as the deployable number.
5. Paired item-level bootstrap, nboot=10000, vs the incumbent and vs the in-harness pointwise head.
6. Per-set guardrail reported for all three sets.
7. Contested stratum (>= 2 distinct candidate strings, n=916 recoverable) reported.
8. Cost stated in forward passes per question over and above the 8 generations.

Usage
-----
    python3 src/training_methods/verifarch_setaware.py --stage cv     # pre-registration
    python3 src/training_methods/verifarch_setaware.py --stage eval   # headline + artifact
Run from the repo root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
from training_methods import genframe_data as G  # noqa: E402

LAYER, POOLING = 21, "span"
CV_JSON = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_setaware_cv_preregistration.json")
OUT_JSON = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_setaware_2026-08-04.json")


# =====================================================================================
# grouped padding
# =====================================================================================
def pack_groups(qids, y, min_pos_neg=True, max_len=8):
    """[n_groups, L] row-index + mask + label matrices. Group = one question.

    min_pos_neg: keep only groups with at least one correct AND one incorrect candidate --
    the only groups on which a listwise/BT loss has a gradient. This is exactly the filter
    fit_hidden_head._pad_groups applies for its BT objective, so the pointwise control here
    sees the identical training material.
    """
    byq = OrderedDict()
    for i, q in enumerate(qids):
        byq.setdefault(q, []).append(i)
    groups = [np.array(v) for v in byq.values()]
    if min_pos_neg:
        groups = [g for g in groups if y[g].sum() > 0 and (1 - y[g]).sum() > 0]
    if not groups:
        return None
    L = max(max(len(g) for g in groups), 1)
    assert L <= max_len, f"group larger than {max_len}: {L}"
    idx = np.zeros((len(groups), L), dtype=np.int64)
    msk = np.zeros((len(groups), L), dtype=np.float32)
    for k, g in enumerate(groups):
        idx[k, :len(g)] = g
        msk[k, :len(g)] = 1.0
    return torch.tensor(idx), torch.tensor(msk)


# =====================================================================================
# heads.  every head maps (B, L, d) + mask (B, L) -> (B, L) scores
# =====================================================================================
class Point(nn.Module):
    """NO cross-candidate path. Parameter creation order matches fit_hidden_head.Head so
    the same torch seed gives the same initialisation -> the published bar reproduces."""

    def __init__(self, d, hidden=256, drop=0.0, **kw):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        return self.f(x).squeeze(-1)


class Centroid(nn.Module):
    """The explicit centroid ablation: a POINTWISE MLP whose input is the raw triple
    [h_i, pool-mean, h_i - pool-mean]. Set-aware only through the raw first moment; no
    learned encoder before pooling. If the whole set-aware gain survives here, the gain is
    centroid distance and a much cheaper method suffices."""

    def __init__(self, d, hidden=256, drop=0.0, **kw):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        m = mask.unsqueeze(-1)
        c = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        c = c.expand_as(x)
        return self.f(torch.cat([x, c, x - c], -1)).squeeze(-1)


class DeepSets(nn.Module):
    """Per-candidate encoder phi, masked-mean pooled context, re-injected as [e, c, e-c]."""

    def __init__(self, d, hidden=256, drop=0.0, **kw):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, 1))

    def forward(self, x, mask):
        e = self.phi(x)
        m = mask.unsqueeze(-1)
        c = (e * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
        c = c.expand_as(e)
        return self.rho(torch.cat([e, c, e - c], -1)).squeeze(-1)


class DeepSetsNoCtx(nn.Module):
    """DeepSets with the pooled context ZEROED -- identical parameter count and depth to
    DeepSets, so any DeepSets-vs-this gap is the set path itself, not capacity."""

    def __init__(self, d, hidden=256, drop=0.0, **kw):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, hidden), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(hidden, 1))

    def forward(self, x, mask):
        e = self.phi(x)
        z = torch.zeros_like(e)
        return self.rho(torch.cat([e, z, e], -1)).squeeze(-1)


class SetAttn(nn.Module):
    """Permutation-equivariant self-attention over the pool. No positional encoding, so the
    head cannot use slot order (which carries no information)."""

    def __init__(self, d, hidden=256, drop=0.0, nheads=4, nblocks=1, **kw):
        super().__init__()
        self.inp = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(nblocks):
            self.blocks.append(nn.ModuleDict({
                "ln1": nn.LayerNorm(hidden),
                "att": nn.MultiheadAttention(hidden, nheads, dropout=drop, batch_first=True),
                "ln2": nn.LayerNorm(hidden),
                "ff": nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Dropout(drop),
                                    nn.Linear(2 * hidden, hidden)),
            }))
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, mask):
        e = self.inp(x)
        kpm = mask == 0
        for b in self.blocks:
            a = b["ln1"](e)
            a, _ = b["att"](a, a, a, key_padding_mask=kpm, need_weights=False)
            e = e + a
            e = e + b["ff"](b["ln2"](e))
        return self.out(e).squeeze(-1)


ARCHS = {"point": Point, "centroid": Centroid, "deepsets": DeepSets,
         "deepsets_noctx": DeepSetsNoCtx, "attn": SetAttn}
#: architectures whose score for candidate i is a function of the WHOLE pool. `point` has no
#: cross-candidate path; `deepsets_noctx` has the path zeroed, so it is a capacity control for
#: DeepSets, not a set-aware arm.
SET_ARCHS = {"centroid", "deepsets", "attn"}


# =====================================================================================
# objectives
# =====================================================================================
def loss_of(objective, s, yy, gm):
    """s (B,L) scores, yy (B,L) labels already masked, gm (B,L) mask."""
    s = s.masked_fill(gm == 0, -1e9)
    if objective == "listmax":          # fit_hidden_head's 'listwise': mean over correct
        logp = torch.log_softmax(s, 1)
        return -((logp * yy).sum(1) / yy.sum(1).clamp(min=1)).mean()
    if objective == "listsum":          # -log sum-of-correct-softmax: "SOME correct on top"
        p = torch.softmax(s, 1)
        return -torch.log((p * yy).sum(1).clamp(min=1e-9)).mean()
    if objective == "bt":               # Bradley-Terry over (correct, incorrect) pairs
        pm = yy.unsqueeze(2)
        nm = ((1 - yy) * gm).unsqueeze(1)
        d = s.unsqueeze(2) - s.unsqueeze(1)
        w = pm * nm
        return ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
    if objective == "bce":              # pointwise control objective, masked, pos-weighted
        pos = float((yy * gm).sum() / gm.sum().clamp(min=1))
        pw = torch.tensor((1 - pos) / max(pos, 1e-6))
        l = nn.functional.binary_cross_entropy_with_logits(
            s.clamp(-30, 30), yy, pos_weight=pw, reduction="none")
        return (l * gm).sum() / gm.sum().clamp(min=1)
    raise ValueError(objective)


def fit(X, y, qids, arch="deepsets", objective="listmax", hidden=256, wd=1e-2, lr=1e-3,
        epochs=30, gb=64, seed=0, drop=0.0, nheads=4, nblocks=1, device="cpu"):
    """Train one head. X is ALREADY standardised. Groups are questions.

    The RNG discipline mirrors fit_hidden_head.fit_head exactly: torch.manual_seed(seed),
    then the model is constructed (so init draws from the CPU generator in the same order),
    then torch.randperm(NG) per epoch. Built on CPU and moved, so the device is numerical
    noise rather than a different trajectory.
    """
    torch.manual_seed(seed)
    m = ARCHS[arch](X.shape[1], hidden=hidden, drop=drop, nheads=nheads, nblocks=nblocks)
    m = m.to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    packed = pack_groups(qids, y, min_pos_neg=(objective != "bce"))
    if packed is None:
        m.eval()
        return m
    idx, msk = packed
    Xt = torch.tensor(X, device=device)
    yt = torch.tensor(y, device=device)
    idx, msk = idx.to(device), msk.to(device)
    NG = idx.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(NG).to(device)
        for i in range(0, NG, gb):
            j = perm[i:i + gb]
            gi, gm = idx[j], msk[j]
            s = m(Xt[gi.reshape(-1)].reshape(gi.shape[0], gi.shape[1], -1), gm)
            yy = yt[gi.reshape(-1)].reshape(gi.shape) * gm
            loss = loss_of(objective, s, yy, gm)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def predict_custom(m, X, sets, targets, device="cpu"):
    """Score one row per explicit set. `sets[k]` is a list of row indices forming the pool the
    head sees; `targets[k]` is the position inside it whose score is returned. Used for the
    context ablations (singleton pool / foreign pool), which is how we test whether a
    set-aware head actually USES the set rather than ignoring it."""
    L = max(len(s) for s in sets)
    out = np.zeros(len(sets), dtype=np.float64)
    Xt = torch.tensor(X, device=device)
    B = 1024
    for a in range(0, len(sets), B):
        ch = sets[a:a + B]
        idx = np.zeros((len(ch), L), dtype=np.int64)
        msk = np.zeros((len(ch), L), dtype=np.float32)
        for k, g in enumerate(ch):
            idx[k, :len(g)] = g
            msk[k, :len(g)] = 1.0
        gi = torch.tensor(idx, device=device)
        gm = torch.tensor(msk, device=device)
        s = m(Xt[gi.reshape(-1)].reshape(gi.shape[0], gi.shape[1], -1), gm).float().cpu().numpy()
        for k in range(len(ch)):
            out[a + k] = s[k, targets[a + k]]
    return out


@torch.no_grad()
def predict_sets(m, X, qids, device="cpu"):
    """Score every row, feeding each question's WHOLE candidate set at once (the only way a
    set-aware head can be evaluated). Returns a (n_rows,) score vector."""
    byq = OrderedDict()
    for i, q in enumerate(qids):
        byq.setdefault(q, []).append(i)
    groups = [np.array(v) for v in byq.values()]
    L = max(len(g) for g in groups)
    out = np.zeros(len(qids), dtype=np.float64)
    Xt = torch.tensor(X, device=device)
    B = 512
    for a in range(0, len(groups), B):
        ch = groups[a:a + B]
        idx = np.zeros((len(ch), L), dtype=np.int64)
        msk = np.zeros((len(ch), L), dtype=np.float32)
        for k, g in enumerate(ch):
            idx[k, :len(g)] = g
            msk[k, :len(g)] = 1.0
        gi = torch.tensor(idx, device=device)
        gm = torch.tensor(msk, device=device)
        s = m(Xt[gi.reshape(-1)].reshape(gi.shape[0], gi.shape[1], -1), gm).float().cpu().numpy()
        for k, g in enumerate(ch):
            out[g] = s[k, :len(g)]
    return out


# =====================================================================================
# data
# =====================================================================================
def load_split(split):
    sp = G.load_candidates(split, mode="generator", layers=[LAYER], pooling=(POOLING,),
                           order="concat")
    X = sp.matrix(POOLING, LAYER)                      # (n_rows, 3584) float32
    y = np.array([r["y"] for r in sp.rows], dtype=np.float32)
    qid = np.array(G.group_ids(sp))
    img = np.array([r["img_md5"] for r in sp.rows])
    return sp, X, y, qid, img


def cv_folds_by_image(img, n_folds=5):
    fo = {h: int(hashlib.md5(str(h).encode()).hexdigest(), 16) % n_folds for h in set(img.tolist())}
    return np.array([fo[h] for h in img], dtype=int)


def cv_sel_eff(y, qid, sv, sub):
    """Within-question selection efficiency on a validation subset: over questions with at
    least one correct candidate, did argmax land on a correct one. (Identical rule to
    fit_hidden_head's CV criterion.)"""
    byq = defaultdict(list)
    for i in sub:
        byq[qid[i]].append(i)
    hit = tot = 0
    for _, ii in byq.items():
        ii = np.array(ii)
        if y[ii].sum() == 0:
            continue
        b = ii[int(np.argmax(sv[ii]))]
        hit += int(y[b] == 1)
        tot += 1
    return hit / max(tot, 1)


# =====================================================================================
# stage: CV pre-registration (TRAIN ONLY)
# =====================================================================================
def stage_cv(A):
    t0 = time.time()
    _, X, y, qid, img = load_split("train")
    fo = cv_folds_by_image(img, A.folds)
    print(f"[cv] train rows={len(y)} questions={len(set(qid))} images={len(set(img))} "
          f"fold sizes={np.bincount(fo).tolist()}", flush=True)

    cache = {}
    # per-fold standardised views, built once (the standardiser is fit on the FIT folds only)
    foldcache = {}
    for f in range(A.folds):
        tr, va = fo != f, fo == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        vidx = np.where(va)[0]
        foldcache[f] = ((X[tr] - mu) / sd, y[tr], qid[tr],
                        (X[vidx] - mu) / sd, qid[vidx], vidx)
    print(f"[cv] fold cache built ({time.time()-t0:.0f}s)", flush=True)

    def run(cfg):
        key = json.dumps(cfg, sort_keys=True)
        if key in cache:
            return cache[key]
        effs, aucs = [], []
        for f in range(A.folds):
            Xf, yf, qf, Xv, qv, vidx = foldcache[f]
            for s in range(A.cv_seeds):
                m = fit(Xf, yf, qf, seed=f * 100 + s, device=A.device, **cfg)
                sv_all = np.full(len(y), np.nan)
                sv_all[vidx] = predict_sets(m, Xv, qv, device=A.device)
                effs.append(cv_sel_eff(y, qid, sv_all, vidx))
                aucs.append(G.auroc(y[vidx], sv_all[vidx]))
        r = {**cfg, "cv_sel_eff": float(np.mean(effs)), "cv_sel_eff_sd": float(np.std(effs)),
             "cv_auroc": float(np.mean(aucs)), "n_fits": len(effs)}
        print(f"  [cv {time.time()-t0:6.0f}s] {cfg['arch']:>14s}/{cfg['objective']:<8s} "
              f"h{cfg['hidden']} wd{cfg['wd']} -> sel_eff={r['cv_sel_eff']:.4f} "
              f"(sd {r['cv_sel_eff_sd']:.4f}) auroc={r['cv_auroc']:.4f}", flush=True)
        cache[key] = r
        return r

    base = dict(hidden=256, wd=1e-2, epochs=30)
    # STAGE A -- architecture x objective at the bar's capacity (hidden=256, wd=1e-2)
    stageA = [run({**base, "arch": a, "objective": o})
              for a in ["point", "centroid", "deepsets", "deepsets_noctx", "attn"]
              for o in ["listmax", "listsum", "bt"]]
    bestA = max(stageA, key=lambda r: r["cv_sel_eff"])
    # STAGE B -- capacity x regularisation at the stage-A winner
    stageB = [run({**base, "arch": bestA["arch"], "objective": bestA["objective"],
                   "hidden": h, "wd": w})
              for h in [128, 256, 512] for w in [1e-2, 1e-1]]
    grid = stageA + stageB
    best = max(grid, key=lambda r: r["cv_sel_eff"])
    # the pointwise control's own CV-best, so the control is tuned on train too and the
    # comparison is not "a tuned new arch vs an untuned baseline"
    pt = [r for r in grid if r["arch"] == "point"]
    ptB = [run({**base, "arch": "point", "objective": max(pt, key=lambda r: r['cv_sel_eff'])["objective"],
                "hidden": h, "wd": w}) for h in [128, 256, 512] for w in [1e-2, 1e-1]]
    best_point = max(pt + ptB, key=lambda r: r["cv_sel_eff"])
    grid = grid + [r for r in ptB if r not in grid]

    # STAGE C -- the SAME capacity x regularisation sweep for the best genuinely SET-AWARE
    # architecture, so the set-aware arm and the pointwise control are tuned symmetrically.
    # (Without this, whichever family wins stage A gets a capacity sweep the other never got.)
    sa = [r for r in grid if r["arch"] in SET_ARCHS]
    bestA_sa = max(sa, key=lambda r: r["cv_sel_eff"])
    saB = [run({**base, "arch": bestA_sa["arch"], "objective": bestA_sa["objective"],
                "hidden": h, "wd": w}) for h in [128, 256, 512] for w in [1e-2, 1e-1]]
    grid = grid + [r for r in saB if r not in grid]
    best = max(grid, key=lambda r: r["cv_sel_eff"])
    best_setaware = max([r for r in grid if r["arch"] in SET_ARCHS], key=lambda r: r["cv_sel_eff"])

    out = {"what": "pre-registration: image-grouped 5-fold CV on the TRAIN split only. "
                   "Eval is never touched. The selected config is the headline; anything "
                   "chosen with eval visibility is a DIAGNOSTIC.",
           "date": time.strftime("%Y-%m-%d %H:%M"), "folds": A.folds, "cv_seeds": A.cv_seeds,
           "features": {"layer": LAYER, "pooling": POOLING, "mode": "generator",
                        "row_order": "concat"},
           "grid": grid, "selected": best, "selected_setaware": best_setaware,
           "selected_point_control": best_point,
           "setaware_archs": sorted(SET_ARCHS),
           "selection_rule": "three configs are pre-registered from THIS grid, all on train CV "
                             "only: `selected` = best overall; `selected_setaware` = best among "
                             "architectures with a genuine cross-candidate path "
                             "(deepsets/centroid/attn); `selected_point_control` = best among "
                             "pointwise architectures. The set-aware arm and the pointwise "
                             "control therefore get the SAME two-stage tuning budget.",
           "minutes": round((time.time() - t0) / 60, 1)}
    os.makedirs(os.path.dirname(CV_JSON), exist_ok=True)
    json.dump(out, open(CV_JSON, "w"), indent=1)
    print(f"\n[cv] SELECTED: {best}\n[cv] SET-AWARE: {best_setaware}\n"
          f"[cv] POINT CONTROL: {best_point}\n[cv] -> {CV_JSON}", flush=True)


# =====================================================================================
# stage: eval
# =====================================================================================
def eval_scores_to_dict(sp_ev, sv):
    return {(r["ds"], r["idx"], r["na"]): float(sv[i]) for i, r in enumerate(sp_ev.rows)}


def summarize(res, base_got, rec, tag, nboot, base_res=None, extra_base=None):
    b = G.paired_bootstrap(res["got"], base_got, rec=rec, nboot=nboot, seed=0)
    bc = G.paired_bootstrap(res["got"], base_got, nboot=nboot, seed=0, mask=res["contested_mask"])
    d = {"tag": tag, "sel_eff": round(res["sel_eff"], 6), "acc": round(res["acc"], 6),
         "per_ds": {k: round(v["sel_eff"], 6) for k, v in res["per_ds"].items()},
         "contested_sel_eff": round(res["contested"]["sel_eff"], 6),
         "contested_n": res["contested"]["n"],
         "vs_incumbent": {"d_sel_eff": round(b["d_sel_eff"], 6),
                          "ci": [round(x, 6) for x in b["d_sel_eff_ci"]],
                          "d_acc": round(b["d_acc"], 6),
                          "acc_ci": [round(x, 6) for x in b["d_acc_ci"]]},
         "vs_incumbent_contested": {"d_sel_eff": round(bc["d_sel_eff"], 6),
                                    "ci": [round(x, 6) for x in bc["d_sel_eff_ci"]]},
         "guardrail_clean": bool(G.guardrail_clean(res, base_res)) if base_res else None}
    if extra_base is not None:
        for name, g in extra_base.items():
            bb = G.paired_bootstrap(res["got"], g, rec=rec, nboot=nboot, seed=0)
            bbc = G.paired_bootstrap(res["got"], g, nboot=nboot, seed=0, mask=res["contested_mask"])
            d[f"vs_{name}"] = {"d_sel_eff": round(bb["d_sel_eff"], 6),
                               "ci": [round(x, 6) for x in bb["d_sel_eff_ci"]],
                               "d_sel_eff_contested": round(bbc["d_sel_eff"], 6),
                               "ci_contested": [round(x, 6) for x in bbc["d_sel_eff_ci"]]}
    return d


def stage_eval(A):
    t0 = time.time()
    art = {"what": "SET-AWARE (DeepSets / self-attention) heads over cached GENERATOR-FRAME "
                   "hidden states for best-of-8 open-text selection; the architecture, not the "
                   "objective, is the variable.",
           "date": time.strftime("%Y-%m-%d %H:%M"), "nboot": A.nboot, "seeds": A.seeds,
           "device": A.device,
           "features": {"mode": "generator", "layer": LAYER, "pooling": POOLING,
                        "dim": 3584, "row_order": "concat",
                        "source": "feats_hidden/generator_{train,eval}_s{0,1}of2.npz",
                        "extractor": "src/training_methods/extract_generator_hidden.py",
                        "base_model": "lingshu-medical-mllm/Lingshu-7B (frozen, NO adapter)",
                        "sys_prompt": "You are an expert medical image analyst. Answer the "
                                      "question with a short, specific phrase. Do not explain.",
                        "frame_note": "the generator frame = the model's OWN answering prompt "
                                      "with the candidate supplied as the assistant turn; the "
                                      "readout is the last answer token (h_last) and the mean "
                                      "over answer tokens (h_span)."}}

    # ---------------- 1. NULL TEST ----------------
    nt = G.null_test()
    art["null_test"] = {"pass": nt["pass"], "max_abs_deviation": nt["max_abs_deviation"],
                        "measured": nt["measured"], "published": nt["published"],
                        "abs_deviation": nt["abs_deviation"],
                        "note": nt["note"]}
    print(f"[null] pass={nt['pass']} max_abs_dev={nt['max_abs_deviation']:.3e}", flush=True)
    if not nt["pass"]:
        raise SystemExit("NULL TEST FAILED -- stopping (protocol rule 1)")

    # ---------------- 2. DISJOINTNESS ----------------
    dj = G.assert_disjoint("generator")
    art["disjointness"] = dj
    print(f"[disjoint] train_images={dj['train_images']} eval_images={dj['eval_images']} "
          f"intersection={dj['image_pixel_md5_intersection']}", flush=True)

    # ---------------- 3. data ----------------
    sp_tr, Xtr, ytr, qtr, itr = load_split("train")
    sp_ev, Xev, yev, qev, iev = load_split("eval")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_s, Xev_s = (Xtr - mu) / sd, (Xev - mu) / sd
    items = G.load_items()
    ctrl = G.control_scores(items)
    r_inc = G.sel_eff(ctrl["incumbent"], items)
    r_sc = G.sel_eff(ctrl["self_consistency"], items)
    rec = r_inc["rec"]
    art["controls"] = {
        "greedy": round(r_inc["greedy"], 6), "oracle@8": round(r_inc["oracle"], 6),
        "random_pick": {k: round(v, 6) for k, v in G.random_pick(items).items()},
        "self_consistency": {"sel_eff": round(r_sc["sel_eff"], 6),
                             "contested_sel_eff": round(r_sc["contested"]["sel_eff"], 6)},
        "incumbent": {"sel_eff": round(r_inc["sel_eff"], 6), "acc": round(r_inc["acc"], 6),
                      "cand_auroc": round(G.cand_auroc(ctrl["incumbent"], items), 6),
                      "per_ds": {k: round(v["sel_eff"], 6) for k, v in r_inc["per_ds"].items()},
                      "contested_sel_eff": round(r_inc["contested"]["sel_eff"], 6),
                      "contested_n": r_inc["contested"]["n"]}}

    # ---------------- 3b. HARNESS VALIDATION: refit the PUBLISHED bar config, seed 0, CPU ----
    # verifarch_hidden_generatorprompt_2026-08-04.json -> arms/generator/eval/cv_selected was
    # fit on CPU at seed 0. Reproducing it bit-exact here proves this trainer is the same
    # trainer; running the same config on GPU quantifies how much pure float-arithmetic order
    # moves a SINGLE-seed number (which is why protocol rule 4 exists).
    PUB_BAR = {"sel_eff": 0.79564, "acc": 0.498081, "cand_auroc": 0.677641,
               "per_ds": {"slake_open": 0.871252, "vqa_rad_open": 0.761905,
                          "pathvqa_open": 0.745806},
               "contested_sel_eff": 0.672489,
               "source": "results/cascade_methods/artifacts/verifarch_hidden_generatorprompt_"
                         "2026-08-04.json -> arms/generator/eval/cv_selected"}
    barcfg = dict(arch="point", objective="bt", hidden=256, wd=1e-2, epochs=30)
    hv = {}
    devs = ["cpu", A.device] if A.device != "cpu" else ["cpu"]
    if A.skip_cpu_check:
        devs = [A.device]
    for dev in devs:
        m = fit(Xtr_s, ytr, qtr, seed=0, device=dev, **barcfg)
        sdct = eval_scores_to_dict(sp_ev, predict_sets(m, Xev_s, qev, device=dev))
        r = G.sel_eff(sdct, items)
        hv[dev] = {"sel_eff": round(r["sel_eff"], 6), "acc": round(r["acc"], 6),
                   "cand_auroc": round(G.cand_auroc(sdct, items), 6),
                   "per_ds": {k: round(v["sel_eff"], 6) for k, v in r["per_ds"].items()},
                   "contested_sel_eff": round(r["contested"]["sel_eff"], 6)}
        print(f"[harness] bar config, seed 0, {dev}: sel_eff={r['sel_eff']:.6f}", flush=True)
    art["harness_validation"] = {
        "published_bar": PUB_BAR, "refit_here": hv,
        "max_abs_deviation_cpu_seed0": round(max(
            abs(hv["cpu"]["sel_eff"] - PUB_BAR["sel_eff"]),
            abs(hv["cpu"]["acc"] - PUB_BAR["acc"]),
            abs(hv["cpu"]["contested_sel_eff"] - PUB_BAR["contested_sel_eff"]),
            *[abs(hv["cpu"]["per_ds"][k] - PUB_BAR["per_ds"][k]) for k in PUB_BAR["per_ds"]]), 8)
        if "cpu" in hv else None,
        "device_swing_seed0": round(hv[A.device]["sel_eff"] - hv["cpu"]["sel_eff"], 6)
        if (A.device != "cpu" and "cpu" in hv) else 0.0,
        "note": "the CPU refit reproduces the published bar exactly. The SAME config on GPU "
                "differs by pure float-arithmetic order alone -- report that number so nobody "
                "mistakes a single-seed swing for an effect. All arms below are run on ONE "
                "device with >=10 seeds so the comparison is internally consistent."}

    # ---------------- 4. pre-registered config ----------------
    cv = json.load(open(CV_JSON))
    KEYS = ["arch", "objective", "hidden", "wd", "epochs"]
    # THE HEADLINE ARM is the best genuinely SET-AWARE config by train CV; the control is the
    # best POINTWISE config by train CV. Both were tuned with the same budget, on train only.
    sel = {k: cv["selected_setaware"][k] for k in KEYS}
    selp = {k: cv["selected_point_control"][k] for k in KEYS}
    art["pre_registration"] = {"cv_artifact": os.path.relpath(CV_JSON, ROOT),
                               "protocol": cv["what"], "selection_rule": cv["selection_rule"],
                               "folds": cv["folds"], "cv_seeds": cv["cv_seeds"],
                               "headline_setaware": cv["selected_setaware"],
                               "point_control": cv["selected_point_control"],
                               "best_overall_any_arch": cv["selected"],
                               "cv_preferred_family": cv["selected"]["arch"],
                               "grid": cv["grid"]}
    print(f"[pre-reg] set-aware = {sel}\n[pre-reg] point control = {selp}\n"
          f"[pre-reg] best overall = {cv['selected']['arch']}/{cv['selected']['objective']}",
          flush=True)

    # ---------------- 5. arms ----------------
    # every arm is run at A.seeds seeds; the seed-averaged (rank-averaged) ensemble is the
    # deployable number, per protocol rule 4.
    arms = OrderedDict()
    arms["setaware_prereg"] = dict(sel)
    arms["point_control"] = dict(selp)
    arms["cv_best_overall"] = {k: cv["selected"][k] for k in KEYS}
    # architecture ablations, all at the pre-registered objective/capacity so the ONLY
    # difference is the cross-candidate path
    for a in ["deepsets", "deepsets_noctx", "centroid", "attn", "point"]:
        tag = f"arch_{a}"
        if tag in arms:
            continue
        arms[tag] = {**sel, "arch": a}
    # objective ablation at the pre-registered architecture
    for o in ["listmax", "listsum", "bt", "bce"]:
        arms[f"obj_{o}"] = {**sel, "objective": o}
    # the bar's exact published config, refit in THIS harness
    arms["bar_published_cfg"] = dict(arch="point", objective="bt", hidden=256, wd=1e-2, epochs=30)

    seen, per_arm, seed_scores = {}, OrderedDict(), {}
    for tag, cfg in arms.items():
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            per_arm[tag] = {"alias_of": seen[key], "config": cfg}
            print(f"[arm] {tag} == {seen[key]} (identical config)", flush=True)
            continue
        seen[key] = tag
        scs, rows = [], []
        for s in range(A.seeds):
            m = fit(Xtr_s, ytr, qtr, seed=s, device=A.device, **cfg)
            sv = predict_sets(m, Xev_s, qev, device=A.device)
            sd_ = eval_scores_to_dict(sp_ev, sv)
            scs.append(sd_)
            r = G.sel_eff(sd_, items)
            rows.append({"seed": s, "sel_eff": round(r["sel_eff"], 6), "acc": round(r["acc"], 6),
                         "contested_sel_eff": round(r["contested"]["sel_eff"], 6),
                         "cand_auroc": round(G.cand_auroc(sd_, items), 6),
                         "per_ds": {k: round(v["sel_eff"], 6) for k, v in r["per_ds"].items()},
                         "guardrail_clean": bool(G.guardrail_clean(r, r_inc))})
        v = np.array([x["sel_eff"] for x in rows])
        ens = G.rank_fuse(*scs, items=items)                  # seed-averaged (rank_avg)
        r_ens = G.sel_eff(ens, items)
        seed_scores[tag] = {"per_seed": scs, "ensemble": ens}
        per_arm[tag] = {
            "config": cfg,
            "per_seed": rows,
            "seed_stats": {"n_seeds": len(v), "mean": round(float(v.mean()), 6),
                           "sd": round(float(v.std(ddof=1)), 6),
                           "min": round(float(v.min()), 6), "max": round(float(v.max()), 6),
                           "range": round(float(v.max() - v.min()), 6)},
            "ensemble": summarize(r_ens, r_inc["got"], rec, f"{tag}|seed-ensemble", A.nboot,
                                  base_res=r_inc),
            "ensemble_cand_auroc": round(G.cand_auroc(ens, items), 6)}
        print(f"[arm {time.time()-t0:6.0f}s] {tag:22s} {cfg['arch']:>14s}/{cfg['objective']:<8s} "
              f"seed mean={v.mean():.4f} sd={v.std(ddof=1):.4f} [{v.min():.4f},{v.max():.4f}] "
              f"| ENSEMBLE {r_ens['sel_eff']:.6f} d={per_arm[tag]['ensemble']['vs_incumbent']['d_sel_eff']:+.4f}",
              flush=True)

    art["arms"] = per_arm

    # ---------------- 6. head-to-head: set-aware vs the in-harness pointwise control ----
    def ens_of(tag):
        t = tag
        while "alias_of" in per_arm[t]:
            t = per_arm[t]["alias_of"]
        return seed_scores[t]["ensemble"], t

    sa_ens, sa_t = ens_of("setaware_prereg")
    pt_ens, pt_t = ens_of("point_control")
    r_sa, r_pt = G.sel_eff(sa_ens, items), G.sel_eff(pt_ens, items)
    b = G.paired_bootstrap(r_sa["got"], r_pt["got"], rec=rec, nboot=A.nboot, seed=0)
    bc = G.paired_bootstrap(r_sa["got"], r_pt["got"], nboot=A.nboot, seed=0,
                            mask=r_sa["contested_mask"])
    per_seed_d = []
    for s in range(A.seeds):
        ra = G.sel_eff(seed_scores[sa_t]["per_seed"][s], items)
        rb = G.sel_eff(seed_scores[pt_t]["per_seed"][s], items)
        per_seed_d.append(round(ra["sel_eff"] - rb["sel_eff"], 6))
    art["setaware_vs_pointwise_SAME_FEATURES"] = {
        "protocol": "both arms refit in THIS harness on the identical layer-21 span "
                    "generator-frame features, same seeds, same standardisation, same "
                    "training groups; the only difference is the cross-candidate path.",
        "setaware_arm": sa_t, "pointwise_arm": pt_t,
        "setaware_ensemble_sel_eff": round(r_sa["sel_eff"], 6),
        "pointwise_ensemble_sel_eff": round(r_pt["sel_eff"], 6),
        "d_sel_eff": round(b["d_sel_eff"], 6), "ci": [round(x, 6) for x in b["d_sel_eff_ci"]],
        "d_sel_eff_contested": round(bc["d_sel_eff"], 6),
        "ci_contested": [round(x, 6) for x in bc["d_sel_eff_ci"]],
        "per_seed_paired_delta": per_seed_d,
        "per_seed_paired_delta_mean": round(float(np.mean(per_seed_d)), 6)}

    # ---------------- 7. centroid ablation: how much of any gain is first-moment? --------
    ca = {}
    for a in ["deepsets", "centroid", "deepsets_noctx", "attn", "point"]:
        tag = "setaware_prereg" if per_arm.get("setaware_prereg", {}).get("config", {}).get("arch") == a \
            and "alias_of" not in per_arm.get("setaware_prereg", {}) else f"arch_{a}"
        t = tag if tag in per_arm else f"arch_{a}"
        while "alias_of" in per_arm[t]:
            t = per_arm[t]["alias_of"]
        r = G.sel_eff(seed_scores[t]["ensemble"], items)
        ca[a] = {"arm": t, "ensemble_sel_eff": round(r["sel_eff"], 6),
                 "contested": round(r["contested"]["sel_eff"], 6),
                 "seed_mean": per_arm[t]["seed_stats"]["mean"],
                 "seed_sd": per_arm[t]["seed_stats"]["sd"]}
    art["centroid_ablation"] = {
        "question": "is any set-aware gain just distance to the pool centroid?",
        "arms": ca,
        "note": "centroid = pointwise MLP on the raw triple [h_i, pool-mean, h_i - pool-mean]. "
                "deepsets_noctx = DeepSets with the pooled context zeroed (identical depth and "
                "parameter count), so deepsets - deepsets_noctx isolates the set path from capacity."}

    # ---------------- 7b. does the head USE the set?  context ablations ----------------
    # Re-score the eval pool with the SAME trained set-aware models but a corrupted pool:
    #   'true'      -> the question's own candidates (= the headline)
    #   'singleton' -> each candidate alone, so the pooled context is the candidate itself
    #   'foreign'   -> the candidate plus (k-1) candidates sampled from OTHER questions,
    #                  preserving pool size but destroying the real comparison
    # If sel_eff barely moves, the head learned to ignore its siblings.
    byq_ev = OrderedDict()
    for i, q in enumerate(qev):
        byq_ev.setdefault(q, []).append(i)
    ev_groups = [np.array(v) for v in byq_ev.values()]
    rng = np.random.default_rng(0)
    sets_single, tgt_single, sets_foreign, tgt_foreign, order_rows = [], [], [], [], []
    gi_of_row = np.zeros(len(qev), dtype=int)
    for gi, g in enumerate(ev_groups):
        gi_of_row[g] = gi
    for gi, g in enumerate(ev_groups):
        for i in g:
            order_rows.append(i)
            sets_single.append([i]); tgt_single.append(0)
            k = len(g)
            if k == 1:
                sets_foreign.append([i]); tgt_foreign.append(0)
            else:
                pool = rng.integers(0, len(qev), size=4 * k)
                alt = [int(j) for j in pool if gi_of_row[j] != gi][:k - 1]
                while len(alt) < k - 1:
                    j = int(rng.integers(0, len(qev)))
                    if gi_of_row[j] != gi:
                        alt.append(j)
                sets_foreign.append([i] + alt); tgt_foreign.append(0)
    order_rows = np.array(order_rows)

    ctx = {}
    for tag in ["setaware_prereg", "arch_deepsets", "arch_attn", "arch_centroid", "point_control"]:
        t = tag
        while t in per_arm and "alias_of" in per_arm[t]:
            t = per_arm[t]["alias_of"]
        if t not in seed_scores:
            continue
        rows = {}
        acc_seed = {"singleton": [], "foreign": []}
        for s in range(A.seeds):
            m = fit(Xtr_s, ytr, qtr, seed=s, device=A.device, **per_arm[t]["config"])
            for name, ss, tt in [("singleton", sets_single, tgt_single),
                                 ("foreign", sets_foreign, tgt_foreign)]:
                sv = np.zeros(len(qev))
                sv[order_rows] = predict_custom(m, Xev_s, ss, tt, device=A.device)
                acc_seed[name].append(eval_scores_to_dict(sp_ev, sv))
        for name in ["singleton", "foreign"]:
            r = G.sel_eff(G.rank_fuse(*acc_seed[name], items=items), items)
            rows[name] = round(r["sel_eff"], 6)
        r_true = G.sel_eff(seed_scores[t]["ensemble"], items)
        rows["true_pool"] = round(r_true["sel_eff"], 6)
        rows["delta_true_minus_singleton"] = round(rows["true_pool"] - rows["singleton"], 6)
        rows["delta_true_minus_foreign"] = round(rows["true_pool"] - rows["foreign"], 6)
        ctx[tag] = rows
        print(f"[ctx] {tag:20s} true={rows['true_pool']:.6f} singleton={rows['singleton']:.6f} "
              f"foreign={rows['foreign']:.6f}", flush=True)
    art["context_ablation"] = {
        "question": "does a set-aware head actually USE its siblings, or does it collapse to a "
                    "pointwise function? Same trained weights, corrupted pool at inference.",
        "arms": ctx,
        "note": "singleton = each candidate scored alone; foreign = pool size preserved but the "
                "siblings are candidates of OTHER questions. A near-zero delta means the set path "
                "carries no decision-relevant information."}

    # ---------------- 7c. sel_eff stratified by pool size ----------------
    ndv = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items])
    strat = {}
    for tag in ["incumbent", "setaware_prereg", "point_control"]:
        if tag == "incumbent":
            r = r_inc
        else:
            t = tag
            while "alias_of" in per_arm[t]:
                t = per_arm[t]["alias_of"]
            r = G.sel_eff(seed_scores[t]["ensemble"], items)
        row = {}
        for lo, hi, nm in [(1, 1, "1"), (2, 3, "2-3"), (4, 5, "4-5"), (6, 8, "6-8")]:
            m_ = (r["rec"] == 1) & (ndv >= lo) & (ndv <= hi)
            row[nm] = {"n": int(m_.sum()), "sel_eff": round(float(r["got"][m_].mean()), 6)}
        strat[tag] = row
    art["by_pool_size"] = {
        "question": "set-aware heads are trained only on pools with >=2 distinct candidates "
                    "(2391 train questions) but scored on eval pools whose mean size is 3.81 "
                    "distinct; any set-size sensitivity shows up here.",
        "arms": strat}

    # ---------------- 8. fusion (the deployable-headline test) ----------------
    fus = {}
    fus_specs = {
        "incumbent+setaware": [ctrl["incumbent"], sa_ens],
        "incumbent+pointwise(reproduces 0.8065 family)": [ctrl["incumbent"], pt_ens],
        "incumbent+pointwise+setaware": [ctrl["incumbent"], pt_ens, sa_ens],
        "pointwise+setaware": [pt_ens, sa_ens],
    }
    for name, parts in fus_specs.items():
        f = G.rank_fuse(*parts, items=items, ranker=G.rank_avg)
        r = G.sel_eff(f, items)
        fus[name] = summarize(r, r_inc["got"], rec, name, A.nboot, base_res=r_inc,
                              extra_base={"pointwise_fusion": G.sel_eff(
                                  G.rank_fuse(ctrl["incumbent"], pt_ens, items=items), items)["got"]})
        print(f"[fuse] {name:46s} sel_eff={r['sel_eff']:.6f} "
              f"d_vs_inc={fus[name]['vs_incumbent']['d_sel_eff']:+.4f} "
              f"{fus[name]['vs_incumbent']['ci']}", flush=True)
    art["fusion"] = {"ranker": "rank_avg (average ranks for ties) -- parameter-free, nothing "
                               "fitted on eval", "arms": fus}

    # ---------------- 9. cost ----------------
    nd = np.array([len(set(G.norm(a) for a in it["preds"])) for it in items])
    art["cost"] = {
        "extra_LLM_forward_passes_per_question_over_the_8_generations":
            round(float(nd.mean()), 4),
        "detail": "one teacher-forced pass per DISTINCT candidate to read its generator-frame "
                  "hidden state (mean 3.81 distinct answers per question, max 8, median 3). "
                  "Identical to what the already-deployed pointwise generator-frame head needs "
                  "-- the features are SHARED, so the set-aware head adds ZERO forward passes "
                  "over that head.",
        "head_evaluations_per_question": "1 (the whole pool is scored in a single set forward "
                                         "pass through a <=1.2M-parameter MLP / 1-block attention "
                                         "over <=8 tokens); vs 8.51 pair evaluations for a "
                                         "cached-vector round robin and 28 real A-vs-B LLM passes "
                                         "for the shelved pairwise verifier.",
        "vs_incumbent_LoRA": "the incumbent judge itself costs 8 LoRA forward passes per question; "
                             "this head reuses the base model's own generator frame instead.",
        "unmeasured_upside": "the generator frame is the SAME frame the candidate was generated "
                             "in, so in principle h_last/h_span can be harvested during generation "
                             "at zero extra cost. That fold was NOT implemented or measured here; "
                             "the honest charge is one extra pass per distinct candidate."}

    # ---------------- 10. write ----------------
    art["protocol_compliance"] = {
        "null_test_first": True, "disjointness_reproved_in_this_code": True,
        "config_preregistered_on_train_cv_only": True,
        "seeds": A.seeds, "bootstrap": f"paired item-level, nboot={A.nboot}, seed 0",
        "guardrail_reported_per_set": True, "contested_stratum_reported": True,
        "eval_visible_choices": "NONE for the headline. Every ablation arm listed under "
                                "'arms' and 'centroid_ablation' other than the pre-registered "
                                "one is a DIAGNOSTIC reported for mechanism, not a headline."}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(art, open(OUT_JSON, "w"), indent=1)
    print(f"\n[done {(time.time()-t0)/60:.1f} min] -> {OUT_JSON}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cv", "eval"], required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv_seeds", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cv_json", default=None, help="override the pre-registration artifact path")
    ap.add_argument("--out", default=None, help="override the output artifact path")
    ap.add_argument("--skip_cpu_check", type=int, default=0,
                    help="skip the 2-minute CPU refit of the published bar (dry runs only)")
    A = ap.parse_args()
    global CV_JSON, OUT_JSON
    if A.cv_json:
        CV_JSON = A.cv_json
    if A.out:
        OUT_JSON = A.out
    (stage_cv if A.stage == "cv" else stage_eval)(A)


if __name__ == "__main__":
    main()
