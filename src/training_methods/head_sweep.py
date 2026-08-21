#!/usr/bin/env python3
"""head_sweep.py -- MAKE THE MLP HEAD AS GOOD AS IT CAN BE, on the features already on disk.

WHY THIS EXISTS.  The deployed head (ckpts/train/genframe_head_ens8) was chosen by a GREEDY
THREE-STAGE search: stage 1 picked layer x pooling with a LINEAR pointwise head, stage 2 picked the
objective at hidden=0, stage 3 added capacity.  That path is provably not the joint optimum -- the
same run's own numbers show bce/hidden=256 at CV 0.68977 against the selected bt/hidden=256 at
CV 0.67846 (verifarch_hidden_cvgap_2026-08-04.json + .../generatorprompt...json), i.e. the objective
choice REVERSES once capacity is added.  It also never used more than ONE of the four cached layers
or more than ONE of the two cached poolings.

WHAT IS SWEPT (all of it on cached features -- zero GPU):
  representation   single layer / multi-layer concat / span+last concat / per-layer gating
  normalisation    z-score (incumbent) / L2-then-z / LayerNorm-first
  architecture     depth, width, dropout, layernorm
  objective        bce / bt / listwise / bce+bt hybrid, JOINTLY with capacity (the fixed defect)
  optimisation     epochs, lr schedule
  set-awareness    none / setrel residual (incumbent) / DeepSets pool / candidate self-attention
  data             per-dataset reweighting (the train pool is 71% PathVQA)

PROTOCOL -- UNCHANGED FROM THE INCUMBENT, AND NON-NEGOTIABLE:
  5-fold CV with folds assigned by md5(image hash) % 5, entirely inside the DISJOINT TRAIN pool.
  THE 2,345-QUESTION EVAL POOL IS NEVER READ BY THIS SCRIPT.  Selection happens on CV only; the
  winner gets exactly one eval measurement afterwards, by a separate script.

NUMERICS.  torch thread count changes CPU batch permutations and moves sel_eff by ~0.005
(CLAUDE.md).  Every config here runs at the SAME pinned thread count, and the incumbent configs are
re-run INSIDE this sweep as matched controls, so all deltas are within-sweep.  Absolute values are
NOT comparable to the published 2026-08-04 numbers.

  python3 src/training_methods/head_sweep.py --stage null
  python3 src/training_methods/head_sweep.py --stage main --workers 8
"""
import argparse, os, sys, json, hashlib, time, itertools

# ---------------------------------------------------------------------------------------------
# THREAD PINNING MUST HAPPEN BEFORE numpy/torch LOAD.  OpenMP reads these at library init, so
# setting them from main() is too late: every process then spawns one OpenMP thread per core.
# Six such processes on 48 cores oversubscribed 6x and segfaulted round 2 (2026-08-18) once the
# matrices reached hidden=1024.  The launcher exports HEAD_SWEEP_THREADS; this honours it.
_T = os.environ.get("HEAD_SWEEP_THREADS", "")
if _T:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_v, _T)
# ---------------------------------------------------------------------------------------------
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
FEATS = os.path.join(ROOT, "feats_hidden")
OUTDIR = os.path.join(ROOT, "results/cascade_methods/artifacts")
JOURNAL = os.path.join(OUTDIR, "_head_sweep_journal.jsonl")   # overridden per shard in main()
LAYERS = [7, 14, 21, 28]
FOLDS = 5


# ------------------------------------------------------------------ features
def load_train():
    """Merge the two shards of the generator-frame TRAIN cache."""
    zs, rows = [], []
    for sh in (0, 1):
        z = np.load(os.path.join(FEATS, f"generator_train_s{sh}of2.npz"))
        m = json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))
        zs.append({"h_last": z["h_last"], "h_span": z["h_span"]})
        rows += m["rows"]
    H = {k: np.concatenate([z[k] for z in zs], 0) for k in ("h_last", "h_span")}
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    H = {k: v[keep] for k, v in H.items()}
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    img = np.array([r["img_md5"] for r in rows])
    ds = np.array([r["ds"] for r in rows])
    return H, y, qid, img, ds


def assemble(H, spec):
    """spec: {'layers': [21], 'pools': ['span'], 'norm': 'z'|'l2z'} -> float32 [n, d]"""
    parts = []
    for p in spec["pools"]:
        key = "h_span" if p == "span" else "h_last"
        for L in spec["layers"]:
            X = H[key][:, LAYERS.index(L)].astype(np.float32)
            if spec.get("norm") == "l2z":
                X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
            parts.append(X)
    return parts[0] if len(parts) == 1 else np.concatenate(parts, 1)


def add_setrel(X, qid, mode):
    """mode: 'none' | 'resid' (incumbent: x - group mean) | 'both' (x, x-mean, group mean)"""
    if mode == "none":
        return X
    byq = defaultdict(list)
    for i, q in enumerate(qid):
        byq[q].append(i)
    M = np.empty_like(X)
    for q, ii in byq.items():
        M[ii] = X[ii].mean(0, keepdims=True)
    if mode == "resid":
        return np.concatenate([X, X - M], 1)
    return np.concatenate([X, X - M, M], 1)


# ------------------------------------------------------------------ head
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d, hidden=0, depth=1, drop=0.0, ln=False):
        super().__init__()
        layers = []
        if ln:
            layers.append(nn.LayerNorm(d))
        if hidden:
            cur = d
            for _ in range(depth):
                layers += [nn.Linear(cur, hidden), nn.GELU(), nn.Dropout(drop)]
                cur = hidden
            layers.append(nn.Linear(cur, 1))
        else:
            layers.append(nn.Linear(d, 1))
        self.f = nn.Sequential(*layers)

    def forward(self, x):
        return self.f(x).squeeze(-1)


def _groups(gtr, ytr, need_both=True):
    byq = defaultdict(list)
    for i, g in enumerate(gtr):
        byq[g].append(i)
    gs = [np.array(v) for v in byq.values()]
    gs = [g for g in gs if ytr[g].sum() > 0 and (not need_both or (1 - ytr[g]).sum() > 0)]
    if not gs:
        return None
    L = max(len(g) for g in gs)
    idx = np.zeros((len(gs), L), dtype=np.int64)
    msk = np.zeros((len(gs), L), dtype=np.float32)
    for k, g in enumerate(gs):
        idx[k, :len(g)] = g
        msk[k, :len(g)] = 1.0
    return torch.tensor(idx), torch.tensor(msk)


def fit(Xtr, ytr, gtr, wtr=None, objective="bce", hidden=0, depth=1, drop=0.0, ln=False,
        wd=1e-2, lr=1e-3, epochs=30, bs=256, seed=0, sched="none", bt_margin=0.0, hybrid=0.0):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1], hidden, depth, drop, ln)
    # foreach=False forces the single-tensor Adam loop.
    #
    # CORRECTION 2026-08-21: this comment used to claim the multi-tensor (_foreach) path was the
    # CAUSE of the 2026-08-18 segfaults and that foreach=False fixed them.  It does not.  With
    # foreach=False in place the crash simply MOVES to _single_tensor_adam (adam.py:535), as
    # faulthandler showed on head_finelayer.py.  The reproducible variable is THREAD COUNT: 10
    # threads dies in ~30s, 4 threads runs clean, and the identical fit in isolation at 10 threads
    # completes 30 epochs in 38.5s.  Keep foreach=False (it is harmless and marginally more
    # predictable), but the real mitigation is to run at <=4-6 threads and shard for parallelism.
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, foreach=False)
    X = torch.tensor(Xtr); y = torch.tensor(ytr)
    W = torch.tensor(wtr) if wtr is not None else None
    packed = _groups(gtr, ytr) if objective in ("bt", "listwise", "hybrid") else None
    if objective in ("bt", "listwise", "hybrid") and packed is None:
        m.eval(); return m
    nsteps = epochs * (max(1, (len(y) // bs)) if objective == "bce" else
                       max(1, packed[0].shape[0] // 64))
    sc = torch.optim.lr_scheduler.CosineAnnealingLR(opt, nsteps) if sched == "cos" else None

    def bce_step(j):
        pos = float(ytr.mean())
        pw = torch.tensor((1 - pos) / max(pos, 1e-6))
        lf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        l = lf(m(X[j]), y[j])
        return (l * W[j]).mean() if W is not None else l.mean()

    if objective == "bce":
        n = len(y)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                opt.zero_grad(); bce_step(perm[i:i + bs]).backward(); opt.step()
                if sc: sc.step()
        m.eval(); return m

    idx, msk = packed
    NG, gb = idx.shape[0], 64
    for _ in range(epochs):
        perm = torch.randperm(NG)
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
                d = s.unsqueeze(2) - s.unsqueeze(1) - bt_margin
                w = pm * nm
                l = ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
                if objective == "hybrid" and hybrid > 0:
                    flat = gi.reshape(-1)[gm.reshape(-1) > 0]
                    l = l + hybrid * bce_step(flat)
            opt.zero_grad(); l.backward(); opt.step()
            if sc: sc.step()
    m.eval(); return m


def predict(m, X, bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(m(torch.tensor(X[i:i + bs])).numpy())
    return np.concatenate(out)


def auroc(y, s):
    o = np.argsort(s); y = np.asarray(y)[o]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ CV
def run_cv(cfg, H, y, qid, img, ds, seeds=(0,)):
    """5-fold image-grouped CV; returns mean sel_eff / auroc over folds (and seeds)."""
    X = assemble(H, cfg)
    X = add_setrel(X, qid, cfg.get("setrel", "none"))
    fold_of = {h: int(hashlib.md5(str(h).encode()).hexdigest(), 16) % FOLDS for h in set(img)}
    fo = np.array([fold_of[h] for h in img])
    if cfg.get("pairs_only"):
        # restrict BCE to exactly the questions BT is given: those with both labels present.
        byq_lab = defaultdict(set)
        for i, q in enumerate(qid):
            byq_lab[q].add(int(y[i]))
        keep = np.array([len(byq_lab[q]) == 2 for q in qid])
        X, y, qid, img, ds = X[keep], y[keep], qid[keep], img[keep], ds[keep]
    w = None
    if cfg.get("reweight") == "per_ds":
        cnt = Counter(ds); n = len(ds)
        w = np.array([n / (len(cnt) * cnt[d]) for d in ds], dtype=np.float32)
    effs, aucs = [], []
    for sd_ in seeds:
        for f in range(FOLDS):
            tr, va = fo != f, fo == f
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xt = (X[tr] - mu) / sg
            m = fit(Xt, y[tr], qid[tr], None if w is None else w[tr],
                    objective=cfg["objective"], hidden=cfg["hidden"], depth=cfg.get("depth", 1),
                    drop=cfg.get("drop", 0.0), ln=cfg.get("ln", False), wd=cfg["wd"],
                    lr=cfg.get("lr", 1e-3), epochs=cfg["epochs"], seed=f + 100 * sd_,
                    sched=cfg.get("sched", "none"), bt_margin=cfg.get("bt_margin", 0.0),
                    hybrid=cfg.get("hybrid", 0.0))
            sv = predict(m, (X[va] - mu) / sg)
            aucs.append(auroc(y[va], sv))
            vidx = np.where(va)[0]
            loc = {i: j for j, i in enumerate(vidx)}
            byq = defaultdict(list)
            for i in vidx:
                byq[qid[i]].append(i)
            hit = tot = 0
            for q, ii in byq.items():
                if y[ii].sum() == 0:
                    continue
                b = ii[int(np.argmax([sv[loc[i]] for i in ii]))]
                hit += int(y[b] == 1); tot += 1
            effs.append(hit / max(tot, 1))
    return {"cv_sel_eff": float(np.mean(effs)), "cv_sel_eff_sd": float(np.std(effs)),
            "cv_auroc": float(np.nanmean(aucs)), "n_feat": int(X.shape[1]),
            "n_fits": len(effs)}


# ------------------------------------------------------------------ grids
def key(cfg):
    return json.dumps(cfg, sort_keys=True)


BASE = dict(layers=[21], pools=["span"], norm="z", setrel="none", objective="bt",
            hidden=256, depth=1, drop=0.0, ln=False, wd=1e-2, lr=1e-3, epochs=30,
            sched="none", reweight="none")


def C(**kw):
    c = dict(BASE); c.update(kw); return c


def grid_null():
    """Matched controls: the incumbent's own CV cells, re-run inside this sweep."""
    return [
        C(objective="bce", hidden=0, tag="ctl_L21span_bce_h0"),          # published 0.6643
        C(objective="bt",  hidden=0, tag="ctl_L21span_bt_h0"),           # published 0.6663
        C(objective="listwise", hidden=0, tag="ctl_L21span_listwise_h0"),# published 0.6615
        C(objective="bt",  hidden=256, tag="ctl_DEPLOYED_bt_h256"),      # published 0.6785
        C(objective="bce", hidden=256, tag="ctl_bce_h256"),              # published 0.6898
        C(layers=[28], objective="bce", hidden=0, tag="ctl_L28span_bce_h0"),
    ]


def grid_main():
    g = []
    # -- A. objective x capacity JOINTLY (the greedy-search defect) -----------
    for obj in ("bce", "bt", "listwise"):
        for h in (0, 128, 256, 512, 1024):
            g.append(C(objective=obj, hidden=h, tag=f"A_obj{obj}_h{h}"))
    # -- B. representation: multi-layer and multi-pooling ---------------------
    for ls in ([21], [28], [14, 21], [21, 28], [14, 21, 28], [7, 14, 21, 28]):
        for ps in (["span"], ["last"], ["span", "last"]):
            g.append(C(layers=ls, pools=ps, tag=f"B_L{'-'.join(map(str,ls))}_{'+'.join(ps)}"))
    # -- C. normalisation -----------------------------------------------------
    for nm in ("z", "l2z"):
        for ln in (False, True):
            g.append(C(norm=nm, ln=ln, tag=f"C_norm{nm}_ln{int(ln)}"))
    # -- D. architecture ------------------------------------------------------
    for h in (256, 512, 1024):
        for dp in (1, 2):
            for dr in (0.0, 0.1, 0.3):
                g.append(C(hidden=h, depth=dp, drop=dr, tag=f"D_h{h}_d{dp}_dr{dr}"))
    # -- E. optimisation ------------------------------------------------------
    for ep in (15, 30, 60, 120):
        for sc in ("none", "cos"):
            g.append(C(epochs=ep, sched=sc, tag=f"E_ep{ep}_{sc}"))
    for lr in (3e-4, 1e-3, 3e-3):
        g.append(C(lr=lr, tag=f"E_lr{lr}"))
    for wd in (1e-3, 1e-2, 1e-1):
        g.append(C(wd=wd, tag=f"E_wd{wd}"))
    # -- F. loss refinements --------------------------------------------------
    for mg in (0.0, 0.25, 0.5, 1.0):
        g.append(C(bt_margin=mg, tag=f"F_btmargin{mg}"))
    for hy in (0.1, 0.3, 1.0):
        g.append(C(objective="hybrid", hybrid=hy, tag=f"F_hybrid{hy}"))
    # -- G. set awareness -----------------------------------------------------
    for sr in ("none", "resid", "both"):
        for obj in ("bce", "bt"):
            g.append(C(setrel=sr, objective=obj, tag=f"G_sr{sr}_{obj}"))
    # -- H. training-data balance (pool is 71% PathVQA) -----------------------
    for rw in ("none", "per_ds"):
        for obj in ("bce", "bt"):
            g.append(C(reweight=rw, objective=obj, tag=f"H_rw{rw}_{obj}"))
    # dedupe
    seen, out = set(), []
    for c in g:
        k = key({a: b for a, b in c.items() if a != "tag"})
        if k in seen:
            continue
        seen.add(k); out.append(c)
    return out


def best_from_journals(prefix="_head_sweep_journal_main_", metric="cv_sel_eff"):
    """Read every finished shard journal and return the best record."""
    import glob
    recs = []
    for f in glob.glob(os.path.join(OUTDIR, prefix + "*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                if metric in r:
                    recs.append(r)
            except Exception:
                pass
    return max(recs, key=lambda r: r[metric]) if recs else None


def grid_main2():
    """ROUND 2.  Round 1's A-series showed the objective the deployed head uses (bt) is beaten by
    every bce cell -- but B/C/D/E/F/G/H all inherited bt from BASE, so every other axis was explored
    on top of the WRONG objective.  Round 2 re-runs those axes on the round-1 winner."""
    b = best_from_journals()
    if b is None:
        raise SystemExit("round 2 needs round 1's journals")
    keep = ("layers", "pools", "norm", "setrel", "objective", "hidden", "depth", "drop", "ln",
            "wd", "lr", "epochs", "sched", "reweight", "bt_margin", "hybrid")
    W = {k: b[k] for k in keep if k in b}
    print(f"round-2 base = {b.get('tag')} at cv_sel_eff={b['cv_sel_eff']:.5f}", flush=True)

    def W_(**kw):
        c = dict(BASE); c.update(W); c.update(kw); return c

    g = []
    # representation, now on the winning objective
    for ls in ([21], [14, 21], [21, 28], [14, 21, 28], [7, 14, 21, 28]):
        for ps in (["span"], ["span", "last"]):
            g.append(W_(layers=ls, pools=ps, tag=f"R2_L{'-'.join(map(str,ls))}_{'+'.join(ps)}"))
    # normalisation / layernorm
    for nm in ("z", "l2z"):
        for ln in (False, True):
            g.append(W_(norm=nm, ln=ln, tag=f"R2_norm{nm}_ln{int(ln)}"))
    # depth / dropout at the winning width
    for dp in (1, 2):
        for dr in (0.0, 0.1, 0.3):
            g.append(W_(depth=dp, drop=dr, tag=f"R2_d{dp}_dr{dr}"))
    # schedule / length / regularisation
    for ep, sc in ((30, "none"), (60, "cos"), (120, "cos")):
        g.append(W_(epochs=ep, sched=sc, tag=f"R2_ep{ep}_{sc}"))
    for wd in (1e-3, 1e-2, 1e-1):
        g.append(W_(wd=wd, tag=f"R2_wd{wd}"))
    # set-awareness and data balance, on the winning objective
    for sr in ("resid", "both"):
        g.append(W_(setrel=sr, tag=f"R2_sr{sr}"))
    g.append(W_(reweight="per_ds", tag="R2_rw_per_ds"))
    seen, out = set(), []
    for c in g:
        k = key({a: b2 for a, b2 in c.items() if a != "tag"})
        if k in seen:
            continue
        seen.add(k); out.append(c)
    return out


def grid_budget():
    """SETTLE OR DROP THE OBJECTIVE CLAIM (2026-08-19 audit).

    The reported +0.01695 for bce over bt is confounded two ways, both inside fit():
      DATA   _groups(need_both=True) drops every question lacking both a positive and a negative,
             so BT trains on 12,244 of 31,498 rows (38.9%) and 2,391 of 6,029 questions (39.7%),
             while BCE sees all of them.
      STEPS  at equal `epochs`, BCE takes epochs*n/256 steps and BT epochs*NG/64 -- 3,720 vs 1,140,
             a factor of 3.26.
    This project's own data curve says the last data doubling is worth +0.024-0.036 sel_eff, so a
    2.57x data deficit alone predicts the entire gap.  This grid removes both confounds:
      bt at epochs 30/60/98/120   -- step-matched at ~98
      bce at epochs 9             -- step-matched DOWN to bt's budget instead
      bce_pairsonly               -- BCE restricted to the same both-label questions BT sees
    If bce still wins with the budget matched, the claim stands.  If not, it is dropped.
    """
    B = {**BASE, "hidden": 256}
    g = []
    for ep in (30, 60, 98, 120):
        g.append({**B, "objective": "bt", "epochs": ep, "tag": f"BUD_bt_ep{ep}"})
    for ep in (9, 30):
        g.append({**B, "objective": "bce", "epochs": ep, "tag": f"BUD_bce_ep{ep}"})
    for ep in (30, 98):
        g.append({**B, "objective": "bce", "epochs": ep, "pairs_only": 1,
                  "tag": f"BUD_bce_pairsonly_ep{ep}"})
    return g


def grid_reg():
    """REGULARISATION.  head_curve_bce_2026-08-18.json shows the head MEMORISES: train sel_eff
    0.96-0.99 against CV 0.59-0.70, a gap of +0.27 to +0.40 at every width, and the gap shrinks
    with data rather than with capacity.  Round 1 only swept dropout and weight decay on the BT
    objective, which the same sweep then refuted.  This sweeps them properly on BCE, plus the
    directions a memorising model actually responds to: shorter training, smaller heads, label
    smoothing via a soft target, and heavy weight decay."""
    g = []
    B = {**BASE, "objective": "bce", "hidden": 256}
    for dr in (0.0, 0.1, 0.3, 0.5, 0.7):
        g.append({**B, "drop": dr, "tag": f"REG_drop{dr}"})
    for wd in (1e-3, 1e-2, 1e-1, 3e-1, 1.0):
        g.append({**B, "wd": wd, "tag": f"REG_wd{wd}"})
    for h in (16, 32, 64, 128, 256):
        g.append({**B, "hidden": h, "tag": f"REG_h{h}"})
    for ep in (5, 10, 20, 30, 60):
        g.append({**B, "epochs": ep, "tag": f"REG_ep{ep}"})
    for h in (32, 64):
        for dr in (0.3, 0.5):
            for wd in (1e-1, 3e-1):
                g.append({**B, "hidden": h, "drop": dr, "wd": wd,
                          "tag": f"REG_h{h}_dr{dr}_wd{wd}"})
    for ln in (True,):
        for dr in (0.3, 0.5):
            g.append({**B, "ln": ln, "drop": dr, "tag": f"REG_ln_dr{dr}"})
    seen, out = set(), []
    for c in g:
        k = key({a: b for a, b in c.items() if a != "tag"})
        if k in seen:
            continue
        seen.add(k); out.append(c)
    return out


def grid_final(topk=8):
    """Round 3.  Seed spread on this head is ~0.005, so single-seed CV cannot separate configs that
    differ by less than ~0.01.  Re-run the top-K from rounds 1-2 at 4 seeds x 5 folds = 20 fits each
    and rank on the seed-averaged number."""
    import glob
    recs = []
    for f in glob.glob(os.path.join(OUTDIR, "_head_sweep_journal_main*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
                if "cv_sel_eff" in r:
                    recs.append(r)
            except Exception:
                pass
    recs.sort(key=lambda r: -r["cv_sel_eff"])
    keep = ("layers", "pools", "norm", "setrel", "objective", "hidden", "depth", "drop", "ln",
            "wd", "lr", "epochs", "sched", "reweight", "bt_margin", "hybrid")
    out, seen = [], set()
    # always carry the deployed config as a matched control
    ctl = dict(BASE); ctl["tag"] = "F_CONTROL_DEPLOYED_bt_h256"
    out.append(ctl); seen.add(key({k: v for k, v in ctl.items() if k != "tag"}))
    for r in recs:
        c = {k: r[k] for k in keep if k in r}
        c = {**BASE, **c}
        k = key(c)
        if k in seen:
            continue
        seen.add(k); c["tag"] = "F_" + str(r.get("tag", ""))
        out.append(c)
        if len(out) > topk:
            break
    return out


# ------------------------------------------------------------------ driver
def load_done():
    done = {}
    if os.path.exists(JOURNAL):
        for line in open(JOURNAL):
            try:
                r = json.loads(line)
                done[r["_key"]] = r
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="null",
                    choices=["null", "main", "main2", "final", "reg", "budget"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    A = ap.parse_args()
    torch.set_num_threads(A.threads)
    _omp = os.environ.get("OMP_NUM_THREADS")
    if _omp is None or _omp != str(A.threads):
        print(f"  [warn] OMP_NUM_THREADS={_omp} but --threads={A.threads}; export "
              f"HEAD_SWEEP_THREADS before launching so BLAS and torch agree", flush=True)
    global JOURNAL
    JOURNAL = os.path.join(OUTDIR, f"_head_sweep_journal_{A.stage}_s{A.shard}of{A.nshard}.jsonl")

    print(f"[{time.strftime('%H:%M:%S')}] loading train cache ...", flush=True)
    H, y, qid, img, ds = load_train()
    print(f"  rows={len(y)} questions={len(set(qid))} images={len(set(img))} "
          f"pos_rate={y.mean():.4f}", flush=True)

    cfgs = {"null": grid_null, "main": grid_main, "main2": grid_main2,
            "final": grid_final, "reg": grid_reg, "budget": grid_budget}[A.stage]()
    cfgs = [c for i, c in enumerate(cfgs) if i % A.nshard == A.shard]
    done = load_done()
    print(f"[{A.stage}] {len(cfgs)} configs for shard {A.shard}/{A.nshard}; "
          f"{len(done)} already in journal", flush=True)

    for n, cfg in enumerate(cfgs):
        tag = cfg.pop("tag", "")
        k = key(cfg)
        if k in done:
            print(f"  [{n+1}/{len(cfgs)}] SKIP {tag}", flush=True)
            continue
        t0 = time.time()
        try:
            r = run_cv(cfg, H, y, qid, img, ds, seeds=tuple(range(A.seeds)))
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"}
        rec = {"_key": k, "tag": tag, "stage": A.stage, "threads": A.threads,
               "secs": round(time.time() - t0, 1), **cfg, **r}
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  [{n+1}/{len(cfgs)}] {tag:34} sel_eff={r.get('cv_sel_eff', float('nan')):.5f} "
              f"auroc={r.get('cv_auroc', float('nan')):.4f} d={r.get('n_feat','?')} "
              f"{rec['secs']}s", flush=True)


if __name__ == "__main__":
    main()
