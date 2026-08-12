#!/usr/bin/env python3
"""visverif_lib.py -- shared machinery for the VISION-AWARE VERIFIER round (2026-08-12).

One place for: (a) loading the language-side cache and the new vision-token cache and aligning
them, (b) the arm definitions (what "injecting vision" concretely means), (c) the training loop
(reused VERBATIM from fit_hidden_head.fit_head so the language-side bar reproduces bit-exact),
(d) the strata (laterality / short-answer) that this attack is supposed to move first and largest,
(e) the map from open-text selection to the 8-cell MACRO headline.

Frozen metric, item order, disjointness assertion and bootstrap all come from
src/training_methods/genframe_data.py -- this module adds nothing to them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G  # noqa: E402
import fit_hidden_head as FH  # noqa: E402

VISDIR = os.path.join(ROOT, "feats_vision")
MACRO_VEC = os.path.join(ROOT, "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz")
MACRO_CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
               "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN_CELL = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open", "pathvqa_open": "PATH_VQA_open"}

# ---- the deployed language-side recipe (ckpts/train/genframe_head_ens8/recipe.json), verbatim
RECIPE = dict(mode="generator", layer=21, pooling="span", objective="bt", hidden=256,
              wd=0.01, lr=0.001, epochs=30, bs=256, drop=0.0, row_order="concat")


# ======================================================================================
# 1. caches
# ======================================================================================
def load_vision(split: str, ablate: str = "none", visdir: str = VISDIR):
    """{img_md5 -> row index} plus the arrays. Vision features are per IMAGE, by construction."""
    tag = "" if ablate == "none" else f"_{ablate}"
    p = os.path.join(visdir, f"vis_{split}{tag}.npz")
    z = np.load(p)
    meta = json.load(open(p.replace(".npz", ".meta.json")))
    rows = [r for r in meta["rows"] if "img_md5" in r]
    assert len(rows) == z["v_mean"].shape[0], f"{len(rows)} meta rows vs {z['v_mean'].shape[0]} arrays"
    idx = {r["img_md5"]: i for i, r in enumerate(rows)}
    assert len(idx) == len(rows), "duplicate img_md5 in vision cache"
    return dict(index=idx, v_mean=z["v_mean"], v_grid=z["v_grid"], grid_hw=z["grid_hw"],
                n_vis=z["n_vis"], layers=[int(x) for x in z["layers"]],
                grid_layers=[int(x) for x in z["grid_layers"]], meta=meta, rows=rows)


def align(split_obj, vis, layer: int, grid_layer: int):
    """Per-CACHE-ROW vision features: V_mean (n_rows, 3584), V_grid (n_rows, P*P, 3584).

    Duplicated by image on purpose -- every candidate of a question gets the SAME vision vector,
    which is precisely the structural fact this round has to work around.
    """
    li = vis["layers"].index(layer)
    gi = vis["grid_layers"].index(grid_layer)
    ii = np.array([vis["index"][r["img_md5"]] for r in split_obj.rows], dtype=int)
    return vis["v_mean"][ii, li].astype(np.float32), vis["v_grid"][ii, gi].astype(np.float32), ii


# ======================================================================================
# 2. arms -- what "inject the vision signal" concretely means
# ======================================================================================
def zstd(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def build_features(arm: str, Hc, Vm, Vg, P=6, rng_proj=None):
    """Hc (n,3584) language-side candidate vector; Vm (n,3584) image mean; Vg (n,P*P,3584) grid.

    Arms:
      L            language-side only                       -- THE BAR
      Vmean        image mean only                          -- DEGENERATE by construction
      L_Vmean      concat                                   -- additive; linear head cannot use it
      L_prod       concat(L, L*Vmean)                        -- explicit multiplicative interaction
      L_simgrid    concat(L, cosine(L, each of the P*P patches))  -- late interaction, P*P extra dims
      L_prod_sim   concat(L, L*Vmean, cosine grid)
      L_maxsim     concat(L, [max,mean,top3,std,argmax-x,argmax-y] of the cosine grid)
    'L*Vmean' and the cosines are the only candidate-varying vision quantities available without a
    learned attention (see xattn head for that).
    """
    n = Hc.shape[0]
    if arm == "L":
        return Hc
    if arm == "Vmean":
        return Vm
    if arm == "L_Vmean":
        return np.concatenate([Hc, Vm], 1)
    if arm == "L_prod":
        return np.concatenate([Hc, Hc * Vm], 1)
    sim = cos_grid(Hc, Vg)                                     # (n, P*P)
    if arm == "L_simgrid":
        return np.concatenate([Hc, sim], 1)
    if arm == "L_prod_sim":
        return np.concatenate([Hc, Hc * Vm, sim], 1)
    if arm == "L_maxsim":
        return np.concatenate([Hc, sim_stats(sim, P)], 1)
    raise ValueError(f"unknown arm {arm}")


def cos_grid(Hc, Vg):
    """(n, P*P) cosine between the candidate vector and every pooled patch."""
    h = Hc / (np.linalg.norm(Hc, axis=1, keepdims=True) + 1e-6)
    v = Vg / (np.linalg.norm(Vg, axis=2, keepdims=True) + 1e-6)
    return np.einsum("nd,npd->np", h, v).astype(np.float32)


def sim_stats(sim, P):
    a = np.sort(sim, 1)
    am = np.argmax(sim, 1)
    return np.stack([sim.max(1), sim.mean(1), a[:, -3:].mean(1), sim.std(1),
                     (am % P) / (P - 1.0), (am // P) / (P - 1.0)], 1).astype(np.float32)


# ======================================================================================
# 3. training (reuses fit_hidden_head verbatim)
# ======================================================================================
def fit_and_score(Xtr, ytr, gtr, Xev, seed, objective=None, hidden=None, wd=None, lr=None,
                  epochs=None, bs=None, drop=None):
    m = FH.fit_head(Xtr, ytr, gtr,
                    objective=objective or RECIPE["objective"],
                    hidden=RECIPE["hidden"] if hidden is None else hidden,
                    wd=RECIPE["wd"] if wd is None else wd,
                    lr=RECIPE["lr"] if lr is None else lr,
                    epochs=RECIPE["epochs"] if epochs is None else epochs,
                    bs=RECIPE["bs"] if bs is None else bs,
                    seed=seed, drop=RECIPE["drop"] if drop is None else drop)
    return FH.predict(m, Xev), m


def scores_by_cand(ev_split, s):
    """(n_rows,) head logits -> {(ds, idx, na): score} for genframe_data.sel_eff."""
    return {(r["ds"], r["idx"], r["na"]): float(s[i]) for i, r in enumerate(ev_split.rows)}


# ======================================================================================
# 4. strata: where a VISUAL grounding fix must show up first
# ======================================================================================
LATERAL = re.compile(r"\b(left|right|bilateral|both sides?|unilateral|lateral|medial|"
                     r"superior|inferior|anterior|posterior|upper|lower|proximal|distal)\b", re.I)


GOLDQ_CKPT = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")


def gold_and_question(items=None):
    """The REAL gold string and question text per eval item, from the greedy cheap-leg checkpoint
    ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b.jsonl (fields 'gold', 'question').
    The transfer dumps carry neither, so this is the only source; keyed by idx within ds."""
    items = items if items is not None else G.load_items()
    by = {}
    for ds in G.EVAL_DS:
        p = os.path.join(GOLDQ_CKPT, f"ckpt_{ds}_lingshu7b.jsonl")
        by[ds] = {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())}
    gold, ques, miss = [], [], 0
    for it in items:
        r = by[it["ds"]].get(it["idx"])
        if r is None:
            miss += 1
            gold.append(""); ques.append("")
        else:
            gold.append(str(r.get("gold", ""))); ques.append(str(r.get("question", "")))
    return gold, ques, miss


def strata(items=None):
    """Item masks over the canonical 2345.

    short3      REAL gold answer <= 3 words -- the documented failure stratum
    laterality  the QUESTION or any CANDIDATE carries a laterality / orientation token; this is
                the "Right. vs Left. vs Both." stratum a visually-grounded verifier must fix
    """
    items = items if items is not None else G.load_items()
    gold, ques, miss = gold_and_question(items)
    goldlen = np.array([len(g.split()) for g in gold])
    lat_q = np.array([bool(LATERAL.search(q)) for q in ques])
    lat_c = np.array([bool(LATERAL.search(" ".join(str(a) for a in it["preds"]))) for it in items])
    lat_g = np.array([bool(LATERAL.search(g)) for g in gold])
    return {"short3": goldlen <= 3, "long4plus": goldlen >= 4,
            "laterality": lat_q | lat_c | lat_g,
            "laterality_question": lat_q, "laterality_candidate": lat_c,
            "short3_and_laterality": (goldlen <= 3) & (lat_q | lat_c | lat_g),
            "gold_words": goldlen, "n_gold_missing": miss}


def stratum_sel_eff(got, rec, mask):
    m = (rec == 1) & mask
    return {"n": int(m.sum()), "sel_eff": float(got[m].mean()) if m.sum() else float("nan")}


# ======================================================================================
# 5. open-text selection -> the 8-cell MACRO headline
# ======================================================================================
def macro_table():
    z = np.load(MACRO_VEC, allow_pickle=True)
    return {c: {a: z[f"{c}|{a}"] for a in ["always_7b", "always_32b_direct"]} for c in MACRO_CELLS}


def macro_from_open(got, items=None, mt=None):
    """Replace the three OPEN cells' 7B-greedy vectors with this selector's per-item outcome,
    keep the five MCQ cells at always-7B, and return the 8-cell macro.

    The open cells of the macro table ARE this pool: SLAKE_open n=645, VQA_RAD_open n=200,
    PATH_VQA_open n=1500, same item order (asserted).
    """
    items = items if items is not None else G.load_items()
    mt = mt if mt is not None else macro_table()
    got = np.asarray(got, dtype=float)
    per = {}
    off = 0
    for ds in G.EVAL_DS:
        k = OPEN_CELL[ds]
        n = int(sum(1 for it in items if it["ds"] == ds))
        assert len(mt[k]["always_7b"]) == n, f"{k}: macro n={len(mt[k]['always_7b'])} vs pool n={n}"
        sub = got[off:off + n]
        assert all(items[off + j]["ds"] == ds for j in range(n)), "item order broken"
        per[k] = float(sub.mean())
        off += n
    assert off == len(items)
    for c in MACRO_CELLS:
        if c not in per:
            per[c] = float(mt[c]["always_7b"].mean())
    return {"macro": float(np.mean([per[c] for c in MACRO_CELLS])), "per_cell": per}


def macro_reference(mt=None):
    mt = mt if mt is not None else macro_table()
    b7 = {c: float(mt[c]["always_7b"].mean()) for c in MACRO_CELLS}
    b32 = {c: float(mt[c]["always_32b_direct"].mean()) for c in MACRO_CELLS}
    return {"always_7b": {"per_cell": b7, "macro": float(np.mean(list(b7.values())))},
            "always_32b_direct": {"per_cell": b32, "macro": float(np.mean(list(b32.values())))},
            "gap": float(np.mean(list(b32.values())) - np.mean(list(b7.values())))}


def macro_bootstrap(got_a, got_b, nboot=10000, seed=0, items=None, mt=None):
    """Paired item bootstrap of the MACRO difference between two open-arm selectors.

    Only the three open cells differ, so the five MCQ cells are constants and are resampled as
    fixed values (their contribution to the difference is exactly 0); the open cells are resampled
    WITHIN cell, which is the correct paired draw for a per-cell-equal-weight average.
    """
    items = items if items is not None else G.load_items()
    mt = mt if mt is not None else macro_table()
    a = np.asarray(got_a, float); b = np.asarray(got_b, float)
    ds_index = np.array([G.EVAL_DS.index(it["ds"]) for it in items])
    rng = np.random.default_rng(seed)
    subs = [np.where(ds_index == j)[0] for j in range(3)]
    d = np.empty(nboot)
    for k in range(nboot):
        acc = 0.0
        for s in subs:
            j = s[rng.integers(0, len(s), len(s))]
            acc += (a[j].mean() - b[j].mean())
        d[k] = acc / 8.0
    point = sum((a[s].mean() - b[s].mean()) for s in subs) / 8.0
    return {"d_macro": float(point),
            "d_macro_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "nboot": nboot, "seed": seed}


# ======================================================================================
# 6. misc
# ======================================================================================
def img_folds(rows, n_folds=5):
    return np.array([int(hashlib.md5(str(r["img_md5"]).encode()).hexdigest(), 16) % n_folds
                     for r in rows], dtype=int)


def qid(rows):
    return [f"{r['ds']}|{r['idx']}" for r in rows]


def train_cv_sel_eff(X, y, q, folds, seed=0, **kw):
    """The pre-registration endpoint: within-question argmax hit-rate on held-out TRAIN folds.
    Never looks at eval. Identical protocol to fit_hidden_head.cv()."""
    effs, aucs = [], []
    for f in sorted(set(folds.tolist())):
        tr, va = folds != f, folds == f
        Xt, mu, sd = zstd(X[tr])
        Xv = (X[va] - mu) / sd
        s, _ = fit_and_score(Xt, y[tr], [q[i] for i in np.where(tr)[0]], Xv, seed=seed, **kw)
        aucs.append(G.auroc(y[va], s))
        vidx = np.where(va)[0]
        loc = {i: j for j, i in enumerate(vidx)}
        byq = defaultdict(list)
        for i in vidx:
            byq[q[i]].append(i)
        hit = tot = 0
        for _, ii in byq.items():
            if y[ii].sum() == 0:
                continue
            b = ii[int(np.argmax([s[loc[i]] for i in ii]))]
            hit += int(y[b] == 1); tot += 1
        effs.append(hit / max(tot, 1))
    return {"cv_sel_eff": float(np.mean(effs)), "cv_auroc": float(np.mean(aucs)),
            "per_fold": [round(e, 6) for e in effs]}
