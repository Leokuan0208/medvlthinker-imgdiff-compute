#!/usr/bin/env python3
"""integrate_lib.py -- shared pieces for the INTEGRATION round (2026-08-05).

The round's question is not "does a new architecture win" (four sibling rounds answered
that: pairwise-over-cached-vectors, set-aware, cheap-contrast features and real A-vs-B
forward passes are all nulls). It is: given what actually earned its place, what is the
ONE deployable selector, measured honestly against the CURRENTLY DEPLOYED fusion (0.806540)
rather than against the incumbent verifier (0.775204), which is no longer news.

Everything here is a thin layer over the frozen metric in genframe_data.py. No new metric,
no new pick rule, no new bootstrap.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import genframe_data as G  # noqa: E402

ROOT = G.ROOT
SC8_DIR = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_integrate_parts")
os.makedirs(PARTS, exist_ok=True)

#: pre-registered head configuration -- CV-selected on the TRAIN split in the 2026-08-04
#: round (verifarch_hidden_generatorprompt_2026-08-04.json -> arms.generator.cv_selected).
#: NOT re-tuned here: this round changes only the readout (seed ensembling) and the fusion.
BASE_CFG = dict(layer=21, pooling="span", objective="bt", hidden=256, wd=1e-2, epochs=30,
                lr=1e-3, bs=256)

SC8_FILE = {
    "slake_open": "ckpt_slake_open_lingshu7b_sc8.jsonl",
    "vqa_rad_open": "ckpt_vqa_rad_open_lingshu7b_sc8.jsonl",
    "pathvqa_open": "ckpt_pathvqa_open_lingshu7b_sc8.jsonl",
}


# ======================================================================================
# gold answers -> the short-answer stratum
# ======================================================================================
def gold_by_item():
    """{(ds, str(idx)) -> gold answer string} read from the sc8 pools the features came from."""
    out = {}
    for ds, f in SC8_FILE.items():
        p = os.path.join(SC8_DIR, f)
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                out[(ds, str(r["idx"]))] = str(r["gold"])
    return out


def short_mask(items, max_words=3):
    """Boolean item mask: gold answer word count <= max_words (the project's short stratum)."""
    gb = gold_by_item()
    m = np.zeros(len(items), dtype=bool)
    miss = 0
    for i, it in enumerate(items):
        g = gb.get((it["ds"], str(it["idx"])))
        if g is None:
            miss += 1
            continue
        m[i] = len(str(g).split()) <= max_words
    if miss:
        raise KeyError(f"{miss} items have no gold answer in the sc8 pools")
    return m


# ======================================================================================
# scoring helpers
# ======================================================================================
def score_map(keys, sv):
    return {keys[i]: float(sv[i]) for i in range(len(keys))}


def slot_mats(keys, svs, items):
    """List of (n_items, 8) slot-score matrices, one per seed."""
    return [G._slot_scores(score_map(keys, s), items) for s in svs]


def ens_rank(keys, svs, items):
    """Seed ensemble by MEAN WITHIN-POOL rank_avg (the convention the siblings used)."""
    mats = slot_mats(keys, svs, items)
    return {(it["ds"], it["idx"]): list(np.mean([G.rank_avg(m[i]) for m in mats], 0))
            for i, it in enumerate(items)}


def ens_score(keys, svs, items):
    """Seed ensemble by MEAN RAW SCORE."""
    return score_map(keys, np.mean(np.stack(svs), 0))


def ens_z(keys, svs, items):
    """Seed ensemble by mean of per-seed globally z-scored logits."""
    S = np.stack(svs)
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-8)
    return score_map(keys, Z.mean(0))


ENSEMBLERS = {"rank": ens_rank, "score": ens_score, "z": ens_z}


def report(smap, items, base_r, tag, nboot=10000, picks=None, extra_masks=None):
    """One selector -> the full frozen report vs a baseline result dict."""
    r = G.sel_eff(smap, items, picks=picks)
    bo = G.paired_bootstrap(r["got"], base_r["got"], rec=r["rec"], nboot=nboot)
    bc = G.paired_bootstrap(r["got"], base_r["got"], nboot=nboot, mask=r["contested_mask"])
    out = {
        "tag": tag, "sel_eff": r["sel_eff"], "acc": r["acc"],
        "per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()},
        "guardrail_clean": G.guardrail_clean(r, base_r),
        "contested_sel_eff": r["contested"]["sel_eff"], "contested_n": r["contested"]["n"],
        "d_sel_eff": bo["d_sel_eff"], "d_sel_eff_ci": bo["d_sel_eff_ci"],
        "d_acc": bo["d_acc"], "d_acc_ci": bo["d_acc_ci"],
        "d_contested": bc["d_sel_eff"], "d_contested_ci": bc["d_sel_eff_ci"],
    }
    if extra_masks:
        for name, m in extra_masks.items():
            mm = np.asarray(m, bool) & (r["rec"] == 1)
            b = G.paired_bootstrap(r["got"], base_r["got"], nboot=nboot, mask=mm)
            out[f"{name}_sel_eff"] = float(r["got"][mm].mean())
            out[f"{name}_n"] = int(mm.sum())
            out[f"{name}_d"] = b["d_sel_eff"]
            out[f"{name}_d_ci"] = b["d_sel_eff_ci"]
    return out, r


# ======================================================================================
# train-split selection efficiency (for CV pre-registration -- no incumbent exists there)
# ======================================================================================
def train_question_index(tr):
    """[(rows, y)] per TRAIN question, restricted to questions with >=2 distinct candidates
    and at least one correct one -- the only ones on which a selector can be scored."""
    out = []
    for q in tr.questions:
        if len(q.cands) < 2:
            continue
        y = np.array([c.y for c in q.cands], dtype=int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        out.append((np.array([c.row for c in q.cands]), y))
    return out


def train_sel_eff(scores, qidx):
    """Within-question selection efficiency on the train split: mean over contested,
    recoverable train questions of 1[argmax score is correct]. First-index tie-break."""
    if not qidx:
        return float("nan")
    got = [int(y[int(np.argmax(scores[rows]))] == 1) for rows, y in qidx]
    return float(np.mean(got))


def fold_of_group(groups, k=5):
    """md5(image-hash) % k -- identical assignment to fit_hidden_head.py."""
    u = sorted(set(groups))
    fo = {g: int(hashlib.md5(str(g).encode()).hexdigest(), 16) % k for g in u}
    return np.array([fo[g] for g in groups])


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    return path
