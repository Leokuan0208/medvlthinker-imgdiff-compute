#!/usr/bin/env python3
"""genframe_data.py -- THE SHARED LOADER + THE FROZEN METRIC for the generator-frame
best-of-8 selection round.

Four agents are working on the same endpoint. This module is the single definition of
(a) how a row of ``feats_hidden/*.npz`` maps to a (question, candidate) pair, (b) what
selection efficiency means, and (c) how the paired bootstrap is drawn -- so four agents
cannot drift into four metrics.

Nothing here touches a GPU, reads a model, or writes a file. Import is side-effect free.

--------------------------------------------------------------------------------------
THE ENDPOINT (frozen)
--------------------------------------------------------------------------------------
2345 open-text eval questions (slake_open 645 / vqa_rad_open 200 / pathvqa_open 1500),
each with a pool of 8 sampled Lingshu-7B answers and a per-candidate judge label.

    greedy   = mean(greedy_ok)                       0.449467
    oracle@8 = P(at least one correct in the pool)   0.626013
    selected = mean(pick is correct)
    sel_eff  = mean(pick is correct | pool recoverable) = (selected-greedy)/(oracle-greedy)
               -- computed as the conditional mean, which is the definition used by every
                  published cell; the difference form is an identity only when
                  greedy == random-slot accuracy, so DO NOT recompute it that way.

Incumbent (the bar): the CLEAN disjoint-trained LoRA verifier, per-candidate scores stored
in ckpts/train/lora_verifier_disjoint/transfer_dump_{slake,vqa_rad,pathvqa}_open_lingshu7b.json.
    sel_eff 0.775204 | selected 0.485288 | cand AUROC 0.885592
    per set: slake 0.850088 / vqa_rad 0.761905 / pathvqa 0.722581

--------------------------------------------------------------------------------------
THE FEATURE CACHE (feats_hidden/, 4.4 GB, gitignored)
--------------------------------------------------------------------------------------
Written by src/training_methods/extract_generator_hidden.py:
    {generator,grader}_{train,eval}_s{0,1}of2.npz + .meta.json

  * A row is one (ds, idx, normalized-answer) triple -- candidates are DEDUPLICATED by
    normalized answer text, so a question contributes 1..8 rows, not always 8.
    eval: 8943 rows over 2345 questions.  train: 31498 rows over 6029 questions.
  * Row order within a shard is the global order  sorted by (ds, str(idx), na)  taken as
    ``rows[shard::nshard]``.  Interleaving shard0/shard1 reconstructs that global order
    exactly (verified).  This module does not rely on it -- it keys by (ds, idx, na).

    !! ROW ORDER IS PART OF THE CONFIG.  The published 2026-08-04 heads were fit on the
    SHARD-CONCATENATED order (all of shard0, then all of shard1), because that is what
    fit_hidden_head.load_cache produced.  SGD sees rows in that order, so a different
    order is a different trajectory: the identical config at seed 0 gives sel_eff
    0.795640 in concat order and 0.799728 in sorted order -- a +0.0041 swing, larger
    than several of the effects being chased, and it even flips the guardrail flag.
    ``order='concat'`` is therefore the DEFAULT here, so the bar reproduces bit-exact.
    Treat any order/seed-0 single fit as noise-dominated; obey protocol rule 4 (>=10 seeds).
  * ``yesno`` (the base model's Yes/No logits) is populated ONLY in mode='grader'.
    In mode='generator' it is all-NaN by construction -- there is no Yes/No readout in
    the generator frame.  Do not silently use it.
  * 0 extraction failures in all 8 files (n_failed == 0, every row n_tok > 0).

--------------------------------------------------------------------------------------
QUICK USE
--------------------------------------------------------------------------------------
    from src.training_methods import genframe_data as G      # or add src/ to sys.path

    G.assert_disjoint()                       # raises unless pixel-md5 intersection == 0
    ev = G.load_candidates("eval",  layers=[21], pooling=("span",))
    tr = G.load_candidates("train", layers=[21], pooling=("span",))

    for q in ev.questions:                    # canonical item order, len 2345
        for c in q.cands:                     # distinct candidates of this question
            x = ev.h_span[c.row, 0]           # (3584,) float16
            y = c.y                           # judge label
        s = q.inc_scores                      # the incumbent's 8 slot scores

    r = G.sel_eff({(q.ds, q.idx): q.inc_scores for q in ev.questions})
    print(r["sel_eff"], r["per_ds"], r["contested"])
    b = G.paired_bootstrap(r["got"], base["got"], rec=r["rec"], nboot=10000)
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
DUMP_DIR = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
FEATDIR = os.path.join(ROOT, "feats_hidden")
EVAL_IMGHASH = os.path.join(ROOT, "data/verifarch/eval_imghash.json")

#: The three eval sets, in the canonical *reporting* order.
EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
#: Item order = the concatenation of the transfer dumps in THIS order. Every published
#: cell (and every bootstrap seed) was drawn against this order -- do not change it.
DUMP_ORDER = ["slake", "vqa_rad", "pathvqa"]

#: Published cells that any harness must reproduce before reporting a new number.
PUBLISHED = {
    "n": 2345,
    "n_recoverable": 1468,
    "oracle@8": 0.626013,
    "selected": 0.485288,
    "greedy": 0.449467,
    "sel_eff": 0.775204,
    "cand_auroc": 0.885592,
    "per_ds": {"slake_open": 0.850088, "vqa_rad_open": 0.761905, "pathvqa_open": 0.722581},
}

MISSING_SCORE = -1e9  # what an unscored slot gets, matching fit_hidden_head.py


def norm(s) -> str:
    """The project's answer-identity normalizer. Used for candidate dedup everywhere."""
    return str(s).strip().lower()


# ======================================================================================
# 1. containers
# ======================================================================================
@dataclass
class Cand:
    """One DISTINCT candidate answer of one question -- i.e. one row of the feature cache.

    na       normalized answer text (the dedup key)
    text     the surface string as generated (first occurrence in pool order)
    y        judge label, 1 = correct
    row      index into Split.h_last / h_span / yesno
    slots    the slot positions (0..7) of the 8-sample pool that carry this answer
    mult     len(slots) -- how many of the 8 samples produced this answer (a
             self-consistency count, free)
    in_draw  train only: 1 if this example was in the incumbent LoRA's matched draw
    """
    na: str
    text: str
    y: int
    row: int
    slots: Tuple[int, ...] = ()
    mult: int = 0
    in_draw: int = 1


@dataclass
class Question:
    """One question and its candidate pool.

    ds, idx      identity; (ds, idx) is unique and is the question key used everywhere
    img_md5      md5 of the DECODED RGB pixels (extract_generator_hidden convention:
                 no size prefix). Group by this for image-disjoint folds.
    cands        distinct candidates (see Cand)
    n_distinct   len(cands); n_distinct >= 2 defines the CONTESTED stratum
    -- eval only --
    preds        the 8 surface answers in pool order
    sl           the 8 judge labels in pool order
    inc_scores   the incumbent LoRA verifier's 8 scores (5-dp rounded, as stored)
    greedy_ok    whether the greedy (non-sampled) answer was correct
    slot_rows    slot position -> feature-cache row index (duplicated answers share a row)
    recoverable  1 if any of the 8 slots is correct
    """
    ds: str
    idx: object
    img_md5: Optional[str]
    cands: List[Cand] = field(default_factory=list)
    preds: Optional[List[str]] = None
    sl: Optional[List[int]] = None
    inc_scores: Optional[List[float]] = None
    greedy_ok: Optional[int] = None
    slot_rows: Optional[List[int]] = None

    @property
    def key(self):
        return (self.ds, self.idx)

    @property
    def n_distinct(self) -> int:
        return len(self.cands)

    @property
    def recoverable(self) -> int:
        return int(1 in self.sl) if self.sl is not None else int(any(c.y == 1 for c in self.cands))


@dataclass
class Split:
    """A loaded feature split.

    h_last / h_span   (n_rows, n_layers_selected, 3584) float16, or None if not requested
    yesno             (n_rows, 2) float32 -- ALL NaN when mode == 'generator'
    layers            the layer ids actually materialized, e.g. [21]
    rows              the per-row meta dicts, global (ds, str(idx), na)-sorted order
    questions         list[Question]; for split='eval' this is the CANONICAL item order
    key_to_row        {(ds, idx, na) -> row index}
    """
    split: str
    mode: str
    layers: List[int]
    pooling: Tuple[str, ...]
    h_last: Optional[np.ndarray]
    h_span: Optional[np.ndarray]
    yesno: np.ndarray
    rows: List[dict]
    questions: List[Question]
    key_to_row: Dict[tuple, int]
    meta: dict

    def layer_index(self, layer: int) -> int:
        """Position of `layer` inside the materialized layer axis."""
        return self.layers.index(int(layer))

    def matrix(self, pooling: str = "span", layer: int = 21, dtype=np.float32) -> np.ndarray:
        """(n_rows, 3584) feature matrix for one (pooling, layer)."""
        h = self.h_span if pooling == "span" else self.h_last
        if h is None:
            raise ValueError(f"pooling={pooling!r} was not loaded (loaded: {self.pooling})")
        return h[:, self.layer_index(layer)].astype(dtype)


# ======================================================================================
# 2. the incumbent / ground truth
# ======================================================================================
def load_items() -> List[dict]:
    """The 2345 eval items, raw, in CANONICAL order (slake, then vqa_rad, then pathvqa).

    Each is the transfer-dump record: ds, idx, sl[8], scores[8], pick, greedy_ok, preds[8].
    """
    items = []
    for d in DUMP_ORDER:
        p = os.path.join(DUMP_DIR, f"transfer_dump_{d}_open_lingshu7b.json")
        items.extend(json.load(open(p)))
    return items


def incumbent_scores() -> Dict[tuple, List[float]]:
    """{(ds, idx) -> the incumbent's 8 slot scores}. Feed straight to sel_eff()."""
    return {(it["ds"], it["idx"]): list(it["scores"]) for it in load_items()}


# ======================================================================================
# 3. the feature cache
# ======================================================================================
def _cache_parts(mode: str, split: str, featdir: str) -> List[str]:
    d = featdir if os.path.isabs(featdir) else os.path.join(ROOT, featdir)
    parts = sorted(glob.glob(os.path.join(d, f"{mode}_{split}.npz"))) or \
        sorted(glob.glob(os.path.join(d, f"{mode}_{split}_s*of*.npz")))
    if not parts:
        raise FileNotFoundError(f"no feature cache for {mode}/{split} in {d}")
    return parts


def _reconstruct_order(shard_rows: List[List[dict]]) -> List[int]:
    """Interleave indices so concatenated shards are put back into global sorted order.

    extract_generator_hidden.py sorts rows by (ds, str(idx), na) then takes rows[s::N].
    Row i of shard s is therefore global position i*N + s.
    """
    N = len(shard_rows)
    offs, pos = [], []
    o = 0
    for s, rr in enumerate(shard_rows):
        offs.append(o)
        for i in range(len(rr)):
            pos.append((i * N + s, o + i))
        o += len(rr)
    pos.sort()
    return [j for _, j in pos]


def load_cache(mode: str = "generator", split: str = "eval",
               layers: Optional[Sequence[int]] = None,
               pooling: Sequence[str] = ("last", "span"),
               featdir: str = FEATDIR, order: str = "concat") -> Tuple[dict, List[dict], dict]:
    """Merge the shards of one cache. Returns (arrays, rows, meta).

    layers   list of layer ids to materialize (subset of [7,14,21,28]); None = all,
             [] = none (metadata-only, cheap). Subsetting early is the difference between
             1.8 GB and 0.2 GB of RAM, which matters when several agents run at once.
    pooling  which of h_last / h_span to materialize; () = neither.
    order    'concat' (DEFAULT) shard0-then-shard1, i.e. exactly fit_hidden_head.load_cache
             -- required to reproduce the published bar bit-exact.
             'sorted' the global (ds, str(idx), na) order. See the module docstring: this
             is not cosmetic, it changes the SGD trajectory by ~0.004 sel_eff.
    """
    if order not in ("concat", "sorted"):
        raise ValueError("order must be 'concat' or 'sorted'")
    parts = _cache_parts(mode, split, featdir)
    metas = [json.load(open(p.replace(".npz", ".meta.json"))) for p in parts]
    all_layers = list(int(x) for x in metas[0]["layers"])
    want = all_layers if layers is None else [int(x) for x in layers]
    for L in want:
        if L not in all_layers:
            raise ValueError(f"layer {L} not in cache (has {all_layers})")
    li = [all_layers.index(L) for L in want]

    shard_rows = [m["rows"] for m in metas]
    perm = list(range(sum(len(r) for r in shard_rows))) if order == "concat" \
        else _reconstruct_order(shard_rows)

    out = {"h_last": None, "h_span": None}
    for name in ("h_last", "h_span"):
        if name.split("_")[1] not in pooling:
            continue
        chunks = []
        for p in parts:
            with np.load(p) as z:
                chunks.append(np.ascontiguousarray(z[name][:, li]))
        out[name] = np.concatenate(chunks, 0)[perm]
        del chunks
    yn = []
    for p in parts:
        with np.load(p) as z:
            yn.append(z["yesno"])
    out["yesno"] = np.concatenate(yn, 0)[perm]

    rows = [r for rr in shard_rows for r in rr]
    rows = [rows[i] for i in perm]
    meta = {k: metas[0][k] for k in metas[0] if k != "rows"}
    meta.update({"n": sum(m["n"] for m in metas), "n_failed": sum(m["n_failed"] for m in metas),
                 "n_shards": len(parts), "layers_loaded": want, "pooling_loaded": tuple(pooling),
                 "row_order": order, "files": [os.path.basename(p) for p in parts]})
    return out, rows, meta


def load_candidates(split: str = "eval", mode: str = "generator",
                    layers: Optional[Sequence[int]] = None,
                    pooling: Sequence[str] = ("last", "span"),
                    featdir: str = FEATDIR, order: str = "concat") -> Split:
    """Per-question grouped view of one feature split.

    split='eval'  -> questions are the 2345 transfer-dump items IN CANONICAL ORDER, each
                     carrying preds/sl/inc_scores/greedy_ok/slot_rows plus its distinct
                     candidates. Every one of the 18760 slots resolves to a cache row
                     (verified: 0 gaps, 0 extras, 18760/18760 label agreement).
    split='train' -> questions are the 6029 image-disjoint training questions; candidates
                     carry y and in_draw but there is no 8-slot pool and no incumbent score.

    Raises if any eval slot fails to resolve, so a silent coverage regression is impossible.
    """
    if split not in ("eval", "train"):
        raise ValueError("split must be 'eval' or 'train'")
    arrays, rows, meta = load_cache(mode, split, layers, pooling, featdir, order)
    key_to_row = {}
    for i, r in enumerate(rows):
        k = (r["ds"], r["idx"], r["na"])
        if k in key_to_row:
            raise ValueError(f"duplicate cache key {k}")
        key_to_row[k] = i

    questions: List[Question] = []
    if split == "eval":
        for it in load_items():
            ds, idx = it["ds"], it["idx"]
            slot_rows, seen, cands = [], {}, []
            for s, a in enumerate(it["preds"]):
                na = norm(a)
                k = (ds, idx, na)
                if k not in key_to_row:
                    raise KeyError(f"COVERAGE GAP: no feature row for {k}")
                row = key_to_row[k]
                slot_rows.append(row)
                if na not in seen:
                    seen[na] = len(cands)
                    cands.append(Cand(na=na, text=a, y=int(rows[row]["y"]), row=row, slots=(s,)))
                else:
                    c = cands[seen[na]]
                    c.slots = c.slots + (s,)
                if int(rows[row]["y"]) != int(it["sl"][s]):
                    raise ValueError(f"LABEL MISMATCH at {k}: cache y={rows[row]['y']} dump sl={it['sl'][s]}")
            for c in cands:
                c.mult = len(c.slots)
            questions.append(Question(
                ds=ds, idx=idx, img_md5=rows[slot_rows[0]].get("img_md5"), cands=cands,
                preds=list(it["preds"]), sl=[int(v) for v in it["sl"]],
                inc_scores=[float(v) for v in it["scores"]], greedy_ok=int(it["greedy_ok"]),
                slot_rows=slot_rows))
    else:
        byq: Dict[tuple, List[int]] = defaultdict(list)
        for i, r in enumerate(rows):
            byq[(r["ds"], r["idx"])].append(i)
        for (ds, idx), ii in byq.items():
            cands = [Cand(na=rows[i]["na"], text=rows[i]["ans"], y=int(rows[i]["y"]), row=i,
                          slots=(), mult=0, in_draw=int(rows[i].get("in_draw", 1))) for i in ii]
            questions.append(Question(ds=ds, idx=idx, img_md5=rows[ii[0]].get("img_md5"), cands=cands))

    return Split(split=split, mode=mode, layers=meta["layers_loaded"], pooling=tuple(pooling),
                 h_last=arrays["h_last"], h_span=arrays["h_span"], yesno=arrays["yesno"],
                 rows=rows, questions=questions, key_to_row=key_to_row, meta=meta)


# ======================================================================================
# 4. folds
# ======================================================================================
def eval_folds(n_folds: int = 5, items: Optional[List[dict]] = None) -> np.ndarray:
    """Image-disjoint CV fold id per EVAL item, in canonical item order.

    2345 items share only 528 images, so a fold boundary must be an IMAGE boundary or a
    cross-fitted score is contaminated by its own image. Uses
    data/verifarch/eval_imghash.json (written by verifarch_eval_imghash.py /
    verifarch_assert_disjoint.py) and the project's md5(hash)%k assignment, identical to
    fit_hidden_head.py's protocol.

    Returns int array (n_items,). Any in-eval fitting is a CONTAMINATED DIAGNOSTIC
    (shares images and question distribution with the test set) and must be labelled so.
    """
    items = items if items is not None else load_items()
    eh = json.load(open(EVAL_IMGHASH))
    out = np.empty(len(items), dtype=int)
    for i, it in enumerate(items):
        h = eh[it["ds"]][str(it["idx"])]
        out[i] = int(hashlib.md5(str(h).encode()).hexdigest(), 16) % n_folds
    return out


def train_folds(split_obj: Split, n_folds: int = 5) -> np.ndarray:
    """Image-disjoint fold id per TRAIN ROW (not per question), matching
    fit_hidden_head.py: md5(img_md5) % n_folds. Use this -- and only this -- to
    pre-register a configuration without eval visibility."""
    return np.array([int(hashlib.md5(str(r["img_md5"]).encode()).hexdigest(), 16) % n_folds
                     for r in split_obj.rows], dtype=int)


def group_ids(split_obj: Split) -> List[str]:
    """Per-row question id 'ds|idx' -- the grouping any listwise / pairwise loss needs."""
    return [f"{r['ds']}|{r['idx']}" for r in split_obj.rows]


# ======================================================================================
# 5. disjointness
# ======================================================================================
def assert_disjoint(mode: str = "generator", featdir: str = FEATDIR, verbose: bool = False) -> dict:
    """Re-prove, from the caches themselves, that no eval image is a train image.

    Currency = md5 of DECODED RGB pixels, recorded per row at extraction time. Raises
    AssertionError on any intersection. Also checks (ds, idx) item-level disjointness and
    that no row failed extraction.

    NOTE ON CONVENTIONS: extract_generator_hidden.img_md5 hashes raw RGB bytes;
    verifarch_assert_disjoint.pixhash prefixes 'WxH|'. They are different strings for the
    same image but induce the SAME partition -- verified: both give exactly 528 eval image
    blocks with identical membership. Either is a valid image identity; do not mix them
    inside one comparison.
    """
    _, tr_rows, tr_meta = load_cache(mode, "train", layers=[], pooling=(), featdir=featdir)
    _, ev_rows, ev_meta = load_cache(mode, "eval", layers=[], pooling=(), featdir=featdir)
    tr_h = set(r["img_md5"] for r in tr_rows if r.get("img_md5"))
    ev_h = set(r["img_md5"] for r in ev_rows if r.get("img_md5"))
    inter = tr_h & ev_h
    tr_i = set((r["ds"].replace("_train", ""), r["idx"]) for r in tr_rows)
    ev_i = set((r["ds"], r["idx"]) for r in ev_rows)
    tr_fail = sum(1 for r in tr_rows if r.get("n_tok", -1) <= 0)
    ev_fail = sum(1 for r in ev_rows if r.get("n_tok", -1) <= 0)
    assert len(inter) == 0, f"CONTAMINATION: {len(inter)} shared decoded-RGB image hashes"
    assert tr_fail == 0 and ev_fail == 0, f"extraction failures: train={tr_fail} eval={ev_fail}"
    out = {"mode": mode, "train_rows": len(tr_rows), "eval_rows": len(ev_rows),
           "train_images": len(tr_h), "eval_images": len(ev_h),
           "image_pixel_md5_intersection": 0,
           "train_questions": len(set((r["ds"], r["idx"]) for r in tr_rows)),
           "eval_questions": len(ev_i),
           "item_key_intersection_note": f"{len(tr_i & ev_i)} (ds,idx) pairs collide numerically "
                                         f"across the train/test parquet index spaces; the binding "
                                         f"assertion is the image pixel-md5 intersection == 0",
           "train_rows_failed": tr_fail, "eval_rows_failed": ev_fail,
           "method": "md5 of decoded RGB pixels recorded at extraction time"}
    if verbose:
        print(json.dumps(out, indent=1))
    return out


# ======================================================================================
# 6. THE FROZEN METRIC  -- one definition, four agents
# ======================================================================================
def _slot_scores(scores_by_question, items: List[dict]) -> np.ndarray:
    """Normalize any accepted score container into an (n_items, 8) array.

    Accepted:
      dict {(ds, idx): sequence of 8}          -- per-slot scores
      dict {(ds, idx, na): float}              -- per DISTINCT candidate, broadcast to slots
      array-like (n_items, 8)                  -- in canonical item order
      list of Question objects with .inc_scores (convenience)
    Missing entries become MISSING_SCORE (-1e9), matching fit_hidden_head.py.
    """
    n = len(items)
    if isinstance(scores_by_question, dict):
        ks = next(iter(scores_by_question)) if scores_by_question else None
        per_cand = ks is not None and isinstance(ks, tuple) and len(ks) == 3
        S = np.full((n, 8), MISSING_SCORE, dtype=float)
        for i, it in enumerate(items):
            if per_cand:
                for k, a in enumerate(it["preds"]):
                    v = scores_by_question.get((it["ds"], it["idx"], norm(a)), None)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        S[i, k] = float(v)
            else:
                v = scores_by_question.get((it["ds"], it["idx"]), None)
                if v is not None:
                    vv = np.asarray(v, dtype=float)
                    vv = np.where(np.isnan(vv), MISSING_SCORE, vv)
                    S[i, :len(vv)] = vv
        return S
    S = np.asarray(scores_by_question, dtype=float)
    if S.shape != (n, 8):
        raise ValueError(f"scores array must be ({n}, 8), got {S.shape}")
    return np.where(np.isnan(S), MISSING_SCORE, S)


def picks_from_scores(scores_by_question, items: Optional[List[dict]] = None) -> np.ndarray:
    """argmax over the 8 slots with FIRST-INDEX tie-break -- the frozen pick rule.

    Tie-break matters: the incumbent's stored 'pick' field disagrees with argmax on
    26/2345 items (5-dp score ties) and gives sel_eff 0.774523 instead of 0.775204.
    That +-0.0007 is tie-break sensitivity, not an effect. Always use argmax.
    """
    items = items if items is not None else load_items()
    return np.argmax(_slot_scores(scores_by_question, items), axis=1).astype(int)


def sel_eff(scores_by_question, items: Optional[List[dict]] = None,
            picks: Optional[Sequence[int]] = None) -> dict:
    """THE metric. Pass per-question scores (any accepted form) or explicit picks.

    Returns a dict with, all computed on the SAME 2345 items:
      n, n_recoverable, oracle, greedy, acc (= 'selected'), sel_eff
      per_ds       {ds: {n, n_recoverable, oracle, acc, sel_eff}}   -- the GUARDRAIL:
                   a method that wins pooled but loses on any single set is guardrail-dirty
      contested    {n, sel_eff, unanimous_n, unanimous_sel_eff} -- the more sensitive
                   endpoint, restricted to recoverable items with >= 2 DISTINCT candidate
                   strings; the unanimous stratum is 1.0 for every selector by construction
      picks, got, rec, contested_mask, ds_index   -- arrays, for paired_bootstrap()
    """
    items = items if items is not None else load_items()
    n = len(items)
    picks = np.asarray(picks, dtype=int) if picks is not None else picks_from_scores(scores_by_question, items)
    if picks.shape != (n,):
        raise ValueError(f"picks must have shape ({n},)")
    sl = [it["sl"] for it in items]
    rec = np.array([1 if 1 in x else 0 for x in sl], dtype=int)
    got = np.array([1 if sl[i][picks[i]] == 1 else 0 for i in range(n)], dtype=int)
    nd = np.array([len(set(norm(a) for a in it["preds"])) for it in items], dtype=int)
    greedy = float(np.mean([it["greedy_ok"] for it in items]))
    ds_index = np.array([EVAL_DS.index(it["ds"]) for it in items], dtype=int)

    per_ds = {}
    for j, ds in enumerate(EVAL_DS):
        m = ds_index == j
        per_ds[ds] = {"n": int(m.sum()), "n_recoverable": int(rec[m].sum()),
                      "oracle": float(rec[m].mean()), "acc": float(got[m].mean()),
                      "sel_eff": float(got[m & (rec == 1)].mean())}
    con = (rec == 1) & (nd >= 2)
    una = (rec == 1) & (nd == 1)
    return {"n": n, "n_recoverable": int(rec.sum()),
            "oracle": float(rec.mean()), "greedy": greedy,
            "acc": float(got.mean()), "sel_eff": float(got[rec == 1].mean()),
            "per_ds": per_ds,
            "contested": {"n": int(con.sum()), "sel_eff": float(got[con].mean()),
                          "unanimous_n": int(una.sum()),
                          "unanimous_sel_eff": float(got[una].mean())},
            "picks": picks, "got": got, "rec": rec, "n_distinct": nd,
            "contested_mask": con, "ds_index": ds_index}


def cand_auroc(scores_by_question, items: Optional[List[dict]] = None) -> float:
    """Candidate-level AUROC over all 18760 labelled slots (duplicated answers counted
    once per slot, exactly as in the published cell 0.885592). Tie-aware rank AUC."""
    items = items if items is not None else load_items()
    S = _slot_scores(scores_by_question, items)
    y, s = [], []
    for i, it in enumerate(items):
        for k, v in enumerate(it["sl"]):
            if v >= 0:
                y.append(v); s.append(S[i, k])
    return auroc(y, s)


def auroc(y, s) -> float:
    """Tie-aware Mann-Whitney AUROC. Verbatim behaviour of fit_hidden_head.auroc."""
    y = np.asarray(y, dtype=float); s = np.asarray(s, dtype=float)
    m = ~np.isnan(s)
    y, s = y[m], s[m]
    if y.min() == y.max():
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    ss = s[o]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos, nneg = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def paired_bootstrap(a, b, rec=None, nboot: int = 10000, seed: int = 0, mask=None) -> dict:
    """Paired item-level bootstrap of (a - b), where a, b are the per-item 0/1 'got'
    vectors of two selectors on the SAME items.

    Draws exactly as fit_hidden_head.paired_boot: an accuracy resample over all n items
    and a SEPARATE resample over the recoverable subset for sel_eff (they are not the
    same draw -- keep it that way so numbers stay comparable to published cells).

    mask  optional boolean item mask (e.g. r['contested_mask']) -- when given, the
          sel_eff arm is bootstrapped inside that stratum instead of over rec == 1.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be the same length (paired over items)")
    rng = np.random.default_rng(seed)
    n = len(a)
    sub = np.where(np.asarray(mask, dtype=bool))[0] if mask is not None else \
        np.where(np.asarray(rec, dtype=int) == 1)[0]
    de, da = np.empty(nboot), np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        da[k] = a[s].mean() - b[s].mean()
        j = sub[rng.integers(0, len(sub), len(sub))]
        de[k] = a[j].mean() - b[j].mean()
    return {"d_sel_eff": float(a[sub].mean() - b[sub].mean()),
            "d_sel_eff_ci": [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
            "d_acc": float(a.mean() - b.mean()),
            "d_acc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "nboot": nboot, "seed": seed, "n_items": n, "n_stratum": int(len(sub))}


def guardrail_clean(res: dict, base: dict) -> bool:
    """Project rule: a win must not be a loss on any single eval set."""
    return all(res["per_ds"][d]["sel_eff"] >= base["per_ds"][d]["sel_eff"] for d in EVAL_DS)


# ---------------------------------------------------------------------- rank fusion
def rank_avg(v) -> np.ndarray:
    """Average ranks for ties, scaled to [0,1]. THE canonical ranking convention here.

    Tie handling is load-bearing, not cosmetic: the incumbent's stored scores are 5-dp
    rounded and duplicated answers get identical scores, so a pool of 8 routinely has
    exact ties. Fusing with average ranks gives sel_eff 0.806540; fusing the same two
    score vectors with argsort ranks gives 0.798365. Always say which you used.
    """
    v = np.asarray(v, dtype=float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), dtype=float)
    r[o] = np.arange(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[o[j + 1]] == v[o[i]]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / max(len(v) - 1, 1)


def rank_argsort(v) -> np.ndarray:
    """argsort-of-argsort ranks in [0,1]; breaks ties by position. NOT the default."""
    return np.argsort(np.argsort(np.asarray(v, dtype=float))) / max(len(np.asarray(v)) - 1, 1)


def rank_fuse(*score_dicts, items: Optional[List[dict]] = None,
              ranker=rank_avg) -> Dict[tuple, List[float]]:
    """Parameter-free rank-average fusion of two or more slot-score containers.

    No weight is fit on eval -- that is what makes it reportable. Returns a slot-score
    dict ready for sel_eff(). Default ranker is rank_avg (see its docstring).
    """
    items = items if items is not None else load_items()
    mats = [_slot_scores(s, items) for s in score_dicts]
    out = {}
    for i, it in enumerate(items):
        out[(it["ds"], it["idx"])] = list(sum(ranker(m[i]) for m in mats))
    return out


# ======================================================================================
# 7. controls that everyone should quote against
# ======================================================================================
def control_scores(items: Optional[List[dict]] = None) -> Dict[str, Dict[tuple, List[float]]]:
    """Reference selectors as slot-score dicts: incumbent, self-consistency (plurality of
    the 8 surface strings), and greedy-slot-0. Random-pick has no score vector -- it is a
    closed-form expectation, see random_pick()."""
    items = items if items is not None else load_items()
    inc, sc = {}, {}
    for it in items:
        inc[(it["ds"], it["idx"])] = list(it["scores"])
        cnt = defaultdict(int)
        for a in it["preds"]:
            cnt[norm(a)] += 1
        sc[(it["ds"], it["idx"])] = [float(cnt[norm(a)]) for a in it["preds"]]
    return {"incumbent": inc, "self_consistency": sc}


def random_pick(items: Optional[List[dict]] = None) -> dict:
    """Closed-form expected accuracy / sel_eff of picking a uniformly random slot."""
    items = items if items is not None else load_items()
    rec = np.array([1 if 1 in it["sl"] else 0 for it in items])
    p = np.array([np.mean([1 if v == 1 else 0 for v in it["sl"]]) for it in items], dtype=float)
    return {"acc": float(p.mean()), "sel_eff": float(p[rec == 1].mean())}


def null_test(tol: float = 1e-6) -> dict:
    """Reproduce every published incumbent cell from the transfer dumps with THIS metric
    code. Returns measured / published / abs-deviation and a pass flag."""
    items = load_items()
    r = sel_eff(incumbent_scores(), items)
    a = cand_auroc(incumbent_scores(), items)
    measured = {"n": r["n"], "n_recoverable": r["n_recoverable"], "oracle@8": r["oracle"],
                "selected": r["acc"], "greedy": r["greedy"], "sel_eff": r["sel_eff"],
                "cand_auroc": a,
                "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in EVAL_DS}}
    dev = {k: abs(measured[k] - PUBLISHED[k]) for k in
           ["oracle@8", "selected", "greedy", "sel_eff", "cand_auroc"]}
    dev.update({f"per_ds.{d}": abs(measured["per_ds"][d] - PUBLISHED["per_ds"][d]) for d in EVAL_DS})
    dev["n"] = abs(measured["n"] - PUBLISHED["n"])
    dev["n_recoverable"] = abs(measured["n_recoverable"] - PUBLISHED["n_recoverable"])
    mx = max(dev.values())
    return {"measured": measured, "published": PUBLISHED, "abs_deviation": dev,
            "max_abs_deviation": mx, "pass": bool(mx <= tol),
            "tolerance": tol,
            "note": "published cells are 6-dp rounded; deviations at 1e-6 are rounding, "
                    "not disagreement."}


if __name__ == "__main__":
    import pprint
    nt = null_test()
    print("NULL TEST pass =", nt["pass"], " max abs deviation =", nt["max_abs_deviation"])
    pprint.pprint(nt["measured"])
    print()
    pprint.pprint(assert_disjoint())
