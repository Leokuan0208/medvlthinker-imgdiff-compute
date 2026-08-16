#!/usr/bin/env python3
"""
veto_binning_sweep.py -- KNOB 2: sweep the CERTIFIED VETO's two never-swept defaults,
`n_bins` and `alpha_z`, in src/cascade_methods/beat32b_more.py:98  `f8_veto(d, n_bins=5, alpha_z=1.645)`.

THE SITUATION.  The certified veto partitions a cell's items into `n_bins` quantile bins of 7B top-1
confidence, and certifies a bin when a one-sided Wilson lower bound at `alpha_z` on the 7B's precision
inside that bin exceeds the 32B's point accuracy in the same bin.  Certified bins keep the 7B answer and
NEVER call the 32B -- so a firing veto is cheaper AND (if the certificate holds) at least as accurate.
Both defaults are arbitrary and have never been swept.  On the shipped arm the veto fires on PMC_VQA
only (rate 0.4002, +0.00954) and has veto rate EXACTLY 0.0000 on SLAKE-closed / VQA-RAD-closed /
PathVQA-closed / MedXpert -- four of the five multiple-choice cells, half the 8-cell macro weight,
contributing exactly nothing.

WHAT THIS SCRIPT DOES (NO GPU, no new inference; everything recomputed from the saved MedEvalKit dumps):

  S0  NULL TEST.  Reproduce the PUBLISHED veto bit-for-bit through the published code path
      (beat32b_fusion.mcq + beat32b_more.f8_veto) AND through this file's re-implementation on the
      published fold structure (arange(n) % 5).  Max abs deviation reported.  Also re-derives the
      published 8-cell macro from artifacts/_selector_rerun_parts/vec_disjoint.npz and asserts the
      frozen open-text identity  selected = oracle@8 x sel_eff  (never the additive form).

  S1  FULL GRID, fixed setting (no selection => no leakage).  15 n_bins x 9 alpha_z x 5 MCQ cells.
      Per cell: veto rate, accuracy, delta vs always-32B-direct with a paired item bootstrap
      (nboot=10000), FLOP-eq, guardrail flag.  Published fold structure.

  S2  PMC HONESTY.  The PMC delta for EVERY grid setting is also reported letter-balanced (macro over
      the four gold letters, 1/4 each) and on the gold-A stratum alone -- 44% of the veto's PMC gain is
      attributable to test_2.csv's answer-position skew (B+C = 73.6%, constant-C floor 37.8%) and the
      shipped veto is significantly WORSE on gold-A (-0.0118).  Full bootstrap CIs on a short list.
      Reference: artifacts/pmcvqa_answer_bias_audit_2026-08-11.json.

  S3  NESTED CV of the SELECTION of (n_bins, alpha_z).  Outer 5-fold x 10 seeds; inside each outer
      training block the setting is chosen on an INNER 5-fold cross-fit, then certified on the whole
      outer-train block and evaluated on the untouched outer-test block.  Selection rules:
        R0 SHIPPED      fixed (5, 1.645), veto on PMC only -- run through the SAME outer folds
        R1 GLOBAL       one (n_bins, alpha_z) for all 5 MCQ cells, maximising the summed cell delta
        R2 PER-CELL     per cell, argmax over the grid U {no-veto}   <-- the KNOWN fake-win route
        R3 COST-GUARDED per cell, argmax over settings whose inner veto rate >= 0.21882 (the FLOP
                        break-even: 1.0 + (1-v)*4.57 <= 4.57) and whose inner delta >= 0, else no-veto
        R4 GLOBAL+GUARD one (n_bins, alpha_z), deployed ONLY on cells the inner one-sided McNemar
                        guardrail admits (z > 1.645); the objective sums admitted cells only
        R5 PER-CELL+GUARD  per cell, argmax among settings the inner guardrail admits, else no-veto
      The guardrail is the project's F1 slice-router discipline: never deviate from always-32B-direct
      on a cell unless a held-out test says the deviation wins there.

  S4  PERMUTATION NULL (mandatory).  The paired labels (ok7, ok32) are permuted jointly WITHIN each
      cell -- 7B confidence, fold structure, both marginals and the ok7/ok32 correlation held fixed,
      only the item-level link between confidence and outcome destroyed -- and the ENTIRE S3 pipeline
      is re-run.  This measures directly how much macro a "sweep the veto knobs and keep the best"
      rule manufactures from nothing.  The project has already measured a per-cell pick-the-best rule
      earning +0.0109 macro from shuffled labels alone against +0.0090 on real data (p = 0.67).

  S5  MACRO ASSEMBLY.  The knob touches the 5 multiple-choice cells only, so the 3 open cells are
      carried unchanged from the shipped clean-verifier arm (vec_disjoint.npz).  Macro = equal weight
      per cell, 8 cells at 1/8, Variant B (MMMU excluded), CLEAN disjoint verifier.  Reported vs
      always-32B-direct (0.656672, THE BAR) and vs the shipped accuracy-max arm, plus the MCQ-only
      frame (open half frozen at always-32B-direct) matching armcombine_mcqonly_2026-08-11.json.
      Cost = as-charged FLOP-eq, equal weight per cell, veto cell = 1.0 + (1-v)*4.57.

NUMERICS PINNED: OMP_NUM_THREADS=1, PYTHONHASHSEED=0, numpy recorded, no torch/TF32 involved (pure
CPU numpy), row order = MedEvalKit results.json file order (load-bearing: the published fold map is
arange(n) % 5), paired item bootstrap nboot=10000 (exact sparse multinomial form), seeds pinned below.

Launch from the repo root (CPU only):
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_sweep.py
"""
import collections
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import beat32b_fusion as BF      # noqa: E402  the published MCQ loader
import beat32b_more as BM        # noqa: E402  the published f8_veto

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "veto_binning_2026-08-15.json")
VEC = os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz")

MCQ_CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN_CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
ALL8 = MCQ_CELLS + OPEN_CELLS

N_BINS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64]
ALPHA_Z = [0.0, 0.25, 0.5, 0.842, 1.0, 1.282, 1.645, 1.96, 2.326]
SETTINGS = [(nb, az) for nb in N_BINS for az in ALPHA_Z]
SHIPPED = (5, 1.645)
MIN_N = 30                      # the published per-bin minimum train count guard (beat32b_more.py:114)

K_PUB = 5
O_FOLDS = 5
I_FOLDS = 5
N_SEEDS = 10
N_PERM = 200
NBOOT = 10000
SEED = 20260815

FLOP_7B = 1.0
FLOP_32B = 4.57
BREAKEVEN_V = FLOP_7B / FLOP_32B        # 0.2188...  veto rate at which the arm is FLOP-neutral

# shipped as-charged per-cell FLOPs of the accuracy-max arm, from
# artifacts/_selector_rerun_parts/macro_disjoint.json -> cost.per_cell_as_charged
OPEN_FLOPS_SHIPPED = {"SLAKE_open": 13.971, "VQA_RAD_open": 17.304, "PATH_VQA_open": 10.312}

LETTERS = ["A", "B", "C", "D"]


def r5(x):
    try:
        return round(float(x), 5)
    except (TypeError, ValueError):
        return x


def r4(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


# ===================================================================================================
# core veto (fold map generalised; the label-INDEPENDENT binning is cacheable)
# ===================================================================================================
def wilson_lb_vec(k, n, z):
    """One-sided Wilson lower bound, vectorised.  Same formula as beat32b_more.wilson_lb."""
    n = np.asarray(n, float)
    k = np.asarray(k, float)
    safe = np.maximum(n, 1.0)
    p = k / safe
    d = 1.0 + z * z / safe
    c = (p + z * z / (2 * safe)) / d
    h = z * np.sqrt(p * (1 - p) / safe + z * z / (4 * safe * safe)) / d
    return np.where(n > 0, np.maximum(0.0, c - h), 0.0)


def bin_edges(c7_train, n_bins):
    qs = np.quantile(c7_train, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    qs = np.unique(qs)
    return qs[1:-1], len(qs) - 1


def binning_plan(c7, fold_id, n_bins):
    """Reproduces beat32b_more.f8_veto's partitioning exactly when fold_id == arange(n) % 5."""
    plan = []
    for f in np.unique(fold_id):
        te = fold_id == f
        tr = ~te
        edges, nb = bin_edges(c7[tr], n_bins)
        plan.append(dict(te=np.flatnonzero(te), tr=np.flatnonzero(tr), nb=nb,
                         btr=np.clip(np.digitize(c7[tr], edges), 0, nb - 1),
                         bte=np.clip(np.digitize(c7[te], edges), 0, nb - 1)))
    return plan


def certify(btr, ok7_tr, ok32_tr, nb, alpha_z, min_n=MIN_N):
    n_b = np.bincount(btr, minlength=nb).astype(float)
    k7 = np.bincount(btr, weights=ok7_tr, minlength=nb)
    s32 = np.bincount(btr, weights=ok32_tr, minlength=nb)
    a32 = np.divide(s32, np.maximum(n_b, 1.0))
    return (n_b >= min_n) & (wilson_lb_vec(k7, n_b, alpha_z) >= a32)


def veto_from_plan(plan, ok7, ok32, alpha_z, min_n=MIN_N):
    """Returns (delivered ok, veto flag, per-fold deployment records for the cost model)."""
    out = ok32.copy()
    veto = np.zeros(len(ok32), bool)
    folds = []
    for p in plan:
        cert = certify(p["btr"], ok7[p["tr"]], ok32[p["tr"]], p["nb"], alpha_z, min_n)
        hit = cert[p["bte"]]
        idx = p["te"][hit]
        out[idx] = ok7[idx]
        veto[idx] = True
        folds.append(dict(n=int(len(p["te"])), deployed=bool(cert.any()), veto=float(hit.mean())))
    return out, veto, folds


# ===================================================================================================
# exact sparse paired bootstrap (multinomial counts over the non-zero difference support)
# ===================================================================================================
def _boot_means(d, nboot, rng):
    """nboot bootstrap means of the per-item difference vector d, drawn exactly."""
    n = len(d)
    nz = np.flatnonzero(d)
    if len(nz) == 0:
        return np.zeros(nboot)
    dv = d[nz]
    p = np.full(len(nz) + 1, 1.0 / n)
    p[-1] = (n - len(nz)) / n
    out = np.empty(nboot)
    chunk = max(1, int(4.0e6 // (len(nz) + 1)))
    for s in range(0, nboot, chunk):
        m = min(chunk, nboot - s)
        cnt = rng.multinomial(n, p, size=m)[:, :-1]
        out[s:s + m] = cnt @ dv / n
    return out


def paired_boot(diff, nboot=NBOOT, seed=SEED):
    d = np.asarray(diff, float)
    n = len(d)
    if n == 0:
        return dict(delta=None, lo=None, hi=None, sig=False, n=0, verdict="EMPTY")
    b = _boot_means(d, nboot, np.random.default_rng(seed))
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    return dict(delta=r5(d.mean()), lo=r5(lo), hi=r5(hi), sig=bool(lo > 0 or hi < 0), n=int(n),
                verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


def macro_boot(diffs, nboot=NBOOT, seed=SEED):
    """Macro (equal weight per cell) paired bootstrap: resample items within each cell independently."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(nboot)
    point = 0.0
    for d in diffs:
        d = np.asarray(d, float)
        point += d.mean()
        acc += _boot_means(d, nboot, rng)
    K = len(diffs)
    acc /= K
    point /= K
    lo, hi = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
    return dict(delta=r5(point), lo=r5(lo), hi=r5(hi), sig=bool(lo > 0 or hi < 0), cells=K,
                verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


def strat_macro_boot(diff, strat, keys, nboot=NBOOT, seed=SEED):
    """Macro over strata INSIDE one cell (the PMC letter-balanced delta)."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strat == k) for k in keys]
    groups = [g for g in groups if len(g) > 0]
    point = float(np.mean([d[g].mean() for g in groups]))
    acc = np.zeros(nboot)
    for g in groups:
        acc += _boot_means(d[g], nboot, rng)
    acc /= len(groups)
    lo, hi = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
    return dict(delta=r5(point), lo=r5(lo), hi=r5(hi), sig=bool(lo > 0 or hi < 0),
                n_strata=len(groups), verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


# ===================================================================================================
# data
# ===================================================================================================
def load_cells():
    spec = [("PMC_VQA", lambda: BF.mcq("PMC_VQA")),
            ("SLAKE_closed", lambda: BF.mcq("SLAKE", "SLAKE")),
            ("VQA_RAD_closed", lambda: BF.mcq("VQA_RAD", "YESNO")),
            ("PATH_VQA_closed", lambda: BF.mcq("PATH_VQA", "YESNO")),
            ("MedXpertQA-MM", lambda: BF.mcq("MedXpertQA-MM", None, think_tag="lingshu32b_reason"))]
    cells = {}
    for name, fn in spec:
        d = fn()
        assert d is not None, name
        cells[name] = dict(ok7=np.asarray(d["ok7"], float), ok32=np.asarray(d["ok32"], float),
                           c7=np.asarray(d["c7"], float), n=len(d["ok7"]), raw=d)
    return cells


def flops_of(veto_rate):
    return FLOP_7B + (1.0 - veto_rate) * FLOP_32B


# ===================================================================================================
# nested CV
# ===================================================================================================
def make_folds(n, n_folds, rng):
    a = np.arange(n) % n_folds
    rng.shuffle(a)
    return a


class CellCache:
    """Label-INDEPENDENT cache for one cell + one fold seed: outer folds, inner folds inside every
    outer-train block, and every binning.  Permutations of the labels reuse all of it unchanged."""

    def __init__(self, c7, n, rng):
        self.c7 = c7
        self.n = n
        self.outer = make_folds(n, O_FOLDS, rng)
        self.apply_idx = {o: (np.flatnonzero(self.outer != o), np.flatnonzero(self.outer == o))
                          for o in range(O_FOLDS)}
        self.inner_fold = {o: make_folds(len(self.apply_idx[o][0]), I_FOLDS, rng)
                           for o in range(O_FOLDS)}
        self._inner = {}
        self._apply = {}

    def inner_plan(self, o, n_bins):
        key = (o, n_bins)
        if key not in self._inner:
            tr = self.apply_idx[o][0]
            plan = binning_plan(self.c7[tr], self.inner_fold[o], n_bins)
            for p in plan:
                p["te"] = tr[p["te"]]
                p["tr"] = tr[p["tr"]]
            self._inner[key] = plan
        return self._inner[key]

    def apply_plan(self, o, n_bins):
        key = (o, n_bins)
        if key not in self._apply:
            tr, te = self.apply_idx[o]
            edges, nb = bin_edges(self.c7[tr], n_bins)
            self._apply[key] = dict(tr=tr, te=te, nb=nb,
                                    btr=np.clip(np.digitize(self.c7[tr], edges), 0, nb - 1),
                                    bte=np.clip(np.digitize(self.c7[te], edges), 0, nb - 1))
        return self._apply[key]


GUARD_Z = 1.645     # one-sided ~95% McNemar admission threshold for the inner guardrail


def inner_scores(cache, ok7, ok32, o):
    """Inner-cross-fit statistics for every grid setting on outer-TRAIN.
    Returns setting -> (delta vs always-32B-direct, veto rate, n_win, n_loss)."""
    tr = cache.apply_idx[o][0]
    ntr = len(tr)
    res = {}
    for nb_ in N_BINS:
        plan = cache.inner_plan(o, nb_)
        pre = []
        for p in plan:
            nb = p["nb"]
            n_b = np.bincount(p["btr"], minlength=nb).astype(float)
            k7 = np.bincount(p["btr"], weights=ok7[p["tr"]], minlength=nb)
            s32 = np.bincount(p["btr"], weights=ok32[p["tr"]], minlength=nb)
            a32 = np.divide(s32, np.maximum(n_b, 1.0))
            dte = ok7[p["te"]] - ok32[p["te"]]
            cnt_te = np.bincount(p["bte"], minlength=nb).astype(float)
            win_te = np.bincount(p["bte"], weights=(dte > 0).astype(float), minlength=nb)
            los_te = np.bincount(p["bte"], weights=(dte < 0).astype(float), minlength=nb)
            pre.append((n_b, k7, a32, cnt_te, win_te, los_te))
        for az in ALPHA_Z:
            h = w = l = 0.0
            for n_b, k7, a32, cnt_te, win_te, los_te in pre:
                cert = (n_b >= MIN_N) & (wilson_lb_vec(k7, n_b, az) >= a32)
                h += float(cnt_te[cert].sum())
                w += float(win_te[cert].sum())
                l += float(los_te[cert].sum())
            res[(nb_, az)] = ((w - l) / ntr, h / ntr, w, l)
    return res


def guard_ok(stat):
    """One-sided McNemar admission: the discordant pairs must favour the 7B by > GUARD_Z sigma."""
    _, _, w, l = stat
    if w + l == 0:
        return False
    return (w - l) / np.sqrt(w + l) > GUARD_Z


def apply_setting(cache, ok7, ok32, o, nb_, az):
    ap = cache.apply_plan(o, nb_)
    tr, te = ap["tr"], ap["te"]
    cert = certify(ap["btr"], ok7[tr], ok32[tr], ap["nb"], az)
    hit = cert[ap["bte"]]
    return np.where(hit, ok7[te], ok32[te]), hit, te, bool(cert.any())


RULES = ("R0", "R1", "R2", "R3", "R4", "R5")


def _blank(caches):
    return {c: dict(ok=np.empty(caches[c].n), veto=np.zeros(caches[c].n, bool), picks=[], folds=[])
            for c in MCQ_CELLS}


def _place(store, c, te, ok, hit, deployed, pick):
    store[c]["ok"][te] = ok
    if hit is not None:
        store[c]["veto"][te] = hit
    store[c]["picks"].append(pick)
    store[c]["folds"].append(dict(n=int(len(te)), deployed=bool(deployed),
                                  veto=float(hit.mean()) if hit is not None else 0.0))


def nested_cv(caches, labels):
    out = {r: _blank(caches) for r in RULES}
    for o in range(O_FOLDS):
        inner = {c: inner_scores(caches[c], labels[c][0], labels[c][1], o) for c in MCQ_CELLS}

        # ---- R1 GLOBAL: one setting for all 5 cells ----
        best = max(SETTINGS, key=lambda k: sum(inner[c][k][0] for c in MCQ_CELLS))
        # ---- R4 GLOBAL + inner guardrail: one setting, deployed only on admitted cells ----
        def obj4(k):
            return sum(inner[c][k][0] for c in MCQ_CELLS if guard_ok(inner[c][k]))
        best4 = max(SETTINGS, key=obj4)
        adm4 = [c for c in MCQ_CELLS if guard_ok(inner[c][best4])]

        for c in MCQ_CELLS:
            ok7, ok32 = labels[c]
            tr, te = caches[c].apply_idx[o]

            ok, hit, t1, dep = apply_setting(caches[c], ok7, ok32, o, best[0], best[1])
            _place(out["R1"], c, t1, ok, hit, dep, best)

            if c in adm4:
                ok, hit, t4, dep = apply_setting(caches[c], ok7, ok32, o, best4[0], best4[1])
                _place(out["R4"], c, t4, ok, hit, dep, best4)
            else:
                _place(out["R4"], c, te, ok32[te], None, False, None)

            # ---- R2 PER-CELL (grid U {no-veto}), accuracy only ----
            key, v = max(inner[c].items(), key=lambda kv: kv[1][0])
            if v[0] <= 0:
                _place(out["R2"], c, te, ok32[te], None, False, None)
            else:
                ok, hit, t2, dep = apply_setting(caches[c], ok7, ok32, o, key[0], key[1])
                _place(out["R2"], c, t2, ok, hit, dep, key)

            # ---- R3 COST-GUARDED PER-CELL ----
            elig = [(k, val) for k, val in inner[c].items() if val[1] >= BREAKEVEN_V and val[0] >= 0.0]
            if not elig:
                _place(out["R3"], c, te, ok32[te], None, False, None)
            else:
                key3 = max(elig, key=lambda kv: kv[1][0])[0]
                ok, hit, t3, dep = apply_setting(caches[c], ok7, ok32, o, key3[0], key3[1])
                _place(out["R3"], c, t3, ok, hit, dep, key3)

            # ---- R5 GUARDRAIL-GATED PER-CELL ----
            elig5 = [(k, val) for k, val in inner[c].items() if guard_ok(val)]
            if not elig5:
                _place(out["R5"], c, te, ok32[te], None, False, None)
            else:
                key5 = max(elig5, key=lambda kv: kv[1][0])[0]
                ok, hit, t5, dep = apply_setting(caches[c], ok7, ok32, o, key5[0], key5[1])
                _place(out["R5"], c, t5, ok, hit, dep, key5)

            # ---- R0 SHIPPED (fixed setting, deployed on PMC only) ----
            if c == "PMC_VQA":
                ok, hit, t0, dep = apply_setting(caches[c], ok7, ok32, o, SHIPPED[0], SHIPPED[1])
                _place(out["R0"], c, t0, ok, hit, dep, SHIPPED)
            else:
                _place(out["R0"], c, te, ok32[te], None, False, None)
    return out


def cell_flops(folds):
    """Honest as-charged FLOP-eq for one cell: a fold where the certificate accepted NO bin never runs
    the 7B at all (the certificate is decided on TRAIN, before any test item is served), so it costs
    exactly always-32B-direct.  A fold with at least one certified bin runs the 7B on every item and
    the 32B on the non-certified ones."""
    tot = sum(f["n"] for f in folds)
    if tot == 0:
        return FLOP_32B
    s = 0.0
    for f in folds:
        s += f["n"] * (FLOP_7B + (1.0 - f["veto"]) * FLOP_32B if f["deployed"] else FLOP_32B)
    return s / tot


def seedstat(vals):
    a = np.asarray(vals, float)
    return dict(mean=r5(a.mean()), sd=r5(a.std(ddof=1)) if len(a) > 1 else 0.0,
                range=[r5(a.min()), r5(a.max())], n_seeds=int(len(a)))


# ===================================================================================================
def main():
    t0 = time.time()
    out = {
        "title": "KNOB 2 -- the certified veto's binning (n_bins) and confidence level (alpha_z), swept jointly",
        "date": "2026-08-15",
        "knob": "src/cascade_methods/beat32b_more.py:98  f8_veto(d, n_bins=5, alpha_z=1.645)",
        "no_gpu": True,
        "no_fabricated_numbers": True,
        "reproduce": "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_sweep.py",
        "numerics_pinned": dict(
            OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset"),
            PYTHONHASHSEED=os.environ.get("PYTHONHASHSEED", "unset"),
            numpy=np.__version__, python=sys.version.split()[0],
            torch_TF32="not applicable -- pure CPU numpy, torch is never imported",
            row_order="MedEvalKit results.json file order; the PUBLISHED fold map is arange(n) % 5, so "
                      "row order is load-bearing for S0/S1 (standing caveat: row order +0.0041)",
            ranker="np.quantile(linear interpolation) + np.digitize on interior edges -- identical to f8_veto",
            bootstrap="exact paired item bootstrap via multinomial counts on the non-zero difference support",
            nboot=NBOOT, seed=SEED, n_seeds=N_SEEDS, n_perm=N_PERM,
            outer_folds=O_FOLDS, inner_folds=I_FOLDS, min_bin_train_count=MIN_N),
        "grid": dict(n_bins=N_BINS, alpha_z=ALPHA_Z, n_settings=len(SETTINGS), shipped=list(SHIPPED),
                     note="alpha_z = 0.0 is the loosest still-meaningful certificate (the 7B's POINT "
                          "precision must beat the 32B's); negative z was not swept because it would "
                          "certify bins where the point estimate already says the 7B is worse, which is "
                          "no longer a certificate."),
        "conventions": dict(
            macro="equal weight per reporting cell, 8 cells at 1/8, Variant B (MMMU excluded), CLEAN "
                  "disjoint verifier",
            bar_always_32b_direct=0.656672,
            shipped_accuracy_max=0.6575,
            significant_macro_delta_needs="+0.0029 (the CI half-width) == summed per-cell gain +0.0235",
            cost="as-charged FLOP-eq; a veto cell costs 1.0 + (1-v)*4.57 -- the 7B runs on EVERY item to "
                 "decide the veto, certified items never call the 32B. Break-even veto rate = 1.0/4.57 = "
                 "%.5f; BELOW that the veto arm is MORE expensive than always-32B-direct." % BREAKEVEN_V),
    }

    cells = load_cells()
    zv = np.load(VEC)
    open_acc_shipped = {c: float(zv[f"{c}|method_accuracy_max_veto"].mean()) for c in OPEN_CELLS}
    open_acc_direct = {c: float(zv[f"{c}|always_32b_direct"].mean()) for c in OPEN_CELLS}
    open_vec_shipped = {c: zv[f"{c}|method_accuracy_max_veto"].astype(float) for c in OPEN_CELLS}
    open_vec_direct = {c: zv[f"{c}|always_32b_direct"].astype(float) for c in OPEN_CELLS}
    shipped_mcq_vec = {c: zv[f"{c}|method_accuracy_max_veto"].astype(float) for c in MCQ_CELLS}

    # ============================== S0  NULL TEST =================================================
    null = {}
    ok_pub, v_pub = BM.f8_veto(cells["PMC_VQA"]["raw"])
    ref = json.load(open(os.path.join(ART, "pmc_label_noise_audit_2026-07-29.json")))["reproduction"]
    got = dict(acc_7b=float(cells["PMC_VQA"]["ok7"].mean()),
               acc_32b_nt=float(cells["PMC_VQA"]["ok32"].mean()),
               veto_acc=float(ok_pub.mean()),
               veto_delta=float(ok_pub.mean() - cells["PMC_VQA"]["ok32"].mean()),
               veto_rate=float(v_pub.mean()),
               n_win_veto=int((ok_pub > cells["PMC_VQA"]["ok32"]).sum()),
               n_loss_veto=int((ok_pub < cells["PMC_VQA"]["ok32"]).sum()),
               n_pmc=int(cells["PMC_VQA"]["n"]))
    dev_a = {k: abs(round(v, 4) - ref[k]) for k, v in got.items() if k in ref}
    null["A_published_code_path_PMC"] = dict(
        source="artifacts/pmc_label_noise_audit_2026-07-29.json -> reproduction",
        recomputed={k: r5(v) for k, v in got.items()}, published={k: ref[k] for k in got if k in ref},
        per_field_abs_deviation={k: r5(v) for k, v in dev_a.items()},
        max_abs_deviation=r5(max(dev_a.values())), passed=bool(max(dev_a.values()) < 1e-4))

    dev_b = {}
    for c in MCQ_CELLS:
        C = cells[c]
        plan = binning_plan(C["c7"], np.arange(C["n"]) % K_PUB, SHIPPED[0])
        ok_mine, v_mine, _f = veto_from_plan(plan, C["ok7"], C["ok32"], SHIPPED[1])
        ok_ref, v_ref = BM.f8_veto(C["raw"])
        dev_b[c] = dict(max_abs_dev_ok=r5(float(np.abs(ok_mine - ok_ref).max())),
                        max_abs_dev_veto=r5(float(np.abs(v_mine.astype(float) - v_ref.astype(float)).max())),
                        veto_rate=r5(v_mine.mean()), delta=r5(ok_mine.mean() - C["ok32"].mean()),
                        matches_frozen_eval_vector=bool(
                            np.array_equal(ok_mine.astype(np.int8), shipped_mcq_vec[c].astype(np.int8))
                            if c == "PMC_VQA" else
                            np.array_equal(C["ok32"].astype(np.int8), shipped_mcq_vec[c].astype(np.int8))))
    mx_b = max(max(v["max_abs_dev_ok"], v["max_abs_dev_veto"]) for v in dev_b.values())
    null["B_reimplementation_vs_f8_veto_all5_cells"] = dict(
        what="this file's binning_plan/veto_from_plan on fold_id = arange(n) % 5 must be BYTE-IDENTICAL "
             "to beat32b_more.f8_veto on every MCQ cell, and the PMC vector must equal the frozen "
             "vec_disjoint.npz 'method_accuracy_max_veto' column",
        per_cell=dev_b, max_abs_deviation=r5(mx_b),
        all_cells_match_frozen_vectors=bool(all(v["matches_frozen_eval_vector"] for v in dev_b.values())),
        passed=bool(mx_b == 0.0 and all(v["matches_frozen_eval_vector"] for v in dev_b.values())))

    macro_ship = float(np.mean([zv[f"{c}|method_accuracy_max_veto"].mean() for c in ALL8]))
    macro_bar = float(np.mean([zv[f"{c}|always_32b_direct"].mean() for c in ALL8]))
    null["C_macro_reconstruction"] = dict(
        source="artifacts/_selector_rerun_parts/vec_disjoint.npz",
        shipped_accuracy_max_macro=r5(macro_ship), published_value=0.6575,
        abs_deviation_shipped=r5(abs(round(macro_ship, 4) - 0.6575)),
        always_32b_direct_macro=r5(macro_bar), published_bar=0.656672,
        abs_deviation_bar=r5(abs(macro_bar - 0.656672)),
        passed=bool(abs(round(macro_ship, 4) - 0.6575) < 1e-4 and abs(macro_bar - 0.656672) < 1e-5))

    sel_eff, oracle8, greedy = 0.775204, 0.626013, 0.449467
    add_form = greedy + sel_eff * (oracle8 - greedy)
    null["D_frozen_open_text_identity"] = dict(
        source_of_definition="src/training_methods/genframe_data.py (sel_eff 0.775204, oracle@8 0.626013, "
                             "greedy 0.449467, n=2345/1468)",
        identity="selected = oracle@8 x sel_eff",
        selected_multiplicative=r5(oracle8 * sel_eff),
        additive_form_FORBIDDEN=r5(add_form),
        additive_over_prediction=r5(add_form - oracle8 * sel_eff),
        asserted=True,
        note="Asserted, not used: this knob changes only the multiple-choice half, so the open-text "
             "selection metric is untouched and every open cell is carried byte-identical.")

    # (e) the COST model must reproduce the frozen per-cell and macro FLOPs of the shipped arm
    C = cells["PMC_VQA"]
    _, _, fds_pmc = veto_from_plan(binning_plan(C["c7"], np.arange(C["n"]) % K_PUB, SHIPPED[0]),
                                   C["ok7"], C["ok32"], SHIPPED[1])
    pmc_flops = cell_flops(fds_pmc)
    ship_cellflops = ([pmc_flops] + [FLOP_32B] * 4 + [OPEN_FLOPS_SHIPPED[c] for c in OPEN_CELLS])
    ship_macro_flops = float(np.mean(ship_cellflops))
    null["E_cost_model_reconstruction"] = dict(
        source="artifacts/_selector_rerun_parts/macro_disjoint.json -> cost.per_cell_as_charged / "
               "cost.as_charged.macro_cells.method_accuracy_max_veto",
        pmc_cell_flops_recomputed=r4(pmc_flops), pmc_cell_flops_published=3.741,
        macro_flops_recomputed=r4(ship_macro_flops), macro_flops_published=7.951,
        x_direct_recomputed=r4(ship_macro_flops / FLOP_32B), x_direct_published=1.7398,
        abs_deviation=r5(abs(round(ship_macro_flops, 3) - 7.951)),
        passed=bool(abs(round(ship_macro_flops, 3) - 7.951) < 1e-3),
        note="Confirms the cost convention used everywhere below: a cell whose certificate is empty "
             "costs 4.57 (always-32B-direct, the 7B is never run), NOT 5.57.")

    null["PASSED"] = bool(null["A_published_code_path_PMC"]["passed"]
                          and null["B_reimplementation_vs_f8_veto_all5_cells"]["passed"]
                          and null["C_macro_reconstruction"]["passed"]
                          and null["E_cost_model_reconstruction"]["passed"])
    null["max_abs_deviation_overall"] = r5(max(
        null["A_published_code_path_PMC"]["max_abs_deviation"],
        null["B_reimplementation_vs_f8_veto_all5_cells"]["max_abs_deviation"],
        null["C_macro_reconstruction"]["abs_deviation_shipped"],
        null["C_macro_reconstruction"]["abs_deviation_bar"],
        null["E_cost_model_reconstruction"]["abs_deviation"]))
    out["S0_NULL_TEST"] = null
    print(f"[S0] null test passed={null['PASSED']}  max abs dev={null['max_abs_deviation_overall']}")
    if not null["PASSED"]:
        json.dump(out, open(OUT, "w"), indent=2, default=str)
        raise SystemExit("NULL TEST FAILED -- stopping before any sweep result is produced.")

    # ============================== S1 + S2  FULL GRID ============================================
    print("[S1] grid ...")
    r7 = BF.load_raw("lingshu7b_full", "PMC_VQA")
    gold = np.array([str(r["answer"]).strip() for r in r7[:cells["PMC_VQA"]["n"]]])
    assert set(np.unique(gold)) <= set(LETTERS), np.unique(gold)

    grid = {c: {} for c in MCQ_CELLS}
    pmc_letter = {}
    for c in MCQ_CELLS:
        C = cells[c]
        fold = np.arange(C["n"]) % K_PUB
        for nb_ in N_BINS:
            plan = binning_plan(C["c7"], fold, nb_)
            for az in ALPHA_Z:
                ok, v, fds = veto_from_plan(plan, C["ok7"], C["ok32"], az)
                diff = ok - C["ok32"]
                bs = paired_boot(diff)
                vr = float(v.mean())
                grid[c][f"{nb_}|{az}"] = dict(
                    veto_rate=r5(vr), acc=r5(ok.mean()), acc_32b_direct=r5(C["ok32"].mean()),
                    delta=bs["delta"], ci=[bs["lo"], bs["hi"]], sig=bs["sig"], verdict=bs["verdict"],
                    flops=r4(cell_flops(fds)), x_direct=r4(cell_flops(fds) / FLOP_32B),
                    n_folds_deployed=int(sum(f["deployed"] for f in fds)),
                    cheaper_than_direct=bool(vr >= BREAKEVEN_V),
                    n_win=int((ok > C["ok32"]).sum()), n_loss=int((ok < C["ok32"]).sum()))
                if c == "PMC_VQA":
                    pmc_letter[f"{nb_}|{az}"] = dict(
                        veto_rate=r5(vr), raw_delta=bs["delta"], raw_ci=[bs["lo"], bs["hi"]],
                        raw_verdict=bs["verdict"],
                        letter_balanced_delta_point=r5(np.mean([diff[gold == L].mean() for L in LETTERS])),
                        goldA_delta_point=r5(diff[gold == "A"].mean()))
    out["S1_grid_fixed_settings"] = dict(
        protocol="every setting evaluated on its OWN 5-fold cross-fit with the PUBLISHED fold map "
                 "(arange(n) % 5). No selection happens here, so a PRE-SPECIFIED row is leakage-free -- "
                 "the numbers become optimistic the instant you pick the best row (that is what S3/S4 fix).",
        per_cell=grid)
    print(f"[S1] done  {time.time()-t0:.0f}s")

    fires = {}
    for c in MCQ_CELLS:
        f = [(k, v["veto_rate"], v["delta"], v["verdict"], v["cheaper_than_direct"])
             for k, v in grid[c].items() if v["veto_rate"] > 0]
        f.sort(key=lambda t: -t[1])
        best = max(f, key=lambda t: t[2]) if f else None
        fires[c] = dict(n_settings_with_nonzero_veto=len(f), n_settings=len(grid[c]),
                        max_veto_rate=r5(max([x[1] for x in f], default=0.0)),
                        top8_by_veto_rate=[dict(setting=x[0], veto_rate=x[1], delta=x[2], verdict=x[3],
                                                cheaper=x[4]) for x in f[:8]],
                        best_delta_in_sample=(best[2] if best else 0.0),
                        best_delta_setting=(best[0] if best else None),
                        n_settings_sig_loss=sum(1 for v in grid[c].values() if v["verdict"] == "LOSS"),
                        n_settings_sig_win=sum(1 for v in grid[c].values() if v["verdict"] == "WIN"))
    out["S1_summary_does_the_veto_ever_fire"] = fires

    # ---- S2 short-list bootstraps ----
    shortlist = {f"{SHIPPED[0]}|{SHIPPED[1]}": "SHIPPED"}
    if fires["PMC_VQA"]["best_delta_setting"]:
        shortlist[fires["PMC_VQA"]["best_delta_setting"]] = "PMC best in-sample delta"
    if fires["PMC_VQA"]["top8_by_veto_rate"]:
        shortlist[fires["PMC_VQA"]["top8_by_veto_rate"][0]["setting"]] = "PMC max veto rate"
    s2 = {}
    C = cells["PMC_VQA"]
    for key, lab in shortlist.items():
        nb_, az = key.split("|")
        plan = binning_plan(C["c7"], np.arange(C["n"]) % K_PUB, int(nb_))
        ok, v, _f = veto_from_plan(plan, C["ok7"], C["ok32"], float(az))
        diff = ok - C["ok32"]
        lb = strat_macro_boot(diff, gold, LETTERS)
        gA = paired_boot(diff[gold == "A"], seed=SEED + 3)
        raw = paired_boot(diff, seed=SEED + 4)
        s2[key] = dict(label=lab, veto_rate=r5(float(v.mean())),
                       raw=dict(delta=raw["delta"], ci=[raw["lo"], raw["hi"]], verdict=raw["verdict"]),
                       letter_balanced=dict(delta=lb["delta"], ci=[lb["lo"], lb["hi"]], verdict=lb["verdict"]),
                       gold_A_stratum=dict(delta=gA["delta"], ci=[gA["lo"], gA["hi"]],
                                           verdict=gA["verdict"], n=gA["n"]))
    out["S2_PMC_letter_honesty"] = dict(
        why="44% of the veto's PMC gain is attributable to test_2.csv's answer-position skew (B+C = 73.6%, "
            "constant-C floor 37.8%); the shipped veto is significantly WORSE on the gold-A stratum "
            "(-0.0118). A setting that fires more on PMC while hurting gold-A more is NOT an improvement.",
        reference="artifacts/pmcvqa_answer_bias_audit_2026-08-11.json -- T5 letter-balanced +0.00534 "
                  "[0.00276, 0.00793]; T4 gold-A -0.01176 [-0.01809, -0.00543]",
        point_estimates_every_setting=pmc_letter,
        bootstrapped_shortlist=s2)

    # ============================== S3  NESTED CV =================================================
    print("[S3] nested CV ...")
    labels_real = {c: (cells[c]["ok7"], cells[c]["ok32"]) for c in MCQ_CELLS}
    seed_results, caches_by_seed = [], []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(SEED + 1000 * s)
        caches = {c: CellCache(cells[c]["c7"], cells[c]["n"], rng) for c in MCQ_CELLS}
        caches_by_seed.append(caches)
        res = nested_cv(caches, labels_real)
        rec = {}
        for r, per in res.items():
            mcq_acc = {c: float(per[c]["ok"].mean()) for c in MCQ_CELLS}
            rec[r] = dict(mcq_acc=mcq_acc,
                          veto_rate={c: float(per[c]["veto"].mean()) for c in MCQ_CELLS},
                          flops={c: cell_flops(per[c]["folds"]) for c in MCQ_CELLS},
                          deploy_rate={c: float(np.mean([f["deployed"] for f in per[c]["folds"]]))
                                       for c in MCQ_CELLS},
                          macro=float(sum(mcq_acc.values()) + sum(open_acc_shipped.values())) / 8.0,
                          macro_mcqonly=float(sum(mcq_acc.values()) + sum(open_acc_direct.values())) / 8.0,
                          picks={c: [None if p is None else list(p) for p in per[c]["picks"]]
                                 for c in MCQ_CELLS},
                          ok={c: per[c]["ok"] for c in MCQ_CELLS})
        seed_results.append(rec)
        print("   seed %d: " % s + "  ".join(f"{r} {rec[r]['macro']:.5f}" for r in RULES)
              + f"   ({time.time()-t0:.0f}s)")

    s3 = {}
    for r in RULES:
        avg_ok = {c: np.mean([sr[r]["ok"][c] for sr in seed_results], axis=0) for c in MCQ_CELLS}
        diffs_bar = ([avg_ok[c] - cells[c]["ok32"] for c in MCQ_CELLS]
                     + [open_vec_shipped[c] - open_vec_direct[c] for c in OPEN_CELLS])
        diffs_ship = ([avg_ok[c] - shipped_mcq_vec[c] for c in MCQ_CELLS]
                      + [np.zeros(len(open_vec_shipped[c])) for c in OPEN_CELLS])
        diffs_mcqonly = [avg_ok[c] - cells[c]["ok32"] for c in MCQ_CELLS] + \
                        [np.zeros(len(open_vec_direct[c])) for c in OPEN_CELLS]
        percell = {}
        for c in MCQ_CELLS:
            bs = paired_boot(avg_ok[c] - cells[c]["ok32"], seed=SEED + 13)
            vrs = [sr[r]["veto_rate"][c] for sr in seed_results]
            fl = float(np.mean([sr[r]["flops"][c] for sr in seed_results]))
            percell[c] = dict(acc=seedstat([sr[r]["mcq_acc"][c] for sr in seed_results]),
                              veto_rate=seedstat(vrs),
                              deploy_rate=seedstat([sr[r]["deploy_rate"][c] for sr in seed_results]),
                              delta_vs_direct=bs["delta"], ci=[bs["lo"], bs["hi"]], verdict=bs["verdict"],
                              flops=r4(fl), x_direct=r4(fl / FLOP_32B))
        for c in OPEN_CELLS:
            percell[c] = dict(acc=dict(mean=r5(open_acc_shipped[c]), sd=0.0,
                                       range=[r5(open_acc_shipped[c])] * 2, n_seeds=N_SEEDS),
                              veto_rate=dict(mean=0.0),
                              delta_vs_direct=r5(open_acc_shipped[c] - open_acc_direct[c]),
                              ci=[None, None], verdict="CARRIED (unchanged by this knob)",
                              flops=OPEN_FLOPS_SHIPPED[c],
                              x_direct=r4(OPEN_FLOPS_SHIPPED[c] / FLOP_32B))
        mflops = float(np.mean([percell[c]["flops"] for c in ALL8]))
        mflops_mcq = float(np.mean([percell[c]["flops"] for c in MCQ_CELLS] + [FLOP_32B] * 3))
        picks = collections.Counter()
        for sr in seed_results:
            for c in MCQ_CELLS:
                for p in sr[r]["picks"][c]:
                    picks[f"{c}|{'none' if p is None else str(p[0]) + ',' + str(p[1])}"] += 1
        s3[r] = dict(
            macro_seedstat=seedstat([sr[r]["macro"] for sr in seed_results]),
            macro_mcqonly_seedstat=seedstat([sr[r]["macro_mcqonly"] for sr in seed_results]),
            macro_vs_always_32b_direct=macro_boot(diffs_bar),
            macro_vs_shipped_accuracy_max=macro_boot(diffs_ship, seed=SEED + 7),
            macro_vs_always_32b_direct_mcqonly_frame=macro_boot(diffs_mcqonly, seed=SEED + 11),
            per_cell=percell,
            macro_flops_as_charged=r4(mflops), macro_x_direct=r4(mflops / FLOP_32B),
            macro_flops_mcqonly_frame=r4(mflops_mcq), macro_x_direct_mcqonly_frame=r4(mflops_mcq / FLOP_32B),
            guardrail_flags_sig_loss=[c for c in MCQ_CELLS if percell[c]["verdict"] == "LOSS"],
            guardrail_flags_point_negative=[c for c in MCQ_CELLS
                                            if percell[c]["delta_vs_direct"] is not None
                                            and percell[c]["delta_vs_direct"] < 0],
            selected_settings_frequency=dict(sorted(picks.items(), key=lambda t: -t[1])))
    out["S3_nested_cv"] = dict(
        protocol=dict(
            outer_folds=O_FOLDS, inner_folds=I_FOLDS, seeds=N_SEEDS,
            what="For each outer fold, every grid setting is scored by an INNER 5-fold cross-fit that only "
                 "ever sees outer-TRAIN items; the winning setting is then certified on the whole outer-TRAIN "
                 "block and delivered on the untouched outer-TEST block. No evaluation item ever influences "
                 "the setting that scores it. Seeds re-draw the outer AND inner fold assignments.",
            per_cell_ci_note="the per-cell / macro CIs are item bootstraps of the SEED-AVERAGED delivered "
                             "vector; fold-assignment variance is reported separately as the seed sd.",
            rules=dict(
                R0="SHIPPED fixed (n_bins=5, alpha_z=1.645), veto deployed on PMC only -- run through the "
                   "SAME random outer folds so the contrast is like-for-like, not published-folds-vs-random",
                R1="GLOBAL: one (n_bins, alpha_z) for all 5 MCQ cells, argmax of the summed cell delta",
                R2="PER-CELL: argmax over the 135-setting grid U {no-veto} -- the KNOWN fake-win route",
                R3="COST-GUARDED PER-CELL: argmax among settings whose inner veto rate >= %.5f (FLOP "
                   "break-even) and whose inner delta >= 0, else no-veto" % BREAKEVEN_V,
                R4="GLOBAL + GUARDRAIL: one (n_bins, alpha_z) deployed only on the cells an inner "
                   "one-sided McNemar test admits (z > %.3f); the selection objective sums admitted "
                   "cells only. This is the project's F1 slice-router discipline applied to the knob."
                   % GUARD_Z,
                R5="PER-CELL + GUARDRAIL: per cell, argmax among settings the inner McNemar guardrail "
                   "admits, else no-veto"),
            cost_model="a fold whose certificate accepted NO bin never runs the 7B at all (the "
                       "certificate is decided on TRAIN before any test item is served) and costs "
                       "exactly always-32B-direct (4.57); a fold with >=1 certified bin costs "
                       "1.0 + (1-v)*4.57. This is what makes the shipped arm 7.951 macro FLOPs and "
                       "not 8.45 -- see S0 null test E."),
        results=s3)

    # ============================== S4  PERMUTATION NULL ==========================================
    print("[S4] permutation null ...")
    perm_rng = np.random.default_rng(SEED + 99)
    n_perm_seeds = 3
    null_rows = {r: [] for r in RULES}
    for ip in range(N_PERM):
        labels_p = {}
        for c in MCQ_CELLS:
            idx = perm_rng.permutation(cells[c]["n"])
            labels_p[c] = (cells[c]["ok7"][idx], cells[c]["ok32"][idx])
        per_seed = {r: [] for r in RULES}
        for s in range(n_perm_seeds):
            res = nested_cv(caches_by_seed[s], labels_p)
            for r, per in res.items():
                d = sum(float(per[c]["ok"].mean()) - float(labels_p[c][1].mean())
                        for c in MCQ_CELLS) / 8.0
                per_seed[r].append(d)
        for r in RULES:
            null_rows[r].append(float(np.mean(per_seed[r])))
        if (ip + 1) % 25 == 0:
            print(f"   perm {ip+1}/{N_PERM}  ({time.time()-t0:.0f}s)")

    obs = {r: float(np.mean([sum(sr[r]["mcq_acc"][c] - float(cells[c]["ok32"].mean())
                                 for c in MCQ_CELLS) / 8.0 for sr in seed_results[:n_perm_seeds]]))
           for r in RULES}
    s4 = {}
    for r in RULES:
        a = np.asarray(null_rows[r], float)
        sd = a.std(ddof=1)
        s4[r] = dict(observed_mcq_macro_contribution=r5(obs[r]),
                     null_mean=r5(a.mean()), null_sd=r5(sd), null_min=r5(a.min()), null_max=r5(a.max()),
                     null_p95=r5(np.percentile(a, 95.0)), null_p975=r5(np.percentile(a, 97.5)),
                     p_value_one_sided=r5((float((a >= obs[r]).sum()) + 1) / (len(a) + 1)),
                     z=(r5((obs[r] - a.mean()) / sd) if sd > 0 else None),
                     n_perm=int(len(a)), beats_null_p975=bool(obs[r] > np.percentile(a, 97.5)))
    out["S4_permutation_null"] = dict(
        design="The paired labels (ok7, ok32) are permuted JOINTLY within each cell: 7B confidence, the fold "
               "structure, both accuracy marginals and the ok7/ok32 correlation are all preserved; only the "
               "item-level link between confidence and outcome is destroyed. The ENTIRE nested-CV selection "
               "pipeline (all four rules) is re-run on the permuted labels. Statistic = the MCQ-side "
               "contribution to the 8-cell macro = sum over the 5 MCQ cells of (arm acc - always-32B-direct "
               "acc), divided by 8. Because the open half is untouched by this knob, this statistic IS the "
               "whole macro delta of the arm.",
        why_mandatory="This project has already measured a per-cell pick-the-best rule earning +0.0109 macro "
                      "from SHUFFLED LABELS ALONE against +0.0090 on real data (p = 0.67).",
        n_perm=N_PERM, fold_seeds_averaged_per_perm=n_perm_seeds, results=s4)
    for r in RULES:
        print(f"   {r}: obs {s4[r]['observed_mcq_macro_contribution']:+.5f}  null "
              f"{s4[r]['null_mean']:+.5f} +/- {s4[r]['null_sd']:.5f}  p={s4[r]['p_value_one_sided']}")

    # ============================== S5  selected settings + PMC honesty ===========================
    sel = {}
    for r in ("R1", "R2", "R3", "R4", "R5"):
        cnt = collections.Counter()
        for sr in seed_results:
            for p in sr[r]["picks"]["PMC_VQA"]:
                if p:
                    cnt[f"{p[0]}|{p[1]}"] += 1
        sel[r] = cnt.most_common(5)
    out["S5_selected_settings_pmc_honesty"] = dict(
        modal_pmc_settings={r: [[k, int(v)] for k, v in v5] for r, v5 in sel.items()},
        letter_point_estimate_at_modal_setting={r: (pmc_letter.get(v5[0][0]) if v5 else None)
                                                for r, v5 in sel.items()},
        shipped_setting_letter_audit=s2.get(f"{SHIPPED[0]}|{SHIPPED[1]}"))

    out["runtime_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(f"\nwrote {OUT}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
