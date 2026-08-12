#!/usr/bin/env python3
"""
ATTACK C / PART 4 -- THE PRE-GENERATION ROUTER FRONTIER POINT.

WHY.  LITERATURE_UPDATE_2026-08-11.md reports a published structural theorem: a pre-generation
router beat the best cascade on 4/5 datasets "mainly because it avoids the cheap model's generation
cost -- cascades pay the cheap model before any escalation decision."  That is EXACTLY this
project's macro cost reversal (CLAUDE.md section 0: under equal weight NO operating point is
compute-cheaper than always-32B-direct).  A pre-generation router is the one architecture whose
cost can go below the cascade's floor, and it costs nothing to evaluate offline.

THE ARITHMETIC THAT MOTIVATES IT.  At routing rate e:
    cascade cost   = FLOP_7B + e * FLOP_32B          (the 7B always runs first)
    router cost    = (1-e) * FLOP_7B + e * FLOP_32B  (only ONE model ever runs)
so the router is cheaper by e * FLOP_7B at every e -- but it must decide WITHOUT having seen the
7B's answer or its confidence, which is the cascade's whole advantage.  Measuring that trade is
the deliverable.

METHOD (pre-registered in stats_recertification_2026-08-11_preregistration.json).
  features   PRIMARY: prompt text only (word 1-2 gram + char_wb 3-5 gram TF-IDF -> 256-d LSA,
             label-free) + question length.  SECONDARY (labelled an upper bound): + one-hot cell.
  target     the routing gain g = y32 - y7 in {-1, 0, +1}
  estimator  ridge, 5-fold CROSS-FITTED, folds GROUPED BY IMAGE CLUSTER so no image is ever in
             both a fit and its own prediction; rows weighted 1/(8*n_cell) so the fit optimises
             the MACRO objective rather than the PMC-VQA-dominated pooled one
  frontier   sweep the routing threshold; report macro accuracy vs macro FLOP-eq cost
  null       shuffle g WITHIN cell (preserving every cell's base rate) and refit the entire
             cross-fitted pipeline.  Because within-cell shuffling PRESERVES each cell's mean
             gain, a null frontier that matches the real one proves the router's value is entirely
             CROSS-cell (it has learned which benchmark it is looking at) and carries no
             per-question signal.  That distinction is the point of the test.

Ridge is refit under permutation in closed form: (X'X + lambda I) is fixed per fold, so only X'y
changes.  1000+ refits are therefore free.

NULL TEST.  The ORACLE router (route each item to whichever model is right) must reproduce the
brief's free upper bound of +0.0661 macro over always-32B-direct.

Launch from the repo root:  python3 src/cascade_methods/stats_recert_p4_router.py
Writes results/cascade_methods/artifacts/_stats_recert/part4_router.json
"""
import json
import os
import sys

# NUMERICS: the thread count is a known +-0.0048 axis in this project and another research round is
# sharing the box, so it is PINNED here (before numpy/BLAS is imported) and recorded in the artifact.
N_THREADS = os.environ.get("STATS_RECERT_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = N_THREADS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_recert_common import (ART, CELLS, FLOP_32B, FLOP_32B_HONEST, FLOP_7B, META, NBOOT, SEED,
                                 ci, cluster_ids, jdump, load_meta, load_vec)

W = 1.0 / len(CELLS)
NPERM = 200
NSVD = 256
FOLDS = 5

# Shipped operating points, verbatim from
# results/cascade_methods/artifacts/_selector_rerun_parts/summary_disjoint.json
SHIPPED = {"always_7b": (0.5971, 1.0), "always_32b_direct": (0.6567, 4.57),
           "always_32b_reasoning": (0.5974, 4.57),
           "method_compute_lean": (0.6443, 6.674), "method_accuracy_max_veto": (0.6575, 7.951),
           "method_accuracy_max_fusion": (0.6503, 7.766)}


def build_features(cellid_onehot=False):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union

    qs, cid, grp = [], [], []
    for k, c in enumerate(CELLS):
        m = load_meta(c)
        qs.extend(m["question"])
        cid.extend([k] * m["n"])
        grp.append(cluster_ids(c))
    cid = np.array(cid)
    print(f"[feat] {len(qs)} prompts; vectorising (label-free)", flush=True)
    vec = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=300000,
                        sublinear_tf=True, strip_accents="unicode", lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000,
                        sublinear_tf=True, lowercase=True))
    T = vec.fit_transform(qs)
    print(f"[feat] tfidf {T.shape}; LSA -> {NSVD}", flush=True)
    Z = TruncatedSVD(n_components=NSVD, random_state=SEED).fit_transform(T)
    Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    ln = np.array([[len(q), len(q.split())] for q in qs], float)
    ln = (ln - ln.mean(0)) / (ln.std(0) + 1e-9)
    cols = [Z, ln, np.ones((len(qs), 1))]
    if cellid_onehot:
        oh = np.zeros((len(qs), len(CELLS)))
        oh[np.arange(len(qs)), cid] = 1.0
        cols.append(oh)
    X = np.hstack(cols)
    return X.astype(np.float64), cid, np.concatenate(grp), qs


def make_folds(cid, grp, rng):
    """Fold id per item; clusters are kept whole and folds are struck WITHIN each cell."""
    f = np.empty(len(cid), np.int64)
    for k in range(len(CELLS)):
        m = cid == k
        g = grp[m]
        K = int(g.max()) + 1
        f[m] = (rng.permutation(K) % FOLDS)[g]
    return f


class CrossFitRidge:
    """Cross-fitted weighted ridge with the per-fold normal equations cached, so refitting under a
    permuted target costs one X'y and one solve."""

    def __init__(self, X, fold, w, lam):
        self.X, self.fold, self.w, self.lam = X, fold, w, lam
        d = X.shape[1]
        self.inv, self.Xw = {}, {}
        for k in range(FOLDS):
            tr = fold != k
            Xt = X[tr]
            Wt = w[tr]
            A = Xt.T @ (Xt * Wt[:, None]) + lam * np.eye(d)
            self.inv[k] = np.linalg.inv(A)
            self.Xw[k] = Xt * Wt[:, None]

    def predict(self, y):
        out = np.empty(len(y))
        for k in range(FOLDS):
            tr = self.fold != k
            te = ~tr
            b = self.inv[k] @ (self.Xw[k].T @ y[tr])
            out[te] = self.X[te] @ b
        return out


class Sweep:
    """Every routing threshold at once, via cumulative sums over the score-sorted items.

    Routing 'score >= t' is the same family as 'the top-j items by score', so sweeping j from 0
    (all 7B) to n (all 32B) enumerates the WHOLE achievable frontier exactly -- no quantile grid,
    and O(n) per evaluation, which makes the permutation null cheap.
    """

    def __init__(self, y7, y32, cid):
        self.y7, self.y32, self.cid, self.n = y7, y32, cid, len(y7)
        self.K = len(CELLS)
        self.ncell = np.array([(cid == k).sum() for k in range(self.K)], float)
        self.onehot = np.zeros((self.K, self.n))
        self.onehot[cid, np.arange(self.n)] = 1.0
        self.S7 = np.array([y7[cid == k].sum() for k in range(self.K)], float)

    def curves(self, score):
        o = np.argsort(-score, kind="stable")
        oh = self.onehot[:, o]                                   # (K, n) membership in sorted order
        z = np.zeros((self.K, 1))
        cnt = np.hstack([z, np.cumsum(oh, axis=1)])              # (K, n+1) items sent to 32B
        c32 = np.hstack([z, np.cumsum(oh * self.y32[o], axis=1)])
        c7 = np.hstack([z, np.cumsum(oh * self.y7[o], axis=1)])
        acc = (c32 + (self.S7[:, None] - c7)) / self.ncell[:, None]
        e = cnt / self.ncell[:, None]
        macro_acc = acc.mean(0)
        macro_cost = ((1 - e) * FLOP_7B + e * FLOP_32B).mean(0)
        macro_cost_h = ((1 - e) * FLOP_7B + e * FLOP_32B_HONEST).mean(0)
        return macro_acc, macro_cost, macro_cost_h, e, acc, o

    def front_arrays(self, score):
        """Pareto front as ARRAYS (cost ascending, accuracy strictly increasing).

        Fully vectorised -- this is what the permutation null calls, so it must not build dicts.
        """
        macro_acc, macro_cost, macro_cost_h, e, acc, o = self.curves(score)
        order = np.lexsort((-macro_acc, macro_cost))
        a, c = macro_acc[order], macro_cost[order]
        run = np.maximum.accumulate(a)
        keep = np.empty(len(a), bool)
        keep[0] = True
        keep[1:] = run[1:] > run[:-1] + 1e-12
        j = order[keep]
        return c[keep], a[keep], j, (macro_acc, macro_cost, macro_cost_h, e, acc)

    def frontier_points(self, score, max_points=400):
        """The real fit's frontier, as reportable dicts (thinned to `max_points` along cost)."""
        c, a, j, (macro_acc, macro_cost, macro_cost_h, e, acc) = self.front_arrays(score)
        sel = np.unique(np.linspace(0, len(j) - 1, min(max_points, len(j))).round().astype(int))
        pts = [dict(j=int(j[i]), macro_acc=round(float(a[i]), 6), macro_cost=round(float(c[i]), 4),
                    macro_cost_honest=round(float(macro_cost_h[j[i]]), 4),
                    macro_route32=round(float(e[:, j[i]].mean()), 4),
                    pooled_route32=round(float(j[i] / self.n), 4),
                    per_cell_route32=[round(float(x), 4) for x in e[:, j[i]]],
                    per_cell_acc=[round(float(x), 5) for x in acc[:, j[i]]]) for i in sel]
        return pts, (c, a)


def at_cost_arr(c, a, budget):
    """Best macro accuracy reachable at or under `budget` macro FLOP-eq."""
    m = c <= budget + 1e-9
    return float(a[m].max()) if m.any() else np.nan


def at_acc_arr(c, a, target):
    """Cheapest macro FLOP-eq cost reaching `target` macro accuracy (1e-9 tie tolerance)."""
    m = a >= target - 1e-9
    return float(c[m].min()) if m.any() else np.nan


def at_cost(front, budget):
    ok = [p for p in front if p["macro_cost"] <= budget + 1e-9]
    return max(ok, key=lambda p: p["macro_acc"]) if ok else None


def at_acc(front, target):
    """Cheapest reported point reaching `target`; 1e-6 tolerance because the reported accuracies
    are 6-dp rounded and an exact-tie endpoint (route everything to the 32B versus
    always-32B-direct) must not be lost to rounding."""
    ok = [p for p in front if p["macro_acc"] >= target - 1e-6]
    return min(ok, key=lambda p: p["macro_cost"]) if ok else None


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else NBOOT
    vec = load_vec("disjoint")
    y7 = np.concatenate([vec[c]["always_7b"] for c in CELLS])
    y32 = np.concatenate([vec[c]["always_32b_direct"] for c in CELLS])
    res = {"what": "ATTACK C part 4 -- the pre-generation router frontier",
           "date": "2026-08-11", "seed": SEED, "n_folds": FOLDS, "n_perm": NPERM,
           "numerics": dict(threads_pinned=N_THREADS, numpy=np.__version__,
                            note="thread count is pinned because it is a known +-0.0048 axis in "
                                 "this project and the box was shared with another research round"),
           "preregistration": "results/cascade_methods/artifacts/"
                              "stats_recertification_2026-08-11_preregistration.json",
           "cost_model": dict(flop_7b=FLOP_7B, flop_32b_as_charged=FLOP_32B,
                              flop_32b_honest=FLOP_32B_HONEST,
                              router="(1-e)*flop_7b + e*flop_32b -- only ONE model ever runs",
                              cascade="flop_7b + e*flop_32b -- the 7B always runs first",
                              source="_selector_rerun_parts/summary_disjoint.json:cost_macro"),
           "shipped_operating_points": {k: dict(macro_acc=a, macro_flops=f)
                                        for k, (a, f) in SHIPPED.items()}}

    # ------------------------------------------------------------------------- NULL TEST -----
    ora = np.maximum(y7, y32)
    cid_all = np.concatenate([[k] * len(vec[c]["always_7b"]) for k, c in enumerate(CELLS)])
    m_ora = float(np.mean([ora[cid_all == k].mean() for k in range(len(CELLS))]))
    m_dir = float(np.mean([y32[cid_all == k].mean() for k in range(len(CELLS))]))
    m_7b = float(np.mean([y7[cid_all == k].mean() for k in range(len(CELLS))]))
    res["null_test"] = dict(
        what="the ORACLE router (route each item to whichever model is right) must reproduce the "
             "brief's free upper bound of +0.0661 macro over always-32B-direct",
        oracle_router_macro=round(m_ora, 6), always_32b_direct_macro=round(m_dir, 6),
        always_7b_macro=round(m_7b, 6), measured_upper_bound=round(m_ora - m_dir, 6),
        published_upper_bound=0.0661,
        abs_deviation=round(abs((m_ora - m_dir) - 0.0661), 6),
        also_reproduces=dict(direct=round(abs(m_dir - 0.6567), 6), b7=round(abs(m_7b - 0.5971), 6)),
        passed=bool(abs((m_ora - m_dir) - 0.0661) <= 1e-3 and abs(m_dir - 0.6567) <= 1e-4))
    print(f"[null] oracle router {m_ora:.4f} - direct {m_dir:.4f} = {m_ora-m_dir:+.4f} "
          f"(published +0.0661)  passed={res['null_test']['passed']}")

    g = (y32 - y7).astype(np.float64)
    rng = np.random.default_rng(SEED)

    res["variants"] = {}
    fronts = {}
    for vname, useid in (("prompt_text_only_PRIMARY", False), ("plus_cell_identity_UPPER_BOUND", True)):
        X, cid, grp, qs = build_features(useid)
        assert len(cid) == len(y7) == len(g)
        fold = make_folds(cid, grp, rng)
        ncell = np.array([(cid == k).sum() for k in range(len(CELLS))], float)
        w = (1.0 / (len(CELLS) * ncell))[cid]                    # macro-objective row weights
        w = w / w.mean()
        print(f"[fit] {vname}: X {X.shape}", flush=True)
        cf = CrossFitRidge(X, fold, w, lam=1.0)
        s = cf.predict(g)
        sw = Sweep(y7, y32, cid)
        front, (rc, ra) = sw.frontier_points(s)

        # ---------------- within-cell permutation null (refits are closed-form) --------------
        rp = np.random.default_rng(SEED + 5)
        null_at = {b: [] for b in (2.0, 3.0, 4.57)}
        null_cost_at_dir = []
        for _ in range(NPERM):
            gp = g.copy()
            for k in range(len(CELLS)):
                m = cid == k
                gp[m] = rp.permutation(gp[m])
            c_, a_, _, _ = sw.front_arrays(cf.predict(gp))[:4]
            for b in null_at:
                null_at[b].append(at_cost_arr(c_, a_, b))
            null_cost_at_dir.append(at_acc_arr(c_, a_, m_dir))

        # the real numbers come from the FULL front arrays, never the thinned reporting list
        real_at = {b: at_cost_arr(rc, ra, b) for b in (2.0, 3.0, 4.57)}
        real_cost_at_dir = at_acc_arr(rc, ra, m_dir)
        perm = {}
        for b in null_at:
            a = np.array(null_at[b], float)
            r = float(real_at[b]) if np.isfinite(real_at[b]) else None
            perm[f"acc_at_cost_{b}"] = dict(
                real=r, null_mean=round(float(np.nanmean(a)), 6), null_sd=round(float(np.nanstd(a)), 6),
                null_p95=round(float(np.nanpercentile(a, 95)), 6),
                p_value=(None if r is None else round(float((np.nansum(a >= r) + 1) / (NPERM + 1)), 4)),
                beats_within_cell_null=(None if r is None else bool(np.nanmean(a >= r) < 0.05)))
        res["variants"][vname] = dict(
            n_features=int(X.shape[1]),
            frontier=front, n_frontier_points=int(len(rc)),
            acc_at_cost={str(b): round(float(real_at[b]), 6) for b in real_at},
            cheapest_cost_matching_always_32b_direct=round(float(real_cost_at_dir), 4),
            permutation_null=perm,
            null_cost_to_match_direct=dict(
                real=round(float(real_cost_at_dir), 4),
                null_mean=round(float(np.nanmean(null_cost_at_dir)), 4),
                null_sd=round(float(np.nanstd(null_cost_at_dir)), 4)))
        print(f"  [{vname}] cheapest cost matching always-32B-direct: {real_cost_at_dir:.4f} "
              f"FLOP-eq (null {np.nanmean(null_cost_at_dir):.3f})")
        for b in (2.0, 3.0, 4.57):
            print(f"    cost<={b}: acc {real_at[b]:.6f} "
                  f"(null mean {perm[f'acc_at_cost_{b}']['null_mean']}, "
                  f"p={perm[f'acc_at_cost_{b}']['p_value']})")
        fronts[vname] = (rc, ra)

    # --------------------------- headline comparison against the shipped cascade --------------
    rc, ra = fronts["prompt_text_only_PRIMARY"]
    fr = res["variants"]["prompt_text_only_PRIMARY"]["frontier"]

    # ---- RANDOM-ALLOCATION FLOOR -------------------------------------------------------------
    # Send a uniformly random fraction e of EVERY cell's traffic to the 32B. Expected macro
    # accuracy is then exactly the straight line (1-e)*acc7 + e*acc32 per cell, and the macro cost
    # is (1-e)*1 + e*4.57. This needs no model, no features and no training. This project has a
    # history of oracle gaps that sit below a random-allocation floor, so the router MUST be
    # scored against it, not only against the shipped arms.
    a7 = np.array([vec[c]["always_7b"].mean() for c in CELLS])
    a32 = np.array([vec[c]["always_32b_direct"].mean() for c in CELLS])
    ee = np.linspace(0, 1, 2001)
    rand_acc = np.array([float(((1 - e) * a7 + e * a32).mean()) for e in ee])
    rand_cost = (1 - ee) * FLOP_7B + ee * FLOP_32B

    def rand_acc_at_cost(b):
        m = rand_cost <= b + 1e-9
        return float(rand_acc[m].max()) if m.any() else np.nan

    def rand_cost_at_acc(t):
        m = rand_acc >= t - 1e-9
        return float(rand_cost[m].min()) if m.any() else np.nan

    res["random_allocation_floor"] = dict(
        what="route a uniformly random fraction of every cell's traffic to the 32B -- no model, no "
             "features, no training. Expected macro accuracy is exactly linear between "
             "always-7B (1.0 FLOP-eq) and always-32B-direct (4.57).",
        acc_at_cost={str(b): round(rand_acc_at_cost(b), 6) for b in (2.0, 3.0, 4.57)},
        router_excess_over_random={
            str(b): round(at_cost_arr(rc, ra, b) - rand_acc_at_cost(b), 6) for b in (2.0, 3.0, 4.57)},
        interpretation="the learned router's margin over this floor is the ONLY part of its "
                       "frontier that required any modelling at all; compare it with "
                       "permutation_null, which says even that margin is not significant.")

    rows = {}
    for name, (a, c) in SHIPPED.items():
        cost = at_acc_arr(rc, ra, a)
        pt = at_acc(fr, a)
        rcost = rand_cost_at_acc(a)
        rows[name] = dict(
            shipped_macro_acc=a, shipped_macro_flops=c,
            router_cheapest_cost_at_that_accuracy=(None if not np.isfinite(cost) else round(cost, 4)),
            router_saving_x=(None if not np.isfinite(cost) else round(c / cost, 3)),
            router_reaches_it=bool(np.isfinite(cost)),
            random_allocation_cost_at_that_accuracy=(None if not np.isfinite(rcost) else round(rcost, 4)),
            random_allocation_saving_x=(None if not np.isfinite(rcost) else round(c / rcost, 3)),
            router_operating_point=pt,
            router_acc_at_that_arm_s_cost=round(at_cost_arr(rc, ra, c), 6))
    res["vs_shipped_cascade"] = dict(
        per_shipped_arm=rows,
        note="Shipped values from _selector_rerun_parts/summary_disjoint.json: accuracy-max 0.6575 "
             "at 7.951 macro FLOP-eq, compute-lean 0.6443 at 6.674. The router's whole frontier "
             "lies in [1.0, 4.57] by construction (it runs exactly ONE model), so wherever it "
             "reaches a shipped arm's accuracy it does so at a cost the cascade cannot reach at "
             "all. router_acc_at_that_arm_s_cost is capped by the all-32B endpoint 0.6567.",
        CRITICAL_CAVEAT="the router has NO within-cell skill (see permutation_null); everything it "
                        "does is choose a per-cell mixing rate. It is a cheaper frontier point, "
                        "NOT a better router.")
    jdump(res, os.path.join(META, "part4_router.json"))


if __name__ == "__main__":
    main()
