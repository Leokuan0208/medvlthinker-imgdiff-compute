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


def frontier(score, y7, y32, cid, n_points=61):
    """Sweep the routing threshold; return macro accuracy and macro FLOP-eq cost at each point."""
    qs = np.unique(np.quantile(score, np.linspace(0, 1, n_points)))
    ncell = np.array([(cid == k).sum() for k in range(len(CELLS))], float)
    pts = []
    for t in np.concatenate([[np.inf], qs[::-1], [-np.inf]]):
        to32 = score >= t
        ok = np.where(to32, y32, y7)
        acc = np.array([ok[cid == k].mean() for k in range(len(CELLS))])
        e = np.array([to32[cid == k].mean() for k in range(len(CELLS))])
        cost = (1 - e) * FLOP_7B + e * FLOP_32B
        cost_h = (1 - e) * FLOP_7B + e * FLOP_32B_HONEST
        pts.append(dict(thr=(None if not np.isfinite(t) else round(float(t), 6)),
                        macro_acc=round(float(acc.mean()), 6),
                        macro_cost=round(float(cost.mean()), 4),
                        macro_cost_honest=round(float(cost_h.mean()), 4),
                        macro_route32=round(float(e.mean()), 4),
                        pooled_route32=round(float(to32.mean()), 4),
                        per_cell_route32=[round(float(x), 4) for x in e],
                        per_cell_acc=[round(float(x), 5) for x in acc]))
    # keep the Pareto-optimal front (max accuracy at each cost, min cost at each accuracy)
    pts.sort(key=lambda p: (p["macro_cost"], -p["macro_acc"]))
    front, best = [], -np.inf
    for p in pts:
        if p["macro_acc"] > best:
            front.append(p)
            best = p["macro_acc"]
    return pts, front


def at_cost(front, budget):
    ok = [p for p in front if p["macro_cost"] <= budget + 1e-9]
    return max(ok, key=lambda p: p["macro_acc"]) if ok else None


def at_acc(front, target):
    ok = [p for p in front if p["macro_acc"] >= target - 1e-12]
    return min(ok, key=lambda p: p["macro_cost"]) if ok else None


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else NBOOT
    vec = load_vec("disjoint")
    y7 = np.concatenate([vec[c]["always_7b"] for c in CELLS])
    y32 = np.concatenate([vec[c]["always_32b_direct"] for c in CELLS])
    res = {"what": "ATTACK C part 4 -- the pre-generation router frontier",
           "date": "2026-08-11", "seed": SEED, "n_folds": FOLDS, "n_perm": NPERM,
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
        pts, front = frontier(s, y7, y32, cid)

        # ---------------- within-cell permutation null (refits are closed-form) --------------
        rp = np.random.default_rng(SEED + 5)
        null_at = {b: [] for b in (2.0, 3.0, 4.57)}
        null_cost_at_dir = []
        for _ in range(NPERM):
            gp = g.copy()
            for k in range(len(CELLS)):
                m = cid == k
                gp[m] = rp.permutation(gp[m])
            sp = cf.predict(gp)
            _, fp = frontier(sp, y7, y32, cid, n_points=41)
            for b in null_at:
                p = at_cost(fp, b)
                null_at[b].append(p["macro_acc"] if p else np.nan)
            p = at_acc(fp, m_dir)
            null_cost_at_dir.append(p["macro_cost"] if p else np.nan)

        real_at = {b: at_cost(front, b) for b in (2.0, 3.0, 4.57)}
        real_cost_at_dir = at_acc(front, m_dir)
        perm = {}
        for b in null_at:
            a = np.array(null_at[b], float)
            r = real_at[b]["macro_acc"] if real_at[b] else None
            perm[f"acc_at_cost_{b}"] = dict(
                real=r, null_mean=round(float(np.nanmean(a)), 6), null_sd=round(float(np.nanstd(a)), 6),
                null_p95=round(float(np.nanpercentile(a, 95)), 6),
                p_value=(None if r is None else round(float((np.nansum(a >= r) + 1) / (NPERM + 1)), 4)),
                beats_within_cell_null=(None if r is None else bool(np.nanmean(a >= r) < 0.05)))
        res["variants"][vname] = dict(
            n_features=int(X.shape[1]),
            frontier=front, all_points=pts,
            acc_at_cost=({str(b): real_at[b] for b in real_at}),
            cheapest_point_matching_always_32b_direct=real_cost_at_dir,
            permutation_null=perm,
            null_cost_to_match_direct=dict(
                real=(real_cost_at_dir["macro_cost"] if real_cost_at_dir else None),
                null_mean=round(float(np.nanmean(null_cost_at_dir)), 4),
                null_sd=round(float(np.nanstd(null_cost_at_dir)), 4)))
        print(f"  [{vname}] cheapest point matching always-32B-direct: "
              f"{real_cost_at_dir['macro_cost'] if real_cost_at_dir else None} FLOP-eq "
              f"(null {np.nanmean(null_cost_at_dir):.3f})")
        for b in (2.0, 3.0, 4.57):
            r = real_at[b]
            print(f"    cost<={b}: acc {r['macro_acc'] if r else None} "
                  f"(null mean {perm[f'acc_at_cost_{b}']['null_mean']}, "
                  f"p={perm[f'acc_at_cost_{b}']['p_value']})")

    # --------------------------- headline comparison against the shipped cascade --------------
    P = res["variants"]["prompt_text_only_PRIMARY"]
    fr = P["frontier"]
    cas_a, cas_c = SHIPPED["method_accuracy_max_veto"]
    cl_a, cl_c = SHIPPED["method_compute_lean"]
    res["vs_shipped_cascade"] = dict(
        router_at_cascade_cost=at_cost(fr, cas_c),
        router_at_compute_lean_cost=at_cost(fr, cl_c),
        router_cost_to_match_accuracy_max=at_acc(fr, cas_a),
        router_cost_to_match_compute_lean=at_acc(fr, cl_a),
        note="cascade accuracy-max is 0.6575 at 7.951 macro FLOP-eq; compute-lean 0.6443 at 6.674 "
             "(_selector_rerun_parts/summary_disjoint.json). The router's whole frontier lies "
             "between 1.0 and 4.57 by construction, so any point that matches those accuracies "
             "does so at a strictly lower cost than the cascade can reach.")
    jdump(res, os.path.join(META, "part4_router.json"))


if __name__ == "__main__":
    main()
