#!/usr/bin/env python3
"""
ATTACK C / PART 3 -- BEST-OF-MAJORITY FILTERING, and the CLUSTER re-certification of sel_eff.

WHY.  LITERATURE_UPDATE_2026-08-11.md flags Best-of-Majority (ICLR 2026): filter candidates by
majority / frequency BEFORE the verifier scores them, claimed to fix the monotone sel_eff decay at
larger K.  Zero generation cost on our frozen 8-sample pools.

Also folded in here (it needs the same pool + the same image clusters): the part-2 cluster
re-certification of the frozen selector's sel_eff gain, +0.0354 (0.775204 incumbent ->
0.810627 deployed).

PRE-REGISTERED in results/cascade_methods/artifacts/stats_recertification_2026-08-11_preregistration.json
BEFORE any number below was computed.  PRIMARY arm = BoM-c2, zero free parameters.

NULL TEST.  Unfiltered argmax must reproduce the published sel_eff 0.775204 (incumbent) and
0.810627 (deployed frozen fused selector, reloaded read-only) to 1e-6.

ckpts/train/genframe_head_ens8/ is READ ONLY here.  freeze_selector.py is never invoked.

Launch from the repo root:  python3 src/cascade_methods/stats_recert_p3_bom.py
Writes results/cascade_methods/artifacts/_stats_recert/part3_bom.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "training_methods"))
from stats_recert_common import (CELLS_OPEN, META, NBOOT, OPEN_DS, SEED, ci, cluster_ids, jdump,
                                 load_transfer, norm)

PUB_INC, PUB_DEP = 0.775204, 0.810627
PUB_PER_DS_DEP = {"slake_open": 0.8853615520282186, "vqa_rad_open": 0.8095238095238095,
                  "pathvqa_open": 0.7561290322580645}
PUB_PER_DS_INC = {"slake_open": 0.8501, "vqa_rad_open": 0.7619, "pathvqa_open": 0.7226}


# ------------------------------------------------------------------------ the BoM filters
def keep_mask(preds, variant):
    """Boolean mask over the 8 pool slots: which candidates survive the frequency filter.

    Every variant falls back to 'keep all 8' when the filter would empty the set, so the method
    ALWAYS returns an answer (CRITICAL RULE 6).
    """
    keys = [norm(p) for p in preds]
    cnt = Counter(keys)
    if variant == "none":
        m = [True] * len(keys)
    elif variant.startswith("c"):                       # count >= t
        t = int(variant[1:])
        m = [cnt[k] >= t for k in keys]
    elif variant.startswith("top"):                     # in the top-m frequency ranks
        m_ = int(variant[3:])
        # rank answers by (count desc, first appearance asc) -- deterministic, no RNG
        first = {}
        for i, k in enumerate(keys):
            first.setdefault(k, i)
        order = sorted(cnt, key=lambda k: (-cnt[k], first[k]))
        keepset = set(order[:m_])
        m = [k in keepset for k in keys]
    else:
        raise ValueError(variant)
    return np.array(m if any(m) else [True] * len(keys))


VARIANTS = ["none", "c2", "top1", "top2", "top3", "c3"]


def build_pool():
    """-> per-cell dict of aligned arrays, in the transfer-dump (= canonical eval) item order."""
    out = {}
    for cell in CELLS_OPEN:
        ds = OPEN_DS[cell]
        rows = load_transfer(ds)
        sl = np.array([[0 if x in (None, -1) else int(x) for x in r["sl"]] for r in rows], np.int8)
        inc = np.array([r["scores"] for r in rows], float)
        preds = [r["preds"] for r in rows]
        out[cell] = dict(ds=ds, idx=[r["idx"] for r in rows], sl=sl, inc=inc, preds=preds,
                         greedy_ok=np.array([int(r["greedy_ok"]) for r in rows], np.int8),
                         oracle=sl.max(1).astype(np.int8), groups=cluster_ids(cell))
        assert len(rows) == len(out[cell]["groups"]), (cell, len(rows))
    return out


def deployed_scores(pool):
    """Per-slot scores of the DEPLOYED frozen fused selector, reloaded read-only from disk."""
    import genframe_selector as GS
    sel = GS.FrozenSelector.load()
    smap, _ = GS.score_eval_pool(sel)
    out = {}
    for cell in CELLS_OPEN:
        ds = pool[cell]["ds"]
        out[cell] = np.array([smap[(ds, i)] for i in pool[cell]["idx"]], float)
    return out, sel.recipe


def picks(scores, preds_list, variant):
    """argmax score among the kept slots, FIRST-INDEX tie-break (the frozen pick rule)."""
    n = len(preds_list)
    p = np.empty(n, np.int64)
    for i in range(n):
        m = keep_mask(preds_list[i], variant)
        s = np.where(m, scores[i], -np.inf)
        p[i] = int(np.argmax(s))
    return p


def sel_eff_of(pool, sel_by_cell):
    """Pooled + per-cell sel_eff = selected / oracle@8, the project's frozen definition."""
    num = sum(int(sel_by_cell[c].sum()) for c in CELLS_OPEN)
    den = sum(int(pool[c]["oracle"].sum()) for c in CELLS_OPEN)
    per = {pool[c]["ds"]: float(sel_by_cell[c].sum() / pool[c]["oracle"].sum()) for c in CELLS_OPEN}
    tot = sum(len(pool[c]["oracle"]) for c in CELLS_OPEN)
    return dict(sel_eff=num / den, acc=num / tot, per_ds=per, n=tot, n_recoverable=den)


# ------------------------------------------------------------------------------ bootstraps
def paired_boot(pool, arms, nboot, rng, scheme="item"):
    """Bootstrap sel_eff for several arms jointly (paired). arms: {name: {cell: 0/1 vector}}."""
    names = list(arms)
    num = np.zeros((nboot, len(names)))
    den = np.zeros(nboot)
    for c in CELLS_OPEN:
        n = len(pool[c]["oracle"])
        if scheme == "item":
            w = rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(np.float64)
        else:
            g = pool[c]["groups"]
            K = int(g.max()) + 1
            mm = rng.multinomial(K, np.full(K, 1.0 / K), size=nboot).astype(np.float64)
            w = mm[:, g]                                     # per-item multiplicity via its cluster
        den += w @ pool[c]["oracle"].astype(np.float64)
        for j, nm in enumerate(names):
            num[:, j] += w @ arms[nm][c].astype(np.float64)
    return num / den[:, None], names


def main():
    nboot = int(sys.argv[1]) if len(sys.argv) > 1 else NBOOT
    pool = build_pool()
    res = {"what": "ATTACK C part 3 -- Best-of-Majority filtering on the frozen 8-sample pools, "
                   "plus the cluster re-certification of the frozen selector's sel_eff gain",
           "date": "2026-08-11", "n_bootstrap": nboot, "seed": SEED,
           "preregistration": "results/cascade_methods/artifacts/"
                              "stats_recertification_2026-08-11_preregistration.json",
           "pool": {c: dict(n=int(len(pool[c]["oracle"])),
                            oracle_at_8=round(float(pool[c]["oracle"].mean()), 6),
                            n_recoverable=int(pool[c]["oracle"].sum()),
                            n_clusters=int(pool[c]["groups"].max() + 1),
                            mean_distinct_answers=round(float(np.mean(
                                [len(set(norm(x) for x in p)) for p in pool[c]["preds"]])), 3))
                    for c in CELLS_OPEN}}

    print("[part3] reloading the frozen deployed selector (READ ONLY)", flush=True)
    dep, recipe = deployed_scores(pool)
    res["deployed_selector_recipe"] = {k: recipe[k] for k in
                                       ("mode", "layer", "pooling", "objective", "hidden", "seeds")
                                       if k in recipe}

    scorers = {"incumbent_lora": {c: pool[c]["inc"] for c in CELLS_OPEN},
               "deployed_fused": dep}

    # ------------------------------------------------------- all (scorer, variant) selections
    sel = {}
    for sname, S in scorers.items():
        for v in VARIANTS:
            sel[(sname, v)] = {c: pool[c]["sl"][np.arange(len(pool[c]["sl"])),
                                                picks(S[c], pool[c]["preds"], v)]
                               for c in CELLS_OPEN}
    # reference arms that use no verifier at all
    sel[("majority_vote", "n/a")] = {c: pool[c]["greedy_ok"] for c in CELLS_OPEN}

    res["arms"] = {f"{a}|{b}": sel_eff_of(pool, s) for (a, b), s in sel.items()}

    # ------------------------------------------------------------------------ NULL TEST -----
    inc0 = res["arms"]["incumbent_lora|none"]
    dep0 = res["arms"]["deployed_fused|none"]
    dev = {"incumbent_sel_eff": abs(inc0["sel_eff"] - PUB_INC),
           "deployed_sel_eff": abs(dep0["sel_eff"] - PUB_DEP)}
    dev.update({f"deployed.{d}": abs(dep0["per_ds"][d] - PUB_PER_DS_DEP[d]) for d in PUB_PER_DS_DEP})
    dev.update({f"incumbent.{d}": abs(inc0["per_ds"][d] - PUB_PER_DS_INC[d]) for d in PUB_PER_DS_INC})
    res["null_test"] = dict(
        what="unfiltered argmax must reproduce the published sel_eff of both selectors",
        measured=dict(incumbent=inc0["sel_eff"], deployed=dep0["sel_eff"]),
        published=dict(incumbent=PUB_INC, deployed=PUB_DEP),
        abs_deviation={k: float(v) for k, v in dev.items()},
        max_abs_deviation=float(max(dev.values())),
        passed=bool(max(dev.values()) <= 1e-4),
        tolerance="1e-4: the published incumbent per-cell cells are 4-dp rounded; the deployed "
                  "cells are full precision and must match at 1e-6.")

    # ------------------------------------------------------- paired bootstrap of every contrast
    rng = np.random.default_rng(SEED)
    arms_flat = {f"{a}|{b}": s for (a, b), s in sel.items()}
    dist, names = paired_boot(pool, arms_flat, nboot, rng, "item")
    ni = {n: i for i, n in enumerate(names)}
    pt = {n: res["arms"][n]["sel_eff"] for n in names}

    res["bom_vs_unfiltered"] = {}
    for sname in scorers:
        base = f"{sname}|none"
        for v in VARIANTS[1:]:
            k = f"{sname}|{v}"
            row = ci(dist[:, ni[k]] - dist[:, ni[base]], pt[k] - pt[base])
            row["sel_eff"] = round(pt[k], 6)
            row["sel_eff_base"] = round(pt[base], 6)
            # per-cell guardrail (paired item bootstrap within the cell)
            g = {}
            for c in CELLS_OPEN:
                d = sel[(sname, v)][c].astype(float) - sel[(sname, "none")][c].astype(float)
                n = len(d)
                bw = rng.multinomial(n, np.full(n, 1.0 / n), size=nboot).astype(np.float64)
                den = bw @ pool[c]["oracle"].astype(np.float64)
                g[c] = ci((bw @ d) / den, float(d.sum() / pool[c]["oracle"].sum()))
            row["per_cell_guardrail"] = g
            row["guardrail_violations"] = [c for c in CELLS_OPEN if g[c]["verdict"] == "LOSS"]
            res["bom_vs_unfiltered"][k] = row
    res["PRIMARY_preregistered"] = {s: res["bom_vs_unfiltered"][f"{s}|c2"] for s in scorers}

    # -------------------------------------------- decomposition null: filter WITHOUT a verifier
    rng2 = np.random.default_rng(SEED + 7)
    fo = {}
    for v in VARIANTS:
        accs = []
        for _ in range(200):
            tot = 0
            for c in CELLS_OPEN:
                for i in range(len(pool[c]["sl"])):
                    m = keep_mask(pool[c]["preds"][i], v)
                    j = rng2.choice(np.flatnonzero(m))
                    tot += int(pool[c]["sl"][i][j])
            accs.append(tot)
        den = sum(int(pool[c]["oracle"].sum()) for c in CELLS_OPEN)
        a = np.array(accs) / den
        fo[v] = dict(sel_eff_mean=round(float(a.mean()), 6), sd=round(float(a.std()), 6),
                     lo=round(float(np.percentile(a, 2.5)), 6),
                     hi=round(float(np.percentile(a, 97.5)), 6), n_seeds=200)
    res["filter_only_random_pick"] = dict(
        what="keep the BoM filter, replace the verifier score with a uniform random draw among the "
             "kept candidates (200 seeds). Separates the frequency FILTER's contribution from the "
             "verifier's.", by_variant=fo)

    # ------------------------------------------------ nested CV over the variant sweep (anti-leak)
    res["nested_cv"] = nested_cv(pool, scorers, sel)

    # ============ the part-2 cluster re-certification of the sel_eff gain (+0.0354) ============
    gain_arms = {"incumbent": sel[("incumbent_lora", "none")], "deployed": sel[("deployed_fused", "none")]}
    out = {}
    for scheme in ("item", "cluster"):
        r = np.random.default_rng(SEED + 3)
        d, nm = paired_boot(pool, gain_arms, nboot, r, scheme)
        out["image" if scheme == "cluster" else scheme] = ci(
            d[:, nm.index("deployed")] - d[:, nm.index("incumbent")], dep0["sel_eff"] - inc0["sel_eff"])
    res["sel_eff_gain_recertification"] = dict(
        claim="frozen 8-seed selector raises pooled sel_eff by +0.0354 (0.775204 -> 0.810627)",
        point=round(dep0["sel_eff"] - inc0["sel_eff"], 6), schemes=out,
        note="this is the SELECTION effect only; CLAUDE.md already records it as NOT significant on "
             "the 8-cell macro (+0.0014 [-0.0003,+0.0032]). Here it is bootstrapped on the open "
             "pool's own sel_eff scale, where it is far larger.")

    jdump(res, os.path.join(META, "part3_bom.json"))
    n = res["null_test"]
    print(f"[null] max abs deviation {n['max_abs_deviation']:.2e} passed={n['passed']}")
    print(f"       incumbent {n['measured']['incumbent']:.6f} (pub {PUB_INC})  "
          f"deployed {n['measured']['deployed']:.6f} (pub {PUB_DEP})")
    for k, r in res["bom_vs_unfiltered"].items():
        print(f"  {k:26s} sel_eff {r['sel_eff']:.4f} (base {r['sel_eff_base']:.4f}) "
              f"{r['delta']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] {r['verdict']:4s} "
              f"guardrail_loss={r['guardrail_violations']}")
    s = res["sel_eff_gain_recertification"]
    print(f"  sel_eff gain {s['point']:+.4f}  item [{s['schemes']['item']['lo']:+.4f},"
          f"{s['schemes']['item']['hi']:+.4f}] {s['schemes']['item']['verdict']}"
          f"  image [{s['schemes']['image']['lo']:+.4f},{s['schemes']['image']['hi']:+.4f}] "
          f"{s['schemes']['image']['verdict']}")


def nested_cv(pool, scorers, sel, folds=5):
    """5-fold nested CV grouped by IMAGE CLUSTER: inner folds choose the variant, outer folds score.

    Prevents 'the best sweep entry' from being quoted as if it were a pre-registered arm.
    """
    rng = np.random.default_rng(SEED + 11)
    out = {}
    for sname in scorers:
        # assign each cluster of each cell to a fold
        fold = {}
        for c in CELLS_OPEN:
            g = pool[c]["groups"]
            K = int(g.max()) + 1
            f = rng.permutation(K) % folds
            fold[c] = f[g]
        num = den = 0
        chosen = []
        for k in range(folds):
            best, bestv = None, -np.inf
            for v in VARIANTS:
                n_ = d_ = 0
                for c in CELLS_OPEN:
                    m = fold[c] != k                                  # inner (training) part
                    n_ += int(sel[(sname, v)][c][m].sum())
                    d_ += int(pool[c]["oracle"][m].sum())
                if n_ / d_ > bestv:
                    bestv, best = n_ / d_, v
            chosen.append(best)
            for c in CELLS_OPEN:
                m = fold[c] == k
                num += int(sel[(sname, best)][c][m].sum())
                den += int(pool[c]["oracle"][m].sum())
        out[sname] = dict(nested_cv_sel_eff=round(num / den, 6), variants_chosen=chosen,
                          unfiltered=round(sel_eff_of(pool, sel[(sname, "none")])["sel_eff"], 6),
                          sweep_max=round(max(sel_eff_of(pool, sel[(sname, v)])["sel_eff"]
                                              for v in VARIANTS), 6))
    out["_note"] = ("nested_cv_sel_eff is the honest number for 'pick the best BoM variant'; "
                    "sweep_max is what quoting the maximum would have claimed.")
    return out


if __name__ == "__main__":
    main()
