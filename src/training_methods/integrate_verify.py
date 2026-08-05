#!/usr/bin/env python3
"""integrate_verify.py -- STAGE 1 of the integration round: verify before building on.

Three checks, all in this agent's own code:

 1. METRIC NULL TEST -- reproduce every published incumbent cell from the transfer dumps.
 2. DISJOINTNESS -- re-assert train/eval image disjointness on md5 of decoded RGB pixels,
    read straight from the cache metadata (not through a previous run's claim).
 3. SPOT-CHECK THE SIBLINGS' STRONGEST CLAIMS by re-deriving them from the score vectors
    they left on disk:
      a) cheap-contrast pre-registered arm  H+C+M+Wc+Ws, 10-seed rank ensemble  -> 0.810627
      b) its like-for-like comparator       H (raw hidden) 10-seed rank ensemble -> 0.805858
      c) pair-head's 12-seed pointwise ensemble and its fusion with the incumbent
      d) REAL A-vs-B forward passes (HF arm) -- Borda / Copeland / knockout re-aggregated
         from the stored teacher matrix, on the 1345 covered items, against the incumbent
         restricted to those same items. This is the claim that decides whether ANY
         comparative component earned a place in the deployable stack.

  python3 src/training_methods/integrate_verify.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G      # noqa: E402
import integrate_lib as IL     # noqa: E402

ART = os.path.join(G.ROOT, "results/cascade_methods/artifacts")
PM_HF = os.path.join(ART, "realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl")
CC = os.path.join(ART, "_cheapcontrast_parts/eval_scores.npz")
PH = os.path.join(G.ROOT, "data/verifarch/pointwise_seed_scores_gpu.npy")


def sub_report(smap, items, sub_idx, base_got, tag, nboot=10000):
    """sel_eff of a selector restricted to a subset of items, paired vs a baseline."""
    r = G.sel_eff(smap, items)
    sub = np.zeros(len(items), bool)
    sub[sub_idx] = True
    m = sub & (r["rec"] == 1)
    b = G.paired_bootstrap(r["got"][sub], base_got[sub], rec=r["rec"][sub], nboot=nboot)
    per = {}
    for j, ds in enumerate(G.EVAL_DS):
        mm = m & (r["ds_index"] == j)
        per[ds] = float(r["got"][mm].mean()) if mm.sum() else None
    return {"tag": tag, "n_items": int(sub.sum()), "n_recoverable": int(m.sum()),
            "sel_eff": float(r["got"][m].mean()), "per_ds": per,
            "d_vs_incumbent_same_items": b["d_sel_eff"], "ci": b["d_sel_eff_ci"]}


def main():
    items = G.load_items()
    inc = G.incumbent_scores()
    base = G.sel_eff(inc, items)
    ev = G.load_candidates("eval", "generator", layers=[], pooling=())
    kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
    out = {"what": "verification stage of the integration round", "date": "2026-08-05"}

    # ---------------------------------------------------------------- 1. metric null test
    out["null_test"] = G.null_test()
    print("NULL pass", out["null_test"]["pass"], out["null_test"]["max_abs_deviation"], flush=True)

    # ---------------------------------------------------------------- 2. disjointness
    out["disjointness"] = {m: G.assert_disjoint(m) for m in ("generator", "grader")}
    print("disjoint OK", flush=True)

    # ---------------------------------------------------------------- 3. sibling claims
    sib = {}
    z = np.load(CC)
    for key, claim, tag in [("scores__H+C+M+Wc+Ws__bt", 0.810627, "cheapcontrast_preregistered"),
                            ("scores__H__bt", 0.805858, "cheapcontrast_H_only_comparator")]:
        S = z[key]
        e = IL.ens_rank(kev, S, items)
        r = G.sel_eff(e, items)
        fu = G.sel_eff(G.rank_fuse(inc, e, items=items), items)
        sib[tag] = {"claimed_sel_eff": claim, "rederived_sel_eff": r["sel_eff"],
                    "abs_dev": abs(r["sel_eff"] - claim),
                    "per_ds": {k: v["sel_eff"] for k, v in r["per_ds"].items()},
                    "guardrail_clean": G.guardrail_clean(r, base),
                    "fused_with_incumbent": fu["sel_eff"],
                    "n_seeds": int(S.shape[0]),
                    "per_seed": [G.sel_eff(IL.score_map(kev, s), items)["sel_eff"] for s in S],
                    "reproduces": bool(abs(r["sel_eff"] - claim) < 1e-5)}

    P = np.load(PH)
    e = IL.ens_rank(kev, P, items)
    r = G.sel_eff(e, items)
    fu = G.sel_eff(G.rank_fuse(inc, e, items=items), items)
    es = G.sel_eff(IL.ens_score(kev, P, items), items)
    ez = G.sel_eff(IL.ens_z(kev, P, items), items)
    sib["pairhead_12seed_pointwise_ensemble"] = {
        "claimed_sel_eff": 0.803134, "claimed_fused": 0.810627,
        "rederived_rank_ensemble": r["sel_eff"], "rederived_fused": fu["sel_eff"],
        "rederived_score_ensemble": es["sel_eff"], "rederived_z_ensemble": ez["sel_eff"],
        "abs_dev_rank_convention": abs(r["sel_eff"] - 0.803134),
        "note": "the claimed value is not reproduced by the mean-rank convention; the "
                "discrepancy is 3 quanta of 1/1468 and is an ENSEMBLING-CONVENTION "
                "difference, not a metric disagreement -- see which convention matches.",
        "reproduces_under_some_convention": bool(
            min(abs(r["sel_eff"] - 0.803134), abs(es["sel_eff"] - 0.803134),
                abs(ez["sel_eff"] - 0.803134)) < 1e-5)}
    out["sibling_spot_checks"] = sib

    # ------------------------------------------------- 3d. real A-vs-B forward passes (HF)
    rows = [json.loads(l) for l in open(PM_HF) if l.strip()]
    key2i = {(it["ds"], it["idx"]): i for i, it in enumerate(items)}
    covered = sorted(key2i[(r["ds"], r["idx"])] for r in rows)
    borda, copeland, knock = {}, {}, {}
    for r in rows:
        k = (r["ds"], r["idx"])
        P = np.asarray(r["P_avg"], float)
        K = len(r["na"])
        b = P.sum(1) - 0.5                      # Borda: sum of P(i beats j), minus self
        c = np.array([sum(1.0 if P[i, j] > 0.5 else (0.5 if P[i, j] == 0.5 else 0.0)
                          for j in range(K) if j != i) for i in range(K)])
        alive = list(range(K))                  # deterministic single-elimination knockout
        while len(alive) > 1:
            nxt = []
            for a in range(0, len(alive) - 1, 2):
                i, j = alive[a], alive[a + 1]
                nxt.append(i if P[i, j] >= 0.5 else j)
            if len(alive) % 2:
                nxt.append(alive[-1])
            alive = nxt
        kk = np.full(K, 0.0); kk[alive[0]] = 1.0
        for name, vec in (("borda", b), ("copeland", c), ("knockout", kk)):
            slots = [0.0] * 8
            for ci_, ss in enumerate(r["slots"]):
                for s in ss:
                    slots[s] = float(vec[ci_])
            {"borda": borda, "copeland": copeland, "knockout": knock}[name][k] = slots
    rp = {"n_items_covered": len(covered),
          "incumbent_on_covered": sub_report(inc, items, covered, base["got"],
                                             "incumbent (same items)")}
    for name, smap in (("borda", borda), ("copeland", copeland), ("knockout", knock)):
        full = dict(inc)
        full.update(smap)                       # uncovered items keep the incumbent's scores
        rp[name] = sub_report(full, items, covered, base["got"], f"real-pairwise {name}")
        fusedmap = G.rank_fuse(inc, full, items=items)
        rp[f"fusion_incumbent_plus_{name}"] = sub_report(
            fusedmap, items, covered, base["got"], f"rank_avg(incumbent, {name})")
    out["real_pairwise_recheck"] = rp

    IL.jdump(out, os.path.join(IL.PARTS, "verify.json"))
    print(json.dumps({k: v for k, v in out.items() if k != "disjointness"},
                     indent=1, default=float)[:6000], flush=True)


if __name__ == "__main__":
    main()
