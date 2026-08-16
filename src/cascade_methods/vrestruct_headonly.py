#!/usr/bin/env python3
"""vrestruct_headonly.py -- THE METHOD IS HEAD-ONLY: characterise it on its own terms.

The deliverable changed on 2026-08-16: the LoRA adapter is dropped and the selector is the frozen
generator-frame head alone, reading a layer-21 state CAPTURED during generation.  Verification cost
then goes to ~0 (a 918,529-parameter MLP over a vector the generator already produced), so the open
arm's cost IS the generation cost.

This script reports, for HEAD-ONLY and nothing else:
  1. sel_eff and SELECTED accuracy in BOTH currencies, per cell, guardrails, paired CIs, against
     the always-7B greedy baseline -- at N=8 and at every fixed N from 1 to 8.
  2. the same on the CAPTURED-DURING-GENERATION states at the generator's own cap320, which is
     what a deployment actually gets (feats_free/free_cap320_L21.h_span_ar.npy).
  3. the Weitzman refit with inspection cost ~0.  A head-only method has no LoRA score to stop on,
     so the box value must come from the head itself: the RAW head logit (mean over seeds) is
     cardinal and cross-question comparable, unlike the within-question rank_avg the selector uses
     for picking.  Both are tried; the rank version is the degenerate control.
  4. the head's own FLOP cost, stated against the always-7B baseline of 1.0.

    OMP_NUM_THREADS=8 python3 src/cascade_methods/vrestruct_headonly.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import genframe_data as G          # noqa: E402
import weitzman_lib as W           # noqa: E402
import vrestruct_lib as V          # noqa: E402
import vrestruct_freehead_eval as FH   # noqa: E402

PARTS = V.PARTS
CELL = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
        "pathvqa_open": "PATH_VQA_open"}
LAMS = np.concatenate([[0.0], np.logspace(-5, 0.5, 90)])
CV_SEEDS = list(range(10))
NFOLD = 5
HEAD_PARAMS = 918529


def head_scores(P, Xall, slot_rows, seeds=range(8)):
    """(raw mean-logit slot scores, rank-ensemble slot scores) for the frozen head."""
    S = V.selector()
    Z = S.standardize(np.asarray(Xall, np.float32))
    L = S.head_logits(Z, standardized=True)          # (8 seeds, n_rows)
    sub = L[list(seeds)]
    n = slot_rows.shape[0]
    raw = np.empty((n, 8), float)
    rnk = np.empty((n, 8), float)
    mean_logit = sub.mean(0)
    for i in range(n):
        r = slot_rows[i]
        raw[i] = mean_logit[r]
        rnk[i] = np.mean([G.rank_avg(sub[s][r]) for s in range(sub.shape[0])], axis=0)
    return raw, rnk


def fixed_N(P, S, label):
    """Accuracy at fixed pool depth N=1..8 using score matrix S, both currencies."""
    n = P["n"]
    rows = np.arange(n)
    pa = np.empty((n, 8), int)
    cur = np.zeros(n, int)
    best = S[:, 0].copy()
    pa[:, 0] = 0
    for j in range(1, 8):
        b = S[:, j] > best
        cur = np.where(b, j, cur)
        best = np.where(b, S[:, j], best)
        pa[:, j] = cur
    out = {}
    got8 = {cur_: P[cur_][rows, pa[:, 7]].astype(int) for cur_ in ("judge", "em")}
    g = P["greedy_ok"].astype(int)
    for N in range(1, 9):
        d = N - 1
        r = {}
        for cur_ in ("judge", "em"):
            got = P[cur_][rows, pa[:, d]].astype(int)
            rec = (P[cur_][:, :N].max(1) == 1)
            r[cur_] = dict(
                acc=float(got.mean()), oracle_at_N=float(rec.mean()),
                sel_eff=float(got[rec].mean()),
                macro3=float(np.mean([got[P["ds_index"] == j].mean() for j in range(3)])),
                per_cell={CELL[ds]: float(got[P["ds_index"] == j].mean())
                          for j, ds in enumerate(G.EVAL_DS)},
                vs_N8=V.paired_boot(got, got8[cur_]),
                per_cell_vs_greedy={CELL[ds]: V.paired_boot(
                    got[P["ds_index"] == j], g[P["ds_index"] == j])
                    for j, ds in enumerate(G.EVAL_DS)} if cur_ == "judge" else None,
                vs_greedy7b=V.paired_boot(got, g) if cur_ == "judge" else None)
            if cur_ == "judge":
                r[cur_]["guardrail_clean_vs_greedy"] = bool(all(
                    r[cur_]["per_cell_vs_greedy"][CELL[ds]]["delta"] >= 0 for ds in G.EVAL_DS))
        out[N] = r
    return {"label": label, "by_N": out}


def weitzman_headonly(P, box, pick, label):
    """Weitzman with inspection cost ~0: stop on `box`, pick with `pick`. No strong box."""
    labs = {"judge": P["judge"].astype(float), "em": P["em"].astype(float)}
    zeros = {"judge": np.zeros(P["n"]), "em": np.zeros(P["n"])}
    keys = [(it["ds"], it["idx"]) for it in P["items"]]

    class SV(W.PoolView):
        def __init__(self):
            super().__init__(box, labs, zeros, P["ds_index"], keys)
            pa = np.empty(pick.shape, int)
            cur = np.zeros(self.n, int)
            best = pick[:, 0].copy()
            pa[:, 0] = 0
            for j in range(1, 8):
                b = pick[:, j] > best
                cur = np.where(b, j, cur)
                best = np.where(b, pick[:, j], best)
                pa[:, j] = cur
            self.prefix_argmax = pa
            r = np.arange(self.n)[:, None]
            self.prefix_lab = {k: np.asarray(v)[r, pa] for k, v in self.labs.items()}

    v = SV()
    # inspection cost: the head is 918,529 params over an already-computed vector.
    # 2*P FLOP = 1.837 MFLOP against a 5.6927 TFLOP forward.
    c_head = 2.0 * HEAD_PARAMS / (V.cost_constants()["unit_tflop"] * 1e12)
    out = {}
    for cname, c_cheap in (("c_cheap_1.0_generation_only", 1.0),
                           ("c_cheap_0.011544_shared_prefill_marginal", 0.011544),
                           ("c_cheap_head_only_inspection", c_head)):
        per_seed = []
        for s in CV_SEEDS:
            folds = W.image_folds_for_keys(s, keys, NFOLD)
            res = {li: dict(ok={k: np.zeros(P["n"]) for k in labs}, N=np.zeros(P["n"]))
                   for li in range(len(LAMS))}
            for f in range(NFOLD):
                te = folds == f
                tr = ~te
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                xs = v.raw[tr].ravel()
                ys = labs["judge"][tr].ravel()
                m = xs > -1e8
                iso.fit(xs[m], ys[m])
                cal = iso.predict(v.raw.ravel()).reshape(v.raw.shape)
                zc = W.zeta_cheap_many(cal[tr].ravel(), LAMS, c=c_cheap)
                for li in range(len(LAMS)):
                    r = v.run(cal, float(zc[li]), -np.inf)
                    res[li]["N"][te] = r["N"][te]
                    for k in labs:
                        res[li]["ok"][k][te] = r["ok"][k][te]
            per_seed.append(res)
        agg = []
        for li in range(len(LAMS)):
            agg.append(dict(
                lam=float(LAMS[li]),
                meanN=float(np.mean([p[li]["N"].mean() for p in per_seed])),
                meanN_sd=float(np.std([p[li]["N"].mean() for p in per_seed], ddof=1)),
                acc_judge=float(np.mean([p[li]["ok"]["judge"].mean() for p in per_seed])),
                acc_em=float(np.mean([p[li]["ok"]["em"].mean() for p in per_seed])),
                macro3_judge=float(np.mean([np.mean([p[li]["ok"]["judge"][P["ds_index"] == j].mean()
                                                     for j in range(3)]) for p in per_seed]))))
        bi = int(np.argmax([a["acc_judge"] for a in agg]))
        # the cheapest lambda whose accuracy ties the best (within its own seed sd)
        tol = agg[bi]["acc_judge"] - 2 * float(np.std(
            [a["acc_judge"] for a in agg[max(bi - 3, 0):bi + 4]], ddof=1) or 1e-9)
        cheap = min([a for a in agg if a["acc_judge"] >= tol], key=lambda a: a["meanN"],
                    default=agg[bi])
        out[cname] = dict(c_cheap=c_cheap, frontier=agg, best=agg[bi],
                          cheapest_tying=cheap, argmax=bi)
    return {"label": label, "scenarios": out,
            "head_inspection_cost_flopeq": c_head,
            "_note": "no strong box: under the always-7B baseline the pipeline never calls the 32B, "
                     "so Weitzman reduces to a pure stopping rule over the cheap draws."}


def main():
    os.makedirs(PARTS, exist_ok=True)
    P = V.load_pool()
    c = V.cost_constants()
    g = P["greedy_ok"].astype(int)

    # ---- the two feature sources -------------------------------------------------------------
    sources = {}
    Lref = V.head_logits(P)
    sources["deployed_cache_fullres_TF"] = (P["X"], P["slot_rows"])
    for cap, which in (("cap320", "ar"), ("fullres", "ar")):
        try:
            Xa, sr, diag = FH.load_free(cap, which, P)
            sources[f"captured_{cap}_{which}"] = (Xa, sr)
        except Exception as e:
            print(f"  [skip] {cap}/{which}: {e}", flush=True)

    out = {}
    for name, (Xall, sr) in sources.items():
        raw, rnk = head_scores(P, Xall, sr)
        r8 = V.evaluate(P, V.picks_of(rnk), f"head_only_{name}")
        row = dict(
            sel_eff_judge=r8["judge"]["sel_eff"], acc_judge=r8["judge"]["acc"],
            sel_eff_em=r8["em"]["sel_eff"], acc_em=r8["em"]["acc"],
            macro3_judge=r8["judge"]["macro_cells"], macro3_em=r8["em"]["macro_cells"],
            per_cell_judge={CELL[d]: r8["judge"]["per_ds"][d]["acc"] for d in G.EVAL_DS},
            per_cell_em={CELL[d]: r8["em"]["per_ds"][d]["acc"] for d in G.EVAL_DS},
            vs_greedy7b_judge=V.paired_boot(r8["judge"]["got"], g),
            per_cell_vs_greedy_judge={CELL[ds]: V.paired_boot(
                r8["judge"]["got"][P["ds_index"] == j], g[P["ds_index"] == j])
                for j, ds in enumerate(G.EVAL_DS)},
            identity_dev=r8["identity_dev_judge"],
            fixed_N=fixed_N(P, rnk, name))
        row["guardrail_clean_vs_greedy"] = bool(all(
            row["per_cell_vs_greedy_judge"][CELL[ds]]["delta"] >= 0 for ds in G.EVAL_DS))
        if name == "captured_cap320_ar" or name == "deployed_cache_fullres_TF":
            row["weitzman_box_raw_logit"] = weitzman_headonly(P, raw, rnk, f"{name}|raw_logit_box")
            row["weitzman_box_rank_DEGENERATE_CONTROL"] = weitzman_headonly(
                P, rnk, rnk, f"{name}|rank_box")
        out[name] = row
        print(f"  {name:28s} sel_eff {r8['judge']['sel_eff']:.6f}  accJ {r8['judge']['acc']:.6f}  "
              f"accEM {r8['em']['acc']:.6f}  macro3J {r8['judge']['macro_cells']:.6f}", flush=True)

    # ---- cost, against always-7B = 1.0 -------------------------------------------------------
    unit = c["unit_tflop"] * 1e12
    cost = dict(
        unit=c["unit_definition"],
        head_forward_flops=2.0 * HEAD_PARAMS,
        head_forward_flopeq=2.0 * HEAD_PARAMS / unit,
        head_params=HEAD_PARAMS,
        n_head_evaluations_per_question=8 * 8,
        total_head_flopeq_per_question=8 * 8 * 2.0 * HEAD_PARAMS / unit,
        _read="8 seeds x 8 slots of a 918,529-parameter MLP per question. Against one Lingshu-7B "
              "cap320 forward this is ~1.65e-5 FLOP-eq -- five orders of magnitude below the "
              "measurement noise on anything else in this table. HEAD-ONLY VERIFICATION IS FREE.",
        teacher_forced_pass_removed_flopeq=c["head_1003520_flopeq"],
        passes_removed_per_question=3.8136460554371,
        removed_total_flopeq=3.8136460554371 * c["head_1003520_flopeq"])

    json.dump(dict(
        title="HEAD-ONLY: the selector on its own terms, both currencies, per cell, with its cost",
        date="2026-08-16", cpu_only=True, no_refit=True,
        arms=out, cost=cost,
        T04_status="NOT MEASURED. The generator-frame feature cache and the free-head capture both "
                   "exist only for the DEPLOYED T=0.7 pool; no layer-21 states have ever been "
                   "extracted for the T=0.4 pools, so head-only at T=0.4 cannot be evaluated "
                   "without a new ~90-minute capture per arm. The LoRA-box Weitzman refit at T=0.4 "
                   "IS measured (weitzman.json) but it is not the head-only method.",
        null_test=dict(deployed_cache_sel_eff=out["deployed_cache_fullres_TF"]["sel_eff_judge"],
                       expected_8seed_rank_head_only=0.8010899182561307,
                       abs_deviation=abs(out["deployed_cache_fullres_TF"]["sel_eff_judge"]
                                         - 0.8010899182561307))),
        open(os.path.join(PARTS, "headonly.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(PARTS, "headonly.json"))


if __name__ == "__main__":
    main()
