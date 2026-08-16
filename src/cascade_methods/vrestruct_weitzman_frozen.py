#!/usr/bin/env python3
"""vrestruct_weitzman_frozen.py -- QUESTION 3, on the RECOMMENDED selector.

vrestruct_weitzman.py refits lambda on the T=0.4 pools, but those pools have only the LoRA
verifier's scores (the generator-frame feature cache was extracted for the deployed T=0.7 pool, so
the FUSED selector does not exist at T=0.4).  This script therefore runs the same cost-scenario
refit on the FROZEN T=0.7 pool with the FUSED selector -- the structure this round recommends --
and adds the question the cost objective actually turns on:

    HOW MANY SAMPLES DO WE NEED?  Once verification is free, generation is ~100% of the cost, so
    the N curve IS the cost curve.  We report fixed-N accuracy for N = 1..8 in BOTH currencies,
    per cell, with paired CIs against N=8, and then ask whether the adaptive controller buys
    anything over the best fixed N at equal cost.

    OMP_NUM_THREADS=8 python3 src/cascade_methods/vrestruct_weitzman_frozen.py
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

import genframe_data as G      # noqa: E402
import weitzman_lib as W       # noqa: E402
import vrestruct_lib as V      # noqa: E402
from vrestruct_weitzman import scenarios, LAMS, CV_SEEDS, NFOLD   # noqa: E402

PARTS = V.PARTS
OPEN_CELL_NAME = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
                  "pathvqa_open": "PATH_VQA_open"}


class SplitView(W.PoolView):
    """Weitzman over a pool where the STOPPING signal and the PICK signal are different vectors.

    This is what the deployed pipeline actually does and what a rank-fusion selector REQUIRES.
    The controller's box value must be comparable ACROSS questions -- it is the thing the isotonic
    calibration turns into P(correct) and the thing the reservation value is a quantile of.  The
    fused selector score is a WITHIN-QUESTION rank sum: it is deliberately not comparable across
    questions, and feeding it to the controller collapses the policy (measured: isotonic maps the
    rank plateaus onto a flat top, so the first slot clears any reservation value and meanN falls
    to 1.5 with accuracy 0.4755, far BELOW fixed N=8).  So: stop on the LoRA verifier's raw
    P(correct); pick with the fused score.
    """

    def __init__(self, stop_scores, pick_scores, labs, strongs, ds_index, item_keys):
        super().__init__(stop_scores, labs, strongs, ds_index, item_keys)
        pa = np.empty(pick_scores.shape, int)
        cur = np.zeros(self.n, int)
        best = pick_scores[:, 0].copy()
        pa[:, 0] = 0
        for j in range(1, self.Nmax):
            better = pick_scores[:, j] > best
            cur = np.where(better, j, cur)
            best = np.where(better, pick_scores[:, j], best)
            pa[:, j] = cur
        self.prefix_argmax = pa
        rows = np.arange(self.n)[:, None]
        self.prefix_lab = {k: np.asarray(v)[rows, pa] for k, v in self.labs.items()}


def build_fused_view(P, L):
    """Stopping signal = the LoRA verifier's raw scores; pick signal = the FUSED selector."""
    S = V.rank_rows(P["inc"]) + V.rank_rows(V.head_rank_slots(P, L, range(8)))
    strong = W._strong_labels()
    n = P["n"]
    sj = np.zeros(n)
    se = np.zeros(n)
    for i, it in enumerate(P["items"]):
        sj[i] = strong[it["ds"]]["judge"][it["idx"]]
        se[i] = strong[it["ds"]]["em"][it["idx"]]
    labs = {"judge": P["judge"].astype(float), "em": P["em"].astype(float)}
    strongs = {"judge": sj, "em": se}
    keys = [(it["ds"], it["idx"]) for it in P["items"]]
    v = SplitView(P["inc"].copy(), S, labs, strongs, P["ds_index"], keys)
    # control: the degenerate design in which the fused rank score is ALSO the box value
    v_bad = W.PoolView(S, labs, strongs, P["ds_index"], keys)
    return v, S, v_bad


def fixed_N_curve(P, v):
    """Accuracy of the fused selector at fixed pool depth N = 1..8, both currencies, per cell."""
    out = {}
    rows = np.arange(P["n"])
    got8 = {cur: v.prefix_lab[cur][rows, 7].astype(int) for cur in ("judge", "em")}
    for N in range(1, 9):
        d = N - 1
        r = {}
        for cur in ("judge", "em"):
            got = v.prefix_lab[cur][rows, d].astype(int)
            rec = (v.labs[cur][:, :N].max(1) == 1)
            r[cur] = dict(
                acc=float(got.mean()),
                oracle_at_N=float(rec.mean()),
                sel_eff=float(got[rec].mean()),
                macro3=float(np.mean([got[P["ds_index"] == j].mean() for j in range(3)])),
                per_cell={OPEN_CELL_NAME[ds]: float(got[P["ds_index"] == j].mean())
                          for j, ds in enumerate(G.EVAL_DS)},
                vs_N8=V.paired_boot(got, got8[cur]),
                vs_greedy7b=(V.paired_boot(got, P["greedy_ok"].astype(int))
                             if cur == "judge" else None))
        out[N] = r
    return out


def run_scenario_view(v, sc, folds, lam_grid=LAMS):
    n = v.n
    res = {li: dict(ok={k: np.zeros(n) for k in v.labs}, N=np.zeros(n), esc=np.zeros(n))
           for li in range(len(lam_grid))}
    for f in range(NFOLD):
        te = folds == f
        tr = ~te
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        xs = v.raw[tr].ravel()
        ys = v.labs["judge"][tr].ravel()
        m = xs > -1e8
        iso.fit(xs[m], ys[m])
        cal = iso.predict(v.raw.ravel()).reshape(v.raw.shape)
        zc = W.zeta_cheap_many(cal[tr].ravel(), lam_grid, c=sc["c_cheap"])
        q = float(v.strongs["judge"][tr].mean())
        for li, lam in enumerate(lam_grid):
            zs = W.zeta_strong(q, float(lam), c=sc["c_strong"]) if sc["strong"] else -np.inf
            r = v.run(cal, float(zc[li]), float(zs))
            res[li]["N"][te] = r["N"][te]
            res[li]["esc"][te] = r["esc"][te]
            for k in v.labs:
                res[li]["ok"][k][te] = r["ok"][k][te]
    return res


def main():
    os.makedirs(PARTS, exist_ok=True)
    c = V.cost_constants()
    SC = scenarios(c)
    P = V.load_pool()
    L = V.head_logits(P)
    v, S, v_bad = build_fused_view(P, L)

    # NULL TEST: depth-8 pick of this view == the frozen selector's published endpoint
    got8 = v.prefix_lab["judge"][np.arange(P["n"]), 7]
    r8 = V.evaluate(P, V.picks_of(S))
    nt = {"NT_depth8_equals_frozen_selector":
          {"acc_from_view": float(got8.mean()), "acc_from_argmax": r8["judge"]["acc"],
           "published": 0.507463,
           "max_abs_deviation": float(max(abs(got8.mean() - 0.507463),
                                          abs(r8["judge"]["acc"] - 0.507463))),
           "pass": bool(abs(got8.mean() - 0.507463) < 1e-5)},
          "NT_frozen_metric": G.null_test()}
    print("null:", nt["NT_depth8_equals_frozen_selector"], flush=True)

    curve = fixed_N_curve(P, v)
    print("\n=== fixed-N with the FUSED selector ===")
    for N in range(1, 9):
        j, e = curve[N]["judge"], curve[N]["em"]
        print(f"  N={N}: accJ {j['acc']:.6f} (macro3 {j['macro3']:.6f})  accEM {e['acc']:.6f}  "
              f"vsN8 J {j['vs_N8']['delta']:+.6f} [{j['vs_N8']['lo']:+.6f},{j['vs_N8']['hi']:+.6f}]",
              flush=True)

    keys = [(it["ds"], it["idx"]) for it in P["items"]]
    arms = {}
    for sname, sc in SC.items():
        per_seed = []
        for s in CV_SEEDS:
            folds = W.image_folds_for_keys(s, keys, NFOLD)
            per_seed.append(run_scenario_view(v, sc, folds))
        agg = []
        for li in range(len(LAMS)):
            meanN = float(np.mean([p[li]["N"].mean() for p in per_seed]))
            esc = float(np.mean([p[li]["esc"].mean() for p in per_seed]))
            agg.append(dict(
                lam=float(LAMS[li]), meanN=meanN, esc=esc,
                meanN_sd=float(np.std([p[li]["N"].mean() for p in per_seed], ddof=1)),
                acc_judge=float(np.mean([p[li]["ok"]["judge"].mean() for p in per_seed])),
                acc_judge_sd=float(np.std([p[li]["ok"]["judge"].mean() for p in per_seed], ddof=1)),
                acc_em=float(np.mean([p[li]["ok"]["em"].mean() for p in per_seed])),
                macro3_judge=float(np.mean([np.mean([p[li]["ok"]["judge"][P["ds_index"] == j].mean()
                                                     for j in range(3)]) for p in per_seed])),
                per_cell_judge={OPEN_CELL_NAME[ds]: float(np.mean(
                    [p[li]["ok"]["judge"][P["ds_index"] == j].mean() for p in per_seed]))
                    for j, ds in enumerate(G.EVAL_DS)},
                policy_flopeq=meanN * sc["c_cheap"] + (esc * sc["c_strong"] if sc["strong"] else 0.0)))
        bi = int(np.argmax([a["acc_judge"] for a in agg]))
        arms[sname] = dict(scenario=sc["what"], c_cheap=sc["c_cheap"], c_strong=sc["c_strong"],
                           frontier=agg, argmax_acc_judge=bi, best=agg[bi])
        b = agg[bi]
        print(f"  {sname:20s} best lam={b['lam']:.5f} meanN={b['meanN']:.3f} esc={b['esc']:.3f} "
              f"accJ={b['acc_judge']:.6f} accEM={b['acc_em']:.6f}", flush=True)

    # ---- the degenerate control: fused rank score used AS the Weitzman box value -------------
    bad = {}
    for sname in ("W3_free_head_shared", "W5_7b_only"):
        sc = SC[sname]
        ps = [run_scenario_view(v_bad, sc, W.image_folds_for_keys(s, keys, NFOLD))
              for s in CV_SEEDS]
        agg = [dict(lam=float(LAMS[li]),
                    meanN=float(np.mean([p[li]["N"].mean() for p in ps])),
                    acc_judge=float(np.mean([p[li]["ok"]["judge"].mean() for p in ps])))
               for li in range(len(LAMS))]
        bi = int(np.argmax([a["acc_judge"] for a in agg]))
        bad[sname] = dict(best=agg[bi],
                          _why="the fused rank score is a WITHIN-QUESTION rank sum and is not "
                               "comparable across questions, so it cannot be a Weitzman box value")
    print("  [degenerate control] fused-score-as-box-value:",
          {k: (round(x['best']['meanN'], 3), round(x['best']['acc_judge'], 6))
           for k, x in bad.items()}, flush=True)

    json.dump(dict(null_tests=nt, cost_constants=c, scenarios=SC,
                   fixed_N_curve=curve, arms=arms,
                   degenerate_control_fused_as_box_value=bad,
                   cv_seeds=CV_SEEDS, nfold=NFOLD,
                   pool="the FROZEN deployed T=0.7 pool (2,345 questions), fused selector",
                   currency_note="judge labels from the transfer dumps; EM labels from "
                                 "_verifier_hparams_parts/em_slots.npz on identical picks"),
              open(os.path.join(PARTS, "weitzman_frozen.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(PARTS, "weitzman_frozen.json"), flush=True)


if __name__ == "__main__":
    main()
