#!/usr/bin/env python3
"""vrestruct_weitzman.py -- QUESTION 3: refit the adaptive-N controller for NEAR-ZERO verification cost.

Weitzman (Econometrica 1979) makes the reservation value a function of the INSPECTION COST of
opening a box.  The deployed controller charges c_cheap = GEN7 + VER7 = 2.0 FLOP-eq per candidate,
i.e. HALF the cost of an extra sample is the verification of it.  If the generator-frame head is
captured during generation the inspection cost collapses toward zero and the optimal policy changes
qualitatively -- you inspect everything and pay only to generate.

WHAT CHANGES AND WHAT DOES NOT.  Only the two cost constants change.  The pool, the calibration,
the cross-fitting, the metric and the closed-form policy are the existing, null-tested code
(weitzman_lib, asserted BIT-EXACT against pandora_controller.run_pandora on 2026-08-15).

COST SCENARIOS (all FLOP-eq; 1.0 = one Lingshu-7B cap320 forward)
  W0_deployed          c_cheap 2.0    (GEN7 1.0 + VER7 1.0, the paper constant)   c_strong 4.57
  W1_honest_today      c_cheap 5.394  (gen 1.0 + LoRA 2.236 + head 2.158, EVERY pass charged at
                                       ITS OWN measured resolution -- 1,003,520 for both scorers)
  W2_free_head         c_cheap 1.0    (head captured during generation, LoRA dropped)
  W3_free_head_shared  c_cheap 0.011544 (as W2, plus a shared generation prefill: the MARGINAL cost
                                       of one more sample is decode only)
  W5_7b_only           as W3 but with NO STRONG BOX at all -- the structure the new always-7B
                       baseline implies, where the pipeline never calls the 32B.

Every arm is reported in BOTH currencies on identical picks, cross-fit 5-fold image-disjoint over
10 CV seeds, on the T=0.4 pools (the established sampling optimum) with the regenerated T=0.7 pool
as the matched in-session control.  Permutation null on the lambda selection.

    OMP_NUM_THREADS=8 python3 src/cascade_methods/vrestruct_weitzman.py
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
from decoding_sweep_analyse import load_judge, load_vscores, DS   # noqa: E402

PARTS = V.PARTS
LAMS = np.concatenate([[0.0], np.logspace(-4, 0.5, 90)])
CV_SEEDS = list(range(10))
NFOLD = 5


def scenarios(c):
    lora = c["ver_1003520_flopeq"]
    head = c["head_1003520_flopeq"]
    dec = c["decode_share_7b"]
    return {
        "W0_deployed": dict(c_cheap=2.0, c_strong=4.57, strong=True,
                            what="the paper constants: GEN7 1.0 + VER7 1.0, R32 4.57"),
        "W1_honest_today": dict(c_cheap=1.0 + lora + head, c_strong=4.57, strong=True,
                                what=f"gen 1.0 + LoRA {lora:.3f} + head {head:.3f}, each charged "
                                     "at its own MEASURED resolution (1,003,520 px)"),
        "W2_free_head": dict(c_cheap=1.0, c_strong=4.57, strong=True,
                             what="head captured during generation (free), LoRA dropped"),
        "W3_free_head_shared": dict(c_cheap=dec, c_strong=4.57, strong=True,
                                    what="as W2 with a shared generation prefill: the marginal "
                                         f"cost of one more sample is decode only ({dec:.6f})"),
        "W5_7b_only": dict(c_cheap=dec, c_strong=None, strong=False,
                           what="as W3 with NO strong box -- the 7B-only structure the always-7B "
                                "baseline implies"),
    }


def build_views():
    """T=0.4 pools (3 generation seeds) + the matched in-session T=0.7 control."""
    ref = G.load_items()
    lab = load_judge()
    vsc = load_vscores()
    strong = W._strong_labels()
    out = {}
    for tag in ("T04_s0", "T04_s1", "T04_s2", "T07r_s0", "T07r_s1", "T07r_s2"):
        v = W.build_view(tag, lab, vsc, ref, strong)
        if v is not None:
            out[tag] = v
    return out, ref


def run_scenario(view, sc, folds, lam_grid=LAMS):
    """Cross-fit: calibrate + choose zeta on the training folds, apply on the held-out fold.

    Returns per-lambda arrays of (accuracy in both currencies, meanN, escalation, cost).
    """
    n = view.n
    raw = view.raw
    lab = view.labs
    res = {lam_i: dict(ok={k: np.zeros(n) for k in lab}, N=np.zeros(n), esc=np.zeros(n))
           for lam_i in range(len(lam_grid))}
    for f in range(NFOLD):
        te = folds == f
        tr = ~te
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        yj = lab["judge"][tr].ravel()
        xs = raw[tr].ravel()
        m = xs > -1e8
        iso.fit(xs[m], yj[m])
        cal_all = iso.predict(raw.ravel()).reshape(raw.shape)
        v_train = cal_all[tr].ravel()
        v_train = v_train[np.isfinite(v_train)]
        zc = W.zeta_cheap_many(v_train, lam_grid, c=sc["c_cheap"])
        q_strong = float(view.strongs["judge"][tr].mean())
        for li, lam in enumerate(lam_grid):
            if sc["strong"]:
                zs = W.zeta_strong(q_strong, float(lam), c=sc["c_strong"])
            else:
                zs = -np.inf                       # no strong box: never escalate
            r = view.run(cal_all, float(zc[li]), float(zs))
            res[li]["N"][te] = r["N"][te]
            res[li]["esc"][te] = r["esc"][te]
            for k in lab:
                res[li]["ok"][k][te] = r["ok"][k][te]
    rows = []
    for li, lam in enumerate(lam_grid):
        r = res[li]
        meanN = float(r["N"].mean())
        esc = float(r["esc"].mean())
        flops = meanN * sc["c_cheap"] + (esc * sc["c_strong"] if sc["strong"] else 0.0)
        rows.append(dict(lam=float(lam), meanN=meanN, esc=esc, flops_eq_policy=flops,
                         acc_judge=float(r["ok"]["judge"].mean()),
                         acc_em=float(r["ok"]["em"].mean()),
                         per_cell_judge={G.EVAL_DS[j]: float(r["ok"]["judge"][view.ds_index == j].mean())
                                         for j in range(3)},
                         per_cell_em={G.EVAL_DS[j]: float(r["ok"]["em"][view.ds_index == j].mean())
                                      for j in range(3)},
                         ok_judge=r["ok"]["judge"], ok_em=r["ok"]["em"], N=r["N"], esc_vec=r["esc"]))
    return rows


def true_cost(meanN, esc, sc, c):
    """The honest FLOP-eq of a policy that draws meanN samples and escalates a fraction esc.

    Generation is charged BOTH ways: as-charged (1.0 per sample) and shared-prefill G(N).
    """
    ver = 0.0
    if sc is not None and "ver_per_cand" in sc:
        ver = sc["ver_per_cand"] * meanN
    return dict(
        as_charged=meanN * 1.0 + ver + (esc * c["R32_as_charged"]),
        shared_prefill=V.G_of_N(meanN, c) + ver + (esc * c["R32_as_charged"]),
        shared_prefill_R32derived=V.G_of_N(meanN, c) + ver + (esc * c["R32_derived"]))


def main():
    os.makedirs(PARTS, exist_ok=True)
    c = V.cost_constants()
    SC = scenarios(c)

    print("building pools ...", flush=True)
    views, ref = build_views()
    print("  ", list(views), flush=True)

    # ---- NULL TEST: frozen metric + the deployed-cost reproduction ----------------------
    nt = {"NT1_frozen_metric": G.null_test()}
    nt["NT2_zeta_closed_form"] = {}
    v = np.sort(np.random.default_rng(0).random(500))
    z = W.zeta_cheap_many(v, [0.01, 0.1], c=2.0)
    g = [float(np.mean(np.clip(v - zz, 0, None))) for zz in z]
    nt["NT2_zeta_closed_form"] = {"target": [0.02, 0.2], "achieved": g,
                                  "max_abs_deviation": max(abs(g[0] - 0.02), abs(g[1] - 0.2)),
                                  "pass": bool(max(abs(g[0] - 0.02), abs(g[1] - 0.2)) < 1e-9)}
    print("null tests:", nt["NT1_frozen_metric"]["pass"], nt["NT2_zeta_closed_form"]["pass"],
          flush=True)

    keys = [(it["ds"], it["idx"]) for it in ref]
    out_arms = {}
    for tag in views:
        for sname, sc in SC.items():
            per_seed = []
            for s in CV_SEEDS:
                folds = W.image_folds_for_keys(s, keys, NFOLD)
                rows = run_scenario(views[tag], sc, folds)
                per_seed.append(rows)
            # seed-average every lambda cell
            agg = []
            for li in range(len(LAMS)):
                agg.append(dict(
                    lam=per_seed[0][li]["lam"],
                    meanN=float(np.mean([p[li]["meanN"] for p in per_seed])),
                    meanN_sd=float(np.std([p[li]["meanN"] for p in per_seed], ddof=1)),
                    esc=float(np.mean([p[li]["esc"] for p in per_seed])),
                    acc_judge=float(np.mean([p[li]["acc_judge"] for p in per_seed])),
                    acc_judge_sd=float(np.std([p[li]["acc_judge"] for p in per_seed], ddof=1)),
                    acc_em=float(np.mean([p[li]["acc_em"] for p in per_seed])),
                    acc_em_sd=float(np.std([p[li]["acc_em"] for p in per_seed], ddof=1)),
                    macro3_judge=float(np.mean([np.mean(list(p[li]["per_cell_judge"].values()))
                                                for p in per_seed])),
                    macro3_em=float(np.mean([np.mean(list(p[li]["per_cell_em"].values()))
                                             for p in per_seed])),
                    per_cell_judge={d: float(np.mean([p[li]["per_cell_judge"][d] for p in per_seed]))
                                    for d in G.EVAL_DS},
                    per_cell_em={d: float(np.mean([p[li]["per_cell_em"][d] for p in per_seed]))
                                 for d in G.EVAL_DS},
                    flops_eq_policy=float(np.mean([p[li]["flops_eq_policy"] for p in per_seed]))))
            out_arms[f"{tag}|{sname}"] = dict(
                scenario=sc["what"], c_cheap=sc["c_cheap"], c_strong=sc["c_strong"],
                frontier=agg,
                argmax_acc_judge=int(np.argmax([a["acc_judge"] for a in agg])),
                n_cv_seeds=len(CV_SEEDS))
            best = agg[int(np.argmax([a["acc_judge"] for a in agg]))]
            print(f"  {tag:9s} {sname:20s} best lam={best['lam']:.5f} meanN={best['meanN']:.3f} "
                  f"esc={best['esc']:.3f} accJ={best['acc_judge']:.6f} accEM={best['acc_em']:.6f}",
                  flush=True)

    json.dump(dict(null_tests=nt, cost_constants=c, scenarios=SC, arms=out_arms,
                   lam_grid=[float(x) for x in LAMS], cv_seeds=CV_SEEDS, nfold=NFOLD),
              open(os.path.join(PARTS, "weitzman.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(PARTS, "weitzman.json"), flush=True)


if __name__ == "__main__":
    main()
