#!/usr/bin/env python3
"""
macro_average_headline.py -- RE-BASE every headline comparison on MACRO-AVERAGING (equal weight per
dataset) instead of the pooled sample-weighted average.

WHY.  The project's headlines are pooled sample-weighted averages over the Variant-B pool
(MMMU excluded; 8 reporting cells, n=42,224).  PMC-VQA alone is 33,430 of those items = 79.2% of the
weight, so "pooled accuracy" is very nearly "PMC-VQA accuracy".  The researcher's decision: every
dataset counts as ONE data point.

PRIMARY granularity = the 8 REPORTING CELLS, 1/8 each (PMC-VQA, SLAKE-closed, VQA-RAD-closed,
PathVQA-closed, MedXpert, SLAKE-open, VQA-RAD-open, PathVQA-open).  The reweighting this implies:
    PMC-VQA          79.2% of the weight  ->  12.5%
    open-text arm     5.6% of the items   ->  37.5%  (3 of 8 cells)
    closed/MCQ arm   94.4% of the items   ->  62.5%  (5 of 8 cells)
SECONDARY (robustness only) = the 5 BENCHMARKS, 1/5 each, where a benchmark's closed and open cells
are first averaged into one number (SLAKE / VQA-RAD / PathVQA each have two cells).  A second variant
weights the two cells by their item counts inside the benchmark.

THE DIRECTION IS NOT UNIFORM -- both halves are reported here:
  * vs always-32B-WITH-REASONING the macro delta is LARGER than the sample-weighted one, because the
    3 open-text cells (where the 32B's reasoning mode collapses) stop being drowned by PMC-VQA.
  * on the MULTIPLE-CHOICE-only cells the compute-lean operating point becomes a SIGNIFICANT LOSS
    against always-32B-direct and against oracle-mode-32B: under equal weight per benchmark the MCQ
    arm is significantly WORSE than simply always running the strong model.  This appears in no
    current headline.  It is reported here unsoftened.

STATISTICS.  A macro CI is NOT a pooled CI.  Each bootstrap replicate resamples ITEMS WITHIN each cell
(paired across systems, common random numbers) and the macro average is recomputed on the replicate.
The resampling is done EXACTLY by drawing multinomial counts over the unique per-item outcome PATTERNS
across all 7 systems -- items sharing a pattern are exchangeable, so this is distributionally identical
to resampling items, and it makes all four weightings share one replicate stream.
    ==> The resulting CI reflects WITHIN-dataset sampling noise ONLY.  It does NOT capture
        dataset-selection noise.  With 5-8 datasets, resampling datasets would be hopelessly unstable
        (a single draw can duplicate PathVQA-open three times), so it is not attempted.  Instead a
        leave-one-cell-out range is reported for every headline comparison, which is the honest way to
        show how much a macro number leans on one dataset.

SYSTEMS (all per-sample, held-out 5-fold cross-fit, reconstructed by paper_baselines.build_cells):
    always_7b                     cheap floor
    always_32b_direct             one 32B no-think forward         (the DEPLOYABLE strong baseline)
    always_32b_reasoning          32B reasoning mode everywhere    (MCQ real okT; PATH_VQA_closed has
                                  no reasoning dump -> = direct; OPEN = MEASURED judged think)
    oracle_mode_32b               per-benchmark best of {reasoning, direct}; the honest strong baseline
    method_compute_lean           margin cascade MCQ | Pandora bo-N + verifier open
    method_accuracy_max_veto      F8 certified weak-veto on PMC | F10 team-objective L2D open (0.93x)
    method_accuracy_max_fusion    F3 confidence-advantage fusion on PMC | Pandora open (1.25x)

COST.  Macro-averaged FLOP-eq / batch-1 latency / energy are reported next to accuracy, under BOTH
(a) the paper's as-charged constants and (b) the honest per-cell re-costing of the reasoning baseline
(honest_recosting.py M1_primary), because cost per dataset differs and the equal-weighted cost is
therefore not the pooled cost.

CPU only, no GPU, no new inference.  Launch from the repo root:
    python3 src/cascade_methods/macro_average_headline.py
"""
import os, json
import numpy as np

import paper_baselines as PB                 # build_cells (per-sample vectors + costs), cost helpers
import method_final_mmmu_corrected as MFC    # add_v2_vectors -> am2_ok (F8 veto PMC + F10 L2D open)
import opentext_32b_think_full as OTF        # measured per-sample judged 32B-reasoning on open text
import honest_recosting as HR                # per-cell measured reasoning generation lengths + cost model

ROOT = PB.ROOT
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json")

MMMU     = "MMMU-Medical-val"
ORDER_9  = PB.MCQ_ORDER + PB.OPEN_ORDER
ORDER_B  = [k for k in ORDER_9 if k != MMMU]                        # 8 cells, n=42,224
MCQ_B    = [k for k in PB.MCQ_ORDER if k != MMMU]                   # 5 closed/MCQ cells
OPEN_B   = list(PB.OPEN_ORDER)                                      # 3 open-text cells

BENCH_OF = {"PMC_VQA": "PMC-VQA", "SLAKE_closed": "SLAKE", "SLAKE_open": "SLAKE",
            "VQA_RAD_closed": "VQA-RAD", "VQA_RAD_open": "VQA-RAD",
            "PATH_VQA_closed": "PathVQA", "PATH_VQA_open": "PathVQA",
            "MedXpertQA-MM": "MedXpert", MMMU: "MMMU"}
BENCH_ORDER = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MedXpert"]

SYS_KEY = {"always_7b": "ok7", "always_32b_direct": "ok32", "always_32b_reasoning": "okT",
           "oracle_mode_32b": "oracle_ok", "method_compute_lean": "cl_ok",
           "method_accuracy_max_veto": "am2_ok", "method_accuracy_max_fusion": "am_ok"}
SYSTEMS   = list(SYS_KEY)
METHODS   = ["method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
BASELINES = ["always_32b_reasoning", "always_32b_direct", "oracle_mode_32b", "always_7b"]
HEADLINE_BASELINES = ["always_32b_reasoning", "always_32b_direct", "oracle_mode_32b"]

NBOOT = PB.NBOOT          # 10000
SEED  = 20260730


# ===================================================================================================
# 1.  CELLS -- per-sample vectors for every system (held-out, reconstructed, no GPU)
# ===================================================================================================
def build():
    cells = PB.build_cells()                  # cl_ok / am_ok / oracle_ok / ok7 / ok32 / okT + costs
    MFC.add_v2_vectors(cells)                 # am2_ok (F8 certified veto on PMC, F10 L2D on open)
    meas_open = OTF.measured_open_think()     # MEASURED judged per-sample 32B-reasoning on open text
    for name in PB.OPEN_ORDER:
        c = cells[name]; mk = meas_open[name]
        assert c["n"] == mk["n"], (name, c["n"], mk["n"])
        c["okT"] = mk["okT"]                  # was None (estimate); now a real per-sample vector
        c["think_real"] = True
    # every system must now be a real per-sample vector on every Variant-B cell
    for k in ORDER_B:
        for s in SYSTEMS:
            v = cells[k][SYS_KEY[s]]
            assert v is not None and len(v) == cells[k]["n"], (k, s)
    return cells


# ===================================================================================================
# 2.  WEIGHTINGS -- every scheme is just a weight vector over the 8 cells
# ===================================================================================================
def weightings(cells, order):
    N = sum(cells[k]["n"] for k in order)
    W = {}
    W["sample_weighted"] = {k: cells[k]["n"] / N for k in order}
    W["macro_cells"]     = {k: 1.0 / len(order) for k in order}
    benches = {}
    for k in order:
        benches.setdefault(BENCH_OF[k], []).append(k)
    nb = len(benches)
    W["macro_benchmarks_cellavg"] = {k: (1.0 / nb) / len(ks) for b, ks in benches.items() for k in ks}
    W["macro_benchmarks_itemweighted"] = {
        k: (1.0 / nb) * cells[k]["n"] / sum(cells[j]["n"] for j in ks)
        for b, ks in benches.items() for k in ks}
    for lab, w in W.items():
        assert abs(sum(w.values()) - 1.0) < 1e-12, lab
    return W, benches


# ===================================================================================================
# 3.  BOOTSTRAP -- resample ITEMS WITHIN each cell, recompute the weighted average per replicate
# ===================================================================================================
def cell_boot_means(mat, nboot, rng):
    """mat: (n_items, n_systems) 0/1 matrix for ONE cell.  Returns (nboot, n_systems) of bootstrap
    resampled per-system means, PAIRED (one common resample per replicate, shared by all systems).

    Drawn exactly: items with an identical outcome pattern across the systems are exchangeable, so the
    vector of pattern counts under the item bootstrap is Multinomial(n, empirical pattern freqs), and
    every system's resampled mean is a linear function of those counts.  Distributionally identical to
    gathering n resampled rows, at a tiny fraction of the cost."""
    pats, cnt = np.unique(mat, axis=0, return_counts=True)
    n = mat.shape[0]
    draws = rng.multinomial(n, cnt / n, size=nboot)             # (nboot, n_patterns)
    return (draws @ pats) / n


def cell_boot_means_items(mat, nboot, rng, chunk=None):
    """Reference implementation: literally gather nboot x n resampled rows.  Used once, as a
    verification that the multinomial-pattern shortcut above gives the same CI."""
    n, S = mat.shape
    if chunk is None:
        chunk = max(20, int(2.0e7 // max(n, 1)))
    out = np.empty((nboot, S))
    for s in range(0, nboot, chunk):
        m = min(chunk, nboot - s)
        idx = rng.integers(0, n, size=(m, n))
        out[s:s + m] = mat[idx].mean(axis=1)
    return out


def ci(dist, point=None):
    lo, hi = float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))
    d = float(np.mean(dist)) if point is None else float(point)
    return dict(delta=round(d, 4), lo=round(lo, 4), hi=round(hi, 4),
                sig=bool(lo > 0 or hi < 0),
                verdict=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"))


def wavg(per_cell_value, w):
    return sum(per_cell_value[k] * w[k] for k in w)


# ===================================================================================================
# 4.  COST -- as-charged (paper constants) and honestly re-costed (per-cell reasoning length)
# ===================================================================================================
def cost_blocks(cells, order):
    """Per-cell (system -> cost dict) under two costing conventions."""
    K = HR.verify_constants()
    meas = HR.measure_reasoning_lengths()
    model = HR.cost_models(K)["M1_primary"]

    as_charged, honest = {}, {}
    for k in order:
        c = cells[k]
        base = {
            "always_7b":            PB.cost_fixed(PB.GEN7),
            "always_32b_direct":    PB.cost_fixed(PB.GEN32N),
            "always_32b_reasoning": PB.cost_fixed(PB.GEN32T),
            "oracle_mode_32b":      dict(c["oracle_cost"]),
            "method_compute_lean":        dict(c["cl_cost"]),
            "method_accuracy_max_veto":   dict(c["am2_cost"]),
            "method_accuracy_max_fusion": dict(c["am_cost"]),
        }
        as_charged[k] = base
        rc = HR.cell_baseline_cost(k, meas, K, model, "M1_primary")
        recost = dict(flops=rc["flops"], energy=rc["energy_j"], lat_seq=rc["lat_ms"], lat_par=rc["lat_ms"])
        h = {s: dict(v) for s, v in base.items()}
        h["always_32b_reasoning"] = recost
        if c["oracle_mode"] == "think":          # the oracle also pays the reasoning cost where it picks it
            h["oracle_mode_32b"] = dict(recost)
        honest[k] = h
    prov = dict(
        decode_ms_per_tok=K["decode_ms_per_tok"], decode_j_per_tok=K["decode_j_per_tok"],
        prefill_ms=K["prefill_ms"], prefill_j=K["prefill_j"],
        as_charged_reasoning_constant=dict(lat_ms=PB.GEN32T[0], energy_j=PB.GEN32T[1], flops=PB.GEN32T[2]),
        per_cell_measured_reasoning_gen_tok={k: meas[k]["gen_reasoning"]["mean"] if meas[k]["gen_reasoning"]
                                            else meas[k]["gen_direct"]["mean"] for k in order},
        per_cell_reasoning_verdict={k: meas[k]["verdict"] for k in order},
        source="honest_recosting.py M1_primary (measured decode rate + 665 ms/126.9 J direct anchor)")
    return as_charged, honest, prov


AXES = [("flops", "flops"), ("lat_par", "lat_par_ms"), ("lat_seq", "lat_seq_ms"), ("energy", "energy_j")]


def cost_table(costs, order, W):
    """weighting -> system -> {flops, lat_par_ms, lat_seq_ms, energy_j}"""
    out = {}
    for lab, w in W.items():
        out[lab] = {}
        for s in SYSTEMS:
            out[lab][s] = {name: round(wavg({k: costs[k][s][ax] for k in order}, w), 3 if ax == "flops" else 1)
                           for ax, name in AXES}
    return out


# ===================================================================================================
# 5.  RUN
# ===================================================================================================
def run():
    cells = build()
    W, benches = weightings(cells, ORDER_B)
    order = ORDER_B
    N_B = sum(cells[k]["n"] for k in order)

    # ---------------- per-cell accuracy + one bootstrap replicate stream per cell -------------------
    rng = np.random.default_rng(SEED)
    acc_cell = {k: {s: float(np.asarray(cells[k][SYS_KEY[s]], float).mean()) for s in SYSTEMS} for k in order}
    B = {}
    for k in order:
        mat = np.column_stack([np.asarray(cells[k][SYS_KEY[s]], float) for s in SYSTEMS])
        B[k] = cell_boot_means(mat, NBOOT, rng)                      # (NBOOT, n_systems)
    SI = {s: i for i, s in enumerate(SYSTEMS)}

    # verification of the pattern-multinomial shortcut against literal item resampling (2 cells)
    rng_v = np.random.default_rng(SEED + 1)
    verify = {}
    for k in ("PMC_VQA", "VQA_RAD_open"):
        mat = np.column_stack([np.asarray(cells[k][SYS_KEY[s]], float) for s in SYSTEMS])
        bi = cell_boot_means_items(mat, NBOOT, rng_v)
        d_pat = B[k][:, SI["method_compute_lean"]] - B[k][:, SI["always_32b_direct"]]
        d_itm = bi[:, SI["method_compute_lean"]] - bi[:, SI["always_32b_direct"]]
        verify[k] = dict(pattern_shortcut_ci=[round(float(np.percentile(d_pat, 2.5)), 5),
                                              round(float(np.percentile(d_pat, 97.5)), 5)],
                         literal_item_resample_ci=[round(float(np.percentile(d_itm, 2.5)), 5),
                                                   round(float(np.percentile(d_itm, 97.5)), 5)],
                         n_unique_patterns=int(np.unique(mat, axis=0).shape[0]), n_items=cells[k]["n"])

    # ---------------- accuracy LEVELS under every weighting ----------------------------------------
    acc_levels = {lab: {s: round(wavg({k: acc_cell[k][s] for k in order}, w), 4) for s in SYSTEMS}
                  for lab, w in W.items()}
    # sub-pools (MCQ-only / open-only) under sample-weighted and macro
    def sub_levels(keys):
        n = sum(cells[k]["n"] for k in keys)
        sw = {k: cells[k]["n"] / n for k in keys}
        mc = {k: 1.0 / len(keys) for k in keys}
        return dict(n=int(n), n_cells=len(keys),
                    sample_weighted={s: round(wavg({k: acc_cell[k][s] for k in keys}, sw), 4) for s in SYSTEMS},
                    macro_cells={s: round(wavg({k: acc_cell[k][s] for k in keys}, mc), 4) for s in SYSTEMS})
    acc_subpools = dict(mcq_only=sub_levels(MCQ_B), open_only=sub_levels(OPEN_B))

    # ---------------- deltas: sample-weighted vs macro, every method x every baseline ---------------
    def delta_block(keys, m, b):
        """All weighting schemes for one (method, baseline) pair over `keys`."""
        n = sum(cells[k]["n"] for k in keys)
        schemes = {"sample_weighted": {k: cells[k]["n"] / n for k in keys},
                   "macro_cells": {k: 1.0 / len(keys) for k in keys}}
        if set(keys) == set(order):                                  # bench schemes only for the full pool
            schemes["macro_benchmarks_cellavg"] = W["macro_benchmarks_cellavg"]
            schemes["macro_benchmarks_itemweighted"] = W["macro_benchmarks_itemweighted"]
        else:
            bb = {}
            for k in keys: bb.setdefault(BENCH_OF[k], []).append(k)
            schemes["macro_benchmarks_cellavg"] = {k: (1.0 / len(bb)) / len(ks) for _, ks in bb.items() for k in ks}
            schemes["macro_benchmarks_itemweighted"] = {
                k: (1.0 / len(bb)) * cells[k]["n"] / sum(cells[j]["n"] for j in ks)
                for _, ks in bb.items() for k in ks}
        out = {}
        dcell = {k: B[k][:, SI[m]] - B[k][:, SI[b]] for k in keys}
        pcell = {k: acc_cell[k][m] - acc_cell[k][b] for k in keys}
        for lab, w in schemes.items():
            dist = sum(dcell[k] * w[k] for k in keys)
            out[lab] = ci(dist, point=wavg(pcell, w))
            out[lab]["n_units"] = (len(set(BENCH_OF[k] for k in keys)) if lab.startswith("macro_benchmarks")
                                   else (len(keys) if lab == "macro_cells" else int(n)))
        # unstratified pooled bootstrap (the repo's published protocol) for the sample-weighted row
        big = np.column_stack([np.concatenate([np.asarray(cells[k][SYS_KEY[s]], float) for k in keys])
                               for s in (m, b)])
        rr = np.random.default_rng(SEED + 7)
        bb2 = cell_boot_means(big, NBOOT, rr)
        out["sample_weighted_pooled_unstratified"] = ci(bb2[:, 0] - bb2[:, 1], point=wavg(pcell, schemes["sample_weighted"]))
        out["sample_weighted_pooled_unstratified"]["n_units"] = int(n)
        # per-cell deltas with per-cell CIs
        out["per_cell"] = {k: ci(dcell[k], point=pcell[k]) for k in keys}
        # leave-one-cell-out range of the MACRO (equal-weight) delta -- dataset-leverage diagnostic
        loo = {k: round(float(np.mean([pcell[j] for j in keys if j != k])), 4) for k in keys}
        worst = min(loo, key=lambda k: loo[k]); best = max(loo, key=lambda k: loo[k])
        out["macro_cells_leave_one_out"] = dict(
            per_dropped_cell=loo, range=[loo[worst], loo[best]],
            cell_carrying_the_claim=worst, cell_holding_the_claim_back=best,
            note="macro delta recomputed with each cell removed in turn (point estimates only); shows how "
                 "much the equal-weight number leans on any single dataset. cell_carrying_the_claim is the "
                 "one whose removal LOWERS the macro delta the most (the load-bearing dataset); "
                 "cell_holding_the_claim_back is the one whose removal raises it the most.")
        return out

    deltas = {}
    for m in METHODS:
        deltas[m] = {}
        for b in BASELINES:
            deltas[m][b] = dict(all8=delta_block(order, m, b),
                                mcq_only=delta_block(MCQ_B, m, b),
                                open_only=delta_block(OPEN_B, m, b))

    # ---------------- cost under the same weightings ------------------------------------------------
    cost_ac, cost_hon, cost_prov = cost_blocks(cells, order)
    cost = dict(as_charged=cost_table(cost_ac, order, W), honest_recost=cost_table(cost_hon, order, W),
                per_cell_as_charged={k: {s: {name: round(cost_ac[k][s][ax], 3) for ax, name in AXES}
                                         for s in SYSTEMS} for k in order},
                per_cell_honest_recost={k: {s: {name: round(cost_hon[k][s][ax], 3) for ax, name in AXES}
                                            for s in SYSTEMS} for k in order},
                provenance=cost_prov)
    # method-vs-baseline cost ratios under each weighting / convention
    ratios = {}
    for conv in ("as_charged", "honest_recost"):
        ratios[conv] = {}
        for lab in W:
            ratios[conv][lab] = {}
            for m in METHODS:
                ratios[conv][lab][m] = {}
                for b in HEADLINE_BASELINES:
                    mv, bv = cost[conv][lab][m], cost[conv][lab][b]
                    ratios[conv][lab][m][b] = {
                        "flops_x": round(mv["flops"] / bv["flops"], 3),
                        "lat_par_x": round(mv["lat_par_ms"] / bv["lat_par_ms"], 3),
                        "lat_par_pct": round(-100 * (1 - mv["lat_par_ms"] / bv["lat_par_ms"]), 1),
                        "lat_seq_x": round(mv["lat_seq_ms"] / bv["lat_seq_ms"], 3),
                        "lat_seq_pct": round(-100 * (1 - mv["lat_seq_ms"] / bv["lat_seq_ms"]), 1),
                        "energy_x": round(mv["energy_j"] / bv["energy_j"], 3),
                        "energy_pct": round(-100 * (1 - mv["energy_j"] / bv["energy_j"]), 1)}
    cost["method_vs_baseline_ratios"] = ratios

    # ---------------- Pareto frontier under each weighting (accuracy vs each cost axis) -------------
    pareto = {}
    for conv in ("as_charged", "honest_recost"):
        pareto[conv] = {}
        for lab in W:
            pts = [dict(system=s, acc=acc_levels[lab][s], **cost[conv][lab][s]) for s in SYSTEMS]
            def mark(axis):
                return sorted(p["system"] for p in pts if not any(
                    q[axis] <= p[axis] and q["acc"] >= p["acc"] and q is not p and
                    (q[axis] < p[axis] or q["acc"] > p["acc"]) for q in pts))
            pareto[conv][lab] = dict(points=pts, pareto_optimal_flops=mark("flops"),
                                     pareto_optimal_lat_par=mark("lat_par_ms"),
                                     pareto_optimal_lat_seq=mark("lat_seq_ms"),
                                     pareto_optimal_energy=mark("energy_j"))
    cost["pareto"] = pareto

    # ---------------- the JOINT (accuracy AND cost) claim, re-evaluated under macro -----------------
    joint = {}
    for m in METHODS:
        joint[m] = {}
        for b in HEADLINE_BASELINES:
            row = {}
            for lab in ("sample_weighted", "macro_cells"):
                a = deltas[m][b]["all8"][lab]
                r_ac = ratios["as_charged"][lab][m][b]; r_hn = ratios["honest_recost"][lab][m][b]
                row[lab] = dict(acc_delta=a["delta"], acc_ci95=[a["lo"], a["hi"]], acc_verdict=a["verdict"],
                                compute_x_as_charged=r_ac["flops_x"], compute_x_honest=r_hn["flops_x"],
                                lat_par_x_honest=r_hn["lat_par_x"], lat_seq_x_honest=r_hn["lat_seq_x"],
                                energy_x_honest=r_hn["energy_x"],
                                joint=("cheaper AND better" if r_hn["flops_x"] < 1 and a["verdict"] == "WIN"
                                       else "cheaper, same accuracy" if r_hn["flops_x"] < 1 and a["verdict"] == "TIE"
                                       else "cheaper but WORSE" if r_hn["flops_x"] < 1
                                       else "MORE EXPENSIVE but better" if a["verdict"] == "WIN"
                                       else "MORE EXPENSIVE and WORSE" if a["verdict"] == "LOSS"
                                       else "MORE EXPENSIVE, same accuracy"))
            row["changed_by_macro"] = (row["sample_weighted"]["joint"] != row["macro_cells"]["joint"])
            joint[m][b] = row
    cost["joint_claim"] = dict(
        by_pair=joint,
        basis="The 'joint' label classifies cheaper/more-expensive on the HONESTLY RE-COSTED FLOP-eq axis "
              "(compute_x_honest); compute_x_as_charged is carried alongside because the paper's convention "
              "charges the reasoning baseline a flat 4.57 FLOP-eq regardless of how much it generates. "
              "Latency and energy ratios (also honest) are in the same record -- the accuracy/latency/energy "
              "picture and the FLOP-eq picture do NOT agree for the method's open arm, which spends many "
              "small 7B forwards that are cheap in wall-clock but fully counted in FLOP-eq.",
        why="Cost per dataset differs enormously. The 3 open-text cells cost the method 7.6-12.6 FLOP-eq "
            "(adaptive best-of-N: several 7B draws + a verifier forward + escalation) against always-32B-"
            "direct's flat 4.57, while on PMC-VQA the cascade costs only 1.39. Sample-weighted, PMC's 79% "
            "share hides the open arm's cost; at 1/8 per cell the open arm holds 37.5% of the weight.",
        non_dominated_vs_dominates=(
            "The method points are still NON-DOMINATED under macro weighting (cost.pareto): they sit on the "
            "frontier because they are more ACCURATE than the 32B baselines, not because they are cheaper. "
            "'Pareto-optimal' survives; 'Pareto-DOMINATES' (strictly better on every axis) does NOT hold at "
            "equal weight against always-32B-direct or oracle-mode-32B, and against "
            "always-32B-with-reasoning it holds on accuracy / latency / energy but NOT on FLOP-eq."),
        headline=("Under equal weight per cell NO operating point is compute-cheaper than always-32B-direct: "
                  "compute-lean %.3fx, accuracy-max-veto %.3fx, accuracy-max-fusion %.3fx FLOP-eq (they were "
                  "%.3fx / %.3fx / %.3fx sample-weighted). The efficiency claim survives ONLY against a 32B "
                  "actually made to reason."
                  % (ratios["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]["flops_x"],
                     ratios["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
                     ratios["as_charged"]["macro_cells"]["method_accuracy_max_fusion"]["always_32b_direct"]["flops_x"],
                     ratios["as_charged"]["sample_weighted"]["method_compute_lean"]["always_32b_direct"]["flops_x"],
                     ratios["as_charged"]["sample_weighted"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
                     ratios["as_charged"]["sample_weighted"]["method_accuracy_max_fusion"]["always_32b_direct"]["flops_x"])))

    # ---------------- escalation rate: the driver of every cost claim -------------------------------
    esc_cell = {k: (round(float(cells[k]["esc"]), 4) if cells[k].get("esc") is not None else None)
                for k in order}
    esc = dict(
        per_cell=esc_cell,
        compute_lean_mcq=dict(
            sample_weighted=round(sum(esc_cell[k] * cells[k]["n"] for k in MCQ_B) /
                                  sum(cells[k]["n"] for k in MCQ_B), 4),
            macro_cells=round(float(np.mean([esc_cell[k] for k in MCQ_B])), 4)),
        compute_lean_all8=dict(
            sample_weighted=round(sum(esc_cell[k] * cells[k]["n"] for k in order) / N_B, 4),
            macro_cells=round(float(np.mean([esc_cell[k] for k in order])), 4)),
        note="The escalation rate is what drives '0.49x compute' and '469 ms'. It is wildly heterogeneous "
             "across cells, and PMC-VQA -- the cell with by far the lowest escalation -- carries 79.2% of "
             "the sample-weighted average. Equal-weighting the cells raises the multiple-choice escalation "
             "rate from the pooled value to the macro value below, which is the mechanism behind the cost "
             "reversal in cost.joint_claim.")

    # ---------------- side-by-side claim table -----------------------------------------------------
    def row(label, pool, m, b, axis="all8"):
        d = deltas[m][b][axis]
        sw, mac, m5 = d["sample_weighted"], d["macro_cells"], d["macro_benchmarks_cellavg"]
        return dict(claim=label, pool=pool, method=m, baseline=b, granularity_axis=axis,
                    sample_weighted=dict(delta=sw["delta"], ci95=[sw["lo"], sw["hi"]], verdict=sw["verdict"]),
                    macro_8cells=dict(delta=mac["delta"], ci95=[mac["lo"], mac["hi"]], verdict=mac["verdict"]),
                    macro_5benchmarks=dict(delta=m5["delta"], ci95=[m5["lo"], m5["hi"]], verdict=m5["verdict"]),
                    change=("STRONGER under macro" if abs(mac["delta"]) > abs(sw["delta"]) and
                            np.sign(mac["delta"]) == np.sign(sw["delta"])
                            else "FLIPS SIGN under macro" if np.sign(mac["delta"]) != np.sign(sw["delta"])
                            else "WEAKER under macro"),
                    verdict_change=("unchanged (%s)" % sw["verdict"]) if sw["verdict"] == mac["verdict"]
                                    else "%s -> %s" % (sw["verdict"], mac["verdict"]))
    side_by_side = []
    for m in METHODS:
        for b in HEADLINE_BASELINES:
            for axis, pool in [("all8", "all 8 cells / 5 benchmarks"),
                               ("mcq_only", "multiple-choice cells only"),
                               ("open_only", "open-text cells only")]:
                side_by_side.append(row("%s vs %s" % (m, b), pool, m, b, axis))
    # accuracy-level side-by-side
    acc_side = {s: dict(sample_weighted=acc_levels["sample_weighted"][s],
                        macro_8cells=acc_levels["macro_cells"][s],
                        macro_5benchmarks=acc_levels["macro_benchmarks_cellavg"][s],
                        shift_macro8_minus_sw=round(acc_levels["macro_cells"][s] -
                                                    acc_levels["sample_weighted"][s], 4)) for s in SYSTEMS}

    # ---------------- verdicts + honest headline ---------------------------------------------------
    verdicts = {}
    for m in METHODS:
        verdicts[m] = {}
        for b in HEADLINE_BASELINES:
            d = deltas[m][b]
            verdicts[m][b] = dict(
                all8_macro=dict(delta=d["all8"]["macro_cells"]["delta"],
                                ci95=[d["all8"]["macro_cells"]["lo"], d["all8"]["macro_cells"]["hi"]],
                                verdict=d["all8"]["macro_cells"]["verdict"]),
                mcq_only_macro=dict(delta=d["mcq_only"]["macro_cells"]["delta"],
                                    ci95=[d["mcq_only"]["macro_cells"]["lo"], d["mcq_only"]["macro_cells"]["hi"]],
                                    verdict=d["mcq_only"]["macro_cells"]["verdict"]),
                open_only_macro=dict(delta=d["open_only"]["macro_cells"]["delta"],
                                     ci95=[d["open_only"]["macro_cells"]["lo"], d["open_only"]["macro_cells"]["hi"]],
                                     verdict=d["open_only"]["macro_cells"]["verdict"]),
                all8_sample_weighted_for_contrast=dict(
                    delta=d["all8"]["sample_weighted"]["delta"],
                    ci95=[d["all8"]["sample_weighted"]["lo"], d["all8"]["sample_weighted"]["hi"]],
                    verdict=d["all8"]["sample_weighted"]["verdict"]))

    cl = deltas["method_compute_lean"]; am2 = deltas["method_accuracy_max_veto"]
    r_cl_dir8 = cl["always_32b_direct"]["all8"]["macro_cells"]
    r_cl_dirM = cl["always_32b_direct"]["mcq_only"]["macro_cells"]
    r_cl_orcM = cl["oracle_mode_32b"]["mcq_only"]["macro_cells"]
    r_am_dir8 = am2["always_32b_direct"]["all8"]["macro_cells"]
    r_am_rea8 = am2["always_32b_reasoning"]["all8"]["macro_cells"]
    r_cl_rea8 = cl["always_32b_reasoning"]["all8"]["macro_cells"]
    RAT = cost["method_vs_baseline_ratios"]
    fx_cl = RAT["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]
    fx_am = RAT["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_direct"]
    fx_cl_sw = RAT["as_charged"]["sample_weighted"]["method_compute_lean"]["always_32b_direct"]
    fx_am_rea = RAT["honest_recost"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_reasoning"]
    cl_rea_ac = RAT["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_reasoning"]
    cl_rea_hn = RAT["honest_recost"]["macro_cells"]["method_compute_lean"]["always_32b_reasoning"]
    am_rea_ac = RAT["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_reasoning"]
    am_rea_hn = fx_am_rea

    headline = dict(
        primary_granularity="8 reporting cells, 1/8 each (macro_cells)",
        sentence=(
            "Under equal weight per benchmark cell (8 cells, 1/8 each), the accuracy-max setting beats "
            "always-32B-direct on accuracy by %+.4f [%+.4f, %+.4f] -- but it now costs %.2fx that "
            "baseline's compute, not less, and the compute-lean setting neither matches it on the "
            "multiple-choice half (%+.4f [%+.4f, %+.4f], a SIGNIFICANT LOSS) nor stays cheap (%.2fx "
            "compute, up from %.2fx sample-weighted); the one baseline the method still clearly beats at "
            "equal weight is a 32B actually made to reason (%+.4f [%+.4f, %+.4f] accuracy, %+.0f%% latency, "
            "%+.0f%% energy -- though not fewer FLOP-eq: %.2fx as charged, %.2fx honestly re-costed)."
            % (r_am_dir8["delta"], r_am_dir8["lo"], r_am_dir8["hi"], fx_am["flops_x"],
               r_cl_dirM["delta"], r_cl_dirM["lo"], r_cl_dirM["hi"],
               fx_cl["flops_x"], fx_cl_sw["flops_x"],
               r_am_rea8["delta"], r_am_rea8["lo"], r_am_rea8["hi"],
               am_rea_hn["lat_par_pct"], am_rea_hn["energy_pct"],
               am_rea_ac["flops_x"], am_rea_hn["flops_x"])),
        one_line_if_only_one_number_is_allowed=(
            "Macro-averaged over the 8 benchmark cells, the method's accuracy-max setting is %+.4f "
            "[%+.4f, %+.4f] over always-32B-direct but at %.2fx its compute, and the compute-lean setting "
            "is a %+.4f tie overall %s that is a significant loss on the multiple-choice cells -- so at "
            "equal weight per benchmark the method's remaining claim is over the REASONING baseline, not "
            "over a single 32B forward pass."
            % (r_am_dir8["delta"], r_am_dir8["lo"], r_am_dir8["hi"], fx_am["flops_x"],
               r_cl_dir8["delta"], "[%+.4f, %+.4f]" % (r_cl_dir8["lo"], r_cl_dir8["hi"]))),
        what_gets_stronger=[
            "vs always-32B-WITH-REASONING: compute-lean %+.4f sample-weighted -> %+.4f macro; "
            "accuracy-max-veto %+.4f -> %+.4f. The open-text cells, where the 32B's reasoning mode "
            "collapses (PathVQA-open 0.109 vs 0.376 direct), stop being drowned by PMC-VQA."
            % (cl["always_32b_reasoning"]["all8"]["sample_weighted"]["delta"], r_cl_rea8["delta"],
               am2["always_32b_reasoning"]["all8"]["sample_weighted"]["delta"], r_am_rea8["delta"]),
            "vs always-7B and on the open-text cells generally (the arm the method actually wins on)."],
        what_gets_weaker_or_flips=[
            "compute-lean vs always-32B-direct, MCQ-only: %+.4f sample-weighted -> %+.4f macro "
            "[%+.4f, %+.4f], a SIGNIFICANT LOSS."
            % (cl["always_32b_direct"]["mcq_only"]["sample_weighted"]["delta"], r_cl_dirM["delta"],
               r_cl_dirM["lo"], r_cl_dirM["hi"]),
            "compute-lean vs oracle-mode-32B, MCQ-only: %+.4f -> %+.4f [%+.4f, %+.4f], a SIGNIFICANT LOSS."
            % (cl["oracle_mode_32b"]["mcq_only"]["sample_weighted"]["delta"], r_cl_orcM["delta"],
               r_cl_orcM["lo"], r_cl_orcM["hi"]),
            "accuracy-max vs always-32B-direct on all 8 cells stays a win but its size is carried by "
            "PathVQA-open; see the leave-one-cell-out range."],
        loss_not_softened=(
            "Stated plainly: on the five multiple-choice cells, under equal weight per cell, the "
            "compute-lean operating point is WORSE than simply always running the 32B (%+.4f "
            "[%+.4f, %+.4f] vs always-32B-direct; %+.4f [%+.4f, %+.4f] vs oracle-mode-32B). And it is not "
            "even cheap at equal weight: %.3fx the compute of always-32B-direct, %+.1f%% energy and "
            "%+.1f%% sequential latency. Under macro weighting compute-lean buys neither accuracy nor "
            "compute against a single 32B forward pass -- 'matches the strong model at half the compute' is "
            "false on both halves."
            % (r_cl_dirM["delta"], r_cl_dirM["lo"], r_cl_dirM["hi"],
               r_cl_orcM["delta"], r_cl_orcM["lo"], r_cl_orcM["hi"],
               fx_cl["flops_x"], fx_cl["energy_pct"], fx_cl["lat_seq_pct"])),
        what_survives=(
            "Against always-32B-WITH-REASONING the accuracy margin GROWS ~3x under equal weight "
            "(compute-lean %+.4f, accuracy-max-veto %+.4f) and the latency / energy advantage stays huge "
            "(%+.0f%% / %+.0f%% for compute-lean, %+.0f%% / %+.0f%% for accuracy-max-veto, honestly "
            "re-costed). On FLOP-eq, though, the method is NOT cheaper than the reasoning baseline at equal "
            "weight either: compute-lean %.2fx as-charged / %.2fx honestly re-costed, accuracy-max-veto "
            "%.2fx / %.2fx -- its open arm spends many small forwards, which FLOP-eq counts in full. The "
            "open-text arm is a genuine equal-weight accuracy win (%+.4f [%+.4f, %+.4f] for "
            "accuracy-max-veto vs always-32B-direct on the 3 open cells), and that plus the "
            "'reasoning mode is actively harmful on free text' result is what the project should lead with."
            % (r_cl_rea8["delta"], r_am_rea8["delta"],
               cl_rea_hn["lat_par_pct"], cl_rea_hn["energy_pct"],
               am_rea_hn["lat_par_pct"], am_rea_hn["energy_pct"],
               cl_rea_ac["flops_x"], cl_rea_hn["flops_x"], am_rea_ac["flops_x"], am_rea_hn["flops_x"],
               am2["always_32b_direct"]["open_only"]["macro_cells"]["delta"],
               am2["always_32b_direct"]["open_only"]["macro_cells"]["lo"],
               am2["always_32b_direct"]["open_only"]["macro_cells"]["hi"])))

    # ---------------- validation against previously published artifacts ----------------------------
    def stored(path, *keys):
        d = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts", path)))
        for k in keys: d = d[k]
        return d
    validation = dict(
        note="the sample-weighted column must reproduce the already-published numbers; any drift in the "
             "last decimal is bootstrap Monte-Carlo noise (independent RNG stream).",
        checks=[
            dict(what="accuracy-max-veto vs 32B-reasoning, sample-weighted, Variant B",
                 published=stored("f8_mode_vsthink_ci.json", "headline"),
                 recomputed_here=am2["always_32b_reasoning"]["all8"]["sample_weighted"]),
            dict(what="accuracy-max-veto vs 32B-direct, sample-weighted, Variant B",
                 published=stored("honest_recosting_2026-07-29.json", "part4b_zero_contribution_cells",
                                  "accuracy_max", "pooled_paired_ci"),
                 recomputed_here=am2["always_32b_direct"]["all8"]["sample_weighted"]),
            dict(what="compute-lean vs 32B-direct, sample-weighted, Variant B",
                 published=stored("honest_recosting_2026-07-29.json", "part4b_zero_contribution_cells",
                                  "compute_lean", "pooled_paired_ci"),
                 recomputed_here=cl["always_32b_direct"]["all8"]["sample_weighted"]),
            dict(what="compute-lean MCQ-only macro vs oracle-mode-32B (the known LOSS)",
                 published=stored("honest_recosting_2026-07-29.json", "part4c_macro_average",
                                  "compute_lean", "mcq_only_vs_oracle_mode"),
                 recomputed_here=r_cl_orcM),
            dict(what="compute-lean MCQ-only macro vs 32B-direct (the known LOSS)",
                 published=stored("honest_recosting_2026-07-29.json", "part4c_macro_average",
                                  "compute_lean", "mcq_only_vs_32b_direct"),
                 recomputed_here=r_cl_dirM),
            dict(what="compute-lean all-8 macro vs 32B-reasoning",
                 published=stored("honest_recosting_2026-07-29.json", "part4c_macro_average",
                                  "compute_lean", "all8_vs_32b_reasoning"),
                 recomputed_here=r_cl_rea8),
            dict(what="method cost, sample-weighted (compute-lean / accuracy-max-veto)",
                 published=stored("honest_recosting_2026-07-29.json", "part3_corrected_headline",
                                  "method_cost_variant_b"),
                 recomputed_here={m: cost["as_charged"]["sample_weighted"][m]
                                  for m in ("method_compute_lean", "method_accuracy_max_veto")}),
            dict(what="honestly re-costed reasoning baseline, sample-weighted",
                 published=stored("honest_recosting_2026-07-29.json", "part2_corrected_cost_model",
                                  "M1_primary", "pooled_variant_b"),
                 recomputed_here=cost["honest_recost"]["sample_weighted"]["always_32b_reasoning"]),
        ],
        bootstrap_shortcut_check=verify)

    # ---------------- MMMU-included reference (why the corrected pool matters even more under macro) --
    acc9 = {k: {s: float(np.asarray(cells[k][SYS_KEY[s]], float).mean()) for s in SYSTEMS} for k in ORDER_9}
    mmmu_ref = dict(
        why="MMMU is EXCLUDED from every number above (Variant B) because the Lingshu-7B 0.80 is train-set "
            "contamination. Under macro weighting the danger is larger, not smaller: MMMU would carry 1/9 "
            "of the weight (11.1%) instead of 150/42374 = 0.35% of the items.",
        macro9_vs_macro8={m: {b: dict(
            macro_9cells=round(float(np.mean([acc9[k][m] - acc9[k][b] for k in ORDER_9])), 4),
            macro_8cells=deltas[m][b]["all8"]["macro_cells"]["delta"]) for b in HEADLINE_BASELINES}
            for m in METHODS},
        mmmu_cell_accs={s: round(acc9[MMMU][s], 4) for s in SYSTEMS})

    out = dict(
        title="Macro-averaged (equal-weight-per-dataset) re-basing of every headline comparison. PRIMARY = "
              "8 reporting cells at 1/8 each; SECONDARY = 5 benchmarks at 1/5 each (closed+open averaged "
              "first). Sample-weighted numbers are carried alongside so the effect of the change is "
              "visible. Corrected pool (Variant B, MMMU excluded). Recomputed live per-sample; no GPU.",
        reproduce="python3 src/cascade_methods/macro_average_headline.py",
        date="2026-07-30", no_gpu=True, no_fabricated_numbers=True, n_bootstrap=NBOOT, seed=SEED,
        pool=dict(name="Variant B (MMMU excluded)", n_items=int(N_B), n_cells=len(order),
                  n_benchmarks=len(benches), cells=order,
                  benchmarks={b: ks for b, ks in benches.items()},
                  cell_n={k: cells[k]["n"] for k in order}),
        granularity=dict(
            primary="macro_cells -- the 8 reporting cells, 1/8 each (the researcher's decision)",
            secondary="macro_benchmarks_cellavg -- 5 benchmarks, 1/5 each, a benchmark's closed and open "
                      "cells averaged into one number first (robustness check only)",
            secondary_variant="macro_benchmarks_itemweighted -- same 5 benchmarks, but the two cells of a "
                              "benchmark are weighted by their item counts inside that benchmark",
            also_reported="sample_weighted -- the project's previous convention, kept for contrast"),
        reweighting=dict(
            PMC_VQA=dict(sample_weighted_pct=round(100 * cells["PMC_VQA"]["n"] / N_B, 1), macro_8cell_pct=12.5),
            open_text_arm=dict(cells=OPEN_B,
                               sample_weighted_pct=round(100 * sum(cells[k]["n"] for k in OPEN_B) / N_B, 1),
                               macro_8cell_pct=37.5),
            closed_mcq_arm=dict(cells=MCQ_B,
                                sample_weighted_pct=round(100 * sum(cells[k]["n"] for k in MCQ_B) / N_B, 1),
                                macro_8cell_pct=62.5),
            statement="Macro-averaging moves PMC-VQA from 79.2% of the weight to 12.5%, the open-text arm "
                      "from 5.6% of the items to 37.5% of the weight, and the closed/multiple-choice arm "
                      "from 94.4% to 62.5%. Every number below is a consequence of exactly that."),
        systems=dict(
            always_7b="cheap floor: Lingshu-7B (no-think on MCQ, greedy on open)",
            always_32b_direct="one Lingshu-32B no-think forward -- the DEPLOYABLE strong baseline",
            always_32b_reasoning="32B reasoning mode on everything; MCQ per-sample from the --reasoning dump "
                                 "(PATH_VQA_closed has no dump -> = direct), OPEN from the MEASURED judged "
                                 "think run. NOTE (honest_recosting): on the 5 MCQ cells that run emitted "
                                 "~3 tokens and did not actually reason; only the 3 open cells reason.",
            oracle_mode_32b="per-benchmark best of {reasoning, direct}; oracle upper bound on single-32B "
                            "mode selection (picks reasoning only on SLAKE-closed)",
            method_compute_lean="margin cascade on MCQ | Pandora bo-N + verifier on open",
            method_accuracy_max_veto="F8 certified weak-veto on PMC | F10 team-objective L2D on open (0.93x FLOPs)",
            method_accuracy_max_fusion="F3 confidence-advantage fusion on PMC | Pandora on open (1.25x FLOPs)"),
        per_cell_accuracy={k: dict(n=cells[k]["n"], format=cells[k]["format"], benchmark=BENCH_OF[k],
                                   sample_weight_pct=round(100 * cells[k]["n"] / N_B, 2),
                                   macro_weight_pct=12.5,
                                   oracle_mode=cells[k]["oracle_mode"],
                                   policy_compute_lean=cells[k]["cl_choice"],
                                   policy_accuracy_max_veto=cells[k].get("am2_choice"),
                                   policy_accuracy_max_fusion=cells[k].get("am_choice"),
                                   accuracy={s: round(acc_cell[k][s], 4) for s in SYSTEMS})
                           for k in order},
        per_benchmark_accuracy={b: dict(cells=ks, n=int(sum(cells[k]["n"] for k in ks)),
                                        accuracy_cellavg={s: round(float(np.mean([acc_cell[k][s] for k in ks])), 4)
                                                          for s in SYSTEMS})
                                for b, ks in benches.items()},
        accuracy_levels=dict(full_pool=acc_levels, subpools=acc_subpools, side_by_side=acc_side),
        deltas=deltas,
        escalation=esc,
        cost=cost,
        side_by_side_claims=side_by_side,
        verdicts=verdicts,
        honest_headline=headline,
        docs_wording_changes=docs_changes(deltas, cost, acc_levels, esc),
        mmmu_included_reference=mmmu_ref,
        validation=validation,
        statistics_note=dict(
            what_the_ci_covers="Each of the %d replicates resamples ITEMS WITHIN each cell (paired across "
                               "systems, common random numbers across weightings) and the macro average is "
                               "recomputed on that replicate. So the interval covers WITHIN-dataset "
                               "sampling noise." % NBOOT,
            what_the_ci_does_NOT_cover="DATASET-SELECTION noise. The 8 cells (5 benchmarks) are treated as "
                                       "fixed. With only 5-8 units, resampling datasets would be hopelessly "
                                       "unstable -- a single bootstrap draw can duplicate one cell three "
                                       "times -- so it is not attempted. This is a real limitation of every "
                                       "macro-average CI reported here and must be stated, not hidden.",
            substitute_diagnostic="A leave-one-cell-out range accompanies every comparison "
                                  "(deltas[method][baseline][pool].macro_cells_leave_one_out): the macro "
                                  "delta recomputed with each cell dropped in turn. That is the honest way "
                                  "to show dataset leverage with 8 units.",
            held_out_protocol="All method vectors are held-out 5-fold cross-fit (tau / lambda / veto and L2D "
                              "thresholds fit on the other folds); the bootstrap therefore resamples "
                              "questions only and does not include calibration variance.",
            exact_resampling="Item resampling is executed as a multinomial draw over the unique per-item "
                             "outcome patterns across the 7 systems, which is distributionally identical to "
                             "gathering resampled rows (items with the same pattern are exchangeable); "
                             "validation.bootstrap_shortcut_check compares it against literal item "
                             "resampling on the largest and one of the smallest cells."),
        caveats=[
            "MACRO CIs DO NOT include dataset-selection noise (see statistics_note). With 8 cells the macro "
            "point estimate itself has leverage: the leave-one-cell-out ranges are reported for every claim.",
            "The 8 cells are not 8 independent datasets: SLAKE / VQA-RAD / PathVQA each contribute a closed "
            "AND an open cell from the SAME images, so equal-weighting cells triple-counts three source "
            "datasets' images and gives the open-text FORMAT 37.5% of the weight. That is precisely why the "
            "5-benchmark version is computed too, and it is the strongest argument a reviewer can make "
            "against the 8-cell primary.",
            "always-32B-reasoning is only a genuine reasoning run on the 3 open cells (measured 105-141 "
            "generated tokens); on the 5 MCQ cells the '--reasoning' dump emitted ~3 tokens and is a "
            "direct-mode run under another name (honest_recosting.py part 1). Under macro weighting the "
            "vs-reasoning win is therefore ~entirely the open-text cells -- report it as such.",
            "PATH_VQA_closed has no 32B reasoning dump at all; its reasoning vector is the direct-mode run, "
            "so its vs-reasoning delta is identical to its vs-direct delta by construction.",
            "oracle-mode-32B peeks at which mode wins per benchmark; it is a fair strong baseline, not a "
            "deployable system.",
            "SLAKE / VQA-RAD / PathVQA-open are IN-DOMAIN for the pooled4 verifier that the open arm uses.",
            "Equal-weighted COST is not pooled cost: the open cells are 5.6% of items but 37.5% of the macro "
            "weight, and they are the expensive ones for the method (adaptive draws, 7.6-12.6 FLOP-eq) AND "
            "for the reasoning baseline (100+ generated tokens). Both cost columns are reported. The "
            "consequence is a REVERSAL, not a shift: no operating point is compute-cheaper than "
            "always-32B-direct at equal weight (see cost.joint_claim).",
            "The open arm's PARALLEL latency assumes the adaptive draws can be overlapped, which was never "
            "measured (it is a known open hole); its SEQUENTIAL latency is 1.9-3.1 s per open cell. Both "
            "lat_par and lat_seq are reported, and under macro weighting the gap between them matters far "
            "more than it did pooled, because the open cells now carry 37.5% of the weight.",
            "The method's thresholds (tau, lambda, veto and L2D thresholds) were CALIBRATED to hold POOLED "
            "accuracy at parity with the strong leg. Re-basing the report on macro-averaging changes the "
            "reporting metric but not the objective the method was tuned for; a macro-objective refit is a "
            "separate experiment and is NOT done here.",
            "MMMU is excluded throughout (contamination). Under macro weighting including it would be worse, "
            "not better: 1/9 of the weight instead of 0.35% of the items (mmmu_included_reference).",
        ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    console(out)
    print(f"\nwrote {OUT}")
    return out


# ===================================================================================================
# 6.  WHICH DOCUMENTED CLAIMS MUST CHANGE
# ===================================================================================================
def docs_changes(deltas, cost, acc_levels, esc):
    cl, am2, amf = (deltas["method_compute_lean"], deltas["method_accuracy_max_veto"],
                    deltas["method_accuracy_max_fusion"])
    g = lambda d, pool, w="macro_cells": d[pool][w]
    R = cost["method_vs_baseline_ratios"]
    return [
        dict(claim="THE TITLE and every 'Pareto-dominates' / 'dominates on every axis' statement",
             targets=["paper/adaptive-cascade-medvqa_ieee_2026-07-08.tex:18 (title 'A Small Model that "
                      "Pareto-Dominates a Large Reasoning Model')",
                      "same file :246 (tab:main caption 'Both method points dominate ... on every axis'), "
                      ":242 (section title 'Main result: Pareto-dominance'), :51 (contribution C2), "
                      ":267, :274, :403",
                      "PROJECT_OVERVIEW.md 'match or beat always-Lingshu-32B-with-thinking while using less "
                      "compute than a single 32B forward'",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md "
                      "'Pareto-dominates on all 5 families'",
                      "meetings/progress_report_professor_2026-07-27.html hero + 'Pareto-dominates every "
                      "32B strategy, including an oracle'"],
             currently="dominance asserted on accuracy AND every cost axis, sample-weighted",
             becomes="FALSE at equal weight per cell against always-32B-direct / oracle-mode-32B on the "
                     "COST axis for every operating point (%.3fx / %.3fx / %.3fx FLOP-eq, all > 1), and "
                     "false on the ACCURACY axis for compute-lean on the multiple-choice cells."
                     % (R["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]["flops_x"],
                        R["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
                        R["as_charged"]["macro_cells"]["method_accuracy_max_fusion"]["always_32b_direct"]["flops_x"]),
             new_wording="Restrict 'dominates' to the always-32B-WITH-REASONING baseline, which it still "
                         "does dominate at equal weight on every axis. Against a single 32B forward pass, "
                         "say 'more accurate at higher compute' (accuracy-max) or 'cheaper only on the "
                         "multiple-choice arm, at a significant accuracy cost' (compute-lean)."),
        dict(claim="'matches always-32B-direct' / 'matches the strong model at 0.49x compute' (compute-lean)",
             currently="pooled sample-weighted %+.4f [%+.4f, %+.4f] -- a tie"
                       % (g(cl["always_32b_direct"], "all8", "sample_weighted")["delta"],
                          g(cl["always_32b_direct"], "all8", "sample_weighted")["lo"],
                          g(cl["always_32b_direct"], "all8", "sample_weighted")["hi"]),
             becomes="macro (8 cells) %+.4f [%+.4f, %+.4f] overall -- still a tie -- BUT on the "
                     "multiple-choice cells %+.4f [%+.4f, %+.4f], a SIGNIFICANT LOSS"
                     % (g(cl["always_32b_direct"], "all8")["delta"], g(cl["always_32b_direct"], "all8")["lo"],
                        g(cl["always_32b_direct"], "all8")["hi"], g(cl["always_32b_direct"], "mcq_only")["delta"],
                        g(cl["always_32b_direct"], "mcq_only")["lo"], g(cl["always_32b_direct"], "mcq_only")["hi"]),
             new_wording="Compute-lean trades accuracy for compute: at equal weight per benchmark cell it "
                         "ties the strong model overall only because the open-text arm compensates a "
                         "significant loss on the multiple-choice cells."),
        dict(claim="'Pareto-dominates always-32B' (unqualified)",
             currently="true sample-weighted (PMC-VQA carries 79% of the weight)",
             becomes="FALSE at equal weight on the multiple-choice half: compute-lean is %+.4f "
                     "[%+.4f, %+.4f] vs oracle-mode-32B there"
                     % (g(cl["oracle_mode_32b"], "mcq_only")["delta"], g(cl["oracle_mode_32b"], "mcq_only")["lo"],
                        g(cl["oracle_mode_32b"], "mcq_only")["hi"]),
             new_wording="Drop 'Pareto-dominates'. Say: cheaper at equal or better accuracy when every "
                         "benchmark cell counts equally ONLY for the accuracy-max settings; compute-lean is "
                         "cheaper at a small, and on multiple-choice significant, accuracy cost."),
        dict(claim="'+0.0245 over always-32B-with-reasoning' (accuracy-max headline)",
             currently="sample-weighted %+.4f [%+.4f, %+.4f]"
                       % (g(am2["always_32b_reasoning"], "all8", "sample_weighted")["delta"],
                          g(am2["always_32b_reasoning"], "all8", "sample_weighted")["lo"],
                          g(am2["always_32b_reasoning"], "all8", "sample_weighted")["hi"]),
             becomes="macro (8 cells) %+.4f [%+.4f, %+.4f] -- roughly 3x LARGER"
                     % (g(am2["always_32b_reasoning"], "all8")["delta"], g(am2["always_32b_reasoning"], "all8")["lo"],
                        g(am2["always_32b_reasoning"], "all8")["hi"]),
             new_wording="Quote the macro number, and in the same sentence say that it is carried by the 3 "
                         "open-text cells and that on the 5 multiple-choice cells the 'reasoning' baseline "
                         "never actually reasoned (~3 generated tokens)."),
        dict(claim="Headline accuracy LEVELS ('accuracy 0.5836', 'always-32B 0.5729', 'suite average')",
             currently="sample-weighted: method-veto %.4f, 32B-direct %.4f, 32B-reasoning %.4f, 7B %.4f"
                       % (acc_levels["sample_weighted"]["method_accuracy_max_veto"],
                          acc_levels["sample_weighted"]["always_32b_direct"],
                          acc_levels["sample_weighted"]["always_32b_reasoning"],
                          acc_levels["sample_weighted"]["always_7b"]),
             becomes="macro (8 cells): method-veto %.4f, 32B-direct %.4f, 32B-reasoning %.4f, 7B %.4f"
                     % (acc_levels["macro_cells"]["method_accuracy_max_veto"],
                        acc_levels["macro_cells"]["always_32b_direct"],
                        acc_levels["macro_cells"]["always_32b_reasoning"],
                        acc_levels["macro_cells"]["always_7b"]),
             new_wording="Every reported suite accuracy must be relabelled 'macro-average over 8 benchmark "
                         "cells' and its value replaced; the old numbers are ~PMC-VQA accuracy."),
        dict(claim="'n=42,224' as the unit of the headline",
             currently="the headline is an average over 42,224 items",
             becomes="the headline is an average over 8 cells (5 benchmarks); n=42,224 remains the item "
                     "count that the within-dataset CIs are computed from",
             new_wording="Report 'macro-average over 8 cells from 5 benchmarks (42,224 items)'."),
        dict(claim="Cost claims ('0.49x / 0.93x compute', '-95.5% latency vs reasoning')",
             currently="sample-weighted: compute-lean %.3fx compute / %+.1f%% latency vs 32B-direct"
                       % (R["as_charged"]["sample_weighted"]["method_compute_lean"]["always_32b_direct"]["flops_x"],
                          R["as_charged"]["sample_weighted"]["method_compute_lean"]["always_32b_direct"]["lat_par_pct"]),
             becomes="macro (8 cells): compute-lean %.3fx compute / %+.1f%% latency vs 32B-direct; "
                     "accuracy-max-veto %.3fx / %+.1f%%"
                     % (R["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]["flops_x"],
                        R["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]["lat_par_pct"],
                        R["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
                        R["as_charged"]["macro_cells"]["method_accuracy_max_veto"]["always_32b_direct"]["lat_par_pct"]),
             new_wording="Recompute every cost ratio under the same equal weighting as the accuracy it is "
                         "quoted with; never pair a macro accuracy with a sample-weighted cost."),
        dict(claim="'accuracy-max-fusion beats oracle-mode-32B' (the F3 row)",
             currently="sample-weighted %+.4f [%+.4f, %+.4f]"
                       % (g(amf["oracle_mode_32b"], "all8", "sample_weighted")["delta"],
                          g(amf["oracle_mode_32b"], "all8", "sample_weighted")["lo"],
                          g(amf["oracle_mode_32b"], "all8", "sample_weighted")["hi"]),
             becomes="macro (8 cells) %+.4f [%+.4f, %+.4f]"
                     % (g(amf["oracle_mode_32b"], "all8")["delta"], g(amf["oracle_mode_32b"], "all8")["lo"],
                        g(amf["oracle_mode_32b"], "all8")["hi"]),
             new_wording="Restate on the macro; note the PMC fusion cell drops from 79% to 12.5% of the "
                         "weight, so the fusion trick can no longer carry the headline."),
        dict(claim="'Pooled escalation is 16.2%' -- the number every cost claim rests on",
             targets=["PROJECT_OVERVIEW.md:158 ('Pooled escalation is 16.2%, but it varies enormously per "
                      "benchmark -- PMC-VQA 8.5%, SLAKE-closed 20.5%, PathVQA-closed 45.7%, VQA-RAD-closed "
                      "57.0%, MedXpert 89.6%')",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md 'the "
                      "multiple-choice arm's pooled compute is 1.739 versus 4.57 (0.38x)'"],
             currently="compute-lean escalation, sample-weighted: %.1f%% over the multiple-choice cells "
                       "(%.1f%% over all 8)"
                       % (100 * esc["compute_lean_mcq"]["sample_weighted"],
                          100 * esc["compute_lean_all8"]["sample_weighted"]),
             becomes="macro (equal weight per cell): %.1f%% over the multiple-choice cells (%.1f%% over all "
                     "8) -- because PMC-VQA's %.1f%% escalation no longer carries 79%% of the average"
                     % (100 * esc["compute_lean_mcq"]["macro_cells"],
                        100 * esc["compute_lean_all8"]["macro_cells"],
                        100 * esc["per_cell"]["PMC_VQA"]),
             new_wording="Report the per-cell escalation rates and the macro rate; never quote a single "
                         "'escalation rate' for the suite without saying which weighting it uses."),
        dict(claim="'89% of the headline delta comes from 2 of 8 cells' (the contribution decomposition)",
             targets=["PROJECT_OVERVIEW.md ('89% of the +0.0245 comes from 2 of the 8 cells')",
                      "RESULTS.md:98-101", "CLAUDE.md ('89% of the headline delta comes from 2 of 8 cells')",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md 'Hole 2'",
                      "results/cascade_methods/README.md"],
             currently="a sample-weighted contribution table (weight_c x delta_c); PathVQA-open 51%, "
                       "PMC-VQA 38%",
             becomes="MEANINGLESS under macro: every cell contributes exactly 1/8 of its own delta, so the "
                     "table collapses to the per-cell delta list. Concentration must instead be shown with "
                     "the leave-one-cell-out range (reported here for every comparison).",
             new_wording="Replace the contribution-share table with the per-cell delta table plus the "
                         "leave-one-cell-out range; drop the '89% from 2 cells' phrasing entirely."),
        dict(claim="'SLAKE-open (+0.0016) and VQA-RAD-open (+0.0050) improve but their CIs span zero at "
                   "n = 645 / 200 -- do not quote them as wins'",
             targets=["PROJECT_OVERVIEW.md (per-benchmark paragraph)"],
             currently="advice that small cells should not be quoted, because they barely move the pooled "
                       "average anyway",
             becomes="INCOMPATIBLE with equal weighting: those cells now carry 12.5%% of the headline each, "
                     "and their wide intervals are exactly what widens the macro CI. The macro CI is 2-4x "
                     "wider than the pooled one for this reason (e.g. accuracy-max-veto vs 32B-direct: "
                     "%s pooled vs %s macro)."
                     % ("[%+.4f,%+.4f]" % (g(am2["always_32b_direct"], "all8", "sample_weighted")["lo"],
                                           g(am2["always_32b_direct"], "all8", "sample_weighted")["hi"]),
                        "[%+.4f,%+.4f]" % (g(am2["always_32b_direct"], "all8")["lo"],
                                           g(am2["always_32b_direct"], "all8")["hi"])),
             new_wording="Keep the small cells in, report their individual CIs honestly, and state that the "
                         "macro CI is wider because small benchmarks now count as much as PMC-VQA."),
        dict(claim="The '+0.02xx family' decode table ('three orthogonal axes: lever / pool / "
                   "estimated-vs-measured')",
             targets=["INCONSISTENCIES.md:104 (Y1)", "READING_GUIDE.md:36",
                      "CLAUDE.md:89 ('+0.0245 at 0.93x is the one canonical number')",
                      "RESULTS.md:31-35", "results/cascade_methods/README.md:37-41",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md sec 10.3"],
             currently="three axes; +0.0245 declared THE canonical number",
             becomes="FOUR axes -- lever / pool / measured-vs-estimated / WEIGHTING. The canonical value "
                     "under the new convention is the macro one (%+.4f [%+.4f, %+.4f] for accuracy-max-veto "
                     "vs reasoning), and +0.0245 becomes 'the sample-weighted equivalent'."
                     % (g(am2["always_32b_reasoning"], "all8")["delta"],
                        g(am2["always_32b_reasoning"], "all8")["lo"], g(am2["always_32b_reasoning"], "all8")["hi"]),
             new_wording="Add a 'weighting' column to the decode table and restate the canonicity rule as "
                         "'macro over 8 cells, Variant B, measured, veto lever'."),
        dict(claim="The MMMU-exclusion rationale ('MMMU is 0.35% of the pool, so the headline moves only "
                   "-0.0005')",
             targets=["INCONSISTENCIES.md (Y4)", "PROJECT_OVERVIEW.md", "progress/progress_July_08.md",
                      "meetings/progress_report_professor_2026-07-27.html",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md"],
             currently="exclusion justified as immaterial (0.35% of items)",
             becomes="the justification INVERTS: under macro MMMU would have carried 1/9 = 11.1% of the "
                     "headline, so excluding the contaminated cell is now a large and consequential "
                     "decision, not a rounding error (see mmmu_included_reference).",
             new_wording="Re-argue the exclusion on contamination grounds alone and state the size of its "
                         "effect on the macro headline explicitly."),
        dict(claim="The 'honest one-line claim' currently in README.md / PROJECT_OVERVIEW.md / "
                   "READING_GUIDE.md / retrospective sec 8.5",
             targets=["README.md:24-27", "PROJECT_OVERVIEW.md ('Matches the strong model at roughly half "
                      "the compute...')", "READING_GUIDE.md:210-214", "CLAUDE.md:158-160",
                      "results/cascade_methods/docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md sec 8.5"],
             currently="'Matches the strong model at roughly half the compute, with a significant accuracy "
                       "gain on two specific cells ... plus a measured characterization of why the "
                       "remaining cells are unwinnable.'",
             becomes="both halves fail at equal weight: 'matches' is a significant LOSS on the "
                     "multiple-choice cells (%+.4f [%+.4f, %+.4f] vs 32B-direct) and 'half the compute' "
                     "becomes %.2fx the compute."
                     % (g(cl["always_32b_direct"], "mcq_only")["delta"],
                        g(cl["always_32b_direct"], "mcq_only")["lo"],
                        g(cl["always_32b_direct"], "mcq_only")["hi"],
                        R["as_charged"]["macro_cells"]["method_compute_lean"]["always_32b_direct"]["flops_x"]),
             new_wording="Use honest_headline.sentence from this artifact."),
    ]


# ===================================================================================================
# 7.  CONSOLE
# ===================================================================================================
SHORT = {"always_7b": "7B", "always_32b_direct": "32Bdir", "always_32b_reasoning": "32Brea",
         "oracle_mode_32b": "orc32", "method_compute_lean": "M-CL", "method_accuracy_max_veto": "M-AMv",
         "method_accuracy_max_fusion": "M-AMf"}


def console(o):
    W = 132
    print("=" * W)
    print("MACRO-AVERAGED HEADLINE  --  %s" % o["pool"]["name"])
    print("=" * W)
    print("  " + o["reweighting"]["statement"].replace("%%", "%"))
    print("\n  PRIMARY: %s\n  SECONDARY: %s" % (o["granularity"]["primary"], o["granularity"]["secondary"]))

    print("\n" + "-" * W)
    print("PER-CELL ACCURACY (equal weighting makes these the story)")
    print("-" * W)
    hdr = f"{'cell':<17}{'n':>7}{'sw%':>7}" + "".join(f"{SHORT[s]:>9}" for s in SYSTEMS)
    print(hdr); print("-" * len(hdr))
    for k, r in o["per_cell_accuracy"].items():
        print(f"{k:<17}{r['n']:>7}{r['sample_weight_pct']:>7.1f}" +
              "".join(f"{r['accuracy'][s]:>9.4f}" for s in SYSTEMS))
    print("-" * len(hdr))
    for lab in ("sample_weighted", "macro_cells", "macro_benchmarks_cellavg"):
        v = o["accuracy_levels"]["full_pool"][lab]
        print(f"{lab:<17}{'':>7}{'':>7}" + "".join(f"{v[s]:>9.4f}" for s in SYSTEMS))
    for pool in ("mcq_only", "open_only"):
        p = o["accuracy_levels"]["subpools"][pool]
        for lab in ("sample_weighted", "macro_cells"):
            print(f"{pool + '/' + lab[:5]:<17}{p['n']:>7}{'':>7}" + "".join(f"{p[lab][s]:>9.4f}" for s in SYSTEMS))

    print("\n" + "=" * W)
    print("SIDE BY SIDE -- sample-weighted vs MACRO(8 cells) vs macro(5 benchmarks); [lo,hi] = 95% CI")
    print("=" * W)
    print(f"  {'comparison':<46}{'pool':<24}{'sample-weighted':>26}{'MACRO 8 cells':>26}{'macro 5 bench':>26}")
    for r in o["side_by_side_claims"]:
        sw, m8, m5 = r["sample_weighted"], r["macro_8cells"], r["macro_5benchmarks"]
        f = lambda x: "%+.4f[%+.4f,%+.4f]%s" % (x["delta"], x["ci95"][0], x["ci95"][1],
                                               "*" if x["verdict"] != "TIE" else " ")
        comp = "%s vs %s" % (SHORT[r["method"]], SHORT[r["baseline"]])
        print(f"  {comp:<46}{r['granularity_axis']:<24}{f(sw):>26}{f(m8):>26}{f(m5):>26}")
    print("  (* = CI excludes 0)")

    print("\n" + "=" * W)
    print("PER-CELL DELTAS under equal weighting (macro = the mean of these columns)")
    print("=" * W)
    for m in METHODS:
        print(f"\n  -- {m} --")
        print(f"    {'cell':<17}" + "".join(f"{'vs ' + SHORT[b]:>28}" for b in HEADLINE_BASELINES))
        for k in o["pool"]["cells"]:
            line = f"    {k:<17}"
            for b in HEADLINE_BASELINES:
                c = o["deltas"][m][b]["all8"]["per_cell"][k]
                line += "%+10.4f[%+.4f,%+.4f]%s" % (c["delta"], c["lo"], c["hi"], "*" if c["sig"] else " ")
            print(line)
        line = f"    {'MACRO(8)':<17}"
        for b in HEADLINE_BASELINES:
            c = o["deltas"][m][b]["all8"]["macro_cells"]
            line += "%+10.4f[%+.4f,%+.4f]%s" % (c["delta"], c["lo"], c["hi"], "*" if c["sig"] else " ")
        print(line)
        line = f"    {'sample-weighted':<17}"
        for b in HEADLINE_BASELINES:
            c = o["deltas"][m][b]["all8"]["sample_weighted"]
            line += "%+10.4f[%+.4f,%+.4f]%s" % (c["delta"], c["lo"], c["hi"], "*" if c["sig"] else " ")
        print(line)
        line = f"    {'LOO range(macro)':<17}"
        for b in HEADLINE_BASELINES:
            r = o["deltas"][m][b]["all8"]["macro_cells_leave_one_out"]["range"]
            line += "%28s" % ("[%+.4f,%+.4f]" % (r[0], r[1]))
        print(line)

    print("\n" + "=" * W)
    print("COST under the same weightings (batch-1; as-charged | honest per-cell re-costing of reasoning)")
    print("=" * W)
    for conv in ("as_charged", "honest_recost"):
        print(f"\n  [{conv}]")
        print(f"    {'system':<28}" + "".join(f"{lab[:14]:>16}" for lab in
                                              ("sample_weighted", "macro_cells", "macro_bench")))
        for s in SYSTEMS:
            a = o["cost"][conv]["sample_weighted"][s]; b = o["cost"][conv]["macro_cells"][s]
            c = o["cost"][conv]["macro_benchmarks_cellavg"][s]
            print(f"    {s:<28}" + "".join("%16s" % ("%.2fF/%.0fms/%.0fJ" % (x["flops"], x["lat_par_ms"], x["energy_j"]))
                                           for x in (a, b, c)))
    print("\n  method vs baseline ratios (as_charged):")
    for lab in ("sample_weighted", "macro_cells"):
        for m in METHODS:
            for b in HEADLINE_BASELINES:
                r = o["cost"]["method_vs_baseline_ratios"]["as_charged"][lab][m][b]
                print(f"    {lab:<18}{SHORT[m]:<7}vs {SHORT[b]:<8}"
                      f"compute {r['flops_x']:.3f}x  latency(par) {r['lat_par_pct']:+.1f}%  "
                      f"latency(seq) {r['lat_seq_pct']:+.1f}%  energy {r['energy_pct']:+.1f}%")
    print("\n  ESCALATION (compute-lean) -- the driver of every cost claim:")
    print("    per cell: " + ", ".join("%s %.1f%%" % (k, 100 * v) for k, v in o["escalation"]["per_cell"].items()))
    print("    MCQ cells: sample-weighted %.1f%%  ->  macro %.1f%%   |  all 8: %.1f%% -> %.1f%%"
          % (100 * o["escalation"]["compute_lean_mcq"]["sample_weighted"],
             100 * o["escalation"]["compute_lean_mcq"]["macro_cells"],
             100 * o["escalation"]["compute_lean_all8"]["sample_weighted"],
             100 * o["escalation"]["compute_lean_all8"]["macro_cells"]))
    print("\n  JOINT (accuracy AND cost) CLAIM:")
    print("    " + o["cost"]["joint_claim"]["headline"])
    for m in METHODS:
        for b in HEADLINE_BASELINES:
            j = o["cost"]["joint_claim"]["by_pair"][m][b]
            print(f"    {SHORT[m]:<7}vs {SHORT[b]:<9} sample-weighted: {j['sample_weighted']['joint']:<28}"
                  f"->  MACRO: {j['macro_cells']['joint']:<28}{'CHANGED' if j['changed_by_macro'] else ''}")
    print("\n  PARETO-optimal systems (macro accuracy vs macro cost, honest re-costing):")
    p = o["cost"]["pareto"]["honest_recost"]["macro_cells"]
    print(f"    on FLOPs   : {p['pareto_optimal_flops']}")
    print(f"    on lat_par : {p['pareto_optimal_lat_par']}")
    print(f"    on lat_seq : {p['pareto_optimal_lat_seq']}")
    print(f"    on energy  : {p['pareto_optimal_energy']}")

    print("\n" + "=" * W)
    print("VERDICTS (macro, 8 cells)")
    print("=" * W)
    for m in METHODS:
        for b in HEADLINE_BASELINES:
            v = o["verdicts"][m][b]
            print(f"  {SHORT[m]:<7}vs {SHORT[b]:<9} ALL8 {v['all8_macro']['verdict']:<5}"
                  f"{v['all8_macro']['delta']:+.4f}  |  MCQ {v['mcq_only_macro']['verdict']:<5}"
                  f"{v['mcq_only_macro']['delta']:+.4f}  |  OPEN {v['open_only_macro']['verdict']:<5}"
                  f"{v['open_only_macro']['delta']:+.4f}   (sample-weighted ALL8 was "
                  f"{v['all8_sample_weighted_for_contrast']['verdict']} {v['all8_sample_weighted_for_contrast']['delta']:+.4f})")

    print("\n" + "=" * W)
    print("HONEST HEADLINE")
    print("=" * W)
    print("  " + o["honest_headline"]["sentence"])
    print("\n  LOSS, UNSOFTENED:\n  " + o["honest_headline"]["loss_not_softened"])
    print("\n  WHAT SURVIVES:\n  " + o["honest_headline"]["what_survives"])
    print("\n  " + o["cost"]["joint_claim"]["non_dominated_vs_dominates"])
    print("\n  STATISTICS: " + o["statistics_note"]["what_the_ci_does_NOT_cover"])

    print("\n" + "=" * W)
    print("VALIDATION vs already-published artifacts (sample-weighted column must reproduce)")
    print("=" * W)
    for c in o["validation"]["checks"]:
        print(f"  {c['what']}\n    published : {json.dumps(c['published'], default=str)[:150]}")
        print(f"    recomputed: {json.dumps(c['recomputed_here'], default=str)[:150]}")
    print("  bootstrap shortcut check: " + json.dumps(o["validation"]["bootstrap_shortcut_check"], default=str))


if __name__ == "__main__":
    run()
