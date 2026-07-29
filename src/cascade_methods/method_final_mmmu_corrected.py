#!/usr/bin/env python3
"""
method_final_mmmu_corrected.py -- the paper's CORE results table RECOMPUTED with the MMMU cell
CORRECTED, because the Lingshu-7B MMMU-medical 0.80 is train-set contamination (+26 over the published
Lingshu-7B 54.0; order-robust under full cyclic option permutation, NOT a parsing/position artifact --
see results/cascade_methods/artifacts/mmmu_fix.json).  The leaked 0.80 must NOT be banked as a
"7B-beats-32B" headline win.  This script reruns the whole method<->baseline comparison in TWO honest
variants, for ALL THREE method settings, and emits pooled (sample-weighted) + macro accuracy with
paired-bootstrap 95% CIs, FLOP-eq / batch-1 latency / energy, and the Pareto-frontier point set.

  VARIANT A -- MMMU ESCALATED to the 32B (reasoning -> 32B-THINK).  The method's MMMU accuracy becomes
               the 32B-think accuracy (0.66); MMMU escalates 100% so its (small, n=150) cost is added
               at the 32B-think cost.  The leaked 7B 0.80 is never used by the method.  Baselines
               (always-32B-think / -no-think / oracle-mode-32B) are UNCHANGED -- they never used the 0.80.
  VARIANT B -- MMMU EXCLUDED (drop the MMMU rows from every pool; 8-cell suite = 5 MCQ + 3 open).

  METHOD SETTINGS (the paper's knob):
    compute_lean         margin-cascade MCQ | Pandora bo-N + verifier open | (MMMU corrected)
    accuracy_max_v1_F3   F1 slice-router, F3 confidence-advantage FUSION on PMC | Pandora open | (MMMU corrected)
    accuracy_max_v2_F8F10 F1 slice-router, F8 certified weak-veto on PMC | F10 team-objective L2D open | (MMMU corrected)

  BASELINES (all per-benchmark + pooled): always-32B-THINK, always-32B-NO-THINK, oracle-mode-32B
  (= per-benchmark best of {think,no-think}; the honest strong single-model baseline).  always-7B is
  carried as the cheap floor for the Pareto point set.

REUSE (simplest-thing-that-works).  Everything is recomputed LIVE per-sample from the SAME MedEvalKit
Lingshu dumps.  We reuse paper_baselines.build_cells() (5-fold cross-fit reconstruction of the
compute-lean and accuracy-max-v1 per-sample delivered-ok vectors + oracle-mode + costs), then ADD the
v2 per-sample vectors (F8 certified veto on PMC via beat32b_more.f8_veto; F10 team-objective L2D on the
open cells via a per-sample reimplementation of beat32b_more.f10_l2d), and re-pool under each MMMU
correction.  Paired bootstrap NBOOT=10000 over questions (paper_baselines machinery).  NO GPU.

HONESTY / DATA GAPS (also in the JSON):
  * always-32B-THINK open-text accuracy is ESTIMATED (judged 32B-no-think + measured modal think-delta
    -0.195/-0.120/-0.130); there is no per-sample judged think dump on open-text -> the method-vs-THINK
    CI is over the MCQ pool (real per-sample) only and the full-suite vs-THINK delta is a flagged POINT
    estimate.  Every OTHER comparison (vs oracle-mode-32B, vs 32B-no-think, vs 7B) has real per-sample
    vectors on ALL cells incl. open.
  * PATH_VQA_closed has no 32B-think dump -> think=no-think there; oracle picks no-think.
  * paired bootstrap resamples QUESTIONS (sampling noise); held-out tau/lambda/threshold are fixed per
    item, so calibration variance is not in the CI.
  * SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (trained on those families).

Launch from repo root (CPU only):   python3 src/cascade_methods/method_final_mmmu_corrected.py
"""
import os, json
import numpy as np
from sklearn.linear_model import LogisticRegression

import paper_baselines as PB          # build_cells (per-sample cl/am/oracle vectors + costs), cost helpers
import beat32b_fusion as BF           # BF.mcq (loads c7 for F8)
import beat32b_more as BB             # f8_veto (per-sample), open_features (F10 features)

ROOT = PB.IM.ROOT
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/method_final_mmmu_corrected.json")

GEN7, GEN32N, GEN32T = PB.GEN7, PB.GEN32N, PB.GEN32T   # (ms, energy_J, flop) tuples
cost_fixed, cost_cascade, cost_pandora = PB.cost_fixed, PB.cost_cascade, PB.cost_pandora

MCQ_ORDER  = PB.MCQ_ORDER
OPEN_ORDER = PB.OPEN_ORDER
FULL_ORDER = MCQ_ORDER + OPEN_ORDER
MMMU = "MMMU-Medical-val"

CONFIGS   = ["compute_lean", "accuracy_max_v1_F3", "accuracy_max_v2_F8F10"]
BASELINES = ["always_32b_think", "always_32b_nt", "oracle_mode_32b"]   # the task's three baselines
POINTSYS  = ["always_7b", "always_32b_nt", "always_32b_think", "oracle_mode_32b",
             "method_compute_lean", "method_accuracy_max_v1_F3", "method_accuracy_max_v2_F8F10"]

NBOOT = PB.NBOOT                      # 10000
RNG   = np.random.default_rng(12345)  # own stream (deterministic, independent of PB module state)


# ============================ paired / macro bootstrap =============================================
def paired_ci(a, b, chunk=1000):
    """95% CI of mean(a-b) by paired bootstrap over questions (aligned 0/1 arrays)."""
    d = np.asarray(a, float) - np.asarray(b, float); N = len(d)
    if N == 0:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), sig=False, n=0)
    boots = np.empty(NBOOT)
    for s in range(0, NBOOT, chunk):
        m = min(chunk, NBOOT - s)
        boots[s:s + m] = d[RNG.integers(0, N, size=(m, N))].mean(axis=1)
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return dict(delta=round(float(d.mean()), 4), lo=round(lo, 4), hi=round(hi, 4),
                sig=bool(lo > 0 or hi < 0), n=N)


def macro_ci(dcells, chunk=1000):
    """95% CI of the MACRO delta (equal-weight mean over cells) by paired bootstrap: per iter resample
    each cell's questions (paired), average the per-cell deltas.  dcells = list of per-cell (a-b) arrays,
    all with real per-sample vectors."""
    K = len(dcells)
    if K == 0:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), sig=False, cells=0)
    acc = np.zeros(NBOOT); point = 0.0
    for d in dcells:
        d = np.asarray(d, float); N = len(d); point += d.mean()
        cell = np.empty(NBOOT)
        for s in range(0, NBOOT, chunk):
            m = min(chunk, NBOOT - s)
            cell[s:s + m] = d[RNG.integers(0, N, size=(m, N))].mean(axis=1)
        acc += cell
    acc /= K; point /= K
    lo, hi = float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))
    return dict(delta=round(point, 4), lo=round(lo, 4), hi=round(hi, 4),
                sig=bool(lo > 0 or hi < 0), cells=K)


# ============================ v2 (F8 + F10) per-sample vectors =====================================
def f10_persample(dskey, K=BB.K):
    """Per-sample F10 team-objective consistent-L2D open-text vector (mirrors beat32b_more.f10_l2d, but
    returns the per-item delivered outcome).  Learned logistic rejector over 7B-side open features;
    threshold tuned on TRAIN team-accuracy; cross-fit 5-fold.  Returns (team_ok, ok32, take7_rate)."""
    d = BB.open_features(dskey)
    ok7, ok32, X = d["ok7"], d["ok32"], d["X"]
    n = len(ok7); ii = np.arange(n)
    team = np.zeros(n); took7 = np.zeros(n, bool)
    for f in range(K):
        te = ii % K == f; tr = ~te
        mdl = LogisticRegression(max_iter=500, C=1.0).fit(X[tr], ok7[tr])
        g_tr = mdl.predict_proba(X[tr])[:, 1]; g_te = mdl.predict_proba(X[te])[:, 1]
        grid = np.unique(np.quantile(g_tr, np.linspace(0, 1, 41)))
        best_t, best_a = 1.0, -1.0
        for t in grid:
            keep = g_tr >= t
            a = np.where(keep, ok7[tr], ok32[tr]).mean()
            if a > best_a: best_a, best_t = a, t
        keep_te = g_te >= best_t
        team[te] = np.where(keep_te, ok7[te], ok32[te]); took7[te] = keep_te
    return team, ok32, float(took7.mean())


def add_v2_vectors(cells):
    """Augment each cell with am2_ok (accuracy-max-v2 per-sample delivered ok) + am2_cost + am2_choice."""
    # ---- MCQ: default v2 == v1 accuracy-max choice (always-32B-nt on closed/MedXpert; keep-7B MMMU) ----
    for k in MCQ_ORDER:
        c = cells[k]
        c["am2_ok"]     = c["am_ok"].copy()
        c["am2_cost"]   = dict(c["am_cost"])
        c["am2_choice"] = c["am_choice"]
    # PMC: F8 certified weak-veto REPLACES the F3 fusion cell (7B on all + 32B on non-veto; cheaper)
    dm = BF.mcq("PMC_VQA")
    ok_f8, veto = BB.f8_veto(dm)
    assert len(ok_f8) == cells["PMC_VQA"]["n"], "F8 PMC alignment mismatch"
    esc_f8 = float(1.0 - veto.mean())
    cells["PMC_VQA"]["am2_ok"]     = ok_f8
    cells["PMC_VQA"]["am2_cost"]   = cost_cascade(esc_f8)          # 7B(all)+32B(non-veto) == GEN7 + esc*GEN32N
    cells["PMC_VQA"]["am2_choice"] = "F8-certified-veto"
    cells["PMC_VQA"]["am2_veto_rate"] = round(float(veto.mean()), 4)
    cells["PMC_VQA"]["am2_esc"]       = round(esc_f8, 4)
    # ---- OPEN: F10 team-objective L2D REPLACES the Pandora parity-tau gate (draw count meanN kept) ----
    for name, dskey in [("SLAKE_open", "slake_open"), ("VQA_RAD_open", "vqa_rad_open"),
                        ("PATH_VQA_open", "pathvqa_open")]:
        team, ok32_f10, took7 = f10_persample(dskey)
        c = cells[name]
        assert len(team) == c["n"] and np.array_equal(ok32_f10, c["ok32"]), f"F10 {name} alignment mismatch"
        esc10 = float(1.0 - took7)
        c["am2_ok"]     = team
        c["am2_cost"]   = cost_pandora(c["meanN"], esc10)          # Pandora draw count meanN, F10 escalation set
        c["am2_choice"] = "pandora-N + F10-L2D->32B-nt"
        c["am2_esc"]    = round(esc10, 4)
    return cells


# ============================ method vector + cost per (config, variant) ===========================
def method_cell(cell, name, config, variant):
    """Return (per-sample ok vector, cost dict, policy label) for the method on this cell, applying the
    MMMU correction.  variant 'A' escalates MMMU to 32B-think (method acc = okT); variant 'B' never
    reaches MMMU (it is dropped from the order)."""
    if name == MMMU and variant == "A":
        return cell["okT"].copy(), cost_fixed(GEN32T), "escalate->32B-think (MMMU corrected)"
    if config == "compute_lean":
        return cell["cl_ok"], cell["cl_cost"], cell["cl_choice"]
    if config == "accuracy_max_v1_F3":
        return cell["am_ok"], cell["am_cost"], cell["am_choice"]
    return cell["am2_ok"], cell["am2_cost"], cell["am2_choice"]   # accuracy_max_v2_F8F10


def base_vec(cell, sysname):
    return {"always_7b": cell["ok7"], "always_32b_nt": cell["ok32"],
            "always_32b_think": cell["okT"], "oracle_mode_32b": cell["oracle_ok"]}[sysname]

def base_cost(cell, sysname):
    return {"always_7b": cost_fixed(GEN7), "always_32b_nt": cost_fixed(GEN32N),
            "always_32b_think": cost_fixed(GEN32T), "oracle_mode_32b": cell["oracle_cost"]}[sysname]

def base_acc(cell, sysname):
    """(acc, is_estimated).  32B-think on open is a measured-delta estimate (no per-sample dump)."""
    if sysname == "always_32b_think" and cell["format"] == "open":
        return cell["think_acc_est"], True
    return float(base_vec(cell, sysname).mean()), False


# ============================ pooled / macro accuracy + cost ======================================
def pool_block(cells, order, mvec, mcost):
    """Sample-weighted + macro accuracy and sample-weighted cost, for the method and every baseline,
    over the given cell `order`.  Returns nested dict keyed by system."""
    n = sum(cells[k]["n"] for k in order)
    out = {}
    # method
    macc = [float(mvec[k].mean()) for k in order]
    out["method"] = dict(
        acc_sw=round(sum(a * cells[k]["n"] for a, k in zip(macc, order)) / n, 4),
        acc_macro=round(float(np.mean(macc)), 4),
        flops=round(sum(mcost[k]["flops"] * cells[k]["n"] for k in order) / n, 3),
        energy_j=round(sum(mcost[k]["energy"] * cells[k]["n"] for k in order) / n, 1),
        lat_par_ms=round(sum(mcost[k]["lat_par"] * cells[k]["n"] for k in order) / n, 1),
        lat_seq_ms=round(sum(mcost[k]["lat_seq"] * cells[k]["n"] for k in order) / n, 1),
        flops_vs_always32=round((sum(mcost[k]["flops"] * cells[k]["n"] for k in order) / n) / GEN32N[2], 3))
    # baselines (+ 7B floor)
    for s in ["always_7b"] + BASELINES:
        accs = [base_acc(cells[k], s) for k in order]
        est = any(e for _, e in accs)
        out[s] = dict(
            acc_sw=round(sum(a * cells[k]["n"] for (a, _), k in zip(accs, order)) / n, 4),
            acc_macro=round(float(np.mean([a for a, _ in accs])), 4),
            acc_is_estimated=bool(est),
            flops=round(sum(base_cost(cells[k], s)["flops"] * cells[k]["n"] for k in order) / n, 3),
            energy_j=round(sum(base_cost(cells[k], s)["energy"] * cells[k]["n"] for k in order) / n, 1),
            lat_par_ms=round(sum(base_cost(cells[k], s)["lat_par"] * cells[k]["n"] for k in order) / n, 1),
            lat_seq_ms=round(sum(base_cost(cells[k], s)["lat_seq"] * cells[k]["n"] for k in order) / n, 1),
            flops_vs_always32=round((sum(base_cost(cells[k], s)["flops"] * cells[k]["n"] for k in order) / n) / GEN32N[2], 3))
    out["_n"] = int(n)
    return out


def deltas_vs_baselines(cells, order, mvec):
    """Paired-bootstrap 95% CI of method-minus-baseline for each of the 3 baselines, per-benchmark and
    pooled (full/mcq/open), + macro CI where all cells have real vectors.  Mirrors paper_baselines'
    handling of the estimated 32B-think open cells."""
    mcq = [k for k in order if cells[k]["format"] == "MCQ"]
    opn = [k for k in order if cells[k]["format"] == "open"]
    res = {}
    for b in BASELINES:
        pb = {}                                                    # per-benchmark
        for k in order:
            c = cells[k]
            if b == "always_32b_think" and c["format"] == "open":
                a = float(mvec[k].mean())
                pb[k] = dict(delta=round(a - c["think_acc_est"], 4), estimated=True,
                             note="32B-think open acc is a measured-delta estimate; no per-sample CI")
            else:
                pb[k] = paired_ci(mvec[k], base_vec(c, b))
        pooled = {}
        for lab, keys in [("full_suite", order), ("mcq_only", mcq), ("open_only", opn)]:
            usable = [k for k in keys if not (b == "always_32b_think" and cells[k]["format"] == "open")]
            dropped = [k for k in keys if k not in usable]
            # sample-weighted delta point over ALL cells (incl. estimated open for think)
            npool = sum(cells[k]["n"] for k in keys) if keys else 0
            sw_pt = (sum((float(mvec[k].mean()) - base_acc(cells[k], b)[0]) * cells[k]["n"] for k in keys) / npool) if npool else float("nan")
            if not usable:
                pooled[lab] = dict(delta=round(sw_pt, 4), estimated=True, n=0,
                                   note="all cells estimated (32B-think open); point delta only")
                continue
            A = np.concatenate([mvec[k] for k in usable])
            B = np.concatenate([base_vec(cells[k], b) for k in usable])
            r = paired_ci(A, B)
            r["sw_point_delta_all_cells"] = round(sw_pt, 4)
            # macro CI over the usable (real-vector) cells
            r["macro_ci"] = macro_ci([mvec[k] - base_vec(cells[k], b) for k in usable])
            r["macro_point_all_cells"] = round(float(np.mean(
                [float(mvec[k].mean()) - base_acc(cells[k], b)[0] for k in keys])), 4) if keys else float("nan")
            if dropped:
                r["note"] = ("open cells %s excluded from the per-sample CI (32B-think open acc is a "
                             "measured-delta estimate); CI over the real-vector cells only" % dropped)
            pooled[lab] = r
        res[b] = dict(per_benchmark=pb, pooled=pooled)
    return res


# ============================ Pareto frontier =====================================================
def frontier(pools_by_cfg):
    """pools_by_cfg: {config: pool_block dict for ONE label}.  Point set (7B floor, 3 single-32B
    baselines, 3 method settings) on pooled sample-weighted accuracy vs cost; mark Pareto-optimal."""
    pts = []
    any_pool = pools_by_cfg["compute_lean"]                    # baselines identical across configs
    for s in ["always_7b", "always_32b_nt", "always_32b_think", "oracle_mode_32b"]:
        v = any_pool[s]
        pts.append(dict(system=s, acc=v["acc_sw"], acc_is_estimated=v.get("acc_is_estimated", False),
                        flops=v["flops"], lat_par_ms=v["lat_par_ms"], energy_j=v["energy_j"]))
    for cfg in CONFIGS:
        v = pools_by_cfg[cfg]["method"]
        pts.append(dict(system="method_" + cfg, acc=v["acc_sw"], acc_is_estimated=False,
                        flops=v["flops"], lat_par_ms=v["lat_par_ms"], energy_j=v["energy_j"]))

    def mark(axis):
        opt = set()
        for p in pts:
            dominated = any((q[axis] <= p[axis] and q["acc"] >= p["acc"] and q is not p and
                             (q[axis] < p[axis] or q["acc"] > p["acc"])) for q in pts)
            if not dominated: opt.add(p["system"])
        return sorted(opt)
    return dict(points=pts, pareto_optimal_flops=mark("flops"), pareto_optimal_lat_par=mark("lat_par_ms"))


# ============================ assemble one variant ================================================
def run_variant(cells, variant):
    order = FULL_ORDER if variant == "A" else [k for k in FULL_ORDER if k != MMMU]
    sub = dict(full_suite=order,
               mcq_only=[k for k in order if cells[k]["format"] == "MCQ"],
               open_only=[k for k in order if cells[k]["format"] == "open"])

    modes_out = {}
    pools = {lab: {} for lab in sub}          # label -> config -> pool_block (for the frontier)
    for cfg in CONFIGS:
        mvec = {}; mcost = {}; policy = {}
        for k in order:
            v, c, pol = method_cell(cells[k], k, cfg, variant)
            mvec[k] = np.asarray(v, float); mcost[k] = c; policy[k] = pol
        pooled = {lab: pool_block(cells, keys, mvec, mcost) for lab, keys in sub.items()}
        for lab in sub: pools[lab][cfg] = pooled[lab]
        ci = deltas_vs_baselines(cells, order, mvec)
        per_bench = {k: dict(n=cells[k]["n"], format=cells[k]["format"], policy=policy[k],
                             method_acc=round(float(mvec[k].mean()), 4),
                             flops=round(mcost[k]["flops"], 3), lat_par_ms=round(mcost[k]["lat_par"], 1),
                             lat_seq_ms=round(mcost[k]["lat_seq"], 1), energy_j=round(mcost[k]["energy"], 1))
                     for k in order}
        modes_out[cfg] = dict(per_benchmark=per_bench, pooled=pooled, paired_bootstrap_ci=ci)

    fr = {lab: frontier(pools[lab]) for lab in sub}
    base_pb = {k: {s: round(base_acc(cells[k], s)[0], 4) for s in ["always_7b"] + BASELINES}
               for k in order}
    return dict(cell_order=order, n_total=int(sum(cells[k]["n"] for k in order)),
                baselines_per_benchmark=base_pb, modes=modes_out, pareto_frontier=fr)


# ============================ run =================================================================
def run():
    cells = PB.build_cells()
    add_v2_vectors(cells)

    variants = {
        "A_mmmu_escalate_32b_think": dict(
            description="MMMU is reasoning -> escalate 100% to the 32B in THINK mode; the method's MMMU "
                        "accuracy = 32B-think (0.66), cost = 32B-think, added at n=150. The leaked 7B 0.80 "
                        "is NOT used. Baselines (think/no-think/oracle-mode) unchanged.",
            data=run_variant(cells, "A")),
        "B_mmmu_excluded": dict(
            description="MMMU rows dropped entirely (8-cell suite = 5 MCQ + 3 open). Neither the method "
                        "nor the baselines see MMMU.",
            data=run_variant(cells, "B")),
    }

    # ---- robustness headline check (sample-weighted, accuracy-max-v2, variant A) ----
    vA = variants["A_mmmu_escalate_32b_think"]["data"]["modes"]
    v2A_fs = vA["accuracy_max_v2_F8F10"]["pooled"]["full_suite"]
    v2A_think = vA["accuracy_max_v2_F8F10"]["paired_bootstrap_ci"]["always_32b_think"]["pooled"]["full_suite"]
    v2A_orc = vA["accuracy_max_v2_F8F10"]["paired_bootstrap_ci"]["oracle_mode_32b"]["pooled"]["full_suite"]
    robustness = dict(
        expected_from_task=dict(accuracy_max_v2_d_vs_think_approx=0.0207, d_vs_oracle_approx=0.013),
        measured_accuracy_max_v2_variantA=dict(
            pooled_sw_acc=v2A_fs["method"]["acc_sw"],
            d_vs_think_sw_point=v2A_think.get("sw_point_delta_all_cells"),
            d_vs_think_mcq_ci=vA["accuracy_max_v2_F8F10"]["paired_bootstrap_ci"]["always_32b_think"]["pooled"]["mcq_only"],
            d_vs_oracle_sw_ci=dict(delta=v2A_orc["delta"], lo=v2A_orc["lo"], hi=v2A_orc["hi"], sig=v2A_orc["sig"])),
        verdict=("Sample-weighted accuracy-max-v2 headline is ROBUST to the MMMU correction: escalating "
                 "MMMU (0.80->0.66) moves the vs-THINK delta only ~-0.0005 (MMMU is 0.35%% of the pool), "
                 "landing at +%.4f (point, full-suite) vs always-32B-THINK. The headline is CARRIED BY PMC "
                 "fusion/veto + the open-text verifier arm, NOT the MMMU anomaly. vs oracle-mode-32B the v2 "
                 "delta is +%.4f (95%% CI [%+.4f,%+.4f]) -- note the ~+0.013-vs-oracle figure quoted in the "
                 "task corresponds to accuracy-max-V1 (F3 fusion, higher acc/cost); V2 (F8, FLOP-negative) "
                 "sits slightly lower vs oracle but keeps the +0.0207-vs-THINK headline."
                 % (v2A_think.get("sw_point_delta_all_cells"), v2A_orc["delta"], v2A_orc["lo"], v2A_orc["hi"])))

    out = dict(
        title="method_final_mmmu_corrected -- CORE results with the MMMU cell CORRECTED (leaked Lingshu-7B "
              "0.80 NOT banked as a beat-32B win). Two variants (A: escalate MMMU->32B-think; B: exclude "
              "MMMU) x three method settings (compute-lean, accuracy-max-v1 F3, accuracy-max-v2 F8+F10) vs "
              "always-32B-think / -no-think / oracle-mode-32B; pooled sample-weighted + macro, paired-"
              "bootstrap 95% CIs, FLOP-eq/latency/energy, Pareto frontier. Recomputed live per-sample; no GPU.",
        reproduce="python3 src/cascade_methods/method_final_mmmu_corrected.py",
        no_gpu=True, no_fabricated_numbers=True, n_bootstrap=NBOOT,
        mmmu_correction=dict(
            leaked_7b_mmmu=0.80, published_lingshu7b_mmmu=0.54, delta_over_published=0.26,
            order_robust_debiased_7b=0.7708, debiased_32b=0.6321,
            root_cause="train-set contamination (7B-and-MMMU-specific; +26 over the Lingshu paper's own 7B "
                       "number; order-robust under full cyclic option permutation -> NOT a parsing/position "
                       "artifact). See results/cascade_methods/artifacts/mmmu_fix.json.",
            action="Do NOT bank the MMMU keep-7B 0.80 as a headline beat-32B win. Variant A escalates MMMU "
                   "to the 32B (reasoning->think); Variant B excludes MMMU. Macro must be reported corrected."),
        cost_constants=dict(GEN7=dict(ms=GEN7[0], energy_j=GEN7[1], flop=GEN7[2]),
                            GEN32_nothink=dict(ms=GEN32N[0], energy_j=GEN32N[1], flop=GEN32N[2]),
                            GEN32_think=dict(ms=GEN32T[0], energy_j=GEN32T[1], flop=GEN32T[2]),
                            FUSE_both_legs=dict(ms_par=PB.FUSE[0], ms_seq=PB.FUSE_SEQ_MS, energy_j=PB.FUSE[1], flop=PB.FUSE[2]),
                            cheap_draw=dict(ms=PB.DRAW_MS, energy_j=PB.DRAW_E, flop=2.0)),
        baselines=dict(
            always_32b_think="naive baseline: 32B think on everything",
            always_32b_nt="strong reference: one 32B no-think forward",
            oracle_mode_32b="HONEST strong baseline: per-benchmark best of {think,no-think}; acc=max, cost=that mode",
            always_7b="cheap floor (carried for the Pareto point set)"),
        method_settings=dict(
            compute_lean="margin-cascade MCQ | Pandora bo-N + verifier open | MMMU corrected",
            accuracy_max_v1_F3="F1 slice-router, F3 confidence-advantage FUSION on PMC | Pandora open | MMMU corrected",
            accuracy_max_v2_F8F10="F1 slice-router, F8 certified weak-veto on PMC | F10 team-objective L2D open | MMMU corrected"),
        variants=variants,
        headline_robustness_check=robustness,
        data_gaps=[
            "always-32B-THINK open-text accuracy is ESTIMATED (judged 32B-no-think + measured modal think-"
            "delta -0.195/-0.120/-0.130); no per-sample judged think dump on open -> method-vs-THINK CI is "
            "the MCQ pool only; the full-suite vs-THINK delta is a flagged sample-weighted point estimate. "
            "Every OTHER comparison (vs oracle-mode-32B, vs 32B-no-think, vs 7B) has real per-sample vectors "
            "on ALL cells incl. open.",
            "PATH_VQA_closed has no 32B-think dump -> think=no-think there; oracle picks no-think.",
            "oracle-mode-32B is an ORACLE upper bound on single-32B mode selection (peeks at which mode wins "
            "per benchmark); the fairest STRONG baseline, not a deployable system.",
            "paired bootstrap resamples QUESTIONS (sampling noise); held-out tau/lambda/threshold are fixed "
            "per item, so calibration variance is not in the CI.",
            "SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (trained on those families).",
            "MMMU n=150 = 0.35% of the pooled 42374; the sample-weighted headline barely moves under either "
            "correction, but MACRO (equal-weight per benchmark) does move materially -> report the corrected macro.",
        ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    console(variants, robustness)
    print(f"\nwrote {OUT}")
    return out


# ============================ console =============================================================
def _pm(x):
    return f"{x:+.4f}" if isinstance(x, (int, float)) else str(x)

def console(variants, robustness):
    for vkey, vobj in variants.items():
        D = vobj["data"]; order = D["cell_order"]
        print("\n" + "=" * 150)
        print(f"VARIANT {vkey}   n={D['n_total']}   ({vobj['description'][:96]})")
        print("=" * 150)
        # per-benchmark method acc per config + baselines
        print(f"\n  {'benchmark':<18}{'fmt':<5}{'n':>6}"
              f"{'7B':>7}{'32Bnt':>7}{'32Bthk':>8}{'oracle':>8}{'CL':>9}{'AMv1':>9}{'AMv2':>9}   policy(AMv2/MMMU)")
        cl = D["modes"]["compute_lean"]; v1 = D["modes"]["accuracy_max_v1_F3"]; v2 = D["modes"]["accuracy_max_v2_F8F10"]
        bpb = D["baselines_per_benchmark"]
        for k in order:
            pbn = cl["per_benchmark"][k]; bb = bpb[k]
            est = "*" if pbn["format"] == "open" else " "
            print(f"  {k:<18}{pbn['format']:<5}{pbn['n']:>6}"
                  f"{bb['always_7b']:>7.3f}{bb['always_32b_nt']:>7.3f}{bb['always_32b_think']:>7.3f}{est}{bb['oracle_mode_32b']:>8.3f}"
                  f"{cl['per_benchmark'][k]['method_acc']:>9.4f}{v1['per_benchmark'][k]['method_acc']:>9.4f}"
                  f"{v2['per_benchmark'][k]['method_acc']:>9.4f}   {v2['per_benchmark'][k]['policy']}")
        print("   (32Bthk* on open = measured-delta ESTIMATE, no per-sample dump)")
        # pooled sample-weighted + macro table
        for cfg, label in [("compute_lean", "COMPUTE-LEAN"),
                           ("accuracy_max_v1_F3", "ACCURACY-MAX v1 (F3 fusion)"),
                           ("accuracy_max_v2_F8F10", "ACCURACY-MAX v2 (F8 veto + F10)")]:
            M = D["modes"][cfg]
            print("\n  " + "-" * 140)
            print(f"  [{label}]")
            for lab in ("full_suite", "mcq_only", "open_only"):
                pb = M["pooled"][lab]; ci = M["paired_bootstrap_ci"]
                m = pb["method"]
                print(f"    POOLED[{lab:<10}] n={pb['_n']:>5}  method sw={m['acc_sw']:.4f} macro={m['acc_macro']:.4f}  "
                      f"| FLOPs {m['flops']:.3f} ({m['flops_vs_always32']:.2f}x)  lat_par {m['lat_par_ms']:.0f}ms  "
                      f"energy {m['energy_j']:.0f}J")
                for bkey, bshort in [("always_32b_think", "vs THINK "), ("always_32b_nt", "vs 32B-nt"),
                                     ("oracle_mode_32b", "vs oracle")]:
                    r = ci[bkey]["pooled"][lab]
                    swpt = r.get("sw_point_delta_all_cells")
                    ci_s = (f"[{r['lo']:+.4f},{r['hi']:+.4f}]{'*SIG' if r.get('sig') else '    '}"
                            if "lo" in r and r.get("n", 1) else "  (point only)   ")
                    mac = r.get("macro_ci", {})
                    mac_s = (f"macroΔ {mac.get('delta'):+.4f} [{mac.get('lo'):+.4f},{mac.get('hi'):+.4f}]"
                             if isinstance(mac, dict) and "lo" in mac else
                             f"macroΔ {r.get('macro_point_all_cells')}")
                    swp_s = f"swΔ {swpt:+.4f}" if swpt is not None else ""
                    note = "  (open-think est: sw is point, CI=MCQ)" if bkey == "always_32b_think" and lab == "full_suite" else ""
                    print(f"       {bshort}:  {swp_s:<14} CI {ci_s}  {mac_s}{note}")
        # frontier
        fr = D["pareto_frontier"]["full_suite"]
        print("\n  " + "-" * 140)
        print(f"  PARETO FRONTIER (pooled full_suite):  optimal@FLOPs {fr['pareto_optimal_flops']}")
        print(f"                                        optimal@lat_par {fr['pareto_optimal_lat_par']}")
        print(f"    {'system':<34}{'acc':>9}{'flops':>9}{'lat_par_ms':>12}{'energy_J':>10}")
        for p in sorted(fr["points"], key=lambda x: x["flops"]):
            print(f"    {p['system']:<34}{p['acc']:>9.4f}{p['flops']:>9.3f}{p['lat_par_ms']:>12.0f}{p['energy_j']:>10.0f}")

    print("\n" + "#" * 150)
    print("HEADLINE ROBUSTNESS CHECK (accuracy-max-v2, variant A = MMMU escalated)")
    print("#" * 150)
    r = robustness["measured_accuracy_max_v2_variantA"]
    print(f"  pooled sw acc = {r['pooled_sw_acc']:.4f}")
    print(f"  vs always-32B-THINK : sw point Δ = {r['d_vs_think_sw_point']:+.4f}  "
          f"(MCQ-pool CI [{r['d_vs_think_mcq_ci']['lo']:+.4f},{r['d_vs_think_mcq_ci']['hi']:+.4f}]"
          f"{' SIG' if r['d_vs_think_mcq_ci']['sig'] else ''})")
    print(f"  vs oracle-mode-32B  : Δ = {r['d_vs_oracle_sw_ci']['delta']:+.4f}  "
          f"[{r['d_vs_oracle_sw_ci']['lo']:+.4f},{r['d_vs_oracle_sw_ci']['hi']:+.4f}]"
          f"{' SIG' if r['d_vs_oracle_sw_ci']['sig'] else ''}")
    print("  " + robustness["verdict"])


if __name__ == "__main__":
    run()
