#!/usr/bin/env python3
"""
cost_decomposition_verify.py -- INDEPENDENT VERIFICATION of ATTACK 1
(results/cascade_methods/artifacts/cost_decomposition_2026-08-12.json), plus the one thing the
brief asked for that the artifact does not contain: the FULL macro cost-accuracy FRONTIER.

WHY THIS EXISTS.  cost_decomposition_2026-08-12.json was produced earlier in the same round by
cost_decomposition.py.  This file re-derives its load-bearing numbers from the raw per-item
correctness vectors and the frozen cost tables with an INDEPENDENT implementation and an
INDEPENDENT bootstrap stream, and then adds:

  F1  the exact CI-CONSTRAINED minimum-cost assignment over the whole 6^8 menu, where the
      pre-registered constraint (paired-bootstrap CI lower bound of macro delta vs
      always-32B-direct >= -0.0029) is evaluated EXACTLY for every candidate, not checked after
      a point-estimate argmin.
  F2  the macro cost-accuracy Pareto envelope (the "macro frontier"), each point annotated with
      its CI, its guardrail flags, and its resident-weight FOOTPRINT.
  F3  a permutation null for F1: the same enumerate-under-CI machinery run on arm labels that
      have been permuted independently per cell, so the selection carries no signal.

EVAL-VISIBILITY WARNING.  F1/F2 are fitted with full eval visibility.  They are DIAGNOSTIC UPPER
BOUNDS on what any cross-fit selector could reach.  The honest achievable numbers are the
nested-CV rows in cost_decomposition_2026-08-12.json:Q3.honest_nested_cv, which are quoted here
verbatim and NOT recomputed.

NO GPU.  NO NEW INFERENCE.  Pure numpy over stored vectors.
"""
import json
import os

import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
OUT = os.path.join(ART, "cost_decomposition_2026-08-12_verification.json")

SEED = 20260812777          # DIFFERENT from the artifact's 20260812 -- an independent stream
NBOOT = 10000
TIE_TOL = 0.0029
R32_CHARGED = 4.57
R32_DERIVED = 3.816

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
BASE = "always_32b_direct"
MENU = ["always_7b", "always_32b_direct", "always_32b_reasoning",
        "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]

Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
MACD = json.load(open(os.path.join(ART, "_selector_rerun_parts/macro_disjoint.json")))
PUB = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))["per_arm"]["disjoint"]
CF = json.load(open(os.path.join(ART, "cost_floor_2026-08-10.json")))
FLOPD = json.load(open(os.path.join(ART, "flop_ratio_derivation_2026-08-03.json")))
A1 = json.load(open(os.path.join(ART, "cost_decomposition_2026-08-12.json")))
PCC = MACD["cost"]["per_cell_as_charged"]
ARMDEC = CF["arm_decomposition"]["table"]

OK = {(c, s): Z[f"{c}|{s}"].astype(np.float64) for c in CELLS for s in MENU}
NIT = {c: len(Z[f"{c}|{BASE}"]) for c in CELLS}

rep = {}
OBSERVED = {"x": None, "d": None}


# ---------------------------------------------------------------------------------------------
# cost tables
# ---------------------------------------------------------------------------------------------
def cell_flops(c, arm, r32):
    a = ARMDEC[c][arm]
    return a["n_gen7"] * 1.0 + a["n_ver7"] * 1.0 + a["n_32b"] * r32


COST_A = np.array([[cell_flops(c, a, R32_CHARGED) for a in MENU] for c in CELLS])   # 8 x 6
COST_D = np.array([[cell_flops(c, a, R32_DERIVED) for a in MENU] for c in CELLS])
LATP = np.array([[PCC[c][a]["lat_par_ms"] for a in MENU] for c in CELLS])
LATS = np.array([[PCC[c][a]["lat_seq_ms"] for a in MENU] for c in CELLS])
ENER = np.array([[PCC[c][a]["energy_j"] for a in MENU] for c in CELLS])
ACC = np.array([[OK[(c, a)].mean() for a in MENU] for c in CELLS])                  # 8 x 6
BASEI = MENU.index(BASE)

# which arms ever put a 7B forward on the critical path / ever call the 32B
USES_7B = np.array([any(ARMDEC[c][a]["n_gen7"] > 0 or ARMDEC[c][a]["n_ver7"] > 0 for c in CELLS)
                    for a in MENU])
USES_VERIFIER = np.array([any(ARMDEC[c][a]["n_ver7"] > 0 for c in CELLS) for a in MENU])
USES_32B = np.array([any(ARMDEC[c][a]["n_32b"] > 0 for c in CELLS) for a in MENU])
# per (cell, arm) -- what that CELL actually needs
NEED7 = np.array([[(ARMDEC[c][a]["n_gen7"] > 0 or ARMDEC[c][a]["n_ver7"] > 0) for a in MENU]
                  for c in CELLS])
NEEDV = np.array([[ARMDEC[c][a]["n_ver7"] > 0 for a in MENU] for c in CELLS])
NEED32 = np.array([[ARMDEC[c][a]["n_32b"] > 0 for a in MENU] for c in CELLS])

P7 = 8292166656
P32 = 33452718336
PLORA = 47589376
B7, B32, BLORA = 16584333312, 66905436672, 190442760
GIB = float(2 ** 30)


def footprint(mask7, maskv, mask32):
    p = (P7 if mask7 else 0) + (PLORA if maskv else 0) + (P32 if mask32 else 0)
    b = (B7 if mask7 else 0) + (BLORA if maskv else 0) + (B32 if mask32 else 0)
    res = []
    if mask7:
        res.append("Lingshu-7B base")
    if maskv:
        res.append("verifier LoRA (adapter on the same 7B base)")
    if mask32:
        res.append("Lingshu-32B")
    return dict(resident=res, params=int(p), params_B=round(p / 1e9, 4),
                weight_bytes=int(b), weight_GiB=round(b / GIB, 4), needs_32B=bool(mask32))


# ---------------------------------------------------------------------------------------------
# bootstrap: one shared item-resample stream per cell, reused by every arm (paired)
# ---------------------------------------------------------------------------------------------
def boot_matrices():
    rng = np.random.default_rng(SEED)
    out = {}
    for c in CELLS:
        n = NIT[c]
        cols = np.stack([OK[(c, a)] for a in MENU], axis=1)          # n x 6
        acc = np.empty((NBOOT, len(MENU)))
        step = max(1, int(4e6 // max(n, 1)))
        i = 0
        while i < NBOOT:
            k = min(step, NBOOT - i)
            idx = rng.integers(0, n, size=(k, n))
            acc[i:i + k] = cols[idx].mean(axis=1)
            i += k
        out[c] = acc
    return out


BOOT = boot_matrices()                                                # cell -> NBOOT x 6
BDELTA = np.stack([BOOT[c] - BOOT[c][:, [BASEI]] for c in CELLS], axis=0)   # 8 x NBOOT x 6


def macro_ci(arm_idx):
    """arm_idx: length-8 int array. Returns (delta, lo, hi) of the macro delta vs 32B-direct."""
    d = float(np.mean(ACC[np.arange(8), arm_idx] - ACC[:, BASEI]))
    bs = BDELTA[np.arange(8), :, arm_idx].mean(axis=0)                 # NBOOT
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return d, float(lo), float(hi)


def cell_ci(c, arm):
    j = CELLS.index(c)
    a = MENU.index(arm)
    d = float(ACC[j, a] - ACC[j, BASEI])
    lo, hi = np.percentile(BDELTA[j, :, a], [2.5, 97.5])
    return dict(delta=round(d, 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                worse_sig=bool(hi < 0))


def describe(arm_idx, name, kind):
    a = np.asarray(arm_idx)
    d, lo, hi = macro_ci(a)
    m7 = bool(NEED7[np.arange(8), a].any())
    mv = bool(NEEDV[np.arange(8), a].any())
    m32 = bool(NEED32[np.arange(8), a].any())
    g = {c: cell_ci(c, MENU[a[i]]) for i, c in enumerate(CELLS)}
    return dict(
        name=name, kind=kind,
        assignment={c: MENU[a[i]] for i, c in enumerate(CELLS)},
        macro_acc=round(float(ACC[np.arange(8), a].mean()), 4),
        delta_vs_direct=round(d, 4), lo=round(lo, 4), hi=round(hi, 4),
        meets_constraint=bool(lo >= -TIE_TOL),
        not_significantly_worse=bool(hi >= 0),
        cost=dict(
            as_charged_R32_4p57=dict(
                flopeq=round(float(COST_A[np.arange(8), a].mean()), 4),
                x_direct=round(float(COST_A[np.arange(8), a].mean() / R32_CHARGED), 4)),
            derived_R32_3p816=dict(
                flopeq=round(float(COST_D[np.arange(8), a].mean()), 4),
                x_direct=round(float(COST_D[np.arange(8), a].mean() / R32_DERIVED), 4)),
            lat_par_ms=round(float(LATP[np.arange(8), a].mean()), 1),
            lat_seq_ms=round(float(LATS[np.arange(8), a].mean()), 1),
            energy_j=round(float(ENER[np.arange(8), a].mean()), 1)),
        footprint=footprint(m7, mv, m32),
        guardrail=g,
        guardrail_flags=[c for c in CELLS if g[c]["worse_sig"]])


# =============================================================================================
# V.  NULL TESTS -- independent recomputation
# =============================================================================================
def v1():
    """V1 -- per-cell and macro accuracy for all 6 deployable systems, from the raw vectors."""
    dev_a, dev_c, rows = [], [], {}
    for a in MENU:
        j = MENU.index(a)
        macro = float(ACC[:, j].mean())
        cost = float(COST_A[:, j].mean())
        da = abs(macro - PUB["macro_acc"][a])
        dc = abs(cost - PUB["cost_macro"][a]["flops"])
        percell = {c: round(float(ACC[i, j]), 6) for i, c in enumerate(CELLS)}
        dpc = max(abs(percell[c] - PUB["per_cell_acc"][c][a]) for c in CELLS)
        dev_a.append(max(da, dpc))
        dev_c.append(dc)
        rows[a] = dict(macro_recomputed=round(macro, 6), macro_published=PUB["macro_acc"][a],
                       abs_dev_macro=round(da, 8), max_abs_dev_per_cell=round(dpc, 8),
                       cost_recomputed=round(cost, 6),
                       cost_published=PUB["cost_macro"][a]["flops"], abs_dev_cost=round(dc, 8))
    ratio = float(COST_A[:, MENU.index("method_accuracy_max_veto")].mean() / R32_CHARGED)
    ok = max(dev_a) < 1e-4 and max(dev_c) < 1e-3
    return dict(
        name="V1 -- independent recomputation of per-cell + macro accuracy and as-charged cost",
        source_vectors="results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz",
        source_cost="results/cascade_methods/artifacts/cost_floor_2026-08-10.json"
                    ":arm_decomposition.table  (n_gen7, n_ver7, n_32b)",
        formula="cell_flopeq = n_gen7*1.0 + n_ver7*1.0 + n_32b*R32;  macro = mean over the 8 cells",
        per_system=rows,
        max_abs_dev_accuracy=round(max(dev_a), 8),
        max_abs_dev_cost_flops=round(max(dev_c), 8),
        macro_ratio_accmax_vs_direct=round(ratio, 5),
        published_ratio=1.74,
        artifact_ratio=A1["Q0_cost_decomposition"]["shipped_accuracy_max_as_charged"]["macro_x_direct"],
        verdict="PASS" if ok else "FAIL")


def v2():
    """V2 -- the (cell x stage) table in the artifact re-sums to the per-cell cost and to 1.740x."""
    dev, macro = [], 0.0
    for view, r32 in [("shipped_accuracy_max_as_charged", R32_CHARGED),
                      ("shipped_accuracy_max_derived_R32", R32_DERIVED),
                      ("shipped_compute_lean_as_charged", R32_CHARGED)]:
        blk = A1["Q0_cost_decomposition"][view]["per_cell"]
        arm = "method_compute_lean" if "lean" in view else "method_accuracy_max_veto"
        tot = []
        for c in CELLS:
            s = sum(blk[c]["stages"].values())
            dev.append(abs(s - blk[c]["total_flopeq"]))
            dev.append(abs(s - cell_flops(c, arm, r32)))
            tot.append(s)
        if view == "shipped_accuracy_max_as_charged":
            macro = float(np.mean(tot)) / R32_CHARGED
    return dict(
        name="V2 -- the artifact's per-stage table re-sums to the per-cell cost, independently recomputed",
        max_abs_dev_stage_sum_vs_cell_total=round(max(dev), 8),
        macro_x_direct_from_stage_sums=round(macro, 5),
        published=1.74,
        verdict="PASS" if max(dev) < 1e-3 else "FAIL")


def v3():
    """V3 -- Q1's waste arithmetic reproduces from the per-item conditional means it publishes."""
    dev, rows = [], {}
    for pol in ["shipped_accuracy_max", "shipped_compute_lean"]:
        blk = A1["Q1_waste_on_escalated_questions"][pol]["per_cell"]
        w = []
        for c in CELLS:
            r = blk[c]
            # cheap-side FLOP-eq actually spent on items that are escalated anyway:
            #   open cells  -> esc * meanN_given_escalate * (1 generation + 1 verifier forward)
            #   MCQ cells   -> esc * 1 generation                (no verifier in the MCQ arm)
            mult = 2.0 if c in ("SLAKE_open", "VQA_RAD_open", "PATH_VQA_open") else 1.0
            recon = r["esc_rate"] * r["meanN_given_escalate"] * mult
            dev.append(abs(recon - r["wasted_flopeq"]))
            w.append(r["wasted_flopeq"])
        pub = A1["Q1_waste_on_escalated_questions"][pol]
        mw = float(np.mean(w))
        dev.append(abs(mw - pub["macro_wasted_flopeq"]))
        rows[pol] = dict(macro_wasted_recomputed=round(mw, 5),
                         macro_wasted_published=pub["macro_wasted_flopeq"],
                         macro_wasted_pct_recomputed=round(100 * mw / pub["macro_total_flopeq"], 3),
                         macro_wasted_pct_published=pub["macro_wasted_pct"])
    return dict(
        name="V3 -- Q1 waste = esc_rate x meanN_given_escalate x (gen + verifier), re-derived",
        per_policy=rows, max_abs_dev=round(max(dev), 6),
        verdict="PASS" if max(dev) < 5e-3 else "FAIL")


def v4():
    """V4 -- reproduce the artifact's pre-specified operating points on an independent bootstrap."""
    rows, dev = {}, []
    for op in A1["operating_points_prespecified"]:
        a = np.array([MENU.index(op["assignment"][c]) for c in CELLS]) \
            if "assignment" in op else None
        if a is None:
            continue
        r = describe(a, op["name"], "re-verified")
        dev += [abs(r["delta_vs_direct"] - op["delta_vs_direct"]),
                abs(r["cost"]["as_charged_R32_4p57"]["x_direct"]
                    - op["cost"]["as_charged_R32_4p57"]["x_direct"])]
        rows[op["name"]] = dict(
            delta_recomputed=r["delta_vs_direct"], delta_published=op["delta_vs_direct"],
            lo_recomputed=r["lo"], lo_published=op["lo"],
            hi_recomputed=r["hi"], hi_published=op["hi"],
            x_direct_recomputed=r["cost"]["as_charged_R32_4p57"]["x_direct"],
            x_direct_published=op["cost"]["as_charged_R32_4p57"]["x_direct"],
            meets_recomputed=r["meets_constraint"], meets_published=op["meets_constraint"],
            agrees=bool(r["meets_constraint"] == op["meets_constraint"]))
    return dict(
        name="V4 -- pre-specified operating points reproduced with an INDEPENDENT bootstrap stream "
             f"(seed {SEED} vs the artifact's 20260812)",
        note="point estimates must match exactly; CI bounds may differ by Monte-Carlo noise "
             "(~1e-4 at nboot=10,000)",
        per_point=rows,
        max_abs_dev_point_estimates=round(max(dev), 6),
        all_constraint_verdicts_agree=all(v["agrees"] for v in rows.values()),
        verdict="PASS" if max(dev) < 1e-3 and all(v["agrees"] for v in rows.values()) else "FAIL")


# =============================================================================================
# F1 / F2.  EXACT CI-CONSTRAINED MINIMUM AND THE MACRO FRONTIER
# =============================================================================================
def enumerate_all():
    """All 6^8 assignments as (cost_as_charged, macro_acc) with the index array recoverable."""
    n_arm = len(MENU)
    total = n_arm ** 8
    idx = np.arange(total)
    digits = np.empty((total, 8), dtype=np.int8)
    t = idx.copy()
    for j in range(7, -1, -1):
        digits[:, j] = t % n_arm
        t //= n_arm
    cost = np.zeros(total)
    acc = np.zeros(total)
    for j in range(8):
        cost += COST_A[j][digits[:, j]]
        acc += ACC[j][digits[:, j]]
    return digits, cost / 8.0, acc / 8.0


def f1_f2():
    digits, cost, acc = enumerate_all()
    base_acc = float(ACC[:, BASEI].mean())
    order = np.argsort(cost, kind="stable")

    # ---- F1: a CI lower bound can never exceed the point estimate, so pre-filter on the point
    #          estimate, then walk cost-ascending and evaluate the EXACT CI.
    cand = order[(acc[order] - base_acc) >= -TIE_TOL]
    n_tested, best = 0, None
    for k in cand:
        n_tested += 1
        a = digits[k].astype(int)
        _, lo, _ = macro_ci(a)
        if lo >= -TIE_TOL:
            best = a
            break
    f1 = describe(best, "EXACT CI-constrained minimum-cost assignment",
                  "EVAL-VISIBLE DIAGNOSTIC UPPER BOUND -- not achievable") if best is not None \
        else None
    if f1 is not None:
        f1["n_assignments_enumerated"] = int(len(cost))
        f1["n_passing_the_point_estimate_prefilter"] = int(len(cand))
        f1["n_ci_evaluations_before_first_feasible"] = int(n_tested)

    # ---- F2: Pareto envelope of (cost, macro accuracy)
    acc_o = acc[order]
    run = np.maximum.accumulate(acc_o)
    keep = np.empty(len(order), dtype=bool)
    keep[0] = True
    keep[1:] = acc_o[1:] > run[:-1] + 1e-12
    env_idx = order[keep]
    pts = []
    for k in env_idx:
        a = digits[k].astype(int)
        r = describe(a, f"envelope @ {cost[k] / R32_CHARGED:.4f}x", "eval-visible Pareto envelope")
        pts.append(dict(
            x_direct_as_charged=r["cost"]["as_charged_R32_4p57"]["x_direct"],
            x_direct_derived=r["cost"]["derived_R32_3p816"]["x_direct"],
            macro_acc=r["macro_acc"], delta_vs_direct=r["delta_vs_direct"],
            lo=r["lo"], hi=r["hi"], meets_constraint=r["meets_constraint"],
            lat_par_ms=r["cost"]["lat_par_ms"], energy_j=r["cost"]["energy_j"],
            weight_GiB=r["footprint"]["weight_GiB"], needs_32B=r["footprint"]["needs_32B"],
            n_guardrail_flags=len(r["guardrail_flags"]),
            assignment=r["assignment"]))
    feas = [p for p in pts if p["meets_constraint"]]
    return f1, dict(
        n_envelope_points=len(pts),
        cheapest_feasible_on_the_envelope=(min(feas, key=lambda p: p["x_direct_as_charged"])
                                           if feas else None),
        cheapest_point_any=pts[0] if pts else None,
        points=pts)


# =============================================================================================
# F3.  PERMUTATION NULL for the F1 machinery
# =============================================================================================
def f3(digits, n_draws=100):
    """BREAK the accuracy<->cost pairing.  Within each cell, independently permute WHICH ARM'S
    ACCURACY (and its bootstrap deltas) is attached to WHICH ARM'S COST, leaving the cost column
    untouched.  The per-cell menus of accuracies and of costs are both exactly preserved -- only
    their correspondence is destroyed.  The same 'cheapest assignment whose CI lower bound clears
    -0.0029' search is then run.  If the real answer sits inside this null's distribution, the
    ~0.86x floor is a property of the cost spread alone and says nothing about which arm is
    accurate on which cell.

    NOTE: permuting accuracy AND cost together is a NO-OP (the set of (acc, cost) pairs per cell is
    unchanged, so the optimum is identical); that version was run first and returned the observed
    0.8619x on 100/100 draws, which is the correct behaviour of a null that nulls nothing."""
    rng = np.random.default_rng(SEED + 1)
    base_acc = float(ACC[:, BASEI].mean())
    xs, deltas, feas = [], [], 0
    dg = digits
    cost_p = COST_A                                   # costs are NOT permuted
    for _ in range(n_draws):
        perm = np.stack([rng.permutation(len(MENU)) for _ in range(8)])   # 8 x 6
        acc_p = np.take_along_axis(ACC, perm, axis=1)
        bd_p = np.stack([BDELTA[j][:, perm[j]] for j in range(8)], axis=0)
        cst = np.zeros(len(dg))
        acc_t = np.zeros(len(dg))
        for j in range(8):
            cst += cost_p[j][dg[:, j]]
            acc_t += acc_p[j][dg[:, j]]
        cst /= 8.0
        acc_t /= 8.0
        order = np.argsort(cst, kind="stable")
        cand = order[(acc_t[order] - base_acc) >= -TIE_TOL]
        found, cnt = None, 0
        for k in cand:
            cnt += 1
            a = dg[k].astype(int)
            bs = bd_p[np.arange(8), :, a].mean(axis=0)
            lo = np.percentile(bs, 2.5)
            if lo >= -TIE_TOL:
                found = (cst[k] / R32_CHARGED, acc_t[k] - base_acc)
                break
            if cnt > 20000:
                break
        if found is not None:
            feas += 1
            xs.append(found[0])
            deltas.append(found[1])
    return dict(
        name="F3 -- accuracy<->cost decoupling permutation null for the F1 search",
        n_draws=n_draws,
        observed_x_direct=OBSERVED["x"],
        observed_delta=OBSERVED["d"],
        feasible_rate=round(feas / n_draws, 4),
        x_direct_mean=round(float(np.mean(xs)), 4) if xs else None,
        x_direct_min=round(float(np.min(xs)), 4) if xs else None,
        x_direct_p2p5=round(float(np.percentile(xs, 2.5)), 4) if xs else None,
        x_direct_p97p5=round(float(np.percentile(xs, 97.5)), 4) if xs else None,
        null_draws_at_or_below_observed=int(sum(1 for v in xs if v <= OBSERVED["x"] + 1e-9)),
        empirical_p_one_sided=(round(sum(1 for v in xs if v <= OBSERVED["x"] + 1e-9) / n_draws, 4)
                               if xs else None),
        delta_mean=round(float(np.mean(deltas)), 5) if deltas else None,
        reading="the real F1 solution must be CHEAPER than this null distribution; if it sits inside "
                "it, the cost floor is a property of the arm-cost spread, not of which arm is "
                "accurate on which cell")


# =============================================================================================
def main():
    rep["title"] = ("INDEPENDENT VERIFICATION of ATTACK 1 (cost_decomposition_2026-08-12.json), "
                    "plus the exact CI-constrained minimum and the macro cost-accuracy frontier")
    rep["date"] = "2026-08-12"
    rep["reproduce"] = "python3 src/cascade_methods/cost_decomposition_verify.py"
    rep["verifies"] = "results/cascade_methods/artifacts/cost_decomposition_2026-08-12.json"
    rep["no_gpu"] = True
    rep["no_new_inference"] = True
    rep["no_fabricated_numbers"] = True
    rep["not_abstention"] = ("every policy enumerated here returns an answer on every item; the "
                             "certified veto KEEPS the cheap answer.  CRITICAL RULE 6 respected.")
    rep["convention"] = ("MACRO, equal weight per reporting cell, 8 cells, Variant B (MMMU "
                         "excluded), n=42,224, CLEAN disjoint verifier.  Macro cost is NEVER "
                         "paired with a sample-weighted accuracy.")
    rep["numerics"] = dict(OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "unset"),
                           bootstrap="paired item resample, one shared stream per cell reused by "
                                     "every arm", nboot=NBOOT, seed=SEED,
                           seed_note="deliberately DIFFERENT from the artifact's 20260812",
                           tf32="not applicable -- pure numpy on CPU over stored int8 vectors",
                           row_order="the stored dump order, unchanged")
    rep["null_tests"] = dict(V1=v1(), V2=v2(), V3=v3(), V4=v4())
    rep["null_tests"]["ALL_PASSED"] = all(rep["null_tests"][k]["verdict"] == "PASS"
                                          for k in ["V1", "V2", "V3", "V4"])
    digits, _, _ = enumerate_all()
    f1, f2 = f1_f2()
    rep["F1_exact_CI_constrained_minimum"] = f1
    rep["F1_warning"] = ("EVAL-VISIBLE.  Fitted with full eval visibility; selecting the best arm "
                         "per cell on eval is exactly how a fake win has already been manufactured "
                         "in this project.  Quote the nested-CV row instead.")
    rep["F2_macro_frontier"] = f2
    OBSERVED["x"] = f1["cost"]["as_charged_R32_4p57"]["x_direct"] if f1 else None
    OBSERVED["d"] = f1["delta_vs_direct"] if f1 else None
    rep["F3_permutation_null"] = f3(digits)
    rep["honest_nested_cv_quoted_verbatim"] = A1["Q3_minimum_compute_at_parity"]["honest_nested_cv"]
    rep["honest_nested_cv_source"] = ("cost_decomposition_2026-08-12.json:"
                                      "Q3_minimum_compute_at_parity.honest_nested_cv -- quoted, "
                                      "NOT recomputed here")
    json.dump(rep, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print("null tests:", {k: rep["null_tests"][k]["verdict"] for k in ["V1", "V2", "V3", "V4"]})
    if f1:
        print("F1 exact CI-constrained min: x_direct=%s delta=%s [%s,%s] meets=%s"
              % (f1["cost"]["as_charged_R32_4p57"]["x_direct"], f1["delta_vs_direct"],
                 f1["lo"], f1["hi"], f1["meets_constraint"]))
    print("F2 envelope points:", f2["n_envelope_points"])
    print("F2 cheapest feasible:", (f2["cheapest_feasible_on_the_envelope"] or {}).get("x_direct_as_charged"))
    print("F3 null:", rep["F3_permutation_null"]["feasible_rate"],
          rep["F3_permutation_null"]["x_direct_mean"])


if __name__ == "__main__":
    main()
