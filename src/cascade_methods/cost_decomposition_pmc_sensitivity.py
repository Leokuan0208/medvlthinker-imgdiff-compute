#!/usr/bin/env python3
"""
cost_decomposition_pmc_sensitivity.py -- ATTACK 1 sensitivity.

WHY.  Every cheap policy on the Attack-1 frontier buys its non-inferiority slack from ONE cell:
PMC-VQA, where the certified-veto arm is +0.00954 over always-32B-direct.  On 2026-08-12 two
independent confounds landed on exactly that number:

  * an ANSWER-LETTER PRIOR (gold B+C = 73.6%, constant-C floor 37.8%); the 7B's predicted-letter
    distribution is closer to the gold prior than the 32B's, so any arm that imports 7B answers
    imports a better-matched prior.   [artifacts/pmcvqa_letterbias_audit_2026-08-12.json]
  * a HARNESS GRADING DEFECT at MedEvalKit/utils/utils.py:112 that under-scores the 32B roughly
    twice as hard as the 7B (+1.7559 pp vs +0.9991 pp recovered by a repaired grader; differential
    0.7568 pp).   [artifacts/pmcvqa_grader_defect_2026-08-12.json]

Under BOTH corrections the PMC-VQA cell delta falls +0.00954 -> +0.00305 [+0.00063, +0.00549].

WHAT THIS FILE DOES.  It re-solves the Attack-1 frontier with the PMC certified-veto delta shifted
down by that measured amount, and reports how much of the frontier survives.  This is a
CONSTANT-SHIFT SENSITIVITY on a measured delta, NOT a re-measurement: the per-item repaired-grader
vectors are not exported, so the shift is applied to the cell's delta distribution as a whole.
`method_accuracy_max_fusion` on PMC is REMOVED from the menu because its corrected value is
explicitly NOT MEASURED (pmcvqa_grader_defect_2026-08-12.json:A4.design).

No GPU.  Launch from the repo root:
    python3 src/cascade_methods/cost_decomposition_pmc_sensitivity.py
Appends `PMC_sensitivity` to results/cascade_methods/artifacts/cost_decomposition_2026-08-12.json
"""
import json
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
MAIN = os.path.join(ART, "cost_decomposition_2026-08-12.json")

SEED = 20260812
NBOOT = 10000
TIE_TOL = 0.0029
CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = [c for c in CELLS if c not in OPEN]
BASE = "always_32b_direct"
DEPLOYABLE = ["always_7b", "always_32b_direct", "always_32b_reasoning",
              "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
R32_CHARGED, R32_DERIVED = 4.57, 3.816

Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
CF = json.load(open(os.path.join(ART, "cost_floor_2026-08-10.json")))
GD = json.load(open(os.path.join(ART, "pmcvqa_grader_defect_2026-08-12.json")))
LB = json.load(open(os.path.join(ART, "pmcvqa_letterbias_audit_2026-08-12.json")))
REP = json.load(open(MAIN))
ARMDEC = CF["arm_decomposition"]["table"]
SH7 = REP["null_tests"]["N2"]["stage_shares"]["lingshu_7b"]
SH32 = REP["null_tests"]["N2"]["stage_shares"]["lingshu_32b"]

OK = {(c, a): Z[f"{c}|{a}"].astype(np.float64) for c in CELLS for a in DEPLOYABLE}
NITEM = {c: len(Z[f"{c}|{BASE}"]) for c in CELLS}

PMC_NAT = GD["A4_combined_stress_test"]["pmc_delta"]["harness_grader_natural_prior"]["delta"]
PMC_STRICT = GD["A4_combined_stress_test"]["pmc_delta"]["repaired_grader_letter_balanced"]
SHIFT = PMC_NAT - PMC_STRICT["delta"]


def cell_pattern_boot(mat, nboot, rng):
    pats, cnt = np.unique(mat, axis=0, return_counts=True)
    n = mat.shape[0]
    return (rng.multinomial(n, cnt / n, size=nboot) @ pats) / n


rng = np.random.default_rng(SEED)
ARM_IDX = {a: i for i, a in enumerate(DEPLOYABLE)}
BASE_I = ARM_IDX[BASE]
BOOT, ACC = {}, {}
for c in CELLS:
    mat = np.stack([OK[(c, a)] for a in DEPLOYABLE], axis=1)
    BOOT[c] = cell_pattern_boot(mat, NBOOT, rng)
    ACC[c] = {a: float(OK[(c, a)].mean()) for a in DEPLOYABLE}
DBOOT = {c: BOOT[c] - BOOT[c][:, [BASE_I]] for c in CELLS}

# ---- NULL TEST: the recomputed PMC veto delta must equal the audit's "natural prior" value -------
pmc_recomputed = ACC["PMC_VQA"]["method_accuracy_max_veto"] - ACC["PMC_VQA"][BASE]
null = dict(name="S1 -- the PMC-VQA certified-veto delta recomputed here equals the audited value",
            recomputed=round(pmc_recomputed, 6),
            audit_natural_prior=PMC_NAT,
            abs_dev=round(abs(pmc_recomputed - PMC_NAT), 6),
            verdict="PASS" if abs(pmc_recomputed - PMC_NAT) < 5e-5 else "FAIL")
print(f"S1 {null['verdict']}  |dev| {null['abs_dev']:.2e}  shift to apply {SHIFT:.5f}")

# ---- apply the shift -----------------------------------------------------------------------------
DBOOT_S = {c: DBOOT[c].copy() for c in CELLS}
DBOOT_S["PMC_VQA"][:, ARM_IDX["method_accuracy_max_veto"]] -= SHIFT
ACC_S = {c: dict(ACC[c]) for c in CELLS}
ACC_S["PMC_VQA"]["method_accuracy_max_veto"] -= SHIFT
MENU = {c: list(DEPLOYABLE) for c in CELLS}
MENU["PMC_VQA"] = [a for a in DEPLOYABLE if a != "method_accuracy_max_fusion"]


def stage_total(c, a, r32):
    x = ARMDEC[c][a]
    return (x["n_gen7"] + x["n_ver7"]) * 1.0 + x["n_32b"] * r32


def cost_of(assign, r32=R32_CHARGED):
    return float(np.mean([stage_total(c, assign[c], r32) for c in CELLS]))


def evaluate(assign, dboot, acc):
    d = np.mean([dboot[c][:, ARM_IDX[assign[c]]] for c in CELLS], axis=0)
    pt = float(np.mean([acc[c][assign[c]] - acc[c][BASE] for c in CELLS]))
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    g, fl = {}, []
    for c in CELLS:
        dc = dboot[c][:, ARM_IDX[assign[c]]]
        l, h = float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))
        g[c] = dict(delta=round(acc[c][assign[c]] - acc[c][BASE], 4), lo=round(l, 4), hi=round(h, 4),
                    worse_sig=bool(h < 0))
        if h < 0:
            fl.append(c)
    return dict(macro_acc=round(float(np.mean([acc[c][assign[c]] for c in CELLS])), 4),
                delta_vs_direct=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4),
                meets_constraint=bool(lo >= -TIE_TOL),
                x_direct_as_charged=round(cost_of(assign) / R32_CHARGED, 4),
                x_direct_derived=round(cost_of(assign, R32_DERIVED) / R32_DERIVED, 4),
                needs_32B=bool(any(ARMDEC[c][assign[c]]["n_32b"] > 0 for c in CELLS)),
                guardrail=g, guardrail_flags=fl, assignment=dict(assign))


def cheapest(dboot, acc, menu):
    """Exhaustive search over the (possibly reduced) menu, cheapest feasible assignment first."""
    idxs = [[ARM_IDX[a] for a in menu[c]] for c in CELLS]
    grid = np.array(np.meshgrid(*idxs, indexing="ij")).reshape(len(CELLS), -1).T
    cost = np.zeros(len(grid))
    mu = np.zeros(len(grid))
    var = np.zeros(len(grid))
    for i, c in enumerate(CELLS):
        cst = np.array([stage_total(c, a, R32_CHARGED) for a in DEPLOYABLE])
        cost += cst[grid[:, i]]
        mu += dboot[c].mean(axis=0)[grid[:, i]]
        var += dboot[c].std(axis=0)[grid[:, i]] ** 2
    cost /= len(CELLS)
    mu /= len(CELLS)
    sd = np.sqrt(var) / len(CELLS)
    cand = np.where(mu - 1.96 * sd >= -TIE_TOL - 0.004)[0]
    cand = cand[np.argsort(cost[cand])]
    for j in cand[:40000]:
        assign = {c: DEPLOYABLE[grid[j, i]] for i, c in enumerate(CELLS)}
        r = evaluate(assign, dboot, acc)
        if r["meets_constraint"]:
            return r
    return None


PRESPEC = {
    "SHIPPED accuracy-max": {c: "method_accuracy_max_veto" for c in CELLS},
    "accuracy-max MCQ + always-32B-direct OPEN": {
        c: ("method_accuracy_max_veto" if c in MCQ else BASE) for c in CELLS},
    "compute-lean MCQ + always-32B-direct OPEN": {
        c: ("method_compute_lean" if c in MCQ else BASE) for c in CELLS},
    "cost_floor cross-fit picks (veto PMC | lean SLAKE-cl,PathVQA-cl | direct elsewhere)": {
        c: ("method_accuracy_max_veto" if c == "PMC_VQA" else
            "method_compute_lean" if c in ("SLAKE_closed", "PATH_VQA_closed") else BASE)
        for c in CELLS},
}

out = dict(
    title="PMC-VQA sensitivity for ATTACK 1 -- how much of the minimum-compute frontier survives the "
          "two confounds that landed on the PMC-VQA cell on 2026-08-12",
    date="2026-08-12",
    kind="CONSTANT-SHIFT SENSITIVITY on a measured delta, NOT a re-measurement",
    sources=["results/cascade_methods/artifacts/pmcvqa_letterbias_audit_2026-08-12.json",
             "results/cascade_methods/artifacts/pmcvqa_grader_defect_2026-08-12.json"],
    reproduce="python3 src/cascade_methods/cost_decomposition_pmc_sensitivity.py",
    null_test=null,
    pmc_cell_delta=dict(harness_grader_natural_prior=PMC_NAT,
                        repaired_grader_letter_balanced=PMC_STRICT,
                        shift_applied=round(SHIFT, 6),
                        gold_letter_prior=LB["A1_gold_letter_prior"] if "A1_gold_letter_prior" in LB
                        else None,
                        grader_defect_differential_pp=GD["A1_size_of_the_defect"]["differential_pp"]),
    menu_change="method_accuracy_max_fusion REMOVED from the PMC menu -- its corrected value is "
                "explicitly NOT MEASURED (pmcvqa_grader_defect_2026-08-12.json:A4.design)",
    prespecified_points_before_and_after={},
    cheapest_feasible_before=None,
    cheapest_feasible_after=None)

for name, a in PRESPEC.items():
    before = evaluate(a, DBOOT, ACC)
    after = evaluate(a, DBOOT_S, ACC_S)
    out["prespecified_points_before_and_after"][name] = dict(
        before=dict(macro_acc=before["macro_acc"], delta=before["delta_vs_direct"],
                    lo=before["lo"], hi=before["hi"], meets=before["meets_constraint"],
                    x_direct=before["x_direct_as_charged"]),
        after=dict(macro_acc=after["macro_acc"], delta=after["delta_vs_direct"],
                   lo=after["lo"], hi=after["hi"], meets=after["meets_constraint"],
                   x_direct=after["x_direct_as_charged"]))
    print(f"{name[:56]:56s} before d={before['delta_vs_direct']:+.4f} lo={before['lo']:+.4f} "
          f"meets={before['meets_constraint']}  ->  after d={after['delta_vs_direct']:+.4f} "
          f"lo={after['lo']:+.4f} meets={after['meets_constraint']}  ({before['x_direct_as_charged']}x)")

out["cheapest_feasible_before"] = cheapest(DBOOT, ACC, {c: list(DEPLOYABLE) for c in CELLS})
out["cheapest_feasible_after"] = cheapest(DBOOT_S, ACC_S, MENU)
out["reading"] = (
    "The Attack-1 frontier buys its non-inferiority slack almost entirely from PMC-VQA.  Under the "
    "strictest honest version of that cell (repaired grader + equal weight per gold letter) the cell "
    f"delta falls {PMC_NAT:+.5f} -> {PMC_STRICT['delta']:+.5f}, i.e. the macro gains "
    f"{SHIFT / 8:+.5f} less slack.  Any frontier point that clears the constraint by less than that "
    "margin should be treated as NOT ESTABLISHED.")
out["eval_visible_warning"] = ("cheapest_feasible_* are EVAL-VISIBLE bounds, computed with full eval "
                               "visibility.  They are upper bounds on what a cross-fit selector could "
                               "reach and must not be quoted as achievable results.")

REP["PMC_sensitivity"] = out
json.dump(REP, open(MAIN, "w"), indent=1, default=float)
print(f"\nappended PMC_sensitivity to {MAIN}")
