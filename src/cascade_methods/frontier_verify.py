#!/usr/bin/env python3
"""ADVERSARIAL VERIFICATION for the min-compute frontier round (2026-08-12).

Independent re-implementation. Loads ONLY the stored per-item eval vectors
(_selector_rerun_parts/vec_disjoint.npz) and the published per-cell cost
constants, and re-derives:
  N1  published macro baselines + shipped arm deltas (NULL TEST)
  N2  per-cell macro costs of the shipped arms (NULL TEST)
  V1  the headline of ATTACK 1 (0.8647x pre-specified per-cell arm swap)
  V2  the headline of ATTACK 2 (pre-generation router / hybrid 0.8331x)
  V3  the headline of ATTACK 4 (min-escalation ~1.0074x)
  D   DEGENERACY CHECK for every cheap frontier point
No GPU, no new inference, MedEvalKit untouched.
"""
import os
# pins MUST precede numpy import (defect found in pregen_router.py this round)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
import json, sys
import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
ART = ROOT + "/results/cascade_methods/artifacts"
CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed",
         "MedXpertQA-MM", "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
SEED = 20260812
NBOOT = 10000
EPS = -0.0029   # pre-registered non-inferiority margin


def load_vecs(tag="disjoint"):
    d = np.load(f"{ART}/_selector_rerun_parts/vec_{tag}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        cell, arm = k.split("|")
        out.setdefault(cell, {})[arm] = d[k].astype(np.float64)
    return out


def macro(vecs, pick):
    """pick: cell -> 1D 0/1 vector. macro = mean over cells of cell mean."""
    return float(np.mean([pick[c].mean() for c in CELLS]))


class Boot:
    """One shared resample stream per cell, reused by every policy."""
    def __init__(self, vecs, seed=SEED, nboot=NBOOT):
        rng = np.random.default_rng(seed)
        self.idx = {c: rng.integers(0, len(vecs[c]["always_7b"]),
                                    size=(nboot, len(vecs[c]["always_7b"])))
                    for c in CELLS}
        self.nboot = nboot

    def cellmeans(self, cell, v):
        return v[self.idx[cell]].mean(axis=1)

    def delta_ci(self, a, b):
        """a,b: cell -> vector. Returns (delta, lo, hi) on the 8-cell macro."""
        da = np.zeros(self.nboot)
        for c in CELLS:
            da += (self.cellmeans(c, a[c]) - self.cellmeans(c, b[c])) / len(CELLS)
        pt = macro(None, a) - macro(None, b)
        lo, hi = np.percentile(da, [2.5, 97.5])
        return float(pt), float(lo), float(hi)

    def cell_delta_ci(self, cell, a, b):
        d = self.cellmeans(cell, a) - self.cellmeans(cell, b)
        lo, hi = np.percentile(d, [2.5, 97.5])
        return float(a.mean() - b.mean()), float(lo), float(hi)


def main():
    res = {}
    vecs = load_vecs("disjoint")
    ns = {c: len(vecs[c]["always_7b"]) for c in CELLS}
    res["n_per_cell"] = ns
    res["n_total"] = int(sum(ns.values()))

    # ---------- N1: null test on published macro accuracies ----------
    pub = json.load(open(f"{ART}/_selector_rerun_parts/summary_disjoint.json"))
    arms = ["always_7b", "always_32b_direct", "always_32b_reasoning",
            "oracle_mode_32b", "method_compute_lean",
            "method_accuracy_max_veto", "method_accuracy_max_fusion"]
    n1 = {}
    maxdev = 0.0
    for a in arms:
        mine = float(np.mean([vecs[c][a].mean() for c in CELLS]))
        theirs = pub["macro_acc"][a]
        dev = abs(round(mine, 4) - theirs)
        maxdev = max(maxdev, dev)
        n1[a] = {"mine": round(mine, 6), "published": theirs, "abs_dev": round(dev, 8)}
    res["N1_macro_baselines"] = {"per_arm": n1, "max_abs_dev": maxdev,
                                 "PASS": maxdev < 1e-4}

    # per-cell null test
    pc_dev = 0.0
    for c in CELLS:
        for a in arms:
            pc_dev = max(pc_dev, abs(round(float(vecs[c][a].mean()), 4)
                                     - pub["per_cell_acc"][c][a]))
    res["N1b_per_cell"] = {"max_abs_dev": pc_dev, "PASS": pc_dev < 1e-4}

    # ---------- N1c: reproduce the published CIs with my own bootstrap ----------
    B = Boot(vecs)
    n1c = {}
    for m in ["method_compute_lean", "method_accuracy_max_veto",
              "method_accuracy_max_fusion"]:
        d, lo, hi = B.delta_ci(
            {c: vecs[c][m] for c in CELLS},
            {c: vecs[c]["always_32b_direct"] for c in CELLS})
        p = pub["deltas"][m]["always_32b_direct"]
        n1c[m] = {"mine": [round(d, 4), round(lo, 4), round(hi, 4)],
                  "published": [p["delta"], p["lo"], p["hi"]],
                  "ci_lo_dev": round(abs(round(lo, 4) - p["lo"]), 6)}
    res["N1c_published_CIs"] = n1c

    json.dump(res, open(f"{ART}/_frontier_verify_parts/n1.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    os.makedirs(f"{ART}/_frontier_verify_parts", exist_ok=True)
    main()
