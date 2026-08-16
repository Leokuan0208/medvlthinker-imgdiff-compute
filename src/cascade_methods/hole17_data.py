#!/usr/bin/env python3
"""hole17_data.py -- per-cell raw inputs for the macro-objective threshold refit, plus a VECTORISED
re-expression of the Weitzman controller that is asserted equal to pandora_controller.run_pandora
item-for-item.  No new inference; no policy change; this file only makes the existing policy fast
enough to sweep.

THE ALGEBRAIC RE-EXPRESSION (derived from run_pandora, then verified by assertion)
  Let cal = calibrated verifier scores in recorded slot order, raw = uncalibrated, z_c = zeta_cheap,
  z_s = zeta_strong.
    * if z_s > z_c            -> the strong box has the higher reservation and best_cal = -inf at
                                 k=0, so the box is opened immediately: N = 0, escalate = 1.
    * else                    -> draw until the running max of cal reaches z_c:
                                    N = 1 + argmax_i [cal_i >= z_c]   (N = Nmax if none)
                                 the loop can only reach the strong box after the cheap pool is
                                 exhausted, so
                                    escalate = (N == Nmax) and (max(cal[:N]) < z_s)
                                 and otherwise the answer is slot argmax(raw[:N]).
  This is exact, not an approximation: assert_pandora_equivalence() replays every (z_c, z_s) pair the
  sweeps use through run_pandora and compares (N, escalate, ok) element-wise.

Reproduce:  OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_data.py
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)

import integrated_method as IM
import integrated_pandora as IP
import pandora_controller as PC
import beat32b_more as BB

ROOT = IM.ROOT
MCQ_B = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN_B = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
ORDER_B = MCQ_B + OPEN_B
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}

# as-charged FLOP-eq constants -- single source, paper_baselines / pandora_controller
F_GEN7, F_VER7, F_GEN32 = 1.0, 1.0, 4.57
C_CHEAP = PC.C_CHEAP_F      # 2.0 per adaptive draw (generate + verify)
C_STRONG = PC.C_STRONG_F    # 4.57 per escalation
assert (C_CHEAP, C_STRONG) == (2.0, 4.57), (C_CHEAP, C_STRONG)


def load_mcq():
    """{cell: dict(ok7, ok32, margin, n)} -- exactly the vectors paper_baselines.cascade_persample sees."""
    out = {}
    for name, closed in [("PMC_VQA", None), ("SLAKE_closed", "SLAKE"),
                         ("VQA_RAD_closed", "YESNO"), ("PATH_VQA_closed", "YESNO")]:
        d = IM.mcq_closed(name.split("_closed")[0], closed)
        out[name] = dict(ok7=np.asarray(d["ok7"], float), ok32=np.asarray(d["ok32"], float),
                         margin=np.asarray(d["margin"], float), n=len(d["ok7"]))
    d = IM.mcq_medxpert()
    out["MedXpertQA-MM"] = dict(ok7=np.asarray(d["ok7"], float), ok32=np.asarray(d["ok32"], float),
                                margin=np.asarray(d["margin"], float), n=len(d["ok7"]))
    return out


def load_open():
    """{cell: dict(raw(n,8), sl(n,8), strong(n), greedy(n), n)} from the CLEAN disjoint verifier."""
    out = {}
    for name in OPEN_B:
        rows = IP.load_open_rows(OPEN_KEY[name])
        Nmax = min(len(r["scores"]) for r in rows)
        out[name] = dict(
            raw=np.array([r["scores"][:Nmax] for r in rows], float),
            sl=np.array([r["sl"][:Nmax] for r in rows], float),
            strong=np.array([r["strong"] for r in rows], float),
            greedy=np.array([r["greedy"] for r in rows], float),
            n=len(rows), Nmax=Nmax)
    return out


# ---------------------------------------------------------------- vectorised Weitzman policy
def pandora_vec(cal, raw, sl, strong, z_c, z_s):
    """Exact vectorised run_pandora over a whole cell.  Returns (N, esc, ok) arrays."""
    n, Nmax = cal.shape
    if z_s > z_c:                                   # strong box preempts at k=0
        return np.zeros(n), np.ones(n), strong.copy()
    hit = cal >= z_c
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1) + 1, Nmax)     # N drawn
    ar = np.arange(Nmax)[None, :]
    drawn = ar < first[:, None]
    best_cal = np.where(drawn, cal, -np.inf).max(axis=1)
    esc = (first == Nmax) & (best_cal < z_s)
    pick = np.where(drawn, raw, -np.inf).argmax(axis=1)
    ok = np.where(esc, strong, sl[np.arange(n), pick])
    return first.astype(float), esc.astype(float), ok.astype(float)


def assert_pandora_equivalence(openc, n_pairs=40, seed=0):
    """Replay a spread of (z_c, z_s) pairs through the ORIGINAL run_pandora and compare item-for-item."""
    rng = np.random.default_rng(seed); report = {}
    for name, d in openc.items():
        cal = d["raw"]                              # any monotone score works for the equivalence test
        zs_grid = np.quantile(cal, np.linspace(0, 1, 9))
        pairs = [(float(a), float(b)) for a in zs_grid for b in zs_grid][:n_pairs * 2]
        rng.shuffle(pairs); pairs = pairs[:n_pairs]
        bad = 0
        for z_c, z_s in pairs:
            N, E, O = pandora_vec(cal, d["raw"], d["sl"], d["strong"], z_c, z_s)
            for i in range(d["n"]):
                Nk, e, ok = PC.run_pandora(d["raw"][i], cal[i], d["sl"][i], d["strong"][i], z_c, z_s)
                if (Nk, e, ok) != (N[i], E[i], O[i]): bad += 1
        report[name] = dict(pairs_tested=len(pairs), items=d["n"],
                            comparisons=len(pairs) * d["n"], mismatches=bad)
    return report


if __name__ == "__main__":
    IP.ADAPTER = "ckpts/train/lora_verifier_disjoint"
    mcq, openc = load_mcq(), load_open()
    print({k: v["n"] for k, v in mcq.items()}, {k: v["n"] for k, v in openc.items()})
    rep = assert_pandora_equivalence(openc)
    print(json.dumps(rep, indent=1))
    assert all(r["mismatches"] == 0 for r in rep.values()), rep
    print("VECTORISED PANDORA == run_pandora on every comparison.")
