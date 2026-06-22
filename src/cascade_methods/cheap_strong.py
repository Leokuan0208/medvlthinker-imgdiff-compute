#!/usr/bin/env python3
"""
cheap_strong.py - re-score the cascade with a CHEAPER STRONG LEG (32B no-think), with the cost
denominator FIXED at the deployed baseline = always-32B-think@fullres (what we save compute against).

Finding that motivates this: on the 4 competent benchmarks the 32B in NO-THINK mode is as accurate
or better than in think mode (thinking overthinks perception VQA), at ~2 decode tokens vs ~477.

Cost model (prefill-inclusive, fixed denominator):
  baseline (denominator) = always-32B-THINK = 2*N32*(PF_full + G32_think)   per question
  cascade(q)  = 2*N7*(PY_cap320 + G7)  +  [escalated] 2*N32*(PF_strong + G32_strong)
  backbone%   = cascade_cost / baseline_cost      (directly comparable to the deployed 73.6%/69.5%)
CPU only; launch from repo root. Needs ckpts/gate_32b_modes/nothink_fullres/.
"""
import os, sys, glob, json, re
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (CascadeData, ALL6, COMPETENT, _load_arm, EVAL_DIR, DIR_32B, CACHE, N7, N32)
from frontier import curve_for_score, min_backbone_at_acc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)


def build(strong_dir, strong_cell, strong_cap, cheap_cap="cap320"):
    """Aligned arrays per dataset with a FIXED think-32B denominator and a candidate strong leg."""
    cache = json.load(open(J(CACHE)))
    ev = _load_arm(J(EVAL_DIR[cheap_cap]), "nothink_norag")
    think = _load_arm(J(DIR_32B), "think_norag")               # baseline denominator + think strong leg
    strong = _load_arm(J(strong_dir), strong_cell)             # candidate strong leg
    out = {}
    for ds in ALL6:
        if ds not in ev or ds not in think or ds not in strong:
            continue
        cY = cache[ds][cheap_cap]; cstrong = cache[ds][strong_cap]; cfull = cache[ds]["fullres"]
        idx = sorted(set(ev[ds]) & set(think[ds]) & set(strong[ds])
                     & {int(k) for k in cY} & {int(k) for k in cstrong} & {int(k) for k in cfull})
        if not idx:
            continue
        sig = [ev[ds][i].get("opt_logprobs") or {} for i in idx]
        def margin(d):
            v = sorted((d).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
        out[ds] = dict(
            a7=np.array([ev[ds][i]["ok"] for i in idx], float),
            aS=np.array([strong[ds][i]["ok"] for i in idx], float),     # strong-leg correctness
            aT=np.array([think[ds][i]["ok"] for i in idx], float),      # think correctness (baseline)
            margin=np.array([margin(d) for d in sig], float),
            run7=2*N7*np.array([cY[str(i)][0] + (ev[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runS=2*N32*np.array([cstrong[str(i)][0] + (strong[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runT=2*N32*np.array([cfull[str(i)][0] + (think[ds][i].get("gen_tokens") or 0) for i in idx], float),
        )
    return out


def pool(D, names):
    names = [d for d in names if d in D]
    return {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}


def report(label, strong_dir, strong_cell, strong_cap):
    if not glob.glob(os.path.join(REPO, strong_dir, f"*{strong_cell}*.jsonl")):
        print(f"[skip] {label}: not present"); return
    D = build(strong_dir, strong_cell, strong_cap)
    # margin gate calibrated on PMC-train (same as deployed)
    Dc = CascadeData("cap320")
    cm = Dc.calib["sig"]["margin"].reshape(-1, 1); cy = Dc.calib["ok"]
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(cm, cy)
    tau = float(np.quantile(g.predict_proba(cm)[:, 1], 1.0 - cy.mean()))
    print(f"\n================  STRONG LEG = {label}   (denominator = always-32B-think)  ================")
    for sub, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = pool(D, names)
        base = P["runT"].sum()                              # always-32B-think cost (denominator)
        think_acc = P["aT"].mean(); strong_acc = P["aS"].mean(); a7 = P["a7"].mean()
        # always-strong-leg baseline (no cascade)
        bb_allS = P["runS"].sum() / base
        # margin-gate cascade -> escalate to STRONG leg
        esc = g.predict_proba(P["margin"].reshape(-1, 1))[:, 1] < tau
        casc_acc = np.where(esc, P["aS"], P["a7"]).mean()
        casc_bb = (P["run7"].sum() + P["runS"][esc].sum()) / base
        # frontier: min backbone (vs think baseline) at think-parity, sweeping margin threshold
        order = np.argsort(P["margin"]); a7o, aSo, rSo = P["a7"][order], P["aS"][order], P["runS"][order]
        accs = a7 + np.concatenate([[0], np.cumsum(aSo - a7o)]) / len(a7o)
        costs = (P["run7"].sum() + np.concatenate([[0], np.cumsum(rSo)])) / base
        hit = np.where(accs >= think_acc - 1e-9)[0]
        fr = (costs[hit.min()], accs[hit.min()], hit.min()/len(a7o)) if len(hit) else None
        print(f"\n  [{sub}] always-7B={a7:.4f}  always-32B-think={think_acc:.4f}  "
              f"always-32B-[{label.split('@')[0]}]={strong_acc:.4f} @ backbone={bb_allS*100:.1f}%")
        print(f"        margin-gate cascade -> strong: esc={esc.mean()*100:.0f}%  acc={casc_acc:.4f}  "
              f"backbone={casc_bb*100:.1f}%  (deployed think cascade=69.5%/73.6%)")
        if fr:
            print(f"        frontier @ think-parity ({think_acc:.4f}): backbone={fr[0]*100:.1f}%  esc={fr[2]*100:.0f}%")


def main():
    report("think@fullres", "ckpts/gate_32b", "think_norag", "fullres")
    report("nothink@fullres", "ckpts/gate_32b_modes/nothink_fullres", "nothink_norag", "fullres")
    report("think@cap320", "ckpts/gate_32b_modes/think_cap320", "think_norag", "cap320")
    report("nothink@cap320", "ckpts/gate_32b_modes/nothink_cap320", "nothink_norag", "cap320")


if __name__ == "__main__":
    main()
