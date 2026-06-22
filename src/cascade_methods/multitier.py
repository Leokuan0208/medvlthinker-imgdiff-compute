#!/usr/bin/env python3
"""
multitier.py - MODE-ADAPTIVE multi-tier cascade. Data shows 32B no-think >= 32B think on the
competent perception benchmarks (thinking overthinks) but think wins on the reasoning benchmarks
(MMMU/MedXpert). So allocate think compute only where needed:

  Tier 1: 7B no-think @ cap320         (cheapest)         keep if 7B confident
  Tier 2: 32B no-think @ fullres       (cheap: ~2 tok)    keep if 32B-no-think confident
  Tier 3: 32B think @ fullres          (~477 tok)         only the residual reasoning-hard cases

Honest cost: each tier is a separate forward pass (the think vs no-think SYSTEM PROMPT differs, so
no KV reuse); a question pays every tier it reaches. Denominator = always-32B-think@fullres.
We compare: 2-tier-think (deployed), 2-tier-nothink, and the 3-tier. Thresholds for the eval frontier
are swept on eval (CEILING); the 7B-gate operating point is also reported honestly (PMC-train calib).
The tier-2 (32B-no-think) gate needs 32B-no-think-on-calib for an honest threshold (run pending).
CPU only; launch from repo root.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, EVAL_DIR, DIR_32B, CACHE, ALL6, COMPETENT, N7, N32, CascadeData
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
NOTHINK_DIR = "ckpts/gate_32b_modes/nothink_fullres"


def _margin(d):
    v = sorted((d or {}).values(), reverse=True); return (v[0] - v[1]) if len(v) >= 2 else 0.0


def build():
    cache = json.load(open(J(CACHE)))
    ev = _load_arm(J(EVAL_DIR["cap320"]), "nothink_norag")
    think = _load_arm(J(DIR_32B), "think_norag")
    noth = _load_arm(J(NOTHINK_DIR), "nothink_norag")
    D = {}
    for ds in ALL6:
        if ds not in ev or ds not in think or ds not in noth:
            continue
        cY = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(ev[ds]) & set(think[ds]) & set(noth[ds])
                     & {int(k) for k in cY} & {int(k) for k in cF})
        D[ds] = dict(
            a7=np.array([ev[ds][i]["ok"] for i in idx], float),
            aN=np.array([noth[ds][i]["ok"] for i in idx], float),
            aT=np.array([think[ds][i]["ok"] for i in idx], float),
            m7=np.array([_margin(ev[ds][i].get("opt_logprobs")) for i in idx], float),
            mN=np.array([_margin(noth[ds][i].get("opt_logprobs")) for i in idx], float),
            run7=2*N7*np.array([cY[str(i)][0] + (ev[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runN=2*N32*np.array([cF[str(i)][0] + (noth[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runT=2*N32*np.array([cF[str(i)][0] + (think[ds][i].get("gen_tokens") or 0) for i in idx], float),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    return {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}


def three_tier(P, tau1, tau2):
    """tau1 on 7B margin, tau2 on 32B-nothink margin. Returns acc, backbone (vs think baseline)."""
    base = P["runT"].sum()
    esc1 = P["m7"] < tau1                        # leave tier 1
    esc2 = esc1 & (P["mN"] < tau2)               # leave tier 2 (only those past tier 1)
    final = np.where(~esc1, P["a7"], np.where(~esc2, P["aN"], P["aT"]))
    cost = P["run7"].sum() + P["runN"][esc1].sum() + P["runT"][esc2].sum()
    return float(final.mean()), float(cost / base), float(esc1.mean()), float(esc2.mean())


def frontier_2d(P, target, n1=60, n2=40):
    """min backbone at acc>=target over a (tau1,tau2) grid (eval ceiling)."""
    q7 = np.quantile(P["m7"], np.linspace(0, 1, n1))
    qN = np.quantile(P["mN"], np.linspace(0, 1, n2))
    best = None
    for t1 in q7:
        for t2 in qN:
            acc, bb, e1, e2 = three_tier(P, t1, t2)
            if acc >= target - 1e-9 and (best is None or bb < best[1]):
                best = (acc, bb, e1, e2, t1, t2)
    return best


def main():
    D = build()
    # 7B margin gate honest threshold (PMC-train)
    Dc = CascadeData("cap320")
    cm = Dc.calib["sig"]["margin"].reshape(-1, 1); cy = Dc.calib["ok"]
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(cm, cy)
    # express the gate's decision as a 7B-margin threshold (monotone): escalate lowest-margin err-rate frac
    tau1_honest = float(np.quantile(Dc.calib["sig"]["margin"], 1.0 - cy.mean()))

    for sub, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = pool(D, names); base = P["runT"].sum()
        tgt = P["aT"].mean()
        print(f"\n################  {sub}  n={len(P['a7'])}  think-parity={tgt:.4f}  ################")
        print(f"  always-7B={P['a7'].mean():.4f}  always-32B-nothink={P['aN'].mean():.4f}"
              f"  always-32B-think={tgt:.4f}")
        # 2-tier think (deployed strong leg), honest 7B gate
        e1 = P["m7"] < tau1_honest
        acc_t = np.where(e1, P["aT"], P["a7"]).mean(); bb_t = (P["run7"].sum() + P["runT"][e1].sum())/base
        # 2-tier nothink, honest 7B gate
        acc_n = np.where(e1, P["aN"], P["a7"]).mean(); bb_n = (P["run7"].sum() + P["runN"][e1].sum())/base
        print(f"  [honest 7B-gate, esc={e1.mean()*100:.0f}%]  2-tier->think: acc={acc_t:.4f} bb={bb_t*100:.1f}%"
              f"   |   2-tier->nothink: acc={acc_n:.4f} bb={bb_n*100:.1f}%")
        # frontiers (eval ceiling) at think-parity
        # 2-tier think frontier
        for nm, aS, rS in [("2tier-think", P["aT"], P["runT"]), ("2tier-nothink", P["aN"], P["runN"])]:
            o = np.argsort(P["m7"]); a7o, aSo, rSo = P["a7"][o], aS[o], rS[o]
            accs = P["a7"].mean() + np.concatenate([[0], np.cumsum(aSo-a7o)])/len(a7o)
            costs = (P["run7"].sum() + np.concatenate([[0], np.cumsum(rSo)]))/base
            hit = np.where(accs >= tgt-1e-9)[0]
            if len(hit):
                print(f"  frontier {nm:14s} @parity: bb={costs[hit.min()]*100:5.1f}%  esc={hit.min()/len(a7o)*100:.0f}%")
            else:
                print(f"  frontier {nm:14s} @parity: NEVER (max acc={accs.max():.4f})")
        # 3-tier frontier (eval ceiling)
        b = frontier_2d(P, tgt)
        if b:
            print(f"  frontier 3-tier        @parity: bb={b[1]*100:5.1f}%  esc1={b[2]*100:.0f}% esc2(of all)={b[3]*100:.0f}%"
                  f"  (tau1={b[4]:.2f},tau2={b[5]:.2f})")
        else:
            print(f"  frontier 3-tier        @parity: NEVER")
    print("\nREAD: 2-tier-nothink should dominate on COMPETENT-4; 3-tier should help ALL-6 by using think")
    print("only on the reasoning residual. Honest tier-2 gate needs 32B-nothink-on-calib (pending).")


if __name__ == "__main__":
    main()
