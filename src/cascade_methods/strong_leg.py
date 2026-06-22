#!/usr/bin/env python3
"""
strong_leg.py - compare cheaper escalation TARGETS (32B no-think / reduced resolution) against the
deployed 32B-think@fullres. The strong leg dominates cascade cost, so a cheaper 32B that retains
accuracy is the largest compute lever. For each candidate strong-leg config we report:
  (1) the config's own pooled/per-benchmark accuracy vs 32B-think@fullres (does accuracy survive?);
  (2) the margin-gate cascade re-scored with that strong leg: backbone% and accuracy vs the FIXED
      think-32B parity target (0.572 ALL-6 / 0.645 COMPETENT-4) -- i.e. can a cheaper strong leg
      still reach think-parity at lower cost?
CPU only; launch from repo root. Skips configs whose checkpoints don't exist yet.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, COMPETENT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CONFIGS = [
    ("think@fullres (deployed)", "ckpts/gate_32b", "think_norag", "fullres"),
    ("nothink@fullres", "ckpts/gate_32b_modes/nothink_fullres", "nothink_norag", "fullres"),
    ("think@cap320", "ckpts/gate_32b_modes/think_cap320", "think_norag", "cap320"),
    ("nothink@cap320", "ckpts/gate_32b_modes/nothink_cap320", "nothink_norag", "cap320"),
]


def margin_gate(D):
    cm = D.calib["sig"]["margin"].reshape(-1, 1); cy = D.calib["ok"]
    g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(cm, cy)
    tau = float(np.quantile(g.predict_proba(cm)[:, 1], 1.0 - cy.mean()))
    return g, tau


def exists(d, cell):
    import glob
    return bool(glob.glob(os.path.join(REPO, d, f"*{cell}*.jsonl")))


def main():
    # reference think-32B parity targets (fixed)
    Dref = CascadeData("cap320")
    parity = {"ALL-6": float(Dref.pool(ALL6)["a32"].mean()),
              "COMPETENT-4": float(Dref.pool(COMPETENT)["a32"].mean())}
    g, tau = margin_gate(Dref)
    print(f"think-32B parity targets: ALL-6={parity['ALL-6']:.4f}  COMPETENT-4={parity['COMPETENT-4']:.4f}")
    print(f"margin gate tau={tau:.3f}\n")

    for label, sdir, scell, scap in CONFIGS:
        if not exists(sdir, scell):
            print(f"[skip] {label}: checkpoints not present yet ({sdir})"); continue
        D = CascadeData("cap320", strong_dir=sdir, strong_cell=scell, strong_cap=scap)
        print(f"================  STRONG LEG = {label}  ================")
        for sub, names in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
            P = D.pool(names)
            # this strong leg's own accuracy and mean decode
            print(f"  [{sub}] strong-leg acc={P['a32'].mean():.4f} (think-32B={parity[sub]:.4f})"
                  f"  meanG32={P['g32'].mean():.0f} tok  meanPF={P['PF'].mean():.0f} tok")
            # margin-gate cascade re-scored with this strong leg
            esc = g.predict_proba(P["sig"]["margin"].reshape(-1, 1))[:, 1] < tau
            m = D.score(P, esc)
            # accuracy here uses this strong leg for escalated; compare to think parity
            hit = "Y" if m["acc"] >= parity[sub] - 1e-9 else "n"
            print(f"        margin-gate cascade: esc={m['esc']*100:.0f}%  acc={m['acc']:.4f} (parity {hit})"
                  f"  backbone={m['backbone']*100:.1f}%  (deployed think@fullres=73.6%/69.5%)")
        print()
    print("READ: a cheaper strong leg WINS if its margin-gate cascade still hits think-parity (Y) at a")
    print("lower backbone%. If acc drops below parity, the think strong leg is necessary for that config.")


if __name__ == "__main__":
    main()
