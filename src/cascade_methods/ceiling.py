#!/usr/bin/env python3
"""
ceiling.py - IN-DISTRIBUTION ceiling of recoverability-aware gating, estimated by K-fold CV on the
EVAL set using 32B-on-eval labels (which we already have). This is NOT a deployable result (it peeks
at eval outcomes to train the predictor); it answers a prior question:

    Is there ANY routable escalation-benefit signal in the 7B's cheap-leg features?

We out-of-fold predict, per sample, the deferral value Delta(x)=P(32B right|x)-P(7B right|x) and the
benefit z=1[7B wrong & 32B right], using only 7B features, then sweep the threshold on eval and read
min backbone% at parity. Compare to the best single confidence signal's eval-oracle and to the
outcome-oracle floor. If the CV ceiling >> confidence frontier, the 7B features carry benefit signal
and the only open question is PMC-train -> eval transfer (answered by the real calibrated run). If
not, the direction is capped and we need a new signal (GPU). CPU only; launch from repo root.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, COMPETENT
from frontier import curve_for_score, min_backbone_at_acc, oracle_parity
from methods_deferral import FEATS, _X, _clf
from sklearn.model_selection import StratifiedKFold


def oof_scores(pool, kind="gbm", nfold=5, seed=0):
    """Out-of-fold P(7B right), P(32B right), P(benefit) using only 7B features."""
    X = _X(pool["sig"], FEATS); a7 = pool["a7"]; a32 = pool["a32"]
    z = ((a7 == 0) & (a32 == 1)).astype(int)
    p7 = np.zeros(len(a7)); p32 = np.zeros(len(a7)); pz = np.zeros(len(a7))
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
    strat = a7.astype(int) * 2 + a32.astype(int)          # stratify on the 4-way outcome
    for tr, te in skf.split(X, strat):
        p7[te] = _clf(kind).fit(X[tr], a7[tr]).predict_proba(X[te])[:, 1]
        p32[te] = _clf(kind).fit(X[tr], a32[tr]).predict_proba(X[te])[:, 1]
        zclf = _clf(kind).fit(X[tr], z[tr])
        pz[te] = zclf.predict_proba(X[te])[:, 1]
    return p7, p32, pz


def main():
    D = CascadeData("cap320")
    for sub_name, names in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
        P = D.pool(names); parity = float(P["a32"].mean())
        print(f"\n################  IN-DIST CEILING (CV on eval)  [{sub_name}]  n={len(P['a7'])}  "
              f"parity={parity:.4f}  ################")
        orp = oracle_parity(P, parity)
        print(f"  outcome-oracle floor @ parity:    backbone={orp['backbone']*100:5.1f}%  esc={orp['esc']*100:.1f}%")
        # best confidence signal eval-oracle (margin & entropy & prob_margin)
        for s, sign, nm in [("margin", -1, "margin"), ("prob_margin", -1, "prob_margin"),
                            ("entropy", +1, "entropy")]:
            cur = curve_for_score(P, sign * P["sig"][s]); mb = min_backbone_at_acc(cur, parity)
            print(f"  confidence eval-oracle [{nm:11s}]:  backbone={mb['backbone']*100:5.1f}%  esc={mb['esc']*100:.1f}%"
                  if mb else f"  confidence [{nm}] never reaches parity")
        for kind in ["logistic", "gbm"]:
            p7, p32, pz = oof_scores(P, kind=kind)
            for score, nm in [(p32 - p7, f"Delta(recoverability)-{kind}"), (pz, f"benefit-z-{kind}")]:
                cur = curve_for_score(P, score); mb = min_backbone_at_acc(cur, parity)
                tag = f"backbone={mb['backbone']*100:5.1f}%  esc={mb['esc']*100:.1f}%" if mb else "never reaches parity"
                print(f"  CV-ceiling [{nm:26s}]: {tag}")
    print("\nREAD: if the CV-ceiling backbone is far below the confidence eval-oracle, the 7B features")
    print("DO carry escalation-benefit signal (in-distribution) -> recoverability-aware gating can win,")
    print("and the remaining question is PMC-train -> eval transfer (the real calibrated run).")


if __name__ == "__main__":
    main()
