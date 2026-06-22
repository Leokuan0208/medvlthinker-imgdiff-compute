#!/usr/bin/env python3
"""
diagnostics.py - WHY confidence-only gating wastes compute. Decomposes the 7B/32B outcome
structure that any cascade gate operates inside, and tests whether the cheap model's uncertainty
even predicts the strong model's recoverability (the assumption every confidence gate implicitly
makes). Pure analysis on existing eval checkpoints; CPU only; launch from repo root.

Definitions per sample (a7,a32 in {0,1}):
  beneficial = 7B wrong & 32B right   -> escalating FIXES it (the only escalation that helps acc)
  futile     = 7B wrong & 32B wrong   -> escalating wastes 32B compute, no acc gain
  harmful    = 7B right & 32B wrong   -> escalating BREAKS a correct answer
  redundant  = 7B right & 32B right   -> escalating wastes compute, acc unchanged
Recoverability r = P(32B right | 7B wrong) = beneficial / (beneficial + futile).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, COMPETENT


def quad(a7, a32):
    ben = ((a7 == 0) & (a32 == 1)).mean()
    fut = ((a7 == 0) & (a32 == 0)).mean()
    har = ((a7 == 1) & (a32 == 0)).mean()
    red = ((a7 == 1) & (a32 == 1)).mean()
    rec = ben / (ben + fut) if (ben + fut) > 0 else float("nan")     # recoverability of 7B errors
    flip = har / (har + red) if (har + red) > 0 else float("nan")    # 32B breaks a correct 7B answer
    return dict(beneficial=ben, futile=fut, harmful=har, redundant=red, recoverability=rec, flip=flip)


def main():
    D = CascadeData("cap320")
    for sub_name, names in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
        P = D.pool(names); a7, a32 = P["a7"], P["a32"]
        q = quad(a7, a32)
        print(f"\n################  OUTCOME STRUCTURE  [{sub_name}]  n={len(a7)}  ################")
        print(f"  7B acc={a7.mean():.4f}   32B acc={a32.mean():.4f}")
        print(f"  beneficial(7w,32r)={q['beneficial']*100:5.1f}%   futile(7w,32w)={q['futile']*100:5.1f}%"
              f"   harmful(7r,32w)={q['harmful']*100:5.1f}%   redundant(7r,32r)={q['redundant']*100:5.1f}%")
        print(f"  recoverability P(32B right | 7B wrong) = {q['recoverability']*100:.1f}%"
              f"   (=> {100-q['recoverability']*100:.1f}% of 7B errors are FUTILE to escalate)")
        print(f"  flip rate     P(32B wrong | 7B right)  = {q['flip']*100:.1f}%  (escalation can BREAK these)")

        # what the margin gate's escalations actually buy: stratify escalated set by outcome
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        cm = D.calib["sig"]["margin"].reshape(-1, 1); cy = D.calib["ok"]
        g = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(cm, cy)
        tau = float(np.quantile(g.predict_proba(cm)[:, 1], 1.0 - cy.mean()))
        esc = g.predict_proba(P["sig"]["margin"].reshape(-1, 1))[:, 1] < tau
        e7, e32 = a7[esc], a32[esc]
        eq = quad(e7, e32)
        print(f"  -- margin gate escalates {esc.mean()*100:.1f}% of samples; OF THOSE ESCALATIONS:")
        print(f"     beneficial={eq['beneficial']*100:4.1f}%  futile={eq['futile']*100:4.1f}%"
              f"  harmful={eq['harmful']*100:4.1f}%  redundant={eq['redundant']*100:4.1f}%"
              f"   (only beneficial helps; harmful hurts)")

        # does 7B uncertainty predict 32B recoverability? stratify 7B-WRONG samples by margin decile
        wrong = a7 == 0
        mg = P["sig"]["margin"][wrong]; a32w = a32[wrong]
        order = np.argsort(mg)                 # low margin = most uncertain
        a32w_sorted = a32w[order]
        nb = 5
        print(f"  -- recoverability of 7B ERRORS by 7B-margin quintile (low margin = most uncertain):")
        for b in range(nb):
            lo, hi = b * len(a32w_sorted) // nb, (b + 1) * len(a32w_sorted) // nb
            seg = a32w_sorted[lo:hi]
            print(f"       Q{b+1} ({'most-unc' if b==0 else 'least-unc' if b==nb-1 else '       '}): "
                  f"recoverability={seg.mean()*100:5.1f}%  (n={len(seg)})")

    print("\nREAD: if recoverability is FLAT or DECREASING across margin quintiles, then cheap-model")
    print("uncertainty does NOT tell you whether escalation will help -> a confidence-only gate must")
    print("escalate the whole uncertain mass (incl. the futile part). A recoverability-aware gate that")
    print("separates 'uncertain & recoverable' from 'uncertain & futile' is the lever.")


if __name__ == "__main__":
    main()
