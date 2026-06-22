#!/usr/bin/env python3
"""
escalation_leaderboard.py - PRIMARY METRIC = minimum 32B ESCALATION RATE at iso-accuracy
(cascade acc >= always-32B-think), pooled over ALL-6 and ALL-5 (excl. MedXpert). Strong leg is the
STANDARD 32B-think@fullres, so this isolates the CASCADE DECISION RULE (the gate), independent of
any strong-leg config trick. Lower escalation = better cascade. Reports, per gate:
  EVAL-ORACLE : smallest escalation (threshold swept on eval) reaching parity  [gate-quality ceiling]
  HONEST      : threshold from PMC-train (escalate calib error-rate fraction); escalation, acc, hit?
Also: random baseline and the outcome-oracle floor. CPU only; launch from repo root.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, ALL5, COMPETENT
from frontier import curve_for_score, min_backbone_at_acc, oracle_parity
from methods import registry as conf_registry
try:
    from methods_deferral import deferral_registry
except Exception:
    deferral_registry = lambda: {}

RNG = np.random.default_rng(0)


def min_esc_at_parity(P, score, target):
    """smallest escalation (in score order) reaching acc>=target. esc & backbone minimized together."""
    cur = curve_for_score(P, score)
    return min_backbone_at_acc(cur, target)


def main():
    D = CascadeData("cap320")
    reg = {**{f"conf:{k.split(':')[-1]}": m for k, m in conf_registry().items()}}
    if D.has_strong:
        reg.update(deferral_registry())
    for sub, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5)]:
        P = D.pool(names); parity = float(P["a32"].mean()); a7 = float(P["a7"].mean())
        print(f"\n################  ESCALATION @ iso-accuracy  [{sub}]  n={len(P['a7'])}  ################")
        print(f"  always-7B={a7:.4f}  always-32B-think(parity target)={parity:.4f}  gap={parity-a7:.4f}")
        orp = oracle_parity(P, parity)
        rb = min_esc_at_parity(P, RNG.standard_normal(len(P["a7"])), parity)
        print(f"  outcome-oracle: esc={orp['esc']*100:.1f}%   random: esc={rb['esc']*100:.1f}%" if rb
              else f"  outcome-oracle: esc={orp['esc']*100:.1f}%   random: never")
        print(f"\n  {'gate':<26}{'EVAL-ORACLE esc%':>18}{'   '}{'HONEST esc% / acc / hit':>28}")
        rows = []
        for name, m in reg.items():
            try:
                m.fit(D)
            except Exception as e:
                continue
            se = m.score_eval(P)
            mb = min_esc_at_parity(P, se, parity)
            # honest: escalate calib error-rate fraction by this gate's calib quantile
            err = 1.0 - D.calib["ok"].mean()
            thr = float(np.quantile(m.score_calib, 1.0 - err))
            esc = se > thr
            sc = D.score(P, esc)
            rows.append((name, mb, sc, parity))
        for name, mb, sc, par in sorted(rows, key=lambda r: (r[1]["esc"] if r[1] else 9)):
            o = f"{mb['esc']*100:>16.1f}%" if mb else f"{'never':>17}"
            hit = "Y" if sc["acc"] >= par - 1e-9 else "n"
            print(f"  {name:<26}{o}   {sc['esc']*100:>10.1f}% /{sc['acc']:>7.4f} / {hit}")
        print(f"  (lower escalation = better; HONEST hit=n means it misses parity at the err-rate point)")
    print("\nDeployed margin gate reference: ALL-6 ~63% escalation at parity. Target: BEAT the best EVAL-ORACLE.")


if __name__ == "__main__":
    main()
