#!/usr/bin/env python3
"""
final_3tier.py - HONEST all-6 evaluation of the 3-tier Compute-Configuration Cascade:
  Tier1 7B-nothink@cap320  -> Tier2 32B-nothink@cap320 -> Tier3 32B-think@fullres
Thresholds calibrated on PMC-train (training-free): tau1 on 7B margins (escalate 7B error-rate
fraction), tau2 on 32B-nothink@cap320 margins (escalate that leg's error-rate fraction to think).
Denominator = always-32B-think@fullres. Tier3 re-prefills (think prompt differs from no-think).
CPU only; launch from repo root. Needs ckpts/gate_32b_pmctrain_nothink_cap320/.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (_load_arm, _load_calib, signals_from_logprobs, EVAL_DIR, DIR_32B, CACHE,
                     ALL6, COMPETENT, N7, N32, CascadeData)

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
NOTH_CAP = "ckpts/gate_32b_modes/nothink_cap320"
CALIB_NOTH_CAP = "ckpts/gate_32b_pmctrain_nothink_cap320"


def _margin(d):
    v = sorted((d or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0


def main():
    cache = json.load(open(J(CACHE)))
    ev = _load_arm(J(EVAL_DIR["cap320"]), "nothink_norag")
    nothC = _load_arm(J(NOTH_CAP), "nothink_norag")
    think = _load_arm(J(DIR_32B), "think_norag")
    # honest thresholds
    Dc = CascadeData("cap320"); tau1 = float(np.quantile(Dc.calib["sig"]["margin"], 1.0 - Dc.calib["ok"].mean()))
    cal = _load_calib(J(CALIB_NOTH_CAP))
    if not cal:
        print("nothink@cap320 calib not present yet; rerun after the calibration run."); return
    cm = np.array([_margin(r.get("opt_logprobs")) for r in cal]); cok = np.array([r["ok"] for r in cal], float)
    tau2 = float(np.quantile(cm, 1.0 - cok.mean()))
    print(f"honest thresholds: tau1(7B)={tau1:.3f}  tau2(32B-nothink@cap320)={tau2:.3f}  "
          f"(calib: 7B acc={Dc.calib['ok'].mean():.3f}, 32B-nothink@cap320 acc={cok.mean():.3f})")

    D = {}
    for ds in ALL6:
        idx = sorted(set(ev[ds]) & set(nothC[ds]) & set(think[ds]) & {int(k) for k in cache[ds]["cap320"]})
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        D[ds] = dict(
            ok7=np.array([ev[ds][i]["ok"] for i in idx], float),
            okN=np.array([nothC[ds][i]["ok"] for i in idx], float),
            okT=np.array([think[ds][i]["ok"] for i in idx], float),
            m7=np.array([_margin(ev[ds][i].get("opt_logprobs")) for i in idx], float),
            mN=np.array([_margin(nothC[ds][i].get("opt_logprobs")) for i in idx], float),
            run7=2*N7*np.array([cC[str(i)][0] + (ev[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runN=2*N32*np.array([cC[str(i)][0] + (nothC[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runT=2*N32*np.array([cF[str(i)][0] + (think[ds][i].get("gen_tokens") or 0) for i in idx], float),
        )

    def pool(names):
        names = [d for d in names if d in D]
        return {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}

    for sub, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = pool(names); base = P["runT"].sum(); target = P["okT"].mean()
        e1 = P["m7"] < tau1
        e2 = e1 & (P["mN"] < tau2)
        final = np.where(~e1, P["ok7"], np.where(~e2, P["okN"], P["okT"]))
        cost = P["run7"].sum() + P["runN"][e1].sum() + P["runT"][e2].sum()
        # 2-tier (nothink@cap320 only, no think tier) for reference
        acc2 = np.where(e1, P["okN"], P["ok7"]).mean(); bb2 = (P["run7"].sum() + P["runN"][e1].sum())/base
        # guardrail
        worse = [ds for ds in names if
                 np.where(~(D[ds]["m7"] < tau1), D[ds]["ok7"],
                          np.where(~((D[ds]["m7"] < tau1) & (D[ds]["mN"] < tau2)), D[ds]["okN"], D[ds]["okT"])).mean()
                 < D[ds]["ok7"].mean() - 1e-9]
        print(f"\n[{sub}] think-parity={target:.4f}")
        print(f"  2-tier (->nothink@cap320):  acc={acc2:.4f}  backbone={bb2*100:.1f}%  esc1={e1.mean()*100:.0f}%"
              f"  {'PARITY' if acc2>=target-1e-9 else 'miss'}")
        print(f"  3-tier (+think@fullres):    acc={final.mean():.4f}  backbone={cost/base*100:.1f}%  "
              f"esc1={e1.mean()*100:.0f}% ->think={e2.mean()*100:.0f}%  "
              f"{'PARITY' if final.mean()>=target-1e-9 else 'miss'}  guardrail={'Y' if not worse else worse}")


if __name__ == "__main__":
    main()
