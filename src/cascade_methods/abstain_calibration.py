#!/usr/bin/env python3
"""
abstain_calibration.py - is the safe-abstention threshold DEPLOYABLE? A clinician sets a target risk r*
(max error among auto-answered). We CALIBRATE the confidence threshold on a held-out split to achieve r*,
then measure the REALIZED risk + coverage on fresh data. Two protocols:
  (A) in-distribution: calibrate on 50% of a dataset, deploy on the other 50% (the realistic protocol).
  (B) cross-dataset transfer: calibrate on SLAKE, deploy on VQA-RAD/PathVQA (robustness / shift).
Deployed model = Lingshu-32B (the strongest, judge-scored). If (A) holds r* with good coverage, the method
is genuinely deployable; (B) tests how far a single calibration travels. Offline. 20 seeds.
"""
import os, json
import numpy as np
def load(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def judged(d, jp):
    if os.path.exists(jp):
        j = {r["idx"]: r["judge_ok"] for r in (json.loads(l) for l in open(jp) if l.strip())}
        for i, r in d.items():
            if i in j: r["modal_ok"] = j[i]
    return d
LS = "ckpts/openvqa/strong_lingshu"
def get(ds):
    s = judged(load(f"{LS}/ckpt_{ds}_lingshu32b.jsonl"), f"{LS}/ckpt_{ds}_lingshu32b.judge.jsonl")
    cor = np.array([r["modal_ok"] for r in s.values()]); cf = np.array([r.get("seqlogprob") or 0.0 for r in s.values()])
    return cf, cor
def calibrate_tau(cf, cor, rstar):
    """highest-coverage tau s.t. risk(answer conf>=tau) <= rstar, on calibration data."""
    order = np.argsort(-cf); c = cor[order]; k = np.arange(1, len(c)+1); risk = 1 - np.cumsum(c)/k
    ok = np.where(risk <= rstar)[0]
    if len(ok) == 0: return np.inf            # cannot meet target -> abstain on all
    kbest = ok[-1]; return cf[order][kbest]    # threshold = confidence at the last admissible rank
def deploy(cf, cor, tau):
    ans = cf >= tau
    if ans.sum() == 0: return 0.0, float("nan")
    return float(ans.mean()), float(1 - cor[ans].mean())

DS = ["slake_open", "vqa_rad_open", "pathvqa_open", "kvasir_open"]
LAB = {"slake_open": "SLAKE", "vqa_rad_open": "VQA-RAD", "pathvqa_open": "PathVQA", "kvasir_open": "Kvasir"}
RSTAR = 0.05
data = {ds: get(ds) for ds in DS}

print(f"DEPLOYABILITY of safe abstention (Lingshu-32B, target risk r*={RSTAR:.0%}):\n")
print("(A) IN-DISTRIBUTION (calibrate on 50%, deploy on held-out 50%) — the realistic protocol:")
out = {"in_distribution": {}, "cross_dataset": {}}
for ds in DS:
    cf, cor = data[ds]; n = len(cf); covs, risks = [], []
    for seed in range(20):
        idx = np.random.default_rng(seed).permutation(n); cal, dep = idx[:n//2], idx[n//2:]
        tau = calibrate_tau(cf[cal], cor[cal], RSTAR)
        cov, rk = deploy(cf[dep], cor[dep], tau)
        covs.append(cov); risks.append(rk if rk == rk else 0.0)
    cov_m, risk_m = np.mean(covs), np.nanmean([r for r in risks if r == r])
    out["in_distribution"][ds] = {"coverage": float(cov_m), "realized_risk": float(risk_m)}
    flag = "OK" if risk_m <= RSTAR + 0.02 else "RISK OVERSHOOT"
    print(f"  {LAB[ds]:<9} deployed coverage={cov_m:.2f}  realized risk={risk_m:.3f}  [{flag}]")

print("\n(B) CROSS-DATASET (calibrate tau on SLAKE, deploy elsewhere) — robustness to distribution shift:")
cf_s, cor_s = data["slake_open"]; tau_s = calibrate_tau(cf_s, cor_s, RSTAR)
for ds in DS:
    cf, cor = data[ds]; cov, rk = deploy(cf, cor, tau_s)
    out["cross_dataset"][ds] = {"coverage": float(cov), "realized_risk": float(rk) if rk == rk else None}
    flag = "OK" if (rk == rk and rk <= RSTAR + 0.02) else ("RISK OVERSHOOT" if rk == rk else "abstain-all")
    print(f"  SLAKE-tau on {LAB[ds]:<9} coverage={cov:.2f}  realized risk={rk:.3f}  [{flag}]")
json.dump(out, open("results/cascade_methods/artifacts/abstain_calibration.json", "w"), indent=1)
print("\nREAD: (A) holding risk<=r* with non-trivial coverage => deployable with standard held-out calibration.")
print("      (B) shows how far one calibration travels; a per-deployment calibration set is the safe practice.")
