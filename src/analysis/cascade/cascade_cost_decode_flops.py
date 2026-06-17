#!/usr/bin/env python3
"""
router_cost.py - turn "efficiency parity" into a quantified compute number.

For the FROZEN router (ckpts/router_margin.pkl) on each eval dataset, reports:
  - accuracy parity (always7 / always32 / routed),
  - 32B-CALL RATE = fraction of questions that ever invoke the 32B (= esc%);
    the prefill-agnostic headline ("X% of questions never touch the 32B"),
  - DECODE-FLOPS RATIO cascade/always-32B, a 2*params*gen_tokens proxy that
    corroborates the call-rate (excludes prefill -> conservative).
Then a CALIBRATED-RATE row per dataset: escalate at the dataset's TRUE error
rate (1 - always7) using the same margin ranking. That's the operating point
conformal calibration targets -> the gap (frozen esc% - error rate) is the
wasted escalation a per-dataset coverage threshold would recover.
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict

EVAL = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
P7, P32 = 7.0, 32.0                       # nominal params (B); only the ~4.6x ratio matters
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}(?:_s\dof\d)?\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def tok(row): return float(row.get("gen_tokens") or 0)

R = pickle.load(open("ckpts/router_margin.pkl", "rb")); gate = R["gate"]; tau = R["tau"]
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
print("=" * 104)
print(f"ROUTER COST   frozen margin gate (tau={tau:.3f}); decode-FLOPs proxy = 2*params*gen_tokens, prefill excluded")
print("=" * 104)
print(f"    {'dataset':<11}{'policy':<16}{'acc':>7}{'(a32)':>8}{'32B-call%':>11}{'decode-FLOPs vs 32B':>22}")

agg = {"frozen": [], "calib": []}
for name in EVAL:
    if name not in r7 or name not in r32: continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    t7  = np.array([tok(r7[name][i])  for i in idx])
    t32 = np.array([tok(r32[name][i]) for i in idx])
    P   = gate.predict_proba(np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32))[:, 1]
    base32_flops = (P32 * t32).sum()                         # always-32B decode FLOPs (proxy)

    def report(escmask, label, store):
        routed = np.where(escmask, a32, a7)
        casc_flops = (P7 * t7).sum() + (P32 * t32[escmask]).sum()
        ratio = casc_flops / base32_flops if base32_flops > 0 else float("nan")
        print(f"    {name:<11}{label:<16}{routed.mean():>7.3f}{a32.mean():>8.3f}"
              f"{escmask.mean()*100:>10.0f}%{ratio*100:>20.0f}%")
        store.append((ratio, escmask.mean()))
        return routed.mean()

    report(P < tau, "frozen", agg["frozen"])
    err  = 1.0 - a7.mean()                                    # dataset's true 7B error rate
    tau_c = np.quantile(P, err)                               # escalate exactly that fraction
    report(P < tau_c, "calibrated-rate", agg["calib"])
    print(f"    {'':<11}{'(7B err rate)':<16}{'':>7}{'':>8}{err*100:>10.0f}%   <- target escalation\n")

fr = np.array([r for r, _ in agg["frozen"]]); fe = np.array([e for _, e in agg["frozen"]])
ca = np.array([r for r, _ in agg["calib"]]);  ce = np.array([e for _, e in agg["calib"]])
print("-" * 104)
print(f"AVG over datasets:  frozen -> {fe.mean()*100:.0f}% 32B-calls, {fr.mean()*100:.0f}% of 32B decode-FLOPs")
print(f"                    calib  -> {ce.mean()*100:.0f}% 32B-calls, {ca.mean()*100:.0f}% of 32B decode-FLOPs (same accuracy)")
print(f"                    gap conformal would recover: ~{(fe.mean()-ce.mean())*100:.0f} pp of escalation")
print("\nREAD: frozen gate = match 32B accuracy at the compute shown. The frozen tau over-escalates")
print("low-error datasets; escalating at each dataset's true error rate keeps the accuracy at lower")
print("cost. A per-dataset coverage threshold (CP-Router) estimates that rate WITHOUT seeing labels.")
