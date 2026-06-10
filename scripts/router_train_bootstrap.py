#!/usr/bin/env python3
"""
router_train_bootstrap.py - paired-bootstrap CIs for the FROZEN router's transfer.

Loads the deployed artifact ckpts/router_margin.pkl (margin gate + tau, fit on
PMC-VQA train), applies it UNCHANGED to each eval dataset, and puts a 95% paired
bootstrap CI on (routed - always-32B) per dataset. Distinguishes significant
efficiency-parity (CI contains 0 at esc<100%) from a real gain (CI>0), so we
don't re-claim small transfer gains that are within noise.
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict

EVAL   = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
B_BOOT = 2000
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}_s\dof\d\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def boot(gain_i, B=B_BOOT, seed=0):
    rng = np.random.RandomState(seed); n = len(gain_i)
    idx = rng.randint(0, n, size=(B, n)); means = gain_i[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5]); return float(lo), float(hi), float((means <= 0).mean())

R = pickle.load(open("ckpts/router_margin.pkl", "rb")); gate = R["gate"]; tau = R["tau"]
print("=" * 94)
print(f"FROZEN ROUTER BOOTSTRAP   margin gate from {R['trained_on']}, tau={tau:.3f}, B={B_BOOT}")
print("=" * 94)
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
print(f"    {'dataset':<11}{'n':>6}{'always32':>10}{'routed':>9}{'esc%':>7}{'gain':>9}{'95% CI':>20}{'p':>8}   verdict")
seed = 0
for name in EVAL:
    if name not in r7 or name not in r32: continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    X   = np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32)
    P   = gate.predict_proba(X)[:, 1]; esc = P < tau; routed = np.where(esc, a32, a7)
    gi  = routed - a32; g = gi.mean()
    lo, hi, p = boot(gi, seed=seed); seed += 1
    v = "gain (CI>0)" if lo > 0 else ("worse (CI<0)" if hi < 0 else "parity (CI~0)")
    print(f"    {name:<11}{len(idx):>6}{a32.mean():>10.3f}{routed.mean():>9.3f}{esc.mean()*100:>6.0f}%"
          f"{g:>+9.3f}   [{lo:+.3f},{hi:+.3f}]{p:>8.3f}   {v}")
print("\nREAD: parity (CI~0) at esc<100% = frozen gate matches 32B at reduced cost, honestly.")
print("gain (CI>0) = a real accuracy gain. The deployable claim is parity-or-better transferring")
print("across datasets; conformal calibration (CP-Router) then tightens cost per dataset.")
