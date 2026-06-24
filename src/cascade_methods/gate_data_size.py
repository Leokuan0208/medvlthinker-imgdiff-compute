#!/usr/bin/env python3
"""
gate_data_size.py - does using MORE PMC-VQA-train data for gate design change the result?
The deployed gate / ACC thresholds are calibrated on a 3000-sample slice of PMC-VQA-train. We
cannot label all 337k cheaply, but we CAN test convergence: subsample the existing 3000 calib at
{250,500,1000,2000,3000}, refit the ACC-v3 thresholds (tau0 small-margin, tau1 big-nt-margin) on
each, apply the FROZEN thresholds to the held-out eval competent-4, and see if the gate / its eval
behaviour has converged by 3000. If flat from ~1000->3000, more data would not change it.
20 bootstrap subsamples per size. Offline (pmctrain has all 3 tiers labeled).
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, COMPETENT, CACHE
J = lambda p: os.path.join(os.path.expanduser("~/medvlthinker-imgdiff-compute"), p)
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
def load_jsonl(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
# ---- calib (pmctrain, all 3 tiers) ----
c0c = load_jsonl(J("ckpts/gate_7b_pmctrain_prune/cap320/ckpt_nothink.jsonl"))
c1c = load_jsonl(J("ckpts/gate_32b_pmctrain_nothink_cap320/ckpt_nothink.jsonl"))
c2c = load_jsonl(J("ckpts/gate_32b_pmctrain/ckpt_think.jsonl"))
cidx = sorted(set(c0c) & set(c1c) & set(c2c))
CAL = dict(m0=np.array([margin(c0c[i].get("opt_logprobs")) for i in cidx]),
           m1=np.array([margin(c1c[i].get("opt_logprobs")) for i in cidx]),
           dis=np.array([float(c0c[i]["pred"] != c1c[i]["pred"]) for i in cidx]),
           ok0=np.array([c0c[i]["ok"] for i in cidx], float), ok1=np.array([c1c[i]["ok"] for i in cidx], float),
           ok2=np.array([c2c[i]["ok"] for i in cidx], float))
# ---- eval (competent-4) ----
e0 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
e1 = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")
e2 = _load_arm(J("ckpts/gate_32b"), "think_norag")
E = {"m0": [], "m1": [], "dis": [], "ok0": [], "ok1": [], "ok2": []}
for ds in COMPETENT:
    idx = sorted(set(e0[ds]) & set(e1[ds]) & set(e2[ds]))
    for i in idx:
        E["m0"].append(margin(e0[ds][i].get("opt_logprobs"))); E["m1"].append(margin(e1[ds][i].get("opt_logprobs")))
        E["dis"].append(float(e0[ds][i]["pred"] != e1[ds][i]["pred"]))
        E["ok0"].append(e0[ds][i]["ok"]); E["ok1"].append(e1[ds][i]["ok"]); E["ok2"].append(e2[ds][i]["ok"])
E = {k: np.array(v, float) for k, v in E.items()}

def calib_thr(idx):
    """ACC-v3 thresholds on a calib subset: min think-rate s.t. acc>=calib parity."""
    m0, m1, dis, ok0, ok1, ok2 = (CAL["m0"][idx], CAL["m1"][idx], CAL["dis"][idx], CAL["ok0"][idx], CAL["ok1"][idx], CAL["ok2"][idx])
    tgt = ok2.mean(); q0 = np.quantile(m0, np.linspace(0, 1, 22)); q1 = np.quantile(m1, np.linspace(0, 1, 22)); best = None
    for t0 in q0:
        E0 = m0 < t0
        for t1 in q1:
            E1 = E0 & (dis > 0.5) & (m1 < t1)
            acc = np.where(~E0, ok0, np.where(~E1, ok1, ok2)).mean()
            if acc >= tgt - 1e-9:
                th = E1.mean()
                if best is None or th < best[0]: best = (th, float(t0), float(t1))
    return (best[1], best[2]) if best else (q0[0], q1[0])
def eval_thr(t0, t1):
    E0 = E["m0"] < t0; E1 = E0 & (E["dis"] > 0.5) & (E["m1"] < t1)
    acc = np.where(~E0, E["ok0"], np.where(~E1, E["ok1"], E["ok2"])).mean()
    return acc, E0.mean(), E1.mean()

print("=" * 86)
print("GATE-DESIGN DATA SIZE — convergence of ACC-v3 thresholds + eval behaviour vs #calib samples")
print(f"  calib=PMC-VQA-train (n={len(cidx)} available), eval=competent-4 (n={len(E['m0'])}), 20 subsamples/size")
print("=" * 86)
print(f"  {'#calib':>8}{'tau0':>10}{'tau1':>10}{'eval acc':>11}{'eval esc0':>11}{'eval think':>12}")
rng = np.random.default_rng(0)
for size in [250, 500, 1000, 2000, 3000]:
    t0s, t1s, accs, e0s, e1s = [], [], [], [], []
    for _ in range(20):
        idx = rng.choice(len(cidx), size=min(size, len(cidx)), replace=False)
        t0, t1 = calib_thr(idx); a, ee0, ee1 = eval_thr(t0, t1)
        t0s.append(t0); t1s.append(t1); accs.append(a); e0s.append(ee0); e1s.append(ee1)
    print(f"  {size:>8}{np.mean(t0s):>10.3f}{np.mean(t1s):>10.3f}{np.mean(accs):>11.4f}{np.mean(e0s)*100:>10.0f}%{np.mean(e1s)*100:>11.0f}%"
          f"   (±{np.std(accs):.4f} acc, ±{np.std(t0s):.3f} tau0)")
print("\n  READ: if eval acc / tau0 are flat (within noise) from ~1000->3000, the gate has CONVERGED")
print("  and using the full 337k PMC-VQA-train would NOT change the result (a scalar threshold needs")
print("  few samples). If still drifting at 3000, more calib data could help -> then label more.")
