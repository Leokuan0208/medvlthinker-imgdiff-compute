#!/usr/bin/env python3
"""
stability_rescue_bootstrap.py - paired bootstrap CIs for the Visual-Stability RESCUE headline.
Computes per-sample (flops, correctness) for the DEPLOYED gate and the param-free 3-cap RESCUE,
then bootstraps samples (stratified by benchmark) to get 95% CIs on:
  Δaccuracy (rescue - deployed)   -> non-inferiority check (CI upper-ish; want >= -0.01)
  Δbackbone% (rescue - deployed)  -> compute saving (want clearly < 0)
  Δ32B-call-rate                  -> escalation saving
Reuses the exact FLOPs accounting + real gate rule from stability_rescue_cost.py.
"""
import os, json, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
TC = json.load(open("ckpts/token_cache.json"))
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
N7, N32 = 7.6e9, 33.0e9
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]
EXTRA = ["cap80", "cap160", "cap640"]

def load_jsonl(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def cap_file(cap, ds):
    return (os.path.join(FULLRES, f"ckpt_{ds}_nothink_norag.jsonl") if cap == "fullres"
            else os.path.join(PRUNE, cap, f"ckpt_{ds}_nothink_norag.jsonl"))
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

# per-sample arrays
flops_dep, ok_dep, flops_res, ok_res, esc_dep, esc_res, ds_id = [], [], [], [], [], [], []
for di, ds in enumerate(COMP4):
    caps = {c: load_jsonl(cap_file(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    margins = []
    keep = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        if any(str(i) not in TC[ds][c] for c in CAPS): continue
        keep.append(i); margins.append(margin(caps["cap320"][i]))
    proba = GATE.predict_proba(np.array(margins, dtype=np.float32).reshape(-1, 1))[:, 1]
    for j, i in enumerate(keep):
        P = {c: TC[ds][c][str(i)][0] for c in CAPS}
        G7 = {c: (caps[c][i].get("gen_tokens") or 2) for c in CAPS}
        G32 = strong[i].get("gen_tokens") or 0
        run7 = lambda c: 2 * N7 * (P[c] + G7[c]); run32 = 2 * N32 * (P["fullres"] + G32)
        ok320 = caps["cap320"][i]["ok"]; ok32 = strong[i]["ok"]
        p320 = caps["cap320"][i]["pred"]
        low = bool(proba[j] < TAU)
        stable = all(caps[c][i]["pred"] == p320 for c in EXTRA)
        # deployed
        cd = run7("cap320") + (run32 if low else 0.0)
        flops_dep.append(cd); ok_dep.append(ok32 if low else ok320); esc_dep.append(low)
        # rescue
        do = low and not stable
        cr = run7("cap320") + (sum(run7(c) for c in EXTRA) if low else 0.0) + (run32 if do else 0.0)
        flops_res.append(cr); ok_res.append(ok32 if do else ok320); esc_res.append(do)
        ds_id.append(di)
        # always-32B baseline contribution stored separately
flops_dep = np.array(flops_dep); ok_dep = np.array(ok_dep); esc_dep = np.array(esc_dep)
flops_res = np.array(flops_res); ok_res = np.array(ok_res); esc_res = np.array(esc_res)
ds_id = np.array(ds_id)
# always-32B baseline per sample (run32), aligned to the kept order
base32 = []
for di, ds in enumerate(COMP4):
    caps = {c: load_jsonl(cap_file(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        if any(str(i) not in TC[ds][c] for c in CAPS): continue
        G32 = strong[i].get("gen_tokens") or 0
        base32.append(2 * N32 * (TC[ds]["fullres"][str(i)][0] + G32))
base32 = np.array(base32)

n = len(ok_dep)
print(f"n={n} samples (competent-4). point estimates:")
def backbone(fl): return fl.sum() / base32.sum()
print(f"  deployed : acc={ok_dep.mean():.4f}  backbone={backbone(flops_dep)*100:.1f}%  32B-call={esc_dep.mean()*100:.1f}%")
print(f"  rescue   : acc={ok_res.mean():.4f}  backbone={backbone(flops_res)*100:.1f}%  32B-call={esc_res.mean()*100:.1f}%")

rng = np.random.RandomState(0)
dA, dB, dC = [], [], []
for _ in range(5000):
    idx = rng.randint(0, n, n)  # simple bootstrap (benchmarks already pooled; ds proportions ~preserved)
    dA.append(ok_res[idx].mean() - ok_dep[idx].mean())
    dB.append(flops_res[idx].sum() / base32[idx].sum() - flops_dep[idx].sum() / base32[idx].sum())
    dC.append(esc_res[idx].mean() - esc_dep[idx].mean())
def ci(x): x = np.sort(x); return x[int(.025*len(x))], x[int(.975*len(x))]
for name, d in [("Δacc (rescue-deployed)", dA), ("Δbackbone%", [v*100 for v in dB]), ("Δ32B-call%", [v*100 for v in dC])]:
    lo, hi = ci(d)
    print(f"  {name:<26} mean={np.mean(d):+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]")
print("\nNON-INFERIORITY: rescue accuracy is non-inferior if the Δacc CI lower bound > -0.01 (1 pt).")
json.dump({"n": n, "acc_dep": float(ok_dep.mean()), "acc_res": float(ok_res.mean()),
           "bb_dep": float(backbone(flops_dep)), "bb_res": float(backbone(flops_res)),
           "call_dep": float(esc_dep.mean()), "call_res": float(esc_res.mean()),
           "dacc_ci": list(map(float, ci(dA))), "dbb_ci": list(map(float, ci([v for v in dB]))),
           "dcall_ci": list(map(float, ci(dC)))},
          open("results/cascade_methods/artifacts/stability_rescue_bootstrap.json", "w"), indent=1)
print("-> results/cascade_methods/artifacts/stability_rescue_bootstrap.json")
