#!/usr/bin/env python3
"""
control_think_signals.py - is RESOLUTION-STABILITY the right signal to gate the think tier, or is the
rescue@think "win" just generic think->big-nt rebalancing? Compare think-skip signals at parity
(in-sample frontier, min-LATENCY at acc>=always-big-think parity), MedVLThinker:
  ACC-v2            : think on ALL disagree (no skip)
  +stability        : skip think if small-nt resolution-stable (OUR rescue)
  +random(matched)  : skip think on a random subset of disagree, matched skip-rate (control)
  +bignt-confidence : skip think if big-nt margin high (natural alternative think-gate)
  +inverse-stability: skip think if small-nt UNSTABLE (sanity: should be worse if signal is real)
If stability ~ random, the signal adds nothing; if a confidence gate beats it, stability is suboptimal.
"""
import os, json
import numpy as np
PRUNE = "ckpts/gate_7b_prune"; BIGNT = "ckpts/gate_32b_modes/nothink_cap320"; BIGTH = "ckpts/gate_32b"
LATD = "ckpts/acc_gen/medvlthinker/lat"; CACHE = json.load(open("ckpts/token_cache.json"))
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]; ALL5 = COMP4 + ["MMMU"]
EXTRA = ["cap80", "cap160", "cap640"]; N0, N1 = 7.6e9, 33e9
def load(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
import glob
def fit(tier, key):
    pts = []
    for f in glob.glob(os.path.join(LATD, tier, "ckpt_*lat*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get(key) is not None: pts.append((r.get("gen_tokens") or 0, r[key]))
    g = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    if g.std() < 1: return (float(np.median(y)), 0.0)
    b, a = np.polyfit(g, y, 1); resid = y-(b*g+a); k = np.abs(resid) <= 2.5*resid.std()+1e-9
    if k.sum() >= 4: b, a = np.polyfit(g[k], y[k], 1)
    return (float(a), float(max(b, 0)))
LAT = {t: fit(t, "latency_s") for t in ["small_nt", "big_nt", "big_think"]}
def tl(t, g): a, b = LAT[t]; return np.clip(a+b*g, 0, None)

def build(names):
    rng = np.random.default_rng(0); D = []
    for ds in names:
        c320 = load(f"{PRUNE}/cap320/ckpt_{ds}_nothink_norag.jsonl")
        cex = {c: load(f"{PRUNE}/{c}/ckpt_{ds}_nothink_norag.jsonl") for c in EXTRA}
        bnt = load(f"{BIGNT}/ckpt_{ds}_nothink_norag.jsonl"); bth = load(f"{BIGTH}/ckpt_{ds}_think_norag.jsonl")
        idx = set(c320) & set(bnt) & set(bth) & {int(k) for k in CACHE[ds]["cap320"]} & {int(k) for k in CACHE[ds]["fullres"]}
        for c in EXTRA: idx &= set(cex[c])
        for i in sorted(idx):
            p = c320[i]["pred"]
            D.append(dict(m0=margin(c320[i].get("opt_logprobs")), m1=margin(bnt[i].get("opt_logprobs")),
                disagree=(p != bnt[i]["pred"]), stable=all(cex[c][i]["pred"] == p for c in EXTRA),
                ok0=c320[i]["ok"], ok1=bnt[i]["ok"], ok2=bth[i]["ok"], rnd=rng.random(),
                g0=c320[i].get("gen_tokens") or 2, g1=bnt[i].get("gen_tokens") or 2, g2=bth[i].get("gen_tokens") or 0,
                Pc=CACHE[ds]["cap320"][str(i)][0], Pf=CACHE[ds]["fullres"][str(i)][0]))
    return D

def evalsig(D, names, signal):
    m0 = np.array([d["m0"] for d in D]); m1 = np.array([d["m1"] for d in D])
    dis = np.array([d["disagree"] for d in D]); stab = np.array([d["stable"] for d in D])
    rnd = np.array([d["rnd"] for d in D])
    ok0 = np.array([d["ok0"] for d in D]); ok1 = np.array([d["ok1"] for d in D]); ok2 = np.array([d["ok2"] for d in D])
    g0 = np.array([d["g0"] for d in D]); g1 = np.array([d["g1"] for d in D]); g2 = np.array([d["g2"] for d in D])
    l0 = tl("small_nt", g0); l1 = tl("big_nt", g1); l2 = tl("big_think", g2)
    parity = ok2.mean()
    skip_rate = (stab[dis].mean() if dis.any() else 0)        # match other signals to stability's skip fraction
    # skip-think masks among disagree candidates (True = SKIP think -> use big-nt)
    if signal == "none":      skip = np.zeros(len(D), bool)
    elif signal == "stability":   skip = stab
    elif signal == "inverse":     skip = ~stab
    elif signal == "random":      skip = rnd < skip_rate
    elif signal == "bignt_conf":                              # skip where big-nt most confident, matched rate
        thr = np.quantile(m1[dis], 1-skip_rate) if dis.any() else 0; skip = m1 >= thr
    best = None
    for t0 in np.unique(np.quantile(m0, np.linspace(0, 1, 81))):
        E0 = m0 < t0; E1 = E0 & dis & (~skip)
        ok = np.where(~E0, ok0, np.where(~E1, ok1, ok2))
        if ok.mean() >= parity - 1e-9:
            lat = (l0 + np.where(E0, l1, 0) + np.where(E1, l2, 0)).mean()
            if best is None or lat < best[0]: best = (lat, ok.mean(), E0.mean(), E1.mean())
    return parity, best

def main():
    OUT = {}
    for label, names in [("ALL-5", ALL5), ("COMPETENT-4", COMP4)]:
        D = build(names); OUT[label] = {}
        print(f"\n===== {label} (n={len(D)}) — min-LATENCY @ big-think parity, in-sample =====")
        print(f"  {'think-skip signal':<22}{'lat(s)':>9}{'acc':>9}{'esc0':>8}{'think':>8}")
        for sig in ["none", "stability", "random", "bignt_conf", "inverse"]:
            parity, b = evalsig(D, names, sig)
            OUT[label]["parity"] = float(parity)
            if b is None: print(f"  {sig:<22}{'cannot reach parity':>34}"); continue
            lat, acc, e0, th = b
            tag = " <- OURS" if sig == "stability" else ""
            print(f"  {sig:<22}{lat:>8.2f}s{acc:>9.4f}{e0*100:>7.0f}%{th*100:>7.0f}%{tag}")
            OUT[label][sig] = dict(lat=float(lat), acc=float(acc), esc0=float(e0), think=float(th))
        print(f"  (parity={parity:.4f}; 'none'=ACC-v2. If random/bignt_conf <= stability, the signal is not special.)")
    os.makedirs("results/cascade_methods/artifacts/rescue_allfam", exist_ok=True)
    json.dump(OUT, open("results/cascade_methods/artifacts/rescue_allfam/control_think_signals.json", "w"), indent=1)
    print("\n-> results/cascade_methods/artifacts/rescue_allfam/control_think_signals.json")

if __name__ == "__main__":
    main()
