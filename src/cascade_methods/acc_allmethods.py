#!/usr/bin/env python3
"""
acc_allmethods.py - ALL cascade methods (the full 3-tier bake-off from final_3tier_comparison.py) run on
ANY 2-size family. Methods: Ours (ACC-v2 agreement), CASP-Stability (trained), ACC-v1 (margin),
MSP/Chow, entropy, Gini/DOCTOR, AutoMix (self-verify), FrugalGPT-learned, Jitkrittum L2D, random.
Config = small-nt -> big-nt -> big-think. Honest 50/50 stratified split x20 seeds, calibrated at parity
at MIN FLOPs (model-agnostic cost; latency models are MedVLThinker-only). Prompt tokens from token_cache
for Qwen2.5-VL families (same processor: medvlthinker/qoq/lingshu), constant approx for others.
Usage: python3 src/cascade_methods/acc_allmethods.py --family {medvlthinker,qoq,chiron}
"""
import sys, os, glob, json, re, argparse; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs, ALL6, ALL5
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
J = lambda p: os.path.join("/home/jamesyang/medvlthinker-imgdiff-compute", p)
FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]
FAM = {
  "medvlthinker": dict(N0=7.6e9, N1=33e9, tokcache=True, lat="ckpts/acc_gen/medvlthinker/lat",
      c0=("ckpts/gate_7b_prune/cap320", "nothink_norag"), c1=("ckpts/gate_32b_modes/nothink_cap320", "nothink_norag"),
      c2=("ckpts/gate_32b", "think_norag"), ver=("ckpts/gate_7b_verify", "verify")),
  "lingshu": dict(N0=7.6e9, N1=33e9, tokcache=True, lat="ckpts/acc_gen/lingshu/lat",
      c0=("ckpts/acc_gen/lingshu7b/cap320", "lingshu7b"), c1=("ckpts/acc_gen/lingshu32b/nothink_cap320", "lingshu32b"),
      c2=("ckpts/acc_gen/lingshu32b/think_native", "lingshu32b"), ver=("ckpts/acc_gen/lingshu7b/verify", "verify")),
  "qoq": dict(N0=7.6e9, N1=33e9, tokcache=True, lat="ckpts/acc_gen/qoq/lat",
      c0=("ckpts/acc_gen/qoq7b/cap320", "qoq7b"), c1=("ckpts/acc_gen/qoq32b/nothink_cap320", "qoq32b"),
      c2=("ckpts/acc_gen/qoq32b/think_native", "qoq32b"), ver=("ckpts/acc_gen/qoq7b/verify", "verify")),
  "chiron": dict(N0=2.0e9, N1=8.0e9, tokcache=False, Pconst=1024.0, lat="ckpts/acc_gen/chiron/lat",
      c0=("ckpts/acc_gen/chiron2b/nt", "chiron2b_nt"), c1=("ckpts/acc_gen/chiron8b/nt", "chiron8b_nt"),
      c2=("ckpts/acc_gen/chiron8b/think_native", "chiron8b_think"), ver=("ckpts/acc_gen/chiron2b/verify", "chiron2b_verify")),
  "medgemma": dict(N0=4.3e9, N1=27e9, tokcache=False, Pconst=900.0, lat="ckpts/acc_gen/medgemma/lat",
      c0=("ckpts/acc_gen/medgemma4b/nt", "medgemma4b_nt"), c1=("ckpts/acc_gen/medgemma27b/nt", "medgemma27b_nt"),
      c2=("ckpts/acc_gen/medgemma27b/think_native", "medgemma27b_think"), ver=("ckpts/acc_gen/medgemma4b/verify", "medgemma4b_verify")),
}
ABBR = {"PMC-VQA": "PMC", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD", "PathVQA": "PathV", "MMMU": "MMMU",
        "MedXpert-Reasoning": "MX-R", "MedXpert-Understanding": "MX-U"}
def loadarm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(J(os.path.join(d, f"ckpt_*{tag}*.jsonl"))):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if m:
            for l in open(f):
                if l.strip(): r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out
A = argparse.ArgumentParser(); A.add_argument("--family", required=True, choices=list(FAM)); A = A.parse_args()
C = FAM[A.family]; N0, N1 = C["N0"], C["N1"]
c0 = loadarm(*C["c0"]); c1 = loadarm(*C["c1"]); c2 = loadarm(*C["c2"]); ver = loadarm(*C["ver"])
cache = json.load(open(J("ckpts/token_cache.json"))) if C["tokcache"] else None
def fit_metric(base, tier, key):  # measured batch-1: key = a + b*gen_tokens (per tier); robust to warm-up
    pts = []
    for f in glob.glob(J(os.path.join(base, tier, "ckpt_*lat*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get(key) is not None: pts.append((r.get("gen_tokens") or 0, r[key]))
    if len(pts) < 4: return None
    g = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    if g.std() < 1.0: return (float(np.median(y)), 0.0)  # near-constant gen (e.g. non-reasoning native think): use median, no slope (avoids ill-conditioned fit + out-of-range extrapolation)
    b, a = np.polyfit(g, y, 1)
    resid = y - (b * g + a); sd = resid.std()  # drop warm-up / GC spikes (CUDA-graph capture on 1st sample)
    keep = np.abs(resid) <= 2.5 * sd + 1e-9
    if keep.sum() >= 4: b, a = np.polyfit(g[keep], y[keep], 1)
    return (float(a), float(max(b, 0.0)))
TIERS = ["small_nt", "big_nt", "big_think"]
LAT = {t: (fit_metric(C["lat"], t, "latency_s") if C.get("lat") else None) for t in TIERS}
EN  = {t: (fit_metric(C["lat"], t, "energy_j") if C.get("lat") else None) for t in TIERS}
HAVE_LAT = all(LAT[t] for t in LAT); HAVE_EN = all(EN[t] for t in EN)
def tlat(t, g): a, b = LAT[t]; return np.clip(a + b * g, 0, None)
def ten(t, g): a, b = EN[t]; return np.clip(a + b * g, 0, None)
def load():
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c1, c2]): continue
        if cache is not None and ds in cache:
            cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
            idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
            Pc = np.array([cC[str(i)][0] for i in idx], float); Pf = np.array([cF[str(i)][0] for i in idx], float)
        else:
            idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]))
            Pc = np.full(len(idx), C.get("Pconst", 1024.0)); Pf = Pc.copy()
        if not idx: continue
        s0 = [signals_from_logprobs(c0[ds][i].get("opt_logprobs")) for i in idx]
        s1 = [signals_from_logprobs(c1[ds][i].get("opt_logprobs")) for i in idx]
        D[ds] = dict(
            ok0=np.array([c0[ds][i]["ok"] for i in idx], float), ok1=np.array([c1[ds][i]["ok"] for i in idx], float),
            ok2=np.array([c2[ds][i]["ok"] for i in idx], float),
            disagree=np.array([float(c0[ds][i]["pred"] != c1[ds][i]["pred"]) for i in idx]),
            stab=np.array([float(c0[ds][i]["pred"] == c2[ds][i]["pred"]) for i in idx]),
            verify=np.array([(ver.get(ds, {}).get(i, {}) or {}).get("p_yes_norm", 0.5) for i in idx], float),
            sig0={k: np.array([s[k] for s in s0], float) for k in s0[0]},
            sig1={k: np.array([s[k] for s in s1], float) for k in s1[0]},
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx], float), Pc=Pc, Pf=Pf)
    return D
def pool(D, names):
    names = [d for d in names if d in D]; out = {}
    for k in D[names[0]]:
        out[k] = ({s: np.concatenate([D[d][k][s] for d in names]) for s in D[names[0]][k]}
                  if k.startswith("sig") else np.concatenate([D[d][k] for d in names]))
    out["ds_of"] = np.concatenate([[d] * len(D[d]["ok0"]) for d in names]); return out
def feat(sig): return np.column_stack([sig[f] for f in FEATS])
def policy(name, P, cal):
    s0, s1 = P["sig0"], P["sig1"]
    if name == "Ours (ACC-v2: agreement)":   return -s0["margin"], P["disagree"] + 1e-6 * (-s1["margin"])
    if name == "ACC-v1 (margin)":            return -s0["margin"], -s1["margin"]
    if name == "MSP/Chow":                   return -s0["top1prob"], -s1["top1prob"]
    if name == "entropy":                    return s0["entropy"], s1["entropy"]
    if name == "Gini/DOCTOR":                return s0["gini"], s1["gini"]
    if name == "AutoMix (self-verify)":      return -P["verify"], -s1["margin"]
    if name == "random":
        rng = np.random.default_rng(0); n = len(P["ok0"]); return rng.standard_normal(n), rng.standard_normal(n)
    if name == "FrugalGPT-style learned":
        p0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok0"][cal]).predict_proba(feat(s0))[:, 1]
        p1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok1"][cal]).predict_proba(feat(s1))[:, 1]
        return -p0, -p1
    if name == "Jitkrittum L2D (Diff-Prob)":
        p0c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok0"][cal]).predict_proba(feat(s0))[:, 1]
        p0n = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok1"][cal]).predict_proba(feat(s0))[:, 1]
        p1c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok1"][cal]).predict_proba(feat(s1))[:, 1]
        p1n = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok2"][cal]).predict_proba(feat(s1))[:, 1]
        return (p0n - p0c), (p1n - p1c)
    if name == "CASP-Stability (trained)":
        pstab = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["stab"][cal]).predict_proba(feat(s0))[:, 1]
        return 1 - pstab, P["disagree"] + 1e-6 * (-s1["margin"])
    raise ValueError(name)
METHODS = ["Ours (ACC-v2: agreement)", "CASP-Stability (trained)", "ACC-v1 (margin)", "MSP/Chow",
           "entropy", "Gini/DOCTOR", "AutoMix (self-verify)", "FrugalGPT-style learned",
           "Jitkrittum L2D (Diff-Prob)", "random"]
def main():
    D = load()
    DUMP = {"family": A.family, "pools": {}}
    print(f"\n##########  ALL-METHODS 3-TIER BAKE-OFF: {A.family.upper()}  (FLOPs-calibrated, 20 seeds)  ##########")
    print(f"  config = small-nt -> big-nt -> big-think | esc0=%past-small, think=%reaching-big-think, guard=#bench worse-than-always-small")
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5)]:
        P = pool(D, names)
        f0 = 2 * N0 * (P["Pc"] + P["g0"]); f1 = 2 * N1 * (P["Pc"] + P["g1"]); f2 = 2 * N1 * (P["Pf"] + P["g2"]); F2 = f2.sum()
        l0v = tlat("small_nt", P["g0"]) if HAVE_LAT else np.zeros(len(P["g0"]))
        l1v = tlat("big_nt", P["g1"]) if HAVE_LAT else np.zeros(len(P["g1"]))
        l2v = tlat("big_think", P["g2"]) if HAVE_LAT else np.zeros(len(P["g2"]))
        e0v = ten("small_nt", P["g0"]) if HAVE_EN else np.zeros(len(P["g0"]))
        e1v = ten("big_nt", P["g1"]) if HAVE_EN else np.zeros(len(P["g1"]))
        e2v = ten("big_think", P["g2"]) if HAVE_EN else np.zeros(len(P["g2"]))
        parity = P["ok2"].mean()
        pb = lambda okf, d: okf[P["ds_of"] == d].mean()
        gd = lambda okf: sum(1 for d in names if pb(okf, d) < pb(P["ok0"], d) - 1e-9)
        rows = [("always-small-nt (cheap)", P["ok0"].mean(), 0., 0., f0.sum() / F2 * 100, l0v.mean(), e0v.mean(), 0, {d: pb(P["ok0"], d) for d in names}),
                ("always-big-nt", P["ok1"].mean(), 100., 0., f1.sum() / F2 * 100, l1v.mean(), e1v.mean(), gd(P["ok1"]), {d: pb(P["ok1"], d) for d in names}),
                ("always-big-think [PARITY]", P["ok2"].mean(), 100., 100., 100., l2v.mean(), e2v.mean(), gd(P["ok2"]), {d: pb(P["ok2"], d) for d in names})]
        res = {m: defaultdict(list) for m in METHODS}; resb = {m: {d: [] for d in names} for m in METHODS}
        for s in range(20):
            rng = np.random.default_rng(s); n = len(P["ok0"]); cal = np.zeros(n, bool)
            key = np.array([f"{d}{int(a)}{int(b)}" for d, a, b in zip(P["ds_of"], P["ok0"], P["ok2"])])
            for k in np.unique(key):
                ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix) // 2]] = True
            te = ~cal; tgt = P["ok2"][cal].mean(); dse = P["ds_of"][te]
            for m in METHODS:
                sc0, sc1 = policy(m, P, cal)
                q0 = np.quantile(sc0[cal], np.linspace(0, 1, 22)); q1 = np.quantile(sc1[cal], np.linspace(0, 1, 22)); best = None
                for t0 in q0:
                    e0m = sc0[cal] > t0
                    for t1 in q1:
                        e1m = e0m & (sc1[cal] > t1)
                        ok = np.where(~e0m, P["ok0"][cal], np.where(~e1m, P["ok1"][cal], P["ok2"][cal]))
                        if ok.mean() >= tgt - 1e-9:
                            fl = (f0[cal] + np.where(e0m, f1[cal], 0) + np.where(e1m, f2[cal], 0)).sum()
                            if best is None or fl < best[0]: best = (fl, t0, t1)
                t0b, t1b = (best[1], best[2]) if best else (q0[-1] + 1, q1[-1] + 1)
                E0 = sc0[te] > t0b; E1 = E0 & (sc1[te] > t1b)
                ok = np.where(~E0, P["ok0"][te], np.where(~E1, P["ok1"][te], P["ok2"][te]))
                fl = (f0[te] + np.where(E0, f1[te], 0) + np.where(E1, f2[te], 0)).sum() / f2[te].sum()
                lt = (l0v[te] + np.where(E0, l1v[te], 0) + np.where(E1, l2v[te], 0)).mean()
                en_ = (e0v[te] + np.where(E0, e1v[te], 0) + np.where(E1, e2v[te], 0)).mean()
                bad = sum(1 for d in names if (dse == d).sum() and ok[dse == d].mean() < P["ok0"][te][dse == d].mean() - 1e-9)
                R = res[m]; R["acc"].append(ok.mean()); R["esc0"].append(E0.mean()); R["esc2"].append(E1.mean()); R["fl"].append(fl); R["lat"].append(lt); R["en"].append(en_); R["bad"].append(bad)
                for d in names:
                    md = dse == d
                    if md.any(): resb[m][d].append(ok[md].mean())
        for m in METHODS:
            R = res[m]; mn = lambda k: float(np.mean(R[k]))
            rows.append((m, mn("acc"), mn("esc0") * 100, mn("esc2") * 100, mn("fl") * 100, mn("lat"), mn("en"), mn("bad"), {d: float(np.mean(resb[m][d])) for d in names}))
        print(f"\n  ====== [{label}]  parity(always-big-think)={parity:.4f} ======")
        lath = "lat(s)" if HAVE_LAT else "lat"; enh = "energy(J)" if HAVE_EN else "en"
        print(f"  {'method':<32}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{lath:>9}{enh:>11}{'guard':>7}")
        for nm, acc, e0, th, fl, lat, en_, g_, _ in rows:
            lats = (f"{lat:>8.2f}s" if HAVE_LAT else f"{'n/a':>9}")
            ens = (f"{en_:>9.1f}J" if HAVE_EN else f"{'n/a':>11}")
            print(f"  {nm:<32}{acc:>7.4f}{e0:>6.0f}%{th:>6.0f}%{fl:>7.1f}%{lats}{ens}{g_:>7.2f}")
        DUMP["pools"][label] = {"parity": float(parity), "rows": [
            {"method": nm, "acc": float(acc), "esc0": float(e0), "think": float(th), "flops": float(fl),
             "lat": float(lat), "energy": float(en_), "guard": float(g_), "bench": {d: float(bench[d]) for d in names}}
            for nm, acc, e0, th, fl, lat, en_, g_, bench in rows]}
        print(f"\n  --- per-benchmark accuracy [{label}] ---")
        print("  " + f"{'method':<32}" + "".join(f"{ABBR[d]:>7}" for d in names))
        for nm, *_, bench in rows:
            print("  " + f"{nm:<32}" + "".join(f"{bench[d]:>7.3f}" for d in names))
        ci = lambda R, k: (np.percentile(R[k], 2.5), np.percentile(R[k], 97.5))
        print(f"\n  --- 95% CI over 20 seeds [{label}] (parity acc={parity:.4f}) ---")
        for m in ["Ours (ACC-v2: agreement)", "ACC-v1 (margin)", "AutoMix (self-verify)"]:
            R = res[m]; a = ci(R, "acc"); f = ci(R, "fl"); g = ci(R, "bad")
            print(f"    {m:<30} acc[{a[0]:.4f},{a[1]:.4f}] FLOPs%[{f[0]*100:.1f},{f[1]*100:.1f}] guard[{g[0]:.0f},{g[1]:.0f}]")
    DUMP["lat_fits"] = {t: LAT[t] for t in TIERS}; DUMP["en_fits"] = {t: EN[t] for t in TIERS}
    json.dump(DUMP, open(J(f"results/cascade_methods/allmethods_{A.family}.json"), "w"), indent=1)
    print(f"\n[dump] results/cascade_methods/allmethods_{A.family}.json")
if __name__ == "__main__": main()
