#!/usr/bin/env python3
"""
final_3tier_comparison.py - COMPLETE method comparison under the FIXED 3-tier ACC config
(7B-nothink@cap320 -> 32B-nothink@cap320 -> 32B-think@fullres), pools ALL-6 and ALL-5 (excl MedXpert)
ONLY (competent-4 dropped per request). Each method = a (tier-0 stop, tier-1 think) gating policy;
calibrated jointly at parity / min-latency, honest 50/50 stratified split, 20 seeds.

Methods (post-audit, faithful):
  Ours (ACC-v2: agreement)   = margin stop + cross-model-agreement think-gate   <- OUR METHOD
  ACC-v1 (margin)            = margin stop + margin think
  MSP/Chow, entropy, Gini/DOCTOR (confidence)
  AutoMix (self-verify)      = 7B self-verify P(True) stop  [tier-1 falls back to 32B-nt margin; verify is 0-shot]
  FrugalGPT-style learned    = -P(this-tier correct), calib-fit
  Jitkrittum L2D (Diff-Prob) = P(next right)-P(this right), calib-fit
  CASP-Stability (trained) = 1-P(pred survives more compute) stop + agreement think  [calib-fit]
  random
  CP-Router (faithful set-size+FBE): reported separately (its |C|!=1 rule is non-monotone; see baseline_compare.py)
Run from repo root:  python3 src/cascade_methods/final_3tier_comparison.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs, ALL6, ALL5
from acc_compare import fit_models, predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
N7, N32 = 7.6e9, 33.0e9
J = lambda p: os.path.join("/home/jamesyang/medvlthinker-imgdiff-compute", p)
FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]
def loadarm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(J(os.path.join(d, f"ckpt_*{tag}*.jsonl"))):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if m:
            for l in open(f):
                if l.strip(): r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out

c0 = loadarm("ckpts/gate_7b_prune/cap320", "nothink_norag")
c1 = loadarm("ckpts/gate_32b_modes/nothink_cap320", "nothink_norag")
c2 = loadarm("ckpts/gate_32b", "think_norag")
ver = loadarm("ckpts/gate_7b_verify", "verify")
cache = json.load(open(J("ckpts/token_cache.json")))
def load():
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c1, c2]): continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
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
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx], float),
            Pc=np.array([cC[str(i)][0] for i in idx], float), Pf=np.array([cF[str(i)][0] for i in idx], float))
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
    if name == "Ours (ACC-v2: agreement)":   return -s0["margin"], P["disagree"] + 1e-6 * (-s1["margin"])  # eps tiebreak -> low-margin disagreements
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
    if name == "CASP-Stability (trained)":   # stability stop + agreement think
        pstab = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["stab"][cal]).predict_proba(feat(s0))[:, 1]
        return 1 - pstab, P["disagree"] + 1e-6 * (-s1["margin"])
    raise ValueError(name)

METHODS = ["Ours (ACC-v2: agreement)", "CASP-Stability (trained)", "ACC-v1 (margin)", "MSP/Chow",
           "entropy", "Gini/DOCTOR", "AutoMix (self-verify)", "FrugalGPT-style learned",
           "Jitkrittum L2D (Diff-Prob)", "random"]
ABBR = {"PMC-VQA": "PMC", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD", "PathVQA": "PathV", "MMMU": "MMMU",
        "MedXpert-Reasoning": "MX-R", "MedXpert-Understanding": "MX-U"}
def main():
    LAT, EN = fit_models(); D = load()
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5)]:
        P = pool(D, names)
        f0 = 2 * N7 * (P["Pc"] + P["g0"]); f1 = 2 * N32 * (P["Pc"] + P["g1"]); f2 = 2 * N32 * (P["Pf"] + P["g2"]); F2 = f2.sum()
        l0 = predict(LAT, "7b:nothink@cap320", P["g0"]); l1 = predict(LAT, "32b:nothink@cap320", P["g1"]); l2 = predict(LAT, "32b:think@fullres", P["g2"])
        parity = P["ok2"].mean()
        pb = lambda okf, d: okf[P["ds_of"] == d].mean()
        gd = lambda okf: sum(1 for d in names if pb(okf, d) < pb(P["ok0"], d) - 1e-9)
        # rows: (name, acc, esc0%, think%, FLOPs%, lat, guard, {bench:acc})  -- baselines first
        rows = [("always-7B-nt@cap320 (cheap)", P["ok0"].mean(), 0., 0., f0.sum() / F2 * 100, l0.mean(), 0, {d: pb(P["ok0"], d) for d in names}),
                ("always-32B-nt@cap320", P["ok1"].mean(), 100., 0., f1.sum() / F2 * 100, l1.mean(), gd(P["ok1"]), {d: pb(P["ok1"], d) for d in names}),
                ("always-32B-think@full [PARITY]", P["ok2"].mean(), 100., 100., 100., l2.mean(), gd(P["ok2"]), {d: pb(P["ok2"], d) for d in names})]
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
                            lat = (l0[cal] + np.where(e0m, l1[cal], 0) + np.where(e1m, l2[cal], 0)).mean()
                            if best is None or lat < best[0]: best = (lat, t0, t1)
                t0b, t1b = (best[1], best[2]) if best else (q0[-1] + 1, q1[-1] + 1)
                E0 = sc0[te] > t0b; E1 = E0 & (sc1[te] > t1b)
                ok = np.where(~E0, P["ok0"][te], np.where(~E1, P["ok1"][te], P["ok2"][te]))
                fl = (f0[te] + np.where(E0, f1[te], 0) + np.where(E1, f2[te], 0)).sum() / f2[te].sum()
                lat = (l0[te] + np.where(E0, l1[te], 0) + np.where(E1, l2[te], 0)).mean()
                bad = sum(1 for d in names if (dse == d).sum() and ok[dse == d].mean() < P["ok0"][te][dse == d].mean() - 1e-9)
                R = res[m]; R["acc"].append(ok.mean()); R["esc0"].append(E0.mean()); R["esc2"].append(E1.mean()); R["fl"].append(fl); R["lat"].append(lat); R["bad"].append(bad)
                for d in names:
                    md = dse == d
                    if md.any(): resb[m][d].append(ok[md].mean())
        for m in METHODS:
            R = res[m]; mn = lambda k: float(np.mean(R[k]))
            rows.append((m, mn("acc"), mn("esc0") * 100, mn("esc2") * 100, mn("fl") * 100, mn("lat"), mn("bad"), {d: float(np.mean(resb[m][d])) for d in names}))
        print(f"\n############  COMPLETE COMPARISON  [{label}]  (3-tier config, 20 seeds, parity={parity:.4f})  ############")
        print(f"  config = 7B-nt@cap320 -> 32B-nt@cap320 -> 32B-think@fullres | esc0=%past-7B, think=%reaching-32B-think, guard=#benchmarks worse-than-always-7B")
        print(f"  {'method':<32}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{'lat':>8}{'guard':>7}")
        for nm, acc, e0, th, fl, lat, g_, _ in rows:
            print(f"  {nm:<32}{acc:>7.4f}{e0:>6.0f}%{th:>6.0f}%{fl:>7.1f}%{lat:>7.2f}s{g_:>7.2f}")
        print(f"\n  --- per-benchmark accuracy [{label}] ---")
        print("  " + f"{'method':<32}" + "".join(f"{ABBR[d]:>7}" for d in names))
        for nm, *_, bench in rows:
            print("  " + f"{nm:<32}" + "".join(f"{bench[d]:>7.3f}" for d in names))
        ci = lambda R, k: (np.percentile(R[k], 2.5), np.percentile(R[k], 97.5))
        print(f"\n  --- 95% CI over 20 seeds [{label}] (parity acc={parity:.4f}) ---")
        for m in ["Ours (ACC-v2: agreement)", "ACC-v1 (margin)", "AutoMix (self-verify)"]:
            R = res[m]; a = ci(R, "acc"); f = ci(R, "fl"); l = ci(R, "lat"); g = ci(R, "bad")
            print(f"    {m:<30} acc[{a[0]:.4f},{a[1]:.4f}] FLOPs%[{f[0]*100:.1f},{f[1]*100:.1f}] lat[{l[0]:.2f},{l[1]:.2f}]s guard[{g[0]:.0f},{g[1]:.0f}]")
if __name__ == "__main__": main()
