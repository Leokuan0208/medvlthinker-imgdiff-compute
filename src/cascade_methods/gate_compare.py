#!/usr/bin/env python3
"""
gate_compare.py - hold the ACC 3-tier CONFIG fixed (7B-nothink@cap320 -> 32B-nothink@cap320 ->
32B-think@fullres) and swap in different cascade GATING METHODS to decide escalation at each tier.
This isolates the cascade-METHOD contribution from the config: if all gates tie, the win is the
structure (and ACC's plain margin is as good as anything); if a gate beats margin, that's a real
method improvement to fold in.

Gates compared (each produces a per-tier escalation score; higher = escalate):
  margin (ACC)      : -(top1-top2 logprob)              [the SOTA / deployed signal, what ACC uses]
  MSP / Chow        : -(max softmax prob)
  entropy           : +predictive entropy
  gini / DOCTOR     : +Gini impurity
  conformal(setsize): non-singleton prediction set at a calibrated alpha (CP-Router)
  learned-correct   : -P̂(this tier correct)  (logistic on the tier's signals; calib-fit)
  learned-defer(VADR): +[P̂(next-tier right) - P̂(this tier right)] (per-tier benefit; calib-fit, VADR-style)
  random            : no-signal baseline

All gated tiers (T0=7B-nt, T1=32B-nt) carry logprobs. Honest 50/50 calib/test; (tau0,tau1) chosen on
calib to reach calib-parity (always-32B-think acc) at MIN latency; 20 seeds. Reports acc, escalation
to the think tier, FLOPs%, latency, energy, guardrail. Latency/energy from real measured data
(reuses acc_compare.fit_models). CPU only; launch from repo root.
"""
import os, sys, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, signals_from_logprobs, CACHE, ALL6, ALL5, COMPETENT, N7, N32
from acc_compare import fit_models, predict

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]


def load():
    cache = json.load(open(J(CACHE)))
    c0 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
    c1 = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")
    c2 = _load_arm(J("ckpts/gate_32b"), "think_norag")
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c1, c2]): continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
        s0 = [signals_from_logprobs(c0[ds][i].get("opt_logprobs")) for i in idx]
        s1 = [signals_from_logprobs(c1[ds][i].get("opt_logprobs")) for i in idx]
        D[ds] = dict(
            ok0=np.array([c0[ds][i]["ok"] for i in idx], float),
            ok1=np.array([c1[ds][i]["ok"] for i in idx], float),
            ok2=np.array([c2[ds][i]["ok"] for i in idx], float),
            sig0={k: np.array([s[k] for s in s0], float) for k in s0[0]},
            sig1={k: np.array([s[k] for s in s1], float) for k in s1[0]},
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx], float),
            Pc=np.array([cC[str(i)][0] for i in idx], float), Pf=np.array([cF[str(i)][0] for i in idx], float),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {}
    for k in D[names[0]]:
        if k.startswith("sig"):
            out[k] = {s: np.concatenate([D[d][k][s] for d in names]) for s in D[names[0]][k]}
        else:
            out[k] = np.concatenate([D[d][k] for d in names])
    out["ds_of"] = np.concatenate([[d] * len(D[d]["ok0"]) for d in names]); out["names"] = names
    return out


# --- each gate returns (score0, score1) for ALL samples, fit on the calib mask where needed ---
def gate_scores(method, P, cal):
    s0, s1 = P["sig0"], P["sig1"]
    def feat(sig): return np.column_stack([sig[f] for f in FEATS])
    if method == "margin":      return -s0["margin"], -s1["margin"]
    if method == "MSP/Chow":    return -s0["top1prob"], -s1["top1prob"]
    if method == "entropy":     return s0["entropy"], s1["entropy"]
    if method == "gini":        return s0["gini"], s1["gini"]
    # NOTE: this "conformal" arm is a pure MSP COLLAPSE (1-top1prob, no prediction set) -> identical to
    # MSP/Chow. The FAITHFUL CP-Router (LAC set-size + FBE alpha*) is in baseline_compare.py and
    # over-escalates (69-80%); do not read this row as CP-Router. Kept only to show it ties MSP.
    if method == "conformal(=MSP-collapse)": return 1 - s0["top1prob"], 1 - s1["top1prob"]
    if method == "random":
        rng = np.random.default_rng(0); return rng.standard_normal(len(P["ok0"])), rng.standard_normal(len(P["ok0"]))
    if method == "learned-correct":
        p0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok0"][cal]).predict_proba(feat(s0))[:, 1]
        p1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok1"][cal]).predict_proba(feat(s1))[:, 1]
        return -p0, -p1
    if method == "learned-defer":   # VADR-style per-tier benefit: P(next right) - P(this right)
        # T0 benefit: best downstream (max ok1,ok2 reachable) vs ok0 ; approximate next-tier = ok1 for T0, ok2 for T1
        p0c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok0"][cal]).predict_proba(feat(s0))[:, 1]
        p0n = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s0)[cal], P["ok1"][cal]).predict_proba(feat(s0))[:, 1]
        p1c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok1"][cal]).predict_proba(feat(s1))[:, 1]
        p1n = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(feat(s1)[cal], P["ok2"][cal]).predict_proba(feat(s1))[:, 1]
        return (p0n - p0c), (p1n - p1c)
    raise ValueError(method)


def main():
    LAT, EN = fit_models()
    D = load()
    METHODS = ["margin", "MSP/Chow", "entropy", "gini", "conformal(=MSP-collapse)", "learned-correct", "learned-defer", "random"]
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        P = pool(D, names)
        f0 = 2 * N7 * (P["Pc"] + P["g0"]); f1 = 2 * N32 * (P["Pc"] + P["g1"]); f2 = 2 * N32 * (P["Pf"] + P["g2"])
        l0 = predict(LAT, "7b:nothink@cap320", P["g0"]); l1 = predict(LAT, "32b:nothink@cap320", P["g1"]); l2 = predict(LAT, "32b:think@fullres", P["g2"])
        e0 = predict(EN, "7b:nothink@cap320", P["g0"]); e1 = predict(EN, "32b:nothink@cap320", P["g1"]); e2 = predict(EN, "32b:think@fullres", P["g2"])
        parity = P["ok2"].mean(); res = {m: defaultdict(list) for m in METHODS}
        for s in range(20):
            rng = np.random.default_rng(s); n = len(P["ok0"]); cal = np.zeros(n, bool)
            key = np.array([f"{d}{int(a)}{int(b)}" for d, a, b in zip(P["ds_of"], P["ok0"], P["ok2"])])
            for k in np.unique(key):
                ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]] = True
            te = ~cal; tgt = P["ok2"][cal].mean()
            for m in METHODS:
                sc0, sc1 = gate_scores(m, P, cal)
                q0 = np.quantile(sc0[cal], np.linspace(0, 1, 22)); q1 = np.quantile(sc1[cal], np.linspace(0, 1, 22))
                best = None
                for t0 in q0:
                    e0m = sc0[cal] > t0
                    for t1 in q1:
                        e1m = e0m & (sc1[cal] > t1)
                        ok = np.where(~e0m, P["ok0"][cal], np.where(~e1m, P["ok1"][cal], P["ok2"][cal]))
                        if ok.mean() >= tgt - 1e-9:
                            lat = (l0[cal] + np.where(e0m, l1[cal], 0) + np.where(e1m, l2[cal], 0)).mean()
                            if best is None or lat < best[0]: best = (lat, t0, t1)
                t0b, t1b = (best[1], best[2]) if best else (q0[0], q1[0])
                E0 = sc0[te] > t0b; E1 = E0 & (sc1[te] > t1b)
                ok = np.where(~E0, P["ok0"][te], np.where(~E1, P["ok1"][te], P["ok2"][te]))
                fl = (f0[te] + np.where(E0, f1[te], 0) + np.where(E1, f2[te], 0)).sum() / f2[te].sum()
                lat = l0[te] + np.where(E0, l1[te], 0) + np.where(E1, l2[te], 0)
                eg = e0[te] + np.where(E0, e1[te], 0) + np.where(E1, e2[te], 0)
                bad = sum(1 for d in names if (P["ds_of"][te] == d).sum() and ok[P["ds_of"][te] == d].mean() < P["ok0"][te][P["ds_of"][te] == d].mean() - 1e-9)
                R = res[m]; R["acc"].append(ok.mean()); R["esc2"].append(E1.mean()); R["flops"].append(fl)
                R["lat"].append(lat.mean()); R["energy"].append(eg.mean()); R["bad"].append(bad)
        print(f"\n################  GATE METHODS @ FIXED 3-TIER CONFIG  [{label}]  (20 seeds)  ################")
        print(f"  config = 7B-nt@cap320 -> 32B-nt@cap320 -> 32B-think@fullres ; parity={parity:.4f}")
        print(f"  {'gate method':<18}{'acc':>7}{'esc→think':>11}{'FLOPs%':>8}{'lat':>8}{'energy':>9}{'guard':>7}")
        for m in METHODS:
            R = res[m]; mn = lambda k: np.mean(R[k])
            star = " <-ACC" if m == "margin" else ""
            print(f"  {m:<18}{mn('acc'):>7.4f}{mn('esc2')*100:>10.0f}%{mn('flops')*100:>7.1f}%{mn('lat'):>7.2f}s{mn('energy'):>8.0f}J{mn('bad'):>7.2f}{star}")


if __name__ == "__main__":
    main()
