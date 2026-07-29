#!/usr/bin/env python3
"""
acc_compare.py - head-to-head of three cascade methods on 5 metrics (accuracy, escalation rate,
latency, FLOPs, energy), honest held-out (50/50 calib/test, thresholds chosen on calib to reach
calib-parity = always-32B-think acc at MIN latency, 20 seeds). Latency & energy from REAL measured
batch-1 data (results/cascade_methods/artifacts/latency_*.jsonl + rt_cascade for 32B-think).

  M1 = ACC (ours)   : 7B-nothink@cap320 -> 32B-NOTHINK@cap320 -> 32B-think@fullres
  M2 = SOTA 2-tier  : 7B-THINK@fullres  -> 32B-think@fullres                  (both models reason)
  M3 = SOTA 3-tier  : 7B-nothink@cap320 -> 7B-THINK@fullres  -> 32B-think@fullres  (escalate reasoning, then size)

All tiers confidence-gated by the answer-letter margin (the SOTA gate). CPU only; launch from repo root.
"""
import os, sys, json, glob, re
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, CACHE, ALL6, ALL5, COMPETENT, N7, N32

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0


def load_think_shards(d, cell="think_norag"):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"*{cell}*.jsonl")):
        ds = re.match(r"ckpt_(.+?)_think", os.path.basename(f)).group(1)
        for l in open(f):
            if l.strip():
                try: r = json.loads(l); out[ds][r["idx"]] = r
                except Exception: pass
    return out


def fit_models():
    """lat(config,gen) and energy(config,gen), NAMESPACED by model ('7b:'/'32b:') because the config
    key 'think@fullres' collides across the 7B and 32B files. From measured batch-1 + rt_cascade."""
    lat, en = {}, {}
    files = [("7b", "results/cascade_methods/artifacts/latency_7b.jsonl"),
             ("7b", "results/cascade_methods/artifacts/latency_7b_think.jsonl"),
             ("32b", "results/cascade_methods/artifacts/latency_32b.jsonl")]
    rows = defaultdict(list)
    for model, f in files:
        if os.path.exists(J(f)):
            for l in open(J(f)):
                r = json.loads(l); rows[f"{model}:{r['config']}"].append(r)
    for c, rs in rows.items():
        g = np.array([r["gen_tok"] for r in rs], float); L = np.array([r["latency_s"] for r in rs], float)
        E = np.array([r["energy_j"] for r in rs], float)
        if g.max() - g.min() > 5 and len(rs) >= 6:
            lat[c] = np.polyfit(g, L, 1); en[c] = np.polyfit(g, E, 1)
        else:
            lat[c] = np.array([0.0, float(L.mean())]); en[c] = np.array([0.0, float(E.mean())])
    # 32B-think from rt_cascade (large sample) -> overrides
    rt = [json.loads(l) for l in open(J("ckpts/rt_cascade_cap320.jsonl")) if l.strip()]
    g = np.array([r["gen32"] for r in rt if r.get("escalate") and r.get("gen32", 0) > 0], float)
    L = np.array([r["lat32_s"] for r in rt if r.get("escalate") and r.get("gen32", 0) > 0], float)
    E = np.array([r["gpu32_energy_j"] for r in rt if r.get("escalate") and r.get("gen32", 0) > 0], float)
    lat["32b:think@fullres"] = np.polyfit(g, L, 1); en["32b:think@fullres"] = np.polyfit(g, E, 1)
    return lat, en


def predict(model, config, gen):
    a = model.get(config)
    if a is None: a = np.array([0.0, 0.2])
    return np.maximum(a[0] * gen + a[1], 0.01)


def load_all():
    cache = json.load(open(J(CACHE)))
    c0 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")     # 7B nothink cap320
    c7t = load_think_shards(J("ckpts/gate_7b_think"))                    # 7B think fullres
    c32n = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")  # 32B nothink cap320
    c32t = _load_arm(J("ckpts/gate_32b"), "think_norag")                 # 32B think fullres
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c7t, c32n, c32t]): continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds]) & set(c7t[ds]) & set(c32n[ds]) & set(c32t[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
        def arr(src, key, default=0): return np.array([src[ds][i].get(key, default) for i in idx], float)
        D[ds] = dict(
            ok_0=arr(c0, "ok"), ok_7t=arr(c7t, "ok"), ok_32n=arr(c32n, "ok"), ok_32t=arr(c32t, "ok"),
            disagree=np.array([1.0 if c0[ds][i].get("pred") != c32n[ds][i].get("pred") else 0.0 for i in idx], float),
            m_0=np.array([margin(c0[ds][i].get("opt_logprobs")) for i in idx]),
            m_7t=np.array([margin(c7t[ds][i].get("opt_logprobs")) for i in idx]),
            m_32n=np.array([margin(c32n[ds][i].get("opt_logprobs")) for i in idx]),
            g_0=arr(c0, "gen_tokens", 2), g_7t=arr(c7t, "gen_tokens", 2),
            g_32n=arr(c32n, "gen_tokens", 2), g_32t=arr(c32t, "gen_tokens", 2),
            Pc=np.array([cC[str(i)][0] for i in idx]), Pf=np.array([cF[str(i)][0] for i in idx]),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"] = np.concatenate([[d] * len(D[d]["ok_0"]) for d in names]); out["names"] = names
    return out


def main():
    LAT, EN = fit_models()
    print("latency model (s):", {k: f"{v[0]:.4f}*g+{v[1]:.2f}" for k, v in LAT.items()})
    D = load_all()
    SEEDS = range(20)
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        P = pool(D, names)
        # per-tier per-query cost vectors
        flop = lambda Nb, Pt, g: 2 * Nb * (Pt + g)
        f_0 = flop(N7, P["Pc"], P["g_0"]); f_7t = flop(N7, P["Pf"], P["g_7t"])
        f_32n = flop(N32, P["Pc"], P["g_32n"]); f_32t = flop(N32, P["Pf"], P["g_32t"])
        base = f_32t.sum()
        l_0 = predict(LAT, "7b:nothink@cap320", P["g_0"]); l_7t = predict(LAT, "7b:think@fullres", P["g_7t"])
        l_32n = predict(LAT, "32b:nothink@cap320", P["g_32n"]); l_32t = predict(LAT, "32b:think@fullres", P["g_32t"])
        e_0 = predict(EN, "7b:nothink@cap320", P["g_0"]); e_7t = predict(EN, "7b:think@fullres", P["g_7t"])
        e_32n = predict(EN, "32b:nothink@cap320", P["g_32n"]); e_32t = predict(EN, "32b:think@fullres", P["g_32t"])
        parity = P["ok_32t"].mean()

        def routed(method, t0, t1):
            # returns final ok, flops, latency, energy, esc-to-32Bthink mask, esc-any mask (per query)
            if method == "M2":   # 7B-think -> 32B-think
                e = P["m_7t"] < t0
                ok = np.where(e, P["ok_32t"], P["ok_7t"])
                fl = f_7t + np.where(e, f_32t, 0); lt = l_7t + np.where(e, l_32t, 0); eg = e_7t + np.where(e, e_32t, 0)
                return ok, fl, lt, eg, e, e
            if method in ("M1", "M1b"):   # 7B-nothink -> 32B-nothink -> 32B-think
                e0 = P["m_0"] < t0
                if method == "M1":
                    e1 = e0 & (P["m_32n"] < t1)                 # think gate = 32B-nothink margin
                else:
                    s1 = P["disagree"] + 1e-6 * (-P["m_32n"])   # think gate = 7B/32B no-think DISAGREEMENT
                    e1 = e0 & (s1 > t1)
                ok = np.where(~e0, P["ok_0"], np.where(~e1, P["ok_32n"], P["ok_32t"]))
                fl = f_0 + np.where(e0, f_32n, 0) + np.where(e1, f_32t, 0)
                lt = l_0 + np.where(e0, l_32n, 0) + np.where(e1, l_32t, 0)
                eg = e_0 + np.where(e0, e_32n, 0) + np.where(e1, e_32t, 0)
                return ok, fl, lt, eg, e1, e0
            # M3: 7B-nothink -> 7B-think -> 32B-think
            e0 = P["m_0"] < t0; e1 = e0 & (P["m_7t"] < t1)
            ok = np.where(~e0, P["ok_0"], np.where(~e1, P["ok_7t"], P["ok_32t"]))
            fl = f_0 + np.where(e0, f_7t, 0) + np.where(e1, f_32t, 0)
            lt = l_0 + np.where(e0, l_7t, 0) + np.where(e1, l_32t, 0)
            eg = e_0 + np.where(e0, e_7t, 0) + np.where(e1, e_32t, 0)
            return ok, fl, lt, eg, e1, e0

        res = {m: defaultdict(list) for m in ["M1", "M1b", "M2", "M3"]}
        s_dis = P["disagree"] + 1e-6 * (-P["m_32n"])
        for s in SEEDS:
            rng = np.random.default_rng(s); n = len(P["ok_0"]); cal = np.zeros(n, bool)
            key = np.array([f"{d}{int(a)}{int(b)}" for d, a, b in zip(P["ds_of"], P["ok_0"], P["ok_32t"])])
            for k in np.unique(key):
                ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]] = True
            te = ~cal; tgt = P["ok_32t"][cal].mean()
            q0 = np.quantile(P["m_0"][cal], np.linspace(0, 1, 22)); q7 = np.quantile(P["m_7t"][cal], np.linspace(0, 1, 22))
            q32 = np.quantile(P["m_32n"][cal], np.linspace(0, 1, 22)); qd = np.quantile(s_dis[cal], np.linspace(0, 1, 22))
            grids = {"M1": (q0, q32), "M1b": (q0, qd), "M2": (q7, [0]), "M3": (q0, q7)}
            for m in ["M1", "M1b", "M2", "M3"]:
                g0g, g1g = grids[m]; best = None
                for t0 in g0g:
                    for t1 in g1g:
                        ok, fl, lt, eg, _, _ = routed(m, t0, t1)
                        if ok[cal].mean() >= tgt - 1e-9:
                            ml = lt[cal].mean()
                            if best is None or ml < best[0]: best = (ml, t0, t1)
                t0b, t1b = (best[1], best[2]) if best else (g0g[0], g1g[0])
                ok, fl, lt, eg, e_exp, e_any = routed(m, t0b, t1b)
                # per-benchmark guardrail (final vs 7B-nothink tier0 acc)
                bad = 0
                for d in names:
                    md = (P["ds_of"][te] == d)
                    if md.sum() and ok[te][md].mean() < P["ok_0"][te][md].mean() - 1e-9: bad += 1
                R = res[m]
                R["acc"].append(ok[te].mean()); R["flops"].append(fl[te].sum() / f_32t[te].sum())
                R["lat"].append(lt[te].mean()); R["latp90"].append(np.percentile(lt[te], 90))
                R["energy"].append(eg[te].mean()); R["esc32think"].append(e_exp[te].mean())
                R["escany"].append(e_any[te].mean()); R["bad"].append(bad)
        print(f"\n################  3-METHOD COMPARISON  [{label}]  (honest 50/50, 20 seeds)  ################")
        print(f"  parity target (always-32B-think) = {parity:.4f}")
        print(f"  {'method':<22}{'acc':>7}{'esc→32Bthink':>14}{'esc-any':>9}{'FLOPs%':>8}{'lat mean':>10}{'lat p90':>9}{'energy/q':>10}{'guard':>7}")
        labelmap = {"M1": "M1 ACC (margin)", "M1b": "M1b ACC+agree(ours)",
                    "M2": "M2 SOTA 2-tier(think)", "M3": "M3 SOTA 3-tier"}
        for m in ["M2", "M3", "M1", "M1b"]:
            R = res[m]; mn = lambda k: np.mean(R[k])
            print(f"  {labelmap[m]:<22}{mn('acc'):>7.4f}{mn('esc32think')*100:>12.0f}% {mn('escany')*100:>7.0f}%"
                  f"{mn('flops')*100:>7.1f}%{mn('lat'):>9.2f}s{mn('latp90'):>8.1f}s{mn('energy'):>8.0f}J{mn('bad'):>7.2f}")


if __name__ == "__main__":
    main()
