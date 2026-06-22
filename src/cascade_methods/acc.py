#!/usr/bin/env python3
"""
acc.py - Adaptive-Compute Cascade (ACC): a confidence-gated cascade that routes each query to the
cheapest COMPUTE CONFIGURATION (model-size x reasoning-mode x resolution) predicted to suffice,
reserving the slow think leg only for the reasoning residual.

  Tier 0: 7B  no-think @ cap320      (gate tau0 on 7B margin)
  Tier 1: 32B no-think @ cap320      (gate tau1 on 32B-no-think margin; this leg HAS logprobs)
  Tier 2: 32B think    @ fullres     (terminal)

Baseline (SOTA confidence-gate cascade): 7B no-think@cap320 -> 32B think@fullres (skip Tier 1).
Both reach the SAME accuracy target = always-32B-think (parity). We compare prefill-inclusive FLOPs
AND real-time latency (calibrated on MEASURED batch-1 latencies) AND per-benchmark never-worse-than-7B.
Honest protocol: stratified 50/50 calib/test, thresholds chosen on calib (min latency s.t. calib
acc>=calib-parity), evaluated on test, averaged over seeds. CPU only; launch from repo root.
"""
import os, sys, json, glob, re
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, signals_from_logprobs, CACHE, ALL6, ALL5, COMPETENT, N7, N32

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)


def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0] - v[1]) if len(v) >= 2 else 0.0


def build_latency_model():
    """lat(config, gen_tok) in seconds, from measured batch-1 files + rt_cascade (think@fullres)."""
    model = {}
    for f in ["results/cascade_methods/latency_32b.jsonl", "results/cascade_methods/latency_7b.jsonl"]:
        if not os.path.exists(J(f)): continue
        rows = defaultdict(list)
        for l in open(J(f)):
            r = json.loads(l); rows[r["config"]].append(r)
        for c, rs in rows.items():
            g = np.array([r["gen_tok"] for r in rs]); lat = np.array([r["latency_s"] for r in rs])
            if g.max() - g.min() > 5 and len(rs) > 5:        # enough decode variation -> linear fit
                A = np.polyfit(g, lat, 1); model[c] = ("lin", A[0], A[1])
            else:
                model[c] = ("const", float(lat.mean()), 0.0)  # prefill-bound (no-think)
    # think@fullres: prefer rt_cascade (5440 samples) if measured think is sparse
    rt = [json.loads(l) for l in open(J("ckpts/rt_cascade_cap320.jsonl")) if l.strip()]
    g = np.array([r["gen32"] for r in rt if r.get("escalate") and r.get("gen32", 0) > 0])
    lat = np.array([r["lat32_s"] for r in rt if r.get("escalate") and r.get("gen32", 0) > 0])
    A = np.polyfit(g, lat, 1); model["think@fullres"] = ("lin", A[0], A[1])
    lat7 = np.mean([r["lat7_s"] for r in rt])
    model.setdefault("nothink@cap320_7b", ("const", lat7, 0.0))
    return model


def lat_of(model, config, gen, key=None):
    m = model.get(key or config)
    if m is None: m = model.get(config, ("const", 0.2, 0.0))
    return max(m[1] * gen + m[2] if m[0] == "lin" else m[1], 0.02)


def load():
    cache = json.load(open(J(CACHE)))
    ev7 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
    evN = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")
    evT = _load_arm(J("ckpts/gate_32b"), "think_norag")
    D = {}
    for ds in ALL6:
        if ds not in ev7 or ds not in evN or ds not in evT: continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(ev7[ds]) & set(evN[ds]) & set(evT[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
        D[ds] = dict(
            ok0=np.array([ev7[ds][i]["ok"] for i in idx], float),
            ok1=np.array([evN[ds][i]["ok"] for i in idx], float),
            ok2=np.array([evT[ds][i]["ok"] for i in idx], float),
            m0=np.array([margin(ev7[ds][i].get("opt_logprobs")) for i in idx], float),
            m1=np.array([margin(evN[ds][i].get("opt_logprobs")) for i in idx], float),
            g0=np.array([ev7[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g1=np.array([evN[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g2=np.array([evT[ds][i].get("gen_tokens") or 0 for i in idx], float),
            Pc=np.array([cC[str(i)][0] for i in idx], float),
            Pf=np.array([cF[str(i)][0] for i in idx], float),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"] = np.concatenate([[d] * len(D[d]["ok0"]) for d in names])
    return out


def main():
    LM = build_latency_model()
    print("latency model:", {k: (v[0], round(v[1], 4), round(v[2], 2)) for k, v in LM.items()})
    D = load()
    # per-query cost vectors (FLOPs and latency) for each tier
    def costs(P):
        run0 = 2 * N7 * (P["Pc"] + P["g0"]); run1 = 2 * N32 * (P["Pc"] + P["g1"]); run2 = 2 * N32 * (P["Pf"] + P["g2"])
        l0 = np.full(len(P["g0"]), lat_of(LM, "nothink@cap320", 2, "nothink@cap320"))   # 7B cap320 ~ const
        l0 = np.full(len(P["g0"]), LM.get("nothink@cap320_7b", ("const", 0.19, 0))[1])
        l1 = np.array([lat_of(LM, "nothink@cap320", g) for g in P["g1"]])
        l2 = np.array([lat_of(LM, "think@fullres", g) for g in P["g2"]])
        base = run2.sum()
        return run0, run1, run2, l0, l1, l2, base

    SEEDS = range(20)
    MARGIN = float(os.environ.get("ACC_MARGIN", "0.0"))  # calib acc target = parity + MARGIN (safety)
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        P = pool(D, names); run0, run1, run2, l0, l1, l2, base = costs(P)
        ok0, ok1, ok2, m0, m1 = P["ok0"], P["ok1"], P["ok2"], P["m0"], P["m1"]
        parity = ok2.mean()
        agg = defaultdict(lambda: defaultdict(list))
        for s in SEEDS:
            rng = np.random.default_rng(s); n = len(ok0); cal = np.zeros(n, bool)
            key = np.array([f"{d}{int(a)}{int(b)}{int(c)}" for d, a, b, c in zip(P["ds_of"], ok0, ok1, ok2)])
            for k in np.unique(key):
                ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix) // 2]] = True
            te = ~cal; tgt = ok2[cal].mean()
            q0 = np.quantile(m0[cal], np.linspace(0, 1, 26)); q1 = np.quantile(m1[cal], np.linspace(0, 1, 26))
            # ---- SOTA: 7B -> think, tau on m0 (min latency s.t. calib acc>=parity) ----
            best = None
            for t0 in q0:
                esc = m0[cal] < t0
                acc = np.where(esc, ok2[cal], ok0[cal]).mean()
                if acc >= tgt + MARGIN:
                    lat = (l0[cal] + np.where(esc, l2[cal], 0)).mean()
                    if best is None or lat < best[0]: best = (lat, t0)
            t0s = best[1] if best else q0[0]
            # ---- ACC: 7B -> 32B-nothink -> think, (tau0,tau1) min latency s.t. calib acc>=parity ----
            bestA = None
            for t0 in q0:
                e0 = m0[cal] < t0
                for t1 in q1:
                    e1 = e0 & (m1[cal] < t1)
                    final = np.where(~e0, ok0[cal], np.where(~e1, ok1[cal], ok2[cal]))
                    if final.mean() >= tgt + MARGIN:
                        lat = (l0[cal] + np.where(e0, l1[cal], 0) + np.where(e1, l2[cal], 0)).mean()
                        if bestA is None or lat < bestA[0]: bestA = (lat, t0, t1)
            t0a, t1a = (bestA[1], bestA[2]) if bestA else (q0[0], q1[0])

            def evalpol(which):
                if which == "SOTA":
                    e0 = m0[te] < t0s; e1 = np.zeros(te.sum(), bool)
                    final = np.where(e0, ok2[te], ok0[te])
                    fl = (run0[te] + np.where(e0, run2[te], 0)).sum() / base * (base / run2[te].sum())
                    lat = l0[te] + np.where(e0, l2[te], 0)
                    think = e0.mean()
                else:
                    e0 = m0[te] < t0a; e1 = e0 & (m1[te] < t1a)
                    final = np.where(~e0, ok0[te], np.where(~e1, ok1[te], ok2[te]))
                    fl = (run0[te] + np.where(e0, run1[te], 0) + np.where(e1, run2[te], 0)).sum() / run2[te].sum()
                    lat = l0[te] + np.where(e0, l1[te], 0) + np.where(e1, l2[te], 0)
                    think = e1.mean()
                # per-benchmark guardrail: final cascade acc vs tier-0 (7B) acc, per benchmark
                bad = 0
                for d in names:
                    md = P["ds_of"][te] == d
                    if md.sum() == 0: continue
                    if final[md].mean() < ok0[te][md].mean() - 1e-9: bad += 1
                return dict(acc=final.mean(), fl=fl, lat_mean=lat.mean(), lat_p90=np.percentile(lat, 90),
                            think=think, bad=bad, parity=ok2[te].mean())
            for w in ["SOTA", "ACC"]:
                r = evalpol(w)
                for k, v in r.items(): agg[w][k].append(v)
        print(f"\n################  ADAPTIVE-COMPUTE CASCADE  [{label}]  (honest 50/50, {len(list(SEEDS))} seeds)  ################")
        print(f"  parity target (always-32B-think) acc = {parity:.4f}")
        print(f"  {'policy':<8}{'acc':>8}{'FLOPs%':>9}{'lat mean':>10}{'lat p90':>9}{'think%':>8}{'guardrail':>10}")
        for w in ["SOTA", "ACC"]:
            a = agg[w]
            print(f"  {w:<8}{np.mean(a['acc']):>8.4f}{np.mean(a['fl'])*100:>8.1f}%{np.mean(a['lat_mean']):>9.2f}s"
                  f"{np.mean(a['lat_p90']):>8.1f}s{np.mean(a['think'])*100:>7.0f}%{np.mean(a['bad']):>10.2f}")
        S, Ac = agg["SOTA"], agg["ACC"]
        print(f"  -> ACC vs SOTA: latency {(1-np.mean(Ac['lat_mean'])/np.mean(S['lat_mean']))*100:+.0f}%  "
              f"FLOPs {(np.mean(Ac['fl'])-np.mean(S['fl']))*100:+.1f}pt  think-calls {(np.mean(Ac['think'])-np.mean(S['think']))*100:+.0f}pt  "
              f"acc {np.mean(Ac['acc'])-np.mean(S['acc']):+.4f}")


if __name__ == "__main__":
    main()
