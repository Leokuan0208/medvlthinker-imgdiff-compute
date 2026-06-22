#!/usr/bin/env python3
"""
latency_estimate.py - estimate REAL-TIME (batch-1) latency reduction of VADR vs the SOTA confidence
gate, using the live cascade's MEASURED per-sample latencies (ckpts/rt_cascade_cap320.jsonl):
  lat7_s   ~0.19s  (cheap 7B leg, every query)
  lat32_s  ~28s    (32B think leg, per escalated query; dominated by ~hundreds of SERIAL decode tokens)
So latency is dominated by escalation; the VADR verify pass is one extra ~lat7 cheap pass.

Method: calibrate lat32 ≈ α + β·gen32 on the live run's escalated samples; predict lat32 for any
sample from the full 32B gen_tokens (harness g32). Then, at each policy's pooled-parity operating
point, latency(q) = lat7 + [VADR]·verify(≈lat7) + [escalate]·lat32_hat. Report mean + p50/p90/p99 +
reduction, for ALL-6 and ALL-5. Energy (GPU joules) reported the same way as a bonus. CPU only.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, ALL5
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2",
         "gini", "n_opts", "cap_disagree", "cap_nuniq", "verify"]


def main():
    D = CascadeData("cap320")
    # attach self-verify
    for ds in D.ds_names:
        f = f"ckpts/gate_7b_verify/ckpt_{ds}_verify.jsonl"
        m = {json.loads(l)["idx"]: json.loads(l).get("p_yes_norm") for l in open(f)} if os.path.exists(f) else {}
        D.per_ds[ds]["sig"]["verify"] = np.array([m.get(int(i)) if m.get(int(i)) is not None else 0.5
                                                  for i in D.per_ds[ds]["idx"]], float)
    # live run real latencies/energy by (ds, idx)
    rt = {}
    for l in open("ckpts/rt_cascade_cap320.jsonl"):
        r = json.loads(l); rt[(r["dataset"], r["idx"])] = r
    # calibrate lat32 ≈ α + β·gen32 and e32 ≈ α'+β'·gen32 on ESCALATED live samples
    esc_rows = [r for r in rt.values() if r.get("escalate") and r.get("gen32", 0) > 0]
    g = np.array([[r["gen32"]] for r in esc_rows]); lat = np.array([r["lat32_s"] for r in esc_rows])
    en = np.array([r["gpu32_energy_j"] for r in esc_rows])
    lat_reg = LinearRegression().fit(g, lat); en_reg = LinearRegression().fit(g, en)
    print(f"lat32 ≈ {lat_reg.intercept_:.2f} + {lat_reg.coef_[0]:.4f}·gen32   (R²={lat_reg.score(g,lat):.3f}, "
          f"mean lat32={lat.mean():.1f}s over {len(esc_rows)} escalated)")
    lat7_mean = np.mean([r["lat7_s"] for r in rt.values()])
    e7_mean = np.mean([r["gpu7_energy_j"] for r in rt.values()])
    print(f"lat7≈{lat7_mean:.3f}s, 7B-energy≈{e7_mean:.0f}J per cheap pass (verify pass modeled as one more)\n")

    def per_pool(names):
        P = D.pool(names); P["ds_of"] = np.concatenate([[d] * len(D.per_ds[d]["a7"]) for d in names])
        idxs = np.concatenate([D.per_ds[d]["idx"] for d in names]); dsof = P["ds_of"]
        a7, a32 = P["a7"], P["a32"]; g32 = P["g32"]
        lat32_hat = lat_reg.predict(g32.reshape(-1, 1)); e32_hat = en_reg.predict(g32.reshape(-1, 1))
        lat7 = np.array([rt.get((d, int(i)), {}).get("lat7_s", lat7_mean) for d, i in zip(dsof, idxs)])
        e7 = np.array([rt.get((d, int(i)), {}).get("gpu7_energy_j", e7_mean) for d, i in zip(dsof, idxs)])
        # VADR OOF Δ score
        X = np.column_stack([P["sig"][f] for f in FEATS])
        p7 = np.zeros(len(a7)); p32 = np.zeros(len(a7)); strat = a7.astype(int) * 2 + a32.astype(int)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat):
            p7[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], a7[tr]).predict_proba(X[te])[:, 1]
            p32[te] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], a32[tr]).predict_proba(X[te])[:, 1]
        delta = p32 - p7
        target = a32.mean()
        def esc_set(score):  # escalate top-score until pooled parity
            o = np.argsort(-score, kind="stable")
            acc = a7.mean() + np.concatenate([[0], np.cumsum(a32[o] - a7[o])]) / len(a7)
            k = np.where(acc >= target - 1e-9)[0].min()
            e = np.zeros(len(a7), bool); e[o[:k]] = True; return e
        out = {}
        for nm, score, is_vadr in [("SOTA conf-gate", -P["sig"]["prob_margin"], False), ("VADR", delta, True)]:
            e = esc_set(score)
            lat = lat7 + (lat7 if is_vadr else 0.0) + np.where(e, lat32_hat, 0.0)
            ej = e7 + (e7 if is_vadr else 0.0) + np.where(e, e32_hat, 0.0)
            out[nm] = dict(esc=e.mean(), mean=lat.mean(), p50=np.percentile(lat, 50),
                           p90=np.percentile(lat, 90), p99=np.percentile(lat, 99),
                           energy=ej.mean(), tput=1.0 / lat.mean())
        return out

    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5)]:
        o = per_pool(names)
        print(f"################  REAL-TIME LATENCY  [{label}]  (batch-1, measured-calibrated)  ################")
        print(f"  {'policy':<16}{'esc%':>7}{'mean lat':>10}{'p50':>8}{'p90':>9}{'p99':>9}{'energy/q':>11}")
        for nm in ["SOTA conf-gate", "VADR"]:
            r = o[nm]
            print(f"  {nm:<16}{r['esc']*100:>6.0f}%{r['mean']:>9.2f}s{r['p50']:>7.2f}s{r['p90']:>8.1f}s"
                  f"{r['p99']:>8.1f}s{r['energy']:>9.0f}J")
        s, v = o["SOTA conf-gate"], o["VADR"]
        print(f"  -> VADR mean latency {(1-v['mean']/s['mean'])*100:+.0f}%  ({s['mean']-v['mean']:+.2f}s/query), "
              f"p90 {(1-v['p90']/s['p90'])*100:+.0f}%, throughput {(v['tput']/s['tput']-1)*100:+.0f}%, "
              f"energy {(1-v['energy']/s['energy'])*100:+.0f}%\n")
    print("Note: verify pass modeled as one extra ~lat7 cheap forward (conservative). 32B latency dominates,")
    print("so escalation-rate reduction -> latency reduction; the cheap verify pass barely registers.")


if __name__ == "__main__":
    main()
