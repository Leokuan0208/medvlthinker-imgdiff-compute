#!/usr/bin/env python3
"""
compare.py - unified, fair leaderboard for cascade gating methods. For each method we report:
  (A) EVAL-ORACLE @ parity  : sweep threshold on eval, min backbone% s.t. eval acc >= always-32B
                              (upper bound = pure signal quality, ignores calibration transfer)
  (B) HONEST CALIBRATED     : threshold set on PMC-train (escalate the calib error-rate fraction,
                              the deployed gate's own rule), applied to eval. acc/esc/backbone +
                              a per-benchmark NEVER-WORSE-THAN-7B guardrail (the paper's claim).
Sorted by (A). CPU only; launch from repo root. This is the engine reused every research loop.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, COMPETENT
from frontier import curve_for_score, min_backbone_at_acc
from methods import registry

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")


def eval_one(D, method, names):
    method.fit(D)
    P = D.pool(names)
    parity = float(P["a32"].mean()); a7 = float(P["a7"].mean())
    escore = method.score_eval(P)
    # (A) eval-oracle @ parity
    cur = curve_for_score(P, escore)
    mbk = min_backbone_at_acc(cur, parity)
    # (B) honest calibrated: escalate calib error-rate fraction by this score's calib quantile
    err = 1.0 - D.calib["ok"].mean()
    thr = float(np.quantile(method.score_calib, 1.0 - err))
    esc = escore > thr
    m = D.score(P, esc)
    # per-benchmark guardrail at the calibrated threshold
    worse = []
    for ds in names:
        if ds not in D.per_ds:
            continue
        d = D.per_ds[ds]
        e = method.score_eval(D.pool([ds])) > thr
        final = np.where(e, d["a32"], d["a7"])
        if final.mean() < d["a7"].mean() - 1e-9:
            worse.append((ds, float(d["a7"].mean() - final.mean())))
    return dict(parity=parity, a7=a7, oracle=mbk, calib=m, calib_thr=thr,
                guardrail_ok=(len(worse) == 0), worse=worse)


def main():
    D = CascadeData("cap320")
    reg = registry()
    for sub_name, names in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
        P = D.pool(names)
        parity = float(P["a32"].mean())
        print(f"\n################  LEADERBOARD  [{sub_name}]  n={len(P['a7'])}  parity={parity:.4f}  ################")
        print(f"{'method':<22}{'(A) eval-oracle@parity':>24}{'   '}{'(B) honest calibrated':>34}")
        print(f"{'':22}{'esc%':>9}{'backbone%':>12}{'   '}{'esc%':>8}{'bbone%':>9}{'acc':>8}{'  par?':>6}{' >=7B?':>7}")
        results = {}
        for name, method in reg.items():
            try:
                r = eval_one(D, method, names)
            except Exception as e:
                print(f"{name:<22}  ERROR: {e}"); continue
            results[name] = r
        # sort by eval-oracle backbone (lower better); methods that never reach parity go last
        def keyf(kv):
            o = kv[1]["oracle"]
            return o["backbone"] if o else 9.0
        for name, r in sorted(results.items(), key=keyf):
            o = r["oracle"]; c = r["calib"]
            oa = f"{o['esc']*100:>8.1f}%{o['backbone']*100:>11.1f}%" if o else f"{'--':>9}{'never':>12}"
            par = "Y" if c["acc"] >= r["parity"] - 1e-9 else "n"
            g = "Y" if r["guardrail_ok"] else f"n({len(r['worse'])})"
            print(f"{name:<22}{oa}   {c['esc']*100:>7.1f}%{c['backbone']*100:>8.1f}%{c['acc']:>8.4f}{par:>6}{g:>7}")
        os.makedirs(os.path.join(REPO, "results/cascade_methods"), exist_ok=True)
        out = os.path.join(REPO, f"results/cascade_methods/leaderboard_{sub_name}.json")
        json.dump({k: {"oracle": v["oracle"], "calib": v["calib"], "guardrail_ok": v["guardrail_ok"],
                       "worse": v["worse"], "parity": v["parity"]} for k, v in results.items()},
                  open(out, "w"), indent=2, default=float)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
