#!/usr/bin/env python3
"""
evaluate.py - the DEFINITIVE cascade leaderboard. Runs every method (confidence baselines +
recoverability-aware deferral) under multiple honest protocols and the signal ceiling, with the
per-benchmark never-worse-than-7B guardrail. Requires 32B-on-PMC-train (ckpts/gate_32b_pmctrain/)
for the deferral methods and the proper calib-to-parity threshold. CPU only; launch from repo root.

Protocols (score s; higher = escalate):
  ORACLE      : threshold swept on eval -> min backbone% s.t. eval acc >= parity  (signal ceiling)
  ERR-RATE    : threshold = calib quantile at the calib error rate (the deployed gate's own rule)
  CALIB-PARITY: smallest-escalation threshold (in score order on the CALIBRATION set) that reaches
                calib-cascade-acc >= calib-always-32B-acc. Honest (calib-only), proper selector.
  DELTA>0     : (deferral Delta-scores only) escalate iff predicted accuracy gain > 0. Intrinsic
                operating point, nothing to transfer.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, COMPETENT
from frontier import curve_for_score, min_backbone_at_acc
from methods import registry as conf_registry
try:
    from methods_deferral import deferral_registry, RecoverabilityGate
except Exception:
    deferral_registry = lambda: {}

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")


def calib_parity_threshold(D, calib_score):
    """Smallest-escalation threshold (score order) reaching calib-cascade-acc >= calib-32B-acc."""
    ok = D.calib["ok"]; a32 = D.calib["a32"]
    order = np.argsort(-calib_score, kind="stable")
    a7o, a32o = ok[order], a32[order]
    acc = ok.mean() + np.concatenate([[0.0], np.cumsum(a32o - a7o)]) / len(ok)
    target = a32.mean()
    hit = np.where(acc >= target - 1e-9)[0]
    if len(hit) == 0:
        return -np.inf, 1.0          # escalate all
    k = hit.min()
    thr = calib_score[order[k - 1]] if k >= 1 else np.inf
    return float(thr), float(k / len(ok))


def guardrail(D, method, names, thr):
    worse = []
    for ds in names:
        if ds not in D.per_ds:
            continue
        d = D.per_ds[ds]
        e = method.score_eval(D.pool([ds])) > thr
        final = np.where(e, d["a32"], d["a7"])
        if final.mean() < d["a7"].mean() - 1e-9:
            worse.append(ds)
    return worse


def eval_method(D, method, names, is_delta=False):
    method.fit(D)
    P = D.pool(names); parity = float(P["a32"].mean())
    se = method.score_eval(P)
    out = {"parity": parity}
    # ORACLE
    out["oracle"] = min_backbone_at_acc(curve_for_score(P, se), parity)
    # ERR-RATE
    err = 1.0 - D.calib["ok"].mean()
    thr_e = float(np.quantile(method.score_calib, 1.0 - err))
    out["errrate"] = {**D.score(P, se > thr_e), "thr": thr_e, "worse": guardrail(D, method, names, thr_e)}
    # CALIB-PARITY (needs strong labels)
    if D.has_strong:
        thr_c, cesc = calib_parity_threshold(D, method.score_calib)
        out["calibparity"] = {**D.score(P, se > thr_c), "thr": thr_c, "calib_esc": cesc,
                              "worse": guardrail(D, method, names, thr_c)}
    # DELTA>0
    if is_delta:
        out["deltapos"] = {**D.score(P, se > 0.0), "thr": 0.0, "worse": guardrail(D, method, names, 0.0)}
    return out


def fmt(m, parity):
    if not m:
        return f"{'--':>30}"
    par = "Y" if m["acc"] >= parity - 1e-9 else "n"
    g = "Y" if not m["worse"] else f"n{len(m['worse'])}"
    return f"esc{m['esc']*100:>4.0f}% bb{m['backbone']*100:>5.1f}% a{m['acc']:.4f} p{par} g{g}"


def main():
    D = CascadeData("cap320")
    print(f"has_strong (32B-on-calib) = {D.has_strong}", end="")
    if hasattr(D, "_calib_strong_partial"):
        print(f"  [PARTIAL {D._calib_strong_partial[0]}/{D._calib_strong_partial[1]}]", end="")
    if D.has_strong:
        print(f"  calib: 7B acc={D.calib['ok'].mean():.3f} 32B acc={D.calib['a32'].mean():.3f} "
              f"recoverability={((D.calib['ok']==0)&(D.calib['a32']==1)).sum()/max((D.calib['ok']==0).sum(),1):.3f}")
    else:
        print("  (deferral methods + calib-parity unavailable until the run lands)")

    methods = {f"conf:{k.split(':')[-1]}": (m, False) for k, m in conf_registry().items()}
    if D.has_strong:
        for k, m in deferral_registry().items():
            methods[k] = (m, "recoverability" in k or "Delta" in k)  # Delta-scores support DELTA>0

    for sub_name, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = D.pool(names); parity = float(P["a32"].mean())
        print(f"\n################  {sub_name}  n={len(P['a7'])}  parity={parity:.4f}  ################")
        print(f"{'method':<26}{'ORACLE':>16}   {'ERR-RATE':>34}{'   '}{'CALIB-PARITY':>34}{'   '}{'DELTA>0':>34}")
        results = {}
        for name, (m, is_delta) in methods.items():
            try:
                results[name] = eval_method(D, m, names, is_delta)
            except Exception as e:
                print(f"{name:<26}  ERROR: {e}")
        def keyf(kv):
            cp = kv[1].get("calibparity") or kv[1].get("errrate")
            return cp["backbone"] if cp else 9.0
        for name, r in sorted(results.items(), key=keyf):
            o = r["oracle"]
            ostr = f"esc{o['esc']*100:>4.0f}% bb{o['backbone']*100:>5.1f}%" if o else f"{'never':>16}"
            print(f"{name:<26}{ostr:>16}   {fmt(r.get('errrate'),parity):>34}   "
                  f"{fmt(r.get('calibparity'),parity):>34}   {fmt(r.get('deltapos'),parity):>34}")
        os.makedirs(os.path.join(REPO, "results/cascade_methods"), exist_ok=True)
        json.dump(results, open(os.path.join(REPO, f"results/cascade_methods/evaluate_{sub_name}.json"), "w"),
                  indent=2, default=float)
    print("\nLegend: bb=backbone% (compute vs always-32B), p=hit parity?, g=never-worse-than-7B "
          "guardrail (Y / n<#failed>). Headline = lowest CALIB-PARITY bb with p=Y and g=Y.")


if __name__ == "__main__":
    main()
