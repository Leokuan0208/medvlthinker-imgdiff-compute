#!/usr/bin/env python3
"""
frontier.py - accuracy vs prefill-inclusive-compute frontiers for cascade escalation SCORES,
plus the oracle headroom. Built on harness.CascadeData. CPU only; launch from repo root.

A "score" is a per-sample escalation priority (HIGHER == escalate first == less confident).
For each score we sort samples by it and sweep the escalation count k=0..n; because escalating
any sample only ADDS 32B FLOPs, backbone% is monotone in k, so the cheapest way to reach a
target accuracy along a given ranking is the SMALLEST k that reaches it. We therefore report,
per score:
   - backbone% at the parity crossing  (min compute s.t. eval acc >= always-32B acc)   [headline]
   - the full (esc, backbone, acc) curve
and three oracle references:
   - ORACLE-maxacc : escalate by true benefit (a32-a7) desc -> upper accuracy envelope per cost
   - ORACLE-parity : cheapest escalation set (by 32B FLOPs) reaching parity acc  -> compute floor
   - RANDOM         : escalate a random fraction -> the no-signal line

This isolates SIGNAL QUALITY (oracle threshold on eval). The honest deployable number comes from
calib-set thresholds (see calibrated_operating_point). Both are reported.
"""
import json, os
import numpy as np
from harness import CascadeData, ALL6, COMPETENT, signals_from_logprobs   # noqa (same dir)

RNG = np.random.default_rng(0)


def curve_for_score(pool, score, include_vit=False):
    """Sort by score desc, sweep escalation count. Returns arrays over k=0..n."""
    a7, a32 = pool["a7"], pool["a32"]
    run7 = 2 * 7.6e9 * (pool["PY"] + pool["g7"])
    run32 = 2 * 33.0e9 * (pool["PF"] + pool["g32"])
    if include_vit:
        run7 = run7 + 2 * 0.675e9 * pool["visY"]
        run32 = run32 + 2 * 0.675e9 * pool["visF"]
    base = run32.sum()
    n = len(a7)
    order = np.argsort(-score, kind="stable")          # escalate highest score first
    a7o, a32o, r32o = a7[order], a32[order], run32[order]
    # cumulative as we escalate the first k (in order)
    correct_if_kept = a7.sum()                          # everyone kept on 7B
    # escalating sample j flips its contribution from a7 to a32 and adds r32 cost
    delta_acc = np.concatenate([[0.0], np.cumsum(a32o - a7o)]) / n
    acc = a7.mean() + delta_acc                          # length n+1, k=0..n
    cost = np.concatenate([[0.0], np.cumsum(r32o)])
    backbone = (run7.sum() + cost) / base
    esc = np.arange(n + 1) / n
    return dict(esc=esc, backbone=backbone, acc=acc, n=n)


def min_backbone_at_acc(curve, target):
    """Smallest-k (=> min backbone, since monotone) reaching acc>=target along this ranking."""
    ok = np.where(curve["acc"] >= target - 1e-9)[0]
    if len(ok) == 0:
        return None
    k = ok.min()
    return dict(k=int(k), esc=float(curve["esc"][k]), backbone=float(curve["backbone"][k]),
                acc=float(curve["acc"][k]))


def oracle_parity(pool, target, include_vit=False):
    """Cheapest escalation set (by 32B FLOPs) achieving acc>=target. Escalate only BENEFICIAL
    samples (a32>a7), cheapest-32B-first, until the accuracy gain reaches target. Harmful and
    neutral samples are never escalated (they cannot help). This is the compute floor at parity."""
    a7, a32 = pool["a7"], pool["a32"]
    run7 = 2 * 7.6e9 * (pool["PY"] + pool["g7"])
    run32 = 2 * 33.0e9 * (pool["PF"] + pool["g32"])
    if include_vit:
        run7 = run7 + 2 * 0.675e9 * pool["visY"]; run32 = run32 + 2 * 0.675e9 * pool["visF"]
    base = run32.sum(); n = len(a7)
    benefit = a32 - a7
    ben_idx = np.where(benefit > 0)[0]                  # 7B wrong, 32B right
    ben_idx = ben_idx[np.argsort(run32[ben_idx])]       # cheapest 32B first
    need = (target - a7.mean()) * n                      # number of net fixes needed
    k = int(np.ceil(max(need, 0)))
    k = min(k, len(ben_idx))
    esc = np.zeros(n, bool); esc[ben_idx[:k]] = True
    final = np.where(esc, a32, a7)
    backbone = (run7.sum() + run32[esc].sum()) / base
    return dict(esc=float(esc.mean()), backbone=float(backbone), acc=float(final.mean()),
                beneficial_total=int((benefit > 0).sum()), used=k)


def calibrated_operating_point(D, calib_score, eval_score, pool, esc_frac=None, include_vit=False):
    """Threshold a score on the CALIBRATION set (escalate the calib error-rate fraction by
    default, matching the margin gate's rule), then apply to eval. Honest, no eval peeking."""
    if esc_frac is None:
        esc_frac = 1.0 - D.calib["ok"].mean()           # escalate as many as 7B gets wrong on calib
    thr = float(np.quantile(calib_score, 1.0 - esc_frac))   # top esc_frac by score
    esc = eval_score > thr
    m = D.score(pool, esc, include_vit)
    m["esc_frac_target"] = float(esc_frac); m["thr"] = thr
    return m


# ---------------------------------------------------------------------------- driver
SIGNAL_SCORES = {   # name -> (signal_key, sign) ; escalation score = sign * signal (higher=escalate)
    "margin":       ("margin", -1.0),       # low logprob margin -> escalate
    "prob_margin":  ("prob_margin", -1.0),
    "margin_top2":  ("margin_top2", -1.0),
    "top1prob":     ("top1prob", -1.0),     # low max prob -> escalate (Chow / softmax response)
    "maxlogprob":   ("maxlogprob", -1.0),
    "entropy":      ("entropy", +1.0),      # high entropy -> escalate
    "entropy2":     ("entropy2", +1.0),
    "neg_energy":   ("neg_energy", +1.0),   # = -logsumexp ; higher -> escalate
    "gini":         ("gini", +1.0),
    "top2mass":     ("top2mass", -1.0),
}


def report(cap="cap320", subset=("ALL-6", ALL6)):
    D = CascadeData(cap)
    name, names = subset
    P = D.pool(names)
    parity = float(P["a32"].mean())
    a7 = float(P["a7"].mean())
    print(f"\n################  FRONTIER  [{name}]  cap={cap}  n={len(P['a7'])}  ################")
    print(f"always-7B acc={a7:.4f}   always-32B(parity) acc={parity:.4f}   gap={parity-a7:.4f}")
    floor = (2*7.6e9*(P['PY']+P['g7'])).sum() / (2*33.0e9*(P['PF']+P['g32'])).sum()
    print(f"cheap-leg floor (escalate none) backbone={floor*100:.1f}%   "
          f"always-32B baseline=100.0%  (margin-gate anchor ~73.6%)")

    orp = oracle_parity(P, parity)
    print(f"\nORACLE-parity (cheapest set reaching {parity:.4f}): "
          f"esc={orp['esc']*100:.1f}%  backbone={orp['backbone']*100:.1f}%  acc={orp['acc']:.4f}  "
          f"(beneficial pool={orp['beneficial_total']}, used={orp['used']})")

    # oracle-maxacc curve (benefit-ordered) for reference at the margin gate's escalation
    oc = curve_for_score(P, (P["a32"] - P["a7"]) + 1e-6 * RNG.standard_normal(len(P["a7"])))
    mb = min_backbone_at_acc(oc, parity)
    print(f"ORACLE-maxacc reaching parity: esc={mb['esc']*100:.1f}%  backbone={mb['backbone']*100:.1f}%")

    # random baseline (avg over draws): escalate random fraction -> acc rises linearly; find parity
    rand_scores = RNG.standard_normal(len(P["a7"]))
    rc = curve_for_score(P, rand_scores); rb = min_backbone_at_acc(rc, parity)
    print(f"RANDOM signal reaching parity: " + (f"esc={rb['esc']*100:.1f}%  backbone={rb['backbone']*100:.1f}%"
          if rb else "NEVER reaches parity (random can't exceed always-32B without escalating ~all)"))

    print(f"\n{'signal':<14}{'EVAL-oracle-thr @parity':>26}{'  ':>2}{'CALIBRATED (PMC-train thr)':>30}")
    print(f"{'':14}{'esc%':>8}{'backbone%':>12}{'acc':>6}   {'esc%':>8}{'backbone%':>12}{'acc':>7}{'  >=7B?':>8}")
    rows = []
    for sname, (key, sign) in SIGNAL_SCORES.items():
        escore = sign * P["sig"][key]
        cscore = sign * D.calib["sig"][key]
        cur = curve_for_score(P, escore)
        mbk = min_backbone_at_acc(cur, parity)
        cop = calibrated_operating_point(D, cscore, escore, P)
        # per-benchmark >=7B check at the calibrated point
        ge7 = "n/a"
        line_e = (f"{mbk['esc']*100:>7.1f}%{mbk['backbone']*100:>11.1f}%{mbk['acc']:>6.3f}"
                  if mbk else f"{'--':>8}{'never':>12}{'':>6}")
        print(f"{sname:<14}{line_e}   {cop['esc']*100:>7.1f}%{cop['backbone']*100:>11.1f}%{cop['acc']:>7.4f}")
        rows.append(dict(signal=sname, eval_parity=mbk, calibrated=cop))

    # save
    os.makedirs(os.path.join(os.path.expanduser('~/medvlthinker-imgdiff-compute'),
                             "results/cascade_methods"), exist_ok=True)
    out = os.path.join(os.path.expanduser('~/medvlthinker-imgdiff-compute'),
                       f"results/cascade_methods/frontier_{name}_{cap}.json")
    json.dump(dict(subset=name, cap=cap, parity=parity, always7=a7, floor=floor,
                   oracle_parity=orp, rows=rows), open(out, "w"), indent=2, default=float)
    print(f"\nsaved -> {out}")
    return D, P, parity


if __name__ == "__main__":
    for sub in [("ALL-6", ALL6), ("COMPETENT-4", COMPETENT)]:
        report("cap320", sub)
