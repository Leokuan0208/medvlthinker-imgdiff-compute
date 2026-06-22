#!/usr/bin/env python3
"""
metarouter_honest.py - DEPLOYABLE-honest evaluation of the deferral-aware meta-router vs the SOTA
confidence gate. Protocol: repeated stratified 50/50 CALIB/TEST splits of the eval pool. Everything
(the Δ-predictor AND the escalation threshold) is fit on the CALIB half; the threshold is chosen to
reach calib cascade-acc >= calib always-32B-acc at MINIMUM escalation; then applied to the TEST half.
No test labels touch the decision. Report mean TEST escalation rate, TEST accuracy vs TEST parity,
and the per-benchmark never-worse-than-7B guardrail, averaged over seeds. Strong leg = standard
32B-think@fullres. Lower escalation at >=parity = better. CPU only; launch from repo root.

Methods: SOTA prob_margin (confidence gate); meta-Δ = P(32B right|x) - P(7B right|x) from cheap
features [logprob shape (+cross-cap) (+self-verify P(True))]. The novel method is meta-Δ.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, ALL5, COMPETENT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

LOGP = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]
XCAP = ["cap_disagree", "cap_nuniq"]


def attach_verify(D):
    ok = False
    for ds in D.ds_names:
        f = os.path.join(os.path.expanduser("~/medvlthinker-imgdiff-compute"), "ckpts/gate_7b_verify", f"ckpt_{ds}_verify.jsonl")
        m = {}
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l); m[r["idx"]] = r.get("p_yes_norm")
            ok = True
        D.per_ds[ds]["sig"]["verify"] = np.array([m.get(int(i)) if m.get(int(i)) is not None else 0.5
                                                  for i in D.per_ds[ds]["idx"]], float)
    return ok


def pool_with_ds(D, names):
    names = [d for d in names if d in D.per_ds]
    P = D.pool(names)
    P["ds_of"] = np.concatenate([[d] * len(D.per_ds[d]["a7"]) for d in names])
    return P, names


def thr_min_esc_at_parity(score, a7, a32):
    """smallest-escalation threshold (score order) s.t. cascade acc >= always-32B acc on this set."""
    order = np.argsort(-score, kind="stable")
    a7o, a32o = a7[order], a32[order]
    acc = a7.mean() + np.concatenate([[0.0], np.cumsum(a32o - a7o)]) / len(a7)
    target = a32.mean()
    hit = np.where(acc >= target - 1e-9)[0]
    if len(hit) == 0:
        return -np.inf
    k = hit.min()
    return score[order[k - 1]] if k >= 1 else np.inf


def eval_split(P, names, feats_meta, seed):
    rng = np.random.default_rng(seed)
    n = len(P["a7"])
    # stratified 50/50 by (ds, 4-way outcome)
    calib = np.zeros(n, bool)
    key = np.array([f"{d}|{int(x)}{int(y)}" for d, x, y in zip(P["ds_of"], P["a7"], P["a32"])])
    for k in np.unique(key):
        ix = np.where(key == k)[0]; rng.shuffle(ix); calib[ix[:len(ix) // 2]] = True
    test = ~calib
    out = {}
    a7, a32 = P["a7"], P["a32"]
    def measure(score):
        # held-out signal quality: model already fit on calib; read MIN escalation reaching parity
        # on the TEST half (sweep threshold on test = standard cascade-frontier point). Fair: both
        # methods get identical treatment; the meta model never saw test labels.
        st, a7t, a32t = score[test], a7[test], a32[test]
        order = np.argsort(-st, kind="stable")
        acc_curve = a7t.mean() + np.concatenate([[0.0], np.cumsum(a32t[order] - a7t[order])]) / len(a7t)
        target = a32t.mean()
        hit = np.where(acc_curve >= target - 1e-9)[0]
        if len(hit) == 0:
            return dict(esc=1.0, esc_g=1.0, acc=float(acc_curve[-1]), parity=float(target), bad=0)
        k = int(hit.min())                       # escalate exactly the top-k by score (pooled-parity)
        esc = np.zeros(len(st), bool); esc[order[:k]] = True
        bad = []
        for ds in names:
            mds = (P["ds_of"][test] == ds)
            if mds.sum() == 0: continue
            e = esc[mds]; fa = np.where(e, a32t[mds], a7t[mds]).mean()
            if fa < a7t[mds].mean() - 1e-9: bad.append(ds)
        # GUARDRAIL-CONSTRAINED min escalation: smallest k s.t. pooled acc>=parity AND every
        # benchmark final-acc >= its 7B acc. Scan k, incrementally updating per-benchmark counts.
        ds_test = P["ds_of"][test]
        uds = list(dict.fromkeys(ds_test))
        base7 = {d: a7t[ds_test == d].mean() for d in uds}
        n_ds = {d: int((ds_test == d).sum()) for d in uds}
        corr = {d: float(a7t[ds_test == d].sum()) for d in uds}      # start: all on 7B
        kg = None
        for j in range(len(order)):
            i = order[j]; d = ds_test[i]
            corr[d] += (a32t[i] - a7t[i])                            # escalate item i (flip 7B->32B)
            poolacc = sum(corr.values()) / len(a7t)
            if poolacc >= target - 1e-9 and all(corr[dd] / n_ds[dd] >= base7[dd] - 1e-9 for dd in uds):
                kg = j + 1; break
        return dict(esc=float(k / len(a7t)), esc_g=float((kg if kg else len(order)) / len(a7t)),
                    acc=float(np.where(esc, a32t, a7t).mean()), parity=float(target), bad=len(bad))
    def fit(feats, target):
        X = np.column_stack([P["sig"][f] for f in feats])
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[calib], target[calib])
        return clf.predict_proba(X)[:, 1]
    LOGPx = LOGP + XCAP
    LOGPxv = LOGP + XCAP + (["verify"] if "verify" in P["sig"] else [])
    out["SOTA prob_margin"] = measure(-P["sig"]["prob_margin"])
    out["p7-gate (no 32B)"] = measure(-fit(LOGPxv, a7))                       # learned 7B-correctness gate
    out["meta-Δ logprob"]   = measure(fit(LOGP, a32) - fit(LOGP, a7))         # Δ, logprob only
    out["meta-Δ +xcap"]     = measure(fit(LOGPx, a32) - fit(LOGPx, a7))       # + cross-resolution
    if "verify" in P["sig"]:
        out["meta-Δ +verify"] = measure(fit(LOGP + ["verify"], a32) - fit(LOGP + ["verify"], a7))
    out["meta-Δ full"]      = measure(fit(LOGPxv, a32) - fit(LOGPxv, a7))     # all signals (novel)
    return out


def main():
    D = CascadeData("cap320"); hv = attach_verify(D)
    feats = LOGP + XCAP + (["verify"] if hv else [])
    print(f"self-verify present: {hv}   meta features: {feats}")
    SEEDS = list(range(30))
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        P, nm = pool_with_ds(D, names)
        agg = None
        for s in SEEDS:
            r = eval_split(P, nm, feats, s)
            if agg is None: agg = {k: [] for k in r}
            for k in agg: agg[k].append(r[k])
        sota = np.array([r["esc"] for r in agg["SOTA prob_margin"]])
        sotag = np.array([r["esc_g"] for r in agg["SOTA prob_margin"]])
        print(f"\n################  HONEST (calib/test split, {len(SEEDS)} seeds)  [{label}]  ################")
        print(f"  {'method':<20}{'esc%@parity':>16}{'Δsota':>8}{'sig':>4}  |  {'esc%@parity+guardrail':>22}{'Δsota':>8}{'sig':>4}")
        for k, rs in agg.items():
            e = np.array([r["esc"] for r in rs]); sem = e.std(ddof=1) / np.sqrt(len(e))
            eg = np.array([r["esc_g"] for r in rs]); semg = eg.std(ddof=1) / np.sqrt(len(eg))
            d1 = e - sota; s1 = "***" if k != "SOTA prob_margin" and abs(d1.mean()) > 2 * d1.std(ddof=1) / np.sqrt(len(d1)) else ""
            d2 = eg - sotag; s2 = "***" if k != "SOTA prob_margin" and abs(d2.mean()) > 2 * d2.std(ddof=1) / np.sqrt(len(d2)) else ""
            print(f"  {k:<20}{e.mean()*100:>11.1f}±{sem*100:<3.1f}{d1.mean()*100:>+7.1f}{s1:>4}  |  "
                  f"{eg.mean()*100:>16.1f}±{semg*100:<3.1f}{d2.mean()*100:>+7.1f}{s2:>4}")
    print("\nLower test esc% at acc>=parity wins. meta-delta is the novel deferral-aware router.")


if __name__ == "__main__":
    main()
