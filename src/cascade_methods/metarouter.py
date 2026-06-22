#!/usr/bin/env python3
"""
metarouter.py - NOVEL deferral-aware "Unsolvable-aware" cascade router, evaluated honestly via
CV-pooled calibration (each fold's router fit on the other folds; all benchmarks present in MCQ
form with the 7B+32B labels we already have). Metric = min 32B-think escalation rate at iso-accuracy
(acc >= always-32B-think), ALL-6 and ALL-5; strong leg standard. Compares against the prob_margin
SOTA confidence gate under the SAME protocol.

Router idea: from cheap-only features [7B logprob shape + free cross-resolution agreement + one-pass
self-verification P(True)] predict P(7B right|x) and P(32B right|x); the escalation score is the
predicted ACCURACY GAIN Delta = P(32B right) - P(7B right). Escalating only high-Delta items skips
(a) futile items the 32B also fails and (b) items the 7B already gets right -> fewer escalations at
the same accuracy. Self-verify is the new orthogonal signal that may make Delta predictable on the
reasoning slice where confidence fails. CPU only; launch from repo root.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, ALL5, COMPETENT
from frontier import curve_for_score, min_backbone_at_acc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

VERIFY_DIR = "ckpts/gate_7b_verify"
LOGP = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "n_opts"]
XCAP = ["cap_disagree", "cap_nuniq"]


def load_verify(D):
    """attach self-verify p_yes_norm to each per_ds sig as 'verify' (0.5 if missing)."""
    have = False
    for ds in D.ds_names:
        m = {}
        f = os.path.join(os.path.expanduser("~/medvlthinker-imgdiff-compute"), VERIFY_DIR, f"ckpt_{ds}_verify.jsonl")
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l); m[r["idx"]] = r.get("p_yes_norm")
            have = True
        idx = D.per_ds[ds]["idx"]
        D.per_ds[ds]["sig"]["verify"] = np.array([m.get(int(i)) if m.get(int(i)) is not None else 0.5 for i in idx], float)
    return have


def _clf(kind):
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    return HistGradientBoostingClassifier(max_depth=3, max_iter=250, learning_rate=0.06,
                                          l2_regularization=1.0, early_stopping=True, random_state=42)


def oof_delta(P, feats, kind="logistic", nfold=5, seed=0):
    X = np.column_stack([P["sig"][f] for f in feats]); a7 = P["a7"]; a32 = P["a32"]
    p7 = np.zeros(len(a7)); p32 = np.zeros(len(a7))
    strat = a7.astype(int) * 2 + a32.astype(int)
    for tr, te in StratifiedKFold(nfold, shuffle=True, random_state=seed).split(X, strat):
        p7[te] = _clf(kind).fit(X[tr], a7[tr]).predict_proba(X[te])[:, 1]
        p32[te] = _clf(kind).fit(X[tr], a32[tr]).predict_proba(X[te])[:, 1]
    return p32 - p7, p7, p32


def guardrail_ok(D, names, score, thr):
    bad = []
    # need per-ds slices of the pooled score; recompute by ds offset
    off = 0
    for ds in names:
        n = len(D.per_ds[ds]["a7"]); s = score[off:off + n]; off += n
        d = D.per_ds[ds]; esc = s > thr
        if np.where(esc, d["a32"], d["a7"]).mean() < d["a7"].mean() - 1e-9:
            bad.append(ds)
    return bad


def report(D, names, label):
    P = D.pool(names); parity = float(P["a32"].mean())
    print(f"\n################  META-ROUTER  [{label}]  n={len(P['a7'])}  parity={parity:.4f}  ################")
    # SOTA baseline: prob_margin eval-oracle (single signal, threshold swept)
    mb = min_backbone_at_acc(curve_for_score(P, -P["sig"]["prob_margin"]), parity)
    print(f"  SOTA prob_margin (eval-oracle): esc={mb['esc']*100:.1f}%" if mb else "  prob_margin never")
    feats_sets = [
        ("logprob", LOGP),
        ("logprob+xcap", LOGP + XCAP),
        ("logprob+verify", LOGP + ["verify"]),
        ("logprob+xcap+verify", LOGP + XCAP + ["verify"]),
    ]
    for kind in ["logistic", "gbm"]:
        for fname, feats in feats_sets:
            feats = [f for f in feats if f in P["sig"]]
            delta, p7, p32 = oof_delta(P, feats, kind=kind)
            # min escalation at parity sweeping the OOF Delta score
            cur = curve_for_score(P, delta); m = min_backbone_at_acc(cur, parity)
            if m:
                # guardrail at that operating point: threshold = the delta value at rank k
                order = np.argsort(-delta, kind="stable"); thr = delta[order[m["k"] - 1]] if m["k"] >= 1 else np.inf
                bad = guardrail_ok(D, names, delta, thr)
                print(f"  meta[{kind:8s}|{fname:22s}] esc={m['esc']*100:5.1f}%  acc={m['acc']:.4f}  "
                      f"guardrail={'Y' if not bad else bad}")
            else:
                print(f"  meta[{kind:8s}|{fname:22s}] never reaches parity")


def main():
    D = CascadeData("cap320")
    hv = load_verify(D)
    print(f"self-verify present: {hv}")
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        report(D, names, label)
    print("\nGOAL: meta-router esc% < SOTA prob_margin eval-oracle (ALL-6 ~60%, ALL-5 ~35%) at parity, guardrail-clean.")


if __name__ == "__main__":
    main()
