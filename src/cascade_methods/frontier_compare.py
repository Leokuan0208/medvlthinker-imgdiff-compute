#!/usr/bin/env python3
"""
frontier_compare.py - DEFINITIVE escalation-accuracy frontier: novel Verification-Augmented Deferral
Router (meta-Δ) vs SOTA confidence gate (prob_margin), POOLED accuracy metric (the user's binding
constraint). Held-out: model fit on a stratified calib half, frontier read on the test half, averaged
over seeds. Reports (a) escalation at iso-accuracy (parity = always-32B-think) and (b) accuracy at
fixed escalation budgets, to show frontier dominance. ALL-6 and ALL-5 (excl MedXpert). CPU only.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CascadeData, ALL6, ALL5, COMPETENT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATS = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2",
         "gini", "n_opts", "cap_disagree", "cap_nuniq", "verify"]


def attach_verify(D):
    for ds in D.ds_names:
        f = os.path.join(os.path.expanduser("~/medvlthinker-imgdiff-compute"), "ckpts/gate_7b_verify", f"ckpt_{ds}_verify.jsonl")
        m = {}
        if os.path.exists(f):
            for l in open(f):
                if l.strip():
                    r = json.loads(l); m[r["idx"]] = r.get("p_yes_norm")
        D.per_ds[ds]["sig"]["verify"] = np.array([m.get(int(i)) if m.get(int(i)) is not None else 0.5
                                                  for i in D.per_ds[ds]["idx"]], float)


def curve(score, a7t, a32t):
    o = np.argsort(-score, kind="stable")
    acc = a7t.mean() + np.concatenate([[0.0], np.cumsum(a32t[o] - a7t[o])]) / len(a7t)
    esc = np.arange(len(a7t) + 1) / len(a7t)
    return esc, acc


def min_esc_at(esc, acc, target):
    hit = np.where(acc >= target - 1e-9)[0]
    return esc[hit.min()] if len(hit) else 1.0


def acc_at_esc(esc, acc, budget):
    k = int(round(budget * (len(acc) - 1)))
    return acc[k]


def run(D, names, seeds=30):
    P = D.pool(names); P["ds_of"] = np.concatenate([[d] * len(D.per_ds[d]["a7"]) for d in names])
    a7, a32 = P["a7"], P["a32"]; X = np.column_stack([P["sig"][f] for f in FEATS])
    parity = float(a32.mean())
    budgets = [0.1, 0.2, 0.3, 0.4, 0.5]
    res = {m: {"esc_par": [], "acc_at": {b: [] for b in budgets}} for m in ["SOTA", "meta-Δ"]}
    for s in range(seeds):
        rng = np.random.default_rng(s); n = len(a7); cal = np.zeros(n, bool)
        key = np.array([f"{d}{int(x)}{int(y)}" for d, x, y in zip(P["ds_of"], a7, a32)])
        for k in np.unique(key):
            ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix) // 2]] = True
        te = ~cal; a7t, a32t = a7[te], a32[te]; tgt = a32t.mean()
        p7 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[cal], a7[cal]).predict_proba(X)[:, 1]
        p32 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[cal], a32[cal]).predict_proba(X)[:, 1]
        scores = {"SOTA": -P["sig"]["prob_margin"][te], "meta-Δ": (p32 - p7)[te]}
        for m, sc in scores.items():
            e, a = curve(sc, a7t, a32t)
            res[m]["esc_par"].append(min_esc_at(e, a, tgt))
            for b in budgets:
                res[m]["acc_at"][b].append(acc_at_esc(e, a, b))
    print(f"\n################  FRONTIER  [{names_label(names)}]  parity={parity:.4f}  ################")
    print("  escalation @ iso-accuracy (parity), mean±SEM:")
    for m in ["SOTA", "meta-Δ"]:
        e = np.array(res[m]["esc_par"]); print(f"     {m:<8} {e.mean()*100:5.1f} ± {e.std(ddof=1)/np.sqrt(len(e))*100:.1f} %")
    d = np.array(res["SOTA"]["esc_par"]) - np.array(res["meta-Δ"]["esc_par"])
    print(f"     reduction: {d.mean()*100:.1f}pt  ({d.mean()/np.array(res['SOTA']['esc_par']).mean()*100:.0f}% rel)  "
          f"{'significant' if d.mean()>2*d.std(ddof=1)/np.sqrt(len(d)) else 'n.s.'}")
    print("  accuracy @ fixed escalation budget (meta-Δ should dominate):")
    print(f"     {'budget':<8}" + "".join(f"{int(b*100):>7}%" for b in budgets))
    for m in ["SOTA", "meta-Δ"]:
        print(f"     {m:<8}" + "".join(f"{np.mean(res[m]['acc_at'][b]):>8.4f}" for b in budgets))
    return res


def names_label(names):
    return {tuple(ALL6): "ALL-6", tuple(ALL5): "ALL-5 (excl MedXpert)", tuple(COMPETENT): "COMPETENT-4"}.get(tuple(names), "pool")


def main():
    D = CascadeData("cap320"); attach_verify(D)
    for names in [ALL6, ALL5, COMPETENT]:
        run(D, names)
    print("\nNovel method = Verification-Augmented Deferral Router (meta-Δ). Lower escalation@parity AND")
    print("higher accuracy@budget = frontier dominance over the SOTA confidence gate (pooled metric).")


if __name__ == "__main__":
    main()
