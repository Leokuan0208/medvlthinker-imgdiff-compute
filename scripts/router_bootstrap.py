#!/usr/bin/env python3
"""
router_bootstrap.py - is the cascade's accuracy gain real, or small-sample noise?

router_pareto found dist_full / entropy / margin beat always-32B on the closed-
form sets (PathVQA solidly, VQA-RAD strongly but n=272). This puts a confidence
interval on that gain via a PAIRED bootstrap over the evaluation sample.

For each competent dataset x cheap signal:
  - OOF P(7B correct) from the CV'd probe (computed once; the bootstrap then
    resamples the EVALUATION sample with the gate's scores held fixed -> the CI
    reflects test-set uncertainty, the dominant source given fixed labels).
  - route at two operating points:
      * err-rate : escalate the bottom (1 - always7) fraction   [PRINCIPLED /
                   pre-registerable: a perfect router escalates the 7B's error rate]
      * peak     : the budget that maximised full-data routed acc [SELECTED -> optimistic]
  - gain_i = routed_i - a32_i  (paired, per-sample, in {-1,0,+1}). Bootstrap
    mean(gain_i) B times. Report observed gain, 95% percentile CI, one-sided
    bootstrap p (frac of resamples with gain <= 0), and SIG if the 95% CI excludes 0.

PathVQA (n=3362) is the trustworthy test; VQA-RAD (n=272) is the small-n one.
PMC-VQA / SLAKE are expected to come back NOT significant -> confirms those are
efficiency-only, not accuracy wins. CPU only. Inputs: ckpts/gate_7b_vllm, ckpts/gate_32b.
"""
import json, glob, os, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
DATASETS = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]   # competent only; MedXpert is the efficiency floor
N_SPLITS, N_REPEATS = 5, 5
B_BOOT = 2000
GRID = np.linspace(0, 1, 21)               # 0,5,...,100% — to locate the peak
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}_s\dof\d\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d

def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0

def entropy(row):
    lp = row.get("opt_logprobs") or {}
    v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return 0.0
    p = np.exp(v - v.max()); p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def dist_vec(row, K=10, pad=-20.0):
    lp = row.get("opt_logprobs") or {}
    v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return np.full(K, pad, dtype=np.float32)
    v = v - (v.max() + np.log(np.exp(v - v.max()).sum()))
    v = v[:K]
    if v.size < K: v = np.concatenate([v, np.full(K - v.size, pad)])
    return v.astype(np.float32)

def oof_predict(X, y):
    """Out-of-fold P(7B correct) for every sample, averaged over repeats."""
    P = np.zeros(len(X)); cnt = np.zeros(len(X))
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, y):
            ytr = y[tr]
            if ytr.min() == ytr.max():
                p = np.full(len(te), float(ytr.mean()))
            else:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                mdl.fit(X[tr], ytr)
                p = mdl.predict_proba(X[te])[:, 1]
            P[te] += p; cnt[te] += 1
    return P / np.maximum(cnt, 1)

def routed_outcome(P, a7, a32, b):
    """Escalate the lowest-P(7B-correct) b-fraction. Returns (per-sample routed, esc_frac)."""
    n = len(P); k = int(round(b * n))
    order = np.argsort(P, kind="stable")
    esc = np.zeros(n, bool); esc[order[:k]] = True
    return np.where(esc, a32, a7), esc.mean()

def boot_gain(gain_i, B=B_BOOT, seed=0):
    """Paired bootstrap of mean(gain_i). Returns (lo95, hi95, one_sided_p)."""
    rng = np.random.RandomState(seed); n = len(gain_i)
    idx = rng.randint(0, n, size=(B, n))           # B resamples of the eval set
    means = gain_i[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi), float((means <= 0).mean())

print("=" * 96)
print("PAIRED BOOTSTRAP: is routed-acc - always-32B > 0?   (B=%d; principled err-rate point + peak)" % B_BOOT)
print("=" * 96)
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")

for name in DATASETS:
    if name not in r7 or name not in r32:
        print(f"\n### {name}: missing labels"); continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    if not idx:
        print(f"\n### {name}: no overlap"); continue
    a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    al7, al32 = a7.mean(), a32.mean()
    feats = {
        "margin":    np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32),
        "entropy":   np.array([[entropy(r7[name][i])] for i in idx], dtype=np.float32),
        "dist_full": np.stack([dist_vec(r7[name][i]) for i in idx]).astype(np.float32),
    }
    print(f"\n### {name}  (n={len(idx)})   always-7B={al7:.3f}   always-32B={al32:.3f}   err-rate budget={1-al7:.0%}")
    seed = 0
    for sname, X in feats.items():
        P = oof_predict(X, a7)
        grid_acc = [routed_outcome(P, a7, a32, b)[0].mean() for b in GRID]
        peak_b = float(GRID[int(np.argmax(grid_acc))])
        for label, b in [("err-rate", 1 - al7), ("peak", peak_b)]:
            b = min(max(b, 0.0), 1.0)
            routed, frac = routed_outcome(P, a7, a32, b)
            gain_i = routed - a32
            g = gain_i.mean()
            lo, hi, p1 = boot_gain(gain_i, seed=seed); seed += 1
            sig = "SIG +" if lo > 0 else ("SIG -" if hi < 0 else "ns")
            print(f"    {sname:10s} {label:8s} esc={frac:>4.0%}  routed={routed.mean():.3f}  "
                  f"gain={g:+.3f}  95%CI[{lo:+.3f}, {hi:+.3f}]  p={p1:.3f}  -> {sig}")

print("\n" + "=" * 96)
print("READ: SIG + means the 95% CI for (routed - always-32B) is entirely above 0 -> a real")
print("accuracy gain, not noise. The 'err-rate' row is the honest headline (its budget is set")
print("a priori, not tuned); 'peak' is the optimistic upper estimate (budget chosen on the data).")
print("ns at err-rate = efficiency-only (match 32B, no significant accuracy gain). The CI here is")
print("test-sample uncertainty with the gate fixed; generalization across datasets is the next test.")
