#!/usr/bin/env python3
"""
router_pareto.py - cost vs accuracy operating curve for the 7B->32B cascade.

router_escalate showed dist_full / entropy give a modest ACCURACY win on the
closed-form sets at one accuracy-maximizing tau. This asks the sharper question:
across the WHOLE range of escalation budgets, what accuracy does each cheap
signal buy at each cost? i.e. the Pareto frontier of (% escalated to 32B) vs
(routed accuracy).

For each dataset x cheap signal (margin, entropy, dist_full):
  - out-of-fold P(7B correct) from a CV'd logistic probe (honest, no leakage)
  - escalate the lowest-P(7B-correct) b-fraction first, for b = 0,10,...,100%
  - record routed accuracy at each budget -> the operating curve
  - report break-even% (least escalation to match always-32B), the peak and where
    it occurs, and a shuffle-control curve (labels permuted -> ranking is noise)

x-axis = % escalated to 32B, an assumption-free compute proxy (0 = all cheap-7B,
100 = always-32B). Numbers print verbatim for the site's hand-coded SVG curve.
Hidden-state signals are omitted: they lost to the output-side signals in
router_escalate. CPU only. Same inputs: ckpts/gate_7b_vllm, ckpts/gate_32b.
"""
import json, glob, os, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("OMP_NUM_THREADS", "1")
DATASETS = ["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA","SLAKE","VQA-RAD","PathVQA"]
N_SPLITS, N_REPEATS = 5, 5
BUDGETS = np.linspace(0, 1, 11)            # 0%,10%,...,100% escalated to 32B
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

def oof_predict(X, y, shuffle=False):
    """Out-of-fold P(7B correct) for every sample, averaged over repeats."""
    P = np.zeros(len(X)); cnt = np.zeros(len(X))
    for rep in range(N_REPEATS):
        yy = y.copy()
        if shuffle: yy = yy[np.random.RandomState(rep).permutation(len(yy))]
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, yy):
            ytr = yy[tr]
            if ytr.min() == ytr.max():
                p = np.full(len(te), float(ytr.mean()))
            else:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                mdl.fit(X[tr], ytr)
                p = mdl.predict_proba(X[te])[:, 1]
            P[te] += p; cnt[te] += 1
    return P / np.maximum(cnt, 1)

def curve(P, a7, a32):
    """Escalate the lowest-P(7B-correct) fraction first. acc at each BUDGET."""
    order = np.argsort(P, kind="stable"); n = len(P)   # ascending: least-confident first
    accs = []
    for b in BUDGETS:
        k = int(round(b * n))
        esc = np.zeros(n, bool); esc[order[:k]] = True
        accs.append(float(np.where(esc, a32, a7).mean()))
    return np.array(accs)

def breakeven(accs, target):
    for b, a in zip(BUDGETS, accs):
        if a >= target - 1e-9: return int(round(b * 100))
    return None

print("=" * 96)
print("COST-vs-ACCURACY OPERATING CURVE   (x = % escalated to 32B; cheap output-side signals)")
print("=" * 96)
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
hdr = "".join(f"{int(b*100):>6d}%" for b in BUDGETS)

for name in DATASETS:
    if name not in r7 or name not in r32:
        print(f"\n### {name}: missing labels"); continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    if not idx:
        print(f"\n### {name}: no overlap"); continue
    a7  = np.array([r7[name][i]["ok"] for i in idx])
    a32 = np.array([r32[name][i]["ok"] for i in idx])
    al7, al32 = a7.mean(), a32.mean()
    feats = {
        "margin":    np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32),
        "entropy":   np.array([[entropy(r7[name][i])] for i in idx], dtype=np.float32),
        "dist_full": np.stack([dist_vec(r7[name][i]) for i in idx]).astype(np.float32),
    }
    print(f"\n### {name}  (n={len(idx)})   always-7B={al7:.3f}   always-32B={al32:.3f}")
    print(f"    {'budget':<11s}{hdr}")
    rnd = al7 * (1 - BUDGETS) + al32 * BUDGETS          # random escalation = linear (expected)
    print(f"    {'random':<11s}" + "".join(f"{a:>7.3f}" for a in rnd) + "   (expected baseline)")
    for sname, X in feats.items():
        accs = curve(oof_predict(X, a7), a7, a32)
        sh   = curve(oof_predict(X, a7, shuffle=True), a7, a32)
        be   = breakeven(accs, al32)
        pkb  = int(round(BUDGETS[accs.argmax()] * 100)); pk = accs.max()
        tag  = f"break-even={be}%" if be is not None else "never matches 32B"
        flag = "  <-- ACC-WIN" if (pk - al32) > 0.01 and pkb < 100 else ""
        print(f"    {sname:<11s}" + "".join(f"{a:>7.3f}" for a in accs) +
              f"   {tag}  peak={pk:.3f}@{pkb}%{flag}")
        print(f"    {'  (shuffle)':<11s}" + "".join(f"{a:>7.3f}" for a in sh))

print("\n" + "=" * 96)
print("READ: 'random' is the no-information line (escalate a random fraction). A signal curve")
print("ABOVE it buys accuracy per unit compute. break-even% = least escalation to match always-")
print("32B; a peak above always-32B at <100% = a real accuracy gain, not just cost saving. Each")
print("signal's (shuffle) row is its control: real >> shuffle means the ranking carries real")
print("information. Numbers are verbatim for the site SVG. MedXpert ~ random -> efficiency floor.")
