#!/usr/bin/env python3
"""
router_conformal.py - CP-Router-style split-conformal gate (CORE mechanism).

Calibrates a per-option probability threshold on the CLEAN PMC-VQA-train labels via
split conformal prediction: LAC nonconformity score s = 1 - p_7B(gold); q_hat = the
finite-sample (1-alpha) quantile of calibration scores. At test time the 7B's option
softmax forms a prediction set {opt : p >= 1 - q_hat}; route to the 7B if the set is a
singleton (confident), escalate to 32B if ambiguous (|set| != 1). This is the principled,
coverage-controlled way to SET the escalation threshold -- vs the err-rate heuristic.

The SAME PMC-calibrated threshold is applied cold to all four datasets; reports cost,
parity, and (diagnostic) empirical coverage, swept over alpha, against the frozen-margin
baseline. The routing decision uses NO labels; coverage is a post-hoc check only.

NOTE: this is the CP-Router CORE. The paper's FBE step (adapt alpha per dataset from the
test-set entropy, label-free) is the next refinement -- added after reading 2505.19970.
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict

EVAL         = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
PMCTRAIN_DIR = "ckpts/gate_7b_pmctrain"
ALPHAS       = [0.05, 0.10, 0.15, 0.20, 0.30]
P7, P32      = 7.0, 32.0
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}(?:_s\dof\d)?\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def load_pmctrain(ckdir):
    rows = []
    for f in glob.glob(os.path.join(ckdir, "ckpt_nothink*.jsonl")):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows
def softmax_opts(lp):
    if not lp: return {}
    ks = list(lp); v = np.array([lp[k] for k in ks], dtype=np.float64); v -= v.max()
    p = np.exp(v); p /= p.sum()
    return {k: float(pi) for k, pi in zip(ks, p)}
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def tok(row): return float(row.get("gen_tokens") or 0)

# ---- calibration: split-conformal LAC scores on PMC-train ----
train_rows = load_pmctrain(PMCTRAIN_DIR)
assert train_rows, f"no calibration labels in {PMCTRAIN_DIR}"
cal = np.sort(np.array([1.0 - softmax_opts(r.get("opt_logprobs") or {}).get(r["gold"], 0.0)
                        for r in train_rows]))
n = len(cal)
def qhat(alpha):
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)   # standard split-CP rank
    return float(cal[k - 1])

r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
FR  = pickle.load(open("ckpts/router_margin.pkl", "rb")); fg, ftau = FR["gate"], FR["tau"]

def datasets():
    for name in EVAL:
        if name in r7 and name in r32:
            idx = sorted(set(r7[name]) & set(r32[name]))
            a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(float)
            a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
            t7  = np.array([tok(r7[name][i])  for i in idx])
            t32 = np.array([tok(r32[name][i]) for i in idx])
            yield name, idx, a7, a32, t7, t32

print("=" * 112)
print(f"CONFORMAL ROUTER (split-CP, LAC)   calibrated on {PMCTRAIN_DIR} (n={n})   vs frozen-margin baseline")
print("=" * 112)

print("\n[frozen-margin baseline]")
print(f"    {'dataset':<11}{'routed':>8}{'a32':>8}{'esc%':>7}{'FLOPs%':>8}")
for name, idx, a7, a32, t7, t32 in datasets():
    P = fg.predict_proba(np.array([[margin(r7[name][i])] for i in idx], dtype=np.float32))[:, 1]
    esc = P < ftau; routed = np.where(esc, a32, a7)
    fl = ((P7*t7).sum() + (P32*t32[esc]).sum()) / max((P32*t32).sum(), 1e-9)
    print(f"    {name:<11}{routed.mean():>8.3f}{a32.mean():>8.3f}{esc.mean()*100:>6.0f}%{fl*100:>7.0f}%")

for alpha in ALPHAS:
    qh = qhat(alpha); thr = 1.0 - qh
    print(f"\n[conformal  alpha={alpha:.2f}  target_cov={1-alpha:.2f}  q_hat={qh:.3f}  set-incl p>={thr:.3f}]")
    print(f"    {'dataset':<11}{'routed':>8}{'a32':>8}{'esc%':>7}{'FLOPs%':>8}{'cover':>8}{'av|set|':>8}   vs a32")
    for name, idx, a7, a32, t7, t32 in datasets():
        sets = [[k for k, pi in softmax_opts(r7[name][i].get("opt_logprobs") or {}).items() if pi >= thr]
                for i in idx]
        esc   = np.array([len(s) != 1 for s in sets])
        cov   = float(np.mean([r7[name][i]["gold"] in s for i, s in zip(idx, sets)]))  # diagnostic only
        setsz = float(np.mean([len(s) for s in sets]))
        routed = np.where(esc, a32, a7)
        fl = ((P7*t7).sum() + (P32*t32[esc]).sum()) / max((P32*t32).sum(), 1e-9)
        g = routed.mean() - a32.mean()
        print(f"    {name:<11}{routed.mean():>8.3f}{a32.mean():>8.3f}{esc.mean()*100:>6.0f}%{fl*100:>7.0f}%"
              f"{cov*100:>7.0f}%{setsz:>8.2f}{g:>+8.3f}")

print("\n" + "=" * 112)
print("READ: conformal sets the threshold by a coverage target on PMC-train, not a heuristic. Compare")
print("each alpha to the frozen-margin baseline. If one alpha holds parity across all four at lower")
print("FLOPs than frozen, CP's principled threshold already helps. If the per-dataset spread persists")
print("(SLAKE still wants more escalation), that is exactly what FBE -- adapt alpha per dataset from")
print("test-set entropy, no labels -- is built to fix: the next step after reading 2505.19970.")
