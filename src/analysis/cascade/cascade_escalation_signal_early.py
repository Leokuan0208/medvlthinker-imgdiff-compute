#!/usr/bin/env python3
"""
router_escalate.py - THE escalation-gate experiment.

Tonight's cascade finding: cross-model complementarity is real (+12.5pp oracle
headroom), but the 7B CONFIDENCE MARGIN as an escalation signal failed (punted
96% to 32B, just reproduced 32B accuracy). Question this script answers:

  Can ANY cheap 7B signal decide "keep on cheap-7B vs escalate to 32B" well
  enough to turn the +12.5pp oracle headroom into a real gain? The overnight run
  showed the margin and the layer-14 hidden state both failed; this run adds the
  full predictive distribution (entropy, sorted log-probs), which were untested.

Signals compared head-to-head (identical CV folds):
  (1) margin     : 7B opt_logprobs top1-top2          (the original baseline)
  (2) entropy    : Shannon entropy of the A-J option distribution (cheap output-side)
  (3) dist_full  : full per-option log-distribution, sorted+padded (max output-side)
  (4) hidden     : layer-14 features (last-token and mean-pooled)
  (5) hidden+mgn : hidden concatenated with margin

Routing: probe predicts P(7B correct). Keep-7B if P >= tau, else escalate to 32B.
tau chosen on TRAIN to maximize routed accuracy, applied to TEST. Reported per
dataset + pooled, with a shuffle control.

PRE-REGISTERED DECISION (per dataset): a signal WINS if gated cascade
  (a) beats always-32B by >0.01, OR
  (b) matches always-32B (within 1 std) while escalating clearly less than margin.
Else -> no better than always-32B -> 'use 32B' for that dataset. Verdict rests on
COMPETENT datasets only (7B-pred-AUROC well above 0.5); MedXpert is near-chance.

CPU only. Reads ckpts/gate_7b_vllm (cheap-7B), ckpts/gate_32b (32B), feats_full/.
"""
import json, glob, os, re, math, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS","1")   # avoid BLAS thread thrash (seen before)
DATASETS = ["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA","SLAKE","VQA-RAD","PathVQA"]
N_SPLITS, N_REPEATS, PCA_DIM = 5, 5, 64
np.random.seed(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}(?:_s\dof\d)?\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")

def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0

def entropy(row):
    """Shannon entropy (nats) of the 7B's option distribution. Low = confident, high = torn.
    Unlike the margin (top1-top2 only), this sees the WHOLE distribution's spread."""
    lp = row.get("opt_logprobs") or {}
    v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return 0.0
    p = np.exp(v - v.max()); p = p / p.sum()          # stable softmax -> proper probs
    return float(-(p * np.log(p + 1e-12)).sum())

def dist_vec(row, K=10, pad=-20.0):
    """Full per-option log-distribution: renormalized, sorted high->low, padded to K.
    The maximal output-side signal -- the whole SHAPE of the 7B's uncertainty."""
    lp = row.get("opt_logprobs") or {}
    v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return np.full(K, pad, dtype=np.float32)
    v = v - (v.max() + np.log(np.exp(v - v.max()).sum()))   # proper log-softmax (stable)
    v = v[:K]
    if v.size < K: v = np.concatenate([v, np.full(K - v.size, pad)])
    return v.astype(np.float32)

def load_feats(name):
    fs = glob.glob(f"feats_full/feat_{name}_L*.npz")
    if not fs: return None
    parts = [np.load(f) for f in fs]
    idx = np.concatenate([p["idx"] for p in parts])
    hl  = np.concatenate([p["h_last"] for p in parts])
    hm  = np.concatenate([p["h_mean"] for p in parts])
    return {int(i): (hl[k], hm[k]) for k,i in enumerate(idx)}

def route_eval(P_pred, a7, a32, tr, te):
    """choose tau on train to max routed acc; apply to test. returns (acc, esc_frac)."""
    best_tau, best = None, -1
    for tau in np.quantile(P_pred[tr], np.linspace(0,1,21)):
        keep = P_pred[tr] >= tau
        acc = np.where(keep, a7[tr], a32[tr]).mean()
        if acc > best: best, best_tau = acc, tau
    keep_te = P_pred[te] >= best_tau
    acc_te = np.where(keep_te, a7[te], a32[te]).mean()
    return acc_te, (~keep_te).mean()

def run_signal(X, a7, a32, pca=False, shuffle=False):
    """CV a probe predicting 7B-correctness from X, route on it. returns dict of arrays."""
    strat = a7  # stratify on the thing we predict
    out = defaultdict(list)
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep)
        for tr, te in skf.split(X, strat):
            ytr = a7[tr].copy()
            if shuffle: ytr = ytr[np.random.RandomState(rep).permutation(len(tr))]
            if ytr.min()==ytr.max():
                P = np.full(len(X), float(ytr.mean()))
            else:
                mdl = make_pipeline(StandardScaler(),
                        PCA(min(PCA_DIM, len(tr)-1, X.shape[1]), svd_solver="randomized", random_state=0),
                        LogisticRegression(max_iter=2000, C=0.5)) if pca else \
                      make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
                mdl.fit(X[tr], ytr)
                P = np.zeros(len(X)); P[te] = mdl.predict_proba(X[te])[:,1]; P[tr] = mdl.predict_proba(X[tr])[:,1]
            acc, esc = route_eval(P, a7, a32, tr, te)
            out["acc"].append(acc); out["esc"].append(esc)
            if not shuffle and len(set(a7[te]))>1:
                out["auc"].append(roc_auc_score(a7[te], P[te]))
    return out

print("="*76); print("ESCALATION GATE: cheap 7B signals vs always-32B"); print("="*76)
pool = defaultdict(lambda: defaultdict(list))
for name in DATASETS:
    if name not in r7 or name not in r32: print(f"\n{name}: missing labels"); continue
    feats = load_feats(name)
    if feats is None: print(f"\n{name}: no features"); continue
    idx = sorted(set(r7[name]) & set(r32[name]) & set(feats))
    if not idx: print(f"\n{name}: no overlap"); continue
    a7  = np.array([r7[name][i]["ok"] for i in idx])
    a32 = np.array([r32[name][i]["ok"] for i in idx])
    mg  = np.array([[margin(r7[name][i])] for i in idx])
    en  = np.array([[entropy(r7[name][i])] for i in idx], dtype=np.float32)
    dv  = np.stack([dist_vec(r7[name][i]) for i in idx]).astype(np.float32)
    hl  = np.stack([feats[i][0] for i in idx]).astype(np.float32)
    hm  = np.stack([feats[i][1] for i in idx]).astype(np.float32)

    always32 = a32.mean(); always7 = a7.mean(); oracle = (a7|a32).mean()
    print(f"\n### {name}  (n={len(idx)})")
    print(f"     cheap-7B={always7:.3f}  32B={always32:.3f}  oracle={oracle:.3f} (headroom over 32B {oracle-always32:+.3f})")

    sigs = {
        "margin":      (mg,  False),
        "entropy":     (en,  False),
        "dist_full":   (dv,  False),
        "hidden_last": (hl,  True),
        "hidden_mean": (hm,  True),
        "hidden+mgn":  (np.hstack([hl, mg]), True),
    }
    for sname,(X,pca) in sigs.items():
        res = run_signal(X, a7, a32, pca=pca)
        sh  = run_signal(X, a7, a32, pca=pca, shuffle=True)
        acc, esc = np.mean(res["acc"]), np.mean(res["esc"])
        auc = np.mean(res["auc"]) if res["auc"] else float("nan")
        gain = acc - always32
        # verdict per pre-registered rule
        if gain > 0.01: verdict = "WIN (beats 32B accuracy)"
        elif abs(gain) <= np.std(res["acc"]) and esc < 0.85: verdict = "WIN-efficiency (~32B acc, less escalation)"
        else: verdict = "no better than 32B"
        print(f"     {sname:12s} acc={acc:.3f}±{np.std(res['acc']):.3f}  esc={esc:.0%}  "
              f"7B-pred-AUROC={auc:.2f}  gain_vs_32B={gain:+.3f}  shuffle={np.mean(sh['acc']):.3f}  -> {verdict}")
        for k in ["acc","esc"]: pool[sname][k] += res[k]

print("\n" + "="*76 + "\nPOOLED (all 6 datasets)\n" + "="*76)
for sname in ["margin","entropy","dist_full","hidden_last","hidden_mean","hidden+mgn"]:
    if pool[sname]["acc"]:
        print(f"     {sname:12s} acc={np.mean(pool[sname]['acc']):.3f}  esc={np.mean(pool[sname]['esc']):.0%}")
print("\nREAD (pre-registered): on a COMPETENT dataset (7B-pred-AUROC well above 0.5),")
print("does ANY cheap signal -- margin, entropy, dist_full, or the layer-14 hidden")
print("state -- show WIN or WIN-efficiency vs always-32B? If yes, that signal is a")
print("real escalation gate -> a method. If every signal says 'no better than 32B' on")
print("every competent dataset, the +12.5pp headroom is NOT reachable from cheap 7B")
print("signals, and the honest contribution is the efficiency cascade on competent")
print("benchmarks only. MedXpert is near-chance -> its verdict is non-informative.")
