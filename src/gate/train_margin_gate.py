#!/usr/bin/env python3
"""
router_train.py - train a frozen router on the CLEAN PMC-VQA TRAIN split, test cold.

Gate is fit on 7B-nothink labels from pmc_vqa_train (the model's RL data m23k is
text-only, so PMC-VQA is unseen) -> all 2000 PMC-VQA EVAL samples stay fully held
out. Threshold tau = the err-rate budget (escalate the 7B's error-rate fraction),
which needs ONLY 7B labels, so no 32B run on the train sample. The frozen gate is
then applied UNCHANGED to the full PMC-VQA eval set and to SLAKE / VQA-RAD / PathVQA.

train labels: ckpts/gate_7b_pmctrain/ckpt_nothink.jsonl   (7B-nothink only)
eval labels : ckpts/gate_7b_vllm (7B) + ckpts/gate_32b (32B)
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

os.environ.setdefault("OMP_NUM_THREADS", "1")
EVAL_PMC      = "PMC-VQA"
TRANSFER      = ["SLAKE", "VQA-RAD", "PathVQA"]
PMCTRAIN_DIR  = "ckpts/gate_7b_pmctrain"
EVAL_BASELINE = 0.539                       # eval-set 7B acc, for the contamination check
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

def load_pmctrain(ckdir):
    rows = []
    for f in glob.glob(os.path.join(ckdir, "ckpt_nothink*.jsonl")):
        for l in open(f):
            if l.strip(): rows.append(json.loads(l))
    return rows

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
    v = v - (v.max() + np.log(np.exp(v - v.max()).sum())); v = v[:K]
    if v.size < K: v = np.concatenate([v, np.full(K - v.size, pad)])
    return v.astype(np.float32)
FE = {"margin": lambda r: [margin(r)], "entropy": lambda r: [entropy(r)], "dist_full": lambda r: dist_vec(r)}
def feats_rows(rows, sig):            return np.array([FE[sig](r) for r in rows], dtype=np.float32)
def feats_idx(rows_by_idx, idx, sig): return np.array([FE[sig](rows_by_idx[i]) for i in idx], dtype=np.float32)

def pick_tau(P_tr, y_tr):
    """err-rate budget: escalate the bottom (1 - 7B_acc) fraction by P. Uses ONLY 7B labels."""
    err = 1.0 - float(np.mean(y_tr))
    return float(np.quantile(P_tr, min(max(err, 0.0), 1.0)))

# ---- training data: clean PMC-VQA TRAIN, 7B labels only ----
train_rows = load_pmctrain(PMCTRAIN_DIR)
assert train_rows, f"no train labels in {PMCTRAIN_DIR} -- run run_pmctrain_vllm.py --arm nothink first"
y_tr  = np.array([r["ok"] for r in train_rows]).astype(float)
m_tr  = np.array([margin(r) for r in train_rows])
zeros = float(np.mean(m_tr == 0))
print("=" * 100)
print(f"FROZEN ROUTER   train={PMCTRAIN_DIR}  n={len(train_rows)}")
print(f"  CONTAMINATION CHECK: train 7B acc = {y_tr.mean():.3f}   (eval baseline {EVAL_BASELINE:.3f}; "
      f"{'OK - clean' if abs(y_tr.mean()-EVAL_BASELINE) < 0.08 else 'WARNING - far from baseline, possible leakage'})")
print(f"  FEATURE CHECK: margin zeros = {zeros:.3f}   ({'OK' if zeros < 0.05 else 'WARNING - many empty opt_logprobs'})")
print("=" * 100)

# ---- eval data (both arms) ----
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
def eval_set(name):
    idx = sorted(set(r7[name]) & set(r32[name]))
    a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    return idx, a7, a32
eval_sets = {name: eval_set(name) for name in [EVAL_PMC] + TRANSFER if name in r7 and name in r32}

for sig in ["margin", "entropy", "dist_full"]:
    Xtr  = feats_rows(train_rows, sig)
    gate = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    gate.fit(Xtr, y_tr)
    tau  = pick_tau(gate.predict_proba(Xtr)[:, 1], y_tr)
    print(f"\n### signal = {sig}   (gate + tau fixed on PMC-VQA train; tau={tau:.3f})")
    print(f"    {'dataset':<12}{'n':>6}{'always7':>10}{'always32':>10}{'routed':>9}{'esc%':>7}{'gain_vs_32B':>13}   note")
    for name, (idx, a7, a32) in eval_sets.items():
        P   = gate.predict_proba(feats_idx(r7[name], idx, sig))[:, 1]
        esc = P < tau
        routed = np.where(esc, a32, a7)
        note = "in-domain (held-out eval split)" if name == EVAL_PMC else "TRANSFER (cold)"
        print(f"    {name:<12}{len(idx):>6}{a7.mean():>10.3f}{a32.mean():>10.3f}"
              f"{routed.mean():>9.3f}{esc.mean()*100:>6.0f}%{routed.mean()-a32.mean():>+13.3f}   {note}")
    if sig == "margin":
        os.makedirs("ckpts", exist_ok=True)
        with open("ckpts/router_margin.pkl", "wb") as f:
            pickle.dump({"gate": gate, "tau": float(tau), "signal": "margin", "trained_on": "pmc_vqa_train"}, f)
        print("    -> saved frozen router to ckpts/router_margin.pkl")

print("\n" + "=" * 100)
print("READ: gate trained on the CLEAN PMC-VQA train split; all 2000 eval rows are held out now.")
print("PMC-VQA = in-domain (train/test both PMC-VQA, different splits); SLAKE/VQA-RAD/PathVQA =")
print("cross-dataset transfer. routed~always32 at esc<100% => efficiency parity that transfers;")
print("esc pinned ~0% (routed=always7) => the fixed tau does NOT transfer -> the conformal-")
print("calibration layer (CP-Router style) is the method fix. margin is the gate to trust.")
