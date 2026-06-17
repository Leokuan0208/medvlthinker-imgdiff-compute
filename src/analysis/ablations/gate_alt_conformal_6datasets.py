#!/usr/bin/env python3
"""
router_conformal_6ds.py - CP-Router-style SPLIT-CONFORMAL selective gate, compared head-to-head
with the frozen 1-D margin gate across all 6 evaluation datasets. NEW file.

Nonconformity score = 1 - p_top, where p_top is the top softmax probability over the 7B's
answer options (the "is the confidence set a singleton?" criterion -- it uses the FULL option
distribution via the softmax normaliser, which is what makes it distinct from the raw top1-top2
margin). Calibrate a split-conformal threshold q_cp on the CLEAN PMC-VQA-train scores at the
err-rate budget, FREEZE it, and apply UNCHANGED to all 6 eval datasets -- exactly the transfer
protocol used for the margin gate and the learned (HistGBM) router. Keep the 7B answer iff
score <= q_cp (confident singleton); otherwise the set is ambiguous -> escalate to the 32B.
Prints always-7B / always-32B / margin-gate / conformal / oracle. Saves ckpts/router_conformal_6ds.pkl.

NOTE: this is a fresh, standard split-conformal implementation (the uploaded router_hidden.py is
the layer-L HIDDEN-STATE probe, not the conformal router). If your original CP-Router used a
different nonconformity score, swap the score() function below -- everything else (calibration,
freezing, transfer, reporting) is identical. Sanity check: it should land within noise of the
margin gate on the competent-4 pool (~0.65 routed acc).
"""
import json, glob, os, re, pickle, numpy as np
from collections import defaultdict

EVAL      = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
             "MedXpert-Reasoning", "MedXpert-Understanding"]
COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
MEDX      = ["MedXpert-Reasoning", "MedXpert-Understanding"]
PMCTRAIN_DIR = "ckpts/gate_7b_pmctrain"
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

def softmax_opts(row):
    lp = row.get("opt_logprobs") or {}
    v = np.array(sorted(lp.values(), reverse=True), dtype=np.float64)
    if v.size == 0: return np.array([1.0])
    p = np.exp(v - v.max()); p /= p.sum(); return p
def score(row):                       # nonconformity: 1 - top softmax prob  (HIGH = ambiguous)
    return 1.0 - softmax_opts(row)[0]
def margin(row):                      # for the frozen 1-D margin-gate baseline only
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0

# ---- calibrate the conformal threshold on clean PMC-train ----
train = load_pmctrain(PMCTRAIN_DIR); assert train, f"no labels in {PMCTRAIN_DIR}"
s_tr  = np.array([score(r) for r in train])
ok_tr = np.array([r["ok"] for r in train]).astype(float)
alpha = 1.0 - ok_tr.mean()                                   # err-rate budget (same convention as the others)
n_cal = len(s_tr)
cal_sorted = np.sort(s_tr)                                    # ascending nonconformity scores
k = int(np.ceil((n_cal + 1) * (1 - alpha)))                  # finite-sample split-conformal rank
k = min(max(k, 1), n_cal)
q_cp = float(cal_sorted[k - 1])                              # keep iff score <= q_cp; escalate top-alpha by score

# ---- load arms + frozen 1-D margin gate baseline ----
r7  = load_arm("ckpts/gate_7b_vllm", "nothink_norag")
r32 = load_arm("ckpts/gate_32b",     "think_norag")
FR  = pickle.load(open("ckpts/router_margin.pkl", "rb")); fg, ftau = FR["gate"], FR["tau"]
def m1(by, idx): return np.array([[margin(by[i])] for i in idx], dtype=np.float32)

store = {}
for name in EVAL:
    if name not in r7 or name not in r32:
        print(f"    [missing checkpoints] {name}"); continue
    idx = sorted(set(r7[name]) & set(r32[name]))
    a7  = np.array([r7[name][i]["ok"]  for i in idx]).astype(float)
    a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(float)
    Pb  = fg.predict_proba(m1(r7[name], idx))[:, 1];      eb = Pb < ftau; rb = np.where(eb, a32, a7)
    sc  = np.array([score(r7[name][i]) for i in idx]);    ec = sc > q_cp; rc = np.where(ec, a32, a7)
    store[name] = dict(a7=a7, a32=a32, eb=eb, rb=rb, ec=ec, rc=rc, orac=np.maximum(a7, a32))

def cat(keys, field): return np.concatenate([store[k][field] for k in keys if k in store])
def line(label, keys):
    keys = [k for k in keys if k in store]
    if not keys: return
    a7=cat(keys,'a7'); a32=cat(keys,'a32'); eb=cat(keys,'eb'); rb=cat(keys,'rb')
    ec=cat(keys,'ec'); rc=cat(keys,'rc'); orac=cat(keys,'orac'); n=len(a7)
    print(f"    {label:<20}{n:>6}{a7.mean():>8.3f}{a32.mean():>8.3f} |"
          f"{rb.mean():>8.3f}{eb.mean()*100:>5.0f}%{rb.mean()-a32.mean():>+8.3f} |"
          f"{rc.mean():>8.3f}{ec.mean()*100:>5.0f}%{rc.mean()-a32.mean():>+8.3f} |{orac.mean():>8.3f}")

print("=" * 110)
print(f"CONFORMAL ROUTER (split-CP, score=1-p_top)  train={PMCTRAIN_DIR}  n={len(train)}  "
      f"7B_acc={ok_tr.mean():.3f}  alpha={alpha:.3f}  q_cp={q_cp:.3f}")
print("=" * 110)
print(f"    {'dataset':<20}{'n':>6}{'a7':>8}{'a32':>8} |{'  margin-gate (1-D)':<21} |{'  conformal (CP)':<21} |{'oracle':>8}")
print(f"    {'':<20}{'':>6}{'':>8}{'':>8} |{'     acc  esc    gain':<21} |{'     acc  esc    gain':<21} |{'':>8}")
print("    " + "-" * 106)
for ds in COMPETENT: line(ds, [ds])
line("MMMU", ["MMMU"])
line("MedX-M (comb)", MEDX)
print("    " + "-" * 106)
line("POOLED competent-4", COMPETENT)
line("POOLED all-6", COMPETENT + ["MMMU"] + MEDX)

with open("ckpts/router_conformal_6ds.pkl", "wb") as f:
    pickle.dump({"q_cp": q_cp, "alpha": alpha, "score": "1-p_top", "trained_on": "pmc_vqa_train"}, f)
print("\n    -> saved ckpts/router_conformal_6ds.pkl")
print("=" * 110)
print("READ: 'gain' = routed acc minus always-32B. Compare the conformal block with the margin-gate")
print("block across all 6 datasets. If the conformal router's gain is no better than the 1-D margin")
print("gate, the simple gate is at the frontier. Swap score() if your CP-Router used another score.")
