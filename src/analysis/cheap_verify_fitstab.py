#!/usr/bin/env python3
"""Is the pm_train shift a converged quantity, or an artifact of where a constant-learning-rate
loop happens to stop?  fit_shift_marginal uses lr=0.3 for a fixed 800 iterations with an argmax
inside, so it is a stochastic-approximation iterate, not a fixed point."""
import csv, json, os
import numpy as np
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_cheapverify")
CK = os.path.join(ROOT, "ckpts/output_bias")
csv.field_size_limit(10**9)

def jl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]
def load(stem):
    rows = []
    for s in ("_s0of2", "_s1of2"):
        p = os.path.join(CK, stem + s + ".jsonl")
        if os.path.exists(p): rows += jl(p)
    return rows
def strip_marker(t):
    t = str(t)
    return t[1:] if (t.startswith("Ġ") or t.startswith(" ")) else t
def letter_logits(row, K):
    lp = row.get("first_logprobs") or {}
    best = {}
    for t, v in lp.items():
        s = strip_marker(t)
        if len(s) == 1 and "A" <= s <= "Z": best[s] = max(float(v), best.get(s, -1e9))
    floor = min([float(v) for v in lp.values()], default=-30.0) if lp else -30.0
    return np.array([best.get(chr(65 + i), floor) for i in range(K)])
def fit(logits, target, iters, lr):
    n, K = logits.shape; w = np.zeros(K)
    for _ in range(iters):
        cur = np.bincount((logits - w).argmax(1), minlength=K) / n
        w = w + lr * np.log(np.maximum(cur, 1e-4) / np.maximum(target, 1e-4)); w -= w.mean()
    return w

ev = load("gen_PMC_VQA_id"); ev.sort(key=lambda r: r["i"])
trd = load("gen_PMC_TRAIN_train"); trd.sort(key=lambda r: r["i"])
X = np.array([letter_logits(r, 4) for r in ev])
gold = np.array([ord(str(r["answer"]).strip()[0]) - 65 for r in ev])
Xtr = np.array([letter_logits(r, 4) for r in trd])
tr = list(csv.DictReader(open("/data/dan/dataset/medevalkit/PMC-VQA/train_2.csv")))
tgt_full = np.bincount([ord(r["Answer"].strip()[0]) - 65 for r in tr], minlength=4) / len(tr)
tgt_dump = np.bincount([ord(str(r["answer"]).strip()[0]) - 65 for r in trd], minlength=4) / len(trd)
readout = float((X.argmax(1) == gold).mean())

R = {"readout_acc": readout,
     "target_full_train_2csv": [float(x) for x in tgt_full],
     "target_6000_item_dump": [float(x) for x in tgt_dump],
     "reference_pm_train_acc_in_artifact": 0.575202}
grid = {}
for tname, tgt in (("full_train_2csv", tgt_full), ("dump_6000", tgt_dump)):
    for iters in (200, 400, 800, 801, 1600, 3200, 6400):
        for lr in (0.3, 0.1, 0.03):
            w = fit(Xtr, tgt, iters, lr)
            acc = float(((X - w).argmax(1) == gold).mean())
            grid[f"{tname}|iters={iters}|lr={lr}"] = dict(
                acc=acc, vs_readout=acc - readout, w=[round(float(x), 6) for x in w],
                pred_marg=[round(float(x), 6) for x in
                           np.bincount((X - w).argmax(1), minlength=4) / len(X)])
R["grid"] = grid
accs = np.array([v["acc"] for v in grid.values()])
R["SENSITIVITY"] = dict(
    n_settings=len(accs), min_acc=float(accs.min()), max_acc=float(accs.max()),
    spread=float(accs.max() - accs.min()),
    min_vs_readout=float(accs.min() - readout), max_vs_readout=float(accs.max() - readout),
    note="the fit has no convergence criterion; lr is constant and the objective contains an argmax.")
json.dump(R, open(os.path.join(OUT, "fit_stability.json"), "w"), indent=1)
print(json.dumps({k: R[k] for k in ("readout_acc", "target_full_train_2csv", "target_6000_item_dump",
                                    "reference_pm_train_acc_in_artifact", "SENSITIVITY")}, indent=1))
for k, v in grid.items(): print(f"{k:42s} acc={v['acc']:.6f}  vs_readout={v['vs_readout']:+.6f}  w={v['w']}")
