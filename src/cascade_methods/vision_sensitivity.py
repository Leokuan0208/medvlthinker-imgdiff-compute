#!/usr/bin/env python3
"""
vision_sensitivity.py - is the BLANK-IMAGE counterfactual ("did the 7B actually use the image?") a
usable, OBSERVED deferral signal (sidestepping the recoverability-prediction wall)?

vision_insensitive(item) := 7B's answer is UNCHANGED when the image is replaced by mid-gray
(i.e. the model answered from language priors). Compares, on competent-4:
  - 7B accuracy on vision-SENSITIVE vs INSENSITIVE items
  - whether insensitivity marks 7B errors, and whether those errors are recoverable (32B / InternVL)
  - AUROC for predicting 7B correctness & recoverability from confidence vs +insensitivity (+margin-shift)
  - a simple gate: escalate insensitive-and/or-low-confidence items; cascade acc vs escalation
Run from repo root after the blank run:  python3 src/cascade_methods/vision_sensitivity.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

COMP = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
def load_arm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"ckpt_*{tag}*.jsonl")):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out
real = load_arm("ckpts/gate_7b_prune/cap320", "nothink_norag")
blank = load_arm("ckpts/peer/blank7b", "blank")
c32 = load_arm("ckpts/gate_32b", "think_norag")
iv = load_arm("ckpts/peer/internvl25_8b", "internvl25_8b")

rows = []
for ds in COMP:
    idx = sorted(set(real[ds]) & set(blank[ds]) & set(c32[ds]) & set(iv[ds]))
    for i in idx:
        sr = signals_from_logprobs(real[ds][i].get("opt_logprobs"))
        sb = signals_from_logprobs(blank[ds][i].get("opt_logprobs"))
        rows.append(dict(ds=ds, ok7=real[ds][i]["ok"], ok32=c32[ds][i]["ok"], okiv=iv[ds][i]["ok"],
                         insens=int(real[ds][i]["pred"] == blank[ds][i]["pred"]),
                         m=sr["margin"], e=sr["entropy"], t=sr["top1prob"], g=sr["gini"],
                         dmargin=abs(sr["margin"] - sb["margin"])))
A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "ds"}
ds = np.array([r["ds"] for r in rows]); n = len(rows)
ok7, ok32, okiv, ins = A["ok7"].astype(int), A["ok32"].astype(int), A["okiv"].astype(int), A["insens"].astype(int)
print(f"n={n}  7B acc={ok7.mean():.3f}  vision-INSENSITIVE rate={ins.mean():.3f} (answer unchanged w/ blank image)")
print(f"\n--- accuracy split by vision-sensitivity ---")
print(f"  vision-SENSITIVE  (answer changed): n={(ins==0).sum():4d}  7B acc={ok7[ins==0].mean():.3f}")
print(f"  vision-INSENSITIVE(answer same)   : n={(ins==1).sum():4d}  7B acc={ok7[ins==1].mean():.3f}")
for d in COMP:
    mk = ds == d
    print(f"    {d:<10} insens={ins[mk].mean():.2f}  acc(sens)={ok7[mk&(ins==0)].mean() if (mk&(ins==0)).sum() else float('nan'):.3f}  acc(insens)={ok7[mk&(ins==1)].mean() if (mk&(ins==1)).sum() else float('nan'):.3f}")
err = ok7 == 0
print(f"\n--- recoverability of 7B errors, split by sensitivity ---")
print(f"  among SENSITIVE   errors (n={(err&(ins==0)).sum()}): 32B fixes {ok32[err&(ins==0)].mean():.3f}  InternVL fixes {okiv[err&(ins==0)].mean():.3f}")
print(f"  among INSENSITIVE errors (n={(err&(ins==1)).sum()}): 32B fixes {ok32[err&(ins==1)].mean():.3f}  InternVL fixes {okiv[err&(ins==1)].mean():.3f}")
CONF = np.column_stack([A["m"], A["e"], A["t"], A["g"]])
def auc(X, y):
    if y.sum() < 10 or y.sum() > len(y) - 10: return float("nan")
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)), X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
print(f"\n--- AUROC: predict 7B CORRECTNESS ---")
print(f"  confidence only            : {auc(CONF, ok7):.4f}")
print(f"  + insensitivity + dmargin  : {auc(np.column_stack([CONF, ins, A['dmargin']]), ok7):.4f}")
print(f"  insensitivity+dmargin only : {auc(np.column_stack([ins, A['dmargin']]), ok7):.4f}")
zrec = (err & (ok32 == 1)).astype(int)
print(f"\n--- AUROC: predict RECOVERABILITY 1[7B wrong & 32B right] (base {zrec.mean():.3f}) ---")
print(f"  confidence only            : {auc(CONF, zrec):.4f}")
print(f"  + insensitivity + dmargin  : {auc(np.column_stack([CONF, ins, A['dmargin']]), zrec):.4f}   [beat ~0.60 to be a new signal]")
print(f"\nVERDICT: vision-sensitivity is useful if it raises correctness/recoverability AUROC beyond confidence,")
print(f"or if insensitive errors are a large, distinctly-recoverable bucket (a targeted intervention).")
