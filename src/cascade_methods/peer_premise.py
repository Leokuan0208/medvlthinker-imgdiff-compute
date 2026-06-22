#!/usr/bin/env python3
"""
peer_premise.py - PREMISE TEST for the cross-family complementarity direction.

Compares the same-family pair (MedVLThinker-7B vs its 32B sibling, both Qwen2.5-VL) against a
CROSS-FAMILY peer (e.g. InternVL2.5-8B, ckpts/peer/<tag>) on competent-4, all aligned by idx.
Reports, per-benchmark and pooled:
  - accuracies; error-correlation phi(ok_a, ok_b); P(b wrong | a wrong)  [vs same-family 0.584/0.372]
  - UNION/oracle ceiling max(ok_a, ok_b)  [GO if cross-family union clears always-32B = 0.645]
  - RECOVERABILITY AUROC: predict "peer fixes the 7B error" from CHEAP features {7B margin/entropy/
    top1prob/gini + cross-family DISAGREEMENT + benchmark id}, CV  [GO if > ~0.7, vs same-family ~0.6]
  - zero-leg router upper bound = mean(max(ok_7B, ok_peer))  (the accuracy a perfect router would hit)

Robust to peer logprob noise: uses correctness + predictions (from generated text), not peer logprobs.
Run from repo root:  python3 src/cascade_methods/peer_premise.py --peer_dir ckpts/peer/internvl25_8b --peer_tag internvl25_8b
"""
import sys, os, glob, json, argparse, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
ap = argparse.ArgumentParser()
ap.add_argument("--peer_dir", required=True)
ap.add_argument("--peer_tag", required=True)
ap.add_argument("--c7_dir", default="ckpts/gate_7b_prune/cap320")
ap.add_argument("--c7_cell", default="nothink_norag")
ap.add_argument("--c32_dir", default="ckpts/gate_32b")
ap.add_argument("--c32_cell", default="think_norag")
ap.add_argument("--benchmarks", nargs="+", default=COMPETENT)
A = ap.parse_args()

def load_arm(d, tag):
    """Flatten ckpt_<ds>_<tag>.jsonl -> {ds: {idx: row}} (ds = clean benchmark name)."""
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"ckpt_*{tag}*.jsonl")):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out

peer = load_arm(A.peer_dir, A.peer_tag)
c7d = load_arm(A.c7_dir, A.c7_cell)
c32d = load_arm(A.c32_dir, A.c32_cell)
class _D:  # tiny shim so the rest of the script reads c7/c32 per-ds dicts
    c7 = c7d; c32 = c32d
D = _D()
phi = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0

print(f"\n{'='*78}\nCROSS-FAMILY PREMISE TEST  (peer={A.peer_tag})\n{'='*78}")
print(f"{'benchmark':<14}{'acc7B':>7}{'accPeer':>8}{'acc32B':>7}{'union7+P':>9}{'P(Pw|7w)':>9}{'phi':>7}{'n':>6}")
ROWS = []
for ds in A.benchmarks:
    raw = D.raw.get(ds, {}) if hasattr(D, "raw") else None
    # pull 7B(cap320) + 32B from the harness pool, peer from disk; align by idx
    c7 = D.c7.get(ds, {}); c32 = D.c32.get(ds, {}); pp = peer.get(ds, {})
    idx = sorted(set(c7) & set(c32) & set(pp))
    if not idx: print(f"{ds:<14}  (no aligned idx)"); continue
    for i in idx:
        s = signals_from_logprobs(c7[i].get("opt_logprobs"))
        ROWS.append(dict(ds=ds, ok7=c7[i]["ok"], okP=pp[i]["ok"], ok32=c32[i]["ok"],
                         dis=int(c7[i].get("pred") != pp[i].get("pred")),
                         m=s["margin"], e=s["entropy"], t=s["top1prob"], g=s["gini"]))
    a7 = np.array([r["ok7"] for r in ROWS if r["ds"] == ds]); aP = np.array([r["okP"] for r in ROWS if r["ds"] == ds])
    a32 = np.array([r["ok32"] for r in ROWS if r["ds"] == ds])
    pw = ((aP == 0) & (a7 == 0)).sum() / max((a7 == 0).sum(), 1)
    print(f"{ds:<14}{a7.mean():>7.3f}{aP.mean():>8.3f}{a32.mean():>7.3f}{np.maximum(a7,aP).mean():>9.3f}{pw:>9.3f}{phi(a7,aP):>7.3f}{len(a7):>6}")

R = {k: np.array([r[k] for r in ROWS], float) for k in ["ok7", "okP", "ok32", "dis", "m", "e", "t", "g"]}
dsarr = np.array([r["ds"] for r in ROWS])
ok7, okP, ok32 = R["ok7"], R["okP"], R["ok32"]
print(f"\n{'POOLED':<14}{ok7.mean():>7.3f}{okP.mean():>8.3f}{ok32.mean():>7.3f}{np.maximum(ok7,okP).mean():>9.3f}"
      f"{((okP==0)&(ok7==0)).sum()/max((ok7==0).sum(),1):>9.3f}{phi(ok7,okP):>7.3f}{len(ok7):>6}")
print(f"\n--- ceilings (pooled) ---")
print(f"  always-7B={ok7.mean():.3f}  always-peer={okP.mean():.3f}  always-32B(parity)={ok32.mean():.3f}")
print(f"  UNION 7B|peer (oracle router) = {np.maximum(ok7,okP).mean():.3f}   [GO if > {ok32.mean():.3f}]")
print(f"  UNION 7B|32B (same-family)    = {np.maximum(ok7,ok32).mean():.3f}")
print(f"  triple UNION 7B|peer|32B      = {np.maximum(np.maximum(ok7,okP),ok32).mean():.3f}")

# RECOVERABILITY: predict "peer fixes a 7B error" from cheap features
oneh = np.column_stack([(dsarr == d).astype(float) for d in sorted(set(dsarr))])
CONF = np.column_stack([R["m"], R["e"], R["t"], R["g"]])
def auc(X, y):
    if y.sum() < 10 or y.sum() > len(y) - 10: return float("nan")
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5)),
                          X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
z_rec = ((ok7 == 0) & (okP == 1)).astype(int)   # peer recovers a 7B error
print(f"\n--- RECOVERABILITY: predict 1[7B wrong & PEER right]  (base {z_rec.mean():.3f}) ---")
print(f"  cheap 7B confidence only        : {auc(CONF, z_rec):.4f}")
print(f"  + cross-family DISAGREEMENT      : {auc(np.column_stack([CONF, R['dis']]), z_rec):.4f}")
print(f"  + disagreement + benchmark id    : {auc(np.column_stack([CONF, R['dis'], oneh]), z_rec):.4f}   [GO if > 0.70, same-family ~0.60]")
print(f"  disagreement ALONE               : {auc(R['dis'].reshape(-1,1), z_rec):.4f}")
# same-family comparison: predict 1[7B wrong & 32B right]
z_sf = ((ok7 == 0) & (ok32 == 1)).astype(int)
print(f"\n  [same-family baseline] predict 1[7B wrong & 32B right] (base {z_sf.mean():.3f}) from confidence: {auc(CONF, z_sf):.4f}")
print(f"\nVERDICT: GO if (union {np.maximum(ok7,okP).mean():.3f} > 32B {ok32.mean():.3f}) AND (recoverability AUROC > 0.70).")
