#!/usr/bin/env python3
"""
peer_router_img.py - THE decisive test of the scout's proposed method: can a router on frozen
IMAGE/TEXT content (SigLIP) capture the cross-family complementarity that confidence+agreement could
not? Routes among {7B, InternVL, Phi} by predicting per-model P(correct) from SigLIP embeddings,
honest 5-fold OOF. Reports routed accuracy vs always-7B / always-32B(parity) / oracle union, and the
zero-leg recoverability AUROC (predict 1[7B wrong & peer right] from image+text alone).
Run from repo root after embed_siglip.py:  python3 src/cascade_methods/peer_router_img.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

COMP = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
N7, N_IV, N_PHI, N32 = 7.6e9, 8.0e9, 4.2e9, 33.0e9
def load_arm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"ckpt_*{tag}*.jsonl")):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out
c7 = load_arm("ckpts/gate_7b_prune/cap320", "nothink_norag"); c32 = load_arm("ckpts/gate_32b", "think_norag")
iv = load_arm("ckpts/peer/internvl25_8b", "internvl25_8b"); phi = load_arm("ckpts/peer/phi35v", "phi35v")
emb = {}
for ds in COMP:
    z = np.load(f"feats_peer/siglip_{ds}.npz"); emb[ds] = {int(i): (z["img_emb"][k], z["txt_emb"][k]) for k, i in enumerate(z["idx"])}

rows, IMG, TXT = [], [], []
for ds in COMP:
    idx = sorted(set(c7[ds]) & set(c32[ds]) & set(iv[ds]) & set(phi[ds]) & set(emb[ds]))
    for i in idx:
        s = signals_from_logprobs(c7[ds][i].get("opt_logprobs"))
        rows.append(dict(ds=ds, ok7=c7[ds][i]["ok"], okiv=iv[ds][i]["ok"], okphi=phi[ds][i]["ok"], ok32=c32[ds][i]["ok"],
                         p7=c7[ds][i]["pred"], piv=iv[ds][i]["pred"], m=s["margin"], e=s["entropy"], t=s["top1prob"], g=s["gini"]))
        IMG.append(emb[ds][i][0]); TXT.append(emb[ds][i][1])
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
ok7, okiv, okphi, ok32 = [A[k].astype(int) for k in ["ok7", "okiv", "okphi", "ok32"]]
ds = A["ds"]; n = len(rows); IMG = np.array(IMG, float); TXT = np.array(TXT, float)
oneh = np.column_stack([(ds == d).astype(float) for d in COMP])
CONF = np.column_stack([A["m"], A["e"], A["t"], A["g"]]).astype(float)
ag = (A["p7"] == A["piv"]).astype(float)
print(f"n={n}  img_dim={IMG.shape[1]} txt_dim={TXT.shape[1]}")
print(f"baselines: 7B={ok7.mean():.3f} 32B(parity)={ok32.mean():.3f}  oracle 7B|IV={np.maximum(ok7,okiv).mean():.3f} 7B|IV|Phi={np.maximum.reduce([ok7,okiv,okphi]).mean():.3f}")

# reduce embeddings
IMGp = PCA(128, random_state=0).fit_transform(StandardScaler().fit_transform(IMG))
TXTp = PCA(64, random_state=0).fit_transform(StandardScaler().fit_transform(TXT))

def router(models_ok, feats, name, fwd):
    skf = StratifiedKFold(5, shuffle=True, random_state=0); strat = ok7 + 2 * okiv
    P = np.zeros((n, len(models_ok)))
    for tr, te in skf.split(feats, strat):
        for j, y in enumerate(models_ok):
            if y[tr].sum() in (0, len(tr)): P[te, j] = y[tr].mean(); continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3))
            clf.fit(feats[tr], y[tr]); P[te, j] = clf.predict_proba(feats[te])[:, 1]
    routed = np.choose(P.argmax(1), models_ok); sw = (P.argmax(1) != 0).mean()
    flops32 = 2 * N32 * (685 + 391) * n
    print(f"  {name:<40} acc={routed.mean():.3f}  away-from-7B={sw*100:.0f}%  FLOPs={sum(fwd)*n/flops32*100:.0f}%")
    return routed.mean()

f7 = 2 * N7 * 390; fiv = 2 * N_IV * 390; fphi = 2 * N_PHI * 390
print(f"\n--- IMAGE/TEXT-content router (the proposed zero-leg method; honest OOF) ---")
print(f"  always-7B                                acc={ok7.mean():.3f}")
router([ok7, okiv], np.hstack([IMGp]), "2-leg IMG-only {7B,IV}", [f7, fiv])
router([ok7, okiv], np.hstack([IMGp, TXTp]), "2-leg IMG+TXT {7B,IV}", [f7, fiv])
router([ok7, okiv, okphi], np.hstack([IMGp, TXTp]), "3-leg IMG+TXT {7B,IV,Phi}", [f7, fiv, fphi])
router([ok7, okiv, okphi], np.hstack([IMGp, TXTp, CONF, ag.reshape(-1,1), oneh]), "3-leg IMG+TXT+conf+agree", [f7, fiv, fphi])

def auc(X, y):
    if y.sum() < 10: return float("nan")
    p = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.3)), X, y, cv=5, method="predict_proba")[:, 1]
    return roc_auc_score(y, p)
zrec = ((ok7 == 0) & (okiv == 1)).astype(int)
print(f"\n--- ZERO-LEG recoverability AUROC: predict 1[7B wrong & InternVL right] from content alone (base {zrec.mean():.3f}) ---")
print(f"  image only        : {auc(IMGp, zrec):.4f}")
print(f"  image+text        : {auc(np.hstack([IMGp,TXTp]), zrec):.4f}")
print(f"  image+text+bench  : {auc(np.hstack([IMGp,TXTp,oneh]), zrec):.4f}   [vs same-family 0.60; confidence-only was 0.725]")
print(f"\nVERDICT: method works if an IMG/TXT router beats always-7B {ok7.mean():.3f} toward 32B {ok32.mean():.3f} at sub-32B FLOPs.")
