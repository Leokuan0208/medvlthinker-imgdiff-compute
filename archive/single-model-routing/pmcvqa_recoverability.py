#!/usr/bin/env python3
"""
pmcvqa_recoverability.py - the audit's key pre-check, PMC-VQA ONLY.
Tests whether a routable signal EXISTS on the one competent-regime benchmark
(PMC-VQA @ 0.52), unlike near-chance MedXpert where signal is unrecoverable.
  (1) oracle vs luck floor  -> real per-question complementarity?
  (2) per-arm correctness AUROC (scalar + hidden) -> correctness predictable?
  (3) routed - best_fixed + shuffle control -> does a router actually gain?
CPU-only, uses existing ckpts/ + feats/. No GPU, no 32B needed.
"""
import json, glob, os, re, math, random, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CKDIR, FEATDIR = "archive/single-model-routing/gate_7b_rag_axes", "feats"
DS = "PMC-VQA"
ARMS = ["nothink_norag", "think_norag", "think_rag_StatPearls", "think_rag_Textbooks"]
N_SPLITS, N_REPEATS, B_PERM, PCA_DIMS = 5, 10, 2000, 64
random.seed(42); np.random.seed(42)

pat = re.compile(r"ckpt_(.+?)_(" + "|".join(ARMS) + r")(?:_s\dof\d)?\.jsonl$")
lab = defaultdict(dict); raw = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(CKDIR, "*.jsonl"))):
    m = pat.search(os.path.basename(f))
    if not m or m.group(1) != DS: continue
    for l in open(f):
        if l.strip():
            r = json.loads(l); lab[m.group(2)][r["idx"]] = r["ok"]; raw[m.group(2)][r["idx"]] = r

idxs = sorted(set.intersection(*[set(lab[a]) for a in ARMS]))
Y = np.array([[lab[a][i] for a in ARMS] for i in idxs], int)
N = len(idxs)
print("="*68); print(f"PMC-VQA RECOVERABILITY CHECK   (N={N})"); print("="*68)

# ---- (1) oracle vs luck floor ----
acc = Y.mean(axis=0); oracle = Y.max(axis=1).mean(); best_fixed = acc.max()
def luck():
    hit = np.zeros(N, bool)
    for a in range(len(ARMS)):
        c = Y[:, a].copy(); np.random.shuffle(c); hit |= c.astype(bool)
    return hit.mean()
sims = [luck() for _ in range(B_PERM)]; mu, sd = np.mean(sims), np.std(sims)
z = (oracle - mu)/sd if sd else float('inf')
print("\n(1) ORACLE vs LUCK FLOOR")
for a,c in zip(ARMS, acc): print(f"     {a:24s} acc={c:.3f}")
print(f"     best-fixed = {best_fixed:.3f}   oracle = {oracle:.3f}")
print(f"     luck floor = {mu:.3f} +/- {sd:.3f}")
print(f"     >> oracle - luck = {oracle-mu:+.3f}  (z = {z:+.1f})")
print(f"     {'>>> ABOVE floor: real complementarity exists!' if z>2 else '>>> NOT above floor: redundant arms (same wall as MedXpert)'}")

# ---- (2)+(3) ----
def scalar_feats(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    g, pk = row.get("gen_tokens",0), row.get("parse_ok",0)
    if len(v) < 2: return [0,0,0, math.log(2), .5, len(v), g, pk]
    mx=v[0]; e=[math.exp(x-mx) for x in v]; Z=sum(e); p=[x/Z for x in e]
    ent=-sum(q*math.log(q+1e-12) for q in p)
    return [v[0], v[0]-v[1], v[0]-(v[2] if len(v)>2 else v[1]), ent, p[0], len(v), g, pk]
def load_hidden(pool):
    fs = glob.glob(os.path.join(FEATDIR, f"feat_{DS}_L*.npz"))
    if not fs: return None
    p=[np.load(f) for f in fs]; idx=np.concatenate([x["idx"] for x in p])
    h=np.concatenate([x["h_last" if pool=="last" else "h_mean"] for x in p])
    fmap={int(i):h[k] for k,i in enumerate(idx)}
    return np.stack([fmap[i] for i in idxs]) if set(idxs)<=set(fmap) else None
def evaluate(X, pca=False, shuffle=False):
    strat=Y.max(axis=1); res=defaultdict(list); auc=defaultdict(list)
    for rep in range(N_REPEATS):
        for tr,te in StratifiedKFold(N_SPLITS,shuffle=True,random_state=rep).split(X,strat):
            Xtr,Xte,Ytr,Yte=X[tr],X[te],Y[tr],Y[te]
            if shuffle: Ytr=Ytr[np.random.RandomState(rep).permutation(len(tr))]
            P=np.zeros((len(te),len(ARMS)))
            for a in range(len(ARMS)):
                ya=Ytr[:,a]
                if ya.min()==ya.max(): P[:,a]=float(ya.mean()); continue
                mdl=make_pipeline(StandardScaler(), PCA(min(PCA_DIMS,Xtr.shape[0]-1,Xtr.shape[1]),
                        svd_solver="randomized",random_state=0), LogisticRegression(max_iter=2000,C=0.5)) \
                    if pca else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,C=1.0))
                mdl.fit(Xtr,ya); P[:,a]=mdl.predict_proba(Xte)[:,1]
                if not shuffle and len(set(Yte[:,a]))>1: auc[ARMS[a]].append(roc_auc_score(Yte[:,a],P[:,a]))
            ch=P.argmax(axis=1)
            res["routed"].append(Yte[np.arange(len(te)),ch].mean())
            res["bestfixed"].append(Yte[:,Ytr.mean(0).argmax()].mean())
    return res,auc
def block(name, X, pca):
    res,auc=evaluate(X,pca); sh,_=evaluate(X,pca,shuffle=True)
    d=np.array(res["routed"])-np.array(res["bestfixed"])
    print(f"\n({name})")
    print("     per-arm AUROC: " + "  ".join(f"{a.split('_')[0][:4]}.{a.split('_')[-1][:3]}={np.mean(auc[a]):.2f}" for a in ARMS if auc[a]))
    print(f"     routed={np.mean(res['routed']):.3f}  best_fixed={np.mean(res['bestfixed']):.3f}")
    print(f"     >> routed - best_fixed = {np.mean(d):+.3f} +/- {np.std(d):.3f}")
    print(f"     >> shuffle control     = {np.mean(sh['routed']):.3f}")

print("\n(2)+(3) PREDICTABILITY + ROUTED GAIN")
X_sc = np.array([scalar_feats(raw[ARMS[0]][i]) for i in idxs], float)
block("scalar features", X_sc, pca=False)
for pool in ["last","mean"]:
    Xh = load_hidden(pool)
    if Xh is not None: block(f"hidden L14 {pool}-pool", Xh, pca=True)
    else: print(f"\n(hidden {pool}: feats not found)")

print("\n" + "="*68)
print("READ: (1) z>2 => complementarity on competent data (unlike MedXpert).")
print("(2)/(3) routed-best_fixed>0 beyond std + shuffle collapse => routable.")
print("If PMC-VQA ALSO flat/below-floor => single-model routing dead on this 7B,")
print("=> 32B cascade (different MODELS, real gap) is the path forward.")
