#!/usr/bin/env python3
"""
grid_resolution_tau.py - calibrated resolution x tau grid (Task 5).
5x5: tau fit at TRAIN-cap X (pmctrain-at-X, router_train.py pipeline) x applied at INFER-cap Y
(cap-Y 7B margins). Each cell, pooled over all six benchmarks: cascade accuracy, escalation, and
PREFILL-INCLUSIVE backbone% -- 7B cheap leg priced at cap-Y prompt tokens, 32B strong leg at
full-res. (fullres,fullres) must reproduce router_cost_prefill's backbone% (anchor).
Needs ckpts/token_cache.json. CPU only.
"""
import json, glob, os, re
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

os.environ.setdefault("OMP_NUM_THREADS","1"); np.random.seed(42)
N7, N32 = 7.6e9, 33.0e9
SIX  = ["PMC-VQA","SLAKE","VQA-RAD","PathVQA","MMMU","MedXpert-Reasoning","MedXpert-Understanding"]
CAPS = ["fullres","cap640","cap320","cap160","cap80"]
EVAL_DIR = dict({"fullres":"ckpts/gate_7b_vllm"}, **{c:f"ckpts/gate_7b_prune/{c}" for c in CAPS[1:]})
PMCT_DIR = dict({"fullres":"ckpts/gate_7b_pmctrain"}, **{c:f"ckpts/gate_7b_pmctrain_prune/{c}" for c in CAPS[1:]})
DIR_32B  = "ckpts/gate_32b"
CACHE    = "ckpts/token_cache.json"

def load_arm(ckdir, cell):
    pat=re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$"); d=defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir,f"*{cell}*.jsonl")):
        m=pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                try: r=json.loads(l); d[m.group(1)][r["idx"]]=r
                except Exception: pass
    return d
def load_pmctrain(ckdir):
    rows=[]
    for f in glob.glob(os.path.join(ckdir,"ckpt_nothink*.jsonl")):
        for l in open(f):
            if l.strip():
                try: rows.append(json.loads(l))
                except Exception: pass
    return rows
def margin(row):
    lp=row.get("opt_logprobs") or {}; v=sorted(lp.values(),reverse=True)
    return (v[0]-v[1]) if len(v)>=2 else 0.0
def pick_tau(P,y):
    err=1.0-float(np.mean(y)); return float(np.quantile(P,min(max(err,0.0),1.0)))

def run(repo):
    J=lambda p: os.path.join(repo,p)
    cache=json.load(open(J(CACHE)))
    r32=load_arm(J(DIR_32B),"think_norag")
    ev={c:load_arm(J(EVAL_DIR[c]),"nothink_norag") for c in CAPS}

    gates={}; taus={}
    for X in CAPS:
        tr=load_pmctrain(J(PMCT_DIR[X]))
        y=np.array([r["ok"] for r in tr],float); m=np.array([[margin(r)] for r in tr],float)
        g=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=1.0)); g.fit(m,y)
        gates[X]=g; taus[X]=pick_tau(g.predict_proba(m)[:,1],y)
    print("per-train-cap tau:", {X:round(taus[X],3) for X in CAPS}, "\n")

    def cap_arrays(Y):
        out={}
        for ds in SIX:
            if ds not in r32 or ds not in ev[Y]: continue
            cY=cache.get(ds,{}).get(Y,{}); cF=cache.get(ds,{}).get("fullres",{})
            idx=sorted(set(ev[Y][ds]) & set(r32[ds]) & {int(k) for k in cY} & {int(k) for k in cF})
            if not idx: continue
            out[ds]=(np.array([margin(ev[Y][ds][i]) for i in idx]),
                     np.array([float(ev[Y][ds][i].get("gen_tokens") or 0) for i in idx]),
                     np.array([ev[Y][ds][i]["ok"] for i in idx],float),
                     np.array([cY[str(i)][0] for i in idx],float),
                     np.array([cF[str(i)][0] for i in idx],float),
                     np.array([float(r32[ds][i].get("gen_tokens") or 0) for i in idx]),
                     np.array([r32[ds][i]["ok"] for i in idx],float))
        return out
    CA={Y:cap_arrays(Y) for Y in CAPS}

    a32_all=np.concatenate([CA["fullres"][ds][6] for ds in CA["fullres"]])
    ref=a32_all.mean()
    print(f"always-32B pooled acc (parity reference) = {ref:.3f}  (n={len(a32_all)})")
    print("always-7B pooled acc by infer cap: " +
          "  ".join(f"{Y}={np.concatenate([CA[Y][ds][2] for ds in CA[Y]]).mean():.3f}" for Y in CAPS) + "\n")

    def cell(X,Y):
        g=gates[X]; tau=taus[X]; num7=num32=den=0.0; rok=[]; esca=[]
        for ds,(mg,g7,a7,PY,PF,g32,a32) in CA[Y].items():
            esc=g.predict_proba(mg.reshape(-1,1))[:,1] < tau
            rok.append(np.where(esc,a32,a7)); esca.append(esc)
            num7 += (2*N7*(PY+g7)).sum()
            num32+= (2*N32*(PF+g32))[esc].sum()
            den  += (2*N32*(PF+g32)).sum()
        return np.concatenate(rok).mean(), np.concatenate(esca).mean(), (num7+num32)/den

    M={(X,Y):cell(X,Y) for X in CAPS for Y in CAPS}
    def matrix(i,title,fmt):
        print(title); print("  trainX\\inferY"+"".join(f"{Y:>9}" for Y in CAPS))
        for X in CAPS: print(f"  {X:<12}"+"".join(fmt(M[(X,Y)][i]) for Y in CAPS))
        print()
    matrix(0,"=== cascade accuracy (pooled over six) ===",lambda v:f"{v:>9.3f}")
    matrix(1,"=== escalation % ===",lambda v:f"{100*v:>8.0f}%")
    matrix(2,"=== backbone% (prefill-incl, % of always-32B) ===",lambda v:f"{100*v:>8.0f}%")

    a=M[("fullres","fullres")]
    print(f"ANCHOR (fullres,fullres): acc={a[0]:.3f} esc={100*a[1]:.0f}% backbone={100*a[2]:.0f}%  <- must match router_cost_prefill")
    par=[((X,Y),v) for (X,Y),v in M.items() if v[0]>=ref-0.005]
    if par:
        (bX,bY),bv=min(par,key=lambda t:t[1][2])
        print(f"Best at parity (acc>={ref-0.005:.3f}): train={bX} infer={bY}  acc={bv[0]:.3f} esc={100*bv[1]:.0f}% backbone={100*bv[2]:.0f}%")
    print("\nREAD: rows=where tau trained, cols=where served. Compare a column to its diagonal: stable")
    print("down a column => tau transfers across inference resolution. Cost falls left->right.")

if __name__=="__main__":
    run(os.path.expanduser("~/medvlthinker-imgdiff-compute"))
