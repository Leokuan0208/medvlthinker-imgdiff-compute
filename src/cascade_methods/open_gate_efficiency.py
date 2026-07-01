#!/usr/bin/env python3
"""EFFICIENCY-leg gate bake-off: at ISO-ACCURACY (match always-strong), which gate escalates LEAST -> lowest
FLOPs/latency/energy? Uses MEASURED batch-1 latency/energy (Lingshu). Verifier best-of-N base + gate escalation.
Gates: verifier-conf, margin, self-consistency, TRAINED logit/gbm on [verifier dist + cheap signals], and the
SOTA post-hoc TRAINED deferral rule (Jitkrittum Diff-Prob: predict P(strong right)-P(cheap right), OOF)."""
import os, json, math, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--adapter_dir",default="ckpts/train/lora_verifier_pooled4"); ap.add_argument("--tag",default="lingshu7b")
ap.add_argument("--cheap_dir",default="ckpts/openvqa/cheap_lingshu7b"); ap.add_argument("--cheap_tag",default="lingshu7b")
ap.add_argument("--strong_dir",default="ckpts/openvqa/strong_lingshu"); ap.add_argument("--strong_tag",default="lingshu32b")
ap.add_argument("--datasets",nargs="+",default=["vqa_rad_open","pathvqa_open","kvasir_open","radimagenet_open"])
ap.add_argument("--N",type=int,default=8)
A=ap.parse_args()
# MEASURED batch-1 (Lingshu): (latency_ms, energy_J, FLOPs_7Beq)
GEN7=(347,45.8,1.0); VER7=(175,25.3,1.0); GEN32=(665,127.0,4.57)
def loadj(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=r
    return m
def jload(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=int(r["judge_ok"])
    return m
def vf(scores,N):
    s=sorted(scores[:N],reverse=True); a=np.array(scores[:N],float)
    return [s[0],s[0]-(s[1] if len(s)>1 else 0),float(a.mean()),float(a.std()),float(a.min()),float((a>0.5).mean())]
rows=[]
for ds in A.datasets:
    dp=os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dp): continue
    sc=loadj(os.path.join(ROOT,A.cheap_dir,f"ckpt_{ds}_{A.cheap_tag}_sc8.jsonl")); sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl",f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=jload(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    for r in json.load(open(dp)):
        i=r["idx"]
        if i not in sj: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]; pk=int(np.argmax(r["scores"][:A.N]))
        f=vf(r["scores"],A.N); s=sc.get(i,{})
        f+=[s.get("self_consistency",np.nan),-(s.get("n_distinct",np.nan) or np.nan)]
        rows.append({"pick":sl[pk],"strong":sj[i],"vconf":max(r["scores"][:A.N]),"vmargin":f[1],
                     "sc":s.get("self_consistency",0),"feat":f})
n=len(rows); pick=np.array([r["pick"] for r in rows]); strong=np.array([r["strong"] for r in rows],float)
Xf=np.nan_to_num(np.array([r["feat"] for r in rows],float))
def oof(y,kind):
    o=np.zeros(n)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,y):
        c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")) if kind=="logit" else HistGradientBoostingClassifier(max_iter=200,max_depth=3,class_weight="balanced")
        c.fit(Xf[tr],y[tr]); o[te]=c.predict_proba(Xf[te])[:,1]
    return o
diffpos=((strong- pick)>0).astype(int)  # Diff-Prob target (recoverability)
gates={"verifier-conf":-np.array([r["vconf"] for r in rows]),  # escalate LOW conf (higher gate=escalate)
       "margin":-np.array([r["vmargin"] for r in rows]),
       "self-consistency":-np.array([r["sc"] for r in rows]),
       "TRAINED-logit(pickok)":-oof(pick,"logit"),
       "TRAINED-gbm(pickok)":-oof(pick,"gbm"),
       "SOTA Diff-Prob(gbm)":oof(diffpos,"gbm")}   # escalate HIGH predicted-recoverability
def cost(esc_frac,N):
    fl=N*GEN7[2]+N*VER7[2]+esc_frac*GEN32[2]
    en=N*GEN7[1]+N*VER7[1]+esc_frac*GEN32[1]
    lat_seq=N*GEN7[0]+N*VER7[0]+esc_frac*GEN32[0]; lat_bat=GEN7[0]+VER7[0]+esc_frac*GEN32[0]
    return fl,en,lat_seq,lat_bat
T=strong.mean()   # iso-accuracy target = always-strong
print(f"[{A.tag}] n={n} | verifier-bo{A.N}={pick.mean():.3f} | always-STRONG={T:.3f} (target) | oracle-not-shown")
print(f"  always-STRONG cost: FLOPs={GEN32[2]:.2f}  energy={GEN32[1]:.0f}J  latency={GEN32[0]}ms")
print(f"  {'gate':<24}{'min-esc@iso-acc':>16}{'FLOPs':>8}{'energy(J)':>11}{'lat_seq/bat(ms)':>18}")
for nm,sc in gates.items():
    order=np.argsort(-sc)  # escalate highest-gate first
    best=None
    for e in range(0,n+1):
        esc=np.zeros(n,bool); esc[order[:e]]=True
        acc=np.where(esc,strong,pick).mean()
        if acc>=T-1e-9: best=e/n; break
    if best is None: best=1.0
    fl,en,ls,lb=cost(best,A.N)
    print(f"  {nm:<24}{best*100:>14.0f}%{fl:>8.1f}{en:>11.0f}{f'{ls:.0f}/{lb:.0f}':>18}")
print(f"  (N={A.N} base cost before escalation: FLOPs={A.N*2:.0f}, energy={A.N*(GEN7[1]+VER7[1]):.0f}J, lat_bat={GEN7[0]+VER7[0]}ms)")
