#!/usr/bin/env python3
"""CONTROLLED GATE SWAP: hold the verifier + best-of-8 SELECTION fixed; swap ONLY the escalation gate mechanism.
Does any gate improve over verifier-confidence? Reports the DEFERRAL CURVE (cascade acc vs escalation%), AUROC for
predicting pick-correctness, and area-under-deferral-curve (ADC). Escalate lowest-gate-score first."""
import os, json, re, string, argparse, math
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--adapter_dir",required=True); ap.add_argument("--tag",required=True)
ap.add_argument("--cheap_dir",required=True); ap.add_argument("--cheap_tag",required=True)
ap.add_argument("--strong_dir",required=True); ap.add_argument("--strong_tag",required=True)
ap.add_argument("--datasets",nargs="+",required=True); ap.add_argument("--N",type=int,default=8)
A=ap.parse_args()
def nrm(s):
    s=str(s).lower().strip(); s=re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b"," ",s)
    return re.sub(r"\s+"," ",s.translate(str.maketrans("","",string.punctuation))).strip()
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
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]; sco=r["scores"][:A.N]; preds=r["preds"][:A.N]
        pk=int(np.argmax(sco)); srt=sorted(sco,reverse=True)
        # answer-entropy over normalized answers (semantic-ish consistency)
        c=Counter(nrm(p) for p in preds); ps=np.array(list(c.values()),float)/len(preds); ent=-np.sum(ps*np.log(ps+1e-9))
        s=sc.get(i,{})
        rows.append({"pick":sl[pk],"strong":sj[i],
            "vconf":srt[0],"vmargin":srt[0]-(srt[1] if len(srt)>1 else 0),"vmean":float(np.mean(sco)),
            "vnegstd":-float(np.std(sco)),"vfrac_hi":float(np.mean(np.array(sco)>0.5)),
            "selfcons":s.get("self_consistency",np.nan),"n_distinct_neg":-(s.get("n_distinct",np.nan) or np.nan),
            "ans_entropy_neg":-ent,"c_conf":s.get("conf",np.nan),"c_margin":s.get("margin",np.nan),
            "c_seqlogprob":s.get("seqlogprob",np.nan),"feat":[srt[0],srt[0]-(srt[1] if len(srt)>1 else 0),float(np.mean(sco)),float(np.std(sco)),float(np.min(sco)),s.get("self_consistency",0) or 0,-(s.get("n_distinct",0) or 0),ent]})
n=len(rows); pick=np.array([r["pick"] for r in rows]); strong=np.array([r["strong"] for r in rows],float)
Xf=np.nan_to_num(np.array([r["feat"] for r in rows],float))
def oof(y,kind):
    o=np.zeros(n)
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xf,y):
        c=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)) if kind=="logit" else HistGradientBoostingClassifier(max_iter=200,max_depth=3)
        c.fit(Xf[tr],y[tr]); o[te]=c.predict_proba(Xf[te])[:,1]
    return o
def g(k): return np.array([r[k] for r in rows],float)
# gate score convention: HIGHER = more confident pick is right (escalate the LOWEST)
gates={"verifier-conf [current]":g("vconf"),"verifier-margin":g("vmargin"),"verifier-mean":g("vmean"),
       "verifier-frac>0.5":g("vfrac_hi"),"verifier-negstd":g("vnegstd"),
       "self-consistency":g("selfcons"),"-n_distinct":g("n_distinct_neg"),"-answer-entropy":g("ans_entropy_neg")}
if not np.all(np.isnan(g("c_conf"))): 
    gates["cheap-conf"]=g("c_conf"); gates["cheap-margin"]=g("c_margin"); gates["cheap-seqlogprob"]=g("c_seqlogprob")
gates["TRAINED-logit"]=oof(pick,"logit"); gates["TRAINED-gbm"]=oof(pick,"gbm")
gates["SOTA Diff-Prob(gbm)"]=-oof(((strong-pick)>0).astype(int),"gbm")  # high recoverability=escalate=>low keep-score
BUD=[0,.05,.10,.15,.20,.30,.40,.60,1.0]
def curve(score):
    order=np.argsort(score); accs=[]
    for B in BUD:
        k=int(round(B*n)); esc=np.zeros(n,bool); esc[order[:k]]=True
        accs.append(np.where(esc,strong,pick).mean())
    return accs
base=pick.mean(); T=strong.mean()
print(f"[{A.tag}] n={n} | verifier-bo{A.N} (esc0)={base:.3f} | always-STRONG={T:.3f} | selection FIXED, gate SWAPPED")
print(f"  {'GATE mechanism':<24}{'AUROC':>7}  " + "".join(f"{int(b*100):>5}%" for b in BUD) + "   ADC")
ref=None
for nm,scr in gates.items():
    try: au=roc_auc_score(pick,scr) if len(np.unique(pick))>1 else float('nan')
    except: au=float('nan')
    cv=curve(scr); adc=np.trapz(cv,BUD)  # area under deferral curve
    if nm.startswith("verifier-conf"): ref=adc
    print(f"  {nm:<24}{au:>7.3f}  " + "".join(f"{a:>6.3f}" for a in cv) + f"   {adc:.4f}")
print(f"  => ADC (area under deferral curve): higher=better gate. verifier-conf ADC={ref:.4f}. A gate IMPROVES only if ADC>this AND curve dominates.")
