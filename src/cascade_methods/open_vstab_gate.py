#!/usr/bin/env python3
"""3-family FAITHFUL CASP/CCPS test: does input-perturbation visual-stability add gate signal beyond the verifier?
Parameterized by family. Stability = cheap model's answer agreement across image resolutions (cap320 vs cap160/cap80)."""
import os, json, re, string, argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--adapter_dir",required=True); ap.add_argument("--tag",required=True)
ap.add_argument("--cheap_dir",required=True); ap.add_argument("--cheap_tag",required=True)
ap.add_argument("--perturb_dir",required=True)
ap.add_argument("--strong_dir",required=True); ap.add_argument("--strong_tag",required=True)
ap.add_argument("--datasets",nargs="+",required=True)
A=ap.parse_args()
def norm(s):
    s=str(s).lower().strip(); s=re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b"," ",s)
    s=s.translate(str.maketrans("","",string.punctuation)); return re.sub(r"\s+"," ",s).strip()
def loadj(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=r
    return m
def judge(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=int(r["judge_ok"])
    return m
rows=[]
for ds in A.datasets:
    dp=os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dp): continue
    dump=json.load(open(dp))
    sc=loadj(os.path.join(ROOT,A.cheap_dir,f"ckpt_{ds}_{A.cheap_tag}_sc8.jsonl"))
    c160=loadj(os.path.join(ROOT,A.perturb_dir,f"ckpt_{ds}_{A.cheap_tag}_cap160.jsonl"))
    c80=loadj(os.path.join(ROOT,A.perturb_dir,f"ckpt_{ds}_{A.cheap_tag}_cap80.jsonl"))
    sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=judge(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    for r in dump:
        i=r["idx"]
        if i not in sj or i not in sc or i not in c160 or i not in c80: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]; pk=int(np.argmax(r["scores"][:8]))
        a320=norm(sc[i]["modal_pred"]); a160=norm(c160[i]["modal_pred"]); a80=norm(c80[i]["modal_pred"])
        rows.append({"vconf":max(r["scores"][:8]),"st160":int(a320==a160),"st80":int(a320==a80),
                     "stfrac":(int(a320==a160)+int(a320==a80))/2.0,"pick_ok":sl[pk]})
n=len(rows)
if n==0: print(f"{A.tag}: no rows (perturb data missing?)"); raise SystemExit
y=np.array([r["pick_ok"] for r in rows])
def X(c): return np.array([[r[k] for k in c] for r in rows],float)
def oof(c):
    Xa=X(c); o=np.zeros(n); 
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(Xa,y):
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000)); clf.fit(Xa[tr],y[tr]); o[te]=clf.predict_proba(Xa[te])[:,1]
    return o
def au(s): return roc_auc_score(y,s) if len(np.unique(y))>1 else float("nan")
print(f"[{A.tag}] n={n} pick-acc={y.mean():.3f} | agree@160={np.mean([r['st160'] for r in rows]):.3f} @80={np.mean([r['st80'] for r in rows]):.3f}")
print(f"   verifier-conf={au(X(['vconf'])[:,0]):.3f} | vstab-frac(pure CASP)={au(X(['stfrac'])[:,0]):.3f} | vstab-trained={au(oof(['st160','st80'])):.3f} | verifier+vstab={au(oof(['vconf','st160','st80'])):.3f}")
