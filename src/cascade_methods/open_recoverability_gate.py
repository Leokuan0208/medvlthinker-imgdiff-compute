#!/usr/bin/env python3
"""THE DECISIVE TEST (Jitkrittum NeurIPS'23, arXiv:2307.02764): the OPTIMAL cascade gate thresholds
RECOVERABILITY d(x)=P(strong correct)-P(small correct), NOT confidence. Can ANY trained gate predict it?
We train gates (logit/gbm) on ALL available signals (verifier 8-score distribution + cheap consistency) to
predict the diff target, cross-fit (5-fold OOF), and report:
  (a) AUROC for RECOVERABILITY = 1[strong_ok=1 & pick_ok=0]  (the rows where escalation strictly GAINS);
  (b) the cascade acc-vs-escalation frontier of the trained recoverability gate vs verifier-conf vs ORACLE
      (escalate exactly the gain rows). If trained ~ verifier-conf << oracle, NO gate closes the wall.
"""
import argparse, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--adapter_dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--cheap_dir", required=True); ap.add_argument("--cheap_tag", required=True)
ap.add_argument("--strong_dir", required=True); ap.add_argument("--strong_tag", required=True)
ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--N", type=int, default=8)
A=ap.parse_args()
def loadj(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=r
    return m
def load_judge(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=int(r["judge_ok"])
    return m
def vfeats(scores,N):
    s=sorted(scores[:N],reverse=True); a=np.array(scores[:N],float)
    return [s[0], s[0]-(s[1] if len(s)>1 else 0), float(a.mean()), float(a.std()), float(a.min()), float((a>0.5).mean())]
rows=[]
for ds in A.datasets:
    dpath=os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dpath): continue
    dump=json.load(open(dpath)); sc=loadj(os.path.join(ROOT,A.cheap_dir,f"ckpt_{ds}_{A.cheap_tag}_sc8.jsonl"))
    sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=load_judge(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    for r in dump:
        i=r["idx"]
        if i not in sj: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]; N=A.N; pk=int(np.argmax(r["scores"][:N]))
        f=vfeats(r["scores"],N); s=sc.get(i,{})
        f+=[s.get("self_consistency",np.nan), -(s.get("n_distinct",np.nan) or np.nan), s.get("gen_tokens",np.nan)]
        for k in ("conf","margin","seqlogprob"):   # fixed-length slots (NaN if absent -> nan_to_num zeroes)
            v=s.get(k); f.append(v if v is not None else np.nan)
        rows.append((f, sl[pk], sj[i]))
X=np.nan_to_num(np.array([r[0] for r in rows],float)); pick=np.array([r[1] for r in rows]); strong=np.array([r[2] for r in rows])
n=len(rows)
gain=((strong==1)&(pick==0)).astype(int)   # escalation strictly GAINS
lose=((strong==0)&(pick==1)).astype(int)    # escalation strictly LOSES
diff=strong.astype(int)-pick.astype(int)     # +1 gain, -1 lose, 0 neutral
print(f"\n==== {A.tag} | n={n} | pick-acc={pick.mean():.3f} strong-acc={strong.mean():.3f} | gain-rows={gain.mean():.3f} lose-rows={lose.mean():.3f} ====")
print(f"   oracle cascade (escalate exactly gain rows): acc={np.where(gain==1,strong,pick).mean():.3f} @esc={gain.mean()*100:.0f}%")
# trained recoverability gate: regress diff (escalate when predicted diff high)
def oof_diff(kind):
    oof=np.zeros(n); skf=StratifiedKFold(5,shuffle=True,random_state=0)
    yb=(diff>0).astype(int)  # stratify on gain
    for tr,te in skf.split(X,yb):
        if kind=="logit": clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced"))
        else: clf=HistGradientBoostingClassifier(max_iter=250,max_depth=3,learning_rate=0.06,class_weight="balanced")
        clf.fit(X[tr],yb[tr]); oof[te]=clf.predict_proba(X[te])[:,1]
    return oof
vconf=X[:,0]  # verifier-conf (escalate LOW)
gates={"verifier-conf (escalate low)": -vconf,
       "TRAINED-recover logit": oof_diff("logit"),
       "TRAINED-recover gbm": oof_diff("gbm")}
print(f"  {'gate (higher=escalate)':<28}{'AUROC(gain)':>12}{'AUROC(diff>0)':>14}   cascade-acc @ esc 10/20/30/40%")
def casc(score,B):
    k=int(round(B*n)); 
    if k==0: return pick.mean()
    esc=np.zeros(n,bool); esc[np.argsort(-score)[:k]]=True  # escalate HIGHEST score
    return np.where(esc,strong,pick).mean()
for name,sc in gates.items():
    try:
        au_g=roc_auc_score(gain, sc) if gain.sum()>0 else float("nan")
        au_d=roc_auc_score((diff>0).astype(int), sc)
    except Exception: au_g=au_d=float("nan")
    accs=[casc(sc,B) for B in (0.1,0.2,0.3,0.4)]
    print(f"  {name:<28}{au_g:>12.3f}{au_d:>14.3f}   "+" ".join(f"{a:.3f}" for a in accs))
print(f"  => if TRAINED-recover AUROC(gain) ~= 0.5 and ~= verifier-conf, NO gate closes the wall (Jitkrittum noise/saturation regime).")
