#!/usr/bin/env python3
"""OPEN-TEXT gate x verifier bake-off: does a TRAINED gate beat the verifier-confidence gate?
Per question we have the trained-verifier's 8 answer-scores (dump) + cheap self-consistency signals (sc8).
Target for the escalation gate = verifier_pick_ok (1 if the cheap best-of-N pick is correct => KEEP; else escalate).
Gate quality compared 2 honest ways:
  (1) AUROC for predicting verifier_pick_ok  (threshold-free, 5-fold OUT-OF-FOLD for trained gates);
  (2) cascade accuracy at FIXED escalation budgets (escalate the bottom-B% by gate score -> use STRONG there),
      using OOF scores for trained gates (no leakage). Budgets 0/10/20/30/40/100%.
Single-signal gates (training-free): verifier-confidence(max score), verifier-margin(top1-top2), self_consistency,
  -n_distinct, and (if present) cheap conf/margin/seqlogprob. Trained gates (logistic/GBM/MLP) on feature sets
  {verif-only, cheap-only, verif+cheap}. Also reports the recoverability AUROC (the 'wall').
"""
import argparse, os, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
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
def vfeats(scores, N):
    s=sorted(scores[:N], reverse=True); a=np.array(scores[:N],float)
    top1=s[0]; top2=s[1] if len(s)>1 else 0.0
    return {"v_max":top1, "v_margin":top1-top2, "v_mean":float(a.mean()), "v_std":float(a.std()),
            "v_min":float(a.min()), "v_range":top1-float(a.min()), "v_frac_hi":float((a>0.5).mean())}
# ---- assemble rows ----
rows=[]  # dict per question
for ds in A.datasets:
    dump=json.load(open(os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json"))) if os.path.exists(os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")) else []
    sc=loadj(os.path.join(ROOT,A.cheap_dir,f"ckpt_{ds}_{A.cheap_tag}_sc8.jsonl"))
    sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=load_judge(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    for r in dump:
        i=r["idx"]
        if i not in sj: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]
        N=A.N; pickN=int(np.argmax(r["scores"][:N]))
        feat=vfeats(r["scores"], N)
        s=sc.get(i,{})
        feat["self_consistency"]=s.get("self_consistency", np.nan)
        feat["n_distinct_neg"]=-(s.get("n_distinct", np.nan) or np.nan)
        feat["gen_tokens"]=s.get("gen_tokens", np.nan)
        for k in ("conf","margin","seqlogprob"):
            if s.get(k) is not None: feat["c_"+k]=s.get(k)
        feat["_pick_ok"]=sl[pickN]; feat["_strong_ok"]=sj[i]
        rows.append(feat)
n=len(rows)
CHEAP=[k for k in ("self_consistency","n_distinct_neg","gen_tokens","c_conf","c_margin","c_seqlogprob") if k in rows[0]]
VERIF=["v_max","v_margin","v_mean","v_std","v_min","v_range","v_frac_hi"]
y=np.array([r["_pick_ok"] for r in rows]); strong=np.array([r["_strong_ok"] for r in rows],float)
print(f"\n==== {A.tag} | n={n} | verifier-pick-acc(no-gate)={y.mean():.3f} | always-STRONG={strong.mean():.3f} ====")
print(f"   cheap signals available: {CHEAP}")
def X(cols): return np.nan_to_num(np.array([[r.get(c,np.nan) for c in cols] for r in rows],float))
def oof_scores(cols, kind):
    Xa=X(cols); oof=np.zeros(n); skf=StratifiedKFold(5,shuffle=True,random_state=0)
    for tr,te in skf.split(Xa,y):
        if kind=="logit": clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000))
        elif kind=="gbm": clf=HistGradientBoostingClassifier(max_iter=200,max_depth=3,learning_rate=0.06)
        else: clf=make_pipeline(StandardScaler(),MLPClassifier((64,32),max_iter=400,random_state=0))
        clf.fit(Xa[tr],y[tr]); oof[te]=clf.predict_proba(Xa[te])[:,1]
    return oof
# ---- gate scores: HIGHER = keep (more confident pick is right); escalate the LOWEST ----
gates={}
gates["verifier-conf (max)  [TF]"]=X(["v_max"])[:,0]
gates["verifier-margin      [TF]"]=X(["v_margin"])[:,0]
gates["self-consistency     [TF]"]=X(["self_consistency"])[:,0]
gates["-n_distinct          [TF]"]=X(["n_distinct_neg"])[:,0]
if "c_conf" in rows[0]: gates["cheap-conf           [TF]"]=X(["c_conf"])[:,0]
if "c_margin" in rows[0]: gates["cheap-margin         [TF]"]=X(["c_margin"])[:,0]
if "c_seqlogprob" in rows[0]: gates["cheap-seqlogprob     [TF]"]=X(["c_seqlogprob"])[:,0]
for kind in ("logit","gbm","mlp"):
    gates[f"TRAINED-{kind} verif-only"]=oof_scores(VERIF,kind)
    gates[f"TRAINED-{kind} cheap-only"]=oof_scores(CHEAP,kind)
    gates[f"TRAINED-{kind} verif+cheap"]=oof_scores(VERIF+CHEAP,kind)
# ---- AUROC for predicting pick_ok ----
print(f"\n  {'GATE':<30}{'AUROC(pick_ok)':>15}   cascade-acc @ escalation budget")
print(f"  {'':<30}{'':>15}   {'10%':>7}{'20%':>7}{'30%':>7}{'40%':>7}")
budgets=[0.10,0.20,0.30,0.40]
def casc_at_budget(score, B):
    k=int(round(B*n)); 
    if k==0: return y.mean()
    esc=np.zeros(n,bool); esc[np.argsort(score)[:k]]=True  # escalate the LOWEST-score (least confident)
    return np.where(esc, strong, y).mean()
results={}
for name,sc in gates.items():
    try: au=roc_auc_score(y, sc) if len(np.unique(y))>1 else float("nan")
    except Exception: au=float("nan")
    accs=[casc_at_budget(sc,B) for B in budgets]
    results[name]=(au,accs)
    print(f"  {name:<30}{au:>15.3f}   "+ "".join(f"{a:>7.3f}" for a in accs))
print(f"  {'always-STRONG (esc100%)':<30}{'':<15}   strong-acc={strong.mean():.3f}  | no-gate(esc0%)={y.mean():.3f}")
# recoverability wall: among pick-wrong rows, predict strong-right
wrong=y==0
if wrong.sum()>10 and len(np.unique(strong[wrong]))>1:
    rec=strong[wrong]
    auc_vc=roc_auc_score(rec, -X(["v_max"])[wrong,0])  # low verif-conf -> more recoverable?
    # trained recoverability on verif+cheap (OOF among wrong)
    print(f"\n  RECOVERABILITY 'wall' (among pick-wrong, predict strong-right): n_wrong={wrong.sum()} strong-right-rate={rec.mean():.3f}")
    print(f"    verifier-conf AUROC for recoverability = {auc_vc:.3f}   (MCQ lesson: ~0.6 = hard wall)")
