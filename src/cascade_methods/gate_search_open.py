#!/usr/bin/env python3
"""
gate_search_open.py - hunt for an open-ended routing signal that ROBUSTLY beats confidence, i.e. on
the CALIBRATED cascade Lingshu-7B -> Lingshu-32B (confidence cheap-wrong AUROC 0.866, recover 0.804),
not just the miscalibrated MedVLThinker-7B one. Offline from the existing K=8 samples + seqlogprob.

Signals (all from the cheap model's own outputs):
  confidence       : -mean seq-logprob (temp-0 answer)            [the baseline to beat]
  exact-SC         : largest exact-normalized cluster fraction (the crude self-consistency I used)
  semantic-SC      : largest SEMANTIC cluster fraction (cluster K samples by token-F1>=0.5; merges
                     paraphrases like 'CT'='computed tomography' that exact-SC splits)
  semantic-entropy : Kuhn/Farquhar-style entropy over semantic clusters (lower=more certain)
  mean-pairwise-F1 : average pairwise token-F1 among the K samples (soft agreement)
  FUSION           : logistic(confidence, semantic-SC, semantic-entropy, mean-F1), honest 50/50 calib/test
Reports AUROC for cheap-wrong + recoverable vs confidence; FUSION uses 20-seed calib/test (no leakage).
  python3 src/cascade_methods/gate_search_open.py            # Lingshu-7B->Lingshu-32B (calibrated; the bar)
  python3 src/cascade_methods/gate_search_open.py --xfam     # MedVLThinker-7B->Lingshu-32B (miscalibrated)
"""
import os, sys, json, re, string
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
XFAM = "--xfam" in sys.argv
DSETS = ["slake_open", "vqa_rad_open"]
STRONG = "ckpts/openvqa/strong_lingshu"; STAG = "lingshu32b"
if XFAM: CHEAP, T0, SC, VTAG = "ckpts/openvqa/cheap", "7b_t0", "7b_sc8", "7b_verify"
else:    CHEAP, T0, SC, VTAG = "ckpts/openvqa/cheap_lingshu7b", "lingshu7b", "lingshu7b_sc8", "lingshu7b_verify"
def norm(s):
    s=str(s).lower().strip(); s=re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b"," ",s)
    s=s.translate(str.maketrans("","",string.punctuation)); return re.sub(r"\s+"," ",s).strip()
def f1(a,b):
    pa,pb=set(norm(a).split()),set(norm(b).split())
    if not pa or not pb: return 0.0
    ov=len(pa&pb)
    if not ov: return 0.0
    p,r=ov/len(pa),ov/len(pb); return 2*p*r/(p+r)
def load(p):
    m={}
    for l in open(p):
        if l.strip(): r=json.loads(l); m[r["idx"]]=r
    return m
def auroc(score,y):
    score=np.asarray(score,float); y=np.asarray(y,int); pos,neg=score[y==1],score[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    allv=np.concatenate([pos,neg]); order=allv.argsort(); ranks=np.empty(len(allv)); ranks[order]=np.arange(1,len(allv)+1)
    u,inv,cnt=np.unique(allv,return_inverse=True,return_counts=True); s=np.zeros(len(cnt)); np.add.at(s,inv,ranks); ranks=(s/cnt)[inv]
    return (ranks[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def sem_clusters(preds):
    """greedy semantic clustering of K samples by token-F1>=0.5; return cluster sizes."""
    reps=[]; sizes=[]
    for p in preds:
        placed=False
        for i,rep in enumerate(reps):
            if f1(p,rep)>=0.5: sizes[i]+=1; placed=True; break
        if not placed: reps.append(p); sizes.append(1)
    return sizes

rows=[]; HAVE_V=os.path.exists(f"{CHEAP}/ckpt_slake_open_{VTAG}.jsonl")
for ds in DSETS:
    t0=load(f"{CHEAP}/ckpt_{ds}_{T0}.jsonl"); sc=load(f"{CHEAP}/ckpt_{ds}_{SC}.jsonl"); st=load(f"{STRONG}/ckpt_{ds}_{STAG}.jsonl")
    vf=load(f"{CHEAP}/ckpt_{ds}_{VTAG}.jsonl") if HAVE_V else {}
    for i in set(t0)&set(sc)&set(st):
        preds=sc[i].get("preds",[]); K=len(preds) or 1
        sizes=sem_clusters(preds) if preds else [1]; ps=np.array(sizes)/K
        semSC=max(sizes)/K; semH=float(-(ps*np.log(ps+1e-12)).sum())
        mpf=np.mean([f1(preds[a],preds[b]) for a in range(len(preds)) for b in range(a+1,len(preds))]) if len(preds)>1 else 1.0
        py=(vf.get(i,{}) or {}).get("p_yes_norm"); ptrue=(1.0-py) if py is not None else 0.5
        rows.append(dict(cw=1-t0[i]["modal_ok"], rec=int(t0[i]["modal_ok"]==0 and st[i]["modal_ok"]==1),
            conf=-(t0[i].get("seqlogprob") or 0.0), exactSC=-sc[i]["self_consistency"],
            semSC=-semSC, semH=semH, mpf=-mpf, ptrue=ptrue))
print(f"{'MedVLThinker-7B' if XFAM else 'Lingshu-7B'} -> Lingshu-32B   n={len(rows)}")
cw=np.array([r["cw"] for r in rows]); rec=np.array([r["rec"] for r in rows])
SIGS=["conf","exactSC","semSC","semH","mpf"]+(["ptrue"] if HAVE_V else [])
print(f"  {'signal':<14}{'AUROC cheap-wrong':>20}{'AUROC recoverable':>20}")
base_cw=auroc([r["conf"] for r in rows],cw); base_rec=auroc([r["conf"] for r in rows],rec)
for s in SIGS:
    a=auroc([r[s] for r in rows],cw); b=auroc([r[s] for r in rows],rec)
    tag=" <- baseline" if s=="conf" else (f"  cw{a-base_cw:+.3f} rec{b-base_rec:+.3f}")
    print(f"  {s:<14}{a:>20.3f}{b:>20.3f}{tag}")
# FUSION: logistic(conf, semSC, semH, mpf [, ptrue]), honest 50/50 calib/test x20 seeds
FF=["conf","semSC","semH","mpf"]+(["ptrue"] if HAVE_V else [])
X=np.column_stack([[r[s] for r in rows] for s in FF]); n=len(rows)
def cv_auroc(y):
    aus=[]
    for seed in range(20):
        rng=np.random.default_rng(seed); idx=rng.permutation(n); tr,te=idx[:n//2],idx[n//2:]
        if len(np.unique(y[tr]))<2: continue
        m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(X[tr],y[tr])
        aus.append(auroc(m.predict_proba(X[te])[:,1],y[te]))
    return np.mean(aus),np.std(aus)
fcw=cv_auroc(cw); frec=cv_auroc(rec)
print(f"  {'FUSION (cv)':<14}{fcw[0]:>20.3f}{frec[0]:>20.3f}   cw{fcw[0]-base_cw:+.3f} rec{frec[0]-base_rec:+.3f}  (honest 20-seed cv)")
print(f"\n  BAR = confidence: cheap-wrong {base_cw:.3f}, recoverable {base_rec:.3f}. A signal/fusion ABOVE both")
print("  (especially on Lingshu-7B, the calibrated case) is a ROBUST gate beating confidence.")
