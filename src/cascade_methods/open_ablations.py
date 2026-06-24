#!/usr/bin/env python3
"""
open_ablations.py - ablations for the open-ended ceiling-break finding (§5.7), on the calibrated cascade
Lingshu-7B -> Lingshu-32B, LLM-judge labels, SLAKE+VQA-RAD+PathVQA pooled (n=2345).
(1) K-ABLATION: self-consistency routing AUROC vs #samples K (its cost) against the confidence baseline.
(2) ORACLE GAP: confidence-gated cascade accuracy vs the best-possible (oracle) routing and random, across
    the escalation budget -> how close is confidence to optimal routing.
Emits paper/figs/open/fig_open_ablations.png + prints numbers. Offline.
"""
import os, json, re, string
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CHEAP="ckpts/openvqa/cheap_lingshu7b"; STRONG="ckpts/openvqa/strong_lingshu"
DS=["slake_open","vqa_rad_open","pathvqa_open"]
def _norm(s):
    s=str(s).lower().strip(); s=re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b"," ",s)
    s=s.translate(str.maketrans("","",string.punctuation)); return re.sub(r"\s+"," ",s).strip()
def load(p):
    m={}
    for l in open(p):
        if l.strip(): r=json.loads(l); m[r["idx"]]=r
    return m
def judged(d,jp):
    if os.path.exists(jp):
        j={r["idx"]:r["judge_ok"] for r in (json.loads(l) for l in open(jp) if l.strip())}
        for i,r in d.items():
            if i in j: r["modal_ok"]=j[i]
    return d
def auroc(score,y):
    score=np.asarray(score,float); y=np.asarray(y,int); pos,neg=score[y==1],score[y==0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    a=np.concatenate([pos,neg]); o=a.argsort(); rk=np.empty(len(a)); rk[o]=np.arange(1,len(a)+1)
    u,inv,c=np.unique(a,return_inverse=True,return_counts=True); s=np.zeros(len(c)); np.add.at(s,inv,rk); rk=(s/c)[inv]
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def sc_at_K(preds,K):
    from collections import Counter
    c=Counter(_norm(p) for p in preds[:K]); return c.most_common(1)[0][1]/K

rows=[]; byds={}
for ds in DS:
    t0=judged(load(f"{CHEAP}/ckpt_{ds}_lingshu7b.jsonl"), f"{CHEAP}/ckpt_{ds}_lingshu7b.judge.jsonl")
    sc=load(f"{CHEAP}/ckpt_{ds}_lingshu7b_sc8.jsonl")
    st=judged(load(f"{STRONG}/ckpt_{ds}_lingshu32b.jsonl"), f"{STRONG}/ckpt_{ds}_lingshu32b.judge.jsonl")
    byds[ds]=[]
    for i in set(t0)&set(sc)&set(st):
        r=dict(cw=1-t0[i]["modal_ok"], cheap_ok=t0[i]["modal_ok"], strong_ok=st[i]["modal_ok"],
            conf=-(t0[i].get("seqlogprob") or 0.0), preds=sc[i].get("preds",[]))
        rows.append(r); byds[ds].append(r)
n=len(rows); cw=np.array([r["cw"] for r in rows])
print(f"n={n} (calibrated Lingshu-7B->Lingshu-32B, LLM-judge)")
# per-dataset routing EFFICIENCY: of the achievable (oracle - always-cheap) accuracy gain, how much does
# the confidence gate capture? (high where a real model-gap exists; ~0 where the gap is tiny, e.g. PathVQA)
def efficiency(rs):
    co=np.array([r["cheap_ok"] for r in rs]); so=np.array([r["strong_ok"] for r in rs]); cf=np.array([r["conf"] for r in rs])
    m=len(rs); base=co.mean(); strong=so.mean(); g=so-co
    best=0.0; bestc=0.0
    for b in np.linspace(0,1,21):
        k=int(round(b*m))
        eo=np.zeros(m,bool); eo[np.argsort(g)[::-1][:k]]=True; orc=np.where(eo,so,co).mean()
        ec=np.zeros(m,bool)
        if k>0: ec[np.argsort(cf)[::-1][:k]]=True
        cda=np.where(ec,so,co).mean()
        if orc-base>best-1e-12 and orc>base: best=orc-base; bestc=cda-base
    return base,strong,best,(bestc/best if best>1e-6 else float("nan"))
print("\nROUTING EFFICIENCY (confidence vs oracle), per dataset — bounded by the model-gap:")
for ds in DS:
    base,strong,omax,frac=efficiency(byds[ds])
    print(f"  {ds:<14} cheap={base:.3f} strong={strong:.3f} oracle-gain={omax:+.3f}  confidence captures {frac*100:.0f}% of it")

# (1) K-ablation
conf_au=auroc([r["conf"] for r in rows],cw)
Ks=list(range(2,9)); scau=[auroc([-sc_at_K(r["preds"],K) for r in rows],cw) for K in Ks]
print("K-ablation (self-consistency AUROC for cheap-wrong vs K):", {K:round(a,3) for K,a in zip(Ks,scau)},
      f"| confidence={conf_au:.3f}")

# (2) per-dataset routing efficiency (confidence vs oracle) — bounded by the model-gap
eff={ds:efficiency(byds[ds]) for ds in DS}
LBL={"slake_open":"SLAKE","vqa_rad_open":"VQA-RAD","pathvqa_open":"PathVQA"}
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,4.4))
a1.plot(Ks,scau,"o-",c="#d62728",label="self-consistency"); a1.axhline(conf_au,ls="--",c="#1f77b4",label="confidence")
a1.set_xlabel("#self-consistency samples K"); a1.set_ylabel("routing AUROC (cheap-wrong)")
a1.set_title("K-ablation: self-consistency never\nbeats confidence (even at K=8)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
xs=np.arange(len(DS)); fracs=[eff[ds][3]*100 for ds in DS]; gains=[eff[ds][2] for ds in DS]
b=a2.bar(xs,fracs,0.55,color=["#2ca02c","#2ca02c","#ff7f0e"])
for i,ds in enumerate(DS): a2.text(i,fracs[i]+2,f"{fracs[i]:.0f}%\n(gap {gains[i]:+.2f})",ha="center",fontsize=8)
a2.set_xticks(xs); a2.set_xticklabels([LBL[d] for d in DS]); a2.set_ylim(0,100)
a2.set_ylabel("% of oracle routing-gain captured by confidence")
a2.set_title("Routing efficiency: confidence captures most of\nthe oracle gain where a real model-gap exists")
a2.grid(axis="y",alpha=0.3)
fig.suptitle("Open-ended cascade ablations (Lingshu-7B→Lingshu-32B, LLM-judge, n=%d)"%n,fontsize=11)
fig.tight_layout(rect=[0,0,1,0.94]); os.makedirs("paper/figs/open",exist_ok=True)
fig.savefig("paper/figs/open/fig_open_ablations.png",dpi=140); print("-> paper/figs/open/fig_open_ablations.png")
json.dump(dict(n=n,conf_auroc=conf_au,K_ablation=dict(zip(map(str,Ks),scau)),
    efficiency={ds:{"cheap":eff[ds][0],"strong":eff[ds][1],"oracle_gain":eff[ds][2],"conf_frac":eff[ds][3]} for ds in DS}),
    open("results/cascade_methods/open_ablations.json","w"),indent=1)
