#!/usr/bin/env python3
"""§verifier figure: the trained verifier's score cleanly separates correct from incorrect candidate
answers (AUROC 0.924, n=8512), the opposite of a 'lazy verifier'. Data: per-question scores+labels in
ckpts/train/lora_verifier_pooled4/perq_sc8.json (real run)."""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
d=json.load(open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/perq_sc8.json")))
S=[];L=[]
for r in d:
    for s,l in zip(r["sc"],r["sl"]):
        if l is not None: S.append(s); L.append(int(l))
S=np.array(S);L=np.array(L)
def auroc(sc,lb):
    o=np.argsort(sc); rk=np.empty(len(sc)); rk[o]=np.arange(1,len(sc)+1)
    p=lb.sum(); n=len(lb)-p; return (rk[lb==1].sum()-p*(p+1)/2)/(p*n)
au=auroc(S,L)
OUT=os.path.join(ROOT,"paper/figs/limits"); os.makedirs(OUT,exist_ok=True)
fig,ax=plt.subplots(figsize=(6.2,4.2))
bins=np.linspace(0,1,26)
ax.hist(S[L==1],bins=bins,density=True,alpha=0.7,color="#2e7d32",label=f"correct candidates (mean {S[L==1].mean():.2f})")
ax.hist(S[L==0],bins=bins,density=True,alpha=0.7,color="#c62828",label=f"incorrect candidates (mean {S[L==0].mean():.2f})")
ax.axvline(S[L==1].mean(),color="#2e7d32",ls="--",lw=1.5); ax.axvline(S[L==0].mean(),color="#c62828",ls="--",lw=1.5)
ax.set_xlabel("trained verifier score  s_φ(v,q,a) = P(correct)",fontsize=10)
ax.set_ylabel("density",fontsize=10)
ax.set_title(f"The trained verifier discriminates correct from incorrect candidates\n"
             f"AUROC = {au:.3f}  (n={len(S)} candidate answers, pooled-4) — not a 'lazy verifier'",fontsize=9.8)
ax.legend(fontsize=9,loc="upper center"); ax.grid(alpha=0.25)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); p=os.path.join(OUT,"fig_verifier_discrimination.png"); plt.savefig(p,dpi=150,bbox_inches="tight")
print(f"wrote {p}  (AUROC={au:.4f}, n={len(S)})")
