#!/usr/bin/env python3
"""§5.10 figure: best-of-K test-time-scaling curve. Random selection stays flat while the TRAINED verifier
monotonically converts more samples into accuracy (toward the rising oracle). Pooled-4 free-text verifier,
held-out, n=1064. Numbers from ckpts/train/lora_verifier_pooled4/scaling_curve.json (real run)."""
import os, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
d=json.load(open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/scaling_curve.json")))
Ks=[1,2,4,8]
ver=[d[str(k)]["verifier"] for k in Ks]; orc=[d[str(k)]["oracle"] for k in Ks]; rnd=[d[str(k)]["random"] for k in Ks]
OUT=os.path.join(ROOT,"paper/figs/limits"); os.makedirs(OUT,exist_ok=True)
fig,ax=plt.subplots(figsize=(6.2,4.4))
ax.plot(Ks,orc,"o--",color="#55A868",lw=2,label="oracle@K (best of K)",zorder=3)
ax.plot(Ks,ver,"o-",color="#4C72B0",lw=2.4,label="trained verifier (best-of-K)",zorder=4)
ax.plot(Ks,rnd,"o:",color="#C44E52",lw=2,label="random selection",zorder=3)
for k,v in zip(Ks,ver): ax.annotate(f"{v:.3f}",(k,v),textcoords="offset points",xytext=(4,-12),fontsize=8.5,color="#4C72B0")
ax.set_xscale("log",base=2); ax.set_xticks(Ks); ax.set_xticklabels(Ks)
ax.set_xlabel("number of sampled answers K (test-time compute)",fontsize=10)
ax.set_ylabel("open-ended accuracy (LLM-judge), n=1064",fontsize=10)
ax.set_title("The trained verifier is a real test-time-scaling method:\nrandom selection stays flat, the verifier "
             "converts more samples into accuracy (0.385→0.501)",fontsize=9.8)
ax.legend(fontsize=9,loc="upper left"); ax.grid(alpha=0.3,zorder=0)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); p=os.path.join(OUT,"fig_verifier_scaling.png"); plt.savefig(p,dpi=150,bbox_inches="tight")
print("wrote",p)
