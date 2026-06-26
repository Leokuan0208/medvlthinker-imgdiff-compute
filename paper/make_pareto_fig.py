#!/usr/bin/env python3
"""Accuracy vs compute: verifier-selected best-of-K on the 7B vs the 32B single pass. Test-time compute
on the small model (verifier-bo8 = 0.501) beats a 5x-larger model (32B single = 0.444 pooled). Cost is a
param-FLOP proxy in 7.6e9-param forward-equivalents: bo-K = K generations + K verifications = 2K; 32B
single = 33/7.6 = 4.34. Verifier accuracy from scaling_curve.json; 32B pooled from real judge files."""
import os, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
d=json.load(open(os.path.join(ROOT,"ckpts/train/lora_verifier_pooled4/scaling_curve.json")))
# (compute units, accuracy) for verifier best-of-K
pts={1:(1,d["1"]["verifier"]),2:(4,d["2"]["verifier"]),4:(8,d["4"]["verifier"]),8:(16,d["8"]["verifier"])}
xs=[pts[k][0] for k in [1,2,4,8]]; ys=[pts[k][1] for k in [1,2,4,8]]
OUT=os.path.join(ROOT,"paper/figs/limits"); os.makedirs(OUT,exist_ok=True)
fig,ax=plt.subplots(figsize=(6.4,4.5))
ax.plot(xs,ys,"o-",color="#4C72B0",lw=2.4,ms=8,label="7B + verifier best-of-K",zorder=4)
for k in [1,2,4,8]:
    x,y=pts[k]; ax.annotate(f"K={k}\n{y:.3f}",(x,y),textcoords="offset points",xytext=(6,-2),fontsize=8,color="#4C72B0")
ax.scatter([4.34],[0.444],color="#C44E52",s=130,marker="D",zorder=5,label="32B single pass (0.444)")
ax.annotate("32B single pass\n0.444 (5× params)",(4.34,0.444),textcoords="offset points",xytext=(8,-26),fontsize=8.5,color="#C44E52",fontweight="bold")
ax.scatter([1],[ys[0]],color="#999999",s=80,zorder=5)
ax.set_xscale("log",base=2); ax.set_xlabel("compute (7.6e9-param forward-equivalents, proxy)",fontsize=10)
ax.set_ylabel("open-ended accuracy (pooled, LLM-judge)",fontsize=10)
ax.set_title("Test-time compute beats parameters: verifier-best-of-8 on the 7B (0.501)\n"
             "outperforms the 32B single pass (0.444) — the verifier is the accuracy-optimal point",fontsize=9.6)
ax.legend(fontsize=9,loc="lower right"); ax.grid(alpha=0.3,zorder=0)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout(); p=os.path.join(OUT,"fig_verifier_pareto.png"); plt.savefig(p,dpi=150,bbox_inches="tight")
print("wrote",p)
