#!/usr/bin/env python3
"""Unified figure for the trained-verifier result: across free-text answers AND structured boxes, across
VQA / organ grounding / chest-X-ray pathology grounding, the training-free selector is luck-floored
(at/below greedy) while a TRAINED verifier captures 40-77% of the oracle gap. All numbers from real runs."""
import os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
OUT=os.path.expanduser("~/medvlthinker-imgdiff-compute/paper/figs/limits"); os.makedirs(OUT,exist_ok=True)
# (setting, greedy, training-free selector, trained verifier, oracle, gap%)
data=[
 ("Free-text answers\n(5 VQA datasets, n=1064)", 0.413, 0.411, 0.501, 0.592, 49),
 ("Boxes: SLAKE organs\n(n=487, IoU≥0.3)",   0.197, 0.164, 0.255, 0.343, 40),
 ("Boxes: MS-CXR pathology\n(real benchmark, n=435)",0.041, 0.053, 0.230, 0.285, 77),
]
fig,axes=plt.subplots(1,3,figsize=(12.4,4.3))
labels=["greedy","training-free\nselector","trained\nverifier","oracle@8"]
colors=["#999999","#C44E52","#4C72B0","#55A868"]
for ax,(name,g,sf,tv,orc,pct) in zip(axes,data):
    vals=[g,sf,tv,orc]; bars=ax.bar(range(4),vals,color=colors,width=0.72,zorder=3)
    for i,v in enumerate(vals): ax.text(i,v+max(vals)*0.02,f"{v:.3f}",ha="center",va="bottom",fontsize=8.5)
    ax.axhline(g,color="#999999",ls=":",lw=1.2,zorder=2)
    ax.axhline(orc,color="#55A868",ls="--",lw=1.2,zorder=2)
    ax.set_title(name,fontsize=9.5)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels,fontsize=8)
    ax.set_ylim(0,orc*1.18); ax.grid(axis="y",alpha=0.3,zorder=0)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    # annotate gap captured
    ax.annotate(f"trained captures\n{pct}% of oracle gap",xy=(2,tv),xytext=(0.5,orc*1.02),
                fontsize=8.5,fontweight="bold",color="#4C72B0",ha="left")
axes[0].set_ylabel("accuracy (VQA: LLM-judge; grounding: IoU≥0.3)",fontsize=9.5)
fig.suptitle("Training is the universal ingredient that breaks the luck floor: training-free selection ≈ greedy "
             "(luck-floored), a TRAINED verifier captures 40–77% of the oracle gap — across free-text answers AND "
             "structured boxes, on VQA, organ grounding, and a real chest-X-ray benchmark",fontsize=9.2,y=1.02)
plt.tight_layout()
p=os.path.join(OUT,"fig_trained_verifier_unified.png"); plt.savefig(p,dpi=150,bbox_inches="tight")
print("wrote",p)
