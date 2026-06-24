#!/usr/bin/env python3
"""
make_open_chart.py - charts for the open-ended self-consistency cascade (MedVLThinker-7B -> Lingshu-32B).
Reads results/cascade_methods/open_cascade_lingshu.json. Outputs to paper/figs/open/.
(1) accuracy-vs-escalation frontier: self-consistency gate vs confidence gate.
(2) routing-signal AUROC bars (cheap-wrong + recoverability) with the ~0.6 MCQ ceiling reference.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = json.load(open("results/cascade_methods/open_cascade_lingshu.json"))
OUT = "paper/figs/open"; os.makedirs(OUT, exist_ok=True)

def chart_frontier():
    fc = sorted(D["frontier"]["confidence"]); fs = sorted(D["frontier"]["self_consistency"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot([p[0]*100 for p in fc], [p[1] for p in fc], "o-", c="#1f77b4", ms=3, lw=1.4, label="confidence gate (Chow/Jitkrittum)")
    ax.plot([p[0]*100 for p in fs], [p[1] for p in fs], "^-", c="#d62728", ms=3, lw=1.4, label="self-consistency gate (ours)")
    ax.axhline(D["POOLED"]["cheap_acc"], ls=":", c="gray", lw=0.8); ax.text(2, D["POOLED"]["cheap_acc"]+0.004, "always-cheap (7B)", fontsize=7, color="gray")
    ax.axhline(D["frontier"]["strong_acc"], ls=":", c="green", lw=0.8); ax.text(2, D["frontier"]["strong_acc"]-0.018, "always-strong (Lingshu-32B)", fontsize=7, color="green")
    ax.set_xlabel("escalation rate to the strong model (%)"); ax.set_ylabel("cascade accuracy")
    ax.set_title("Open-ended medical VQA cascade (7B → Lingshu-32B)\nself-consistency routes better than confidence", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout(); p = f"{OUT}/frontier_selfconsistency.png"; fig.savefig(p, dpi=140); plt.close(); return p

def chart_auroc():
    cw = D["POOLED"]["auroc_cw"]; rec = D["POOLED"]["auroc_rec"]
    groups = ["predict cheap-WRONG", "predict RECOVERABLE"]
    conf = [cw["conf"], rec["conf"]]; scs = [cw["selfcons"], rec["selfcons"]]; nd = [cw["ndist"], rec["ndist"]]
    x = np.arange(2); w = 0.25
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(x-w, conf, w, label="confidence (seq-logprob)", color="#1f77b4")
    ax.bar(x, scs, w, label="self-consistency (K=8)", color="#d62728")
    ax.bar(x+w, nd, w, label="answer-diversity (n_distinct)", color="#ff7f0e")
    for i,(a,b,c) in enumerate(zip(conf,scs,nd)):
        for dx,v in [(-w,a),(0,b),(w,c)]: ax.text(i+dx, v+0.008, f"{v:.2f}", ha="center", fontsize=7)
    ax.axhline(0.6, ls="--", c="k", lw=0.9); ax.text(1.35, 0.61, "≈ MCQ ceiling", fontsize=8)
    ax.axhline(0.5, ls=":", c="gray", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylabel("AUROC"); ax.set_ylim(0.45, 0.85)
    ax.set_title("Routing-signal quality (open-ended, pooled n=845)\nself-consistency beats confidence and breaks the MCQ ceiling", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); p = f"{OUT}/auroc_signals.png"; fig.savefig(p, dpi=140); plt.close(); return p

if __name__ == "__main__":
    for p in [chart_frontier(), chart_auroc()]: print("written:", p)
