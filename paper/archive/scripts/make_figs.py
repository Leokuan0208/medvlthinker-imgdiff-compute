#!/usr/bin/env python3
"""Generate the paper's headline figures from REAL values in results/cascade_methods/artifacts/final_3tier_comparison.txt
(ALL-6). No fabricated numbers. Outputs paper/figs/*.png. matplotlib only."""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("paper/figs", exist_ok=True)

# --- Fig 1: latency-vs-accuracy frontier (ALL-6), real values from final_3tier_comparison.txt ---
pts = {  # name: (latency_s, accuracy, marker, is_ours) -- CANONICAL from master_data.csv (June-24, native cost)
    "always-7B-nt": (0.13, 0.5262, "v", 0), "always-32B-nt": (0.23, 0.5573, "s", 0),
    "always-32B-think (parity)": (11.34, 0.5723, "*", 0),
    "Ours (ACC-v2)": (2.27, 0.5693, "o", 1), "ACC-v1 (margin)": (2.69, 0.5687, "o", 0),
    "MSP/Chow": (2.96, 0.5697, "o", 0), "AutoMix": (2.50, 0.5692, "o", 0),
    "CASP-Stability (trained)": (1.77, 0.5698, "D", 1), "Jitkrittum L2D": (2.29, 0.5666, "o", 0),
}
fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.axhline(0.5723, ls="--", c="gray", lw=1, label="parity (always-32B-think)")
for nm, (x, y, mk, ours) in pts.items():
    ax.scatter(x, y, s=140 if ours else 70, marker=mk, c=("crimson" if ours else "steelblue"), zorder=3, edgecolors="k", linewidths=0.5)
    ax.annotate(nm, (x, y), textcoords="offset points", xytext=(6, 5), fontsize=7.5,
                color=("crimson" if ours else "black"), fontweight=("bold" if ours else "normal"))
ax.set_xscale("log"); ax.set_xlabel("batch-1 latency (s, log scale)"); ax.set_ylabel("pooled accuracy (ALL-6)")
ax.set_title("ACC reaches parity at a fraction of the latency"); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("paper/figs/fig1_latency_accuracy_frontier.png", dpi=150); plt.close(fig)

# --- Fig 2: per-benchmark over-thinking (32B-no-think minus 32B-think accuracy); >0 = thinking HURTS ---
bench = ["PMC", "SLAKE", "VQA-RAD", "PathVQA", "MMMU", "MedX-R", "MedX-U"]
nt = [0.551, 0.849, 0.853, 0.661, 0.624, 0.279, 0.292]
th = [0.556, 0.764, 0.776, 0.673, 0.688, 0.326, 0.384]
delta = [a - b for a, b in zip(nt, th)]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
cols = ["seagreen" if d > 0 else "indianred" for d in delta]
ax.bar(bench, delta, color=cols, edgecolor="k", linewidth=0.5)
ax.axhline(0, c="k", lw=0.8)
ax.set_ylabel("acc(32B no-think) − acc(32B think)"); ax.set_title("Thinking over-thinks perception (green = no-think wins)")
for i, d in enumerate(delta): ax.annotate(f"{d:+.2f}", (i, d), ha="center", va="bottom" if d > 0 else "top", fontsize=8)
ax.grid(axis="y", alpha=0.3); fig.tight_layout(); fig.savefig("paper/figs/fig2_overthinking_perbench.png", dpi=150); plt.close(fig)
print("wrote paper/figs/fig1_latency_accuracy_frontier.png and fig2_overthinking_perbench.png")
