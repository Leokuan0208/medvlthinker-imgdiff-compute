#!/usr/bin/env python3
"""§5.9 figure: the open-ended SELECTION luck floor. Every training-free selector over the cheap model's
8 samples sits just above the random-pick floor and far below the oracle — and below the large model's
single pass. All numbers from SLAKE-open, Lingshu-7B samples, one consistent LLM-judge run (this session).
Output: paper/figs/limits/fig_selection_luckfloor.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = os.path.expanduser("~/medvlthinker-imgdiff-compute/paper/figs/limits")
os.makedirs(OUT, exist_ok=True)
# (label, value, kind)  kind: floor/selector/ref
rows = [
    ("random pick\n(floor)", 0.720, "floor"),
    ("self-verify\nP(Yes)", 0.715, "selector"),
    ("greedy", 0.730, "selector"),
    ("self-consistency\n(majority)", 0.736, "selector"),
    ("learned\nfusion", 0.743, "selector"),
    ("32B verify\n(pointwise)", 0.746, "selector"),
    ("32B verify\n(listwise)", 0.758, "selector"),
    ("32B synth\n(primed)", 0.774, "selector"),
]
labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
colors = {"floor": "#888888", "selector": "#4C72B0"}
cols = [colors[r[2]] for r in rows]
fig, ax = plt.subplots(figsize=(9.2, 4.6))
bars = ax.bar(range(len(rows)), vals, color=cols, width=0.7, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, zorder=4)
# reference lines
ax.axhline(0.720, color="#888888", ls=":", lw=1.4, zorder=2)
ax.axhline(0.819, color="#C44E52", ls="--", lw=1.8, zorder=2)
ax.axhline(0.879, color="#55A868", ls="--", lw=1.8, zorder=2)
ax.text(len(rows)-0.4, 0.823, "32B single pass (SOTA) 0.819", color="#C44E52", ha="right", va="bottom", fontsize=9, fontweight="bold")
ax.text(len(rows)-0.4, 0.883, "oracle@8 (luck ceiling) 0.879", color="#55A868", ha="right", va="bottom", fontsize=9, fontweight="bold")
ax.text(0.0, 0.704, "random-pick floor 0.720", color="#555555", ha="left", va="top", fontsize=8.5)
ax.set_ylim(0.69, 0.90)
ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8.3)
ax.set_ylabel("open-ended accuracy (LLM-judge)", fontsize=10)
ax.set_title("Open-ended selection is a luck floor: no training-free selector escapes it\n"
             "(SLAKE-open, Lingshu-7B samples) — best selector captures 24% of the oracle gap, none beats the 32B single pass",
             fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
p = os.path.join(OUT, "fig_selection_luckfloor.png")
plt.savefig(p, dpi=150, bbox_inches="tight")
print("wrote", p)
