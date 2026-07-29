#!/usr/bin/env python3
"""Generate the three figures for paper/main.tex.

All numbers are read directly from the result artifacts (no fabricated values):
  - results/cascade_methods/artifacts/method_final_mmmu_corrected.json  (Pareto points, Variant A)
  - results/cascade_methods/artifacts/finding1_corrected_2026-07-29.json
        (cross-family think-minus-no-think deltas, PROMPT- AND RESOLUTION-MATCHED arms;
         policy P1_audit_best_matched. Replaces reframe_vs_bigthink.json, whose think arms
         were prompt-unmatched and produced the superseded 15/20 count.)

Outputs (PDF, vector) into paper/figs_final/:
  fig_pareto.pdf     -- accuracy vs {FLOP-eq, batch-1 latency} Pareto frontier
  fig_overthink.pdf  -- cross-family think-minus-no-think accuracy heatmap
  fig_schematic.pdf  -- schematic of the format-aware adaptive cascade

Run from repo root:  python3 paper/make_ieee_figs.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "results", "cascade_methods", "artifacts")
OUT = os.path.join(REPO, "paper", "figs_final")
os.makedirs(OUT, exist_ok=True)

# ---- global style: clean, readable, dark text -----------------------------
plt.rcParams.update({
    "font.size": 8,
    "font.family": "serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.7,
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
})

# Okabe-Ito colorblind-safe palette
OI = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "vermil": "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
    "grey":   "#7f7f7f",
}

def load(name):
    with open(os.path.join(ART, name)) as f:
        return json.load(f)

# ===========================================================================
# FIGURE 1 -- Pareto frontier on two cost axes
# ===========================================================================
def fig_pareto():
    d = load("method_final_mmmu_corrected.json")
    # MMMU-excluded suite (Variant B): 5 MCQ + 3 open, n = 42,224
    pts = d["variants"]["B_mmmu_excluded"]["data"]["pareto_frontier"]["full_suite"]["points"]
    P = {p["system"]: p for p in pts}

    # display config: label, marker, color, is_method
    disp = [
        ("always_7b",                    "7B (cheap floor)",       "o", OI["grey"],   False),
        ("always_32b_think",             "32B think",              "X", OI["vermil"], False),
        ("always_32b_nt",                "32B no-think",           "s", OI["orange"], False),
        ("oracle_mode_32b",              "oracle mode-select 32B", "D", OI["purple"], False),
        ("method_compute_lean",          "Ours: compute-lean",     "*", OI["blue"],   True),
        ("method_accuracy_max_v2_F8F10", "Ours: accuracy-max",     "P", OI["green"],  True),
        ("method_accuracy_max_v1_F3",    "Ours: accuracy-max$^{+}$","^", OI["sky"],   True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))

    def frontier(ax, xkey, xlabel, logx):
        for sys, lab, mk, col, meth in disp:
            p = P[sys]
            x, y = p[xkey], p["acc"]
            ax.scatter(x, y, marker=mk, s=(150 if meth else 70),
                       facecolor=col, edgecolor="#111111",
                       linewidth=(0.9 if meth else 0.6),
                       zorder=5 if meth else 4, label=lab)
        # Pareto envelope (upper-left): non-dominated on (min x, max acc)
        allp = [(P[s][xkey], P[s]["acc"], s) for s, *_ in disp]
        allp.sort()
        env, best = [], -1
        for x, y, s in allp:
            if y > best + 1e-12:
                env.append((x, y)); best = y
        ex, ey = zip(*env)
        ax.plot(ex, ey, color="#111111", lw=0.8, ls="--", alpha=0.55, zorder=2)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("accuracy (sample-weighted, n=42,224)")
        ax.set_ylim(0.551, 0.590)

    frontier(axes[0], "flops", "compute  (FLOP-equivalents per query)", False)
    frontier(axes[1], "lat_par_ms", "batch-1 latency (ms, log scale)", True)

    # shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=6.6, handletextpad=0.3, columnspacing=1.0,
               bbox_to_anchor=(0.5, -0.02))
    axes[0].set_title("(a) accuracy vs. FLOPs", fontsize=8)
    axes[1].set_title("(b) accuracy vs. latency", fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pareto.pdf")

# ===========================================================================
# FIGURE 2 -- cross-family "reasoning hurts perception" heatmap
# ===========================================================================
def fig_overthink():
    # 2026-07-29: switched from reframe_vs_bigthink.json (prompt-UNMATCHED think arms,
    # the source of the superseded 15/20 count) to the prompt-matching correction, so
    # that this figure and Table "tab:overthink" report the same pairing. Policy
    # P1_audit_best_matched == the paper's primary policy; 17/20 perception cells are
    # strictly negative here.
    d = load("finding1_corrected_2026-07-29.json")
    cells = {(c["family"], c["benchmark"]): c
             for c in d["per_cell_by_policy"]["P1_audit_best_matched"]}
    fams = [("medvlthinker", "MedVLThinker"), ("lingshu", "Lingshu"),
            ("qoq", "QoQ-Med"), ("chiron", "Chiron"), ("medgemma", "MedGemma")]
    cols = [("PMC-VQA", "PMC-VQA"), ("SLAKE", "SLAKE"), ("VQA-RAD", "VQA-RAD"),
            ("PathVQA", "PathVQA"), ("MedXpert-Reasoning", "MedXpert-R"),
            ("MedXpert-Understanding", "MedXpert-U")]
    M = np.zeros((len(fams), len(cols)))
    for i, (fk, _) in enumerate(fams):
        for j, (ck, _) in enumerate(cols):
            M[i, j] = cells[(fk, ck)]["delta"]

    # diverging, colorblind-safe: blue (think hurts) - light neutral - orange (think helps)
    cmap = LinearSegmentedColormap.from_list(
        "bo", [OI["blue"], "#9ecae1", "#f2f2f2", "#fdd0a2", OI["vermil"]])
    vmax = 0.13   # widened from 0.11: the matched arms reach -0.1274 (MedVLThinker/SLAKE)
    fig, ax = plt.subplots(figsize=(6.4, 2.55))
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=7.5)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f[1] for f in fams], fontsize=7.5)
    ax.grid(False)
    # annotate
    for i in range(len(fams)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=6.6, color=("white" if abs(v) > 0.075 else "#111111"))
    # divider between perception (cols 0-3) and reasoning (cols 4-5)
    ax.axvline(3.5, color="#111111", lw=1.4)
    ax.text(1.5, -0.72, "perception", ha="center", fontsize=7.5, style="italic")
    ax.text(4.5, -0.72, "reasoning", ha="center", fontsize=7.5, style="italic")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(r"$\Delta$ acc = think $-$ no-think" + "\n(orange: think helps; blue: think hurts)",
                 fontsize=6.6)
    cb.ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_overthink.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_overthink.pdf")

# ===========================================================================
# FIGURE 3 -- schematic of the format-aware adaptive cascade
# ===========================================================================
def fig_schematic():
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec=OI["black"], fs=7.2, tc="#111111"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.4,rounding_size=1.2",
                     linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=3))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=4)

    def arrow(x1, y1, x2, y2, text=None, color="#333333"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=10,
                     linewidth=1.0, color=color, zorder=2))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 1.4, text, ha="center", va="bottom",
                    fontsize=6.3, color=color, style="italic")

    LT = "#eaf2fb"   # light blue
    LO = "#fdf0e0"   # light orange
    LG = "#e9f6f0"   # light green
    NEU = "#f2f2f2"

    # input + router
    box(2, 27, 15, 8, "image +\nquestion", NEU, fs=7.5)
    box(21, 26.5, 15, 9, "format router\n(read the prompt)", "#e6e6e6", fs=7.2)
    arrow(17, 31, 21, 31)

    # ---- MCQ arm (top) ----
    box(42, 44, 15, 8, "7B no-think\n(1 greedy pass)", LT)
    box(62, 44, 17, 8, "margin gate\n$m<\\tau$ ?", LT)
    box(84, 44, 14, 8, "32B\nno-think", LO)
    arrow(36, 33, 42, 47, "MCQ / closed", OI["blue"])
    arrow(57, 48, 62, 48)
    arrow(79, 48, 84, 48, "escalate", OI["vermil"])
    ax.text(70.5, 41.2, "keep 7B  if $m\\geq\\tau$", ha="center", fontsize=6.0,
            color=OI["blue"], style="italic")

    # ---- open-text arm (bottom) ----
    box(42, 8, 17, 8, "7B best-of-$N$\n(adaptive stop)", LG)
    box(63, 8, 16, 8, "trained\nverifier: pick", LG)
    box(84, 8, 14, 8, "32B\nno-think", LO)
    arrow(36, 29, 42, 12, "open-ended", OI["green"])
    arrow(59, 12, 63, 12)
    arrow(79, 12, 84, 12, "escalate", OI["vermil"])
    ax.text(70.5, 5.2, "keep 7B pick  if verifier-confident", ha="center",
            fontsize=6.0, color=OI["green"], style="italic")

    # answer collectors
    ax.text(91, 37, "answer", ha="center", fontsize=7.2)
    ax.text(91, 1.5, "answer", ha="center", fontsize=7.2)

    # captions of the two mechanisms
    ax.text(70, 55.5, "MCQ arm  —  confidence-margin escalation", ha="center",
            fontsize=7.4, weight="bold", color=OI["blue"])
    ax.text(70, 19.5, "open-text arm  —  best-of-$N$ + verifier selection", ha="center",
            fontsize=7.4, weight="bold", color=OI["green"])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_schematic.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_schematic.pdf")

if __name__ == "__main__":
    fig_pareto()
    fig_overthink()
    fig_schematic()
    print("all figures written to", OUT)
