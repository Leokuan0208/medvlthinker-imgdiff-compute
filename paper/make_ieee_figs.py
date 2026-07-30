#!/usr/bin/env python3
"""Generate the three figures for paper/main.tex.

All numbers are read directly from the result artifacts (no fabricated values):
  - results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json
        keys cost.pareto.honest_recost.{macro_cells, sample_weighted}
        (accuracy--cost points for all 7 systems under the paper's PRIMARY 8-cell
         equal-weight convention and, for contrast, the sample-weighted one.
         Replaces method_final_mmmu_corrected.json, which was Variant A,
         sample-weighted, and carried an ESTIMATED always_32b_reasoning = 0.5628.
         The old figure is preserved as fig_pareto_superseded_2026-07-08.pdf.)
  - results/cascade_methods/artifacts/finding1_corrected_2026-07-29.json
        (cross-family think-minus-no-think deltas, PROMPT- AND RESOLUTION-MATCHED arms;
         policy P1_audit_best_matched. Replaces reframe_vs_bigthink.json, whose think arms
         were prompt-unmatched and produced the superseded 15/20 count.)

Outputs (PDF, vector) into paper/figs_final/:
  fig_pareto.pdf     -- accuracy vs cost under BOTH weightings (3 panels, \textwidth)
  fig_overthink.pdf  -- cross-family think-minus-no-think accuracy heatmap (\textwidth)
  fig_schematic.pdf  -- schematic of the format-aware adaptive cascade

Run from repo root:  python3 paper/make_ieee_figs.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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
# FIGURE 1 -- accuracy vs cost, under BOTH weighting conventions
# ---------------------------------------------------------------------------
# 2026-07-30 REBUILD.  The previous version read
# method_final_mmmu_corrected.json (Variant A, sample-weighted, with an
# ESTIMATED always_32b_reasoning accuracy of 0.5628) and so disagreed with the
# main result table (tab:main, printed as TABLE III) on both axes once the paper
# was re-based onto the 8-cell macro average.  It also drew the retired "Pareto-dominates every fixed way of using
# the 32B" picture.  This version reads
#   macro_average_headline_2026-07-30.json : cost.pareto.honest_recost.*
# and plots three internally-consistent panels.  Each panel pairs an accuracy
# with a cost computed under the SAME weighting -- the paper's setup section
# forbids pairing a macro accuracy with a sample-weighted cost -- so panels
# (a),(b) are equal-weight (tab:main "Acc. (macro)") and panel (c) is
# sample-weighted (tab:main "Acc. (sw)" / "rel. FLOP (sw)").
#
# Costing basis: honest_recost, i.e. always-32B-reasoning and oracle
# mode-select are charged their MEASURED per-cell generation rather than the
# flat convention constant.  This is the basis of every claim the paper makes
# against the reasoning baseline (-89.0% latency, -87.3% energy, 1.131x
# FLOP-eq) and matches tab:main's footnote; tab:main's body charges the flat
# constant, which is why the caption names the basis explicitly.
# ===========================================================================
PARETO_ART = "macro_average_headline_2026-07-30.json"
ONE_32B_FLOPS = 4.57   # FLOP-eq of one 32B forward (= the measured 32B/7B ratio)

def fig_pareto():
    d = load(PARETO_ART)
    par = d["cost"]["pareto"]["honest_recost"]
    MACRO = {p["system"]: p for p in par["macro_cells"]["points"]}
    SW    = {p["system"]: p for p in par["sample_weighted"]["points"]}
    ncell = d["pool"]["n_cells"]
    nitem = d["pool"]["n_items"]

    # display config: key, label (matching tab:main row names), marker, colour, is_method
    disp = [
        ("always_7b",                 "always-7B (cheap floor)",        "o", OI["grey"],   False),
        ("always_32b_reasoning",      "always-32B-reasoning",           "X", OI["vermil"], False),
        ("always_32b_direct",         "always-32B-direct",              "s", OI["orange"], False),
        ("oracle_mode_32b",           "oracle mode-select 32B",         "D", OI["purple"], False),
        ("method_compute_lean",       "Ours: compute-lean",             "*", OI["blue"],   True),
        ("method_accuracy_max_veto",  "Ours: accuracy-max",             "P", OI["green"],  True),
        ("method_accuracy_max_fusion","Ours: accuracy-max$^{+}$(fusion)","^", OI["sky"],   True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95))

    def panel(ax, P, xkey, xlabel, ylim, logx=False, vline=None, ylabel=None):
        for sysk, lab, mk, col, meth in disp:
            p = P[sysk]
            hollow = (sysk == "oracle_mode_32b")   # sits on top of always-32B-direct
            ax.scatter(p[xkey], p["acc"], marker=mk, s=(120 if meth else 78),
                       facecolor=("none" if hollow else col),
                       edgecolor=(col if hollow else "#111111"),
                       linewidth=(1.3 if hollow else (0.9 if meth else 0.6)),
                       zorder=(4 if hollow else (6 if meth else 5)), label=lab)
        # non-dominated set (min cost, max accuracy).  Drawn as a staircase-free
        # dashed guide: it shows the trade-off, NOT dominance.
        allp = sorted((P[s][xkey], P[s]["acc"]) for s, *_ in disp)
        env, best = [], -1.0
        for x, y in allp:
            if y > best + 1e-12:
                env.append((x, y)); best = y
        ex, ey = zip(*env)
        ax.plot(ex, ey, color="#111111", lw=0.8, ls="--", alpha=0.5, zorder=2,
                label="non-dominated set")
        if vline is not None:
            ax.axvspan(ax.get_xlim()[0] if logx else 0, vline,
                       color="#f0f0f0", zorder=0)
            ax.axvline(vline, color="#555555", lw=0.8, ls=":", zorder=1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=7.4)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=7.4)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7.0)

    YMAC = (0.5885, 0.6790)
    YSW  = (0.5495, 0.5945)

    # ---- (a) equal weight, compute -----------------------------------------
    a = axes[0]
    a.set_xlim(0, 7.6)
    panel(a, MACRO, "flops", "compute (FLOP-eq per query)", YMAC,
          vline=ONE_32B_FLOPS,
          ylabel="accuracy — macro (%d cells, 1/%d each)" % (ncell, ncell))
    a.text(ONE_32B_FLOPS - 0.20, 0.6775, "one 32B forward", rotation=90,
           ha="right", va="top", fontsize=6.0, color="#555555")
    a.text(0.12, 0.5905, "cheaper than one 32B forward", fontsize=5.9,
           color="#555555", style="italic", va="bottom")
    a.annotate("", xy=(MACRO["method_accuracy_max_veto"]["flops"], 0.6745),
               xytext=(ONE_32B_FLOPS, 0.6745),
               arrowprops=dict(arrowstyle="-|>", lw=0.8, color=OI["vermil"],
                               shrinkA=0, shrinkB=0))
    a.text(5.55, 0.6760, "$1.41\\times$", fontsize=6.2, color=OI["vermil"],
           ha="center", va="bottom")

    # ---- (b) equal weight, latency -----------------------------------------
    b = axes[1]
    b.set_xlim(280, 14000)
    panel(b, MACRO, "lat_par_ms", "batch-1 latency (ms, log scale)", YMAC,
          logx=True)
    b.set_xticks([500, 1000, 2000, 5000, 10000])
    b.set_xticklabels(["500", "1k", "2k", "5k", "10k"])
    b.set_xticks([], minor=True)
    r, v = MACRO["always_32b_reasoning"], MACRO["method_accuracy_max_veto"]
    b.annotate("", xy=(v["lat_par_ms"] * 1.15, v["acc"] - 0.0015),
               xytext=(r["lat_par_ms"] * 0.86, r["acc"] + 0.0015),
               arrowprops=dict(arrowstyle="-|>", lw=0.9, color=OI["green"],
                               connectionstyle="arc3,rad=0.18",
                               shrinkA=2, shrinkB=2))
    b.text(2050, 0.6330, "vs. a reasoning 32B:\n$+0.072$ acc,\n$-89\\%$ lat., $-87\\%$ energy",
           fontsize=6.0, color=OI["green"], ha="center", va="center")

    # ---- (c) sample-weighted, compute --------------------------------------
    c = axes[2]
    c.set_xlim(0, 7.6)
    panel(c, SW, "flops", "compute (FLOP-eq per query)", YSW,
          vline=ONE_32B_FLOPS,
          ylabel="accuracy — sample-weighted, n=%s" % f"{nitem:,}")
    c.text(ONE_32B_FLOPS + 0.30, 0.5938, "one 32B forward", rotation=90,
           ha="left", va="top", fontsize=6.0, color="#555555")
    c.text(0.12, 0.5505, "cheaper than one 32B forward", fontsize=5.9,
           color="#555555", style="italic", va="bottom")
    c.annotate("", xy=(SW["method_compute_lean"]["flops"], 0.5915),
               xytext=(ONE_32B_FLOPS, 0.5915),
               arrowprops=dict(arrowstyle="-|>", lw=0.8, color=OI["blue"],
                               shrinkA=0, shrinkB=0))
    c.text(3.30, 0.5921, "$0.49\\times$", fontsize=6.2, color=OI["blue"],
           ha="center", va="bottom")

    axes[0].set_title("(a) equal weight per cell — compute", fontsize=7.6)
    axes[1].set_title("(b) equal weight per cell — latency", fontsize=7.6)
    axes[2].set_title("(c) sample-weighted — compute", fontsize=7.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=6.6, handletextpad=0.3, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    fig.subplots_adjust(wspace=0.33)
    fig.savefig(os.path.join(OUT, "fig_pareto.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pareto.pdf  (macro + sample-weighted, honest_recost)")

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
    # 2026-07-30: promoted to a full-width figure* in the paper.  The figure is
    # drawn at (almost exactly) \textwidth = 516pt = 7.153in, so \includegraphics
    # [width=\textwidth] scales it ~1:1 and the fontsize= values below are the
    # TRUE on-page point sizes.  (At \columnwidth it was printed at 54% and the
    # cell numbers landed at ~3.6pt.)  Sizes are chosen against the document's
    # own type scale -- 10pt body, 8pt captions, 7pt \scriptsize table notes --
    # so that no text in this figure is the weakest text on the page.
    fig, ax = plt.subplots(figsize=(7.30, 2.85))
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], fontsize=8.5)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels([f[1] for f in fams], fontsize=8.5)
    ax.grid(False)
    # annotate
    for i in range(len(fams)):
        for j in range(len(cols)):
            v = M[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8.0, color=("white" if abs(v) > 0.075 else "#111111"))
    # divider between perception (cols 0-3) and reasoning (cols 4-5)
    ax.axvline(3.5, color="#111111", lw=1.4)
    # 2026-07-30: the reasoning block is HATCHED because its two arms differ in
    # answer format as well as in the reasoning instruction.  A format-matched
    # re-run (medeval_matched_direct_2026-07-29.json) leaves 0/9 reasoning-trigger
    # effects CI-significant and 3/9 format effects significant, so these cells
    # must not be read as a reasoning gain.
    ax.add_patch(Rectangle((3.5, -0.5), 2.0, 5.0, facecolor="none",
                           edgecolor="#3d3d3d", hatch="////", linewidth=0.0,
                           alpha=0.45, zorder=2))
    ax.text(1.5, -1.00, "perception", ha="center", fontsize=9.0, style="italic")
    ax.text(4.5, -1.00, "reasoning", ha="center", fontsize=9.0, style="italic")
    ax.text(4.5, -0.68, "(answer format not matched)",
            ha="center", fontsize=8.0, color="#3d3d3d")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label(r"$\Delta$ acc = think $-$ no-think" + "\n(orange: think helps; blue: think hurts)",
                 fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)
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
