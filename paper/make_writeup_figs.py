#!/usr/bin/env python3
"""
make_writeup_figs.py -- the seven figures for the 2026-08 write-up.

RUN FROM THE REPO ROOT:
    python3 paper/make_writeup_figs.py

Output: results/cascade_methods/docs/current/figs/*.png  @ 200 dpi.
Every canvas is EXACTLY 6.5 in wide (1300 px @ 200 dpi) -- the full text width of
a US-Letter Word page with 1 in margins -- so each PNG is inserted at 100% scale
and the point sizes below are the point sizes on paper.  Nothing smaller than
6.2 pt is used, and that only for source footnotes.

NO FABRICATED NUMBERS.  Every data literal carries a comment naming the artifact
file and the JSON key path it came from.  Nothing is recomputed, smoothed,
rounded-for-looks, or filled in.  Where a value is DERIVED or MODELLED rather
than MEASURED the figure says so on its face -- that distinction is load-bearing
and must not be flattened.

Palette: the validated colourblind-safe categorical set (slot1 blue #2a78d6,
slot2 orange #eb6834, slot3 aqua #1baf7a) plus the blue<->red diverging pair over
a neutral gray midpoint.  Adjacent-pair and first-three-slot all-pairs CVD
separation are validated for this set.  Text always wears ink tokens, never a
series colour, and every series is also direct-labelled -- identity is never
carried by colour alone.
"""

import os
import textwrap
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# output + style
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "cascade_methods", "docs", "current", "figs")
os.makedirs(OUT, exist_ok=True)

DPI = 200
W = 6.5  # inches -- exact Word text width; every figure is saved at this width

BLUE = "#2a78d6"    # categorical slot 1
BLUE_L = "#9ec5f4"  # blue ramp step 200 (range fills)
ORANGE = "#eb6834"  # categorical slot 2
AQUA = "#1baf7a"    # categorical slot 3
RED = "#e34948"     # diverging warm pole
INK = "#0b0b0b"     # text-primary
INK2 = "#52514e"    # text-secondary
GRID = "#dcdbd7"
SURF = "#ffffff"

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 9.0, "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.edgecolor": INK2, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def finish(fig, name):
    """Save on a FIXED canvas (no bbox tightening) so the width is exactly 6.5 in."""
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=DPI, facecolor=SURF)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(p).size
    print(f"  {name:28s} {w:5d} x {h:4d} px   "
          f"({w / DPI:.2f} x {h / DPI:.2f} in @ {DPI} dpi)")
    return p


def foot(fig, txt, y=0.010, size=6.3, wrap=128):
    """Bottom-left source/method note, hard-wrapped so it cannot run off the page."""
    lines = []
    for para in txt.split("\n"):
        lines += textwrap.wrap(para, wrap) or [""]
    fig.text(0.012, y, "\n".join(lines), fontsize=size, color=INK2,
             ha="left", va="bottom", linespacing=1.4)


# =============================================================================
# 1. fig_correction_cascade
#    ARTIFACT: results/cascade_methods/artifacts/headline_three_way_2026-08-03.json
#    KEY:      /headline_progression   (col1..col4; col5 leaves the delta unchanged)
# =============================================================================
def fig_correction_cascade():
    # /headline_progression[i]/{delta, ci95, delta_change_from_previous_column}
    lv = [0.0245, 0.0720, 0.0601, 0.0325]      # delta at cols 1,2,3,4
    ci = [(0.0217, 0.0274),                     # col1 /ci95
          (0.0614, 0.0824),                     # col2 /ci95
          (0.0498, 0.0703),                     # col3 /ci95
          (0.0237, 0.0412)]                     # col4 /ci95  (= col5 /ci95)
    step = [0.0475, -0.0119, -0.0276]           # /delta_change_from_previous_column
    xlab = [
        "published\n(sample-weighted,\ncontaminated verifier,\nunmatched reasoning)",
        "+ macro average\n(equal weight per\nreporting cell,\n8 cells)",
        "+ clean verifier\n(retrained on\nstrictly disjoint\ndata, L1)",
        "+ matched\nreasoning baseline\n(prompt-matched,\narm B)",
        "FINAL\n(all three, plus the\ngrounded R32 = 3.816,\nwhich moves no accuracy)",
    ]
    # where each column's value + CI text goes: above the level, or below it
    place = ["above", "above", "below", None, "above"]

    fig = plt.figure(figsize=(W, 4.55))
    ax = fig.add_axes([0.105, 0.255, 0.875, 0.625])
    bw = 0.44

    ax.bar(0, lv[0], bw, color=INK2, zorder=3)                      # level: published
    for i, (x, s) in enumerate(zip([1, 2, 3], step)):               # floating steps
        base, top = lv[i], lv[i] + s
        ax.bar(x, s, bw, bottom=base, color=BLUE if s > 0 else RED, zorder=3)
        ax.text(x, base + s / 2, f"{s:+.4f}", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold", zorder=7)
        ax.plot([x - bw / 2 - 0.28, x - bw / 2], [base, base], color=INK2,
                lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax.plot([x + bw / 2, x + bw / 2 + 0.28], [top, top], color=INK2,
                lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.bar(4, lv[3], bw, color=INK2, zorder=3)                      # level: final

    # 95% CI on each column's resulting level.  The whisker sits just right of the
    # bar (never on it); the value + interval are stacked in clear vertical space.
    for x, v, (lo, hi), where in zip([0, 1, 2, 4], lv, ci,
                                     [p for p in place if p]):
        ax.errorbar(x + bw / 2 + 0.11, v, yerr=[[v - lo], [hi - v]], fmt="none",
                    ecolor=INK, elinewidth=1.1, capsize=3.0, capthick=1.1, zorder=6)
        if where == "above":
            ax.text(x + 0.08, hi + 0.0075, f"{v:+.4f}", ha="center", va="bottom",
                    fontsize=8.2, fontweight="bold", color=INK)
            ax.text(x + 0.08, hi + 0.0022, f"[{lo:+.4f}, {hi:+.4f}]", ha="center",
                    va="bottom", fontsize=6.3, color=INK2)
        else:
            ax.text(x + 0.08, lo - 0.0075, f"{v:+.4f}", ha="center", va="top",
                    fontsize=8.2, fontweight="bold", color=INK)
            ax.text(x + 0.08, lo - 0.0022, f"[{lo:+.4f}, {hi:+.4f}]", ha="center",
                    va="top", fontsize=6.3, color=INK2)

    ax.axhline(0, color=INK, lw=0.9, zorder=4)
    ax.set_xlim(-0.62, 4.72)
    ax.set_ylim(-0.003, 0.108)
    ax.set_xticks(range(5))
    ax.set_xticklabels(xlab, fontsize=6.6, color=INK, linespacing=1.35)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(np.arange(0.0, 0.0951, 0.02))
    ax.set_ylabel("Δ accuracy:  accuracy-max-veto  −  always-32B-with-reasoning",
                  fontsize=7.6)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=INK2, label="headline at that column"),
        Patch(facecolor=BLUE, label="correction that RAISED it"),
        Patch(facecolor=RED, label="correction that LOWERED it"),
    ], loc="upper right", bbox_to_anchor=(1.0, 1.005), ncol=1, handlelength=1.3,
        borderaxespad=0.0, labelspacing=0.32, fontsize=7.2)
    ax.set_title("Every column is still a significant win (CI excludes 0) — "
                 "and the claim halves on the way",
                 fontsize=8.5, color=INK2, pad=5, loc="left")

    foot(fig, "Absolute accuracy difference. Weighting: the published column is "
              "sample-weighted over the pooled evaluation; every later column is a "
              "macro average over the 8 reporting cells (1/8 each). Intervals are 95% "
              "percentile CIs from 10,000 paired bootstrap resamples (seed 20260730).\n"
              "Source: headline_three_way_2026-08-03.json /headline_progression.")
    return finish(fig, "fig_correction_cascade")


# =============================================================================
# 2. fig_finding1_crossfamily
#    ARTIFACT: results/cascade_methods/artifacts/finding1_corrected_2026-07-29.json
#    KEY:      /per_cell_by_policy/P1_audit_best_matched          (35 cells)
#              /counts_by_policy/P1_audit_best_matched/{perception,perception_pooled}
# =============================================================================
def fig_finding1_crossfamily():
    fams = ["MedVLThinker-32B", "Lingshu-32B", "QoQ-Med-32B",
            "Chiron-o1-32B", "MedGemma-27B"]
    bench = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA",
             "MMMU", "MedXpert-\nReasoning", "MedXpert-\nUnderstanding"]
    NP = 4  # first four columns are the perception benchmarks

    # /per_cell_by_policy/P1_audit_best_matched -> (delta, ci95_lo, ci95_hi),
    # rows in registry order: medvlthinker, lingshu, qoq, chiron, medgemma
    D = [
        [(-0.0075, -0.0275,  0.0120), (-0.1274, -0.1659, -0.0913),
         (-0.0846, -0.1360, -0.0368), ( 0.0012, -0.0155,  0.0173),
         ( 0.0882,  0.0235,  0.1529), ( 0.0491,  0.0228,  0.0754),
         ( 0.0884,  0.0469,  0.1300)],
        [(-0.0425, -0.0625, -0.0220), (-0.0649, -0.0986, -0.0312),
         (-0.0919, -0.1471, -0.0368), (-0.1017, -0.1169, -0.0872),
         ( 0.0059, -0.0647,  0.0765), ( 0.0000, -0.0256,  0.0256),
         ( 0.0235, -0.0181,  0.0650)],
        [(-0.0585, -0.0795, -0.0375), (-0.0144, -0.0553,  0.0240),
         (-0.0662, -0.1176, -0.0184), (-0.0523, -0.0681, -0.0366),
         ( 0.0118, -0.0588,  0.0824), (-0.0131, -0.0332,  0.0076),
         (-0.0433, -0.0794, -0.0090)],
        [(-0.0680, -0.0895, -0.0470), (-0.1010, -0.1466, -0.0577),
         (-0.1103, -0.1728, -0.0515), (-0.0654, -0.0842, -0.0467),
         ( 0.0294, -0.0471,  0.1059), ( 0.0021, -0.0230,  0.0265),
         ( 0.0273, -0.0128,  0.0674)],
        [(-0.0135, -0.0365,  0.0085), ( 0.0144, -0.0264,  0.0553),
         (-0.0735, -0.1250, -0.0221), ( 0.0413,  0.0220,  0.0607),
         ( 0.0353, -0.0412,  0.1118), ( 0.0263, -0.0007,  0.0526),
         ( 0.0830,  0.0397,  0.1264)],
    ]
    # /counts_by_policy/P1_audit_best_matched/perception/{n_strictly_negative,n_cells}
    N_NEG, N_CELLS = 17, 20
    # /counts_by_policy/P1_audit_best_matched/perception_pooled
    PD, PLO, PHI, PN = -0.0401, -0.0456, -0.0347, 30250

    M = np.array([[c[0] for c in r] for r in D])
    cmap = LinearSegmentedColormap.from_list(  # diverging blue<->red, neutral gray midpoint
        "dv", ["#104281", "#3987e5", "#9ec5f4", "#f0efec",
               "#f6b3b2", "#e34948", "#a11f1e"])
    vmax = 0.13

    fig = plt.figure(figsize=(W, 3.55))
    ax = fig.add_axes([0.175, 0.255, 0.735, 0.560])
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(len(fams)):
        for j in range(len(bench)):
            d, lo, hi = D[i][j]
            sig = (lo > 0) or (hi < 0)
            ax.text(j, i, f"{d:+.3f}" + ("*" if sig else ""), ha="center",
                    va="center", fontsize=7.2,
                    color="white" if abs(d) > 0.075 else INK,
                    fontweight="bold" if sig else "normal", zorder=6)

    for j in range(NP, len(bench)):  # reasoning columns: answer format not controlled
        ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(fams), facecolor="none",
                               edgecolor=INK, hatch="///", lw=0.0, alpha=0.22,
                               zorder=4))
    for x0, wd in ((-0.5, NP), (NP - 0.5, len(bench) - NP)):
        ax.add_patch(Rectangle((x0, -0.5), wd, len(fams), facecolor="none",
                               edgecolor=INK, lw=1.5, zorder=5))

    ax.set_xticks(range(len(bench)))
    ax.set_xticklabels(bench, fontsize=7.0)
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=7.6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text((NP - 1) / 2, -0.70, "perception — 20 cells", ha="center",
            va="bottom", fontsize=7.6, color=INK, fontweight="bold")
    ax.text(NP + 1, -0.70, "reasoning — 15 cells (hatched)", ha="center",
            va="bottom", fontsize=7.6, color=INK, fontweight="bold")

    cax = fig.add_axes([0.925, 0.255, 0.018, 0.560])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Δ accuracy: reasoning − direct", fontsize=6.9)
    cb.ax.tick_params(labelsize=6.6)
    cb.outline.set_visible(False)

    fig.text(0.012, 0.975,
             f"{N_NEG}/{N_CELLS} perception cells strictly negative;  pooled perception "
             f"Δ = {PD:+.4f}  [{PLO:+.4f}, {PHI:+.4f}],  n = {PN:,} items",
             fontsize=8.4, color=INK2, ha="left", va="top")

    foot(fig, "Policy P1 (audit-recommended best-matched think arm on disk; published "
              "no-think arm retained). * = 95% paired-bootstrap CI excludes 0. Per-cell "
              "n ranges 170–3,362; unweighted within cell.\n"
              "HATCHED: the reasoning columns are NOT answer-format-controlled — there "
              "the think/direct contrast confounds the reasoning trigger with the answer "
              "format. Format-matched, 0/9 trigger effects are significant "
              "(fig_format_vs_trigger).\n"
              "Source: finding1_corrected_2026-07-29.json "
              "/per_cell_by_policy/P1_audit_best_matched.", y=0.008)
    return finish(fig, "fig_finding1_crossfamily")


# =============================================================================
# 3. fig_format_vs_trigger
#    ARTIFACT: results/cascade_methods/artifacts/medeval_matched_direct_2026-07-29.json
#    KEY:      /cells[*]/attribution, /cells[*]/{delta_format,delta_matched}/sig
#              /verdict/{primary_cells, format_effect_sig_positive, sig_positive}
# =============================================================================
def fig_format_vs_trigger():
    # the 9 PRIMARY cells (/verdict/primary_cells = 9): 3 families x 3 benchmarks.
    # total   = /attribution/"total_published_gain(reason - truly_direct)"
    # fmt     = /attribution/"from_answer_format(boxed - bare, both trigger-free)"
    # trigger = /attribution/"from_explicit_reasoning_trigger(marginal)"
    # fsig    = /delta_format/sig       tsig = /delta_matched/sig
    groups = [
        ("Lingshu-32B", [
            ("MMMU-MCQonly",            0.0276, -0.0138, False,  0.0414, False),
            ("MedXpert-Reasoning",     -0.0035, -0.0076, False,  0.0041, False),
            ("MedXpert-Understanding",  0.0000, -0.0018, False,  0.0018, False)]),
        ("MedVLThinker-32B", [
            ("MMMU-MCQonly",            0.1034,  0.0621, False,  0.0414, False),
            ("MedXpert-Reasoning",      0.0463,  0.0456, True,   0.0007, False),
            ("MedXpert-Understanding",  0.0415,  0.0433, True,  -0.0018, False)]),
        ("InternVL3-38B", [
            ("MMMU-MCQonly",            0.1241,  0.0897, True,   0.0345, False),
            ("MedXpert-Reasoning",      0.0353,  0.0221, False,  0.0131, False),
            ("MedXpert-Understanding",  0.0199,  0.0090, False,  0.0108, False)]),
    ]
    N_FMT_SIG = 3   # /verdict/format_effect_sig_positive
    N_TRG_SIG = 0   # /verdict/sig_positive
    N_PRIM = 9      # /verdict/primary_cells

    # y layout: a header row per family, then its three cells
    ypos, ylabels, headers = [], [], []
    y = 11.6
    for fam, cells in groups:
        headers.append((y, fam))
        y -= 0.95
        for c in cells:
            ypos.append(y)
            ylabels.append(c[0])
            y -= 1.0
        y -= 0.35

    fig = plt.figure(figsize=(W, 4.85))
    ax = fig.add_axes([0.235, 0.265, 0.745, 0.600])
    h = 0.33
    flat = [c for _, cells in groups for c in cells]

    for (cell, tot, fmt, fsig, trg, tsig), yy in zip(flat, ypos):
        ax.barh(yy + h / 2 + 0.035, fmt, h, color=BLUE, zorder=3)
        ax.barh(yy - h / 2 - 0.035, trg, h, color=ORANGE, zorder=3)
        ax.plot(tot, yy, marker="D", ms=5.0, color=INK, mec=SURF, mew=0.9,
                zorder=6, ls="none")
        for v, yv, sig in ((fmt, yy + h / 2 + 0.035, fsig),
                           (trg, yy - h / 2 - 0.035, tsig)):
            ax.text(v + (0.0032 if v >= 0 else -0.0032), yv,
                    f"{v:+.4f}" + ("  ✱" if sig else ""), va="center",
                    ha="left" if v >= 0 else "right", fontsize=6.9, color=INK,
                    fontweight="bold" if sig else "normal")

    for yy, fam in headers:
        ax.text(-0.0295, yy, fam, ha="left", va="center", fontsize=7.8,
                color=INK, fontweight="bold")
        ax.axhline(yy + 0.52, color=GRID, lw=0.8, zorder=1)

    ax.axvline(0, color=INK, lw=0.9, zorder=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(min(ypos) - 0.75, 12.35)
    ax.set_xlim(-0.030, 0.142)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Δ accuracy (absolute)", fontsize=8.0, labelpad=3)
    ax.set_title("Under a matched prompt the reasoning trigger is worth ~nothing:\n"
                 f"{N_FMT_SIG}/{N_PRIM} answer-format effects significant, "
                 f"{N_TRG_SIG}/{N_PRIM} reasoning-trigger effects",
                 fontsize=8.5, color=INK2, pad=6, loc="left")

    ax.legend(handles=[
        Patch(facecolor=BLUE, label=r"answer-format component  (\boxed{} vs bare letter, both trigger-free)"),
        Patch(facecolor=ORANGE, label="reasoning-trigger component  (marginal, at matched answer format)"),
        Line2D([], [], marker="D", ls="none", color=INK, ms=5.0,
               label="total published gain  (reasoning arm − truly direct arm)"),
    ], loc="upper left", bbox_to_anchor=(-0.315, -0.100), ncol=1,
        handlelength=1.3, borderaxespad=0.0, labelspacing=0.32, fontsize=6.9)

    foot(fig, "✱ = 95% paired-bootstrap CI excludes 0 (10,000 resamples, seed 12345). "
              "Per-cell n: MMMU-MCQonly 145, MedXpert-Reasoning 1,446, "
              "MedXpert-Understanding 554; unweighted within cell.\n"
              "Source: medeval_matched_direct_2026-07-29.json /cells[*]/attribution.",
         y=0.010)
    return finish(fig, "fig_format_vs_trigger")


# =============================================================================
# 4. fig_accuracy_cost
#    ARTIFACT: results/cascade_methods/artifacts/headline_three_way_2026-08-03.json
#    KEY:      /four_column_progression/col5_all_three_plus_grounded_R32/
#                  {accuracy, cost_honest_recost}
#              /cost_constants/flop_ratio/{used, band}
#              /final_headline/comparison/method_accuracy_max_veto/*/{accuracy,flop_eq}
# =============================================================================
def fig_accuracy_cost():
    # col5 = the FINAL accounting: 8-cell macro, clean L1 verifier, matched
    # reasoning arm B, R32 = 3.816, reasoning baseline honestly re-costed per cell
    # from its own measured generation length.
    # accuracy <- /accuracy/<system>;  flops <- /cost_honest_recost/<system>/flops
    P = [("always-7B",                  0.5971, 1.000, "base"),
         ("always-32B, direct",         0.6567, 3.816, "base"),
         ("always-32B, with reasoning", 0.6250, 4.567, "base"),
         ("oracle mode-choice on 32B",  0.6573, 3.816, "base"),
         ("compute-lean",               0.6443, 6.358, "meth"),
         ("accuracy-max (fusion)",      0.6503, 7.188, "meth"),
         ("accuracy-max (veto)",        0.6575, 7.342, "meth")]
    R32 = 3.816              # /cost_constants/flop_ratio/used   (status: DERIVED)
    BAND = (3.734, 3.859)    # /cost_constants/flop_ratio/band
    # /final_headline/comparison/method_accuracy_max_veto/always_32b_reasoning/accuracy
    H_REA = (0.0325, 0.0237, 0.0412)
    # /final_headline/comparison/method_accuracy_max_veto/always_32b_direct/accuracy
    H_DIR = (0.0008, -0.0022, 0.0036)
    X_REA, X_DIR = 1.608, 1.924   # .../flop_eq/x  vs reasoning and vs direct

    fig = plt.figure(figsize=(W, 4.45))
    ax = fig.add_axes([0.105, 0.235, 0.875, 0.565])

    ax.axvspan(0.0, R32, color="#eaf2fd", zorder=0)
    ax.axvspan(BAND[0], BAND[1], color="#c9dcf6", zorder=1)
    ax.axvline(R32, color=INK, lw=1.1, ls=(0, (5, 3)), zorder=2)
    ax.text(1.88, 0.6470,
            "cheaper than one 32B forward\n— no method operating point lands here",
            ha="center", va="center", fontsize=7.4, color=INK2, style="italic")
    ax.text(R32 - 0.14, 0.6160,
            "one 32B forward:  FLOP-eq = R32 = 3.816\n"
            "DERIVED, not name-plate (4.57 rejected)\n"
            "shaded band = [3.734, 3.859]",
            ha="right", va="top", fontsize=6.9, color=INK, linespacing=1.3)

    for name, acc, fl, kind in P:
        if kind == "meth":
            ax.plot(fl, acc, marker="o", ms=7.0, color=BLUE, mec=SURF, mew=1.1,
                    ls="none", zorder=6)
        else:
            ax.plot(fl, acc, marker="s", ms=6.0, color=INK2, mec=SURF, mew=1.1,
                    ls="none", zorder=6)

    # direct labels, hand-placed (the two systems at x = 3.816 must not collide)
    lab = {"always-7B":                  (1.22, 0.5971, "left",   "center"),
           "oracle mode-choice on 32B":  (3.66, 0.6650, "right",  "bottom"),
           "always-32B, direct":         (3.96, 0.6528, "left",   "top"),
           "always-32B, with reasoning": (4.72, 0.6250, "left",   "center"),
           "compute-lean":               (6.16, 0.6418, "right",  "top"),
           "accuracy-max (fusion)":      (7.32, 0.6462, "left",   "top"),
           "accuracy-max (veto)":        (7.34, 0.6620, "center", "bottom")}
    for name, acc, fl, kind in P:
        tx, ty, ha, va = lab[name]
        ax.text(tx, ty, f"{name}\n{acc:.4f} @ {fl:.3f}×", ha=ha, va=va,
                fontsize=7.0, color=INK, linespacing=1.25, zorder=7)
    ax.annotate("", xy=(3.816, 0.6573), xytext=(3.70, 0.6640),
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7), zorder=5)

    ax.set_xlim(0.0, 9.45)
    ax.set_ylim(0.5915, 0.6765)
    ax.set_xticks(range(0, 10))
    ax.set_yticks(np.arange(0.60, 0.671, 0.01))
    ax.set_xlabel("compute per question, FLOP-eq   "
                  "(1.000× = one Lingshu-7B forward at the measured prompt geometry)",
                  fontsize=7.6, labelpad=3)
    ax.set_ylabel("accuracy\n(macro average over the 8 reporting cells)", fontsize=7.6)
    ax.grid(zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color=BLUE, ms=7.0,
               label="method operating point"),
        Line2D([], [], marker="s", ls="none", color=INK2, ms=6.0,
               label="baseline / oracle"),
    ], loc="lower right", handlelength=1.2, borderaxespad=0.5, labelspacing=0.3)

    ax.set_title(
        f"Final accounting — accuracy-max (veto) beats always-32B-with-reasoning by\n"
        f"{H_REA[0]:+.4f} [{H_REA[1]:+.4f}, {H_REA[2]:+.4f}] and ties always-32B-direct "
        f"({H_DIR[0]:+.4f} [{H_DIR[1]:+.4f}, {H_DIR[2]:+.4f}])\n"
        f"— at {X_REA:.3f}× and {X_DIR:.3f}× their FLOP-eq cost",
        fontsize=8.4, color=INK2, pad=6, loc="left")

    foot(fig, "Accounting: 8-cell macro average, open-text verifier retrained on strictly "
              "disjoint data (L1), reasoning baseline prompt-matched (arm B) and honestly "
              "re-costed per cell from its own measured generation length.\n"
              "R32 = 3.816 is DERIVED from exact safetensors parameter counts "
              "(8,292,166,656 and 33,452,718,336) plus measured prompt geometry "
              "(326.68 prompt / 280.48 image tokens).\n"
              "Source: headline_three_way_2026-08-03.json "
              "/four_column_progression/col5_all_three_plus_grounded_R32.", y=0.008)
    return finish(fig, "fig_accuracy_cost")


# =============================================================================
# 5. fig_verifier_contamination
#    ARTIFACT: results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json
#    KEY:      /levels/{L1_image_disjoint,L2_strict}/selection_stage/<ds>/
#                  {contrasts/*, auroc_candidate_*, oracle_conversion_*}
# =============================================================================
def fig_verifier_contamination():
    DS = ["SLAKE-open\nn = 645", "VQA-RAD-open\nn = 200",
          "PathVQA-open\nn = 1,500", "POOLED\nn = 2,345"]
    # /levels/L1_image_disjoint/selection_stage/<ds>/contrasts/
    #     contaminated_verifier_minus_greedy/{point, ci}
    CONTAM = [(0.04341, 0.01550, 0.07136), (0.11000, 0.06500, 0.16000),
              (0.12933, 0.11000, 0.14867), (0.10405, 0.08912, 0.11898)]
    # /levels/L1_image_disjoint/.../contrasts/clean_verifier_minus_greedy/{point, ci}
    CLEAN1 = [(0.01085, -0.01705, 0.03876), (0.01500, -0.03500, 0.06500),
              (0.04933,  0.03200, 0.06667), (0.03582,  0.02132, 0.05032)]
    # /levels/L2_strict/.../contrasts/clean_verifier_minus_greedy/{point, ci}
    CLEAN2 = [( 0.00000, -0.02791, 0.02791), ( 0.00000, -0.04500, 0.04500),
              (-0.01867, -0.03867, 0.00133), (-0.01194, -0.02772, 0.00341)]
    # POOLED mechanism: /levels/<lvl>/selection_stage/POOLED/
    #     auroc_candidate_{contaminated,clean} and oracle_conversion_{contaminated,clean}
    AUROC = [0.9433, 0.8856, 0.7960]   # contaminated / clean L1 / clean L2
    CONV = [0.5894, 0.2029, -0.0676]   # contaminated / clean L1 / clean L2

    fig = plt.figure(figsize=(W, 4.00))
    axL = fig.add_axes([0.135, 0.410, 0.425, 0.470])
    axR = fig.add_axes([0.705, 0.410, 0.275, 0.470])

    # ---- left: selection gain -------------------------------------------
    y = np.arange(4)[::-1].astype(float)
    off, h = 0.25, 0.22
    series = [("contaminated verifier", CONTAM, BLUE, +off),
              ("clean verifier, L1 (image-disjoint)", CLEAN1, ORANGE, 0.0),
              ("clean verifier, L2 (strict: no eval question text)", CLEAN2, AQUA, -off)]
    for name, vals, col, dy in series:
        pt = [v[0] for v in vals]
        lo = [v[0] - v[1] for v in vals]
        hi = [v[2] - v[0] for v in vals]
        axL.barh(y + dy, pt, h, color=col, zorder=3, label=name)
        axL.errorbar(pt, y + dy, xerr=[lo, hi], fmt="none", ecolor=INK,
                     elinewidth=0.85, capsize=2.2, capthick=0.85, zorder=5)
    axL.axvline(0, color=INK, lw=0.9, zorder=4)
    axL.axhline(0.52, color=GRID, lw=0.9, zorder=1)
    axL.set_yticks(y)
    axL.set_yticklabels(DS, fontsize=7.0)
    axL.tick_params(axis="y", length=0, pad=3)
    axL.spines["left"].set_visible(False)
    axL.set_xlim(-0.058, 0.180)
    axL.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15])
    axL.set_xlabel("selection gain: verifier-selected − greedy\n(absolute accuracy)",
                   fontsize=7.4, labelpad=2)
    axL.grid(axis="x", zorder=0)
    axL.set_axisbelow(True)
    axL.set_title("Gain — per dataset and pooled", fontsize=8.2, color=INK,
                  loc="left", pad=4)
    axL.legend(loc="upper left", bbox_to_anchor=(-0.185, -0.290), fontsize=6.9,
               handlelength=1.3, borderaxespad=0.0, labelspacing=0.28)

    # ---- right: mechanism (pooled) ---------------------------------------
    x = np.arange(2)
    bw = 0.25
    for i, col in enumerate([BLUE, ORANGE, AQUA]):
        vals = [AUROC[i], CONV[i]]
        axR.bar(x + (i - 1) * bw, vals, bw, color=col, zorder=3)
        for xx, vv in zip(x + (i - 1) * bw, vals):
            axR.text(xx, vv + (0.028 if vv >= 0 else -0.028), f"{vv:.3f}",
                     ha="center", va="bottom" if vv >= 0 else "top",
                     fontsize=6.5, color=INK)
    axR.axhline(0, color=INK, lw=0.9, zorder=4)
    axR.axhline(0.5, color=INK2, lw=0.7, ls=(0, (3, 3)), zorder=2)
    axR.text(1.46, 0.515, "chance", ha="right", va="bottom", fontsize=6.4, color=INK2)
    axR.set_xticks(x)
    axR.set_xticklabels(["ranking quality\n(candidate AUROC)",
                         "conversion\n(greedy→oracle@8\nheadroom realised)"],
                        fontsize=6.6)
    axR.tick_params(axis="x", length=0, pad=3)
    axR.set_xlim(-0.5, 1.5)
    axR.set_ylim(-0.22, 1.16)
    axR.set_yticks(np.arange(0.0, 1.01, 0.25))
    axR.set_ylabel("value (unitless)", fontsize=7.4)
    axR.grid(axis="y", zorder=0)
    axR.set_axisbelow(True)
    axR.set_title("Mechanism (pooled): ranking\nsurvived, conversion did not",
                  fontsize=8.0, color=INK, loc="left", pad=4)

    foot(fig, "95% CIs from 10,000 bootstrap resamples (seed 0). Per-dataset rows are "
              "unweighted within dataset; POOLED is item-weighted over 2,345 items / "
              "18,760 candidates. The contaminated arm is the same adapter in both level "
              "blocks — only the clean arm is retrained, so a zero-length bar "
              "(SLAKE-open, VQA-RAD-open at L2) is an exact 0.000 gain, not a missing "
              "value. Judge: MedVLThinker-32B, the same judge as the headline.\n"
              "Source: verifier_disjoint_retrain_2026-07-30.json /levels/*/selection_stage.",
         y=0.008)
    return finish(fig, "fig_verifier_contamination")


# =============================================================================
# 6. fig_pmc_defects
#    ARTIFACT: results/cascade_methods/artifacts/pmc_label_noise_audit_2026-07-29.json
#    KEY:      /classification_distributions/<group>/defective/{k,n,p,ci95}  (Wilson)
#              /bias_tests/<pair>/fisher_p_two_sided
# =============================================================================
def fig_pmc_defects():
    # defective = BAD-GOLD or UNANSWERABLE or MULTI-CORRECT (UNCLEAR never counts)
    G = [("fusion WINS\n(method right, 32B wrong)",       53, 100, 0.53, 0.4329, 0.6249, BLUE),
         ("fusion LOSSES\n(method wrong, 32B right)",     30,  50, 0.60, 0.4618, 0.7239, ORANGE),
         ("control: agree & correct\n(both models right)", 14,  50, 0.28, 0.1747, 0.4167, INK2)]
    P_WL = 0.48701  # /bias_tests/wins_vs_losses/fisher_p_two_sided
    P_LC = 0.00233  # /bias_tests/losses_vs_control/fisher_p_two_sided
    P_WC = 0.00510  # /bias_tests/wins_vs_control/fisher_p_two_sided

    fig = plt.figure(figsize=(W, 4.05))
    ax = fig.add_axes([0.115, 0.230, 0.865, 0.625])
    x = np.arange(3).astype(float)

    for xx, (lab, k, n, p, lo, hi, col) in zip(x, G):
        ax.bar(xx, p, 0.56, color=col, zorder=3)
        ax.errorbar(xx, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK,
                    elinewidth=1.2, capsize=4.0, capthick=1.2, zorder=5)
        ax.text(xx, p / 2 + 0.030, f"{p:.2f}", ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold", zorder=6)
        ax.text(xx, p / 2 - 0.042, f"{k} of {n} audited", ha="center", va="center",
                color="white", fontsize=6.8, zorder=6)

    def bracket(x0, x1, ytop, txt, sig):
        ax.plot([x0, x0, x1, x1], [ytop - 0.017, ytop, ytop, ytop - 0.017],
                color=INK, lw=0.9, zorder=6)
        ax.text((x0 + x1) / 2, ytop + 0.008, txt, ha="center", va="bottom",
                fontsize=7.2, color=INK, fontweight="bold" if sig else "normal")

    bracket(0, 1, 0.790, f"Fisher exact  p = {P_WL:.3f}   n.s.", False)
    bracket(1, 2, 0.885, f"p = {P_LC:.4f}  ✱", True)
    bracket(0, 2, 0.975, f"p = {P_WC:.4f}  ✱", True)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in G], fontsize=7.6)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("PMC-VQA item defect rate\n"
                  "(fraction of audited items, 95% Wilson interval)", fontsize=7.6)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Defects are enriched in the disagreement set — "
                 "but NOT biased toward the wins",
                 fontsize=8.5, color=INK2, pad=6, loc="left")

    foot(fig, "Defect = BAD-GOLD ∪ UNANSWERABLE ∪ MULTI-CORRECT under the audit rubric; "
              "UNCLEAR is never counted as a defect and hard-but-well-posed items count "
              "as GENUINE (a conservative rubric). Strata are drawn from the 33,430-item "
              "PMC-VQA evaluation (seed 20260729). ✱ = significant at α = 0.05.\n"
              "Source: pmc_label_noise_audit_2026-07-29.json "
              "/classification_distributions + /bias_tests.")
    return finish(fig, "fig_pmc_defects")


# =============================================================================
# 7. fig_latency_correction
#    ARTIFACT: results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json
#    KEY:      /reconciliation/contradiction/{claimed_latency_ms,
#                  claimed_latency_construction, claimed_energy_j}
#              /reconciliation/measurement/pooled/bo8_total/*
#              /reconciliation/canonical_constants/GEN32_nothink_ms
#              /reconciliation/resolution/{latency, energy, harness_reproduces_canonical}
# =============================================================================
def fig_latency_correction():
    ASSERT = 522.0    # /reconciliation/contradiction/claimed_latency_ms (MODELLED: 347.1 + 175.5)
    MEAS = 1305.3     # /reconciliation/measurement/pooled/bo8_total/lat_ms_mean (n = 45)
    MED = 1290.7      # /reconciliation/measurement/pooled/bo8_total/lat_ms_median
    FACT = 2.5        # /reconciliation/resolution/latency/factor_wrong
    GEN32 = 665.0     # /reconciliation/canonical_constants/GEN32_nothink_ms
    # /reconciliation/measurement/pooled/bo8_total/{per_replicate_lat_ms_mean, lat_ms_p10_p90}
    REP = [("replicate 1\nn = 20", 1325.7, 910.6, 1761.9),
           ("replicate 2\nn = 25", 1289.0, 946.4, 1656.5)]

    fig = plt.figure(figsize=(W, 3.95))
    ax = fig.add_axes([0.135, 0.290, 0.845, 0.545])
    y = [2.62, 1.20, 0.42]

    ax.barh(y[0], ASSERT, 0.38, color=ORANGE, zorder=3)
    ax.text(ASSERT + 24, y[0], f"{ASSERT:.0f} ms", va="center", ha="left",
            fontsize=8.8, color=INK, fontweight="bold")
    ax.text(24, y[0] - 0.30, "MODELLED, never measured  (GEN7 347.1 + VER7 175.5 ms)",
            va="top", ha="left", fontsize=7.0, color=INK2)

    for (lab, mean, p10, p90), yy in zip(REP, y[1:]):
        ax.barh(yy, p90 - p10, 0.38, left=p10, color=BLUE_L, zorder=3)
        ax.plot([mean, mean], [yy - 0.22, yy + 0.22], color=BLUE, lw=2.6, zorder=5)
        ax.text(p10 - 18, yy, f"p10 {p10:.0f}", va="center", ha="right",
                fontsize=6.6, color=INK2)
        ax.text(p90 + 18, yy, f"p90 {p90:.0f}", va="center", ha="left",
                fontsize=6.6, color=INK2)
        ax.text(mean, yy + 0.25, f"mean {mean:.1f}", va="bottom", ha="center",
                fontsize=7.0, color=INK)

    ax.axvline(MEAS, color=BLUE, lw=1.3, ls=(0, (5, 3)), zorder=6)
    ax.text(MEAS + 22, 3.36, f"MEASURED\npooled mean {MEAS:.1f} ms\n"
                             f"(n = 45, median {MED:.1f})",
            va="top", ha="left", fontsize=7.0, color=INK, fontweight="bold")
    ax.axvline(GEN32, color=INK, lw=1.1, ls=(0, (2, 2)), zorder=6)
    ax.text(GEN32 - 18, 3.36, f"one 32B forward\n{GEN32:.0f} ms", va="top",
            ha="right", fontsize=7.0, color=INK)

    ax.annotate("", xy=(MEAS, 1.94), xytext=(ASSERT, 1.94),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4,
                                shrinkA=0, shrinkB=0), zorder=7)
    ax.text((ASSERT + MEAS) / 2, 2.00,
            f"understated by {FACT:.1f}× — batching 8 does NOT make N drop out",
            ha="center", va="bottom", fontsize=7.4, color=RED, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(["asserted\nbest-of-8"] + [r[0] for r in REP], fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(0.0, 3.46)
    ax.set_xlim(0, 2050)
    ax.set_xticks(range(0, 2001, 250))
    ax.set_xlabel("wall-clock latency per question, batched best-of-8 (ms)",
                  fontsize=7.8, labelpad=3)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=ORANGE, label="asserted (modelled sum)"),
        Patch(facecolor=BLUE_L, label="measured p10–p90 range"),
        Line2D([], [], color=BLUE, lw=2.6, label="measured replicate mean"),
    ], loc="lower left", bbox_to_anchor=(0.002, 0.002), handlelength=1.3,
        borderaxespad=0.0, labelspacing=0.28, fontsize=6.9)

    foot(fig, "MEASURED: Lingshu-7B + LoRA verifier, HuggingFace batch-1 serving, cap320, "
              "real VQA-RAD images, NVML-instrumented on one A100 80GB PCIe (300 W cap), "
              "2 replicates, n = 45 questions. Harness validated against the canonical "
              "constants: single greedy generation 350.0 ms measured vs 347.1 ms canonical "
              "(+0.8%). Energy moves the other way: 316.7 J measured vs 568.6 J modelled "
              "(1.8× overstated).\n"
              "Source: bestofn_latency_energy_2026-08-03.json /reconciliation.", y=0.008)
    return finish(fig, "fig_latency_correction")


if __name__ == "__main__":
    print(f"writing to {OUT}\n")
    for f in (fig_correction_cascade, fig_finding1_crossfamily,
              fig_format_vs_trigger, fig_accuracy_cost,
              fig_verifier_contamination, fig_pmc_defects,
              fig_latency_correction):
        f()
    print(f"\nAll seven rendered at {DPI} dpi on a fixed {W} in canvas "
          f"(= {int(W * DPI)} px wide; insert at 100% scale).")
