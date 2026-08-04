#!/usr/bin/env python3
"""crossfamily_verifier_fig.py -- the de-correlation scatter for the cross-family verifier sweep.

Reads results/cascade_methods/artifacts/crossfamily_verifier_sweep_2026-08-04.json (measured) and
draws the two panels the hypothesis lives or dies on:
  LEFT   x = de-correlation from the generator,  y = REALISED selection efficiency
  RIGHT  x = de-correlation from the generator,  y = pair-oracle headroom over the trained verifier
           (= information the cross-family scorer holds that the same-family verifier lacks)

Single series per panel, every point directly labelled, so no legend and no colour-coded identity.

  python3 src/cascade_methods/crossfamily_verifier_fig.py
  -> results/cascade_methods/artifacts/crossfamily_decorrelation_2026-08-04.png
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)
SRC = J("results/cascade_methods/artifacts/crossfamily_verifier_sweep_2026-08-04.json")
OUT = J("results/cascade_methods/artifacts/crossfamily_decorrelation_2026-08-04.png")

INK, INK2, MUTED, ACCENT, GRID = "#1a1a1a", "#555555", "#8a8a8a", "#2b6cb0", "#e2e2e2"
SHORT = {"lingshu7b_zs": "Lingshu-7B\n(generator itself)", "qwen25vl7b": "Qwen2.5-VL-7B",
         "mvt7b": "MedVLThinker-7B", "internvl3_8b": "InternVL3-8B", "medgemma4b": "MedGemma-4B",
         "qoqmed7b": "QoQ-Med-VL-7B", "chiron_o1_8b": "Chiron-o1-8B", "lingshu32b": "Lingshu-32B"}

d = json.load(open(SRC))
law = d["decorrelation_law"]["x2_score_corr_with_generator"]
base = d["table"]["POOLED"]["verifiers"]["trained7b_clean"]["sel_eff"]

LABELS = []
fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2))
panels = [
    (axes[0], law["scatter"], "sel_eff", "Realised selection efficiency @8 (pooled)",
     f"a. De-correlation does NOT buy selection\nPearson r = {law['pearson']:+.2f}, "
     f"Spearman = {law['spearman']:+.2f}  (n={law['n_points']})", base,
     "trained same-family verifier"),
]
po = law.get("vs_pair_oracle_headroom")
if po:
    panels.append((axes[1], po["scatter"], "pair_oracle_headroom",
                   "Pair-oracle headroom over trained verifier",
                   f"b. …but it DOES buy available information\nPearson r = {po['pearson']:+.2f}, "
                   f"Spearman = {po['spearman']:+.2f}  (n={len(po['scatter'])})", None, None))

for ax, pts, ykey, ylab, title, hline, hlab in panels:
    xs = [p["x"] for p in pts]; ys = [p[ykey] for p in pts]
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED); ax.spines[s].set_linewidth(1.0)
    if hline is not None:
        ax.axhline(hline, color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
        ax.annotate(hlab, xy=(min(xs), hline), xytext=(0, 5), textcoords="offset points",
                    fontsize=8.5, color=INK2, ha="left", va="bottom")
    ax.scatter(xs, ys, s=95, color=ACCENT, edgecolor="white", linewidth=2.0, zorder=3)
    pad = 0.06 * (max(xs) - min(xs) or 1)
    ax.set_xlim(min(xs) - 2.2 * pad, max(xs) + 2.2 * pad)
    lo, hi = min(ys), max(ys); span = (hi - lo) or 0.05
    ax.set_ylim(lo - 0.45 * span, hi + 0.62 * span)
    LABELS.append((ax, pts, ykey))
    ax.set_xlabel("Score correlation with the generator's own P(Yes)\n(low = different information)",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel(ylab, fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left", pad=10)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)

# Lay the figure out FIRST -- label placement below works in display coordinates, so it must run
# against the final axes geometry, not the pre-tight_layout one.
fig.suptitle("Cross-family zero-shot verifiers on a fixed Lingshu-7B best-of-8 pool "
             "(SLAKE/VQA-RAD/PathVQA open, n=2,345)", fontsize=11, color=INK, x=0.011, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))

# Greedy label placement: try candidate offsets in preference order, keep the first that does not
# overlap an already-placed label or a data point and stays inside the panel. A thin leader line is
# drawn when the label is pushed away from its mark.
fig.canvas.draw()
for ax, pts, ykey in LABELS:
    placed = []
    marks = [ax.transData.transform((p["x"], p[ykey])) for p in pts]
    for m in marks:
        placed.append((m[0] - 13, m[1] - 13, m[0] + 13, m[1] + 13))
    order = sorted(range(len(pts)), key=lambda i: -pts[i][ykey])
    CANDS = [(0, 16), (0, -26), (0, 30), (0, -40), (0, 46), (0, -56), (0, 62), (0, -74),
             (46, 8), (-46, 8), (46, -14), (-46, -14), (46, 34), (-46, 34), (46, -44), (-46, -44),
             (0, 80), (0, -92), (78, 20), (-78, 20), (78, -30), (-78, -30)]
    bb = ax.get_window_extent()
    for i in order:
        p = pts[i]; txt = SHORT.get(p["verifier"], p["verifier"])
        lines = txt.split("\n")
        w = 5.4 * max(len(s) for s in lines) + 6; h = 12.5 * len(lines) + 4
        px, py = marks[i]
        best = None
        for dx, dy in CANDS:
            cx, cy = px + dx, py + dy
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if not (bb.x0 + 2 <= box[0] and box[2] <= bb.x1 - 2 and
                    bb.y0 + 2 <= box[1] and box[3] <= bb.y1 - 2):
                continue                                   # keep every label inside its own panel
            if not any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3] for b in placed):
                best = (dx, dy, box); break
        if best is None:
            dx, dy = CANDS[0]; best = (dx, dy, (px + dx - w / 2, py + dy - h / 2, px + dx + w / 2, py + dy + h / 2))
        dx, dy, box = best
        placed.append(box)
        ax.annotate(txt, xy=(p["x"], p[ykey]), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontsize=8.5, color=INK2, linespacing=1.15, zorder=4,
                    arrowprops=(dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkA=1, shrinkB=6)
                                if abs(dx) > 20 or abs(dy) > 18 else None))

fig.savefig(OUT, dpi=200, facecolor="white")
print("->", OUT)
