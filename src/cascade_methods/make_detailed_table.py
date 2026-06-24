#!/usr/bin/env python3
"""
make_detailed_table.py - detailed, readable tables (numbers) for the cascade methods across families.
Reads results/cascade_methods/rescue_allfam/accv3_{family}.json (anchors + ACC-v1/v2/v3, honest
50/50 calib/test x20 seeds, min-think@parity). Renders per-family PNG tables (metrics + per-dataset
accuracy) AND a markdown master table. Outputs to paper/figs/rescue/.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RB = "results/cascade_methods/rescue_allfam"; OUT = "paper/figs/rescue"; os.makedirs(OUT, exist_ok=True)
FAMS = ["medvlthinker", "lingshu", "qoq"]
DATA = {f: json.load(open(f"{RB}/accv3_{f}.json")) for f in FAMS}
ORDER = ["always-small-nt", "always-big-nt", "always-big-think", "ACC-v1 (confidence)",
         "ACC-v2 (agreement)", "ACC-v3 (agree+confidence)"]
SHORT = {"always-small-nt":"always small-nt","always-big-nt":"always big-nt","always-big-think":"always big-think [parity]",
         "ACC-v1 (confidence)":"ACC-v1 confidence","ACC-v2 (agreement)":"ACC-v2 agreement (baseline)","ACC-v3 (agree+confidence)":"ACC-v3 agree+conf (ours)"}
ABBR = {"PMC-VQA":"PMC","SLAKE":"SLAKE","VQA-RAD":"VQARAD","PathVQA":"PathV","MMMU":"MMMU","MedXpert-Reasoning":"MedX-R","MedXpert-Understanding":"MedX-U"}
BENCHES = list(ABBR.keys())

def rowmap(fam, pool):
    return {r["method"]: r for r in DATA[fam]["pools"][pool]["rows"]}

def fmt(r, key, kind):
    if r is None or r.get(key) is None: return "—"
    v = r[key]
    if kind == "acc": return f"{v:.4f}"
    if kind == "pct": return f"{v*100:.0f}%"
    if kind == "lat": return f"{v:.2f}s"
    if kind == "en": return f"{v:.0f}J"
    if kind == "g": return f"{v:.2f}"
    return str(v)

# ---------- PNG: metrics table per family (ALL-6 + ALL-5) ----------
def metrics_table(fam):
    cols = ["method", "acc", "think%", "FLOPs%", "lat", "energy", "guard"]
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.6))
    for ax, pool in zip(axes, ["ALL-6", "ALL-5"]):
        rm = rowmap(fam, pool); parity = DATA[fam]["pools"][pool]["parity"]
        cells = []
        for m in ORDER:
            r = rm.get(m)
            cells.append([SHORT[m], fmt(r,"acc","acc"), fmt(r,"think","pct"), fmt(r,"flops","pct"),
                          fmt(r,"lat","lat"), fmt(r,"energy","en"), fmt(r,"guard","g")])
        ax.axis("off")
        cw = [0.30] + [0.117]*(len(cols)-1)
        t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center", colWidths=cw)
        t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
        for j in range(len(cols)): t[(0, j)].set_facecolor("#34495e"); t[(0, j)].set_text_props(color="white", weight="bold")
        for i, m in enumerate(ORDER, start=1):
            t[(i, 0)].set_text_props(ha="left")
            if "ACC-v3" in m:
                for j in range(len(cols)): t[(i, j)].set_facecolor("#d5f5e3")
            elif "ACC-v2" in m:
                for j in range(len(cols)): t[(i, j)].set_facecolor("#fdebd0")
        ax.set_title(f"{fam} — {pool}  (honest 50/50, 20 seeds; parity=always-big-think={parity:.4f})",
                     fontsize=10, weight="bold")
    fig.text(0.5, 0.005, "“—” = did not reach always-big-think parity in all 20 seeds.  "
             "ACC-v3 (green) = ours; ACC-v2 (orange) = baseline.  lower think%/FLOPs/lat/energy = better.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0,0.02,1,1]); p = f"{OUT}/table_metrics_{fam}.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p

# ---------- PNG: per-dataset accuracy table per family ----------
def perdataset_table(fam):
    pool = "ALL-6"; rm = rowmap(fam, pool)
    cols = ["method"] + [ABBR[b] for b in BENCHES]
    cells = []
    for m in ORDER:
        r = rm.get(m)
        row = [SHORT[m]]
        for b in BENCHES:
            row.append(f"{r['bench'][b]:.3f}" if (r and b in r.get("bench", {})) else "—")
        cells.append(row)
    fig, ax = plt.subplots(figsize=(11.5, 3.4)); ax.axis("off")
    cw = [0.28] + [0.103]*len(BENCHES)
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center", colWidths=cw)
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for j in range(len(cols)): t[(0, j)].set_facecolor("#34495e"); t[(0, j)].set_text_props(color="white", weight="bold")
    for i, m in enumerate(ORDER, start=1):
        t[(i, 0)].set_text_props(ha="left")
        if "ACC-v3" in m:
            for j in range(len(cols)): t[(i, j)].set_facecolor("#d5f5e3")
    ax.set_title(f"{fam} — per-benchmark accuracy @ parity (ALL-6)", fontsize=10, weight="bold")
    fig.tight_layout(); p = f"{OUT}/table_perdataset_{fam}.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close()
    return p

# ---------- Markdown master table ----------
def markdown():
    lines = ["# Detailed cascade results — ACC-v1/v2/v3 across families",
             "", "Honest 50/50 calib/test, 20 seeds, min-think@parity. parity = always-big-think.",
             "Measured batch-1 latency/energy. guard = #benchmarks below always-small.", ""]
    for fam in FAMS:
        lines.append(f"## {fam}")
        for pool in ["ALL-6", "ALL-5", "COMPETENT-4"]:
            rm = rowmap(fam, pool); parity = DATA[fam]["pools"][pool]["parity"]
            lines += [f"\n### {pool}  (parity={parity:.4f})", "",
                      "| method | acc | esc0 | think | FLOPs% | latency | energy | guard |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for m in ORDER:
                r = rm.get(m)
                lines.append(f"| {SHORT[m]} | {fmt(r,'acc','acc')} | {fmt(r,'esc0','pct')} | {fmt(r,'think','pct')} "
                             f"| {fmt(r,'flops','pct')} | {fmt(r,'lat','lat')} | {fmt(r,'energy','en')} | {fmt(r,'guard','g')} |")
            # per-dataset row block
            lines += ["", "| method | " + " | ".join(ABBR[b] for b in BENCHES) + " |",
                      "|---|" + "---:|"*len(BENCHES)]
            for m in ORDER:
                r = rm.get(m)
                cellvals = " | ".join((f"{r['bench'][b]:.3f}" if (r and b in r.get('bench',{})) else "—") for b in BENCHES)
                lines.append(f"| {SHORT[m]} | {cellvals} |")
        lines.append("")
    p = "results/cascade_methods/DETAILED_TABLES.md"; open(p, "w").write("\n".join(lines)); return p

if __name__ == "__main__":
    ps = []
    for fam in FAMS:
        ps.append(metrics_table(fam)); ps.append(perdataset_table(fam))
    ps.append(markdown())
    print("written:")
    for p in ps: print("  " + p)
