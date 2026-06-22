#!/usr/bin/env python3
"""
make_master_charts.py - EVERY data point across all families/methods into comparable charts + tables.
Reads results/cascade_methods/allmethods_<family>.json (written by acc_allmethods.py) + PEAK_VRAM from
logs/lat_*.log. Emits: master_data.csv (one row per family x pool x method, all metrics + per-benchmark),
MASTER_TABLES.md (markdown), and charts in paper/figs/master/. matplotlib + stdlib only.
Run after: for f in medvlthinker lingshu qoq chiron medgemma; do python3 acc_allmethods.py --family $f; done
"""
import os, json, glob, csv, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
RES = os.path.join(REPO, "results/cascade_methods"); FIG = os.path.join(REPO, "paper/figs/master")
os.makedirs(FIG, exist_ok=True)
FAMS = ["medvlthinker", "lingshu", "qoq", "chiron", "medgemma"]
FAM_LABEL = {"medvlthinker": "MedVLThinker 7B/32B", "lingshu": "Lingshu 7B/32B", "qoq": "QoQ-Med 7B/32B",
             "chiron": "Chiron-o1 2B/8B", "medgemma": "MedGemma 4B/27B"}
BENCH = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU", "MedXpert-Reasoning", "MedXpert-Understanding"]
BAB = ["PMC", "SLAKE", "VQARAD", "PathV", "MMMU", "MX-R", "MX-U"]
BASE = ["always-small-nt (cheap)", "always-big-nt", "always-big-think [PARITY]"]
OURS = "Ours (ACC-v2: agreement)"

D = {}
for fam in FAMS:
    p = os.path.join(RES, f"allmethods_{fam}.json")
    if os.path.exists(p): D[fam] = json.load(open(p))
def vram(fam):  # peak VRAM per tier from logs
    out = {}
    for tier, tag in [("small_nt", "s"), ("big_nt", "bnt"), ("big_think", "bt")]:
        f = os.path.join(REPO, f"logs/lat_{fam}_{tag}.log")
        if os.path.exists(f):
            m = re.findall(r"PEAK_VRAM_GB=([\d.]+)", open(f).read())
            if m: out[tier] = float(m[-1])
    return out
VR = {fam: vram(fam) for fam in D}

# ---------- master CSV ----------
cols = ["family", "pool", "method", "acc", "esc0_pct", "think_pct", "flops_pct", "latency_s", "energy_j", "guard"] + BAB
with open(os.path.join(RES, "master_data.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for fam in FAMS:
        if fam not in D: continue
        for pool, pd in D[fam]["pools"].items():
            poolname = "ALL-6" if "6" in pool or pool == "ALL-6" else "ALL-5"
            for r in pd["rows"]:
                bench = [round(r["bench"].get(b, float("nan")), 4) for b in BENCH]
                w.writerow([fam, poolname, r["method"], round(r["acc"], 4), round(r["esc0"], 1), round(r["think"], 1),
                            round(r["flops"], 1), round(r["lat"], 3), round(r["energy"], 1), round(r["guard"], 2)] + bench)
print("wrote master_data.csv")

# ---------- markdown tables ----------
def row_get(fam, pool, method):
    for r in D[fam]["pools"][pool]["rows"]:
        if r["method"] == method: return r
    return None
def poolkey(fam, want6):
    for k in D[fam]["pools"]:
        if (want6 and k == "ALL-6") or (not want6 and "5" in k): return k
    return None
with open(os.path.join(RES, "MASTER_TABLES.md"), "w") as fh:
    fh.write("# Master tables — every data point (5 families × all methods)\n\n")
    fh.write("Cost legend: FLOPs% vs always-big-think; latency = batch-1 (s); energy = batch-1 (J); guard = "
             "#benchmarks worse than always-small (0=clean). Source: `acc_allmethods.py` JSON dumps + NVML logs.\n\n")
    # peak VRAM
    fh.write("## Peak VRAM (GB, batch-1)\n\n| family | small | big-nt | big-think |\n|---|---|---|---|\n")
    for fam in FAMS:
        v = VR.get(fam, {})
        fh.write(f"| {FAM_LABEL[fam]} | {v.get('small_nt','—')} | {v.get('big_nt','—')} | {v.get('big_think','—')} |\n")
    # per-family all-methods tables
    for fam in FAMS:
        if fam not in D: continue
        for want6 in (True, False):
            pk = poolkey(fam, want6)
            if not pk: continue
            pd = D[fam]["pools"][pk]
            fh.write(f"\n## {FAM_LABEL[fam]} — {'ALL-6' if want6 else 'ALL-5'} (parity={pd['parity']:.4f})\n\n")
            fh.write("| method | acc | esc0% | think% | FLOPs% | lat(s) | energy(J) | guard |\n|---|---|---|---|---|---|---|---|\n")
            for r in pd["rows"]:
                fh.write(f"| {r['method']} | {r['acc']:.4f} | {r['esc0']:.0f} | {r['think']:.0f} | {r['flops']:.1f} | "
                         f"{r['lat']:.2f} | {r['energy']:.1f} | {r['guard']:.2f} |\n")
    # per-benchmark accuracy (3 baselines + Ours), ALL-6
    fh.write("\n## Per-benchmark accuracy (ALL-6): baselines + Ours\n\n")
    fh.write("| family | config | " + " | ".join(BAB) + " |\n|" + "---|" * (len(BAB) + 2) + "\n")
    for fam in FAMS:
        if fam not in D: continue
        pk = poolkey(fam, True)
        for cfg in BASE + [OURS]:
            r = row_get(fam, pk, cfg)
            if not r: continue
            nm = cfg.replace(" (cheap)", "").replace(" [PARITY]", "").replace(" (ACC-v2: agreement)", "")
            fh.write(f"| {fam} | {nm} | " + " | ".join(f"{r['bench'].get(b,0):.3f}" for b in BENCH) + " |\n")
print("wrote MASTER_TABLES.md")

COLORS = plt.cm.tab10.colors
# ---------- Chart 1: per-benchmark accuracy heatmap (family x config rows, benchmark cols) ----------
rows_lbl, mat = [], []
for fam in FAMS:
    if fam not in D: continue
    pk = poolkey(fam, True)
    for cfg, short in zip(BASE, ["small-nt", "big-nt", "big-think"]):
        r = row_get(fam, pk, cfg)
        if not r: continue
        rows_lbl.append(f"{fam[:8]}·{short}"); mat.append([r["bench"].get(b, np.nan) for b in BENCH])
mat = np.array(mat)
fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows_lbl) + 1.5))
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=0.9)
ax.set_xticks(range(len(BAB))); ax.set_xticklabels(BAB, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(len(rows_lbl))); ax.set_yticklabels(rows_lbl, fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6)
ax.set_title("Per-benchmark accuracy — every family × config (ALL-6)"); fig.colorbar(im, fraction=0.025)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_perbench_heatmap.png"), dpi=150); plt.close(fig)

# ---------- Chart 2 & 3: accuracy vs latency / energy frontier (5 subplots) ----------
for metric, fname, xlab in [("lat", "fig_acc_vs_latency.png", "batch-1 latency (s, log)"),
                            ("energy", "fig_acc_vs_energy.png", "batch-1 energy (J, log)")]:
    fig, axes = plt.subplots(1, len([f for f in FAMS if f in D]), figsize=(4 * len(D), 3.8), squeeze=False)
    for ax, fam in zip(axes[0], [f for f in FAMS if f in D]):
        pk = poolkey(fam, True); pd = D[fam]["pools"][pk]
        for r in pd["rows"]:
            isb = r["method"] in BASE; iso = r["method"] == OURS
            ax.scatter(max(r[metric], 1e-3), r["acc"], s=(120 if iso else 55),
                       marker=("*" if isb else ("o" if iso else ".")),
                       c=("crimson" if iso else ("black" if isb else "steelblue")), zorder=3)
            if isb or iso:
                ax.annotate(r["method"].split(" (")[0].replace("always-", ""), (max(r[metric], 1e-3), r["acc"]),
                            fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.axhline(pd["parity"], ls="--", c="gray", lw=0.8)
        ax.set_xscale("log"); ax.set_title(FAM_LABEL[fam], fontsize=9); ax.set_xlabel(xlab, fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0][0].set_ylabel("accuracy (ALL-6)")
    fig.suptitle(f"Accuracy vs {xlab.split(' (')[0]} — all methods, all families (ALL-6)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)

# ---------- Chart 4: cost bars (FLOPs%, latency, energy) for the 3 baselines per family ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = [("flops", "FLOPs % of always-big-think"), ("lat", "batch-1 latency (s)"), ("energy", "batch-1 energy (J)")]
x = np.arange(len([f for f in FAMS if f in D])); w = 0.26
for ax, (mk, title) in zip(axes, metrics):
    for i, cfg in enumerate(BASE):
        vals = [(row_get(f, poolkey(f, True), cfg) or {}).get(mk, np.nan) for f in FAMS if f in D]
        ax.bar(x + (i - 1) * w, vals, w, label=cfg.split(" (")[0].split(" [")[0].replace("always-", ""), color=COLORS[i])
    ax.set_xticks(x); ax.set_xticklabels([f[:8] for f in FAMS if f in D], rotation=20, fontsize=8)
    ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=0.3)
    if mk != "flops": ax.set_yscale("log")
axes[0].legend(fontsize=7)
fig.suptitle("Cost per tier across families (ALL-6)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cost_bars.png"), dpi=150); plt.close(fig)

# ---------- Chart 5: ALL-5 vs ALL-6 accuracy (baselines + Ours) per family ----------
fig, axes = plt.subplots(1, len([f for f in FAMS if f in D]), figsize=(3.6 * len(D), 3.8), squeeze=False)
cfgs = BASE + [OURS]; short = ["sm-nt", "bg-nt", "bg-th", "Ours"]
for ax, fam in zip(axes[0], [f for f in FAMS if f in D]):
    a6 = [(row_get(fam, poolkey(fam, True), c) or {}).get("acc", np.nan) for c in cfgs]
    a5 = [(row_get(fam, poolkey(fam, False), c) or {}).get("acc", np.nan) for c in cfgs]
    xx = np.arange(len(cfgs))
    ax.bar(xx - 0.2, a6, 0.4, label="ALL-6", color="steelblue")
    ax.bar(xx + 0.2, a5, 0.4, label="ALL-5", color="orange")
    ax.set_xticks(xx); ax.set_xticklabels(short, fontsize=7, rotation=15); ax.set_title(FAM_LABEL[fam], fontsize=9)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0.4, 0.85)
axes[0][0].set_ylabel("accuracy"); axes[0][0].legend(fontsize=7)
fig.suptitle("ALL-5 vs ALL-6 accuracy (baselines + Ours)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_all5_all6_acc.png"), dpi=150); plt.close(fig)
print("wrote 5 charts to paper/figs/master/")
