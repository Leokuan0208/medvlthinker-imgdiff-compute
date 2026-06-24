#!/usr/bin/env python3
"""
make_rescue_charts.py - charts for integrating the Visual-Stability Rescue into ACC-v2.
Reads results/cascade_methods/rescue_allfam/{medvlthinker,lingshu,qoq,control_think_signals}.json.
Outputs PNGs to paper/figs/rescue/. Honest set: the control (resolution-stability is a WORSE
think-skip signal than random/confidence), the per-family parity cost comparison, the MedVLThinker
accuracy-latency frontier, and per-dataset accuracy at parity.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RB = "results/cascade_methods/rescue_allfam"; OUT = "paper/figs/rescue"; os.makedirs(OUT, exist_ok=True)
def load(f): return json.load(open(os.path.join(RB, f)))
FAMS = ["medvlthinker", "lingshu", "qoq"]
DATA = {f: load(f"{f}.json") for f in FAMS}
CTRL = load("control_think_signals.json")
ABBR = {"PMC-VQA":"PMC","SLAKE":"SLAKE","VQA-RAD":"VQARAD","PathVQA":"PathV","MMMU":"MMMU",
        "MedXpert-Reasoning":"MedX-R","MedXpert-Understanding":"MedX-U"}

# ---- Chart 1: CONTROL — think-skip signal vs latency at parity (the key honest result) ----
def chart_control():
    sigs = ["none", "stability", "random", "bignt_conf", "inverse"]
    nice = {"none":"ACC-v2\n(no skip)","stability":"resolution\nstability (ours)","random":"random\n(matched)",
            "bignt_conf":"big-nt\nconfidence","inverse":"inverse\nstability"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, pool in zip(axes, ["ALL-5", "COMPETENT-4"]):
        lats = [CTRL[pool][s]["lat"] for s in sigs]
        colors = ["#999999","#d62728","#1f77b4","#2ca02c","#9467bd"]
        bars = ax.bar(range(len(sigs)), lats, color=colors)
        for i, v in enumerate(lats): ax.text(i, v+0.02, f"{v:.2f}s", ha="center", fontsize=9)
        ax.set_xticks(range(len(sigs))); ax.set_xticklabels([nice[s] for s in sigs], fontsize=8)
        ax.set_ylabel("batch-1 latency (s) @ parity"); ax.set_title(f"{pool}: which think-skip signal? (lower=better)")
        ax.axhline(lats[0], ls="--", c="#999999", lw=0.8)
    fig.suptitle("Resolution-stability is the WORST think-skip signal: random & big-nt-confidence beat it\n"
                 "(MedVLThinker, min-latency at always-big-think parity) — the rescue does not help ACC-v2's think tier",
                 fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.92]); p = f"{OUT}/control_think_signals.png"; fig.savefig(p, dpi=130); plt.close()
    return p

# ---- Chart 2: MedVLThinker accuracy-vs-latency frontier (ACC-v2 vs rescue placements) ----
def chart_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    modes = [("ACC-v2 (agreement)","#1f77b4","o"),("ACC-v2 + rescue@tier0","#d62728","s"),("ACC-v2 + rescue@think","#2ca02c","^")]
    for ax, pool in zip(axes, ["ALL-5","COMPETENT-4"]):
        P = DATA["medvlthinker"]["pools"][pool]
        for nm, c, mk in modes:
            fr = sorted(P["frontiers"][nm], key=lambda x: x["lat"])
            ax.plot([p["lat"] for p in fr], [p["acc"] for p in fr], mk+"-", c=c, ms=3, lw=1, label=nm)
        ax.axhline(P["parity"], ls="--", c="k", lw=0.8, label="big-think parity")
        ax.set_xlabel("batch-1 latency (s)"); ax.set_ylabel("accuracy"); ax.set_title(f"MedVLThinker {pool}")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Accuracy–latency frontier: rescue@tier0 (red) is dominated; rescue@think (green) shifts left of ACC-v2\n"
                 "but a plain confidence think-gate does better (see control chart)", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.92]); p = f"{OUT}/medvlthinker_frontier.png"; fig.savefig(p, dpi=130); plt.close()
    return p

# ---- Chart 3: cross-family parity cost — ACC-v2 vs rescue@think (latency, energy, FLOPs) ----
def chart_crossfam():
    pool = "ALL-5"; fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    metrics = [("lat","latency (s)"),("energy","energy (J)"),("flops","FLOPs (% of always-big-think)")]
    x = np.arange(len(FAMS)); w = 0.38
    for ax, (mk, ml) in zip(axes, metrics):
        v2, rs = [], []
        for f in FAMS:
            hp = DATA[f]["pools"][pool]["honest_parity"]
            a = hp.get("ACC-v2 (agreement)", {}); b = hp.get("ACC-v2 + rescue@think", {})
            def get(d):
                v = d.get(mk)
                return (v*100 if mk=="flops" and v is not None else v) or 0
            v2.append(get(a)); rs.append(get(b))
        ax.bar(x-w/2, v2, w, label="ACC-v2", color="#1f77b4")
        ax.bar(x+w/2, rs, w, label="ACC-v2 + rescue@think", color="#2ca02c")
        ax.set_xticks(x); ax.set_xticklabels(FAMS, fontsize=9); ax.set_ylabel(ml); ax.set_title(ml.split(" (")[0])
        ax.legend(fontsize=8)
    fig.suptitle("Cross-family at big-think parity (ALL-5): rescue@think helps MedVLThinker (slow think), "
                 "modest for Lingshu (fast think), no-op for QoQ (degenerate cascade)", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.93]); p = f"{OUT}/crossfamily_parity_cost.png"; fig.savefig(p, dpi=130); plt.close()
    return p

# ---- Chart 4: per-dataset accuracy at parity (MedVLThinker ALL-6) ----
def chart_perdataset():
    P = DATA["medvlthinker"]["pools"]["ALL-6"]; names = list(ABBR.keys())
    base = P["base_bench"]; hp = P["honest_parity"]
    series = [("always-small-nt", base["always-small-nt"], "#cccccc"),
              ("always-big-think", base["always-big-think"], "#000000")]
    if hp.get("ACC-v2 (agreement)", {}).get("reach", 0) >= 1 and "bench" in hp.get("ACC-v2 (agreement)", {}):
        series.append(("ACC-v2", hp["ACC-v2 (agreement)"]["bench"], "#1f77b4"))
    fig, ax = plt.subplots(figsize=(11, 4.3)); x = np.arange(len(names)); w = 0.8/len(series)
    for i, (nm, bench, c) in enumerate(series):
        ax.bar(x+i*w, [bench.get(d, 0) for d in names], w, label=nm, color=c)
    ax.set_xticks(x+w*(len(series)-1)/2); ax.set_xticklabels([ABBR[d] for d in names], fontsize=9)
    ax.set_ylabel("accuracy"); ax.set_title("MedVLThinker per-benchmark accuracy (ALL-6) — ACC-v2 at parity vs anchors")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout(); p = f"{OUT}/medvlthinker_perdataset.png"; fig.savefig(p, dpi=130); plt.close()
    return p

if __name__ == "__main__":
    ps = [chart_control(), chart_frontier(), chart_crossfam(), chart_perdataset()]
    print("charts written:")
    for p in ps: print("  " + p)
