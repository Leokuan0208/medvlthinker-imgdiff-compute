#!/usr/bin/env python3
"""
acc_2size.py - ACC generalization to OTHER 2-size medical VLM families (Lingshu 7B/32B, MedGemma 4B/27B).
Per family: (1) the make-or-break PREMISE -- does big-NO-THINK >= big-THINK on perception (per benchmark)?
(2) the size ladder + ACC cascade (small-nt -> big-nt -> big-think, agreement think-gate) at parity:
accuracy, escalation, exact prefill-inclusive FLOPs%. (Latency not measured for the new families -- FLOPs
is the exact, model-agnostic cost.) Honest 50/50 split, 20 seeds.
Usage: python3 src/cascade_methods/acc_2size.py --family lingshu   (or medgemma)
"""
import sys, os, glob, json, re, argparse; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs, ALL6, ALL5
J = lambda p: os.path.join("/home/jamesyang/medvlthinker-imgdiff-compute", p)
FAM = {
    "lingshu": dict(Ns=7.6e9, Nb=33.0e9, small=("ckpts/acc_gen/lingshu7b/cap320", "lingshu7b"),
                    bignt=("ckpts/acc_gen/lingshu32b/nothink_cap320", "lingshu32b"),
                    bigth=("ckpts/acc_gen/lingshu32b/think_native", "lingshu32b"),
                    sweep="ckpts/acc_gen/lingshu7b"),
    "medgemma": dict(Ns=4.3e9, Nb=27.0e9, small=("ckpts/acc_gen/medgemma4b/nt", "medgemma4b_nt"),
                     bignt=("ckpts/acc_gen/medgemma27b/nt", "medgemma27b_nt"),
                     bigth=("ckpts/acc_gen/medgemma27b/think_native", "medgemma27b_think"), sweep=None),
    "medvlthinker": dict(Ns=7.6e9, Nb=33.0e9, small=("ckpts/gate_7b_prune/cap320", "nothink_norag"),
                         bignt=("ckpts/gate_32b_modes/nothink_cap320", "nothink_norag"),
                         bigth=("ckpts/gate_32b", "think_norag"), sweep="ckpts/gate_7b_prune"),
    "qoq": dict(Ns=7.6e9, Nb=33.0e9, small=("ckpts/acc_gen/qoq7b/cap320", "qoq7b"),
                bignt=("ckpts/acc_gen/qoq32b/nothink_cap320", "qoq32b"),
                bigth=("ckpts/acc_gen/qoq32b/think_native", "qoq32b"), sweep="ckpts/acc_gen/qoq7b"),
    "chiron": dict(Ns=2.0e9, Nb=8.0e9, small=("ckpts/acc_gen/chiron2b/nt", "chiron2b_nt"),
                   bignt=("ckpts/acc_gen/chiron8b/nt", "chiron8b_nt"),
                   bigth=("ckpts/acc_gen/chiron8b/think_native", "chiron8b_think"), sweep=None),
}
ABBR = {"PMC-VQA": "PMC", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD", "PathVQA": "PathV", "MMMU": "MMMU",
        "MedXpert-Reasoning": "MX-R", "MedXpert-Understanding": "MX-U"}
def loadarm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(J(os.path.join(d, f"ckpt_*{tag}*.jsonl"))):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if m:
            for l in open(f):
                if l.strip(): r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out
A = argparse.ArgumentParser(); A.add_argument("--family", required=True, choices=list(FAM)); A = A.parse_args()
C = FAM[A.family]; Ns, Nb = C["Ns"], C["Nb"]
s = loadarm(*C["small"]); bn = loadarm(*C["bignt"]); bt = loadarm(*C["bigth"])
rows = []
for ds in ALL6:
    idx = sorted(set(s.get(ds, {})) & set(bn.get(ds, {})) & set(bt.get(ds, {})))
    for i in idx:
        sg = signals_from_logprobs(s[ds][i].get("opt_logprobs"))
        rows.append(dict(ds=ds, oks=s[ds][i]["ok"], okbn=bn[ds][i]["ok"], okbt=bt[ds][i]["ok"],
                         ps=s[ds][i]["pred"], pbn=bn[ds][i]["pred"], margin=sg["margin"],
                         gbt=bt[ds][i].get("gen_tokens") or 300))
R = {k: np.array([r[k] for r in rows]) for k in ["oks", "okbn", "okbt", "margin", "gbt"]}
ds_of = np.array([r["ds"] for r in rows]); dis = np.array([float(r["ps"] != r["pbn"]) for r in rows])
oks, okbn, okbt = R["oks"].astype(int), R["okbn"].astype(int), R["okbt"].astype(int)
print(f"\n############  ACC 2-SIZE GENERALIZATION: {A.family.upper()}  (n={len(rows)})  ############")
print(f"  small={C['small'][0].split('/')[-2]}  big-nt={C['bignt'][0].split('/')[-1]}  big-think={C['bigth'][0].split('/')[-1]}")
print(f"\n  --- PREMISE: per-benchmark accuracy (does big-NO-THINK >= big-THINK on perception?) ---")
print(f"  {'benchmark':<12}{'small-nt':>9}{'big-nt':>8}{'big-think':>10}{'nt-think':>9}{'n':>6}")
for ds in ALL6:
    m = ds_of == ds
    if not m.any(): continue
    d = okbn[m].mean() - okbt[m].mean()
    print(f"  {ds:<12}{oks[m].mean():>9.3f}{okbn[m].mean():>8.3f}{okbt[m].mean():>10.3f}{d:>+9.3f}{m.sum():>6}{'  <-nt wins' if d>0.005 else ''}")
def pooled(names):
    m = np.isin(ds_of, names); return oks[m].mean(), okbn[m].mean(), okbt[m].mean()
for lab, names in [("POOLED ALL-6", ALL6), ("POOLED ALL-5", ALL5)]:
    a, b, c = pooled(names); print(f"  {lab:<12}{a:>9.3f}{b:>8.3f}{c:>10.3f}{b-c:>+9.3f}")
# ACC cascade: small-nt -> big-nt -> big-think, agreement think-gate, FLOPs at parity
print(f"\n  --- ACC cascade (small-nt -> big-nt -> big-think; agreement think-gate) at parity, exact FLOPs ---")
print(f"  {'pool':<8}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{'guard':>7}")
for lab, names in [("ALL-6", ALL6), ("ALL-5", ALL5)]:
    M = np.isin(ds_of, names); idxs = np.where(M)[0]
    o0, o1, o2, mg, dg, gb, dso = oks[M], okbn[M], okbt[M], R["margin"][M], dis[M], R["gbt"][M], ds_of[M]
    # exact prefill-inclusive FLOPs (assume ~400 prompt tok small+bignt cap320, big-think fullres+gen)
    f0 = 2 * Ns * 400; f1 = 2 * Nb * 400; f2 = 2 * Nb * (700 + gb); F2 = f2.sum() if np.ndim(f2) else f2 * len(o0)
    parity = o2.mean(); accs, e0s, e1s, fls, bads = [], [], [], [], []
    for sd in range(20):
        rg = np.random.default_rng(sd); n = len(o0); cal = np.zeros(n, bool); cal[rg.choice(n, n // 2, replace=False)] = True; te = ~cal
        tgt = o2[cal].mean(); best = None
        for t0 in np.quantile(-mg[cal], np.linspace(0, 1, 22)):
            e0 = -mg[cal] > t0; e1 = e0 & (dg[cal] > 0.5)
            ok = np.where(~e0, o0[cal], np.where(~e1, o1[cal], o2[cal]))
            if ok.mean() >= tgt - 1e-9:
                fl = (f0 * len(o0[cal]) + f1 * e0.sum() + (f2[cal] * e1).sum()) / (f2[cal].sum())
                if best is None or fl < best[0]: best = (fl, t0)
        t0b = best[1] if best else (-mg[cal]).max() + 1
        E0 = -mg[te] > t0b; E1 = E0 & (dg[te] > 0.5)
        ok = np.where(~E0, o0[te], np.where(~E1, o1[te], o2[te]))
        fl = (f0 * len(o0[te]) + f1 * E0.sum() + (f2[te] * E1).sum()) / f2[te].sum()
        bad = sum(1 for d in names if (dso[te] == d).sum() and ok[dso[te] == d].mean() < o0[te][dso[te] == d].mean() - 1e-9)
        accs.append(ok.mean()); e0s.append(E0.mean()); e1s.append(E1.mean()); fls.append(fl); bads.append(bad)
    print(f"  {lab:<8}{np.mean(accs):>7.4f}{np.mean(e0s)*100:>6.0f}%{np.mean(e1s)*100:>6.0f}%{np.mean(fls)*100:>7.1f}%{np.mean(bads):>7.2f}  (parity {parity:.4f}; always-big-think=100% FLOPs)")
# Lingshu resolution sweep (small-nt accuracy vs cap)
if C["sweep"]:
    print(f"\n  --- small-model no-think resolution sweep (accuracy vs cap, ALL-5 perception) ---")
    for cap in ["cap80", "cap160", "cap320", "cap640", "fullres"]:
        sc = loadarm(os.path.join(C["sweep"], cap), C["small"][1])
        accs = [sc[ds][i]["ok"] for ds in ALL5 for i in sc.get(ds, {})]
        if accs: print(f"    {cap:<8} acc={np.mean(accs):.3f}  (n={len(accs)})")
