#!/usr/bin/env python3
"""
recompute_energy2.py  --  CORRECTED per-dataset time/energy saved from the live cascade run.

FIX: gpu7_energy_j / gpu32_energy_j are each card's energy over the ENTIRE query window,
NOT per-leg active energy. (Proof: gpu7_energy_j 1864 J / latency_s 23.8 s = 78 W = the idle
GPU0 power while the 32B ran for 24 s; the 7B's real work was ~0.2 s.) So:
  ENERGY[active] (marginal, idle excluded; the fair analog of FLOPs):
     7B active energy from NON-escalated queries (window = the 7B leg there);
     32B active energy from gpu32_energy_j on escalated queries.
       base = mean(gpu32_energy_j | escalated)
       casc = mean(gpu7_energy_j | NON-escalated) + esc_rate * mean(gpu32_energy_j | escalated)
  ENERGY[node] (total board, both models on the same 2-GPU box, symmetric):
       base = mean(energy_j | escalated)
       casc = mean(energy_j | all)
Both should sit within ~0.5 pp of time saved. saved = 1 - cascade / always-32B.
Baseline = measured cost of an escalated (harder) query applied to all -> mild upper bound on
easy datasets (same caveat the deck already states).
"""
import argparse, json, os, sys
from collections import defaultdict

def load(p):
    out=[]
    for l in open(p):
        l=l.strip()
        if l: out.append(json.loads(l))
    return out
def mean(xs):
    xs=[x for x in xs if x is not None]
    return sum(xs)/len(xs) if xs else float("nan")
def flm(p,g): return str(p).strip().upper()[:1]==str(g).strip().upper()[:1]
def norm(s): return str(s).lower().replace("-","").replace("_","").replace(" ","")
MEDX={norm(x) for x in ["MedXpert","MedXpert-Reasoning","MedXpert-Understanding","MedXpertQA","MedX-M"]}
FOUR={norm(x) for x in ["PMC-VQA","SLAKE","VQA-RAD","PathVQA"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--jsonl",default=os.path.expanduser("~/medvlthinker-imgdiff-compute/rt_cascade_cap320.jsonl"))
    a=ap.parse_args()
    if not os.path.exists(a.jsonl): sys.exit(f"!! not found: {a.jsonl}")
    rows=load(a.jsonl)
    if not rows: sys.exit("!! empty")
    def pick(o): return next((k for k in o if k in rows[0]),None)
    F_ds=pick(["dataset","benchmark"]); F_esc=pick(["escalate","escalated"])
    F_ok=pick(["ok","correct"]); F_p7=pick(["pred7","pred_7b"]); F_gold=pick(["gold","answer","label"])
    F_lt=pick(["latency_s","lat_s"]); F_l32=pick(["lat32_s","lat_32b_s"])
    F_e7=pick(["gpu7_energy_j","gpu0_energy_j"]); F_e32=pick(["gpu32_energy_j","gpu1_energy_j"])
    F_etot=pick(["energy_j","total_energy_j"])
    F_p7w=pick(["gpu7_power_w","gpu0_power_w"]); F_p32w=pick(["gpu32_power_w","gpu1_power_w"])
    req=dict(dataset=F_ds,escalate=F_esc,lat32=F_l32,e32=F_e32,etot=F_etot,lat_tot=F_lt)
    if any(v is None for v in req.values()):
        print("!! missing:",[k for k,v in req.items() if v is None]); print("keys:",sorted(rows[0])); sys.exit(1)
    by=defaultdict(list)
    for r in rows: by[r[F_ds]].append(r)

    def block(name,R):
        n=len(R); esc=[r for r in R if r.get(F_esc)]; non=[r for r in R if not r.get(F_esc)]; er=len(esc)/n
        casc_acc=mean([float(r[F_ok]) for r in R]) if F_ok else float("nan")
        a7=mean([1.0 if flm(r[F_p7],r[F_gold]) else 0.0 for r in R]) if (F_p7 and F_gold) else float("nan")
        P7a=mean([r[F_p7w] for r in non]) if (F_p7w and non) else float("nan")
        P32a=mean([r[F_p32w] for r in esc]) if (F_p32w and esc) else float("nan")
        P1i=mean([r[F_p32w] for r in non]) if (F_p32w and non) else float("nan")
        base_lat=mean([r[F_l32] for r in esc]); casc_lat=mean([r[F_lt] for r in R])
        ts=1-casc_lat/base_lat if base_lat else float("nan")
        e32a=mean([r[F_e32] for r in esc])
        if non:
            e7a=mean([r[F_e7] for r in non]); ea=1-(e7a+er*e32a)/e32a if e32a else float("nan")
        else: ea=float("nan")
        be=mean([r[F_etot] for r in esc]); ce=mean([r[F_etot] for r in R])
        en=1-ce/be if be else float("nan")
        def pc(x): return f"{x*100:5.1f}%" if x==x else "   -- "
        def w(x): return f"{x:5.0f}W" if x==x else "  -- "
        print(f"--- {name}  (n={n}) ---")
        print(f"    escalation {er*100:5.1f}%   cascade acc {casc_acc:.4f}   always-7B acc {a7:.4f}")
        print(f"    real power  7B-active {w(P7a)}  32B-active {w(P32a)}  idle~{w(P1i)}")
        print(f"    SAVED vs always-32B :  time {pc(ts)}   energy[active] {pc(ea)}   energy[node] {pc(en)}")
        ra=(1-ea) if ea==ea else float("nan")
        print(f"    energy ratio (cascade/always-32B):  active {ra:.3f}x   node {1-en:.3f}x\n")
        return dict(time=ts,ea=ea,en=en)

    print("#"*70+"\n# PER-DATASET\n"+"#"*70+"\n")
    for ds in sorted(by): block(ds,by[ds])
    print("#"*70+"\n# GROUPED\n"+"#"*70+"\n")
    g=block("OVERALL — ALL 6  (page-1)",rows)
    block("EXCL MedXpert (5)",[r for r in rows if norm(r[F_ds]) not in MEDX])
    block("FOUR COMPETENT (4)",[r for r in rows if norm(r[F_ds]) in FOUR])
    print("="*70)
    print("PAGE-1 (six-benchmark) — copy these:")
    print(f"   time saved   : {g['time']*100:.1f}%")
    print(f"   energy saved : active {g['ea']*100:.1f}%  |  node {g['en']*100:.1f}%")
    print(f"   energy ratio : active {1-g['ea']:.3f}x  |  node {1-g['en']:.3f}x")
    print(f"   FLOPs stays 74% / 26% saved (cap320 grid, already six-benchmark)")
    print("="*70)

if __name__=="__main__": main()
