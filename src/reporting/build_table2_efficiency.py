#!/usr/bin/env python3
"""
extract_efficiency.py -- builds Table 2 (efficiency, four competent benchmarks).
Reports escalation %, latency mean/p50/p95, energy/query, energy/correct, and ESTIMATES the
compute fraction vs always-32B from per-leg energies the co-resident run already logged:
  E7 = mean(energy_j|not esc); E32 = mean(energy_j|esc) - E7; phi = mean(energy_j)/E32 = r + E7/E32.
CAVEAT: energy_j may include idle co-resident draw -> phi is an upper bound (use gap G2 for exact).
Peak VRAM is constant-resident: read it from the run's startup 'resident:' log line.
"""
import argparse, json, re
import numpy as np
FOUR = {"pmcvqa","slake","vqarad","pathvqa"}
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def pct(a,q): return float(np.percentile(a,q)) if len(a) else float("nan")
def report(rows, title):
    esc=np.array([bool(r["escalate"]) for r in rows]); ok=np.array([bool(r["ok"]) for r in rows])
    lat=np.array([float(r["latency_s"]) for r in rows]); en=np.array([float(r["energy_j"]) for r in rows]); n=len(rows)
    E7  = en[~esc].mean() if (~esc).any() else float("nan")
    E32 = (en[esc].mean()-E7) if esc.any() else float("nan")
    casc_e=en.mean(); phi=casc_e/E32 if E32==E32 and E32>0 else float("nan")
    phi_7b=E7/E32 if E32==E32 and E32>0 else float("nan"); r=esc.mean()
    print(f"\n==== {title}  (n={n}) ====")
    print(f"  escalation rate          : {100*r:6.2f} %")
    print(f"  cascade accuracy         : {ok.mean():.4f}")
    print(f"  latency mean / p50 / p95 : {lat.mean():6.2f} / {pct(lat,50):6.2f} / {pct(lat,95):6.2f}  s")
    print(f"  energy / query           : {casc_e:8.0f} J")
    print(f"  energy / correct         : {en.sum()/max(ok.sum(),1):8.0f} J")
    print(f"  E7 (cheap) / E32 (margl) : {E7:8.0f} / {E32:8.0f} J")
    print(f"  Rel. compute  always-7B  : {phi_7b:5.3f} x")
    print(f"  Rel. compute  cascade    : {phi:5.3f} x   <- headline (r + E7/E32)")
    print(f"  Rel. compute  always-32B : 1.000 x")
    print(f"  NOTE: phi is an upper-bound estimate; VRAM not in this file (read startup log).")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cascade", default="rt_cascade_cap320.jsonl")
    ap.add_argument("--all", action="store_true"); A=ap.parse_args()
    rows=[json.loads(l) for l in open(A.cascade) if l.strip()]
    report([r for r in rows if norm(r["dataset"]) in FOUR], "FOUR competent benchmarks")
    if A.all: report(rows, "ALL six (pooled; includes excluded MMMU/MedXpert)")
if __name__ == "__main__": main()
