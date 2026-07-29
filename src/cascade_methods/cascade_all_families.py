#!/usr/bin/env python3
"""Faithful MedEvalKit 2-tier cascade (7B->32B/38B, margin gate) across ALL 6 benchmarks incl PATH_VQA, any family.
Usage: cascade_all_families.py <label> <cheap_dir> <strong_dir>   (dirs under MedEvalKit/)

NOTE on latency: the `lat` column is each sample's `latency_s` = per-sample latency amortized over identical
vLLM batched serving. The batch size is a fixed constant (batch_size=2000, model-independent), so cheap and
strong of the SAME benchmark amortize over the SAME N -> a FAIR cheap-vs-strong RELATIVE comparison. (Not
batch-1, and not a GPU-hours number: tp=2 on the strong leg is a real deployment-config difference whose
direction is conservative — it speeds up the strong leg and thus SHRINKS the reported savings. Batch-1 proper
is in src/cascade/measure_config.py.)
CAVEAT (latency only): a latency cell is clean only where BOTH legs were run serialized at matched tp/gpu_mem.
This holds for Lingshu & MedVLThinker on MMMU/MedXpert/PMC/SLAKE/VQA-RAD. It does NOT hold for (a) the InternVL3
family (cheap tp=1 vs strong tp=2, and the cheap leg was measured under GPU contention) or (b) any PATH_VQA row
(regenerated under contention) -> do not cite those latency cells until the leg is re-run matched. Accuracy,
esc%, and FLOPs for those rows are unaffected.
Cheap/strong rows are joined by position (same seed + dataset order); a count mismatch is warned about."""
import json,glob,sys
import numpy as np
ROOT="/home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit"
label,cheap_dir,strong_dir=sys.argv[1],sys.argv[2],sys.argv[3]
R32=4.57
def rows(md,ds):
    r=[]
    for f in sorted(glob.glob(f"{ROOT}/{md}/*/{ds}/results.json"))+sorted(glob.glob(f"{ROOT}/{md}/*/{ds}/*/output_sample.json"))+sorted(glob.glob(f"{ROOT}/{md}/*/{ds}/output_sample.json")):
        r.extend(json.load(open(f)))
    return r
def ok(r):
    if "correct" in r: return int(bool(r["correct"]))
    return int(str(r.get("answer","")).strip().lower()==str(r.get("parsed_pred",r.get("response",""))).strip().lower())
def closed(ds,rs):
    if ds=="SLAKE": return [x.get("answer_type","")=="CLOSED" for x in rs]
    if ds in ("VQA_RAD","PATH_VQA"): return [str(x.get("answer","")).strip().lower() in ("yes","no") for x in rs]
    return [True]*len(rs)
print(f"===== {label} — faithful 2-tier cascade (margin gate) @ iso-32B, closed subsets =====")
print(f"  {'benchmark':<16}{'7B':>7}{'32B':>7}{'2tier':>8}{'esc%':>6}{'FLOPs':>7}{'vs32B':>7}{'lat vs32B':>10}")
print("  (lat = amortized per-sample latency under identical batched serving -> a FAIR cheap-vs-strong relative number")
print("   (not batch-1; see measure_config.py). NOT clean for the InternVL3 family or any PATH_VQA row -> GPU-contended, do not cite.)")
for ds in ["MMMU-Medical-val","MedXpertQA-MM","PMC_VQA","SLAKE","VQA_RAD","PATH_VQA"]:
    r7=rows(cheap_dir,ds); r32=rows(strong_dir,ds); n=min(len(r7),len(r32))
    if n==0: print(f"  {ds:<16} (no data)"); continue
    if len(r7)!=len(r32): print(f"  # WARNING {ds}: cheap/strong row counts differ ({len(r7)} vs {len(r32)}); positional join may mispair — treat with caution")
    cm=closed(ds,r7[:n]); idx=[i for i in range(n) if cm[i]]
    if not idx: print(f"  {ds:<16} (no closed samples; n={n})"); continue
    o7=np.array([ok(r7[i]) for i in idx]); o32=np.array([ok(r32[i]) for i in idx])
    m7=np.array([float(r7[i].get("margin") or 0) for i in idx])
    if not np.any(m7): print(f"  # WARNING {ds}: cheap-leg margins all zero/missing -> gate signal absent, escalation disabled below")
    l7=np.array([float(r7[i].get("latency_s") or 0) for i in idx]); l32=np.array([float(r32[i].get("latency_s") or 0) for i in idx])
    a7,a32=o7.mean(),o32.mean()
    best=None
    for q in np.linspace(0,1,101):
        esc=m7<np.quantile(m7,q); acc=np.where(esc,o32,o7).mean(); fl=1+esc.mean()*R32; lat=l7.mean()+(l32*esc).mean()
        if acc>=a32-1e-9 and (best is None or fl<best[2]): best=(acc,esc.mean(),fl,lat)
    if best and a32>a7:
        acc,esc,fl,lat=best; lv=(1-lat/l32.mean())*100 if l32.mean()>0 else 0
        print(f"  {ds:<16}{a7:>7.3f}{a32:>7.3f}{acc:>8.3f}{esc*100:>5.0f}%{fl:>7.2f}{f'-{(1-fl/R32)*100:.0f}%':>7}{f'{lv:+.0f}%':>10}  (n={len(idx)})")
    else:
        print(f"  {ds:<16}{a7:>7.3f}{a32:>7.3f}  (7B>=32B or no match -> keep 7B; n={len(idx)})")
