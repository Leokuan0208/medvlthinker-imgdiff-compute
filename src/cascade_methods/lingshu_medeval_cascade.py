#!/usr/bin/env python3
"""CLEAN 2-tier cascade + efficiency on faithful MedEvalKit Lingshu outputs. Uses per-sample 'correct' field;
closed subset for SLAKE/VQA_RAD; prefill-dominated FLOPs (params-ratio x escalation, MCQ decode~0) + measured
latency_s. Reports baselines + 2-tier (7B->32B, margin gate) at iso-32B-nt accuracy: FLOPs & latency saved."""
import json,glob,re,string
import numpy as np
ROOT="/home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit"
def loadrows(model_dir,ds):
    rows=[]
    for f in sorted(glob.glob(f"{ROOT}/{model_dir}/*/{ds}/results.json"))+sorted(glob.glob(f"{ROOT}/{model_dir}/*/{ds}/*/output_sample.json"))+sorted(glob.glob(f"{ROOT}/{model_dir}/*/{ds}/output_sample.json")):
        d=json.load(open(f)); rows.extend(d if isinstance(d,list) else list(d.values()))
    return rows
def ok(r):
    if "correct" in r: return int(bool(r["correct"]))
    g,p=r.get("answer"),r.get("parsed_pred",r.get("response"))
    return int(str(g).strip().lower()==str(p).strip().lower())
def closed_mask(ds,rows):
    if ds=="SLAKE": return [ (r.get("answer_type","")=="CLOSED") for r in rows]
    if ds=="VQA_RAD": return [ str(r.get("answer","")).strip().lower() in ("yes","no") for r in rows]
    return [True]*len(rows)
R7=1.0; R32=4.57
print("CLEAN 2-tier cascade (faithful MedEvalKit Lingshu). FLOPs=prefill-dominated (7B=1,32B=4.57); latency=measured.")
print(f"  {'benchmark':<14}{'7B':>7}{'32B':>7}{'2tier@iso':>10}{'esc%':>6}{'FLOPs':>7}{'FLOPsvs32B':>11}{'lat':>7}{'latvs32B':>9}")
for ds in ["MMMU-Medical-val","MedXpertQA-MM","PMC_VQA","SLAKE","VQA_RAD"]:
    r7=loadrows("eval_results_lingshu7b_full",ds); r32=loadrows("eval_results_lingshu32b_full",ds)
    n=min(len(r7),len(r32))
    if n==0: print(f"  {ds:<14} no data"); continue
    cm=closed_mask(ds,r7[:n]); idx=[i for i in range(n) if cm[i]]
    o7=np.array([ok(r7[i]) for i in idx]); o32=np.array([ok(r32[i]) for i in idx])
    m7=np.array([float(r7[i].get("margin") or 0) for i in idx])
    l7=np.array([float(r7[i].get("latency_s") or 0) for i in idx]); l32=np.array([float(r32[i].get("latency_s") or 0) for i in idx])
    a7,a32=o7.mean(),o32.mean(); nC=len(idx)
    # 2-tier: escalate lowest-margin 7B -> 32B; min-cost config matching always-32B-nt acc
    best=None
    for q in np.linspace(0,1,101):
        th=np.quantile(m7,q); esc=m7<th
        acc=np.where(esc,o32,o7).mean()
        flops=R7 + esc.mean()*R32; lat=l7.mean()+ (l32*esc).mean()
        if acc>=a32-1e-9 and (best is None or flops<best[2]):
            best=(acc,esc.mean(),flops,lat)
    always32_fl=R32; always32_lat=l32.mean()
    if best:
        acc,esc,fl,lat=best
        print(f"  {ds:<14}{a7:>7.3f}{a32:>7.3f}{acc:>10.3f}{esc*100:>5.0f}%{fl:>7.2f}{f'-{(1-fl/always32_fl)*100:.0f}%':>11}{lat:>6.3f}s{f'-{(1-lat/always32_lat)*100:.0f}%':>9}  (n_closed={nC})")
    else:
        print(f"  {ds:<14}{a7:>7.3f}{a32:>7.3f}  cannot match 32B (n_closed={nC})")
print("\nNOTE: MMMU 7B inflated (protocol) -> its cascade keeps 7B; treat MMMU cascade as unreliable.")
print("2-tier margin gate; think-tier(3-tier) pending a working CoT prompt for Lingshu.")
