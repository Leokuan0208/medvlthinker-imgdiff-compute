#!/usr/bin/env python3
"""Publication-ready efficiency metrics for the faithful 2-tier cascade: deferral curve (accuracy vs cost) + APGR
(RouteLLM: area under Performance-Gap-Recovered vs normalized-cost) + CPT (cost at iso-accuracy). Both families,
MedEvalKit closed subsets, margin gate. Cost = prefill-dominated FLOPs (7B=1, 32B=4.57)."""
import json,glob
import numpy as np
ROOT="/home/jamesyang/medvlthinker-imgdiff-compute/MedEvalKit"
def rows(md,ds):
    r=[]
    for f in sorted(glob.glob(f"{ROOT}/{md}/*/{ds}/results.json"))+sorted(glob.glob(f"{ROOT}/{md}/*/{ds}/*/output_sample.json")):
        r.extend(json.load(open(f)))
    return r
def ok(r):
    if "correct" in r: return int(bool(r["correct"]))
    return int(str(r.get("answer","")).strip().lower()==str(r.get("parsed_pred",r.get("response",""))).strip().lower())
R32=4.57
FAM={"Lingshu":("eval_results_lingshu7b_full","eval_results_lingshu32b_full"),
     "MedVLThinker":("eval_results_mvt7b","eval_results_mvt32b")}
BENCH=["PMC_VQA","SLAKE","VQA_RAD","MedXpertQA-MM"]  # exclude MMMU (Lingshu-7B inflated)
for fam,(d7,d32) in FAM.items():
    print(f"\n===== {fam} — deferral curve APGR + CPT (cost@iso-acc), faithful MedEvalKit =====")
    print(f"  {'benchmark':<14}{'7B':>6}{'32B':>6}{'APGR':>7}{'CPT(FLOPs@iso)':>16}{'FLOPs-saved':>12}")
    apgrs=[]
    for ds in BENCH:
        r7=rows(d7,ds); r32=rows(d32,ds); n=min(len(r7),len(r32))
        if n==0: continue
        cm=[True]*n
        if ds=="SLAKE": cm=[r7[i].get("answer_type")=="CLOSED" for i in range(n)]
        if ds=="VQA_RAD": cm=[str(r7[i].get("answer","")).lower() in ("yes","no") for i in range(n)]
        idx=[i for i in range(n) if cm[i]]
        o7=np.array([ok(r7[i]) for i in idx]); o32=np.array([ok(r32[i]) for i in idx])
        m7=np.array([float(r7[i].get("margin") or 0) for i in idx])
        a7,a32=o7.mean(),o32.mean()
        if a32<=a7: print(f"  {ds:<14}{a7:>6.3f}{a32:>6.3f}  (7B>=32B, cascade trivially keeps 7B)"); continue
        # sweep escalation -> (cost, acc)
        curve=[]
        for q in np.linspace(0,1,101):
            esc=m7<np.quantile(m7,q); acc=np.where(esc,o32,o7).mean(); cost=1+esc.mean()*R32
            curve.append((cost,acc))
        curve=sorted(set(curve))
        costs=np.array([c for c,a in curve]); accs=np.array([a for c,a in curve])
        # PGR vs normalized cost in [1,R32]
        pgr=(accs-a7)/(a32-a7); ncost=(costs-1)/(R32-1)
        order=np.argsort(ncost); apgr=np.trapz(pgr[order],ncost[order])
        # CPT: min cost to reach a32
        cpt=min([c for c,a in curve if a>=a32-1e-9], default=R32)
        apgrs.append(apgr)
        print(f"  {ds:<14}{a7:>6.3f}{a32:>6.3f}{apgr:>7.3f}{cpt:>14.2f}  {(1-cpt/R32)*100:>10.0f}%")
    if apgrs: print(f"  mean APGR (higher=better, 1.0=ideal)={np.mean(apgrs):.3f}")
print("\nAPGR = area under Performance-Gap-Recovered vs normalized-cost (RouteLLM metric). CPT = FLOPs to match 32B.")
