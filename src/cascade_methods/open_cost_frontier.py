#!/usr/bin/env python3
"""Cost-optimized verifier-cascade frontier, PER-DATASET + pooled, from saved dumps (no inference).
For each best-of-N (N in 1,2,4,8): verifier picks argmax(scores[:N]); gate=max(scores[:N]); escalate
when gate<tau to STRONG. Cost = prefill-dominated forward-equiv (validated: gen decode ~5 tok << ~320
image-prefill tok, so each gen/verify/strong call ~= one prefill scaled by params).
  cost(verifier-bo-N, esc) = N*cheap_ratio + N*verif_ratio + esc*strong_ratio   [units: 2*7B*P]
  always-STRONG cost = strong_ratio.   ratio = params/7.  verifier base here = Lingshu-7B (ratio 1.0).
Prints, per dataset, the cheapest N that BEATS always-STRONG on accuracy, and whether it also beats on cost.
"""
import argparse, os, json
import numpy as np
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--adapter_dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--strong_dir", required=True); ap.add_argument("--strong_tag", required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--cheap_ratio", type=float, required=True)
ap.add_argument("--strong_ratio", type=float, required=True)
ap.add_argument("--verif_ratio", type=float, default=1.0)
A = ap.parse_args()
def load_judge(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=int(r["judge_ok"])
    return m
def frontier_best(v,s,gate):
    v=np.array(v,float); s=np.array(s,float); gate=np.array(gate,float)
    cand=sorted(set(gate.tolist())); best=(-1,0.0)
    for tau in [cand[0]-1e-9]+cand+[cand[-1]+1e-9]:
        esc=gate<tau; acc=np.where(esc,s,v).mean()
        if acc>best[0]: best=(acc,esc.mean())
    return best
cr,vr,sr=A.cheap_ratio,A.verif_ratio,A.strong_ratio
def analyze(recs, name):
    if not recs: print(f"  {name}: empty"); return
    svec=[r[2] for r in recs]; strong_acc=np.mean(svec); n=len(recs)
    print(f"\n  [{name}] n={n}  always-STRONG acc={strong_acc:.3f} cost={sr:.2f}")
    cheapest_winner=None
    for N in (1,2,4,8):
        vN=[r[0][int(np.argmax(r[1][:N]))] for r in recs]
        gate=[max(r[1][:N]) for r in recs]
        acc_v=np.mean(vN); cost_v=N*cr+N*vr
        acc_b,esc_b=frontier_best(vN,svec,gate); cost_c=N*cr+N*vr+esc_b*sr
        flag_v = "BEATS-acc" + ("+cost" if (acc_v>strong_acc+1e-9 and cost_v<sr) else ("(costlier)" if acc_v>strong_acc+1e-9 else "")) if acc_v>strong_acc+1e-9 else ""
        print(f"    N={N}: verif-boN={acc_v:.3f}(cost {cost_v:.2f}) {flag_v:<16} | +gate-best={acc_b:.3f}@esc{esc_b*100:.0f}%(cost {cost_c:.2f})")
        if cheapest_winner is None and acc_v>strong_acc+1e-9 and cost_v<sr:
            cheapest_winner=(N,acc_v,cost_v)
    if cheapest_winner:
        N,a,c=cheapest_winner
        print(f"    => CHEAPEST no-gate win: N={N} acc={a:.3f}>{strong_acc:.3f} at cost {c:.2f}<{sr:.2f} (BEATS strong on BOTH acc+FLOPs)")
    else:
        print(f"    => no no-gate setting beats strong on BOTH axes; best is via gate (accuracy win, cost>strong)")
pool=[]
for ds in A.datasets:
    dp=os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dp): print(f"  {ds}: MISSING dump"); continue
    sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=load_judge(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    recs=[]
    for r in json.load(open(dp)):
        if r["idx"] not in sj: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]
        recs.append((sl, r["scores"], sj[r["idx"]]))
    analyze(recs, ds); pool+=recs
analyze(pool, "POOLED")
