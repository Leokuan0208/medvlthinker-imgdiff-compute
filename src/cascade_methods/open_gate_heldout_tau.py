#!/usr/bin/env python3
"""Honest (deployable) value of the verifier-confidence GATE vs plain best-of-N selection.
The cost-frontier '+gate-best' uses oracle-tau (swept on the eval set). Here: 5-fold cross-fit -- pick
tau* on 4/5 (maximize cascade acc), evaluate on the held-out 1/5; average folds. Compares, at N=8:
  no-gate bo8 (tau-free) | held-out-tau gate | oracle-tau gate.  If held-out ~= no-gate, the gate adds
nothing deployable and best-of-N selection is the real lever.
"""
import argparse, os, json
import numpy as np
ROOT=os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap=argparse.ArgumentParser()
ap.add_argument("--adapter_dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--strong_dir", required=True); ap.add_argument("--strong_tag", required=True)
ap.add_argument("--datasets", nargs="+", required=True); ap.add_argument("--N", type=int, default=8)
A=ap.parse_args()
def load_judge(p):
    m={}
    if os.path.exists(p):
        for l in open(p):
            if l.strip(): r=json.loads(l); m[r["idx"]]=int(r["judge_ok"])
    return m
def best_tau(v,s,gate):
    cand=sorted(set(gate)); best=(-1,None)
    for tau in [cand[0]-1e-9]+cand+[cand[-1]+1e-9]:
        esc=np.array(gate)<tau; acc=np.where(esc,s,v).mean()
        if acc>best[0]: best=(acc,tau)
    return best[1]
def apply_tau(v,s,gate,tau):
    esc=np.array(gate)<tau; return np.where(esc,s,v).mean(), esc.mean()
# gather pooled recs
V=[]; S=[]; G=[]
for ds in A.datasets:
    dp=os.path.join(ROOT,A.adapter_dir,f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dp): continue
    sj={}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj=load_judge(os.path.join(ROOT,A.strong_dir,cand))
        if sj: break
    for r in json.load(open(dp)):
        if r["idx"] not in sj: continue
        sl=[0 if x is None or x==-1 else int(x) for x in r["sl"]]
        N=A.N; V.append(sl[int(np.argmax(r["scores"][:N]))]); G.append(max(r["scores"][:N])); S.append(sj[r["idx"]])
V=np.array(V,float); S=np.array(S,float); G=np.array(G,float); n=len(V)
strong=S.mean(); nogate=V.mean()
# oracle-tau
ot=best_tau(V,S,list(G)); oacc,oesc=apply_tau(V,S,G,ot)
# 5-fold cross-fit held-out-tau  (deterministic fold by index mod 5; no RNG)
ho_acc=[]; ho_esc=[]
for f in range(5):
    te=np.array([i%5==f for i in range(n)]); tr=~te
    tau=best_tau(V[tr],S[tr],list(G[tr]))
    a,e=apply_tau(V[te],S[te],G[te],tau); ho_acc.append(a); ho_esc.append(e)
print(f"  N={A.N} pooled n={n}  STRONG={strong:.3f}")
print(f"    no-gate bo{A.N} (tau-free)   acc={nogate:.3f}")
print(f"    held-out-tau gate (5-fold)  acc={np.mean(ho_acc):.3f}  esc={np.mean(ho_esc)*100:.0f}%   <-- DEPLOYABLE")
print(f"    oracle-tau gate (optimistic)acc={oacc:.3f}  esc={oesc*100:.0f}%")
print(f"    => gate deployable gain over no-gate: {np.mean(ho_acc)-nogate:+.3f} ;  oracle optimism: {oacc-np.mean(ho_acc):+.3f}")
