#!/usr/bin/env python3
"""
cascade_check2.py - corrected cascade analysis: CHEAP 7B (nothink_norag) vs 32B,
the realistic two-model pairing (not 7B-best-of-4, which is unreachable).

The honest cascade framing is EFFICIENCY: a perfect router over {cheap-7B, 32B}
has oracle accuracy = (a7c | a32). Key quantities:
  - only-7Bcheap: Qs the CHEAP 7B gets that 32B misses (the rescuable pool)
  - oracle vs 32B-alone: headroom a router could add over just using 32B
  - cost framing: every Q routed to 7B instead of 32B saves the 32B's compute
We also report a CONFIDENCE-GATED cascade estimate: if the 7B's confidence
(opt_logprobs margin) predicts when it's right, you escalate only the rest.
"""
import json, glob, os, re, math, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
np.random.seed(42)

DATASETS = ["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"]

# 7B cheap arm: need ok AND opt_logprobs (for confidence gating)
pat7 = re.compile(r"ckpt_(.+?)_nothink_norag(?:_s\dof\d)?\.jsonl$")
r7 = defaultdict(dict)
for f in glob.glob("archive/single-model-routing/gate_7b_rag_axes/*nothink_norag*.jsonl"):
    m=pat7.search(os.path.basename(f))
    if not m: continue
    for l in open(f):
        if l.strip():
            d=json.loads(l); r7[m.group(1)][d["idx"]]=d
pat32 = re.compile(r"ckpt_(.+?)_think_norag(?:_s\dof\d)?\.jsonl$")
r32 = defaultdict(dict)
for f in glob.glob("ckpts/gate_32b/*think_norag*.jsonl"):
    m=pat32.search(os.path.basename(f))
    if not m: continue
    for l in open(f):
        if l.strip():
            d=json.loads(l); r32[m.group(1)][d["idx"]]=d

def margin(row):
    lp=row.get("opt_logprobs") or {}; v=sorted(lp.values(),reverse=True)
    return (v[0]-v[1]) if len(v)>=2 else 0.0

print("="*72); print("CORRECTED CASCADE: cheap-7B (nothink_norag) vs 32B"); print("="*72)
POOL=defaultdict(list)
for name in DATASETS:
    idx=sorted(set(r7[name])&set(r32[name]))
    if not idx: print(f"{name}: no overlap"); continue
    a7=np.array([r7[name][i]["ok"] for i in idx])
    a32=np.array([r32[name][i]["ok"] for i in idx])
    mg=np.array([margin(r7[name][i]) for i in idx])
    oracle=(a7|a32).mean()
    only7=((a7==1)&(a32==0)).mean(); only32=((a7==0)&(a32==1)).mean()
    both=((a7==1)&(a32==1)).mean(); neither=((a7==0)&(a32==0)).mean()
    POOL["a7"]+=a7.tolist(); POOL["a32"]+=a32.tolist(); POOL["mg"]+=mg.tolist()

    print(f"\n### {name}  (n={len(idx)})")
    print(f"     cheap-7B = {a7.mean():.3f}    32B = {a32.mean():.3f}")
    print(f"     cells: both={both:.3f}  only-7B={only7:.3f}  only-32B={only32:.3f}  neither={neither:.3f}")
    print(f"     perfect-router oracle{{7B,32B}} = {oracle:.3f}  (gain over 32B = {oracle-a32.mean():+.3f})")
    print(f"     cheap-7B rescues {only7*len(idx):.0f} Qs the 32B missed")

    # confidence-gated cascade: trust 7B when its margin is high, else escalate to 32B.
    # CV the threshold: route-to-32B when 7B margin < tau; pick tau maximizing acc on train.
    strat=(a7|a32)
    accs=[]; routed_to_32_frac=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(mg.reshape(-1,1),strat):
        best_tau,best_acc=None,-1
        for tau in np.quantile(mg[tr], np.linspace(0,1,21)):
            keep7 = mg[tr]>=tau
            acc = (np.where(keep7, a7[tr], a32[tr])).mean()
            if acc>best_acc: best_acc,best_tau=acc,tau
        keep7_te = mg[te]>=best_tau
        acc_te = (np.where(keep7_te, a7[te], a32[te])).mean()
        accs.append(acc_te); routed_to_32_frac.append((~keep7_te).mean())
    print(f"     confidence-gated cascade acc = {np.mean(accs):.3f} +/- {np.std(accs):.3f}"
          f"  (escalates {np.mean(routed_to_32_frac):.0%} to 32B)")
    print(f"        vs always-32B {a32.mean():.3f}  |  vs always-cheap-7B {a7.mean():.3f}")

a7=np.array(POOL["a7"]); a32=np.array(POOL["a32"]); mg=np.array(POOL["mg"])
oracle=(a7|a32).mean()
print("\n"+"="*72+"\nPOOLED\n"+"="*72)
print(f"     cheap-7B={a7.mean():.3f}  32B={a32.mean():.3f}  oracle={oracle:.3f} (gain over 32B {oracle-a32.mean():+.3f})")
print(f"     only-7B={((a7==1)&(a32==0)).mean():.3f}  only-32B={((a7==0)&(a32==1)).mean():.3f}")
strat=(a7|a32); accs=[]; esc=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=0).split(mg.reshape(-1,1),strat):
    bt,ba=None,-1
    for tau in np.quantile(mg[tr],np.linspace(0,1,21)):
        a=(np.where(mg[tr]>=tau,a7[tr],a32[tr])).mean()
        if a>ba: ba,bt=a,tau
    k=mg[te]>=bt; accs.append((np.where(k,a7[te],a32[te])).mean()); esc.append((~k).mean())
print(f"     confidence-gated cascade = {np.mean(accs):.3f} +/- {np.std(accs):.3f}  (escalates {np.mean(esc):.0%} to 32B)")
print("\nREAD: oracle gain over 32B = ceiling. confidence-gated acc vs always-32B")
print("= what a REAL cheap-margin cascade achieves. If gated >= 32B while escalating")
print("<100%, you beat/match 32B at lower cost = a method. If gated < 32B, the margin")
print("signal is too weak (the same predictability wall) and 'use 32B' wins.")
