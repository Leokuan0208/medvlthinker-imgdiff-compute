#!/usr/bin/env python3
"""
medxpert_impact.py — what does including the near-chance MedXpert-MM benchmarks do to the
cascade's saving? FULL-RES ONLY (the sweep didn't run MedXpert at the caps), no GPU.

Shows with existing data:
  (1) per-MedXpert-set 7B/32B/oracle accuracy -> are both models near chance? could ANY router help?
  (2) held-out-tau parity cost pooled over 4 competent vs all 6 -> the exact dilution.
tau set on held-out full-res pmctrain margins, applied blind to eval (same protocol as the frontier).
"""
import json, glob, os, re, argparse
import numpy as np
from collections import defaultdict

np.random.seed(0)
COMPETENT = ["PMC-VQA","SLAKE","VQA-RAD","PathVQA"]
MEDX      = ["MedXpert-Reasoning","MedXpert-Understanding"]
P7,P32,TXT,VIS32 = 7.0e9,32.0e9,64,640
N_BOOT=1000; TGRID=np.linspace(0,1,41)
EVAL_7B="ckpts/gate_7b_vllm"; DIR_32B="ckpts/gate_32b"; PMTRAIN="ckpts/gate_7b_pmctrain"

def load_arm(ckdir, cell):
    pat=re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$")
    d=defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir,f"*{cell}*.jsonl")):
        m=pat.search(os.path.basename(f))
        if not m: continue
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except Exception: continue
            if "idx" in r: d[m.group(1)][r["idx"]]=r
    return d

def marg(row):
    lp=row.get("opt_logprobs") or {}; v=sorted(lp.values(),reverse=True)
    return (v[0]-v[1]) if len(v)>=2 else 0.0

def pm_margins(dirpath):
    out=[]
    for f in glob.glob(os.path.join(dirpath,"*nothink*.jsonl")):
        for line in open(f):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except Exception: continue
            out.append(marg(r))
    return np.array(out,float)

def mean_gen(arm):
    vals=[r.get("gen_tokens",0) or 0 for ds in arm.values() for r in ds.values()]
    return float(np.mean(vals)) if vals else 0.0

def joined(a7,r32,ds):
    a,b=a7.get(ds,{}),r32.get(ds,{}); idx=sorted(set(a)&set(b))
    return (np.array([a[i]["ok"] for i in idx],float),
            np.array([b[i]["ok"] for i in idx],float),
            np.array([marg(a[i]) for i in idx],float))

def boot_std(ok):
    if len(ok)==0: return 0.0
    n=len(ok); return float(ok[np.random.randint(0,n,(N_BOOT,n))].mean(1).std())

def parity_cost(pool_ds,a7,r32,pm,offset):
    cols=[joined(a7,r32,d) for d in pool_ds]
    ok7 =np.concatenate([c[0] for c in cols])
    ok32=np.concatenate([c[1] for c in cols])
    mg  =np.concatenate([c[2] for c in cols])
    acc32=ok32.mean(); tol=boot_std(ok32)
    for t in TGRID:
        tau=np.quantile(pm,t); esc=mg<tau
        acc=np.where(esc,ok32,ok7).mean()
        if acc>=acc32-tol:
            return dict(acc32=acc32,tol=tol,esc=esc.mean(),cost=offset+esc.mean(),acc=acc)
    return dict(acc32=acc32,tol=tol,esc=1.0,cost=offset+1.0,acc=ok32.mean())

def run():
    ap=argparse.ArgumentParser(); ap.add_argument("--repo",default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A=ap.parse_args(); repo=A.repo
    a7=load_arm(os.path.join(repo,EVAL_7B),"nothink_norag")
    r32=load_arm(os.path.join(repo,DIR_32B),"think_norag")
    pm=pm_margins(os.path.join(repo,PMTRAIN))
    g32=mean_gen(r32); cost32=2*P32*(VIS32+TXT)+2*P32*g32
    offset=(2*P7*(VIS32+TXT))/cost32

    print("="*72)
    print("MedXpert at full-res: are both models near chance? can routing help?")
    print(f"{'dataset':<24}{'n':>6}{'7B':>8}{'32B':>8}{'oracle':>8}{'32B-7B':>8}")
    print("-"*62)
    for ds in MEDX+COMPETENT:
        o7,o32,_=joined(a7,r32,ds)
        if len(o7)==0:
            print(f"{ds:<24}{0:>6}   (no data)"); continue
        orac=np.maximum(o7,o32).mean()
        print(f"{ds:<24}{len(o7):>6}{o7.mean():>8.3f}{o32.mean():>8.3f}{orac:>8.3f}{o32.mean()-o7.mean():>+8.3f}")
    print("  (oracle = best-of-2 per question; large oracle but ~0 realizable = no usable routing signal)")

    print("\n"+"="*72)
    print("Held-out-tau parity cost (full-res): competent-only vs all-6")
    if pm.size==0:
        print(f"  no pmctrain margins at {PMTRAIN}"); return
    comp=parity_cost(COMPETENT,a7,r32,pm,offset)
    all6=parity_cost(COMPETENT+MEDX,a7,r32,pm,offset)
    print(f"  7B overhead (full-res) = {100*offset:.1f}% of always-32B\n")
    for name,res in [("4 competent",comp),("all 6 (+MedXpert)",all6)]:
        print(f"  {name:<20} acc={res['acc']:.3f} (32B {res['acc32']:.3f}±{res['tol']:.3f})  "
              f"escalation={100*res['esc']:.1f}%  ->  COST={100*res['cost']:.1f}% of always-32B")
    print(f"\n  MedXpert moves the saving from {100*comp['cost']:.1f}% to {100*all6['cost']:.1f}% "
          f"(+{100*(all6['cost']-comp['cost']):.1f} cost pts) for ~no accuracy gain on that slice.")

if __name__=="__main__":
    run()
