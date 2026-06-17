#!/usr/bin/env python3
"""
rt_report.py - reconstruct the deployed cascade's real-time cost from the two measured legs.
Joins rt_7b.jsonl + rt_32b.jsonl with the FROZEN gate's escalation decision (margin from
gate_7b_vllm + router_margin.pkl), per query:
  cascade_latency = lat_7B + [escalate] lat_32B   (sequential: 32B can't start until 7B's
                                                    margin says escalate)
Reports always-7B / always-32B / cascade: accuracy, mean/p50/p95 latency, mean power,
energy/query, peak VRAM (cascade = both models co-resident), escalation%. CPU only.
"""
import argparse, json, glob, os, re, pickle
import numpy as np
from collections import defaultdict

def load_jsonl_byidx(path):
    d={}
    for l in open(path):
        if l.strip(): r=json.loads(l); d[r["idx"]]=r
    return d
def load_arm(ckdir, cell):
    pat=re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$"); d=defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir,f"*{cell}*.jsonl")):
        m=pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                try: r=json.loads(l); d[m.group(1)][r["idx"]]=r
                except Exception: pass
    return d
def margin(row):
    lp=row.get("opt_logprobs") or {}; v=sorted(lp.values(),reverse=True)
    return (v[0]-v[1]) if len(v)>=2 else 0.0
def pct(a,q): return float(np.percentile(a,q))

def run(repo, rt7_path, rt32_path):
    J=lambda p: os.path.join(repo,p)
    rt7=load_jsonl_byidx(rt7_path); rt32=load_jsonl_byidx(rt32_path)
    R=pickle.load(open(J("ckpts/router_margin.pkl"),"rb")); gate,tau=R["gate"],R["tau"]
    ev7=load_arm(J("ckpts/gate_7b_vllm"),"nothink_norag")
    m_by_idx={}
    for ds,rows in ev7.items():
        for i,r in rows.items(): m_by_idx[i]=margin(r)

    idx=sorted(set(rt7) & set(rt32))
    miss=[i for i in idx if i not in m_by_idx]
    if miss: print(f"WARN: {len(miss)} rt idx have no gate_7b_vllm margin (default no-escalate)")
    idx=[i for i in idx if i in m_by_idx]
    mg=np.array([[m_by_idx[i]] for i in idx]); esc=gate.predict_proba(mg)[:,1] < tau

    l7  = np.array([rt7[i]["latency_s"] for i in idx]);  l32 = np.array([rt32[i]["latency_s"] for i in idx])
    e7  = np.array([rt7[i]["energy_j"]  for i in idx]);  e32 = np.array([rt32[i]["energy_j"]  for i in idx])
    ok7 = np.array([rt7[i]["ok"] for i in idx]);         ok32= np.array([rt32[i]["ok"] for i in idx])
    vr7 = np.median([rt7[i]["resident_vram_mb"] for i in idx]); vr32=np.median([rt32[i]["resident_vram_mb"] for i in idx])

    lc = l7 + np.where(esc, l32, 0.0)
    ec = e7 + np.where(esc, e32, 0.0)
    okc= np.where(esc, ok32, ok7)

    def line(name, lat, en, ok, vram):
        mp = en.sum()/lat.sum() if lat.sum()>0 else 0.0
        print(f"  {name:<12}{ok.mean():>8.3f}{lat.mean():>10.2f}{pct(lat,50):>9.2f}{pct(lat,95):>9.2f}"
              f"{en.mean():>11.1f}{mp:>9.0f}{vram:>11.0f}")
    print(f"real-time single-query (batch-1), n={len(idx)}   escalation={100*esc.mean():.0f}%  (tau={tau:.3f})\n")
    print(f"  {'policy':<12}{'acc':>8}{'lat_mean':>10}{'p50':>9}{'p95':>9}{'energy/q':>11}{'meanW':>9}{'VRAM_MB':>11}")
    print("  "+"-"*78)
    line("always-7B",  l7, e7, ok7, vr7)
    line("always-32B", l32,e32,ok32, vr32)
    line("cascade",    lc, ec, okc, vr7+vr32)
    print(f"\n  speedup vs always-32B: {l32.mean()/lc.mean():.2f}x mean latency, "
          f"{l32.sum()/lc.sum():.2f}x throughput-equiv;  energy {ec.sum()/e32.sum()*100:.0f}% of always-32B")
    print("\nREAD: cascade latency/energy sit between the two legs, weighted by escalation. VRAM for the")
    print("cascade is both models resident (two GPUs in production). p95 shows tail latency a user sees.")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    ap.add_argument("--rt7", default="rt_7b.jsonl"); ap.add_argument("--rt32", default="rt_32b.jsonl")
    A=ap.parse_args(); run(A.repo, A.rt7, A.rt32)
