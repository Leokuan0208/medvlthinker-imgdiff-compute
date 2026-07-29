#!/usr/bin/env python3
"""
acc_v4_lowres_think.py - ACC-v4 = ACC-v3 (confidence-tightened think gate) + RESOLUTION-DECOUPLED
think tier. Finding: medical-VLM REASONING is resolution-insensitive (think@cap320 >= think@fullres
on the reasoning benchmarks MMMU/MedXpert that actually reach the think tier; MMMU 0.688->0.712),
while PERCEPTION is resolution-sensitive (handled by the no-think tiers anyway). So run the expensive
think tier at cap320, not fullres -> ~28% less think prefill, no loss on the reasoning residual.

Compares (honest 50/50 calib/test x20 seeds, min-think@parity, per-dataset + ALL-5/6 + cost):
  ACC-v2  : agreement think-gate, think@fullres        (the baseline)
  ACC-v3  : agree+confidence think-gate, think@fullres
  ACC-v4  : agree+confidence think-gate, think@CAP320   (resolution-decoupled reasoning tier)
MedVLThinker only (the family with a genuine firing think tier; lingshu fast-think/qoq degenerate).
Parity target = always-big-think@fullres (the strongest baseline). Launch from repo root.
"""
import os, sys, json, glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, ALL6, ALL5, COMPETENT, CACHE
J = lambda p: os.path.join(os.path.expanduser("~/medvlthinker-imgdiff-compute"), p)
ABBR = {"PMC-VQA":"PMC","SLAKE":"SLAKE","VQA-RAD":"VQARAD","PathVQA":"PathV","MMMU":"MMMU","MedXpert-Reasoning":"MedX-R","MedXpert-Understanding":"MedX-U"}
N0, N1 = 7.6e9, 33e9; LATD = "ckpts/acc_gen/medvlthinker/lat"; cache = json.load(open(J(CACHE)))
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
def fitm(tier, key):
    pts = []
    for f in glob.glob(J(os.path.join(LATD, tier, "ckpt_*lat*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get(key) is not None: pts.append((r.get("gen_tokens") or 0, r[key]))
    g = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    if g.std() < 1: return (float(np.median(y)), 0.0)
    b, a = np.polyfit(g, y, 1); rs = y-(b*g+a); k = np.abs(rs) <= 2.5*rs.std()+1e-9
    if k.sum() >= 4: b, a = np.polyfit(g[k], y[k], 1)
    return (float(a), float(max(b, 0)))
TI = ["small_nt","big_nt","big_think"]; LAT = {t: fitm(t,"latency_s") for t in TI}; EN = {t: fitm(t,"energy_j") for t in TI}
def tl(t,g): a,b=LAT[t]; return np.clip(a+b*g,0,None)
def te(t,g): a,b=EN[t]; return np.clip(a+b*g,0,None)
c0 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
c1 = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")
c2f = _load_arm(J("ckpts/gate_32b"), "think_norag")                       # think @ fullres
c2c = _load_arm(J("ckpts/gate_32b_modes/think_cap320"), "think_norag")    # think @ cap320

def load():
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0,c1,c2f,c2c]) or ds not in cache: continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds])&set(c1[ds])&set(c2f[ds])&set(c2c[ds])&{int(k) for k in cC}&{int(k) for k in cF})
        D[ds] = dict(ok0=np.array([c0[ds][i]["ok"] for i in idx],float),ok1=np.array([c1[ds][i]["ok"] for i in idx],float),
            ok2f=np.array([c2f[ds][i]["ok"] for i in idx],float), ok2c=np.array([c2c[ds][i]["ok"] for i in idx],float),
            m0=np.array([margin(c0[ds][i].get("opt_logprobs")) for i in idx]), m1=np.array([margin(c1[ds][i].get("opt_logprobs")) for i in idx]),
            dis=np.array([float(c0[ds][i]["pred"]!=c1[ds][i]["pred"]) for i in idx]),
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx],float), g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx],float),
            g2f=np.array([c2f[ds][i].get("gen_tokens") or 0 for i in idx],float), g2c=np.array([c2c[ds][i].get("gen_tokens") or 0 for i in idx],float),
            Pc=np.array([cC[str(i)][0] for i in idx],float), Pf=np.array([cF[str(i)][0] for i in idx],float))
    return D
def pool(D,names):
    names=[d for d in names if d in D]; out={k:np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"]=np.concatenate([[d]*len(D[d]["ok0"]) for d in names]); out["names"]=names; return out
# method = (think_mode, think_res): think_mode in {agree, agreeconf}; think_res in {full, cap320}
METHODS=[("ACC-v2 (agree, think@full)","agree","full"),("ACC-v3 (agree+conf, think@full)","agreeconf","full"),
         ("ACC-v4 (agree+conf, think@cap320)","agreeconf","cap320")]
def route(P,t0,t1,mode):
    E0=P["m0"]<t0; tm=(P["dis"]>0.5) if mode=="agree" else ((P["dis"]>0.5)&(P["m1"]<t1)); return E0,E0&tm
def accof(P,E0,E1,res): ok2=P["ok2c"] if res=="cap320" else P["ok2f"]; return np.where(~E0,P["ok0"],np.where(~E1,P["ok1"],ok2))
def cost(P,E0,E1,res):
    g2=P["g2c"] if res=="cap320" else P["g2f"]; P2=P["Pc"] if res=="cap320" else P["Pf"]
    f0=2*N0*(P["Pc"]+P["g0"]); f1=2*N1*(P["Pc"]+P["g1"]); f2=2*N1*(P2+g2)
    fl=f0+np.where(E0,f1,0)+np.where(E1,f2,0)
    lt=tl("small_nt",P["g0"])+np.where(E0,tl("big_nt",P["g1"]),0)+np.where(E1,tl("big_think",g2),0)
    en=te("small_nt",P["g0"])+np.where(E0,te("big_nt",P["g1"]),0)+np.where(E1,te("big_think",g2),0)
    return fl,lt,en
def calib(P,cal,mode,res,tgt):
    q0=np.quantile(P["m0"][cal],np.linspace(0,1,22)); q1=np.quantile(P["m1"][cal],np.linspace(0,1,22)) if mode=="agreeconf" else [np.inf]
    Pc={k:(v[cal] if isinstance(v,np.ndarray) else v) for k,v in P.items() if k!="names"}; best=None
    for t0 in q0:
        for t1 in q1:
            E0,E1=route(Pc,t0,t1,mode); ok=accof(Pc,E0,E1,res)
            if ok.mean()>=tgt-1e-9:
                th=E1.mean()
                if best is None or th<best[0]: best=(th,float(t0),float(t1))
    return (best[1],best[2]) if best else None
def main():
    D=load(); DUMP={"family":"medvlthinker","pools":{}}
    print("\n##########  ACC-v4: confidence think-gate + RESOLUTION-DECOUPLED think tier  ##########")
    for label,names in [("ALL-6",ALL6),("ALL-5",ALL5),("COMPETENT-4",COMPETENT)]:
        P=pool(D,names); n=len(P["ok0"]); parity=P["ok2f"].mean(); F2=(2*N1*(P["Pf"]+P["g2f"]))
        res={m[0]:defaultdict(list) for m in METHODS}; resb={m[0]:{d:[] for d in names} for m in METHODS}; reach={m[0]:0 for m in METHODS}
        for s in range(20):
            rng=np.random.default_rng(s); cal=np.zeros(n,bool)
            key=np.array([f"{d}{int(a)}{int(b)}" for d,a,b in zip(P["ds_of"],P["ok0"],P["ok2f"])])
            for k in np.unique(key):
                ix=np.where(key==k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]]=True
            tev=~cal; tgt=P["ok2f"][cal].mean(); dse=P["ds_of"][tev]; F2te=F2[tev].sum()
            for nm,mode,r in METHODS:
                cc=calib(P,cal,mode,r,tgt)
                if cc is None: continue
                reach[nm]+=1; E0,E1=route(P,cc[0],cc[1],mode); ok=accof(P,E0,E1,r); fl,lt,en=cost(P,E0,E1,r)
                bad=sum(1 for d in names if (dse==d).sum() and ok[tev][dse==d].mean()<P["ok0"][tev][dse==d].mean()-1e-9)
                R=res[nm]; R["acc"].append(ok[tev].mean()); R["esc0"].append(E0[tev].mean()); R["think"].append(E1[tev].mean())
                R["flops"].append(fl[tev].sum()/F2te); R["lat"].append(lt[tev].mean()); R["energy"].append(en[tev].mean()); R["bad"].append(bad)
                for d in names:
                    md=dse==d
                    if md.any(): resb[nm][d].append(ok[tev][md].mean())
        print(f"\n  ===== [{label}]  parity(always-big-think@full)={parity:.4f}  (20 seeds, min-think@parity) =====")
        print(f"  {'method':<34}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{'lat(s)':>9}{'energy(J)':>11}{'guard':>7}")
        DUMP["pools"][label]={"parity":float(parity),"rows":[]}
        for nm,_,_ in METHODS:
            if reach[nm]<20: print(f"  {nm:<34}  — parity {reach[nm]}/20 seeds"); continue
            R=res[nm]; mn=lambda k: float(np.mean(R[k]))
            print(f"  {nm:<34}{mn('acc'):>7.4f}{mn('esc0')*100:>6.0f}%{mn('think')*100:>6.0f}%{mn('flops')*100:>7.1f}%{mn('lat'):>8.2f}s{mn('energy'):>9.1f}J{mn('bad'):>7.2f}")
            DUMP["pools"][label]["rows"].append(dict(method=nm,acc=mn("acc"),esc0=mn("esc0"),think=mn("think"),flops=mn("flops"),lat=mn("lat"),energy=mn("energy"),guard=mn("bad"),bench={d:float(np.mean(resb[nm][d])) for d in names}))
        print(f"  --- per-benchmark accuracy @ parity [{label}] ---")
        print("  "+f"{'method':<34}"+"".join(f"{ABBR[d]:>8}" for d in names))
        for nm,_,_ in METHODS:
            if reach[nm]>=20: print("  "+f"{nm:<34}"+"".join(f"{np.mean(resb[nm][d]):>8.3f}" for d in names))
    json.dump(DUMP,open(J("results/cascade_methods/artifacts/rescue_allfam/accv4_medvlthinker.json"),"w"),indent=1)
    print("\n-> results/cascade_methods/artifacts/rescue_allfam/accv4_medvlthinker.json")
if __name__ == "__main__":
    main()
