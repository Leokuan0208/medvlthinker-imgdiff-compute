#!/usr/bin/env python3
"""
acc_v3_confgate.py - validate the CONFIDENCE-TIGHTENED think-gate across families.

ACC-v2 (our baseline) fires the expensive think tier on EVERY small-nt/big-nt DISAGREEMENT. The
control (control_think_signals.py) showed that gate over-fires think. ACC-v3 tightens it: fire think
only when the two no-think models disagree AND the big no-think model is itself UNSURE (its margin <
tau1). i.e. don't pay 28s of reasoning when the big model is already confident, even if the small one
disagreed.

  Tier 0: small-nt@cap320     escalate to big-nt if small margin < tau0
  Tier 1: big-nt@cap320       -> think iff:  v1: big-nt margin < tau1
                                             v2: small-nt != big-nt            (agreement, ACC-v2)
                                             v3: (small-nt != big-nt) AND (big-nt margin < tau1)
  Tier 2: big-think@fullres   terminal

Honest 50/50 calib/test x20 seeds, thresholds chosen on calib at MIN think-rate (canonical ACC
latency proxy) s.t. calib acc >= calib parity (always-big-think). Per-dataset + ALL-5/6 + measured
latency/energy/FLOPs/guard. Families = the 3 Qwen2.5-VL with big-nt logprobs.
  python3 src/cascade_methods/acc_v3_confgate.py --family medvlthinker|lingshu|qoq
"""
import os, sys, json, glob, argparse
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, ALL6, ALL5, COMPETENT, CACHE
REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
ABBR = {"PMC-VQA":"PMC","SLAKE":"SLAKE","VQA-RAD":"VQARAD","PathVQA":"PathV","MMMU":"MMMU",
        "MedXpert-Reasoning":"MedX-R","MedXpert-Understanding":"MedX-U"}
FAM = {
 "medvlthinker": dict(N0=7.6e9,N1=33e9,lat="ckpts/acc_gen/medvlthinker/lat",
    c0=("ckpts/gate_7b_prune/cap320","nothink_norag"), c1=("ckpts/gate_32b_modes/nothink_cap320","nothink_norag"), c2=("ckpts/gate_32b","think_norag")),
 "lingshu": dict(N0=7.6e9,N1=33e9,lat="ckpts/acc_gen/lingshu/lat",
    c0=("ckpts/acc_gen/lingshu7b/cap320","lingshu7b"), c1=("ckpts/acc_gen/lingshu32b/nothink_cap320","lingshu32b"), c2=("ckpts/acc_gen/lingshu32b/think_native","lingshu32b")),
 "qoq": dict(N0=7.6e9,N1=33e9,lat="ckpts/acc_gen/qoq/lat",
    c0=("ckpts/acc_gen/qoq7b/cap320","qoq7b"), c1=("ckpts/acc_gen/qoq32b/nothink_cap320","qoq32b"), c2=("ckpts/acc_gen/qoq32b/think_native","qoq32b")),
}
ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True, choices=list(FAM)); A = ap.parse_args()
C = FAM[A.family]; N0, N1 = C["N0"], C["N1"]; cache = json.load(open(J(CACHE)))
def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
def fitm(base, tier, key):
    pts = []
    for f in glob.glob(J(os.path.join(base, tier, "ckpt_*lat*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get(key) is not None: pts.append((r.get("gen_tokens") or 0, r[key]))
    if len(pts) < 4: return None
    g = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    if g.std() < 1: return (float(np.median(y)), 0.0)
    b, a = np.polyfit(g, y, 1); rs = y-(b*g+a); k = np.abs(rs) <= 2.5*rs.std()+1e-9
    if k.sum() >= 4: b, a = np.polyfit(g[k], y[k], 1)
    return (float(a), float(max(b, 0)))
T = ["small_nt","big_nt","big_think"]; LAT = {t: fitm(C["lat"],t,"latency_s") for t in T}; EN = {t: fitm(C["lat"],t,"energy_j") for t in T}
HAVE = all(LAT[t] for t in T) and all(EN[t] for t in T)
def tl(t,g): a,b=LAT[t]; return np.clip(a+b*g,0,None)
def te(t,g): a,b=EN[t]; return np.clip(a+b*g,0,None)
c0=_load_arm(J(C["c0"][0]),C["c0"][1]); c1=_load_arm(J(C["c1"][0]),C["c1"][1]); c2=_load_arm(J(C["c2"][0]),C["c2"][1])
def load():
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0,c1,c2]) or ds not in cache: continue
        cC=cache[ds]["cap320"]; cF=cache[ds]["fullres"]
        idx=sorted(set(c0[ds])&set(c1[ds])&set(c2[ds])&{int(k) for k in cC}&{int(k) for k in cF})
        if not idx: continue
        D[ds]=dict(ok0=np.array([c0[ds][i]["ok"] for i in idx],float),ok1=np.array([c1[ds][i]["ok"] for i in idx],float),
            ok2=np.array([c2[ds][i]["ok"] for i in idx],float),
            m0=np.array([margin(c0[ds][i].get("opt_logprobs")) for i in idx]),
            m1=np.array([margin(c1[ds][i].get("opt_logprobs")) for i in idx]),
            disagree=np.array([float(c0[ds][i]["pred"]!=c1[ds][i]["pred"]) for i in idx]),
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx],float),
            g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx],float),
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx],float),
            Pc=np.array([cC[str(i)][0] for i in idx],float),Pf=np.array([cF[str(i)][0] for i in idx],float))
    return D
def pool(D,names):
    names=[d for d in names if d in D]; out={k:np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"]=np.concatenate([[d]*len(D[d]["ok0"]) for d in names]); out["names"]=names; return out
def think_mask(P, mode, tau1):
    dis = P["disagree"] > 0.5; conf = P["m1"] < tau1
    if mode == "v1": return conf
    if mode == "v2": return dis
    return dis & conf                                   # v3
def routed(P, tau0, mode, tau1):
    E0 = P["m0"] < tau0; E1 = E0 & think_mask(P, mode, tau1); return E0, E1
def accof(P, E0, E1): return np.where(~E0, P["ok0"], np.where(~E1, P["ok1"], P["ok2"]))
def cost(P, E0, E1):
    f0=2*N0*(P["Pc"]+P["g0"]); f1=2*N1*(P["Pc"]+P["g1"]); f2=2*N1*(P["Pf"]+P["g2"])
    fl=f0+np.where(E0,f1,0)+np.where(E1,f2,0)
    if HAVE:
        lt=tl("small_nt",P["g0"])+np.where(E0,tl("big_nt",P["g1"]),0)+np.where(E1,tl("big_think",P["g2"]),0)
        en=te("small_nt",P["g0"])+np.where(E0,te("big_nt",P["g1"]),0)+np.where(E1,te("big_think",P["g2"]),0)
    else: lt=en=np.zeros(len(fl))
    return fl,f2,lt,en
MODES=[("ACC-v1 (confidence)","v1"),("ACC-v2 (agreement)","v2"),("ACC-v3 (agree+confidence)","v3")]
def calib(P, cal, mode, tgt):
    """min THINK-RATE on calib s.t. calib acc>=tgt (canonical ACC latency proxy). returns (tau0,tau1) or None."""
    q0=np.quantile(P["m0"][cal],np.linspace(0,1,22)); q1=np.quantile(P["m1"][cal],np.linspace(0,1,22)) if mode!="v2" else [np.inf]
    Pc={k:(v[cal] if isinstance(v,np.ndarray) else v) for k,v in P.items() if k!="names"}; best=None
    for t0 in q0:
        for t1 in q1:
            E0,E1=routed(Pc,t0,mode,t1); ok=accof(Pc,E0,E1)
            if ok.mean()>=tgt-1e-9:
                th=E1.mean()
                if best is None or th<best[0]: best=(th,float(t0),float(t1))
    return (best[1],best[2]) if best else None
def main():
    D=load(); DUMP={"family":A.family,"pools":{}}
    print(f"\n##########  ACC-v3 CONFIDENCE-TIGHTENED THINK-GATE : {A.family.upper()}  ##########")
    print(f"  measured latency/energy: {'YES' if HAVE else 'NO'}")
    for label,names in [("ALL-6",ALL6),("ALL-5",ALL5),("COMPETENT-4",COMPETENT)]:
        P=pool(D,names); n=len(P["ok0"]); F2all=2*N1*(P["Pf"]+P["g2"]); parity=P["ok2"].mean()
        res={m[0]:defaultdict(list) for m in MODES}; resb={m[0]:{d:[] for d in names} for m in MODES}; reach={m[0]:0 for m in MODES}
        for s in range(20):
            rng=np.random.default_rng(s); cal=np.zeros(n,bool)
            key=np.array([f"{d}{int(a)}{int(b)}" for d,a,b in zip(P["ds_of"],P["ok0"],P["ok2"])])
            for k in np.unique(key):
                ix=np.where(key==k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]]=True
            tev=~cal; tgt=P["ok2"][cal].mean(); dse=P["ds_of"][tev]; F2te=F2all[tev].sum()
            for nm,mode in MODES:
                cc=calib(P,cal,mode,tgt)
                if cc is None: continue
                reach[nm]+=1; E0,E1=routed(P,cc[0],mode,cc[1]); ok=accof(P,E0,E1); fl,_,lt,en=cost(P,E0,E1)
                bad=sum(1 for d in names if (dse==d).sum() and ok[tev][dse==d].mean()<P["ok0"][tev][dse==d].mean()-1e-9)
                R=res[nm]; R["acc"].append(ok[tev].mean()); R["esc0"].append(E0[tev].mean()); R["think"].append(E1[tev].mean())
                R["flops"].append(fl[tev].sum()/F2te); R["lat"].append(lt[tev].mean()); R["energy"].append(en[tev].mean()); R["bad"].append(bad)
                for d in names:
                    md=dse==d
                    if md.any(): resb[nm][d].append(ok[tev][md].mean())
        print(f"\n  ===== [{label}]  parity(always-big-think)={parity:.4f}  (honest 50/50, 20 seeds, min-think@parity) =====")
        lh="lat(s)" if HAVE else "lat"; eh="energy(J)" if HAVE else "en"
        print(f"  {'method':<28}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{lh:>9}{eh:>11}{'guard':>7}")
        DUMP["pools"][label]={"parity":float(parity),"rows":[]}
        # static anchors (always-small / always-big-nt / always-big-think) with full cost
        F2tot=F2all.sum(); pb=lambda okf,d: float(okf[P["ds_of"]==d].mean())
        def anchor(E0v,E1v):
            E0=np.full(n,E0v,bool); E1=np.full(n,E1v,bool); ok=accof(P,E0,E1); fl,_,lt,en=cost(P,E0,E1)
            return ok.mean(),E0.mean(),E1.mean(),fl.sum()/F2tot,(lt.mean() if HAVE else None),(en.mean() if HAVE else None),ok
        for an,(e0v,e1v) in [("always-small-nt",(0,0)),("always-big-nt",(1,0)),("always-big-think",(1,1))]:
            a,ae0,ae1,afl,alt,aen,aok=anchor(e0v,e1v)
            gbad=sum(1 for d in names if pb(aok,d)<pb(P["ok0"],d)-1e-9) if an!="always-small-nt" else 0
            DUMP["pools"][label]["rows"].append(dict(method=an,acc=float(a),esc0=float(ae0),think=float(ae1),flops=float(afl),
                lat=(float(alt) if HAVE else None),energy=(float(aen) if HAVE else None),guard=float(gbad),
                bench={d:pb(aok,d) for d in names},anchor=True))
        for nm,_ in MODES:
            if reach[nm]<20:
                print(f"  {nm:<28}  — reaches parity {reach[nm]}/20 seeds"); continue
            R=res[nm]; mn=lambda k: float(np.mean(R[k]))
            ls=f"{mn('lat'):>8.2f}s" if HAVE else f"{'n/a':>9}"; es=f"{mn('energy'):>9.1f}J" if HAVE else f"{'n/a':>11}"
            print(f"  {nm:<28}{mn('acc'):>7.4f}{mn('esc0')*100:>6.0f}%{mn('think')*100:>6.0f}%{mn('flops')*100:>7.1f}%{ls}{es}{mn('bad'):>7.2f}")
            DUMP["pools"][label]["rows"].append(dict(method=nm,acc=mn("acc"),esc0=mn("esc0"),think=mn("think"),flops=mn("flops"),
                lat=(mn("lat") if HAVE else None),energy=(mn("energy") if HAVE else None),guard=mn("bad"),
                bench={d:float(np.mean(resb[nm][d])) for d in names}))
        print(f"  --- per-benchmark accuracy @ parity [{label}] ---")
        print("  "+f"{'method':<28}"+"".join(f"{ABBR[d]:>8}" for d in names))
        for nm,_ in MODES:
            if reach[nm]>=20: print("  "+f"{nm:<28}"+"".join(f"{np.mean(resb[nm][d]):>8.3f}" for d in names))
    os.makedirs(J("results/cascade_methods/rescue_allfam"),exist_ok=True)
    json.dump(DUMP,open(J(f"results/cascade_methods/rescue_allfam/accv3_{A.family}.json"),"w"),indent=1)
    print(f"\n-> results/cascade_methods/rescue_allfam/accv3_{A.family}.json")
if __name__ == "__main__":
    main()
