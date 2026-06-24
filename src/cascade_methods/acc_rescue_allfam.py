#!/usr/bin/env python3
"""
acc_rescue_allfam.py - integrate the VISUAL-STABILITY RESCUE into the ACC-v2 (cross-model agreement)
cascade and test it across families, vs plain ACC-v2. Non-destructive (acc_v2.py / acc_allmethods.py
untouched). Reports PER-DATASET + ALL-5 + ALL-6 accuracy with escalation, FLOPs, latency, energy, guard.

THE CASCADE (3 compute tiers; stop at first accepted):
  Tier 0: small no-think @ cap320           gate: escalate if small margin < tau0
          [+RESCUE]: do NOT escalate if the small answer is RESOLUTION-STABLE (same letter across
                     caps {80,160,320,640}); only the visually-fragile low-margin go up. The 3 extra
                     small passes are charged ONLY on the would-escalate (margin<tau0) set.
  Tier 1: big no-think @ cap320             gate: escalate to think iff small-nt != big-nt (DISAGREE)
  Tier 2: big think @ fullres               terminal (reasoning residual)

Methods compared: always-small-nt, always-big-nt, always-big-think[parity], ACC-v2 (agreement),
ACC-v2 + rescue. Honest 50/50 calib/test x20 seeds; thresholds picked on calib at MIN FLOPs s.t.
calib acc >= calib parity (= always-big-think). FLOPs=2N(P+G); latency/energy = measured-batch1 a+b*gen
per tier. Only Qwen2.5-VL families have a max_pixels resolution ladder (multi-cap data); run those.
  python3 src/cascade_methods/acc_rescue_allfam.py --family medvlthinker|lingshu|qoq
"""
import os, sys, json, glob, re, argparse
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, ALL6, ALL5, COMPETENT, CACHE
REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
ABBR = {"PMC-VQA": "PMC", "SLAKE": "SLAKE", "VQA-RAD": "VQARAD", "PathVQA": "PathV", "MMMU": "MMMU",
        "MedXpert-Reasoning": "MedX-R", "MedXpert-Understanding": "MedX-U"}
EXTRA = ["cap80", "cap160", "cap640"]; OPCAP = "cap320"   # rescue: agreement of EXTRA vs OPCAP

# small-model cap dirs differ per family; big-nt/big-think/lat as in acc_allmethods.FAM
FAM = {
 "medvlthinker": dict(N0=7.6e9, N1=33e9, lat="ckpts/acc_gen/medvlthinker/lat",
    small_caps={c: (f"ckpts/gate_7b_prune/{c}", "nothink_norag") for c in ["cap80","cap160","cap320","cap640"]},
    c1=("ckpts/gate_32b_modes/nothink_cap320","nothink_norag"), c2=("ckpts/gate_32b","think_norag")),
 "lingshu": dict(N0=7.6e9, N1=33e9, lat="ckpts/acc_gen/lingshu/lat",
    small_caps={c: (f"ckpts/acc_gen/lingshu7b/{c}", "lingshu7b") for c in ["cap80","cap160","cap320","cap640"]},
    c1=("ckpts/acc_gen/lingshu32b/nothink_cap320","lingshu32b"), c2=("ckpts/acc_gen/lingshu32b/think_native","lingshu32b")),
 "qoq": dict(N0=7.6e9, N1=33e9, lat="ckpts/acc_gen/qoq/lat",
    small_caps={c: (f"ckpts/acc_gen/qoq7b/{c}", "qoq7b") for c in ["cap80","cap160","cap320","cap640"]},
    c1=("ckpts/acc_gen/qoq32b/nothink_cap320","qoq32b"), c2=("ckpts/acc_gen/qoq32b/think_native","qoq32b")),
}
ap = argparse.ArgumentParser(); ap.add_argument("--family", required=True, choices=list(FAM)); A = ap.parse_args()
C = FAM[A.family]; N0, N1 = C["N0"], C["N1"]
cache = json.load(open(J(CACHE)))

def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0]-v[1]) if len(v) >= 2 else 0.0
def fit_metric(base, tier, key):
    pts = []
    for f in glob.glob(J(os.path.join(base, tier, "ckpt_*lat*.jsonl"))):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get(key) is not None: pts.append((r.get("gen_tokens") or 0, r[key]))
    if len(pts) < 4: return None
    g = np.array([p[0] for p in pts], float); y = np.array([p[1] for p in pts], float)
    if g.std() < 1.0: return (float(np.median(y)), 0.0)
    b, a = np.polyfit(g, y, 1); resid = y-(b*g+a); sd = resid.std(); keep = np.abs(resid) <= 2.5*sd+1e-9
    if keep.sum() >= 4: b, a = np.polyfit(g[keep], y[keep], 1)
    return (float(a), float(max(b, 0.0)))
TIERS = ["small_nt", "big_nt", "big_think"]
LAT = {t: fit_metric(C["lat"], t, "latency_s") for t in TIERS}
EN  = {t: fit_metric(C["lat"], t, "energy_j") for t in TIERS}
HAVE = all(LAT[t] for t in TIERS) and all(EN[t] for t in TIERS)
def tlat(t, g): a, b = LAT[t]; return np.clip(a+b*g, 0, None)
def ten(t, g): a, b = EN[t]; return np.clip(a+b*g, 0, None)

# load small-nt at every cap, big-nt, big-think
caps = {c: _load_arm(J(C["small_caps"][c][0]), C["small_caps"][c][1]) for c in C["small_caps"]}
c1 = _load_arm(J(C["c1"][0]), C["c1"][1]); c2 = _load_arm(J(C["c2"][0]), C["c2"][1])
c0 = caps[OPCAP]

def load():
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c1, c2]) or ds not in cache: continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
        idx = [i for i in idx if all(i in caps[c].get(ds, {}) for c in caps)]
        if not idx: continue
        p_op = {i: c0[ds][i]["pred"] for i in idx}
        res_stable = np.array([float(all(caps[c][ds][i]["pred"] == p_op[i] for c in EXTRA)) for i in idx])
        # extra-pass cost (only used when escalating): sum over EXTRA caps of one small pass
        ex_fl = np.zeros(len(idx)); ex_lt = np.zeros(len(idx)); ex_en = np.zeros(len(idx))
        for c in EXTRA:
            Pc = np.array([cache[ds][c][str(i)][0] for i in idx], float)
            gc = np.array([caps[c][ds][i].get("gen_tokens") or 2 for i in idx], float)
            ex_fl += 2*N0*(Pc+gc)
            ex_lt += tlat("small_nt", gc) if HAVE else 0.0
            ex_en += ten("small_nt", gc) if HAVE else 0.0
        D[ds] = dict(
            ok0=np.array([c0[ds][i]["ok"] for i in idx], float), ok1=np.array([c1[ds][i]["ok"] for i in idx], float),
            ok2=np.array([c2[ds][i]["ok"] for i in idx], float),
            m0=np.array([margin(c0[ds][i].get("opt_logprobs")) for i in idx]),
            m1=np.array([margin(c1[ds][i].get("opt_logprobs")) for i in idx]),
            disagree=np.array([float(c0[ds][i]["pred"] != c1[ds][i]["pred"]) for i in idx]),
            res_stable=res_stable, ex_fl=ex_fl, ex_lt=ex_lt, ex_en=ex_en,
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx], float),
            Pc=np.array([cC[str(i)][0] for i in idx], float), Pf=np.array([cF[str(i)][0] for i in idx], float))
    return D
def pool(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"] = np.concatenate([[d]*len(D[d]["ok0"]) for d in names]); out["names"] = names
    return out

# mode: "v2"=agreement; "rescue0"=resolution-rescue on tier-0 (cheap gate); "rescue2"=on think tier (expensive gate)
def cascade(P, tau0, mode):
    elig = P["m0"] < tau0; stable = P["res_stable"] > 0.5; dis = P["disagree"] > 0.5
    if mode == "rescue0":
        E0 = elig & (~stable); E1 = E0 & dis; xmask = elig                 # extra passes on tier-0 eligible
    elif mode == "rescue2":
        E0 = elig; E1 = E0 & dis & (~stable); xmask = E0 & dis             # extra passes only on think-candidates
    else:
        E0 = elig; E1 = E0 & dis; xmask = np.zeros(len(elig), bool)
    return elig, E0, E1, xmask
def costs(P, E0, E1, xmask):
    f0 = 2*N0*(P["Pc"]+P["g0"]); f1 = 2*N1*(P["Pc"]+P["g1"]); f2 = 2*N1*(P["Pf"]+P["g2"])
    fl = f0 + np.where(E0, f1, 0) + np.where(E1, f2, 0) + np.where(xmask, P["ex_fl"], 0)
    if HAVE:
        l0 = tlat("small_nt", P["g0"]); l1 = tlat("big_nt", P["g1"]); l2 = tlat("big_think", P["g2"])
        e0 = ten("small_nt", P["g0"]); e1 = ten("big_nt", P["g1"]); e2 = ten("big_think", P["g2"])
        lt = l0 + np.where(E0, l1, 0) + np.where(E1, l2, 0) + np.where(xmask, P["ex_lt"], 0)
        en = e0 + np.where(E0, e1, 0) + np.where(E1, e2, 0) + np.where(xmask, P["ex_en"], 0)
    else:
        lt = en = np.zeros(len(fl))
    return fl, f2, lt, en
def acc_of(P, E0, E1): return np.where(~E0, P["ok0"], np.where(~E1, P["ok1"], P["ok2"]))

MODES = [("ACC-v2 (agreement)", "v2"), ("ACC-v2 + rescue@tier0", "rescue0"),
         ("ACC-v2 + rescue@think", "rescue2")]
def frontier(P, mode):
    """In-sample frontier: sweep tau0; per point return cost+accuracy (relative compare is fair)."""
    F2 = (2*N1*(P["Pf"]+P["g2"])).sum()
    taus = np.unique(np.quantile(P["m0"], np.linspace(0, 1, 81)))
    pts = []
    for t0 in taus:
        elig, E0, E1, xm = cascade(P, t0, mode)
        ok = acc_of(P, E0, E1); fl, _, lt, en = costs(P, E0, E1, xm)
        bench = {d: float(ok[P["ds_of"] == d].mean()) for d in P["names"]}
        pts.append(dict(tau0=float(t0), acc=float(ok.mean()), flops=float(fl.sum()/F2),
                        lat=float(lt.mean()), energy=float(en.mean()),
                        esc0=float(E0.mean()), think=float(E1.mean()), bench=bench))
    return pts
def min_at(pts, T):
    cand = [p for p in pts if p["acc"] >= T - 1e-9]
    return min(cand, key=lambda p: p["flops"]) if cand else None

def calib_tau(P, cal, mode, tgt):
    """min-FLOPs tau0 on calib s.t. calib acc >= tgt (canonical acc_allmethods objective). None if unreachable."""
    q = np.quantile(P["m0"][cal], np.linspace(0, 1, 26)); best = None
    Pc = {k: (v[cal] if isinstance(v, np.ndarray) else v) for k, v in P.items() if k != "names"}
    for t0 in q:
        _, E0, E1, xm = cascade(Pc, t0, mode)
        ok = acc_of(Pc, E0, E1)
        if ok.mean() >= tgt - 1e-9:
            fl = costs(Pc, E0, E1, xm)[0].sum()
            if best is None or fl < best[0]: best = (fl, float(t0))
    return best[1] if best else None

def honest_parity(P, names, label, DUMP):
    """Honest 50/50 calib/test, 20 seeds, min-FLOPs at always-big-think parity. Per-dataset + costs."""
    n = len(P["ok0"]); F2all = (2*N1*(P["Pf"]+P["g2"]))
    res = {nm: defaultdict(list) for nm, _ in MODES}; resb = {nm: {d: [] for d in names} for nm, _ in MODES}
    reach = {nm: 0 for nm, _ in MODES}
    for s in range(20):
        rng = np.random.default_rng(s); cal = np.zeros(n, bool)
        key = np.array([f"{d}{int(a)}{int(b)}" for d, a, b in zip(P["ds_of"], P["ok0"], P["ok2"])])
        for k in np.unique(key):
            ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]] = True
        te = ~cal; tgt = P["ok2"][cal].mean(); dse = P["ds_of"][te]; F2te = F2all[te].sum()
        for nm, mode in MODES:
            t0 = calib_tau(P, cal, mode, tgt)
            if t0 is None:                                  # cannot reach parity on calib -> mark, skip metrics
                continue
            reach[nm] += 1
            _, E0, E1, xm = cascade(P, t0, mode)
            ok = acc_of(P, E0, E1); fl, _, lt, en = costs(P, E0, E1, xm)
            bad = sum(1 for d in names if (dse == d).sum() and ok[te][dse==d].mean() < P["ok0"][te][dse==d].mean()-1e-9)
            R = res[nm]; R["acc"].append(ok[te].mean()); R["esc0"].append(E0[te].mean()); R["think"].append(E1[te].mean())
            R["flops"].append(fl[te].sum()/F2te); R["lat"].append(lt[te].mean()); R["energy"].append(en[te].mean()); R["bad"].append(bad)
            for d in names:
                md = dse == d
                if md.any(): resb[nm][d].append(ok[te][md].mean())
    print(f"\n  ===== HONEST calib/test @ always-big-think PARITY [{label}] (20 seeds, min-FLOPs) =====")
    lh = "lat(s)" if HAVE else "lat"; eh = "energy(J)" if HAVE else "en"
    print(f"  {'method':<26}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}{lh:>9}{eh:>11}{'guard':>7}")
    DUMP["pools"][label]["honest_parity"] = {}
    for nm, _ in MODES:
        if reach[nm] < 20:
            print(f"  {nm:<26}  — reaches parity in {reach[nm]}/20 seeds (wrong-tier: caps accuracy)")
            DUMP["pools"][label]["honest_parity"][nm] = {"reach": reach[nm]}
            continue
        R = res[nm]; mn = lambda k: float(np.mean(R[k]))
        ls = f"{mn('lat'):>8.2f}s" if HAVE else f"{'n/a':>9}"; es = f"{mn('energy'):>9.1f}J" if HAVE else f"{'n/a':>11}"
        print(f"  {nm:<26}{mn('acc'):>7.4f}{mn('esc0')*100:>6.0f}%{mn('think')*100:>6.0f}%{mn('flops')*100:>7.1f}%{ls}{es}{mn('bad'):>7.2f}")
        DUMP["pools"][label]["honest_parity"][nm] = dict(reach=20, acc=mn("acc"), esc0=mn("esc0"), think=mn("think"),
            flops=mn("flops"), lat=(mn("lat") if HAVE else None), energy=(mn("energy") if HAVE else None),
            guard=mn("bad"), bench={d: float(np.mean(resb[nm][d])) for d in names})
    print(f"  --- per-benchmark accuracy @ parity [{label}] ---")
    print("  " + f"{'method':<26}" + "".join(f"{ABBR[d]:>8}" for d in names))
    for nm, _ in MODES:
        if reach[nm] >= 20:
            print("  " + f"{nm:<26}" + "".join(f"{np.mean(resb[nm][d]):>8.3f}" for d in names))

def run_pool(D, names, label, DUMP):
    P = pool(D, names); parity = P["ok2"].mean(); small = P["ok0"].mean(); bignt = P["ok1"].mean()
    FR = {nm: frontier(P, mode) for nm, mode in MODES}
    ceil = {nm: max(p["acc"] for p in FR[nm]) for nm in FR}
    print(f"\n  ===== [{label}] =====  always-small={small:.4f}  always-big-nt={bignt:.4f}  parity(big-think)={parity:.4f}")
    print("  reachable accuracy ceiling: " + "  ".join(f"{nm.split('+ ')[-1]}={ceil[nm]:.4f}" for nm in ceil))
    common = min(ceil.values()) - 0.002                       # highest target ALL methods reach
    targets = [("common (all reach)", common), ("parity-0.3pt", parity-0.003), ("parity", parity)]
    print(f"  min-cost @ matched accuracy:")
    print(f"  {'target':<18}{'method':<24}{'acc':>7}{'esc0':>7}{'think':>7}{'FLOPs%':>8}" + (f"{'lat(s)':>9}{'energy(J)':>11}" if HAVE else ""))
    op = {}
    for tname, T in targets:
        for nm, _ in MODES:
            p = min_at(FR[nm], T)
            if p is None:
                print(f"  {tname:<18}{nm:<24}{'— cannot reach':>20}"); continue
            ls = f"{p['lat']:>8.2f}s" if HAVE else ""; es = f"{p['energy']:>9.1f}J" if HAVE else ""
            print(f"  {tname:<18}{nm:<24}{p['acc']:>7.4f}{p['esc0']*100:>6.0f}%{p['think']*100:>6.0f}%{p['flops']*100:>7.1f}%{ls}{es}")
            if tname == "common (all reach)": op[nm] = p
        print()
    print(f"  --- per-benchmark accuracy @ common target ({common:.4f}) ---")
    base = {"always-small-nt": {d: float(P["ok0"][P["ds_of"]==d].mean()) for d in names},
            "always-big-nt":   {d: float(P["ok1"][P["ds_of"]==d].mean()) for d in names},
            "always-big-think":{d: float(P["ok2"][P["ds_of"]==d].mean()) for d in names}}
    print("  " + f"{'method':<24}" + "".join(f"{ABBR[d]:>8}" for d in names) + f"{'ALL':>8}")
    upool = lambda b: float(np.mean([b[d] for d in names]))
    for nm, b in base.items():
        print("  " + f"{nm:<24}" + "".join(f"{b[d]:>8.3f}" for d in names) + f"{upool(b):>8.3f}")
    for nm, _ in MODES:
        if nm in op:
            b = op[nm]["bench"]
            print("  " + f"{nm:<24}" + "".join(f"{b[d]:>8.3f}" for d in names) + f"{op[nm]['acc']:>8.3f}")
    DUMP["pools"][label] = dict(parity=float(parity), always_small=float(small), always_big_nt=float(bignt),
        ceil={nm: float(ceil[nm]) for nm in ceil}, common_target=float(common),
        frontiers={nm: FR[nm] for nm in FR}, op_common=op, base_bench=base)
    honest_parity(P, names, label, DUMP)

def main():
    D = load()
    print(f"\n##########  ACC-v2 + VISUAL-STABILITY RESCUE : {A.family.upper()}  ##########")
    print(f"  tiers: small-nt@cap320 -> big-nt@cap320 -> big-think@fullres | rescue extra caps={EXTRA}")
    print(f"  measured latency/energy: {'YES' if HAVE else 'NO (FLOPs only)'}")
    DUMP = {"family": A.family, "pools": {}}
    for label, names in [("ALL-6", ALL6), ("ALL-5", ALL5), ("COMPETENT-4", COMPETENT)]:
        run_pool(D, names, label, DUMP)
    os.makedirs(J("results/cascade_methods/rescue_allfam"), exist_ok=True)
    json.dump(DUMP, open(J(f"results/cascade_methods/rescue_allfam/{A.family}.json"), "w"), indent=1)
    print(f"\n-> results/cascade_methods/rescue_allfam/{A.family}.json")

if __name__ == "__main__":
    main()
