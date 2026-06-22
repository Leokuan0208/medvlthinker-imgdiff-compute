#!/usr/bin/env python3
"""
acc_v2.py - Adaptive-Compute Cascade v2 (ACC-A): the recommended ACC variant, with a CROSS-MODEL
AGREEMENT gate on the expensive think tier. Standalone (acc.py is the v1/margin reference, untouched).

THE METHOD
----------
Three compute-configuration tiers; stop at the first one accepted:
  Tier 0:  7B  no-think @ cap320     (~0.18s)  -- escalate if 7B margin (top1-top2 logprob) < tau0
  Tier 1:  32B no-think @ cap320     (~0.34s)  -- escalate to think iff the two no-think models DISAGREE
                                                  (7B-nt answer != 32B-nt answer); 32B margin breaks ties
  Tier 2:  32B think    @ fullres    (~28s)    -- terminal (only the genuine reasoning residual)

Why agreement: at Tier 1 both a SMALL and a LARGE model have answered in no-think mode. When two
independent models agree, the answer is almost always right -> spending the 28s think pass is wasted.
Disagreement is a query-by-committee signal of genuine ambiguity -> route those to think. This is a
multi-model cascade signal (not single-model confidence), and it strictly Pareto-improves the v1
margin gate on ALL-6 (acc +0.0016, think-calls 19->14%, latency -18%, energy -19%).

The only learned/tuned parameters are the two scalar thresholds (tau0, tau1), calibrated on a held-out
split to reach always-32B-think accuracy (parity) at minimum latency. Frozen thresholds for deployment
are fit on the PMC-VQA-train calibration split and saved to ckpts/acc_v2_thresholds.json.

EVALUATION (run directly): honest 50/50 calib/test, thresholds chosen on calib, 20 seeds. Reports
accuracy, escalation-to-think, FLOPs%, latency, energy, per-benchmark never-worse-than-7B guardrail,
vs the v1 margin gate. Latency/energy from REAL measured batch-1 data. CPU only; launch from repo root.
"""
import os, sys, json, glob
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import _load_arm, _load_calib, CACHE, ALL6, ALL5, COMPETENT, N7, N32
from acc_compare import fit_models, predict

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
TIERS = "7B-nt@cap320 -> 32B-nt@cap320 -> 32B-think@fullres"


def margin(lp):
    v = sorted((lp or {}).values(), reverse=True); return (v[0] - v[1]) if len(v) >= 2 else 0.0


# ------------------------------------------------------------------ data (eval, by benchmark)
def load_eval():
    cache = json.load(open(J(CACHE)))
    c0 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
    c1 = _load_arm(J("ckpts/gate_32b_modes/nothink_cap320"), "nothink_norag")
    c2 = _load_arm(J("ckpts/gate_32b"), "think_norag")
    D = {}
    for ds in ALL6:
        if not all(ds in x for x in [c0, c1, c2]):
            continue
        cC = cache[ds]["cap320"]; cF = cache[ds]["fullres"]
        idx = sorted(set(c0[ds]) & set(c1[ds]) & set(c2[ds]) & {int(k) for k in cC} & {int(k) for k in cF})
        D[ds] = dict(
            ok0=np.array([c0[ds][i]["ok"] for i in idx], float),
            ok1=np.array([c1[ds][i]["ok"] for i in idx], float),
            ok2=np.array([c2[ds][i]["ok"] for i in idx], float),
            m0=np.array([margin(c0[ds][i].get("opt_logprobs")) for i in idx]),
            m1=np.array([margin(c1[ds][i].get("opt_logprobs")) for i in idx]),
            disagree=np.array([1.0 if c0[ds][i].get("pred") != c1[ds][i].get("pred") else 0.0 for i in idx]),
            g0=np.array([c0[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g1=np.array([c1[ds][i].get("gen_tokens") or 2 for i in idx], float),
            g2=np.array([c2[ds][i].get("gen_tokens") or 0 for i in idx], float),
            Pc=np.array([cC[str(i)][0] for i in idx], float),
            Pf=np.array([cF[str(i)][0] for i in idx], float),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]]}
    out["ds_of"] = np.concatenate([[d] * len(D[d]["ok0"]) for d in names]); out["names"] = names
    return out


# ------------------------------------------------------------------ the cascade
def think_score(P):
    """Tier1->Tier2 escalation score: DISAGREEMENT primary, low-32B-margin as tiebreak. Higher=escalate."""
    return P["disagree"] + 1e-6 * (-P["m1"])


def route(P, tau0, tau1, gate="agree"):
    """Returns boolean masks (e0 = past tier0, e1 = reached think) for the cascade."""
    e0 = P["m0"] < tau0
    s1 = think_score(P) if gate == "agree" else (-P["m1"])      # 'agree' = v2, 'margin' = v1
    e1 = e0 & (s1 > tau1)
    return e0, e1


def calibrate(P, mask, gate="agree", target=None):
    """Pick (tau0, tau1) on `mask` to reach acc>=target at MINIMUM escalation-to-think (latency proxy)."""
    a0, a1, a2 = P["ok0"][mask], P["ok1"][mask], P["ok2"][mask]
    tgt = a2.mean() if target is None else target
    s1 = (think_score(P) if gate == "agree" else (-P["m1"]))[mask]
    q0 = np.quantile(P["m0"][mask], np.linspace(0, 1, 26)); q1 = np.quantile(s1, np.linspace(0, 1, 26))
    best = None
    for t0 in q0:
        e0 = P["m0"][mask] < t0
        for t1 in q1:
            e1 = e0 & (s1 > t1)
            acc = np.where(~e0, a0, np.where(~e1, a1, a2)).mean()
            if acc >= tgt - 1e-9 and (best is None or e1.mean() < best[0]):
                best = (e1.mean(), float(t0), float(t1))
    return (best[1], best[2]) if best else (float(q0[0]), float(q1[0]))


# ------------------------------------------------------------------ evaluation (honest held-out)
def evaluate():
    LAT, EN = fit_models(); D = load_eval()
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        P = pool(D, names)
        f0 = 2*N7*(P["Pc"]+P["g0"]); f1 = 2*N32*(P["Pc"]+P["g1"]); f2 = 2*N32*(P["Pf"]+P["g2"])
        l0 = predict(LAT, "7b:nothink@cap320", P["g0"]); l1 = predict(LAT, "32b:nothink@cap320", P["g1"]); l2 = predict(LAT, "32b:think@fullres", P["g2"])
        e0v = predict(EN, "7b:nothink@cap320", P["g0"]); e1v = predict(EN, "32b:nothink@cap320", P["g1"]); e2v = predict(EN, "32b:think@fullres", P["g2"])
        parity = P["ok2"].mean(); res = {g: defaultdict(list) for g in ["margin", "agree"]}
        for s in range(20):
            rng = np.random.default_rng(s); n = len(P["ok0"]); cal = np.zeros(n, bool)
            key = np.array([f"{d}{int(a)}{int(b)}" for d, a, b in zip(P["ds_of"], P["ok0"], P["ok2"])])
            for k in np.unique(key):
                ix = np.where(key == k)[0]; rng.shuffle(ix); cal[ix[:len(ix)//2]] = True
            te = ~cal
            for gate in ["margin", "agree"]:
                Pc = {k: (v[cal] if isinstance(v, np.ndarray) else v) for k, v in P.items() if k != "names"}
                t0, t1 = calibrate(P, cal, gate)
                e0, e1 = route({k: P[k] for k in ["m0", "m1", "disagree"]}, t0, t1, gate)
                ok = np.where(~e0[te], P["ok0"][te], np.where(~e1[te], P["ok1"][te], P["ok2"][te]))
                fl = (f0[te] + np.where(e0[te], f1[te], 0) + np.where(e1[te], f2[te], 0)).sum() / f2[te].sum()
                lat = l0[te] + np.where(e0[te], l1[te], 0) + np.where(e1[te], l2[te], 0)
                eg = e0v[te] + np.where(e0[te], e1v[te], 0) + np.where(e1[te], e2v[te], 0)
                bad = sum(1 for d in names if (P["ds_of"][te] == d).sum() and
                          ok[P["ds_of"][te] == d].mean() < P["ok0"][te][P["ds_of"][te] == d].mean() - 1e-9)
                R = res[gate]; R["acc"].append(ok.mean()); R["esc2"].append(e1[te].mean()); R["flops"].append(fl)
                R["lat"].append(lat.mean()); R["latp90"].append(np.percentile(lat, 90)); R["energy"].append(eg.mean()); R["bad"].append(bad)
        print(f"\n################  ACC-v2  [{label}]  config: {TIERS}  ################")
        print(f"  parity (always-32B-think) = {parity:.4f}   (honest 50/50, 20 seeds)")
        print(f"  {'gate':<22}{'acc':>7}{'esc→think':>11}{'FLOPs%':>8}{'lat mean':>10}{'lat p90':>9}{'energy/q':>10}{'guard':>7}")
        for gate, nm in [("margin", "ACC v1 (margin)"), ("agree", "ACC v2 (agreement)")]:
            R = res[gate]; mn = lambda k: np.mean(R[k])
            print(f"  {nm:<22}{mn('acc'):>7.4f}{mn('esc2')*100:>10.0f}%{mn('flops')*100:>7.1f}%{mn('lat'):>9.2f}s"
                  f"{mn('latp90'):>8.1f}s{mn('energy'):>8.0f}J{mn('bad'):>7.2f}")


# ------------------------------------------------------------------ freeze deployable thresholds
def freeze():
    """Calibrate (tau0,tau1) on the held-out PMC-VQA-train split and save for deployment."""
    rows7 = _load_calib(J("ckpts/gate_7b_pmctrain_prune/cap320"))                 # 7B-nt @ cap320
    m32n = {r["idx"]: r for r in (json.loads(l) for l in open(J("ckpts/gate_32b_pmctrain_nothink_cap320/ckpt_nothink.jsonl")) if l.strip())}
    m32t = {r["idx"]: r for r in (json.loads(l) for l in open(J("ckpts/gate_32b_pmctrain/ckpt_think.jsonl")) if l.strip())}
    idx = [r["idx"] for r in rows7 if r["idx"] in m32n and r["idx"] in m32t]
    by = {r["idx"]: r for r in rows7}
    Pc = dict(
        ok0=np.array([by[i]["ok"] for i in idx], float), ok1=np.array([m32n[i]["ok"] for i in idx], float),
        ok2=np.array([m32t[i]["ok"] for i in idx], float),
        m0=np.array([margin(by[i].get("opt_logprobs")) for i in idx]),
        m1=np.array([margin(m32n[i].get("opt_logprobs")) for i in idx]),
        disagree=np.array([1.0 if by[i].get("pred") != m32n[i].get("pred") else 0.0 for i in idx]),
    )
    mask = np.ones(len(idx), bool)
    t0, t1 = calibrate(Pc, mask, "agree")
    e0c = Pc["m0"] < t0; e1c = e0c & (think_score(Pc) > t1)
    out = dict(tier0_gate="7B-nothink margin < tau0", tier1_gate="escalate to think iff disagree (7B-nt != 32B-nt) [tau1 on disagree score]",
               tau0=t0, tau1=t1, calibrated_on="pmc_vqa_train", n=len(idx),
               calib_parity=float(Pc["ok2"].mean()), config=TIERS,
               note=("PMC-train is perception-only, so its parity is reachable WITHOUT the think tier -> "
                     "tau1 here disables think (think_fires~0%). Correct for perception/COMPETENT-4 deployment. "
                     "For reasoning-inclusive (ALL-6) deployment, calibrate tau1 on a mixed set that contains "
                     "reasoning examples (the held-out 50/50 eval calibration fires think ~14% on ALL-6)."),
               calib_tier1_fires=float(e0c.mean()), calib_think_fires=float(e1c.mean()))
    path = J("ckpts/acc_v2_thresholds.json"); json.dump(out, open(path, "w"), indent=2)
    e0 = Pc["m0"] < t0; e1 = e0 & (think_score(Pc) > t1)
    print(f"\nFROZEN thresholds (calibrated on PMC-train, n={len(idx)}) -> {path}")
    print(f"  tau0={t0:.4f} (7B margin)  tau1={t1:.4f} (disagree score)")
    print(f"  on calib: tier1 fires {e0.mean()*100:.0f}%, think fires {e1.mean()*100:.0f}%, "
          f"cascade acc {np.where(~e0,Pc['ok0'],np.where(~e1,Pc['ok1'],Pc['ok2'])).mean():.4f} "
          f"(calib parity {Pc['ok2'].mean():.4f})")


if __name__ == "__main__":
    evaluate()
    try:
        freeze()
    except Exception as e:
        print(f"\n[freeze skipped: {e}]")
