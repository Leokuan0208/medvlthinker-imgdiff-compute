#!/usr/bin/env python3
"""
sota_comparison.py - head-to-head of CURRENT SOTA training-free cascade gates against our
mode-adaptive cascade, all measured identically: prefill-inclusive backbone% at think-parity
(denominator = always-32B-think@fullres), with the per-benchmark never-worse-than-7B guardrail,
honest PMC-train calibration, plus the eval-oracle frontier (gate-quality ceiling).

SOTA gate families implemented (all training-free, calibration-only):
  margin (FrugalGPT-style confidence cascade) | MSP/Chow's rule | predictive entropy |
  Gini/DOCTOR D_alpha | CP-Router (conformal APS prediction-set size) | post-hoc deferral
  (recoverability, learning-to-defer; needs 32B-on-calib). Temperature-scaled "calibrated
  confidence" (2026) is monotone under quantile thresholding => identical decision to its base
  signal, so it is reported as a note, not a separate row.

Each gate is run on THREE strong-leg configs:
  think    : 7B-nothink-cap320 -> 32B-THINK@fullres        (the standard 2-tier cascade SOTA uses)
  nothink  : 7B-nothink-cap320 -> 32B-NO-THINK@fullres     (ours: cheaper+better strong leg)
  3tier    : 7B -> 32B-nothink -> 32B-think                (ours: think only on residual; gate=margin)
CPU only; launch from repo root.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (_load_arm, signals_from_logprobs, EVAL_DIR, DIR_32B, CACHE,
                     ALL6, COMPETENT, N7, N32, CascadeData)

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
NOTHINK_DIR = "ckpts/gate_32b_modes/nothink_fullres"


def load_all(cheap_cap="cap320"):
    cache = json.load(open(J(CACHE)))
    ev = _load_arm(J(EVAL_DIR[cheap_cap]), "nothink_norag")
    think = _load_arm(J(DIR_32B), "think_norag")
    noth = _load_arm(J(NOTHINK_DIR), "nothink_norag")
    D = {}
    for ds in ALL6:
        if ds not in ev or ds not in think or ds not in noth:
            continue
        cY = cache[ds][cheap_cap]; cF = cache[ds]["fullres"]
        idx = sorted(set(ev[ds]) & set(think[ds]) & set(noth[ds])
                     & {int(k) for k in cY} & {int(k) for k in cF})
        sig = [signals_from_logprobs(ev[ds][i].get("opt_logprobs")) for i in idx]
        D[ds] = dict(
            ok7=np.array([ev[ds][i]["ok"] for i in idx], float),
            okN=np.array([noth[ds][i]["ok"] for i in idx], float),
            okT=np.array([think[ds][i]["ok"] for i in idx], float),
            mN=np.array([signals_from_logprobs(noth[ds][i].get("opt_logprobs"))["margin"] for i in idx], float),
            sig={k: np.array([s[k] for s in sig], float) for k in sig[0]},
            run7=2*N7*np.array([cY[str(i)][0] + (ev[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runN=2*N32*np.array([cF[str(i)][0] + (noth[ds][i].get("gen_tokens") or 0) for i in idx], float),
            runT=2*N32*np.array([cF[str(i)][0] + (think[ds][i].get("gen_tokens") or 0) for i in idx], float),
        )
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in D[names[0]] if k != "sig"}
    out["sig"] = {k: np.concatenate([D[d]["sig"][k] for d in names]) for k in D[names[0]]["sig"]}
    return out


# --- SOTA gate scores (higher = escalate). Calib uses 7B-PMC-train signals. ---
def gate_scores(sig):
    return {
        "margin(FrugalGPT)": -sig["margin"],
        "MSP/Chow":          -sig["top1prob"],
        "entropy":           +sig["entropy"],
        "Gini/DOCTOR":       +sig["gini"],
        "prob_margin":       -sig["prob_margin"],
    }


def min_bb_at_parity(run7_sum, runS, ok7, okS, score, base, target):
    """sweep threshold (escalate highest score first); smallest-escalation backbone reaching parity."""
    o = np.argsort(-score, kind="stable")
    a7o, aSo, rSo = ok7[o], okS[o], runS[o]
    acc = ok7.mean() + np.concatenate([[0], np.cumsum(aSo - a7o)]) / len(ok7)
    cost = (run7_sum + np.concatenate([[0], np.cumsum(rSo)])) / base
    hit = np.where(acc >= target - 1e-9)[0]
    if not len(hit):
        return None
    k = hit.min()
    return dict(bb=float(cost[k]), esc=float(k / len(ok7)), acc=float(acc[k]))


def honest_point(D, names, score_key, sign, leg, base_target):
    """PMC-train err-rate threshold on the chosen signal, applied to eval; backbone + guardrail."""
    Dc = CascadeData("cap320")
    err = 1.0 - Dc.calib["ok"].mean()
    cscore = sign * Dc.calib["sig"][score_key]
    thr = float(np.quantile(cscore, 1.0 - err))
    P = pool(D, names); base = P["runT"].sum()
    okS = P["okN"] if leg == "nothink" else P["okT"]
    runS = P["runN"] if leg == "nothink" else P["runT"]
    esc = sign * P["sig"][score_key] > thr
    acc = np.where(esc, okS, P["ok7"]).mean()
    bb = (P["run7"].sum() + runS[esc].sum()) / base
    worse = []
    for ds in names:
        d = D[ds]; e = sign * d["sig"][score_key] > thr
        oS = d["okN"] if leg == "nothink" else d["okT"]
        if np.where(e, oS, d["ok7"]).mean() < d["ok7"].mean() - 1e-9:
            worse.append(ds)
    return dict(acc=float(acc), esc=float(esc.mean()), bb=float(bb), worse=worse)


SIGN = {"margin(FrugalGPT)": ("margin", -1), "MSP/Chow": ("top1prob", -1), "entropy": ("entropy", +1),
        "Gini/DOCTOR": ("gini", +1), "prob_margin": ("prob_margin", -1)}


def three_tier_frontier(P, base, target, n1=70, n2=45):
    q7 = np.quantile(P["sig"]["margin"], np.linspace(0, 1, n1))
    qN = np.quantile(P["mN"], np.linspace(0, 1, n2))
    best = None
    for t1 in q7:
        e1 = P["sig"]["margin"] < t1
        for t2 in qN:
            e2 = e1 & (P["mN"] < t2)
            acc = np.where(~e1, P["ok7"], np.where(~e2, P["okN"], P["okT"])).mean()
            if acc >= target - 1e-9:
                bb = (P["run7"].sum() + P["runN"][e1].sum() + P["runT"][e2].sum()) / base
                if best is None or bb < best["bb"]:
                    best = dict(bb=float(bb), acc=float(acc), esc1=float(e1.mean()), esc2=float(e2.mean()))
    return best


def main():
    D = load_all()
    for sub, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = pool(D, names); base = P["runT"].sum(); target = P["okT"].mean()
        print(f"\n################  SOTA COMPARISON  [{sub}]  n={len(P['ok7'])}  "
              f"think-parity={target:.4f}  ################")
        print(f"  baselines: always-7B={P['ok7'].mean():.4f}  always-32B-nothink={P['okN'].mean():.4f}  "
              f"always-32B-think={target:.4f} (=100% backbone)")
        print(f"\n  {'GATE METHOD':<20}{'STRONG=think':>22}{'   '}{'STRONG=nothink(ours)':>26}")
        print(f"  {'(SOTA gates)':<20}{'oracle@par':>12}{'honest':>10}{'   '}{'oracle@par':>12}{'honest':>14}")
        for gname, (key, sign) in SIGN.items():
            row = f"  {gname:<20}"
            for leg, okS, runS in [("think", P["okT"], P["runT"]), ("nothink", P["okN"], P["runN"])]:
                fr = min_bb_at_parity(P["run7"].sum(), runS, P["ok7"], okS, sign * P["sig"][key], base, target)
                hp = honest_point(D, names, key, sign, leg, target)
                gflag = "" if not hp["worse"] else f"!{len(hp['worse'])}"
                obb = f"{fr['bb']*100:>10.1f}%" if fr else f"{'never':>11}"
                hbb = f"{hp['bb']*100:.1f}%{('p' if hp['acc']>=target-1e-9 else 'x')}{gflag}"
                row += f"{obb}{'':1}{hbb:>10}" if leg == "think" else f"   {obb}{'':1}{hbb:>13}"
            print(row)
        # 3-tier (ours): gate = margin at both tiers
        ft = three_tier_frontier(P, base, target)
        if ft:
            print(f"\n  {'3-TIER (ours)':<20}-> oracle@parity bb={ft['bb']*100:.1f}%  "
                  f"esc1={ft['esc1']*100:.0f}% esc2(think)={ft['esc2']*100:.0f}%")
        # CP-Router note (CORRECTED after audit -> results/cascade_methods/baseline_audit.json):
        # The earlier "set-size monotone in top1prob => == MSP/Chow" claim was WRONG: our eval has 5-10
        # options (not <=5), and LAC set-size equals MSP only when 1-q_hat>=0.5 (rules diverge up to ~18%
        # at the alphas CP-Router targets). Faithful CP-Router (LAC set-size + FBE alpha*) lives in
        # src/cascade_methods/baseline_compare.py and OVER-escalates (69-80%), losing to confidence here.
        print(f"\n  CP-Router (conformal): see baseline_compare.py for the FAITHFUL set-size+FBE row")
        print(f"    (it over-escalates 69-80%; the old 'monotone == MSP' note was wrong and is removed).")
    print("\nLegend: oracle@par = min backbone%% at parity (eval-swept threshold, gate ceiling); honest =")
    print("PMC-train err-rate threshold (p=hit parity, x=miss, !k=fails guardrail on k benchmarks).")
    print("Headline: compare the best SOTA gate on STRONG=think vs the SAME gates on STRONG=nothink (ours).")


if __name__ == "__main__":
    main()
