#!/usr/bin/env python3
"""
final_comparison.py - the DEFINITIVE leaderboard. Every current-SOTA training-free cascade gate x
every strong-leg config (32B reasoning-mode x image-resolution), all scored identically:
prefill-inclusive backbone% at think-parity (denominator = always-32B-think@fullres), honest
PMC-train calibration, per-benchmark never-worse-than-7B guardrail, plus the eval-oracle frontier.

Strong-leg configs (the escalation TARGET): think@fullres (deployed/SOTA), nothink@fullres,
think@cap320, nothink@cap320 (ours). Gate signals come from the 7B cheap leg only (training-free).
CPU only; launch from repo root.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (_load_arm, signals_from_logprobs, EVAL_DIR, DIR_32B, CACHE,
                     ALL6, COMPETENT, N7, N32, CascadeData)

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
LEGS = {  # name -> (dir, cell, prefill-cap)
    "think@full(SOTA)": ("ckpts/gate_32b", "think_norag", "fullres"),
    "nothink@full":     ("ckpts/gate_32b_modes/nothink_fullres", "nothink_norag", "fullres"),
    "think@cap320":     ("ckpts/gate_32b_modes/think_cap320", "think_norag", "cap320"),
    "nothink@cap320":   ("ckpts/gate_32b_modes/nothink_cap320", "nothink_norag", "cap320"),
}
GATES = {"margin": ("margin", -1), "MSP/Chow": ("top1prob", -1), "prob_margin": ("prob_margin", -1),
         "entropy": ("entropy", +1), "Gini": ("gini", +1)}


def load():
    cache = json.load(open(J(CACHE)))
    ev = _load_arm(J(EVAL_DIR["cap320"]), "nothink_norag")
    legs = {n: _load_arm(J(d), c) for n, (d, c, _) in LEGS.items()}
    D = {}
    for ds in ALL6:
        if ds not in ev or any(ds not in legs[n] for n in LEGS):
            continue
        cY = cache[ds]["cap320"]
        common = set(ev[ds])
        for n in LEGS:
            common &= set(legs[n][ds])
        idx = sorted(common & {int(k) for k in cY})
        sig = [signals_from_logprobs(ev[ds][i].get("opt_logprobs")) for i in idx]
        rec = dict(ok7=np.array([ev[ds][i]["ok"] for i in idx], float),
                   sig={k: np.array([s[k] for s in sig], float) for k in sig[0]},
                   run7=2*N7*np.array([cY[str(i)][0] + (ev[ds][i].get("gen_tokens") or 0) for i in idx], float))
        for n, (d, c, pcap) in LEGS.items():
            cp = cache[ds][pcap]
            rec[f"ok_{n}"] = np.array([legs[n][ds][i]["ok"] for i in idx], float)
            rec[f"run_{n}"] = 2*N32*np.array([cp[str(i)][0] + (legs[n][ds][i].get("gen_tokens") or 0) for i in idx], float)
        D[ds] = rec
    return D


def pool(D, names):
    names = [d for d in names if d in D]
    out = {}
    for k in D[names[0]]:
        if k == "sig":
            out["sig"] = {s: np.concatenate([D[d]["sig"][s] for d in names]) for s in D[names[0]]["sig"]}
        else:
            out[k] = np.concatenate([D[d][k] for d in names])
    return out


def main():
    D = load()
    Dc = CascadeData("cap320"); err = 1.0 - Dc.calib["ok"].mean()
    for sub, names in [("COMPETENT-4", COMPETENT), ("ALL-6", ALL6)]:
        P = pool(D, names); base = P["run_think@full(SOTA)"].sum(); target = P["ok_think@full(SOTA)"].mean()
        print(f"\n################  FINAL  [{sub}]  n={len(P['ok7'])}  think-parity={target:.4f}  ################")
        print(f"  always-7B={P['ok7'].mean():.4f}   denominator = always-32B-think@fullres (=100%)")
        for leg in LEGS:
            print(f"     always-{leg:<16}= {P['ok_'+leg].mean():.4f}  @ {P['run_'+leg].sum()/base*100:5.1f}% backbone")
        print(f"\n  {'gate':<12}" + "".join(f"{leg:>20}" for leg in LEGS))
        print(f"  {'':12}" + "".join(f"{'honest / oracle':>20}" for _ in LEGS))
        for gname, (key, sign) in GATES.items():
            thr = float(np.quantile(sign * Dc.calib["sig"][key], 1.0 - err))
            row = f"  {gname:<12}"
            for leg in LEGS:
                okS = P["ok_" + leg]; runS = P["run_" + leg]
                esc = sign * P["sig"][key] > thr
                acc = np.where(esc, okS, P["ok7"]).mean()
                bb = (P["run7"].sum() + runS[esc].sum()) / base
                # guardrail
                worse = 0
                for ds in names:
                    e = sign * D[ds]["sig"][key] > thr
                    if np.where(e, D[ds]["ok_" + leg], D[ds]["ok7"]).mean() < D[ds]["ok7"].mean() - 1e-9:
                        worse += 1
                # oracle frontier
                o = np.argsort(-sign * P["sig"][key], kind="stable")
                accs = P["ok7"].mean() + np.concatenate([[0], np.cumsum(okS[o] - P["ok7"][o])]) / len(okS)
                costs = (P["run7"].sum() + np.concatenate([[0], np.cumsum(runS[o])])) / base
                hit = np.where(accs >= target - 1e-9)[0]
                orc = f"{costs[hit.min()]*100:.0f}" if len(hit) else "--"
                flag = "" if (acc >= target - 1e-9 and worse == 0) else ("x" if acc < target - 1e-9 else f"!{worse}")
                row += f"{bb*100:>11.1f}{flag:<2}/{orc:>4}"
            print(row)
        print(f"  (cell = honest backbone% [x=miss parity, !k=guardrail fail on k] / eval-oracle backbone%)")
    print("\nHeadline: best SOTA = prob_margin/margin -> think@full; best OURS = MSP or margin -> nothink@cap320.")


if __name__ == "__main__":
    main()
