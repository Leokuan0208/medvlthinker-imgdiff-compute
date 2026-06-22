#!/usr/bin/env python3
"""
baseline_compare.py - CORRECTED training-free gate bake-off (post-audit, results/cascade_methods/
baseline_audit.json). Canonical 2-tier cascade 7B-nothink@cap320 -> 32B-think@fullres; each gate
escalates the 7B's answer. Honest 50/50 calib/test split (threshold/alpha chosen on calib to reach
calib-parity at MIN escalation; metrics on test), 20 seeds. Reports escalation% + FLOPs% at iso-accuracy.

Fixes from the audit:
  - CP-Router/conformal: FAITHFUL now -> LAC nonconformity s=1-softmax(gold), finite-sample q_hat,
    real prediction set C={p>=1-q_hat}, escalate iff |C|!=1, with FBE alpha* selection (label-free,
    argmax of full+binary set-size entropy). NOT the old "1-top1prob (approx MSP)" collapse.
  - AutoMix: standalone row -> escalate the 7B when its SELF-VERIFICATION confidence p_yes_norm is low
    (the meta-verifier threshold variant). NB: our self-verify is ZERO-SHOT single-pass (faithful
    AutoMix is few-shot); labeled accordingly.
  - DOCTOR(Gini), MSP/Chow, entropy: faithful as-is (audit confirmed; DOCTOR D_alpha monotone-equiv to Gini).
  - learned-correct = FrugalGPT-STYLE learned scorer (logprob-feature variant; not text scorer).
  - learned-defer = Jitkrittum Diff-Prob target (P(32B right)-P(7B right)), calib-fit.
Run from repo root:  python3 src/cascade_methods/baseline_compare.py
"""
import sys, os, glob, json, re; sys.path.insert(0, "src/cascade_methods")
import numpy as np
from collections import defaultdict
from harness import signals_from_logprobs, ALL6, ALL5, COMPETENT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
N7, N32 = 7.6e9, 33.0e9
J = lambda p: os.path.join("/home/jamesyang/medvlthinker-imgdiff-compute", p)
def loadarm(d, tag):
    out = defaultdict(dict)
    for f in glob.glob(J(os.path.join(d, f"ckpt_*{tag}*.jsonl"))):
        m = re.match(rf"ckpt_(.+?)_{re.escape(tag.split('_')[0])}", os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); out[m.group(1)][r["idx"]] = r
    return out

def softmax_probs(lp):
    if not lp: return np.array([1.0])
    v = np.array(sorted(lp.values(), reverse=True)); e = np.exp(v - v.max()); return e / e.sum()
def lac_gold_score(lp, g):  # nonconformity = 1 - softmax(gold)
    if not lp or g not in lp: return None
    v = np.array(list(lp.values())); e = np.exp(v - v.max()); p = e / e.sum()
    return 1.0 - float(p[list(lp.keys()).index(g)])
def setsize_at(lp, thr):  # |C| = #options with softmax prob >= thr (=1-q_hat)
    return int((softmax_probs(lp) >= thr).sum())

c7 = loadarm("ckpts/gate_7b_prune/cap320", "nothink_norag")
c32t = loadarm("ckpts/gate_32b", "think_norag")
ver = loadarm("ckpts/gate_7b_verify", "verify")
cache = json.load(open(J("ckpts/token_cache.json")))

def build(names):
    rows = []
    for ds in names:
        cC = cache.get(ds, {}).get("cap320", {}); cF = cache.get(ds, {}).get("fullres", {})
        idx = sorted(set(c7.get(ds, {})) & set(c32t.get(ds, {})) & {int(k) for k in cC} & {int(k) for k in cF})
        for i in idx:
            r = c7[ds][i]; s = signals_from_logprobs(r.get("opt_logprobs"))
            vrow = ver.get(ds, {}).get(i, {})
            rows.append(dict(ds=ds, ok7=r["ok"], ok32=c32t[ds][i]["ok"], lp=r.get("opt_logprobs") or {}, gold=r.get("gold"),
                margin=s["margin"], top1=s["top1prob"], ent=s["entropy"], gini=s["gini"], pm=s["prob_margin"],
                verify=vrow.get("p_yes_norm", 0.5) if vrow else 0.5,
                Pc=cC[str(i)][0], Pf=cF[str(i)][0], g2=c32t[ds][i].get("gen_tokens") or 0))
    return rows

def two_tier(esc, ok7, ok32):  # final answer + escalation rate
    return np.where(esc, ok32, ok7).mean(), esc.mean()

def fbe(setsizes):  # Full+Binary set-size entropy (label-free); higher = better easy/hard separability
    ss = np.asarray(setsizes);
    if len(ss) == 0: return -1
    vals, cnt = np.unique(ss, return_counts=True); p = cnt / cnt.sum(); Hfull = -(p * np.log(p + 1e-12)).sum()
    q = (ss == 1).mean(); Hbin = -(q * np.log(q + 1e-12) + (1 - q) * np.log(1 - q + 1e-12))
    return Hfull + Hbin

ALPHAS = np.round(np.arange(0.02, 0.51, 0.02), 3)
def main():
    rng_master = np.random.default_rng(0)
    for label, names in [("ALL-6", ALL6), ("ALL-5 (excl MedXpert)", ALL5), ("COMPETENT-4", COMPETENT)]:
        R = build(names); n = len(R)
        ok7 = np.array([r["ok7"] for r in R]); ok32 = np.array([r["ok32"] for r in R])
        parity = ok32.mean()
        feats = lambda keys: np.column_stack([[r[k] for r in R] for k in keys])
        Xconf = feats(["margin", "top1", "ent", "gini", "pm"])
        f0 = 2 * N7 * (np.array([r["Pc"] for r in R]) + 2); f2 = 2 * N32 * (np.array([r["Pf"] for r in R]) + np.array([r["g2"] for r in R]))
        F2tot = f2.sum()
        # precompute conformal calib scores + per-row softmax for set-size
        lac = np.array([(lambda s: s if s is not None else np.nan)(lac_gold_score(r["lp"], r["gold"])) for r in R])
        print(f"\n################  CORRECTED 2-TIER GATE BAKE-OFF  [{label}]  (20 seeds)  ################")
        print(f"  7B-nt@cap320 -> 32B-think@fullres ; always-7B={ok7.mean():.3f}  parity(always-32B-think)={parity:.3f}")
        print(f"  {'gate':<34}{'test-acc':>9}{'esc%':>7}{'FLOPs%':>8}")
        SCORES = {  # gate -> escalation score (higher = escalate); thresholded on calib at parity
            "margin (=Chow/margin-samp, ACC)": -np.array([r["margin"] for r in R]),
            "MSP/Chow": -np.array([r["top1"] for r in R]),
            "entropy": np.array([r["ent"] for r in R]),
            "Gini/DOCTOR (D_alpha)": np.array([r["gini"] for r in R]),
            "AutoMix (0-shot self-verify thr)": -np.array([r["verify"] for r in R]),
        }
        seeds = range(20)
        def eval_score(score, learned=None):
            accs, escs, fl = [], [], []
            for sd in seeds:
                rg = np.random.default_rng(sd); cal = np.zeros(n, bool); cal[rg.choice(n, n // 2, replace=False)] = True; te = ~cal
                sc = learned(cal) if learned else score
                # threshold on calib: min escalation s.t. calib-acc >= calib-parity
                pcal = ok32[cal].mean(); order = np.argsort(-sc[cal]); best_t = sc[cal].max() + 1
                # sweep thresholds (escalate where sc>=t); find smallest escalation reaching parity
                cand = np.unique(sc[cal])[::-1]; chosen = cand[0] + 1
                for t in cand:
                    e = sc[cal] >= t; a = np.where(e, ok32[cal], ok7[cal]).mean()
                    if a >= pcal - 1e-9: chosen = t; break
                e_te = sc[te] >= chosen
                accs.append(np.where(e_te, ok32[te], ok7[te]).mean()); escs.append(e_te.mean())
                fl.append((f0[te].sum() + f2[te][e_te].sum()) / f2[te].sum())
            return np.mean(accs), np.mean(escs), np.mean(fl) * 100
        for name, sc in SCORES.items():
            a, e, fp = eval_score(sc); print(f"  {name:<34}{a:>9.3f}{e*100:>6.0f}%{fp:>7.1f}%")
        # learned-correct (FrugalGPT-style) and learned-defer (Jitkrittum) — calib-fit
        def lc(cal): return -make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xconf[cal], ok7[cal]).predict_proba(Xconf)[:, 1]
        def ld(cal):
            pc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xconf[cal], ok7[cal]).predict_proba(Xconf)[:, 1]
            pn = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xconf[cal], ok32[cal]).predict_proba(Xconf)[:, 1]
            return pn - pc
        a, e, fp = eval_score(None, lc); print(f"  {'FrugalGPT-style learned scorer':<34}{a:>9.3f}{e*100:>6.0f}%{fp:>7.1f}%")
        a, e, fp = eval_score(None, ld); print(f"  {'Jitkrittum L2D (Diff-Prob defer)':<34}{a:>9.3f}{e*100:>6.0f}%{fp:>7.1f}%")
        # CP-Router faithful: set-size + FBE alpha* (label-free), 20-seed calib/test
        accs, escs, fl, alstars, escFBE = [], [], [], [], []
        valid = ~np.isnan(lac)
        for sd in seeds:
            rg = np.random.default_rng(100 + sd); cal = np.zeros(n, bool); cal[rg.choice(n, n // 2, replace=False)] = True; te = ~cal
            cal_lac = lac[cal & valid]
            # FBE alpha* on calib (label-free set-sizes)
            best_a, best_fbe = ALPHAS[0], -1
            for al in ALPHAS:
                k = int(np.ceil((len(cal_lac) + 1) * (1 - al))); k = min(max(k, 1), len(cal_lac))
                qh = np.sort(cal_lac)[k - 1]; thr = 1 - qh
                ss = [setsize_at(R[i]["lp"], thr) for i in np.where(cal)[0]]
                fb = fbe(ss)
                if fb > best_fbe: best_fbe, best_a = fb, al
            k = int(np.ceil((len(cal_lac) + 1) * (1 - best_a))); k = min(max(k, 1), len(cal_lac))
            thr = 1 - np.sort(cal_lac)[k - 1]
            e_te = np.array([setsize_at(R[i]["lp"], thr) != 1 for i in np.where(te)[0]])
            accs.append(np.where(e_te, ok32[te], ok7[te]).mean()); escs.append(e_te.mean()); alstars.append(best_a)
            fl.append((f0[te].sum() + f2[te][e_te].sum()) / f2[te].sum())
        print(f"  {'CP-Router (LAC set-size + FBE)':<34}{np.mean(accs):>9.3f}{np.mean(escs)*100:>6.0f}%{np.mean(fl)*100:>7.1f}%   [FBE alpha*~{np.mean(alstars):.2f}]")
        print(f"  (CP-Router faithful: real prediction set |C|!=1, NOT the old 1-top1prob/MSP collapse)")
if __name__ == "__main__": main()
