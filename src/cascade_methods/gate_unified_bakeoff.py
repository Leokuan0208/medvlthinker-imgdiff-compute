#!/usr/bin/env python3
"""
gate_unified_bakeoff.py - OFFLINE, CPU-only clean gate bake-off for the cheap->strong medical-VQA
cascade, in BOTH settings, plus our NEW gate (EG-RC: Expected-Gain via Recovery-Calibration).

Reads ONLY existing checkpoints (no GPU / no new inference). Launch from repo root:
    python3 src/cascade_methods/gate_unified_bakeoff.py

------------------------------------------------------------------------------------------------
Settings
  (A) MCQ    : cheap MedVLThinker-7B-nt@cap320  ->  strong 32B-think.  (competent-4 + ALL-5)
               cheap signals per sample from the 7B per-letter logprobs + a 7B SELF-VERIFY head
               (p_yes_norm) + a free resolution-perturbation (cap80/160/320/640) stability signal.
  (B) OPEN   : cheap Lingshu-7B best-of-8 + trained verifier PICK  ->  strong Lingshu-32B.
               gate signals from the verifier's 8 answer-scores + cheap self-consistency.

Every gate is a per-sample KEEP score (higher = keep with the cheap answer; escalate the LOWEST).
Trained gates use 5-fold OUT-OF-FOLD scores on the eval pool (no leakage).

Metrics (all threshold choices made explicit):
  * AUROC_detect     : AUROC( keep-score , cheap_ok )      -- selective-prediction / error detection.
  * AUROC_recover    : AUROC( -keep-score , strong_ok | cheap_WRONG ) -- Jitkrittum recoverability "wall".
  * ADC              : area under the deferral curve (cascade acc vs escalation budget in [0,1]);
                       threshold-FREE; higher = better; oracle & random reported for scale.
  * routing_eff      : (ADC_gate - ADC_random) / (ADC_oracle - ADC_random)  in [0,1] (APGR-style).
  * min_esc_parity   : smallest escalation budget whose cascade acc >= always-strong acc
                       (frontier / oracle-on-eval; flagged as optimistic).

NEW GATE  EG-RC  (deliverable 4): the Bayes-optimal deferral score is the expected accuracy GAIN
  g(x)=P(strong right|x)-P(cheap right|x) (Jitkrittum NeurIPS'23, arXiv:2307.02764). Learning g
  directly is the "recoverability wall" (unlearnable, AUROC~0.6, overfits). EG-RC instead DECOMPOSES
  it into a LEARNABLE detector times a LOW-VARIANCE calibrated recovery prior:
      g(x) ~=  P_cal(cheap wrong|x) * r_hi(bin)  -  P_cal(cheap right|x) * (1 - r_lo(bin))
  where P_cal = isotonic/logit-calibrated cheap-confidence detector (the learnable part), and
  r_hi(bin)=P(strong right | cheap wrong, bin), r_lo(bin)=P(strong right | cheap right, bin) are
  estimated per confidence-decile on the OTHER folds (the recovery prior; low variance, not point-wise).
  It reduces to the plain confidence gate exactly when recovery is bin-independent -> explains WHY
  confidence is near-optimal, and can only win where recovery varies with a cheap-observable stratum.
"""
import os, sys, json, glob, re
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import signals_from_logprobs, _load_arm, ALL6, ALL5, COMPETENT
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
RNG = np.random.default_rng(0)

# ----------------------------------------------------------------------------- metric helpers
def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    P = score[y == 1]; N = score[y == 0]
    if len(P) == 0 or len(N) == 0: return float("nan")
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    u, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss / c)[inv]
    return (rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N))

def deferral_curve(keep, cheap_ok, strong_ok, grid=41):
    """escalate the LOWEST keep-score first. returns (B, acc_gate, acc_oracle, acc_random)."""
    keep = np.asarray(keep, float); co = np.asarray(cheap_ok, float); so = np.asarray(strong_ok, float)
    n = len(co); B = np.linspace(0, 1, grid)
    order = np.argsort(keep)                     # ascending: least-confident first
    gain = so - co
    oracle_order = np.argsort(gain)[::-1]        # highest escalation-benefit first
    at = lambda mask: float(np.where(mask, so, co).mean())
    ag, ao, ar = [], [], []
    for b in B:
        k = int(round(b * n))
        m = np.zeros(n, bool); m[order[:k]] = True; ag.append(at(m))
        m2 = np.zeros(n, bool); m2[oracle_order[:k]] = True; ao.append(at(m2))
        rr = []
        for _ in range(20):
            m3 = np.zeros(n, bool); m3[RNG.permutation(n)[:k]] = True; rr.append(at(m3))
        ar.append(float(np.mean(rr)))
    return B, np.array(ag), np.array(ao), np.array(ar)

def _adc_of(keep, co, so, B):
    n = len(co); order = np.argsort(keep)
    acc = []
    for b in B:
        k = int(round(b * n)); m = np.zeros(n, bool); m[order[:k]] = True
        acc.append(float(np.where(m, so, co).mean()))
    return float(np.trapz(acc, B))

def bootstrap_adc_diff(keep_a, keep_b, cheap_ok, strong_ok, nboot=500, grid=21, seed=7):
    """paired bootstrap of ADC(a)-ADC(b) over resampled samples. a=challenger, b=deployed."""
    co = np.asarray(cheap_ok, float); so = np.asarray(strong_ok, float)
    ka = np.asarray(keep_a, float); kb = np.asarray(keep_b, float)
    n = len(co); B = np.linspace(0, 1, grid); rng = np.random.default_rng(seed); diffs = []
    for _ in range(nboot):
        ix = rng.integers(0, n, n)
        diffs.append(_adc_of(ka[ix], co[ix], so[ix], B) - _adc_of(kb[ix], co[ix], so[ix], B))
    diffs = np.array(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs > 0).mean())

def gate_metrics(keep, cheap_ok, strong_ok):
    co = np.asarray(cheap_ok, float); so = np.asarray(strong_ok, float)
    B, ag, ao, ar = deferral_curve(keep, co, so)
    adc = float(np.trapz(ag, B)); adc_o = float(np.trapz(ao, B)); adc_r = float(np.trapz(ar, B))
    eff = (adc - adc_r) / (adc_o - adc_r) if adc_o - adc_r > 1e-9 else float("nan")
    au_det = auroc(keep, co)                                   # high keep -> cheap right
    wrong = co == 0
    au_rec = auroc(-np.asarray(keep, float)[wrong], so[wrong]) if wrong.sum() > 5 and len(np.unique(so[wrong])) > 1 else float("nan")
    strong_acc = float(so.mean())
    hit = np.where(ag >= strong_acc - 1e-9)[0]
    min_esc = float(B[hit[0]]) if len(hit) else float("nan")   # nan => never matches strong below 100%
    return dict(auroc_detect=au_det, auroc_recover=au_rec, adc=adc, adc_oracle=adc_o,
                adc_random=adc_r, routing_eff=eff, min_esc_parity=min_esc,
                cheap=float(co.mean()), strong=strong_acc)

def oof(Xa, y, kind="logit", target=None):
    """5-fold out-of-fold P(target=1). target defaults to y. Returns prob vector aligned to rows."""
    t = y if target is None else np.asarray(target, int)
    n = len(y); out = np.zeros(n); skf = StratifiedKFold(5, shuffle=True, random_state=0)
    strat = y if len(np.unique(y)) > 1 else np.zeros(n, int)
    for tr, te in skf.split(Xa, strat):
        if kind == "gbm":
            clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        yt = t[tr]
        if len(np.unique(yt)) < 2:                # degenerate fold
            out[te] = yt.mean(); continue
        clf.fit(Xa[tr], yt); out[te] = clf.predict_proba(Xa[te])[:, 1]
    return out

def eg_rc_gate(conf_cal, cheap_ok, strong_ok, nbin=10, folds=5):
    """OUR NEW GATE. conf_cal = OOF calibrated P(cheap right|x) (learnable detector).
    Estimate per-decile recovery priors on OTHER folds, form expected-gain deferral score.
    Return a KEEP score (higher = keep). CPU, no leakage (priors cross-fit by bin)."""
    conf_cal = np.asarray(conf_cal, float); co = np.asarray(cheap_ok, float); so = np.asarray(strong_ok, float)
    n = len(co); esc_score = np.zeros(n)
    edges = np.quantile(conf_cal, np.linspace(0, 1, nbin + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
    binid = np.clip(np.digitize(conf_cal, edges) - 1, 0, nbin - 1)
    skf = StratifiedKFold(folds, shuffle=True, random_state=1)
    strat = co if len(np.unique(co)) > 1 else np.zeros(n, int)
    for tr, te in skf.split(conf_cal.reshape(-1, 1), strat):
        # recovery priors estimated on the TRAIN fold, per bin
        r_hi = np.full(nbin, np.nan); r_lo = np.full(nbin, np.nan)
        for b in range(nbin):
            m = (binid[tr] == b)
            wr = m & (co[tr] == 0); rt = m & (co[tr] == 1)
            r_hi[b] = so[tr][wr].mean() if wr.sum() >= 5 else np.nan
            r_lo[b] = so[tr][rt].mean() if rt.sum() >= 5 else np.nan
        # fallbacks: global rates
        gwr = so[tr][co[tr] == 0].mean() if (co[tr] == 0).any() else 0.0
        grt = so[tr][co[tr] == 1].mean() if (co[tr] == 1).any() else 1.0
        r_hi = np.where(np.isnan(r_hi), gwr, r_hi); r_lo = np.where(np.isnan(r_lo), grt, r_lo)
        p_wrong = 1.0 - conf_cal[te]; p_right = conf_cal[te]; bt = binid[te]
        # expected accuracy gain of escalating this sample
        esc_score[te] = p_wrong * r_hi[bt] - p_right * (1.0 - r_lo[bt])
    return -esc_score            # keep-score: high => low escalation benefit => keep


# ============================================================ (A) MCQ setting
def load_mcq():
    cache = json.load(open(J("ckpts/token_cache.json")))
    ev7 = _load_arm(J("ckpts/gate_7b_prune/cap320"), "nothink_norag")
    evT = _load_arm(J("ckpts/gate_32b"), "think_norag")
    ver = defaultdict(dict)
    for f in glob.glob(J("ckpts/gate_7b_verify/ckpt_*_verify.jsonl")):
        m = re.match(r"ckpt_(.+?)_verify", os.path.basename(f))
        for l in open(f):
            if l.strip(): r = json.loads(l); ver[m.group(1)][r["idx"]] = r
    caps = ["cap80", "cap160", "cap320", "cap640"]
    evc = {c: _load_arm(J(f"ckpts/gate_7b_prune/{c}"), "nothink_norag") for c in caps}
    D = {}
    for ds in ALL6:
        if ds not in ev7 or ds not in evT: continue
        idx = sorted(set(ev7[ds]) & set(evT[ds]))
        sig = [signals_from_logprobs(ev7[ds][i].get("opt_logprobs")) for i in idx]
        # resolution-perturbation stability: frac of other caps whose pred != cap320 pred
        base = {i: ev7[ds][i]["pred"] for i in idx}
        capdis = []
        for i in idx:
            ps = [evc[c][ds][i]["pred"] for c in caps if ds in evc[c] and i in evc[c][ds]]
            capdis.append(np.mean([p != base[i] for p in ps]) if ps else 0.0)
        D[ds] = dict(
            cheap_ok=np.array([ev7[ds][i]["ok"] for i in idx], float),
            strong_ok=np.array([evT[ds][i]["ok"] for i in idx], float),
            self_verify=np.array([(ver.get(ds, {}).get(i, {}) or {}).get("p_yes_norm", 0.5) for i in idx], float),
            cap_stable=1.0 - np.array(capdis, float),
            sig={k: np.array([s[k] for s in sig], float) for k in sig[0]},
        )
    return D

def pool_mcq(D, names):
    names = [d for d in names if d in D]
    out = {k: np.concatenate([D[d][k] for d in names]) for k in ("cheap_ok", "strong_ok", "self_verify", "cap_stable")}
    out["sig"] = {k: np.concatenate([D[d]["sig"][k] for d in names]) for k in D[names[0]]["sig"]}
    return out

def mcq_gates(P):
    s = P["sig"]; co = P["cheap_ok"]; so = P["strong_ok"]
    FEAT = ["margin", "maxlogprob", "top1prob", "prob_margin", "entropy", "entropy2", "gini", "neg_energy", "n_opts"]
    Xcheap = np.column_stack([s[f] for f in FEAT])
    Xrich = np.column_stack([s[f] for f in FEAT] + [P["self_verify"], P["cap_stable"]])
    gates = {}
    gates["margin (DEPLOYED)"] = s["margin"]
    gates["maxprob (MSP/Chow)"] = s["top1prob"]
    gates["neg-entropy"] = -s["entropy"]
    gates["neg-gini (DOCTOR)"] = -s["gini"]
    gates["cum_logprob (-energy)"] = -s["neg_energy"]
    gates["resolution-stability"] = P["cap_stable"]
    gates["7B self-verify (AutoMix)"] = P["self_verify"]
    gates["learned-logit (logprob feats)"] = oof(Xcheap, co, "logit")
    gates["learned-gbm (logprob feats)"] = oof(Xcheap, co, "gbm")
    gates["Jitkrittum Diff-Prob"] = -(oof(Xcheap, co, "logit", target=so) - oof(Xcheap, co, "logit"))  # -(P(strong)-P(cheap))
    gates["CASP-stability (trained)"] = oof(Xcheap, co, "logit", target=(co == so).astype(int))          # P(pred7==pred32) proxy
    # ---- OUR NEW GATE ----
    conf_cal = oof(Xrich, co, "logit")               # calibrated detector on the RICH cheap features
    gates["EG-RC (ours)"] = eg_rc_gate(conf_cal, co, so)
    conf_sv = oof(P["self_verify"].reshape(-1, 1), co, "logit")   # calibrate the self-verify head
    gates["EG-RC/self-verify (ours)"] = eg_rc_gate(conf_sv, co, so)
    gates["learned-RICH (ours, fused)"] = conf_cal    # ablation: richer feature detector, no recovery prior
    gates["random"] = RNG.standard_normal(len(co))
    return gates


# ============================================================ (B) OPEN-TEXT setting
def load_open(datasets):
    root = "ckpts/train/lora_verifier_pooled4"
    cheapd = "ckpts/openvqa/cheap_lingshu7b"; strongd = "ckpts/openvqa/strong_lingshu"
    rows = []
    for ds in datasets:
        dp = J(os.path.join(root, f"transfer_dump_{ds}_open_lingshu7b.json"))
        if not os.path.exists(dp): continue
        dump = json.load(open(dp))
        sc = {}
        scp = J(os.path.join(cheapd, f"ckpt_{ds}_open_lingshu7b_sc8.jsonl"))
        if os.path.exists(scp):
            for l in open(scp):
                if l.strip(): r = json.loads(l); sc[r["idx"]] = r
        sj = {}
        for cand in (f"ckpt_{ds}_open_lingshu32b.judge.jsonl", f"ckpt_{ds}_open_lingshu32b_t0.judge.jsonl"):
            p = J(os.path.join(strongd, cand))
            if os.path.exists(p):
                for l in open(p):
                    if l.strip(): r = json.loads(l); sj[r["idx"]] = int(r["judge_ok"])
                break
        for r in dump:
            i = r["idx"]
            if i not in sj: continue
            sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
            N = 8; scores = r["scores"][:N]; a = np.array(scores, float); ssort = sorted(scores, reverse=True)
            pick = int(np.argmax(scores))
            rows.append(dict(ds=ds, pick_ok=sl[pick], strong_ok=sj[i],
                             v_max=ssort[0], v_margin=ssort[0] - (ssort[1] if len(ssort) > 1 else 0.0),
                             v_mean=float(a.mean()), v_std=float(a.std()), v_min=float(a.min()),
                             self_consistency=(sc.get(i, {}).get("self_consistency") or np.nan),
                             n_distinct_neg=-(sc.get(i, {}).get("n_distinct") or np.nan)))
    return rows

def open_gates(rows):
    co = np.array([r["pick_ok"] for r in rows], float); so = np.array([r["strong_ok"] for r in rows], float)
    col = lambda k: np.nan_to_num(np.array([r[k] for r in rows], float))
    Xverif = np.column_stack([col(k) for k in ("v_max", "v_margin", "v_mean", "v_std", "v_min")])
    Xall = np.column_stack([col(k) for k in ("v_max", "v_margin", "v_mean", "v_std", "v_min",
                                             "self_consistency", "n_distinct_neg")])
    gates = {}
    gates["verifier-conf (DEPLOYED)"] = col("v_max")
    gates["verifier-margin"] = col("v_margin")
    gates["verifier-mean"] = col("v_mean")
    gates["self-consistency"] = col("self_consistency")
    gates["-n_distinct"] = col("n_distinct_neg")
    gates["learned-logit (verif+cheap)"] = oof(Xall, co, "logit")
    gates["learned-gbm (verif+cheap)"] = oof(Xall, co, "gbm")
    gates["Jitkrittum Diff-Prob"] = -(oof(Xall, co, "logit", target=so) - oof(Xall, co, "logit"))
    conf_cal = oof(Xall, co, "logit")
    gates["EG-RC (ours)"] = eg_rc_gate(conf_cal, co, so)
    gates["random"] = RNG.standard_normal(len(co))
    return gates, co, so


# ============================================================ driver
def report(title, gates, cheap_ok, strong_ok, deployed):
    print(f"\n{'='*104}\n{title}   (n={len(cheap_ok)})\n{'='*104}")
    print(f"  cheap acc = {float(np.mean(cheap_ok)):.4f}   strong acc = {float(np.mean(strong_ok)):.4f}   "
          f"headroom = {float(np.mean(strong_ok))-float(np.mean(cheap_ok)):+.4f}")
    rows = {}
    for name, keep in gates.items():
        rows[name] = gate_metrics(keep, cheap_ok, strong_ok)
    o = rows[list(gates)[0]]
    print(f"  ORACLE ADC = {o['adc_oracle']:.4f}   RANDOM ADC = {o['adc_random']:.4f}\n")
    hdr = f"  {'gate':<32}{'AUROC_det':>10}{'AUROC_rec':>10}{'ADC':>9}{'rout_eff':>9}{'minEsc@par':>11}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    order = sorted(gates, key=lambda k: -rows[k]["adc"])
    for name in order:
        m = rows[name]
        me = f"{m['min_esc_parity']*100:.0f}%" if not np.isnan(m["min_esc_parity"]) else "never"
        star = " *" if name == deployed else ("  <=OURS" if "ours" in name.lower() else "")
        print(f"  {name:<32}{m['auroc_detect']:>10.3f}{m['auroc_recover']:>10.3f}{m['adc']:>9.4f}"
              f"{m['routing_eff']:>9.3f}{me:>11}{star}")
    # paired bootstrap of ADC(challenger) - ADC(deployed): is the top-of-table edge real?
    print(f"\n  paired bootstrap  ADC(gate) - ADC({deployed})  [500 resamples, 95% CI, P(>0)]")
    for name in order:
        if name == deployed or name == "random": continue
        d, lo, hi, p = bootstrap_adc_diff(gates[name], gates[deployed], cheap_ok, strong_ok)
        flag = "  SIG+" if lo > 0 else ("  sig-" if hi < 0 else "")
        rows[name]["adc_vs_deployed"] = [d, lo, hi, p]
        print(f"    {name:<32}{d:>+9.4f}   [{lo:>+.4f}, {hi:>+.4f}]   P(>0)={p:.2f}{flag}")
    return rows

def main():
    DUMP = {}
    print("\n########################  UNIFIED CASCADE-GATE BAKE-OFF (offline, CPU)  ########################")

    # (A) MCQ
    Dm = load_mcq()
    for label, names in [("MCQ  competent-4 (MedVLThinker 7B-nt@cap320 -> 32B-think)", COMPETENT),
                         ("MCQ  ALL-5 incl MMMU", ALL5)]:
        P = pool_mcq(Dm, names)
        g = mcq_gates(P)
        r = report(label, g, P["cheap_ok"], P["strong_ok"], deployed="margin (DEPLOYED)")
        DUMP[label] = {k: v for k, v in r.items()}

    # (B) OPEN
    rows = load_open(["vqa_rad", "pathvqa", "kvasir", "radimagenet"])
    if rows:
        for label, dss in [("OPEN pooled (Lingshu-7B bo8+verifier -> 32B)", None),
                           ("OPEN vqa_rad only (strong competitive)", ["vqa_rad"])]:
            sub = rows if dss is None else [x for x in rows if x["ds"] in dss]
            g, co, so = open_gates(sub)
            r = report(label, g, co, so, deployed="verifier-conf (DEPLOYED)")
            DUMP[label] = {k: v for k, v in r.items()}

    out = J("results/cascade_methods/artifacts/gate_unified_bakeoff.json")
    json.dump({k: {gk: {mk: (None if isinstance(mv, float) and np.isnan(mv) else mv)
                        for mk, mv in gv.items()} for gk, gv in v.items()} for k, v in DUMP.items()},
              open(out, "w"), indent=1)
    print(f"\n[dump] {out}")

if __name__ == "__main__":
    main()
