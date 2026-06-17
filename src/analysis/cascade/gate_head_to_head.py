#!/usr/bin/env python3
"""
head_to_head.py - CPU-only head-to-head of escalation GATES on the competent-four benchmarks.
No GPU: reads the JSONL checkpoints you already have.

Fixed legs (only the GATE varies): kept answer = deployed cap320 7B (cascade pred7),
escalated answer = validated vLLM-32B label (gate_32b). Each gate decides escalate/keep from
7B-side signals; on escalation the answer is the 32B label by lookup. Nothing re-runs a model.

GATES, compared at a MATCHED escalation budget (the deployed cascade's own escalation rate):
  always-7B             floor (never escalate)
  margin (cap320)       the DEPLOYED frozen gate (rank by -margin from the cascade log)
  margin (fullres)      same idea on the full-res 7B logits (resolution control for conformal)
  conformal (CP-Router) split-conformal LAC prediction sets; rank by 2nd-option prob
  learned (HistGBM)     the multi-feature router from the probe
  always-32B            ceiling (always escalate)
  oracle                rank by the true rescue label (unbeatable upper bound)

CONSERVATIVE BY DESIGN: conformal and learned are given the RICHER full-res 7B logits as their
decision signal; the deployed margin gate sees only the cap320 margin. So a failure of
conformal/learned to beat the margin gate is conservative.

Decisive test: bootstrap CI on (router acc - margin_cap320 acc) at the deployed budget. If every
CI spans 0, the simple margin gate MATCHES conformal and learned routing at iso-compute.
"""
import argparse, json, glob, os, re, math
import numpy as np

FOUR = {"PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"}

def load_arm(ckdir, cell):
    rec, dsn = {}, {}
    if not ckdir or not os.path.isdir(ckdir): return rec, dsn
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$")
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        ds = m.group(1)
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            rec[r["idx"]] = r; dsn[r["idx"]] = ds
    return rec, dsn

def conf_feats(lp, pre):
    if not lp:
        return {f"{pre}_margin":0.0, f"{pre}_top1lp":-20.0, f"{pre}_top1p":1/3, f"{pre}_entropy":2.0,
                f"{pre}_nopts":0, f"{pre}_top1top3":0.0, f"{pre}_2ndp":1/3, f"{pre}_haslp":0}
    vals = sorted((float(v) for v in lp.values()), reverse=True)
    mx = vals[0]; ex = [math.exp(v-mx) for v in vals]; Z = sum(ex); p = [e/Z for e in ex]
    margin = vals[0]-vals[1] if len(vals) >= 2 else 0.0
    t1t3   = vals[0]-vals[2] if len(vals) >= 3 else (vals[0]-vals[-1] if len(vals) >= 2 else 0.0)
    ent    = -sum(pi*math.log(pi+1e-12) for pi in p)
    return {f"{pre}_margin":margin, f"{pre}_top1lp":vals[0], f"{pre}_top1p":p[0], f"{pre}_entropy":ent,
            f"{pre}_nopts":len(vals), f"{pre}_top1top3":t1t3,
            f"{pre}_2ndp":(p[1] if len(p) >= 2 else 0.0), f"{pre}_haslp":1}

def prob_dict(lp):
    if not lp: return {}
    vals = {k: float(v) for k, v in lp.items()}; mx = max(vals.values())
    ex = {k: math.exp(v-mx) for k, v in vals.items()}; Z = sum(ex.values())
    return {k: e/Z for k, e in ex.items()}

def top_two(lp):
    if not lp: return 1.0, 0.0, 20.0
    vals = sorted((float(v) for v in lp.values()), reverse=True)
    pr = sorted(prob_dict(lp).values(), reverse=True)
    p1 = pr[0]; p2 = pr[1] if len(pr) >= 2 else 0.0
    mg = vals[0]-vals[1] if len(vals) >= 2 else 20.0
    return p1, p2, mg

def policy(score, r7, r32, budget):
    n = len(score); k = int(round(budget*n)); esc = np.zeros(n, bool)
    if k > 0: esc[np.argsort(-score)[:k]] = True
    rescued = int((esc & ~r7 & r32).sum()); broken = int((esc & r7 & ~r32).sum())
    wasted  = int((esc & ~r7 & ~r32).sum()); redun = int((esc & r7 & r32).sum())
    rescuable = int((~r7 & r32).sum())
    acc = float(((~esc & r7) | (esc & r32)).mean())
    return dict(esc=int(esc.sum()), rescued=rescued, broken=broken, wasted=wasted,
                redundant=redun, recall=rescued/max(rescuable,1), acc=acc)

def boot(sa, sb, r7, r32, budget, nboot, seed=0):
    rng = np.random.default_rng(seed); n = len(r7); d = np.empty(nboot)
    for b in range(nboot):
        ix = rng.integers(0, n, n)
        d[b] = policy(sa[ix], r7[ix], r32[ix], budget)["acc"] - policy(sb[ix], r7[ix], r32[ix], budget)["acc"]
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def conf_calibrate(pds, golds, alpha):
    """LAC split-conformal: nonconformity = 1 - p(true class). Returns membership threshold thr=1-qhat."""
    s = np.array([1.0 - (pd.get(g, 0.0) if pd else 0.0) for pd, g in zip(pds, golds)])
    n = len(s); qlevel = min(1.0, math.ceil((n+1)*(1-alpha))/n)
    qhat = float(np.quantile(s, qlevel, method="higher"))
    return 1.0 - qhat

def conf_escalate(pds, thr):
    out = np.zeros(len(pds), bool)
    for i, pd in enumerate(pds):
        if not pd: out[i] = True; continue
        out[i] = (sum(1 for v in pd.values() if v >= thr) > 1)    # set size > 1 (multiple plausible) -> escalate
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", default="rt_cascade_cap320.jsonl")
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    ap.add_argument("--d32", default=None); ap.add_argument("--d7", default=None); ap.add_argument("--d7think", default=None)
    ap.add_argument("--all", action="store_true", help="include all benchmarks (default: competent four)")
    ap.add_argument("--test_frac", type=float, default=0.30); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nboot", type=int, default=2000)
    A = ap.parse_args()
    d32 = A.d32 or os.path.join(A.repo, "ckpts/gate_32b")
    d7  = A.d7  or os.path.join(A.repo, "ckpts/gate_7b_vllm")
    d7t = A.d7think or os.path.join(A.repo, "ckpts/gate_7b_think")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split
    except Exception as e:
        print(f"need scikit-learn: {e}"); return

    casc = {}
    if os.path.exists(A.cascade):
        for l in open(A.cascade):
            if l.strip(): r = json.loads(l); casc[r["idx"]] = r
    if not casc: print(f"no cascade file at {A.cascade}"); return
    g32, _ = load_arm(d32, "think_norag"); g7, _ = load_arm(d7, "nothink_norag"); g7t, _ = load_arm(d7t, "think_norag")
    print(f"loaded: cascade={len(casc)}  gate_32b={len(g32)}  gate_7b_vllm={len(g7)}  gate_7b_think={len(g7t)}")
    print(f"scope: {'ALL benchmarks' if A.all else 'competent four (PMC-VQA, SLAKE, VQA-RAD, PathVQA)'}\n")

    rows = []
    for i, c in casc.items():
        if i not in g32: continue
        ds = c["dataset"]
        if not A.all and ds not in FOUR: continue
        gold = c["gold"]; pred7 = c["pred7"]
        ok7 = int(pred7 == gold); ok32 = int(g32[i].get("ok", int(g32[i].get("pred") == gold)))
        lp_fr = g7[i].get("opt_logprobs", {}) if i in g7 else {}
        _, p2, mg_fr = top_two(lp_fr)
        feat = {"dep_margin": float(c.get("margin",0.0)), "dep_gen7": int(c.get("gen7",0)), "dep_parsed": int(pred7 != "?")}
        feat.update(conf_feats(lp_fr, "fr"))
        feat["fr_gen"] = int(g7[i].get("gen_tokens",0)) if i in g7 else 0
        feat["fr_parse"] = int(g7[i].get("parse_ok",0)) if i in g7 else 0
        if i in g7t:
            feat.update(conf_feats(g7t[i].get("opt_logprobs",{}), "th"))
            feat["th_gen"] = int(g7t[i].get("gen_tokens",0)); feat["disagree_think"] = int(g7t[i].get("pred") != pred7); feat["has_think"] = 1
        else:
            feat.update(conf_feats({}, "th")); feat["th_gen"]=0; feat["disagree_think"]=0; feat["has_think"]=0
        rows.append(dict(ds=ds, gold=gold, ok7=ok7, ok32=ok32, rescue=int(ok7==0 and ok32==1),
                         margin_cap=float(c.get("margin",0.0)), margin_fr=mg_fr, p2=p2,
                         pd=prob_dict(lp_fr), dep_esc=int(c.get("escalate",0)), feat=feat))
    if not rows: print("no rows after filtering."); return

    feat_names = sorted(rows[0]["feat"].keys())
    X = np.array([[r["feat"][f] for f in feat_names] for r in rows], float)
    y = np.array([r["rescue"] for r in rows]); idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=A.test_frac, random_state=A.seed, stratify=y)

    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=A.seed)
    gb.fit(X[tr], y[tr]); learned_p = gb.predict_proba(X[te])[:, 1]

    ok7 = np.array([rows[i]["ok7"] for i in te]).astype(bool); ok32 = np.array([rows[i]["ok32"] for i in te]).astype(bool)
    s_mcap = -np.array([rows[i]["margin_cap"] for i in te]); s_mfr = -np.array([rows[i]["margin_fr"] for i in te])
    s_conf = np.array([rows[i]["p2"] for i in te]); s_learn = learned_p; s_oracle = y[te].astype(float)
    dep_esc = np.array([rows[i]["dep_esc"] for i in te]).astype(bool)
    nT = len(te); rescuable = int((~ok7 & ok32).sum()); Estar = float(dep_esc.mean())
    print(f"test n={nT}   rescue base rate={y[te].mean():.3f} ({rescuable} rescuable)")
    print(f"deployed escalation rate (matched budget E*) = {Estar:.1%}\n")

    arms = [("always-7B", None, 0.0), ("margin cap320", s_mcap, Estar), ("margin fullres", s_mfr, Estar),
            ("conformal", s_conf, Estar), ("learned", s_learn, Estar), ("always-32B", None, 1.0), ("oracle", s_oracle, Estar)]
    print(f"HEAD-TO-HEAD at matched escalation E*={Estar:.0%}  (test n={nT})")
    print(f"  {'gate':<16}{'esc%':>6}{'acc':>8}{'resc':>6}{'broke':>6}{'waste':>6}{'redun':>6}{'recall':>8}")
    for nm, sc, bud in arms:
        P = policy(np.zeros(nT) if sc is None else sc, ok7, ok32, bud)
        print(f"  {nm:<16}{P['esc']/nT*100:>5.0f}%{P['acc']:>8.3f}{P['rescued']:>6}{P['broken']:>6}{P['wasted']:>6}{P['redundant']:>6}{P['recall']:>8.2f}")

    print(f"\nBOOTSTRAP CI vs the DEPLOYED margin (cap320) gate at E*={Estar:.0%} ({A.nboot} resamples):")
    any_beat = False
    for nm, sc in [("margin fullres", s_mfr), ("conformal", s_conf), ("learned", s_learn)]:
        md, lo, hi = boot(sc, s_mcap, ok7, ok32, Estar, A.nboot, seed=A.seed)
        v = "beats margin" if lo > 0 else ("worse than margin" if hi < 0 else "MATCHES margin (CI spans 0)")
        any_beat = any_beat or (lo > 0)
        print(f"  {nm:<16} delta acc = {md:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   -> {v}")

    print(f"\nCONFORMAL natural operating points (LAC split-conformal, escalate if set size > 1):")
    print(f"  {'alpha':>6}{'thr':>8}{'esc%':>7}{'acc':>8}")
    for al in (0.30, 0.20, 0.10, 0.05):
        th = conf_calibrate([rows[i]["pd"] for i in tr], [rows[i]["gold"] for i in tr], al)
        esc = conf_escalate([rows[i]["pd"] for i in te], th)
        acc = float(((~esc & ok7) | (esc & ok32)).mean())
        print(f"  {al:>6.2f}{th:>8.3f}{esc.mean()*100:>6.0f}%{acc:>8.3f}")

    print(f"\nDEFERRAL CURVE across budgets (acc):")
    print(f"  {'budget':>7}{'margin':>9}{'conformal':>11}{'learned':>9}{'oracle':>9}")
    for b in sorted(set([round(Estar,2), 0.20, 0.40, 0.55, 0.80])):
        print(f"  {b:>7.0%}{policy(s_mcap,ok7,ok32,b)['acc']:>9.3f}{policy(s_conf,ok7,ok32,b)['acc']:>11.3f}"
              f"{policy(s_learn,ok7,ok32,b)['acc']:>9.3f}{policy(s_oracle,ok7,ok32,b)['acc']:>9.3f}")

    print("\n" + "="*66)
    if not any_beat:
        print("RESULT: at the deployed escalation budget, NO router beats the frozen margin gate")
        print("  (every 95% CI spans 0). The parameter-free margin gate MATCHES a CP-Router-style")
        print("  conformal router and a learned multi-feature router at iso-compute on the")
        print("  competent benchmarks -- the credible negative: sophistication buys no deployable")
        print("  gain in this regime.")
    else:
        print("RESULT: a router's CI excludes 0 -- inspect the table; a real gain may exist here.")

if __name__ == "__main__":
    main()
