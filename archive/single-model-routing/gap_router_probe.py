#!/usr/bin/env python3
"""
gap_router_probe.py - OFFLINE falsification probe for a "gap router". CPU-only, no GPU.

QUESTION: can we PREDICT which queries the 32B will RESCUE (7B wrong AND 32B right)
from signals available WITHOUT running the 32B? If yes, a smarter gate could escalate
fewer queries and keep more rescues. If no, the frozen margin gate is at the frontier.

PRE-REGISTERED GATE: held-out ROC-AUC for rescue must exceed --gate_auroc (default 0.60).
But AUROC is not the deployment test -- the DEFERRAL CURVE + the bootstrap CI on the
router-minus-margin accuracy delta AT YOUR ESCALATION BUDGET are. A router that wins on
AUROC but not on accuracy-at-budget is not worth deploying.

FLAGS added for the iso-accuracy ablation:
  --four-only   restrict to PMC-VQA, SLAKE, VQA-RAD, PathVQA (drop the excluded MedXpert/MMMU)
  --no-dataset  drop the dataset one-hot from features (an open-world deployment lacks it;
                this isolates the per-query signal from per-benchmark priors)
  --budget B    the operating point to bootstrap the acc delta at (default 0.65)
  --nboot N     bootstrap resamples for the acc-delta CI (default 2000)

INPUTS: --cascade rt_cascade_*.jsonl ; --d32 gate_32b ; --d7 gate_7b_vllm ; --d7think gate_7b_think
LABEL : rescue = 1[ 7B(cap320) wrong AND 32B(vLLM) right ].   BASELINE: -margin_cap320.
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

def policy(score, r7, r32, budget):
    n = len(score); k = int(round(budget*n)); esc = np.zeros(n, bool)
    if k > 0: esc[np.argsort(-score)[:k]] = True
    rescued = int((esc & ~r7 & r32).sum()); broken = int((esc & r7 & ~r32).sum())
    wasted  = int((esc & ~r7 & ~r32).sum()); redun = int((esc & r7 & r32).sum())
    rescuable = int((~r7 & r32).sum())
    acc = float(((~esc & r7) | (esc & r32)).mean())
    return dict(esc=int(esc.sum()), rescued=rescued, broken=broken, wasted=wasted,
                redundant=redun, recall=rescued/max(rescuable,1), acc=acc)

def boot_delta(sr, sm, r7, r32, budget, nboot, seed=0):
    """Bootstrap the (router acc - margin acc) at `budget` over the test set."""
    rng = np.random.default_rng(seed); n = len(r7); d = np.empty(nboot)
    for b in range(nboot):
        ix = rng.integers(0, n, n)
        d[b] = policy(sr[ix], r7[ix], r32[ix], budget)["acc"] - policy(sm[ix], r7[ix], r32[ix], budget)["acc"]
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", default="rt_cascade_cap320.jsonl")
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    ap.add_argument("--d32", default=None); ap.add_argument("--d7", default=None); ap.add_argument("--d7think", default=None)
    ap.add_argument("--gate_auroc", type=float, default=0.60)
    ap.add_argument("--test_frac", type=float, default=0.30); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--four-only", action="store_true", help="restrict to PMC-VQA/SLAKE/VQA-RAD/PathVQA")
    ap.add_argument("--no-dataset", action="store_true", help="drop dataset one-hot from features")
    ap.add_argument("--budget", type=float, default=0.65, help="operating point to bootstrap the acc delta at")
    ap.add_argument("--nboot", type=int, default=2000)
    A = ap.parse_args()
    d32 = A.d32 or os.path.join(A.repo, "ckpts/gate_32b")
    d7  = A.d7  or os.path.join(A.repo, "ckpts/gate_7b_vllm")
    d7t = A.d7think or os.path.join(A.repo, "ckpts/gate_7b_think")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception as e:
        print(f"need scikit-learn (pip install scikit-learn): {e}"); return

    casc = {}
    if os.path.exists(A.cascade):
        for l in open(A.cascade):
            if l.strip(): r = json.loads(l); casc[r["idx"]] = r
    g32, _    = load_arm(d32, "think_norag")
    g7,  ds7  = load_arm(d7,  "nothink_norag")
    g7t, _    = load_arm(d7t, "think_norag")
    print(f"loaded: cascade={len(casc)}  gate_32b={len(g32)}  gate_7b_vllm={len(g7)}  gate_7b_think={len(g7t)}")
    flags = []
    if A.four_only: flags.append("--four-only")
    if A.no_dataset: flags.append("--no-dataset")
    if flags: print(f"ablation: {' '.join(flags)}")

    use_casc = len(casc) > 0
    src7 = casc if use_casc else g7
    idxs = [i for i in src7 if i in g32]
    if not idxs: print("no overlap 7B<->gate_32b; check paths."); return

    rows = []
    for i in idxs:
        if use_casc:
            pred7 = casc[i]["pred7"]; gold = casc[i]["gold"]; ds = casc[i]["dataset"]
            margin_dep = float(casc[i].get("margin", 0.0)); gen7_dep = int(casc[i].get("gen7", 0))
        else:
            r = g7[i]; pred7 = r["pred"]; gold = r["gold"]; ds = ds7.get(i, "?")
            margin_dep = conf_feats(r.get("opt_logprobs", {}), "fr")["fr_margin"]; gen7_dep = int(r.get("gen_tokens", 0))
        if A.four_only and ds not in FOUR: continue
        ok7  = int(pred7 == gold); ok32 = int(g32[i].get("ok", int(g32[i].get("pred") == gold)))
        feat = {"dep_margin": margin_dep, "dep_gen7": gen7_dep, "dep_parsed": int(pred7 != "?")}
        feat.update(conf_feats(g7[i].get("opt_logprobs", {}) if i in g7 else {}, "fr"))
        feat["fr_gen"]   = int(g7[i].get("gen_tokens", 0)) if i in g7 else 0
        feat["fr_parse"] = int(g7[i].get("parse_ok", 0)) if i in g7 else 0
        if i in g7t:
            feat.update(conf_feats(g7t[i].get("opt_logprobs", {}), "th"))
            feat["th_gen"] = int(g7t[i].get("gen_tokens", 0)); feat["disagree_think"] = int(g7t[i].get("pred") != pred7); feat["has_think"] = 1
        else:
            feat.update(conf_feats({}, "th")); feat["th_gen"] = 0; feat["disagree_think"] = 0; feat["has_think"] = 0
        rows.append((i, ds, ok7, ok32, int(ok7 == 0 and ok32 == 1), feat))

    if not rows: print("no rows after filtering."); return
    feat_names = sorted(rows[0][5].keys())
    datasets   = sorted(set(r[1] for r in rows))
    ds_cols = [] if A.no_dataset else datasets
    X = np.array([[r[5][f] for f in feat_names] + [1.0 if r[1] == d else 0.0 for d in ds_cols] for r in rows], float)
    y    = np.array([r[4] for r in rows]); ok7 = np.array([r[2] for r in rows]); ok32 = np.array([r[3] for r in rows])
    m_all = np.array([r[5]["dep_margin"] for r in rows]); dsarr = np.array([r[1] for r in rows])

    n = len(y); base = float(y.mean()); r7 = ok7.astype(bool); r32 = ok32.astype(bool)
    print(f"\nrows usable: {n}  ({len(feat_names)} feats{' + '+str(len(ds_cols))+' dataset cols' if ds_cols else ', no dataset cols'})"
          f"   rescue base rate: {base:.3f}  ({int(y.sum())} of {n})")
    print(f"buckets over rows: 7B right {r7.mean():.3f} | 32B right {r32.mean():.3f} | "
          f"rescuable {int((~r7&r32).sum())}  both-right {int((r7&r32).sum())}  both-wrong {int((~r7&~r32).sum())}  7Bright-32Bwrong {int((r7&~r32).sum())}")

    Xtr,Xte,ytr,yte,m_tr,m_te,o7tr,o7te,o32tr,o32te,ds_tr,ds_te = train_test_split(
        X, y, m_all, ok7, ok32, dsarr, test_size=A.test_frac, random_state=A.seed, stratify=y)

    auc_margin = roc_auc_score(yte, -m_te)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=A.seed)
    gb.fit(Xtr, ytr); p_gb = gb.predict_proba(Xte)[:, 1]; auc_gb = roc_auc_score(yte, p_gb)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    lr.fit(Xtr, ytr); p_lr = lr.predict_proba(Xte)[:, 1]; auc_lr = roc_auc_score(yte, p_lr)
    print(f"\nHELD-OUT ROC-AUC for rescue (test n={len(yte)}):")
    print(f"  baseline  -margin (deployed signal) : {auc_margin:.3f}")
    print(f"  LogReg    full feature union        : {auc_lr:.3f}")
    print(f"  HistGBM   full feature union        : {auc_gb:.3f}")
    best = max(auc_gb, auc_lr); best_name = "HistGBM" if auc_gb >= auc_lr else "LogReg"
    best_p = p_gb if auc_gb >= auc_lr else p_lr

    print(f"\nper-dataset held-out AUROC ({best_name} vs margin):")
    for d in datasets:
        mk = (ds_te == d)
        if mk.sum() >= 10 and 0 < yte[mk].mean() < 1:
            print(f"  {d:<22} n={int(mk.sum()):4d}  rescue_rate={yte[mk].mean():.3f}  "
                  f"router={roc_auc_score(yte[mk], best_p[mk]):.3f}  margin={roc_auc_score(yte[mk], -m_te[mk]):.3f}")
        else:
            print(f"  {d:<22} n={int(mk.sum()):4d}  (too few / single-class)")

    print(f"\nDEFERRAL CURVE on held-out test (n={len(yte)}):")
    print(f"  {'budget':>7} | {'policy':>7} | {'esc':>4} {'resc':>5} {'broke':>5} {'waste':>5} {'redun':>5} | {'recall':>6} {'acc':>6}")
    for b in sorted(set([round(base, 2), 0.20, 0.40, 0.55, round(A.budget,2), 0.80])):
        for nm in ("margin", "router", "oracle"):
            sc = {"margin": -m_te, "router": best_p, "oracle": yte.astype(float)}[nm]
            P = policy(sc, o7te.astype(bool), o32te.astype(bool), b)
            print(f"  {b:>7.0%} | {nm:>7} | {P['esc']:>4} {P['rescued']:>5} {P['broken']:>5} {P['wasted']:>5} {P['redundant']:>5} | {P['recall']:>6.2f} {P['acc']:>6.3f}")
        print()

    md, lo, hi = boot_delta(best_p, -m_te, o7te.astype(bool), o32te.astype(bool), A.budget, A.nboot, seed=A.seed)
    mP = policy(-m_te, o7te.astype(bool), o32te.astype(bool), A.budget)
    rP = policy(best_p, o7te.astype(bool), o32te.astype(bool), A.budget)
    print(f"ISO-BUDGET TEST at escalation={A.budget:.0%}  ({A.nboot} bootstrap resamples):")
    print(f"  margin acc {mP['acc']:.3f}   router acc {rP['acc']:.3f}")
    print(f"  router - margin acc = {md:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    sig = (lo > 0)
    print(f"  -> {'SIGNIFICANT: router beats margin at the operating point' if sig else 'NOT significant: CI spans 0 -> no real edge at the operating point'}")

    print("=" * 66)
    if best > A.gate_auroc and sig:
        print(f"PASS (AUROC {best:.3f}>{A.gate_auroc:.2f}) AND router wins accuracy at {A.budget:.0%} (CI excludes 0).")
        print("  A richer router is worth building: real, deployable efficiency headroom.")
    elif best > A.gate_auroc:
        print(f"AUROC PASS ({best:.3f}>{A.gate_auroc:.2f}) but NO accuracy edge at {A.budget:.0%} (CI spans 0).")
        print("  Rescue is weakly predictable, but the predictability does NOT convert into")
        print("  lower escalation at iso-accuracy where you deploy. The frozen margin gate")
        print("  stands as the deployable; the negative is clean and consistent with the")
        print("  non-uniform-model-2 regime (Jitkrittum et al., NeurIPS 2023).")
    else:
        print(f"FAIL: AUROC {best:.3f} <= {A.gate_auroc:.2f}. Rescue not separable; frozen gate stands.")

if __name__ == "__main__":
    main()
