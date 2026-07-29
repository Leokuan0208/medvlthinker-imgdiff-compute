#!/usr/bin/env python3
"""
pandora_correlated.py - CORRELATED-PANDORA: a correlation-aware variant of the validated
Pandora's-Box controller (src/cascade_methods/pandora_controller.py). OFFLINE, CPU-only, reuses
the exact same per-sample dumps, cost model, held-out cross-fit and baselines.

WHY. The validated Pandora controller is a real both-axes win (-27% FLOPs @ iso-bo8, -61% @
iso-strong) but it OVER-DRAWS (meanN ~5.4) because independent-Weitzman assumes the within-question
cheap draws are i.i.d. draws from the marginal verifier-score distribution. They are NOT: samples
from one model on one question are CORRELATED -- once several draws AGREE (same answer / same
verifier score), another draw is very likely to just repeat what we already have and adds almost
no information. The independent reservation value therefore over-states the option value of "one
more draw" and the controller keeps drawing past the point of diminishing returns.

THE FIX (correlation-aware reservation value). Weitzman's cheap-box reservation z solves
    lambda * c_cheap = E_v[(v - z)^+]                       (option value == inspection cost).
Model the next draw as, with prob (1 - rho_k) a REPEAT of an already-seen answer (contributes ~0
new option value) and with prob rho_k a genuinely fresh draw ~ the marginal. Then the option value
of the next draw is discounted:  lambda*c = rho_k * E_v[(v - z)^+], i.e. the SAME reservation
equation with an INFLATED effective inspection cost  c_eff = c_cheap / rho_k.  rho_k in (0,1] is the
observed DIVERSITY of the samples drawn so far:
    rho_k = 1 - collision_k,  collision_k = sum_a n_a(n_a-1) / (k(k-1))   (unbiased P[two draws match])
(k<2 -> rho=1, i.e. no info yet -> behaves EXACTLY like independent Pandora). As agreement rises
rho_k falls -> c_eff rises -> z_cheap falls -> the STOP condition (best-so-far >= z_cheap) fires
SOONER. When the drawn samples fully agree, rho->0, z_cheap->-inf, and the controller stops (or,
if best-so-far still < z_strong, escalates). Guarantee by construction: at a fixed lambda,
correlated meanN <= independent meanN; the honest question the frontier answers is whether the
draws it removes were REDUNDANT (accuracy held -> free meanN cut) or USEFUL (accuracy drops).

Everything else (cost model, isotonic cross-fit calibration, q_strong, the marginal cheap-score
pool, the 5-fold held-out protocol, the escalation box z_strong = q_strong - lambda*c_strong, the
gate / adaptive-N / bo8 / always-32B baselines) is IDENTICAL to pandora_controller.py and imported
from it, so the ONLY change under test is the correlation discount. The diversity discount uses the
recorded answer TEXT (r['preds'], normalized) -- observable at deploy time, no correctness peek.

Robustness: also runs an entropy-based and a Good-Turing-novelty-based discount; simpson is primary.

Launch from repo root:  python3 src/cascade_methods/pandora_correlated.py
"""
import os, sys, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandora_controller import (  # reuse the EXACT validated pieces
    REPO, J, C_CHEAP_F, C_STRONG_F, cost_of, OPEN_DSETS, ADAPTER, FAM, load_judge,
    zeta_cheap, zeta_strong, run_pandora, run_adaptiveN, pareto, min_flops_at,
    LAMS, GATE_TAUS, ANS_STOP, ANS_ESC, agg_points, pool_bases, pool_aggs, fmt)

MEASURES = ["simpson", "entropy", "novelty"]   # simpson = primary
PRIMARY = "simpson"
RHO_FLOOR = 1e-3

def normpred(s):
    return str(s).strip().lower()

# ------------------------------------------------------------------ data loader (adds preds)
def load_family_preds(cfg):
    """same as pandora_controller.load_family but ALSO keeps normalized answer text (preds),
    needed for the diversity discount."""
    rows = []
    for ds in OPEN_DSETS:
        dp = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{cfg['vtag']}.json"))
        if not os.path.exists(dp):
            continue
        dump = json.load(open(dp))
        sj = {}
        for t in cfg["sj"]:
            sj = load_judge(J(os.path.join(cfg["sdir"], t.format(ds=ds))))
            if sj:
                break
        for r in dump:
            i = r["idx"]
            if i not in sj:
                continue
            sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
            sc = list(r["scores"])
            pr = list(r.get("preds", []))
            n = min(len(sl), len(sc))
            if pr:
                n = min(n, len(pr))
            if n < 1:
                continue
            preds = [normpred(x) for x in pr[:n]] if pr else [f"__u{j}" for j in range(n)]
            rows.append(dict(ds=ds, scores=sc[:n], sl=sl[:n], preds=preds, strong=int(sj[i])))
    return rows

# ------------------------------------------------------------------ diversity discount rho_k
def diversity_rho(drawn_preds, measure):
    """rho in [RHO_FLOOR, 1]. 1 = fully diverse so far (== independent); ->0 = fully agreed."""
    k = len(drawn_preds)
    if k < 2:
        return 1.0
    cnt = Counter(drawn_preds)
    return _rho_from_counts(cnt.values(), k, measure)

def _rho_from_counts(counts, k, measure):
    if k < 2:
        return 1.0
    counts = list(counts)
    if measure == "simpson":                       # 1 - unbiased P(two random draws collide)
        coll = sum(c * (c - 1) for c in counts) / (k * (k - 1))
        rho = 1.0 - coll
    elif measure == "entropy":                     # normalized Shannon entropy of the answer mix
        p = np.array(counts, float) / k
        rho = float(-(p * np.log(p)).sum() / np.log(k))
    elif measure == "novelty":                     # Good-Turing missing-mass = singletons / k
        rho = sum(1 for c in counts if c == 1) / k
    else:
        raise ValueError(measure)
    return float(max(RHO_FLOOR, min(1.0, rho)))

def rho_sequence(preds, measure):
    """rho_seq[k] = diversity_rho(preds[:k]) for k=0..len(preds). Depends only on the recorded pred
    prefix (NOT on lambda) -> precompute once per question instead of per (lambda x draw)."""
    seq = [1.0, 1.0]                                # k=0 and k=1 -> no discount
    cnt = Counter()
    for k, p in enumerate(preds):
        cnt[p] += 1
        if k + 1 >= 2:
            seq.append(_rho_from_counts(cnt.values(), k + 1, measure))
    return seq[:len(preds) + 1]

# ------------------------------------------------------------------ correlation-aware reservation
_ZPREP = {}   # id(v) -> (v_ref, sorted-desc w, prefix P, breakpoint E's Eb, n, mean, vmin, vmax)

def _zprep(v):
    """precompute the sorted marginal + prefix sums once per array object (id-keyed; the stored ref
    keeps v alive so ids are never reused -> safe). Enables analytic O(log n) reservation queries."""
    key = id(v)
    ent = _ZPREP.get(key)
    if ent is not None and ent[0] is v:
        return ent
    w = np.sort(np.asarray(v, float))[::-1]
    n = len(w)
    P = np.concatenate([[0.0], np.cumsum(w)])          # P[k] = sum of top k
    i = np.arange(n + 1)
    Eb = (P - i * np.concatenate([w, [w[-1]]])) / n     # Eb[i] = E[(v - w[i])^+] (increasing in i)
    ent = (v, w, P, Eb, n, float(w.mean()), float(w[-1]), float(w[0]))
    _ZPREP[key] = ent
    return ent

def zeta_from_t(v, t):
    """Weitzman cheap reservation solving t = E_v[(v - z)^+]; z decreasing in t. Analytic (piecewise-
    linear) solution: with the top (i+1) marginal values active, z = (P[i+1] - n*t)/(i+1). Matches the
    old 80-iteration bisection to ~1e-9 but is O(log n) per query (id-cached sort+prefix)."""
    _, w, P, Eb, n, mean, vmin, vmax = _zprep(v)
    if t <= 0.0:
        return vmax
    if t >= mean - vmin:                                # target below the support -> closed-form tail
        return mean - t
    i = int(np.searchsorted(Eb, t, side="right")) - 1   # largest i with Eb[i] <= t
    if i < 0:
        return vmax
    return (P[i + 1] - n * t) / (i + 1)

def run_pandora_corr(scores_raw, scores_cal, rho_seq, sl, strong_ok, marg_pool, lam, z_strong, cache):
    """CORRELATED Pandora on one question. Identical to run_pandora except z_cheap is recomputed
    each step with effective cost c_cheap / rho_k (rho_k = diversity of the samples drawn so far,
    precomputed in rho_seq). marg_pool: the fold's calibrated cheap-score marginal (np array).
    cache: per-fold dict t->z."""
    Nmax = len(scores_raw)
    best_cal = -1e18; best_raw = -1e18; pick = -1; k = 0
    while True:
        if k < Nmax:
            rho = rho_seq[k]
            t = lam * C_CHEAP_F / rho
            key = round(t, 10)
            z_cheap = cache.get(key)
            if z_cheap is None:
                z_cheap = zeta_from_t(marg_pool, t); cache[key] = z_cheap
        else:
            z_cheap = -1e18                                  # cheap boxes exhausted
        boxtype, maxres = max((("cheap", z_cheap), ("strong", z_strong)), key=lambda x: x[1])
        if best_cal >= maxres:                               # STOP: best-so-far beats every reservation
            break
        if boxtype == "cheap":
            s_cal = scores_cal[k]; s_raw = scores_raw[k]
            if s_cal > best_cal: best_cal = s_cal
            if s_raw > best_raw: best_raw = s_raw; pick = k
            k += 1
        else:
            return k, 1, int(strong_ok)                      # escalate (commit to strong)
    return k, 0, int(sl[pick] if pick >= 0 else 0)

# ------------------------------------------------------------------ per-dataset eval (cross-fit)
def eval_dataset(rows, seed=0, nfold=5):
    """returns base, {method: {knob:{N,esc,acc,n}}} for pandora_indep, pandora_corr_<measure>,
    gate, adaptiveN. Baselines + independent pandora are IDENTICAL to pandora_controller."""
    n = len(rows)
    raw = [r["scores"] for r in rows]
    sl = [r["sl"] for r in rows]
    preds = [r["preds"] for r in rows]
    strong = np.array([r["strong"] for r in rows], float)
    Nmax = min(len(r["scores"]) for r in rows)

    bo8_ok = np.array([sl[i][int(np.argmax(raw[i][:Nmax]))] for i in range(n)], float)
    oracle_ok = np.array([max(sl[i][:Nmax]) for i in range(n)], float)
    base = dict(strong_acc=float(strong.mean()), bo8_acc=float(bo8_ok.mean()),
                oracle_acc=float(oracle_ok.mean()), n=n, Nmax=Nmax)

    # precompute per-question diversity rho sequences (depend only on the pred prefix, not lambda)
    rho_seqs = {m: [rho_sequence(preds[i][:Nmax], m) for i in range(n)] for m in MEASURES}

    kf = KFold(nfold if n >= nfold else 2, shuffle=True, random_state=seed)
    ind = {float(l): dict(N=np.zeros(n), esc=np.zeros(n), ok=np.zeros(n)) for l in LAMS}
    cor = {m: {float(l): dict(N=np.zeros(n), esc=np.zeros(n), ok=np.zeros(n)) for l in LAMS}
           for m in MEASURES}
    for tr, te in kf.split(np.arange(n)):
        xs = np.concatenate([raw[i][:Nmax] for i in tr])
        ys = np.concatenate([np.array(sl[i][:Nmax], float) for i in tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); iso.fit(xs, ys)
        train_pool = iso.predict(xs)
        q = float(strong[tr].mean())
        te_cal = {i: iso.predict(np.asarray(raw[i][:Nmax], float)) for i in te}
        cache = {m: {} for m in MEASURES}
        for l in LAMS:
            zc = zeta_cheap(train_pool, l); zs = zeta_strong(q, l)
            for i in te:
                N, e, ok = run_pandora(raw[i][:Nmax], te_cal[i], sl[i][:Nmax], strong[i], zc, zs)
                d = ind[float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok
                for m in MEASURES:
                    N, e, ok = run_pandora_corr(raw[i][:Nmax], te_cal[i], rho_seqs[m][i],
                                                sl[i][:Nmax], strong[i], train_pool, l, zs, cache[m])
                    d = cor[m][float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok

    def agg(dd):
        return {l: dict(N=float(d["N"].mean()), esc=float(d["esc"].mean()),
                        acc=float(d["ok"].mean()), n=n) for l, d in dd.items()}
    methods = {"pandora_indep": agg(ind)}
    for m in MEASURES:
        methods[f"pandora_corr_{m}"] = agg(cor[m])

    # ---- baselines (identical to pandora_controller) ----
    gate = {}
    vmax = np.array([max(raw[i][:Nmax]) for i in range(n)], float)
    for tau in GATE_TAUS:
        esc = (vmax < tau); ok = np.where(esc, strong, bo8_ok)
        gate[float(tau)] = dict(N=float(Nmax), esc=float(esc.mean()), acc=float(ok.mean()), n=n)
    adn = {}
    for ts in ANS_STOP:
        for tesc in ANS_ESC:
            Ns = np.zeros(n); es = np.zeros(n); oks = np.zeros(n)
            for i in range(n):
                N, e, ok = run_adaptiveN(raw[i][:Nmax], sl[i][:Nmax], strong[i], ts, tesc)
                Ns[i] = N; es[i] = e; oks[i] = ok
            adn[(ts, tesc)] = dict(N=float(Ns.mean()), esc=float(es.mean()), acc=float(oks.mean()), n=n)
    methods["gate"] = gate
    methods["adaptiveN"] = adn
    return base, methods

# ------------------------------------------------------------------ reporting
METHOD_ORDER = ["pandora_indep", f"pandora_corr_{PRIMARY}"] + \
               [f"pandora_corr_{m}" for m in MEASURES if m != PRIMARY] + ["gate", "adaptiveN"]

def build_out(base, methods):
    Nmax = base["Nmax"]
    fr = {k: pareto(agg_points(v)) for k, v in methods.items()}
    always32 = dict(acc=base["strong_acc"], meanN=0.0, esc=1.0, **cost_of(0.0, 1.0))
    bo8 = dict(acc=base["bo8_acc"], meanN=float(Nmax), esc=0.0, **cost_of(Nmax, 0.0))
    out = dict(n=base["n"], Nmax=Nmax, strong_acc=base["strong_acc"], bo8_acc=base["bo8_acc"],
               oracle_acc=base["oracle_acc"], always32=always32, bo8=bo8,
               frontiers={k: [dict(p) for p in v] for k, v in fr.items()})
    iso = {}
    for tname, tacc in [("iso_bo8", base["bo8_acc"]), ("iso_strong", base["strong_acc"])]:
        iso[tname] = dict(target=tacc)
        for k in methods:
            iso[tname][k] = min_flops_at(fr[k], tacc)
        iso[tname]["bo8"] = bo8 if bo8["acc"] >= tacc - 3e-3 else None
        iso[tname]["always32"] = always32 if always32["acc"] >= tacc - 3e-3 else None
    out["iso"] = iso
    return out

def print_block(title, out):
    print(f"\n{'='*120}\n{title}   n={out['n']}  (verifier best-of-{out['Nmax']})")
    print(f"  strong={out['strong_acc']:.3f}  bo8={out['bo8_acc']:.3f}  oracle@{out['Nmax']}={out['oracle_acc']:.3f}")
    for tname in ("iso_bo8", "iso_strong"):
        it = out["iso"][tname]; tgt = it["target"]
        print(f"  -- iso-accuracy @ {tname.replace('iso_','match ')} (acc>={tgt:.3f}) : cheapest op-point --")
        for k in METHOD_ORDER:
            tag = "  <== CORRELATED (ours)" if k == f"pandora_corr_{PRIMARY}" else \
                  ("  (validated indep)" if k == "pandora_indep" else "")
            print(f"       {k:<24}{fmt(it[k])}{tag}")

def summarize(all_ds):
    """per-domain-tuned aggregate: pick cheapest op-point per domain that hits the target, n-weight."""
    SUMMARY = {}
    for tname, tlabel in [("iso_bo8", "match per-domain verifier-bo8 accuracy"),
                          ("iso_strong", "match per-domain always-32B accuracy")]:
        agg = {}
        for meth in METHOD_ORDER:
            hit = [(o["n"], o["iso"][tname][meth]) for _, o in all_ds if o["iso"][tname][meth] is not None]
            Ntot = sum(nn for nn, _ in hit)
            if Ntot == 0:
                agg[meth] = None; continue
            agg[meth] = dict(datasets_covered=len(hit), datasets_total=len(all_ds),
                             flops=sum(nn * p["flops"] for nn, p in hit) / Ntot,
                             energy=sum(nn * p["energy"] for nn, p in hit) / Ntot,
                             meanN=sum(nn * p["meanN"] for nn, p in hit) / Ntot,
                             esc=sum(nn * p["esc"] for nn, p in hit) / Ntot,
                             lat_seq=sum(nn * p["lat_seq"] for nn, p in hit) / Ntot)
        SUMMARY[tname] = dict(label=tlabel, methods=agg, ref_bo8_flops=16.0, ref_always32_flops=C_STRONG_F)
        print(f"\n  [{tname}] {tlabel}   (n-weighted over {len(all_ds)} datasets; bo8 FLOPs=16.0)")
        for meth in METHOD_ORDER:
            a = agg[meth]
            if a is None:
                print(f"    {meth:<24} n/a"); continue
            tag = "  <== CORRELATED (ours)" if meth == f"pandora_corr_{PRIMARY}" else \
                  ("  (validated indep)" if meth == "pandora_indep" else "  (oracle-tau)")
            print(f"    {meth:<20} cover {a['datasets_covered']}/{a['datasets_total']}  "
                  f"FLOPs={a['flops']:5.2f} (-{(1-a['flops']/16.0)*100:2.0f}% vs bo8)  "
                  f"meanN={a['meanN']:.2f}  esc={a['esc']*100:3.0f}%  Lseq={a['lat_seq']:5.0f}ms{tag}")
    # explicit correlated-vs-independent meanN delta
    print(f"\n  {'-'*90}\n  CORRELATED ({PRIMARY}) vs INDEPENDENT Pandora  (meanN / FLOPs cut at iso-accuracy):")
    for tname in ("iso_bo8", "iso_strong"):
        ind = SUMMARY[tname]["methods"]["pandora_indep"]
        cor = SUMMARY[tname]["methods"][f"pandora_corr_{PRIMARY}"]
        if ind and cor:
            dN = (1 - cor["meanN"] / ind["meanN"]) * 100 if ind["meanN"] else 0.0
            dF = (1 - cor["flops"] / ind["flops"]) * 100 if ind["flops"] else 0.0
            print(f"    [{tname}]  indep meanN={ind['meanN']:.2f} F={ind['flops']:.2f}  ->  "
                  f"corr meanN={cor['meanN']:.2f} F={cor['flops']:.2f}   "
                  f"(meanN -{dN:.0f}%, FLOPs -{dF:.0f}%)")
    return SUMMARY

# ------------------------------------------------------------------ main
def main():
    print("#" * 120)
    print("CORRELATED-PANDORA - correlation-aware reservation value (discount option value by sample agreement)")
    print(f"  discount rho = diversity of drawn answers; primary measure = {PRIMARY}; c_eff = c_cheap / rho")
    print("  everything else (cost model, cross-fit calibration, held-out, baselines) == pandora_controller.py")
    print("#" * 120)
    DUMP = {}
    all_ds = []
    for fam, cfg in FAM.items():
        rows = load_family_preds(cfg)
        if not rows:
            print(f"\n### {fam}: no data"); continue
        by_ds = defaultdict(list)
        for r in rows:
            by_ds[r["ds"]].append(r)
        print(f"\n{'#'*120}\n############  FAMILY: {fam}  ({len(rows)} q over {len(by_ds)} datasets)  ############")
        fam_out = {}
        per_ds = {}
        for ds in OPEN_DSETS:
            if ds not in by_ds:
                continue
            base, methods = eval_dataset(by_ds[ds])
            per_ds[ds] = (base, methods)
            out = build_out(base, methods); fam_out[ds] = out
            all_ds.append((f"{fam}/{ds}", out))
            print_block(f"{fam}/{ds}", out)
        evals = list(per_ds.values())
        pooled_base = pool_bases([e[0] for e in evals])
        pooled_methods = {k: pool_aggs([e[1][k] for e in evals]) for k in evals[0][1]}
        out_pd = build_out(pooled_base, pooled_methods)
        fam_out["POOLED_per_domain"] = out_pd
        print_block(f"{fam}/POOLED (per-domain calibration)", out_pd)
        DUMP[fam] = fam_out

    print(f"\n{'#'*120}\n#####  HEADLINE: per-domain-tuned aggregate (correlated vs independent vs baselines)  #####\n{'#'*120}")
    DUMP["SUMMARY_per_domain_tuned"] = summarize(all_ds)
    DUMP["_meta"] = dict(measures=MEASURES, primary=PRIMARY, rho_floor=RHO_FLOOR,
                         discount="c_eff = c_cheap / rho ; rho = diversity of drawn answers (preds)")

    outp = J("results/cascade_methods/artifacts/pandora_correlated.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(DUMP, open(outp, "w"), indent=1,
              default=lambda o: (float(o) if isinstance(o, np.floating) else
                                 (o.tolist() if isinstance(o, np.ndarray) else o)))
    print(f"\n[dump] {outp}")

if __name__ == "__main__":
    main()
