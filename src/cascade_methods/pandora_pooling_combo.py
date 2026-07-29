#!/usr/bin/env python3
"""
pandora_pooling_combo.py - PANDORA x POOLING: stack the project's TWO validated levers.

Lever 1 (pandora_controller.py): the Weitzman Pandora's-Box adaptive controller -- one lambda gives
both the STOP-DRAWING and the ESCALATE thresholds; a real both-axes win over gate / adaptive-N.
Lever 2 (generator_portfolio.py): a CROSS-MODEL cheap pool {Lingshu-7B, MedVLThinker-7B, InternVL3-8B}
raises the answer-coverage CEILING at fixed compute BECAUSE the models fail on DIFFERENT questions
(low error-correlation) -- the validated +0.06..0.13 oracle lever.

COMBO. Let the Pandora controller draw its cheap samples from the 3-MODEL POOL instead of a single
generator. Because pooled draws de-correlate, a CORRECT (high-verifier-score) answer tends to appear
after FEWER draws -> the running-max crosses the stop-reservation sooner AND the oracle ceiling on the
drawn set is higher -> a better accuracy-vs-cost frontier than single-model Pandora.

Routing of the cheap draws across the pool (all use the SAME Weitzman stop/escalate rule + cost model):
  * rr  (round-robin) : interleave the 3 models in descending train-accuracy order (A0,B0,C0,A1,...).
                        Realizes de-correlation even with the *validated independent* controller.
  * res (reservation) : correlated-only. Each model is its own cheap box with reservation z_m from its
                        OWN marginal, discounted by that model's self-agreement (correlated-Pandora,
                        pandora_correlated.py); Weitzman opens the highest-z model, so it naturally
                        ROTATES to a fresh model once the current one's draws start agreeing.
z_cheap uses the POOLED calibrated verifier-score marginal (rr) / per-model marginals (res); the
correlated discount uses the recorded answer TEXT. FAIRNESS: for isolation, the single-model baselines
AND the pool escalate to the SAME strong box (per-fold train-best of {MVT-32B, Lingshu-32B, IV3-38B}),
so the ONLY thing that varies is the cheap leg (single vs pool). Same measured cost model, same 5-fold
held-out cross-fit calibration, same lambda sweep as pandora_controller.py. Datasets = the 3 where all
three generators ran: kvasir_open, radimagenet_open, vqa_rad_open. Cheap-draw cost is 2.0 FLOP-eq for
every model (project's single 7B-equiv cheap cost; InternVL3-8B ~1.14x is folded into that approx).

Reads ONLY existing dumps. CPU-only, no GPU/inference. Launch from repo root:
    python3 src/cascade_methods/pandora_pooling_combo.py
"""
import os, sys, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pandora_controller import (
    REPO, J, C_CHEAP_F, C_STRONG_F, cost_of, load_judge,
    zeta_cheap, zeta_strong, run_pandora, pareto, min_flops_at, LAMS, agg_points, fmt)
from pandora_correlated import (normpred, _rho_from_counts, rho_sequence, zeta_from_t,
                                run_pandora_corr, RHO_FLOOR)

DUMP = "ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_{tag}.json"
GENS = [("lingshu7b", "Lingshu-7B"), ("7b", "MedVLThinker-7B"), ("iv3_8b", "InternVL3-8B")]
GTAGS = [t for t, _ in GENS]
STRONGS = {
    "MVT-32B":     ("ckpts/openvqa/strong",         ["ckpt_{ds}_32b_t0.judge.jsonl", "ckpt_{ds}_32b.judge.jsonl"]),
    "Lingshu-32B": ("ckpts/openvqa/strong_lingshu", ["ckpt_{ds}_lingshu32b_t0.judge.jsonl", "ckpt_{ds}_lingshu32b.judge.jsonl"]),
    "IV3-38B":     ("ckpts/openvqa/internvl3_38b",  ["ckpt_{ds}_iv3_38b_t0.judge.jsonl", "ckpt_{ds}_iv3_38b.judge.jsonl"]),
}
DS3 = ["kvasir_open", "radimagenet_open", "vqa_rad_open"]
MEASURE = "simpson"   # correlated discount for the pool (matches pandora_correlated primary)

# ------------------------------------------------------------------ loading (aligned multi-model)
def load_pool_ds(ds):
    gens = {}
    for tag in GTAGS:
        p = J(DUMP.format(ds=ds, tag=tag))
        if not os.path.exists(p):
            return None
        gens[tag] = {r["idx"]: r for r in json.load(open(p))}
    strongs = {}
    for sn, (sdir, tmpls) in STRONGS.items():
        sj = {}
        for t in tmpls:
            sj = load_judge(J(os.path.join(sdir, t.format(ds=ds))))
            if sj:
                break
        strongs[sn] = sj
    idxsets = [set(gens[t].keys()) for t in GTAGS] + [set(s.keys()) for s in strongs.values()]
    idx = sorted(set.intersection(*idxsets))
    rows = []
    for i in idx:
        mods = {}; ok = True
        for tag in GTAGS:
            r = gens[tag][i]
            sl = [0 if x in (None, -1) else int(x) for x in r["sl"]]
            sc = list(r["scores"]); pr = [normpred(x) for x in r.get("preds", [])]
            n = min(len(sl), len(sc), len(pr) if pr else len(sl))
            if n < 1:
                ok = False; break
            if not pr:
                pr = [f"__{tag}{j}" for j in range(n)]
            mods[tag] = dict(raw=sc[:n], sl=sl[:n], preds=pr[:n])
        if not ok:
            continue
        rows.append(dict(idx=i, mods=mods, strong={sn: int(strongs[sn][i]) for sn in STRONGS}))
    return rows

# ------------------------------------------------------------------ the pool controller (one question)
def run_pool(mods_cal, mods_raw, mods_preds, mods_sl, strong_ok, lam, z_strong,
            pooled_marg, per_marg, order, routing, corr, cache):
    """Pandora over the cheap POOL. mods_* keyed by tag -> per-model sample lists (calibrated score,
    raw score, pred text, judge_ok). order: tag priority (train-accuracy desc). routing: 'rr'|'res'.
    corr: apply the correlation discount. Returns (N cheap draws, escalated, judge_ok)."""
    ptr = {t: 0 for t in GTAGS}
    length = {t: len(mods_raw[t]) for t in GTAGS}
    best_cal = -1e18; best_raw = -1e18; pick = None
    cnt_global = Counter(); cnt_self = {t: Counter() for t in GTAGS}
    ndrawn_self = {t: 0 for t in GTAGS}
    N = 0
    while True:
        avail = [t for t in GTAGS if ptr[t] < length[t]]
        z_cheap = -1e18; chosen = None
        if avail:
            if routing == "rr":
                # next model in round-robin priority among those with samples left
                nxt = min(avail, key=lambda t: (ptr[t], order.index(t)))
                rho = _rho_from_counts(cnt_global.values(), N, MEASURE) if corr else 1.0
                t_eff = lam * C_CHEAP_F / rho
                key = ("rr", round(t_eff, 10))
                z = cache.get(key)
                if z is None:
                    z = zeta_from_t(pooled_marg, t_eff); cache[key] = z
                z_cheap = z; chosen = nxt
            else:  # reservation-routed (correlated): per-model discounted reservation, open the max
                best_z = -1e18; best_t = None
                for t in avail:
                    rho = _rho_from_counts(cnt_self[t].values(), ndrawn_self[t], MEASURE) if corr else 1.0
                    t_eff = lam * C_CHEAP_F / rho
                    key = (t, round(t_eff, 10))
                    z = cache.get(key)
                    if z is None:
                        z = zeta_from_t(per_marg[t], t_eff); cache[key] = z
                    if z > best_z:
                        best_z = z; best_t = t
                z_cheap = best_z; chosen = best_t
        boxtype, maxres = max((("cheap", z_cheap), ("strong", z_strong)), key=lambda x: x[1])
        if best_cal >= maxres:
            break
        if boxtype == "cheap":
            j = ptr[chosen]
            s_cal = mods_cal[chosen][j]; s_raw = mods_raw[chosen][j]
            if s_cal > best_cal: best_cal = s_cal
            if s_raw > best_raw: best_raw = s_raw; pick = (chosen, j)
            p = mods_preds[chosen][j]
            cnt_global[p] += 1; cnt_self[chosen][p] += 1; ndrawn_self[chosen] += 1
            ptr[chosen] += 1; N += 1
        else:
            return N, 1, int(strong_ok)
    if pick is None:
        return N, 0, 0
    return N, 0, int(mods_sl[pick[0]][pick[1]])

# ------------------------------------------------------------------ per-dataset eval (cross-fit)
def eval_pool_ds(rows, seed=0, nfold=5):
    n = len(rows)
    Nmax = {t: min(len(r["mods"][t]["raw"]) for r in rows) for t in GTAGS}
    strong_names = list(STRONGS)

    # base references (per-model bo8 + single/pool oracle ceilings @ full 8-each)
    base = dict(n=n, Nmax=Nmax)
    for t in GTAGS:
        bo8 = np.array([r["mods"][t]["sl"][int(np.argmax(r["mods"][t]["raw"]))] for r in rows], float)
        base[f"bo8_{t}"] = float(bo8.mean())
    # single oracle@8 per model, pool oracle (8 each) -- realized coverage on the drawn labels
    for t in GTAGS:
        base[f"oracle8_{t}"] = float(np.mean([max(r["mods"][t]["sl"]) for r in rows]))
    base["oracle_pool_8each"] = float(np.mean([max(max(r["mods"][t]["sl"]) for t in GTAGS) for r in rows]))
    for sn in strong_names:
        base[f"strong_{sn}"] = float(np.mean([r["strong"][sn] for r in rows]))

    methods = ["pool_indep_rr", "pool_corr_rr", "pool_corr_res"] + \
              [f"single_{t}_indep" for t in GTAGS] + [f"single_{t}_corr" for t in GTAGS]
    acc = {m: {float(l): dict(N=np.zeros(n), esc=np.zeros(n), ok=np.zeros(n)) for l in LAMS} for m in methods}
    strong_used = np.zeros(n, dtype=object)
    strong_deployed_ok = np.zeros(n, float)
    # per-model diversity rho sequences (depend only on the pred prefix, not lambda/fold)
    rho_seq_m = {t: [rho_sequence(rows[i]["mods"][t]["preds"][:Nmax[t]], MEASURE) for i in range(n)] for t in GTAGS}

    kf = KFold(nfold if n >= nfold else 2, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        # per-fold train-best strong (held-out choice); q_strong from train
        sc_strong = {sn: np.mean([rows[i]["strong"][sn] for i in tr]) for sn in strong_names}
        sname = max(strong_names, key=lambda s: sc_strong[s])
        q = float(sc_strong[sname])
        for i in te:
            strong_used[i] = sname
            strong_deployed_ok[i] = rows[i]["strong"][sname]   # correctness of the chosen strong on held-out i

        # pooled calibration (all models' train samples) + per-model calibration
        xs_all = np.concatenate([np.asarray(rows[i]["mods"][t]["raw"][:Nmax[t]], float) for i in tr for t in GTAGS])
        ys_all = np.concatenate([np.asarray(rows[i]["mods"][t]["sl"][:Nmax[t]], float) for i in tr for t in GTAGS])
        iso_pool = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); iso_pool.fit(xs_all, ys_all)
        pooled_marg = iso_pool.predict(xs_all)
        iso_m = {}
        per_marg = {}          # model t scores under model-t's OWN isotonic (for single-model baselines)
        per_marg_pool = {}     # model t scores under the SHARED pooled isotonic (for res-routed pool: same value scale as best_cal)
        for t in GTAGS:
            xs = np.concatenate([np.asarray(rows[i]["mods"][t]["raw"][:Nmax[t]], float) for i in tr])
            ys = np.concatenate([np.asarray(rows[i]["mods"][t]["sl"][:Nmax[t]], float) for i in tr])
            im = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); im.fit(xs, ys)
            iso_m[t] = im; per_marg[t] = im.predict(xs); per_marg_pool[t] = iso_pool.predict(xs)
        # train-accuracy order for round-robin priority
        order = sorted(GTAGS, key=lambda t: -np.mean([np.mean(rows[i]["mods"][t]["sl"]) for i in tr]))

        # precompute test calibrations
        te_cal_pool = {i: {t: iso_pool.predict(np.asarray(rows[i]["mods"][t]["raw"][:Nmax[t]], float)) for t in GTAGS} for i in te}
        te_cal_m = {i: {t: iso_m[t].predict(np.asarray(rows[i]["mods"][t]["raw"][:Nmax[t]], float)) for t in GTAGS} for i in te}
        cache_pool = {}                      # run_pool: keys are tuples ('rr',t) / (tag,t) -> marginal-safe
        cache_single = {t: {} for t in GTAGS}  # run_pandora_corr keys by bare t -> needs one cache per marginal
        for l in LAMS:
            zs = zeta_strong(q, l)
            zc_single = {t: zeta_cheap(per_marg[t], l) for t in GTAGS}
            for i in te:
                r = rows[i]; sok = r["strong"][sname]
                mraw = {t: r["mods"][t]["raw"][:Nmax[t]] for t in GTAGS}
                mpr = {t: r["mods"][t]["preds"][:Nmax[t]] for t in GTAGS}
                msl = {t: r["mods"][t]["sl"][:Nmax[t]] for t in GTAGS}
                mcalp = te_cal_pool[i]; mcalm = te_cal_m[i]
                # ---- pool methods (pooled calibration for cal scores, pooled/per marginal for z) ----
                for mname, routing, corr in [("pool_indep_rr", "rr", False),
                                             ("pool_corr_rr", "rr", True),
                                             ("pool_corr_res", "res", True)]:
                    N, e, ok = run_pool(mcalp, mraw, mpr, msl, sok, l, zs,
                                        pooled_marg, per_marg_pool, order, routing, corr, cache_pool)
                    d = acc[mname][float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok
                # ---- single-model baselines (own calibration, SAME strong) ----
                for t in GTAGS:
                    N, e, ok = run_pandora(mraw[t], mcalm[t], msl[t], sok, zc_single[t], zs)
                    d = acc[f"single_{t}_indep"][float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok
                    N, e, ok = run_pandora_corr(mraw[t], mcalm[t], rho_seq_m[t][i], msl[t], sok,
                                                per_marg[t], l, zs, cache_single[t])
                    d = acc[f"single_{t}_corr"][float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok

    base["strong_used"] = Counter(strong_used.tolist())
    base["strong_deployed_acc"] = float(strong_deployed_ok.mean())
    aggm = {m: {l: dict(N=float(d["N"].mean()), esc=float(d["esc"].mean()),
                        acc=float(d["ok"].mean()), n=n) for l, d in acc[m].items()} for m in methods}
    return base, aggm

# ------------------------------------------------------------------ reporting
def build_out(base, aggm):
    n = base["n"]
    fr = {m: pareto(agg_points(v)) for m, v in aggm.items()}
    strong_used = base["strong_used"]
    strong_acc = base["strong_deployed_acc"]   # accuracy of the per-fold train-best strong actually escalated to
    best_single_bo8 = max(base[f"bo8_{t}"] for t in GTAGS)
    out = dict(n=n, best_single_bo8=best_single_bo8,
               oracle8_single_best=max(base[f"oracle8_{t}"] for t in GTAGS),
               oracle_pool_8each=base["oracle_pool_8each"],
               strong_acc_by_model={sn: base[f"strong_{sn}"] for sn in STRONGS},
               strong_deployed_acc=strong_acc,
               strong_used={k: v for k, v in strong_used.items()},
               bo8_by_model={t: base[f"bo8_{t}"] for t in GTAGS},
               frontiers={m: [dict(p) for p in v] for m, v in fr.items()})
    # iso targets: match the best single-model bo8, and match the deployed strong accuracy
    iso = {}
    for tname, tacc in [("iso_bo8", best_single_bo8), ("iso_strong", strong_acc)]:
        iso[tname] = dict(target=tacc)
        for m in aggm:
            iso[tname][m] = min_flops_at(fr[m], tacc)
        # best single-model op-point (min FLOPs over the 3 singles, per controller)
        for ctrl in ("indep", "corr"):
            cands = [iso[tname][f"single_{t}_{ctrl}"] for t in GTAGS if iso[tname][f"single_{t}_{ctrl}"]]
            iso[tname][f"best_single_{ctrl}"] = min(cands, key=lambda p: p["flops"]) if cands else None
    out["iso"] = iso
    return out

POOL_METHODS = ["pool_indep_rr", "pool_corr_rr", "pool_corr_res"]

def print_block(title, out):
    print(f"\n{'='*120}\n{title}   n={out['n']}")
    print(f"  bo8 by model: " + "  ".join(f"{t}={out['bo8_by_model'][t]:.3f}" for t in GTAGS) +
          f"   best-single-bo8={out['best_single_bo8']:.3f}")
    print(f"  oracle@8 single-best={out['oracle8_single_best']:.3f}   POOL oracle(8 each)={out['oracle_pool_8each']:.3f}"
          f"   (coverage lever = {out['oracle_pool_8each']-out['oracle8_single_best']:+.3f})")
    print(f"  strong by model: " + "  ".join(f"{sn}={out['strong_acc_by_model'][sn]:.3f}" for sn in STRONGS) +
          f"   deployed={out['strong_deployed_acc']:.3f}  used(folds): {out['strong_used']}")
    for tname in ("iso_bo8", "iso_strong"):
        it = out["iso"][tname]; tgt = it["target"]
        print(f"  -- iso-accuracy @ {tname.replace('iso_','match ')} (acc>={tgt:.3f}) : cheapest op-point --")
        for m in POOL_METHODS:
            tag = "  <== POOL" if m.startswith("pool") else ""
            print(f"       {m:<22}{fmt(it[m])}{tag}")
        print(f"       {'best_single_indep':<22}{fmt(it['best_single_indep'])}  (validated single-model Pandora)")
        print(f"       {'best_single_corr':<22}{fmt(it['best_single_corr'])}  (correlated single-model)")

def summarize(all_ds):
    methods = POOL_METHODS + ["best_single_indep", "best_single_corr"]
    SUMMARY = {}
    for tname, tlabel in [("iso_bo8", "match per-domain best-single verifier-bo8 accuracy"),
                          ("iso_strong", "match per-domain deployed-strong accuracy")]:
        agg = {}
        for meth in methods:
            hit = [(o["n"], o["iso"][tname][meth]) for _, o in all_ds if o["iso"][tname].get(meth) is not None]
            Ntot = sum(nn for nn, _ in hit)
            if Ntot == 0:
                agg[meth] = None; continue
            agg[meth] = dict(datasets_covered=len(hit), datasets_total=len(all_ds),
                             flops=sum(nn * p["flops"] for nn, p in hit) / Ntot,
                             meanN=sum(nn * p["meanN"] for nn, p in hit) / Ntot,
                             esc=sum(nn * p["esc"] for nn, p in hit) / Ntot,
                             lat_seq=sum(nn * p["lat_seq"] for nn, p in hit) / Ntot)
        SUMMARY[tname] = dict(label=tlabel, methods=agg)
        print(f"\n  [{tname}] {tlabel}   (n-weighted over {len(all_ds)} datasets)")
        for meth in methods:
            a = agg[meth]
            if a is None:
                print(f"    {meth:<22} n/a"); continue
            tag = "  <== POOL" if meth.startswith("pool") else "  (single-model baseline)"
            print(f"    {meth:<20} cover {a['datasets_covered']}/{a['datasets_total']}  "
                  f"FLOPs={a['flops']:5.2f}  meanN={a['meanN']:.2f}  esc={a['esc']*100:3.0f}%  "
                  f"Lseq={a['lat_seq']:5.0f}ms{tag}")
        # pool vs best-single deltas
        for pm in POOL_METHODS:
            a = agg[pm]; b = agg["best_single_indep"]
            if a and b:
                dF = (1 - a["flops"] / b["flops"]) * 100
                print(f"      {pm} vs best_single_indep: FLOPs {b['flops']:.2f}->{a['flops']:.2f} ({dF:+.0f}%)")
    return SUMMARY

def main():
    print("#" * 120)
    print("PANDORA x POOLING COMBO - Pandora controller drawing cheap samples from the 3-model cross-model pool")
    print("  pool={Lingshu-7B, MedVLThinker-7B, InternVL3-8B}; same strong / cost / cross-fit / lambda as pandora_controller")
    print("#" * 120)
    DUMP_OUT = {}
    all_ds = []
    for ds in DS3:
        rows = load_pool_ds(ds)
        if not rows:
            print(f"\n### {ds}: no aligned data"); continue
        base, aggm = eval_pool_ds(rows)
        out = build_out(base, aggm)
        DUMP_OUT[ds] = out
        all_ds.append((ds, out))
        print_block(ds, out)

    print(f"\n{'#'*120}\n#####  HEADLINE: per-domain-tuned aggregate (POOL vs best single-model Pandora)  #####\n{'#'*120}")
    DUMP_OUT["SUMMARY_per_domain_tuned"] = summarize(all_ds)
    DUMP_OUT["_meta"] = dict(datasets=DS3, generators=[nm for _, nm in GENS], strongs=list(STRONGS),
                             measure=MEASURE, cheap_cost_flop=C_CHEAP_F, strong_cost_flop=C_STRONG_F,
                             note="single-model baselines and pool escalate to the SAME per-fold train-best strong")

    outp = J("results/cascade_methods/artifacts/pandora_pooling_combo.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(DUMP_OUT, open(outp, "w"), indent=1,
              default=lambda o: (float(o) if isinstance(o, np.floating) else
                                 (int(o) if isinstance(o, np.integer) else
                                  (o.tolist() if isinstance(o, np.ndarray) else o))))
    print(f"\n[dump] {outp}")

if __name__ == "__main__":
    main()
