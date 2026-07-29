#!/usr/bin/env python3
"""
pandora_controller.py - OFFLINE test of the Pandora's-Box adaptive candidate+escalation controller
(idea C1 / TOP-5 #1 in results/cascade_methods/METHOD_IDEAS_BACKLOG.md). CPU-only, no GPU/vLLM/eval.

WHAT IT IS (Weitzman, Econometrica 1979). Each question is a search over "boxes" to open:
  * cheap box #k  = draw one more 7B sample.  inspection cost c_cheap FLOP-eq (GEN7+VER7);
                    realized reward = the verifier's calibrated P(correct) of that sample.
  * strong box    = escalate to the 32B.        inspection cost c_strong FLOP-eq (GEN32);
                    realized reward = q_strong = calibrated P(strong correct)  (deterministic box,
                    because at deploy time we cannot inspect the strong answer's correctness -- once
                    we pay for it we must output it).
Weitzman's reservation value zeta_i solves  lambda*c_i = E[(V_i - zeta_i)^+]  (lambda = the accuracy<->
compute exchange rate, the single frontier knob). Optimal rule: open the unopened box of highest zeta;
STOP once the best realized reward so far >= the highest remaining zeta; then output the best box.
  - cheap boxes are i.i.d. draws from the marginal verifier-score distribution  => one shared zeta_cheap
    (a STOP-DRAWING threshold; unifies "how many cheap samples").
  - the deterministic strong box has  zeta_strong = q_strong - lambda*c_strong  (an ESCALATION threshold;
    escalate iff the best cheap reward < zeta_strong; unifies "when to defer").
ONE lambda yields BOTH thresholds -> a single optimal rule that replaces (adaptive-N + confidence gate).

OFFLINE TEST. We reuse the existing per-sample dumps (schema from open_bestofN_adaptive.py /
open_gate_efficiency.py): per question, scores[8] (verifier P(correct)), sl[8] (per-sample judge_ok),
and the strong model's judge_ok. We draw samples in RECORDED order, act only on OBSERVABLE (calibrated)
verifier scores (no peeking at correctness), and SCORE the controller by true judge_ok. Reservation
values use HELD-OUT calibration: 5-fold cross-fit isotonic score->P(correct), the cheap-score
distribution, and q_strong on the OTHER folds. lambda is swept to trace the accuracy-vs-cost frontier.

Cost model = the project's canonical MEASURED batch-1 model (open_gate_efficiency.py):
  GEN7=(347ms,45.8J,1.0 FLOP-7Beq)  VER7=(175ms,25.3J,1.0)  GEN32=(665ms,127.0J,4.57).
  one cheap draw = GEN7+VER7 (generate+verify);  escalation = GEN32.  FLOPs in 7B-forward-equivalents.
  Adaptive draws are inherently SEQUENTIAL (draw->check->draw) => latency is batch-1/sequential; a fixed
  best-of-N can batch its N draws, so we report both lat_seq and lat_bat and flag the trade honestly.

Baselines compared on BOTH axes: always-32B; verifier-bo8 (fixed N=8, no escalation); verifier-bo8 +
confidence gate (sweep tau); fixed adaptive-N (open_bestofN_adaptive: running-max stop + escalate).

Launch from repo root:  python3 src/cascade_methods/pandora_controller.py
"""
import os, json
import numpy as np
from collections import defaultdict
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)

# ------------------------------------------------------------------ cost model (measured, batch-1)
GEN7 = (347.0, 45.8, 1.0)      # (latency_ms, energy_J, FLOPs in 7B-forward-equivalents)
VER7 = (175.0, 25.3, 1.0)
GEN32 = (665.0, 127.0, 4.57)
C_CHEAP_F = GEN7[2] + VER7[2]   # 2.0  FLOP-eq per cheap draw (generate + verify)
C_STRONG_F = GEN32[2]           # 4.57 FLOP-eq per escalation

def cost_of(meanN, esc):
    """FLOPs / energy / latency for a policy that draws meanN cheap samples and escalates frac 'esc'.
    lat_seq: batch-1 / adaptive-sequential (the honest number for adaptive drawing).
    lat_bat: if the N cheap draws could be batched in parallel (only valid for FIXED N)."""
    flops = meanN * C_CHEAP_F + esc * C_STRONG_F
    energy = meanN * (GEN7[1] + VER7[1]) + esc * GEN32[1]
    lat_seq = meanN * (GEN7[0] + VER7[0]) + esc * GEN32[0]
    lat_bat = (GEN7[0] + VER7[0] if meanN > 0 else 0.0) + esc * GEN32[0]
    return dict(flops=flops, energy=energy, lat_seq=lat_seq, lat_bat=lat_bat)

# ------------------------------------------------------------------ data (reuse existing dumps)
OPEN_DSETS = ["kvasir_open", "vqa_rad_open", "pathvqa_open", "radimagenet_open"]
ADAPTER = "ckpts/train/lora_verifier_pooled4"
FAM = {
    "Lingshu":      dict(vtag="lingshu7b", sdir="ckpts/openvqa/strong_lingshu",
                         sj=["ckpt_{ds}_lingshu32b_t0.judge.jsonl", "ckpt_{ds}_lingshu32b.judge.jsonl"]),
    "MedVLThinker": dict(vtag="7b", sdir="ckpts/openvqa/strong",
                         sj=["ckpt_{ds}_32b_t0.judge.jsonl", "ckpt_{ds}_32b.judge.jsonl"]),
    "InternVL3":    dict(vtag="iv3_8b", sdir="ckpts/openvqa/internvl3_38b",
                         sj=["ckpt_{ds}_iv3_38b_t0.judge.jsonl", "ckpt_{ds}_iv3_38b.judge.jsonl"]),
}

def load_judge(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def load_family(cfg):
    """returns list of dicts: ds, scores(raw list), sl(list 0/1), strong(0/1)."""
    rows = []
    for ds in OPEN_DSETS:
        dp = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{cfg['vtag']}.json"))
        if not os.path.exists(dp): continue
        dump = json.load(open(dp))
        sj = {}
        for t in cfg["sj"]:
            sj = load_judge(J(os.path.join(cfg["sdir"], t.format(ds=ds))))
            if sj: break
        for r in dump:
            i = r["idx"]
            if i not in sj: continue
            sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
            sc = list(r["scores"]); n = min(len(sl), len(sc))
            if n < 1: continue
            rows.append(dict(ds=ds, scores=sc[:n], sl=sl[:n], strong=int(sj[i])))
    return rows

# ------------------------------------------------------------------ Weitzman reservation values
def zeta_cheap(scores_cal, lam, c=C_CHEAP_F):
    """solve  lam*c = mean_v (v - z)^+  for z. g(z) is continuous, strictly decreasing for z<max(v).
    z=max(v) at lam=0; for large lam, z = mean(v) - lam*c (support-linear tail)."""
    v = np.asarray(scores_cal, float); t = lam * c
    if t <= 0.0: return float(v.max())
    if t >= v.mean() - v.min():          # target beyond the linear tail -> closed form
        return float(v.mean() - t)
    lo, hi = float(v.min()), float(v.max())
    for _ in range(80):
        mid = 0.5 * (lo + hi); g = np.maximum(v - mid, 0.0).mean()
        if g > t: lo = mid                # g decreasing -> need larger z
        else: hi = mid
    return 0.5 * (lo + hi)

def zeta_strong(q_strong, lam, c=C_STRONG_F):
    """deterministic box: lam*c = (q_strong - z)^+  =>  z = q_strong - lam*c."""
    return q_strong - lam * c

# ------------------------------------------------------------------ the controller (one question)
def run_pandora(scores_raw, scores_cal, sl, strong_ok, z_cheap, z_strong):
    """Weitzman policy. Draw cheap samples in recorded order; act on calibrated verifier scores.
    Returns (N cheap drawn, escalated 0/1, final judge_ok)."""
    Nmax = len(scores_raw)
    best_cal = -1e18; best_raw = -1e18; pick = -1; k = 0
    while True:
        res = []
        if k < Nmax: res.append(("cheap", z_cheap))
        res.append(("strong", z_strong))      # strong box always available until opened (then we return)
        boxtype, maxres = max(res, key=lambda x: x[1])
        if best_cal >= maxres:                 # STOP: best-so-far beats every remaining reservation
            break
        if boxtype == "cheap":
            s_cal = scores_cal[k]; s_raw = scores_raw[k]
            if s_cal > best_cal: best_cal = s_cal
            if s_raw > best_raw: best_raw = s_raw; pick = k
            k += 1
        else:                                  # open strong box == escalate (commit to strong answer)
            return k, 1, int(strong_ok)
    return k, 0, int(sl[pick] if pick >= 0 else 0)

# ------------------------------------------------------------------ fixed adaptive-N baseline (open_bestofN_adaptive)
def run_adaptiveN(scores, sl, strong_ok, tau_stop, tau_esc):
    """running-max stop then escalate-if-unsure (draws in recorded order)."""
    Nmax = len(scores); stop_k = Nmax
    for k in range(1, Nmax + 1):
        if max(scores[:k]) >= tau_stop:
            stop_k = k; break
    runmax = max(scores[:stop_k])
    if runmax < tau_esc:
        return stop_k, 1, int(strong_ok)
    j = int(np.argmax(scores[:stop_k]))
    return stop_k, 0, int(sl[j])

# ------------------------------------------------------------------ frontier / Pareto helpers
def pareto(points):
    """points: list of dict(acc, flops, ...). keep non-dominated (max acc, min flops)."""
    pts = sorted(points, key=lambda p: (p["flops"], -p["acc"]))
    out = []; best_acc = -1.0
    for p in pts:
        if p["acc"] > best_acc + 1e-12:
            out.append(p); best_acc = p["acc"]
    return out

def min_flops_at(frontier, target_acc, tol=0.003):
    """cheapest (min-FLOPs) operating point that reaches >= target_acc - tol."""
    ok = [p for p in frontier if p["acc"] >= target_acc - tol]
    if not ok: return None
    return min(ok, key=lambda p: p["flops"])

# ------------------------------------------------------------------ per-dataset evaluation (cross-fit)
LAMS = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 90)])
GATE_TAUS = np.linspace(0.0, 1.0, 101)
ANS_STOP = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]
ANS_ESC = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def eval_dataset(rows, seed=0, nfold=5):
    """rows for ONE dataset. Cross-fit calibration; return per-question outcome arrays for every method."""
    n = len(rows)
    raw = [r["scores"] for r in rows]
    sl = [r["sl"] for r in rows]
    strong = np.array([r["strong"] for r in rows], float)
    Nmax = min(len(r["scores"]) for r in rows)

    # ---- static baselines (no calibration needed) ----
    bo8_ok = np.array([sl[i][int(np.argmax(raw[i][:Nmax]))] for i in range(n)], float)
    oracle_ok = np.array([max(sl[i][:Nmax]) for i in range(n)], float)
    base = dict(strong_acc=float(strong.mean()), bo8_acc=float(bo8_ok.mean()),
                oracle_acc=float(oracle_ok.mean()), n=n, Nmax=Nmax)

    # ---- PANDORA frontier: for each lambda, thresholds come from the TRAIN fold, applied held-out ----
    # cross-fit isotonic score->P(correct), the cheap-score distribution, and q_strong (no leakage).
    kf = KFold(nfold if n >= nfold else 2, shuffle=True, random_state=seed)
    pan = {float(l): dict(N=np.zeros(n), esc=np.zeros(n), ok=np.zeros(n)) for l in LAMS}
    for tr, te in kf.split(np.arange(n)):
        xs = np.concatenate([raw[i][:Nmax] for i in tr])
        ys = np.concatenate([np.array(sl[i][:Nmax], float) for i in tr])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0); iso.fit(xs, ys)
        train_pool = iso.predict(xs)                       # calibrated cheap-score distribution (train)
        q = float(strong[tr].mean())
        te_cal = {i: iso.predict(np.asarray(raw[i][:Nmax], float)) for i in te}
        for l in LAMS:
            zc = zeta_cheap(train_pool, l); zs = zeta_strong(q, l)
            for i in te:
                N, e, ok = run_pandora(raw[i][:Nmax], te_cal[i], sl[i][:Nmax], strong[i], zc, zs)
                d = pan[float(l)]; d["N"][i] = N; d["esc"][i] = e; d["ok"][i] = ok

    pan_agg = {l: dict(N=float(d["N"].mean()), esc=float(d["esc"].mean()), acc=float(d["ok"].mean()), n=n)
               for l, d in pan.items()}

    # ---- baseline frontiers (swept knobs, evaluated on full data = optimistic 'oracle-tau') ----
    gate = {}                                              # verifier-bo8 + confidence gate
    vmax = np.array([max(raw[i][:Nmax]) for i in range(n)], float)
    for tau in GATE_TAUS:
        esc = (vmax < tau)
        ok = np.where(esc, strong, bo8_ok)
        gate[float(tau)] = dict(N=float(Nmax), esc=float(esc.mean()), acc=float(ok.mean()), n=n)
    adn = {}                                               # fixed adaptive-N
    for ts in ANS_STOP:
        for tesc in ANS_ESC:
            Ns = np.zeros(n); es = np.zeros(n); oks = np.zeros(n)
            for i in range(n):
                N, e, ok = run_adaptiveN(raw[i][:Nmax], sl[i][:Nmax], strong[i], ts, tesc)
                Ns[i] = N; es[i] = e; oks[i] = ok
            adn[(ts, tesc)] = dict(N=float(Ns.mean()), esc=float(es.mean()), acc=float(oks.mean()), n=n)

    return base, pan_agg, gate, adn

# ------------------------------------------------------------------ assemble frontiers into points
def agg_points(agg):
    """agg: {knob: {N,esc,acc,n}} -> list of point dicts with cost attached."""
    pts = []
    for k, d in agg.items():
        pts.append(dict(knob=k, acc=d["acc"], meanN=d["N"], esc=d["esc"], **cost_of(d["N"], d["esc"])))
    return pts

def pool_bases(bases):
    """n-weighted pooling of per-dataset base metrics."""
    N = sum(b["n"] for b in bases)
    w = lambda key: sum(b["n"] * b[key] for b in bases) / N
    return dict(n=N, Nmax=bases[0]["Nmax"], strong_acc=w("strong_acc"), bo8_acc=w("bo8_acc"), oracle_acc=w("oracle_acc"))

def pool_aggs(aggs):
    """n-weighted pooling of a list of per-dataset {knob:{N,esc,acc,n}} dicts (shared knobs)."""
    out = {}
    keys = list(aggs[0].keys())
    for k in keys:
        N = sum(a[k]["n"] for a in aggs)
        out[k] = dict(N=sum(a[k]["n"] * a[k]["N"] for a in aggs) / N,
                      esc=sum(a[k]["n"] * a[k]["esc"] for a in aggs) / N,
                      acc=sum(a[k]["n"] * a[k]["acc"] for a in aggs) / N, n=N)
    return out

# ------------------------------------------------------------------ reporting
def build_out(base, pan_agg, gate, adn):
    Nmax = base["Nmax"]
    Pf, Gf, Af = pareto(agg_points(pan_agg)), pareto(agg_points(gate)), pareto(agg_points(adn))
    always32 = dict(acc=base["strong_acc"], meanN=0.0, esc=1.0, **cost_of(0.0, 1.0))
    bo8 = dict(acc=base["bo8_acc"], meanN=float(Nmax), esc=0.0, **cost_of(Nmax, 0.0))

    out = dict(n=base["n"], Nmax=Nmax, strong_acc=base["strong_acc"], bo8_acc=base["bo8_acc"],
               oracle_acc=base["oracle_acc"], always32=always32, bo8=bo8,
               pandora_frontier=[{k: v for k, v in p.items()} for p in Pf],
               gate_frontier=[{k: v for k, v in p.items()} for p in Gf],
               adaptiveN_frontier=[{k: v for k, v in p.items()} for p in Af])

    # iso-accuracy comparisons at two targets: match verifier-bo8, and match always-32B
    iso = {}
    for tname, tacc in [("iso_bo8", base["bo8_acc"]), ("iso_strong", base["strong_acc"])]:
        iso[tname] = dict(target=tacc,
                          pandora=min_flops_at(Pf, tacc), gate=min_flops_at(Gf, tacc),
                          adaptiveN=min_flops_at(Af, tacc),
                          bo8=(bo8 if bo8["acc"] >= tacc - 3e-3 else None),
                          always32=(always32 if always32["acc"] >= tacc - 3e-3 else None))
    out["iso"] = iso
    return out

def fmt(p):
    if p is None: return "     n/a"
    return f"acc={p['acc']:.3f} N={p.get('meanN',0):.2f} esc={p.get('esc',0)*100:3.0f}% F={p['flops']:5.2f} E={p['energy']:5.0f}J Lseq={p['lat_seq']:5.0f}ms"

def print_block(title, out):
    print(f"\n{'='*118}\n{title}   n={out['n']}  (verifier best-of-{out['Nmax']})")
    print(f"  greedy/strong/bo8/oracle acc:  strong={out['strong_acc']:.3f}  bo8={out['bo8_acc']:.3f}  oracle@{out['Nmax']}={out['oracle_acc']:.3f}")
    print(f"  {'always-32B':<22}{fmt(out['always32'])}")
    print(f"  {'verifier-bo'+str(out['Nmax'])+' (fixedN)':<22}{fmt(out['bo8'])}")
    for tname in ("iso_bo8", "iso_strong"):
        it = out["iso"][tname]; tgt = it["target"]
        print(f"  -- iso-accuracy @ {tname.replace('iso_','match ')} (acc>={tgt:.3f}) : cheapest op-point per method --")
        print(f"       {'PANDORA (ours, held-out)':<28}{fmt(it['pandora'])}")
        print(f"       {'bo'+str(out['Nmax'])+'+gate (oracle-tau)':<28}{fmt(it['gate'])}")
        print(f"       {'adaptive-N (oracle-tau)':<28}{fmt(it['adaptiveN'])}")

def main():
    print("#" * 118)
    print("PANDORA'S-BOX ADAPTIVE CONTROLLER (Weitzman) - offline both-axes test on open-text verifier dumps")
    print("  cost: cheap draw = GEN7+VER7 = 2.0 FLOP-eq ; escalate = GEN32 = 4.57 FLOP-eq ; latency batch-1/sequential")
    print("  PANDORA = held-out cross-fit thresholds (no peek). baselines' tau swept on full data (optimistic).")
    print("#" * 118)
    DUMP = {}
    for fam, cfg in FAM.items():
        rows = load_family(cfg)
        if not rows:
            print(f"\n### {fam}: no data"); continue
        by_ds = defaultdict(list)
        for r in rows: by_ds[r["ds"]].append(r)
        fam_out = {}
        print(f"\n{'#'*118}\n############  FAMILY: {fam}  ({len(rows)} q over {len(by_ds)} datasets)  ############")
        per_ds_eval = {}
        for ds in OPEN_DSETS:
            if ds not in by_ds: continue
            base, pan_agg, gate, adn = eval_dataset(by_ds[ds])
            per_ds_eval[ds] = (base, pan_agg, gate, adn)
            out = build_out(base, pan_agg, gate, adn)
            fam_out[ds] = out
            print_block(f"{fam}/{ds}", out)
        # POOLED (per-domain): concat per-dataset held-out outcomes (deploy per-domain, aggregate) = the FAIR pooled
        evals = list(per_ds_eval.values())
        pooled_base = pool_bases([e[0] for e in evals])
        out_pd = build_out(pooled_base, pool_aggs([e[1] for e in evals]),
                           pool_aggs([e[2] for e in evals]), pool_aggs([e[3] for e in evals]))
        fam_out["POOLED_per_domain"] = out_pd
        print_block(f"{fam}/POOLED (per-domain calibration)", out_pd)
        # POOLED (domain-blind): one calibration over the mixture = pessimistic, shows the heterogeneity cost
        base, pan_agg, gate, adn = eval_dataset(rows)
        out_db = build_out(base, pan_agg, gate, adn)
        fam_out["POOLED_domain_blind"] = out_db
        print_block(f"{fam}/POOLED (domain-blind calibration)", out_db)
        DUMP[fam] = fam_out

    # ------------------------------------------------------------------ per-domain-tuned SUMMARY
    # Realistic deployment: calibrate + pick the operating point PER DOMAIN to hit an accuracy target,
    # then n-weight-aggregate the cost. Every method gets per-domain tuning (fair; baselines still peek).
    print(f"\n{'#'*118}\n#####  HEADLINE: per-domain-tuned aggregate (deploy+tune per domain, then aggregate cost)  #####\n{'#'*118}")
    SUMMARY = {}
    all_ds = []
    for fam in DUMP:
        for ds, out in DUMP[fam].items():
            if ds.startswith("POOLED"): continue
            all_ds.append((f"{fam}/{ds}", out))
    for tname, tlabel in [("iso_bo8", "match per-domain verifier-bo8 accuracy"),
                          ("iso_strong", "match per-domain always-32B accuracy")]:
        agg = {}
        for meth in ("pandora", "gate", "adaptiveN"):
            hit = [(o["n"], o["iso"][tname][meth]) for _, o in all_ds if o["iso"][tname][meth] is not None]
            Ntot = sum(n for n, _ in hit)
            if Ntot == 0:
                agg[meth] = None; continue
            agg[meth] = dict(
                datasets_covered=len(hit), datasets_total=len(all_ds),
                flops=sum(n * p["flops"] for n, p in hit) / Ntot,
                energy=sum(n * p["energy"] for n, p in hit) / Ntot,
                meanN=sum(n * p["meanN"] for n, p in hit) / Ntot,
                esc=sum(n * p["esc"] for n, p in hit) / Ntot,
                lat_seq=sum(n * p["lat_seq"] for n, p in hit) / Ntot)
        # fixed references (always reach their own defn): bo8 F=16 ; always-32B F=4.57
        SUMMARY[tname] = dict(label=tlabel, methods=agg,
                              ref_bo8_flops=16.0, ref_always32_flops=C_STRONG_F)
        print(f"\n  [{tname}] {tlabel}   (n-weighted over {len(all_ds)} datasets; verifier-bo8 baseline FLOPs=16.0)")
        for meth in ("pandora", "gate", "adaptiveN"):
            a = agg[meth]
            if a is None: print(f"    {meth:<26} n/a"); continue
            tag = "  <== OURS (held-out)" if meth == "pandora" else "  (oracle-tau, optimistic)"
            print(f"    {meth:<12} cover {a['datasets_covered']}/{a['datasets_total']}  "
                  f"FLOPs={a['flops']:5.2f} (-{(1-a['flops']/16.0)*100:2.0f}% vs bo8)  "
                  f"energy={a['energy']:4.0f}J  meanN={a['meanN']:.2f}  esc={a['esc']*100:3.0f}%  "
                  f"Lseq={a['lat_seq']:5.0f}ms{tag}")
    DUMP["SUMMARY_per_domain_tuned"] = SUMMARY

    outp = J("results/cascade_methods/artifacts/pandora_controller.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(DUMP, open(outp, "w"), indent=1, default=lambda o: (float(o) if isinstance(o, (np.floating,)) else
                                                                  (o.tolist() if isinstance(o, np.ndarray) else o)))
    print(f"\n[dump] {outp}")

if __name__ == "__main__":
    main()
