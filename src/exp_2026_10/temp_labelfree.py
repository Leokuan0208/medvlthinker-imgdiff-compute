#!/usr/bin/env python3
"""temp_labelfree.py -- choosing the sampling temperature WITHOUT knowing which benchmark a question is from.

Owed since the 2026-10-19 deck ("temperature without the dataset label: per-modality T; per-question T or N
from the probe's scores"). The final method's frozen temperature map (frozen_temperature_map_2026-10-15.json)
needs the benchmark's identity, which a deployed system does not have. This measures label-free alternatives
under the FINAL protocol: frozen final probes (ckpts/train/final_probe_2026-10-16/fold{0,1}), two-fold image
cross-fitting, rank_vote readout, the existing N=8 pools at T = 0.2 / 0.4 / 0.7 / 1.0 for all 8 benchmarks.
No probe is refitted and no new sample is drawn: a policy only decides WHICH of the stored samples it uses.

POLICIES (all 8 samples per question unless stated -- the same budget as the deck's method):
  fixed_T{02,04,07,10}  all 8 samples at one temperature            (fixed_T07 = the deck's headline method)
  tmap                  the frozen per-benchmark map                  (= FINAL_METHOD's method; needs the label)
  mix8                  2 samples at each temperature                 (tested 2026-10-18; label-free)
  explore_if_unsure     4 at T=0.7; if the probe's confidence in its stage-1 pick is below tau, the next 4 at
                        T=1.0, else samples 5-8 at T=0.7. PRE-SPECIFIED primary label-free policy. tau is
                        cross-fitted: chosen on image-fold 0 (best training-judge gain), applied to fold 1, and
                        vice versa, so no question is scored with a tau chosen on itself.
  best_pair_xfit        the same two-stage rule, but the stage-2 temperature for unsure AND for confident
                        questions are chosen too (4 x 4 pairs x tau), all by the same cross-fitting.
                        EXPLORATORY -- many more choices, so expect optimism even with cross-fitting.
  oracle_T              per question, the temperature whose 8-sample pick is judged correct (upper bound only)
ADAPTIVE N (a curve, no selection): 4 samples at T=0.7; if confidence < tau, samples 5-8 at T=0.7 (or at T=1.0),
  else stop at 4. Reported as (mean samples per question, gain) for every tau, against fixed N = 1..8 at T=0.7
  (the first k stored samples), so it either dominates the fixed-N curve or it does not.
CONFIDENCE (pre-specified): the mean over the 24 probes of sigmoid(probe logit) for the stage-1 pick -- the
  "max P(correct)" gate that won the project's July gate bake-off.
PER-MODALITY T needs a modality for every image (two benchmarks have none in their metadata), so it is scored
  later from this script's records, with the model's own modality prediction; see the records file.

CHECKS before any new policy is believed: fixed_T07, tmap and mix8 must reproduce the FINAL / MIX8 records' pick
for every question (same rows, same row order, same np.mean-over-rank_avg readout).
CURRENCIES: training judge (MedVLThinker-32B: the meta row y of the pool the picked row came from) and lenient EM,
for every policy. MedGemma-27B labels exist only for the T07 and Tmap pools, so they are reported ONLY for
policies whose picks are all labelled, and every unlabelled (question, answer) is written to
/data/dan/exp_2026_10/xjudge_missing_temp_labelfree.jsonl for a later MedGemma pass.

  cd <exp worktree> && OMP_NUM_THREADS=4 python3 -u src/exp_2026_10/temp_labelfree.py
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import itertools, json, time
from collections import defaultdict
import numpy as np
import torch

WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/exp_2026_10")
import final_method_eval as FME                     # noqa: E402

ER, rank_avg, OPEN8, ENS, CK = FME.ER, FME.rank_avg, FME.OPEN8, FME.ENS, FME.CK
FROZEN = FME.MAIN + "/ckpts/train/final_probe_2026-10-16"
XJ = "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"
TS = [0.2, 0.4, 0.7, 1.0]
TAG = {0.2: "lingshu7bT02_sc8", 0.4: "lingshu7bT04_sc8", 0.7: "lingshu7b_sc8", 1.0: "lingshu7bT10_sc8"}
MIXORDER = [0.2, 0.4, 0.7, 1.0]                     # = FME.MIX_TAGS order (first pool wins a shared answer)
ART = WT + "/results/cascade_methods/artifacts"
OUT = ART + "/temp_labelfree_2026-10-22.json"
REC = "/data/dan/exp_2026_10/temp_labelfree_records.json"
MISS = "/data/dan/exp_2026_10/xjudge_missing_temp_labelfree.jsonl"
NB, SEED = 10000, 20261022
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

xlab, xj = {}, {}
# MedGemma-27B labels: the October T07/Tmap pass, plus the 2026-10-22 pass on this script's own unlabelled picks
XJS = [XJ] + [p for p in ["/data/dan/exp_2026_10/xjudge_1022/merged.jsonl"] if os.path.exists(p)]
for X in XJS:
    for l in open(X.replace(".jsonl", ".judge.jsonl")):
        if l.strip():
            d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))
for X in XJS:   # (cell, q, answer) -> its first LABELLED idx: one answer string can sit in several work lists
    for l in open(X):  # (e.g. a 7B pick identical to a 32B greedy answer, judged in only one of them)
        if l.strip():
            d = json.loads(l)
            if d["idx"] in xj: xlab.setdefault(tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],), d["idx"])


def xlabel(c, q, ans):
    k = xlab.get((c, str(q), ans))
    return xj.get(k) if k is not None else None


def readout(draws, Q):
    """draws: list of (T, start, end) slices of each pool's 8 stored samples, in draw order.
    Returns (pick record, conf). Distinct answers are taken pool by pool in MIXORDER, rows in cache order,
    first pool wins a shared answer -- exactly final_method_eval's single-pool and MIX handling."""
    drawn = {}
    for T, a, b in draws:
        drawn.setdefault(T, []).extend(Q["samples"][T][a:b])
    allv = [x for T in drawn for x in drawn[T]]
    seen, cand = set(), []
    for T in MIXORDER:
        if T not in drawn:
            continue
        allowed = set(drawn[T])
        for row in Q["rows"][T]:                     # cache order
            if row["na"] in allowed and row["na"] not in seen:
                cand.append(row); seen.add(row["na"])
    if not cand:
        return None, None
    S = np.stack([r["s"] for r in cand], 1)          # 24 x n
    rk = np.mean([rank_avg(S[k]) for k in range(S.shape[0])], 0)
    votes = np.array([allv.count(r["na"]) for r in cand], float)
    top = np.flatnonzero(rk == rk.max())
    p = cand[int(top[np.argmax(votes[top])])]
    return p, float(np.mean(1 / (1 + np.exp(-p["s"]))))


# ---------------------------------------------------------------- load + score every pool once
Qs = {c: [] for c in OPEN8}
for c in OPEN8:
    pools = {T: FME.loadj(f"{CK}/ckpt_{c}_{TAG[T]}.jsonl") for T in TS}
    caches = {T: FME.load_cache(*FME.eval_stem(c, TAG[T], True)) for T in TS}
    greedy = FME.loadj(f"{CK}/ckpt_{c}_lingshu7b.jsonl")
    gj = {k: int(v["judge_ok"]) for k, v in FME.loadj(f"{CK}/ckpt_{c}_lingshu7b.judge.jsonl").items()}
    for f in (0, 1):
        heads, stats, _ = ER.load_probes(f"{FROZEN}/fold{f}")
        byq = {T: defaultdict(list) for T in TS}
        for T in TS:
            X, rows = caches[T]
            idx = [i for i, r in enumerate(rows) if ER.half(r["img_md5"]) == f]
            S = []
            with torch.no_grad():
                for L in ENS:
                    mu, sd = stats[L]
                    Z = torch.tensor(((X[L][idx] - mu) / sd).astype(np.float32))
                    for m in heads[L]:
                        m.eval(); S.append(m(Z).numpy())
            S = np.stack(S)
            for j, i in enumerate(idx):
                r = rows[i]
                byq[T][r["idx"]].append({"na": r["na"], "ans": r["ans"], "y": int(r["y"]), "s": S[:, j], "T": T,
                                         "img": r["img_md5"]})
        qset = set.intersection(*[set(byq[T]) for T in TS]) & set(gj) & set(greedy)
        qset &= set.intersection(*[set(pools[T]) for T in TS])
        for q in sorted(qset, key=str):
            gold = pools[0.7][q]["gold"]; gp = greedy[q]["preds"][0]
            Qs[c].append({"idx": q, "fold": f, "img": byq[0.7][q][0]["img"], "gold": gold,
                          "question": pools[0.7][q]["question"],
                          "samples": {T: [ER.row_norm(a) for a in pools[T][q]["preds"]] for T in TS},
                          "rows": {T: byq[T][q] for T in TS},
                          "greedy": {"judge": gj[q], "em": ER.em_score(gp, gold), "xjudge": xlabel(c, q, gp)}})
        log(f"{c:17} fold {f}: {sum(1 for x in Qs[c] if x['fold'] == f)} questions with all four pools")

# ---------------------------------------------------------------- policies -> picks
TMAP = FME.TMAP
picks = defaultdict(dict)        # policy -> (c, q) -> pick row (or None)
confs = {}                        # (c, q) -> stage-1 confidence
stage2 = defaultdict(dict)        # (T_stage2) -> (c, q) -> pick row of [4@0.7 + 4@T2]
for c in OPEN8:
    for Q in Qs[c]:
        k = (c, Q["idx"])
        for T in TS:
            picks[f"fixed_T{int(T * 10):02d}"][k] = readout([(T, 0, 8)], Q)[0]
        picks["tmap"][k] = readout([(TMAP.get(c, 0.7), 0, 8)], Q)[0]
        picks["mix8"][k] = readout([(T, 0, 2) for T in MIXORDER], Q)[0]
        p1, conf = readout([(0.7, 0, 4)], Q); confs[k] = conf
        for T2 in TS:
            stage2[T2][k] = readout([(0.7, 0, 4), (0.7, 4, 8)] if T2 == 0.7 else [(0.7, 0, 4), (T2, 0, 4)], Q)[0]
        for n in range(1, 9):
            picks[f"fixedN{n}_T07"][k] = readout([(0.7, 0, n)], Q)[0]
        picks["_stage1_N4"][k] = p1
log("all fixed / stage-2 picks computed")

# ---------------------------------------------------------------- checks against the FINAL records
chk = {}
for pol, recf in (("fixed_T07", "final_eval_FINAL_cap320_T07"), ("tmap", "final_eval_FINAL_cap320_Tmap"),
                  ("mix8", "final_eval_FINAL_cap320_MIX_MIX8")):
    R = json.load(open(f"/data/dan/exp_2026_10/{recf}_records.json"))
    ref = {(c, r["idx"]): r["pick_ans"] for c in OPEN8 for r in R[c]}
    both = [k for k in picks[pol] if k in ref]
    mm = sum(1 for k in both if (picks[pol][k] or {}).get("ans") != ref[k])
    chk[pol] = {"compared": len(both), "of_ref": len(ref), "mismatches": mm}
    log(f"CHECK {pol:9} vs {recf}: {mm} mismatches over {len(both)} shared questions (ref has {len(ref)})")
assert all(v["mismatches"] == 0 for v in chk.values()), "a reference policy does not reproduce its FINAL picks"

# ---------------------------------------------------------------- metrics + cross-fitted two-stage policies
KEYS = [(c, Q["idx"], Q["fold"], Q["img"], Q["gold"], Q["greedy"]) for c in OPEN8 for Q in Qs[c]]


def gain_vec(pol_picks, cu):
    out = []
    for c, q, f, img, gold, g in KEYS:
        row = pol_picks.get((c, q))
        v = np.nan if row is None else (float(row["y"]) if cu == "judge" else float(ER.em_score(row["ans"], gold))
                                         if cu == "em" else (lambda x: np.nan if x is None else float(x))(xlabel(c, q, row["ans"])))
        gv = g[cu]; gv = np.nan if gv is None else float(gv)
        out.append(v - gv)
    return np.array(out)


def macro_pt(vec, mask=None):
    cells = np.array([k[0] for k in KEYS]); folds = np.array([k[2] for k in KEYS])
    m = np.ones(len(vec), bool) if mask is None else mask
    return float(np.mean([np.nanmean(vec[(cells == c) & m]) for c in OPEN8]))


cells_a = np.array([k[0] for k in KEYS]); folds_a = np.array([k[2] for k in KEYS])
TAUQ = np.quantile([confs[(k[0], k[1])] for k in KEYS], np.linspace(0.05, 0.95, 19))


def two_stage(lo_T, hi_T, tau):
    """unsure (conf < tau) -> stage-2 at lo_T; confident -> stage-2 at hi_T"""
    return {k: (stage2[lo_T][k] if confs[k] < tau else stage2[hi_T][k]) for k in confs}


def xfit(cands):
    """cands: list of (label, picks-dict). Choose on fold 0 by training-judge gain, apply to fold 1; and vice versa."""
    chosen, final = {}, {}
    for f in (0, 1):
        best = max(cands, key=lambda lc: macro_pt(gain_vec(lc[1], "judge"), folds_a == f))
        chosen[f"chosen_on_fold{f}"] = best[0]
        for c, q, ff, *_ in KEYS:
            if ff == 1 - f:                          # applied to the OTHER fold only
                final[(c, q)] = best[1][(c, q)]
    return final, chosen


primary = [(f"tau={t:.4f}", two_stage(1.0, 0.7, t)) for t in TAUQ]
picks["explore_if_unsure"], chose_primary = xfit(primary)
grid = [(f"unsure->T{lo}, confident->T{hi}, tau={t:.4f}", two_stage(lo, hi, t)) for lo in TS for hi in TS for t in TAUQ[::3]]
picks["best_pair_xfit"], chose_pair = xfit(grid)
log(f"cross-fitted choices: primary {chose_primary}; best pair {chose_pair}")
# oracle temperature (upper bound): per question, any temperature whose 8-sample pick the judge accepts
picks["oracle_T"] = {k: next((picks[f"fixed_T{int(T * 10):02d}"][k] for T in TS
                              if (picks[f"fixed_T{int(T * 10):02d}"][k] or {}).get("y") == 1),
                             picks["fixed_T07"][k]) for k in confs}

# ---------------------------------------------------------------- paired bootstrap, macro over open8 (both folds)
rng = np.random.default_rng(SEED); BW = {}
for c in OPEN8:
    sel = np.flatnonzero(cells_a == c)
    keys_, inv = np.unique([KEYS[i][3] for i in sel], return_inverse=True)
    BW[c] = (sel, inv, rng.multinomial(len(keys_), np.full(len(keys_), 1 / len(keys_)), size=NB).astype(float), len(keys_))


def paired(vec_a, vec_b=None):
    reps, pts = [], []
    for c in OPEN8:
        sel, inv, W, nk = BW[c]
        d = vec_a[sel] - (0 if vec_b is None else vec_b[sel]); m = ~np.isnan(d); d = np.where(m, d, 0.0)
        reps.append((W @ np.bincount(inv, weights=d, minlength=nk)) / np.maximum(W @ np.bincount(inv, weights=m.astype(float), minlength=nk), 1e-12))
        pts.append(d[m].mean() if m.any() else np.nan)
    Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.nanmean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}


res, missing = {}, set()
for pol in [p for p in picks if not p.startswith("_")]:
    e = {}
    for cu in ("judge", "em", "xjudge"):
        g = gain_vec(picks[pol], cu)
        cov = float(np.mean(~np.isnan(g)))
        if cu == "xjudge":
            for (c, q, *_), v in zip(KEYS, g):
                if np.isnan(v) and picks[pol].get((c, q)) is not None:
                    missing.add((c, str(q), picks[pol][(c, q)]["ans"]))
            if cov < 0.999:
                e[cu] = {"coverage": cov, "note": "MedGemma labels missing for some picks; pending a MedGemma pass"}
                continue
        e[cu] = {"gain_over_greedy": paired(g), "vs_fixed_T07": paired(g, gain_vec(picks["fixed_T07"], cu)),
                 "vs_tmap": paired(g, gain_vec(picks["tmap"], cu)), "coverage": cov}
    res[pol] = e
    log(f"{pol:18} judge {e['judge']['gain_over_greedy']['macro']:+.4f}  vs T07 {e['judge']['vs_fixed_T07']['macro']:+.4f} "
        f"{e['judge']['vs_fixed_T07']['verdict']}  | EM {e['em']['gain_over_greedy']['macro']:+.4f}  vs T07 "
        f"{e['em']['vs_fixed_T07']['macro']:+.4f} {e['em']['vs_fixed_T07']['verdict']}")

# adaptive-N curves: (mean samples, gain) for every tau, two stage-2 temperatures, vs fixed N
curves = {}
for T2 in (0.7, 1.0):
    pts_ = []
    for t in TAUQ:
        pk = {k: (stage2[T2][k] if confs[k] < t else picks["_stage1_N4"][k]) for k in confs}
        meanN = float(np.mean([8 if confs[k] < t else 4 for k in confs]))
        pts_.append({"tau": float(t), "mean_samples": meanN,
                     "judge": macro_pt(gain_vec(pk, "judge")), "em": macro_pt(gain_vec(pk, "em"))})
    curves[f"adaptiveN_stage2_T{int(T2 * 10):02d}"] = pts_
curves["fixedN_T07"] = [{"n": n, "judge": macro_pt(gain_vec(picks[f"fixedN{n}_T07"], "judge")),
                         "em": macro_pt(gain_vec(picks[f"fixedN{n}_T07"], "em"))} for n in range(1, 9)]
with open(MISS, "w") as fh:
    for c, q, a in sorted(missing):
        Q = next(x for x in Qs[c] if str(x["idx"]) == q)
        fh.write(json.dumps({"idx": f"{c}|{q}|lf", "cell": c, "q": q, "question": Q["question"], "gold": Q["gold"],
                             "modal_pred": a}) + "\n")
json.dump({c: [{"idx": Q["idx"], "fold": Q["fold"], "img": Q["img"],
                "greedy": Q["greedy"],
                "pick_by_T": {str(T): {"ans": (picks[f"fixed_T{int(T * 10):02d}"][(c, Q["idx"])] or {}).get("ans"),
                                       "judge": (picks[f"fixed_T{int(T * 10):02d}"][(c, Q["idx"])] or {}).get("y"),
                                       "em": ER.em_score((picks[f"fixed_T{int(T * 10):02d}"][(c, Q["idx"])] or {}).get("ans", ""), Q["gold"])}
                              for T in TS}}
               for Q in Qs[c]] for c in OPEN8}, open(REC, "w"))
art = {"title": "Sampling temperature without the dataset label (final protocol, frozen final probes)",
       "date": "2026-10-22", "no_fabricated_numbers": True, "script": os.path.abspath(__file__),
       "n_questions": len(KEYS), "checks": chk, "confidence": "mean over 24 probes of sigmoid(logit) of the stage-1 pick",
       "tau_grid_quantiles": [float(x) for x in TAUQ], "cross_fitted_choices": {"explore_if_unsure": chose_primary,
       "best_pair_xfit": chose_pair}, "ci_caveat": "the cross-fitted choice is not re-made inside each bootstrap resample",
       "policies": res, "adaptive_N_curves": curves, "xjudge_missing_written": {"file": MISS, "n": len(missing)},
       "records": REC, "seconds": round(time.time() - t0, 1), "COMPLETE": True}
json.dump(art, open(OUT, "w"), indent=1)
log(f"wrote {OUT}; {len(missing)} picks need MedGemma labels -> {MISS}")
