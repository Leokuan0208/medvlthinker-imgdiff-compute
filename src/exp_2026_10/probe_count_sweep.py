#!/usr/bin/env python3
"""probe_count_sweep.py -- WHY 24 PROBES? Score the final method with SUBSETS of its own frozen probes.

Leo, 2026-10-19: "why 24 and not 21, 27 etc"; 2026-10-21: "do we actually need 8 seeds, do a sweep on that".
The shipped ensemble is 3 layers (18/20/22) x 8 seeds = 24 one-hidden-layer probes, rank-averaged with a vote
tie-break. Nothing ever measured what each of those 24 buys. This script answers it WITHOUT refitting
anything: it re-scores the final method's exact questions with every subset of its frozen probes.

WHAT IS HELD FIXED -- everything except which probes vote. Same frozen probes
(ckpts/train/final_probe_2026-10-16/fold{0,1}), same two-fold image cross-fitting, same Tmap pools, same
rank_vote readout, same currencies, same paired image-clustered bootstrap. Data loading is NOT
re-implemented: it calls final_method_eval.py's own pool_tag / eval_stem / load_cache / loadj.

THE SWEEP. Seeds are interchangeable (each is an independent random init of the same recipe), so "K seeds"
is not one configuration but C(8,K) of them. Every subset is scored:
  3 layers x K seeds per layer (same K seed indices on each layer)  -> 3K probes, 255 subsets
  1 layer  x K seeds, for each of layers 18 / 20 / 22               ->  K probes, 765 subsets
For each (layers, K): the mean / min / max over ALL subsets of the macro gain over greedy, and of the
difference from the shipped 24 -- so "how much does a probe buy" and "how much does WHICH seeds you keep
matter" are both visible. Paired image-clustered 95% CIs vs the shipped 24 are computed for one
pre-specified subset per (layers, K): seeds {0..K-1}, i.e. what you would have if you had trained only K.

TWO BUILT-IN CHECKS, both must pass before any subset is believed:
  1. the 24-probe configuration must reproduce final_eval_FINAL_cap320_Tmap_records.json's pick for EVERY
     question (it uses the same np.mean-over-rank_avg expression, so even float ties resolve identically);
  2. its macro gain over greedy, with the same bootstrap RNG stream, must equal final_eval_FINAL_cap320_Tmap.json.

Currencies (identical picks): judge = MedVLThinker-32B (training judge), xjudge = MedGemma-27B-it
(cross-family), em = lenient exact match. CLAUDE.md: a verifier comparison is not interpretable in one.

  cd <exp worktree> && OMP_NUM_THREADS=4 python3 -u src/exp_2026_10/probe_count_sweep.py
CPU only, 4 threads, ~30 min. Writes results/cascade_methods/artifacts/probe_count_sweep_2026-10-21.json.
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
import final_method_eval as FME                     # noqa: E402  module-level helpers only; main() is not run

ER, rank_avg, OPEN8, ENS = FME.ER, FME.rank_avg, FME.OPEN8, FME.ENS
MAIN = FME.MAIN
FROZEN = MAIN + "/ckpts/train/final_probe_2026-10-16"
XJUDGE = "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"
# SWEEP_POLICY=T07 (the shipped method and the 2026-10-19 deck's headline) or Tmap (the domain-calibrated variant,
# not shipped since 2026-10-18); both are FINAL-run policies. The T07 artifact of record is probe_count_sweep_T07.py's.
POLICY, LEGACY, GTAG = os.environ.get("SWEEP_POLICY", "Tmap"), True, "lingshu7b"
REF_ART = WT + f"/results/cascade_methods/artifacts/final_eval_FINAL_cap320_{POLICY}.json"
REF_REC = f"/data/dan/exp_2026_10/final_eval_FINAL_cap320_{POLICY}_records.json"
OUT = WT + ("/results/cascade_methods/artifacts/probe_count_sweep_2026-10-21.json" if POLICY == "Tmap" else
            f"/results/cascade_methods/artifacts/probe_count_sweep_{POLICY}_2026-10-22.json")
CUR = ("judge", "xjudge", "em")
NSEED = 8
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
t0 = time.time()


def log(s):
    print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)


# ---------------------------------------------------------------- configurations (rows of S = layer*8 + seed)
CFG = []                                             # (layers_label, K, seed_tuple, probe_row_tuple)
for K in range(1, NSEED + 1):
    for T in itertools.combinations(range(NSEED), K):
        CFG.append(("3 layers", K, T, tuple(sorted(l * NSEED + s for l in range(3) for s in T))))
        for li, L in enumerate(ENS):
            CFG.append((f"layer {L}", K, T, tuple(li * NSEED + s for s in T)))
FULL = next(i for i, c in enumerate(CFG) if c[0] == "3 layers" and c[1] == NSEED)
ROWS = [np.array(c[3]) for c in CFG]               # index arrays built once, not per question
assert CFG[FULL][3] == tuple(range(3 * NSEED))
log(f"{len(CFG)} probe subsets to score (shipped 24 = config {FULL})")

# ---------------------------------------------------------------- cross-family judge labels (as FME.main)
xlab, xj = {}, {}
for l in open(XJUDGE):
    if l.strip():
        d = json.loads(l); xlab[tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],)] = d["idx"]
for l in open(XJUDGE.replace(".jsonl", ".judge.jsonl")):
    if l.strip():
        d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))


def xlabel(c, q, ans):
    k = xlab.get((c, str(q), ans))
    return xj.get(k) if k is not None else None


# ---------------------------------------------------------------- score every question once per fold
ref = json.load(open(REF_REC))
ref_pick = {(c, r["fold"], r["idx"]): r["pick_ans"] for c in OPEN8 for r in ref[c]}
Q = {c: [] for c in OPEN8}       # per question: img, greedy currencies, candidate currency arrays, picks per config
mismatch = 0
for c in OPEN8:
    tag = FME.pool_tag(c, POLICY, LEGACY)
    stem, dsf = FME.eval_stem(c, tag, LEGACY)
    X, rows = FME.load_cache(stem, dsf)
    sc8 = FME.loadj(f"{FME.CK}/ckpt_{c}_{tag}.jsonl")
    greedy = FME.loadj(f"{FME.CK}/ckpt_{c}_{GTAG}.jsonl")
    gj = {k: int(v["judge_ok"]) for k, v in FME.loadj(f"{FME.CK}/ckpt_{c}_{GTAG}.judge.jsonl").items()}
    for f in (0, 1):
        heads, stats, _ = ER.load_probes(f"{FROZEN}/fold{f}")
        idx = [i for i, r in enumerate(rows) if ER.half(r["img_md5"]) == f]
        rr = [rows[i] for i in idx]
        S = []
        with torch.no_grad():
            for L in ENS:
                mu, sd = stats[L]
                Z = torch.tensor(((X[L][idx] - mu) / sd).astype(np.float32))
                for m in heads[L]:
                    m.eval(); S.append(m(Z).numpy())
        S = np.stack(S)
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        for q, ii in byq.items():
            if q not in gj or q not in greedy or q not in sc8:
                continue
            ii = np.array(ii)
            R = np.stack([rank_avg(S[k][ii]) for k in range(S.shape[0])])   # 24 x n, the shipped expression
            pn = [ER.row_norm(a) for a in sc8[q]["preds"]]
            votes = np.array([pn.count(rr[i]["na"]) for i in ii], float)
            picks = np.empty(len(CFG), np.int16)
            for ci, cfg in enumerate(CFG):
                rk = np.mean(R[ROWS[ci]], 0)
                top = np.flatnonzero(rk == rk.max())
                picks[ci] = top[np.argmax(votes[top])]
            gold, gpred = sc8[q]["gold"], greedy[q]["preds"][0]
            ans = [rr[i]["ans"] for i in ii]
            cand = {"judge": np.array([rr[i]["y"] for i in ii], float),
                    "em": np.array([ER.em_score(a, gold) for a in ans], float),
                    "xjudge": np.array([np.nan if xlabel(c, q, a) is None else xlabel(c, q, a) for a in ans])}
            gx = xlabel(c, q, gpred)
            Q[c].append({"img": rr[ii[0]]["img_md5"], "fold": f, "picks": picks, "cand": cand,
                         "greedy": {"judge": float(gj[q]), "em": float(ER.em_score(gpred, gold)),
                                    "xjudge": np.nan if gx is None else float(gx)}})
            if ans[picks[FULL]] != ref_pick.get((c, f, q)):
                mismatch += 1
        log(f"{c:17} fold {f}: {sum(1 for r in Q[c] if r['fold'] == f):5} questions scored, running mismatches vs shipped picks: {mismatch}")
nq = sum(len(v) for v in Q.values())
log(f"CHECK 1 -- shipped-24 picks vs the FINAL run's records: {mismatch} mismatches over {nq} questions")
assert len(ref_pick) == nq, f"question count {nq} != records {len(ref_pick)}"
assert mismatch == 0, "the 24-probe configuration does not reproduce the shipped picks -- stop"

# ---------------------------------------------------------------- per-config, per-question metric matrices
M = {}                                              # (cell, currency) -> (configs x questions) pick values
G = {}                                              # (cell, currency) -> (questions,) greedy values
PC = {}                                             # cell -> (configs x questions) picked candidate index
for c in OPEN8:
    P = PC[c] = np.stack([r["picks"] for r in Q[c]], 1)
    for cu in CUR:
        M[(c, cu)] = np.stack([r["cand"][cu][P[:, j]] for j, r in enumerate(Q[c])], 1)
        G[(c, cu)] = np.array([r["greedy"][cu] for r in Q[c]])
log("metric matrices built")


def cell_point(c, cu, vals, base):
    m = ~(np.isnan(vals) | np.isnan(base))
    return float((vals - base)[m].mean())


# point estimates for EVERY config: macro (pick - greedy) and macro (config - shipped), per currency
pt_g = {cu: np.array([np.mean([cell_point(c, cu, M[(c, cu)][k], G[(c, cu)]) for c in OPEN8])
                      for k in range(len(CFG))]) for cu in CUR}
pt_s = {cu: np.array([np.mean([cell_point(c, cu, M[(c, cu)][k], M[(c, cu)][FULL]) for c in OPEN8])
                      for k in range(len(CFG))]) for cu in CUR}
allP = np.concatenate([PC[c] for c in OPEN8], 1)   # configs x all questions
agree = (allP == allP[FULL]).mean(1)
log("point estimates for all configs done")

# ---------------------------------------------------------------- paired bootstrap, same RNG stream as FME
ss = np.random.SeedSequence(FME.SEED)
W, INV, NKEY = {}, {}, {}
for c in OPEN8:                                     # FME spawns one RNG per cell in records order (OPEN8 first),
    rng = np.random.default_rng(ss.spawn(1)[0])     # then draws the fold0 matrix BEFORE the "all" matrix
    for scope in ("fold0", "all"):
        rows_ = [r for r in Q[c] if scope == "all" or r["fold"] == 0]
        keys, inv = np.unique([r["img"] for r in rows_], return_inverse=True)
        w = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=FME.NBOOT).astype(float)
        if scope == "all":
            W[c], INV[c], NKEY[c] = w, inv, len(keys)


def boot_macro(cu, vals_of, base_of):
    reps, pts = [], []
    for c in OPEN8:
        v, b = vals_of(c), base_of(c)
        m = ~(np.isnan(v) | np.isnan(b)); d = np.where(m, v - b, 0.0)
        num = W[c] @ np.bincount(INV[c], weights=d, minlength=NKEY[c])
        den = W[c] @ np.bincount(INV[c], weights=m.astype(float), minlength=NKEY[c])
        reps.append(num / np.maximum(den, 1e-12)); pts.append(d[m].mean())
    Rm = np.stack(reps, 1).mean(1)
    lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}


refart = json.load(open(REF_ART))["macro_over_open8"]["all"]
check2 = {}
for cu in CUR:
    got = boot_macro(cu, lambda c: M[(c, cu)][FULL], lambda c: G[(c, cu)])
    want = refart[f"pick_minus_greedy_{cu}"]
    check2[cu] = {"this_script": got, f"final_eval_FINAL_cap320_{POLICY}": {"macro": want["macro"], "ci": want["ci"]},
                  "point_matches": abs(got["macro"] - want["macro"]) < 1e-12,
                  "ci_matches": max(abs(a - b) for a, b in zip(got["ci"], want["ci"])) < 1e-12}
    log(f"CHECK 2 -- {cu:6} shipped {got['macro']:+.4f} {got['ci']} vs FINAL {want['macro']:+.4f} {want['ci']}")
assert all(v["point_matches"] for v in check2.values()), "shipped macro does not reproduce the FINAL artifact"

# ---------------------------------------------------------------- summarise per (layers, K)
summary = {}
for lab in ["3 layers"] + [f"layer {L}" for L in ENS]:
    summary[lab] = {}
    for K in range(1, NSEED + 1):
        ks = [i for i, c in enumerate(CFG) if c[0] == lab and c[1] == K]
        pre = next(i for i in ks if CFG[i][2] == tuple(range(K)))
        e = {"n_probes": len(CFG[pre][3]), "n_subsets": len(ks), "pick_agreement_with_shipped_mean": float(agree[ks].mean())}
        for cu in CUR:
            g, s = pt_g[cu][ks], pt_s[cu][ks]
            e[cu] = {"gain_over_greedy_mean_over_subsets": float(g.mean()), "gain_min": float(g.min()), "gain_max": float(g.max()),
                     "minus_shipped_mean_over_subsets": float(s.mean()), "minus_shipped_min": float(s.min()),
                     "minus_shipped_max": float(s.max()),
                     "prespecified_seeds_0_to_K-1": {
                         "gain_over_greedy": boot_macro(cu, lambda c: M[(c, cu)][pre], lambda c: G[(c, cu)]),
                         "minus_shipped": boot_macro(cu, lambda c: M[(c, cu)][pre], lambda c: M[(c, cu)][FULL])}}
        summary[lab][str(K)] = e
    log(f"summarised {lab}")

art = {"title": "Probe-count sweep: what each of the final method's 24 probes buys (why 24?)",
       "date": "2026-10-21", "no_fabricated_numbers": True, "script": os.path.abspath(__file__),
       "question": "Leo 2026-10-19 'why 24 and not 21, 27 etc'; 2026-10-21 'do we actually need 8 seeds'",
       "held_fixed": {"frozen_probes": FROZEN, "policy": POLICY, "readout": "rank_vote", "folds": [0, 1],
                      "legacy_cap320": LEGACY, "xjudge": XJUDGE, "bootstrap": {"n": FME.NBOOT, "seed": FME.SEED,
                      "cluster": "img_md5 within cell, paired", "scope": "both folds (all)"}},
       "n_questions": nq, "n_configs": len(CFG),
       "checks": {"1_shipped_picks_reproduced": {"mismatches": mismatch, "questions": nq},
                  "2_shipped_macro_reproduced": check2},
       "currencies": {"judge": "MedVLThinker-32B (training judge)", "xjudge": "MedGemma-27B-it (cross-family)",
                      "em": "lenient exact match"},
       "summary": summary, "seconds": round(time.time() - t0, 1), "COMPLETE": True}
json.dump(art, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
