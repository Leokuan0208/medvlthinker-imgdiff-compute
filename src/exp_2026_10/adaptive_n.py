#!/usr/bin/env python3
"""adaptive_n.py -- per-question sample count from the probe's own confidence, with paired CIs.

Follow-up to temp_labelfree.py, whose adaptive-N curve showed that drawing 4 samples and topping up only the
questions the probe is least sure of reaches the fixed N=8 gain with fewer samples. That was a curve of point
estimates. This pre-specifies three budgets and tests them in all three currencies.

POLICY. Draw the first 4 of the stored 8 samples (T=0.7). Readout = the shipped rank_vote over the frozen final
probes. Confidence = mean over the 24 probes of sigmoid(logit) of the stage-1 pick. Re-sample (samples 5-8) the
least-confident fraction p of questions; the rest keep their stage-1 pick. p in {0.25, 0.5, 0.75} -> mean
budget 5, 6, 7 samples. The cut is the p-quantile of confidence over all questions: it uses NO labels, so there
is nothing to cross-fit. Compared, paired and image-clustered, with fixed N = 5 / 6 / 7 (the first k samples,
same budget) and with fixed N = 8 (the deck's method). All picks come from the T=0.7 pool, which MedGemma-27B
has labelled in full, so all three currencies are complete. Frozen final probes, both folds, all 8 benchmarks.

  OMP_NUM_THREADS=4 python3 src/exp_2026_10/adaptive_n.py
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json, time
from collections import defaultdict
import numpy as np, torch

WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/exp_2026_10")
import final_method_eval as FME                     # noqa: E402

ER, rank_avg, OPEN8, ENS = FME.ER, FME.rank_avg, FME.OPEN8, FME.ENS
FROZEN = FME.MAIN + "/ckpts/train/final_probe_2026-10-16"
XJ = "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"
OUT = WT + "/results/cascade_methods/artifacts/adaptive_n_2026-10-22.json"
NB, SEED = 10000, 20261022
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"])); t0 = time.time()
xlab, xj = {}, {}
for l in open(XJ):
    if l.strip():
        d = json.loads(l); xlab[tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],)] = d["idx"]
for l in open(XJ.replace(".jsonl", ".judge.jsonl")):
    if l.strip():
        d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))
xl = lambda c, q, a: (lambda k: xj.get(k) if k is not None else None)(xlab.get((c, str(q), a)))


def pick(rows, S, samples):
    """shipped rank_vote over the distinct answers present in `samples` (cache row order)."""
    al = set(samples); keep = [j for j, r in enumerate(rows) if r["na"] in al]
    if not keep:
        return None, None
    R = np.stack([rank_avg(S[k][keep]) for k in range(S.shape[0])]); rk = R.mean(0)
    votes = np.array([samples.count(rows[j]["na"]) for j in keep], float)
    top = np.flatnonzero(rk == rk.max()); p = keep[int(top[np.argmax(votes[top])])]
    return p, float(np.mean(1 / (1 + np.exp(-S[:, p]))))


Q = []
for c in OPEN8:
    X, rows = FME.load_cache(*FME.eval_stem(c, "lingshu7b_sc8", True))
    sc8 = FME.loadj(f"{FME.CK}/ckpt_{c}_lingshu7b_sc8.jsonl"); greedy = FME.loadj(f"{FME.CK}/ckpt_{c}_lingshu7b.jsonl")
    gj = {k: int(v["judge_ok"]) for k, v in FME.loadj(f"{FME.CK}/ckpt_{c}_lingshu7b.judge.jsonl").items()}
    for f in (0, 1):
        heads, stats, _ = ER.load_probes(f"{FROZEN}/fold{f}")
        idx = [i for i, r in enumerate(rows) if ER.half(r["img_md5"]) == f]; rr = [rows[i] for i in idx]
        S = []
        with torch.no_grad():
            for L in ENS:
                mu, sd = stats[L]; Z = torch.tensor(((X[L][idx] - mu) / sd).astype(np.float32))
                for m in heads[L]:
                    m.eval(); S.append(m(Z).numpy())
        S = np.stack(S); byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        for q, ii in byq.items():
            if q not in gj or q not in greedy or q not in sc8:
                continue
            sub = [rr[i] for i in ii]; Ss = S[:, ii]; smp = [ER.row_norm(a) for a in sc8[q]["preds"]]
            gold, gp = sc8[q]["gold"], greedy[q]["preds"][0]
            val = lambda j: {"judge": float(sub[j]["y"]), "em": float(ER.em_score(sub[j]["ans"], gold)),
                             "xjudge": (lambda x: np.nan if x is None else float(x))(xl(c, q, sub[j]["ans"]))}
            fixed = {n: pick(sub, Ss, smp[:n])[0] for n in range(1, 9)}
            p1, conf = pick(sub, Ss, smp[:4])
            Q.append({"c": c, "f": f, "img": sub[0]["img_md5"], "conf": conf,
                      "fixed": {n: val(j) for n, j in fixed.items()},
                      "g": {"judge": float(gj[q]), "em": float(ER.em_score(gp, gold)),
                            "xjudge": (lambda x: np.nan if x is None else float(x))(xl(c, q, gp))}})
    print(f"[{time.time() - t0:5.0f}s] {c}: {sum(1 for x in Q if x['c'] == c)} questions", flush=True)

conf = np.array([x["conf"] for x in Q]); cells = np.array([x["c"] for x in Q])
rng = np.random.default_rng(SEED); W = {}
for c in OPEN8:
    sel = np.flatnonzero(cells == c); keys, inv = np.unique([Q[i]["img"] for i in sel], return_inverse=True)
    W[c] = (sel, inv, rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NB).astype(float), len(keys))


def paired(a, b):
    reps, pts = [], []
    for c in OPEN8:
        sel, inv, w, nk = W[c]; d = a[sel] - b[sel]; m = ~np.isnan(d); d = np.where(m, d, 0.0)
        reps.append((w @ np.bincount(inv, weights=d, minlength=nk)) / np.maximum(w @ np.bincount(inv, weights=m.astype(float), minlength=nk), 1e-12))
        pts.append(d[m].mean())
    Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
    return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}


vec = lambda f, cu: np.array([f(x)[cu] for x in Q])
res = {}
for p in (0.25, 0.5, 0.75):
    cut = np.quantile(conf, p); top = conf < cut
    budget = 4 + 4 * float(top.mean()); nb = int(round(budget))
    ad = lambda x, cut=cut: x["fixed"][8] if x["conf"] < cut else x["fixed"][4]
    e = {"resampled_fraction": float(top.mean()), "mean_samples": budget, "confidence_cut": float(cut)}
    for cu in ("judge", "xjudge", "em"):
        a, g = vec(ad, cu), vec(lambda x: x["g"], cu)
        e[cu] = {"gain_over_greedy": paired(a, g),
                 f"vs_fixed_N{nb}": paired(a, vec(lambda x, n=nb: x["fixed"][n], cu)),
                 "vs_fixed_N8": paired(a, vec(lambda x: x["fixed"][8], cu))}
    res[f"resample_{int(p * 100)}pct"] = e
    print(f"  re-sample {p:.0%} (mean N {budget:.2f}): " + " | ".join(
        f"{cu} {e[cu]['gain_over_greedy']['macro']:+.4f}, vs N{nb} {e[cu][f'vs_fixed_N{nb}']['macro']:+.4f} {e[cu][f'vs_fixed_N{nb}']['verdict']}, "
        f"vs N8 {e[cu]['vs_fixed_N8']['macro']:+.4f} {e[cu]['vs_fixed_N8']['verdict']}" for cu in ("judge", "xjudge", "em")), flush=True)
json.dump({"title": "Adaptive sample count from probe confidence (pre-specified budgets 5/6/7)", "date": "2026-10-22",
           "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "n_questions": len(Q),
           "policy": "4 samples at T=0.7; re-sample (samples 5-8) the least-confident fraction p; confidence = mean sigmoid over 24 probes",
           "label_free": "the confidence cut is a quantile of confidence over all questions -- no labels used",
           "results": res, "bootstrap": {"n": NB, "seed": SEED, "cluster": "img_md5 within cell, paired"},
           "seconds": round(time.time() - t0, 1), "COMPLETE": True}, open(OUT, "w"), indent=1)
print("wrote", OUT)
