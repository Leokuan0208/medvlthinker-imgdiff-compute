#!/usr/bin/env python3
"""layer_fusion.py -- which layers, and which COMBINATIONS of layers, should the probe verifier read?

Second half of the layer study (layer_fit.py fits the probes). Every layer of 16 and 18-28 has 8 probes per
fold, all fitted in one session with the FINAL recipe and fitting set, so layers are compared like with like.
A "fusion" is a set of layers whose 8 probes each vote through the shipped readout (rank average over all chosen
probes, vote tie-break) -- exactly how the shipped 18/20/22 ensemble combines its three layers.

CONFIGS: every layer subset of size 1-4, every contiguous window of the depth order, and all 12 layers (~830).
CONTROL: {18, 20, 22} refitted in this same session (NOT the frozen probes, which were fitted at 4 threads).
REPORTED: (1) point estimates for every config; (2) paired image-clustered 95% CIs vs the control for a
PRE-SPECIFIED list -- each single layer (the depth curve), the deeper triples {22,24,26} {23,25,27} {24,26,28}
{26,27,28}, the spread triples {16,22,28} {20,24,28}, all deep layers {23..28}, and all 12; (3) the best
subset chosen HONESTLY by cross-fitting -- picked on image-fold 0 by training-judge gain and scored on fold 1,
and vice versa -- because picking the best of ~830 on the same questions is the selection-on-eval bug the
2026-09-18 audit fixed. Both pools: T07 (the deck's headline) and Tmap (FINAL_METHOD's method).
Currencies: training judge (MedVLThinker-32B), MedGemma-27B (labels exist for T07 and Tmap pools), lenient EM.

  OMP_NUM_THREADS=4 python3 src/exp_2026_10/layer_fusion.py
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import itertools, json, time
from collections import defaultdict
import numpy as np, torch

WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/exp_2026_10")
import final_method_eval as FME                     # noqa: E402
import layer_fit as LF                              # noqa: E402  stems, loader

ER, HS, rank_avg, OPEN8 = FME.ER, FME.HS, FME.rank_avg, FME.OPEN8
LAYERS = [16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
PD = LF.OUTD
XJ = "/data/dan/exp_2026_10/xjudge_c320/merged.jsonl"
OUT = WT + "/results/cascade_methods/artifacts/layer_fusion_2026-10-22.json"
CUR = ("judge", "xjudge", "em")
NB, SEED = 10000, 20261022
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"])); t0 = time.time()
log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

CFG = []
for k in range(1, 5):
    CFG += [tuple(c) for c in itertools.combinations(LAYERS, k)]
for k in range(5, len(LAYERS) + 1):
    CFG += [tuple(LAYERS[i:i + k]) for i in range(len(LAYERS) - k + 1)]
CFG = list(dict.fromkeys(CFG))
CONTROL = (18, 20, 22)
PRE = [(L,) for L in LAYERS] + [(22, 24, 26), (23, 25, 27), (24, 26, 28), (26, 27, 28), (16, 22, 28), (20, 24, 28),
                                 tuple(range(23, 29)), tuple(LAYERS)]
assert CONTROL in CFG and all(p in CFG for p in PRE)
ROWS = {cfg: np.array([LAYERS.index(L) * 8 + s for L in cfg for s in range(8)]) for cfg in CFG}
log(f"{len(CFG)} layer configurations")

xlab, xj = {}, {}
for l in open(XJ):
    if l.strip():
        d = json.loads(l); xlab[tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],)] = d["idx"]
for l in open(XJ.replace(".jsonl", ".judge.jsonl")):
    if l.strip():
        d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))
xl = lambda c, q, a: (lambda k: xj.get(k) if k is not None else None)(xlab.get((c, str(q), a)))


def probes(f):
    H, ST = {}, {}
    for L in LAYERS:
        z = np.load(f"{PD}/fold{f}/standardizer_L{L}.npz"); ST[L] = (z["mu"], z["sd"]); H[L] = []
        for s in range(8):
            m = HS.MLP(3584, hidden=256); m.load_state_dict(torch.load(f"{PD}/fold{f}/head_L{L}_seed{s}.pt", map_location="cpu"))
            m.eval(); H[L].append(m)
    return H, ST


res_all = {}
for pol in ("T07", "Tmap"):
    Q = {c: [] for c in OPEN8}
    for f in (0, 1):
        H, ST = probes(f)
        for c in OPEN8:
            stem = LF.stem07(c) if pol == "T07" else LF.stemT(c)
            tag = FME.pool_tag(c, pol, True)
            sc8 = FME.loadj(f"{FME.CK}/ckpt_{c}_{tag}.jsonl"); greedy = FME.loadj(f"{FME.CK}/ckpt_{c}_lingshu7b.jsonl")
            gj = {k: int(v["judge_ok"]) for k, v in FME.loadj(f"{FME.CK}/ckpt_{c}_lingshu7b.judge.jsonl").items()}
            S, rr = [], None
            for L in LAYERS:
                X, rows = LF.load(stem, L)
                idx = [i for i, r in enumerate(rows) if ER.half(r["img_md5"]) == f]; rr = [rows[i] for i in idx]
                mu, sd = ST[L]; Z = torch.tensor(((X[idx] - mu) / sd).astype(np.float32))
                with torch.no_grad():
                    for m in H[L]:
                        S.append(m(Z).numpy())
            S = np.stack(S)                                     # 96 x n
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            for q, ii in byq.items():
                if q not in gj or q not in greedy or q not in sc8:
                    continue
                ii = np.array(ii); R = np.stack([rank_avg(S[k][ii]) for k in range(S.shape[0])])
                pn = [ER.row_norm(a) for a in sc8[q]["preds"]]; votes = np.array([pn.count(rr[i]["na"]) for i in ii], float)
                pk = np.empty(len(CFG), np.int16)
                for ci, cfg in enumerate(CFG):
                    rk = np.mean(R[ROWS[cfg]], 0); top = np.flatnonzero(rk == rk.max()); pk[ci] = top[np.argmax(votes[top])]
                gold, gp = sc8[q]["gold"], greedy[q]["preds"][0]; ans = [rr[i]["ans"] for i in ii]
                cand = {"judge": np.array([rr[i]["y"] for i in ii], float), "em": np.array([ER.em_score(a, gold) for a in ans], float),
                        "xjudge": np.array([np.nan if xl(c, q, a) is None else xl(c, q, a) for a in ans], float)}
                g = {"judge": float(gj[q]), "em": float(ER.em_score(gp, gold)), "xjudge": np.nan if xl(c, q, gp) is None else float(xl(c, q, gp))}
                Q[c].append({"img": rr[ii[0]]["img_md5"], "fold": f, "pk": pk, "cand": cand, "g": g})
            log(f"[{pol}] fold {f} {c:17} {sum(1 for x in Q[c] if x['fold'] == f)} questions")
    # metrics
    M = {(c, cu): np.stack([x["cand"][cu][x["pk"]] for x in Q[c]], 1) for c in OPEN8 for cu in CUR}   # cfg x q
    G = {(c, cu): np.array([x["g"][cu] for x in Q[c]]) for c in OPEN8 for cu in CUR}
    F = {c: np.array([x["fold"] for x in Q[c]]) for c in OPEN8}
    ci_ = CFG.index(CONTROL)

    def pt(cu, k, base=None, fold=None):
        out = []
        for c in OPEN8:
            v = M[(c, cu)][k]; b = G[(c, cu)] if base is None else M[(c, cu)][base]
            m = ~(np.isnan(v) | np.isnan(b)) & (True if fold is None else F[c] == fold)
            out.append(np.mean((v - b)[m]))
        return float(np.mean(out))

    rng = np.random.default_rng(SEED); W = {}
    for c in OPEN8:
        keys, inv = np.unique([x["img"] for x in Q[c]], return_inverse=True)
        W[c] = (inv, rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NB).astype(float), len(keys))

    def boot(cu, vals, base):
        reps, pts = [], []
        for c in OPEN8:
            inv, w, nk = W[c]; v, b = vals(c), base(c)
            m = ~(np.isnan(v) | np.isnan(b)); d = np.where(m, v - b, 0.0)
            reps.append((w @ np.bincount(inv, weights=d, minlength=nk)) / np.maximum(w @ np.bincount(inv, weights=m.astype(float), minlength=nk), 1e-12))
            pts.append(d[m].mean())
        Rm = np.stack(reps, 1).mean(1); lo, hi = float(np.percentile(Rm, 2.5)), float(np.percentile(Rm, 97.5))
        return {"macro": float(np.mean(pts)), "ci": [lo, hi], "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")}

    allpts = {",".join(map(str, cfg)): {cu: {"gain": pt(cu, k), "minus_control": pt(cu, k, ci_)} for cu in CUR}
              for k, cfg in enumerate(CFG)}
    pre = {}
    for cfg in [CONTROL] + PRE:
        k = CFG.index(cfg)
        pre[",".join(map(str, cfg))] = {cu: {"gain_over_greedy": boot(cu, lambda c: M[(c, cu)][k], lambda c: G[(c, cu)]),
                                             "minus_control": boot(cu, lambda c: M[(c, cu)][k], lambda c: M[(c, cu)][ci_])} for cu in CUR}
    # cross-fitted best subset (chosen on one fold by training judge, scored on the other)
    choice = {f: max(range(len(CFG)), key=lambda k: pt("judge", k, fold=f)) for f in (0, 1)}
    # fold-0 questions get the config chosen on fold 1, fold-1 questions the one chosen on fold 0 (question order unchanged)
    xf = {(c, cu): np.where(F[c] == 0, M[(c, cu)][choice[1]], M[(c, cu)][choice[0]]) for c in OPEN8 for cu in CUR}
    xbest = {"chosen_on_fold0": ",".join(map(str, CFG[choice[0]])), "chosen_on_fold1": ",".join(map(str, CFG[choice[1]])),
             **{cu: {"gain_over_greedy": boot(cu, lambda c: xf[(c, cu)], lambda c: G[(c, cu)]),
                     "minus_control": boot(cu, lambda c: xf[(c, cu)], lambda c: M[(c, cu)][ci_])} for cu in CUR},
             "caveat": "the choice is not re-made inside each bootstrap resample"}
    res_all[pol] = {"control": ",".join(map(str, CONTROL)), "prespecified": pre, "cross_fitted_best": xbest,
                    "all_configs_point": allpts, "n_questions": sum(len(v) for v in Q.values())}
    log(f"[{pol}] control {pre['18,20,22']['judge']['gain_over_greedy']['macro']:+.4f}; xfit best {xbest['chosen_on_fold0']} / "
        f"{xbest['chosen_on_fold1']}: {xbest['judge']['minus_control']['macro']:+.4f} {xbest['judge']['minus_control']['verdict']}")
json.dump({"title": "Layer study: single layers 16, 18-28 and layer fusions under the final protocol", "date": "2026-10-22",
           "no_fabricated_numbers": True, "script": os.path.abspath(__file__), "layers": LAYERS, "n_configs": len(CFG),
           "probes": PD, "bootstrap": {"n": NB, "seed": SEED, "cluster": "img_md5 within cell, paired"},
           "policies": res_all, "seconds": round(time.time() - t0, 1), "COMPLETE": True}, open(OUT, "w"), indent=1)
log(f"wrote {OUT}")
