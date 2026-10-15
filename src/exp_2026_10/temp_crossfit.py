#!/usr/bin/env python3
"""temp_crossfit.py -- is choosing the sampling temperature PER BENCHMARK a real improvement, or only
the in-sample +0.0103 that head_temp_ensemble_2026-08-30.json reports as an upper bound?

Honest version: cross-fitting inside the held-out half. Held-out images are split into two folds
(md5("tf"+img_md5) % 2). For each benchmark, the temperature in {0.2, 0.4, 0.7, 1.0} with the best
judge gain over greedy on fold A is applied to fold B's questions, and vice versa. The probe never
saw any of these questions (it was fitted on the other image half), and no question is scored with a
temperature chosen on itself.

Probes: the SHIPPED 24 (genframe_head_pooled_ens_v2), rank readout -- head_temp_ensemble used the
pre-v2 build. Questions: those with a pool at all four temperatures and a greedy judge label, so every
arm is scored on the same questions. Currencies: judge of record (MedVLThinker-32B, meta row y) and
lenient EM (run_openvqa scorer). The cross-family judge has labels only for the T=0.7 pools, so it is
not available here. CI: paired image-clustered bootstrap of (cross-fit minus fixed T=0.7); the T choice
is NOT re-made inside each resample, so the interval is somewhat anticonservative (stated in output).

  OMP_NUM_THREADS=4 python3 temp_crossfit.py   (from the repo root)
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import hashlib, json, time
from collections import defaultdict
import numpy as np
import torch

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, ".claude/worktrees/exp-2026-10-15/src/audit_2026_09_18"))
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
import em_rescore_pooled_probe as ER          # noqa: E402
from genframe_data import rank_avg            # noqa: E402

TT = {0.2: ("_T02", "lingshu7bT02"), 0.4: ("_T04", "lingshu7bT04"),
      0.7: ("", "lingshu7b"), 1.0: ("_T10", "lingshu7bT10")}
OUT = os.path.join(MAIN, "results/cascade_methods/artifacts/temp_crossfit_2026-10-15.json")
NBOOT, SEED = 10000, 20261016


def fold(img):
    return int(hashlib.md5(("tf" + str(img)).encode()).hexdigest(), 16) % 2


def main():
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    t0 = time.time()
    heads, stats, _ = ER.load_probes(ER.PROBE_DIR)
    per, deltas = {}, {}
    rng_master = np.random.SeedSequence(SEED).spawn(len(ER.BENCH))
    for b, cell in enumerate(ER.BENCH):
        greedy = ER.loadj(f"{ER.CK}/ckpt_{cell}_lingshu7b.jsonl")
        gj = {k: int(v["judge_ok"]) for k, v in ER.loadj(f"{ER.CK}/ckpt_{cell}_lingshu7b.judge.jsonl").items()}
        res = {}                          # T -> {q: (judge, em)}
        img_of = {}
        for T, (suf, tag) in TT.items():
            stem = (f"{ER.FEATS}/generator_eval_ens{suf}_{cell}" if suf else
                    (f"{ER.FEATS}/generator_eval_finelayer" if cell in ER.SHARED
                     else f"{ER.FEATS}/generator_eval_finelayer_{cell}"))
            z = np.load(stem + ".npz"); m = json.load(open(stem + ".meta.json"))
            lay = [int(x) for x in z["layers"]]
            dsf = cell if (not suf and cell in ER.SHARED) else None
            keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and ER.half(r["img_md5"]) == 0
                    and (dsf is None or r.get("ds") == dsf)]
            rr = [m["rows"][i] for i in keep]
            H = z["h_span"]
            S = ER.batched_scores(heads, stats, {L: H[keep, lay.index(L)].astype(np.float32) for L in ER.ENS})
            del H
            sc = ER.loadj(f"{ER.CK}/ckpt_{cell}_{tag}_sc8.jsonl")
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            out = {}
            for q, ii in byq.items():
                if q not in gj or q not in sc:
                    continue
                ii = np.array(ii)
                p = ii[int(np.argmax(np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], 0)))]
                out[q] = (int(rr[p]["y"]), ER.em_score(rr[p]["ans"], sc[q]["gold"]))
                img_of[q] = rr[ii[0]]["img_md5"]
            res[T] = out
            print(f"  {cell:17} T={T} questions {len(out)} ({time.time() - t0:.0f}s)", flush=True)
        qs = sorted(set.intersection(*[set(v) for v in res.values()]), key=str)
        qs = [q for q in qs if q in greedy]
        g = {q: (gj[q], ER.em_score(greedy[q]["preds"][0], greedy[q]["gold"])) for q in qs}
        fq = {q: fold(img_of[q]) for q in qs}
        chosen = {}
        for f in (0, 1):                  # choose on fold f by judge gain, apply to the other fold
            qf = [q for q in qs if fq[q] == f]
            gain = {T: np.mean([res[T][q][0] - g[q][0] for q in qf]) if qf else -1 for T in TT}
            chosen[1 - f] = max(gain, key=gain.get)
        arms = {"greedy": lambda q: g[q], "crossfit": lambda q: res[chosen[fq[q]]][q]}
        for T in TT:
            arms[f"fixed_{T}"] = (lambda T_: (lambda q: res[T_][q]))(T)
        o = {"n": len(qs), "T_chosen_for_fold": {str(k): v for k, v in chosen.items()}}
        cols = {a: np.array([fn(q) for q in qs], float) for a, fn in arms.items()}   # (n, 2): judge, em
        for a in arms:
            if a != "greedy":
                o[f"{a}_minus_greedy_judge"] = float((cols[a][:, 0] - cols["greedy"][:, 0]).mean())
                o[f"{a}_minus_greedy_em"] = float((cols[a][:, 1] - cols["greedy"][:, 1]).mean())
        keys, inv = np.unique([img_of[q] for q in qs], return_inverse=True)
        W = np.random.default_rng(rng_master[b]).multinomial(
            len(keys), np.full(len(keys), 1 / len(keys)), size=NBOOT).astype(float)
        cnt = np.bincount(inv, minlength=len(keys)).astype(float)
        deltas[cell] = {}
        for cur, j in (("judge", 0), ("em", 1)):
            d = cols["crossfit"][:, j] - cols["fixed_0.7"][:, j]
            R = (W @ np.bincount(inv, weights=d, minlength=len(keys))) / (W @ cnt)
            o[f"crossfit_minus_fixed07_{cur}"] = float(d.mean())
            o[f"ci_crossfit_minus_fixed07_{cur}"] = [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))]
            deltas[cell][cur] = (float(d.mean()), R)
        per[cell] = o
        print(f"  {cell:17} chosen {o['T_chosen_for_fold']}  crossfit-fixed0.7 judge "
              f"{o['crossfit_minus_fixed07_judge']:+.4f} em {o['crossfit_minus_fixed07_em']:+.4f}", flush=True)
    macro = {}
    for cur in ("judge", "em"):
        pts = np.array([deltas[c][cur][0] for c in ER.BENCH])
        R = np.stack([deltas[c][cur][1] for c in ER.BENCH], 1).mean(1)
        lo, hi = float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))
        macro[f"crossfit_minus_fixed07_{cur}"] = {"macro": float(pts.mean()), "ci_image_clustered": [lo, hi],
                                                  "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                                                  "n_positive_benchmarks": int((pts > 0).sum())}
        for a in ["crossfit"] + [f"fixed_{T}" for T in TT]:
            macro[f"{a}_minus_greedy_{cur}"] = float(np.mean([per[c][f"{a}_minus_greedy_{cur}"] for c in ER.BENCH]))
    art = {"title": "Per-benchmark sampling temperature, chosen by cross-fitting inside the held-out half",
           "date": "2026-10-15", "no_fabricated_numbers": True, "cpu_only": True, "no_refit": True,
           "script": os.path.abspath(__file__), "probe_dir": ER.PROBE_DIR, "readout": "rank_avg (shipped)",
           "selection_currency": "judge of record (MedVLThinker-32B)",
           "ci_caveat": "T choice not re-made inside bootstrap resamples: interval somewhat anticonservative",
           "macro": macro, "per_benchmark": per, "seconds": round(time.time() - t0, 1)}
    json.dump(art, open(OUT, "w"), indent=1)
    print("\nMACRO:", json.dumps(macro, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
