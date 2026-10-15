#!/usr/bin/env python3
"""readout_variants.py -- backlog item 0.4 and two parameter-free neighbours: how should the 24 frozen
probes' scores be combined into one pick?  CPU only, nothing refit, shipped probes
(ckpts/train/genframe_head_pooled_ens_v2), held-out halves, 18,452 questions.

Variants (fixed before any result was seen, 2026-10-15):
  rank        SHIPPED. rank_avg per head within the candidate set, mean of 24, argmax (first index).
  logit_mean  mean of the 24 raw logits.                     (the known-bad baseline: scales differ)
  z_mean      each head's logit z-scored with that head's mean/sd over TRAIN-half rows, then mean.
              Backlog 0.4's "calibrate first, then average".  Uses no labels.
  prob_mean   mean of sigmoid(logit).
  platt_mean  each head Platt-scaled (1-D logistic regression of y on its logit) on TRAIN-half rows,
              then mean probability. NOTE: the heads were fitted on those rows, so this calibration
              is in-sample for the heads -- reported, but flagged.
  rank_vote   shipped rank readout, ties broken by the candidate's vote count (how many of the 8
              samples produced that string), then first index. Parameter-free.
  vote_prob   probe-weighted majority vote: argmax over candidates of votes x prob_mean
              (= summing each SAMPLE's probability over the 8 samples). Parameter-free.
Reference arms: greedy, self-consistency (the em-rescore definition), oracle.

Currencies on identical picks: judge of record (MedVLThinker-32B, meta row y), cross-family judge
(MedGemma-27B-it, audit 2026-09-20 labels), lenient EM and token-F1 (run_openvqa's scorer verbatim,
via em_rescore_pooled_probe). Bootstrap: 10,000 paired resamples clustered by image within benchmark;
macro = mean of 8 benchmark deltas.

PRE-SPECIFIED ADOPTION RULE: a variant replaces `rank` only if its paired macro delta over `rank` is
> 0 under BOTH judges, its image-clustered CI excludes 0 under at least one of them, and its EM delta
over `rank` is not significantly negative.

  OMP_NUM_THREADS=4 python3 readout_variants.py   (run from the repo root)
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json, time
from collections import defaultdict
import numpy as np
import torch

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, ".claude/worktrees/exp-2026-10-15/src/audit_2026_09_18"))
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
import em_rescore_pooled_probe as ER          # noqa: E402  (scorers, loaders, split, bootstrap)
from genframe_data import rank_avg            # noqa: E402

CACHE = "/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json"
XPRED = "/data/dan/audit_2026-09-18/tmp/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"
OUT = os.path.join(MAIN, "results/cascade_methods/artifacts/readout_variants_2026-10-15.json")
VARIANTS = ["rank", "logit_mean", "z_mean", "prob_mean", "platt_mean", "rank_vote", "vote_prob"]
NBOOT, SEED = 10000, 20261015


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def platt(s, y, iters=50):
    """1-D logistic regression y ~ sigmoid(a*s+b), Newton's method."""
    a, b = 1.0, 0.0
    X = np.stack([s, np.ones_like(s)], 1)
    w = np.array([a, b])
    for _ in range(iters):
        p = sig(X @ w)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w = w + np.linalg.solve(H, g)
    return w


def pick(v, S, votes, cal):
    if v == "rank":
        return int(np.argmax(np.mean([rank_avg(S[k]) for k in range(S.shape[0])], 0)))
    if v == "rank_vote":
        r = np.mean([rank_avg(S[k]) for k in range(S.shape[0])], 0)
        top = np.flatnonzero(r == r.max())
        return int(top[np.argmax(votes[top])])
    if v == "logit_mean":
        return int(np.argmax(S.mean(0)))
    if v == "z_mean":
        return int(np.argmax(((S - cal["mu"][:, None]) / cal["sd"][:, None]).mean(0)))
    if v == "prob_mean":
        return int(np.argmax(sig(S).mean(0)))
    if v == "platt_mean":
        return int(np.argmax(sig(cal["pa"][:, None] * S + cal["pb"][:, None]).mean(0)))
    if v == "vote_prob":
        return int(np.argmax(votes * sig(S).mean(0)))
    raise ValueError(v)


def main():
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    t0 = time.time()
    heads, stats, recipe = ER.load_probes(ER.PROBE_DIR)
    cache = json.load(open(CACHE))["records"]
    xlab = {}
    for l in open(XPRED):
        if l.strip():
            d = json.loads(l); c, i, _ = d["idx"].split("|"); xlab[(c, i, d["modal_pred"])] = d["idx"]
    xj = {}
    for l in open(XPRED.replace(".jsonl", ".judge.jsonl")):
        if l.strip():
            d = json.loads(l); xj.setdefault(d["idx"], int(d["judge_ok"]))

    # ---- pass 1: scores for every held-out and train-half row
    held, tr_s, tr_y = {}, [], []
    for cell in ER.BENCH:
        stem = "generator_eval_finelayer" if cell in ER.SHARED else f"generator_eval_finelayer_{cell}"
        z = np.load(f"{ER.FEATS}/{stem}.npz"); meta = json.load(open(f"{ER.FEATS}/{stem}.meta.json"))
        lay = [int(x) for x in z["layers"]]
        dsf = {cell} if cell in ER.SHARED else None
        ok = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
        H = z["h_span"]
        for hv in (0, 1):
            keep = [i for i in ok if ER.half(meta["rows"][i]["img_md5"]) == hv]
            X = {L: H[:, lay.index(L)][keep].astype(np.float32) for L in ER.ENS}
            S = ER.batched_scores(heads, stats, X)
            rr = [meta["rows"][i] for i in keep]
            if hv == 1:
                tr_s.append(S); tr_y.append(np.array([int(r["y"]) for r in rr]))
            else:
                held[cell] = (S, rr)
        del H
        print(f"  scored {cell} ({time.time() - t0:.0f}s)", flush=True)
    TS, TY = np.concatenate(tr_s, 1), np.concatenate(tr_y)
    cal = {"mu": TS.mean(1), "sd": TS.std(1)}
    w = np.array([platt(TS[k], TY) for k in range(TS.shape[0])])
    cal["pa"], cal["pb"] = w[:, 0], w[:, 1]
    print(f"  calibration on {TS.shape[1]} train-half rows; logit sd range "
          f"{cal['sd'].min():.3f}..{cal['sd'].max():.3f}", flush=True)

    # ---- pass 2: per-question picks, aligned to the em-rescore cache's candidate order
    per, mats = {}, {}
    ss = np.random.SeedSequence(SEED); child = ss.spawn(len(ER.BENCH))
    misalign = defaultdict(int)
    for b, cell in enumerate(ER.BENCH):
        S, rr = held[cell]
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        crec = {r["idx"]: r for r in cache[cell]}
        rows, imgs = [], []
        for q, rec in crec.items():
            ii = np.array(byq.get(q, []))
            if len(ii) != len(rec["cands"]) or any(rr[i]["ans"] != c["ans"] for i, c in zip(ii, rec["cands"])):
                misalign[cell] += 1
                continue
            Sq = S[:, ii]
            votes = np.array([c["votes"] for c in rec["cands"]], float)
            picks = {v: pick(v, Sq, votes, cal) for v in VARIANTS}
            assert picks["rank"] == rec["pick"], (cell, q)          # reproduces the shipped readout exactly

            def xb(a):
                k = xlab.get((cell, str(q), a))
                return xj.get(k) if k is not None else None
            cx = [xb(c["ans"]) for c in rec["cands"]]
            gx = xb(rec["greedy"]["ans"])
            row = {"greedy_j": rec["greedy"]["y"], "greedy_em": rec["greedy"]["em"], "greedy_f1": rec["greedy"]["f1"],
                   "greedy_x": gx}
            for v, p in list(picks.items()) + [("sc", rec["sc_pick"])]:
                c = rec["cands"][p]
                row[f"{v}_j"], row[f"{v}_em"], row[f"{v}_f1"], row[f"{v}_x"] = c["y"], c["em"], c["f1"], cx[p]
            rows.append(row); imgs.append(rec["img"])
        arms = VARIANTS + ["sc", "greedy"]
        o = {"n": len(rows)}
        cols = {}
        for a in arms:
            for cur in ("j", "em", "f1", "x"):
                vals = [r[f"{a}_{cur}"] for r in rows]
                cols[(a, cur)] = np.array([np.nan if x is None else float(x) for x in vals])
        # cross-judge: restrict to questions where every arm's pick has a MedGemma label
        xok = np.all([~np.isnan(cols[(a, "x")]) for a in arms], 0)
        o["n_xjudge_complete"] = int(xok.sum())
        for a in arms:
            for cur in ("j", "em", "f1"):
                o[f"{a}_{cur}"] = float(np.nanmean(cols[(a, cur)]))
            o[f"{a}_x"] = float(cols[(a, "x")][xok].mean())
        # paired bootstrap: each variant minus rank, and each variant minus greedy
        keys, inv = np.unique(np.asarray(imgs), return_inverse=True)
        rng = np.random.default_rng(child[b])
        W = rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=NBOOT).astype(float)

        def boot(diff, mask=None):
            m = np.ones(len(diff), bool) if mask is None else mask
            Ssum = np.bincount(inv[m], weights=diff[m], minlength=len(keys))
            cnt = np.bincount(inv[m], minlength=len(keys)).astype(float)
            return (W @ Ssum) / np.maximum(W @ cnt, 1e-12)
        mats[cell] = {}
        for a in VARIANTS + ["sc"]:
            for base in ("rank", "greedy"):
                if a == base:
                    continue
                for cur in ("j", "em", "f1", "x"):
                    d = cols[(a, cur)] - cols[(base, cur)]
                    mask = xok if cur == "x" else None
                    pt = float(d[mask].mean()) if mask is not None else float(d.mean())
                    R = boot(np.nan_to_num(d), mask)
                    key = f"{a}_minus_{base}_{cur}"
                    o[key] = pt; o["ci_" + key] = [float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))]
                    mats[cell][key] = (pt, R)
        per[cell] = o
        print(f"  {cell:17} n {o['n']:5} | judge rank {o['rank_j'] - o['greedy_j']:+.4f} z {o['z_mean_j'] - o['greedy_j']:+.4f} "
              f"vote_prob {o['vote_prob_j'] - o['greedy_j']:+.4f} rank_vote {o['rank_vote_j'] - o['greedy_j']:+.4f}", flush=True)

    macro = {}
    for key in mats[ER.BENCH[0]]:
        pts = np.array([mats[c][key][0] for c in ER.BENCH])
        R = np.stack([mats[c][key][1] for c in ER.BENCH], 1).mean(1)
        lo, hi = float(np.percentile(R, 2.5)), float(np.percentile(R, 97.5))
        macro[key] = {"macro": float(pts.mean()), "ci_image_clustered": [lo, hi],
                      "verdict": "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE"),
                      "n_positive_benchmarks": int((pts > 0).sum())}
    adopt = {}
    for a in VARIANTS[1:]:
        mj, mx, me = macro[f"{a}_minus_rank_j"], macro[f"{a}_minus_rank_x"], macro[f"{a}_minus_rank_em"]
        adopt[a] = bool(mj["macro"] > 0 and mx["macro"] > 0 and (mj["verdict"] == "WIN" or mx["verdict"] == "WIN")
                        and me["verdict"] != "LOSS")
    art = {"title": "How to combine the 24 frozen probes: readout variants on identical held-out questions",
           "date": "2026-10-15", "no_fabricated_numbers": True, "cpu_only": True, "no_refit": True,
           "script": os.path.abspath(__file__), "probe_dir": ER.PROBE_DIR, "em_rescore_cache": CACHE,
           "xjudge_labels": XPRED, "variants": VARIANTS,
           "adoption_rule": "variant replaces rank only if macro delta over rank > 0 under BOTH judges, image-clustered CI "
                            "excludes 0 under at least one, and EM delta over rank is not a significant LOSS",
           "adopt": adopt, "calibration_rows": int(TS.shape[1]), "misaligned_questions_skipped": dict(misalign),
           "threads": int(os.environ["OMP_NUM_THREADS"]), "bootstrap": {"n": NBOOT, "seed": SEED,
           "cluster": "img_md5 within benchmark, paired"}, "macro": macro, "per_benchmark": per,
           "seconds": round(time.time() - t0, 1)}
    json.dump(art, open(OUT, "w"), indent=1)
    print("\nMACRO (variant minus shipped rank):")
    for a in VARIANTS[1:] + ["sc"]:
        print(f"  {a:11}" + "  ".join(
            f"{cur} {macro[f'{a}_minus_rank_{cur}']['macro']:+.4f} {macro[f'{a}_minus_rank_{cur}']['verdict']:4}"
            for cur in ("j", "x", "em", "f1")))
    print("\nMACRO vs greedy:")
    for a in VARIANTS + ["sc"]:
        print(f"  {a:11}" + "  ".join(f"{cur} {macro[f'{a}_minus_greedy_{cur}']['macro']:+.4f}" for cur in ("j", "x", "em", "f1")))
    print("adopt:", adopt, "| misaligned:", dict(misalign), f"| wrote {OUT} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
