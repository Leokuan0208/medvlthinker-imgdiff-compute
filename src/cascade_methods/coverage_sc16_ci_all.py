#!/usr/bin/env python3
"""
coverage_sc16_ci_all.py -- image-clustered CIs on the 8->16 sampling-budget question, on ALL EIGHT
benchmarks.

WHY.  coverage_scaling_ALL_2026-09-01.json reports the matched-budget 8->16 gain per benchmark as a
POINT ESTIMATE, and its own VERDICT says significance must be read from
coverage_sc16_ci_2026-08-25.json -- which covers **two** benchmarks, vqamed_open and vqa_rad_open,
both TIES, and both the LEAST sampling-responsive of the eight.  So the project's headline
"doubling the sampling budget is worth +0.0147 macro" rests on six benchmarks with no interval at
all, and the only two that HAVE intervals are the two least able to speak for the rest.  No script
on disk produces that artifact; the two-benchmark version was computed ad hoc.  This produces it
for all eight, from one implementation.

WHAT IS COMPUTED, per benchmark, on questions present in BOTH the sc8 and sc16 pools (matched --
the same questions, so the only thing that differs is how many candidates were drawn):
    acc@8      the shipped probe's pick out of the 8-candidate set
    acc@16     the same probe's pick out of the 16-candidate set
    greedy     the 7B's greedy answer
  and three paired, IMAGE-CLUSTERED bootstrap intervals (never i.i.d. over questions -- a project
  landmine: an i.i.d. CI once turned a TIE into a "WIN"):
    acc@16 - acc@8        <- the budget question the deck hedges on
    acc@8  - greedy
    acc@16 - greedy

CURRENCY.  32B judge throughout (`judge_ok`), never exact match.  Mixing the two inflated a
sampling penalty 12.4x in this repo before.

READOUT.  The shipped readout: rank_avg over the 24 probes (3 layers x 8 seeds), which is
scale-free within a candidate set.  Raw-score averaging across separately fitted probes is wrong.

WHICH VERIFIER, AND ON WHICH QUESTIONS -- the thing that makes or breaks the comparison.
  --verifier pooled   (DEFAULT) genframe_head_pooled_ens_v2, the SHIPPED probe, scored on HELD-OUT
                      image halves ONLY, because it was fitted on the other half. This is the arm
                      coverage_scaling_ALL_2026-09-01.json measures, so its point estimates and
                      these intervals describe the same thing.
  --verifier incumbent  genframe_head_ens8, the old four-domain layer-21 probe, which was never
                      fitted on any of these benchmarks and so can legitimately be scored on ALL
                      questions. A different arm. Do NOT compare its number to the pooled one.
  The first run of this script used `incumbent` on ALL questions and got macro -0.0046; that is a
  real measurement of a DIFFERENT arm and must not be read as overturning the pooled result.

  python3 src/cascade_methods/coverage_sc16_ci_all.py --nboot 10000
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
ENS = [18, 20, 22]
HELDOUT = [False]          # set from --heldout before any load()
HALF = [None]
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
# slake and vqa_rad live in the combined finelayer cache; the rest have their own.
SHARED = {"slake_open", "vqa_rad_open"}


def load(stem, dsf, want):
    """Return {layer: X}, rows -- only rows the generator actually produced (n_tok > 0)."""
    z = np.load(f"{FEATS}/{stem}.npz")
    m = json.load(open(f"{FEATS}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    miss = [L for L in want if L not in lay]
    if miss:
        return None, None
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") == dsf)
            and (not HELDOUT[0] or HALF[0](r["img_md5"]) == 0)]
    if not keep:
        return None, None
    rows = [m["rows"][i] for i in keep]
    return {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in want}, rows


def picks(sel, rank_avg, X, rows, gok):
    """Per question: 1 if the probe's top-ranked candidate is judge-correct. Returns arrays aligned
    on a sorted question list, plus that list and its image ids."""
    if hasattr(sel, "head_logits"):
        L = {l: sel.head_logits(X[l]) for l in ENS}          # frozen incumbent: [n_heads, n_rows]
        def score(ii):
            return np.mean([rank_avg(L[l][k][ii]) for l in ENS for k in range(L[l].shape[0])], axis=0)
    else:
        SC = sel.scores(X)                                    # pooled: already the stacked scores
        SC = np.asarray(SC)
        if SC.ndim == 1: SC = SC[None, :]
        def score(ii):
            return np.mean([rank_avg(SC[k][ii]) for k in range(SC.shape[0])], axis=0)
    y = np.array([r["y"] for r in rows], dtype=int)
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[r["idx"]].append(i)
    img = {r["idx"]: r["img_md5"] for r in rows}
    qs = sorted([q for q in byq if q in gok], key=lambda k: (len(str(k)), str(k)))
    out = []
    for q in qs:
        ii = np.array(byq[q])
        hr = score(ii)
        out.append(int(y[ii][int(np.argmax(hr))]))
    return np.array(out), qs, np.array([img[q] for q in qs])


def boot(delta, clusters, nboot, seed=0):
    """Paired bootstrap resampling IMAGES, not questions."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    keys = list(by)
    idx = [np.array(by[k]) for k in keys]
    out = np.empty(nboot)
    for b in range(nboot):
        s = rng.integers(0, len(keys), len(keys))
        out[b] = delta[np.concatenate([idx[j] for j in s])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def verdict(d, lo, hi):
    if lo > 0: return "WIN"
    if hi < 0: return "LOSS"
    return "TIE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--verifier", choices=["pooled", "incumbent"], default="pooled")
    ap.add_argument("--heldout", dest="heldout", action="store_true", default=None,
                    help="score held-out image halves only (forced ON for --verifier pooled)")
    ap.add_argument("--all_questions", dest="heldout", action="store_false")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results/cascade_methods/artifacts/coverage_sc16_ci_ALL_2026-09-16.json"))
    A = ap.parse_args()
    from genframe_data import rank_avg
    import hashlib
    if A.verifier == "pooled":
        from pooled_selector import PooledSelector
        sel = PooledSelector.load()
        if A.heldout is None:
            A.heldout = True
        assert A.heldout, ("the pooled probe was FITTED on the train halves; scoring it on all "
                           "questions would be train contamination")
    else:
        from genframe_selector import FrozenSelector
        sel = FrozenSelector.load()
        if A.heldout is None:
            A.heldout = False
    def half(img):
        return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2
    HELDOUT[0] = bool(A.heldout); HALF[0] = half
    print(f"verifier={A.verifier}  heldout_only={A.heldout}", flush=True)

    art = {"title": "Image-clustered CIs on the 8->16 sampling budget, all eight benchmarks",
           "date": "2026-09-16", "no_fabricated_numbers": True, "nboot": A.nboot,
           "currency": "32B judge (judge_ok) throughout",
           "readout": "rank_avg over the 24 frozen probes (3 layers x 8 seeds)",
           "clustering": "paired bootstrap over IMAGES within benchmark, never i.i.d. over questions",
           "supersedes": ("coverage_sc16_ci_2026-08-25.json, which covered vqamed_open and "
                          "vqa_rad_open only -- the two least sampling-responsive benchmarks."),
           "verifier": A.verifier, "heldout_only": bool(A.heldout),
           "cells": {}, "COMPLETE": False}

    for b in BENCH:
        gjp = f"{CK}/ckpt_{b}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            print(f"  [skip] {b}: no greedy judge", flush=True); continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])

        s8 = (f"generator_eval_finelayer_{b}"
              if os.path.exists(f"{FEATS}/generator_eval_finelayer_{b}.npz")
              else "generator_eval_finelayer")
        X8, r8 = load(s8, b if s8 == "generator_eval_finelayer" else None, ENS)
        X16, r16 = load(f"generator_eval_{b}_sc16", None, ENS)
        if X8 is None or X16 is None:
            print(f"  [skip] {b}: missing a cache at layers {ENS}", flush=True); continue

        a8, q8, c8 = picks(sel, rank_avg, X8, r8, gok)
        a16, q16, c16 = picks(sel, rank_avg, X16, r16, gok)

        # MATCHED: only questions present in both pools, so budget is the only difference.
        m8 = {q: i for i, q in enumerate(q8)}
        m16 = {q: i for i, q in enumerate(q16)}
        common = [q for q in q8 if q in m16]
        if len(common) < 40:
            print(f"  [skip] {b}: only {len(common)} matched questions", flush=True); continue
        i8 = np.array([m8[q] for q in common]); i16 = np.array([m16[q] for q in common])
        A8, A16 = a8[i8], a16[i16]
        G = np.array([gok[q] for q in common])
        CL = c8[i8]

        rec = {"n_matched_questions": len(common), "n_images": len(set(CL.tolist())),
               "greedy": float(G.mean()), "acc_at_8": float(A8.mean()), "acc_at_16": float(A16.mean())}
        for nm, dl in (("budget_16_minus_8", A16 - A8),
                       ("at8_minus_greedy", A8 - G),
                       ("at16_minus_greedy", A16 - G)):
            lo, hi = boot(dl.astype(float), CL, A.nboot)
            rec[nm] = {"delta": float(dl.mean()), "ci": [lo, hi], "verdict": verdict(dl.mean(), lo, hi),
                       "discordant": int((dl != 0).sum())}
        art["cells"][b] = rec
        print(f"  {b:18s} n {len(common):5d} | greedy {rec['greedy']:.4f} @8 {rec['acc_at_8']:.4f} "
              f"@16 {rec['acc_at_16']:.4f} | 16-8 {rec['budget_16_minus_8']['delta']:+.4f} "
              f"[{rec['budget_16_minus_8']['ci'][0]:+.4f},{rec['budget_16_minus_8']['ci'][1]:+.4f}] "
              f"{rec['budget_16_minus_8']['verdict']}", flush=True)
        art["COMPLETE"] = False
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        d = [v["budget_16_minus_8"]["delta"] for v in cs.values()]
        w = [k for k, v in cs.items() if v["budget_16_minus_8"]["verdict"] == "WIN"]
        l = [k for k, v in cs.items() if v["budget_16_minus_8"]["verdict"] == "LOSS"]
        art["macro_budget_16_minus_8"] = float(np.mean(d))
        art["VERDICT"] = (
            f"Doubling the sampling budget 8->16 is worth {np.mean(d):+.4f} macro across "
            f"{len(cs)} benchmarks. With image-clustered intervals: {len(w)} WIN ({', '.join(w) or 'none'}), "
            f"{len(l)} LOSS ({', '.join(l) or 'none'}), {len(cs)-len(w)-len(l)} TIE. "
            "This replaces the two-benchmark read that the deck and coverage_scaling_ALL both hedge on.")
        print(f"\n  MACRO 16-8 {np.mean(d):+.4f} | WIN {len(w)} LOSS {len(l)} TIE {len(cs)-len(w)-len(l)}")
        print(f"=> {art['VERDICT']}")
    art["COMPLETE"] = True
    tmp = A.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(art, f, indent=1)
    os.replace(tmp, A.out)
    print(f"wrote {A.out}  COMPLETE")


if __name__ == "__main__":
    main()
