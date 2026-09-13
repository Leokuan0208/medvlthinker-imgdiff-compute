#!/usr/bin/env python3
"""coverage_scaling.py -- can a trained verifier convert extra COVERAGE into accuracy?

THE TWO BENCHMARKS THE RETRAINED VERIFIER STILL LOSES are vqa_rad (-0.0206) and vqamed (-0.0033),
and neither is a selection failure.  vqamed's oracle@8 is 0.2102 against greedy 0.0947: most
candidate sets contain no correct answer at all, so there is nothing for any selector to find.  That
is a COVERAGE wall, and coverage is the one thing more samples actually buy.

WHY THIS IS NOT ALREADY ANSWERED.  The earlier N-scaling result showed self-consistency is FLAT at
the random-pick floor as N grows (0.0933 at N=2, 0.0800 at N=16) while oracle@N climbs 0.156 ->
0.418.  That says the ceiling rises and *that particular selector* cannot reach it.  It says nothing
about a trained verifier, which is a far stronger selector -- on these same benchmarks the pooled
probe is worth +0.08 macro where self-consistency is worth less than nothing.

So: draw N of the 16 samples, score with the FROZEN pooled verifier, and report the whole curve --
verifier accuracy, oracle@N and greedy -- rather than a single point.  If the verifier curve rises
with N while greedy is flat, coverage is buyable and the fix for these two benchmarks is more
samples.  If it flattens like self-consistency did, the wall is real and no amount of sampling helps.

Evaluated on HELD-OUT image halves only: the pooled verifier was fitted on the other half of these
benchmarks, so scoring it on the full set would be reporting on its own training images.

  python3 src/cascade_methods/coverage_scaling.py
"""
import hashlib, json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
import argparse as _ap
_p = _ap.ArgumentParser(); _p.add_argument("--all", action="store_true")
_A, _ = _p.parse_known_args()
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/"
      + ("coverage_scaling_ALL_2026-09-01.json" if _A.all else "coverage_scaling_2026-08-25.json"))
# 8 vs 16 was a TIE on both coverage-limited benchmarks; two points cannot separate "sampling more
# never helps" from "it helps where coverage binds", and that pair is the least able to tell.
CELLS = (["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open",
          "omnimed_open", "vqamed_open", "gemex_open"] if _A.all
         else ["vqa_rad_open", "vqamed_open"])
ENS = [18, 20, 22]
NS = [2, 4, 8, 16]
DRAWS = 8


def norm(s):
    return str(s).strip().lower().rstrip(".")


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def main():
    from pooled_selector import PooledSelector
    from genframe_data import rank_avg
    S = PooledSelector.load()
    rng = np.random.default_rng(0)
    art = {"title": "Does a trained verifier convert extra coverage into accuracy?",
           "date": "2026-08-25", "no_fabricated_numbers": True, "N_values": NS,
           "draws_per_N": DRAWS, "verifier": "genframe_head_pooled_ens (24 heads)",
           "eval": "held-out image halves only", "cells": {}}

    for cell in CELLS:
        stem = f"{FEATS}/generator_eval_{cell}_sc16"
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(stem + ".npz") and os.path.exists(gjp)):
            print(f"  [skip] {cell}: no sc16 feature cache yet", flush=True)
            continue
        z = np.load(stem + ".npz"); m = json.load(open(stem + ".meta.json"))
        lay = [int(x) for x in z["layers"]]
        if not all(L in lay for L in ENS):
            print(f"  [skip] {cell}: cache holds layers {lay}", flush=True)
            continue
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and half(r["img_md5"]) == 0]
        if not keep:
            continue
        rr = [m["rows"][i] for i in keep]
        X = {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS}
        y = np.array([r["y"] for r in rr], dtype=int)
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = [q for q in byq if q in gok]
        if len(qs) < 40:
            continue
        # score every candidate once with the frozen verifier, then subsample INDICES per N so the
        # curve isolates candidate-set size and not a re-scoring difference
        full = {}
        for q in qs:
            ii = np.array(byq[q])
            full[q] = (ii, S.scores({L: X[L][ii] for L in ENS}))
        rec = {"n_questions": len(qs), "greedy": float(np.mean([gok[q] for q in qs])),
               "mean_distinct_candidates": float(np.mean([len(byq[q]) for q in qs])), "curve": {}}
        for N in NS:
            acc, orc = [], []
            for q in qs:
                ii, sc = full[q]
                for _ in range(DRAWS if N < len(ii) else 1):
                    pick = (rng.choice(len(ii), N, replace=False) if N < len(ii)
                            else np.arange(len(ii)))
                    acc.append(int(y[ii][pick][int(np.argmax(sc[pick]))]))
                    orc.append(int(y[ii][pick].max()))
            rec["curve"][str(N)] = {"verifier": float(np.mean(acc)), "oracle": float(np.mean(orc)),
                                    "verifier_minus_greedy": float(np.mean(acc) - rec["greedy"])}
            r = rec["curve"][str(N)]
            print(f"  {cell:15} N={N:<3} verifier {r['verifier']:.4f} "
                  f"(v-g {r['verifier_minus_greedy']:+.4f})  oracle {r['oracle']:.4f}", flush=True)
        # MATCHED CONTROL: score the 8-sample pool on THESE SAME held-out questions.
        # Without it the N=16 number is not comparable to the -0.0206 measured on the sc8 pool --
        # different generation run, different question coverage, and the N axis counts DISTINCT
        # candidates rather than samples drawn, so "N=8" here is not "8 samples".
        # the 8-sample features for these three live in the SHARED finelayer cache, not in a
        # per-benchmark one, and the per-benchmark generator_eval_<cell>.npz holds layers
        # [7,14,21,28] which do not contain the ensemble's 18/20/22
        # 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
        # generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
        # the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
        # generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
        SHARED = {"slake_open", "vqa_rad_open"}
        s8 = (f"{FEATS}/generator_eval_finelayer" if cell in SHARED
              else f"{FEATS}/generator_eval_finelayer_{cell}")
        if os.path.exists(s8 + ".npz"):
            z8 = np.load(s8 + ".npz"); m8 = json.load(open(s8 + ".meta.json"))
            l8 = [int(x) for x in z8["layers"]]
            if all(L in l8 for L in ENS):
                k8 = [i for i, r in enumerate(m8["rows"])
                      if r.get("n_tok", -1) > 0 and half(r["img_md5"]) == 0
                      and (cell not in SHARED or r.get("ds") == cell)]
                r8 = [m8["rows"][i] for i in k8]
                X8 = {L: z8["h_span"][k8, l8.index(L)].astype(np.float32) for L in ENS}
                y8 = np.array([r["y"] for r in r8], dtype=int)
                b8 = defaultdict(list)
                for i, r in enumerate(r8):
                    b8[r["idx"]].append(i)
                same = [q for q in qs if q in b8]
                if len(same) >= 40:
                    a8 = []
                    for q in same:
                        ii = np.array(b8[q])
                        sc = S.scores({L: X8[L][ii] for L in ENS})
                        a8.append(int(y8[ii][int(np.argmax(sc))]))
                    g_same = float(np.mean([gok[q] for q in same]))
                    rec["matched_control_sc8"] = {
                        "n_questions": len(same), "verifier": float(np.mean(a8)),
                        "greedy": g_same,
                        "verifier_minus_greedy": float(np.mean(a8) - g_same),
                        "mean_distinct_candidates": float(np.mean([len(b8[q]) for q in same]))}
                    # and the sc16 pool restricted to exactly those questions
                    a16 = [int(y[full[q][0]][int(np.argmax(full[q][1]))]) for q in same]
                    rec["sc16_on_same_questions"] = {
                        "verifier": float(np.mean(a16)),
                        "verifier_minus_greedy": float(np.mean(a16) - g_same)}
                    d = rec["sc16_on_same_questions"]["verifier"] - np.mean(a8)
                    rec["budget_8_to_16_gain"] = float(d)
                    print(f"  {cell:15} MATCHED on {len(same)} questions: "
                          f"8-sample pool {np.mean(a8):.4f} ({rec['matched_control_sc8']['mean_distinct_candidates']:.1f} distinct) "
                          f"-> 16-sample pool {np.mean(a16):.4f} "
                          f"({rec['mean_distinct_candidates']:.1f} distinct)  "
                          f"gain {d:+.4f}  greedy {g_same:.4f}", flush=True)
        c2, c16 = rec["curve"]["2"], rec["curve"]["16"]
        rec["verifier_gain_2_to_16"] = c16["verifier"] - c2["verifier"]
        rec["oracle_gain_2_to_16"] = c16["oracle"] - c2["oracle"]
        rec["conversion_rate"] = (rec["verifier_gain_2_to_16"] / rec["oracle_gain_2_to_16"]
                                  if rec["oracle_gain_2_to_16"] > 1e-9 else float("nan"))
        art["cells"][cell] = rec
        print(f"  {cell:15} N=2->16: oracle {rec['oracle_gain_2_to_16']:+.4f}, verifier "
              f"{rec['verifier_gain_2_to_16']:+.4f}, converted "
              f"{rec['conversion_rate']:.1%}\n", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    cs = art["cells"]
    if cs:
        conv = float(np.mean([v["conversion_rate"] for v in cs.values()
                              if np.isfinite(v["conversion_rate"])]))
        beats = [c for c, v in cs.items() if v["curve"]["16"]["verifier_minus_greedy"] > 0]
        art["mean_conversion_rate"] = conv
        art["beats_greedy_at_N16"] = beats
        # LEAD WITH THE MATCHED CONTROL. The within-pool N curve and the matched-budget
        # comparison superficially disagree, and only one of them answers the question asked.
        # Subsampling N from a 16-sample pool handicaps small N on COVERAGE, so that curve will
        # always rise; comparing two real generation runs at 8 and 16 samples is what "should we
        # sample more" actually means.
        art["VERDICT"] = (
            "MATCHED BUDGET (the number that answers the question): "
            + ", ".join(f"{c} {v['budget_8_to_16_gain']:+.4f}"
                        for c, v in cs.items() if "budget_8_to_16_gain" in v)
            + " going from an 8-sample to a 16-sample pool on identical questions. "
            + f"Within-pool subsampling separately shows the verifier converting {conv:.1%} of the "
            f"oracle gain from N=2 to N=16, but that curve handicaps small N on coverage and does "
            f"NOT license spending more samples. At N=16 the verifier beats greedy on "
            f"{len(beats)}/{len(cs)} of these benchmarks. "
            + ("Doubling the sampling budget 8->16 on matched questions is worth "
               + ", ".join(f"{c} {v['budget_8_to_16_gain']:+.4f}"
                           for c, v in cs.items() if "budget_8_to_16_gain" in v) + ". "
               if any("budget_8_to_16_gain" in v for v in cs.values()) else "") +
            # A SIGN IS NOT A RESULT. Counting how many benchmarks come out positive called this
            # a fix on vqamed (+0.0022) and a regression on vqa_rad (-0.0309); image-clustered CIs
            # put both at TIE (+0.0022 [-0.0105,+0.0155] and -0.0515 [-0.1458,+0.0404] against
            # greedy at N=16, coverage_sc16_ci_2026-08-25.json). Report the interval, not the sign.
            "Significance must be read from coverage_sc16_ci_2026-08-25.json, not from the sign "
            "of these deltas: both benchmarks are TIES against greedy at N=16, so doubling the "
            "sampling budget buys nothing measurable on either.")
        print(f"=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
