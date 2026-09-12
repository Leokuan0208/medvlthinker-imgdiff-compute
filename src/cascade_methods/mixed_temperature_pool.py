#!/usr/bin/env python3
"""mixed_temperature_pool.py -- a UNION pool over temperatures: cold quality, hot coverage.

WHAT THE TEMPERATURE SWEEP SHOWED (head_temperature_2026-08-22.json).  The two terms of
`head - greedy = selection skill - sampling penalty` move in OPPOSITE directions with temperature:

    vqamed   T=0.2   h-g +0.0011   penalty ~0.008   oracle 0.1590
             T=0.4   h-g -0.0079   penalty  0.0157  oracle 0.1873
             T=0.7   h-g -0.0259   penalty  0.0385  oracle 0.2102
             T=1.0   h-g -0.0379   penalty  0.0595  oracle 0.1837

Cold pools are cheap to select over but contain fewer correct answers; hot pools carry the coverage
but a random draw from them is far worse than greedy.  Picking one temperature per cell captures
whichever term dominates there -- and picking it PER CELL means reading the eval labels, which is
selection on the test set and not a deployable rule.

A UNION over temperatures needs no such choice.  Take every distinct candidate the model produced at
any temperature, let the head rank them, and both terms improve at once: oracle can only rise, and
the cold samples put a greedy-quality answer in the pool for the head to find.

TWO THINGS THIS MUST REPORT HONESTLY OR IT IS NOT A RESULT:
  compute      a union of k temperatures costs k x 8 generations, so it is NOT free.  The
               budget-matched comparison is the union SUBSAMPLED BACK TO 8 candidates, and that is
               reported beside the full union.
  oracle       a bigger pool trivially raises oracle@N, so the union must be judged on what the
               HEAD actually picks, against greedy, not on the ceiling it unlocks.

  python3 src/cascade_methods/mixed_temperature_pool.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/mixed_temperature_2026-08-22.json")
LAYERS = [7, 14, 21, 28]
TAGS = {"lingshu7b": (0.7, ""), "lingshu7bT02": (0.2, "_T02"),
        "lingshu7bT04": (0.4, "_T04"), "lingshu7bT10": (1.0, "_T10")}
CELLS = ["vqa_rad_open", "slake_open", "pathvqa_open", "radimagenet_open",
         "vqamed_open", "gemex_open", "omnimed_open", "kvasir_x1_open"]
from free_signal_bakeoff import norm


def boot(a, b, clusters, nboot=4000, seed=11):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    groups = [np.where(np.asarray(clusters) == c)[0] for c in np.unique(clusters)]
    d = np.empty(nboot)
    for i in range(nboot):
        s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        d[i] = a[s].mean() - b[s].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": float(a.mean() - b.mean()), "ci": [float(lo), float(hi)],
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"}


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    art = {"title": "Union-over-temperature candidate pools", "date": "2026-08-22",
           "no_fabricated_numbers": True, "cells": {}}
    rng = np.random.default_rng(0)

    for cell in CELLS:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        # gather every temperature that has BOTH features and a judged pool
        pools = defaultdict(dict)       # qid -> normalized answer -> (y, head_logit_vector)
        temps = []
        for tag, (T, suf) in TAGS.items():
            stem = f"{FEATS}/generator_eval_{cell}{suf}"
            if not (os.path.exists(stem + ".npz") and os.path.getsize(stem + ".npz") > 1e7):
                continue
            meta = json.load(open(stem + ".meta.json"))
            keep = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
            if not keep:
                continue
            rows = [meta["rows"][i] for i in keep]
            H = np.load(stem + ".npz")["h_span"][keep, LAYERS.index(21)].astype(np.float32)
            L = sel.head_logits(H)                       # (seeds, n)
            temps.append(T)
            for i, r in enumerate(rows):
                a = norm(r["na"])
                # first temperature to produce an answer owns it; identical strings score identically
                pools[r["idx"]].setdefault(a, (int(r["y"]), L[:, i]))
        if len(temps) < 2:
            continue
        # CLUSTER BUGFIX 2026-09-12: boot() was called with UNIQUE QUESTION IDS as the cluster
        # key, so every group was a singleton and the cluster machinery collapsed to i.i.d. --
        # while reading as clustered. These benchmarks put up to 21 questions on one image, so the
        # unit has to be the image. 8 WIN/LOSS verdicts in mixed_temperature_2026-08-22.json were
        # published on that degenerate interval.
        q_img = {}
        for _t, (_T, _suf) in TAGS.items():
            _st = f"{FEATS}/generator_eval_{cell}{_suf}"
            if os.path.exists(_st + ".meta.json"):
                for _r in json.load(open(_st + ".meta.json"))["rows"]:
                    if _r.get("n_tok", -1) > 0:
                        q_img.setdefault(_r["idx"], _r["img_md5"])
        qs = [q for q in pools if q in gok and len(pools[q]) >= 1 and q in q_img]
        if len(qs) < 50:
            continue

        # per-temperature reference (each cell's own single-T pools), computed on the SAME questions
        single = {}
        for tag, (T, suf) in TAGS.items():
            stem = f"{FEATS}/generator_eval_{cell}{suf}"
            if not (os.path.exists(stem + ".npz") and os.path.getsize(stem + ".npz") > 1e7):
                continue
            meta = json.load(open(stem + ".meta.json"))
            keep = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
            rows = [meta["rows"][i] for i in keep]
            H = np.load(stem + ".npz")["h_span"][keep, LAYERS.index(21)].astype(np.float32)
            L = sel.head_logits(H)
            by = defaultdict(list)
            for i, r in enumerate(rows):
                by[r["idx"]].append(i)
            hd, orc = [], []
            for q in qs:
                ii = np.array(by.get(q, []))
                if len(ii) == 0:
                    hd.append(gok[q]); orc.append(0); continue
                hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
                yy = np.array([rows[i]["y"] for i in ii])
                hd.append(int(yy[int(np.argmax(hr))])); orc.append(int(yy.max()))
            single[str(T)] = {"head": float(np.mean(hd)), "oracle": float(np.mean(orc))}

        uni_hd, uni_or, sub_hd, sub_or, gr, npool = [], [], [], [], [], []
        for q in qs:
            items = list(pools[q].items())
            y = np.array([v[0] for _, v in items])
            Lq = np.stack([v[1] for _, v in items], axis=1)          # (seeds, n)
            hr = np.mean([rank_avg(Lq[k]) for k in range(Lq.shape[0])], axis=0)
            uni_hd.append(int(y[int(np.argmax(hr))])); uni_or.append(int(y.max()))
            npool.append(len(items))
            # budget-matched: 8 candidates drawn uniformly from the union
            k = min(8, len(items))
            pick = rng.choice(len(items), k, replace=False)
            hs = np.mean([rank_avg(Lq[j][pick]) for j in range(Lq.shape[0])], axis=0)
            sub_hd.append(int(y[pick][int(np.argmax(hs))])); sub_or.append(int(y[pick].max()))
            gr.append(gok[q])
        uni_hd, uni_or, sub_hd, sub_or, gr = map(np.array,
                                                 (uni_hd, uni_or, sub_hd, sub_or, gr))
        best_single_T = max(single, key=lambda t: single[t]["head"])
        art["cells"][cell] = {
            "n_questions": len(qs), "temperatures_unioned": sorted(temps),
            "mean_union_pool_size": float(np.mean(npool)),
            "greedy": float(gr.mean()),
            "per_temperature_head": {t: v["head"] for t, v in single.items()},
            "per_temperature_oracle": {t: v["oracle"] for t, v in single.items()},
            "best_single_T_in_sample": best_single_T,
            "union_head": float(uni_hd.mean()), "union_oracle": float(uni_or.mean()),
            "union_sub8_head": float(sub_hd.mean()), "union_sub8_oracle": float(sub_or.mean()),
            "union_vs_greedy": boot(uni_hd, gr, [q_img[q] for q in qs]),
            "union_vs_best_single_T": float(uni_hd.mean() - single[best_single_T]["head"]),
            "sub8_vs_greedy": boot(sub_hd, gr, [q_img[q] for q in qs]),
            "sub8_vs_T07": (float(sub_hd.mean() - single["0.7"]["head"])
                            if "0.7" in single else None)}
        c = art["cells"][cell]
        print(f"  {cell:17} T{sorted(temps)} pool {c['mean_union_pool_size']:.1f}  "
              f"greedy {c['greedy']:.4f}  union {c['union_head']:.4f} "
              f"({c['union_vs_greedy']['delta']:+.4f} {c['union_vs_greedy']['verdict']})  "
              f"sub8 {c['union_sub8_head']:.4f} "
              f"({c['sub8_vs_greedy']['delta']:+.4f} {c['sub8_vs_greedy']['verdict']})  "
              f"bestT {best_single_T} {single[best_single_T]['head']:.4f}", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mu = float(np.mean([v["union_head"] - v["greedy"] for v in cs.values()]))
        ms = float(np.mean([v["union_sub8_head"] - v["greedy"] for v in cs.values()]))
        nb = sum(1 for v in cs.values() if v["union_vs_greedy"]["delta"] > 0)
        ns = sum(1 for v in cs.values() if v["union_sub8_head"] >= v["greedy"])
        art["summary"] = {"n_cells": len(cs), "macro_union_minus_greedy": mu,
                          "macro_sub8_minus_greedy": ms,
                          "cells_union_beats_greedy": f"{nb}/{len(cs)}",
                          "cells_sub8_not_worse_than_greedy": f"{ns}/{len(cs)}"}
        art["VERDICT"] = (
            f"union of temperatures: macro head-minus-greedy {mu:+.4f} at {len(cs)} cells "
            f"({nb}/{len(cs)} beat greedy) but it costs k x 8 generations; BUDGET-MATCHED to 8 "
            f"candidates it is {ms:+.4f} with {ns}/{len(cs)} cells not worse than greedy. "
            + ("The budget-matched union is the deployable form and it helps."
               if ms > 0.005 else
               "The gain is bought with extra generation, not with a better pool: once the budget "
               "is matched it does not survive."))
        print(f"\n  macro union-greedy {mu:+.4f} | budget-matched sub8 {ms:+.4f}")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
