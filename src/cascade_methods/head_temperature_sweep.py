#!/usr/bin/env python3
"""head_temperature_sweep.py -- does a COLDER pool make selection beat greedy?

THE DECOMPOSITION THIS TESTS (2026-08-22).  head - greedy factors exactly into

    (head - one sample)  -  (greedy - one sample)  =  selection skill - sampling penalty

and selection skill is positive on 7 of 8 cells.  Where the head loses it is not because it selects
badly but because the T=0.7 pool it selects over is worse than greedy decoding: vqa_rad has +0.0436
of skill and a 0.0686 penalty.  Temperature acts directly on the penalty, so the question is whether
a colder pool can bring the penalty under the skill.

THE CATCH, and why this needs measuring rather than reasoning.  Lowering T shrinks the penalty AND
the pool's diversity, and diversity is what puts a correct answer in the pool at all.  Measured on
vqamed going T=0.7 -> T=0.4: penalty 0.0434 -> 0.0214 (good) but headroom oracle-greedy
0.1155 -> 0.0925 (bad).  Both terms move the same way, so the net is empirical.

Reports, per cell and per temperature, on a FIXED question set (the questions greedy also covers):
greedy, one-sample, oracle@8, the head's deployed pick, self-consistency, and the string prior --
so the temperature that maximises head-minus-greedy can be read off directly, with the null beside
it.  Cells are included at whichever temperatures have BOTH a judge file and a feature cache.

  python3 src/cascade_methods/head_temperature_sweep.py
"""
import glob, json, os, re, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/head_temperature_2026-08-22.json")
LAYERS = [7, 14, 21, 28]
TAGS = {"lingshu7b": 0.7, "lingshu7bT02": 0.2, "lingshu7bT04": 0.4, "lingshu7bT10": 1.0}
SUF = {"lingshu7b": "", "lingshu7bT02": "_T02", "lingshu7bT04": "_T04", "lingshu7bT10": "_T10"}
from free_signal_bakeoff import norm


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    art = {"title": "Head, greedy, oracle and the nulls as functions of pool temperature",
           "date": "2026-08-22", "no_fabricated_numbers": True, "cells": defaultdict(dict)}

    # string prior from the head's own training rows, fitted once
    pos, tot, ntr = defaultdict(int), defaultdict(int), 0
    for sh in (0, 1):
        for r in json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]:
            if r.get("n_tok", -1) > 0:
                a = norm(r["na"]); tot[a] += 1; pos[a] += int(r["y"]); ntr += 1
    gp = sum(pos.values()) / max(ntr, 1)

    cells = sorted({re.match(r"generator_eval_(.+?)(_T0\d|_T10)?$", os.path.basename(p)[:-4]).group(1)
                    for p in glob.glob(f"{FEATS}/generator_eval_*.npz")
                    if re.match(r"generator_eval_(.+?)(_T0\d|_T10)?$", os.path.basename(p)[:-4])})
    for cell in cells:
        if cell in ("s0of2", "s1of2", "finelayer") or cell.startswith("finelayer"):
            continue
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        for tag, T in sorted(TAGS.items(), key=lambda kv: kv[1]):
            stem = f"{FEATS}/generator_eval_{cell}{SUF[tag]}"
            rawp = f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl"
            if not (os.path.exists(stem + ".npz") and os.path.exists(rawp)):
                continue
            meta = json.load(open(stem + ".meta.json"))
            keep = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
            rows = [meta["rows"][i] for i in keep]
            if not rows:
                continue
            H = np.load(stem + ".npz")["h_span"][keep, LAYERS.index(21)].astype(np.float32)
            y = np.array([r["y"] for r in rows], dtype=int)
            na = np.array([norm(r["na"]) for r in rows])
            byq = defaultdict(list)
            for i, r in enumerate(rows):
                byq[r["idx"]].append(i)
            raw = {}
            for l in open(rawp):
                if l.strip():
                    d = json.loads(l); raw[d["idx"]] = d
            # exploded judge labels, keyed question -> normalised answer -> judge_ok. The exploded
            # file dedups identical (question, answer) pairs, so it has one row per DISTINCT answer;
            # multiplicity has to come from the raw preds list above.
            lab = {}
            expp = rawp.replace(".jsonl", "_scexploded.jsonl")
            judp = rawp.replace(".jsonl", "_scexploded.judge.jsonl")
            if os.path.exists(expp) and os.path.exists(judp):
                _j = {}
                for l in open(judp):
                    if l.strip():
                        _d = json.loads(l); _j[_d["idx"]] = int(_d["judge_ok"])
                for l in open(expp):
                    if l.strip():
                        _d = json.loads(l)
                        if _d["idx"] in _j:
                            lab.setdefault(str(_d["idx"]).rsplit("#", 1)[0], {})[
                                norm(_d["modal_pred"])] = _j[_d["idx"]]
            L = sel.head_logits(H)
            sp = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])
            sc = np.zeros(len(rows), np.float32)
            for i, r in enumerate(rows):
                d = raw.get(r["idx"])
                if d:
                    c = Counter(norm(p) for p in d["preds"])
                    sc[i] = c.get(na[i], 0) / max(len(d["preds"]), 1)
            hd, sd_, pr, orc, gr = [], [], [], [], []
            for q in sorted(byq, key=lambda k: (len(str(k)), str(k))):
                if q not in gok:
                    continue
                ii = np.array(byq[q])
                hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
                hd.append(int(y[ii][int(np.argmax(hr))]))
                sd_.append(int(y[ii][int(np.argmax(rank_avg(sc[ii])))]))
                pr.append(int(y[ii][int(np.argmax(sp[ii]))]))
                orc.append(int(y[ii].max())); gr.append(gok[q])
            if len(hd) < 50:
                continue
            hd, sd_, pr, orc, gr = map(np.array, (hd, sd_, pr, orc, gr))
            art["cells"][cell][str(T)] = {
                "n_questions": int(len(hd)), "greedy": float(gr.mean()),
                "head": float(hd.mean()),
                "self_consistency": float(sd_.mean()), "string_prior": float(pr.mean()),
                "oracle_at_8": float(orc.mean()),
                "head_minus_greedy": float(hd.mean() - gr.mean()),
                # selection_skill and sampling_penalty are NOT computed here any more.
                # 2026-09-12: this script and decomposition_report.py both derived them and
                # disagreed on all 8 benchmarks even after the currency fix, because this one fell
                # back to a distinct-answer mean whenever an exploded judge label was missing while
                # the other skipped those questions. One quantity, two implementations, two answers
                # is the bug class. decomposition_2026-08-24.json is the single source of truth for
                # the skill/penalty decomposition; this file owns head-vs-greedy across temperature.
                "one_sample_NOT_COMPUTED_see": "decomposition_2026-08-24.json",
                "headroom_above_greedy": float(orc.mean() - gr.mean()),
                "mean_distinct_candidates": float(len(rows) / max(len(byq), 1))}
            r = art["cells"][cell][str(T)]
            print(f"  {cell:17} T={T:<4} head {r['head']:.4f} greedy {r['greedy']:.4f} "
                  f"h-g {r['head_minus_greedy']:+.4f}  oracle {r['oracle_at_8']:.4f}", flush=True)
            json.dump({**art, "cells": dict(art["cells"])}, open(OUT, "w"), indent=1)

    art["cells"] = dict(art["cells"])
    multi = {c: v for c, v in art["cells"].items() if len(v) > 1}
    best = {}
    for c, v in multi.items():
        Ts = sorted(float(t) for t in v)
        bt = max(Ts, key=lambda t: v[str(t)]["head_minus_greedy"])
        best[c] = {"temperatures": Ts, "best_T": bt,
                   "head_minus_greedy_by_T": [v[str(t)]["head_minus_greedy"] for t in Ts],
                   "gain_over_T07": (v[str(bt)]["head_minus_greedy"] - v["0.7"]["head_minus_greedy"]
                                     if "0.7" in v else None)}
        print(f"\n  {c}: T {Ts}  h-g {[round(x,4) for x in best[c]['head_minus_greedy_by_T']]}"
              f"  -> best T={bt}")
    art["best_temperature_per_cell"] = best
    if best:
        g = [b["gain_over_T07"] for b in best.values() if b["gain_over_T07"] is not None]
        art["VERDICT"] = (
            f"tuning temperature per cell is worth {np.mean(g):+.4f} mean head-minus-greedy over "
            f"{len(g)} cells measured at more than one T; best T is 0.7 for "
            f"{sum(1 for b in best.values() if b['best_T'] == 0.7)}/{len(best)} of them."
            if g else "not enough cells measured at more than one temperature yet")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
