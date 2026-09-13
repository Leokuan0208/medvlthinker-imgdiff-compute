#!/usr/bin/env python3
"""tiebreak_bakeoff.py -- the readout is not order-invariant, and that is a live bug.

FOUND 2026-09-13, chasing a 2-question disagreement between free_signal_bakeoff.py and
head_temperature_sweep.py on slake_open (frozen head 0.768992 vs 0.772093 = exactly 2/645).
The two scripts read the SAME candidates with the SAME labels out of two different feature
caches -- generator_eval_slake_open vs the generator_eval_s{0,1}of2 shards -- and 219 of the
645 questions carry their candidates in a DIFFERENT ROW ORDER.  rank_avg can tie exactly
(ranks are integers averaged over heads, and near-duplicate candidates get identical ranks),
and np.argmax breaks a tie by taking the FIRST row.  So the selected answer depends on which
cache the analysis happened to read.  This is the "feature row order" landmine in CLAUDE.md
section 0 showing up in a reported number rather than in a fit.

This script measures the size of the problem and bakes off deterministic, cache-independent
tie-break rules against the status quo:

  first_row   status quo -- np.argmax, i.e. whichever candidate the cache lists first
  lexical     lowest normalised answer string.  Neutral: uses no signal, only determinism.
  selfcons    most-sampled candidate in the pool (multiplicity), lexical as the inner tie-break.
  longest     longest normalised answer, lexical inner.  A verbosity control -- if `selfcons`
              wins, this says whether it won on multiplicity or just on string length.

and reports the ambiguity band -- accuracy if every tie broke in the best possible way vs the
worst -- which upper-bounds what ANY tie-break rule can be worth.

  python3 src/cascade_methods/tiebreak_bakeoff.py
"""
import json, os, re, sys
from collections import Counter, defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/tiebreak_2026-09-13.json")
LAYERS = [7, 14, 21, 28]
CELLS = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
from free_signal_bakeoff import norm


def boot(pick, gr, clusters, n=10000, seed=0):
    """Clustered bootstrap over IMAGES of the paired delta pick-minus-greedy."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, c in enumerate(clusters):
        by[c].append(i)
    keys = list(by)
    idx = [np.array(by[k]) for k in keys]
    d = pick - gr
    out = np.empty(n)
    for b in range(n):
        s = rng.integers(0, len(keys), len(keys))
        out[b] = d[np.concatenate([idx[j] for j in s])].mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    art = {"title": "Tie-break dependence of the best-of-N readout, and deterministic replacements",
           "date": "2026-09-13", "no_fabricated_numbers": True,
           "why": "np.argmax over rank_avg breaks exact ties by cache row order; two caches for the "
                  "same benchmark list candidates in different orders, so the reported number moved.",
           "cells": {}}
    RULES = ["first_row", "lexical", "selfcons", "longest"]
    agg = {r: [] for r in RULES}
    agg["best_case"], agg["worst_case"], agg["greedy"] = [], [], []
    tie_tot = tie_q = 0

    for cell in CELLS:
        stem = f"{FEATS}/generator_eval_{cell}"
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        rawp = f"{CK}/ckpt_{cell}_lingshu7b_sc8.jsonl"
        if not (os.path.exists(stem + ".npz") and os.path.exists(gjp)):
            print(f"  [skip] {cell}", flush=True)
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        meta = json.load(open(stem + ".meta.json"))
        keep = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
        rows = [meta["rows"][i] for i in keep]
        H = np.load(stem + ".npz")["h_span"][keep, LAYERS.index(21)].astype(np.float32)
        y = np.array([r["y"] for r in rows], dtype=int)
        na = [norm(r["na"]) for r in rows]
        L = sel.head_logits(H)

        # pool multiplicity, for the selfcons rule
        raw = {}
        if os.path.exists(rawp):
            for l in open(rawp):
                if l.strip():
                    d = json.loads(l); raw[d["idx"]] = d
        mult = np.zeros(len(rows), np.float32)
        for i, r in enumerate(rows):
            d = raw.get(r["idx"])
            if d:
                mult[i] = Counter(norm(p) for p in d["preds"]).get(na[i], 0)

        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        q_img = {r["idx"]: r["img_md5"] for r in rows}

        res = {r: [] for r in RULES}
        best, worst, gr, clus = [], [], [], []
        nt = 0
        for q in sorted(byq):
            if q not in gok:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            top = hr.max()
            tied = np.flatnonzero(hr >= top - 1e-9)
            if len(tied) > 1:
                nt += 1
            yy = y[ii]
            res["first_row"].append(int(yy[tied[0]]))
            res["lexical"].append(int(yy[tied[np.argmin([na[ii[t]] for t in tied])]]))
            # selfcons: max multiplicity, lexical inner tie-break
            sc_t = [(-mult[ii[t]], na[ii[t]], t) for t in tied]
            res["selfcons"].append(int(yy[sorted(sc_t)[0][2]]))
            lg_t = [(-len(na[ii[t]]), na[ii[t]], t) for t in tied]
            res["longest"].append(int(yy[sorted(lg_t)[0][2]]))
            best.append(int(yy[tied].max())); worst.append(int(yy[tied].min()))
            gr.append(gok[q]); clus.append(q_img[q])
        if len(gr) < 50:
            continue
        gr = np.array(gr)
        rec = {"n_questions": len(gr), "n_questions_with_a_top1_TIE": nt,
               "frac_tied": nt / len(gr), "greedy": float(gr.mean())}
        for r in RULES:
            a = np.array(res[r]); rec[r] = float(a.mean()); agg[r].append(float(a.mean()))
        rec["tie_ambiguity_band"] = [float(np.mean(worst)), float(np.mean(best))]
        rec["band_width"] = float(np.mean(best) - np.mean(worst))
        # CI on the best rule vs first_row, paired, clustered on image
        bestrule = max(RULES, key=lambda r: rec[r])
        rec["best_rule"] = bestrule
        rec["ci_best_rule_minus_first_row"] = list(
            boot(np.array(res[bestrule]), np.array(res["first_row"]), clus))
        agg["best_case"].append(float(np.mean(best))); agg["worst_case"].append(float(np.mean(worst)))
        agg["greedy"].append(float(gr.mean()))
        art["cells"][cell] = rec
        tie_tot += len(gr); tie_q += nt
        print(f"  {cell:17} n {len(gr):5d}  tied {nt:4d} ({nt/len(gr):5.1%})  "
              f"first {rec['first_row']:.4f}  lex {rec['lexical']:.4f}  "
              f"sc {rec['selfcons']:.4f}  long {rec['longest']:.4f}  "
              f"band {rec['band_width']:+.4f}", flush=True)

    art["macro"] = {r: float(np.mean(agg[r])) for r in RULES}
    art["macro"]["greedy"] = float(np.mean(agg["greedy"]))
    art["macro"]["tie_ambiguity_band"] = [float(np.mean(agg["worst_case"])),
                                          float(np.mean(agg["best_case"]))]
    art["macro"]["band_width"] = float(np.mean(agg["best_case"]) - np.mean(agg["worst_case"]))
    art["overall_frac_tied"] = tie_q / max(tie_tot, 1)
    mx = max(RULES, key=lambda r: art["macro"][r])
    art["VERDICT"] = {
        "fraction_of_questions_whose_pick_is_decided_by_a_TIE": art["overall_frac_tied"],
        "macro_spread_across_the_four_deterministic_rules":
            float(max(art["macro"][r] for r in RULES) - min(art["macro"][r] for r in RULES)),
        "upper_bound_any_tiebreak_rule_can_be_worth": art["macro"]["band_width"],
        "best_rule": mx,
        "best_rule_macro_vs_first_row": art["macro"][mx] - art["macro"]["first_row"],
        "reading": "first_row is not a rule, it is whichever cache was read. Any of the other three "
                   "is reproducible. Adopt one on determinism grounds even if the accuracy is a tie.",
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(art, f, indent=1)
    os.replace(tmp, OUT)
    print("\nMACRO:", {r: round(art["macro"][r], 4) for r in RULES})
    print("ambiguity band:", [round(x, 4) for x in art["macro"]["tie_ambiguity_band"]],
          f"width {art['macro']['band_width']:+.4f}")
    print(f"{art['overall_frac_tied']:.2%} of all questions are decided by a tie")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
