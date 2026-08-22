#!/usr/bin/env python3
"""answer_kind_analysis.py -- does the KIND of answer decide whether selection helps?

THE QUESTION.  Measured 2026-08-22 over eight cells, the frozen head beats greedy on four
(pathvqa +0.047, slake +0.039, kvasir_x1 +0.078, gemex +0.015), ties on two (vqa_rad, radimagenet)
and loses on two (omnimed -0.049, vqamed -0.026).  "Per-domain" does not explain that split, and
neither does distance from the training distribution: GEMeX has the LARGEST kNN distance (7.228)
and the LOWEST domain-classifier confidence (0.693) of any cell and the head still wins on it.

The remaining pattern visible by eye is the kind of answer.  The two losing cells ask for short
category labels -- omnimed's kept types are Modality Recognition and Anatomy Identification,
vqamed C4 is single-term abnormality naming -- while GEMeX and PathVQA are descriptive free text.

WHY THIS SCRIPT DOES NOT JUST CORRELATE EIGHT CELLS.  That is exactly the analysis that already
failed here: a cell-level Spearman over seven cells put knn_train at -0.821 and domclf_maxprob at
+0.857, and ONE additional cell moved them to -0.643 and +0.667 and flipped the verdict.  Eight
points and a free choice of feature will find something whether or not anything is there.  So the
primary test is WITHIN cells, where there are thousands of questions and the cell's own identity is
held fixed:

    for each cell, split its questions at the median mean-candidate-length and compute
    (head - greedy) in the short half and the long half separately.

If answer kind drives the effect, the long half should favour the head in cell after cell, and the
sign test over eight cells is then the evidence -- not a correlation through eight scatter points.
The cell-level correlation is still reported, clearly labelled as the weak version.

  python3 src/cascade_methods/answer_kind_analysis.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/answer_kind_2026-08-22.json")
LAYERS = [7, 14, 21, 28]
from free_signal_bakeoff import CELLS, RAW, norm, load_cell


def boot_diff(a, b, nboot=4000, seed=7):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.empty(nboot)
    for i in range(nboot):
        ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
        d[i] = a[ia].mean() - b[ib].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    art = {"title": "Does the KIND of answer decide whether selection helps?",
           "date": "2026-08-22", "no_fabricated_numbers": True,
           "primary_test": "WITHIN-cell median split on mean candidate answer length; the "
                           "cell-level correlation over 8 points is reported but is the weak "
                           "version, and an 8-point ordering already collapsed once in this project",
           "cells": {}}

    for cell in CELLS:
        got = load_cell(cell)
        if got is None:
            continue
        H, rows = got
        stem = RAW.get(cell, cell)
        rawp = os.path.join(CK, f"ckpt_{stem}_lingshu7b_sc8.jsonl")
        jp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.judge.jsonl")
        if not (os.path.exists(rawp) and os.path.exists(jp)):
            continue
        y = np.array([r["y"] for r in rows], dtype=int)
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        raw, gj = {}, {}
        for l in open(rawp):
            if l.strip():
                d = json.loads(l); raw[d["idx"]] = d
        for l in open(jp):
            if l.strip():
                d = json.loads(l); gj[d["idx"]] = d
        L = sel.head_logits(H)

        hq, gq, ln, oneword = [], [], [], []
        for q in sorted(byq, key=lambda k: (len(str(k)), str(k))):
            if q not in gj or q not in raw:
                continue
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            hq.append(int(y[ii][int(np.argmax(hr))])); gq.append(int(gj[q]["judge_ok"]))
            preds = raw[q]["preds"]
            w = [len(str(p).split()) for p in preds]
            ln.append(float(np.mean(w)))
            oneword.append(float(np.mean([x <= 1 for x in w])))
        hq, gq, ln, oneword = map(np.array, (hq, gq, ln, oneword))
        if len(hq) < 100:
            continue
        med = np.median(ln)
        short, long = ln <= med, ln > med
        if short.sum() < 40 or long.sum() < 40:
            art["cells"][cell] = {"skipped": "answer length has no spread in this cell",
                                  "mean_candidate_words": float(ln.mean())}
            print(f"  {cell:17} SKIPPED -- no length spread (mean {ln.mean():.2f} words)", flush=True)
            continue
        ds, dlo, dhi = boot_diff(hq[long] - gq[long], hq[short] - gq[short])
        art["cells"][cell] = {
            "n_questions": int(len(hq)),
            "mean_candidate_words": float(ln.mean()),
            "frac_single_word_candidates": float(oneword.mean()),
            "head_minus_greedy_overall": float(hq.mean() - gq.mean()),
            "head_minus_greedy_SHORT_half": float((hq[short] - gq[short]).mean()),
            "head_minus_greedy_LONG_half": float((hq[long] - gq[long]).mean()),
            "long_minus_short": ds, "long_minus_short_ci": [dlo, dhi]}
        c = art["cells"][cell]
        print(f"  {cell:17} words {ln.mean():5.2f}  1-word {oneword.mean():.2f}  "
              f"h-g overall {c['head_minus_greedy_overall']:+.4f}  "
              f"short {c['head_minus_greedy_SHORT_half']:+.4f}  "
              f"long {c['head_minus_greedy_LONG_half']:+.4f}  "
              f"long-short {ds:+.4f} [{dlo:+.4f},{dhi:+.4f}]", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    ok = {k: v for k, v in art["cells"].items() if "long_minus_short" in v}
    lms = np.array([v["long_minus_short"] for v in ok.values()])
    npos = int((lms > 0).sum())
    art["within_cell_sign_test"] = {
        "n_cells": len(lms), "cells_where_long_favours_head": npos,
        "mean_long_minus_short": float(lms.mean())}
    # the weak version, kept only so nobody re-derives it and thinks it is new
    w = np.array([v["mean_candidate_words"] for v in ok.values()])
    hg = np.array([v["head_minus_greedy_overall"] for v in ok.values()])
    rw, rh = np.argsort(np.argsort(w)), np.argsort(np.argsort(hg))
    art["WEAK_cell_level_spearman_words_vs_head_minus_greedy"] = float(np.corrcoef(rw, rh)[0, 1])
    art["VERDICT"] = (
        f"within-cell: longer answers favour the head in {npos}/{len(lms)} cells, mean "
        f"{lms.mean():+.4f}. " +
        ("Answer KIND is a real within-cell driver and survives holding the cell fixed."
         if npos >= len(lms) - 1 and lms.mean() > 0.005 else
         "Answer kind does NOT explain the split: it fails the within-cell sign test, so the "
         "cell-level appearance is not reproduced where the cell is held fixed."))
    print(f"\n  within-cell sign test: long favours head in {npos}/{len(lms)} cells "
          f"(mean {lms.mean():+.4f})")
    print(f"  WEAK cell-level spearman (words vs head-greedy): "
          f"{art['WEAK_cell_level_spearman_words_vs_head_minus_greedy']:+.3f}")
    print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
