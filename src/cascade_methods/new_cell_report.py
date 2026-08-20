#!/usr/bin/env python3
"""new_cell_report.py -- measure a newly built open-text cell, with the controls that the
2026-08-19 audit made mandatory.

EVERY number here carries its null.  The audit found that a counter over answer STRINGS -- no image,
no hidden state, no head -- reproduced 101% of the head's measured gain on RadImageNet and 69% on
PathVQA, because both have small closed answer vocabularies.  Four of the five new cells are
template-heavy, which is exactly that failure mode, so a cell report without the string prior beside
it is not interpretable.

Reported per cell:
  always-7B greedy         the baseline
  oracle@8                 the ceiling any selector could reach
  random-pick floor        what picking at random from the same pool gets
  string-prior selection   the null: P(y=1 | answer string) fitted on the HEAD'S OWN TRAINING ROWS
  frozen head selection    deployed readout (per-seed within-pool rank_avg, then mean over seeds)
  head - string prior      THE number that says whether the head did anything
Plus templating diagnostics (distinct questions / golds / top-10 coverage), because they predict
how much of any gain the string prior will eat.

  python3 src/cascade_methods/new_cell_report.py --cell vqamed_open
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
LAYERS = [7, 14, 21, 28]


def norm(s):
    return str(s).strip().lower().rstrip(".")


def loadj(p):
    return {json.loads(l)["idx"]: json.loads(l) for l in open(p) if l.strip()}


def boot(a, b, clusters, nboot=10000, seed=20260819):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    if A.out is None:
        A.out = os.path.join(ROOT, f"results/cascade_methods/artifacts/cell_{A.cell}_2026-08-19.json")

    stem = os.path.join(FEATS, f"generator_eval_{A.cell}")
    z = np.load(stem + ".npz")
    meta = json.load(open(stem + ".meta.json"))
    rows = [r for r in meta["rows"] if r.get("n_tok", -1) > 0]
    keep = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
    H = z["h_span"][keep, LAYERS.index(21)].astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=int)
    na = np.array([norm(r["na"]) for r in rows])
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[r["idx"]].append(i)
    qids = sorted(byq, key=lambda k: (len(str(k)), str(k)))
    q_img = {r["idx"]: r["img_md5"] for r in rows}

    # --- the frozen head, deployed readout -------------------------------------------------
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    L = sel.head_logits(H)

    # --- the string prior, fitted on the head's OWN training rows (never on this cell) ------
    pos, tot = defaultdict(int), defaultdict(int)
    ntr = 0
    for sh in (0, 1):
        for r in json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]:
            if r.get("n_tok", -1) <= 0:
                continue
            a = norm(r["na"]); tot[a] += 1; pos[a] += int(r["y"]); ntr += 1
    gp = sum(pos.values()) / max(ntr, 1)
    sp_all = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])

    head_ok, prior_ok, orc = [], [], []
    for q in qids:
        ii = np.array(byq[q])
        s = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
        head_ok.append(int(y[ii][int(np.argmax(s))]))
        prior_ok.append(int(y[ii][int(np.argmax(sp_all[ii]))]))
        orc.append(int(y[ii].max()))
    head_ok, prior_ok, orc = map(np.array, (head_ok, prior_ok, orc))

    fl_n = fl_d = 0
    for q in qids:
        yy = y[np.array(byq[q])]
        if yy.max() == 1:
            fl_n += yy.mean(); fl_d += 1
    floor = fl_n / max(fl_d, 1)

    g = loadj(os.path.join(CK, f"ckpt_{A.cell}_lingshu7b.judge.jsonl"))
    greedy = np.array([int(g[q]["judge_ok"]) if q in g else
                       (int(g[int(q)]["judge_ok"]) if str(q).lstrip("-").isdigit() and int(q) in g else -1)
                       for q in qids])
    ok = greedy >= 0
    cl = np.array([q_img[q] for q in qids])

    src = json.load(open({"vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
                          "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
                          "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json",
                          "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json"}[A.cell]))
    qc = Counter(r["question"] for r in src); gc = Counter(norm(r["answer"]) for r in src)

    art = {"title": f"New open-text cell: {A.cell}", "date": "2026-08-19",
           "no_fabricated_numbers": True,
           "n_questions": len(qids), "n_candidate_rows": len(rows),
           "mean_distinct_candidates": round(len(rows) / max(len(qids), 1), 3),
           "templating": {"distinct_questions": len(qc), "distinct_golds": len(gc),
                          "top10_gold_coverage": sum(n for _, n in gc.most_common(10)) / len(src),
                          "note": "radimagenet, whose donor gain the string prior explained 101%, "
                                  "has 15 distinct questions / 105 golds / 66.5% top-10"},
           "arms_judge": {"always_7b_greedy": float(greedy[ok].mean()),
                          "string_prior_selection": float(prior_ok[ok].mean()),
                          "frozen_head_selection": float(head_ok[ok].mean()),
                          "oracle_at_8": float(orc[ok].mean()),
                          "random_pick_floor_sel_eff": floor,
                          "head_sel_eff": float(head_ok[ok][orc[ok] == 1].mean()),
                          "string_prior_sel_eff": float(prior_ok[ok][orc[ok] == 1].mean())},
           "deltas": {"head_vs_greedy": boot(head_ok[ok], greedy[ok], cl[ok]),
                      "head_vs_string_prior": boot(head_ok[ok], prior_ok[ok], cl[ok]),
                      "string_prior_vs_greedy": boot(prior_ok[ok], greedy[ok], cl[ok])},
           "caveats": ["judge currency only",
                       "the string prior is fitted on the HEAD'S training rows, so it is the "
                       "matched null for a frozen head applied out of domain",
                       "bootstrap resamples IMAGES, not questions"]}
    json.dump(art, open(A.out, "w"), indent=1)
    a = art["arms_judge"]
    print(f"\n{A.cell}: {len(qids)} q, {art['mean_distinct_candidates']} distinct cand/q")
    print(f"  templating: {len(qc)} questions / {len(gc)} golds / "
          f"top-10 {art['templating']['top10_gold_coverage']:.1%}")
    print(f"  always-7B greedy   {a['always_7b_greedy']:.4f}")
    print(f"  string prior       {a['string_prior_selection']:.4f}   (sel_eff {a['string_prior_sel_eff']:.4f})")
    print(f"  frozen head        {a['frozen_head_selection']:.4f}   (sel_eff {a['head_sel_eff']:.4f})")
    print(f"  oracle@8           {a['oracle_at_8']:.4f}   floor {a['random_pick_floor_sel_eff']:.4f}")
    for k, v in art["deltas"].items():
        print(f"  {k:24} {v['delta']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}] {v['verdict']}")
    print(f"  wrote {A.out}")


if __name__ == "__main__":
    main()
