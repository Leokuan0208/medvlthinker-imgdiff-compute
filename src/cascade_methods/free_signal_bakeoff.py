#!/usr/bin/env python3
"""free_signal_bakeoff.py -- the head against per-candidate signals that are FREE and TRAINING-FREE.

WHY THIS EXISTS.  The frozen head's entire input is `h_span` at layer 21: the mean layer-21 hidden
state over the candidate answer's tokens.  Measured 2026-08-19 on three new cells with the deployed
readout, that is not enough:

    kvasir_x1 (IN-domain)   head 0.3629 vs greedy 0.2849   +0.0781 WIN
    vqamed C4 (new domain)  head 0.0688 vs greedy 0.0947   -0.0259 LOSS
    omnimed   (new domain)  head 0.3396 vs greedy 0.3885   -0.0489 LOSS, sel_eff BELOW the floor

Seven architectures were then ranked on out-of-domain transfer (head_arch_transfer_2026-08-19.json)
and every one of them -- pool normalisation, PCA to 128 and 32, within-pool ranks, domain-adversarial
training -- transferred WORSE than the plain baseline.  Architecture is not the lever.

WHAT WAS NEVER TRIED.  The generator already emits, for free, per-candidate evidence that the head
never sees, because `extract_generator_hidden.py` DEDUPLICATES the candidate pool before storing
features.  Measured on the caches: within-pool string duplication in the feature rows is 0.000, while
the raw dumps (`ckpt_<cell>_lingshu7b_sc8.jsonl`, field `preds`) keep all 8 samples WITH their
multiplicity.  The sampling frequency of an answer -- self-consistency -- was computed by the
generator, written to disk, and then thrown away at feature-extraction time.

That signal has the exact property the head lacks.  It is a within-pool count, so it carries no
answer vocabulary and no domain identity, and it needs no training rows, so there is nothing for it
to overfit to pathology and endoscopy.  If it transfers where the head does not, the transfer wall is
a consequence of WHICH FEATURES were kept, not of the head's capacity, objective or architecture.

ARMS.  Each is a per-candidate score; the pick is the within-pool argmax under the deployed readout.
  greedy                  the model's own answer -- the bar every arm must clear
  random-pick floor       picking uniformly from the same pool
  oracle@8                the ceiling
  string prior            P(y=1 | answer string) counted on the HEAD'S OWN training rows.  Mandatory:
                          it reproduced 101% of the head's donor gain on RadImageNet (2026-08-19).
  frozen head             deployed readout: per-seed within-pool rank_avg, mean over 8 seeds
  self-consistency        count of this candidate's normalised string among the raw 8 samples
  head + SC               equal-weight fusion of the two within-pool rank_avg vectors.  Equal weight
                          is PRE-SPECIFIED so the headline is not selected on the eval cell; a weight
                          sweep is reported separately and labelled EXPLORATORY.

Bootstrap resamples IMAGES, not questions, because several questions share one image.

  python3 src/cascade_methods/free_signal_bakeoff.py --cells all
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUTDIR = os.path.join(ROOT, "results/cascade_methods/artifacts")
LAYERS = [7, 14, 21, 28]

# cell -> (feature stem, ds filter or None). The three original open cells share two shard caches.
CELLS = {
    # 2026-09-13: pathvqa now reads its OWN complete cache. The s*of2 shards hold only the
    # truncated 1,500 questions (run_openvqa.py:158 capped that run); generator_eval_pathvqa_open
    # was re-extracted at the same layers [7,14,21,28] and covers all 3,357. Measured bias of the
    # truncation on this benchmark: +0.0174 overstated (pathvqa_truncation_2026-09-13.json).
    "pathvqa_open":   (["generator_eval_pathvqa_open"], None),
    "slake_open":     (["generator_eval_s0of2", "generator_eval_s1of2"], "slake_open"),
    "vqa_rad_open":   (["generator_eval_s0of2", "generator_eval_s1of2"], "vqa_rad_open"),
    "radimagenet_open": (["generator_eval_radimagenet"], None),
    "kvasir_x1_open": (["generator_eval_kvasir_x1_open"], None),
    "omnimed_open":   (["generator_eval_omnimed_open"], None),
    "vqamed_open":    (["generator_eval_vqamed_open"], None),
    "gemex_open":     (["generator_eval_gemex_open"], None),
}
RAW = {"radimagenet_open": "radimagenet_open"}      # dumps whose stem differs from the cell name


def norm(s):
    return str(s).strip().lower().rstrip(".")


def boot(a, b, clusters, nboot=10000, seed=20260821):
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


def load_cell(cell):
    stems, dsf = CELLS[cell]
    H, rows = [], []
    for st in stems:
        mp = os.path.join(FEATS, st + ".meta.json")
        if not os.path.exists(mp):
            return None
        meta = json.load(open(mp))
        z = np.load(os.path.join(FEATS, st + ".npz"))
        keep = [i for i, r in enumerate(meta["rows"])
                if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") == dsf)]
        if not keep:
            continue
        H.append(z["h_span"][keep, LAYERS.index(21)].astype(np.float32))
        rows += [meta["rows"][i] for i in keep]
    if not rows:
        return None
    return np.concatenate(H), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=["all"])
    ap.add_argument("--out", default=os.path.join(OUTDIR, "free_signal_bakeoff_2026-08-21.json"))
    A = ap.parse_args()
    cells = list(CELLS) if A.cells == ["all"] else A.cells

    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    # the string prior, counted once on the head's own training rows
    pos, tot, ntr = defaultdict(int), defaultdict(int), 0
    for sh in (0, 1):
        for r in json.load(open(os.path.join(FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]:
            if r.get("n_tok", -1) > 0:
                a = norm(r["na"]); tot[a] += 1; pos[a] += int(r["y"]); ntr += 1
    gp = sum(pos.values()) / max(ntr, 1)

    art = {"title": "The head vs FREE per-candidate signals the feature cache discarded",
           "date": "2026-08-21", "no_fabricated_numbers": True,
           "endpoint": "judge-currency accuracy of the within-pool argmax, per cell",
           "fusion_weight": "equal, PRE-SPECIFIED (not tuned on any eval cell)",
           "cells": {}}

    for cell in cells:
        got = load_cell(cell)
        if got is None:
            print(f"  [skip] {cell}: no feature cache", flush=True)
            continue
        H, rows = got
        y = np.array([r["y"] for r in rows], dtype=int)
        na = np.array([norm(r["na"]) for r in rows])
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        qids = sorted(byq, key=lambda k: (len(str(k)), str(k)))
        q_img = {r["idx"]: r["img_md5"] for r in rows}

        # --- self-consistency, recovered from the raw pre-dedup dump ---------------------
        stem = RAW.get(cell, cell)
        rawp = os.path.join(CK, f"ckpt_{stem}_lingshu7b_sc8.jsonl")
        if not os.path.exists(rawp):
            print(f"  [skip] {cell}: no raw sc8 dump at {rawp}", flush=True)
            continue
        raw = {}
        with open(rawp) as fh:
            for l in fh:
                if l.strip():
                    d = json.loads(l); raw[d["idx"]] = d
        sc = np.zeros(len(rows), dtype=np.float32)
        unmatched = 0
        for i, r in enumerate(rows):
            d = raw.get(r["idx"])
            if d is None:
                unmatched += 1; continue
            c = Counter(norm(p) for p in d["preds"])
            sc[i] = c.get(na[i], 0) / max(len(d["preds"]), 1)

        L = sel.head_logits(H)
        sp = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])

        head_ok, sc_ok, fus_ok, pri_ok, orc = [], [], [], [], []
        wsweep = {w: [] for w in (0.0, 0.25, 0.5, 0.75, 1.0)}
        for q in qids:
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
            sr = rank_avg(sc[ii])
            head_ok.append(int(y[ii][int(np.argmax(hr))]))
            sc_ok.append(int(y[ii][int(np.argmax(sr))]))
            fus_ok.append(int(y[ii][int(np.argmax(0.5 * hr + 0.5 * sr))]))
            pri_ok.append(int(y[ii][int(np.argmax(sp[ii]))]))
            orc.append(int(y[ii].max()))
            for w in wsweep:
                wsweep[w].append(int(y[ii][int(np.argmax((1 - w) * hr + w * sr))]))
        head_ok, sc_ok, fus_ok, pri_ok, orc = map(np.array,
                                                  (head_ok, sc_ok, fus_ok, pri_ok, orc))

        g = {}
        jp = os.path.join(CK, f"ckpt_{stem}_lingshu7b.judge.jsonl")
        if os.path.exists(jp):
            for l in open(jp):
                if l.strip():
                    d = json.loads(l); g[d["idx"]] = d
        greedy = np.array([int(g[q]["judge_ok"]) if q in g else -1 for q in qids])
        ok = greedy >= 0
        if ok.sum() == 0:
            print(f"  [skip] {cell}: no greedy judge file", flush=True)
            continue
        cl = np.array([q_img[q] for q in qids])

        fl_n = fl_d = 0
        for q in qids:
            yy = y[np.array(byq[q])]
            if yy.max() == 1:
                fl_n += yy.mean(); fl_d += 1
        floor = fl_n / max(fl_d, 1)
        rec = orc[ok] == 1

        art["cells"][cell] = {
            "n_questions": int(ok.sum()), "unmatched_rows": unmatched,
            "mean_self_consistency_of_pool": float(sc.mean()),
            "arms_judge": {
                "always_7b_greedy": float(greedy[ok].mean()),
                "string_prior": float(pri_ok[ok].mean()),
                "frozen_head": float(head_ok[ok].mean()),
                "self_consistency": float(sc_ok[ok].mean()),
                "head_plus_sc": float(fus_ok[ok].mean()),
                "oracle_at_8": float(orc[ok].mean()),
                "random_pick_floor_sel_eff": float(floor)},
            "sel_eff": {"frozen_head": float(head_ok[ok][rec].mean()),
                        "self_consistency": float(sc_ok[ok][rec].mean()),
                        "head_plus_sc": float(fus_ok[ok][rec].mean()),
                        "string_prior": float(pri_ok[ok][rec].mean())},
            "deltas": {
                "sc_vs_greedy": boot(sc_ok[ok], greedy[ok], cl[ok]),
                "sc_vs_head": boot(sc_ok[ok], head_ok[ok], cl[ok]),
                "fusion_vs_greedy": boot(fus_ok[ok], greedy[ok], cl[ok]),
                "fusion_vs_head": boot(fus_ok[ok], head_ok[ok], cl[ok]),
                "fusion_vs_sc": boot(fus_ok[ok], sc_ok[ok], cl[ok])},
            "EXPLORATORY_weight_sweep": {str(w): float(np.array(v)[ok].mean())
                                         for w, v in wsweep.items()}}
        a = art["cells"][cell]["arms_judge"]; d = art["cells"][cell]["deltas"]
        print(f"\n{cell}  ({int(ok.sum())} q)")
        print(f"  greedy {a['always_7b_greedy']:.4f} | prior {a['string_prior']:.4f} | "
              f"head {a['frozen_head']:.4f} | SC {a['self_consistency']:.4f} | "
              f"head+SC {a['head_plus_sc']:.4f} | oracle {a['oracle_at_8']:.4f}")
        for k in ("sc_vs_greedy", "sc_vs_head", "fusion_vs_greedy", "fusion_vs_head"):
            print(f"    {k:18} {d[k]['delta']:+.4f} [{d[k]['ci'][0]:+.4f},{d[k]['ci'][1]:+.4f}] "
                  f"{d[k]['verdict']}")
        json.dump(art, open(A.out, "w"), indent=1)

    json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
