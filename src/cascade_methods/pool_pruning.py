#!/usr/bin/env python3
"""pool_pruning.py -- if a BIGGER pool hurts selection, does pruning before ranking help?

WHERE THIS COMES FROM.  mixed_temperature_2026-08-22.json united the T=0.2/0.4/0.7/1.0 pools, which
strictly raises coverage, and the head got WORSE on 6 of 7 cells against its own best single
temperature (vqamed 0.0543 vs 0.0958, vqa_rad 0.4500 vs 0.4950, omnimed 0.4981 vs 0.5094).  Adding
candidates the head cannot rank costs more than the correct answers among them are worth.

The converse is then worth testing directly: REMOVE candidates before ranking.  If selection
degrades with pool size, a pruned pool should beat the full one -- and pruning is free at inference,
because every criterion here is computable without labels.

PRUNERS (each keeps k of the pool, then the frozen head ranks what survives):
  sc          keep the k most-sampled answers -- self-consistency as a filter rather than a scorer,
              which is the role it may be better suited to given it LOSES as a scorer on 6/8 cells
  prior       keep the k with the highest training-set answer-string prior
  head        keep the head's own top-k, then re-rank -- a no-op by construction for argmax, so it
              is the CONTROL that shows the harness is measuring what it claims
  random      keep k at random -- the null that says whether any skill is involved in the pruning
  length      keep the k shortest answers (verbosity is a known confound in this project)

Reported against the unpruned pool at the same k sweep, per cell, with greedy alongside.  A pruner
that only matches the full pool is worthless; the bar is beating it.

  python3 src/cascade_methods/pool_pruning.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/pool_pruning_2026-08-24.json")
LAYERS = [7, 14, 21, 28]
CELLS = ["vqa_rad_open", "slake_open", "pathvqa_open", "radimagenet_open",
         "vqamed_open", "gemex_open", "omnimed_open", "kvasir_x1_open"]
KEEPS = [2, 3, 4, 6]
from free_signal_bakeoff import norm


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()
    pos, tot, ntr = defaultdict(int), defaultdict(int), 0
    for sh in (0, 1):
        for r in json.load(open(f"{FEATS}/generator_train_s{sh}of2.meta.json"))["rows"]:
            if r.get("n_tok", -1) > 0:
                a = norm(r["na"]); tot[a] += 1; pos[a] += int(r["y"]); ntr += 1
    gp = sum(pos.values()) / max(ntr, 1)
    rng = np.random.default_rng(0)

    art = {"title": "Pruning the candidate pool before the head ranks it", "date": "2026-08-24",
           "no_fabricated_numbers": True, "keeps": KEEPS, "cells": {}}
    for cell in CELLS:
        stem = f"{FEATS}/generator_eval_{cell}"
        jp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        rawp = f"{CK}/ckpt_{cell}_lingshu7b_sc8.jsonl"
        if not all(os.path.exists(x) for x in (stem + ".npz", jp, rawp)):
            continue
        meta = json.load(open(stem + ".meta.json"))
        keep_i = [i for i, r in enumerate(meta["rows"]) if r.get("n_tok", -1) > 0]
        rows = [meta["rows"][i] for i in keep_i]
        H = np.load(stem + ".npz")["h_span"][keep_i, LAYERS.index(21)].astype(np.float32)
        y = np.array([r["y"] for r in rows], dtype=int)
        na = np.array([norm(r["na"]) for r in rows])
        byq = defaultdict(list)
        for i, r in enumerate(rows):
            byq[r["idx"]].append(i)
        gok, raw = {}, {}
        for l in open(jp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        for l in open(rawp):
            if l.strip():
                d = json.loads(l); raw[d["idx"]] = d
        L = sel.head_logits(H)
        sp = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])
        sc = np.zeros(len(rows), np.float32)
        wl = np.zeros(len(rows), np.float32)
        for i, r in enumerate(rows):
            d = raw.get(r["idx"])
            if d:
                c = Counter(norm(p) for p in d["preds"])
                sc[i] = c.get(na[i], 0) / max(len(d["preds"]), 1)
            wl[i] = len(str(r.get("ans", na[i])).split())
        qs = [q for q in byq if q in gok]
        res = {p: {k: [] for k in KEEPS} for p in ("sc", "prior", "head", "random", "length")}
        full, gr = [], []
        for q in qs:
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(L[j][ii]) for j in range(L.shape[0])], axis=0)
            full.append(int(y[ii][int(np.argmax(hr))])); gr.append(gok[q])
            for k in KEEPS:
                kk = min(k, len(ii))
                order = {"sc": np.argsort(-sc[ii]), "prior": np.argsort(-sp[ii]),
                         "head": np.argsort(-hr), "random": rng.permutation(len(ii)),
                         "length": np.argsort(wl[ii])}
                for p, o in order.items():
                    keep = ii[o[:kk]]
                    hk = np.mean([rank_avg(L[j][keep]) for j in range(L.shape[0])], axis=0)
                    res[p][k].append(int(y[keep][int(np.argmax(hk))]))
        full, gr = np.array(full), np.array(gr)
        art["cells"][cell] = {"n_questions": len(qs), "greedy": float(gr.mean()),
                              "head_full_pool": float(full.mean()),
                              "mean_pool_size": float(len(rows) / max(len(byq), 1)),
                              "pruned": {p: {str(k): float(np.mean(v)) for k, v in d.items()}
                                         for p, d in res.items()}}
        c = art["cells"][cell]
        best = max(((p, k, np.mean(v)) for p, d in res.items() for k, v in d.items()),
                   key=lambda t: t[2])
        c["best_pruner"] = {"pruner": best[0], "keep": best[1], "acc": float(best[2]),
                            "vs_full_pool": float(best[2] - full.mean())}
        print(f"  {cell:17} greedy {c['greedy']:.4f} full {c['head_full_pool']:.4f}  "
              f"best {best[0]}@{best[1]} {best[2]:.4f} ({best[2]-full.mean():+.4f})", flush=True)
        json.dump(art, open(OUT, "w"), indent=1)

    cs = art["cells"]
    if cs:
        # the honest read: a pruner+k chosen per cell is chosen on eval, so also report the single
        # (pruner, k) that is best on AVERAGE, which is what a deployment would have to fix in advance
        combos = {(p, k): np.mean([v["pruned"][p][str(k)] - v["head_full_pool"] for v in cs.values()])
                  for p in ("sc", "prior", "head", "random", "length") for k in KEEPS}
        bp, bv = max(combos.items(), key=lambda kv: kv[1])
        art["best_fixed_combo"] = {"pruner": bp[0], "keep": bp[1], "macro_gain_vs_full": float(bv)}
        art["per_cell_best_is_selected_on_eval"] = True
        art["VERDICT"] = (
            f"best FIXED pruner is {bp[0]} keeping {bp[1]}, worth {bv:+.4f} macro against the full "
            f"pool. " + ("Pruning helps and should be part of the method."
                         if bv > 0.005 else
                         "Pruning does NOT help once the rule is fixed in advance: the pool size is "
                         "not what limits selection here."))
        print(f"\n  best FIXED combo: {bp[0]} keep {bp[1]}  macro {bv:+.4f}")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
