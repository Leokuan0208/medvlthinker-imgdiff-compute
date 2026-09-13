#!/usr/bin/env python3
"""head_lobo_pooled.py -- does pooled training generalise to an UNSEEN benchmark?

THE QUESTION THE POOLED RESULT LEAVES OPEN.  Fitting on the training half of all eight benchmarks
took macro verifier-minus-greedy from +0.0213 to +0.0778.  But every benchmark it was scored on had
contributed its own training half, so that number cannot distinguish two very different worlds:

  (a) the verifier learned something general from seeing eight domains, and would help on a NINTH
      benchmark it has never seen;
  (b) the verifier learned eight per-benchmark decision rules, and a ninth benchmark gets nothing.

They have opposite consequences.  Under (a) the artifact ships and works on new data.  Under (b) the
method requires a labelled training split for every benchmark it will ever run on, which is a very
different claim and a much weaker one.

LEAVE-ONE-BENCHMARK-OUT settles it.  For each benchmark: train on the four original domains plus the
training halves of the OTHER SEVEN, and score the held-out half of the excluded one -- so the
evaluation benchmark contributes nothing to training, while the training set is still large and
broad.  Three arms on identical questions:

    four_domain   the July recipe: four domains only            (the floor)
    lobo          four domains + seven other benchmarks         (breadth without this benchmark)
    pooled        four domains + all eight, including this one  (the ceiling, and what we shipped)

lobo - four_domain is what breadth buys on unseen data.  pooled - lobo is what a benchmark's OWN
training half buys, i.e. the price of a new benchmark measured from the other side.

Note this is a fairer test of breadth than head_domain_scaling, which held the row budget fixed at
20k to isolate the mechanism; here the budget grows, which is what a practitioner would actually do.

  python3 src/training_methods/head_lobo_pooled.py --threads 4 --seeds 5
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
# 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
# generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
# the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
# generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    z = np.load(f"{HS.FEATS}/{stem}.npz"); m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    return ({L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS},
            [m["rows"][i] for i in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_lobo_pooled_2026-08-25.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    Xo, rows_o = load("generator_train_finelayer", TRAIN_DOMAINS)
    blocks = {"__orig__": (Xo, rows_o)}
    ev = {}
    for cell in BENCH:
        stem = ("generator_eval_finelayer" if cell in SHARED
                else f"generator_eval_finelayer_{cell}")
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(f"{HS.FEATS}/{stem}.npz") and os.path.exists(gjp)):
            continue
        Xc, rr = load(stem, {cell} if cell in SHARED else None)
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        blocks[cell] = ({L: Xc[L][istr] for L in ENS},
                        [rr[i] for i in np.where(istr)[0]])
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = ({L: Xc[L][~istr] for L in ENS},
                    [rr[i] for i in np.where(~istr)[0]], gok)
        print(f"  {cell:17} train {int(istr.sum()):6,} / held-out {int((~istr).sum()):6,}",
              flush=True)

    evimgs = set()
    for _, rr, _ in ev.values():
        evimgs |= {r["img_md5"] for r in rr}

    def assemble(names):
        Xs = {L: [] for L in ENS}; rows, src = [], []
        for nm in names:
            Xb, rb = blocks[nm]
            m = np.array([r["img_md5"] not in evimgs for r in rb])
            for L in ENS:
                Xs[L].append(Xb[L][m])
            rows += [r for r, k in zip(rb, m) if k]
            src += [(r["ds"] if nm == "__orig__" else nm) for r, k in zip(rb, m) if k]
        return ({L: np.concatenate(Xs[L]) for L in ENS},
                np.array([r["y"] for r in rows], dtype=np.float32),
                np.array([f"{s}|{r['idx']}" for s, r in zip(src, rows)]))

    def fit_score(names, target):
        X, y, q = assemble(names)
        Xe, rr, gok = ev[target]
        ye = np.array([r["y"] for r in rr], dtype=int)
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = [x for x in byq if x in gok]
        S = []
        for L in ENS:
            mu, sg = X[L].mean(0), X[L].std(0) + 1e-6
            ms = [HS.fit((X[L] - mu) / sg, y, q, None, objective="bce", hidden=256, wd=1e-2,
                         epochs=30, seed=s) for s in range(A.seeds)]
            S.append(np.stack([HS.predict(mm, (Xe[L] - mu) / sg) for mm in ms]))
        acc = []
        for x in qs:
            ii = np.array(byq[x])
            hr = np.mean([rank_avg(s[k][ii]) for s in S for k in range(s.shape[0])], axis=0)
            acc.append(int(ye[ii][int(np.argmax(hr))]))
        return float(np.mean(acc)), float(np.mean([gok[x] for x in qs])), len(qs), int(len(y))

    art = {"title": "Does pooled training generalise to an unseen benchmark?",
           "date": "2026-08-25", "no_fabricated_numbers": True, "seeds": A.seeds, "cells": {}}
    for target in ev:
        others = [c for c in ev if c != target]
        r = {}
        for nm, names in (("four_domain", ["__orig__"]),
                          ("lobo", ["__orig__"] + others),
                          ("pooled", ["__orig__"] + list(ev))):
            # HEARTBEAT. This script used to print only once per TARGET, but each target costs
            # 3 arms x 3 layers x 5 seeds = 45 fits on up to 112k rows, so nothing reached the log
            # for well over an hour. On 2026-09-13 the supervisor's stall detector -- which reads a
            # silent log as a hung process -- killed it twice at 40 min, and the queue then passed
            # its `expect` check against a stale artifact from that morning and reported success.
            # A long job must say something while it works.
            print(f"    [{target}] fitting {nm} ...", flush=True)
            a, g, n, ntr = fit_score(names, target)
            r[nm] = a; r[nm + "_rows"] = ntr; r["greedy"] = g; r["n_questions"] = n
            r[nm + "_minus_greedy"] = a - g
        r["breadth_gain"] = r["lobo"] - r["four_domain"]
        r["own_data_gain"] = r["pooled"] - r["lobo"]
        art["cells"][target] = r
        print(f"  {target:17} n{r['n_questions']:6} greedy {r['greedy']:.4f} | 4dom "
              f"{r['four_domain_minus_greedy']:+.4f} | LOBO {r['lobo_minus_greedy']:+.4f} "
              f"| pooled {r['pooled_minus_greedy']:+.4f} || breadth {r['breadth_gain']:+.4f} "
              f"own-data {r['own_data_gain']:+.4f}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mac = {k: float(np.mean([v[k + "_minus_greedy"] for v in cs.values()]))
               for k in ("four_domain", "lobo", "pooled")}
        art["macro"] = mac
        art["macro_breadth_gain"] = mac["lobo"] - mac["four_domain"]
        art["macro_own_data_gain"] = mac["pooled"] - mac["lobo"]
        art["VERDICT"] = (
            f"four-domain {mac['four_domain']:+.4f} -> leave-one-benchmark-out {mac['lobo']:+.4f} "
            f"-> pooled {mac['pooled']:+.4f}. Breadth alone, on a benchmark the verifier has never "
            f"seen, is worth {art['macro_breadth_gain']:+.4f}; that benchmark's own training half "
            f"adds a further {art['macro_own_data_gain']:+.4f}. " +
            ("Pooling GENERALISES -- most of the gain survives on unseen benchmarks, so the "
             "artifact is worth shipping as-is."
             if art["macro_breadth_gain"] > art["macro_own_data_gain"] else
             "Pooling is mostly PER-BENCHMARK -- the gain comes from a benchmark's own training "
             "half, so the method needs labelled data for each benchmark it will run on."))
        for k, v in mac.items():
            print(f"  MACRO {k:14} {v:+.4f}")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
