#!/usr/bin/env python3
"""head_price_from_lobo.py -- the price of a new benchmark, measured from the deployable base.

WHY THE EXISTING PRICE CURVES ARE THE WRONG BASELINE NOW.  head_newdomain_curve.py added k labelled
questions on top of the FOUR July domains.  head_lobo_pooled_2026-08-25.json then showed that
breadth is worth nothing on an unseen benchmark -- four-domain +0.0312, leave-one-benchmark-out
+0.0304, pooled +0.0803 -- so the gain is entirely a benchmark's own training half (+0.0500), and
what a practitioner deploys is a verifier trained on every benchmark they already have.

The question that matters is therefore: starting from THAT base, how many labelled questions does a
NEW benchmark need?  This holds out one benchmark, trains on the other seven plus the four original
domains, adds k questions of the held-out one, and finds where it overtakes greedy decoding.

The answer is the operational cost of onboarding a dataset, and it is not the same number as the
old price curve: the base is stronger, but LOBO says the base does not transfer, so the curve may
start no higher and simply be the same climb from a different place. That is worth knowing either
way, and it is the difference between "the method needs data per benchmark" and "the method needs
THIS MUCH data per benchmark".

  python3 src/training_methods/head_price_from_lobo.py --threads 4 --seeds 3
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
SHARED = {"pathvqa_open", "slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]
KS = [0, 50, 100, 250, 500, 1000, 2000]


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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR,
                                                  "head_price_from_lobo_2026-08-30.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    Xo, rows_o = load("generator_train_finelayer", TRAIN_DOMAINS)
    blocks = {"__orig__": (Xo, rows_o, ["__orig__"] * len(rows_o))}
    ev = {}
    for cell in BENCH:
        stem = ("generator_eval_finelayer" if cell in SHARED
                else f"generator_eval_finelayer_{cell}")
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(f"{HS.FEATS}/{stem}.npz") and os.path.exists(gjp)):
            continue
        Xc, rr = load(stem, {cell} if cell in SHARED else None)
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        ti = np.where(istr)[0]
        blocks[cell] = ({L: Xc[L][istr] for L in ENS}, [rr[i] for i in ti], [cell] * len(ti))
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = ({L: Xc[L][~istr] for L in ENS},
                    [rr[i] for i in np.where(~istr)[0]], gok)
    evimgs = set()
    for _, rr, _ in ev.values():
        evimgs |= {r["img_md5"] for r in rr}

    art = {"title": "Price of a new benchmark, starting from a verifier trained on the others",
           "date": "2026-08-30", "no_fabricated_numbers": True, "ks": KS, "seeds": A.seeds,
           "base": "four original domains + the training halves of the seven OTHER benchmarks",
           "cells": {}}
    for target in ev:
        others = [c for c in ev if c != target]
        # base rows, minus anything whose image is in any held-out half
        Xs = {L: [] for L in ENS}; rows = []; src = []
        for nm in ["__orig__"] + others:
            Xb, rb, sb = blocks[nm]
            keep = np.array([r["img_md5"] not in evimgs for r in rb])
            for L in ENS:
                Xs[L].append(Xb[L][keep])
            rows += [r for r, k in zip(rb, keep) if k]
            src += [s for s, k in zip(sb, keep) if k]
        Xt, rt, st = blocks[target]
        qs_t = sorted({r["idx"] for r in rt})
        rng = np.random.default_rng(0); rng.shuffle(qs_t)
        Xe, rr_e, gok = ev[target]
        ye = np.array([r["y"] for r in rr_e], dtype=int)
        byq = defaultdict(list)
        for i, r in enumerate(rr_e):
            byq[r["idx"]].append(i)
        qs_e = [q for q in byq if q in gok]
        greedy = float(np.mean([gok[q] for q in qs_e]))
        res = {"n_eval_questions": len(qs_e), "greedy": greedy,
               "donor_pool_questions": len(qs_t), "curve": {}}
        for k in sorted({min(k, len(qs_t)) for k in KS}):
            take = set(qs_t[:k])
            add = np.array([r["idx"] in take for r in rt])
            X = {L: np.concatenate(Xs[L] + [Xt[L][add]]) for L in ENS}
            rw = rows + [r for r, a in zip(rt, add) if a]
            sc = src + [target] * int(add.sum())
            y = np.array([r["y"] for r in rw], dtype=np.float32)
            q = np.array([f"{s}|{r['idx']}" for s, r in zip(sc, rw)])
            S = []
            for L in ENS:
                mu, sg = X[L].mean(0), X[L].std(0) + 1e-6
                ms = [HS.fit((X[L] - mu) / sg, y, q, None, objective="bce", hidden=256, wd=1e-2,
                             epochs=30, seed=s) for s in range(A.seeds)]
                S.append(np.stack([HS.predict(mm, (Xe[L] - mu) / sg) for mm in ms]))
            acc = []
            for x in qs_e:
                ii = np.array(byq[x])
                hr = np.mean([rank_avg(s[j][ii]) for s in S for j in range(s.shape[0])], axis=0)
                acc.append(int(ye[ii][int(np.argmax(hr))]))
            res["curve"][str(k)] = {"donor_questions": k, "verifier": float(np.mean(acc)),
                                    "minus_greedy": float(np.mean(acc) - greedy),
                                    "train_rows": int(len(y))}
            r = res["curve"][str(k)]
            print(f"  {target:17} k={k:5} verifier {r['verifier']:.4f} "
                  f"(v-g {r['minus_greedy']:+.4f})  {r['train_rows']:,} rows", flush=True)
        cross = [int(k) for k in res["curve"] if res["curve"][k]["minus_greedy"] > 0]
        res["crossover_k"] = min(cross) if cross else None
        art["cells"][target] = res
        print(f"  {target:17} crosses greedy at k={res['crossover_k']}\n", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        got = [v["crossover_k"] for v in cs.values() if v["crossover_k"] is not None]
        art["summary"] = {
            "benchmarks": len(cs), "crossed": len(got),
            "median_crossover_k": float(np.median(got)) if got else None,
            "zero_shot_macro": float(np.mean([v["curve"]["0"]["minus_greedy"]
                                              for v in cs.values()]))}
        art["VERDICT"] = (
            f"starting from a verifier trained on the other seven benchmarks, {len(got)}/{len(cs)} "
            f"cross greedy, median at k={art['summary']['median_crossover_k']} labelled questions; "
            f"at k=0 the macro is {art['summary']['zero_shot_macro']:+.4f}. This is the operational "
            f"cost of onboarding a new benchmark into a deployed verifier.")
        print(f"=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
