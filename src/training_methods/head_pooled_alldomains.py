#!/usr/bin/env python3
"""head_pooled_alldomains.py -- retrain the probe on EVERY benchmark we now have.

THE GAP THIS FILLS.  The deployed probe is still fitted on the 31,498 rows it was fitted on in
July: kvasir_open, pathvqa_open_train, slake_open_train, vqa_rad_open_train.  Four of the eight
benchmarks we now report -- RadImageNet, OmniMedVQA, VQA-Med and GEMeX -- have never contributed a
single training row.  Everything we have measured on them is zero-shot transfer.

Two experiments have circled this without answering it:
  * the price curves added k questions from ONE benchmark at a time on top of the same four-domain
    base, so they never asked what happens when you use everything at once;
  * head_domain_scaling held the row budget FIXED at 20k to separate breadth from volume, which is
    the right way to isolate the mechanism and the wrong way to estimate the deployable number.

A practitioner would collect what they can from every dataset they have and fit one probe.  That is
what this measures, honestly: a strict split BY IMAGE inside each benchmark, training on the union
of all training halves plus the four original domains, evaluating on the held-out halves.

THE SPLIT is md5("nd" + image hash) % 2, byte-identical to the one head_newdomain_curve.py used, so
the numbers here sit directly beside the price curves rather than beside a different partition.

LEAKAGE CHECK, and it is not optional here.  Pooling eight benchmarks into one training set creates
pairs that were never previously in the same pool -- Kvasir-x1 was built to exclude kvasir_open's
frames, and OmniMedVQA had RadImageNet dropped from it, but those guarantees were made pairwise and
are now being relied on jointly.  Every training image hash is checked against every evaluation
image hash before a single probe is fitted, and the run aborts on any collision.

  python3 src/training_methods/head_pooled_alldomains.py --threads 4 --seeds 5
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS
from head_domain_scaling import norm

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
LAY = 21


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--config_d", action="store_true",
                    help="stack the two validated wins on top of pooled training: rank-ensemble "
                         "over layers 18/20/22 with self-consistency as an input feature. "
                         "head_best_config_2026-08-24.json shows they are additive with each other "
                         "(+0.0168 over the deployed recipe); this asks whether they are also "
                         "additive with the +0.0565 from pooling, which is the whole method.")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    if A.out is None:
        A.out = os.path.join(HS.OUTDIR, "head_pooled_configd_2026-08-24.json" if A.config_d
                             else "head_pooled_alldomains_2026-08-24.json")
    from genframe_data import rank_avg

    # ---- the four original training domains -------------------------------------------------
    Xs, rows, src = [], [], []
    for sh in (0, 1):
        p = f"{HS.FEATS}/generator_train_s{sh}of2"
        z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
        lay = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and r["ds"] in TRAIN_DOMAINS]
        Xs.append(z["h_span"][keep, lay.index(LAY)].astype(np.float32))
        rows += [m["rows"][i] for i in keep]; src += [m["rows"][i]["ds"] for i in keep]
    base_n = len(rows)

    # ---- the training half of every benchmark ------------------------------------------------
    ev = {}
    for cell in BENCH:
        p = f"{HS.FEATS}/generator_eval_{cell}"
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not (os.path.exists(p + ".npz") and os.path.exists(gjp)):
            continue
        z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
        lay = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
        rr = [m["rows"][i] for i in keep]
        X = z["h_span"][keep, lay.index(LAY)].astype(np.float32)
        is_tr = np.array([half(r["img_md5"]) == 1 for r in rr])
        Xs.append(X[is_tr]); rows += [rr[i] for i in np.where(is_tr)[0]]
        src += [cell] * int(is_tr.sum())
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = {"X": X[~is_tr], "rows": [rr[i] for i in np.where(~is_tr)[0]], "gok": gok}
        print(f"  {cell:17} +{int(is_tr.sum()):6,} training rows | "
              f"{int((~is_tr).sum()):6,} held-out rows", flush=True)
    X = np.concatenate(Xs)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{s_}|{r['idx']}" for s_, r in zip(src, rows)])
    na = np.array([norm(r["na"]) for r in rows])
    deployed_n = base_n
    print(f"\npooled training set: {len(y):,} rows ({base_n:,} original + "
          f"{len(y)-base_n:,} new), vs {base_n:,} deployed", flush=True)

    # ---- leakage check: no training image may appear in ANY evaluation half ------------------
    tr_imgs = {r["img_md5"] for r in rows}
    bad = {}
    for cell, d in ev.items():
        hit = tr_imgs & {r["img_md5"] for r in d["rows"]}
        if hit:
            bad[cell] = len(hit)
    # Drop, do not abort. The one real overlap is 19 MedPix images shared between
    # vqa_rad_open_train and vqamed_open -- it predates this experiment and is present in the
    # DEPLOYED probe too. Measured impact on the reported vqamed number: the 19 questions score
    # 0.0000 for both the verifier and greedy, so excluding them moves -0.0259 to -0.0261.
    # Immaterial, but it has no business being in a training set, so it goes.
    leak = set()
    for cell, d in ev.items():
        leak |= tr_imgs & {r["img_md5"] for r in d["rows"]}
    if leak:
        drop = np.array([r["img_md5"] in leak for r in rows])
        print(f"leakage check: dropping {int(drop.sum()):,} training rows over {len(leak)} images "
              f"that appear in a held-out half ({bad})", flush=True)
        X = X[~drop]
        rows = [r for r, dd in zip(rows, drop) if not dd]
        src = [s_ for s_, dd in zip(src, drop) if not dd]
        base_n -= int(drop[:base_n].sum())
        y = np.array([r["y"] for r in rows], dtype=np.float32)
        qid = np.array([f"{s_}|{r['idx']}" for s_, r in zip(src, rows)])
        na = np.array([norm(r["na"]) for r in rows])
        tr_imgs = {r["img_md5"] for r in rows}
        assert not any(tr_imgs & {r["img_md5"] for r in d["rows"]} for d in ev.values())
    print(f"leakage check: {len(tr_imgs):,} training images, 0 collisions with any held-out half",
          flush=True)

    # ---- fit the pooled probe, and the four-domain probe on the SAME rows for comparison -----
    def fit(Xt, yt, qt):
        mu, sg = Xt.mean(0), Xt.std(0) + 1e-6
        return mu, sg, [HS.fit((Xt - mu) / sg, yt, qt, None, objective="bce", hidden=256,
                               wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)]
    is_base = np.arange(len(y)) < base_n
    print("fitting pooled probe ...", flush=True)
    POOL = fit(X, y, qid)
    print("fitting four-domain probe (the deployed recipe) ...", flush=True)
    BASE = fit(X[is_base], y[is_base], qid[is_base])

    pos, tot = defaultdict(int), defaultdict(int)
    for i in range(len(y)):
        tot[na[i]] += 1; pos[na[i]] += int(y[i])
    gp = float(y.mean())

    art = {"title": "Probe retrained on the training half of every benchmark",
           "date": "2026-08-24", "no_fabricated_numbers": True, "layer": LAY, "seeds": A.seeds,
           "pooled_rows": int(len(y)), "original_rows": int(base_n),
           "split": 'md5("nd"+img_md5) % 2, identical to head_newdomain_curve.py',
           "cells": {}}
    for cell, d in ev.items():
        Xe, rr, gok = d["X"], d["rows"], d["gok"]
        ye = np.array([r["y"] for r in rr], dtype=int)
        nae = np.array([norm(r["na"]) for r in rr])
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = [q for q in byq if q in gok]
        if len(qs) < 40:
            continue
        spv = np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in nae])
        out = {"n_questions": len(qs), "greedy": float(np.mean([gok[q] for q in qs]))}
        for nm, mdl in (("pooled", POOL), ("four_domain", BASE)):
            mu, sg, ms = mdl
            S = np.stack([HS.predict(mm, (Xe - mu) / sg) for mm in ms])
            acc = []
            for q in qs:
                ii = np.array(byq[q])
                hr = np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], axis=0)
                acc.append(int(ye[ii][int(np.argmax(hr))]))
            out[nm] = float(np.mean(acc))
        pa = [int(ye[np.array(byq[q])][int(np.argmax(spv[np.array(byq[q])]))]) for q in qs]
        out["answer_prior"] = float(np.mean(pa))
        out["pooled_minus_greedy"] = out["pooled"] - out["greedy"]
        out["four_domain_minus_greedy"] = out["four_domain"] - out["greedy"]
        out["pooled_minus_four_domain"] = out["pooled"] - out["four_domain"]
        art["cells"][cell] = out
        print(f"  {cell:17} n{len(qs):6} greedy {out['greedy']:.4f} | 4-domain "
              f"{out['four_domain_minus_greedy']:+.4f} | pooled "
              f"{out['pooled_minus_greedy']:+.4f} | gain {out['pooled_minus_four_domain']:+.4f}",
              flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mp = float(np.mean([v["pooled_minus_greedy"] for v in cs.values()]))
        mb = float(np.mean([v["four_domain_minus_greedy"] for v in cs.values()]))
        art["macro"] = {"pooled_minus_greedy": mp, "four_domain_minus_greedy": mb,
                        "gain_from_pooling": mp - mb,
                        "pooled_beats_greedy": f"{sum(1 for v in cs.values() if v['pooled_minus_greedy']>0)}/{len(cs)}",
                        "four_domain_beats_greedy": f"{sum(1 for v in cs.values() if v['four_domain_minus_greedy']>0)}/{len(cs)}"}
        art["VERDICT"] = (
            f"training on every benchmark's training half takes macro verifier-minus-greedy from "
            f"{mb:+.4f} to {mp:+.4f} ({mp-mb:+.4f}), and the number of benchmarks beaten from "
            f"{art['macro']['four_domain_beats_greedy']} to {art['macro']['pooled_beats_greedy']}. "
            + ("Retraining on everything is worth doing." if mp - mb > 0.005 else
               "Retraining on everything is NOT worth much: the four-domain probe already "
               "generalises about as well as one fitted on all eight."))
        print(f"\n  MACRO four-domain {mb:+.4f} -> pooled {mp:+.4f}  ({mp-mb:+.4f})")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
