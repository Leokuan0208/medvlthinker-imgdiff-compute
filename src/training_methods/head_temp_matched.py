#!/usr/bin/env python3
"""head_temp_matched.py -- is the probe verifier hurt by a TRAIN/DEPLOY temperature mismatch?

THE SETUP.  The probe is fitted on candidate sets sampled at T=0.7 and then deployed over whatever
temperature the generator ran at.  head_temperature_2026-08-22.json shows the best deployment
temperature is 0.2 on three benchmarks and 0.4 on a fourth, so on half of them the probe scores a
candidate distribution it was never fitted to.  Every explanation of the cold-pool gain so far has
needed the eval labels to pick T per benchmark; this one does not.

THE DESIGN is a 2x2 over train temperature and eval temperature, so the mismatch term is separable
from the plain effect of a colder candidate set:

                     eval T=0.2      eval T=0.7
    train T=0.7      MISMATCHED      the deployed setting
    train T=0.2      MATCHED         mismatched the other way

If matching matters, the diagonal beats the off-diagonal.  If only the eval temperature matters, the
columns move together and the training temperature is irrelevant.

CONTAMINATION GUARD.  The T=0.2 training extraction pulled in 1,296 rows of **radimagenet_open**,
which is one of the eight EVALUATION benchmarks -- the T=0.7 pool contains only the four legitimate
training domains (kvasir_open, pathvqa_open_train, slake_open_train, vqa_rad_open_train).  Training
on those rows and then reporting radimagenet would be training on test.  They are dropped here, and
the script refuses to run if the two training pools do not end up with the same domain set.

  python3 src/training_methods/head_temp_matched.py --threads 4 --seeds 5
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D); sys.path.insert(0, os.path.join(os.path.dirname(D), "cascade_methods"))
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SUF = {0.2: "_T02", 0.7: ""}
LAY = 21


def norm(s):
    return str(s).strip().lower().rstrip(".")


def load_train(temp):
    """(X, y, qid) for the training pool at `temp`, restricted to the four training domains."""
    Xs, rows = [], []
    stems = ([f"generator_train_s{sh}of2" for sh in (0, 1)] if temp == 0.7
             else ["generator_train_T02"])
    for st in stems:
        p = f"{HS.FEATS}/{st}"
        z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
        layers = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and r["ds"] in TRAIN_DOMAINS]
        Xs.append(z["h_span"][keep, layers.index(LAY)].astype(np.float32))
        rows += [m["rows"][i] for i in keep]
    X = np.concatenate(Xs)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    return X, y, qid, sorted({r["ds"] for r in rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_temp_matched_2026-08-24.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    models, doms = {}, {}
    for T in (0.7, 0.2):
        X, y, qid, ds = load_train(T)
        doms[T] = ds
        mu, sg = X.mean(0), X.std(0) + 1e-6
        print(f"  train T={T}: {len(y)} rows over {ds}", flush=True)
        models[T] = (mu, sg, [HS.fit((X - mu) / sg, y, qid, None, objective="bce", hidden=256,
                                     wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)])
    if doms[0.7] != doms[0.2]:
        raise SystemExit(f"ABORT: training pools differ -- T0.7 {doms[0.7]} vs T0.2 {doms[0.2]}. "
                         "A domain in one and not the other makes the 2x2 uninterpretable.")

    art = {"title": "Train/deploy temperature mismatch for the probe verifier",
           "date": "2026-08-24", "no_fabricated_numbers": True, "layer": LAY,
           "train_domains": doms[0.7], "seeds": A.seeds,
           "excluded_from_training": "radimagenet_open (1,296 rows) appeared in the T=0.2 train "
                                     "extraction and is an EVALUATION benchmark",
           "cells": {}}
    for cell in BENCH:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        rec = {}
        for evalT, suf in SUF.items():
            p = f"{HS.FEATS}/generator_eval_{cell}{suf}"
            if not (os.path.exists(p + ".npz") and os.path.getsize(p + ".npz") > 5e6):
                continue
            z = np.load(p + ".npz"); m = json.load(open(p + ".meta.json"))
            layers = [int(x) for x in z["layers"]]
            if LAY not in layers:
                continue
            keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
            rr = [m["rows"][i] for i in keep]
            Xe = z["h_span"][keep, layers.index(LAY)].astype(np.float32)
            ye = np.array([r["y"] for r in rr], dtype=int)
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            qs = [q for q in byq if q in gok]
            if len(qs) < 50:
                continue
            rec[f"greedy@{evalT}"] = float(np.mean([gok[q] for q in qs]))
            for trainT, (mu, sg, ms) in models.items():
                S = np.stack([HS.predict(mm, (Xe - mu) / sg) for mm in ms])
                acc = []
                for q in qs:
                    ii = np.array(byq[q])
                    hr = np.mean([rank_avg(S[k][ii]) for k in range(S.shape[0])], axis=0)
                    acc.append(int(ye[ii][int(np.argmax(hr))]))
                rec[f"train{trainT}_eval{evalT}"] = float(np.mean(acc))
            rec[f"n@{evalT}"] = len(qs)
        if len(rec) < 4:
            continue
        art["cells"][cell] = rec
        g = rec.get("greedy@0.7")
        line = "  ".join(f"tr{t}/ev{e} {rec.get(f'train{t}_eval{e}', float('nan')):.4f}"
                         for e in (0.2, 0.7) for t in (0.2, 0.7))
        print(f"  {cell:17} greedy {g:.4f}  {line}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        def mac(k):
            v = [c[k] for c in cs.values() if k in c]
            return float(np.mean(v)) if v else float("nan")
        art["macro"] = {k: mac(k) for k in
                        ("greedy@0.2", "greedy@0.7", "train0.2_eval0.2", "train0.7_eval0.2",
                         "train0.2_eval0.7", "train0.7_eval0.7")}
        m = art["macro"]
        gain_cold = m["train0.2_eval0.2"] - m["train0.7_eval0.2"]
        gain_hot = m["train0.7_eval0.7"] - m["train0.2_eval0.7"]
        art["matching_gain_at_eval02"] = gain_cold
        art["matching_gain_at_eval07"] = gain_hot
        art["VERDICT"] = (
            f"matching the training temperature to the deployment temperature is worth "
            f"{gain_cold:+.4f} at eval T=0.2 and {gain_hot:+.4f} at eval T=0.7. " +
            ("Mismatch is real and the probe should be fitted at the temperature it will be "
             "deployed at." if min(gain_cold, gain_hot) > 0.005 else
             "Train/deploy temperature mismatch does NOT explain the cold-pool effect: what "
             "matters is the temperature of the candidate set, not the one the probe was fitted "
             "on."))
        for k, v in m.items():
            print(f"  MACRO {k:20} {v:.4f}")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
