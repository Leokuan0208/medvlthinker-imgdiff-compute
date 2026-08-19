#!/usr/bin/env python3
"""head_donor_hetero.py -- the donor curve again, on a HETEROGENEOUS domain, because the first one
was run on a dataset whose structure inflates it.

WHAT THE FIRST DONOR CURVE SAID (head_domain_curve_2026-08-18.json).  Adding RadImageNet donor
questions to the head's training pool took selection efficiency on a held-out RadImageNet half from
0.6573 to 0.8996 -- 14.6% of above-floor headroom to 75.0% -- and 100 donor questions alone were
worth +0.151.

WHY THAT NUMBER IS AN UPPER BOUND.  RadImageNet is unusually templated and closed-vocabulary:

    dataset        distinct questions   distinct golds   top-10 golds cover
    radimagenet                    15              105               66.5%
    slake_open                    169              128               41.9%
    pathvqa_open                  312              504               41.3%

Fifteen templates over 2,000 questions.  A head shown 100 of them has seen essentially every
template and much of the 105-answer vocabulary, so part of the jump is LEARNING THE LABEL SET
rather than learning to verify answers in a new modality.  Reporting +0.151-per-100-questions as
the value of a new domain would overstate what a heterogeneous dataset will actually buy.

THIS SCRIPT re-runs the identical design on PathVQA, the most heterogeneous pool we have (312
distinct questions, 504 distinct golds), holding out half its images and feeding the other half in
at 0/10/25/50/100%.  The two curves bracket the truth: RadImageNet is the optimistic end, PathVQA
the realistic one.  Zero GPU -- every feature already exists in the frozen train cache.

  python3 src/training_methods/head_donor_hetero.py --domain pathvqa_open_train --threads 8
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS

FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
CONFIGS = {"deployed_bt_h256": dict(HS.BASE),
           "sweep_bce_h256": {**HS.BASE, "objective": "bce", "hidden": 256}}


def pick_stats(scores, y, qid, mask):
    idx = np.where(mask)[0]
    byq = defaultdict(list)
    for i in idx:
        byq[qid[i]].append(i)
    got, rec = [], []
    for q, ii in byq.items():
        ii = np.array(ii)
        got.append(int(y[ii][int(np.argmax(scores[ii]))]))
        rec.append(int(y[ii].max()))
    got, rec = np.array(got), np.array(rec)
    return {"n_q": len(got), "acc": float(got.mean()),
            "sel_eff": float(got[rec == 1].mean()), "oracle": float(rec.mean())}


def string_prior_scores(y, na, train_mask):
    """P(y=1 | normalised answer STRING) counted on the training rows -- no image, no head.

    MANDATORY 2026-08-19.  On RadImageNet this counter alone reproduced 72.6% of the head's entire
    donor gain, and 62.8% on PathVQA, because both have small closed answer vocabularies.  A donor
    curve reported without it measures label-vocabulary memorisation as if it were verification.
    """
    pos, tot = defaultdict(int), defaultdict(int)
    idx = np.where(train_mask)[0]
    for i in idx:
        tot[na[i]] += 1
        pos[na[i]] += int(y[i])
    gp = float(y[idx].mean()) if len(idx) else 0.5
    return np.array([(pos[a] + gp * 2) / (tot[a] + 2) if tot[a] else gp for a in na])


def random_floor(y, qid, mask):
    idx = np.where(mask)[0]
    byq = defaultdict(list)
    for i in idx:
        byq[qid[i]].append(i)
    num = den = 0
    for q, ii in byq.items():
        yy = y[np.array(ii)]
        if yy.max() == 1:
            num += yy.mean(); den += 1
    return float(num / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="pathvqa_open_train")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    if A.out is None:
        A.out = os.path.join(HS.OUTDIR, f"head_donor_hetero_{A.domain}_2026-08-18.json")

    H, y, qid, img, ds = HS.load_train()
    rows = []
    for sh in (0, 1):
        rows += json.load(open(os.path.join(HS.FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]
    rows = [r for r in rows if r.get("n_tok", -1) > 0]
    na = np.array([str(r["na"]).strip().lower().rstrip(".") for r in rows])
    assert len(na) == len(y)
    isD = ds == A.domain
    assert isD.sum() > 0, f"{A.domain} not in the pool: {sorted(set(ds))}"

    half = {h: int(hashlib.md5(("het" + str(h)).encode()).hexdigest(), 16) % 2
            for h in set(img[isD])}
    is_eval = np.array([isD[i] and half[img[i]] == 0 for i in range(len(y))])
    is_donor = np.array([isD[i] and half[img[i]] == 1 for i in range(len(y))])
    base = ~isD                                     # the other domains, always present
    floor = random_floor(y, qid, is_eval)
    print(f"{A.domain}: {isD.sum()} rows | eval half {is_eval.sum()} rows / "
          f"{len({img[i] for i in np.where(is_eval)[0]})} images | donor half {is_donor.sum()} rows",
          flush=True)
    print(f"  base pool (other domains) {base.sum()} rows | random-pick floor {floor:.4f}", flush=True)

    art = {"title": f"Donor curve on a HETEROGENEOUS domain ({A.domain})",
           "date": "2026-08-18", "no_fabricated_numbers": True,
           "why": "the RadImageNet donor curve is an upper bound: 15 question templates over 2,000 "
                  "questions and 105 gold answers, top-10 covering 66.5%. PathVQA has 312 distinct "
                  "questions and 504 golds, top-10 covering 41.3%, so its curve is the realistic end.",
           "eval_half": {"n_rows": int(is_eval.sum()),
                         "n_questions": int(len(set(qid[is_eval]))),
                         "random_pick_floor_sel_eff": floor},
           "results": {}}

    donor_imgs = sorted({img[i] for i in np.where(is_donor)[0]})
    rng = np.random.default_rng(0)
    order = list(donor_imgs); rng.shuffle(order)

    for cname, cfg in CONFIGS.items():
        X = HS.assemble(H, cfg)
        art["results"][cname] = {}
        for fr in FRACS:
            k = int(round(fr * len(order)))
            take = set(order[:k])
            use = base | np.array([is_donor[i] and img[i] in take for i in range(len(y))])
            runs = []
            for s in range(A.seeds):
                mu, sg = X[use].mean(0), X[use].std(0) + 1e-6
                m = HS.fit((X[use] - mu) / sg, y[use], qid[use], None,
                           objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                           lr=cfg.get("lr", 1e-3), epochs=cfg["epochs"], seed=s)
                sv = np.empty(len(X), dtype=np.float32)
                for b0 in range(0, len(X), 4096):
                    b1 = min(b0 + 4096, len(X))
                    sv[b0:b1] = HS.predict(m, (X[b0:b1] - mu) / sg)
                runs.append(pick_stats(sv, y, qid, is_eval))
            se = [r["sel_eff"] for r in runs]; ac = [r["acc"] for r in runs]
            nq = int(len({qid[i] for i in np.where(is_donor)[0] if img[i] in take})) if k else 0
            sp = pick_stats(string_prior_scores(y, na, use), y, qid, is_eval)
            art["results"][cname][f"{fr:.2f}"] = {
                "donor_images": k, "donor_questions": nq,
                "string_prior_sel_eff": sp["sel_eff"],
                "head_minus_string_prior": float(np.mean(se) - sp["sel_eff"]),
                "donor_rows": int(sum(1 for i in range(len(y)) if is_donor[i] and img[i] in take)),
                "sel_eff_mean": float(np.mean(se)), "sel_eff_sd": float(np.std(se)),
                "acc_mean": float(np.mean(ac)),
                "headroom_above_floor": float((np.mean(se) - floor) / (1 - floor))}
            r = art["results"][cname][f"{fr:.2f}"]
            print(f"  [{cname}] frac {fr:4.2f}  {r['donor_questions']:6} donor q -> "
                  f"sel_eff {r['sel_eff_mean']:.4f} (sd {r['sel_eff_sd']:.4f})  "
                  f"acc {r['acc_mean']:.4f}  hdrm {r['headroom_above_floor']:.1%}  "
                  f"strprior {r['string_prior_sel_eff']:.4f}  "
                  f"head-prior {r['head_minus_string_prior']:+.4f}", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
