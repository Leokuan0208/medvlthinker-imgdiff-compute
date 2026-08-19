#!/usr/bin/env python3
"""head_domain_curve.py -- HOW MUCH in-domain data does the head need to stop being useless?

THE QUESTION.  radimagenet_cell_2026-08-18.json showed the frozen head is a TIE against greedy on
RadImageNet (sel_eff 0.6396 vs a 0.5823 random floor) because it never trained on that domain.  The
obvious follow-up is not "does it transfer" -- we know it does not -- but "how little in-domain
data would fix it".  That decides whether every new eval cell must be paired with a big train
split or merely a small one, which is the difference between an afternoon and a week per dataset.

THE DESIGN, and it costs ZERO GPU because every feature already exists.  RadImageNet's 1,000 images
are split by md5 into a held-out EVAL half and a donor half.  The head is then fitted on the four
existing training datasets PLUS a growing fraction of the donor half's images, and always scored on
the same held-out eval half.  Fraction 0.0 reproduces the frozen head's out-of-domain condition;
the curve from there is the value of in-domain data, measured rather than assumed.

Image-grouped throughout: an image is wholly donor or wholly eval, never split, so no question can
leak across.

  python3 src/training_methods/head_domain_curve.py --threads 24
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS

FEATS = HS.FEATS
OUT = os.path.join(HS.OUTDIR, "head_domain_curve_2026-08-18.json")
FRACS = [0.0, 0.1, 0.25, 0.5, 1.0]
CONFIGS = {"deployed_bt_h256": dict(HS.BASE),
           "sweep_bce_h256": {**HS.BASE, "objective": "bce", "hidden": 256}}


def load_radimagenet():
    z = np.load(os.path.join(FEATS, "generator_eval_radimagenet.npz"))
    meta = json.load(open(os.path.join(FEATS, "generator_eval_radimagenet.meta.json")))
    rows = meta["rows"]
    H = {"h_span": z["h_span"], "h_last": z["h_last"]}
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"radimagenet|{r['idx']}" for r in rows])
    img = np.array([r["img_md5"] for r in rows])
    ds = np.array(["radimagenet_open"] * len(rows))
    return H, y, qid, img, ds


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
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    Hb, yb, qb, ib, db = HS.load_train()
    Hr, yr, qr, ir, dr = load_radimagenet()
    brows = []
    for sh in (0, 1):
        brows += json.load(open(os.path.join(HS.FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]
    brows = [r for r in brows if r.get("n_tok", -1) > 0]
    nab = np.array([str(r["na"]).strip().lower().rstrip(".") for r in brows])
    rmeta = json.load(open(os.path.join(FEATS, "generator_eval_radimagenet.meta.json")))["rows"]
    rmeta = [r for r in rmeta if r.get("n_tok", -1) > 0]
    nar = np.array([str(r["na"]).strip().lower().rstrip(".") for r in rmeta])
    assert len(nab) == len(yb) and len(nar) == len(yr)
    print(f"base pool {len(yb)} rows | radimagenet {len(yr)} rows / {len(set(ir))} images",
          flush=True)

    # image-grouped halving of radimagenet: even md5 -> EVAL (never trained on), odd -> donor
    half = {h: int(hashlib.md5(("rad" + str(h)).encode()).hexdigest(), 16) % 2 for h in set(ir)}
    is_eval = np.array([half[h] == 0 for h in ir])
    donor_imgs = sorted({h for h in set(ir) if half[h] == 1})
    print(f"  eval half {is_eval.sum()} rows / {len({h for h in set(ir) if half[h]==0})} images | "
          f"donor half {(~is_eval).sum()} rows / {len(donor_imgs)} images", flush=True)

    floor = random_floor(yr, qr, is_eval)
    art = {"title": "How much in-domain data does the head need? RadImageNet donor curve",
           "date": "2026-08-18", "no_fabricated_numbers": True,
           "design": "RadImageNet's images are halved by md5. The EVAL half is never trained on. "
                     "The head is fitted on the four existing training datasets plus a growing "
                     "fraction of the DONOR half's images, and always scored on the eval half. "
                     "Fraction 0.0 IS the frozen head's out-of-domain condition. Zero GPU: every "
                     "feature already existed.",
           "eval_half": {"n_rows": int(is_eval.sum()),
                         "n_questions": int(len(set(qr[is_eval]))),
                         "random_pick_floor_sel_eff": floor},
           "results": {}}
    print(f"  random-pick floor on the eval half: {floor:.4f}", flush=True)

    rng = np.random.default_rng(0)
    order = list(donor_imgs); rng.shuffle(order)

    for cname, cfg in CONFIGS.items():
        Xb = HS.assemble(Hb, cfg)
        Xr = HS.assemble(Hr, cfg)
        art["results"][cname] = {}
        for fr in FRACS:
            k = int(round(fr * len(order)))
            take = set(order[:k])
            use = (~is_eval) & np.array([h in take for h in ir])
            X = np.concatenate([Xb, Xr[use]], 0) if k else Xb
            Y = np.concatenate([yb, yr[use]], 0) if k else yb
            Q = np.concatenate([qb, qr[use]], 0) if k else qb
            runs = []
            for s in range(A.seeds):
                mu, sg = X.mean(0), X.std(0) + 1e-6
                m = HS.fit((X - mu) / sg, Y, Q, None, objective=cfg["objective"],
                           hidden=cfg["hidden"], depth=cfg.get("depth", 1),
                           drop=cfg.get("drop", 0.0), ln=cfg.get("ln", False), wd=cfg["wd"],
                           lr=cfg.get("lr", 1e-3), epochs=cfg["epochs"], seed=s)
                sc = HS.predict(m, (Xr - mu) / sg)
                runs.append(pick_stats(sc, yr, qr, is_eval))
            se = [r["sel_eff"] for r in runs]; ac = [r["acc"] for r in runs]
            # the control: counted on the SAME rows the head trained on, scored on the eval half
            na_all = np.concatenate([nab, nar]); y_all = np.concatenate([yb, yr])
            tr_mask = np.concatenate([np.ones(len(yb), bool), use])
            spv = string_prior_scores(y_all, na_all, tr_mask)[len(yb):]
            sp = pick_stats(spv, yr, qr, is_eval)
            art["results"][cname][f"{fr:.2f}"] = {
                "string_prior_sel_eff": sp["sel_eff"],
                "head_minus_string_prior": float(np.mean(se) - sp["sel_eff"]),
                "donor_images": k, "donor_rows": int(use.sum()),
                "donor_questions": int(len(set(qr[use]))) if k else 0,
                "sel_eff_mean": float(np.mean(se)), "sel_eff_sd": float(np.std(se)),
                "acc_mean": float(np.mean(ac)), "acc_sd": float(np.std(ac)),
                "headroom_above_floor": float((np.mean(se) - floor) / (1 - floor)),
            }
            r = art["results"][cname][f"{fr:.2f}"]
            print(f"  [{cname}] frac {fr:4.2f}  {r['donor_questions']:5} donor questions -> "
                  f"sel_eff {r['sel_eff_mean']:.4f} (sd {r['sel_eff_sd']:.4f})  "
                  f"acc {r['acc_mean']:.4f}  strprior {r['string_prior_sel_eff']:.4f}  "
                  f"head-prior {r['head_minus_string_prior']:+.4f}", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


if __name__ == "__main__":
    main()
