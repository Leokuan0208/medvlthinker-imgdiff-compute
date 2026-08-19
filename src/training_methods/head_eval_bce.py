#!/usr/bin/env python3
"""head_eval_bce.py -- THE ONE-SHOT EVAL MEASUREMENT of the CV-selected head, plus the two
diagnostics that separate "we need more data" from "we need a bigger head".

WHAT THE SWEEP ESTABLISHED (CV only, eval untouched, 74 configs):
    deployed  bt /h256   0.68047
    winner    bce/h1024  0.69742   (+0.01695)
  and the bce MEDIAN (0.69002) beats the bt BEST (0.68890) over 58 bt configs.

--stage eval   fits the selected configs on the FULL training pool at 8 seeds, applies the frozen
               readout (per-seed within-question rank_avg, mean over seeds, argmax, first-index
               tie-break), and scores the 2,345-question eval pool ONCE.  bt/h256 is refit in the
               same run at the same thread count as a MATCHED control, so the delta is not
               confounded by refit noise; the frozen artifact's published value is printed
               alongside for provenance only.

--stage curve  DATA x CAPACITY, with the TRAIN fit reported next to the CV fit.  This is the
               diagnostic the roadmap turns on:
                 train >> CV and the gap grows with capacity  -> OVERFITTING, more DATA helps
                 train ~= CV and both rise with capacity       -> UNDERFITTING, a BIGGER HEAD helps
                 CV flat in data at every capacity             -> neither; the ceiling is elsewhere
               Training questions are subsampled BY IMAGE so no question straddles a split.

  python3 src/training_methods/head_eval_bce.py --stage eval  --threads 24
  python3 src/training_methods/head_eval_bce.py --stage curve --threads 24
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import head_sweep as HS

EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
FROZEN_PUBLISHED = {"head_alone_sel_eff": 0.8010899182561307, "head_alone_acc": 0.5014925373134328}

ARMS = {
    "control_bt_h256_REFIT": dict(HS.BASE),
    "cv_winner_bce_h1024": {**HS.BASE, "objective": "bce", "hidden": 1024},
    "cv_runnerup_bce_h256": {**HS.BASE, "objective": "bce", "hidden": 256},
}


def load_train_big():
    """The BIG train cache: every judged candidate (60,384 rows), not the 31,498-row matched draw
    the head inherited from the LoRA verifier.  radimagenet is excluded -- it is an eval cell."""
    zs, rows = [], []
    for sh in (0, 1):
        z = np.load(os.path.join(HS.FEATS, f"generator_train_bigtrain_s{sh}of2.npz"))
        m = json.load(open(os.path.join(HS.FEATS, f"generator_train_bigtrain_s{sh}of2.meta.json")))
        zs.append({"h_last": z["h_last"], "h_span": z["h_span"]})
        rows += m["rows"]
    H = {k: np.concatenate([z[k] for z in zs], 0) for k in ("h_last", "h_span")}
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    H = {k: v[keep] for k, v in H.items()}
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    img = np.array([r["img_md5"] for r in rows])
    ds = np.array([r["ds"] for r in rows])
    return H, y, qid, img, ds


def load_eval():
    zs, rows = [], []
    for sh in (0, 1):
        z = np.load(os.path.join(HS.FEATS, f"generator_eval_s{sh}of2.npz"))
        m = json.load(open(os.path.join(HS.FEATS, f"generator_eval_s{sh}of2.meta.json")))
        zs.append({"h_last": z["h_last"], "h_span": z["h_span"]})
        rows += m["rows"]
    H = {k: np.concatenate([z[k] for z in zs], 0) for k in ("h_last", "h_span")}
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    H = {k: v[keep] for k, v in H.items()}
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    ds = np.array([r["ds"] for r in rows])
    img = np.array([r["img_md5"] for r in rows])
    return H, y, qid, ds, img


# THE canonical ranking convention -- imported, not reimplemented, because tie handling here is
# load-bearing (rank_avg gives the frozen fusion 0.810627, rank_argsort gives 0.798365).
from genframe_data import rank_avg


def ensemble_pick(L, qid):
    """L (n_seeds, n_rows) logits -> {question: picked row}, frozen readout."""
    byq = defaultdict(list)
    for i, q in enumerate(qid):
        byq[q].append(i)
    picks = {}
    for q, ii in byq.items():
        ii = np.array(ii)
        s = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
        picks[q] = int(ii[int(np.argmax(s))])
    return picks


def score(picks, y, qid, ds, imgs=None):
    byq_ds, byq_img = {}, {}
    for i, q in enumerate(qid):
        byq_ds.setdefault(q, ds[i])
        byq_img.setdefault(q, imgs[i] if imgs is not None else q)
    rec = {}
    byq = defaultdict(list)
    for i, q in enumerate(qid):
        byq[q].append(i)
    for q, ii in byq.items():
        rec[q] = int(y[np.array(ii)].max())
    qs = sorted(picks)
    got = np.array([int(y[picks[q]]) for q in qs])
    r = np.array([rec[q] for q in qs])
    d = np.array([byq_ds[q] for q in qs])
    cl = np.array([byq_img[q] for q in qs])
    out = {"n_q": len(qs), "acc": float(got.mean()),
           "sel_eff": float(got[r == 1].mean()), "oracle": float(r.mean()), "per_ds": {}}
    for c in EVAL_DS:
        m = d == c
        out["per_ds"][c] = {"n": int(m.sum()), "acc": float(got[m].mean()),
                            "sel_eff": float(got[m & (r == 1)].mean())}
    return out, got, r, d, cl


def paired_boot(a, b, rec=None, nboot=10000, seed=20260818, clusters=None):
    """Paired bootstrap. If `clusters` is given, resample CLUSTERS (images) and take all their
    questions -- questions about the same image are not independent.

    BUGFIX 2026-08-19: this resampled questions i.i.d., but the eval pool is 2,345 questions over
    528 images (4.44 per image, max 21). At an intra-cluster correlation of 0.2 the design effect is
    1.69 and every interval was ~30% too narrow. No claim flipped, because every delta was a TIE --
    but the first positive result would have been overstated.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    da = np.empty(nboot); de = np.empty(nboot)
    ri = np.where(rec == 1)[0] if rec is not None else None
    if clusters is not None:
        cl = np.asarray(clusters)
        groups = [np.where(cl == c)[0] for c in np.unique(cl)]
        gr_rec = ([np.array([i for i in g if rec[i] == 1]) for g in groups]
                  if ri is not None else None)
    for i in range(nboot):
        if clusters is None:
            s = rng.integers(0, n, n)
        else:
            pick = rng.integers(0, len(groups), len(groups))
            s = np.concatenate([groups[k] for k in pick])
        da[i] = a[s].mean() - b[s].mean()
        if ri is not None:
            if clusters is None:
                j = ri[rng.integers(0, len(ri), len(ri))]
            else:
                sel = [gr_rec[k] for k in pick if len(gr_rec[k])]
                j = np.concatenate(sel) if sel else ri
            de[i] = a[j].mean() - b[j].mean()
    o = {"d_acc": float(a.mean() - b.mean()),
         "d_acc_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))]}
    if ri is not None:
        o["d_sel_eff"] = float(a[ri].mean() - b[ri].mean())
        o["d_sel_eff_ci"] = [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))]
    o["verdict"] = ("WIN" if o["d_acc_ci"][0] > 0 else
                    "LOSS" if o["d_acc_ci"][1] < 0 else "TIE")
    return o


# ------------------------------------------------------------------ stage: eval
def stage_eval(A):
    Hb, yb, qb, ib, db = (load_train_big() if getattr(A, "bigtrain", False) else HS.load_train())
    He, ye, qe, de, ie = load_eval()
    print(f"train {len(yb)} rows / {len(set(qb))} q | eval {len(ye)} rows / {len(set(qe))} q",
          flush=True)
    art = {"title": "One-shot eval of the CV-selected head objective",
           "date": "2026-08-18", "no_fabricated_numbers": True,
           "selection": "configs chosen on 5-fold image-grouped CV inside the TRAIN pool only "
                        "(head_sweep round 1, 74 configs). Eval was read for the first time here.",
           "multiplicity": "THREE arms are scored on eval, not one: the CV winner, the CV "
                           "runner-up at a deployable width, and the deployed objective refit as a "
                           "matched control. Declared rather than hidden.",
           "readout": "8 seeds; per-seed within-question rank_avg (ties averaged); mean over seeds; "
                      "argmax with first-index tie-break -- the frozen deployed convention.",
           "frozen_artifact_published": FROZEN_PUBLISHED,
           "train_pool": ("bigtrain: every judged row" if getattr(A, "bigtrain", False)
                          else "the inherited 31,498-row matched draw"),
           "threads": A.threads, "seeds": A.seeds, "arms": {}}
    store = {}
    for name, cfg in ARMS.items():
        Xb = HS.assemble(Hb, cfg)
        Xe = HS.assemble(He, cfg)
        mu, sg = Xb.mean(0), Xb.std(0) + 1e-6
        Xbs, Xes = (Xb - mu) / sg, (Xe - mu) / sg
        L = []
        for s in range(A.seeds):
            m = HS.fit(Xbs, yb, qb, None, objective=cfg["objective"], hidden=cfg["hidden"],
                       depth=cfg.get("depth", 1), drop=cfg.get("drop", 0.0),
                       ln=cfg.get("ln", False), wd=cfg["wd"], lr=cfg.get("lr", 1e-3),
                       epochs=cfg["epochs"], seed=s)
            L.append(HS.predict(m, Xes))
            print(f"    [{name}] seed {s} fitted", flush=True)
        L = np.stack(L)
        picks = ensemble_pick(L, qe)
        st, got, rec, dsv, clv = score(picks, ye, qe, de, ie)
        store[name] = (got, rec, dsv, clv)
        art["arms"][name] = {**{k: v for k, v in st.items()}, "config":
                             {k: cfg[k] for k in ("objective", "hidden", "layers", "pools",
                                                  "epochs", "wd", "lr")}}
        print(f"  {name:26} sel_eff {st['sel_eff']:.4f}  acc {st['acc']:.4f}  "
              f"per_ds {[round(st['per_ds'][c]['sel_eff'],4) for c in EVAL_DS]}", flush=True)

    ctl = "control_bt_h256_REFIT"
    art["deltas_vs_refit_control"] = {}
    for name in ARMS:
        if name == ctl:
            continue
        a, rec, dsv, clv = store[name]
        b = store[ctl][0]
        bt = paired_boot(a, b, rec, clusters=clv)   # image-clustered
        bt["bootstrap"] = "image-clustered (528 images / 2,345 questions)"
        clean = all(art["arms"][name]["per_ds"][c]["sel_eff"] >=
                    art["arms"][ctl]["per_ds"][c]["sel_eff"] for c in EVAL_DS)
        bt["guardrail_clean_sel_eff"] = bool(clean)
        bt["per_ds_d_sel_eff"] = {c: art["arms"][name]["per_ds"][c]["sel_eff"] -
                                  art["arms"][ctl]["per_ds"][c]["sel_eff"] for c in EVAL_DS}
        art["deltas_vs_refit_control"][name] = bt
        print(f"\n  {name} vs refit control:")
        print(f"    d_acc     {bt['d_acc']:+.4f} [{bt['d_acc_ci'][0]:+.4f},{bt['d_acc_ci'][1]:+.4f}] {bt['verdict']}")
        print(f"    d_sel_eff {bt['d_sel_eff']:+.4f} [{bt['d_sel_eff_ci'][0]:+.4f},{bt['d_sel_eff_ci'][1]:+.4f}]")
        print(f"    guardrail clean: {clean}  per-cell {[(c, round(v,4)) for c,v in bt['per_ds_d_sel_eff'].items()]}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


# ------------------------------------------------------------------ stage: curve
def stage_curve(A):
    """DATA x CAPACITY with the train fit beside the CV fit."""
    H, y, qid, img, ds = HS.load_train()
    fold_of = {h: int(hashlib.md5(str(h).encode()).hexdigest(), 16) % HS.FOLDS for h in set(img)}
    fo = np.array([fold_of[h] for h in img])
    rng = np.random.default_rng(0)
    art = {"title": "Data x capacity: is the head data-limited or capacity-limited?",
           "date": "2026-08-18", "no_fabricated_numbers": True,
           "reads": "train >> CV, gap widening with capacity -> overfitting, MORE DATA helps. "
                    "train ~= CV, both rising with capacity -> underfitting, a BIGGER HEAD helps. "
                    "CV flat in data at every capacity -> the ceiling is neither.",
           "subsampling": "training images are drawn at random; a question never straddles a split. "
                          "The CV fold assignment is unchanged, so held-out rows are identical "
                          "across every cell of the grid.",
           "grid": {}}
    for hid in A.widths:
        cfg = {**HS.BASE, "objective": "bce", "hidden": hid}
        X = HS.assemble(H, cfg)
        for frac in A.fracs:
            key = f"h{hid}_frac{frac:.2f}"
            cvs, trs = [], []
            for f in range(HS.FOLDS):
                trmask = fo != f
                tr_imgs = sorted({img[i] for i in np.where(trmask)[0]})
                rng2 = np.random.default_rng(1000 * f + hid)
                rng2.shuffle(tr_imgs)
                take = set(tr_imgs[:max(1, int(round(frac * len(tr_imgs))))])
                sub = trmask & np.array([h in take for h in img])
                mu, sg = X[sub].mean(0), X[sub].std(0) + 1e-6
                m = HS.fit((X[sub] - mu) / sg, y[sub], qid[sub], None, objective="bce",
                           hidden=hid, wd=cfg["wd"], lr=cfg.get("lr", 1e-3),
                           epochs=cfg["epochs"], seed=f)
                # chunked, so the full standardised matrix is never materialised -- the
                # 451 MB temporary per fold x width x fraction was churning the allocator and
                # segfaulting the run (2026-08-18).
                sv_all = np.empty(len(X), dtype=np.float32)
                for b0 in range(0, len(X), 4096):
                    b1 = min(b0 + 4096, len(X))
                    sv_all[b0:b1] = HS.predict(m, (X[b0:b1] - mu) / sg)

                def se(mask):
                    byq = defaultdict(list)
                    for i in np.where(mask)[0]:
                        byq[qid[i]].append(i)
                    hit = tot = 0
                    for q, ii in byq.items():
                        ii = np.array(ii)
                        if y[ii].sum() == 0:
                            continue
                        hit += int(y[ii][int(np.argmax(sv_all[ii]))]); tot += 1
                    return hit / max(tot, 1)

                cvs.append(se(fo == f)); trs.append(se(sub))
            art["grid"][key] = {
                "hidden": hid, "frac": frac,
                "train_images": int(round(frac * len({h for h in set(img)}) * 0.8)),
                "cv_sel_eff": float(np.mean(cvs)), "cv_sd": float(np.std(cvs)),
                "train_sel_eff": float(np.mean(trs)), "train_sd": float(np.std(trs)),
                "gap": float(np.mean(trs) - np.mean(cvs))}
            r = art["grid"][key]
            print(f"  h{hid:<5} frac {frac:4.2f}  CV {r['cv_sel_eff']:.4f}  "
                  f"TRAIN {r['train_sel_eff']:.4f}  gap {r['gap']:+.4f}", flush=True)
            json.dump(art, open(A.out, "w"), indent=1)
    print(f"\nwrote {A.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["eval", "curve"], required=True)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[128, 256, 1024])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--out", default=None)
    ap.add_argument("--bigtrain", action="store_true",
                    help="fit on the FULL 60,384-row judged pool instead of the 31,498-row draw")
    A = ap.parse_args()
    if A.out is None:
        A.out = os.path.join(HS.OUTDIR, f"head_{A.stage}_bce_2026-08-18.json")
    HS.torch.set_num_threads(A.threads)
    (stage_eval if A.stage == "eval" else stage_curve)(A)


if __name__ == "__main__":
    main()
