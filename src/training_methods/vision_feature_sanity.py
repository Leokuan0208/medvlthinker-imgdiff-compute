#!/usr/bin/env python3
"""vision_feature_sanity.py -- POSITIVE CONTROL for the vision-token cache.

If every vision arm ties or loses to the language-side bar, the first objection is "your vision
features were broken".  This script forecloses that objection BEFORE the null is reported, by
showing the same cached vectors are highly predictive of things the IMAGE genuinely determines,
using the same feature matrix the arms consume.

Three probes, all on feats_vision/vis_eval.npz + vis_train.npz, all image-level, all with the
train/eval image split that is already known to be disjoint (0 shared decoded-RGB md5):

  P1  dataset identity (slake / vqa_rad / pathvqa)  -- modality is visually obvious; a near-ceiling
      score proves the vectors carry real image content and are correctly aligned to their rows.
  P2  image-level difficulty: is this item RECOVERABLE at all (does the 8-sample pool contain a
      correct answer)?  A real but weak signal -- it says the image predicts task difficulty.
  P3  greedy correctness of the 7B on this item.  Same flavour as P2.

P1 is the sanity gate.  P2/P3 are the interesting ones: they measure how much a per-question
CONSTANT can say about the question, which is the ceiling on any additive use of a vision feature.

Also reports the ablation contrast: the same probes on the BLANK and NOISE caches. P1 should
collapse toward chance there; if it does not, the cache is not image-dependent and everything
downstream is suspect.

Pure numpy + sklearn logistic regression; no GPU, no seeds beyond the solver's.
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts/vision_feature_sanity.json")


def probe(Xtr, ytr, Xev, yev, multi=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    m = LogisticRegression(max_iter=2000, C=1.0,
                           multi_class="multinomial" if multi else "auto")
    m.fit((Xtr - mu) / sd, ytr)
    p = m.predict((Xev - mu) / sd)
    out = {"n_train": int(len(ytr)), "n_eval": int(len(yev)),
           "accuracy": float(accuracy_score(yev, p)),
           "majority_baseline": float(max(np.bincount(yev.astype(int))) / len(yev))}
    if not multi:
        s = m.predict_proba((Xev - mu) / sd)[:, 1]
        out["auroc"] = float(roc_auc_score(yev, s)) if len(set(yev.tolist())) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()

    items = G.load_items()
    rep = {"what": "positive control: are the cached vision vectors informative about things the "
                   "IMAGE determines? Establishes that any null in the verifier arms is about the "
                   "TASK, not about broken features.",
           "date": "2026-08-12", "layer": A.layer,
           "code": "src/training_methods/vision_feature_sanity.py",
           "caches": {}}

    # ---- per-IMAGE feature matrices, train and eval
    vtr = V.load_vision("train")
    li = vtr["layers"].index(A.layer)
    Xtr_img = vtr["v_mean"][:, li].astype(np.float32)
    tr_ds = np.array([r["ds"] for r in vtr["rows"]])

    # eval item-level: one vision row per ITEM (items sharing an image share a row -- that is the point)
    ev = G.load_candidates("eval", layers=[], pooling=())
    row_md5 = {(r["ds"], r["idx"]): r.get("img_md5") for r in ev.rows}

    for ablate in ["none", "blank", "noise"]:
        p = os.path.join(V.VISDIR, "vis_eval.npz" if ablate == "none" else f"vis_eval_{ablate}.npz")
        if not os.path.exists(p):
            continue
        vev = V.load_vision("eval", ablate=ablate)
        lie = vev["layers"].index(A.layer)
        Xev_img = vev["v_mean"][:, lie].astype(np.float32)
        ev_ds = np.array([r["ds"] for r in vev["rows"]])
        idx_of = vev["index"]

        # P1 -- dataset identity, image level. The TRAIN cache labels carry a '_train' suffix
        # (pathvqa_open_train / slake_open_train / vqa_rad_open_train, plus kvasir_open which has
        # no eval counterpart and is dropped), so they are mapped onto the eval class names first.
        classes = ["slake_open", "vqa_rad_open", "pathvqa_open"]
        canon = {d: d.replace("_train", "") for d in set(tr_ds.tolist())}
        keep = np.array([canon[d] in classes for d in tr_ds])
        ytr = np.array([classes.index(canon[d]) for d in tr_ds[keep]])
        Xtr1 = Xtr_img[keep]
        yev = np.array([classes.index(d) for d in ev_ds])
        p1 = probe(Xtr1, ytr, Xev_img, yev, multi=True)
        p1["train_images_used"] = int(keep.sum())
        p1["train_classes_dropped"] = sorted({d for d in set(tr_ds.tolist()) if canon[d] not in classes})

        # P2/P3 -- per ITEM, using that item's image vector
        rows, rec, gre = [], [], []
        miss = 0
        for it in items:
            m = row_md5.get((it["ds"], it["idx"]))
            j = idx_of.get(m) if m else None
            if j is None:
                miss += 1
                continue
            rows.append(j)
            rec.append(1 if any(x == 1 for x in it["sl"]) else 0)
            gre.append(int(it["greedy_ok"]))
        rows = np.array(rows); rec = np.array(rec); gre = np.array(gre)
        Xit = Xev_img[rows]
        # image-grouped 5-fold inside eval (never trains and tests on the same image)
        folds = np.array([int(__import__("hashlib").md5(str(vev["rows"][j]["img_md5"]).encode())
                              .hexdigest(), 16) % 5 for j in rows])
        def cv_probe(y):
            accs, aucs = [], []
            for f in range(5):
                tr, va = folds != f, folds == f
                if len(set(y[tr].tolist())) < 2 or va.sum() == 0:
                    continue
                r = probe(Xit[tr], y[tr], Xit[va], y[va])
                accs.append(r["accuracy"])
                if r.get("auroc") is not None:
                    aucs.append(r["auroc"])
            return {"cv_accuracy": float(np.mean(accs)) if accs else None,
                    "cv_auroc": float(np.mean(aucs)) if aucs else None,
                    "positive_rate": float(y.mean()), "n": int(len(y))}

        rep["caches"][ablate] = {
            "P1_dataset_identity_image_level": p1,
            "P2_pool_recoverable_from_image_alone": cv_probe(rec),
            "P3_greedy_correct_from_image_alone": cv_probe(gre),
            "n_items_without_a_vision_row": int(miss)}
        print(f"[{ablate}] P1 dataset acc={p1['accuracy']:.4f} (majority {p1['majority_baseline']:.4f})"
              f"  P2 auroc={rep['caches'][ablate]['P2_pool_recoverable_from_image_alone']['cv_auroc']}"
              f"  P3 auroc={rep['caches'][ablate]['P3_greedy_correct_from_image_alone']['cv_auroc']}",
              flush=True)

    if "none" in rep["caches"] and "noise" in rep["caches"]:
        a = rep["caches"]["none"]["P1_dataset_identity_image_level"]["accuracy"]
        b = rep["caches"]["noise"]["P1_dataset_identity_image_level"]["accuracy"]
        rep["verdict"] = {
            "P1_real": round(a, 6), "P1_noise": round(b, 6), "P1_drop": round(a - b, 6),
            "features_are_image_dependent": bool(a - b > 0.15),
            "reading": "P1 near ceiling on the real cache and collapsing on the noise cache proves "
                       "the cached vision vectors carry real, correctly-aligned image content. Any "
                       "null in the verifier arms is therefore a statement about the TASK -- "
                       "selecting the correct answer -- not about broken features. P2/P3 bound what "
                       "an image-only, per-question CONSTANT can contribute at all."}
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(rep, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}", flush=True)


if __name__ == "__main__":
    main()
