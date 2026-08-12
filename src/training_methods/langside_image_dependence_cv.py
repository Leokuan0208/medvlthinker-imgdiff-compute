#!/usr/bin/env python3
"""langside_image_dependence_cv.py -- THE CONFOUND-FREE PREMISE TEST for the vision-aware verifier
round (attack 1).

THE PREMISE UNDER TEST.  The round's hypothesis was: "the verifier is not really looking at the
image, so injecting the vision signal should help."  The `langnoise` ablation already re-scored the
language-side head on eval features extracted from noise images and saw a large drop -- but that
number is CONFOUNDED, because the head was TRAINED on real-image features and TESTED on
noise-image features, so the drop mixes "lost the image information" with "the features are now out
of distribution".  The pre-registration says so explicitly
(image_ablations._what._langnoise_interpretation_is_ASYMMETRIC).

WHAT THIS SCRIPT DOES INSTEAD.  It trains and tests ENTIRELY IN-DISTRIBUTION on each cache:

    real  : fit and evaluate on feats_hidden        (real images)
    noise : fit and evaluate on feats_hidden_noise  (image replaced by uniform RGB noise)

by 5-fold IMAGE-GROUPED cross-validation INSIDE THE EVAL POOL, identical folds, identical seeds,
identical trainer (visverif_lib.train_cv_sel_eff -> the deployed language-side recipe).  Neither
arm ever sees an out-of-distribution feature, so the difference is a clean measurement of

    how much of the language-side verifier's selection power is IMAGE-DERIVED.

WHY THIS IS A DIAGNOSTIC AND NOT A HEADLINE.  It trains on eval rows (held out fold-wise, grouped
by image md5 so no image and no question straddles a fold).  Its LEVEL is therefore not comparable
to the deployed sel_eff 0.775204 / 0.793869 numbers, which are fitted on the disjoint TRAIN pool.
Only the real-minus-noise CONTRAST is used, and both sides are produced by the same protocol.

  OMP_NUM_THREADS=8 python3 -u src/training_methods/langside_image_dependence_cv.py --seeds 0 1 2
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402

PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")
OUT = os.path.join(PARTS, "langside_image_dependence.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()
    torch.set_num_threads(A.threads)

    caches = {"real": "feats_hidden", "noise": "feats_hidden_noise"}
    res, t0 = {}, time.time()
    folds_ref = rows_ref = None

    for name, fd in caches.items():
        ev = G.load_candidates("eval", layers=[A.layer], pooling=("span",),
                               featdir=os.path.join(ROOT, fd))
        X = ev.matrix("span", A.layer)
        y = np.array([r["y"] for r in ev.rows], dtype=np.float32)
        q = V.qid(ev.rows)
        # the two caches MUST be row-aligned, or the contrast is meaningless
        key = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
        if rows_ref is None:
            rows_ref = key
            # FOLDS ARE DEFINED ONCE, ON THE REAL CACHE, AND REUSED VERBATIM FOR BOTH ARMS.
            # img_folds hashes r['img_md5'], and the noise cache records the md5 of the NOISE
            # image, so recomputing folds per cache would give the two arms DIFFERENT splits and
            # the contrast would confound the arm with the split. (The first version of this
            # script did exactly that; the row-alignment assertion below caught it.)
            folds_ref = V.img_folds(ev.rows, A.folds)
        else:
            assert key == rows_ref, "the two caches are not row-aligned"
        folds = folds_ref
        per_seed = []
        for s in A.seeds:
            r = V.train_cv_sel_eff(X, y, q, folds, seed=s)
            per_seed.append(r)
            print(f"  [{name} seed {s}] cv_sel_eff={r['cv_sel_eff']:.6f} "
                  f"cv_auroc={r['cv_auroc']:.6f} ({(time.time()-t0)/60:.1f} min)", flush=True)
        se = [r["cv_sel_eff"] for r in per_seed]
        au = [r["cv_auroc"] for r in per_seed]
        res[name] = {"featdir": fd, "n_rows": int(X.shape[0]), "d": int(X.shape[1]),
                     "n_folds": A.folds, "seeds": A.seeds,
                     "cv_sel_eff_mean": round(float(np.mean(se)), 6),
                     "cv_sel_eff_sd": round(float(np.std(se, ddof=1)), 6) if len(se) > 1 else None,
                     "cv_sel_eff_per_seed": [round(x, 6) for x in se],
                     "cv_auroc_mean": round(float(np.mean(au)), 6),
                     "cv_auroc_per_seed": [round(x, 6) for x in au],
                     "fold_counts": np.bincount(folds_ref).tolist()}

    d = [res["real"]["cv_sel_eff_per_seed"][i] - res["noise"]["cv_sel_eff_per_seed"][i]
         for i in range(len(A.seeds))]
    da = [res["real"]["cv_auroc_per_seed"][i] - res["noise"]["cv_auroc_per_seed"][i]
          for i in range(len(A.seeds))]
    out = {
        "what": "CONFOUND-FREE premise test: how much of the LANGUAGE-SIDE verifier's selection "
                "power is image-derived? Both arms are trained AND tested in-distribution on their "
                "own cache, by identical image-grouped 5-fold CV inside the eval pool.",
        "date": "2026-08-12",
        "code": "src/training_methods/langside_image_dependence_cv.py",
        "why_this_exists": "the langnoise ablation (image_ablations.L.langnoise) trains on real "
                           "features and tests on noise features, so its drop mixes lost image "
                           "information with an out-of-distribution shift. This one does not.",
        "IS_A_DIAGNOSTIC_NOT_A_HEADLINE": "fitted on eval rows (fold-held-out, grouped by image "
                                          "md5). Its LEVEL is not comparable to the deployed "
                                          "sel_eff figures, which are fitted on the disjoint TRAIN "
                                          "pool. Only the real-minus-noise contrast is used.",
        "grouping": "5-fold on md5(img_md5) % 5 -- no image and therefore no question straddles a "
                    "fold boundary. The folds are computed ONCE from the REAL cache and reused "
                    "verbatim for the noise arm, because the noise cache records the md5 of the "
                    "NOISE image; recomputing per cache would hand the two arms different splits.",
        "numerics": {"layer": A.layer, "pooling": "span", "threads": A.threads,
                     "trainer": "visverif_lib.train_cv_sel_eff (deployed language-side recipe)"},
        "arms": res,
        "contrast_real_minus_noise": {
            "d_cv_sel_eff_mean": round(float(np.mean(d)), 6),
            "d_cv_sel_eff_per_seed": [round(x, 6) for x in d],
            "d_cv_sel_eff_sd": round(float(np.std(d, ddof=1)), 6) if len(d) > 1 else None,
            "n_positive": int(sum(1 for x in d if x > 0)), "n_seeds": len(d),
            "d_cv_auroc_mean": round(float(np.mean(da)), 6),
            "d_cv_auroc_per_seed": [round(x, 6) for x in da]},
        "how_to_read": {
            "if_the_contrast_is_LARGE": "the language-side representation already carries "
                                        "substantial image information, and the round's premise "
                                        "-- 'the verifier is not looking at the image' -- is FALSE. "
                                        "That would explain the null on every vision-injection arm: "
                                        "there is no vision blindness left to fix.",
            "if_the_contrast_is_~ZERO": "the language-side head really is scoring from text priors "
                                        "alone, the premise holds, and the vision-injection null "
                                        "would then be a statement about the INJECTION MECHANISMS "
                                        "tried rather than about the information being redundant."},
    }
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(out, open(A.out, "w"), indent=1)
    print(json.dumps(out["contrast_real_minus_noise"], indent=1))
    print("wrote", A.out)


if __name__ == "__main__":
    main()
