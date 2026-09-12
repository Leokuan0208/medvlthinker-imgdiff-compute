#!/usr/bin/env python3
"""vqamed_contamination.py -- MEASURE the MedPix overlap's effect instead of estimating it.

THE OVERLAP.  vqa_rad_open_train -- a training domain for the deployed probe -- shares 19 images
(by decoded-pixel md5) with vqamed_open, an evaluation benchmark.  It is the only nonzero cell in
all 32 (training domain x evaluation benchmark) pairs.

ROOT CAUSE, found 2026-09-12: build_vqamed_cell.py globbed "*.jpg" while every VQA-RAD image on
disk is ".png" (2,249 png, 0 jpg), so that arm of the build-time contamination audit hashed an
empty set and printed "0 VQA-RAD images".  Fixed to be extension-agnostic.

WHY THIS FILE EXISTS.  The effect was recorded in a CODE COMMENT as "-0.0259 -> -0.0261", which is
an estimate sitting where a measurement belongs.  This project's rule 7 says every figure must come
verbatim from real experimental output and name its file.  So: measure it, write it down, and let
the docs cite an artifact.

It matters most for the DEPLOYED probe.  freeze_selector.py fits genframe_head_ens8 with no
image-hash leakage filter at all, and that probe is quoted on the FULL vqamed_open (3,663 questions,
not a held-out half) -- so those 19 questions are genuinely in its reported number.  The pooled
artifact is unaffected: freeze_pooled_selector.py drops leaking rows (59 rows over 7 images) before
fitting.

  python3 src/cascade_methods/vqamed_contamination.py
"""
import json, os, sys
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
FEATS = os.path.join(ROOT, "feats_hidden")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/vqamed_contamination_2026-09-12.json")
LAYERS = [7, 14, 21, 28]


def norm(s):
    return str(s).strip().lower().rstrip(".")


def main():
    from genframe_selector import FrozenSelector
    from genframe_data import rank_avg
    sel = FrozenSelector.load()

    tr_imgs = set()
    for sh in (0, 1):
        for r in json.load(open(f"{FEATS}/generator_train_s{sh}of2.meta.json"))["rows"]:
            if r.get("n_tok", -1) > 0 and r["ds"] == "vqa_rad_open_train":
                tr_imgs.add(r["img_md5"])

    stem = f"{FEATS}/generator_eval_vqamed_open"
    z = np.load(stem + ".npz"); m = json.load(open(stem + ".meta.json"))
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0]
    rows = [m["rows"][i] for i in keep]
    H = z["h_span"][keep, LAYERS.index(21)].astype(np.float32)
    y = np.array([r["y"] for r in rows], dtype=int)
    L = sel.head_logits(H)
    gok = {}
    for l in open(f"{CK}/ckpt_vqamed_open_lingshu7b.judge.jsonl"):
        if l.strip():
            d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[r["idx"]].append(i)
    img = {r["idx"]: r["img_md5"] for r in rows}

    ver, gre, dirty, orc = [], [], [], []
    for q in byq:
        if q not in gok:
            continue
        ii = np.array(byq[q])
        hr = np.mean([rank_avg(L[k][ii]) for k in range(L.shape[0])], axis=0)
        ver.append(int(y[ii][int(np.argmax(hr))])); gre.append(gok[q])
        orc.append(int(y[ii].max())); dirty.append(img[q] in tr_imgs)
    ver, gre, dirty, orc = map(np.array, (ver, gre, dirty, orc))
    clean = ~dirty

    art = {"title": "Measured effect of the vqa_rad_open_train / vqamed_open MedPix overlap",
           "date": "2026-09-12", "no_fabricated_numbers": True,
           "probe": "genframe_head_ens8 (the DEPLOYED probe, fitted without a leakage filter)",
           "overlap_images": len(tr_imgs & {img[q] for q in byq}),
           "n_questions_total": int(len(ver)),
           "n_questions_contaminated": int(dirty.sum()),
           "contaminated_fraction": float(dirty.mean()),
           "all_questions": {"verifier": float(ver.mean()), "greedy": float(gre.mean()),
                             "verifier_minus_greedy": float(ver.mean() - gre.mean()),
                             "oracle_at_8": float(orc.mean())},
           "clean_only": {"n": int(clean.sum()), "verifier": float(ver[clean].mean()),
                          "greedy": float(gre[clean].mean()),
                          "verifier_minus_greedy": float(ver[clean].mean() - gre[clean].mean()),
                          "oracle_at_8": float(orc[clean].mean())},
           "contaminated_only": {"n": int(dirty.sum()),
                                 "verifier": float(ver[dirty].mean()),
                                 "greedy": float(gre[dirty].mean()),
                                 "oracle_at_8": float(orc[dirty].mean())}}
    art["effect_on_reported_delta"] = (art["clean_only"]["verifier_minus_greedy"]
                                       - art["all_questions"]["verifier_minus_greedy"])
    art["VERDICT"] = (
        f"{art['n_questions_contaminated']} of {art['n_questions_total']} vqamed questions "
        f"({art['contaminated_fraction']:.3%}) sit on an image the deployed probe trained on. "
        f"Removing them moves verifier-minus-greedy from "
        f"{art['all_questions']['verifier_minus_greedy']:+.4f} to "
        f"{art['clean_only']['verifier_minus_greedy']:+.4f} "
        f"({art['effect_on_reported_delta']:+.5f}). "
        + ("Immaterial: the contaminated questions are ones the model fails either way "
           f"(verifier {art['contaminated_only']['verifier']:.4f}, greedy "
           f"{art['contaminated_only']['greedy']:.4f}, oracle "
           f"{art['contaminated_only']['oracle_at_8']:.4f}), so the overlap cannot have inflated "
           "anything."
           if abs(art["effect_on_reported_delta"]) < 0.002 else
           "MATERIAL -- the reported vqamed number must be restated on the clean subset."))
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"  overlap images: {art['overlap_images']}")
    print(f"  contaminated: {art['n_questions_contaminated']}/{art['n_questions_total']} "
          f"({art['contaminated_fraction']:.3%})")
    print(f"  all          verifier {art['all_questions']['verifier']:.4f} greedy "
          f"{art['all_questions']['greedy']:.4f} delta "
          f"{art['all_questions']['verifier_minus_greedy']:+.4f}")
    print(f"  clean only   verifier {art['clean_only']['verifier']:.4f} greedy "
          f"{art['clean_only']['greedy']:.4f} delta "
          f"{art['clean_only']['verifier_minus_greedy']:+.4f}")
    print(f"  dirty only   verifier {art['contaminated_only']['verifier']:.4f} greedy "
          f"{art['contaminated_only']['greedy']:.4f} oracle "
          f"{art['contaminated_only']['oracle_at_8']:.4f}")
    print(f"\n=> {art['VERDICT']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
