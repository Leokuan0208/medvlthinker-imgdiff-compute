#!/usr/bin/env python3
"""vision_relevance_vs_correctness.py -- WHY the vision arms behave the way they do.

BACKGROUND (docs/current/VERIFIER_ARCHITECTURES_2026-08-04.md, family D).  The project already
tested vision-aware verification with EXTERNAL dual encoders (SigLIP / PubMedCLIP / BiomedCLIP) and
measured the mechanism with a two-test control on the identical score function:

    score               on-topic CORRECTNESS AUROC        off-topic RELEVANCE AUROC
    SigLIP              0.540                             0.764
    PubMedCLIP          0.527                             0.651
    BiomedCLIP          0.554                             0.726
    incumbent verifier  0.756                             --
  and on the laterality slice (n=342) zero-shot SigLIP scored 0.464 -- BELOW chance -- vs the
  incumbent's 0.635.

Conclusion recorded there: an image-text similarity has real resolution for "is this text ABOUT this
image" and essentially none for "is this the RIGHT ANSWER", and it fails worst exactly where a
one-token contrast (left/right) decides the item.

WHAT THIS SCRIPT ADDS.  Those encoders were trained with a contrastive objective that never had to
encode laterality.  The generator's OWN vision tower is a different object: it is the representation
Lingshu-7B actually conditions on, inside the model that produced the candidates.  So the fair
question is whether the SAME decomposition comes out differently when the similarity is computed in
the generator's own space -- candidate language-side vector vs the generator's own pooled patch grid.

Two tests on the identical score function, exactly mirroring the family-D control:
  CORRECTNESS  within a question's pool, correct candidate vs incorrect candidate
  RELEVANCE    a question's own candidate vs a candidate drawn from a DIFFERENT question
               (the image stays the same; only the text is foreign)
plus the laterality slice, so the numbers are directly comparable to 0.464 / 0.635.

Scores compared:
  cos_max / cos_mean / cos_top3   within-model cosine to the pooled patch grid (the L_simgrid /
                                  L_maxsim raw material)
  cos_vmean                       cosine to the whole-image mean vision vector
  incumbent                       the deployed LoRA verifier's stored score (reference)

Pure numpy over cached features; no GPU, no training, no seeds.
"""
import argparse, json, os, sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G   # noqa: E402
import visverif_lib as V    # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts/relevance_vs_correctness.json")


def auroc(pos, neg):
    """P(score(pos) > score(neg)) with ties at 0.5, over PAIRED draws."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0:
        return float("nan")
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--grid_layer", type=int, default=21)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    A = ap.parse_args()

    items = G.load_items()
    ev = G.load_candidates("eval", layers=[A.layer], pooling=("span",))
    vev = V.load_vision("eval")
    Vm, Vg, iev = V.align(ev, vev, A.layer, A.grid_layer)
    H = ev.matrix("span", A.layer)
    y = np.array([r["y"] for r in ev.rows], dtype=int)

    sim = V.cos_grid(H, Vg)                      # (n_rows, 36) candidate vs each pooled patch
    hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-6)
    vn = Vm / (np.linalg.norm(Vm, axis=1, keepdims=True) + 1e-6)
    srt = np.sort(sim, 1)
    SCORES = {"cos_max": sim.max(1), "cos_mean": sim.mean(1), "cos_top3": srt[:, -3:].mean(1),
              "cos_vmean": (hn * vn).sum(1)}

    # row index per (ds, idx, na); and the incumbent score per row
    row_of = {(r["ds"], r["idx"], r["na"]): i for i, r in enumerate(ev.rows)}
    inc_row = np.full(len(ev.rows), np.nan)
    for it in items:
        for s, a in enumerate(it["preds"]):
            i = row_of.get((it["ds"], it["idx"], G.norm(a)))
            if i is not None:
                inc_row[i] = float(it["scores"][s])
    SCORES["incumbent"] = inc_row

    # ------------------------------------------------------------------ strata over ROWS
    st = V.strata(items)
    lat_item = st["laterality"]
    item_of_row = {}
    for k, it in enumerate(items):
        for a in it["preds"]:
            i = row_of.get((it["ds"], it["idx"], G.norm(a)))
            if i is not None:
                item_of_row[i] = k
    lat_row = np.zeros(len(ev.rows), bool)
    for i, k in item_of_row.items():
        lat_row[i] = lat_item[k]

    # ------------------------------------------------------------------ pair construction
    byq = defaultdict(list)
    for i, r in enumerate(ev.rows):
        byq[(r["ds"], r["idx"])].append(i)

    rng = np.random.default_rng(A.seed)
    corr_pairs, corr_pairs_lat = [], []
    for q, ii in byq.items():
        pos = [i for i in ii if y[i] == 1]
        neg = [i for i in ii if y[i] == 0]
        for p in pos:
            for n in neg:
                corr_pairs.append((p, n))
                if lat_row[p]:
                    corr_pairs_lat.append((p, n))

    # RELEVANCE: own candidate vs a candidate text from a DIFFERENT question, same image.
    # The "foreign" row is another question's row, so its language vector is foreign while the
    # patch grid it is compared against is this question's -- which is exactly the off-topic
    # contrast family D used.
    allrows = np.arange(len(ev.rows))
    rel_pairs, rel_pairs_lat = [], []
    for q, ii in byq.items():
        for i in ii:
            j = int(rng.choice(allrows))
            while item_of_row.get(j) == item_of_row.get(i):
                j = int(rng.choice(allrows))
            rel_pairs.append((i, j, i))     # (own row, foreign row, image-owning row)
            if lat_row[i]:
                rel_pairs_lat.append((i, j, i))

    def corr_auroc(s, pairs):
        return auroc([s[p] for p, n in pairs], [s[n] for p, n in pairs])

    def _grid_stats(text_rows, img_rows, chunk=2048):
        """cos(text vector of row t, each pooled patch of the image owned by row k), reduced to
        max / mean / top3. Chunked so the (n, 36, 3584) intermediate never materialises."""
        out = {k: np.empty(len(text_rows), np.float32) for k in ("cos_max", "cos_mean", "cos_top3")}
        for s in range(0, len(text_rows), chunk):
            ti = text_rows[s:s + chunk]; ki = img_rows[s:s + chunk]
            g = Vg[ki]                                              # (c, 36, 3584)
            g = g / (np.linalg.norm(g, axis=2, keepdims=True) + 1e-6)
            c = np.einsum("nd,npd->np", hn[ti], g).astype(np.float32)
            out["cos_max"][s:s + chunk] = c.max(1)
            out["cos_mean"][s:s + chunk] = c.mean(1)
            out["cos_top3"][s:s + chunk] = np.sort(c, 1)[:, -3:].mean(1)
        return out

    def rel_auroc_all(pairs):
        """Own text vs a FOREIGN question's text, both scored against the OWN image."""
        own_i = np.array([p[0] for p in pairs]); for_i = np.array([p[1] for p in pairs])
        img_i = np.array([p[2] for p in pairs])
        go = _grid_stats(own_i, img_i); gf = _grid_stats(for_i, img_i)
        r = {k: auroc(go[k], gf[k]) for k in go}
        r["cos_vmean"] = auroc((hn[own_i] * vn[img_i]).sum(1), (hn[for_i] * vn[img_i]).sum(1))
        r["incumbent"] = float("nan")   # no incumbent score exists for a foreign text on this image
        return r

    rel_all = rel_auroc_all(rel_pairs)
    rel_lat = rel_auroc_all(rel_pairs_lat)
    res = {}
    for name, s in SCORES.items():
        res[name] = {
            "correctness_auroc": corr_auroc(s, corr_pairs),
            "correctness_auroc_laterality": corr_auroc(s, corr_pairs_lat),
            "relevance_auroc": rel_all[name],
            "relevance_auroc_laterality": rel_lat[name],
        }
        print(f"  {name:<11} correctness={res[name]['correctness_auroc']:.4f} "
              f"(lat {res[name]['correctness_auroc_laterality']:.4f})  "
              f"relevance={res[name]['relevance_auroc']:.4f}", flush=True)

    rep = {
        "what": "Does a similarity computed in the GENERATOR'S OWN vision space separate CORRECTNESS "
                "any better than an external contrastive encoder did? Mirrors the family-D control "
                "table in docs/current/VERIFIER_ARCHITECTURES_2026-08-04.md.",
        "date": "2026-08-12",
        "code": "src/training_methods/vision_relevance_vs_correctness.py",
        "layer": A.layer, "grid_layer": A.grid_layer, "pairing_seed": A.seed,
        "n_correctness_pairs": len(corr_pairs), "n_correctness_pairs_laterality": len(corr_pairs_lat),
        "n_relevance_pairs": len(rel_pairs),
        "definitions": {
            "correctness_auroc": "within a question's candidate pool, P(score(correct) > score(wrong)) "
                                 "over all correct x wrong pairs, ties 0.5",
            "relevance_auroc": "P(score(own candidate, own image) > score(a DIFFERENT question's "
                               "candidate, own image)) -- 'is this text about this image'",
            "laterality": "restricted to items whose question / gold / any candidate carries a "
                          "laterality or orientation token"},
        "published_external_encoder_reference": {
            "source": "results/cascade_methods/docs/current/VERIFIER_ARCHITECTURES_2026-08-04.md §4.4",
            "SigLIP": {"correctness": 0.540, "relevance": 0.764, "laterality_slice_n342": 0.464},
            "PubMedCLIP": {"correctness": 0.527, "relevance": 0.651},
            "BiomedCLIP": {"correctness": 0.554, "relevance": 0.726},
            "incumbent_verifier": {"correctness": 0.756, "laterality_slice_n342": 0.635}},
        "measured_within_model": res,
    }
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(rep, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}", flush=True)


if __name__ == "__main__":
    main()
