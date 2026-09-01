#!/usr/bin/env python3
"""head_visual_features.py -- does giving the probe the IMAGE tokens directly help?

WHAT THE PROBE HAS ALWAYS READ.  h_span: the mean hidden state over the candidate ANSWER's tokens.
The image reaches it only through whatever the language model already mixed into those positions.
Nothing in this project has ever fed the probe a visual representation directly.

WHAT THE LITERATURE DOES.  VLM probing work pools the image placeholder tokens and combines that
with pooled text tokens -- "Bridging Hidden States in Vision-Language Models" (arXiv 2511.11526)
computes exactly this: image-token representations by mean pooling over the model's image
placeholder positions, text-token representations by mean pooling the non-image tokens, and the
combined representation as their average. It reports that image-token states carry localised visual
information that the text states do not expose. Separately, the hallucination-detection line
(HaloProbe arXiv 2604.06165, HALP arXiv 2603.05465, VIB-Probe arXiv 2601.05547) treats a generated
token's relationship to the image as the signal that separates grounded answers from hallucinated
ones -- which is precisely the judgement our verifier has to make.

MEASURED FIRST, before spending the GPU: on a 709-row slice, cos(h_img, h_span) = 0.62 and
cos(h_q, h_span) = 0.74, so these are genuinely different vectors rather than re-encodings.

VARIANTS (all at layers 18/20/22, rank-ensembled, the shipped readout):
  h_span                 the deployed input
  h_span + h_img         + mean over the image placeholder tokens
  h_span + h_q           THE CONTROL -- mean over the QUESTION tokens. If h_q helps as much as
                         h_img, the gain is "more pooled context", not vision, and the visual
                         story is dead.
  h_span + h_img + h_q   everything
  h_span + cos           an explicit grounding scalar: cosine(h_span, h_img) per candidate, the
                         cheapest possible form of "is this answer aligned with the image"
  h_img alone            can the image tokens alone rank candidates? (they do not depend on the
                         candidate at all, so this MUST be near the random floor -- it is the
                         sanity check that the harness is not leaking)

That last arm matters: h_img is identical for every candidate of a question, so a probe given only
h_img cannot rank within a candidate set. If it scores above the floor, something is wrong with the
evaluation rather than interesting about vision.

  python3 src/training_methods/head_visual_features.py --threads 4 --seeds 3
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
ENS = [18, 20, 22]
VARIANTS = ["h_span", "h_span+h_img", "h_span+h_q", "h_span+h_img+h_q", "h_span+cos", "h_img_only"]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    zp = f"{HS.FEATS}/{stem}.npz"
    if not os.path.exists(zp):
        return None, None
    z = np.load(zp); m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    if not keep:
        return None, None
    out = {}
    for k in ("h_span", "h_img", "h_q"):
        if k in z:
            out[k] = {L: z[k][keep, lay.index(L)].astype(np.float32) for L in ENS}
    return out, [m["rows"][i] for i in keep]


def build(v, F, L):
    sp, im, q = F["h_span"][L], F.get("h_img", {}).get(L), F.get("h_q", {}).get(L)
    if v == "h_span":
        return sp
    if v == "h_span+h_img":
        return np.concatenate([sp, im], 1)
    if v == "h_span+h_q":
        return np.concatenate([sp, q], 1)
    if v == "h_span+h_img+h_q":
        return np.concatenate([sp, im, q], 1)
    if v == "h_span+cos":
        c = ((sp * im).sum(1) / (np.linalg.norm(sp, axis=1) * np.linalg.norm(im, axis=1) + 1e-9))
        return np.concatenate([sp, c[:, None].astype(np.float32)], 1)
    if v == "h_img_only":
        return im
    raise ValueError(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR,
                                                  "head_visual_features_2026-09-01.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    Ftr, rows = load("generator_train_vis", TRAIN_DOMAINS)
    if Ftr is None:
        raise SystemExit("missing generator_train_vis.npz -- run the --visual_feats extraction")
    src = [r["ds"] for r in rows]
    ev = {}
    add = {k: {L: [Ftr[k][L]] for L in ENS} for k in Ftr}
    for cell in BENCH:
        Fc, rr = load(f"generator_eval_vis_{cell}")
        if Fc is None:
            continue
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        for k in Fc:
            for L in ENS:
                add[k][L].append(Fc[k][L][istr])
        rows += [rr[i] for i in np.where(istr)[0]]
        src += [cell] * int(istr.sum())
        gok = {}
        for l in open(f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = ({k: {L: Fc[k][L][~istr] for L in ENS} for k in Fc},
                    [rr[i] for i in np.where(~istr)[0]], gok)
        print(f"  {cell:17} train {int(istr.sum()):6,} / held-out {int((~istr).sum()):6,}",
              flush=True)
    F = {k: {L: np.concatenate(add[k][L]) for L in ENS} for k in add}

    evimgs = set()
    for _, rr, _ in ev.values():
        evimgs |= {r["img_md5"] for r in rr}
    drop = np.array([r["img_md5"] in evimgs for r in rows])
    if drop.any():
        for k in F:
            for L in ENS:
                F[k][L] = F[k][L][~drop]
        rows = [r for r, d in zip(rows, drop) if not d]
        src = [s for s, d in zip(src, drop) if not d]
        print(f"dropped {int(drop.sum())} leaking training rows", flush=True)
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src, rows)])
    print(f"pooled training set: {len(y):,} rows over {len(ev)} benchmarks\n", flush=True)

    art = {"title": "Do image-token hidden states help the verifier?", "date": "2026-09-01",
           "no_fabricated_numbers": True, "variants": VARIANTS, "seeds": A.seeds,
           "layers": ENS, "pooled_rows": int(len(y)), "cells": {}}
    for v in VARIANTS:
        M = {}
        for L in ENS:
            X = build(v, F, L)
            mu, sg = X.mean(0), X.std(0) + 1e-6
            M[L] = (mu, sg, [HS.fit((X - mu) / sg, y, qid, None, objective="bce", hidden=256,
                                    wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)])
        for cell, (Fe, rr, gok) in ev.items():
            ye = np.array([r["y"] for r in rr], dtype=int)
            byq = defaultdict(list)
            for i, r in enumerate(rr):
                byq[r["idx"]].append(i)
            qs = [q for q in byq if q in gok]
            if len(qs) < 40:
                continue
            S = []
            for L in ENS:
                mu, sg, ms = M[L]
                Xe = build(v, Fe, L)
                S.append(np.stack([HS.predict(mm, (Xe - mu) / sg) for mm in ms]))
            acc = []
            for q in qs:
                ii = np.array(byq[q])
                hr = np.mean([s[k][ii] for s in S for k in range(s.shape[0])], axis=0) \
                    if False else np.mean([rank_avg(s[k][ii]) for s in S
                                           for k in range(s.shape[0])], axis=0)
                acc.append(int(ye[ii][int(np.argmax(hr))]))
            g = float(np.mean([gok[q] for q in qs]))
            art["cells"].setdefault(cell, {"greedy": g, "n_questions": len(qs)})
            art["cells"][cell][v] = float(np.mean(acc)) - g
        print(f"  {v:20} " + "  ".join(f"{c[:9]} {art['cells'][c][v]:+.4f}"
                                       for c in art["cells"]), flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mac = {v: float(np.mean([c[v] for c in cs.values() if v in c])) for v in VARIANTS}
        art["macro"] = mac
        base, img, q = mac["h_span"], mac["h_span+h_img"], mac["h_span+h_q"]
        art["visual_gain"] = img - base
        art["context_control_gain"] = q - base
        art["visual_minus_control"] = img - q
        best = max(mac, key=mac.get)
        for v in VARIANTS:
            print(f"  MACRO {v:20} {mac[v]:+.4f}")
        art["VERDICT"] = (
            f"h_span {base:+.4f}; +h_img {img:+.4f} ({img-base:+.4f}); +h_q {q:+.4f} "
            f"({q-base:+.4f}); best {best}. Image tokens beat the question-token control by "
            f"{img-q:+.4f}. " +
            ("Visual features help and the gain is specifically VISUAL, not just extra context."
             if img - base > 0.005 and img - q > 0.005 else
             "Adding pooled question tokens helps about as much as pooled image tokens, so the "
             "gain is extra context rather than vision." if img - base > 0.005 else
             "Feeding the image tokens directly does NOT help: whatever the verifier needs from "
             "the image, the language model has already mixed into the answer positions."))
        print(f"\n  sanity: h_img_only {mac['h_img_only']:+.4f} -- must be near the floor, since "
              f"h_img is identical for every candidate of a question")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
