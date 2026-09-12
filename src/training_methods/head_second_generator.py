#!/usr/bin/env python3
"""head_second_generator.py -- does the method work on a DIFFERENT generator, or only on Lingshu?

Every number in this project comes from Lingshu-7B.  Nothing in it separates "a probe on frozen
hidden states is a good best-of-N verifier" from "a probe on LINGSHU's hidden states is".  That is
the difference between a method and an observation about one checkpoint.

Qwen2.5-VL-7B-Instruct is the right second model: it is the base Lingshu was finetuned from, so the
architecture is identical and extract_generator_hidden runs unchanged, and it isolates the variable
that matters -- whether the MEDICAL finetuning is what makes the hidden states probe-able.
(InternVL3-8B was tried first and is incompatible: AutoProcessor returns a bare tokenizer with no
image_processor and AutoModelForImageTextToText rejects InternVLChatConfig.)

The whole pipeline was re-run on Qwen: 8-sample candidate sets at T=0.7, greedy answers, 32B judging
with the SAME judge, and hidden-state extraction at layers 18/20/22.  The probe is then fitted on
Qwen states and scored against QWEN's own greedy decoding -- an entirely self-contained replication.

Reported beside the Lingshu numbers on the same benchmarks, but note the two are NOT a paired
comparison: the generators have different accuracy, so what replicates is the SHAPE (does a probe
beat that model's own greedy decoding, and by how much), not the absolute delta.

  python3 src/training_methods/head_second_generator.py --threads 4 --seeds 5
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TAG = "qwen25vl7b"
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
# EXTENDED to all eight 2026-09-12. The Qwen pipeline finished every benchmark during the week; this
# list was still the four it was scoped to for a first answer, so the artifact reported a 4-benchmark
# replication when 8-benchmark data was already on disk. Scoring the same set as the Lingshu result
# makes the replication claim directly comparable instead of a subset.
EVAL = ["pathvqa_open", "slake_open", "vqa_rad_open", "kvasir_x1_open",
        "radimagenet_open", "omnimed_open", "vqamed_open", "gemex_open"]
ENS = [18, 20, 22]


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
    return ({L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS},
            [m["rows"][i] for i in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR,
                                                  "head_second_generator_2026-09-04.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    Xtr, rows = load("generator_train_qwen_kvasir_open")
    parts = {}
    for c in TRAIN_DOMAINS:
        Xc, rc = load(f"generator_train_qwen_{c}")
        if Xc is not None:
            parts[c] = (Xc, rc)
    if not parts:
        raise SystemExit("no Qwen training caches found")
    ev = {}
    for c in EVAL:
        Xc, rc = load(f"generator_eval_qwen_{c}")
        gjp = f"{CK}/ckpt_{c}_{TAG}.judge.jsonl"
        if Xc is None or not os.path.exists(gjp):
            print(f"  [skip] {c}: features or greedy judge missing", flush=True); continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        istr = np.array([half(r["img_md5"]) == 1 for r in rc])
        parts[f"__ev__{c}"] = ({L: Xc[L][istr] for L in ENS},
                               [rc[i] for i in np.where(istr)[0]])
        ev[c] = ({L: Xc[L][~istr] for L in ENS},
                 [rc[i] for i in np.where(~istr)[0]], gok)
        print(f"  {c:17} train {int(istr.sum()):6,} / held-out {int((~istr).sum()):6,}", flush=True)

    evimgs = set()
    for _, rr, _ in ev.values():
        evimgs |= {r["img_md5"] for r in rr}
    Xs = {L: [] for L in ENS}; allr, src = [], []
    for nm, (Xc, rc) in parts.items():
        keep = np.array([r["img_md5"] not in evimgs for r in rc])
        for L in ENS:
            Xs[L].append(Xc[L][keep])
        allr += [r for r, k in zip(rc, keep) if k]
        src += [(r["ds"] if not nm.startswith("__ev__") else nm[6:])
                for r, k in zip(rc, keep) if k]
    X = {L: np.concatenate(Xs[L]) for L in ENS}
    y = np.array([r["y"] for r in allr], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src, allr)])
    print(f"\nQwen pooled training set: {len(y):,} rows\n", flush=True)

    M = {}
    for L in ENS:
        mu, sg = X[L].mean(0), X[L].std(0) + 1e-6
        M[L] = (mu, sg, [HS.fit((X[L] - mu) / sg, y, qid, None, objective="bce", hidden=256,
                                wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)])
        print(f"  fitted layer {L}", flush=True)

    art = {"title": "Does the probe-verifier method replicate on a second generator?",
           "date": "2026-09-04", "no_fabricated_numbers": True, "generator": "Qwen2.5-VL-7B-Instruct",
           "judge": "unchanged (MedVLThinker-32B)", "seeds": A.seeds, "layers": ENS,
           "train_rows": int(len(y)), "cells": {}}
    for c, (Xe, rr, gok) in ev.items():
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
            S.append(np.stack([HS.predict(mm, (Xe[L] - mu) / sg) for mm in ms]))
        acc, orc = [], []
        for q in qs:
            ii = np.array(byq[q])
            hr = np.mean([rank_avg(s[k][ii]) for s in S for k in range(s.shape[0])], axis=0)
            acc.append(int(ye[ii][int(np.argmax(hr))])); orc.append(int(ye[ii].max()))
        g = float(np.mean([gok[q] for q in qs]))
        art["cells"][c] = {"n_questions": len(qs), "greedy": g, "verifier": float(np.mean(acc)),
                           "verifier_minus_greedy": float(np.mean(acc) - g),
                           "oracle_at_8": float(np.mean(orc))}
        r = art["cells"][c]
        print(f"  {c:17} n{len(qs):6} greedy {g:.4f}  verifier {r['verifier']:.4f}  "
              f"delta {r['verifier_minus_greedy']:+.4f}  oracle {r['oracle_at_8']:.4f}", flush=True)
        json.dump(art, open(A.out, "w"), indent=1)

    cs = art["cells"]
    if cs:
        mac = float(np.mean([v["verifier_minus_greedy"] for v in cs.values()]))
        art["macro_verifier_minus_greedy"] = mac
        art["beats_greedy"] = f"{sum(1 for v in cs.values() if v['verifier_minus_greedy']>0)}/{len(cs)}"
        art["lingshu_reference"] = {
            "note": "NOT a paired comparison -- different generators have different accuracy, so "
                    "what replicates is the SHAPE (a probe beats that model's own greedy), not the "
                    "absolute delta",
            "lingshu_pooled_macro_on_8_benchmarks": 0.0802,
            "source": "head_final_stack_2026-08-24.json"}
        art["VERDICT"] = (
            f"on Qwen2.5-VL-7B the same recipe gives {mac:+.4f} macro over Qwen's own greedy "
            f"decoding, beating it on {art['beats_greedy']} benchmarks. " +
            ("The method REPLICATES on a second generator -- it is not a property of Lingshu or of "
             "medical finetuning." if mac > 0.01 else
             "The method does NOT replicate on the base model: the gain appears to depend on "
             "Lingshu's medical finetuning, which would make every result here a statement about "
             "one checkpoint rather than about a method."))
        print(f"\n  MACRO {mac:+.4f}  ({art['beats_greedy']} beat greedy)")
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
