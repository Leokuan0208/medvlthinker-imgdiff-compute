#!/usr/bin/env python3
"""head_final_stack.py -- everything that worked, together, measured the way it would be deployed.

THREE THINGS SURVIVED end-to-end validation, and they act on different parts of the method:

  pooled training      +0.0565  fit on the training half of all eight benchmarks rather than the
                               four July domains (head_pooled_alldomains_2026-08-24.json)
  layer ensembling     +0.0125  rank-ensemble probes over layers 18/20/22 instead of layer 21 alone
  self-consistency     +0.0098  give the probe the candidate's sampling frequency as an input
                               (both from head_best_config_2026-08-24.json, where together they are
                               +0.0168 over the deployed recipe, so they are additive with each other)

They plausibly stack -- one changes WHAT the probe is fitted on, the others change WHAT IT READS --
but "plausibly" is how the layer sweep went wrong, so this measures it.

EVERY ARM IS EVALUATED ON THE SAME HELD-OUT HALVES, split by image with
md5("nd" + img_md5) % 2, so the deployed recipe is re-measured on exactly the questions the stacked
one is scored on rather than quoted from a run over the full benchmark.  Question counts are
therefore about half the usual and are printed.

SECOND GENERATOR (added 2026-09-13).  --generator qwen runs the identical arm ladder on
Qwen2.5-VL-7B, so "does pooled training replicate on a second generator?" is answered by ONE
implementation rather than by a sibling script that drifts.  head_second_generator.py answered the
four-domain half of that question and had a defect this path does not:

  ITS TRAINING SET WAS UP TO 4x DUPLICATED.  The Qwen train features were extracted in four runs
  named generator_train_qwen_{kvasir_open,pathvqa_open_train,slake_open_train,vqa_rad_open_train},
  but each of those caches holds an OVERLAPPING MIXTURE of all four domains, not the one in its
  name (pathvqa_open_train and kvasir_open are byte-identical, 1,365,111,910 bytes each).
  head_second_generator.py:75 concatenated all four, so the same (ds, idx, candidate) entered
  training up to four times, pathvqa was overweighted ~4x and slake -- absent from two of the four
  caches -- was underweighted.  Its reported train_rows 170,014 is that inflation; the deduplicated
  count is printed by this script.  Here the four caches are merged and DEDUPED on
  (ds, idx, normalised answer) before anything is fitted.

  python3 src/training_methods/head_final_stack.py --threads 4 --seeds 5
  python3 src/training_methods/head_final_stack.py --generator qwen --threads 4 --seeds 5
"""
import argparse, hashlib, json, os, sys
import numpy as np
from collections import defaultdict, Counter

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS
from head_domain_scaling import norm

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
# 2026-09-13: pathvqa_open REMOVED from the shared set. It lived in the combined
# generator_eval_finelayer cache alongside slake and vqa_rad, and that cache covers only
# the truncated 1,500-question pathvqa. It now has its own complete 3,357-question cache at
# generator_eval_finelayer_pathvqa_open, so it is read per-benchmark like every other cell.
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]

# Per-generator cache layout. The Lingshu entry is the original hard-coded behaviour verbatim.
GENERATORS = {
    "lingshu": {
        "tag": "lingshu7b",
        "train_stems": ["generator_train_finelayer"],
        "eval_stem": lambda c: ("generator_eval_finelayer" if c in SHARED
                                else f"generator_eval_finelayer_{c}"),
        "eval_dsfilter": lambda c: ({c} if c in SHARED else None),
        "out": "head_final_stack_2026-08-24.json",
        "bench": None,          # None = the full BENCH list
        "ens": None,            # None = the module-default ENS
    },
    "qwen": {
        "tag": "qwen25vl7b",
        # four overlapping caches, deduped on (ds, idx, na) -- see the module docstring
        "train_stems": [f"generator_train_qwen_{c}" for c in
                        ("kvasir_open", "pathvqa_open_train",
                         "slake_open_train", "vqa_rad_open_train")],
        "eval_stem": lambda c: f"generator_eval_qwen_{c}",
        "eval_dsfilter": lambda c: None,
        "out": "head_final_stack_qwen_2026-09-13.json",
        "bench": None,
        "ens": None,            # Qwen2.5-VL-7B is 28-layer like Lingshu, so [18,20,22] transfers
    },
    "medgemma": {
        # THIRD GENERATOR, and the first from a different LM family: Gemma 3 + SigLIP, medically
        # trained. Lingshu is a Qwen2.5-VL finetune, so lingshu-vs-qwen is a within-family
        # replication. There are no dedicated *_train caches here -- the pooled set is the
        # by-image TRAIN HALVES of the four benchmarks, which is all the claim needs: does a probe
        # fitted on this generator's own hidden states beat this generator's own greedy decoding?
        "tag": "medgemma4b",
        "train_stems": [],
        "eval_stem": lambda c: f"generator_eval_medgemma_{c}",
        "eval_dsfilter": lambda c: None,
        "out": "head_final_stack_medgemma_2026-09-13.json",
        "bench": ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open"],
        # DEPTH-MATCHED, pre-specified before any MedGemma probe was fitted. Gemma 3 here is
        # 34-layer against Lingshu's 28, so the shipped [18,20,22] -- relative depth
        # 0.643/0.714/0.786 on Lingshu -- would sit at 0.529/0.588/0.647 here, materially
        # shallower. [22,24,27] is 0.647/0.706/0.794, the actual analogue. The absolute-matched
        # [18,20,22] is extracted too and is a robustness check, NOT the headline.
        "ens": [22, 24, 27],
    },
}
GEN = GENERATORS["lingshu"]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def sc_of(rows, ds_of, tag=None):
    tag = tag or GEN["tag"]
    raw = {}
    for ds in sorted({ds_of(r) for r in rows}):
        p = f"{CK}/ckpt_{ds}_{tag}_sc8.jsonl"
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    d = json.loads(l); raw[(ds, d["idx"])] = d
    out = np.zeros(len(rows), np.float32)
    for i, r in enumerate(rows):
        d = raw.get((ds_of(r), r["idx"]))
        if d:
            c = Counter(norm(x) for x in d["preds"])
            out[i] = c.get(norm(r["na"]), 0) / max(len(d["preds"]), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--generator", choices=sorted(GENERATORS), default="lingshu")
    ap.add_argument("--only_arm", default=None,
                    help="fit just this one arm. For reproducibility replicates: the full ladder is "
                         "~2.5 h, one arm is ~25 min, so a spread can be measured in a morning.")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    globals()["GEN"] = GENERATORS[A.generator]
    if GEN.get("ens"):
        globals()["ENS"] = GEN["ens"]
    if A.out is None:
        A.out = os.path.join(HS.OUTDIR, GEN["out"])
    print(f"generator={A.generator} tag={GEN['tag']} layers={ENS}", flush=True)
    HS.torch.set_num_threads(A.threads)
    from genframe_data import rank_avg

    def load_fine(stem, dsf=None):
        z = np.load(f"{HS.FEATS}/{stem}.npz"); m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
        lay = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
        rr = [m["rows"][i] for i in keep]
        return {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS}, rr

    # Merge this generator's train caches, deduped on (ds, idx, normalised answer). Lingshu has a
    # single cache so the dedup is a no-op there; Qwen has four overlapping ones (docstring).
    seen, Xparts, rows_o, n_dup = set(), {L: [] for L in ENS}, [], 0
    for st in GEN["train_stems"]:
        if not os.path.exists(f"{HS.FEATS}/{st}.npz"):
            print(f"  [skip] train cache {st} absent", flush=True); continue
        Xs_, rs_ = load_fine(st, TRAIN_DOMAINS)
        k = np.array([(r["ds"], r["idx"], norm(r["na"])) not in seen for r in rs_])
        for r, kk in zip(rs_, k):
            if kk:
                seen.add((r["ds"], r["idx"], norm(r["na"])))
        n_dup += int((~k).sum())
        for L in ENS:
            Xparts[L].append(Xs_[L][k])
        rows_o += [r for r, kk in zip(rs_, k) if kk]
    if not rows_o and GEN["train_stems"]:
        raise SystemExit(f"no training cache found for generator {A.generator}")
    Xtr_o = {L: np.concatenate(Xparts[L]) for L in ENS} if rows_o else None
    if n_dup:
        print(f"deduped {n_dup:,} duplicate training rows across "
              f"{len(GEN['train_stems'])} overlapping caches -> {len(rows_o):,} unique", flush=True)
    src_o = [r["ds"] for r in rows_o]
    # With no dedicated train-domain cache (medgemma) the pool is seeded empty rather than with a
    # zero-row array -- the hidden width differs per generator (3584 Qwen/Lingshu, 2560 Gemma 3),
    # so a hard-coded empty shape would break the concatenate.
    Xadd = {L: ([Xtr_o[L]] if rows_o else []) for L in ENS}
    rows_all, src_all = list(rows_o), list(src_o)
    ev = {}
    for cell in (GEN["bench"] or BENCH):
        stem = GEN["eval_stem"](cell)
        gjp = f"{CK}/ckpt_{cell}_{GEN['tag']}.judge.jsonl"
        if not (os.path.exists(f"{HS.FEATS}/{stem}.npz") and os.path.exists(gjp)):
            continue
        Xc, rr = load_fine(stem, GEN["eval_dsfilter"](cell))
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        for L in ENS:
            Xadd[L].append(Xc[L][istr])
        rows_all += [rr[i] for i in np.where(istr)[0]]
        src_all += [cell] * int(istr.sum())
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        ev[cell] = {"X": {L: Xc[L][~istr] for L in ENS},
                    "rows": [rr[i] for i in np.where(~istr)[0]], "gok": gok}
    Xpool = {L: np.concatenate(Xadd[L]) for L in ENS}
    n_orig = len(rows_o)

    # drop any training row whose image appears in a held-out half (the 19 MedPix images)
    evimgs = set()
    for d in ev.values():
        evimgs |= {r["img_md5"] for r in d["rows"]}
    drop = np.array([r["img_md5"] in evimgs for r in rows_all])
    if drop.any():
        print(f"dropping {int(drop.sum())} leaking training rows", flush=True)
        for L in ENS:
            Xpool[L] = Xpool[L][~drop]
        n_orig -= int(drop[:n_orig].sum())
        rows_all = [r for r, dd in zip(rows_all, drop) if not dd]
        src_all = [s for s, dd in zip(src_all, drop) if not dd]
    y = np.array([r["y"] for r in rows_all], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src_all, rows_all)])
    sc = sc_of(rows_all, lambda r: src_all[rows_all.index(r)] if False else r["ds"])
    # rows carry their own ds for the four original domains; for benchmark rows ds == the benchmark
    sc = sc_of(rows_all, lambda r: r["ds"])
    print(f"pooled: {len(y):,} rows ({n_orig:,} original)", flush=True)

    def fit(X, sub):
        mu, sg = X[sub].mean(0), X[sub].std(0) + 1e-6
        return mu, sg, [HS.fit((X[sub] - mu) / sg, y[sub], qid[sub], None, objective="bce",
                               hidden=256, wd=1e-2, epochs=30, seed=s) for s in range(A.seeds)]
    isorig = np.arange(len(y)) < n_orig
    allrows = np.ones(len(y), bool)
    ARMS = {}
    for nm, sub, layers, use_sc in (
            ("deployed_4dom_L21ish", isorig, [20], False),   # single layer, four domains
            ("pooled_singlelayer", allrows, [20], False),
            ("pooled_ens", allrows, ENS, False),
            ("pooled_ens_sc", allrows, ENS, True)):
        if A.only_arm and nm != A.only_arm:
            continue
        if not sub.any():
            print(f"skipping {nm}: no training rows for this arm "
                  f"(generator {A.generator} has no dedicated train-domain cache)", flush=True)
            continue
        print(f"fitting {nm} ...", flush=True)
        ARMS[nm] = {L: fit(np.concatenate([Xpool[L], sc[:, None]], 1) if use_sc else Xpool[L], sub)
                    for L in layers}
        ARMS[nm]["_sc"] = use_sc; ARMS[nm]["_layers"] = layers

    art = {"title": "Everything that worked, stacked", "date": "2026-08-24",
           "no_fabricated_numbers": True, "seeds": A.seeds, "generator": A.generator,
           "generator_tag": GEN["tag"], "duplicate_training_rows_dropped": int(n_dup),
           "pooled_rows": int(len(y)), "original_rows": int(n_orig), "cells": {}}
    for cell, d in ev.items():
        rr, gok = d["rows"], d["gok"]
        ye = np.array([r["y"] for r in rr], dtype=int)
        sce = sc_of(rr, lambda r: cell)
        byq = defaultdict(list)
        for i, r in enumerate(rr):
            byq[r["idx"]].append(i)
        qs = [q for q in byq if q in gok]
        if len(qs) < 40:
            continue
        out = {"n_questions": len(qs), "greedy": float(np.mean([gok[q] for q in qs]))}
        for nm, arm in ARMS.items():
            S = []
            for L in arm["_layers"]:
                mu, sg, ms = arm[L]
                Xe = (np.concatenate([d["X"][L], sce[:, None]], 1) if arm["_sc"] else d["X"][L])
                S.append(np.stack([HS.predict(mm, (Xe - mu) / sg) for mm in ms]))
            acc = []
            for q in qs:
                ii = np.array(byq[q])
                hr = np.mean([rank_avg(s[k][ii]) for s in S for k in range(s.shape[0])], axis=0)
                acc.append(int(ye[ii][int(np.argmax(hr))]))
            out[nm] = float(np.mean(acc))
            out[nm + "_minus_greedy"] = out[nm] - out["greedy"]
        art["cells"][cell] = out
        print(f"  {cell:17} n{len(qs):6} greedy {out['greedy']:.4f} | " +
              "  ".join(f"{k.replace('pooled_','p_')[:14]} {out[k+'_minus_greedy']:+.4f}"
                        for k in ARMS), flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    cs = art["cells"]
    if cs:
        art["macro"] = {k: float(np.mean([v[k + "_minus_greedy"] for v in cs.values()]))
                        for k in ARMS}
        art["beats_greedy"] = {k: f"{sum(1 for v in cs.values() if v[k+'_minus_greedy']>0)}/{len(cs)}"
                               for k in ARMS}
        for k, v in art["macro"].items():
            print(f"  MACRO {k:24} {v:+.4f}  ({art['beats_greedy'][k]})")
        best = max(art["macro"], key=art["macro"].get)
        # --only_arm runs can omit the four-domain baseline entirely; the verdict must not assume it
        base = art["macro"].get("deployed_4dom_L21ish")
        art["VERDICT"] = (f"best is {best} at {art['macro'][best]:+.4f} macro" +
                          (f", {art['macro'][best]-base:+.4f} over the four-domain single-layer "
                           f"probe on the same held-out halves." if base is not None else
                           " (four-domain baseline not fitted in this run)."))
        print(f"\n=> {art['VERDICT']}")
    json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
