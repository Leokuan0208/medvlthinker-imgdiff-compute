#!/usr/bin/env python3
"""AUDIT 2026-09-20 -- does the probe learn CORRECTNESS or does it DISTIL ITS JUDGE?  A 2x2.

  probe_A  trained on judge-A labels (MedVLThinker-32B, the labels of record)
  probe_B  trained on judge-B labels (MedGemma-27B-it, cross-family), SAME rows, SAME recipe, SAME seeds
  each probe's held-out picks are graded by judge A, judge B and the project's lenient exact match.

If the probe reads correctness, the off-diagonal cells (A-trained graded by B, B-trained graded by A) keep most
of the diagonal gain. If it distils judge idiosyncrasy, each probe looks best under its own judge.

Recipe = the shipped one (head_sweep.fit: Linear-256-GELU-Linear, BCE, AdamW 1e-3, wd 1e-2, 30 epochs),
single layer 20 (the 'pooled_singlelayer' arm, +0.0729 vs the 3-layer ensemble's +0.0736), --seeds heads each.
CPU, 4 threads, read-only on the repo. Judge currency A reproduces head_final_stack's pooled_singlelayer arm.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--layer", type=int, default=20)
ap.add_argument("--out", required=True)
A_ = ap.parse_args()
torch.set_num_threads(4)
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import head_sweep as HS                      # noqa: E402
from genframe_data import rank_avg           # noqa: E402
T = "/data/dan/audit_2026-09-18/tmp"
FE = f"{ROOT}/feats_hidden"
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
L = A_.layer
rng = np.random.default_rng(20260920); NB = 10000


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    z = np.load(f"{FE}/{stem}.npz"); m = json.load(open(f"{FE}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    return z["h_span"][keep, lay.index(L)].astype(np.float32), [m["rows"][i] for i in keep]


def read_labels(pred, jud):
    lab = {}
    for l in open(jud):
        if l.strip():
            d = json.loads(l); lab.setdefault(d["idx"], int(d["judge_ok"]))
    return lab


t0 = time.time()
# ---- assemble the training pool exactly as build_xjudge_train_preds.py / head_final_stack.py do
X0, r0 = load("generator_train_finelayer", TRAIN_DOMAINS)
Xp, rows = [X0], [(r["ds"], r) for r in r0]
ev, evimgs = {}, set()
for cell in BENCH:
    stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
    Xc, rr = load(stem, {cell} if cell in SHARED else None)
    istr = np.array([half(r["img_md5"]) == 1 for r in rr])
    Xp.append(Xc[istr]); rows += [(cell, rr[i]) for i in np.where(istr)[0]]
    ev[cell] = (Xc[~istr], [rr[i] for i in np.where(~istr)[0]])
    evimgs |= {rr[i]["img_md5"] for i in np.where(~istr)[0]}
    print(f"loaded {cell}: train {int(istr.sum())} held-out {int((~istr).sum())} ({time.time()-t0:.0f}s)", flush=True)
X = np.concatenate(Xp); del Xp
keep = np.array([r["img_md5"] not in evimgs for _, r in rows])
X = X[keep]; rows = [x for x, k in zip(rows, keep) if k]
labT = read_labels(None, f"{T}/me/xjudge/train_lingshu7b_medgemma27b.judge.jsonl")
ids = [json.loads(l)["idx"] for l in open(f"{T}/me/xjudge/train_lingshu7b_medgemma27b.jsonl")]
assert len(ids) == len(rows), (len(ids), len(rows))
for k, ((ds, r), i) in enumerate(zip(rows, ids)):
    assert i == f"{ds}|{r['idx']}|{k}", (i, ds, r["idx"], k)
hasB = np.array([i in labT for i in ids])
yA = np.array([r["y"] for _, r in rows], np.float32)
yB = np.array([labT.get(i, 0) for i in ids], np.float32)
print(f"training rows {len(rows):,}; judge-B labels present for {int(hasB.sum()):,}; "
      f"pos rate A {yA.mean():.4f} B {yB[hasB].mean():.4f}; agreement {(yA[hasB]==yB[hasB]).mean():.4f}", flush=True)
X, yA, yB = X[hasB], yA[hasB], yB[hasB]
qid = np.array([f"{ds}|{r['idx']}" for (ds, r), h in zip(rows, hasB) if h])
mu, sg = X.mean(0), X.std(0) + 1e-6
Xs = (X - mu) / sg
P = {}
for nm, y in (("A", yA), ("B", yB)):
    P[nm] = [HS.fit(Xs, y, qid, None, objective="bce", hidden=256, wd=1e-2, epochs=30, seed=s) for s in range(A_.seeds)]
    print(f"fitted probe_{nm} x{A_.seeds} ({time.time()-t0:.0f}s)", flush=True)
del X, Xs

# ---- held-out: labels under A (meta y), B (held-out judge file), lenient EM + greedy (em-rescore cache)
labH = read_labels(None, f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.judge.jsonl")
a2i = {}
for l in open(f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"):
    d = json.loads(l); c, i, _ = d["idx"].split("|"); a2i[(c, i, d["modal_pred"])] = d["idx"]
C = json.load(open(f"{T}/em-rescore/em_rescore_per_question_cache.json"))["records"]


def boot(imgs, v):
    ks = {}
    for im, x in zip(imgs, v):
        ks.setdefault(im, []).append(x)
    s = np.array([sum(x) for x in ks.values()], float); n = np.array([len(x) for x in ks.values()], float)
    ix = rng.integers(0, len(s), size=(NB, len(s))); b = s[ix].sum(1) / n[ix].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


out = {"title": "2x2: probe trained on judge A vs judge B labels, graded by A, B and lenient EM", "date": "2026-09-20",
       "no_fabricated_numbers": True, "judge_A": "MedVLThinker-32B-RL_m23k (labels of record)",
       "judge_B": "google/medgemma-27b-it (same run_judge.py prompt/decoding)", "layer": L, "seeds": A_.seeds,
       "recipe": "head_sweep.fit bce hidden=256 wd=1e-2 epochs=30; standardizer on train rows; rank_avg over seeds; "
                 "first-index tie-break; 4 threads", "train_rows": int(len(yA)),
       "train_label_pos_rate": {"A": float(yA.mean()), "B": float(yB.mean())},
       "train_label_agreement_AB": float((yA == yB).mean()), "cells": {}}
for cell in BENCH:
    Xe, rr = ev[cell]
    cache = {r["idx"]: r for r in C[cell]}
    Xe = (Xe - mu) / sg
    S = {nm: np.stack([HS.predict(m, Xe) for m in P[nm]]) for nm in P}
    byq = {}
    for i, r in enumerate(rr):
        byq.setdefault(r["idx"], []).append(i)
    rec = []
    for q, ii in byq.items():
        cq = cache.get(q)
        if cq is None:
            continue
        em = {c["ans"]: c["em"] for c in cq["cands"]}
        gB = labH.get(a2i.get((cell, str(q), cq["greedy"]["ans"])))
        bl = [labH.get(a2i.get((cell, str(q), rr[i].get("ans")))) for i in ii]
        if gB is None or any(b is None for b in bl) or any(rr[i].get("ans") not in em for i in ii):
            continue
        ii = np.array(ii); row = {"img": rr[ii[0]]["img_md5"], "g": (cq["greedy"]["y"], gB, cq["greedy"]["em"])}
        for nm in P:
            k = int(np.argmax(np.mean([rank_avg(s[ii]) for s in S[nm]], axis=0)))
            row[nm] = (int(rr[ii[k]]["y"]), bl[k], em[rr[ii[k]]["ans"]])
        rec.append(row)
    imgs = [r["img"] for r in rec]; c = {"n": len(rec), "n_questions_in_cache": len(cache)}
    for nm in P:
        for j, cur in enumerate(("A", "B", "EM")):
            d = np.array([r[nm][j] - r["g"][j] for r in rec], float)
            c[f"probe_{nm}_graded_{cur}"], c[f"probe_{nm}_graded_{cur}_ci"] = boot(imgs, d)
    c["greedy"] = {cur: float(np.mean([r["g"][j] for r in rec])) for j, cur in enumerate(("A", "B", "EM"))}
    c["same_pick_rate_A_vs_B_probe"] = float(np.mean([r["A"] == r["B"] for r in rec]))
    out["cells"][cell] = c
    print(f"{cell:17} n{c['n']:5} | probe_A: A {c['probe_A_graded_A']:+.4f} B {c['probe_A_graded_B']:+.4f} EM {c['probe_A_graded_EM']:+.4f}"
          f" | probe_B: A {c['probe_B_graded_A']:+.4f} B {c['probe_B_graded_B']:+.4f} EM {c['probe_B_graded_EM']:+.4f}", flush=True)
    json.dump(out, open(A_.out, "w"), indent=1)
ks = [k for k in out["cells"][BENCH[0]] if k.startswith("probe_") and not k.endswith("_ci")]
out["macro"] = {k: float(np.mean([v[k] for v in out["cells"].values()])) for k in ks}
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out["macro"], indent=1))
json.dump(out, open(A_.out, "w"), indent=1)
