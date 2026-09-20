#!/usr/bin/env python3
"""PILOT (2026-09-20) -- can a LABEL-FREE linear map close the cross-lineage transfer gap?

pilot_cross_model_transfer.py: the frozen LINGSHU probes applied zero-shot to QWEN2.5-VL-7B hidden states give
+0.0451 macro over Qwen greedy, against +0.0820 for a Qwen-native probe. Lingshu is a fine-tune of Qwen2.5-VL-7B,
so the two residual streams are related. Here a ridge map  Z_qwen -> Z_lingshu  is fitted WITHOUT ANY LABEL on
paired inputs: rows where both generators produced the SAME normalised answer to the SAME question (so the two
models were teacher-forced on identical image+question+answer text). Pairs come from the by-image TRAIN halves
only; the ridge strength is chosen on a 10% split of those pairs by alignment error (label-free). The mapped
held-out Qwen states are then scored by the frozen Lingshu probes.

  arm A      zero-shot, Lingshu standardizer                (reproduces the earlier pilot)
  arm B      zero-shot, Qwen's own train-half standardizer
  arm RIDGE  B followed by the fitted linear map
  ref        Qwen-native pooled_ens, read from head_final_stack_qwen_2026-09-13.json (not refit)

Read-only, CPU, 4 threads. Judge currency (labels of record) + the image-clustered bootstrap."""
import hashlib, json, os, sys, time
import numpy as np
import torch

torch.set_num_threads(4)
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import head_sweep as HS                      # noqa: E402
from genframe_data import rank_avg           # noqa: E402
FE, CK = f"{ROOT}/feats_hidden", f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"
PD = f"{ROOT}/ckpts/train/genframe_head_pooled_ens_v2"
ART = f"{ROOT}/results/cascade_methods/artifacts"
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]
rng = np.random.default_rng(20260920); NB = 10000


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load(stem, dsf=None):
    z = np.load(f"{FE}/{stem}.npz"); m = json.load(open(f"{FE}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    H = z["h_span"]
    return {L: H[keep, lay.index(L)].astype(np.float32) for L in ENS}, [m["rows"][i] for i in keep]


def key(s):
    """Pairing key. Lingshu ends answers with a period ("ct."), Qwen does not ("ct"), so an exact match pairs
    almost nothing. The two teacher-forced inputs then differ by that one trailing token -- noise for the map, noted."""
    return " ".join(str(s).lower().split()).rstrip(".!?, ")


def boot(res, a, b):
    ks = {}
    for r in res:
        ks.setdefault(r[0], []).append(r[a] - r[b])
    s = np.array([sum(v) for v in ks.values()], float); n = np.array([len(v) for v in ks.values()], float)
    ix = rng.integers(0, len(s), size=(NB, len(s))); bb = s[ix].sum(1) / n[ix].sum(1)
    return float(s.sum() / n.sum()), [float(np.quantile(bb, .025)), float(np.quantile(bb, .975))]


t0 = time.time()
H = {}
for L in ENS:
    st = np.load(f"{PD}/standardizer_L{L}.npz"); ms = []
    for s in range(8):
        m = HS.MLP(3584, 256, 1, 0.0, False)
        m.load_state_dict(torch.load(f"{PD}/head_L{L}_seed{s}.pt", map_location="cpu")); m.eval(); ms.append(m)
    H[L] = (st["mu"].astype(np.float32), st["sd"].astype(np.float32), ms)

pairs = {L: ([], []) for L in ENS}; qtr = {L: [] for L in ENS}; held = {}
for cell in BENCH:
    Xl, rl = load("generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}",
                  {cell} if cell in SHARED else None)
    Xq, rq = load(f"generator_eval_qwen_{cell}")
    il = {(r["idx"], key(r["na"])): i for i, r in enumerate(rl) if half(r["img_md5"]) == 1}
    iq = [(i, il[(r["idx"], key(r["na"]))]) for i, r in enumerate(rq) if half(r["img_md5"]) == 1 and (r["idx"], key(r["na"])) in il]
    trq = np.array([half(r["img_md5"]) == 1 for r in rq])
    for L in ENS:
        if iq:
            pairs[L][0].append(Xq[L][[a for a, _ in iq]]); pairs[L][1].append(Xl[L][[b for _, b in iq]])
        qtr[L].append(Xq[L][trq])
    g = {}
    for l in open(f"{CK}/ckpt_{cell}_qwen25vl7b.judge.jsonl"):
        if l.strip():
            d = json.loads(l); g[d["idx"]] = int(d["judge_ok"])
    held[cell] = ({L: Xq[L][~trq] for L in ENS}, [rq[i] for i in np.where(~trq)[0]], g)
    print(f"{cell:17} train-half paired rows {len(iq):6} (of {int(trq.sum())} qwen train rows) ({time.time()-t0:.0f}s)", flush=True)
    del Xl, Xq

W, stq, info = {}, {}, {}
for L in ENS:
    Aq = np.concatenate(pairs[L][0]); Al = np.concatenate(pairs[L][1]); allq = np.concatenate(qtr[L])
    muq, sdq = allq.mean(0), allq.std(0) + 1e-6
    mul, sdl, _ = H[L]
    Zq, Zl = (Aq - muq) / sdq, (Al - mul) / sdl
    perm = rng.permutation(len(Zq)); k = len(Zq) // 10; va, tr = perm[:k], perm[k:]
    G = (Zq[tr].T @ Zq[tr]).astype(np.float64); C = (Zq[tr].T @ Zl[tr]).astype(np.float64)
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4):
        w = np.linalg.solve(G + lam * np.eye(G.shape[0]), C).astype(np.float32)
        err = float(((Zq[va] @ w - Zl[va]) ** 2).mean()); base = float(((Zq[va] - Zl[va]) ** 2).mean())
        if best is None or err < best[1]:
            best = (lam, err, w, base)
    W[L], stq[L] = best[2], (muq, sdq)
    info[L] = {"pairs": int(len(Zq)), "ridge_lambda": best[0], "val_mse_mapped": best[1], "val_mse_identity": best[3]}
    print(f"layer {L}: pairs {len(Zq):,} lambda {best[0]:g} val-MSE mapped {best[1]:.4f} vs identity {best[3]:.4f} ({time.time()-t0:.0f}s)", flush=True)

ref = json.load(open(f"{ART}/head_final_stack_qwen_2026-09-13.json"))
out = {"title": "PILOT: label-free linear alignment Qwen2.5-VL-7B -> Lingshu-7B hidden space, scored by frozen Lingshu probes",
       "date": "2026-09-20", "no_fabricated_numbers": True, "currency": "judge of record (MedVLThinker-32B)",
       "alignment": {str(L): info[L] for L in ENS}, "pairing": "same (benchmark, idx, answer up to case/whitespace/trailing punctuation) in both generators' caches, TRAIN halves only, no labels",
       "threads": 4, "bootstrap": f"{NB}, clustered by img_md5", "cells": {}}
for cell in BENCH:
    X, rows, g = held[cell]
    y = np.array([r["y"] for r in rows], int)
    byq = {}
    for i, r in enumerate(rows):
        byq.setdefault(r["idx"], []).append(i)
    S = {}
    for arm in ("A", "B", "RIDGE"):
        sc = []
        for L in ENS:
            mul, sdl, ms = H[L]; muq, sdq = stq[L]
            Z = (X[L] - mul) / sdl if arm == "A" else (X[L] - muq) / sdq
            if arm == "RIDGE":
                Z = Z @ W[L]
            sc.append(np.stack([HS.predict(m, Z.astype(np.float32)) for m in ms]))
        S[arm] = sc
    res = []
    for q, ii in byq.items():
        if q not in g:
            continue
        ii = np.array(ii); row = [rows[ii[0]]["img_md5"], g[q]]
        for arm in ("A", "B", "RIDGE"):
            hr = np.mean([rank_avg(s[k][ii]) for s in S[arm] for k in range(s.shape[0])], axis=0)
            row.append(int(y[ii][int(np.argmax(hr))]))
        res.append(row)
    c = {"n": len(res), "qwen_greedy": float(np.mean([r[1] for r in res]))}
    for j, arm in enumerate(("A_zeroshot", "B_restandardised", "RIDGE_aligned")):
        c[f"{arm}_minus_greedy"], c[f"{arm}_ci"] = boot(res, 2 + j, 1)
    c["RIDGE_minus_A"], c["RIDGE_minus_A_ci"] = boot(res, 4, 2)
    c["ref_qwen_native_pooled_ens_minus_greedy"] = ref["cells"].get(cell, {}).get("pooled_ens_minus_greedy")
    out["cells"][cell] = c
    print(f"{cell:17} n{c['n']:5} A {c['A_zeroshot_minus_greedy']:+.4f} B {c['B_restandardised_minus_greedy']:+.4f} "
          f"RIDGE {c['RIDGE_aligned_minus_greedy']:+.4f} {np.round(c['RIDGE_aligned_ci'], 4).tolist()} | native {c['ref_qwen_native_pooled_ens_minus_greedy']:+.4f}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
ks = ("A_zeroshot_minus_greedy", "B_restandardised_minus_greedy", "RIDGE_aligned_minus_greedy", "RIDGE_minus_A",
      "ref_qwen_native_pooled_ens_minus_greedy")
out["macro"] = {k: float(np.mean([v[k] for v in out["cells"].values()])) for k in ks}
out["seconds"] = round(time.time() - t0, 1)
print(json.dumps(out["macro"], indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
