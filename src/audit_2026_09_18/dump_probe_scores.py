#!/usr/bin/env python3
"""Dump the frozen 24-probe ensemble's per-candidate scores on the Lingshu held-out halves (read-only, CPU).

Writes {cell: [{idx, img, ans, y, p_mean (mean sigmoid over 24 heads), rank_mean (mean within-question
rank_avg over 24 heads), n_tok}]} so later analyses (pseudo-label precision, weighted vote, greedy-anchored
selection, adaptive-N) need not touch the 20 GB of feature caches again."""
import hashlib, json, os, sys
import numpy as np
import torch

torch.set_num_threads(4)
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import head_sweep as HS                      # noqa: E402
from genframe_data import rank_avg           # noqa: E402

FE, PD = f"{ROOT}/feats_hidden", f"{ROOT}/ckpts/train/genframe_head_pooled_ens_v2"
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


H = {}
for L in ENS:
    st = np.load(f"{PD}/standardizer_L{L}.npz"); ms = []
    for s in range(8):
        m = HS.MLP(3584, 256, 1, 0.0, False)
        m.load_state_dict(torch.load(f"{PD}/head_L{L}_seed{s}.pt", map_location="cpu")); m.eval(); ms.append(m)
    H[L] = (st["mu"], st["sd"], ms)

out = {}
for cell in BENCH:
    stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
    z = np.load(f"{FE}/{stem}.npz"); m = json.load(open(f"{FE}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0
            and (cell not in SHARED or r.get("ds") == cell) and half(r["img_md5"]) == 0]
    rows = [m["rows"][i] for i in keep]
    Hs = z["h_span"]
    logits = []
    for L in ENS:
        mu, sd, ms = H[L]
        X = (Hs[keep, lay.index(L)].astype(np.float32) - mu) / sd
        logits += [HS.predict(mm, X) for mm in ms]
    logits = np.stack(logits)                       # [24, rows]
    p = 1 / (1 + np.exp(-logits))
    byq = {}
    for i, r in enumerate(rows):
        byq.setdefault(r["idx"], []).append(i)
    rk = np.zeros(len(rows))
    for ii in byq.values():
        ii = np.array(ii); rk[ii] = np.mean([rank_avg(l[ii]) for l in logits], axis=0)
    out[cell] = [{"idx": r["idx"], "img": r["img_md5"], "ans": r["na"], "y": int(r["y"]), "n_tok": r.get("n_tok"),
                  "p_mean": float(p[:, i].mean()), "p_sd": float(p[:, i].std()), "rank_mean": float(rk[i])}
                 for i, r in enumerate(rows)]
    print(cell, len(rows), "rows", len(byq), "questions", flush=True)
    del Hs, z
json.dump(out, open(sys.argv[1], "w"))
print("wrote", sys.argv[1])
