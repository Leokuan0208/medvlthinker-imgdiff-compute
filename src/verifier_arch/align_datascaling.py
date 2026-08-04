#!/usr/bin/env python3
"""align_datascaling.py -- the fairness control the headline needs.

The trained alignment head sees 85,544 labelled (answer, judge_ok) examples from 10,693 image-disjoint
train items.  The incumbent LoRA verifier (ckpts/train/lora_verifier_disjoint) was trained on 10,364
such examples (its composition was matched to the contaminated pooled4 verifier; see
run_lora_verifier_disjoint.py:CONTAMINATED_QUOTA).  So the pooled win could be a DATA-VOLUME effect
rather than an architecture effect.  This traces sel_eff vs training-example count on the same features,
5 seeds per point, and marks the incumbent's own budget.

  CUDA_VISIBLE_DEVICES=0 python3 src/verifier_arch/align_datascaling.py
  -> merges "addendum_data_scaling" into results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json
"""
import os, json
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CACHE = os.path.join(ROOT, "data/align_cache")
OUTP = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json")
BEST = "decl"
TMPL = {"decl": lambda q, a: f"medical image. {q} answer: {a}"}

man = json.load(open(os.path.join(CACHE, "manifest.json")))
EVAL, TRAIN = man["eval"], man["train"]
inc = [np.array(r["incumbent"], float) for r in EVAL]
z = np.load(os.path.join(CACHE, "emb_siglip.npz"))
IH = {h: i for i, h in enumerate(z["img_hash"])}
TK = {k: i for i, k in enumerate(z["txt_key"])}
V = z["img_emb"].astype(np.float64); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
T = z["txt_emb"].astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9


def feats(rows):
    X, Y, ITEM = [], [], []
    for n, r in enumerate(rows):
        v = V[IH[r["img"]]]
        for a, lab in zip(r["preds"], r["sl"]):
            t = T[TK[BEST + "\x00" + TMPL[BEST](r["q"], a)[:400]]]
            X.append(np.concatenate([v, t, v * t, np.abs(v - t)])); Y.append(lab); ITEM.append(n)
    return np.asarray(X, np.float32), np.asarray(Y), np.asarray(ITEM)


def train_mlp(X, Y, Xv, Yv, seed=0):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); dev = "cuda"
    m = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4); lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    best, bs_, n = -1, None, len(Xt)
    for _ in range(12):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            k = perm[i:i + 512]; opt.zero_grad(); lf(m(Xt[k]).squeeze(-1), Yt[k]).backward(); opt.step()
        au = roc_auc_score(Yv, predict(m, Xv))
        if au > best:
            best, bs_ = au, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs_); return m


def predict(m, X):
    import torch
    dev = next(m.parameters()).device; m.eval(); o = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            o.append(torch.sigmoid(m(torch.tensor(X[i:i + 4096], device=dev)).squeeze(-1)).cpu().numpy())
    return np.concatenate(o)


def to_items(p, ITEM, n):
    out = [[] for _ in range(n)]
    for v, i in zip(p, ITEM):
        out[i].append(v)
    return [np.array(x) for x in out]


def sel_eff(rows, s):
    k = [(r, x) for r, x in zip(rows, s) if max(r["sl"]) == 1]
    return float(np.mean([r["sl"][int(np.argmax(x))] == 1 for r, x in k]))


Xe, _, ITEMe = feats(EVAL)
imgs = sorted({r["img"] for r in TRAIN}); np.random.default_rng(3).shuffle(imgs)
va = set(imgs[:max(1, len(imgs) // 5)])
tr_all = [r for r in TRAIN if r["img"] not in va]
vr = [r for r in TRAIN if r["img"] in va]
Xv, Yv, _ = feats(vr); ok = Yv >= 0; Xv, Yv = Xv[ok], Yv[ok]

out = {"incumbent_training_examples": 10364, "incumbent_sel_eff": sel_eff(EVAL, inc), "points": []}
for n_ex in [5000, 10364, 20000, 40000, 85544]:
    effs, real = [], None
    for seed in range(5):
        rs = np.random.default_rng(200 + seed)
        order = rs.permutation(len(tr_all))
        # take items until the example budget is reached
        sub, tot = [], 0
        for i in order:
            if tot >= n_ex: break
            sub.append(tr_all[i]); tot += sum(1 for x in tr_all[i]["sl"] if x >= 0)
        Xt, Yt, _ = feats(sub); ok = Yt >= 0; Xt, Yt = Xt[ok], Yt[ok]
        real = len(Xt)
        mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
        m = train_mlp((Xt - mu) / sd, Yt, (Xv - mu) / sd, Yv, seed=seed)
        effs.append(sel_eff(EVAL, to_items(predict(m, (Xe - mu) / sd), ITEMe, len(EVAL))))
        del m
    out["points"].append({"target_examples": n_ex, "actual_examples": int(real), "n_items": len(sub),
                          "sel_eff_mean": float(np.mean(effs)), "sel_eff_sd": float(np.std(effs)),
                          "sel_eff_per_seed": effs,
                          "delta_vs_incumbent_mean": float(np.mean(effs)) - out["incumbent_sel_eff"]})
    print(f"  {n_ex:>6} ex (actual {real}, {len(sub)} items): sel_eff {np.mean(effs):.4f} "
          f"+- {np.std(effs):.4f}  d={np.mean(effs)-out['incumbent_sel_eff']:+.4f}", flush=True)

res = json.load(open(OUTP)); res["addendum_data_scaling"] = out
json.dump(res, open(OUTP, "w"), indent=1)
print("merged addendum_data_scaling into", OUTP)
