#!/usr/bin/env python3
"""align_followup.py -- the controls that decide how to read align_score.py's two surprises:

  1. the L2-strict head collapsed to 0.597, but L2 also cut the training pool 10693 -> 2753 items.
     SIZE-MATCHED CONTROL: retrain the L1 head on a random 2753-item L1 subsample (5 seeds).
     If L1@2753 also collapses, the L2 drop is a DATA-SIZE effect, not a question-text shortcut.
  2. the pooled win is guardrail-dirty (loses on vqa_rad_open).  Per-set paired-bootstrap CIs.

Also: fuse the incumbent with the TEXT-ONLY head (does the pooled fusion gain need the image at all?)
and report the head/incumbent correlation + pair-oracle headroom.

  CUDA_VISIBLE_DEVICES=0 python3 src/verifier_arch/align_followup.py
  -> merges an "addendum" block into results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json
Run from the repo root.
"""
import os, json
import numpy as np
import importlib.util

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
spec = importlib.util.spec_from_file_location("_as", os.path.join(ROOT, "src/verifier_arch/align_score.py"))
# align_score.py runs its whole analysis on import, so re-implement the few helpers here instead.
CACHE = os.path.join(ROOT, "data/align_cache")
OUTP = os.path.join(ROOT, "results/cascade_methods/artifacts/verifarch_alignment_2026-08-04.json")
ENC = "siglip"
TMPL = {"ans": lambda q, a: a, "qa": lambda q, a: f"{q} {a}",
        "decl": lambda q, a: f"medical image. {q} answer: {a}"}
BEST = "decl"   # chosen on TRAIN by align_score.py

man = json.load(open(os.path.join(CACHE, "manifest.json")))
EVAL, TRAIN = man["eval"], man["train"]
inc = [np.array(r["incumbent"], float) for r in EVAL]

z = np.load(os.path.join(CACHE, f"emb_{ENC}.npz"))
IH = {h: i for i, h in enumerate(z["img_hash"])}
TK = {k: i for i, k in enumerate(z["txt_key"])}
V = z["img_emb"].astype(np.float64); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
T = z["txt_emb"].astype(np.float64); T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9


def tvec(q, a):
    return T[TK[BEST + "\x00" + TMPL[BEST](q, a)[:400]]]


def feats(rows, blind=False):
    X, Y, ITEM = [], [], []
    for n, r in enumerate(rows):
        v = V[IH[r["img"]]] * (0.0 if blind else 1.0)
        for a, lab in zip(r["preds"], r["sl"]):
            t = tvec(r["q"], a)
            X.append(np.concatenate([v, t, v * t, np.abs(v - t)])); Y.append(lab); ITEM.append(n)
    return np.asarray(X, np.float32), np.asarray(Y), np.asarray(ITEM)


def train_mlp(X, Y, Xv, Yv, hid=256, epochs=12, lr=1e-3, wd=1e-4, seed=0):
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = nn.Sequential(nn.Linear(X.shape[1], hid), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hid, 1)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); lf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(Y, dtype=torch.float32, device=dev)
    best, bs_, n = -1, None, len(Xt)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, 512):
            k = perm[i:i + 512]; opt.zero_grad(); lf(m(Xt[k]).squeeze(-1), Yt[k]).backward(); opt.step()
        au = roc_auc_score(Yv, predict(m, Xv))
        if au > best:
            best, bs_ = au, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs_)
    return m, best


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
    return float(np.mean([r["sl"][int(np.argmax(x))] == 1 for r, x in k])), len(k)


def boot_paired(rows, sA, sB, B=2000, seed=0):
    idx = [i for i, r in enumerate(rows) if max(r["sl"]) == 1]
    a = np.array([rows[i]["sl"][int(np.argmax(sA[i]))] == 1 for i in idx], float)
    b = np.array([rows[i]["sl"][int(np.argmax(sB[i]))] == 1 for i in idx], float)
    rng = np.random.default_rng(seed); n = len(idx)
    d = np.array([a[k].mean() - b[k].mean() for k in (rng.integers(0, n, n) for _ in range(B))])
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), n


Xe, Ye, ITEMe = feats(EVAL)
Xeb, _, _ = feats(EVAL, blind=True)
add = {}

# ---- 1. size-matched L1 control at the L2 training size
strict = {ds: set(json.load(open(os.path.join(ROOT, f"data/disjoint_split/strict_idx_{ds}.json"))))
          for ds in ("slake_open_train", "vqa_rad_open_train", "pathvqa_open_train")}
TR2 = [r for r in TRAIN if r["idx"] in strict[r["ds"]]]
imgs2 = sorted({r["img"] for r in TR2}); np.random.default_rng(3).shuffle(imgs2)
va2 = set(imgs2[:max(1, len(imgs2) // 5)])
n_l2_train = len([r for r in TR2 if r["img"] not in va2])
print(f"L2 train-side items = {n_l2_train}", flush=True)

sizes = {"L1_full": None, "L1_sizematched_to_L2": n_l2_train}
add["size_matched_control"] = {"n_l2_train_items": n_l2_train, "runs": {}}
for tag, cap in sizes.items():
    effs, persets = [], []
    for seed in range(5):
        imgs = sorted({r["img"] for r in TRAIN}); np.random.default_rng(3).shuffle(imgs)
        va = set(imgs[:max(1, len(imgs) // 5)])
        tr = [r for r in TRAIN if r["img"] not in va]
        vr = [r for r in TRAIN if r["img"] in va]
        if cap is not None:
            rs = np.random.default_rng(100 + seed); k = rs.permutation(len(tr))[:cap]
            tr = [tr[i] for i in k]
        Xt, Yt, _ = feats(tr); ok = Yt >= 0; Xt, Yt = Xt[ok], Yt[ok]
        Xv, Yv, _ = feats(vr); ok = Yv >= 0; Xv, Yv = Xv[ok], Yv[ok]
        mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
        m, _ = train_mlp((Xt - mu) / sd, Yt, (Xv - mu) / sd, Yv, seed=seed)
        s = to_items(predict(m, (Xe - mu) / sd), ITEMe, len(EVAL))
        effs.append(sel_eff(EVAL, s)[0])
        persets.append({ds: sel_eff([r for r in EVAL if r["ds"] == ds],
                                    [x for r, x in zip(EVAL, s) if r["ds"] == ds])[0]
                        for ds in sorted({r["ds"] for r in EVAL})})
        del m
    add["size_matched_control"]["runs"][tag] = {
        "n_train_items": cap if cap else len(tr), "sel_eff_mean": float(np.mean(effs)),
        "sel_eff_sd": float(np.std(effs)), "sel_eff_per_seed": effs, "sel_eff_per_set_seed0": persets[0]}
    print(f"  {tag}: sel_eff {np.mean(effs):.4f} +- {np.std(effs):.4f}", flush=True)

# ---- 2. per-set paired-bootstrap deltas for the headline (seed 0) head
imgs = sorted({r["img"] for r in TRAIN}); np.random.default_rng(3).shuffle(imgs)
va = set(imgs[:max(1, len(imgs) // 5)])
tr = [r for r in TRAIN if r["img"] not in va]; vr = [r for r in TRAIN if r["img"] in va]
Xt, Yt, _ = feats(tr); ok = Yt >= 0; Xt, Yt = Xt[ok], Yt[ok]
Xv, Yv, _ = feats(vr); ok = Yv >= 0; Xv, Yv = Xv[ok], Yv[ok]
mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
m, _ = train_mlp((Xt - mu) / sd, Yt, (Xv - mu) / sd, Yv, seed=0)
s_head = to_items(predict(m, (Xe - mu) / sd), ITEMe, len(EVAL))
Xtb, Ytb, _ = feats(tr, blind=True); ok = Ytb >= 0; Xtb, Ytb = Xtb[ok], Ytb[ok]
Xvb, Yvb, _ = feats(vr, blind=True); ok = Yvb >= 0; Xvb, Yvb = Xvb[ok], Yvb[ok]
mub, sdb = Xtb.mean(0, keepdims=True), Xtb.std(0, keepdims=True) + 1e-6
mb, _ = train_mlp((Xtb - mub) / sdb, Ytb, (Xvb - mub) / sdb, Yvb, seed=0)
s_blind = to_items(predict(mb, (Xeb - mub) / sdb), ITEMe, len(EVAL))

add["per_set_paired_delta_vs_incumbent"] = {}
for ds in sorted({r["ds"] for r in EVAL}):
    rs_ = [r for r in EVAL if r["ds"] == ds]
    a = [x for r, x in zip(EVAL, s_head) if r["ds"] == ds]
    b = [x for r, x in zip(EVAL, inc) if r["ds"] == ds]
    d, lo, hi, n = boot_paired(rs_, a, b)
    add["per_set_paired_delta_vs_incumbent"][ds] = {"n_recoverable": n, "delta": d, "ci95": [lo, hi]}
    print(f"  {ds}: d={d:+.4f} [{lo:+.4f},{hi:+.4f}] (n_rec={n})", flush=True)

# ---- 3. does the DIAGNOSTIC fusion gain need the image?
from sklearn.linear_model import LogisticRegression
Yc = np.concatenate([r["sl"] for r in EVAL])
ITEMc = np.concatenate([[i] * len(r["preds"]) for i, r in enumerate(EVAL)])
folds = np.array([int(EVAL[i]["img"][:8], 16) % 5 for i in ITEMc])
add["fusion_ablation_DIAGNOSTIC_ONLY"] = {}
for tag, cols in (("incumbent+head(image)", [np.concatenate(inc), np.concatenate(s_head)]),
                  ("incumbent+head(text-only)", [np.concatenate(inc), np.concatenate(s_blind)])):
    Z = np.stack(cols, 1); pf = np.zeros(len(Z))
    for f in range(5):
        trm = folds != f
        pf[~trm] = LogisticRegression(max_iter=2000).fit(Z[trm], Yc[trm]).predict_proba(Z[~trm])[:, 1]
    sf = to_items(pf, ITEMc, len(EVAL))
    d, lo, hi, _ = boot_paired(EVAL, sf, inc)
    add["fusion_ablation_DIAGNOSTIC_ONLY"][tag] = {"sel_eff": sel_eff(EVAL, sf)[0],
                                                   "delta_vs_incumbent": d, "delta_ci95": [lo, hi]}
    print(f"  fusion[{tag}]: sel_eff={sel_eff(EVAL, sf)[0]:.4f} d={d:+.4f} [{lo:+.4f},{hi:+.4f}]", flush=True)

# ---- 4. decorrelation + pair-oracle headroom vs the incumbent
po = float(np.mean([1.0 if (r["sl"][int(np.argmax(a))] == 1 or r["sl"][int(np.argmax(b))] == 1) else 0.0
                    for r, a, b in zip(EVAL, s_head, inc) if max(r["sl"]) == 1]))
agree = float(np.mean([int(np.argmax(a)) == int(np.argmax(b)) for a, b in zip(s_head, inc)]))
add["decorrelation"] = {"pearson_r_candidate_scores": float(np.corrcoef(np.concatenate(s_head),
                                                                       np.concatenate(inc))[0, 1]),
                        "argmax_agreement_rate": agree, "pair_oracle_sel_eff": po,
                        "incumbent_sel_eff": sel_eff(EVAL, inc)[0], "head_sel_eff": sel_eff(EVAL, s_head)[0]}
print(f"  pair-oracle(head, incumbent)={po:.4f}  argmax agreement={agree:.4f}", flush=True)

res = json.load(open(OUTP))
res["addendum_followup"] = add
json.dump(res, open(OUTP, "w"), indent=1)
print("merged addendum into", OUTP)
