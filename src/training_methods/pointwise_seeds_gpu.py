#!/usr/bin/env python3
"""pointwise_seeds_gpu.py -- the pointwise bar at >=10 seeds, on GPU.

Same architecture, objective, features, standardization and CV-selected hyperparameters as the
published generator-frame pointwise head (fit_hidden_head.py: L21 / span / Bradley-Terry over
within-question (correct, incorrect) pairs / hidden 256 / wd 1e-2 / 30 epochs / setrel 0), moved
to GPU so 12 seeds cost minutes instead of hours on a contended box.

Numerics caveat, stated because it is load-bearing: floating-point reduction order differs
between the CPU and GPU kernels (and, on CPU, between thread counts -- measured here:
torch.set_num_threads(8) gives sel_eff 0.800409 for the SAME seed-0 config that gives 0.795640
at the published thread count). So a GPU seed 0 is a different draw from the same distribution,
not a failed reproduction. The bit-exact reproduction of the published cell is done separately
(pointwise_seeds.py, CPU) and the incumbent null test is bit-exact regardless.

  python3 src/training_methods/pointwise_seeds_gpu.py --seeds 12
"""
import argparse, json, os, sys, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genframe_data as G
import pairhead_lib as P

CFG = {"layer": 21, "pooling": "span", "objective": "bt", "hidden": 256, "wd": 1e-2, "epochs": 30}


class Head(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.f = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
                  if hidden else nn.Linear(d, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def fit_bt(X, y, groups, cfg, seed):
    """Bradley-Terry over within-question (pos, neg) pairs -- fit_hidden_head's 'bt' objective,
    batched by group exactly the same way (64 groups per step, softplus(-(s_pos - s_neg)))."""
    torch.manual_seed(seed)
    m = Head(X.shape[1], cfg["hidden"]).to(P.DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=cfg["wd"])
    keep = [g for g in groups if y[g].sum() > 0 and (1 - y[g]).sum() > 0]
    L = max(len(g) for g in keep)
    idx = np.zeros((len(keep), L), dtype=np.int64)
    msk = np.zeros((len(keep), L), dtype=np.float32)
    for k, g in enumerate(keep):
        idx[k, :len(g)] = g
        msk[k, :len(g)] = 1.0
    idx = torch.tensor(idx, device=P.DEV)
    msk = torch.tensor(msk, device=P.DEV)
    yt = torch.tensor(y, device=P.DEV)
    NG, gb = idx.shape[0], 64
    gen = torch.Generator(); gen.manual_seed(seed)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(NG, generator=gen).to(P.DEV)
        for i in range(0, NG, gb):
            j = perm[i:i + gb]
            gi, gm = idx[j], msk[j]
            s = m(X[gi.reshape(-1)]).reshape(gi.shape)
            yy = yt[gi.reshape(-1)].reshape(gi.shape) * gm
            s = s.masked_fill(gm == 0, -1e9)
            pm, nm = yy.unsqueeze(2), ((1 - yy) * gm).unsqueeze(1)
            d = s.unsqueeze(2) - s.unsqueeze(1)
            w = pm * nm
            loss = ((nn.functional.softplus(-d) * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="data/verifarch/pointwise_seeds_gpu.json")
    A = ap.parse_args()
    t0 = time.time()
    items = G.load_items()
    tr = G.load_candidates("train", "generator", [CFG["layer"]], ("span",), order="concat")
    ev = G.load_candidates("eval", "generator", [CFG["layer"]], ("span",), order="concat")
    Xtr, Xev = P.base_matrix(tr, [CFG["layer"]], "span"), P.base_matrix(ev, [CFG["layer"]], "span")
    y = np.array([float(r["y"]) for r in tr.rows], dtype=np.float32)
    byq = defaultdict(list)
    for i, r in enumerate(tr.rows):
        byq[f"{r['ds']}|{r['idx']}"].append(i)
    groups = [np.array(v) for v in byq.values()]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_d, Xev_d = P.to_dev(Xtr, mu, sd), P.to_dev(Xev, mu, sd)
    kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]

    S = np.zeros((A.seeds, len(kev)))
    out = {"config": CFG, "device": str(P.DEV), "seeds": {},
           "keys": [list(k) for k in kev],
           "numerics_note": "GPU kernels; see module docstring -- CPU/GPU and CPU thread count all "
                            "shift the SGD trajectory. Measured CPU sensitivity: same seed-0 config "
                            "gives 0.795640 (published thread count) vs 0.800409 at 8 threads."}
    for s in range(A.seeds):
        t = time.time()
        m = fit_bt(Xtr_d, y, groups, CFG, s)
        with torch.no_grad():
            S[s] = m(Xev_d).double().cpu().numpy()
        smap = {kev[i]: float(S[s, i]) for i in range(len(kev))}
        r = G.sel_eff(smap, items)
        out["seeds"][str(s)] = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                                "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                                "contested": r["contested"]["sel_eff"]}
        print(f"seed {s}: sel_eff={r['sel_eff']:.6f} contested={r['contested']['sel_eff']:.6f} "
              f"({time.time()-t:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()

    np.save(os.path.join(G.ROOT, "data/verifarch/pointwise_seed_scores_gpu.npy"), S)
    v = np.array([out["seeds"][str(s)]["sel_eff"] for s in range(A.seeds)])
    out["seed_spread"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "min": float(v.min()), "max": float(v.max()), "n": A.seeds}
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
    smap = {kev[i]: float(x) for i, x in enumerate(Z.mean(0))}
    r = G.sel_eff(smap, items)
    inc = G.sel_eff(G.incumbent_scores(), items)
    out["ensemble_zmean"] = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                             "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                             "contested": r["contested"]["sel_eff"],
                             "cand_auroc": G.cand_auroc(smap, items),
                             "vs_incumbent": G.paired_bootstrap(r["got"], inc["got"], rec=r["rec"]),
                             "guardrail_clean": G.guardrail_clean(r, inc)}
    out["seed0_vs_published"] = {"measured_gpu": out["seeds"]["0"]["sel_eff"], "published_cpu": 0.795640}
    out["minutes"] = round((time.time() - t0) / 60, 1)
    print("spread:", json.dumps(out["seed_spread"]), flush=True)
    print("ensemble:", out["ensemble_zmean"]["sel_eff"],
          "d vs inc", out["ensemble_zmean"]["vs_incumbent"]["d_sel_eff"],
          out["ensemble_zmean"]["vs_incumbent"]["d_sel_eff_ci"], flush=True)
    op = os.path.join(G.ROOT, A.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op, flush=True)


if __name__ == "__main__":
    main()
