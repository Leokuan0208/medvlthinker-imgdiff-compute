#!/usr/bin/env python3
"""pointwise_seeds.py -- the BAR, run at >=10 seeds.

The published generator-frame pointwise head (sel_eff 0.795640) is a SINGLE seed-0 fit. Protocol
rule 4 says a single seed is not a result, and the apples-to-apples comparison for a pairwise
contrast head is "comparative vs pointwise on IDENTICAL representations". So refit the published
CV-selected pointwise config (L21 / span / bt / h256 / wd 1e-2 / 30 epochs / setrel 0) at 12 seeds
through the published code path (fit_hidden_head.fit_head, CPU), and store every seed's
per-candidate eval scores so the pairwise round can fuse against, and bootstrap against, the
seed-AVERAGED pointwise head rather than its luckiest draw.

  python3 src/training_methods/pointwise_seeds.py --seeds 12 --out data/verifarch/pointwise_seeds.json
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import fit_hidden_head as F  # noqa: E402
import genframe_data as G  # noqa: E402

CFG = {"layer": 21, "pooling": "span", "objective": "bt", "hidden": 256, "wd": 1e-2, "epochs": 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="data/verifarch/pointwise_seeds.json")
    A = ap.parse_args()
    t0 = time.time()
    print("torch threads", torch.get_num_threads(), flush=True)

    ztr, mtr = F.load_cache("feats_hidden", "generator", "train")
    zev, mev = F.load_cache("feats_hidden", "generator", "eval")
    layers = list(ztr["layers"]); li = layers.index(CFG["layer"])
    Xtr, ytr, ktr, gtr, _ = F.build_matrix(ztr, mtr, li, CFG["pooling"], 0)
    Xev, yev, kev, gev, _ = F.build_matrix(zev, mev, li, CFG["pooling"], 0)
    qid = [f"{a}|{b}" for (a, b, c) in ktr]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_s, Xev_s = (Xtr - mu) / sd, (Xev - mu) / sd
    items = G.load_items()

    out = {"config": CFG, "torch_threads": int(torch.get_num_threads()),
           "published_seed0": 0.795640, "seeds": {}, "keys": [list(k) for k in kev]}
    S = np.zeros((A.seeds, len(kev)))
    for s in range(A.seeds):
        t = time.time()
        m = F.fit_head(Xtr_s, ytr, qid, objective=CFG["objective"], hidden=CFG["hidden"],
                       wd=CFG["wd"], epochs=CFG["epochs"], seed=s)
        S[s] = F.predict(m, Xev_s)
        smap = {kev[i]: float(S[s, i]) for i in range(len(kev))}
        r = G.sel_eff(smap, items)
        out["seeds"][str(s)] = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                                "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                                "contested": r["contested"]["sel_eff"]}
        print(f"seed {s}: sel_eff={r['sel_eff']:.6f} acc={r['acc']:.6f} "
              f"contested={r['contested']['sel_eff']:.6f}  ({time.time()-t:.0f}s)", flush=True)

    np.save(os.path.join(G.ROOT, "data/verifarch/pointwise_seed_scores.npy"), S)
    v = np.array([out["seeds"][str(s)]["sel_eff"] for s in range(A.seeds)])
    out["seed_spread"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                          "min": float(v.min()), "max": float(v.max()), "n": A.seeds}
    out["seed0_reproduction"] = {"measured": out["seeds"]["0"]["sel_eff"],
                                 "published": 0.795640,
                                 "abs_dev": abs(out["seeds"]["0"]["sel_eff"] - 0.795640)}
    # seed ENSEMBLE: per-seed global z-score, then mean (scale-free; BT logits have arbitrary offset)
    Z = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
    ens = Z.mean(0)
    smap = {kev[i]: float(ens[i]) for i in range(len(kev))}
    r = G.sel_eff(smap, items)
    inc = G.sel_eff(G.incumbent_scores(), items)
    b = G.paired_bootstrap(r["got"], inc["got"], rec=r["rec"], nboot=10000)
    out["ensemble_zmean"] = {"sel_eff": r["sel_eff"], "acc": r["acc"],
                             "per_ds": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                             "contested": r["contested"]["sel_eff"],
                             "cand_auroc": G.cand_auroc(smap, items),
                             "vs_incumbent": b,
                             "guardrail_clean": G.guardrail_clean(r, inc)}
    print("ENSEMBLE(z-mean):", json.dumps({k: v for k, v in out["ensemble_zmean"].items()
                                           if k != "vs_incumbent"}), flush=True)
    print("  vs incumbent d=%.6f %s" % (b["d_sel_eff"], b["d_sel_eff_ci"]), flush=True)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    op = os.path.join(G.ROOT, A.out)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op, flush=True)


if __name__ == "__main__":
    main()
