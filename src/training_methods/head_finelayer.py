#!/usr/bin/env python3
"""head_finelayer.py -- locate the actual best layer, on transfer as well as on CV.

The deployed head reads layer 21, chosen from a FOUR-point grid (7/14/21/28) where 21 beat 28 by
0.017 CV.  The optimum therefore lies somewhere in 15..27 and was never located.  Layers
10/12/16/18/20/22/24/26 were extracted for exactly this.  Both endpoints are reported, because the
2026-08-19 audit showed in-domain CV is blind to the failure that matters.
"""
import argparse, json, os, sys
import numpy as np
from collections import defaultdict
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import head_sweep as HS
from head_domain_scaling import pick_sel_eff, string_prior, norm

FINE = [10, 12, 16, 18, 20, 22, 24, 26]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_finelayer_2026-08-19.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    zs, rows = [], []
    for sh in (0, 1):
        p = os.path.join(HS.FEATS, f"generator_train_finelayer_s{sh}of2")
        if not os.path.exists(p + ".npz"):
            raise SystemExit(f"missing {p}.npz -- run runners/run_finelayer_extract.sh first")
        zs.append(np.load(p + ".npz"))
        rows += json.load(open(p + ".meta.json"))["rows"]
    keep = [i for i, r in enumerate(rows) if r.get("n_tok", -1) > 0]
    rows = [rows[i] for i in keep]
    hs = np.concatenate([z["h_span"] for z in zs], 0)[keep]
    layers = list(zs[0]["layers"])
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{r['ds']}|{r['idx']}" for r in rows])
    img = np.array([r["img_md5"] for r in rows])
    ds = np.array([r["ds"] for r in rows])
    na = np.array([norm(r["na"]) for r in rows])
    print(f"{len(y)} rows | layers on disk {layers}", flush=True)

    import hashlib
    fold = np.array([int(hashlib.md5(str(h).encode()).hexdigest(), 16) % 5 for h in img])
    art = {"title": "Fine layer sweep: CV and transfer", "date": "2026-08-19",
           "no_fabricated_numbers": True, "layers_tested": layers, "results": {}}
    for li, L in enumerate(layers):
        X = hs[:, li].astype(np.float32)
        cvs = []
        for f in range(5):
            tr = fold != f
            mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
            m = HS.fit((X[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce", hidden=256,
                       wd=1e-2, epochs=30, seed=f)
            sc = np.empty(len(X), np.float32)
            for b0 in range(0, len(X), 4096):
                b1 = min(b0 + 4096, len(X)); sc[b0:b1] = HS.predict(m, (X[b0:b1] - mu) / sg)
            cvs.append(pick_sel_eff(sc, y, qid, fold == f))
        tr_hp = []
        for d in sorted(set(ds)):
            isD = ds == d; tr = ~isD
            sp = pick_sel_eff(string_prior(y, na, tr), y, qid, isD)
            vv = []
            for s in range(A.seeds):
                mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
                m = HS.fit((X[tr] - mu) / sg, y[tr], qid[tr], None, objective="bce", hidden=256,
                           wd=1e-2, epochs=30, seed=s)
                sc = np.empty(len(X), np.float32)
                for b0 in range(0, len(X), 4096):
                    b1 = min(b0 + 4096, len(X)); sc[b0:b1] = HS.predict(m, (X[b0:b1] - mu) / sg)
                vv.append(pick_sel_eff(sc, y, qid, isD))
            tr_hp.append(float(np.mean(vv)) - sp)
        art["results"][str(L)] = {"cv_sel_eff": float(np.mean(cvs)),
                                  "transfer_head_minus_prior": float(np.mean(tr_hp))}
        print(f"  layer {L:3}  CV {np.mean(cvs):.5f}   transfer head-prior {np.mean(tr_hp):+.5f}",
              flush=True)
        json.dump(art, open(A.out, "w"), indent=1)
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
