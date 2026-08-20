#!/usr/bin/env python3
"""head_best_arch_eval.py -- take whichever architecture won on TRANSFER and score it properly.

Reads head_arch_transfer_2026-08-19.json, picks the architecture with the best mean
head-minus-string-prior over the four held-out domains, retrains it on the full pool at 8 seeds, and
reports it on the 2,345-question eval set against the deployed baseline -- with the image-clustered
bootstrap and the string prior beside it.
"""
import argparse, json, os, sys
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import head_sweep as HS
import head_arch_transfer as HA
from head_domain_scaling import pick_sel_eff, string_prior, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HS.OUTDIR, "head_best_arch_eval_2026-08-19.json"))
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)

    src = os.path.join(HS.OUTDIR, "head_arch_transfer_2026-08-19.json")
    if not os.path.exists(src):
        raise SystemExit("head_arch_transfer must run first")
    tr = json.load(open(src))["results"]
    ranked = sorted(((v.get("MEAN_head_minus_prior", -9), k) for k, v in tr.items()), reverse=True)
    best = ranked[0][1]
    print(f"best on transfer: {best} (mean head-minus-prior {ranked[0][0]:+.4f})", flush=True)
    print("  ranking: " + ", ".join(f"{k} {s:+.4f}" for s, k in ranked), flush=True)

    X, y, qid, na, ds = HA.__dict__["np"], None, None, None, None  # placeholder to keep linters quiet
    H, y, qid, img, ds = HS.load_train()
    X0 = HS.assemble(H, HS.BASE)
    rows = []
    for sh in (0, 1):
        rows += json.load(open(os.path.join(HS.FEATS, f"generator_train_s{sh}of2.meta.json")))["rows"]
    rows = [r for r in rows if r.get("n_tok", -1) > 0]
    na = np.array([norm(r["na"]) for r in rows])

    sys.path.insert(0, os.path.join(HS.ROOT, "src/training_methods"))
    from head_eval_bce import load_eval, ensemble_pick, score, paired_boot
    He, ye, qe, de, ie = load_eval()
    Xe0 = HS.assemble(He, HS.BASE)

    out = {"title": f"Best transfer architecture ({best}) on the eval set", "date": "2026-08-19",
           "no_fabricated_numbers": True, "arch": best,
           "transfer_ranking": {k: s for s, k in ranked}, "arms": {}}
    for tag, arch in (("deployed_raw", "raw"), (f"best_{best}", best)):
        Xa, pca = HA.build(arch, X0, qid, None)
        Xea, _ = HA.build(arch, Xe0, qe, pca)
        mu, sg = Xa.mean(0), Xa.std(0) + 1e-6
        L = []
        for s in range(A.seeds):
            m = HS.fit((Xa - mu) / sg, y, qid, None, objective="bce", hidden=256, wd=1e-2,
                       epochs=30, seed=s)
            sc = np.empty(len(Xea), np.float32)
            for b0 in range(0, len(Xea), 4096):
                b1 = min(b0 + 4096, len(Xea)); sc[b0:b1] = HS.predict(m, (Xea[b0:b1] - mu) / sg)
            L.append(sc)
            print(f"    [{tag}] seed {s}", flush=True)
        picks = ensemble_pick(np.stack(L), qe)
        st, got, rec, dsv, clv = score(picks, ye, qe, de, ie)
        out["arms"][tag] = {k: v for k, v in st.items()}
        out["arms"][tag]["_got"] = got.tolist()
        out["_cl"] = clv.tolist(); out["_rec"] = rec.tolist()
        print(f"  {tag:18} sel_eff {st['sel_eff']:.4f} acc {st['acc']:.4f}", flush=True)
    a = np.array(out["arms"][f"best_{best}"].pop("_got"))
    b = np.array(out["arms"]["deployed_raw"].pop("_got"))
    out["delta_best_vs_deployed"] = paired_boot(a, b, np.array(out.pop("_rec")),
                                                clusters=np.array(out.pop("_cl")))
    json.dump(out, open(A.out, "w"), indent=1)
    print(f"  delta {out['delta_best_vs_deployed']}")
    print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
