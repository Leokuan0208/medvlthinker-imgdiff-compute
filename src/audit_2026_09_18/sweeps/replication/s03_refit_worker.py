#!/usr/bin/env python3
"""READ-ONLY re-fit of ONE layer of head_final_stack.py's pooled arm for a second/third generator, saving
the per-row held-out scores of every seed so that per-question picks (and therefore clustered CIs, EM
currency, etc.) can be computed.  Mirrors head_final_stack.py main() line by line for row order, dedupe,
leak-drop, standardisation and the HS.fit call.  Writes ONLY under the audit tmp dir.

  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 s03_refit_worker.py --generator qwen --layer 20
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(MAIN, "src/training_methods"))
import head_sweep as HS                      # no import-time writes (checked)
from head_domain_scaling import norm

OUT = "/data/dan/audit_2026-09-18/tmp/replication/refit"
CK = os.path.join(MAIN, "ckpts/openvqa/cheap_lingshu7b")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
GENS = {
    "qwen": {"tag": "qwen25vl7b",
             "train_stems": [f"generator_train_qwen_{c}" for c in
                             ("kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train")],
             "eval_stem": lambda c: f"generator_eval_qwen_{c}"},
    "medgemma": {"tag": "medgemma4b", "train_stems": [],
                 "eval_stem": lambda c: f"generator_eval_medgemma_{c}"},
}


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=sorted(GENS))
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    A = ap.parse_args()
    GEN = GENS[A.generator]
    HS.torch.set_num_threads(A.threads)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    L = A.layer

    def load_fine(stem, dsf=None):
        z = np.load(f"{HS.FEATS}/{stem}.npz"); m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
        lay = [int(x) for x in z["layers"]]
        keep = [i for i, r in enumerate(m["rows"])
                if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
        rr = [m["rows"][i] for i in keep]
        return z["h_span"][keep, lay.index(L)].astype(np.float32), rr

    seen, Xparts, rows_o, n_dup = set(), [], [], 0
    for st in GEN["train_stems"]:
        Xs_, rs_ = load_fine(st, TRAIN_DOMAINS)
        k = np.array([(r["ds"], r["idx"], norm(r["na"])) not in seen for r in rs_])
        for r, kk in zip(rs_, k):
            if kk:
                seen.add((r["ds"], r["idx"], norm(r["na"])))
        n_dup += int((~k).sum())
        Xparts.append(Xs_[k])
        rows_o += [r for r, kk in zip(rs_, k) if kk]
        print(f"  train cache {st}: kept {int(k.sum())}  [{time.time()-t0:.0f}s]", flush=True)
    print(f"deduped {n_dup:,} -> {len(rows_o):,} unique", flush=True)
    Xadd = list(Xparts)
    rows_all, src_all = list(rows_o), [r["ds"] for r in rows_o]
    ev = {}
    for cell in BENCH:
        stem = GEN["eval_stem"](cell)
        Xc, rr = load_fine(stem, None)
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        Xadd.append(Xc[istr])
        rows_all += [rr[i] for i in np.where(istr)[0]]
        src_all += [cell] * int(istr.sum())
        ev[cell] = {"X": Xc[~istr], "rows": [rr[i] for i in np.where(~istr)[0]]}
        print(f"  eval cache {cell}: train-half rows {int(istr.sum())}, held-out rows {int((~istr).sum())} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    Xpool = np.concatenate(Xadd)
    n_orig = len(rows_o)
    evimgs = set()
    for d in ev.values():
        evimgs |= {r["img_md5"] for r in d["rows"]}
    drop = np.array([r["img_md5"] in evimgs for r in rows_all])
    if drop.any():
        print(f"dropping {int(drop.sum())} leaking training rows", flush=True)
        Xpool = Xpool[~drop]
        n_orig -= int(drop[:n_orig].sum())
        rows_all = [r for r, dd in zip(rows_all, drop) if not dd]
        src_all = [s for s, dd in zip(src_all, drop) if not dd]
    y = np.array([r["y"] for r in rows_all], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src_all, rows_all)])
    print(f"pooled: {len(y):,} rows ({n_orig:,} original); non-finite in X: {int((~np.isfinite(Xpool)).sum())}",
          flush=True)
    sub = np.ones(len(y), bool)
    mu, sg = Xpool[sub].mean(0), Xpool[sub].std(0) + 1e-6
    Xs = (Xpool[sub] - mu) / sg
    scores = {c: [] for c in ev}
    for s in range(A.seeds):
        m = HS.fit(Xs, y[sub], qid[sub], None, objective="bce", hidden=256, wd=1e-2, epochs=30, seed=s)
        for c, d in ev.items():
            scores[c].append(HS.predict(m, (d["X"] - mu) / sg))
        print(f"  seed {s} fitted [{time.time()-t0:.0f}s]", flush=True)
        np.savez(f"{OUT}/{A.generator}_L{L}.npz", **{c: np.stack(v) for c, v in scores.items()})
    meta = {"generator": A.generator, "layer": L, "seeds": A.seeds, "threads": A.threads,
            "pooled_rows": int(len(y)), "original_rows": int(n_orig), "dup_dropped": int(n_dup),
            "torch_threads": HS.torch.get_num_threads(), "OMP": os.environ.get("OMP_NUM_THREADS"),
            "minutes": (time.time() - t0) / 60,
            "heldout_rows": {c: [{"idx": r["idx"], "na": r["na"], "ans": r["ans"], "y": r["y"],
                                  "img_md5": r["img_md5"]} for r in d["rows"]] for c, d in ev.items()}}
    json.dump(meta, open(f"{OUT}/{A.generator}_L{L}.meta.json", "w"))
    print(f"DONE {A.generator} L{L} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
