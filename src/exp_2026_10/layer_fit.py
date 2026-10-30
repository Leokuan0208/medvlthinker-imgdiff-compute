#!/usr/bin/env python3
"""layer_fit.py -- fit the shipped probe recipe on ONE layer, both cross-fitting folds, for the layer study.

Leo, 2026-10-21: "did we test for deeper layers other than 18 to 22, lets test for deeper layers and run different
fusions of different layers". The only earlier depth sweep (head_finelayer_shard*.json, 2026-08-19) used the
old four-domain recipe, one currency and a different baseline. This refits the FINAL recipe -- Linear(3584,256)
-> GELU -> Linear(256,1), BCE, AdamW wd 1e-2, 30 epochs, 8 seeds, per-layer standardisation, two-fold image
cross-fitting, the FINAL fitting set (four original domains + the other image half of every open and
comparability cell's T=0.7 cache, minus every image scored in the fold) -- on layer L of the 2026-10-21
re-extraction (/data/dan/feats_deep_2026-10-21, layers 16 and 18-28, bitwise identical to the old caches at
18/20/22). Layers 18/20/22 are refitted HERE too, at the same thread count, so the study's control is a
same-session refit rather than the frozen probes (whose 4-thread fit cannot be reproduced at 2 threads).

  OMP_NUM_THREADS=2 python3 src/exp_2026_10/layer_fit.py --layer 24
Writes ckpts/train/layer_study_2026-10-22/fold{0,1}/head_L<L>_seed<s>.pt + standardizer_L<L>.npz + done_L<L>.json
"""
import os, sys
sys.dont_write_bytecode = True
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import argparse, json, time
import numpy as np, torch

WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
sys.path.insert(0, WT + "/src/exp_2026_10")
import final_method_eval as FME                  # noqa: E402  (ER.half, HS.fit, cell lists, TMAP)

ER, HS = FME.ER, FME.HS
D = "/data/dan/feats_deep_2026-10-21"
OUTD = FME.MAIN + "/ckpts/train/layer_study_2026-10-22"
TMAP, TMAP_COMP = FME.TMAP, FME.TMAP_COMP
CELLS = FME.OPEN8 + FME.COMP
TT = {0.2: "T02_", 0.4: "T04_", 0.7: "", 1.0: "T10_"}


def stem07(c): return f"generator_eval_deep_{c}"


def stemT(c):
    """The Tmap pool's cache. STRICT for the open benchmarks: a missing cache must not silently become T07.
    The comparability cells' Tmap pools were not re-extracted; they are only used here for the excluded-image
    set, and their images are the same images as their T07 pools, so the T07 cache stands in exactly."""
    T = TMAP.get(c, TMAP_COMP.get(c, 0.7))
    s = f"generator_eval_deep_{TT[T]}{c}"
    if os.path.exists(f"{D}/{s}.npz") or T == 0.7:
        return s
    if c in FME.OPEN8:
        raise FileNotFoundError(f"Tmap cache {s} missing for {c} (T={T}) -- refusing to fall back to T=0.7")
    return stem07(c)


def load(stem, L, dsf=None):
    z = np.load(f"{D}/{stem}.npz"); m = json.load(open(f"{D}/{stem}.meta.json"))
    li = [int(x) for x in z["layers"]].index(L)
    keep = [i for i, r in enumerate(m["rows"]) if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    return z["h_span"][keep, li].astype(np.float32), [m["rows"][i] for i in keep]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layer", type=int, required=True); ap.add_argument("--seeds", type=int, default=8)
    A = ap.parse_args(); L = A.layer; t0 = time.time()
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    Xo, ro = load("generator_train_deep", L, FME.TRAIN_DOMAINS)
    FX, rows, orig, src = [Xo], list(ro), [True] * len(ro), [r["ds"] for r in ro]     # = FME's fit_src
    evimg = {0: set(), 1: set()}
    for c in CELLS:
        X7, r7 = load(stem07(c), L); FX.append(X7); rows += r7; orig += [False] * len(r7); src += [c] * len(r7)
        for st in {stem07(c), stemT(c)}:                         # every image scored under T07 or Tmap
            for r in json.load(open(f"{D}/{st}.meta.json"))["rows"]:
                if r.get("n_tok", -1) > 0:                       # = FME: only rows load_cache keeps
                    evimg[ER.half(r["img_md5"])].add(r["img_md5"])
    FX = np.concatenate(FX); y = np.array([r["y"] for r in rows], np.float32)
    q = np.array([f"{s_}|{r['idx']}" for s_, r in zip(src, rows)]); img = np.array([r["img_md5"] for r in rows])
    half = np.array([ER.half(i) for i in img]); orig = np.array(orig)
    info = {"layer": L, "seeds": A.seeds, "threads": int(os.environ["OMP_NUM_THREADS"]), "folds": {}}
    for f in (0, 1):
        tr = (orig | (half == 1 - f)) & ~np.isin(img, list(evimg[f]))
        mu, sd = FX[tr].mean(0), FX[tr].std(0) + 1e-6
        Xs = (FX[tr] - mu) / sd
        od = f"{OUTD}/fold{f}"; os.makedirs(od, exist_ok=True)
        for s in range(A.seeds):
            m = HS.fit(Xs, y[tr], q[tr], None, objective="bce", hidden=256, wd=1e-2, epochs=30, sched="none", seed=s)
            torch.save(m.state_dict(), f"{od}/head_L{L}_seed{s}.pt")
        np.savez(f"{od}/standardizer_L{L}.npz", mu=mu.astype(np.float32), sd=sd.astype(np.float32))
        info["folds"][f] = {"train_rows": int(tr.sum()), "original_domain_rows": int((tr & orig).sum())}
        print(f"[{time.time() - t0:6.0f}s] layer {L} fold {f}: {A.seeds} probes fitted on {int(tr.sum()):,} rows", flush=True)
    info["seconds"] = round(time.time() - t0, 1)
    json.dump(info, open(f"{OUTD}/done_L{L}.json", "w"), indent=1)
    print("DONE", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
