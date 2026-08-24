#!/usr/bin/env python3
"""freeze_pooled_selector.py -- retrain the probe verifier and FREEZE it as a reloadable artifact.

WHAT CHANGES FROM THE INCUMBENT (ckpts/train/genframe_head_ens8, 2026-08-05)

  training data   4 July domains, 31,439 rows      ->  those PLUS the training half of all eight
                                                       benchmarks, 108,126 rows
  features        layer 21, h_span                 ->  layers 18/20/22, h_span, rank-ensembled
  objective       Bradley-Terry pairs              ->  BCE (settled 2026-08-19: BCE survives
                                                       budget-matching, and BT gets worse with
                                                       more epochs while BCE does not)
  seeds           8                                ->  8 per layer, 24 heads total

MEASURED, on held-out image halves the probe never saw (head_final_stack_2026-08-24.json):

    four-domain, single layer   +0.0243 macro verifier-minus-greedy   6/8 benchmarks beaten
    pooled, single layer        +0.0765
    pooled + layer ensemble     +0.0802                               <- this artifact
    pooled + ensemble + SC      +0.0797

Self-consistency as an input feature is NOT included even though it was worth +0.0098 from the
four-domain base: once the probe is trained on in-domain data it is worth -0.0005.  It was
compensating for missing training data, not adding independent signal, and shipping it would add an
inference-time dependency on the raw candidate set for nothing.

⚠ WRITES TO A NEW DIRECTORY.  CLAUDE.md records that freeze_selector.py REWRITES
ckpts/train/genframe_head_ens8 and that those .pt files are the artifact of record.  This script
never touches it; the incumbent stays exactly where it is so the two can be compared.

⚠ WHAT THIS ARTIFACT MAY AND MAY NOT BE EVALUATED ON.  It is fitted on the training halves, so the
held-out halves are a clean test set and the numbers above are honest.  It has seen the OTHER half
of every benchmark, so it must never be quoted on a full benchmark -- that would be reporting on
questions whose images are in its training set.

  python3 src/training_methods/freeze_pooled_selector.py --seeds 8 --threads 4
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

D_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D_)
import head_sweep as HS
from head_domain_scaling import norm

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUTDIR = os.path.join(ROOT, "ckpts/train/genframe_head_pooled_ens")
INCUMBENT = os.path.join(ROOT, "ckpts/train/genframe_head_ens8")
TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
SHARED = {"pathvqa_open", "slake_open", "vqa_rad_open"}
ENS = [18, 20, 22]


def half(img):
    return int(hashlib.md5(("nd" + str(img)).encode()).hexdigest(), 16) % 2


def load_fine(stem, dsf=None):
    z = np.load(f"{HS.FEATS}/{stem}.npz"); m = json.load(open(f"{HS.FEATS}/{stem}.meta.json"))
    lay = [int(x) for x in z["layers"]]
    keep = [i for i, r in enumerate(m["rows"])
            if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]
    rr = [m["rows"][i] for i in keep]
    return {L: z["h_span"][keep, lay.index(L)].astype(np.float32) for L in ENS}, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=OUTDIR)
    A = ap.parse_args()
    HS.torch.set_num_threads(A.threads)
    t0 = time.time()
    if os.path.abspath(A.out) == os.path.abspath(INCUMBENT):
        raise SystemExit("REFUSING to overwrite the incumbent artifact at " + INCUMBENT)

    Xo, rows = load_fine("generator_train_finelayer", TRAIN_DOMAINS)
    Xadd = {L: [Xo[L]] for L in ENS}
    src = [r["ds"] for r in rows]
    n_orig = len(rows)
    evimgs, per_bench = set(), {}
    for cell in BENCH:
        stem = "generator_eval_finelayer" if cell in SHARED else f"generator_eval_finelayer_{cell}"
        if not os.path.exists(f"{HS.FEATS}/{stem}.npz"):
            continue
        Xc, rr = load_fine(stem, {cell} if cell in SHARED else None)
        istr = np.array([half(r["img_md5"]) == 1 for r in rr])
        for L in ENS:
            Xadd[L].append(Xc[L][istr])
        rows += [rr[i] for i in np.where(istr)[0]]
        src += [cell] * int(istr.sum())
        per_bench[cell] = int(istr.sum())
        evimgs |= {rr[i]["img_md5"] for i in np.where(~istr)[0]}
    X = {L: np.concatenate(Xadd[L]) for L in ENS}

    drop = np.array([r["img_md5"] in evimgs for r in rows])
    n_dropped = int(drop.sum())
    if n_dropped:
        for L in ENS:
            X[L] = X[L][~drop]
        n_orig -= int(drop[:n_orig].sum())
        rows = [r for r, d in zip(rows, drop) if not d]
        src = [s for s, d in zip(src, drop) if not d]
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    qid = np.array([f"{s}|{r['idx']}" for s, r in zip(src, rows)])
    assert not ({r["img_md5"] for r in rows} & evimgs), "held-out image in the training set"
    print(f"training rows {len(y):,} ({n_orig:,} original + {len(y)-n_orig:,} benchmark halves); "
          f"dropped {n_dropped} leaking rows", flush=True)

    os.makedirs(A.out, exist_ok=True)
    stats = {}
    for L in ENS:
        mu, sg = X[L].mean(0), X[L].std(0) + 1e-6
        Xs = (X[L] - mu) / sg
        for s in range(A.seeds):
            m = HS.fit(Xs, y, qid, None, objective="bce", hidden=256, wd=1e-2, epochs=30, seed=s)
            torch.save(m.state_dict(), os.path.join(A.out, f"head_L{L}_seed{s}.pt"))
        np.savez(os.path.join(A.out, f"standardizer_L{L}.npz"), mu=mu, sd=sg)
        stats[L] = {"mu_mean": float(mu.mean()), "sd_mean": float(sg.mean())}
        print(f"  layer {L}: {A.seeds} heads saved", flush=True)

    recipe = {
        "artifact": "genframe_head_pooled_ens", "date": "2026-08-24",
        "supersedes_for_open_text": "genframe_head_ens8 (NOT overwritten; still on disk)",
        "features": "generator-frame h_span, layers 18/20/22, 3584-d each",
        "architecture": "Linear(3584,256) -> GELU -> Linear(256,1) per layer per seed",
        "objective": "bce", "optimizer": "AdamW lr 1e-3 wd 1e-2, 30 epochs, batch 256",
        "seeds_per_layer": A.seeds, "n_heads": len(ENS) * A.seeds,
        "standardise": "per layer, x <- (x - mu_train)/sd_train, frozen in standardizer_L*.npz",
        "readout": "per head: rank_avg over the candidate set; score = mean over all 24 rank "
                   "vectors; pick = argmax, first-index tie-break",
        "training_rows": int(len(y)), "original_domain_rows": int(n_orig),
        "rows_per_benchmark_half": per_bench, "leaking_rows_dropped": n_dropped,
        "split": 'md5("nd"+img_md5) % 2 == 1 is TRAIN, == 0 is held out',
        "measured_on_held_out_halves": {
            "macro_verifier_minus_greedy": 0.0802, "benchmarks_beaten": "6/8",
            "four_domain_single_layer_baseline": 0.0243,
            "source": "results/cascade_methods/artifacts/head_final_stack_2026-08-24.json"},
        "excluded": "self-consistency input feature -- worth +0.0098 from the four-domain base "
                    "but -0.0005 once pooled, so it was compensating for missing data",
        "MAY_NOT_BE_EVALUATED_ON": "the full benchmarks -- this probe has seen the other image "
                                   "half of each. Held-out halves only.",
        "standardizer_stats": stats,
        "fit_seconds": round(time.time() - t0, 1)}
    json.dump(recipe, open(os.path.join(A.out, "recipe.json"), "w"), indent=1)
    open(os.path.join(A.out, "README.md"), "w").write(
        "# genframe_head_pooled_ens\n\n"
        "Probe verifier retrained 2026-08-24 on the training half of all eight open-ended medical "
        "VQA benchmarks, rank-ensembled over layers 18/20/22.\n\n"
        f"- {len(y):,} training rows ({n_orig:,} from the four original domains)\n"
        f"- {len(ENS) * A.seeds} heads: {len(ENS)} layers x {A.seeds} seeds\n"
        "- measured **+0.0802** macro verifier-minus-greedy on held-out image halves, against "
        "**+0.0243** for the four-domain single-layer recipe on the same halves\n\n"
        "**Do not evaluate this on a full benchmark.** It has seen the other image half of every "
        "one of them. The held-out halves (md5(\"nd\"+img_md5) %% 2 == 0) are the only clean test "
        "set for it.\n\n"
        "The incumbent `genframe_head_ens8` is untouched and remains the artifact of record for "
        "every number published before 2026-08-24.\n")
    print(f"\nwrote {A.out}")
    print(f"  {len(os.listdir(A.out))} files, "
          f"{sum(os.path.getsize(os.path.join(A.out, f)) for f in os.listdir(A.out))/1e6:.1f} MB")
    print(f"  incumbent untouched: {INCUMBENT} "
          f"({len(os.listdir(INCUMBENT))} files)")


if __name__ == "__main__":
    main()
