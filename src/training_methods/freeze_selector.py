#!/usr/bin/env python3
"""freeze_selector.py -- persist the recommended best-of-8 open-text selector as a RELOADABLE
artifact under ckpts/train/genframe_head_ens8/.

Item 2 of COMPARATIVE_VERIFIER_2026-08-05.md sec.7: before this script the deployed selector was a
scoring PROCEDURE inside integrate_eval.py, so using it meant re-training it, and a re-trained head
is a draw from a seed distribution (the 1-seed recipe's own 16-seed range is [0.802452, 0.814033]).

What it writes
  head_seed{0..7}.pt   torch state_dicts, 918,529 params each (~3.5 MB)
  standardizer.npz     the TRAIN mu / sd (sd = std + 1e-6).  Omitting this step costs 0.0075.
  recipe.json          every element of the config + the numerics environment + provenance
  README.md            what it is, how to reproduce, what it must reproduce

Determinism note.  fit_hidden_head.fit_head draws its batch permutations on the CPU under
torch.manual_seed(seed), so the fit is reproducible for a FIXED thread count -- the round measured
that torch.set_num_threads(8) moves seed-0 sel_eff 0.795640 -> 0.800409.  This script therefore
records torch.get_num_threads() in recipe.json and checks the refit against the training run's
stored per-seed eval logits (_integrate_parts/eval_head_seed_scores_cpu.npy).

    python3 src/training_methods/freeze_selector.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G            # noqa: E402
import integrate_lib as IL           # noqa: E402  (BASE_CFG, the pre-registered head config)
import cheapcontrast as CC           # noqa: E402  (standardize)
import genframe_selector as GS       # noqa: E402
from fit_hidden_head import fit_head, predict, Head  # noqa: E402

OUTDIR = os.path.join(G.ROOT, "ckpts/train/genframe_head_ens8")
REF = os.path.join(G.ROOT, "results/cascade_methods/artifacts/_integrate_parts",
                   "eval_head_seed_scores_cpu.npy")
PREREG = os.path.join(IL.PARTS, "prereg.json")
SEEDS = list(range(8))


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cfg = dict(IL.BASE_CFG)
    dec = json.load(open(PREREG))["PREREGISTERED_DECISION"]
    assert dec["ensembling_convention"] == "rank" and int(dec["n_seeds"]) == 8, dec
    print("PRE-REGISTERED:", json.dumps(dec), flush=True)

    # ---- data, exactly as integrate_eval.py builds it ------------------------------------------
    tr = G.load_candidates("train", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    ev = G.load_candidates("eval", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    Xtr_raw = tr.matrix(cfg["pooling"], cfg["layer"])
    Xev_raw = ev.matrix(cfg["pooling"], cfg["layer"])
    Xa, Xb = CC.standardize(Xtr_raw, Xev_raw)
    mu, sd = Xtr_raw.mean(0), Xtr_raw.std(0) + 1e-6
    assert np.allclose((Xev_raw - mu) / sd, Xb, atol=0, rtol=0), "standardizer round-trip mismatch"
    ytr = np.array([r["y"] for r in tr.rows], np.float32)
    gtr = G.group_ids(tr)
    kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]

    # ---- fit + persist -------------------------------------------------------------------------
    S = []
    for sd_seed in SEEDS:
        t = time.time()
        m = fit_head(Xa, ytr, gtr, objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                     lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"], seed=sd_seed)
        S.append(predict(m, Xb).astype(np.float64))
        torch.save(m.state_dict(), os.path.join(OUTDIR, f"head_seed{sd_seed}.pt"))
        print(f"  seed {sd_seed} fit {time.time()-t:.1f}s -> head_seed{sd_seed}.pt", flush=True)
    S = np.stack(S)
    np.savez(os.path.join(OUTDIR, "standardizer.npz"), mu=mu, sd=sd)

    n_par = sum(p.numel() for p in Head(Xa.shape[1], cfg["hidden"], 0.0).parameters())
    recipe = dict(
        what="frozen 8-seed generator-frame discriminative head + the parameter-free rank fusion "
             "with the incumbent LoRA verifier -- the deployable best-of-N open-text selector",
        date="2026-08-05", mode="generator", layer=cfg["layer"], pooling=cfg["pooling"],
        objective=cfg["objective"], hidden=cfg["hidden"], in_dim=int(Xa.shape[1]),
        wd=cfg["wd"], lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"], drop=0.0,
        seeds=SEEDS, n_params_per_head=int(n_par),
        row_order="concat",
        standardisation="x <- (x - mu_TRAIN)/(sd_TRAIN); sd_TRAIN = train std + 1e-6; frozen in "
                        "standardizer.npz. Omitting it costs 0.0075 sel_eff.",
        seed_readout="per seed rank_avg over the question's slots, then MEAN over the 8 seeds",
        ranker="rank_avg (average ranks for ties; rank_argsort gives 0.798365 and is NOT this)",
        fusion="final = rank_avg(incumbent slot scores) + rank_avg(head_rank); parameter-free, w=0.5",
        pick_rule="np.argmax over slots, FIRST-INDEX tie-break",
        incumbent="ckpts/train/lora_verifier_disjoint (the CLEAN disjoint-trained LoRA verifier)",
        train_data=dict(rows=len(tr.rows), questions=len(tr.questions),
                        cache="feats_hidden/generator_train_s*of2.npz"),
        eval_data=dict(rows=len(ev.rows), questions=len(ev.questions),
                       cache="feats_hidden/generator_eval_s*of2.npz"),
        provenance=dict(
            head_config_frozen_from="verifarch_hidden_generatorprompt_2026-08-04.json -> "
                                    "arms.generator.cv_selected (TRAIN-CV selected, not eval-tuned)",
            preregistration="results/cascade_methods/artifacts/_integrate_parts/prereg.json",
            round_doc="results/cascade_methods/docs/current/COMPARATIVE_VERIFIER_2026-08-05.md",
            training_run_artifact="results/cascade_methods/artifacts/verifarch_integrated_2026-08-04.json"),
        numerics=dict(device="cpu", torch=torch.__version__,
                      torch_num_threads=int(torch.get_num_threads()),
                      numpy=np.__version__,
                      note="the CPU trainer's batch permutations are thread-count sensitive: the "
                           "round measured seed-0 sel_eff 0.795640 at the published thread count "
                           "vs 0.800409 at torch.set_num_threads(8). Inference (this artifact's "
                           "only use) is a plain forward pass and is NOT thread-sensitive."),
        must_reproduce=dict(GS.PUBLISHED),
        cost="zero extra VLM forward passes over the deployed recipe (7.636674 per question total)")
    json.dump(recipe, open(os.path.join(OUTDIR, "recipe.json"), "w"), indent=1, default=float)

    # ---- checks -------------------------------------------------------------------------------
    items = G.load_items()
    inc = G.incumbent_scores()
    ens = IL.ENSEMBLERS["rank"](kev, S, items)
    r_train_run = G.sel_eff(G.rank_fuse(inc, ens, items=items, ranker=G.rank_avg), items)

    v = GS.verify(OUTDIR)                                   # reload from disk, score, compare
    ref_dev = None
    if os.path.exists(REF):
        ref_dev = float(np.max(np.abs(np.load(REF)[:len(SEEDS)] - S)))

    out = dict(
        what="freeze + verify the deployable 8-seed generator-frame selector",
        outdir=OUTDIR, recipe=recipe,
        refit_in_this_script=dict(
            sel_eff=r_train_run["sel_eff"], acc=r_train_run["acc"],
            per_ds={d: r_train_run["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
            abs_dev_vs_published=abs(r_train_run["sel_eff"] - GS.PUBLISHED["sel_eff"])),
        refit_vs_training_run_max_abs_logit_deviation=ref_dev,
        reload_from_disk_verify=v,
        files=sorted(os.listdir(OUTDIR)),
        bytes_on_disk=int(sum(os.path.getsize(os.path.join(OUTDIR, f))
                              for f in os.listdir(OUTDIR))))
    p = os.path.join(G.ROOT, "results/cascade_methods/artifacts",
                     "frozen_selector_ens8_2026-08-05.json")
    json.dump(out, open(p, "w"), indent=1, default=float)

    print("\n--- REFIT IN THIS SCRIPT ---")
    print("  sel_eff", repr(r_train_run["sel_eff"]),
          " |dev vs 0.810627| =", out["refit_in_this_script"]["abs_dev_vs_published"])
    print("  max abs logit deviation vs the 2026-08-05 training run:", ref_dev)
    print("--- RELOAD FROM DISK ---")
    print("  pass =", v["pass"], " max abs deviation =", v["max_abs_deviation"])
    print("  measured:", json.dumps(v["measured"], default=float))
    print("  reloaded-vs-refit head logits max abs dev:",
          v["reloaded_vs_training_run_max_abs_logit_deviation"])
    print(f"\nwrote {OUTDIR}  ({out['bytes_on_disk']/1e6:.1f} MB)\nwrote {p}")
    return out


if __name__ == "__main__":
    main()
