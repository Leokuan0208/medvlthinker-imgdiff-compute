#!/usr/bin/env python3
"""integrate_eval.py -- STAGE 3: the deployable selector, measured against the CURRENTLY
DEPLOYED fusion (0.806540), not just against the incumbent verifier (0.775204).

Everything here runs on CPU with the published trainer (fit_hidden_head.fit_head) and the
published standardization, so the reference cells reproduce EXACTLY inside this same script:
seed 0 head = 0.795640 and rank_avg(incumbent, seed-0 head) = 0.806540. Every arm is
therefore device-matched to the number it is being compared against.

Arms
  INC             the incumbent clean disjoint-trained LoRA verifier          (the old bar)
  HEAD_s          the frozen generator-frame head, one per seed
  HEAD_ens        the pre-registered seed ensemble of those heads
  DEPLOYED        rank_avg(INC, HEAD_0)  -- the currently deployed fusion, reconstructed
  DEPLOYED_dist   rank_avg(INC, HEAD_s) for every seed -- the deployed number's OWN spread
  HEADLINE        rank_avg(INC, HEAD_ens)                          <- the deployable answer
  + diagnostics:  eval-fitted weight sweep, cross-fitted learned combiner, self-consistency
                  as a third member, and real A-vs-B pairwise as a fourth member.

  python3 src/training_methods/integrate_eval.py --seeds 16
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G          # noqa: E402
import integrate_lib as IL         # noqa: E402
import cheapcontrast as CC         # noqa: E402  (standardize + multiplicity)
from fit_hidden_head import fit_head, predict  # noqa: E402

ART = os.path.join(G.ROOT, "results/cascade_methods/artifacts")
PM_HF = os.path.join(ART, "realpairwise_teacher_pmatrix_hf_2026-08-05.jsonl")


def fit_seeds(Xa, ytr, gtr, Xb, cfg, seeds):
    S = []
    for sd in seeds:
        t = time.time()
        m = fit_head(Xa, ytr, gtr, objective=cfg["objective"], hidden=cfg["hidden"],
                     wd=cfg["wd"], lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"], seed=sd)
        S.append(predict(m, Xb).astype(np.float64))
        print(f"  seed {sd} fit {time.time()-t:.1f}s", flush=True)
    return np.stack(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--nboot", type=int, default=10000)
    A = ap.parse_args()
    cfg = IL.BASE_CFG
    prereg = json.load(open(os.path.join(IL.PARTS, "prereg.json")))
    dec = prereg["PREREGISTERED_DECISION"]
    conv, k = dec["ensembling_convention"], int(dec["n_seeds"])
    print("PRE-REGISTERED:", json.dumps(dec), flush=True)

    items = G.load_items()
    inc = G.incumbent_scores()
    base = G.sel_eff(inc, items)                    # THE OLD BAR (0.775204)
    tr = G.load_candidates("train", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    ev = G.load_candidates("eval", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
    Xa, Xb = CC.standardize(tr.matrix(cfg["pooling"], cfg["layer"]),
                            ev.matrix(cfg["pooling"], cfg["layer"]))
    ytr = np.array([r["y"] for r in tr.rows], np.float32)
    gtr = G.group_ids(tr)

    seeds = list(range(max(A.seeds, k)))
    S = fit_seeds(Xa, ytr, gtr, Xb, cfg, seeds)
    np.save(os.path.join(IL.PARTS, "eval_head_seed_scores_cpu.npy"), S)

    out = {"what": "the integrated deployable selector for open-text best-of-8 selection",
           "date": "2026-08-05",
           "preregistration": {"source": "results/cascade_methods/artifacts/_integrate_parts/prereg.json",
                               "decision": dec,
                               "head_config": cfg,
                               "head_config_provenance": prereg["head_config_frozen_from"]},
           "device": "cpu (published trainer, published standardization)",
           "n_seeds_run": len(seeds), "nboot": A.nboot}

    # ------------------------------------------------------------------ reference cells
    head0 = IL.score_map(kev, S[0])
    r0 = G.sel_eff(head0, items)
    dep0map = G.rank_fuse(inc, head0, items=items, ranker=G.rank_avg)
    rdep0 = G.sel_eff(dep0map, items)
    out["reference_cells_in_this_script"] = {
        "head_seed0": {"measured": r0["sel_eff"], "published": 0.795640,
                       "abs_dev": abs(r0["sel_eff"] - 0.795640)},
        "deployed_fusion_seed0": {"measured": rdep0["sel_eff"], "published": 0.806540,
                                  "abs_dev": abs(rdep0["sel_eff"] - 0.806540)},
        "pass": bool(abs(r0["sel_eff"] - 0.795640) < 1e-5 and abs(rdep0["sel_eff"] - 0.806540) < 1e-5)}
    print("reference cells:", json.dumps(out["reference_cells_in_this_script"], default=float),
          flush=True)

    # ------------------------------------------------------------------ strata
    short = IL.short_mask(items, 3)
    masks = {"short_answer_le3w": short, "long_answer_gt3w": ~short}
    out["strata_definition"] = {
        "contested": "recoverable AND >=2 distinct candidate strings (n=916 of 1468 recoverable)",
        "short_answer_le3w": {"rule": "gold answer word count <= 3, gold read from "
                                      "ckpts/openvqa/cheap_lingshu7b/*_sc8.jsonl",
                              "n_items": int(short.sum()),
                              "n_recoverable": int((short & (base["rec"] == 1)).sum())},
        "long_answer_gt3w": {"n_items": int((~short).sum()),
                             "n_recoverable": int(((~short) & (base["rec"] == 1)).sum())}}

    def rep(smap, tag, picks=None):
        return IL.report(smap, items, base, tag, nboot=A.nboot, picks=picks, extra_masks=masks)

    # ------------------------------------------------------------------ per-seed + ensembles
    per_seed = [G.sel_eff(IL.score_map(kev, s), items)["sel_eff"] for s in S]
    dep_seed = [G.sel_eff(G.rank_fuse(inc, IL.score_map(kev, s), items=items), items)["sel_eff"]
                for s in S]
    out["seed_spread"] = {
        "head_single_seed": {"per_seed": per_seed, "mean": float(np.mean(per_seed)),
                             "sd": float(np.std(per_seed, ddof=1)),
                             "range": [float(np.min(per_seed)), float(np.max(per_seed))]},
        "deployed_recipe_single_seed": {
            "definition": "rank_avg(incumbent, ONE-seed head) -- i.e. the published 0.806540 "
                          "recipe, re-run at every seed",
            "per_seed": dep_seed, "mean": float(np.mean(dep_seed)),
            "sd": float(np.std(dep_seed, ddof=1)),
            "range": [float(np.min(dep_seed)), float(np.max(dep_seed))],
            "published_0.806540_percentile": float(np.mean(np.array(dep_seed) <= 0.806540))}}

    ensmap = IL.ENSEMBLERS[conv](kev, S[:k], items)
    r_ens, _ = rep(ensmap, f"HEAD_ens ({k}-seed {conv})")
    head_ens_res = G.sel_eff(ensmap, items)

    headline_map = G.rank_fuse(inc, ensmap, items=items, ranker=G.rank_avg)
    r_head, r_head_full = rep(headline_map, "HEADLINE rank_avg(incumbent, HEAD_ens)")

    # paired bootstrap vs the DEPLOYED fusion (the comparison that matters)
    def vs(a_res, b_got, tag):
        b1 = G.paired_bootstrap(a_res["got"], b_got, rec=a_res["rec"], nboot=A.nboot)
        b2 = G.paired_bootstrap(a_res["got"], b_got, nboot=A.nboot, mask=a_res["contested_mask"])
        o = {"tag": tag, "d_sel_eff": b1["d_sel_eff"], "d_sel_eff_ci": b1["d_sel_eff_ci"],
             "d_acc": b1["d_acc"], "d_acc_ci": b1["d_acc_ci"],
             "d_contested": b2["d_sel_eff"], "d_contested_ci": b2["d_sel_eff_ci"]}
        for nm, m in masks.items():
            mm = np.asarray(m, bool) & (a_res["rec"] == 1)
            bb = G.paired_bootstrap(a_res["got"], b_got, nboot=A.nboot, mask=mm)
            o[f"d_{nm}"] = bb["d_sel_eff"]; o[f"d_{nm}_ci"] = bb["d_sel_eff_ci"]
        return o

    out["arms"] = {
        "incumbent": {"sel_eff": base["sel_eff"], "acc": base["acc"],
                      "per_ds": {d: v["sel_eff"] for d, v in base["per_ds"].items()},
                      "contested_sel_eff": base["contested"]["sel_eff"],
                      "short_sel_eff": float(base["got"][short & (base["rec"] == 1)].mean()),
                      "long_sel_eff": float(base["got"][(~short) & (base["rec"] == 1)].mean())},
        "deployed_fusion_seed0": {
            "sel_eff": rdep0["sel_eff"], "acc": rdep0["acc"],
            "per_ds": {d: v["sel_eff"] for d, v in rdep0["per_ds"].items()},
            "contested_sel_eff": rdep0["contested"]["sel_eff"],
            "short_sel_eff": float(rdep0["got"][short & (rdep0["rec"] == 1)].mean()),
            "long_sel_eff": float(rdep0["got"][(~short) & (rdep0["rec"] == 1)].mean()),
            "guardrail_clean": G.guardrail_clean(rdep0, base)},
        "head_ensemble_alone": r_ens,
        "HEADLINE": r_head,
    }
    out["HEADLINE_vs_deployed_fusion"] = vs(r_head_full, rdep0["got"],
                                            "HEADLINE vs deployed fusion (seed-0, = 0.806540)")
    out["head_ensemble_vs_deployed_fusion"] = vs(head_ens_res, rdep0["got"],
                                                 "HEAD_ens alone vs deployed fusion")

    # ------------------------------------------------- ensemble stability (blocks of seeds)
    blocks = {}
    for kk in (1, 2, 4, 8, 16):
        if kk > len(seeds):
            continue
        vals, fvals = [], []
        for i in range(0, len(seeds) - kk + 1, kk):
            e = IL.ENSEMBLERS[conv](kev, S[i:i + kk], items)
            vals.append(G.sel_eff(e, items)["sel_eff"])
            fvals.append(G.sel_eff(G.rank_fuse(inc, e, items=items), items)["sel_eff"])
        blocks[f"k={kk}"] = {"head_ens": vals, "head_ens_mean": float(np.mean(vals)),
                             "fused": fvals, "fused_mean": float(np.mean(fvals)),
                             "n_disjoint_blocks": len(vals)}
    out["ensemble_size_curve_disjoint_blocks"] = blocks

    # ------------------------------------------------------------------ DIAGNOSTICS
    diag = {}
    # (a) eval-visible weight sweep on the rank blend -- how much the parameter-free choice costs
    RI = np.stack([G.rank_avg(m) for m in G._slot_scores(inc, items)])
    RH = np.stack([G.rank_avg(m) for m in G._slot_scores(ensmap, items)])
    sweep = {}
    for w in np.round(np.arange(0, 1.001, 0.1), 2):
        sc = {(it["ds"], it["idx"]): list(w * RH[i] + (1 - w) * RI[i]) for i, it in enumerate(items)}
        sweep[float(w)] = G.sel_eff(sc, items)["sel_eff"]
    diag["eval_visible_weight_sweep"] = {
        "note": "DIAGNOSTIC ONLY -- w is chosen with eval visibility. w=0.5 is the "
                "parameter-free headline.", "sel_eff_by_w": sweep,
        "best_w": float(max(sweep, key=sweep.get)), "best_sel_eff": max(sweep.values()),
        "headline_w0.5": sweep[0.5]}

    # (b) cross-fitted learned combiner (image-disjoint eval folds) -- contaminated diagnostic
    import torch
    folds = G.eval_folds(5, items)
    mult = CC.pool_multiplicity(set(G.EVAL_DS))
    MU = np.array([[mult.get((it["ds"], it["idx"], G.norm(a)), 0) / 8.0 for a in it["preds"]]
                   for it in items])
    ND = np.array([[len(set(G.norm(a) for a in it["preds"]))] * 8 for it in items], float)
    F = np.stack([RI, RH, RI * RH, MU, ND / 8.0], -1).astype(np.float32)   # (n,8,5)
    Y = np.array([it["sl"] for it in items], dtype=np.float32)
    pred = np.zeros((len(items), 8))
    for f in range(5):
        trm = folds != f
        Ft = torch.tensor(F[trm].reshape(-1, F.shape[-1])); yt = torch.tensor(Y[trm].reshape(-1))
        w = torch.zeros(F.shape[-1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(Ft @ w + b, yt).backward()
            opt.step()
        pred[folds == f] = (F[folds == f] @ w.detach().numpy()) + float(b.detach().numpy()[0])
    lc = {(it["ds"], it["idx"]): list(pred[i]) for i, it in enumerate(items)}
    r_lc, r_lc_full = rep(lc, "learned combiner (cross-fitted ON EVAL -- diagnostic)")
    r_lc["vs_headline"] = vs(r_lc_full, r_head_full["got"], "learned combiner vs HEADLINE")
    r_lc["status"] = ("CONTAMINATED DIAGNOSTIC: fitted on eval with image-disjoint folds. It "
                      "shares the eval question distribution and cannot be a headline. Reported "
                      "because if even an eval-fitted combiner does not beat the parameter-free "
                      "fusion, no deployable learned combiner will.")
    diag["learned_combiner_crossfitted_on_eval"] = r_lc

    # (c) third member: self-consistency (free)
    sc_map = G.control_scores(items)["self_consistency"]
    r_sc3, _ = rep(G.rank_fuse(inc, ensmap, sc_map, items=items), "3-way rank_avg + self-consistency")
    diag["add_self_consistency_as_third_member"] = r_sc3

    # (d) fourth member: REAL A-vs-B pairwise, on the 1345 items it covers
    rows = [json.loads(l) for l in open(PM_HF) if l.strip()]
    key2i = {(it["ds"], it["idx"]): i for i, it in enumerate(items)}
    cov = np.zeros(len(items), bool)
    borda = {}
    for r in rows:
        cov[key2i[(r["ds"], r["idx"])]] = True
        P = np.asarray(r["P_avg"], float)
        b = P.sum(1) - 0.5
        slots = [0.0] * 8
        for ci_, ss in enumerate(r["slots"]):
            for s in ss:
                slots[s] = float(b[ci_])
        borda[(r["ds"], r["idx"])] = slots
    bmap = dict(inc); bmap.update(borda)
    fuse4 = G.rank_fuse(inc, ensmap, bmap, items=items)
    r4 = G.sel_eff(fuse4, items)
    m_cov = cov & (r_head_full["rec"] == 1)
    bb = G.paired_bootstrap(r4["got"], r_head_full["got"], nboot=A.nboot, mask=m_cov)
    diag["add_real_pairwise_as_fourth_member_on_covered_items"] = {
        "n_items_covered": int(cov.sum()), "n_recoverable_covered": int(m_cov.sum()),
        "headline_on_covered": float(r_head_full["got"][m_cov].mean()),
        "headline_plus_pairwise_on_covered": float(r4["got"][m_cov].mean()),
        "d": bb["d_sel_eff"], "ci": bb["d_sel_eff_ci"],
        "note": "real A-vs-B forward passes cost ~13 extra full VLM passes per question; this "
                "is what buying them adds on top of the headline, on the items they cover."}
    out["diagnostics"] = diag

    IL.jdump(out, os.path.join(IL.PARTS, "eval.json"))
    np.savez(os.path.join(IL.PARTS, "eval_got.npz"),
             headline=r_head_full["got"], deployed=rdep0["got"], incumbent=base["got"],
             head_ens=head_ens_res["got"], rec=base["rec"], short=short,
             contested=r_head_full["contested_mask"], ds_index=base["ds_index"])
    print(json.dumps({"HEADLINE": r_head, "vs_deployed": out["HEADLINE_vs_deployed_fusion"]},
                     indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
