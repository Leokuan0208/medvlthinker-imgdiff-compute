#!/usr/bin/env python3
"""resolution_strata_ci.py -- SWEEP 2 supplement: paired bootstrap CIs on the LATERALITY stratum
and the macro-8 arithmetic for the open half, per generator resolution.

resolution_open_analyze.py reports the strata as point estimates only. The laterality stratum
(n=283) is the project's weakest verifier stratum and the one a resolution change is most likely
to touch, so it needs an interval, not a point. This recomputes the same arm vectors on the same
canonical item order and bootstraps:

  * sel_eff on the laterality stratum, native vs the cap320 control, PAIRED on the items that are
    recoverable in BOTH arms (identical conditioning set), seed-matched;
  * oracle@8 and selected on the laterality stratum, paired on all stratum items;
  * the laterality-vs-non-laterality sel_eff GAP per cap, and its change vs control.

and prints the macro-8 arithmetic that converts an open-3 selected delta into macro-8 points.

    python3 src/cascade_methods/resolution_strata_ci.py
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
CAPS = [("cap80", 62720), ("cap160", 125440), ("cap320", 250880), ("cap640", 501760),
        ("fullres", 1003520), ("native", 12845056)]
CONTROL = "cap320"
NBOOT, BSEED = 10000, 20260814
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}

from src.training_methods.visverif_lib import LATERAL  # noqa: E402


def norm(s):
    return str(s).strip().lower()


def load_arm(cap, tag):
    out = {}
    for ds in DS:
        p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{tag}.jsonl")
        if not os.path.exists(p):
            return None
        d = {}
        for l in open(p):
            if l.strip():
                try:
                    r = json.loads(l)
                    d[r["idx"]] = r
                except Exception:
                    pass
        if len(d) < NEXP[ds]:
            return None
        out[ds] = d
    return out


def boot(d, nboot=NBOOT, seed=BSEED):
    d = np.asarray(d, float)
    if len(d) == 0:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    J = json.load(open(os.path.join(SWEEP, "judge_cache.json")))
    V = json.load(open(os.path.join(SWEEP, "verifier_score_cache.json")))

    order = []
    for ds, nm in [("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"), ("pathvqa_open", "pathvqa")]:
        p = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint",
                         f"transfer_dump_{nm}_open_lingshu7b.json")
        for r in json.load(open(p)):
            order.append((ds, r["idx"]))
    assert len(order) == 2345, len(order)

    def arm_stats(arm):
        rec, sel, orc = [], [], []
        gold, ques = [], []
        for ds, idx in order:
            r = arm[ds].get(idx)
            if r is None:
                rec.append(0); sel.append(0); orc.append(0); gold.append(""); ques.append("")
                continue
            preds = r["preds"]
            y = [J.get(f"{ds}|{idx}|{norm(a)}") for a in preds]
            sc = [V.get(f"{ds}|{idx}|{a}") for a in preds]
            yv = [0 if v is None else int(v) for v in y]
            svv = [-1e9 if v is None else float(v) for v in sc]
            k = int(np.argmax(svv))
            labelled = [int(v) for v in y if v is not None]
            o = int(max(labelled) == 1) if labelled else 0
            orc.append(o); rec.append(o); sel.append(int(yv[k] == 1))
            gold.append(str(r["gold"])); ques.append(str(r["question"]))
        return dict(rec=np.array(rec), sel=np.array(sel), orc=np.array(orc), gold=gold, ques=ques)

    arms = {}
    for cap, _ in CAPS:
        for tag in ["s0", "s1", "s2"]:
            A = load_arm(cap, tag)
            if A is not None:
                arms[(cap, tag)] = arm_stats(A)

    ref = arms[(CONTROL, "s0")]
    goldlen = np.array([len(g.split()) for g in ref["gold"]])
    lat = np.array([bool(LATERAL.search(q)) or bool(LATERAL.search(g))
                    for q, g in zip(ref["ques"], ref["gold"])])
    nonlat = (goldlen <= 3) & ~lat

    res = {"_meta": {
        "what": "paired bootstrap CIs on the laterality stratum, per generator resolution, on the "
                "judge-labelled endpoint. Masks are built from QUESTION and GOLD only, so they are "
                "ARM-INVARIANT and the same items are compared at every cap.",
        "control": CONTROL, "nboot": NBOOT, "bootstrap_seed": BSEED,
        "laterality_regex": "src/training_methods/visverif_lib.LATERAL",
        "n_laterality": int(lat.sum()), "n_short3_not_laterality": int(nonlat.sum()),
        "published_reference": "the project's laterality sel_eff 0.613043 vs 0.817186 on short "
                               "non-laterality items comes from the DEPLOYED transfer dump, whose "
                               "laterality mask also ORs in candidate text; these masks are "
                               "question/gold-only so the absolute levels are not identical to it.",
    }, "per_cap": {}, "vs_control": {}}

    for cap, px in CAPS:
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms]
        if not seeds:
            continue
        row = {"max_pixels": px, "seeds": seeds}
        for nm, msk in [("laterality", lat), ("short3_not_laterality", nonlat)]:
            se, o8, sl = [], [], []
            for t in seeds:
                S = arms[(cap, t)]
                rr = (S["rec"] == 1) & msk
                se.append(float(S["sel"][rr].mean()) if rr.sum() else float("nan"))
                o8.append(float(S["orc"][msk].mean()))
                sl.append(float(S["sel"][msk].mean()))
            row[nm] = {"n_items": int(msk.sum()),
                       "sel_eff_mean": round(float(np.mean(se)), 6),
                       "sel_eff_per_seed": [round(x, 6) for x in se],
                       "oracle8_mean": round(float(np.mean(o8)), 6),
                       "selected_mean": round(float(np.mean(sl)), 6)}
        row["sel_eff_gap_nonlat_minus_lat"] = round(
            row["short3_not_laterality"]["sel_eff_mean"] - row["laterality"]["sel_eff_mean"], 6)
        res["per_cap"][cap] = row

    for cap, px in CAPS:
        if cap == CONTROL or not any((cap, t) in arms for t in ["s0", "s1", "s2"]):
            continue
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms and (CONTROL, t) in arms]
        if not seeds:
            continue
        blk = {"max_pixels": px, "seeds_paired": seeds, "laterality": {}}
        for q, fld in [("oracle8", "orc"), ("selected", "sel")]:
            ds_, cis = [], []
            for t in seeds:
                a, b = arms[(CONTROL, t)], arms[(cap, t)]
                d, ci = boot(np.asarray(b[fld][lat], float) - np.asarray(a[fld][lat], float))
                ds_.append(d); cis.append(ci)
            blk["laterality"][q] = {
                "delta_mean_over_seeds": round(float(np.mean(ds_)), 6),
                "delta_per_seed": [round(x, 6) for x in ds_],
                "ci95_per_seed": [[round(c[0], 6), round(c[1], 6)] for c in cis],
                "all_seeds_ci_exclude_zero": bool(all(c[0] > 0 or c[1] < 0 for c in cis))}
        se_d, se_ci = [], []
        for t in seeds:
            a, b = arms[(CONTROL, t)], arms[(cap, t)]
            m = (a["rec"] == 1) & (b["rec"] == 1) & lat
            d, ci = boot(np.asarray(b["sel"][m], float) - np.asarray(a["sel"][m], float))
            se_d.append(d); se_ci.append(ci)
        blk["laterality"]["sel_eff_on_jointly_recoverable"] = {
            "n_jointly_recoverable_mean": float(np.mean(
                [int((((arms[(CONTROL, t)]["rec"] == 1) & (arms[(cap, t)]["rec"] == 1) & lat)).sum())
                 for t in seeds])),
            "delta_mean_over_seeds": round(float(np.mean(se_d)), 6),
            "delta_per_seed": [round(x, 6) for x in se_d],
            "ci95_per_seed": [[round(c[0], 6), round(c[1], 6)] for c in se_ci],
            "all_seeds_ci_exclude_zero": bool(all(c[0] > 0 or c[1] < 0 for c in se_ci))}
        res["vs_control"][cap] = blk

    # ---- macro-8 arithmetic for the open half ------------------------------------------------
    op = json.load(open(os.path.join(OUT, "open_generator_resolution.json")))
    macro = {}
    for cap, blk in op["vs_control"].items():
        d = blk["per_metric"]["selected"]
        macro[cap] = {
            "max_pixels": blk["max_pixels"],
            "delta_selected_open3_mean_over_cells": d["delta_mean_over_seeds"],
            "ci95_per_seed": d["ci95_per_seed"],
            "open_cells_weight_in_macro8": 0.375,
            "macro8_equivalent": round(d["delta_mean_over_seeds"] * 0.375, 6),
            "macro8_equivalent_ci95_per_seed": [[round(c[0] * 0.375, 6), round(c[1] * 0.375, 6)]
                                                for c in d["ci95_per_seed"]],
            "project_significance_threshold_on_macro8": 0.0029,
            "_read": "the 3 open cells are 3/8 of the macro-8 and the delta above is their MEAN, so "
                     "the macro-8 contribution is delta x 3/8. The other 5 cells are held at zero.",
        }
    res["macro8_arithmetic_open_half"] = macro

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "strata_ci.json"), "w"), indent=1)
    print(json.dumps(res["per_cap"], indent=1))
    print(json.dumps(res["vs_control"], indent=1))
    print(json.dumps(macro, indent=1))
    print("wrote", os.path.join(OUT, "strata_ci.json"))


if __name__ == "__main__":
    main()
