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


def _matched(cap, tag, control="cap320"):
    """True when this cap's arm and the control's arm for the same seed were generated in the same
    session (see resolution_arm_provenance.py). A False here means the delta carries the project's
    +-0.008 serving-config caveat on top of its own sampling noise and must not be a headline."""
    p = os.path.join(OUT, "arm_provenance.json")
    if not os.path.exists(p):
        return None
    pr = json.load(open(p)).get("pairs", {})
    return pr.get(f"{cap}_vs_{control}_{tag}", {}).get("WITHIN_SESSION_MATCHED")


def _pooled_map(op_vs, cap, seeds):
    """the POOLED (item-weighted) selected delta per seed, aligned to that cap's own seed list in
    open_generator_resolution -- reported next to the macro figure so the two bases are visibly
    different quantities, never silently swapped."""
    blk = op_vs.get(cap, {})
    order_seeds = blk.get("seeds_paired", [])
    vals = blk.get("per_metric", {}).get("selected", {}).get("delta_per_seed", [])
    out = {}
    for t in seeds:
        out[t] = vals[order_seeds.index(t)] if t in order_seeds and \
            order_seeds.index(t) < len(vals) else None
    return out


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

    # An arm whose candidates are not fully judge-labelled is biased DOWNWARD (an unlabelled
    # candidate is excluded from oracle@8 and counts as wrong if the verifier picks it), so such an
    # arm is DROPPED here rather than analysed. resolution_open_analyze.py reports the same arms
    # with a RELIABLE flag; this script simply refuses them, because its outputs (the macro-8
    # arithmetic and the laterality CIs) are quoted directly.
    MAXMISS = 0.01
    arms, skipped = {}, {}
    for cap, _ in CAPS:
        for tag in ["s0", "s1", "s2"]:
            A = load_arm(cap, tag)
            if A is None:
                continue
            slots = miss = 0
            for ds in DS:
                for r in A[ds].values():
                    for a in r["preds"]:
                        slots += 1
                        miss += int(f"{ds}|{r['idx']}|{norm(a)}" not in J)
            frac = miss / max(slots, 1)
            if frac > MAXMISS:
                skipped[f"{cap}_{tag}"] = {
                    "n_candidate_slots": slots, "slots_without_a_judge_label": miss,
                    "frac_unlabelled": round(frac, 6),
                    "_why": "dropped: >1% of this arm's candidates have no judge label, which "
                            "biases its oracle@8 and selected DOWNWARD. Label it and re-run."}
                continue
            arms[(cap, tag)] = arm_stats(A)

    op = json.load(open(os.path.join(OUT, "open_generator_resolution.json")))
    op_vs = op.get("vs_control", {})

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
    }, "per_cap": {}, "vs_control": {},
        "arms_dropped_for_incomplete_judge_labelling": skipped}

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

    # ---- macro-8 arithmetic for the open half, CELL-STRATIFIED --------------------------------
    # CORRECTION (2026-08-14): an earlier version of this block multiplied the POOLED
    # (item-weighted) delta by 3/8. That is wrong. The project's macro convention is EQUAL WEIGHT
    # PER CELL (8 cells, 1/8 each), while the pooled delta weights items -- and the open pool is
    # 645/200/1500, so pooling gives PathVQA 64% of the weight where the macro gives it 33%.
    # PathVQA is the cell with the SMALLEST gain, so pooling systematically understates the macro
    # effect here (native: pooled 0.005117 vs cell-mean 0.009623). The macro delta is therefore
    # rebuilt as the mean of the three per-cell deltas, with a CELL-STRATIFIED bootstrap: each
    # replicate resamples items WITHIN each cell, takes that cell's mean delta, and averages the
    # three cells equally -- the same weighting the reported macro-8 uses.
    dsmask = {ds: np.array([d == ds for d, _ in order]) for ds in DS}

    def macro_boot(a_vec, b_vec, nboot=NBOOT, seed=BSEED):
        rng = np.random.default_rng(seed)
        d = np.asarray(b_vec, float) - np.asarray(a_vec, float)
        cells = [d[dsmask[ds]] for ds in DS]
        point = float(np.mean([c.mean() for c in cells]))
        reps = np.zeros(nboot)
        for ci, c in enumerate(cells):
            idx = rng.integers(0, len(c), size=(nboot, len(c)))
            reps += c[idx].mean(axis=1)
        reps /= len(cells)
        return point, [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    macro = {}
    for cap, _px in CAPS:
        if cap == CONTROL:
            continue
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms and (CONTROL, t) in arms]
        if not seeds:
            continue
        rows = {}
        for t in seeds:
            a, b = arms[(CONTROL, t)], arms[(cap, t)]
            pt, ci = macro_boot(a["sel"], b["sel"])
            pc = {ds: round(float(b["sel"][dsmask[ds]].mean() - a["sel"][dsmask[ds]].mean()), 6)
                  for ds in DS}
            rows[t] = {
                "per_cell_delta_selected": pc,
                "open3_equal_weight_mean_delta": round(pt, 6),
                "open3_ci95": [round(ci[0], 6), round(ci[1], 6)],
                "macro8_equivalent": round(pt * 0.375, 6),
                "macro8_equivalent_ci95": [round(ci[0] * 0.375, 6), round(ci[1] * 0.375, 6)],
                "macro8_ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                "macro8_exceeds_project_threshold": bool(abs(pt * 0.375) > 0.0029),
                "WITHIN_SESSION_MATCHED": _matched(cap, t),
                # the project's standard robustness check: equal cell weighting hands a 200-item
                # cell a full third of the open weight, so a handful of items there can carry the
                # macro. Drop each cell in turn and re-weight the survivors equally.
                "leave_one_cell_out_open_mean_delta": {
                    ds: round(float(np.mean([pc[o] for o in DS if o != ds])), 6) for ds in DS},
                "leave_one_cell_out_macro8": {
                    ds: round(float(np.mean([pc[o] for o in DS if o != ds]) * 0.375), 6)
                    for ds in DS},
                "n_items_behind_each_cell_delta": {
                    ds: {"n": int(dsmask[ds].sum()),
                         "net_items_changed": int(round(pc[ds] * int(dsmask[ds].sum())))}
                    for ds in DS},
            }
        macro[cap] = {
            "max_pixels": _px, "seeds": seeds, "per_seed": rows,
            "open_cells_weight_in_macro8": 0.375,
            "project_significance_threshold_on_macro8": 0.0029,
            "_weighting": "EQUAL WEIGHT PER CELL, matching the project's macro-8 convention. The "
                          "bootstrap is stratified by cell: each replicate resamples items within "
                          "slake_open (645), vqa_rad_open (200) and pathvqa_open (1500) "
                          "separately, then averages the three cell means equally.",
            "_the_other_five_cells_are_held_at_zero": "resolution cannot move the 5 MCQ cells "
                                                      "upward -- they already run uncapped -- so "
                                                      "this is the whole macro-8 effect of a "
                                                      "generator-side resolution change.",
            "_pooled_for_contrast": _pooled_map(op_vs, cap, seeds),
            "_why_pooled_differs": "the pooled (item-weighted) delta is also reported in "
                                   "open_generator_resolution.vs_control. It answers a different "
                                   "question -- the average over QUESTIONS rather than over CELLS "
                                   "-- and it is NOT the macro-8 basis.",
        }
    res["macro8_arithmetic_open_half"] = macro

    # ---- per-cell deltas with their own paired CIs, for oracle@8 / selected / sel_eff ---------
    # open_generator_resolution reports per-cell LEVELS and a POOLED delta; this adds the per-cell
    # DELTA with an interval, which is what a per-cell guardrail actually needs.
    percell = {}
    for cap, px in CAPS:
        if cap == CONTROL:
            continue
        seeds = [t for t in ["s0", "s1", "s2"] if (cap, t) in arms and (CONTROL, t) in arms]
        if not seeds:
            continue
        blk = {"max_pixels": px, "seeds": seeds, "per_seed": {}}
        for t in seeds:
            a, b = arms[(CONTROL, t)], arms[(cap, t)]
            row = {}
            for ds in DS:
                m = dsmask[ds]
                e = {}
                for q, fld in [("oracle8", "orc"), ("selected", "sel")]:
                    d, ci = boot(np.asarray(b[fld][m], float) - np.asarray(a[fld][m], float))
                    e[q] = {"delta": round(d, 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
                            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
                rr = (a["rec"] == 1) & (b["rec"] == 1) & m
                d, ci = boot(np.asarray(b["sel"][rr], float) - np.asarray(a["sel"][rr], float))
                e["sel_eff_on_jointly_recoverable"] = {
                    "n_jointly_recoverable": int(rr.sum()),
                    "delta": round(d, 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
                    "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
                row[ds] = e
            blk["per_seed"][t] = row
            blk["per_seed"][t]["_WITHIN_SESSION_MATCHED"] = _matched(cap, t)
        percell[cap] = blk
    res["per_cell_deltas_with_ci"] = percell
    res["per_cell_deltas_with_ci"]["_read"] = (
        "paired item bootstrap (nboot=10000) of each cell's delta against the SAME-SEED cap320 "
        "control. sel_eff is bootstrapped inside the items recoverable in BOTH arms so the "
        "conditioning set is identical. Check _WITHIN_SESSION_MATCHED before quoting.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "strata_ci.json"), "w"), indent=1)
    print(json.dumps(res["per_cap"], indent=1))
    print(json.dumps(res["vs_control"], indent=1))
    print(json.dumps(macro, indent=1))
    print("wrote", os.path.join(OUT, "strata_ci.json"))


if __name__ == "__main__":
    main()
