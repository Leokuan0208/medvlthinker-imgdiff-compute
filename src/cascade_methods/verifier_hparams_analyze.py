#!/usr/bin/env python3
"""verifier_hparams_analyze.py -- KNOB 3 analysis: the verifier's scoring-resolution ladder.

Reads ckpts/openvqa/verifier_hparams/scores_px*.jsonl (one arm per max_pixels, produced by
verifier_hparams_score.py on the FROZEN deployed candidate pool) and reports, for every arm:

  * SELECTION EFFICIENCY and SELECTED accuracy in BOTH CURRENCIES
      - judge : the project's primary label (`sl` in the transfer dumps, 32B judge)
      - EM    : run_openvqa.py's normalized exact match, recomputed on the SAME picks
  * candidate-level AUROC in both currencies
  * paired item bootstrap (nboot=10000) against the IN-SESSION 1,003,520 control -- never
    against the stored dumps, because a batch-1 re-score of the stored pairs at the deployed
    resolution already deviates by max 6.03e-2 / mean 5.86e-3 (null_test_rescore.json)
  * the per-set GUARDRAIL
  * measured FLOPs / latency geometry per candidate

and then two leakage controls for the act of CHOOSING a rung:

  * NESTED CV      -- 5-fold: the rung is chosen on 4 folds, scored on the held-out fold, so
                      the reported number is the SELECTION PROCEDURE's, not the best arm's.
  * PERMUTATION NULL -- the same selection procedure run on labels shuffled WITHIN each item's
                      pool (which preserves every pool's recoverability and every arm's score
                      vector, and destroys only the score-label association). This project has
                      measured a per-cell "pick the best" rule earning +0.0109 macro from
                      shuffled labels alone, so this control is mandatory, not decorative.

CPU only.   python3 src/cascade_methods/verifier_hparams_analyze.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from src.training_methods import genframe_data as G   # noqa: E402

SCOREDIR = os.path.join(ROOT, "ckpts/openvqa/verifier_hparams")
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
CONTROL_PX = 1003520          # the DEPLOYED rung == the TRAINED rung (train_config.json)
GEN_PX = 250880               # what the GENERATOR renders at (run_openvqa.py cap320)
NBOOT = 10000
BSEED = 20260815
NPERM = 2000

LADDER = {62720: "cap80", 125440: "cap160", 250880: "cap320 (= GENERATOR's resolution)",
          376320: "cap480 (EXPLORATORY knee rung, added post-hoc)",
          501760: "cap640", 1003520: "fullres (DEPLOYED = TRAINED)",
          12845056: "native (no effective cap on these images)"}
#: the six rungs named in the pre-registration (_verifier_hparams_parts/prereg.json).
PREREG_RUNGS = [62720, 125440, 250880, 501760, 1003520, 12845056]


# =====================================================================================
# loading
# =====================================================================================
def load_arm(px, pfx=""):
    p = os.path.join(SCOREDIR, f"scores_{pfx}px{px}.jsonl")
    if not os.path.exists(p):
        return None
    sc, geo, nfail, patch = {}, [], 0, {}
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("p") is None:
            nfail += 1
            continue
        sc[(r["ds"], r["idx"], r["ans"])] = float(r["p"])
        patch[(r["ds"], r["idx"])] = int(r["patch"])
        geo.append((r["in_tok"], r["patch"], r["wall_s"]))
    g = np.array(geo, float)
    return {"scores": sc, "n_scored": len(sc), "n_failed": nfail, "patch": patch,
            "geometry": {"n": int(g.shape[0]),
                         "mean_input_tokens": float(g[:, 0].mean()),
                         "mean_premerge_patches": float(g[:, 1].mean()),
                         "mean_vision_tokens": float(g[:, 1].mean() / 4.0),
                         "max_vision_tokens": float(g[:, 1].max() / 4.0),
                         "mean_wall_s_batch1": float(g[:, 2].mean()),
                         "median_wall_s_batch1": float(np.median(g[:, 2])),
                         "_vision_token_note": "pixel_values rows are PRE-MERGE patches; "
                                               "Qwen2.5-VL applies a 2x2 spatial merge, so "
                                               "vision tokens = patches / 4."}}


def slot_scores(sc, items):
    """{(ds, idx) -> [8 scores]} by mapping each slot's RAW answer string to its score."""
    out, nmiss = {}, 0
    for it in items:
        v = []
        for a in it["preds"]:
            s = sc.get((it["ds"], it["idx"], a))
            if s is None:
                nmiss += 1
                s = np.nan
            v.append(s)
        out[(it["ds"], it["idx"])] = v
    return out, nmiss


# =====================================================================================
# the two currencies
# =====================================================================================
def endpoint(sc, items, em, judge):
    """Judge-currency via the frozen metric; EM-currency on the IDENTICAL picks."""
    sq, nmiss = slot_scores(sc, items)
    r = G.sel_eff(sq, items)
    picks = r["picks"]
    n = len(items)
    got_em = np.array([em[i, picks[i]] for i in range(n)], int)
    rec_em = (em.max(axis=1) == 1).astype(int)
    nd = r["n_distinct"]
    con_em = (rec_em == 1) & (nd >= 2)
    ds_index = r["ds_index"]
    per_ds_em = {}
    for j, ds in enumerate(G.EVAL_DS):
        m = ds_index == j
        per_ds_em[ds] = {"n": int(m.sum()), "acc": float(got_em[m].mean()),
                         "sel_eff": float(got_em[m & (rec_em == 1)].mean()),
                         "oracle": float(rec_em[m].mean())}
    # AUROC in both currencies over all 18,760 labelled slots
    S = G._slot_scores(sq, items)
    au_j = G.auroc(judge.ravel(), S.ravel())
    au_e = G.auroc(em.ravel(), S.ravel())
    return {
        "n_missing_slot_scores": nmiss,
        "judge": {"sel_eff": r["sel_eff"], "selected": r["acc"], "oracle@8": r["oracle"],
                  "greedy": r["greedy"], "n_recoverable": r["n_recoverable"],
                  "cand_auroc": float(au_j),
                  "per_ds_sel_eff": {d: r["per_ds"][d]["sel_eff"] for d in G.EVAL_DS},
                  "per_ds_selected": {d: r["per_ds"][d]["acc"] for d in G.EVAL_DS},
                  "contested_sel_eff": r["contested"]["sel_eff"],
                  "identity_oracle_x_sel_eff": float(r["oracle"] * r["sel_eff"]),
                  "identity_abs_dev": float(abs(r["oracle"] * r["sel_eff"] - r["acc"]))},
        "em": {"sel_eff": float(got_em[rec_em == 1].mean()), "selected": float(got_em.mean()),
               "oracle@8": float(rec_em.mean()), "n_recoverable": int(rec_em.sum()),
               "cand_auroc": float(au_e),
               "per_ds_sel_eff": {d: per_ds_em[d]["sel_eff"] for d in G.EVAL_DS},
               "per_ds_selected": {d: per_ds_em[d]["acc"] for d in G.EVAL_DS},
               "contested_sel_eff": float(got_em[con_em].mean()),
               "contested_n": int(con_em.sum())},
        "_vec": {"picks": picks, "got_j": r["got"], "rec_j": r["rec"],
                 "got_em": got_em, "rec_em": rec_em, "ds_index": ds_index,
                 "contested": r["contested_mask"], "contested_em": con_em, "S": S},
    }


def pboot(a, b, sub, nboot=NBOOT, seed=BSEED):
    """Paired item bootstrap of (a-b): accuracy over all items, sel_eff inside `sub`."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx_sub = np.where(np.asarray(sub, bool))[0]
    da = np.empty(nboot); de = np.empty(nboot)
    for k in range(nboot):
        s = rng.integers(0, n, n)
        da[k] = a[s].mean() - b[s].mean()
        j = idx_sub[rng.integers(0, len(idx_sub), len(idx_sub))]
        de[k] = a[j].mean() - b[j].mean()
    return {"d_selected": float(a.mean() - b.mean()),
            "d_selected_ci": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "d_sel_eff": float(a[idx_sub].mean() - b[idx_sub].mean()),
            "d_sel_eff_ci": [float(np.percentile(de, 2.5)), float(np.percentile(de, 97.5))],
            "nboot": nboot, "seed": seed}


# =====================================================================================
# leakage controls for CHOOSING a rung
# =====================================================================================
def picks_from_S(S):
    return np.argmax(S, axis=1).astype(int)


def sel_eff_from(picks, lab, rec, mask=None):
    got = np.array([lab[i, picks[i]] for i in range(len(picks))], int)
    m = (rec == 1) if mask is None else ((rec == 1) & mask)
    return float(got[m].mean()), got


def nested_cv(Smap, lab, rec, folds, ds_index, per_cell=False):
    """5-fold nested selection.  Per fold: choose the rung (globally, or per open cell) on the
    OTHER 4 folds; score the choice on the held-out fold.  Returns the pooled held-out sel_eff
    of the PROCEDURE and the chosen rungs."""
    names = list(Smap)
    picks = {k: picks_from_S(Smap[k]) for k in names}
    got = {k: np.array([lab[i, picks[k][i]] for i in range(lab.shape[0])], int) for k in names}
    n = lab.shape[0]
    out = np.zeros(n, int)
    chosen = []
    for f in sorted(set(folds)):
        te = folds == f
        tr = ~te
        if per_cell:
            ch = {}
            for j in range(3):
                cm = tr & (ds_index == j) & (rec == 1)
                ch[j] = max(names, key=lambda k: got[k][cm].mean() if cm.sum() else 0.0)
            for j in range(3):
                m = te & (ds_index == j)
                out[m] = got[ch[j]][m]
            chosen.append({G.EVAL_DS[j]: ch[j] for j in range(3)})
        else:
            cm = tr & (rec == 1)
            ch = max(names, key=lambda k: got[k][cm].mean())
            out[te] = got[ch][te]
            chosen.append(ch)
    m = rec == 1
    return float(out[m].mean()), float(out.mean()), chosen


def permutation_null(Smap, lab, rec, folds, ds_index, nperm=NPERM, seed=BSEED, per_cell=False):
    """The SAME selection procedure on labels shuffled WITHIN each item's pool.

    Shuffling within the pool preserves (a) every item's recoverability, (b) every item's number
    of correct slots, and (c) every arm's score vector -- and destroys ONLY the score<->label
    association.  Under this null every arm has the same expected sel_eff (random pick), so any
    positive 'gain' the selection procedure earns is manufactured by the act of choosing.
    """
    rng = np.random.default_rng(seed)
    names = list(Smap)
    picks = {k: picks_from_S(Smap[k]) for k in names}
    n = lab.shape[0]
    ar = np.arange(n)
    ctrl = names.index(CONTROL_PX) if CONTROL_PX in names else 0
    gains_insample, gains_nested = [], []
    for _ in range(nperm):
        # independent within-row permutation of the 8 slots, vectorized
        L = np.take_along_axis(lab, np.argsort(rng.random(lab.shape), axis=1), axis=1)
        got = {k: L[ar, picks[k]].astype(int) for k in names}
        m = rec == 1
        base = got[names[ctrl]][m].mean()
        gains_insample.append(max(got[k][m].mean() for k in names) - base)
        # the nested version of the same procedure
        out = np.zeros(n, int)
        for f in sorted(set(folds)):
            te = folds == f
            tr = ~te
            if per_cell:
                for j in range(3):
                    cm = tr & (ds_index == j) & m
                    ch = max(names, key=lambda k: got[k][cm].mean() if cm.sum() else 0.0)
                    mm = te & (ds_index == j)
                    out[mm] = got[ch][mm]
            else:
                cm = tr & m
                ch = max(names, key=lambda k: got[k][cm].mean())
                out[te] = got[ch][te]
        gains_nested.append(out[m].mean() - base)
    gi = np.array(gains_insample); gn = np.array(gains_nested)
    return {"nperm": nperm,
            "in_sample_max_over_rungs_gain": {"mean": float(gi.mean()), "sd": float(gi.std(ddof=1)),
                                              "p95": float(np.percentile(gi, 95)),
                                              "p99": float(np.percentile(gi, 99)),
                                              "max": float(gi.max())},
            "nested_cv_gain": {"mean": float(gn.mean()), "sd": float(gn.std(ddof=1)),
                               "p95": float(np.percentile(gn, 95)),
                               "p99": float(np.percentile(gn, 99)),
                               "max": float(gn.max())},
            "_dist_insample": gi, "_dist_nested": gn}


# =====================================================================================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="", help="'base_' selects the no-adapter control arms")
    ap.add_argument("--out", default="ladder.json")
    ap.add_argument("--no_leakage", action="store_true",
                    help="skip nested CV / permutation null (nothing is selected on that arm set)")
    ap.add_argument("--no_dumps", action="store_true",
                    help="skip writing per-rung transfer dumps for the macro re-run")
    ap.add_argument("--rungs", default="",
                    help="comma-separated max_pixels to restrict to. Used to reproduce the "
                         "PRE-REGISTERED six-rung arm set exactly after the exploratory "
                         "376,320 knee rung was added, because the nested-CV / permutation-null "
                         "leakage controls are properties of the ARM SET, not of one arm.")
    A = ap.parse_args()
    pfx = A.prefix
    only = {int(x) for x in A.rungs.split(",") if x.strip()} if A.rungs.strip() else None

    os.makedirs(PARTS, exist_ok=True)
    items = G.load_items()
    d = np.load(os.path.join(PARTS, "em_slots.npz"))
    em, judge = d["em"], d["judge"]

    pat = f"scores_{pfx}px*.jsonl"
    pxs = sorted(int(os.path.basename(f)[len(f"scores_{pfx}px"):-len(".jsonl")])
                 for f in glob.glob(os.path.join(SCOREDIR, pat)))
    if only is not None:
        pxs = [p for p in pxs if p in only]
    arms, EP = {}, {}
    for px in pxs:
        a = load_arm(px, pfx)
        if a is None or a["n_scored"] < 8965:
            print(f"  px{px}: INCOMPLETE ({a['n_scored'] if a else 0}/8965) -- skipped")
            continue
        arms[px] = a
        EP[px] = endpoint(a["scores"], items, em, judge)
        print(f"  px{px:>9}  judge sel_eff {EP[px]['judge']['sel_eff']:.6f}  "
              f"EM sel_eff {EP[px]['em']['sel_eff']:.6f}  "
              f"vis_tok {a['geometry']['mean_vision_tokens']:.1f}  "
              f"{a['geometry']['mean_wall_s_batch1']*1000:.1f} ms")
    if CONTROL_PX not in EP:
        print("CONTROL ARM px1003520 NOT COMPLETE -- cannot report matched deltas yet")
        json.dump({"_status": "incomplete", "arms_complete": sorted(EP)},
                  open(os.path.join(PARTS, A.out), "w"), indent=1)
        return

    ctrl = EP[CONTROL_PX]["_vec"]
    rep = {"_arms": {}, "_control_px": CONTROL_PX}
    for px in sorted(EP):
        e = EP[px]
        v = e["_vec"]
        row = {"max_pixels": px, "label": LADDER.get(px, str(px)),
               "trained_at_this_resolution": px == CONTROL_PX,
               "matches_generator_resolution": px == GEN_PX,
               "n_scored": arms[px]["n_scored"], "n_failed": arms[px]["n_failed"],
               "n_missing_slot_scores": e["n_missing_slot_scores"],
               "geometry": arms[px]["geometry"],
               "judge": e["judge"], "em": e["em"],
               "n_picks_differing_from_control": int((v["picks"] != ctrl["picks"]).sum())}
        if px != CONTROL_PX:
            row["vs_control_judge"] = pboot(v["got_j"], ctrl["got_j"], ctrl["rec_j"] == 1)
            row["vs_control_em"] = pboot(v["got_em"], ctrl["got_em"], ctrl["rec_em"] == 1)
            row["vs_control_judge_contested"] = pboot(v["got_j"], ctrl["got_j"],
                                                      ctrl["contested"])
            row["vs_control_em_contested"] = pboot(v["got_em"], ctrl["got_em"],
                                                   ctrl["contested_em"])
            gr = {ds: e["judge"]["per_ds_sel_eff"][ds] -
                      EP[CONTROL_PX]["judge"]["per_ds_sel_eff"][ds] for ds in G.EVAL_DS}
            gre = {ds: e["em"]["per_ds_sel_eff"][ds] -
                       EP[CONTROL_PX]["em"]["per_ds_sel_eff"][ds] for ds in G.EVAL_DS}
            row["guardrail_judge"] = {"per_ds_delta": gr,
                                      "clean": all(x >= 0 for x in gr.values()),
                                      "flags": [d_ for d_, x in gr.items() if x < 0]}
            row["guardrail_em"] = {"per_ds_delta": gre,
                                   "clean": all(x >= 0 for x in gre.values()),
                                   "flags": [d_ for d_, x in gre.items() if x < 0]}

            # ---- THE FREE INTERNAL PLACEBO ------------------------------------------
            # An image whose native pixel count is already at or below BOTH caps is
            # rendered IDENTICALLY by both arms (qwen smart_resize only shrinks), so the
            # verifier sees the same tensor and must return the same score. Those items
            # are a placebo stratum measured inside this very comparison: whatever delta
            # appears there is the numerical noise floor, and the real effect must live
            # entirely in the stratum where the rendering actually changed.
            pc, pk = arms[CONTROL_PX]["patch"], arms[px]["patch"]
            same = np.array([1 if pc.get((it["ds"], it["idx"])) ==
                             pk.get((it["ds"], it["idx"])) else 0 for it in items], int)
            row["identical_rendering_placebo"] = {
                "_what": "items whose image renders to the SAME number of patches at both "
                         "resolutions -- the verifier input is byte-identical there.",
                "n_items_identical_rendering": int(same.sum()),
                "n_items_rendering_changed": int((1 - same).sum()),
                "frac_identical": float(same.mean()),
                "n_score_differences_on_identical_rendering": int(sum(
                    1 for i_, it in enumerate(items) if same[i_] == 1
                    for a in it["preds"]
                    if abs(arms[px]["scores"].get((it["ds"], it["idx"], a), 0.0) -
                           arms[CONTROL_PX]["scores"].get((it["ds"], it["idx"], a), 0.0)) > 1e-9)),
                "n_slots_on_identical_rendering": int(sum(
                    len(it["preds"]) for i_, it in enumerate(items) if same[i_] == 1)),
                "d_sel_eff_judge_on_identical_rendering": (
                    float(v["got_j"][(same == 1) & (ctrl["rec_j"] == 1)].mean() -
                          ctrl["got_j"][(same == 1) & (ctrl["rec_j"] == 1)].mean())
                    if ((same == 1) & (ctrl["rec_j"] == 1)).sum() else None),
                "d_sel_eff_judge_on_changed_rendering": (
                    float(v["got_j"][(same == 0) & (ctrl["rec_j"] == 1)].mean() -
                          ctrl["got_j"][(same == 0) & (ctrl["rec_j"] == 1)].mean())
                    if ((same == 0) & (ctrl["rec_j"] == 1)).sum() else None),
                "n_recoverable_identical": int(((same == 1) & (ctrl["rec_j"] == 1)).sum()),
                "n_recoverable_changed": int(((same == 0) & (ctrl["rec_j"] == 1)).sum()),
            }
        rep["_arms"][str(px)] = row

    # ---- leakage controls -------------------------------------------------------------
    if A.no_leakage:
        json.dump(rep, open(os.path.join(PARTS, A.out), "w"), indent=1, default=float)
        print(f"\nwrote {PARTS}/{A.out}  (leakage controls skipped: nothing is selected here)")
        return
    Smap = {px: EP[px]["_vec"]["S"] for px in EP}
    folds = G.eval_folds(5, items)
    rec_j = ctrl["rec_j"]
    ds_index = ctrl["ds_index"]
    base_j = EP[CONTROL_PX]["judge"]["sel_eff"]
    best_px = max(EP, key=lambda k: EP[k]["judge"]["sel_eff"])
    ncv_g, ncv_acc_g, ch_g = nested_cv(Smap, judge, rec_j, folds, ds_index, per_cell=False)
    ncv_c, ncv_acc_c, ch_c = nested_cv(Smap, judge, rec_j, folds, ds_index, per_cell=True)
    print("  running permutation null (judge, within-pool label shuffle)...", flush=True)
    pn_g = permutation_null(Smap, judge, rec_j, folds, ds_index, per_cell=False)
    pn_c = permutation_null(Smap, judge, rec_j, folds, ds_index, per_cell=True)

    def pval(obs, dist):
        return float((np.asarray(dist) >= obs).mean())

    rep["selection_leakage_controls"] = {
        "_what": "the ONLY thing selected in this round is WHICH RUNG to deploy. The headline "
                 "comparison (cap320 vs the deployed 1,003,520) is PRE-SPECIFIED by the task and "
                 "involves no selection; these controls price the extra claim 'and this rung is "
                 "the best one'.",
        "control_rung_sel_eff_judge": base_j,
        "in_sample_best_rung": {"px": best_px, "sel_eff": EP[best_px]["judge"]["sel_eff"],
                                "gain_vs_control": EP[best_px]["judge"]["sel_eff"] - base_j},
        "nested_cv_global_rung": {"sel_eff": ncv_g, "selected": ncv_acc_g,
                                  "gain_vs_control": ncv_g - base_j, "chosen_per_fold": ch_g},
        "nested_cv_per_cell_rung": {"sel_eff": ncv_c, "selected": ncv_acc_c,
                                    "gain_vs_control": ncv_c - base_j, "chosen_per_fold": ch_c},
        "permutation_null_global_rung": {k: v for k, v in pn_g.items() if not k.startswith("_")},
        "permutation_null_per_cell_rung": {k: v for k, v in pn_c.items() if not k.startswith("_")},
        "p_values": {
            "in_sample_best_vs_null": pval(EP[best_px]["judge"]["sel_eff"] - base_j,
                                           pn_g["_dist_insample"]),
            "nested_global_vs_null": pval(ncv_g - base_j, pn_g["_dist_nested"]),
            "nested_per_cell_vs_null": pval(ncv_c - base_j, pn_c["_dist_nested"]),
        },
        "_null_construction": "labels shuffled WITHIN each item's 8-slot pool (2000 permutations); "
                              "preserves recoverability and per-item positive count, destroys only "
                              "the score-label association.",
        "_folds": "src/training_methods/genframe_data.eval_folds(5) -- the project's canonical "
                  "item folds.",
    }

    np.savez_compressed(os.path.join(PARTS, "perm_null_dists.npz"),
                        global_insample=pn_g["_dist_insample"], global_nested=pn_g["_dist_nested"],
                        cell_insample=pn_c["_dist_insample"], cell_nested=pn_c["_dist_nested"])
    json.dump(rep, open(os.path.join(PARTS, A.out), "w"), indent=1, default=float)
    print(f"\nwrote {PARTS}/{A.out}")

    # ---- write transfer dumps for the macro re-run ------------------------------------
    if A.no_dumps:
        return
    for px in EP:
        outd = os.path.join(ROOT, f"ckpts/train/verifhp_px{px}")
        os.makedirs(outd, exist_ok=True)
        sq, _ = slot_scores(arms[px]["scores"], items)
        for nm, ds in [("slake", "slake_open"), ("vqa_rad", "vqa_rad_open"),
                       ("pathvqa", "pathvqa_open")]:
            src = json.load(open(os.path.join(G.DUMP_DIR,
                                              f"transfer_dump_{nm}_open_lingshu7b.json")))
            for r in src:
                s = sq[(r["ds"], r["idx"])]
                r["scores"] = [float(x) for x in s]
                r["pick"] = int(np.argmax(s))
            json.dump(src, open(os.path.join(outd, f"transfer_dump_{ds}_lingshu7b.json"), "w"))
    print(f"wrote per-rung transfer dumps to ckpts/train/verifhp_px*/")


if __name__ == "__main__":
    main()
