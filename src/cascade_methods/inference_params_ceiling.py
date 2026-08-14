#!/usr/bin/env python3
"""inference_params_ceiling.py -- CLOSE THE LEVER.

The user's question is "can changing the 7B's inference parameters improve the samples
enough to matter?". A per-setting null answers it only for the settings that were run.
To close the lever rather than merely fail to move it, this measures the FREE UPPER
BOUNDS of the whole decoding grid:

  1. BEST-SINGLE-SETTING ceiling   -- the max over settings of SELECTED (what you could
     ship if you picked the best of the 8 knowing the answer key).
  2. PER-ITEM ORACLE-OVER-SETTINGS -- an oracle that picks the decoding parameter per
     question. Strictly above anything a router over decoding parameters could reach.
  3. UNION-POOL ORACLE             -- oracle over the union of all 8 settings x 3 seeds
     (up to 192 candidates per item). The coverage ceiling of the entire grid.
  4. UNION-POOL SELECTED           -- what the FROZEN verifier actually converts from
     that union. The gap between 3 and 4 is the selection wall, measured on the richest
     candidate distribution this project has ever built for these items.

Also runs the judge-source confound test: labels come from a preload cache (prior runs)
and from this session's judge, and the two sets are DISJOINT, so their agreement cannot
be measured directly. Hotter settings emit more novel answers and therefore draw a larger
share of their labels from the fresh judge. This measures that share per setting and
re-runs the headline delta restricted to slots whose label came from the PRELOAD cache
only -- a source-matched comparison.

Writes results/cascade_methods/artifacts/_infparams_ceiling.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.cascade_methods.inference_params_verify import (  # noqa: E402
    DEC, load_judge_map, load_vscores, build_items)

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
EVAL_DS = G.EVAL_DS
SETTINGS = ["T03", "T05", "T07", "T10", "T13", "minp01", "rp105", "rp11"]
CTRL = "T07"
NBOOT = 10000
SEED = 20260814
# equal weight per reporting cell, 8 cells (CLAUDE.md canonical convention)
MACRO_W = 1.0 / 8.0


def label_sources():
    """(ds, idx, na) -> 'preload' | 'fresh'."""
    src = {}
    for ds in EVAL_DS:
        with open(os.path.join(DEC, f"judgecache_preload_{ds}.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    src[(ds, str(o["idx"]), o["na"])] = "preload"
        jm = json.load(open(os.path.join(DEC, f"judgemap_{ds}.json")))
        for key in jm:
            idx, na = key.split("|", 1)
            src[(ds, idx, na)] = "fresh"
    return src


def main():
    judge = load_judge_map()
    vsc = load_vscores()
    src = label_sources()
    ref = G.load_items()
    order = [(it["ds"], str(it["idx"])) for it in ref]
    n = len(order)
    dsidx = np.array([EVAL_DS.index(d) for d, _ in order])

    pools = {}   # (setting, seed) -> items
    for s in SETTINGS:
        for sd in (0, 1, 2):
            it, mj, mv = build_items(s, sd, judge, vsc)
            assert it is not None and mj == 0 and mv == 0, (s, sd, mj, mv)
            pools[(s, sd)] = it
    print("pools built", len(pools), flush=True)

    out = {"title": "CEILING AND CONFOUND AUDIT of the 7B decoding-parameter lever",
           "date": "2026-08-14", "no_fabricated_numbers": True,
           "n_items": n, "cells": {d: int((dsidx == j).sum())
                                   for j, d in enumerate(EVAL_DS)}}

    # ---------------------------------------------------------------- confound: label source
    comp, restricted = {}, {}
    for s in SETTINGS:
        npre = nfre = 0
        for sd in (0, 1, 2):
            for it in pools[(s, sd)]:
                for a in it["preds"]:
                    k = (it["ds"], str(it["idx"]), G.norm(a))
                    if src.get(k) == "fresh":
                        nfre += 1
                    else:
                        npre += 1
        comp[s] = {"slots": npre + nfre, "from_preload_cache": npre,
                   "from_this_session_judge": nfre,
                   "fresh_share": nfre / (npre + nfre)}
    out["judge_label_source_composition"] = comp
    out["judge_source_confound"] = {
        "issue": "preload-cache labels and this-session judge labels are DISJOINT sets "
                 "(overlap = 0 keys), so their mutual agreement is unmeasurable. Hotter "
                 "settings emit more novel strings and draw more labels from the fresh "
                 "judge, so any systematic leniency difference between the two label "
                 "sources loads onto the temperature axis.",
        "preload_validation_cited": "judgepreload_report.json: 18733 agree / 0 disagree / "
                                    "27 not-in-preload against the frozen transfer dump "
                                    "(agreement_rate 1.0) -- the cache reproduces the "
                                    "frozen pool's labels exactly, which bounds but does "
                                    "not eliminate the concern.",
    }

    # source-matched delta: items where EVERY slot of BOTH arms is preload-labelled
    def all_preload(it):
        return all(src.get((it["ds"], str(it["idx"]), G.norm(a))) != "fresh"
                   for a in it["preds"])
    for s in SETTINGS:
        m = np.ones(n, dtype=bool)
        for sd in (0, 1, 2):
            m &= np.array([all_preload(it) for it in pools[(s, sd)]])
            m &= np.array([all_preload(it) for it in pools[(CTRL, sd)]])
        restricted[s] = m
    ms = restricted["T05"]
    out["judge_source_confound"]["source_matched_T05_vs_T07"] = {
        "n_items_all_slots_preload_in_both_arms": int(ms.sum()),
        "note": "restricting to items whose every candidate in BOTH arms was labelled by "
                "the SAME (cached) judge pass removes the source asymmetry entirely.",
    }

    # ---------------------------------------------------------------- build got/rec matrices
    got, rec = {}, {}
    for s in SETTINGS:
        gg, rr = [], []
        for sd in (0, 1, 2):
            it = pools[(s, sd)]
            sb = {(x["ds"], x["idx"]): x["scores"] for x in it}
            r = G.sel_eff(sb, items=it)
            gg.append(r["got"]); rr.append(r["rec"])
        got[s] = np.stack(gg); rec[s] = np.stack(rr)

    rng = np.random.default_rng(SEED)
    a, b = got["T05"], got[CTRL]
    idx = np.where(ms)[0]
    pt = a[:, idx].mean() - b[:, idx].mean()
    d = np.empty(NBOOT)
    for k in range(NBOOT):
        j = idx[rng.integers(0, len(idx), len(idx))]
        sa, sb2 = rng.integers(0, 3, 3), rng.integers(0, 3, 3)
        d[k] = a[np.ix_(sa, j)].mean() - b[np.ix_(sb2, j)].mean()
    out["judge_source_confound"]["source_matched_T05_vs_T07"].update({
        "d_selected": float(pt),
        "d_selected_ci95_seed_aware": [float(np.percentile(d, 2.5)),
                                       float(np.percentile(d, 97.5))]})
    print("source-matched T05:", pt, out["judge_source_confound"]
          ["source_matched_T05_vs_T07"]["d_selected_ci95_seed_aware"], flush=True)

    # ---------------------------------------------------------------- ceiling 1: best single
    lev = {}
    for s in SETTINGS:
        lev[s] = {"selected": float(got[s].mean()),
                  "oracle@8": float(rec[s].mean()),
                  "sel_eff": float(got[s][rec[s] == 1].mean()),
                  "per_cell_selected": {d_: float(got[s][:, dsidx == j].mean())
                                        for j, d_ in enumerate(EVAL_DS)}}
    out["levels_3seed_mean"] = lev
    best = max(SETTINGS, key=lambda s: lev[s]["selected"])
    out["ceiling_1_best_single_setting"] = {
        "argmax": best, "selected": lev[best]["selected"],
        "control_selected": lev[CTRL]["selected"],
        "gain_over_control": lev[best]["selected"] - lev[CTRL]["selected"],
        "note": "an ORACLE choice of setting -- picked by reading the eval endpoint. The "
                "honest ceiling of 'tune the decoding parameters' on this grid.",
    }

    # ---------------------------------------------------------------- ceiling 2: per-item oracle over settings
    # per item: correct if ANY setting's verifier pick is correct (seed-averaged as
    # 'correct on the majority of its seeds' AND as 'correct on any seed', both reported)
    stk_any = np.stack([got[s].max(axis=0) for s in SETTINGS])       # per setting: any seed
    stk_maj = np.stack([(got[s].mean(axis=0) >= 0.5).astype(int) for s in SETTINGS])
    out["ceiling_2_per_item_oracle_over_settings"] = {
        "oracle_over_settings_majority_seed": float(stk_maj.max(axis=0).mean()),
        "oracle_over_settings_any_seed": float(stk_any.max(axis=0).mean()),
        "gain_over_control_majority": float(stk_maj.max(axis=0).mean()
                                            - (got[CTRL].mean(axis=0) >= 0.5).mean()),
        "note": "an oracle that chooses the decoding parameter PER QUESTION. Strictly "
                "above anything a router over decoding parameters could achieve, and it "
                "is measured, not assumed.",
    }

    # ---------------------------------------------------------------- ceiling 3/4: union pool
    # union of all 8 settings x 3 seeds; dedup by normalized answer
    uni_rec, uni_sel, uni_ncand = [], [], []
    for i in range(n):
        ds, idx_ = order[i]
        seen = {}
        for s in SETTINGS:
            for sd in (0, 1, 2):
                it = pools[(s, sd)][i]
                for k2, aa in enumerate(it["preds"]):
                    na = G.norm(aa)
                    if na not in seen:
                        seen[na] = (it["sl"][k2], it["scores"][k2])
        ys = [v[0] for v in seen.values()]
        ss = [v[1] for v in seen.values()]
        uni_ncand.append(len(seen))
        uni_rec.append(1 if 1 in ys else 0)
        uni_sel.append(ys[int(np.argmax(ss))])
    uni_rec = np.array(uni_rec); uni_sel = np.array(uni_sel)
    out["ceiling_3_union_pool"] = {
        "mean_distinct_candidates_per_item": float(np.mean(uni_ncand)),
        "max_distinct": int(np.max(uni_ncand)),
        "oracle_union": float(uni_rec.mean()),
        "selected_union_frozen_verifier": float(uni_sel.mean()),
        "sel_eff_union": float(uni_sel[uni_rec == 1].mean()),
        "control_oracle@8": lev[CTRL]["oracle@8"],
        "control_selected": lev[CTRL]["selected"],
        "d_oracle_union_vs_control": float(uni_rec.mean() - lev[CTRL]["oracle@8"]),
        "d_selected_union_vs_control": float(uni_sel.mean() - lev[CTRL]["selected"]),
        "per_cell": {d_: {"oracle": float(uni_rec[dsidx == j].mean()),
                          "selected": float(uni_sel[dsidx == j].mean())}
                     for j, d_ in enumerate(EVAL_DS)},
        "note": "THE COVERAGE CEILING OF THE ENTIRE DECODING GRID: every candidate any "
                "swept setting produced at any seed, deduplicated, scored by the frozen "
                "verifier. 24x the deployed sampling budget.",
    }
    print("union pool:", out["ceiling_3_union_pool"], flush=True)

    # ---------------------------------------------------------------- macro translation
    macro = {}
    for s in SETTINGS:
        dc = {d_: lev[s]["per_cell_selected"][d_] - lev[CTRL]["per_cell_selected"][d_]
              for d_ in EVAL_DS}
        macro[s] = {"per_cell_d_selected": dc,
                    "macro8_contribution": float(sum(dc.values()) * MACRO_W),
                    "macro8_contribution_excl_vqa_rad": float(
                        (dc["slake_open"] + dc["pathvqa_open"]) * MACRO_W)}
    out["macro8_translation"] = macro
    out["macro8_note"] = ("equal weight per reporting cell, 8 cells, 1/8 each "
                          "(CLAUDE.md canonical). The open arm is 3 of 8 cells, so a "
                          "pooled open-text delta enters the macro at 3/8 weight and "
                          "only if the open arm is fully deployed with no escalation -- "
                          "an upper bound. Project significance threshold: +0.0029.")
    uc = {d_: out["ceiling_3_union_pool"]["per_cell"][d_]["selected"]
          - lev[CTRL]["per_cell_selected"][d_] for d_ in EVAL_DS}
    out["ceiling_macro8"] = {
        "best_single_setting": macro[best]["macro8_contribution"],
        "per_item_oracle_over_settings": float(
            (out["ceiling_2_per_item_oracle_over_settings"]
             ["oracle_over_settings_majority_seed"] - lev[CTRL]["selected"]) * 3 * MACRO_W),
        "union_pool_frozen_verifier": float(sum(uc.values()) * MACRO_W),
        "union_pool_oracle_coverage": float(
            (out["ceiling_3_union_pool"]["oracle_union"] - lev[CTRL]["oracle@8"])
            * 3 * MACRO_W),
    }
    print("ceilings (macro8):", out["ceiling_macro8"], flush=True)

    p = os.path.join(ART, "_infparams_ceiling.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
