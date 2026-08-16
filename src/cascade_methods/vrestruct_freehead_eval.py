#!/usr/bin/env python3
"""vrestruct_freehead_eval.py -- de-condition the "free head" row of the cost table.

Rows D-H of verifier_restructure_2026-08-16.json assume the generator-frame head's layer-21 state
can be CAPTURED during generation instead of recomputed in a separate teacher-forced pass, which
removes 3.8136 passes/question.  This round flagged a risk nobody had priced: the frozen heads were
fit on states extracted at max_pixels 1,003,520 (feats_hidden/*.meta.json) while generation runs at
250,880, so a real capture hands the heads a resolution they have never seen.

The sibling round's src/training_methods/free_head_capture.py has now written BOTH budgets, each
with the teacher-forced (TF) and captured-during-generation (AR) state for the SAME 8,943 rows:
    feats_free/free_fullres_L21.h_span_{tf,ar}.npy   (max_pixels 1,003,520)
    feats_free/free_cap320_L21.h_span_{tf,ar}.npy    (max_pixels   250,880)

This script runs the FROZEN 8-seed head + the frozen rank fusion over all four and reports sel_eff
and selected accuracy in BOTH currencies, per cell, with paired item bootstraps against the
deployed selector.  CPU only, no GPU, nothing refit.

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_freehead_eval.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import genframe_data as G      # noqa: E402
import vrestruct_lib as V      # noqa: E402

FEATDIR = os.path.join(ROOT, "feats_free")
OPEN_CELL_NAME = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
                  "pathvqa_open": "PATH_VQA_open"}
ARMS = [("fullres", "tf"), ("fullres", "ar"), ("cap320", "tf"), ("cap320", "ar")]


def load_free(cap, which, P):
    """(n_items, 8, 3584) -> the h_span vector of each SLOT, from the free-capture cache."""
    stem = os.path.join(FEATDIR, f"free_{cap}_L21")
    rows = [json.loads(l) for l in open(stem + ".rows.jsonl") if l.strip()]
    meta = json.load(open(stem + ".meta.json"))
    X = np.load(stem + f".h_span_{which}.npy", mmap_mode="r")
    key_to_row = {}
    for r in rows:
        key_to_row[(r["ds"], r["idx"], r["na"])] = r["row"]
    n = P["n"]
    slot_rows = np.empty((n, 8), int)
    for i, it in enumerate(P["items"]):
        for s, a in enumerate(it["preds"]):
            k = (it["ds"], it["idx"], G.norm(a))
            if k not in key_to_row:
                raise KeyError(f"{cap}/{which}: no captured row for {k}")
            slot_rows[i, s] = key_to_row[k]
    diag = dict(
        n_rows=len(rows), minutes=meta.get("minutes"), max_pixels=meta.get("max_pixels"),
        n_boundary_bad=int(sum(1 for r in rows if not r.get("boundary_ok", True))),
        n_span_bad=int(sum(1 for r in rows if not r.get("span_ok", True))),
        n_forced_bad=int(sum(1 for r in rows if not r.get("forced_ok", True))),
        d_span_max_p50=float(np.median([r.get("d_span_max", np.nan) for r in rows])),
        d_span_max_p99=float(np.percentile([r.get("d_span_max", 0.0) for r in rows], 99)),
        d_span_max_max=float(np.max([r.get("d_span_max", 0.0) for r in rows])))
    return np.asarray(X), slot_rows, diag


def head_rank_from(Xall, slot_rows, seeds=range(8)):
    S = V.selector()
    Z = S.standardize(np.asarray(Xall, np.float32))
    L = S.head_logits(Z, standardized=True)
    sub = L[list(seeds)]
    n = slot_rows.shape[0]
    out = np.empty((n, 8), float)
    for i in range(n):
        r = slot_rows[i]
        out[i] = np.mean([G.rank_avg(sub[s][r]) for s in range(sub.shape[0])], axis=0)
    return out


def main():
    P = V.load_pool()
    inc_rank = V.rank_rows(P["inc"])

    # the deployed reference: the frozen feats_hidden cache (teacher forced, 1,003,520)
    Lref = V.head_logits(P)
    HRref = V.head_rank_slots(P, Lref, range(8))
    ref_head = V.evaluate(P, V.picks_of(HRref), "head_only_deployed_cache")
    ref_fused = V.evaluate(P, V.picks_of(inc_rank + V.rank_rows(HRref)), "fused_deployed_cache")
    assert abs(ref_fused["judge"]["acc"] - 0.507463) < 1e-5, ref_fused["judge"]["acc"]

    out = {}
    for cap, which in ARMS:
        Xall, slot_rows, diag = load_free(cap, which, P)
        HR = head_rank_from(Xall, slot_rows)
        r_head = V.evaluate(P, V.picks_of(HR), f"head_only_{cap}_{which}")
        r_fus = V.evaluate(P, V.picks_of(inc_rank + V.rank_rows(HR)), f"fused_{cap}_{which}")
        row = dict(capture_diagnostics=diag)
        for nm, r, ref in (("head_only", r_head, ref_head), ("fused", r_fus, ref_fused)):
            row[nm] = {}
            for cur in ("judge", "em"):
                row[nm][cur] = dict(
                    sel_eff=r[cur]["sel_eff"], acc=r[cur]["acc"],
                    macro3=r[cur]["macro_cells"],
                    per_cell={OPEN_CELL_NAME[ds]: r[cur]["per_ds"][ds]["acc"]
                              for ds in G.EVAL_DS},
                    vs_deployed_cache=V.paired_boot(r[cur]["got"], ref[cur]["got"]),
                    per_cell_vs_deployed={OPEN_CELL_NAME[ds]: V.paired_boot(
                        r[cur]["got"][P["ds_index"] == j], ref[cur]["got"][P["ds_index"] == j])
                        for j, ds in enumerate(G.EVAL_DS)})
                row[nm][cur]["guardrail_clean"] = bool(all(
                    row[nm][cur]["per_cell_vs_deployed"][OPEN_CELL_NAME[ds]]["delta"] >= 0
                    for ds in G.EVAL_DS))
            row[nm]["n_picks_differing_from_deployed"] = int((r["picks"] != ref["picks"]).sum())
        out[f"{cap}_{which}"] = row
        print(f"  {cap:8s} {which}: head sel_eff {r_head['judge']['sel_eff']:.6f}  "
              f"fused sel_eff {r_fus['judge']['sel_eff']:.6f}  "
              f"fused accJ {r_fus['judge']['acc']:.6f}  accEM {r_fus['em']['acc']:.6f}  "
              f"picks!= {int((r_fus['picks'] != ref_fused['picks']).sum())}", flush=True)

    nt = {"NT_deployed_cache_reproduces_published": {
        "fused_acc": ref_fused["judge"]["acc"], "published": 0.507463,
        "fused_sel_eff": ref_fused["judge"]["sel_eff"], "published_sel_eff": 0.810627,
        "pass": bool(abs(ref_fused["judge"]["sel_eff"] - 0.810627) < 1e-5)},
        "NT_fullres_TF_should_match_the_deployed_cache": {
        "_what": "free_head_capture's TF path is extract_generator_hidden's path at the SAME "
                 "resolution, so fullres/tf is the harness null test: any gap is bf16/attention-"
                 "kernel noise, not a mechanism.",
        "fused_sel_eff_fullres_tf": out["fullres_tf"]["fused"]["judge"]["sel_eff"],
        "fused_sel_eff_deployed_cache": ref_fused["judge"]["sel_eff"],
        "abs_deviation": abs(out["fullres_tf"]["fused"]["judge"]["sel_eff"]
                             - ref_fused["judge"]["sel_eff"]),
        "n_picks_differing": out["fullres_tf"]["fused"]["n_picks_differing_from_deployed"]}}

    verdict = dict(
        the_question="does capturing the layer-21 state during generation, at the resolution "
                     "generation actually runs at, keep the frozen head working?",
        AR_vs_TF_at_fullres=out["fullres_ar"]["fused"]["judge"]["vs_deployed_cache"],
        AR_vs_TF_at_cap320=out["cap320_ar"]["fused"]["judge"]["vs_deployed_cache"],
        deployed_reference=dict(fused_acc_judge=ref_fused["judge"]["acc"],
                                fused_acc_em=ref_fused["em"]["acc"],
                                fused_sel_eff=ref_fused["judge"]["sel_eff"]))
    json.dump(dict(
        title="Does the FREE head survive? The frozen 8-seed head + fusion on captured-during-"
              "generation states, at both image budgets",
        date="2026-08-16", cpu_only=True, no_refit=True,
        inputs="feats_free/free_{fullres,cap320}_L21.h_span_{tf,ar}.npy, written by the sibling "
               "round's src/training_methods/free_head_capture.py (fullres 167.7 min / cap320 "
               "91.7 min, 8,943 rows each, 0 failed)",
        null_tests=nt, arms=out, verdict=verdict),
        open(os.path.join(V.PARTS, "freehead.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(V.PARTS, "freehead.json"))


if __name__ == "__main__":
    main()
