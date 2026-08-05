#!/usr/bin/env python3
"""export_selector_dumps.py -- write the frozen 8-seed selector's per-slot scores into the
transfer-dump schema the CASCADE reads, so the open-text arm can be re-run end-to-end without
changing any cascade mechanics.

The cascade's open arm (integrated_method.open_bestof8, integrated_pandora.load_open_rows,
beat32b_more.open_features) all read

    ckpts/train/<verifier>/transfer_dump_{ds}_lingshu7b.json
      -> [{ds, idx, sl[8], scores[8], pick, greedy_ok, preds[8]}, ...]

and use `scores` for BOTH the pick and the escalation gate.  This script emits two new
directories with `sl`, `preds`, `greedy_ok` and item order copied verbatim from the CLEAN
disjoint dumps -- only `scores`/`pick` change:

  selector_ens8_rank    scores = rank_avg(incumbent) + rank_avg(8-seed head rank)
                        THE RECOMMENDATION, VERBATIM.  Range [0,2].  Note what this means for
                        the cascade: a rank is a WITHIN-POOL quantity, so `max(scores)` -- the
                        escalation gate -- is very nearly constant across questions and carries
                        almost no cross-question confidence.  Reported as the UNTUNED re-run.

  selector_ens8_scaled  the SAME within-pool ORDERING, with the magnitudes quantile-matched onto
                        the incumbent's own 8 scores for that pool (descending stable sort;
                        s'[o[i]] = sorted(inc, desc)[i]).  The pool's score MULTISET is therefore
                        byte-identical to the incumbent's, so every cross-question gate feature
                        (max / mean / std / range, and the isotonic calibration domain) is
                        unchanged and ONLY the pick moves.  This is the honest refit that
                        isolates the selection change from the scale change.

Both are derived from the CLEAN `lora_verifier_disjoint` incumbent, not the contaminated
`lora_verifier_pooled4` the published cascade reads.

    python3 src/training_methods/export_selector_dumps.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G          # noqa: E402
import genframe_selector as GS     # noqa: E402

CASCADE_DS = {"slake_open": "slake", "vqa_rad_open": "vqa_rad", "pathvqa_open": "pathvqa"}
OUT_RANK = os.path.join(G.ROOT, "ckpts/train/selector_ens8_rank")
OUT_SCAL = os.path.join(G.ROOT, "ckpts/train/selector_ens8_scaled")


TIE_EPS = 1e-9   # incumbent scores are stored 5-dp rounded, so 1e-9 is 4 orders below resolution


def scale_match(c, inc):
    """Keep c's within-pool ORDER, take inc's magnitudes.

    Descending stable sort so first-index ties resolve the same way np.argmax does.  The
    incumbent's own scores contain exact ties (5-dp rounding + duplicated answers share a
    score), so a bare quantile match can hand the max to a slot the selector did not pick;
    a strictly decreasing 1e-9 offset in selector order removes that without perturbing any
    gate feature at the 5th decimal.
    """
    c = np.asarray(c, float); inc = np.asarray(inc, float)
    n = len(c)
    o = np.argsort(-c, kind="mergesort")
    out = np.empty_like(inc)
    out[o] = np.sort(inc)[::-1] + TIE_EPS * np.arange(n - 1, -1, -1, dtype=float)
    return out


def main():
    sel = GS.FrozenSelector.load()
    items = G.load_items()
    smap, _ = GS.score_eval_pool(sel)

    # sanity: the selector reproduces its published endpoint before anything is exported
    r = G.sel_eff(smap, items)
    dev = abs(r["sel_eff"] - GS.PUBLISHED["sel_eff"])
    assert dev < 1e-5, f"selector does not reproduce: {r['sel_eff']} vs {GS.PUBLISHED['sel_eff']}"
    print(f"selector reproduces sel_eff {r['sel_eff']!r} (dev {dev:.3e})", flush=True)

    os.makedirs(OUT_RANK, exist_ok=True)
    os.makedirs(OUT_SCAL, exist_ok=True)
    src = G.DUMP_DIR
    stats = {}
    for ds, tag in CASCADE_DS.items():
        recs = json.load(open(os.path.join(src, f"transfer_dump_{tag}_open_lingshu7b.json")))
        rank_recs, scal_recs = [], []
        n_pickdiff_inc = n_pickdiff_scale = n_outcome_diff = 0
        for rec in recs:
            key = (rec["ds"], rec["idx"])
            assert key in smap, f"missing selector score for {key}"
            c = np.asarray(smap[key], float)
            inc = np.asarray(rec["scores"], float)
            s = scale_match(c, inc)
            p_c, p_i, p_s = int(np.argmax(c)), int(np.argmax(inc)), int(np.argmax(s))
            n_pickdiff_inc += int(p_c != p_i)
            n_pickdiff_scale += int(p_s != p_c)
            n_outcome_diff += int(int(rec["sl"][p_s]) != int(rec["sl"][p_c]))
            a = dict(rec); a["scores"] = [float(v) for v in c]; a["pick"] = p_c
            b = dict(rec); b["scores"] = [float(v) for v in s]; b["pick"] = p_s
            rank_recs.append(a); scal_recs.append(b)
        json.dump(rank_recs, open(os.path.join(OUT_RANK, f"transfer_dump_{ds}_lingshu7b.json"), "w"))
        json.dump(scal_recs, open(os.path.join(OUT_SCAL, f"transfer_dump_{ds}_lingshu7b.json"), "w"))
        acc = lambda rr: float(np.mean([r_["sl"][int(np.argmax(r_["scores"]))] for r_ in rr]))
        stats[ds] = dict(
            n=len(recs), acc_incumbent=acc(recs), acc_rank=acc(rank_recs), acc_scaled=acc(scal_recs),
            n_pick_differs_from_incumbent=n_pickdiff_inc,
            n_scalematch_pick_differs_from_rank_pick=n_pickdiff_scale,
            n_of_those_that_change_the_OUTCOME=n_outcome_diff)
        print(ds, json.dumps(stats[ds]), flush=True)

    # gate-scale diagnostic: what max(scores) looks like under each scheme
    diag = {}
    for name, d in (("incumbent_disjoint", src), ("ens8_rank", OUT_RANK), ("ens8_scaled", OUT_SCAL)):
        mx = []
        for ds, tag in CASCADE_DS.items():
            f = (f"transfer_dump_{tag}_open_lingshu7b.json" if name == "incumbent_disjoint"
                 else f"transfer_dump_{ds}_lingshu7b.json")
            mx += [float(np.max(r_["scores"])) for r_ in json.load(open(os.path.join(d, f)))]
        mx = np.array(mx)
        diag[name] = dict(mean=float(mx.mean()), sd=float(mx.std()),
                          n_distinct_values=int(len(np.unique(np.round(mx, 6)))),
                          pct_at_the_modal_value=float(
                              np.mean(mx == float(np.round(np.median(mx), 12)))),
                          p05=float(np.percentile(mx, 5)), p95=float(np.percentile(mx, 95)))
    out = dict(
        what="selector score vectors exported into the cascade's transfer-dump schema",
        date="2026-08-05",
        source_incumbent=src, outputs=dict(rank=OUT_RANK, scaled=OUT_SCAL),
        selector_endpoint_check=dict(sel_eff=r["sel_eff"], acc=r["acc"],
                                     published=GS.PUBLISHED["sel_eff"], abs_dev=dev),
        per_dataset=stats,
        escalation_gate_scale_diagnostic=dict(
            values=diag,
            why_it_matters="the cascade's open-text escalation gate is max(scores) (and, for the "
                           "F10 rejector, max/mean/std/range of scores). A rank_avg fusion is a "
                           "WITHIN-POOL quantity, so its max is nearly a constant and carries "
                           "almost no cross-question confidence -- which is why the rank export is "
                           "reported as the UNTUNED re-run and the scale-matched export as the "
                           "refit."))
    p = os.path.join(G.ROOT, "results/cascade_methods/artifacts",
                     "selector_cascade_dumps_2026-08-05.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("\ngate-scale diagnostic:", json.dumps(diag, indent=1, default=float))
    print(f"\nwrote {OUT_RANK}\nwrote {OUT_SCAL}\nwrote {p}")


if __name__ == "__main__":
    main()
