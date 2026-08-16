#!/usr/bin/env python3
"""vrestruct_resolution_fused.py -- QUESTION 4, for the structure this round recommends.

The concurrent resolution round (verifier_hparams_2026-08-15) measured the LoRA verifier ALONE at
six scoring resolutions and found max_pixels 501,760 is a TIE with the deployed 1,003,520 while
250,880 (the generator's own resolution) is a significant LOSS.  But the deployed selector is the
FUSION of that verifier with the generator-frame head, and a rank fusion can absorb or amplify a
change in one of its two inputs.  Nobody has measured the ladder through the fusion.

This script re-scores every rung's stored per-candidate dump through the frozen 8-seed head
fusion and reports sel_eff and selected accuracy in BOTH currencies, per cell, with paired CIs
against the in-session 1,003,520 control.

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_resolution_fused.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import genframe_data as G      # noqa: E402
import vrestruct_lib as V      # noqa: E402

OPEN_CELL_NAME = {"slake_open": "SLAKE_open", "vqa_rad_open": "VQA_RAD_open",
                  "pathvqa_open": "PATH_VQA_open"}


def load_rung_scores(d, items):
    """(n, 8) verifier scores from a verifhp_px* dump directory, in canonical item order."""
    by = {}
    for short in G.DUMP_ORDER:
        p = os.path.join(ROOT, d, f"transfer_dump_{short}_open_lingshu7b.json")
        for it in json.load(open(p)):
            by[(it["ds"], it["idx"])] = it
    S = np.full((len(items), 8), G.MISSING_SCORE, float)
    npred = 0
    for i, it in enumerate(items):
        r = by.get((it["ds"], it["idx"]))
        if r is None:
            raise KeyError(f"{d}: missing {(it['ds'], it['idx'])}")
        if list(r["preds"]) != list(it["preds"]):
            npred += 1
        sc = list(r["scores"])
        S[i, :len(sc)] = sc
    if npred:
        raise ValueError(f"{d}: {npred} items whose candidate strings differ from the frozen pool")
    return S


def main():
    P = V.load_pool()
    L = V.head_logits(P)
    HR = V.head_rank_slots(P, L, range(8))
    head_rank = V.rank_rows(HR)
    c = V.cost_constants()

    rungs = sorted(int(os.path.basename(x).split("px")[1])
                   for x in glob.glob(os.path.join(ROOT, "ckpts/train/verifhp_px*")))
    out = {}
    ctrl_px = 1003520
    Sctrl = None
    res = {}
    for px in rungs:
        S = load_rung_scores(f"ckpts/train/verifhp_px{px}", P["items"])
        if px == ctrl_px:
            Sctrl = S
        res[px] = dict(lora=V.evaluate(P, V.picks_of(S), f"lora_px{px}"),
                       fused=V.evaluate(P, V.picks_of(V.rank_rows(S) + head_rank), f"fused_px{px}"))

    # null test: the STORED deployed dump reproduces 0.775204; the in-session control need not
    stored = V.evaluate(P, V.picks_of(P["inc"]), "stored_deployed")
    nt = {"NT_stored_deployed_dump": {
        "sel_eff": stored["judge"]["sel_eff"], "published": G.PUBLISHED["sel_eff"],
        "abs_deviation": abs(stored["judge"]["sel_eff"] - G.PUBLISHED["sel_eff"]),
        "pass": bool(abs(stored["judge"]["sel_eff"] - G.PUBLISHED["sel_eff"]) < 1e-6)},
        "NT_in_session_control_vs_stored": {
            "in_session_1003520_lora_sel_eff": res[ctrl_px]["lora"]["judge"]["sel_eff"],
            "stored_lora_sel_eff": stored["judge"]["sel_eff"],
            "difference": res[ctrl_px]["lora"]["judge"]["sel_eff"] - stored["judge"]["sel_eff"],
            "_read": "a batch-1 re-score of the SAME pairs with the SAME adapter at the SAME "
                     "resolution is not bit-identical (the concurrent round measured max 6.03e-2 "
                     "per-candidate deviation). The IN-SESSION 1,003,520 arm is the control for "
                     "the ladder; the stored dump is the anchor for the published number."}}
    print(json.dumps(nt, indent=1, default=float), flush=True)

    for px in rungs:
        row = {"max_pixels": px,
               "verifier_forward_flopeq": c["ver_flopeq_by_max_pixels"].get(px),
               "geometry": c["ver_geometry_by_max_pixels"].get(px)}
        for arm in ("lora", "fused"):
            r = res[px][arm]
            ctl = res[ctrl_px][arm]
            row[arm] = {}
            for cur in ("judge", "em"):
                row[arm][cur] = dict(
                    sel_eff=r[cur]["sel_eff"], acc=r[cur]["acc"],
                    macro3=r[cur]["macro_cells"],
                    per_cell={OPEN_CELL_NAME[ds]: r[cur]["per_ds"][ds]["acc"]
                              for ds in G.EVAL_DS},
                    vs_control=V.paired_boot(r[cur]["got"], ctl[cur]["got"]),
                    per_cell_vs_control={OPEN_CELL_NAME[ds]: V.paired_boot(
                        r[cur]["got"][P["ds_index"] == j], ctl[cur]["got"][P["ds_index"] == j])
                        for j, ds in enumerate(G.EVAL_DS)})
                row[arm][cur]["guardrail_clean_vs_control"] = bool(all(
                    row[arm][cur]["per_cell_vs_control"][OPEN_CELL_NAME[ds]]["delta"] >= 0
                    for ds in G.EVAL_DS))
            row[arm]["n_picks_differing_from_control"] = int(
                (r["picks"] != ctl["picks"]).sum())
        out[str(px)] = row

    json.dump(dict(
        title="QUESTION 4 -- the verifier's scoring resolution, measured THROUGH the fusion",
        date="2026-08-16", cpu_only=True,
        control_max_pixels=ctrl_px, null_tests=nt, rungs=out,
        sources=dict(scores="ckpts/train/verifhp_px*/transfer_dump_*_lingshu7b.json "
                            "(concurrent round verifier_hparams_2026-08-15)",
                     head="ckpts/train/genframe_head_ens8 (frozen)",
                     flops="artifacts/_verifier_hparams_parts/{cost,recost}.json")),
        open(os.path.join(V.PARTS, "resolution_fused.json"), "w"), indent=1, default=float)

    print(f"\n{'px':>10s} {'verFLOPeq':>9s} | {'loraSelEff':>10s} {'loraAcc':>9s} {'dCtrl':>9s} "
          f"| {'fusedSelEff':>11s} {'fusedAcc':>9s} {'dCtrl':>9s} {'ci':>24s} {'guard':>6s}")
    for px in rungs:
        r = out[str(px)]
        lv = r["lora"]["judge"]
        fv = r["fused"]["judge"]
        print(f"{px:>10d} {r['verifier_forward_flopeq']:9.4f} | {lv['sel_eff']:10.6f} "
              f"{lv['acc']:9.6f} {lv['vs_control']['delta']:+9.6f} | {fv['sel_eff']:11.6f} "
              f"{fv['acc']:9.6f} {fv['vs_control']['delta']:+9.6f} "
              f"[{fv['vs_control']['lo']:+.6f},{fv['vs_control']['hi']:+.6f}] "
              f"{str(fv['guardrail_clean_vs_control']):>6s}")


if __name__ == "__main__":
    main()
