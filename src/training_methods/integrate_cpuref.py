#!/usr/bin/env python3
"""integrate_cpuref.py -- TRAINER null test.

The metric null test (genframe_data.null_test) only proves the harness reads the incumbent's
stored scores correctly. This proves the TRAINER is the same trainer: it refits the published
generator-frame head end-to-end on CPU at seed 0 with the pre-registered config and must
reproduce sel_eff 0.795640, and it reconstructs the currently-DEPLOYED rank_avg fusion, which
must reproduce 0.806540. Both score vectors are saved so the eval stage can bootstrap
against the deployed fusion item-by-item rather than against a quoted constant.

  python3 src/training_methods/integrate_cpuref.py           # ~15-25 min on CPU
"""
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import genframe_data as G           # noqa: E402
import integrate_lib as IL          # noqa: E402
from fit_hidden_head import fit_head, predict  # noqa: E402  (the CPU original, unmodified)
import cheapcontrast as CC          # noqa: E402  (only for standardize())
import torch                        # noqa: E402


def main():
    t0 = time.time()
    cfg = IL.BASE_CFG
    tr = G.load_candidates("train", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    ev = G.load_candidates("eval", "generator", layers=[cfg["layer"]], pooling=(cfg["pooling"],))
    Xtr = tr.matrix(cfg["pooling"], cfg["layer"])
    Xev = ev.matrix(cfg["pooling"], cfg["layer"])
    ytr = np.array([r["y"] for r in tr.rows], np.float32)
    gtr = G.group_ids(tr)
    kev = [(r["ds"], r["idx"], r["na"]) for r in ev.rows]
    items = G.load_items()
    print(f"loaded train {Xtr.shape} eval {Xev.shape} in {time.time()-t0:.0f}s "
          f"threads={torch.get_num_threads()}", flush=True)

    # feature standardization with TRAIN mu/sd -- part of the published recipe
    # (fit_hidden_head.py:507 fits on (Xtr - mu)/sd and scores (Xev - mu)/sd)
    t = time.time()
    Xa, Xb = CC.standardize(Xtr, Xev)
    m = fit_head(Xa, ytr, gtr, objective=cfg["objective"], hidden=cfg["hidden"], wd=cfg["wd"],
                 lr=cfg["lr"], epochs=cfg["epochs"], bs=cfg["bs"], seed=0)
    sv = predict(m, Xb)
    fit_s = time.time() - t

    inc = G.incumbent_scores()
    base = G.sel_eff(inc, items)
    head = IL.score_map(kev, sv)
    r_head = G.sel_eff(head, items)
    fus = G.rank_fuse(inc, head, items=items, ranker=G.rank_avg)
    r_fus = G.sel_eff(fus, items)

    out = {
        "what": "TRAINER null test: published head + published deployed fusion refit on CPU, seed 0",
        "config": cfg, "torch_threads": torch.get_num_threads(), "fit_seconds": fit_s,
        "published_head": {
            "measured_sel_eff": r_head["sel_eff"], "published": 0.795640,
            "abs_dev": abs(r_head["sel_eff"] - 0.795640),
            "acc": r_head["acc"],
            "per_ds": {k: v["sel_eff"] for k, v in r_head["per_ds"].items()},
            "contested": r_head["contested"]["sel_eff"]},
        "published_deployed_fusion": {
            "measured_sel_eff": r_fus["sel_eff"], "published": 0.806540,
            "abs_dev": abs(r_fus["sel_eff"] - 0.806540),
            "acc": r_fus["acc"],
            "per_ds": {k: v["sel_eff"] for k, v in r_fus["per_ds"].items()},
            "contested": r_fus["contested"]["sel_eff"]},
        "pass": bool(abs(r_head["sel_eff"] - 0.795640) < 1e-5 and
                     abs(r_fus["sel_eff"] - 0.806540) < 1e-5),
    }
    np.savez(os.path.join(IL.PARTS, "cpuref_scores.npz"),
             head_cpu_seed0=sv.astype(np.float64),
             head_slots=G._slot_scores(head, items),
             fusion_slots=G._slot_scores(fus, items),
             head_got=r_head["got"], fusion_got=r_fus["got"], rec=r_fus["rec"])
    IL.jdump(out, os.path.join(IL.PARTS, "cpuref.json"))
    print(json.dumps(out, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
