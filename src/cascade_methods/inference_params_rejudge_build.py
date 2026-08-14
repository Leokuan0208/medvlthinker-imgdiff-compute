#!/usr/bin/env python3
"""inference_params_rejudge_build.py -- build the input for THE DECISIVE CONFOUND TEST.

The 2026-08-13 decoding sweep labelled its candidate slots from two disjoint sources:
a PRELOAD cache of judge labels harvested from earlier runs, and a fresh judge pass run
in that session. Which source a slot draws from is a monotone function of the treatment
(hotter sampling -> more novel answer strings -> larger fresh share: 0.080 at T=0.3 rising
to 0.500 at T=1.3). If the two label sources disagree at all systematically, that gradient
manufactures a temperature effect out of nothing.

The two sets share zero keys, so their agreement cannot be measured from the files. It CAN
be measured by re-judging: take a stratified sample of slots whose label came from the
PRELOAD cache and run them through THIS session's judge harness, then compare.

This writes the judge input file. runners/run_rejudge_confound.sh runs the judge on it;
src/cascade_methods/inference_params_rejudge_analyze.py reads the result back.
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
from src.cascade_methods.inference_params_verify import DEC, build_items, load_judge_map, load_vscores  # noqa: E402
from src.cascade_methods.inference_params_ceiling import label_sources  # noqa: E402

OUTDIR = os.path.join(ROOT, "ckpts/openvqa/decoding_sweep")
N_PER_CELL = 1500
SEED = 20260814


def main():
    src = label_sources()
    judge = load_judge_map()
    vsc = load_vscores()
    ref = G.load_items()
    meta = {}          # (ds, idx) -> (question, gold)
    for ds in G.EVAL_DS:
        with open(os.path.join(DEC, f"ckpt_{ds}_T07_s0.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    meta[(ds, str(o["idx"]))] = (o["question"], o["gold"])

    # every (ds, idx, na) that (a) is PRELOAD-labelled and (b) actually appears in a pool
    seen = set()
    for s in ["T03", "T05", "T07", "T10", "T13", "minp01", "rp105", "rp11"]:
        for sd in (0, 1, 2):
            it, _, _ = build_items(s, sd, judge, vsc)
            for x in it:
                for a in x["preds"]:
                    k = (x["ds"], str(x["idx"]), G.norm(a))
                    if src.get(k) != "fresh":
                        seen.add((k, a))
    bycell = defaultdict(list)
    for k, raw in sorted(seen, key=lambda z: (z[0][0], z[0][1], z[0][2])):
        bycell[k[0]].append((k, raw))
    rng = np.random.default_rng(SEED)
    rows, keys = [], []
    for ds in G.EVAL_DS:
        pool = bycell[ds]
        take = rng.choice(len(pool), size=min(N_PER_CELL, len(pool)), replace=False)
        for t in sorted(take):
            (dsx, idx, na), raw = pool[t]
            q, gold = meta[(dsx, idx)]
            jid = f"{idx}#RJ{len(rows)}"
            rows.append({"idx": jid, "question": q, "gold": gold, "modal_pred": raw})
            keys.append({"jid": jid, "ds": dsx, "idx": idx, "na": na,
                         "cached_label": judge[(dsx, idx, na)]})
    p = os.path.join(OUTDIR, "rejudge_confound_in.jsonl")
    with open(p, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    json.dump(keys, open(os.path.join(OUTDIR, "rejudge_confound_keys.json"), "w"))
    print(f"wrote {p}  n={len(rows)}")
    print({ds: sum(1 for k in keys if k["ds"] == ds) for ds in G.EVAL_DS})


if __name__ == "__main__":
    main()
