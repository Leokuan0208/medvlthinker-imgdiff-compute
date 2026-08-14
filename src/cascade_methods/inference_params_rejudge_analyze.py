#!/usr/bin/env python3
"""inference_params_rejudge_analyze.py -- read back THE DECISIVE CONFOUND TEST.

4,047 slots whose label the decoding sweep took from its PRELOAD cache, re-judged with the
same harness the sweep used for its FRESH labels. If the two label sources agree, the
cache-share gradient across temperature (0.080 fresh at T=0.3 -> 0.500 at T=1.3) cannot
manufacture a temperature effect and the sweep's grading is sound. If they disagree in the
direction that flatters cached labels, part of the reported "colder is better" result is
the label source rather than the model.

Then: PROPAGATE the measured disagreement rate into the headline delta, by re-labelling
the preload-sourced slots with the re-judged label where they differ and recomputing the
T=0.5 vs T=0.7 delta end to end.

Writes results/cascade_methods/artifacts/_infparams_rejudge.json
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
from src.cascade_methods.inference_params_ceiling import label_sources  # noqa: E402

ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
EVAL_DS = G.EVAL_DS
SETTINGS = ["T03", "T05", "T07", "T10", "T13", "minp01", "rp105", "rp11"]
CTRL = "T07"
NBOOT = 10000
SEED = 20260814


def sboot(a, b, nboot=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = a.shape[1]
    pt = a.mean() - b.mean()
    d = np.empty(nboot)
    for k in range(nboot):
        j = rng.integers(0, n, n)
        d[k] = (a[np.ix_(rng.integers(0, 3, 3), j)].mean()
                - b[np.ix_(rng.integers(0, 3, 3), j)].mean())
    return {"d": float(pt), "ci95": [float(np.percentile(d, 2.5)),
                                     float(np.percentile(d, 97.5))]}


def main():
    keys = json.load(open(os.path.join(DEC, "rejudge_confound_keys.json")))
    lab = {}
    with open(os.path.join(DEC, "rejudge_confound_in.judge.jsonl")) as fh:
        for ln in fh:
            if ln.strip():
                o = json.loads(ln)
                lab[o["idx"]] = int(o["judge_ok"])
    rows = [(k, lab[k["jid"]]) for k in keys if k["jid"] in lab]
    n = len(rows)
    agree = sum(1 for k, v in rows if k["cached_label"] == v)
    # directional: cached said correct, re-judge says wrong  (cache more generous)
    c1r0 = sum(1 for k, v in rows if k["cached_label"] == 1 and v == 0)
    c0r1 = sum(1 for k, v in rows if k["cached_label"] == 0 and v == 1)
    out = {
        "title": "RE-JUDGE CONFOUND TEST -- do the preload cache and this session's judge "
                 "agree?",
        "date": "2026-08-14", "no_fabricated_numbers": True,
        "judge": "src/labeling/run_judge.py, MedVLThinker-32B (Qwen2.5-32B backbone), "
                 "text-only, temperature 0.0, max_tokens 2, Yes/No logprob comparison, "
                 "tp=2 -- the SAME harness and model the sweep used for its fresh labels",
        "n_slots_rejudged": n,
        "n_requested": len(keys),
        "per_cell_n": {d: sum(1 for k, _ in rows if k["ds"] == d) for d in EVAL_DS},
        "agreement": agree / n,
        "n_disagree": n - agree,
        "cache_says_correct_rejudge_says_wrong": c1r0,
        "cache_says_wrong_rejudge_says_correct": c0r1,
        "net_cache_generosity_pp": (c1r0 - c0r1) / n,
        "cached_positive_rate": sum(k["cached_label"] for k, _ in rows) / n,
        "rejudged_positive_rate": sum(v for _, v in rows) / n,
        "per_cell_agreement": {
            d: (sum(1 for k, v in rows if k["ds"] == d and k["cached_label"] == v)
                / max(1, sum(1 for k, _ in rows if k["ds"] == d)))
            for d in EVAL_DS},
    }
    print(json.dumps({k: v for k, v in out.items() if k != "title"}, indent=1))

    # ---------------- propagate: rebuild the headline delta with re-judged labels applied
    judge = load_judge_map()
    vsc = load_vscores()
    src = label_sources()
    patched = dict(judge)
    npatch = 0
    for k, v in rows:
        key = (k["ds"], k["idx"], k["na"])
        if judge.get(key) != v:
            patched[key] = v
            npatch += 1
    out["n_labels_patched"] = npatch

    def run(jmap):
        got, rec = {}, {}
        for s in SETTINGS:
            gg, rr = [], []
            for sd in (0, 1, 2):
                it, mj, mv = build_items(s, sd, jmap, vsc)
                sb = {(x["ds"], x["idx"]): x["scores"] for x in it}
                r = G.sel_eff(sb, items=it)
                gg.append(r["got"]); rr.append(r["rec"])
            got[s] = np.stack(gg); rec[s] = np.stack(rr)
        return got, rec

    g0, r0 = run(judge)
    g1, r1 = run(patched)
    prop = {}
    for s in SETTINGS:
        if s == CTRL:
            continue
        prop[s] = {"as_published": sboot(g0[s], g0[CTRL]),
                   "with_rejudged_labels": sboot(g1[s], g1[CTRL])}
        print(f"{s:7s} published {prop[s]['as_published']['d']:+.6f} "
              f"{prop[s]['as_published']['ci95']}  patched "
              f"{prop[s]['with_rejudged_labels']['d']:+.6f} "
              f"{prop[s]['with_rejudged_labels']['ci95']}", flush=True)
    out["headline_delta_with_rejudged_labels"] = prop
    out["propagation_note"] = (
        "the patch covers the SAMPLED preload slots only (%d of the cache), so this is a "
        "partial correction and its magnitude is a lower bound on the full effect of "
        "re-labelling every cached slot." % n)

    p = os.path.join(ART, "_infparams_rejudge.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
