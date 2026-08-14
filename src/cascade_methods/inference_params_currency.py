#!/usr/bin/env python3
"""inference_params_currency.py -- is the T=0.5 effect a GENERATION effect or a GRADING one?

Three independent probes, none of which needs a GPU:

 1. CONTRAST CHECK on my own source-matched test. Restricting to items whose every
    candidate in both arms came from the SAME cached judge pass gave d_selected = 0.0
    exactly. That is only evidence if the subset still CONTAINS a treatment contrast --
    if the two arms produce identical pools there, d = 0 is vacuous. Measured here as the
    pool Jaccard and the oracle delta inside the subset.

 2. EXACT-MATCH currency, recomputed from the generation dumps' own oks_em field on the
    IDENTICAL verifier picks. EM is a deterministic function of (answer, gold): no judge,
    no cache, no session. If the effect is a judge-source artifact it must vanish in EM.
    Reported with the SEED-AWARE interval, not the item-only one.

 3. LENIENCY DECOMPOSITION. For every setting, the rate at which the judge rescues an
    answer that exact match rejects (judge_ok=1, em=0), split by label source. A cold
    setting that wins by producing MORE cache-labelled answers, when cache labels happen
    to be more generous, would show up as a source-dependent rescue rate.

Writes results/cascade_methods/artifacts/_infparams_currency.json
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


def raw_dump(setting, seed):
    ref = G.load_items()
    order = [(it["ds"], str(it["idx"])) for it in ref]
    byds = defaultdict(dict)
    for ds in EVAL_DS:
        with open(os.path.join(DEC, f"ckpt_{ds}_{setting}_s{seed}.jsonl")) as fh:
            for ln in fh:
                if ln.strip():
                    o = json.loads(ln)
                    byds[ds][str(o["idx"])] = o
    return [byds[ds][i] for ds, i in order]


def sboot(a, b, mask=None, nboot=NBOOT, seed=SEED):
    """seed-aware paired bootstrap on (3, n) 0/1 matrices."""
    rng = np.random.default_rng(seed)
    idx = np.arange(a.shape[1]) if mask is None else np.where(mask)[0]
    pt = a[:, idx].mean() - b[:, idx].mean()
    d = np.empty(nboot)
    for k in range(nboot):
        j = idx[rng.integers(0, len(idx), len(idx))]
        d[k] = (a[np.ix_(rng.integers(0, 3, 3), j)].mean()
                - b[np.ix_(rng.integers(0, 3, 3), j)].mean())
    return {"d": float(pt), "ci95": [float(np.percentile(d, 2.5)),
                                     float(np.percentile(d, 97.5))]}


def main():
    judge = load_judge_map()
    vsc = load_vscores()
    src = label_sources()
    ref = G.load_items()
    order = [(it["ds"], str(it["idx"])) for it in ref]
    n = len(order)
    dsidx = np.array([EVAL_DS.index(d) for d, _ in order])

    pools, raws = {}, {}
    for s in SETTINGS:
        for sd in (0, 1, 2):
            it, mj, mv = build_items(s, sd, judge, vsc)
            pools[(s, sd)] = it
            raws[(s, sd)] = raw_dump(s, sd)
    out = {"title": "GENERATION EFFECT OR GRADING EFFECT? currency + confound decomposition",
           "date": "2026-08-14", "no_fabricated_numbers": True,
           "nboot": NBOOT, "bootstrap_seed": SEED}

    # picks + got in both currencies
    picks, gotJ, gotE, rec = {}, {}, {}, {}
    for s in SETTINGS:
        P, J, E, R = [], [], [], []
        for sd in (0, 1, 2):
            it = pools[(s, sd)]
            sb = {(x["ds"], x["idx"]): x["scores"] for x in it}
            r = G.sel_eff(sb, items=it)
            P.append(r["picks"]); J.append(r["got"]); R.append(r["rec"])
            rw = raws[(s, sd)]
            E.append(np.array([rw[i]["oks_em"][r["picks"][i]] for i in range(n)], dtype=int))
        picks[s] = np.stack(P); gotJ[s] = np.stack(J)
        gotE[s] = np.stack(E); rec[s] = np.stack(R)

    # ---- 1. contrast check on the source-matched subset
    def all_pre(it):
        return all(src.get((it["ds"], str(it["idx"])), G.norm(a)) != "fresh"
                   if False else
                   src.get((it["ds"], str(it["idx"]), G.norm(a))) != "fresh"
                   for a in it["preds"])
    m = np.ones(n, dtype=bool)
    for sd in (0, 1, 2):
        m &= np.array([all_pre(it) for it in pools[("T05", sd)]])
        m &= np.array([all_pre(it) for it in pools[(CTRL, sd)]])
    jac = []
    for i in np.where(m)[0]:
        A = set(); B = set()
        for sd in (0, 1, 2):
            A |= {G.norm(a) for a in pools[("T05", sd)][i]["preds"]}
            B |= {G.norm(a) for a in pools[(CTRL, sd)][i]["preds"]}
        jac.append(len(A & B) / len(A | B))
    jac_all = []
    for i in range(n):
        A = set(); B = set()
        for sd in (0, 1, 2):
            A |= {G.norm(a) for a in pools[("T05", sd)][i]["preds"]}
            B |= {G.norm(a) for a in pools[(CTRL, sd)][i]["preds"]}
        jac_all.append(len(A & B) / len(A | B))
    out["contrast_check_on_source_matched_subset"] = {
        "n_items": int(m.sum()),
        "pool_jaccard_T05_vs_T07_inside_subset": float(np.mean(jac)),
        "pool_jaccard_T05_vs_T07_all_items": float(np.mean(jac_all)),
        "d_oracle_inside_subset": sboot(rec["T05"], rec[CTRL], mask=m),
        "d_oracle_all_items": sboot(rec["T05"], rec[CTRL]),
        "d_selected_judge_inside_subset": sboot(gotJ["T05"], gotJ[CTRL], mask=m),
        "d_selected_em_inside_subset": sboot(gotE["T05"], gotE[CTRL], mask=m),
        "mean_distinct_inside_subset": {
            s: float(np.mean([len({G.norm(a) for a in pools[(s, sd)][i]["preds"]})
                              for sd in (0, 1, 2) for i in np.where(m)[0]]))
            for s in ("T05", CTRL)},
        "mean_distinct_all_items": {
            s: float(np.mean([len({G.norm(a) for a in pools[(s, sd)][i]["preds"]})
                              for sd in (0, 1, 2) for i in range(n)]))
            for s in ("T05", CTRL)},
        "interpretation_rule": "d=0 inside the subset is evidence ONLY if the subset still "
                               "carries a treatment contrast (jaccard < 1, oracle delta "
                               "non-zero). If the two arms produce near-identical pools "
                               "there, the test is vacuous and must be reported as such.",
    }
    cc = out["contrast_check_on_source_matched_subset"]
    print("CONTRAST CHECK n=%d jac_subset=%.4f jac_all=%.4f d_oracle_subset=%s"
          % (cc["n_items"], cc["pool_jaccard_T05_vs_T07_inside_subset"],
             cc["pool_jaccard_T05_vs_T07_all_items"],
             cc["d_oracle_inside_subset"]), flush=True)

    # ---- 2. EM currency, seed-aware, all settings
    cur = {}
    for s in SETTINGS:
        if s == CTRL:
            continue
        cur[s] = {"judge": sboot(gotJ[s], gotJ[CTRL]),
                  "exact_match": sboot(gotE[s], gotE[CTRL]),
                  "levels": {"judge": float(gotJ[s].mean()),
                             "exact_match": float(gotE[s].mean())},
                  "per_cell_em": {d_: sboot(gotE[s], gotE[CTRL], mask=(dsidx == j))
                                  for j, d_ in enumerate(EVAL_DS)}}
        print(f"{s:7s} judge {cur[s]['judge']['d']:+.6f} {cur[s]['judge']['ci95']}"
              f"   EM {cur[s]['exact_match']['d']:+.6f} {cur[s]['exact_match']['ci95']}",
              flush=True)
    cur["_control_levels"] = {"judge": float(gotJ[CTRL].mean()),
                              "exact_match": float(gotE[CTRL].mean())}
    out["dual_currency_seed_aware"] = cur

    # ---- 3. leniency decomposition by label source
    len_dec = {}
    for s in SETTINGS:
        tot = defaultdict(int); resc = defaultdict(int)
        for sd in (0, 1, 2):
            it = pools[(s, sd)]; rw = raws[(s, sd)]
            for i in range(n):
                for k, a in enumerate(it[i]["preds"]):
                    so = src.get((it[i]["ds"], str(it[i]["idx"]), G.norm(a)), "preload")
                    em = rw[i]["oks_em"][k]
                    jd = it[i]["sl"][k]
                    if em == 0:
                        tot[so] += 1
                        if jd == 1:
                            resc[so] += 1
        len_dec[s] = {so: {"n_em_negative_slots": tot[so],
                           "judge_rescued": resc[so],
                           "rescue_rate": (resc[so] / tot[so]) if tot[so] else None}
                      for so in ("preload", "fresh")}
    out["judge_leniency_by_label_source"] = len_dec
    out["judge_leniency_note"] = (
        "rescue rate = P(judge_ok=1 | exact_match=0), computed per LABEL SOURCE. A gap "
        "between the two sources is NOT by itself proof of judge drift -- novel strings "
        "genuinely differ from cached ones -- but a gap in the direction that favours "
        "cold settings is the mechanism by which a cache-share gradient could manufacture "
        "the observed temperature effect.")
    for s in SETTINGS:
        pr = len_dec[s]["preload"]["rescue_rate"]; fr = len_dec[s]["fresh"]["rescue_rate"]
        print(f"{s:7s} rescue preload {pr if pr is None else round(pr,4)}  "
              f"fresh {fr if fr is None else round(fr,4)}", flush=True)

    p = os.path.join(ART, "_infparams_currency.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
