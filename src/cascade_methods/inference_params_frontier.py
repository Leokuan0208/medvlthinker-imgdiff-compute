#!/usr/bin/env python3
"""inference_params_frontier.py -- THE FRONTIER for the 7B inference-parameter round.

The project's current objective is MINIMUM COST AT PARITY, so accuracy alone does not
rank these settings: a setting that holds accuracy while cutting compute is a win and has
to be scored as one. This builds the full table -- oracle@8, sel_eff, SELECTED (judge and
exact-match), macro-8 contribution, generator FLOPs, verifier FLOPs, whole-arm FLOPs --
and marks the non-dominated points.

IT ALSO CORRECTS THE ROUND'S COST MODEL. The resolution sweep charges the deployed open
arm as "8 generator forwards at cap320 + 8 VERIFIER forwards at 1,003,520". The verifier
does not run 8 times. Candidates are deduplicated by normalized answer string before
scoring -- that is how the frozen feature cache was built and how the score cache is keyed
(feats_hidden/generator_eval_s{0,1}of2.meta.json: 4472 + 4471 = 8943 rows over 2345
questions = 3.813 distinct candidates per question, not 8). Charging 8 overstates the
deployed arm's cost by 1.57x and therefore overstates every saving measured against it.

The correction matters twice over, because the distinct-candidate count is itself a
function of the decoding parameter being swept (2.18 at T=0.3, 3.67 at T=0.7, 5.48 at
T=1.3). Under dedup-aware costing the temperature axis moves COST, not just accuracy --
which the sweep did not report, and which is the axis the project actually cares about.

Writes results/cascade_methods/artifacts/_infparams_frontier.json
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
MACRO_W = 1.0 / 8.0
NBOOT = 10000
SEED = 20260814

# per-candidate FLOPs, taken verbatim from the round's own measured cost artifact
COST_SRC = os.path.join(ART, "_resolution_parts/cost_by_resolution.json")


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
    cost = json.load(open(COST_SRC))
    per = cost["open_half_per_candidate"]
    f_gen_cap320 = per["cap320"]["flops_per_candidate"]
    f_gen_native = per["native"]["flops_per_candidate"]
    f_ver = cost["where_the_open_arm_spends_its_flops"][
        "at_the_deployed_operating_point"]["verifier_per_candidate_at_1003520"]

    judge = load_judge_map()
    vsc = load_vscores()
    ref = G.load_items()
    n = len(ref)
    dsidx = np.array([EVAL_DS.index(it["ds"]) for it in ref])

    got, gotE, rec, distinct, gtok = {}, {}, {}, {}, {}
    for s in SETTINGS:
        gg, ee, rr, dd, tt = [], [], [], [], []
        for sd in (0, 1, 2):
            it, mj, mv = build_items(s, sd, judge, vsc)
            assert mj == 0 and mv == 0
            sb = {(x["ds"], x["idx"]): x["scores"] for x in it}
            r = G.sel_eff(sb, items=it)
            gg.append(r["got"]); rr.append(r["rec"])
            byds = defaultdict(dict)
            for ds in EVAL_DS:
                with open(os.path.join(DEC, f"ckpt_{ds}_{s}_s{sd}.jsonl")) as fh:
                    for ln in fh:
                        if ln.strip():
                            o = json.loads(ln)
                            byds[ds][str(o["idx"])] = o
            raws = [byds[x["ds"]][str(x["idx"])] for x in it]
            ee.append(np.array([raws[i]["oks_em"][r["picks"][i]] for i in range(n)], int))
            dd.append(np.array([len({G.norm(a) for a in x["preds"]}) for x in it], float))
            tt.append(np.array([np.mean(raws[i]["gen_tokens_all"]) for i in range(n)], float))
        got[s] = np.stack(gg); gotE[s] = np.stack(ee); rec[s] = np.stack(rr)
        distinct[s] = np.stack(dd); gtok[s] = np.stack(tt)

    out = {
        "title": "THE FRONTIER -- 7B inference parameters, accuracy AND cost",
        "date": "2026-08-14", "no_fabricated_numbers": True,
        "objective": "minimum cost at parity (CLAUDE.md / current project objective), so a "
                     "setting that holds accuracy and cuts compute is a WIN",
        "cost_model": {
            "source": "results/cascade_methods/artifacts/_resolution_parts/"
                      "cost_by_resolution.json -- prefill-inclusive whole-forward FLOPs on "
                      "MEASURED token geometry (src/cascade_methods/flop_ratio_derivation."
                      "forward_flops); a CPU re-costing, not a wall-clock run",
            "flops_generator_per_candidate_cap320": f_gen_cap320,
            "flops_generator_per_candidate_native": f_gen_native,
            "flops_verifier_per_candidate_at_1003520": f_ver,
        },
        "COST_MODEL_CORRECTION": {
            "what_the_round_charged": "8 generator forwards + 8 VERIFIER forwards "
                                      "(resolution_greedy_vs_arm.py:125, arm_deployed = "
                                      "8*f_gen_cap320 + 8*f_ver)",
            "why_that_is_wrong": "the verifier scores DEDUPLICATED candidates, not slots. "
                                 "Evidence: feats_hidden/generator_eval_s0of2.meta.json n=4472 "
                                 "+ generator_eval_s1of2.meta.json n=4471 = 8943 rows over "
                                 "2345 questions; the frozen transfer dumps contain 18760 "
                                 "slots but only 8943 distinct normalized answer strings; "
                                 "and the sweep's own vscore cache is keyed (ds, idx, ans).",
            "measured_distinct_per_question_deployed_pool": 8943 / 2345,
            "overcharge_factor_on_the_deployed_arm": None,  # filled below
            "consequence": "every saving quoted against the deployed open arm is inflated, "
                           "and the temperature axis moves VERIFIER cost as well as "
                           "accuracy -- an effect the round did not report.",
        },
    }

    rows = {}
    ctrl_sel = float(got[CTRL].mean())
    for s in SETTINGS:
        nd = float(distinct[s].mean())
        arm8 = 8 * f_gen_cap320 + 8 * f_ver                 # as the round charged it
        armd = 8 * f_gen_cap320 + nd * f_ver                # dedup-aware
        per_cell_sel = {d: float(got[s][:, dsidx == j].mean())
                        for j, d in enumerate(EVAL_DS)}
        rows[s] = {
            "params": {"T03": "temperature 0.3", "T05": "temperature 0.5",
                       "T07": "temperature 0.7 (DEPLOYED)", "T10": "temperature 1.0",
                       "T13": "temperature 1.3", "minp01": "min_p 0.10 @ T=0.7",
                       "rp105": "repetition_penalty 1.05 @ T=0.7",
                       "rp11": "repetition_penalty 1.10 @ T=0.7"}[s],
            "oracle@8": float(rec[s].mean()),
            "sel_eff": float(got[s][rec[s] == 1].mean()),
            "selected_judge": float(got[s].mean()),
            "selected_em": float(gotE[s].mean()),
            "mean_distinct_candidates": nd,
            "mean_generated_tokens": float(gtok[s].mean()),
            "per_cell_selected_judge": per_cell_sel,
            "macro8_contribution": float(sum(
                per_cell_sel[d] - float(got[CTRL][:, dsidx == j].mean())
                for j, d in enumerate(EVAL_DS)) * MACRO_W),
            "cost": {
                "generator_flops_per_question": 8 * f_gen_cap320,
                "verifier_flops_per_question_dedup_aware": nd * f_ver,
                "arm_flops_per_question_dedup_aware": armd,
                "arm_flops_per_question_as_the_round_charged": arm8,
                "verifier_share_of_arm_dedup_aware": nd * f_ver / armd,
            },
        }
    ctrl_armd = rows[CTRL]["cost"]["arm_flops_per_question_dedup_aware"]
    out["COST_MODEL_CORRECTION"]["overcharge_factor_on_the_deployed_arm"] = float(
        rows[CTRL]["cost"]["arm_flops_per_question_as_the_round_charged"] / ctrl_armd)

    for s in SETTINGS:
        r = rows[s]
        r["cost"]["arm_flops_relative_to_deployed"] = (
            r["cost"]["arm_flops_per_question_dedup_aware"] / ctrl_armd)
        if s != CTRL:
            r["d_selected_judge_seed_aware"] = sboot(got[s], got[CTRL])
            r["d_selected_em_seed_aware"] = sboot(gotE[s], gotE[CTRL])
            r["d_oracle_seed_aware"] = sboot(rec[s], rec[CTRL])
        r["latency_s"] = "NOT MEASURED -- no wall-clock instrumentation in this round"
        r["vram_gib"] = ("NOT MEASURED for any decoding setting. Same model, same N, same "
                         "max_tokens, same image cap means no mechanism for it to move, "
                         "but that is an argument, not a measurement.")
    out["settings"] = rows

    # ---------------- Pareto frontier on (SELECTED judge, arm FLOPs), maximise / minimise
    pts = [(s, rows[s]["selected_judge"],
            rows[s]["cost"]["arm_flops_relative_to_deployed"]) for s in SETTINGS]
    nd_set = []
    for s, a, c in pts:
        dominated = any((a2 >= a and c2 <= c and (a2 > a or c2 < c))
                        for s2, a2, c2 in pts if s2 != s)
        if not dominated:
            nd_set.append(s)
    out["pareto_non_dominated_on_selected_vs_arm_flops"] = {
        "non_dominated": nd_set,
        "axes": "maximise SELECTED (judge, 3-seed mean); minimise dedup-aware whole-arm "
                "FLOPs per question relative to the deployed T=0.7",
        "table": {s: {"selected_judge": rows[s]["selected_judge"],
                      "arm_flops_rel": rows[s]["cost"]["arm_flops_relative_to_deployed"]}
                  for s in SETTINGS},
        "caveat": "dominance here is on POINT ESTIMATES. Every accuracy difference in this "
                  "grid except the T>=1.0 losses has a seed-aware CI spanning zero, so the "
                  "frontier's accuracy axis is not separated -- the COST axis is what "
                  "actually distinguishes these points.",
    }

    # ---------------- the cost-at-parity reading
    par = {}
    for s in SETTINGS:
        if s == CTRL:
            continue
        d = rows[s]["d_selected_judge_seed_aware"]
        par[s] = {
            "d_selected_judge": d["d"], "ci95": d["ci95"],
            "accuracy_verdict": ("TIE (CI spans 0)" if d["ci95"][0] < 0 < d["ci95"][1]
                                 else ("WIN" if d["d"] > 0 else "LOSS")),
            "arm_flops_rel": rows[s]["cost"]["arm_flops_relative_to_deployed"],
            "compute_saving_pct": 100 * (1 - rows[s]["cost"]["arm_flops_relative_to_deployed"]),
        }
    out["cost_at_parity"] = par
    out["cost_at_parity_note"] = (
        "read this table with the objective in mind: a row that is a TIE on accuracy and "
        "below 1.0 on arm_flops_rel is a COST WIN. The saving is entirely in the verifier "
        "term -- generation cost is identical across every setting here (same N, same "
        "max_tokens, same image cap), and generated tokens are 1.2% of compute.")

    p = os.path.join(ART, "_infparams_frontier.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({"overcharge": out["COST_MODEL_CORRECTION"]
                      ["overcharge_factor_on_the_deployed_arm"],
                      "non_dominated": nd_set}, indent=1))
    for s in SETTINGS:
        r = rows[s]
        print(f"{s:7s} sel {r['selected_judge']:.6f}  or@8 {r['oracle@8']:.6f}  "
              f"se {r['sel_eff']:.6f}  dist {r['mean_distinct_candidates']:.3f}  "
              f"armFLOPs_rel {r['cost']['arm_flops_relative_to_deployed']:.4f}  "
              f"macro8 {r['macro8_contribution']:+.6f}")
    print("wrote", p)


if __name__ == "__main__":
    main()
