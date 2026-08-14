#!/usr/bin/env python3
"""resolution_greedy_vs_arm.py -- SWEEP 2: the cross-arm comparison the per-cap tables do not make.

The per-cap tables compare like with like (native best-of-8 vs cap320 best-of-8).  This script asks
the deployment question instead:

    can ONE greedy decode at NATIVE resolution replace EIGHT samples at cap320 plus the LoRA
    verifier that ranks them?

Both quantities already exist on the same 2,345-question endpoint, under the same judge, generated
by the same script in the SAME session (2026-08-13), so the comparison is matched and is not
exposed to the +-0.008 serving-config caveat.  It is scored two ways:

    accuracy  paired item bootstrap of (native greedy_t0) - (cap320 selected), per cell and pooled
    compute   the round's own measured-geometry FLOP model, 1 generator forward at native against
              8 generator forwards at cap320 + 8 verifier forwards at 1,003,520

The greedy arm is deterministic, so it carries no seed noise; the cap320 selected arm has three
seeds and the delta is reported against each.

    python3 src/cascade_methods/resolution_greedy_vs_arm.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}
NBOOT, BSEED = 10000, 20260814


def norm(s):
    return str(s).strip().lower()


def load_arm(cap, tag):
    out = {}
    for ds in DS:
        p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{tag}.jsonl")
        if not os.path.exists(p):
            return None
        d = {}
        for l in open(p):
            if l.strip():
                try:
                    r = json.loads(l)
                    d[r["idx"]] = r
                except Exception:
                    pass
        if len(d) < NEXP[ds]:
            return None
        out[ds] = d
    return out


def boot(d, nboot=NBOOT, seed=BSEED):
    d = np.asarray(d, float)
    if len(d) == 0:
        return None, [None, None]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(nboot, len(d)))
    s = d[idx].mean(axis=1)
    return float(d.mean()), [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    J = json.load(open(os.path.join(SWEEP, "judge_cache.json")))
    V = json.load(open(os.path.join(SWEEP, "verifier_score_cache.json")))
    cost = json.load(open(os.path.join(OUT, "cost_by_resolution.json")))

    order = []
    for ds, nm in [("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"), ("pathvqa_open", "pathvqa")]:
        p = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint",
                         f"transfer_dump_{nm}_open_lingshu7b.json")
        for r in json.load(open(p)):
            order.append((ds, r["idx"]))
    assert len(order) == 2345

    def greedy_vec(arm):
        v = []
        for ds, idx in order:
            r = arm[ds].get(idx)
            if r is None:
                v.append(0); continue
            a = r["preds"][0] if r["preds"] else ""
            v.append(int(J.get(f"{ds}|{idx}|{norm(a)}", 0) or 0))
        return np.array(v)

    def selected_vec(arm):
        v = []
        for ds, idx in order:
            r = arm[ds].get(idx)
            if r is None:
                v.append(0); continue
            preds = r["preds"]
            y = [J.get(f"{ds}|{idx}|{norm(a)}") for a in preds]
            sc = [V.get(f"{ds}|{idx}|{a}") for a in preds]
            yv = [0 if t is None else int(t) for t in y]
            svv = [-1e9 if t is None else float(t) for t in sc]
            v.append(int(yv[int(np.argmax(svv))] == 1))
        return np.array(v)

    nat_t0 = load_arm("native", "t0")
    c320_t0 = load_arm("cap320", "t0")
    if nat_t0 is None or c320_t0 is None:
        raise SystemExit("need native t0 and cap320 t0")
    gN, gC = greedy_vec(nat_t0), greedy_vec(c320_t0)

    seeds = [t for t in ["s0", "s1", "s2"] if load_arm("cap320", t) is not None]
    sel = {t: selected_vec(load_arm("cap320", t)) for t in seeds}
    dsm = {ds: np.array([d == ds for d, _ in order]) for ds in DS}

    # ---- compute --------------------------------------------------------------------------
    per = cost["open_half_per_candidate"]
    f_gen_native = per["native"]["flops_per_candidate"]
    f_gen_cap320 = per["cap320"]["flops_per_candidate"]
    f_ver = cost["where_the_open_arm_spends_its_flops"]["at_the_deployed_operating_point"][
        "verifier_per_candidate_at_1003520"]
    arm_deployed = 8 * f_gen_cap320 + 8 * f_ver
    arm_greedy_native = 1 * f_gen_native

    res = {
        "_question": "can ONE greedy decode at NATIVE resolution replace EIGHT samples at cap320 "
                     "plus the LoRA verifier that ranks them?",
        "_matched": "native t0 and every cap320 arm were generated by the same script in the SAME "
                    "session (2026-08-13, gpu_mem 0.30) -- see arm_provenance.json. The comparison "
                    "is therefore not exposed to the +-0.008 serving-config caveat.",
        "_label": "LLM judge (src/labeling/run_judge.py, MedVLThinker-32B, text-only).",
        "_endpoint": "the frozen 2,345-question open pool, same item order as "
                     "src/training_methods/genframe_data.py.",
        "accuracy": {}, "compute": {}, "per_cell": {},
    }

    pooled = {}
    for t in seeds:
        d, ci = boot(gN.astype(float) - sel[t].astype(float))
        pooled[t] = {"native_greedy_t0": round(float(gN.mean()), 6),
                     "cap320_selected_best_of_8_plus_verifier": round(float(sel[t].mean()), 6),
                     "delta": round(d, 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
                     "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    res["accuracy"]["native_greedy_MINUS_cap320_selected"] = {
        "per_control_seed": pooled,
        "delta_mean_over_seeds": round(float(np.mean([pooled[t]["delta"] for t in seeds])), 6),
        "all_seeds_ci_exclude_zero": bool(all(pooled[t]["ci_excludes_zero"] for t in seeds)),
        "_read": "positive means one greedy decode at native resolution is MORE accurate than the "
                 "deployed eight-sample cap320 arm with its trained verifier.",
    }
    d, ci = boot(gN.astype(float) - gC.astype(float))
    res["accuracy"]["native_greedy_MINUS_cap320_greedy"] = {
        "cap320_greedy_t0": round(float(gC.mean()), 6),
        "native_greedy_t0": round(float(gN.mean()), 6),
        "delta": round(d, 6), "ci95": [round(ci[0], 6), round(ci[1], 6)],
        "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        "_read": "the pure resolution effect on a single deterministic decode: no sampling, no "
                 "selection, no verifier.",
    }
    for ds in DS:
        m = dsm[ds]
        pc = {}
        for t in seeds:
            dd, cc = boot(gN[m].astype(float) - sel[t][m].astype(float))
            pc[t] = {"delta": round(dd, 6), "ci95": [round(cc[0], 6), round(cc[1], 6)],
                     "ci_excludes_zero": bool(cc[0] > 0 or cc[1] < 0)}
        res["per_cell"][ds] = {
            "n": int(m.sum()),
            "native_greedy_t0": round(float(gN[m].mean()), 6),
            "cap320_selected_per_seed": {t: round(float(sel[t][m].mean()), 6) for t in seeds},
            "delta_vs_cap320_selected": pc,
            "worse_than_deployed_on_this_cell": bool(
                np.mean([pc[t]["delta"] for t in seeds]) < 0)}

    # ---- the macro-8 basis: EQUAL WEIGHT PER CELL, cell-stratified bootstrap ------------------
    # The pooled delta above averages over QUESTIONS; the macro-8 averages over CELLS. With
    # 645/200/1500 items the two differ a lot here, and the macro basis is the one the project
    # reports, so both are given -- never one alone.
    def macro_boot(dvec, nboot=NBOOT, seed=BSEED):
        rng = np.random.default_rng(seed)
        cells = [np.asarray(dvec, float)[dsm[ds]] for ds in DS]
        point = float(np.mean([c.mean() for c in cells]))
        reps = np.zeros(nboot)
        for c in cells:
            idx = rng.integers(0, len(c), size=(nboot, len(c)))
            reps += c[idx].mean(axis=1)
        reps /= len(cells)
        return point, [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]

    macro = {}
    for t in seeds:
        dv = gN.astype(float) - sel[t].astype(float)
        pt, ci = macro_boot(dv)
        pc = {ds: float(dv[dsm[ds]].mean()) for ds in DS}
        macro[t] = {
            "per_cell_delta": {k: round(v, 6) for k, v in pc.items()},
            "open3_equal_weight_mean_delta": round(pt, 6),
            "open3_ci95": [round(ci[0], 6), round(ci[1], 6)],
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
            "macro8_equivalent": round(pt * 0.375, 6),
            "macro8_equivalent_ci95": [round(ci[0] * 0.375, 6), round(ci[1] * 0.375, 6)],
            "leave_one_cell_out_macro8": {
                ds: round(float(np.mean([pc[o] for o in DS if o != ds]) * 0.375), 6) for ds in DS},
            "net_items_changed_per_cell": {
                ds: {"n": int(dsm[ds].sum()),
                     "net_items_changed": int(round(pc[ds] * int(dsm[ds].sum())))} for ds in DS},
        }
    res["accuracy"]["macro8_basis_equal_weight_per_cell"] = {
        "per_control_seed": macro,
        "project_significance_threshold_on_macro8": 0.0029,
        "_read": "on the macro's own equal-weight-per-cell basis the swap looks much BETTER than "
                 "pooled, because vqa_rad_open (n=200), where the greedy native arm is "
                 "significantly ahead, carries a full third of the open weight instead of 8.5%. "
                 "Both bases are reported; neither is quoted alone.",
        "_fragility": "read net_items_changed_per_cell before quoting any of this: vqa_rad_open's "
                      "contribution is a small number of items in a 200-item cell, and "
                      "leave_one_cell_out_macro8 shows what the macro becomes without it.",
    }

    res["compute"] = {
        "flops_generator_per_candidate_native": f_gen_native,
        "flops_generator_per_candidate_cap320": f_gen_cap320,
        "flops_verifier_per_candidate_1003520": f_ver,
        "deployed_open_arm_8gen_cap320_plus_8verify": arm_deployed,
        "one_greedy_decode_at_native": arm_greedy_native,
        "ratio_greedy_native_over_deployed_arm": round(arm_greedy_native / arm_deployed, 6),
        "compute_reduction_x": round(arm_deployed / arm_greedy_native, 4),
        "_flop_model": cost["_meta"]["flop_model"],
        "_geometry": cost["_meta"]["geometry"],
        "_caveat": "prefill-inclusive whole-forward FLOPs on this round's own MEASURED token "
                   "geometry. It is an arithmetic re-costing of measured token counts, not a "
                   "wall-clock or energy measurement; no end-to-end run was performed, which is "
                   "the standing caveat on every operating point in this project.",
    }
    res["_the_catch"] = (
        "this compares a GREEDY arm against a SAMPLED arm, so it is not a resolution result -- it "
        "is a result about what the sampling+verification machinery is worth once the generator "
        "sees the image properly. It also removes the verifier's confidence signal, which the "
        "deployed open policy uses to decide when to escalate to the 32B. That escalation gate is "
        "NOT reproducible from a single greedy decode, so this is not a drop-in replacement for "
        "the deployed open arm -- it is a bound on what the arm's sampling half is buying.")
    res["_the_8_is_the_frozen_metric_not_the_deployed_average_N"] = (
        "the 'selected' endpoint is best-of-EIGHT, which is what the project's frozen open-text "
        "metric (src/training_methods/genframe_data.py) defines and what every number in this "
        "round's accuracy tables uses. The DEPLOYED open policy draws an ADAPTIVE N (Weitzman), so "
        "its average N -- and therefore its true per-item compute -- is at most 8 and in general "
        "lower. The 13.17x compute reduction is therefore an UPPER bound on what this swap saves "
        "in the deployed policy: at an average N of 4 it would be ~6.6x, at N=2 ~3.3x. The "
        "ACCURACY comparison is unaffected, because it is against the best-of-8 number the metric "
        "actually reports.")
    res["_not_measured"] = (
        "the 32B escalation leg, latency, energy, the deployed policy's average N, and any effect "
        "on the MCQ half. The open pool's n is 2,345 and vqa_rad_open is n=200, so its per-cell "
        "interval is wide.")

    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "greedy_native_vs_deployed_arm.json"), "w"), indent=1)
    print(json.dumps(res["accuracy"], indent=1))
    print(json.dumps(res["compute"], indent=1))
    print(json.dumps({k: {"n": v["n"], "native_greedy": v["native_greedy_t0"],
                          "cap320_selected": v["cap320_selected_per_seed"],
                          "worse": v["worse_than_deployed_on_this_cell"]}
                      for k, v in res["per_cell"].items()}, indent=1))
    print("wrote", os.path.join(OUT, "greedy_native_vs_deployed_arm.json"))


if __name__ == "__main__":
    main()
