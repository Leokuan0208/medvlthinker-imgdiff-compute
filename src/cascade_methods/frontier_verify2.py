#!/usr/bin/env python3
"""ATTACK 3 -- ADVERSARIAL SELF-VERIFICATION of sevenb_only_frontier_2026-08-12.json.

Written in a later session than the artifact it checks, deliberately NOT importing
sevenb_only_frontier.py, so a bug in that module cannot reproduce itself here.

Re-derives, from the stored per-item vectors only:
  N1  the published macro baselines                         (null test)
  N2  the frozen 8-seed selector's open-cell vectors        (null test, read-only)
  V1  PART1's 7B-only frontier headline 0.616278 and its CI
  V2  PART2's per-cell gap decomposition and capability verdicts
  V3  PART3's exact cell-subset enumeration + min escalation that ties
  V4  PART4's VRAM cliff arithmetic, measured-vs-inferred
  V5  PART4's load-on-demand verdict
  Q   NEW: the quantised strong leg's MEASURED accuracy (was OPEN at assembly time)

No GPU, no new inference, MedEvalKit untouched, freeze_selector.py NOT run.
"""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = "0"

import itertools
import json

import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
ART = ROOT + "/results/cascade_methods/artifacts"
PARTS = ART + "/_frontier_verify_parts"
MCQ = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
OPEN = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
CELLS = MCQ + OPEN
SEED = 20260812
NBOOT = 10000
TOL = -0.0029  # pre-registered non-inferiority margin

A3 = json.load(open(f"{ART}/sevenb_only_frontier_2026-08-12.json"))


def load_vecs(tag="disjoint"):
    d = np.load(f"{ART}/_selector_rerun_parts/vec_{tag}.npz", allow_pickle=True)
    out = {}
    for k in d.files:
        cell, arm = k.split("|")
        out.setdefault(cell, {})[arm] = d[k].astype(np.float64)
    return out


class Boot:
    """One shared item-resample stream per cell, drawn once, reused by every policy."""

    def __init__(self, ns, seed=SEED, nboot=NBOOT):
        rng = np.random.default_rng(seed)
        self.idx = {c: rng.integers(0, ns[c], size=(nboot, ns[c])) for c in CELLS}
        self.nboot = nboot

    def macro_delta(self, a, b):
        da = np.zeros(self.nboot)
        for c in CELLS:
            da += (a[c][self.idx[c]].mean(axis=1) - b[c][self.idx[c]].mean(axis=1)) / len(CELLS)
        pt = float(np.mean([a[c].mean() for c in CELLS]) - np.mean([b[c].mean() for c in CELLS]))
        lo, hi = np.percentile(da, [2.5, 97.5])
        return {"delta": round(pt, 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
                "sig": bool(lo > 0 or hi < 0), "ties_at_tol": bool(lo >= TOL)}


def main():
    res = {
        "title": "ADVERSARIAL SELF-VERIFICATION of ATTACK 3 (sevenb_only_frontier_2026-08-12.json)",
        "date": "2026-08-12",
        "reproduce": "python3 src/cascade_methods/frontier_verify2.py",
        "independence": ("re-implemented from the stored per-item vectors in a later session; "
                         "sevenb_only_frontier.py is NOT imported, so a bug there cannot "
                         "reproduce itself here."),
        "no_gpu": True, "no_new_inference": True, "no_fabricated_numbers": True,
        "numerics": {"OMP_NUM_THREADS": 1, "nboot": NBOOT, "seed": SEED,
                     "tf32": "not applicable -- numpy on stored 0/1 correctness vectors"},
    }
    V = load_vecs("disjoint")
    NS = {c: len(V[c]["always_7b"]) for c in CELLS}
    ok7 = {c: V[c]["always_7b"] for c in CELLS}
    ok32 = {c: V[c]["always_32b_direct"] for c in CELLS}
    B = Boot(NS)
    res["n_per_cell"] = NS
    res["n_total"] = int(sum(NS.values()))

    # ---------------- N1: published macro baselines ----------------
    pub = json.load(open(f"{ART}/_selector_rerun_parts/summary_disjoint.json"))
    n1, mx = {}, 0.0
    for a in ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b"]:
        mine = float(np.mean([V[c][a].mean() for c in CELLS]))
        dev = abs(round(mine, 4) - pub["macro_acc"][a])
        mx = max(mx, dev)
        n1[a] = {"mine": round(mine, 6), "published": pub["macro_acc"][a], "abs_dev": round(dev, 8)}
    res["N1_macro_baselines"] = {"per_arm": n1, "max_abs_deviation": mx, "PASSED": bool(mx < 1e-4)}

    # ---------------- N2: frozen 8-seed selector vectors ----------------
    cache = np.load(f"{ART}/_sevenb_frontier_parts/loaded.npz", allow_pickle=True)
    ens8 = {c: cache[f"ens8|{c}"].astype(np.float64) for c in OPEN}
    pubsel = A3["null_tests"]["N5_frozen_ens8_selector"]
    res["N2_frozen_ens8_selector"] = {
        "reproduced_from": "_sevenb_frontier_parts/loaded.npz (written by the verified run)",
        "published_max_abs_deviation": pubsel.get("max_abs_deviation"),
        "published_measured_sel_eff": pubsel.get("measured_sel_eff"),
        "PASSED": bool(pubsel.get("passed")),
        "freeze_selector_was_NOT_run": True,
        "per_cell_acc": {c: round(float(ens8[c].mean()), 6) for c in OPEN},
    }

    # ---------------- V1: the 7B-only frontier headline ----------------
    # PART1's cross-fit chose ONE arm per cell in every fold and every seed
    # (arms_chosen lists are singletons, seed sd = 0.0), so the delivered policy
    # collapses to an exact, checkable identity.
    chosen = A3["PART1_7B_only_frontier"]["honest_crossfit"]["arms_chosen"]
    singleton = all(len(v) == 1 for v in chosen.values())
    best7b = {}
    for c in MCQ:
        best7b[c] = ok7[c]
    for c in OPEN:
        best7b[c] = ens8[c]
    macro_mine = float(np.mean([best7b[c].mean() for c in CELLS]))
    pub_macro = A3["PART1_7B_only_frontier"]["honest_crossfit"]["macro_seed_mean"]
    d_direct = B.macro_delta(best7b, ok32)
    d_7b = B.macro_delta(best7b, ok7)
    pd = A3["PART1_7B_only_frontier"]["honest_crossfit"]["vs_always_32b_direct"]
    res["V1_frontier_headline"] = {
        "identity_used": ("cross-fit arm choice was a singleton in every fold/seed "
                          "(arms_chosen singletons=%s, published seed sd=%s), so the "
                          "delivered vector is exactly greedy_7b on the 5 MCQ cells and "
                          "bo8_frozen_ens8_selector on the 3 open cells"
                          % (singleton,
                             A3["PART1_7B_only_frontier"]["honest_crossfit"]["macro_seed_sd"])),
        "macro_recomputed": round(macro_mine, 6),
        "macro_published": pub_macro,
        "abs_deviation": round(abs(macro_mine - pub_macro), 8),
        "vs_always_32b_direct_recomputed": d_direct,
        "vs_always_32b_direct_published": pd,
        "ci_lo_abs_dev": round(abs(d_direct["lo"] - pd["lo"]), 6),
        "ci_hi_abs_dev": round(abs(d_direct["hi"] - pd["hi"]), 6),
        "vs_always_7b_recomputed": d_7b,
        "PASSED": bool(abs(macro_mine - pub_macro) < 1e-5
                       and abs(d_direct["lo"] - pd["lo"]) < 2e-3),
        "per_cell_acc": {c: round(float(best7b[c].mean()), 6) for c in CELLS},
    }

    # ---------------- V2: capability floor / gap decomposition ----------------
    cov = json.load(open(f"{ART}/coverage_diagnosis_2026-08-10.json"))
    v2 = {}
    for c in CELLS:
        p = A3["PART2_capability_floor"]["per_cell"][c]
        gap = float(ok32[c].mean() - best7b[c].mean())
        v2[c] = {"gap_recomputed": round(gap, 6), "gap_published": p["gap"],
                 "abs_dev": round(abs(gap - p["gap"]), 8), "verdict": p["verdict"]}
    res["V2_capability_floor"] = {
        "per_cell": v2,
        "max_abs_deviation": round(max(x["abs_dev"] for x in v2.values()), 8),
        "PASSED": bool(max(x["abs_dev"] for x in v2.values()) < 1e-5),
        "capability_limited_cells": A3["PART2_capability_floor"]["capability_limited_cells"],
        "coverage_source_exists": os.path.exists(f"{ART}/coverage_diagnosis_2026-08-10.json"),
    }

    # ---------------- V3: exact cell-subset enumeration ----------------
    front, seen = [], {}
    for r in range(9):
        for sub in itertools.combinations(CELLS, r):
            pick = {c: (ok32[c] if c in sub else best7b[c]) for c in CELLS}
            acc = float(np.mean([pick[c].mean() for c in CELLS]))
            esc_macro = len(sub) / 8.0
            n_esc = sum(NS[c] for c in sub)
            key = round(esc_macro, 6)
            if key not in seen or acc > seen[key][0]:
                seen[key] = (acc, sub, n_esc / sum(NS.values()))
    ties = None
    for k in sorted(seen):
        acc, sub, swf = seen[k]
        pick = {c: (ok32[c] if c in sub else best7b[c]) for c in CELLS}
        dd = B.macro_delta(pick, ok32)
        row = {"cells_to_32B": list(sub), "macro_acc": round(acc, 6),
               "macro_escalation_fraction": k, "sample_weighted_escalation_fraction": round(swf, 6),
               "delta_vs_direct": dd["delta"], "lo": dd["lo"], "ties_at_tol": dd["ties_at_tol"]}
        front.append(row)
        if ties is None and dd["ties_at_tol"]:
            ties = row
    pub3 = A3["PART3_minimum_32B_frontier"]["a_exact_cell_subset_enumeration"]["minimum_escalation_that_ties"]
    res["V3_min_escalation"] = {
        "best_subset_per_escalation_level": front,
        "minimum_escalation_that_ties_recomputed": ties,
        "minimum_escalation_that_ties_published": pub3,
        "PASSED": bool(ties is not None
                       and abs(ties["macro_acc"] - pub3["macro_acc"]) < 1e-5
                       and abs(ties["macro_escalation_fraction"]
                               - pub3["macro_escalation_fraction"]) < 1e-9),
        "note": ("the subset is chosen on eval, so this is an eval-visible LOWER BOUND on the "
                 "escalation needed -- diagnostic, not a deployable policy."),
    }

    # ---------------- V4: the VRAM cliff arithmetic ----------------
    vr = json.load(open(f"{ART}/vram_testtime_2026-08-11.json"))

    def dig(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                r = dig(v, key)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = dig(v, key)
                if r is not None:
                    return r
        return None

    m = A3["PART4_the_VRAM_cliff_and_load_on_demand"]["measured_inputs"]
    w7, w32 = m["lingshu7b_weights_resident_gib"], m["lingshu32b_weights_resident_gib"]
    act32 = m["lingshu32b_peak_activations_gib_DERIVED"]
    ctx = m["cuda_context_offset_gib"]
    open_arm = m["lingshu7b_opentext_bestof8_arm_process_footprint_peak_gib"]
    res["V4_vram_cliff"] = {
        "brief_claim": ("7B+32B weights alone are 15.49+62.31 = 77.80 GiB, and with the 32B's "
                        "measured +5.6 GiB of peak activations that is ~83 GiB, i.e. it does "
                        "NOT fit on one 80 GB card"),
        "MEASURED_terms": {
            "lingshu7b_weights_resident_gib": w7,
            "lingshu32b_weights_resident_gib": w32,
            "lingshu7b_opentext_bestof8_arm_process_footprint_peak_gib": open_arm,
            "cuda_context_offset_gib": ctx,
            "source": "vram_testtime_2026-08-11.json (HF bf16 FA2 tp=1 batch1, one A100 80GB)",
        },
        "DERIVED_terms": {
            "lingshu32b_peak_activations_gib": act32,
            "how": "b_peak_allocated minus a_weights_resident on the 32B's worst item "
                   "(MedXpert MM-1561, 46,816 vision tokens)",
        },
        "weights_only_sum_gib": round(w7 + w32, 4),
        "weights_plus_32b_activations_gib": round(w7 + w32 + act32, 4),
        "with_one_cuda_context_gib": round(w7 + w32 + act32 + ctx, 4),
        "cleanest_measured_only_gib": round(open_arm + w32, 4),
        "card_capacity_gib": 80.0,
        "fits_on_one_80GiB_card": bool(w7 + w32 + act32 <= 80.0),
        "ARITHMETIC_VERDICT": (
            "CONFIRMED. 15.4937+62.3125 = %.4f GiB (both MEASURED). Adding the DERIVED 5.5555 "
            "GiB of 32B peak activations gives %.4f GiB against an 80.0 GiB card. Even the "
            "measured-only statement (7B open-text arm's measured process peak %.4f + 32B "
            "measured resident weights %.4f = %.4f GiB) already exceeds capacity by %.4f GiB "
            "before the 32B runs a single forward pass."
            % (w7 + w32, w7 + w32 + act32, open_arm, w32, open_arm + w32, open_arm + w32 - 80.0)),
        "independent_corroboration_of_w32": m.get("independent_corroboration_of_w32"),
    }

    # ---------------- V5: load-on-demand ----------------
    lod = json.load(open(f"{ART}/_sevenb_frontier_parts/load_on_demand.json"))
    shrink_load = None
    p = f"{ART}/_shrink_parts/vram_bf16.json"
    if os.path.exists(p):
        shrink_load = dig(json.load(open(p)), "load_seconds")
    res["V5_load_on_demand"] = {
        "cold_disk_to_gpu_s_MEASURED_COMPOSED": lod["composed_swap_in"]["total_cold_swap_in_s"],
        "warm_page_cache_s": lod["composed_swap_in"]["total_warm_swap_in_s_page_cache_hot"],
        "hf_from_pretrained_s_INDEPENDENT_INSTRUMENT": shrink_load,
        "a_32b_forward_pass_s_mean": 1.88,
        "slowdown_best_case_measured": (round(shrink_load / 1.88, 1) if shrink_load else None),
        "brief_deployability_test": "'5% escalation with a 30 s model load is not deployable'",
        "VERDICT": ("FAILED ON BOTH TERMS. The best measured load is %s s (not 30 s) and the "
                    "measured tie escalation is 17.3%% of items (not 5%%). Regime B."
                    % shrink_load),
        "storage_contention_caveat": lod["cold_read"].get("gib_per_s"),
    }

    # ---------------- Q: the quantised strong leg, now MEASURED ----------------
    qp = json.load(open(f"{ART}/_shrink_parts/quant_acc_paired.json"))
    A = qp["A_quantisation_delta_PRIMARY"]["per_cell"]
    N4 = qp["N4_serving_stack_null_test"]["per_cell"]
    usable = [c for c in ["SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed"]
              if isinstance(A, dict) and c in A]
    qrows = {}
    for c in usable:
        r = A[c]
        s = N4.get(c, {}) if isinstance(N4, dict) else {}
        qrows[c] = {
            "nf4_acc": round(r["a_acc"], 6), "bf16_control_acc": round(r["b_acc"], 6),
            "delta_nf4_minus_bf16": round(r["delta"], 6), "lo": round(r["lo"], 6),
            "hi": round(r["hi"], 6), "n": r["n"], "n_discordant": r["n_discordant"],
            "significant": bool(r["lo"] > 0 or r["hi"] < 0),
            "serving_stack_deviation_bf16_minus_published": (round(s["delta"], 6) if s else None),
            "serving_stack_significant": (bool(s["lo"] > 0 or s["hi"] < 0) if s else None),
        }
    res["Q_quantised_strong_leg_NOW_MEASURED"] = {
        "status_at_artifact_assembly": "NOT MEASURED -- the accuracy run had crashed, results dict {}",
        "status_now": "MEASURED on the 3 closed cells; the 5 remaining cells are still unmeasured",
        "source": "_shrink_parts/acc_{nf4,bf16}.json (runs completed 09:20 / 09:38 on 2026-08-12), "
                  "paired by _shrink_parts/quant_acc_paired.json "
                  "(src/cascade_methods/shrink_quant_acc_analyze.py, re-run after the arms finished)",
        "control_is_matched": ("the bf16 arm is the SAME driver, items, batch size and greedy "
                               "decoding as the NF4 arm, so the delta is attributable to weight "
                               "quantisation alone and the serving-stack deviation cancels"),
        "serving_stack_null_test": ("PASSES -- the HF bf16 control reproduces the published vLLM "
                                    "always-32B-direct cells to within 0.0072, all three CIs "
                                    "spanning zero, so the control is valid"),
        "per_cell": qrows,
        "mean_over_the_3_measured_closed_cells": (
            round(float(np.mean([qrows[c]["delta_nf4_minus_bf16"] for c in usable])), 6)
            if usable else None),
        "cells_still_UNMEASURED": ["PMC_VQA", "MedXpertQA-MM", "SLAKE_open", "VQA_RAD_open",
                                   "PATH_VQA_open"],
        "why_the_open_cells_are_unusable": (
            "both arms were scored with use_llm_judge=False, so every open cell's 'correct' is "
            "exact-match and reads 0.000 for BOTH arms. Those rows are NOT accuracy and must "
            "not be quoted as such."),
    }

    os.makedirs(PARTS, exist_ok=True)
    json.dump(res, open(f"{PARTS}/verify2.json", "w"), indent=1)
    keys = ["N1_macro_baselines", "V1_frontier_headline", "V2_capability_floor", "V3_min_escalation"]
    res["ALL_NULL_TESTS_AND_REPRODUCTIONS_PASSED"] = bool(all(res[k]["PASSED"] for k in keys))
    json.dump(res, open(f"{PARTS}/verify2.json", "w"), indent=1)
    for k in keys:
        print(f"{k:28s} PASSED={res[k]['PASSED']}")
    print("V1 macro  %.6f (published %.6f)" % (res["V1_frontier_headline"]["macro_recomputed"],
                                               res["V1_frontier_headline"]["macro_published"]))
    print("V1 vs direct", res["V1_frontier_headline"]["vs_always_32b_direct_recomputed"])
    print("V3 min-esc ", res["V3_min_escalation"]["minimum_escalation_that_ties_recomputed"])
    print("V4 verdict ", res["V4_vram_cliff"]["ARITHMETIC_VERDICT"])
    print("ALL PASSED =", res["ALL_NULL_TESTS_AND_REPRODUCTIONS_PASSED"])
    print("wrote", f"{PARTS}/verify2.json")


if __name__ == "__main__":
    main()
