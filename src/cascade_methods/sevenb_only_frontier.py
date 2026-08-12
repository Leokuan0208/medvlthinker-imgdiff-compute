#!/usr/bin/env python3
"""sevenb_only_frontier.py -- ATTACK 3: THE 7B-ONLY FRONTIER.

How close does a Lingshu-7B-plus-verifier pipeline with NO 32B AT TEST TIME get to
always-32B-direct (macro 0.6567, 8 cells, Variant B), what part of the shortfall is
selection vs coverage vs raw capability, and what is the MINIMUM 32B usage that closes the
rest -- with its VRAM consequence, which is the fact the user's goal actually turns on.

NO GPU.  NO NEW INFERENCE.  Every number is read from an existing per-item dump or from a
dated artifact that is named inline.

    python3 src/cascade_methods/sevenb_only_frontier.py
"""
import json, os, sys, itertools
import numpy as np

ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_sevenb_frontier_parts")
OUT = os.path.join(ART, "sevenb_only_frontier_2026-08-12.json")
os.makedirs(PARTS, exist_ok=True)

NBOOT = 10000
SEED = 20260812
N_FOLD_SEEDS = 12
N_FOLDS = 5
TIE_TOL = -0.0029          # the round's pre-registered tie tolerance (= published CI half-width)

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
MCQ = CELLS[:5]
OPEN = CELLS[5:]
OPEN_KEY = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}

os.environ.setdefault("OMP_NUM_THREADS", "1")

import integrated_method as IM       # noqa: E402
import genframe_data as G           # noqa: E402


# ======================================================================================
# bootstrap -- one shared item-resample stream per cell, reused by every policy
# ======================================================================================
def make_boot_index(ns, nboot=NBOOT, seed=SEED):
    """ONE shared item-resample stream per cell, drawn once and reused by every policy."""
    rng = np.random.default_rng(seed)
    return {c: rng.integers(0, n, size=(nboot, n), dtype=np.int32) for c, n in ns.items()}


def macro_boot_delta(A, B, BI):
    """paired item bootstrap of macro(A) - macro(B).  ONE gather per cell, on the DIFFERENCE
    vector d = A - B, which is exact (the resample is shared and paired) and half the work."""
    d = np.mean([A[c].mean() - B[c].mean() for c in CELLS])
    draws = np.zeros(NBOOT)
    for c in CELLS:
        dv = (A[c] - B[c]).astype(np.float32)
        draws += dv[BI[c]].mean(axis=1)
    draws /= len(CELLS)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": round(float(d), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "sig": bool(lo > 0 or hi < 0), "ties_at_tol": bool(lo >= TIE_TOL)}


def cell_boot_delta(a, b, ix):
    d = a.mean() - b.mean()
    draws = (a - b).astype(np.float32)[ix].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta": round(float(d), 6), "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "worse_sig": bool(hi < 0)}


def main():
    out = {
        "title": ("ATTACK 3 -- THE 7B-ONLY FRONTIER: how close does a 7B+verifier pipeline with NO 32B "
                  "at test time get to always-32B-direct, what limits it, and what is the minimum 32B "
                  "usage (and VRAM class) that closes the rest?"),
        "date": "2026-08-12",
        "reproduce": "python3 src/cascade_methods/sevenb_only_frontier.py",
        "no_gpu": True, "no_new_inference": True, "no_fabricated_numbers": True,
        "not_abstention": ("every policy here returns an answer on every item.  No reject option, no "
                           "defer-to-human, no selective prediction.  CRITICAL RULE 6 respected."),
        "convention": ("MACRO, equal weight per reporting cell, Variant B (MMMU excluded), 8 cells, "
                       "1/8 each, n=42,224.  CLEAN disjoint open-text verifier "
                       "(ckpts/train/lora_verifier_disjoint)."),
        "numerics": {"OMP_NUM_THREADS": 1, "tf32": "not applicable -- pure numpy/CPU on stored vectors",
                     "row_order": "the stored dump order, unchanged (genframe_data canonical item order)",
                     "bootstrap": "one shared item-resample stream per cell, drawn once, reused by every policy",
                     "nboot": NBOOT, "seed": SEED, "n_fold_seeds": N_FOLD_SEEDS, "n_folds": N_FOLDS},
        "sources": {
            "per_item_correctness": "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz (cascade_selector_rerun_2026-08-05.json, arm 'disjoint')",
            "open_pool": "src/training_methods/genframe_data.py FROZEN block -> ckpts/train/lora_verifier_disjoint/transfer_dump_*.json",
            "frozen_selector": "ckpts/train/genframe_head_ens8 via src/training_methods/genframe_selector.py (READ ONLY -- freeze_selector.py was NOT run)",
            "mcq_gate_features": "MedEvalKit/eval_results_lingshu7b_full via src/cascade_methods/integrated_method.py mcq_closed/mcq_medxpert",
            "coverage_ceilings": "results/cascade_methods/artifacts/coverage_diagnosis_2026-08-10.json (SCOUT B) -- reused verbatim, not recomputed",
            "mcq_7b_only_tta": "results/cascade_methods/artifacts/mcq_tta_2026-08-10.json -- reused verbatim",
            "vram": "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json",
            "cost_frontier_sibling": "results/cascade_methods/artifacts/cost_floor_2026-08-10.json and min_escalation_2026-08-12*.json (ATTACK 4, same round)",
        },
    }

    # ==================================================================================
    # LOAD
    # ==================================================================================
    CACHE = os.path.join(PARTS, "loaded.npz")
    V = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))
    ok7 = {c: V[f"{c}|always_7b"].astype(float) for c in CELLS}
    ok32 = {c: V[f"{c}|always_32b_direct"].astype(float) for c in CELLS}
    NS = {c: len(ok7[c]) for c in CELLS}
    BI = make_boot_index(NS)

    items = G.load_items()
    SL = {}          # cell -> (n,8) judge labels in pool order
    GREEDY = {}      # cell -> (n,) greedy correctness
    INC = {}         # cell -> (n,8) incumbent verifier scores
    for cell in OPEN:
        ds = OPEN_KEY[cell]
        rows = [it for it in items if it["ds"] == ds]
        SL[cell] = np.array([it["sl"] for it in rows], dtype=float)
        GREEDY[cell] = np.array([it["greedy_ok"] for it in rows], dtype=float)
        INC[cell] = np.array([it["scores"] for it in rows], dtype=float)

    # ==================================================================================
    # NULL TESTS
    # ==================================================================================
    nulls = {}

    # N1 -- per-cell published accuracies reproduce from the stored per-item vectors
    pub = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))
    pm = pub["per_arm"]["disjoint"]["macro_acc"]
    dev = max(abs(np.mean([ok7[c].mean() for c in CELLS]) - pm["always_7b"]),
              abs(np.mean([ok32[c].mean() for c in CELLS]) - pm["always_32b_direct"]))
    nulls["N1_published_macro_baselines"] = {
        "always_7b_recomputed": round(float(np.mean([ok7[c].mean() for c in CELLS])), 6),
        "always_7b_published": pm["always_7b"],
        "always_32b_direct_recomputed": round(float(np.mean([ok32[c].mean() for c in CELLS])), 6),
        "always_32b_direct_published": pm["always_32b_direct"],
        "max_abs_deviation": round(float(dev), 8),
        "passed": bool(dev < 5e-5),
        "note": "published values are rounded to 4 dp in the source artifact; tolerance is 5e-5.",
    }

    # N2 -- the frozen open-text bar reproduces
    r = G.sel_eff(G.incumbent_scores(), items)
    P = G.PUBLISHED
    d2 = max(abs(r["oracle"] - P["oracle@8"]), abs(r["acc"] - P["selected"]),
             abs(r["greedy"] - P["greedy"]), abs(r["sel_eff"] - P["sel_eff"]),
             *[abs(r["per_ds"][k]["sel_eff"] - v) for k, v in P["per_ds"].items()])
    nulls["N2_frozen_opentext_bar"] = {
        "n": r["n"], "oracle@8": round(r["oracle"], 6), "selected": round(r["acc"], 6),
        "greedy": round(r["greedy"], 6), "sel_eff": round(r["sel_eff"], 6),
        "per_ds_sel_eff": {k: round(v["sel_eff"], 6) for k, v in r["per_ds"].items()},
        "max_abs_deviation": round(float(d2), 9), "passed": bool(d2 < 1e-6),
    }

    # N3 -- the EXACT identity selected = oracle x sel_eff (the form the brief mandates)
    ident = {}
    for k, v in r["per_ds"].items():
        ident[k] = abs(v["oracle"] * v["sel_eff"] - v["acc"])
    ident["_pooled"] = abs(r["oracle"] * r["sel_eff"] - r["acc"])
    nulls["N3_multiplicative_identity"] = {
        "identity": "selected = oracle@8 x sel_eff, with sel_eff the CONDITIONAL mean P(pick correct | pool recoverable)",
        "max_abs_deviation": float(max(ident.values())),
        "passed": bool(max(ident.values()) < 1e-12),
        "corroborates": "coverage_diagnosis_2026-08-10.json HEADLINE.1 (max |err| 5.6e-17)",
        "never_use": "the additive form selected ~= greedy + sel_eff*(oracle-greedy); it over-predicts by +0.090 to +0.111 per cell",
    }

    # N4 -- my independent MCQ load from the MedEvalKit dumps matches the stored vectors
    gate = {}
    mcq_dev = {}
    _cache = dict(np.load(CACHE, allow_pickle=True)) if os.path.exists(CACHE) else {}
    for cell in MCQ:
        if f"margin|{cell}" in _cache:
            gate[cell] = _cache[f"margin|{cell}"]
            mcq_dev[cell] = {"ok7": float(_cache[f"dev7|{cell}"]), "ok32": float(_cache[f"dev32|{cell}"])}
            continue
        if cell == "MedXpertQA-MM":
            d = IM.mcq_medxpert()
        else:
            ds = cell.split("_closed")[0]
            closed = {"SLAKE_closed": "SLAKE", "VQA_RAD_closed": "YESNO",
                      "PATH_VQA_closed": "YESNO", "PMC_VQA": None}[cell]
            d = IM.mcq_closed(ds, closed)
        assert len(d["ok7"]) == NS[cell], (cell, len(d["ok7"]), NS[cell])
        mcq_dev[cell] = {"ok7": float(np.max(np.abs(d["ok7"] - ok7[cell]))),
                         "ok32": float(np.max(np.abs(d["ok32"] - ok32[cell])))}
        gate[cell] = np.asarray(d["margin"], dtype=float)
        _cache[f"margin|{cell}"] = gate[cell]
        _cache[f"dev7|{cell}"] = mcq_dev[cell]["ok7"]
        _cache[f"dev32|{cell}"] = mcq_dev[cell]["ok32"]
    mx = max(max(v.values()) for v in mcq_dev.values())
    nulls["N4_independent_mcq_reload"] = {
        "what": ("the 5 MCQ cells were re-loaded from MedEvalKit/eval_results_lingshu7b_full and "
                 "eval_results_lingshu32b_full through integrated_method's own loaders and compared "
                 "element-wise to _selector_rerun_parts/vec_disjoint.npz.  This validates that the "
                 "gate scalar (7B margin) I attach is in the SAME item order as the correctness vectors."),
        "per_cell_max_abs_deviation": mcq_dev, "max_abs_deviation": mx, "passed": bool(mx == 0.0),
    }

    # N5 -- the frozen 8-seed selector reproduces from disk (read-only)
    try:
        import genframe_selector as GS
        if "ens8|SLAKE_open" in _cache:
            ver = json.loads(str(_cache["ens8_verify"]))
            ok_ens = {c: _cache[f"ens8|{c}"] for c in OPEN}
            nulls["N5_frozen_ens8_selector"] = ver
            raise StopIteration
        ver = GS.verify()
        nulls["N5_frozen_ens8_selector"] = {
            "max_abs_deviation": ver.get("max_abs_deviation"), "passed": bool(ver.get("pass")),
            "measured_sel_eff": ver.get("measured", {}).get("sel_eff"),
            "note": "READ ONLY.  src/training_methods/freeze_selector.py was NOT run -- it REWRITES the frozen artifact.",
        }
        SELE = GS.FrozenSelector.load()
        ens_scores, _Llogits = GS.score_eval_pool(SELE)
        ens_r = G.sel_eff(ens_scores, items)
        ens_picks = ens_r["picks"]
        ok_ens = {}
        o = 0
        for cell in OPEN:
            n = NS[cell]
            ok_ens[cell] = np.array([SL[cell][i, ens_picks[o + i]] for i in range(n)], dtype=float)
            o += n
        for c in OPEN:
            _cache[f"ens8|{c}"] = ok_ens[c]
        _cache["ens8_verify"] = json.dumps(nulls["N5_frozen_ens8_selector"])
    except StopIteration:
        pass
    except Exception as e:
        nulls["N5_frozen_ens8_selector"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
        ok_ens = None
    np.savez(CACHE, **_cache)
    out["null_tests"] = nulls
    out["null_tests"]["ALL_PASSED"] = bool(all(v.get("passed", True) for v in nulls.values()))

    # ==================================================================================
    # PART 1 -- THE 7B-ONLY FRONTIER
    # ==================================================================================
    # 7B-only policy menu, per cell.  Every entry is an EXISTING measurement.
    def bok(cell, k, scores):
        """best-of-k over the first k pool slots, picked by `scores` (argmax, first-index tie-break)."""
        s = scores[cell][:, :k]
        p = np.argmax(s, axis=1)
        return SL[cell][np.arange(len(p)), p]

    menu = {}
    for cell in MCQ:
        menu[cell] = {"greedy_7b": ok7[cell]}
    for cell in OPEN:
        m = {"greedy_7b": GREEDY[cell]}
        for k in range(1, 9):
            m[f"bo{k}_incumbent_verifier"] = bok(cell, k, INC)
        if ok_ens is not None:
            m["bo8_frozen_ens8_selector"] = ok_ens[cell]
        menu[cell] = m

    menu_acc = {c: {k: round(float(v.mean()), 6) for k, v in m.items()} for c, m in menu.items()}

    # honest per-cell arm choice: 5-fold cross-fit, 12 fold seeds
    rng_master = np.random.default_rng(SEED)
    seed_macros, seed_picks, per_seed_vec = [], [], []
    for si in range(N_FOLD_SEEDS):
        rs = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        deliv, picks = {}, {}
        for cell in CELLS:
            n = NS[cell]
            fold = rs.permutation(n) % N_FOLDS
            v = np.zeros(n)
            chosen = []
            for f in range(N_FOLDS):
                tr, te = fold != f, fold == f
                best = max(menu[cell], key=lambda k: menu[cell][k][tr].mean())
                chosen.append(best)
                v[te] = menu[cell][best][te]
            deliv[cell] = v
            picks[cell] = chosen
        seed_macros.append(float(np.mean([deliv[c].mean() for c in CELLS])))
        seed_picks.append(picks)
        per_seed_vec.append(deliv)

    # the deployable per-item vector = the seed-MEAN policy is not a vector, so report the
    # seed mean scalar AND bootstrap the modal-seed vector for a CI
    mid = int(np.argsort(seed_macros)[len(seed_macros) // 2])
    best7b = per_seed_vec[mid]
    d_vs_direct = macro_boot_delta(best7b, ok32, BI)
    d_vs_7b = macro_boot_delta(best7b, ok7, BI)

    # eval-visible ceilings (UPPER BOUNDS, unreachable)
    ceil_perfect = {c: (ok7[c] if c in MCQ else SL[c].max(axis=1)) for c in CELLS}
    macro_7b_greedy = float(np.mean([ok7[c].mean() for c in CELLS]))
    macro_direct = float(np.mean([ok32[c].mean() for c in CELLS]))

    out["PART1_7B_only_frontier"] = {
        "question": "best achievable 8-cell macro with NO 32B at test time",
        "menu_per_cell_accuracy_EVAL_VISIBLE": menu_acc,
        "menu_note": ("MCQ cells carry ONE 7B-only entry because no other 7B-only MCQ mechanism has ever "
                      "measured positive on this pool: MCQ test-time augmentation is measured NEGATIVE "
                      "(always-K summed MCQ gain -0.0078; gated cross-fit +0.0000062) in "
                      "mcq_tta_2026-08-10.json, and sampling-based best-of-N is structurally dead on MCQ "
                      "(PMC verifier pick 0.4325 BELOW greedy 0.5060; MedXpert oracle@8 0.5365 BELOW its "
                      "own luck floor 0.6808)."),
        "honest_crossfit": {
            "protocol": f"{N_FOLDS}-fold cross-fit arm choice per cell, {N_FOLD_SEEDS} fold-split seeds; "
                        "the arm is chosen on the 4 training folds and applied to the held-out fold",
            "macro_seed_mean": round(float(np.mean(seed_macros)), 6),
            "macro_seed_sd": round(float(np.std(seed_macros, ddof=1)), 6),
            "macro_seed_min": round(float(np.min(seed_macros)), 6),
            "macro_seed_max": round(float(np.max(seed_macros)), 6),
            "median_seed_macro": round(seed_macros[mid], 6),
            "arms_chosen": {c: sorted(set(sum([sp[c] for sp in seed_picks], [])))
                            for c in CELLS},
            "vs_always_32b_direct": d_vs_direct,
            "vs_always_7b_greedy": d_vs_7b,
        },
        "baselines": {"always_7b_macro": round(macro_7b_greedy, 6),
                      "always_32b_direct_macro": round(macro_direct, 6),
                      "gap_to_close": round(macro_direct - macro_7b_greedy, 6)},
        "eval_visible_upper_bounds_UNREACHABLE": {
            "perfect_selector_over_the_current_8_pool": {
                "macro": round(float(np.mean([ceil_perfect[c].mean() for c in CELLS])), 6),
                "per_cell": {c: round(float(ceil_perfect[c].mean()), 6) for c in CELLS},
                "what": "MCQ cells held at 7B greedy (a perfect selector over the OPTION set is trivially "
                        "1.0 and carries no information); open cells at oracle@8.  UPPER BOUND.",
            },
        },
    }

    # ==================================================================================
    # PART 2 -- THE CAPABILITY FLOOR
    # ==================================================================================
    cov = json.load(open(os.path.join(ART, "coverage_diagnosis_2026-08-10.json")))
    lp = cov["part2_ceiling"]["per_cell"]
    best7_percell = {c: float(best7b[c].mean()) for c in CELLS}

    percell = {}
    for c in CELLS:
        g = float(ok32[c].mean()) - best7_percell[c]
        if c in OPEN:
            ds = OPEN_KEY[c]
            oracle8 = float(SL[c].max(axis=1).mean())
            ceil = lp[ds]["LP_estimated_reachable_share"]
            sel_head = max(0.0, oracle8 - best7_percell[c])
            cov_head = max(0.0, ceil - oracle8)
            coverage_wall = 1.0 - oracle8
            kind = ("selection-limited" if sel_head >= g else
                    ("coverage-then-capability" if sel_head + cov_head >= g else "CAPABILITY-LIMITED"))
        else:
            oracle8 = None
            ceil = None
            sel_head = 0.0     # measured: every 7B-only MCQ re-selection mechanism is <= 0
            cov_head = 0.0
            coverage_wall = 0.0
            kind = ("already at/above the 32B" if g <= 0 else "CAPABILITY-LIMITED")
        sel_used = min(max(g, 0.0), sel_head)
        cov_used = min(max(g, 0.0) - sel_used, cov_head)
        resid = max(0.0, g) - sel_used - cov_used
        percell[c] = {
            "n": NS[c], "format": "MCQ" if c in MCQ else "open-text",
            "acc_7b_only_best": round(best7_percell[c], 6),
            "acc_always_32b_direct": round(float(ok32[c].mean()), 6),
            "gap": round(g, 6),
            "macro_share_of_the_gap": round(g / 8.0, 6),
            "oracle@8": None if oracle8 is None else round(oracle8, 6),
            "coverage_wall_P_no_correct_answer_available": round(coverage_wall, 6),
            "iid_sampling_ceiling_LP_N_infinity": None if ceil is None else round(ceil, 6),
            "headroom_from_better_SELECTION_over_the_current_pool": round(sel_head, 6),
            "headroom_from_better_COVERAGE_up_to_the_iid_ceiling": round(cov_head, 6),
            "gap_closable_by_SELECTION_capped_at_the_gap": round(sel_used, 6),
            "gap_closable_by_COVERAGE_capped_at_the_gap": round(cov_used, 6),
            "residual_pure_capability": round(resid, 6),
            "verdict": kind,
        }

    out["PART2_capability_floor"] = {
        "question": "for each cell, P(no correct answer is available to the 7B under ANY selection), and "
                    "which cells are capability-limited rather than selection-limited",
        "mcq_coverage_note": (
            "On all 5 MCQ cells the candidate set is COMPLETE BY CONSTRUCTION -- the gold answer is always "
            "one of the presented options (PMC-VQA 4 options, MedXpert 5 options) or one of {yes, no} "
            "(SLAKE-closed / VQA-RAD-closed / PathVQA-closed, which are free-form yes/no, verified in "
            "MedEvalKit/eval_results_lingshu7b_full/*/results.json: choices==None and answer in {yes,no}).  "
            "So the coverage wall is 0 and the oracle-over-candidates is trivially 1.0.  That upper bound "
            "is VACUOUS: it says only 'if the 7B could tell which option is right, it would be right'.  "
            "The honest MCQ ceiling is therefore the best MEASURED 7B-only mechanism, and every one tried "
            "is <= greedy.  MCQ cells are entered here with selection headroom 0, which is a MEASUREMENT, "
            "not an assumption -- see mcq_tta_2026-08-10.json and the luck-floor result in the round brief."),
        "per_cell": percell,
        "capability_limited_cells": [c for c in CELLS if percell[c]["verdict"] == "CAPABILITY-LIMITED"],
        "cells_where_the_7B_already_matches_or_beats_the_32B": [c for c in CELLS if percell[c]["gap"] <= 0],
        "macro_decomposition_of_the_gap": {
            "total_gap_best7Bonly_to_direct": round(macro_direct - float(np.mean(list(best7_percell.values()))), 6),
            "closable_by_SELECTION_over_the_current_pool": round(
                float(np.mean([percell[c]["gap_closable_by_SELECTION_capped_at_the_gap"] for c in CELLS])), 6),
            "closable_by_COVERAGE_up_to_the_iid_sampling_ceiling": round(
                float(np.mean([percell[c]["gap_closable_by_COVERAGE_capped_at_the_gap"] for c in CELLS])), 6),
            "residual_PURE_CAPABILITY": round(
                float(np.mean([percell[c]["residual_pure_capability"] for c in CELLS])), 6),
            "caveat": ("each lever is CAPPED at the cell's own gap, in the order selection -> coverage -> "
                       "residual, before averaging, so the three parts sum EXACTLY to the total gap.  The "
                       "UNCAPPED headroom is reported per cell alongside."),
            "uncapped_headroom_for_reference": {
                "SELECTION": round(float(np.mean([percell[c]["headroom_from_better_SELECTION_over_the_current_pool"] for c in CELLS])), 6),
                "COVERAGE": round(float(np.mean([percell[c]["headroom_from_better_COVERAGE_up_to_the_iid_ceiling"] for c in CELLS])), 6)},
        },
        "reused_verbatim_from": "coverage_diagnosis_2026-08-10.json part2_ceiling (Lincoln-Petersen / Chao "
                               "two-sample, endpoint-8 vs independent-16, judge-labelled; a LOWER bound on "
                               "the reachable share, so an UPPER bound on the coverage headroom).",
    }

    # ==================================================================================
    # PART 3 -- THE MINIMUM-32B FRONTIER
    # ==================================================================================
    # gate scalars: MCQ = 7B margin; open = max incumbent verifier score (the deployed convention)
    for cell in OPEN:
        gate[cell] = INC[cell].max(axis=1)

    # (a) EXACT cell-subset enumeration -- eval-visible, DIAGNOSTIC, but exact and leakage-free
    #     in the sense that no per-item fitting happens: a whole cell is routed or not.
    subs = []
    for mask in itertools.product([0, 1], repeat=8):
        acc = np.mean([ok32[c].mean() if m else best7_percell[c] for c, m in zip(CELLS, mask)])
        esc_macro = float(np.mean(mask))
        esc_sw = float(sum(NS[c] for c, m in zip(CELLS, mask) if m) / sum(NS.values()))
        subs.append({"cells_to_32B": [c for c, m in zip(CELLS, mask) if m],
                     "macro_acc": round(float(acc), 6),
                     "macro_escalation_fraction": round(esc_macro, 6),
                     "sample_weighted_escalation_fraction": round(esc_sw, 6)})
    subs.sort(key=lambda r: (r["macro_escalation_fraction"], -r["macro_acc"]))
    pareto, bestacc = [], -1
    for r_ in sorted(subs, key=lambda r: r["macro_escalation_fraction"]):
        if r_["macro_acc"] > bestacc:
            pareto.append(r_); bestacc = r_["macro_acc"]
    tie_cell = [r_ for r_ in pareto if r_["macro_acc"] >= macro_direct + TIE_TOL]
    min_tie = min(tie_cell, key=lambda r: r["macro_escalation_fraction"]) if tie_cell else None

    # (b) ITEM-LEVEL cross-fit escalation-budget sweep
    BUDGETS = [0.0, 0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]
    CI_AT = set(BUDGETS)   # every budget gets a full CI; the grid is coarse enough to afford it
    sweep = []
    for B in BUDGETS:
        seed_acc, vecs = [], []
        for si in range(N_FOLD_SEEDS):
            rs = np.random.default_rng(SEED + 1000 + si)
            deliv = {}
            for cell in CELLS:
                n = NS[cell]
                fold = rs.permutation(n) % N_FOLDS
                base = best7b[cell]
                v = np.zeros(n)
                for f in range(N_FOLDS):
                    tr, te = fold != f, fold == f
                    # label-free: escalate the lowest-gate B fraction, threshold set on TRAIN
                    thr = np.quantile(gate[cell][tr], B) if B > 0 else -np.inf
                    esc = gate[cell][te] <= thr if B > 0 else np.zeros(te.sum(), bool)
                    v[te] = np.where(esc, ok32[cell][te], base[te])
                deliv[cell] = v
            seed_acc.append(float(np.mean([deliv[c].mean() for c in CELLS])))
            vecs.append(deliv)
        m = int(np.argsort(seed_acc)[len(seed_acc) // 2])
        dd = macro_boot_delta(vecs[m], ok32, BI)
        # actual realised escalation (quantile on train applied to test is not exactly B)
        realised = float(np.mean([np.mean(gate[c] <= np.quantile(gate[c], B)) if B > 0 else 0.0 for c in CELLS]))
        sweep.append({"budget": B, "macro_escalation_realised": round(realised, 4),
                      "macro_acc_seed_mean": round(float(np.mean(seed_acc)), 6),
                      "macro_acc_seed_sd": round(float(np.std(seed_acc, ddof=1)), 6),
                      "delta_vs_direct": dd["delta"], "lo": dd["lo"], "hi": dd["hi"],
                      "ties_at_pre_registered_tol": dd["ties_at_tol"],
                      "not_significantly_worse": bool(dd["lo"] <= 0 <= dd["hi"]) or dd["delta"] > 0})
    first_tie = next((s for s in sweep if s["ties_at_pre_registered_tol"]), None)
    first_ns = next((s for s in sweep if s["not_significantly_worse"]), None)

    # permutation null on the gate
    perm = []
    for B in [0.2, 0.4]:
        rp = np.random.default_rng(SEED + 77)
        pg = {c: gate[c][rp.permutation(NS[c])] for c in CELLS}
        deliv = {}
        for cell in CELLS:
            thr = np.quantile(pg[cell], B)
            deliv[cell] = np.where(pg[cell] <= thr, ok32[cell], best7b[cell])
        perm.append({"budget": B, "macro_acc_permuted_gate": round(float(np.mean([deliv[c].mean() for c in CELLS])), 6),
                     "macro_acc_real_gate": next(s["macro_acc_seed_mean"] for s in sweep if s["budget"] == B)})

    R32 = {"paper_4.57": 4.57, "derived_3.816": 3.816}
    def simple_cost(esc, r32):
        """SIMPLE forward-pass count, in units of one always-32B-direct pass.  NOT cost_floor's
        prefill-inclusive as-charged convention -- use cost_floor_2026-08-10.json for cost headlines."""
        return round((1.0 + esc * r32) / r32, 4)

    out["PART3_minimum_32B_frontier"] = {
        "question": "the minimum fraction of questions that must go to the 32B to restore the tie with "
                    "always-32B-direct, and what it costs in compute and in VRAM CLASS",
        "tie_definition": f"paired item bootstrap, nboot={NBOOT}; 95% CI lower bound of (policy - always-32B-direct) >= {TIE_TOL} "
                          "(the round's pre-registered tolerance, = the published CI half-width)",
        "a_exact_cell_subset_enumeration": {
            "what": "all 2^8 = 256 subsets of cells routed WHOLESALE to always-32B-direct, the rest on the "
                    "best 7B-only policy.  Exact, no per-item fitting, but the subset is chosen on eval -> "
                    "DIAGNOSTIC (an eval-visible lower bound on the escalation needed).",
            "pareto_frontier": pareto,
            "minimum_escalation_that_ties": min_tie,
        },
        "b_item_level_crossfit_budget_sweep": {
            "what": "per-cell escalation of the lowest-gate fraction B; gate = 7B answer margin (MCQ) or "
                    "max incumbent verifier score over the 8 candidates (open).  Threshold is a TRAIN-fold "
                    "quantile (label-free), 5-fold cross-fit, 12 fold seeds.  The 7B-only fallback is "
                    "PART1's cross-fit policy.",
            "sweep": sweep,
            "first_budget_that_ties_at_pre_registered_tol": first_tie,
            "first_budget_not_significantly_worse": first_ns,
            "permutation_null_on_the_gate": {
                "protocol": "the per-item gate scalar is permuted within each cell; correctness vectors untouched",
                "rows": perm,
                "reading": "if the permuted gate matches the real gate the frontier is manufactured by the "
                           "escalation RATE alone and the gate carries nothing.",
            },
            "cost_SIMPLE_CONVENTION": {
                "definition": "(one 7B pass + escalation x R32 7B-equivalents) / R32, i.e. in units of one "
                              "always-32B-direct forward pass.  A COUNT OF FORWARD PASSES, deliberately "
                              "simpler than cost_floor_2026-08-10.json's prefill-inclusive as-charged "
                              "convention; do not mix the two.  Open-cell best-of-k generation is NOT "
                              "charged here -- see cost_floor for that.",
                "rows": [{"escalation": s["macro_escalation_realised"],
                          **{f"x_direct|R32={k}": simple_cost(s["macro_escalation_realised"], v)
                             for k, v in R32.items()}} for s in sweep if s["budget"] in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)],
            },
        },
        "cross_reference": ("ATTACK 4 of this same round (min_escalation_2026-08-12*.json) sweeps the same "
                           "question with a richer per-cell policy menu and a nested-CV lambda.  Where the "
                           "two disagree, prefer ATTACK 4 for the COST endpoint and this file for the VRAM "
                           "consequence.  cost_floor_2026-08-10.json is the prior art on the cost side: its "
                           "verdict is that the pre-registered tie survives only at 1.029x-1.118x "
                           "always-32B-direct compute across 12 seeds, i.e. NOT cheaper."),
    }


    # ==================================================================================
    # PART 1b -- WHAT WOULD HAVE TO BE TRUE: the sel_eff a 7B-only pipeline needs to TIE
    # ==================================================================================
    mcq_sum = float(sum(ok7[c].mean() for c in MCQ))
    o8 = {c: float(SL[c].max(axis=1).mean()) for c in OPEN}
    lp_ceiling = {c: lp[OPEN_KEY[c]]["LP_estimated_reachable_share"] for c in OPEN}
    target = macro_direct
    need_sum_open = 8.0 * target - mcq_sum
    req_now = need_sum_open / sum(o8.values())
    req_at_ceiling = need_sum_open / sum(lp_ceiling.values())
    cur_sel_eff = {c: (best7_percell[c] / o8[c]) for c in OPEN}
    out["PART1b_what_would_have_to_be_true"] = {
        "framing": ("Hold the 5 MCQ cells at 7B greedy -- measured, and with no 7B-only lever that improves "
                    "them.  Then the whole tie rests on the 3 open cells, where the EXACT identity "
                    "selected = oracle@8 x sel_eff applies per cell.  Solve for the sel_eff that ties."),
        "mcq_five_cell_sum_at_7B_greedy": round(mcq_sum, 6),
        "open_three_cell_sum_required_to_tie": round(need_sum_open, 6),
        "oracle@8_per_open_cell": {c: round(v, 6) for c, v in o8.items()},
        "iid_sampling_ceiling_per_open_cell": {c: round(v, 6) for c, v in lp_ceiling.items()},
        "REQUIRED_sel_eff_over_the_CURRENT_8_sample_pool": round(req_now, 6),
        "REQUIRED_sel_eff_if_coverage_is_ALSO_pushed_to_the_iid_ceiling": round(req_at_ceiling, 6),
        "sel_eff_ACHIEVED_today_per_open_cell": {c: round(v, 6) for c, v in cur_sel_eff.items()},
        "field_constant": ("~20 independent verifier/selection architectures converge on sel_eff 0.80-0.81 "
                           "(docs/current/COMPARATIVE_VERIFIER_2026-08-05.md); the incumbent clean verifier "
                           "is 0.775204 and the frozen 8-seed ensemble 0.810627 "
                           "(genframe_data.py FROZEN block / frozen_selector_ens8_2026-08-05.json).  The "
                           "seed spread alone (~0.021) exceeds every architectural effect measured."),
        "reading": ("the required sel_eff is the number to compare against 0.81.  If it exceeds 1.0 the tie is "
                    "IMPOSSIBLE from selection alone at the current coverage; if it exceeds 0.81 by more than "
                    "the seed spread, no measured architecture reaches it."),
        "perfect_selector_at_the_iid_ceiling_macro": round((mcq_sum + sum(lp_ceiling.values())) / 8.0, 6),
    }

    # ==================================================================================
    # PART 4 -- THE VRAM CLIFF, and LOAD-ON-DEMAND
    # ==================================================================================
    vr = json.load(open(os.path.join(ART, "vram_testtime_2026-08-11.json")))
    SC = vr["scenarios"]
    w7 = 15.4937      # S1 a_weights_resident_gib
    w32 = 62.3125     # S2 a_weights_resident_gib
    ctx = SC["S2_lingshu32b_direct_mcq"]["contamination_audit"]["cuda_context_offset_gib"]
    p7 = SC["S1_lingshu7b_direct_mcq"]["b_peak_allocated_gib"]["peak"]
    p32 = SC["S2_lingshu32b_direct_mcq"]["b_peak_allocated_gib"]["peak"]
    f7_mcq = SC["S1_lingshu7b_direct_mcq"]["d_process_footprint_gib"]["peak"]
    f32 = SC["S2_lingshu32b_direct_mcq"]["d_process_footprint_gib"]["peak"]
    f_open = SC["S4_opentext_bestof8_full_arm"]["d_process_footprint_gib"]["peak"]
    act32 = p32 - w32
    CARD = 80.0
    lod = json.load(open(os.path.join(PARTS, "load_on_demand.json")))

    out["PART4_the_VRAM_cliff_and_load_on_demand"] = {
        "why_this_is_the_decisive_section": (
            "the user's goal is 'match the 32B with LESS VRAM than the 32B'.  Accuracy and compute are "
            "continuous in the escalation rate; VRAM is NOT.  A policy that escalates 0% of queries and a "
            "policy that escalates 0.1% are in DIFFERENT HARDWARE CLASSES, because the second one has to "
            "have the 32B somewhere."),
        "measured_inputs": {
            "source": "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json (HF transformers, bf16, FA2, tp=1, batch 1, one A100 80GB)",
            "lingshu7b_weights_resident_gib": w7,
            "lingshu32b_weights_resident_gib": w32,
            "lingshu7b_peak_allocated_gib": p7,
            "lingshu32b_peak_allocated_gib": p32,
            "lingshu32b_peak_activations_gib_DERIVED": round(act32, 4),
            "cuda_context_offset_gib": ctx,
            "lingshu7b_mcq_process_footprint_peak_gib": f7_mcq,
            "lingshu7b_opentext_bestof8_arm_process_footprint_peak_gib": f_open,
            "lingshu32b_process_footprint_peak_gib": f32,
            "card_capacity_gib": CARD,
            "independent_corroboration_of_w32": {
                "on_disk_bf16_safetensors_gib": lod["total_gib"],
                "vram_a_weights_resident_gib": w32,
                "abs_difference_gib": round(abs(lod["total_gib"] - w32), 4),
                "reading": "the 14 safetensors shards on disk and the CUDA allocation after from_pretrained "
                           "agree to 0.002 GiB.  Two independent instruments, same number.",
            },
        },
        "arithmetic_verification_requested_by_the_brief": {
            "brief_claim": "7B+32B weights alone are 15.49+62.31 = 77.80 GiB, and with the 32B's measured "
                           "+5.6 GiB of peak activations that is ~83 GiB, i.e. it does NOT fit on one 80 GB card",
            "weights_only_sum_gib": round(w7 + w32, 4),
            "thirtytwob_peak_activations_gib": round(act32, 4),
            "weights_plus_32b_activations_gib": round(w7 + w32 + act32, 4),
            "VERDICT": "CONFIRMED.  Both addends are MEASURED (a_weights_resident_gib); the 5.5555 GiB "
                       "activation term is DERIVED as (b_peak_allocated peak - a_weights_resident) on the "
                       "32B's own worst item, MedXpert MM-1561 (46,816 vision tokens).  The sum is "
                       f"{round(w7 + w32 + act32, 2)} GiB against an 80.0 GiB card.",
            "the_arithmetic_is_conservative_in_TWO_ways": [
                "it omits the CUDA context (measured 1.3835 GiB per process), which every real process pays",
                "it omits caching-allocator fragmentation (c_peak_reserved exceeds b_peak_allocated by up to 3.35 GiB on the 32B)",
            ],
            "with_the_cuda_context_gib": round(w7 + w32 + act32 + ctx, 4),
            "cleanest_measured_only_statement": {
                "expression": "the 7B open-text best-of-8 arm's MEASURED process footprint + the 32B's MEASURED resident weights",
                "gib": round(f_open + w32, 4),
                "over_capacity_by_gib": round(f_open + w32 - CARD, 4),
                "reading": "both terms measured, the only operation is addition, and it ALREADY exceeds the "
                           "card before the 32B executes a single forward pass.",
            },
            "even_the_most_favourable_arithmetic_fails": {
                "weights_only_plus_one_shared_context_gib": round(w7 + w32 + ctx, 4),
                "headroom_left_on_an_80_GiB_card_gib": round(CARD - (w7 + w32 + ctx), 4),
                "reading": "0.81 GiB of headroom is not enough for a single 32B prefill (measured 5.56 GiB) "
                           "nor for a single 7B prefill (measured 2.53 GiB).  The most generous accounting "
                           "still does not run.",
            },
        },
        "the_two_deployment_REGIMES": {
            "A_never_loads_the_32B": {
                "policies": "escalation rate exactly 0",
                "measured_footprint_gib": f_open,
                "smallest_card": "24 GB (RTX 4090 / L4) -- vram_testtime_2026-08-11.json deployer_guidance",
                "pct_of_one_A100_80GB": round(100 * f_open / CARD, 1),
                "best_macro_accuracy_ATTAINABLE": out["PART1_7B_only_frontier"]["honest_crossfit"]["macro_seed_mean"],
                "delta_vs_always_32b_direct": out["PART1_7B_only_frontier"]["honest_crossfit"]["vs_always_32b_direct"],
            },
            "B_needs_the_32B_resident": {
                "policies": "ANY escalation rate > 0, however small, on a single-card deployment",
                "single_card_one_process_gib_DERIVED": round(w7 + w32 + act32 + ctx, 4),
                "fits_on_one_80_GiB_card": False,
                "two_process_additive_footprint_gib": round(f7_mcq + f32, 4),
                "minimum_hardware": "TWO 80 GB cards (or one >= 96 GB card, e.g. H100 NVL / H200); the "
                                    "deployment is no longer 'a 7B box'",
                "note": "vram_testtime_2026-08-11.json lists co-residency explicitly under not_measured "
                        "('cut for time.  Additive from (d)').  This section supplies it, as DERIVED "
                        "arithmetic over that file's measurements -- not as a new measurement.",
            },
            "C_load_on_demand": {
                "idea": "keep only the 7B resident and page the 32B in from disk when a query escalates",
                "measured": {
                    "model_bytes_on_disk_gib": lod["total_gib"],
                    "cold_sequential_read_s": lod["cold_read"]["seconds"],
                    "cold_read_gib_per_s": lod["cold_read"]["gib_per_s"],
                    "page_cache_warm_read_gib_per_s": lod["warm_read_one_shard"]["gib_per_s"],
                    "host_to_device_gib_per_s": lod["h2d_bandwidth"].get("gib_per_s"),
                    "composed_cold_swap_in_s": lod["composed_swap_in"]["total_cold_swap_in_s"],
                    "composed_warm_swap_in_s": lod["composed_swap_in"]["total_warm_swap_in_s_page_cache_hot"],
                },
                "measurement_caveat": (
                    "measured on 2026-08-12 on a SHARED /data mount while two sibling GPU jobs of this same "
                    "round were running (both A100s at 59-100% util).  0.152 GiB/s is therefore a "
                    "CONTENDED-STORAGE figure, and per-shard throughput drifts downward across the run "
                    "(0.210 -> 0.098 GiB/s), which is the signature of growing contention.  On idle NVMe "
                    "this would be far faster.  It is reported as measured, under the conditions stated, and "
                    "must not be quoted as an idle-system number.  The H2D leg (22.9 GiB/s) is healthy and "
                    "is NOT the bottleneck -- the 32B's weights cross PCIe in 2.72 s."),
                "VERDICT": (
                    "NOT DEPLOYABLE as an on-demand swap in this environment.  A cold swap-in is "
                    f"{lod['composed_swap_in']['total_cold_swap_in_s']:.0f} s "
                    f"(~{lod['composed_swap_in']['total_cold_swap_in_s']/60:.1f} min) against a 32B forward "
                    "pass of 1.88 s mean / 6.01 s max (vram_testtime S2 wall_s).  Even the page-cache-warm "
                    f"path is {lod['composed_swap_in']['total_warm_swap_in_s_page_cache_hot']:.0f} s -- and "
                    "'warm' means 62.31 GiB of HOST RAM is dedicated to holding the model, which is a "
                    "larger, not smaller, resource commitment than keeping it on a second card."),
                "which_regime_are_we_in": (
                    "Regime B.  The measured minimum escalation that ties (PART3) is 75% of macro weight / "
                    "17.3% of items -- not 5% -- so the 32B is not a rare guest but a co-equal tier, and "
                    "load-on-demand is not even a candidate.  The honest options are: (i) accept regime A "
                    "and its measured accuracy shortfall, or (ii) accept a two-card deployment."),
            },
        },
        "consequence_for_the_users_goal": (
            "'less VRAM than the 32B' is achievable ONLY in regime A -- 18.76 GiB measured, 3.9x smaller "
            "than always-32B-direct's 72.60 GiB, on a 24 GB card instead of an 80 GB card.  But regime A's "
            "measured accuracy is 0.6030 macro against the 32B's 0.6567, a SIGNIFICANT shortfall of "
            "-0.0528 [-0.0646, -0.0408].  Every policy that recovers that accuracy is in regime B, where "
            "the footprint is not 'less than the 32B' -- it is the 32B PLUS the 7B, on more hardware than "
            "always-32B-direct needs.  There is no measured operating point in between."),
    }

    json.dump(out, open(os.path.join(PARTS, "core.json"), "w"), indent=2, default=str)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    print(json.dumps({"nulls": {k: v.get("passed") for k, v in nulls.items() if isinstance(v, dict)},
                      "part1_macro_seed_mean": out["PART1_7B_only_frontier"]["honest_crossfit"]["macro_seed_mean"],
                      "vs_direct": d_vs_direct,
                      "min_tie_cellsubset": min_tie,
                      "first_tie_budget": first_tie}, indent=2))
    print("WROTE", os.path.join(PARTS, "core.json"))
    return out


if __name__ == "__main__":
    main()
