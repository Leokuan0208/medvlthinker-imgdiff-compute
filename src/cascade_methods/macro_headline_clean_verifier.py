#!/usr/bin/env python3
"""
macro_headline_clean_verifier.py -- recompute the ENTIRE 8-cell MACRO headline (accuracy AND cost)
with the open-text arm's verifier swapped from the CONTAMINATED adapter to the CLEAN
(disjoint-trained) one.

WHY THIS EXISTS.
  * `verifier_disjoint_retrain_2026-07-30.json` measured the de-contamination cost as
    0.5750 -> 0.5722 (-0.0028) on the full suite.  That is a SAMPLE-WEIGHTED number, where the three
    open-text cells are 2,345 / 42,374 = 5.5% of the items.
  * `macro_average_headline_2026-07-30.json` re-based the project's primary headline on the 8-cell
    MACRO average, where the same three open cells carry 3/8 = 37.5% of the weight.
  * Nobody had computed the macro headline ON A CLEAN VERIFIER.  The two corrections
    (equal weighting; uncontaminated verifier) have never been applied together, and the honest
    re-costing is a third correction that interacts with both.  This script applies all three.

WHAT IS SWAPPED, AND ONLY THAT.
  The open-text candidate lists (`preds`), the per-candidate judge labels (`sl`), the greedy label
  (`greedy_ok`) and the 32B strong-leg judge labels are IDENTICAL across arms -- verified here by
  assertion, item by item.  ONLY the verifier `scores` differ.  The swap is implemented as an exact
  path redirect of the three `transfer_dump_{ds}_lingshu7b.json` files at the `builtins.open` level,
  so NOT ONE LINE of the scoring/aggregation machinery is duplicated or re-implemented:
  `paper_baselines.build_cells`, `method_final_mmmu_corrected.add_v2_vectors`,
  `opentext_32b_think_full.measured_open_think`, `honest_recosting`, and every helper inside
  `macro_average_headline` are imported and called unchanged.

  Three read sites consume the dump, all covered by the redirect:
      integrated_method.open_bestof8            (bo8 pick + verifier-confidence gate; sets the
                                                 Pandora iso-accuracy target a_fix)
      integrated_pandora.load_open_rows         (per-candidate scores for the Pandora controller)
      beat32b_more.open_features                (7B-side features for the F10 L2D rejector)

LEVELS.
  L1 (headline)  = no eval IMAGE and no eval ITEM in verifier training; question TEMPLATES may recur.
  L2 (lower bnd) = L1 + no eval question TEXT at all.  On these templated sets L2 deletes legitimate
                   in-distribution coverage (PathVQA: 7,306 of 9,903 train items dropped), so it
                   conflates de-contamination with distribution shift.  Reported, labelled, NOT the
                   headline.

VALIDATION (run aborts on failure).
  The `contaminated` arm must reproduce `macro_average_headline_2026-07-30.json` EXACTLY -- accuracy
  levels, every delta point estimate AND both CI bounds, per-cell escalation, and every cost ratio.
  The RNG stream is replayed in the same order, so agreement is exact, not "within Monte-Carlo noise";
  any drift is a wiring error, not a finding.

STATISTICS.  Identical protocol to macro_average_headline.py: items are resampled WITHIN each cell
(paired across systems, common random numbers, exact pattern-multinomial draw) and the macro average
is recomputed per replicate.  ==> The CIs reflect WITHIN-dataset sampling noise ONLY; they do NOT
capture dataset-selection noise.  With 5-8 datasets, resampling datasets is hopelessly unstable, so a
leave-one-cell-out point range is reported instead.

NO GPU, no new inference, no fabricated numbers.  Launch from the repo root:
    python3 src/cascade_methods/macro_headline_clean_verifier.py
"""
import builtins, os, sys, json, time

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")

# =================================================================================================
# 0.  VERIFIER SWAP -- an exact-path redirect installed on builtins.open BEFORE anything is read.
#     Nothing else in the pipeline is modified.
# =================================================================================================
_REAL_OPEN = builtins.open
_REDIRECT = {}


def _redirecting_open(file, *a, **kw):
    try:
        p = os.path.abspath(os.fspath(file))
    except Exception:
        return _REAL_OPEN(file, *a, **kw)
    return _REAL_OPEN(_REDIRECT.get(p, file), *a, **kw)


builtins.open = _redirecting_open

CONTAM_ADAPTER = "ckpts/train/lora_verifier_pooled4"
ARMS = [("contaminated", CONTAM_ADAPTER,
         "deployed verifier -- trained on 67-73% of the very items it scores"),
        ("clean_L1",     "ckpts/train/lora_verifier_disjoint",
         "HEADLINE: no eval image, no eval item; question templates may recur"),
        ("clean_L2",     "ckpts/train/lora_verifier_disjoint_l2",
         "LOWER BOUND ONLY: L1 + no eval question text (starves the in-domain pools)")]
OPEN_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
DUMP = "{ad}/transfer_dump_{ds}_lingshu7b.json"


def dump_path(adapter, ds):
    return os.path.abspath(os.path.join(ROOT, DUMP.format(ad=adapter, ds=ds)))


def set_arm(adapter):
    """Point the three open-text verifier dumps at `adapter`.  No-op for the contaminated arm."""
    _REDIRECT.clear()
    if adapter == CONTAM_ADAPTER:
        return
    for ds in OPEN_DS:
        src, dst = dump_path(CONTAM_ADAPTER, ds), dump_path(adapter, ds)
        if not os.path.exists(dst):
            raise SystemExit(f"missing clean dump: {dst}")
        _REDIRECT[src] = dst


def assert_arms_are_paired():
    """Candidates / judge labels / greedy must be byte-identical across arms; only scores differ."""
    rep = {}
    for ds in OPEN_DS:
        base = json.load(_REAL_OPEN(dump_path(CONTAM_ADAPTER, ds)))
        row = dict(n=len(base))
        for name, ad, _ in ARMS[1:]:
            oth = json.load(_REAL_OPEN(dump_path(ad, ds)))
            assert len(oth) == len(base), (ds, name, "row count")
            assert [r["idx"] for r in oth] == [r["idx"] for r in base], (ds, name, "idx ORDER")
            assert all(a["preds"] == b["preds"] for a, b in zip(base, oth)), (ds, name, "preds")
            assert all(a["sl"] == b["sl"] for a, b in zip(base, oth)), (ds, name, "judge labels sl")
            assert all(a["greedy_ok"] == b["greedy_ok"] for a, b in zip(base, oth)), (ds, name, "greedy")
            row[name + "_scores_changed"] = int(sum(1 for a, b in zip(base, oth)
                                                    if a["scores"] != b["scores"]))
        rep[ds] = row
    return rep


# =================================================================================================
# 1.  the unchanged machinery
# =================================================================================================
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import macro_average_headline as MAH          # helpers reused verbatim: build/weightings/boot/ci/cost
import paper_baselines as PB
import integrated_method as IM

ORDER   = MAH.ORDER_B                          # 8 Variant-B cells
MCQ_B   = MAH.MCQ_B
OPEN_B  = MAH.OPEN_B
SYSTEMS = MAH.SYSTEMS
SYS_KEY = MAH.SYS_KEY
METHODS = MAH.METHODS
BASELINES = MAH.BASELINES
HEADLINE_BASELINES = MAH.HEADLINE_BASELINES
BENCH_OF = MAH.BENCH_OF
NBOOT, SEED = MAH.NBOOT, MAH.SEED
OPEN_KEY = PB.OPEN_KEY

PUBLISHED = os.path.join(ROOT, "results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json")
RETRAIN   = os.path.join(ROOT, "results/cascade_methods/artifacts/verifier_disjoint_retrain_2026-07-30.json")
OUT       = os.path.join(ROOT, "results/cascade_methods/artifacts/macro_headline_clean_verifier_2026-07-30.json")

# published open-arm efficiency claim being audited (METHOD_FINAL_2026-07.md 5.x)
PUB_OPEN_ESC   = 0.0397
PUB_OPEN_LAT   = 548.3      # ms, = BO8 522 + esc*665
PUB_OPEN_PCT   = -94.8      # vs 32B-think 10521.6 ms
BO8_MS, BO8_F  = IM.BO8["ms"], IM.BO8["flop"]
G32N_MS, G32N_F = IM.GEN32N["ms"], IM.GEN32N["flop"]
G32T_MS         = IM.GEN32T["ms"]
BO8_J = 8 * (PB.GEN7[1] + PB.VER7[1])          # 8 gens + 8 verifier forwards, measured batch-1 energy
G32N_J = PB.GEN32N[1]


# =================================================================================================
# 2.  one arm -> every number
# =================================================================================================
def compute_arm(adapter):
    set_arm(adapter)
    cells = MAH.build()                                   # unchanged pipeline
    W, benches = MAH.weightings(cells, ORDER)
    N_B = sum(cells[k]["n"] for k in ORDER)

    # ---- per-cell accuracy + one paired bootstrap replicate stream per cell (same order as MAH) ----
    rng = np.random.default_rng(SEED)
    acc_cell = {k: {s: float(np.asarray(cells[k][SYS_KEY[s]], float).mean()) for s in SYSTEMS}
                for k in ORDER}
    B = {}
    for k in ORDER:
        mat = np.column_stack([np.asarray(cells[k][SYS_KEY[s]], float) for s in SYSTEMS])
        B[k] = MAH.cell_boot_means(mat, NBOOT, rng)
    SI = {s: i for i, s in enumerate(SYSTEMS)}

    acc_levels = {lab: {s: round(MAH.wavg({k: acc_cell[k][s] for k in ORDER}, w), 4) for s in SYSTEMS}
                  for lab, w in W.items()}

    def sub_levels(keys):
        n = sum(cells[k]["n"] for k in keys)
        sw = {k: cells[k]["n"] / n for k in keys}
        mc = {k: 1.0 / len(keys) for k in keys}
        return dict(n=int(n), n_cells=len(keys),
                    sample_weighted={s: round(MAH.wavg({k: acc_cell[k][s] for k in keys}, sw), 4)
                                     for s in SYSTEMS},
                    macro_cells={s: round(MAH.wavg({k: acc_cell[k][s] for k in keys}, mc), 4)
                                 for s in SYSTEMS})
    subpools = dict(mcq_only=sub_levels(MCQ_B), open_only=sub_levels(OPEN_B))

    # ---- deltas (identical construction to MAH.run's delta_block) --------------------------------
    def delta_block(keys, m, b):
        n = sum(cells[k]["n"] for k in keys)
        schemes = {"sample_weighted": {k: cells[k]["n"] / n for k in keys},
                   "macro_cells": {k: 1.0 / len(keys) for k in keys}}
        if set(keys) == set(ORDER):
            schemes["macro_benchmarks_cellavg"] = W["macro_benchmarks_cellavg"]
            schemes["macro_benchmarks_itemweighted"] = W["macro_benchmarks_itemweighted"]
        else:
            bb = {}
            for k in keys:
                bb.setdefault(BENCH_OF[k], []).append(k)
            schemes["macro_benchmarks_cellavg"] = {k: (1.0 / len(bb)) / len(ks)
                                                   for _, ks in bb.items() for k in ks}
            schemes["macro_benchmarks_itemweighted"] = {
                k: (1.0 / len(bb)) * cells[k]["n"] / sum(cells[j]["n"] for j in ks)
                for _, ks in bb.items() for k in ks}
        out = {}
        dcell = {k: B[k][:, SI[m]] - B[k][:, SI[b]] for k in keys}
        pcell = {k: acc_cell[k][m] - acc_cell[k][b] for k in keys}
        for lab, w in schemes.items():
            dist = sum(dcell[k] * w[k] for k in keys)
            out[lab] = MAH.ci(dist, point=MAH.wavg(pcell, w))
            out[lab]["n_units"] = (len(set(BENCH_OF[k] for k in keys)) if lab.startswith("macro_benchmarks")
                                   else (len(keys) if lab == "macro_cells" else int(n)))
        big = np.column_stack([np.concatenate([np.asarray(cells[k][SYS_KEY[s]], float) for k in keys])
                               for s in (m, b)])
        rr = np.random.default_rng(SEED + 7)
        bb2 = MAH.cell_boot_means(big, NBOOT, rr)
        out["sample_weighted_pooled_unstratified"] = MAH.ci(bb2[:, 0] - bb2[:, 1],
                                                            point=MAH.wavg(pcell, schemes["sample_weighted"]))
        out["sample_weighted_pooled_unstratified"]["n_units"] = int(n)
        out["per_cell"] = {k: MAH.ci(dcell[k], point=pcell[k]) for k in keys}
        loo = {k: round(float(np.mean([pcell[j] for j in keys if j != k])), 4) for k in keys}
        worst = min(loo, key=lambda k: loo[k]); best = max(loo, key=lambda k: loo[k])
        out["macro_cells_leave_one_out"] = dict(per_dropped_cell=loo, range=[loo[worst], loo[best]],
                                                cell_carrying_the_claim=worst,
                                                cell_holding_the_claim_back=best)
        return out

    deltas = {m: {b: dict(all8=delta_block(ORDER, m, b), mcq_only=delta_block(MCQ_B, m, b),
                          open_only=delta_block(OPEN_B, m, b))
                  for b in BASELINES} for m in METHODS}

    # ---- cost (as-charged + honest re-cost), same helpers -----------------------------------------
    cost_ac, cost_hon, cost_prov = MAH.cost_blocks(cells, ORDER)
    cost = dict(as_charged=MAH.cost_table(cost_ac, ORDER, W),
                honest_recost=MAH.cost_table(cost_hon, ORDER, W),
                per_cell_as_charged={k: {s: {nm: round(cost_ac[k][s][ax], 3) for ax, nm in MAH.AXES}
                                         for s in SYSTEMS} for k in ORDER},
                per_cell_honest_recost={k: {s: {nm: round(cost_hon[k][s][ax], 3) for ax, nm in MAH.AXES}
                                            for s in SYSTEMS} for k in ORDER})
    # open-only and mcq-only cost aggregates (the open arm's efficiency claim is open-only)
    for tag, keys in (("open_only", OPEN_B), ("mcq_only", MCQ_B)):
        n = sum(cells[k]["n"] for k in keys)
        sw = {k: cells[k]["n"] / n for k in keys}; mc = {k: 1.0 / len(keys) for k in keys}
        cost[tag] = {conv: {lab: {s: {nm: round(MAH.wavg({k: src[k][s][ax] for k in keys}, w), 3
                                                if ax == "flops" else 1)
                                      for ax, nm in MAH.AXES} for s in SYSTEMS}
                            for lab, w in (("sample_weighted", sw), ("macro_cells", mc))}
                     for conv, src in (("as_charged", cost_ac), ("honest_recost", cost_hon))}

    ratios = {}
    for conv in ("as_charged", "honest_recost"):
        ratios[conv] = {}
        for lab in W:
            ratios[conv][lab] = {m: {} for m in METHODS}
            for m in METHODS:
                for b in HEADLINE_BASELINES:
                    mv, bv = cost[conv][lab][m], cost[conv][lab][b]
                    ratios[conv][lab][m][b] = {
                        "flops_x": round(mv["flops"] / bv["flops"], 3),
                        "lat_par_x": round(mv["lat_par_ms"] / bv["lat_par_ms"], 3),
                        "lat_par_pct": round(-100 * (1 - mv["lat_par_ms"] / bv["lat_par_ms"]), 1),
                        "lat_seq_x": round(mv["lat_seq_ms"] / bv["lat_seq_ms"], 3),
                        "lat_seq_pct": round(-100 * (1 - mv["lat_seq_ms"] / bv["lat_seq_ms"]), 1),
                        "energy_x": round(mv["energy_j"] / bv["energy_j"], 3),
                        "energy_pct": round(-100 * (1 - mv["energy_j"] / bv["energy_j"]), 1)}
    cost["method_vs_baseline_ratios"] = ratios
    # open-only ratios (the open arm claim, isolated)
    open_ratios = {}
    for conv in ("as_charged", "honest_recost"):
        open_ratios[conv] = {}
        for lab in ("sample_weighted", "macro_cells"):
            open_ratios[conv][lab] = {}
            for m in METHODS:
                open_ratios[conv][lab][m] = {}
                for b in HEADLINE_BASELINES:
                    mv, bv = cost["open_only"][conv][lab][m], cost["open_only"][conv][lab][b]
                    open_ratios[conv][lab][m][b] = {
                        "flops_x": round(mv["flops"] / bv["flops"], 3),
                        "lat_par_pct": round(-100 * (1 - mv["lat_par_ms"] / bv["lat_par_ms"]), 1),
                        "lat_seq_pct": round(-100 * (1 - mv["lat_seq_ms"] / bv["lat_seq_ms"]), 1),
                        "energy_pct": round(-100 * (1 - mv["energy_j"] / bv["energy_j"]), 1)}
    cost["open_only_ratios"] = open_ratios

    # ---- Pareto (same rule as MAH) ---------------------------------------------------------------
    pareto = {}
    for conv in ("as_charged", "honest_recost"):
        pareto[conv] = {}
        for lab in ("sample_weighted", "macro_cells"):
            pts = [dict(system=s, acc=acc_levels[lab][s], **cost[conv][lab][s]) for s in SYSTEMS]

            def mark(axis, pts=pts):
                return sorted(p["system"] for p in pts if not any(
                    q[axis] <= p[axis] and q["acc"] >= p["acc"] and q is not p and
                    (q[axis] < p[axis] or q["acc"] > p["acc"]) for q in pts))
            pareto[conv][lab] = dict(points=pts, pareto_optimal_flops=mark("flops"),
                                     pareto_optimal_lat_par=mark("lat_par_ms"),
                                     pareto_optimal_lat_seq=mark("lat_seq_ms"),
                                     pareto_optimal_energy=mark("energy_j"))
    cost["pareto"] = pareto

    # ---- escalation ------------------------------------------------------------------------------
    esc_cell = {k: (round(float(cells[k]["esc"]), 4) if cells[k].get("esc") is not None else None)
                for k in ORDER}
    esc = dict(per_cell=esc_cell,
               compute_lean_mcq=dict(
                   sample_weighted=round(sum(esc_cell[k] * cells[k]["n"] for k in MCQ_B) /
                                         sum(cells[k]["n"] for k in MCQ_B), 4),
                   macro_cells=round(float(np.mean([esc_cell[k] for k in MCQ_B])), 4)),
               compute_lean_open=dict(
                   sample_weighted=round(sum(esc_cell[k] * cells[k]["n"] for k in OPEN_B) /
                                         sum(cells[k]["n"] for k in OPEN_B), 4),
                   macro_cells=round(float(np.mean([esc_cell[k] for k in OPEN_B])), 4)),
               compute_lean_all8=dict(
                   sample_weighted=round(sum(esc_cell[k] * cells[k]["n"] for k in ORDER) / N_B, 4),
                   macro_cells=round(float(np.mean([esc_cell[k] for k in ORDER])), 4)),
               accuracy_max_veto_open=dict(
                   per_cell={k: cells[k].get("am2_esc") for k in OPEN_B},
                   sample_weighted=round(sum(cells[k]["am2_esc"] * cells[k]["n"] for k in OPEN_B) /
                                         sum(cells[k]["n"] for k in OPEN_B), 4),
                   macro_cells=round(float(np.mean([cells[k]["am2_esc"] for k in OPEN_B])), 4)),
               open_meanN={k: round(float(cells[k]["meanN"]), 3) for k in OPEN_B})

    # ---- the published-style FIXED best-of-8 + verifier-confidence-gate open arm -------------------
    # This is the arm the "3.97% escalation / 548.3 ms / -94.8%" claim was made about
    # (METHOD_FINAL_2026-07.md 5.x); the macro pipeline above instead runs Pandora adaptive-N.
    bo8 = {}
    for cell in OPEN_B:
        dskey = OPEN_KEY[cell]
        d = IM.open_bestof8(dskey)
        a_arm, e_arm = IM.heldout(d["ok7"], d["ok32"], d["gate"])
        bo8[cell] = dict(n=int(len(d["ok7"])),
                         bo8_verifier_pick_acc=round(float(d["ok7"].mean()), 4),
                         greedy=round(float(d["greedy"]), 4),
                         strong_32b_nothink=round(float(d["ok32"].mean()), 4),
                         arm_accuracy=round(a_arm, 4), escalation=round(e_arm, 4),
                         lat_par_ms=round(BO8_MS + e_arm * G32N_MS, 1),
                         flops=round(BO8_F + e_arm * G32N_F, 3),
                         energy_j=round(BO8_J + e_arm * G32N_J, 1))
    n_open = sum(bo8[k]["n"] for k in OPEN_B)
    for lab, w in (("sample_weighted", {k: bo8[k]["n"] / n_open for k in OPEN_B}),
                   ("macro_cells", {k: 1.0 / len(OPEN_B) for k in OPEN_B})):
        e = sum(bo8[k]["escalation"] * w[k] for k in OPEN_B)
        bo8[lab] = dict(escalation=round(e, 4),
                        arm_accuracy=round(sum(bo8[k]["arm_accuracy"] * w[k] for k in OPEN_B), 4),
                        strong_32b_nothink=round(sum(bo8[k]["strong_32b_nothink"] * w[k] for k in OPEN_B), 4),
                        lat_par_ms=round(BO8_MS + e * G32N_MS, 1),
                        flops=round(BO8_F + e * G32N_F, 3),
                        energy_j=round(BO8_J + e * G32N_J, 1),
                        pct_vs_32b_think_lat=round(-100 * (1 - (BO8_MS + e * G32N_MS) / G32T_MS), 1),
                        pct_vs_32b_direct_lat=round(-100 * (1 - (BO8_MS + e * G32N_MS) / G32N_MS), 1),
                        x_vs_32b_direct_flops=round((BO8_F + e * G32N_F) / G32N_F, 3))

    # ---- per-item vectors retained for the paired clean-vs-contaminated bootstrap -----------------
    vecs = {k: {s: np.asarray(cells[k][SYS_KEY[s]], float).copy() for s in SYSTEMS} for k in OPEN_B}

    return dict(acc_cell=acc_cell, acc_levels=acc_levels, subpools=subpools, deltas=deltas,
                cost=cost, cost_provenance=cost_prov, escalation=esc, bo8_open_arm=bo8,
                cell_n={k: cells[k]["n"] for k in ORDER},
                policies={k: dict(compute_lean=cells[k]["cl_choice"], veto=cells[k]["am2_choice"],
                                  fusion=cells[k]["am_choice"], oracle_mode=cells[k]["oracle_mode"])
                          for k in ORDER}), vecs


# =================================================================================================
# 3.  reproduction check of the contaminated arm against the published artifact
# =================================================================================================
def validate_against_published(res):
    pub = json.load(_REAL_OPEN(PUBLISHED))
    diffs = []
    n_checked = [0]

    def cmp(path, got, exp):
        n_checked[0] += 1
        if got != exp:
            diffs.append(dict(path=path, recomputed=got, published=exp))

    for lab, d in pub["accuracy_levels"]["full_pool"].items():
        for s, v in d.items():
            cmp(f"accuracy_levels.full_pool.{lab}.{s}", res["acc_levels"][lab][s], v)
    for sp in ("mcq_only", "open_only"):
        for lab in ("sample_weighted", "macro_cells"):
            for s, v in pub["accuracy_levels"]["subpools"][sp][lab].items():
                cmp(f"subpools.{sp}.{lab}.{s}", res["subpools"][sp][lab][s], v)
    for m in METHODS:
        for b in BASELINES:
            for ax in ("all8", "mcq_only", "open_only"):
                for lab in ("sample_weighted", "macro_cells", "macro_benchmarks_cellavg"):
                    g, e = res["deltas"][m][b][ax][lab], pub["deltas"][m][b][ax][lab]
                    for f in ("delta", "lo", "hi", "verdict"):
                        cmp(f"deltas.{m}.{b}.{ax}.{lab}.{f}", g[f], e[f])
    for k, v in pub["escalation"]["per_cell"].items():
        cmp(f"escalation.per_cell.{k}", res["escalation"]["per_cell"][k], v)
    for conv in ("as_charged", "honest_recost"):
        for lab in pub["cost"][conv]:
            for s in SYSTEMS:
                for nm in ("flops", "lat_par_ms", "lat_seq_ms", "energy_j"):
                    cmp(f"cost.{conv}.{lab}.{s}.{nm}", res["cost"][conv][lab][s][nm],
                        pub["cost"][conv][lab][s][nm])
        for lab in pub["cost"]["method_vs_baseline_ratios"][conv]:
            for m in METHODS:
                for b in HEADLINE_BASELINES:
                    for nm, v in pub["cost"]["method_vs_baseline_ratios"][conv][lab][m][b].items():
                        cmp(f"ratios.{conv}.{lab}.{m}.{b}.{nm}",
                            res["cost"]["method_vs_baseline_ratios"][conv][lab][m][b][nm], v)
    return diffs, n_checked[0]


def validate_bo8_against_retrain(arms):
    """The bo8+verifier-gate open arm must reproduce verifier_disjoint_retrain_2026-07-30.json."""
    rt = json.load(_REAL_OPEN(RETRAIN))
    out, diffs = {}, []
    for lvl, armname in (("L1_image_disjoint", "clean_L1"), ("L2_strict", "clean_L2")):
        e2e = rt["levels"][lvl]["end_to_end_open_arm"]
        for cell, dskey in zip(OPEN_B, ["slake_open", "vqa_rad_open", "pathvqa_open"]):
            for side, arm in (("clean", armname), ("contaminated", "contaminated")):
                exp = e2e[dskey][side]
                got = arms[arm]["bo8_open_arm"][cell]
                for f_exp, f_got in (("arm_accuracy", "arm_accuracy"),
                                     ("escalation_rate", "escalation"),
                                     ("cheap_leg_bo8_verifier", "bo8_verifier_pick_acc"),
                                     ("strong_32b_nothink", "strong_32b_nothink")):
                    if abs(round(exp[f_exp], 4) - got[f_got]) > 1e-4:
                        diffs.append(dict(level=lvl, cell=cell, side=side, field=f_exp,
                                          recomputed=got[f_got], retrain_artifact=round(exp[f_exp], 4)))
        out[lvl] = "matches verifier_disjoint_retrain_2026-07-30.json"
    return out, diffs


# =================================================================================================
# 4.  paired clean-vs-contaminated bootstrap on the open cells (items are aligned 1:1)
# =================================================================================================
def paired_open_delta(vec_a, vec_b, systems=SYSTEMS):
    """vec_*: cell -> system -> per-item 0/1.  Items are the SAME items in the SAME order across arms
    (asserted upstream), so a single item resample can be shared by both arms -- a genuinely paired
    contrast of the two verifiers."""
    rng = np.random.default_rng(SEED + 991)
    out = {}
    for lab in ("sample_weighted", "macro_cells"):
        n = {k: len(vec_a[k][systems[0]]) for k in OPEN_B}
        tot = sum(n.values())
        w = {k: (n[k] / tot if lab == "sample_weighted" else 1.0 / len(OPEN_B)) for k in OPEN_B}
        out[lab] = {}
        for s in systems:
            dist = 0.0; point = 0.0
            for k in OPEN_B:
                mat = np.column_stack([vec_a[k][s], vec_b[k][s]])
                bb = MAH.cell_boot_means(mat, NBOOT, rng)
                dist = dist + w[k] * (bb[:, 0] - bb[:, 1])
                point += w[k] * (float(vec_a[k][s].mean()) - float(vec_b[k][s].mean()))
            out[lab][s] = MAH.ci(dist, point=point)
    return out


# =================================================================================================
# 5.  RUN
# =================================================================================================
def run():
    t0 = time.time()
    pairing = assert_arms_are_paired()

    arms, vecs = {}, {}
    for name, adapter, _desc in ARMS:
        t = time.time()
        arms[name], vecs[name] = compute_arm(adapter)
        print(f"[{name}] built in {time.time()-t:.1f}s", flush=True)
    set_arm(CONTAM_ADAPTER)

    repro_diffs, n_checked = validate_against_published(arms["contaminated"])
    if repro_diffs:
        print("\n!!! CONTAMINATED ARM DOES NOT REPRODUCE THE PUBLISHED MACRO ARTIFACT !!!")
        for d in repro_diffs[:40]:
            print("   ", d)
        raise SystemExit("wiring error -- aborting rather than reporting a corrupted clean column")
    bo8_ok, bo8_diffs = validate_bo8_against_retrain(arms)
    if bo8_diffs:
        print("\n!!! bo8 open arm does not reproduce verifier_disjoint_retrain !!!")
        for d in bo8_diffs[:40]:
            print("   ", d)
        raise SystemExit("wiring error")

    # ---------------- 3-column side-by-side for every claim ----------------------------------------
    COLS = [("A_published_contaminated_sample_weighted", "contaminated", "sample_weighted"),
            ("B_macro_only_contaminated",                "contaminated", "macro_cells"),
            ("C_macro_plus_clean_verifier_L1",           "clean_L1",     "macro_cells"),
            ("D_macro_plus_clean_verifier_L2_lower_bound", "clean_L2",   "macro_cells")]

    def cell_of(arm, lab, m, b, ax):
        d = arms[arm]["deltas"][m][b][ax][lab]
        return dict(delta=d["delta"], ci95=[d["lo"], d["hi"]], verdict=d["verdict"])

    side_by_side = []
    for m in METHODS:
        for b in HEADLINE_BASELINES:
            for ax, pool in (("all8", "all 8 cells"), ("mcq_only", "5 multiple-choice cells"),
                             ("open_only", "3 open-text cells")):
                row = dict(claim=f"{m} vs {b}", pool=pool, method=m, baseline=b, axis=ax)
                for col, arm, lab in COLS:
                    row[col] = cell_of(arm, lab, m, b, ax)
                a, c = row[COLS[0][0]], row[COLS[2][0]]
                bcol = row[COLS[1][0]]
                row["effect_of_macro_alone"] = round(bcol["delta"] - a["delta"], 4)
                row["effect_of_clean_verifier_alone_under_macro"] = round(c["delta"] - bcol["delta"], 4)
                row["net_A_to_C"] = round(c["delta"] - a["delta"], 4)
                row["sign_flip_A_to_C"] = bool(np.sign(a["delta"]) != np.sign(c["delta"]) and
                                               a["delta"] != 0 and c["delta"] != 0)
                row["significance_change_A_to_C"] = (f"{a['verdict']} -> {c['verdict']}"
                                                     if a["verdict"] != c["verdict"] else
                                                     f"unchanged ({a['verdict']})")
                row["loses_significance_from_B_to_C"] = bool(bcol["verdict"] != "TIE" and c["verdict"] == "TIE")
                side_by_side.append(row)

    acc_side = {s: {col: arms[arm]["acc_levels"][lab][s] for col, arm, lab in COLS} for s in SYSTEMS}

    def cost_side(conv, field):
        return {m: {b: {col: arms[arm]["cost"]["method_vs_baseline_ratios"][conv][lab][m][b][field]
                        for col, arm, lab in COLS} for b in HEADLINE_BASELINES} for m in METHODS}

    cost_sbs = {conv: {f: cost_side(conv, f) for f in ("flops_x", "lat_par_pct", "lat_seq_pct",
                                                       "energy_pct")}
                for conv in ("as_charged", "honest_recost")}

    esc_side = {"compute_lean_open": {col: arms[arm]["escalation"]["compute_lean_open"][
                                          "sample_weighted" if lab == "sample_weighted" else "macro_cells"]
                                      for col, arm, lab in COLS},
                "compute_lean_all8": {col: arms[arm]["escalation"]["compute_lean_all8"][
                                          "sample_weighted" if lab == "sample_weighted" else "macro_cells"]
                                      for col, arm, lab in COLS},
                "bo8_parity_open_arm": {col: arms[arm]["bo8_open_arm"][
                                            "sample_weighted" if lab == "sample_weighted" else "macro_cells"]["escalation"]
                                        for col, arm, lab in COLS}}

    # ---------------- open-arm honest latency (the published -94.8% claim) --------------------------
    open_lat = {}
    for name in ("contaminated", "clean_L1", "clean_L2"):
        b8 = arms[name]["bo8_open_arm"]
        open_lat[name] = {lab: dict(escalation=b8[lab]["escalation"], lat_par_ms=b8[lab]["lat_par_ms"],
                                    pct_vs_32b_think=b8[lab]["pct_vs_32b_think_lat"],
                                    pct_vs_32b_direct=b8[lab]["pct_vs_32b_direct_lat"],
                                    flops=b8[lab]["flops"],
                                    flops_x_vs_32b_direct=b8[lab]["x_vs_32b_direct_flops"],
                                    energy_j=b8[lab]["energy_j"])
                          for lab in ("sample_weighted", "macro_cells")}
        open_lat[name]["per_cell"] = {k: {f: b8[k][f] for f in
                                          ("escalation", "lat_par_ms", "flops", "energy_j", "arm_accuracy")}
                                      for k in OPEN_B}
    # the deployed pipeline's own open arm (Pandora adaptive-N), which is what the macro cost uses
    pandora_lat = {name: {lab: {m: {f: arms[name]["cost"]["open_only"]["as_charged"][lab][m][f]
                                    for f in ("flops", "lat_par_ms", "lat_seq_ms", "energy_j")}
                                for m in METHODS + ["always_32b_direct", "always_32b_reasoning"]}
                          for lab in ("sample_weighted", "macro_cells")}
                   for name in ("contaminated", "clean_L1", "clean_L2")}

    # ---------------- paired clean-minus-contaminated on the open cells -----------------------------
    paired = {lvl: paired_open_delta(vecs[lvl], vecs["contaminated"])
              for lvl in ("clean_L1", "clean_L2")}

    # ---------------- per-cell table ----------------------------------------------------------------
    per_cell = {}
    for k in ORDER:
        rec = dict(n=arms["contaminated"]["cell_n"][k], format=("open" if k in OPEN_B else "MCQ"),
                   macro_weight_pct=12.5,
                   sample_weight_pct=round(100 * arms["contaminated"]["cell_n"][k] /
                                           sum(arms["contaminated"]["cell_n"].values()), 2),
                   policies=arms["contaminated"]["policies"][k],
                   accuracy={name: {s: round(arms[name]["acc_cell"][k][s], 4) for s in SYSTEMS}
                             for name in arms},
                   escalation={name: arms[name]["escalation"]["per_cell"][k] for name in arms},
                   changed_by_verifier=bool(k in OPEN_B))
        per_cell[k] = rec

    # ---------------- verdicts --------------------------------------------------------------------
    def V(m, b, ax, arm, lab="macro_cells"):
        return arms[arm]["deltas"][m][b][ax][lab]

    am_direct_C = V("method_accuracy_max_veto", "always_32b_direct", "all8", "clean_L1")
    am_direct_B = V("method_accuracy_max_veto", "always_32b_direct", "all8", "contaminated")
    am_direct_A = arms["contaminated"]["deltas"]["method_accuracy_max_veto"]["always_32b_direct"]["all8"]["sample_weighted"]
    fu_direct_C = V("method_accuracy_max_fusion", "always_32b_direct", "all8", "clean_L1")
    cl_direct_C = V("method_compute_lean", "always_32b_direct", "all8", "clean_L1")
    cl_mcq_C    = V("method_compute_lean", "always_32b_direct", "mcq_only", "clean_L1")
    am_oracle_C = V("method_accuracy_max_veto", "oracle_mode_32b", "all8", "clean_L1")
    am_reason_C = V("method_accuracy_max_veto", "always_32b_reasoning", "all8", "clean_L1")
    open_am_C   = V("method_accuracy_max_veto", "always_32b_direct", "open_only", "clean_L1")
    open_cl_C   = V("method_compute_lean", "always_32b_direct", "open_only", "clean_L1")
    am_direct_D = V("method_accuracy_max_veto", "always_32b_direct", "all8", "clean_L2")

    r_hon_C = arms["clean_L1"]["cost"]["method_vs_baseline_ratios"]["honest_recost"]["macro_cells"]
    r_ac_C  = arms["clean_L1"]["cost"]["method_vs_baseline_ratios"]["as_charged"]["macro_cells"]

    verdicts = dict(
        does_accuracy_max_still_beat_32b_direct_under_macro_with_a_clean_verifier=dict(
            answer=("YES -- it survives" if am_direct_C["verdict"] == "WIN" else
                    "NO -- it no longer beats it" if am_direct_C["verdict"] == "TIE" else
                    "NO -- it is now significantly WORSE"),
            published_sample_weighted_contaminated=dict(delta=am_direct_A["delta"],
                                                        ci95=[am_direct_A["lo"], am_direct_A["hi"]],
                                                        verdict=am_direct_A["verdict"]),
            macro_only_contaminated=dict(delta=am_direct_B["delta"],
                                         ci95=[am_direct_B["lo"], am_direct_B["hi"]],
                                         verdict=am_direct_B["verdict"]),
            macro_plus_clean_L1=dict(delta=am_direct_C["delta"],
                                     ci95=[am_direct_C["lo"], am_direct_C["hi"]],
                                     verdict=am_direct_C["verdict"]),
            macro_plus_clean_L2_lower_bound=dict(delta=am_direct_D["delta"],
                                                 ci95=[am_direct_D["lo"], am_direct_D["hi"]],
                                                 verdict=am_direct_D["verdict"]),
            leave_one_cell_out_range_L1=arms["clean_L1"]["deltas"]["method_accuracy_max_veto"][
                "always_32b_direct"]["all8"]["macro_cells_leave_one_out"]["range"],
            load_bearing_cell_L1=arms["clean_L1"]["deltas"]["method_accuracy_max_veto"][
                "always_32b_direct"]["all8"]["macro_cells_leave_one_out"]["cell_carrying_the_claim"]),
        compute_lean_multiple_choice_loss=dict(
            note="unchanged by the verifier -- the MCQ cells never touch it; listed so the reader sees "
                 "the macro-only correction is what created this loss, not the de-contamination.",
            macro_clean_L1=dict(delta=cl_mcq_C["delta"], ci95=[cl_mcq_C["lo"], cl_mcq_C["hi"]],
                                verdict=cl_mcq_C["verdict"]),
            macro_contaminated=dict(**{f: V("method_compute_lean", "always_32b_direct", "mcq_only",
                                            "contaminated")[f] for f in ("delta", "lo", "hi", "verdict")}),
            sample_weighted_contaminated=dict(**{f: arms["contaminated"]["deltas"]["method_compute_lean"][
                "always_32b_direct"]["mcq_only"]["sample_weighted"][f]
                for f in ("delta", "lo", "hi", "verdict")})),
        sign_flips=[r["claim"] + " | " + r["pool"] for r in side_by_side if r["sign_flip_A_to_C"]],
        significance_lost_from_macro_only_to_macro_plus_clean=[
            r["claim"] + " | " + r["pool"] for r in side_by_side if r["loses_significance_from_B_to_C"]],
        significance_changes_A_to_C=[dict(claim=r["claim"], pool=r["pool"],
                                          change=r["significance_change_A_to_C"])
                                     for r in side_by_side if r["significance_change_A_to_C"].startswith(
                                         ("WIN", "LOSS", "TIE")) and "unchanged" not in
                                     r["significance_change_A_to_C"]])

    # ---------------- Pareto DOMINATION, explicitly (frontier membership is not the same thing) -----
    def domination(arm, lab="macro_cells", conv="honest_recost"):
        pts = {p["system"]: p for p in arms[arm]["cost"]["pareto"][conv][lab]["points"]}
        AX = ("flops", "lat_par_ms", "lat_seq_ms", "energy_j")
        out = {}
        for m in METHODS:
            out[m] = {}
            for b in HEADLINE_BASELINES:
                pm, pb = pts[m], pts[b]
                dominated_by_baseline = (pb["acc"] >= pm["acc"] and all(pb[a] <= pm[a] for a in AX)
                                         and (pb["acc"] > pm["acc"] or any(pb[a] < pm[a] for a in AX)))
                dominates_baseline = (pm["acc"] >= pb["acc"] and all(pm[a] <= pb[a] for a in AX)
                                      and (pm["acc"] > pb["acc"] or any(pm[a] < pb[a] for a in AX)))
                out[m][b] = dict(
                    method_acc=pm["acc"], baseline_acc=pb["acc"],
                    acc_gap=round(pm["acc"] - pb["acc"], 4),
                    cheaper_on=[a for a in AX if pm[a] < pb[a]],
                    more_expensive_on=[a for a in AX if pm[a] > pb[a]],
                    method_dominates_baseline=bool(dominates_baseline),
                    method_is_DOMINATED_by_baseline=bool(dominated_by_baseline))
            out[m]["on_frontier"] = {ax: (m in arms[arm]["cost"]["pareto"][conv][lab][key])
                                     for ax, key in (("flops", "pareto_optimal_flops"),
                                                     ("lat_par", "pareto_optimal_lat_par"),
                                                     ("lat_seq", "pareto_optimal_lat_seq"),
                                                     ("energy", "pareto_optimal_energy"))}
        return out
    pareto_dom = {arm: domination(arm) for arm in ("contaminated", "clean_L1", "clean_L2")}

    joint = dict(
        basis="accuracy verdict from the macro 8-cell paired bootstrap; cost from the honestly "
              "re-costed FLOP-eq / batch-1 latency / energy, macro-averaged over the same 8 cells.",
        pareto_domination_macro_honest=pareto_dom,
        accuracy_max_veto_vs_always_32b_direct=dict(
            acc=dict(delta=am_direct_C["delta"], ci95=[am_direct_C["lo"], am_direct_C["hi"]],
                     verdict=am_direct_C["verdict"]),
            flops_x_as_charged=r_ac_C["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
            flops_x_honest=r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"],
            lat_par_pct=r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["lat_par_pct"],
            lat_seq_pct=r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["lat_seq_pct"],
            energy_pct=r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["energy_pct"]),
        accuracy_max_veto_vs_oracle_mode_32b=dict(
            acc=dict(delta=am_oracle_C["delta"], ci95=[am_oracle_C["lo"], am_oracle_C["hi"]],
                     verdict=am_oracle_C["verdict"]),
            flops_x_honest=r_hon_C["method_accuracy_max_veto"]["oracle_mode_32b"]["flops_x"],
            lat_par_pct=r_hon_C["method_accuracy_max_veto"]["oracle_mode_32b"]["lat_par_pct"],
            energy_pct=r_hon_C["method_accuracy_max_veto"]["oracle_mode_32b"]["energy_pct"]),
        accuracy_max_veto_vs_always_32b_reasoning=dict(
            acc=dict(delta=am_reason_C["delta"], ci95=[am_reason_C["lo"], am_reason_C["hi"]],
                     verdict=am_reason_C["verdict"]),
            flops_x_as_charged=r_ac_C["method_accuracy_max_veto"]["always_32b_reasoning"]["flops_x"],
            flops_x_honest=r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["flops_x"],
            lat_par_pct=r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["lat_par_pct"],
            energy_pct=r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["energy_pct"]),
        compute_lean_vs_always_32b_direct=dict(
            acc=dict(delta=cl_direct_C["delta"], ci95=[cl_direct_C["lo"], cl_direct_C["hi"]],
                     verdict=cl_direct_C["verdict"]),
            acc_mcq_only=dict(delta=cl_mcq_C["delta"], ci95=[cl_mcq_C["lo"], cl_mcq_C["hi"]],
                              verdict=cl_mcq_C["verdict"]),
            flops_x_honest=r_hon_C["method_compute_lean"]["always_32b_direct"]["flops_x"],
            lat_seq_pct=r_hon_C["method_compute_lean"]["always_32b_direct"]["lat_seq_pct"],
            energy_pct=r_hon_C["method_compute_lean"]["always_32b_direct"]["energy_pct"]),
        open_arm_only=dict(
            accuracy_max_veto_vs_32b_direct=dict(delta=open_am_C["delta"],
                                                 ci95=[open_am_C["lo"], open_am_C["hi"]],
                                                 verdict=open_am_C["verdict"]),
            compute_lean_vs_32b_direct=dict(delta=open_cl_C["delta"],
                                            ci95=[open_cl_C["lo"], open_cl_C["hi"]],
                                            verdict=open_cl_C["verdict"]),
            pandora_cost=arms["clean_L1"]["cost"]["open_only_ratios"]["as_charged"]["macro_cells"]))

    # ---------------- the corrected headline sentence, assembled from the numbers above -------------
    b8c, b8k = arms["contaminated"]["bo8_open_arm"], arms["clean_L1"]["bo8_open_arm"]
    b8l = arms["clean_L2"]["bo8_open_arm"]
    am_flops_C = r_ac_C["method_accuracy_max_veto"]["always_32b_direct"]["flops_x"]
    am_lat_C   = r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["lat_par_pct"]
    am_en_C    = r_hon_C["method_accuracy_max_veto"]["always_32b_direct"]["energy_pct"]
    cl_flops_C = r_ac_C["method_compute_lean"]["always_32b_direct"]["flops_x"]
    reason_ac  = r_ac_C["method_accuracy_max_veto"]["always_32b_reasoning"]["flops_x"]
    reason_hn  = r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["flops_x"]
    reason_lat = r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["lat_par_pct"]
    reason_en  = r_hon_C["method_accuracy_max_veto"]["always_32b_reasoning"]["energy_pct"]

    honest_headline = dict(
        primary_granularity="8 reporting cells, 1/8 each; open-text verifier trained on strictly "
                            "disjoint data (L1: no eval image, no eval item)",
        corrected_joint_claim_sentence=(
            "Macro-averaged over the 8 benchmark cells and with an uncontaminated open-text verifier, "
            "the method NO LONGER beats a single 32B forward pass: accuracy-max is +%.4f [%+.4f, %+.4f] "
            "vs always-32B-direct (a statistical TIE, down from the published +%.4f [%+.4f, %+.4f]) while "
            "costing %.2fx its FLOP-eq, %+.1f%% batch-1 latency and %+.1f%% energy; compute-lean is now a "
            "SIGNIFICANT LOSS at %+.4f [%+.4f, %+.4f] and %.2fx the compute. What survives is the "
            "comparison against a 32B actually made to reason: %+.4f [%+.4f, %+.4f] accuracy at %+.1f%% "
            "latency and %+.1f%% energy (though %.2fx as-charged / %.2fx honestly re-costed FLOP-eq, i.e. "
            "not fewer FLOPs), plus the standalone finding that reasoning mode is actively harmful on "
            "free-text medical VQA."
            % (am_direct_C["delta"], am_direct_C["lo"], am_direct_C["hi"],
               am_direct_A["delta"], am_direct_A["lo"], am_direct_A["hi"],
               am_flops_C, am_lat_C, am_en_C,
               cl_direct_C["delta"], cl_direct_C["lo"], cl_direct_C["hi"], cl_flops_C,
               am_reason_C["delta"], am_reason_C["lo"], am_reason_C["hi"], reason_lat, reason_en,
               reason_ac, reason_hn)),
        one_line_if_only_one_number_is_allowed=(
            "8-cell macro, clean verifier: accuracy-max %+.4f [%+.4f, %+.4f] vs always-32B-direct at "
            "%.2fx its compute -- a tie bought with more compute, not a win."
            % (am_direct_C["delta"], am_direct_C["lo"], am_direct_C["hi"], am_flops_C)),
        what_each_correction_costs=dict(
            macro_reweighting_alone=round(am_direct_B["delta"] - am_direct_A["delta"], 4),
            clean_verifier_alone_under_macro=round(am_direct_C["delta"] - am_direct_B["delta"], 4),
            net=round(am_direct_C["delta"] - am_direct_A["delta"], 4),
            note="on the headline comparison (accuracy-max vs always-32B-direct, all 8 cells). "
                 "Macro re-weighting alone slightly HELPED this comparison (+0.0021); the clean "
                 "verifier is what removed it (-0.0120). The two corrections are separable and the "
                 "table in side_by_side_claims separates them for every claim."),
        open_arm_honest_latency=dict(
            claim_audited="open arm: 3.97% escalation, 548.3 ms batch-1, -94.8% vs always-32B-reasoning",
            reproduced_contaminated=dict(escalation=b8c["sample_weighted"]["escalation"],
                                         lat_par_ms=b8c["sample_weighted"]["lat_par_ms"],
                                         pct_vs_32b_reasoning=b8c["sample_weighted"]["pct_vs_32b_think_lat"],
                                         pct_vs_32b_direct=b8c["sample_weighted"]["pct_vs_32b_direct_lat"]),
            honest_clean_L1_sample_weighted=dict(escalation=b8k["sample_weighted"]["escalation"],
                                                 lat_par_ms=b8k["sample_weighted"]["lat_par_ms"],
                                                 pct_vs_32b_reasoning=b8k["sample_weighted"]["pct_vs_32b_think_lat"],
                                                 pct_vs_32b_direct=b8k["sample_weighted"]["pct_vs_32b_direct_lat"]),
            honest_clean_L1_macro=dict(escalation=b8k["macro_cells"]["escalation"],
                                       lat_par_ms=b8k["macro_cells"]["lat_par_ms"],
                                       pct_vs_32b_reasoning=b8k["macro_cells"]["pct_vs_32b_think_lat"],
                                       pct_vs_32b_direct=b8k["macro_cells"]["pct_vs_32b_direct_lat"]),
            clean_L2_lower_bound=dict(escalation=b8l["sample_weighted"]["escalation"],
                                      lat_par_ms=b8l["sample_weighted"]["lat_par_ms"],
                                      pct_vs_32b_reasoning=b8l["sample_weighted"]["pct_vs_32b_think_lat"],
                                      pct_vs_32b_direct=b8l["sample_weighted"]["pct_vs_32b_direct_lat"]),
            sentence=(
                "The open arm's published '-94.8%% batch-1 latency' becomes %.1f%% sample-weighted / "
                "%.1f%% macro against always-32B-with-reasoning -- the headline percentage barely moves "
                "because the reasoning baseline is 10.5 s. The figure that actually breaks is the one "
                "against a single 32B forward: at 3.97%% escalation the arm was %.1f%% faster than "
                "always-32B-direct (548.4 ms vs 665 ms); a clean verifier needs %.1f%% escalation "
                "(%.1f%% macro) to hold the same parity target, so the arm becomes %+.1f%% "
                "sample-weighted / %+.1f%% macro -- i.e. SLOWER than just running the 32B once, at "
                "%.2fx its FLOP-eq."
                % (b8k["sample_weighted"]["pct_vs_32b_think_lat"], b8k["macro_cells"]["pct_vs_32b_think_lat"],
                   -b8c["sample_weighted"]["pct_vs_32b_direct_lat"],
                   100 * b8k["sample_weighted"]["escalation"], 100 * b8k["macro_cells"]["escalation"],
                   b8k["sample_weighted"]["pct_vs_32b_direct_lat"], b8k["macro_cells"]["pct_vs_32b_direct_lat"],
                   b8k["sample_weighted"]["x_vs_32b_direct_flops"]))),
        what_survives=[
            "vs always-32B-WITH-REASONING the accuracy margin still GROWS under macro even after "
            "de-contamination (accuracy-max %+.4f [%+.4f, %+.4f] macro+clean vs %+.4f sample-weighted "
            "contaminated) and the latency/energy advantage stays large (%+.1f%% / %+.1f%%)."
            % (am_reason_C["delta"], am_reason_C["lo"], am_reason_C["hi"],
               arms["contaminated"]["deltas"]["method_accuracy_max_veto"]["always_32b_reasoning"]["all8"]
               ["sample_weighted"]["delta"], reason_lat, reason_en),
            "The 'reasoning mode is actively harmful on free-text medical VQA' result is untouched by "
            "the verifier and by the weighting: PathVQA-open 0.1087 reasoning vs 0.3760 direct, "
            "SLAKE-open 0.6791 vs 0.8186, VQA-RAD-open 0.5450 vs 0.6000.",
            "On the 5 multiple-choice cells accuracy-max still beats always-32B-direct (%+.4f "
            "[%+.4f, %+.4f]); that half never used the verifier."
            % (V("method_accuracy_max_veto", "always_32b_direct", "mcq_only", "clean_L1")["delta"],
               V("method_accuracy_max_veto", "always_32b_direct", "mcq_only", "clean_L1")["lo"],
               V("method_accuracy_max_veto", "always_32b_direct", "mcq_only", "clean_L1")["hi"]),
            "The verifier is still a real ranker after de-contamination (candidate AUROC 0.943 -> 0.886 "
            "at L1, per verifier_disjoint_retrain_2026-07-30.json); what collapsed is oracle CONVERSION, "
            "and with it the arm's ability to hold parity at low escalation."],
        what_does_not_survive=[
            "'The method beats a single 32B forward pass.' Under macro + clean L1 accuracy-max is "
            "%+.4f [%+.4f, %+.4f] vs always-32B-direct -- a TIE -- and accuracy-max-fusion is a "
            "significant LOSS (%+.4f [%+.4f, %+.4f])."
            % (am_direct_C["delta"], am_direct_C["lo"], am_direct_C["hi"],
               fu_direct_C["delta"], fu_direct_C["lo"], fu_direct_C["hi"]),
            "'Compute-lean matches the strong model at half the compute.' Under macro + clean L1 it is "
            "%+.4f [%+.4f, %+.4f] (a significant LOSS) at %.2fx the compute of always-32B-direct."
            % (cl_direct_C["delta"], cl_direct_C["lo"], cl_direct_C["hi"], cl_flops_C),
            "'The open-text arm beats always-32B-direct.' Macro + clean L1: accuracy-max %+.4f "
            "[%+.4f, %+.4f] (TIE), compute-lean %+.4f [%+.4f, %+.4f] (LOSS)."
            % (open_am_C["delta"], open_am_C["lo"], open_am_C["hi"],
               open_cl_C["delta"], open_cl_C["lo"], open_cl_C["hi"]),
            "'The method Pareto-dominates the 32B baselines.' Under macro + clean L1 (honestly "
            "re-costed) compute-lean and accuracy-max-fusion are both strictly DOMINATED by "
            "always-32B-direct; only accuracy-max-veto stays on the frontier, and only by a "
            "%+.4f accuracy edge that is not statistically distinguishable from zero."
            % (am_direct_C["delta"],)])

    out = dict(
        title="8-cell MACRO headline recomputed with an UNCONTAMINATED (disjoint-trained) open-text "
              "verifier -- accuracy AND cost, with the sample-weighted column kept for contrast.",
        reproduce="python3 src/cascade_methods/macro_headline_clean_verifier.py",
        date="2026-07-30", no_gpu=True, no_fabricated_numbers=True, n_bootstrap=NBOOT, seed=SEED,
        headline_level="clean_L1 (no eval image, no eval item)",
        l2_status="LOWER BOUND ONLY -- L2 additionally strips every training item sharing an eval "
                  "question text, which on these templated sets removes legitimate in-distribution "
                  "coverage and conflates de-contamination with distribution shift.",
        arms={n: dict(adapter=a, description=d) for n, a, d in ARMS},
        what_was_swapped=dict(
            only="the verifier P(correct) scores on the three open-text cells",
            identical_across_arms=["candidate answer lists (preds)", "per-candidate judge labels (sl)",
                                   "greedy label (greedy_ok)", "32B strong-leg judge labels",
                                   "every multiple-choice cell (never touches the verifier)"],
            pairing_check=pairing,
            mechanism="exact-path redirect of transfer_dump_{ds}_lingshu7b.json at builtins.open; "
                      "no pipeline code is duplicated or re-implemented"),
        validation=dict(
            contaminated_arm_reproduces_published_macro_artifact=True,
            n_fields_compared_exactly=n_checked,
            published_artifact=os.path.relpath(PUBLISHED, ROOT),
            bo8_open_arm_reproduces_retrain_artifact=bo8_ok,
            retrain_artifact=os.path.relpath(RETRAIN, ROOT),
            note="Every accuracy level, every delta point estimate AND both CI bounds, every per-cell "
                 "escalation rate and every cost ratio in the contaminated column matched the "
                 "published artifact EXACTLY (same RNG stream replayed), so the clean column differs "
                 "only because the verifier differs."),
        per_cell_table=per_cell,
        accuracy_levels={n: arms[n]["acc_levels"] for n in arms},
        accuracy_subpools={n: arms[n]["subpools"] for n in arms},
        accuracy_side_by_side=acc_side,
        deltas={n: arms[n]["deltas"] for n in arms},
        side_by_side_claims=side_by_side,
        cost={n: arms[n]["cost"] for n in arms},
        cost_side_by_side=cost_sbs,
        cost_provenance=arms["contaminated"]["cost_provenance"],
        escalation={n: arms[n]["escalation"] for n in arms},
        escalation_side_by_side=esc_side,
        open_arm_bo8_parity=dict(
            what="the FIXED best-of-8 + verifier-confidence-gate open arm -- the arm the published "
                 "'3.97% escalation / 548.3 ms / -94.8% latency' claim describes "
                 "(METHOD_FINAL_2026-07.md 5.x). tau is cross-fit to reach 32B-no-think parity at "
                 "minimum escalation, so a weaker verifier buys parity by escalating more.",
            published_claim=dict(escalation=PUB_OPEN_ESC, lat_par_ms=PUB_OPEN_LAT,
                                 pct_vs_32b_think=PUB_OPEN_PCT),
            cost_model=f"lat = BO8 {BO8_MS} ms + esc x {G32N_MS} ms; FLOP-eq = {BO8_F} + esc x {G32N_F}; "
                       f"energy = {BO8_J} J + esc x {G32N_J} J (all measured batch-1 constants)",
            by_arm=open_lat),
        open_arm_pandora_pipeline=dict(
            what="the arm the macro headline actually costs: Pandora adaptive-N (draw-and-stop) + gate. "
                 "This is what cost.open_only reports.",
            by_arm=pandora_lat),
        paired_clean_minus_contaminated_open_cells=paired,
        joint_claim=joint,
        honest_headline=honest_headline,
        verdicts=verdicts,
        statistics_note="Items are bootstrapped WITHIN each cell (paired across systems, exact "
                        "pattern-multinomial draw) and the macro average is recomputed per replicate. "
                        "The CIs therefore capture WITHIN-dataset sampling noise ONLY -- they do NOT "
                        "capture dataset-selection noise. With 5-8 datasets, resampling datasets is "
                        "hopelessly unstable, so a leave-one-cell-out point range is given instead.",
        caveats=[
            "The de-contamination changes ONLY the three open-text cells. Every multiple-choice result "
            "in this file is identical across arms by construction; where a multiple-choice claim moves, "
            "the macro re-weighting caused it, not the verifier.",
            "L1 keeps question TEMPLATES that recur with different images. That is ordinary "
            "generalization on templated medical VQA, not leakage; L2 removes them and is reported as a "
            "conservative lower bound only.",
            "The Pandora open arm's iso-accuracy target a_fix is itself derived from the bo8+verifier "
            "arm, so a weaker verifier lowers the target as well as the delivered accuracy.",
            "32B-with-reasoning on the open cells is a MEASURED judged per-sample vector "
            "(opentext_32b_think_full); on PATH_VQA_closed there is no reasoning dump, so reasoning = "
            "direct there.",
            "FLOP-eq counts every 7B forward in full; batch-1 latency does not, because the best-of-N "
            "draws are issued in parallel. The two axes disagree for the open arm and both are given."],
        runtime_s=round(time.time() - t0, 1))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with _REAL_OPEN(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


# =================================================================================================
# 6.  console
# =================================================================================================
def console(o):
    P = print
    P("\n" + "=" * 108)
    P("MACRO HEADLINE ON A CLEAN VERIFIER  --  8 cells, 1/8 each (PRIMARY);  L1 = headline, L2 = lower bound")
    P("=" * 108)
    P(f"validation: contaminated column reproduces {os.path.basename(PUBLISHED)} EXACTLY -> "
      f"the clean column differs only because the verifier differs")

    P("\n---- 1. PER-CELL ACCURACY (only the 3 open cells move) " + "-" * 46)
    hdr = f"{'cell':<16}{'n':>7}  {'system':<26}{'contam':>9}{'cleanL1':>9}{'cleanL2':>9}"
    P(hdr); P("-" * len(hdr))
    for k, rec in o["per_cell_table"].items():
        for i, s in enumerate(SYSTEMS):
            a = rec["accuracy"]
            P(f"{k if i==0 else '':<16}{rec['n'] if i==0 else '':>7}  {s:<26}"
              f"{a['contaminated'][s]:>9.4f}{a['clean_L1'][s]:>9.4f}{a['clean_L2'][s]:>9.4f}")
        P("")

    P("---- 2. ACCURACY LEVELS, 4 columns " + "-" * 68)
    P(f"{'system':<28}{'A pub(sw,contam)':>18}{'B macro(contam)':>18}{'C macro+cleanL1':>18}{'D macro+cleanL2':>18}")
    for s, r in o["accuracy_side_by_side"].items():
        P(f"{s:<28}" + "".join(f"{r[c]:>18.4f}" for c, _, _ in
                               [(k, 0, 0) for k in ("A_published_contaminated_sample_weighted",
                                                    "B_macro_only_contaminated",
                                                    "C_macro_plus_clean_verifier_L1",
                                                    "D_macro_plus_clean_verifier_L2_lower_bound")]))

    P("\n---- 3. EVERY HEADLINE ACCURACY CLAIM, 3(+1) COLUMNS " + "-" * 50)
    for r in o["side_by_side_claims"]:
        def f(c):
            x = r[c]
            return f"{x['delta']:+.4f} [{x['ci95'][0]:+.4f},{x['ci95'][1]:+.4f}] {x['verdict']:<4}"
        P(f"\n{r['claim']}   |   {r['pool']}")
        P(f"   A published (sample-wtd, contaminated) : {f('A_published_contaminated_sample_weighted')}")
        P(f"   B macro only (contaminated)           : {f('B_macro_only_contaminated')}")
        P(f"   C macro + CLEAN verifier  (L1, HEAD)  : {f('C_macro_plus_clean_verifier_L1')}")
        P(f"   D macro + CLEAN verifier  (L2, bound) : {f('D_macro_plus_clean_verifier_L2_lower_bound')}")
        P(f"   effect of macro alone {r['effect_of_macro_alone']:+.4f} | of clean verifier alone "
          f"{r['effect_of_clean_verifier_alone_under_macro']:+.4f} | net {r['net_A_to_C']:+.4f} | "
          f"{r['significance_change_A_to_C']}")

    P("\n---- 4. ESCALATION " + "-" * 84)
    for k, r in o["escalation_side_by_side"].items():
        P(f"{k:<26}" + "  ".join(f"{c.split('_')[0]}={r[c]:.4f}" for c in r))

    P("\n---- 5. OPEN-ARM COST (bo8 + verifier-confidence gate: the '-94.8%' claim) " + "-" * 28)
    for name, r in o["open_arm_bo8_parity"]["by_arm"].items():
        for lab in ("sample_weighted", "macro_cells"):
            x = r[lab]
            P(f"{name:<14}{lab:<17} esc={x['escalation']:.4f}  lat={x['lat_par_ms']:>7.1f} ms  "
              f"vs32B-think {x['pct_vs_32b_think']:>6.1f}%  vs32B-direct {x['pct_vs_32b_direct']:>+6.1f}%  "
              f"FLOP-eq {x['flops']:.2f} ({x['flops_x_vs_32b_direct']:.2f}x direct)")

    P("\n---- 6. JOINT CLAIM (macro + clean L1) " + "-" * 64)
    P(json.dumps(o["joint_claim"], indent=1, default=float))
    P("\n---- 7. VERDICTS " + "-" * 86)
    P(json.dumps(o["verdicts"], indent=1, default=float))
    P("\n---- 8. CORRECTED HEADLINE " + "-" * 76)
    P(json.dumps(o["honest_headline"], indent=1, default=float))
    P(f"\nwrote {os.path.relpath(OUT, ROOT)}   ({o['runtime_s']}s)")


if __name__ == "__main__":
    console(run())
