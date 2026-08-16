#!/usr/bin/env python3
"""
veto_binning_patch_s8.py -- appends S8 to results/cascade_methods/artifacts/veto_binning_2026-08-15.json.

S8a  SEED-PAIRED DIFFERENCES.  The S3 nested CV is re-run with the SAME pinned seeds (deterministic,
     verified against the S3 seedstats already in the artifact) so the per-seed macro of every rule is
     recorded, together with the PAIRED per-seed difference against the shipped arm R0.  A rule whose
     10 seed-paired differences are all the same sign is a stable change; a rule whose item bootstrap
     is "significant" but whose seed sd is larger than the CI half-width is not.

S8b  THE LETTER-BALANCED FRONTIER.  The whole 135-setting grid ranked by the PMC delta computed
     LETTER-BALANCED (macro over the four gold letters, 1/4 each -- the answer-prior-removed currency)
     and on the gold-A stratum, next to the raw delta.  Answers directly: does any setting that raises
     the RAW PMC delta also raise the letter-balanced one, or is the whole tuning gain answer-prior?

S8c  THE SELECTED ARMS IN BOTH PMC CURRENCIES.  For each selection rule, the seed-averaged nested-CV
     PMC vector re-scored raw / letter-balanced / gold-A with full bootstrap CIs.

Run AFTER veto_binning_sweep.py, from the repo root:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_patch_s8.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import veto_binning_sweep as V   # noqa: E402
import beat32b_fusion as BF      # noqa: E402


def main():
    art = json.load(open(V.OUT))
    cells = V.load_cells()
    zv = np.load(V.VEC)
    open_acc_shipped = {c: float(zv[f"{c}|method_accuracy_max_veto"].mean()) for c in V.OPEN_CELLS}
    open_acc_direct = {c: float(zv[f"{c}|always_32b_direct"].mean()) for c in V.OPEN_CELLS}

    C = cells["PMC_VQA"]
    r7 = BF.load_raw("lingshu7b_full", "PMC_VQA")
    gold = np.array([str(r["answer"]).strip() for r in r7[:C["n"]]])

    # ------------------------------------------------------------------ S8a
    labels = {c: (cells[c]["ok7"], cells[c]["ok32"]) for c in V.MCQ_CELLS}
    per_seed = {r: [] for r in V.RULES}
    acc_ok = {r: {c: [] for c in V.MCQ_CELLS} for r in V.RULES}
    for s in range(V.N_SEEDS):
        rng = np.random.default_rng(V.SEED + 1000 * s)
        caches = {c: V.CellCache(cells[c]["c7"], cells[c]["n"], rng) for c in V.MCQ_CELLS}
        res = V.nested_cv(caches, labels)
        for r in V.RULES:
            acc = {c: float(res[r][c]["ok"].mean()) for c in V.MCQ_CELLS}
            per_seed[r].append(dict(
                macro=V.r5(float(sum(acc.values()) + sum(open_acc_shipped.values())) / 8.0),
                macro_mcqonly=V.r5(float(sum(acc.values()) + sum(open_acc_direct.values())) / 8.0)))
            for c in V.MCQ_CELLS:
                acc_ok[r][c].append(res[r][c]["ok"])
        print(f"  seed {s} done")

    # EXACT seed mean (an incremental += x/N leaves ~1e-16 residue that turns a genuinely
    # unchanged cell into a spurious non-zero difference vector)
    avg_ok = {r: {c: np.mean(np.stack(acc_ok[r][c]), axis=0) for c in V.MCQ_CELLS} for r in V.RULES}

    s8a = {}
    base = np.array([x["macro"] for x in per_seed["R0"]])
    for r in V.RULES:
        m = np.array([x["macro"] for x in per_seed[r]])
        dif = m - base
        s8a[r] = dict(
            per_seed_macro=[V.r5(x) for x in m],
            per_seed_macro_mcqonly=[x["macro_mcqonly"] for x in per_seed[r]],
            mean=V.r5(m.mean()), sd=V.r5(m.std(ddof=1)), range=[V.r5(m.min()), V.r5(m.max())],
            paired_vs_R0_shipped=dict(
                mean=V.r5(dif.mean()), sd=V.r5(dif.std(ddof=1)),
                range=[V.r5(dif.min()), V.r5(dif.max())],
                n_seeds_positive=int((dif > 0).sum()), n_seeds=int(len(dif)),
                all_same_sign=bool((dif > 0).all() or (dif < 0).all())))
    # consistency check against the S3 block already in the artifact
    chk = {r: V.r5(abs(s8a[r]["mean"] - art["S3_nested_cv"]["results"][r]["macro_seedstat"]["mean"]))
           for r in V.RULES}
    s8a["_reproduction_check_vs_S3"] = dict(per_rule_abs_deviation=chk,
                                            max_abs_deviation=V.r5(max(chk.values())),
                                            passed=bool(max(chk.values()) < 1e-5))

    # ------------------------------------------------------------------ S8b
    grid = art["S1_grid_fixed_settings"]["per_cell"]["PMC_VQA"]
    pe = art["S2_PMC_letter_honesty"]["point_estimates_every_setting"]
    rows = []
    for k, v in pe.items():
        rows.append(dict(setting=k, veto_rate=v["veto_rate"], raw_delta=v["raw_delta"],
                         raw_verdict=v["raw_verdict"],
                         letter_balanced_delta=v["letter_balanced_delta_point"],
                         goldA_delta=v["goldA_delta_point"],
                         x_direct=grid[k]["x_direct"]))
    by_raw = sorted(rows, key=lambda r: -r["raw_delta"])
    by_lb = sorted(rows, key=lambda r: -r["letter_balanced_delta"])
    by_gA = sorted(rows, key=lambda r: -r["goldA_delta"])
    ship = f"{V.SHIPPED[0]}|{V.SHIPPED[1]}"
    rank = lambda lst: 1 + [r["setting"] for r in lst].index(ship)
    corr = np.corrcoef([r["raw_delta"] for r in rows], [r["letter_balanced_delta"] for r in rows])[0, 1]
    s8b = dict(
        what="Every one of the 135 settings' PMC delta in BOTH currencies: raw (sample-weighted over "
             "test_2.csv, whose gold letters are 13/36/38/13 % A/B/C/D) and LETTER-BALANCED (macro over "
             "the four gold letters, 1/4 each -- under a uniform gold marginal a constant-letter policy "
             "scores exactly 0.25 and confers no advantage, so any surviving delta is not a letter prior).",
        shipped_setting=ship,
        shipped_rank_by_raw_delta=rank(by_raw),
        shipped_rank_by_letter_balanced_delta=rank(by_lb),
        shipped_rank_by_goldA_delta=rank(by_gA),
        n_settings=len(rows),
        best_by_raw=by_raw[:6], best_by_letter_balanced=by_lb[:6], best_by_goldA=by_gA[:6],
        pearson_r_raw_vs_letter_balanced=V.r5(corr),
        n_settings_letter_balanced_above_shipped=int(sum(
            1 for r in rows if r["letter_balanced_delta"] > pe[ship]["letter_balanced_delta_point"])),
        n_settings_goldA_above_shipped=int(sum(
            1 for r in rows if r["goldA_delta"] > pe[ship]["goldA_delta_point"])))

    # ------------------------------------------------------------------ S8c
    s8c = {}
    for r in V.RULES:
        diff = avg_ok[r]["PMC_VQA"] - C["ok32"]
        s8c[r] = dict(
            raw=V.paired_boot(diff, seed=V.SEED + 21),
            letter_balanced=V.strat_macro_boot(diff, gold, V.LETTERS, seed=V.SEED + 22),
            gold_A=V.paired_boot(diff[gold == "A"], seed=V.SEED + 23),
            gold_BC=V.paired_boot(diff[(gold == "B") | (gold == "C")], seed=V.SEED + 24))

    art["S8_seed_pairing_and_letter_currency"] = dict(
        S8a_seed_paired=dict(
            what="per-seed nested-CV macro for every rule and the PAIRED per-seed difference vs the "
                 "shipped arm R0 run through the identical outer folds",
            results=s8a),
        S8b_letter_balanced_frontier=s8b,
        S8c_selected_arms_both_currencies=dict(
            what="the seed-averaged nested-CV PMC vector of each rule, re-scored in BOTH currencies "
                 "with full paired item bootstraps (nboot=%d)" % V.NBOOT,
            reference="artifacts/pmcvqa_answer_bias_audit_2026-08-11.json",
            results=s8c))
    art.setdefault("generated_by", []).append("src/cascade_methods/veto_binning_patch_s8.py (S8)")
    json.dump(art, open(V.OUT, "w"), indent=2, default=str)
    print(f"\nappended S8 to {V.OUT}")

    print("\nS8a seed-paired macro vs shipped R0:")
    for r in V.RULES:
        p = s8a[r]["paired_vs_R0_shipped"]
        print(f"  {r}: {p['mean']:+.5f} sd {p['sd']:.5f} range {p['range']} "
              f"positive {p['n_seeds_positive']}/{p['n_seeds']} same-sign {p['all_same_sign']}")
    print(f"  reproduction check vs S3: max abs dev {s8a['_reproduction_check_vs_S3']['max_abs_deviation']}")
    print("\nS8b: shipped setting rank -- raw %d/135, letter-balanced %d/135, gold-A %d/135; "
          "pearson(raw, letter-balanced) = %.3f"
          % (s8b["shipped_rank_by_raw_delta"], s8b["shipped_rank_by_letter_balanced_delta"],
             s8b["shipped_rank_by_goldA_delta"], s8b["pearson_r_raw_vs_letter_balanced"]))
    print("  best by letter-balanced:", [(r["setting"], r["letter_balanced_delta"], r["raw_delta"])
                                         for r in s8b["best_by_letter_balanced"][:4]])
    print("\nS8c PMC in both currencies (seed-averaged nested-CV vectors):")
    for r in V.RULES:
        v = s8c[r]
        print(f"  {r}: raw {v['raw']['delta']:+.5f} {v['raw']['verdict']:<5} | letter-bal "
              f"{v['letter_balanced']['delta']:+.5f} [{v['letter_balanced']['lo']:+.5f},{v['letter_balanced']['hi']:+.5f}] "
              f"{v['letter_balanced']['verdict']:<5} | gold-A {v['gold_A']['delta']:+.5f} "
              f"[{v['gold_A']['lo']:+.5f},{v['gold_A']['hi']:+.5f}] {v['gold_A']['verdict']}")


if __name__ == "__main__":
    main()
