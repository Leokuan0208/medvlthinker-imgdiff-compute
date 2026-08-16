#!/usr/bin/env python3
"""
veto_binning_patch_s9.py -- appends S9 to results/cascade_methods/artifacts/veto_binning_2026-08-15.json.

S9  THE PMC ANSWER-PRIOR CONTROL, APPLIED TO THE OTHER CLOSED CELLS.  The certified veto's PMC gain is
    44% attributable to that split's answer-letter skew, so the same question has to be asked of the
    cell the knob newly switches on -- SLAKE-closed -- and of the other closed cells the sweep can
    reach.  For each closed cell: the gold-answer marginal and the constant-answer floor straight from
    the MedEvalKit dumps, then the nested-CV delta of every selection rule reported BOTH raw
    (sample-weighted) and ANSWER-BALANCED (macro over gold-answer strata with n >= 30, equal weight),
    with stratified paired bootstraps.  Under a balanced gold marginal a constant-answer policy carries
    no advantage, so a delta that survives is item-level competence and not an answer prior.

Run AFTER veto_binning_sweep.py, from the repo root:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_patch_s9.py
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import veto_binning_sweep as V   # noqa: E402
import beat32b_fusion as BF      # noqa: E402

CLOSED = {"SLAKE_closed": ("SLAKE", "SLAKE"),
          "VQA_RAD_closed": ("VQA_RAD", "YESNO"),
          "PATH_VQA_closed": ("PATH_VQA", "YESNO"),
          "MedXpertQA-MM": ("MedXpertQA-MM", None)}
MIN_STRATUM = 30


def gold_for(cell):
    ds, mode = CLOSED[cell]
    r7 = BF.load_raw("lingshu7b_full", ds)
    if mode == "SLAKE":
        idx = [i for i in range(len(r7)) if r7[i].get("answer_type") == "CLOSED"]
    elif mode == "YESNO":
        idx = [i for i in range(len(r7))
               if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(len(r7)))
    return np.array([str(r7[i].get("answer", "")).strip().lower() for i in idx])


def main():
    art = json.load(open(V.OUT))
    cells = V.load_cells()
    labels = {c: (cells[c]["ok7"], cells[c]["ok32"]) for c in V.MCQ_CELLS}

    acc_ok = {r: {c: [] for c in V.MCQ_CELLS} for r in V.RULES}
    for s in range(V.N_SEEDS):
        rng = np.random.default_rng(V.SEED + 1000 * s)
        caches = {c: V.CellCache(cells[c]["c7"], cells[c]["n"], rng) for c in V.MCQ_CELLS}
        res = V.nested_cv(caches, labels)
        for r in V.RULES:
            for c in V.MCQ_CELLS:
                acc_ok[r][c].append(res[r][c]["ok"])
        print(f"  seed {s} done")

    # EXACT seed mean (an incremental += x/N leaves ~1e-16 residue that turns a genuinely
    # unchanged cell into a spurious non-zero difference vector)
    avg_ok = {r: {c: np.mean(np.stack(acc_ok[r][c]), axis=0) for c in V.MCQ_CELLS} for r in V.RULES}

    out = {}
    for cell in CLOSED:
        gold = gold_for(cell)
        n = cells[cell]["n"]
        assert len(gold) == n, (cell, len(gold), n)
        cnt = collections.Counter(gold)
        keys = sorted([k for k, v in cnt.items() if v >= MIN_STRATUM], key=lambda k: -cnt[k])
        cov = sum(cnt[k] for k in keys) / n
        rec = dict(
            n=int(n),
            gold_answer_marginal={k: dict(n=int(v), frac=V.r4(v / n)) for k, v in cnt.most_common(8)},
            n_distinct_gold_answers=int(len(cnt)),
            strata_used=keys, strata_min_n=MIN_STRATUM,
            item_coverage_of_strata=V.r4(cov),
            constant_answer_floor=V.r4(max(cnt.values()) / n),
            balanced_floor_under_equal_strata=V.r4(1.0 / len(keys)) if keys else None,
            rules={})
        for r in V.RULES:
            diff = avg_ok[r][cell] - cells[cell]["ok32"]
            raw = V.paired_boot(diff, seed=V.SEED + 31)
            bal = (V.strat_macro_boot(diff, gold, keys, seed=V.SEED + 32) if len(keys) >= 2 else None)
            rec["rules"][r] = dict(
                raw=dict(delta=raw["delta"], ci=[raw["lo"], raw["hi"]], verdict=raw["verdict"]),
                answer_balanced=(dict(delta=bal["delta"], ci=[bal["lo"], bal["hi"]],
                                      verdict=bal["verdict"], n_strata=bal["n_strata"])
                                 if bal else None))
        out[cell] = rec

    art["S9_answer_prior_control_closed_cells"] = dict(
        what="the PMC answer-letter control, applied to every closed multiple-choice cell the knob can "
             "reach -- above all SLAKE-closed, the cell the tuned veto newly switches on. Raw vs "
             "ANSWER-BALANCED (macro over gold-answer strata with n >= %d)." % MIN_STRATUM,
        why="the certified veto's PMC gain is 44% attributable to test_2.csv's answer-letter skew "
            "(artifacts/pmcvqa_answer_bias_audit_2026-08-11.json); a new cell must clear the same bar "
            "before it is treated as competence rather than a prior.",
        per_cell=out)
    art.setdefault("generated_by", []).append("src/cascade_methods/veto_binning_patch_s9.py (S9)")
    json.dump(art, open(V.OUT, "w"), indent=2, default=str)
    print(f"\nappended S9 to {V.OUT}\n")

    for cell, rec in out.items():
        print(f"== {cell}  n={rec['n']}  strata={rec['strata_used'][:4]}... cov={rec['item_coverage_of_strata']}"
              f"  const-floor={rec['constant_answer_floor']}")
        for r in V.RULES:
            v = rec["rules"][r]
            b = v["answer_balanced"]
            bs = (f"{b['delta']:+.5f} {b['ci']} {b['verdict']}" if b else "n/a")
            print(f"   {r}: raw {v['raw']['delta']:+.5f} {v['raw']['ci']} {v['raw']['verdict']:<5} | "
                  f"balanced {bs}")


if __name__ == "__main__":
    main()
