#!/usr/bin/env python3
"""
veto_binning_followup.py -- appends S6 (the mis-certification trade) and S7 (the global-setting
frontier) to results/cascade_methods/artifacts/veto_binning_2026-08-15.json.

S6  THE TRADE, MEASURED.  "A looser bound certifies more bins but certifies some of them wrongly."
    For every (cell, n_bins, alpha_z) on the PUBLISHED fold map, walk each cross-fit fold and every
    bin the certificate accepted, then look at what that bin actually delivered on the HELD-OUT test
    items of that fold:
      - n certified (fold, bin) pairs and the item share they cover  (= the veto rate)
      - MIS-CERTIFIED pairs = certified bins whose held-out (ok7 - ok32) mean is NEGATIVE, i.e. the
        certificate was wrong there
      - the gain summed over correctly-certified bins and the loss summed over mis-certified bins,
        so the net delta is decomposed into "what the certificate bought" and "what it cost".
    This is the honest form of the accuracy/coverage trade the knob controls.

S7  GLOBAL-SETTING FRONTIER.  For each of the 135 settings, the arm "apply this one setting to all 5
    multiple-choice cells, carry the 3 open cells from the shipped accuracy-max arm": 8-cell macro
    accuracy, delta vs always-32B-direct, macro FLOP-eq, and the per-cell guardrail flags.
    IN-SAMPLE for the SETTING CHOICE -- each row is individually leakage-free (its own 5-fold
    cross-fit) but picking the best row is exactly the leakage the nested CV in S3 and the permutation
    null in S4 exist to price.  Reported so the shape of the knob is visible, never as a claim.

Run AFTER veto_binning_sweep.py, from the repo root:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/veto_binning_followup.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
import veto_binning_sweep as V   # noqa: E402

OUT = V.OUT


def main():
    art = json.load(open(OUT))
    cells = V.load_cells()
    zv = np.load(V.VEC)
    open_acc_shipped = {c: float(zv[f"{c}|method_accuracy_max_veto"].mean()) for c in V.OPEN_CELLS}
    open_flops = V.OPEN_FLOPS_SHIPPED
    bar_cell = {c: float(zv[f"{c}|always_32b_direct"].mean()) for c in V.ALL8}
    bar_macro = float(np.mean([bar_cell[c] for c in V.ALL8]))

    # ---------------------------------------------------------------- S6
    s6 = {c: {} for c in V.MCQ_CELLS}
    for c in V.MCQ_CELLS:
        C = cells[c]
        fold = np.arange(C["n"]) % V.K_PUB
        for nb_ in V.N_BINS:
            plan = V.binning_plan(C["c7"], fold, nb_)
            for az in V.ALPHA_Z:
                n_cert = n_mis = 0
                items_cert = items_mis = 0
                gain_pos = gain_neg = 0.0
                for p in plan:
                    cert = V.certify(p["btr"], C["ok7"][p["tr"]], C["ok32"][p["tr"]], p["nb"], az)
                    for b in np.flatnonzero(cert):
                        sel = p["te"][p["bte"] == b]
                        if len(sel) == 0:
                            continue
                        g = float((C["ok7"][sel] - C["ok32"][sel]).sum())
                        n_cert += 1
                        items_cert += len(sel)
                        if g < 0:
                            n_mis += 1
                            items_mis += len(sel)
                            gain_neg += g
                        else:
                            gain_pos += g
                s6[c][f"{nb_}|{az}"] = dict(
                    n_certified_bin_folds=int(n_cert),
                    n_miscertified_bin_folds=int(n_mis),
                    miscertification_rate_bins=(V.r5(n_mis / n_cert) if n_cert else None),
                    items_certified=int(items_cert),
                    items_in_miscertified_bins=int(items_mis),
                    item_share_miscertified=(V.r5(items_mis / items_cert) if items_cert else None),
                    veto_rate=V.r5(items_cert / C["n"]),
                    gain_from_correct_bins=V.r5(gain_pos / C["n"]),
                    loss_from_miscertified_bins=V.r5(gain_neg / C["n"]),
                    net_delta=V.r5((gain_pos + gain_neg) / C["n"]))
    art["S6_miscertification_trade"] = dict(
        what="For every setting, each certified (fold, bin) is checked against what it actually "
             "delivered on that fold's HELD-OUT items. A certified bin whose held-out (ok7 - ok32) mean "
             "is negative is a MIS-CERTIFICATION: the Wilson bound accepted it and the 7B was worse "
             "there. net_delta = gain_from_correct_bins + loss_from_miscertified_bins, and equals the "
             "S1 delta for the same setting.",
        note="A looser alpha_z buys coverage (more items vetoed = more compute saved) and pays for it "
             "in mis-certified bins. This block is the price list.",
        per_cell=s6)

    # ---------------------------------------------------------------- S7
    grid = art["S1_grid_fixed_settings"]["per_cell"]
    rows = []
    for nb_ in V.N_BINS:
        for az in V.ALPHA_Z:
            key = f"{nb_}|{az}"
            acc = {c: grid[c][key]["acc"] for c in V.MCQ_CELLS}
            vr = {c: grid[c][key]["veto_rate"] for c in V.MCQ_CELLS}
            macro = float(sum(acc.values()) + sum(open_acc_shipped.values())) / 8.0
            cellfl = {c: grid[c][key]["flops"] for c in V.MCQ_CELLS}   # per-fold honest cost from S1
            fl = [cellfl[c] for c in V.MCQ_CELLS] + [open_flops[c] for c in V.OPEN_CELLS]
            mfl = float(np.mean(fl))
            fl_m = [cellfl[c] for c in V.MCQ_CELLS] + [V.FLOP_32B] * 3
            rows.append(dict(
                setting=key, n_bins=nb_, alpha_z=az,
                macro=V.r5(macro), macro_delta_vs_direct=V.r5(macro - bar_macro),
                macro_flops=V.r4(mfl), macro_x_direct=V.r4(mfl / V.FLOP_32B),
                macro_mcqonly=V.r5(float(sum(acc.values()) + sum(bar_cell[c] for c in V.OPEN_CELLS)) / 8.0),
                macro_mcqonly_delta=V.r5(float(sum(acc[c] - bar_cell[c] for c in V.MCQ_CELLS)) / 8.0),
                macro_x_direct_mcqonly=V.r4(float(np.mean(fl_m)) / V.FLOP_32B),
                veto_rate={c: vr[c] for c in V.MCQ_CELLS},
                per_cell_flops=cellfl,
                per_cell_delta={c: grid[c][key]["delta"] for c in V.MCQ_CELLS},
                guardrail_sig_loss=[c for c in V.MCQ_CELLS if grid[c][key]["verdict"] == "LOSS"],
                guardrail_point_negative=[c for c in V.MCQ_CELLS if grid[c][key]["delta"] < 0]))
    rows_sorted = sorted(rows, key=lambda r: -r["macro_delta_vs_direct"])
    cheap = [r for r in rows if r["macro_x_direct_mcqonly"] <= 1.0 and not r["guardrail_sig_loss"]]
    art["S7_global_setting_frontier"] = dict(
        what="Arm = apply ONE (n_bins, alpha_z) to all 5 multiple-choice cells, carry the 3 open cells "
             "unchanged from the shipped accuracy-max arm. 8-cell macro, Variant B, clean verifier.",
        leakage_warning="IN-SAMPLE FOR THE SETTING CHOICE. Every individual row is a leakage-free "
                        "5-fold cross-fit of a pre-specified setting; picking the best row is the "
                        "leakage that S3 (nested CV) and S4 (permutation null) price. Do not quote a "
                        "top row as a result.",
        bar_macro_always_32b_direct=V.r5(bar_macro),
        shipped_accuracy_max_macro=V.r5(float(np.mean(
            [zv[f"{c}|method_accuracy_max_veto"].mean() for c in V.ALL8]))),
        top15_by_macro=rows_sorted[:15],
        bottom5_by_macro=rows_sorted[-5:],
        n_rows_guardrail_clean=int(sum(1 for r in rows if not r["guardrail_sig_loss"])),
        n_rows=int(len(rows)),
        cheapest_guardrail_clean_rows=sorted(cheap, key=lambda r: r["macro_x_direct_mcqonly"])[:10],
        all_rows=rows)

    art.setdefault("generated_by", []).append(
        "src/cascade_methods/veto_binning_followup.py (S6, S7)")
    json.dump(art, open(OUT, "w"), indent=2, default=str)
    print(f"appended S6 + S7 to {OUT}")

    # console
    print("\nTOP 10 global settings by 8-cell macro (IN-SAMPLE for the choice -- not a claim):")
    print(f"  {'setting':>12}{'macro':>9}{'dmacro':>9}{'x_direct':>10}{'x_mcqonly':>11}  guardrail_sig_loss")
    for r in rows_sorted[:10]:
        print(f"  {r['setting']:>12}{r['macro']:>9.5f}{r['macro_delta_vs_direct']:>+9.5f}"
              f"{r['macro_x_direct']:>10.3f}{r['macro_x_direct_mcqonly']:>11.3f}  {r['guardrail_sig_loss']}")
    print("\nCHEAPEST guardrail-clean settings (MCQ-only frame):")
    for r in sorted(cheap, key=lambda x: x["macro_x_direct_mcqonly"])[:10]:
        print(f"  {r['setting']:>12}{r['macro']:>9.5f}{r['macro_delta_vs_direct']:>+9.5f}"
              f"{r['macro_x_direct_mcqonly']:>11.3f}  veto={ {c: r['veto_rate'][c] for c in V.MCQ_CELLS} }")


if __name__ == "__main__":
    main()
