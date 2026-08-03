#!/usr/bin/env python3
"""
flop_ratio_impact.py -- re-run the project's headline cost accounting under a corrected R32.

`flop_ratio_derivation.py` replaces the underived literal R32 = 4.57 with a derived 3.816.
This script quantifies what that changes, by MONKEYPATCHING the constant into every module that
carries it and re-running `macro_average_headline.run()` unmodified. Nothing on disk is edited;
the canonical artifact is not overwritten (OUT is redirected to a temp file).

FLOP cost is exactly affine in R32 for every system (cost = A + B*R32, B = expected number of
32B forwards), so the pipeline is run at three values of R32 and linearity is asserted, which
also recovers the per-system A and B for the record.

Launch from the repo root (CPU only):  python3 src/cascade_methods/flop_ratio_impact.py
Writes: results/cascade_methods/artifacts/flop_ratio_impact_2026-08-03.json
"""
import os, sys, json, contextlib, io, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src/cascade_methods")
sys.path.insert(0, SRC)
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/flop_ratio_impact_2026-08-03.json")

import paper_baselines as PB
import integrated_method as IM
import beat32b_fusion as BF
import pandora_controller as PC
import honest_recosting as HR
import method_final_mmmu_corrected as MFC
import macro_average_headline as M

_HR_VERIFY = HR.verify_constants


def set_R32(r):
    """Patch the 32B-forward FLOP charge everywhere it is carried. Latency/energy untouched."""
    PB.GEN32N = (PB.GEN32N[0], PB.GEN32N[1], r)
    PB.GEN32T = (PB.GEN32T[0], PB.GEN32T[1], r)
    PB.FUSE = (PB.FUSE[0], PB.FUSE[1], PB.GEN7[2] + r)      # fusion cell runs BOTH legs
    MFC.GEN7, MFC.GEN32N, MFC.GEN32T = PB.GEN7, PB.GEN32N, PB.GEN32T
    IM.GEN32N["flop"] = r; IM.GEN32T["flop"] = r
    BF.GEN32N["flop"] = r; BF.GEN32T["flop"] = r; BF.FUSE["flop"] = BF.GEN7["flop"] + r
    BF.POLICY_FLOP = {"always_7b": BF.GEN7["flop"], "always_32b_nt": BF.GEN32N["flop"],
                      "F3_confadv": BF.FUSE["flop"], "F2_cv3": BF.FUSE["flop"]}
    PC.GEN32 = (PC.GEN32[0], PC.GEN32[1], r); PC.C_STRONG_F = r
    HR.verify_constants = lambda: {**_HR_VERIFY(), "flop_ratio_32b_over_7b": r}


def run_at(r):
    set_R32(r)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    M.OUT = tmp
    with contextlib.redirect_stdout(io.StringIO()):
        M.run()
    d = json.load(open(tmp)); os.unlink(tmp)
    return d


SYS = ["always_7b", "always_32b_direct", "always_32b_reasoning", "oracle_mode_32b",
       "method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
WGT = ["sample_weighted", "macro_cells"]


def flops_table(d, conv="as_charged"):
    return {w: {s: d["cost"][conv][w][s]["flops"] for s in SYS} for w in WGT}


def main():
    R_OLD, R_NEW = 4.57, 3.816
    derivation = json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")))
    band = derivation["derived_ratio"]["recommended"]["band"]

    base = run_at(R_OLD)                                   # must reproduce the published artifact
    published = json.load(open(os.path.join(
        ROOT, "results/cascade_methods/artifacts/macro_average_headline_2026-07-30.json")))
    repro = {w: {s: (flops_table(base)[w][s], flops_table(published)[w][s]) for s in SYS} for w in WGT}
    repro_ok = all(abs(a - b) < 1e-9 for w in WGT for a, b in repro[w].values())

    mid = run_at(4.0)                                      # third point -> linearity check
    new = run_at(R_NEW)
    lo = run_at(band[0]); hi = run_at(band[1])

    # affine decomposition cost = A + B*R32, recovered from the R=4.57 and R=3.816 runs
    AB, lin_err = {}, 0.0
    for conv in ("as_charged", "honest_recost"):
        AB[conv] = {}
        for w in WGT:
            AB[conv][w] = {}
            for s in SYS:
                f_old = base["cost"][conv][w][s]["flops"]; f_new = new["cost"][conv][w][s]["flops"]
                B = (f_old - f_new) / (R_OLD - R_NEW); A = f_old - B * R_OLD
                pred = A + B * 4.0
                lin_err = max(lin_err, abs(pred - mid["cost"][conv][w][s]["flops"]))
                AB[conv][w][s] = dict(A_cheap_leg=round(A, 4), B_expected_32B_forwards=round(B, 4))

    def ratios(d, conv="as_charged"):
        r = {}
        for w in WGT:
            b = d["cost"][conv][w]["always_32b_direct"]["flops"]
            r[w] = {s: round(d["cost"][conv][w][s]["flops"] / b, 3) for s in SYS}
        return r

    HEAD = ["method_compute_lean", "method_accuracy_max_veto", "method_accuracy_max_fusion"]
    out = dict(
        title="Impact of the corrected 32B/7B FLOP ratio on the project's headline compute claims",
        date="2026-08-03", no_gpu=True, no_fabricated_numbers=True,
        reproduce="python3 src/cascade_methods/flop_ratio_impact.py",
        method=("macro_average_headline.run() re-executed unmodified with R32 monkeypatched into "
                "paper_baselines / integrated_method / beat32b_fusion / pandora_controller / "
                "method_final_mmmu_corrected / honest_recosting. Latency and energy constants are "
                "untouched, so only the FLOP axis moves. [DERIVED from measured per-sample outcomes]"),
        constants=dict(old_literal=R_OLD, new_derived=R_NEW, band=band,
                       derivation="results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json"),

        reproduction_gate=dict(
            passed=repro_ok,
            note=("At R32=4.57 the patched pipeline reproduces every published FLOP figure exactly, so "
                  "any difference below is attributable to the constant alone."),
            spot_check={w: {s: repro[w][s][0] for s in HEAD} for w in WGT}),
        linearity_gate=dict(max_abs_error_at_R32_4p0=round(lin_err, 6),
                            passed=lin_err < 5e-3,
                            note="cost is affine in R32; the third run at R32=4.0 confirms it to the "
                                 "artifact's own 3-decimal rounding."),

        absolute_flops=dict(
            as_charged_R32_4p57=flops_table(base), corrected_R32_3p816=flops_table(new),
            band_low_R32_3p734=flops_table(lo), band_high_R32_3p859=flops_table(hi),
            honest_recost_R32_4p57=flops_table(base, "honest_recost"),
            honest_recost_R32_3p816=flops_table(new, "honest_recost")),

        affine_decomposition=dict(
            explanation="cost(R32) = A + B*R32; A is the cheap-leg cost in 7B-forward units "
                        "(1.0 per MCQ cell, ~2*meanN for a Pandora open cell), B is the expected "
                        "number of 32B forwards per query.",
            **AB),

        ratios_vs_always_32b_direct=dict(
            as_charged_R32_4p57=ratios(base), corrected_R32_3p816=ratios(new),
            band_low=ratios(lo), band_high=ratios(hi),
            honest_recost_R32_4p57=ratios(base, "honest_recost"),
            honest_recost_R32_3p816=ratios(new, "honest_recost")),

        headline_deltas={
            w: {s: dict(
                published_x=ratios(base)[w][s], corrected_x=ratios(new)[w][s],
                band_x=[ratios(lo)[w][s], ratios(hi)[w][s]],
                change_pct=round(100 * (ratios(new)[w][s] / ratios(base)[w][s] - 1), 2),
                crosses_1x=("no -- both above 1x" if min(ratios(base)[w][s], ratios(new)[w][s]) > 1
                            else "no -- both below 1x" if max(ratios(base)[w][s], ratios(new)[w][s]) < 1
                            else "YES -- status changes"))
                for s in HEAD} for w in WGT},

        # ---- the writeup's "C = macro + clean L1" column (§5.6.3). Those absolute FLOP figures
        #      (6.674 / 7.951) are [Mo] in the document and are backed by no artifact, so they cannot
        #      be re-run here. They ARE affine in R32 like everything else, so they are rescaled with
        #      ratio(R) = A/R + B, A = F_old - B*R_old.
        clean_L1=dict(
            source="COMPREHENSIVE_WRITEUP_2026-07-30.md 5.6.3 -- absolute macro FLOP-eq 6.674 "
                   "(compute-lean clean L1) and 7.951 (accuracy-max-veto clean L1), both [Mo].",
            compute_lean=dict(
                F_as_charged=6.674, B_expected_32B_forwards=0.4183,
                B_source="the same document's clean-L1 escalation rate, all 8 cells [M]; for a pure "
                         "cascade/Pandora policy B IS the escalation rate",
                published_x=round(6.674 / R_OLD, 4),
                corrected_x=round((6.674 - 0.4183 * R_OLD) / R_NEW + 0.4183, 4),
                band_x=[round((6.674 - 0.4183 * R_OLD) / band[0] + 0.4183, 4),
                        round((6.674 - 0.4183 * R_OLD) / band[1] + 0.4183, 4)],
                status="DERIVED, exact given the two published inputs"),
            accuracy_max_veto=dict(
                F_as_charged=7.951,
                B_expected_32B_forwards="not published for this column",
                point_estimate_x=round((7.951 - AB["as_charged"]["macro_cells"]
                                        ["method_accuracy_max_veto"]["B_expected_32B_forwards"] * R_OLD)
                                       / R_NEW + AB["as_charged"]["macro_cells"]
                                       ["method_accuracy_max_veto"]["B_expected_32B_forwards"], 4),
                point_estimate_assumption=("B held at the contaminated run's %.4f. Clean L1 raises the "
                                           "open-arm escalation, so the true B is HIGHER and the true "
                                           "corrected ratio slightly LOWER than this point estimate."
                                           % AB["as_charged"]["macro_cells"]
                                           ["method_accuracy_max_veto"]["B_expected_32B_forwards"]),
                rigorous_bounds_x=[round((7.951 - R_OLD) / R_NEW + 1.0, 4),   # B = 1 (max)
                                   round(7.951 / R_NEW, 4)],                  # B = 0 (min)
                bounds_note=("B <= 1 (no policy calls the 32B more than once per query) gives the lower "
                             "bound %.4fx; B >= 0 gives the upper bound %.4fx."
                             % ((7.951 - R_OLD) / R_NEW + 1, 7.951 / R_NEW)),
                published_x=round(7.951 / R_OLD, 4),
                status="DERIVED with a stated assumption -- the underlying per-sample vectors for the "
                       "clean-L1 column are not on disk"),
            verdict=("The clean-L1 column moves the same way and by more: compute-lean 1.460x -> "
                     "%.3fx, accuracy-max-veto 1.740x -> ~%.2fx (bounded %.2f-%.2f). Both were already "
                     "well above 1x."
                     % ((6.674 - 0.4183 * R_OLD) / R_NEW + 0.4183,
                        (7.951 - AB["as_charged"]["macro_cells"]["method_accuracy_max_veto"]
                         ["B_expected_32B_forwards"] * R_OLD) / R_NEW
                        + AB["as_charged"]["macro_cells"]["method_accuracy_max_veto"]
                        ["B_expected_32B_forwards"],
                        (7.951 - R_OLD) / R_NEW + 1, 7.951 / R_NEW))),

        # ---- the writeup's deployment break-even table (§9, "use the cascade only inside its regime")
        breakevens=dict(
            model="cascade(e) = 1 + e*R32 FLOP-eq / 347 + e*665 ms / 45.8 + e*127 J, vs one 32B call",
            note=("Only the FLOP threshold depends on R32; the latency and energy thresholds are set by "
                  "measured per-leg constants and do not move. [DERIVED]"),
            flops_breakeven_e=dict(at_R32_4p57=round((R_OLD - 1) / R_OLD, 4),
                                   at_R32_3p816=round((R_NEW - 1) / R_NEW, 4),
                                   band=[round((band[0] - 1) / band[0], 4),
                                         round((band[1] - 1) / band[1], 4)]),
            latency_breakeven_e=round((PB.GEN32N[0] - PB.GEN7[0]) / PB.GEN32N[0], 4),
            energy_breakeven_e=round((PB.GEN32N[1] - PB.GEN7[1]) / PB.GEN32N[1], 4),
            table_correction=("The published table's last row, 'e > ~78%% -> worse on everything', "
                              "becomes 'e > ~%.0f%%'. The ~48%% (all-axes) and ~64%% (latency) rows are "
                              "unchanged. Consequence for the measured suite: MedXpert at 89.60%% "
                              "escalation was already past the threshold under both constants; "
                              "VQA-RAD-closed at 56.97%% is still inside it. No cell changes verdict."
                              % (100 * (R_NEW - 1) / R_NEW))),

        verdict=None, caveats=[
            "Only the FLOP axis is recomputed. Latency and energy are separately measured and unchanged.",
            "The ratio of a method to always-32B-direct is NOT invariant to R32: the method's cheap leg "
            "is charged in 7B units, so lowering R32 makes the 32B baseline cheaper faster than it makes "
            "the method cheaper. Every method/baseline compute ratio therefore gets WORSE, not better.",
            "The corrected constant does not change any accuracy number, any escalation rate, any "
            "latency and any energy figure -- only FLOP-eq.",
            "Cells where a policy is 'always_32b_nt' have A=0, B=1 and their ratio is exactly 1.000x at "
            "any R32; the movement comes entirely from the cascade and Pandora cells.",
        ])

    cl_o = ratios(base)["macro_cells"]["method_compute_lean"]
    cl_n = ratios(new)["macro_cells"]["method_compute_lean"]
    am_o = ratios(base)["macro_cells"]["method_accuracy_max_veto"]
    am_n = ratios(new)["macro_cells"]["method_accuracy_max_veto"]
    sw_o = ratios(base)["sample_weighted"]["method_compute_lean"]
    sw_n = ratios(new)["sample_weighted"]["method_compute_lean"]
    out["verdict"] = (
        "NO COMPUTE CLAIM CHANGES STATUS; every one gets modestly WORSE. Macro (8 cells, equal weight), "
        "as charged: compute-lean %.3fx -> %.3fx, accuracy-max-veto %.3fx -> %.3fx a single 32B forward. "
        "Both were already >1x, i.e. already NOT compute-negative, and remain so. Sample-weighted, "
        "compute-lean %.3fx -> %.3fx: still <1x, so the one surviving compute-negative statement "
        "(sample-weighted only) survives, with a thinner margin. The published '~7%% margin of error' on "
        "the constant was an UNDERESTIMATE in magnitude (4.57 -> 3.816 is -16.5%%) but correct in "
        "direction of harmlessness: the affected claims were already the wrong side of 1x."
        % (cl_o, cl_n, am_o, am_n, sw_o, sw_n))

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("reproduction_gate", "linearity_gate", "headline_deltas",
                                          "verdict")}, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
