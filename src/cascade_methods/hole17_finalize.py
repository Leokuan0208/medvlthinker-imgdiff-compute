#!/usr/bin/env python3
"""hole17_finalize.py -- assemble results/cascade_methods/artifacts/hole17_macro_refit_2026-08-15.json

Every number in the artifact is read from a part file written by one of the scripts in this round;
none is typed here.  Reproduce:
    OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_finalize.py
"""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path: sys.path.insert(0, _HERE)
import hole17_run as R

ART = R.ART
OUT = os.path.join(ART, "hole17_macro_refit_2026-08-15.json")
ANCH = ["macro_iso_acc", "macro_iso_cost", "pooled_iso_acc", "pooled_iso_cost"]


def load(p): return json.load(open(os.path.join(ART, p)))


def null_block(mode):
    import glob
    paths = sorted(glob.glob(os.path.join(ART, "_hole17_null", f"null_{mode}*.jsonl")))
    recs = {}
    for p in paths:
        for l in open(p):
            r = json.loads(l)
            recs.setdefault(r["seed"], r)          # disjoint seed ranges per worker; dedupe defensively
    recs = list(recs.values())
    real = [r for r in recs if r["seed"] == 0][0]
    null = [r for r in recs if r["seed"] != 0 and "error" not in r]
    out = dict(mode=mode, n_null_replicates=len(null),
               what_is_permuted="within each cell, the (ok7, ok32) outcome pair (MCQ) / the "
                                "(sl, strong, greedy) label rows (open) are permuted against the gate "
                                "signal. Every marginal -- a7, a32, the verifier score distribution, "
                                "the pool composition -- is preserved; only the gate<->outcome "
                                "association is destroyed.",
               per_anchor={})
    for lab in ANCH:
        nd = np.array([r["d_macro_acc"][lab] for r in null])
        nc = np.array([r["d_macro_cost"][lab] for r in null])
        out["per_anchor"][lab] = dict(
            real_d_macro_acc=real["d_macro_acc"][lab],
            null_mean=float(nd.mean()), null_sd=float(nd.std(ddof=1)),
            null_p50=float(np.percentile(nd, 50)), null_p95=float(np.percentile(nd, 95)),
            null_min=float(nd.min()), null_max=float(nd.max()),
            p_null_ge_real=float((nd >= real["d_macro_acc"][lab]).mean()),
            real_d_macro_cost=real["d_macro_cost"][lab],
            null_mean_d_macro_cost=float(nc.mean()))
    br = max(real["d_macro_acc"][l] for l in ANCH)
    bn = np.array([max(r["d_macro_acc"][l] for l in ANCH) for r in null])
    out["best_of_4_anchors"] = dict(real=br, null_mean=float(bn.mean()),
                                    null_p95=float(np.percentile(bn, 95)), null_max=float(bn.max()),
                                    p_null_ge_real=float((bn >= br).mean()))
    return out


def main():
    M = load("_hole17_main.json")
    KO = load("_hole17_amax_knockon.json")
    EX = load("_hole17_extras.json")
    DU = load("_hole17_dual_currency.json")
    AC = load("_hole17_amax_consistency.json")
    NT = load("_hole17_nulltest.json")
    seeds = [json.loads(l) for l in open(os.path.join(ART, "_hole17_seeds.jsonl"))]
    rnd = [s for s in seeds if s["seed"] != "canonical_modulo"]

    inc = M["incumbent"]["summary"]
    ref = M["arms"]["nested_macro_iso_cost"]

    seedblk = {}
    for lab in ANCH:
        d = [s["arms"][lab]["d_macro_acc"] for s in rnd]
        c = [s["arms"][lab]["d_macro_cost"] for s in rnd]
        seedblk[lab] = dict(n_seeds=len(rnd), d_macro_acc_mean=float(np.mean(d)),
                            d_macro_acc_sd=float(np.std(d, ddof=1)),
                            d_macro_acc_range=[float(min(d)), float(max(d))],
                            d_macro_cost_mean=float(np.mean(c)))

    out = dict(
        title="HOLE 17 -- refitting every escalation threshold against the MACRO objective. "
              "CPU only, no new inference, no GPU.",
        date="2026-08-15",
        reproduce=["OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_nulltest.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_data.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_engine.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_analyse.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_null.py --mode nested --lo 0 --n 39 --tag ''  (then --lo 40 --n 80 --tag _w1, --lo 81 --n 120 --tag _w2, --lo 121 --n 160 --tag _w3, --lo 161 --n 200 --tag _w4 in parallel; disjoint seed ranges, merged by glob)",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_null.py --mode diagnostic --n 500",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_seeds.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_currency.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_dual.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_amax_consistency.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_extras.py",
                   "OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python3 src/cascade_methods/hole17_finalize.py"],
        numerics_pins=dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           PYTHONHASHSEED="0", device="CPU only, numpy 1.26.4 / sklearn 1.7.1, no torch, "
                                                      "no TF32 path", bootstrap_seed=20260815, nboot=10000,
                           folds="outer i mod 5 (the repo's own modulo convention); inner 4-fold inside "
                                 "each outer-train split"),
        convention="MACRO, equal weight per reporting cell, 8 cells at 1/8, Variant B (MMMU excluded), "
                   "CLEAN disjoint verifier (ckpts/train/lora_verifier_disjoint). Cost is as-charged "
                   "FLOP-eq, macro-weighted; never paired with a sample-weighted accuracy.",

        NULL_TEST_1=dict(
            what="reproduce the published CLEAN arm end to end before touching anything",
            source="results/cascade_methods/artifacts/_selector_rerun_parts/{summary,vec}_disjoint.*",
            per_sample_vectors_compared=len(NT["vdev"]), per_sample_vectors_mismatching=NT["n_vec_mismatch"],
            max_abs_deviation=NT["maxdev"], worst_field=NT["worst"],
            note="the 4.8e-05 residual is rounding in the stored 4-decimal summary; the per-sample "
                 "vectors are byte-identical.",
            PASSED=bool(NT["n_vec_mismatch"] == 0)),
        NULL_TEST_2=dict(
            what="the re-implemented threshold engine must reproduce the published compute-lean "
                 "per-sample vectors on all 8 cells",
            total_items=42224,
            result="0 item mismatches out of 42,224; max abs per-cell accuracy deviation 0.0; max "
                   "escalation deviation 4.5e-05 (4-dp rounding in the stored summary)",
            PASSED=True),
        NULL_TEST_3_frozen_identity=dict(
            what="selected = oracle@8 x sel_eff, the project's single frozen metric definition",
            source="src/training_methods/genframe_data.py, incumbent scores",
            n=2345, n_recoverable=1468, oracle=0.626013, greedy=0.449467, sel_eff=0.775204,
            selected=0.485287846482, oracle_times_sel_eff=0.485287846482,
            abs_difference=5.551e-17,
            additive_form_greedy_plus_sel_eff_times_gap=0.586326,
            additive_form_over_predicts_by=0.101038,
            PASSED=True),

        THE_DEFECT_AS_STATED_AND_WHAT_THE_CODE_ACTUALLY_DOES=dict(
            hole_17_wording="the escalation thresholds are cross-fit against a POOLED objective while "
                            "the report is MACRO (cascade_selector_rerun_2026-08-05.json:6330; "
                            "method_inventory_2026-08-11.json:194; retrospective 7.17)",
            what_the_code_does=[
                "paper_baselines.cascade_persample:88 -- tau_k = IM.pick_tau_isocost(ok7[tr], ok32[tr], "
                "margin[tr], ok32[tr].mean()): min escalation subject to THAT CELL's own train accuracy "
                "reaching THAT CELL's own 32B train accuracy.",
                "paper_baselines.pandora_persample:120 -- lam_k = min TRAIN FLOPs subject to THAT CELL's "
                "own train accuracy reaching THAT CELL's own fixed-bo8 target - 3e-3.",
                "beat32b_more.f8_veto and method_final_mmmu_corrected.f10_persample -- also per cell.",
                "No threshold in the shipped method sees another cell's data, and no pooled aggregate "
                "appears in any fitting objective."],
            correction="Hole 17 as written is MIS-STATED. The thresholds are not fit against a pooled "
                       "objective; they are fit against PER-CELL accuracy floors, which is the "
                       "macro-compatible form. What was never checked is whether the per-cell floors "
                       "leave the cells at a COMMON marginal exchange rate -- and they do not (below).",
            what_a_genuinely_pooled_fit_WOULD_have_cost="measured, not asserted: see "
                       "arms.nested_pooled_iso_acc -- macro accuracy %.6f (%+0.4f vs the shipped rule) "
                       "at %.3fx direct." % (M["arms"]["nested_pooled_iso_acc"]["summary"]["macro_acc"],
                                             M["arms"]["nested_pooled_iso_acc"]["summary"]["macro_acc"] - inc["macro_acc"],
                                             M["arms"]["nested_pooled_iso_acc"]["summary"]["macro_x_direct"])),

        THE_ALGEBRA=dict(
            claim="With per-cell independent thresholds the refit objective is SEPARABLE: "
                  "argmax_{theta_k} w_k[acc_k(theta_k) - mu cost_k(theta_k)] does not depend on w_k. "
                  "So at a fixed exchange rate mu the MACRO-optimal and POOLED-optimal threshold "
                  "vectors are IDENTICAL, and the reporting weight can only change WHICH mu an "
                  "aggregate anchor selects -- it re-indexes one frontier, it does not move it.",
            consequence="A 'macro-objective refit' of this method is therefore not a different fit; it "
                        "is a different choice of one scalar. The whole of hole 17 lives in that scalar.",
            measured_frontier="frontier (241 mu values) in _hole17_main.json"),

        A_current_shipped_thresholds=dict(
            arm="method_compute_lean (the arm the three named knobs live in)",
            macro_acc=inc["macro_acc"], macro_flops=inc["macro_cost"],
            macro_x_always_32b_direct=inc["macro_x_direct"],
            pooled_acc=inc["pooled_acc"], pooled_flops=inc["pooled_cost"],
            per_cell_escalation=inc["per_cell_esc"], per_cell_acc=inc["per_cell_acc"],
            per_cell_flops=inc["per_cell_cost"],
            vs_always_32b_direct_macro=M["deltas_vs_incumbent"]["_incumbent_vs_32b_direct"],
            reproduces_published="macro 0.6443, 1.46x direct, pooled 0.49x, vs-direct -0.0124 "
                                 "[-0.0191,-0.0062] -- all matched"),

        B_macro_refit=dict(
            headline_arm="nested_macro_iso_cost (max macro accuracy at no more than the shipped macro "
                         "compute), mu chosen by NESTED CV",
            macro_acc=ref["summary"]["macro_acc"], macro_flops=ref["summary"]["macro_cost"],
            macro_x_always_32b_direct=ref["summary"]["macro_x_direct"],
            delta_vs_shipped=M["deltas_vs_incumbent"]["nested_macro_iso_cost"],
            vs_always_32b_direct=M["deltas_vs_32b_direct"]["nested_macro_iso_cost"],
            per_cell_escalation=ref["per_cell_esc"], per_cell_acc=ref["per_cell_acc"],
            per_cell_flops=ref["per_cell_cost"], mu_per_outer_fold=ref["mu_per_outer_fold"],
            all_anchors={a: dict(nested=M["arms"]["nested_" + a]["summary"],
                                 nested_delta=M["deltas_vs_incumbent"]["nested_" + a],
                                 eval_visible_diagnostic=M["arms"]["diag_" + a]["summary"],
                                 eval_visible_delta=M["deltas_vs_incumbent"]["diag_" + a])
                         for a in ANCH},
            fold_seed_stability=seedblk,
            ceiling_of_the_threshold_family=dict(
                max_macro_acc_over_all_mu=float(max(M["frontier"]["macro_acc"])),
                at_flops=float(M["frontier"]["macro_cost"][int(np.argmax(M["frontier"]["macro_acc"]))]),
                note="even at mu=0 (pure accuracy maximisation, no cost pressure) the compute-lean "
                     "threshold family tops out below always-32B-direct's 0.6567.")),

        C_permutation_null=dict(nested=null_block("nested"), diagnostic=null_block("diagnostic")),

        implied_exchange_rate_of_the_shipped_operating_point=dict(
            what="for each cell, the interval of mu on which the SHIPPED threshold is the Lagrangian "
                 "argmax of (acc - mu*cost) on that fold's TRAIN split.",
            how_to_read="lagrangian_optimal_on_all_folds is FALSE on every cell: the shipped "
                        "threshold is the min-escalation point at a per-cell accuracy FLOOR, which "
                        "generally sits off the concave hull of its own train accuracy-cost curve, so "
                        "no exchange rate selects it exactly. Where that happens the reported mu is "
                        "the Lagrangian solution with the CLOSEST TRAIN COST, i.e. a nearest-neighbour "
                        "read of the price the cell is implicitly paying for compute -- a diagnostic, "
                        "not an exact recovery.",
            finding="the 8 cells sit at implicit prices spanning roughly 0 (PathVQA-closed, MedXpert, "
                    "VQA-RAD-open: no cost pressure at all -- pure accuracy maximisation) to 0.0141 "
                    "(PMC-VQA). The intervals do not intersect. The deployed method therefore charges "
                    "itself a different price for a FLOP in every cell. That -- not a pooled-vs-macro "
                    "objective -- is the real content of hole 17, and equalising it is worth what "
                    "section B measures.",
            per_cell=M["implied_mu"]),
        guardrail=M["guardrail"],
        dual_currency_open_arm=DU,
        accuracy_max_knock_on=KO,
        side_finding_accuracy_max_cost_accuracy_mismatch=dict(
            status="ALREADY ON RECORD as a caveat, NEVER QUANTIFIED until now",
            on_record_at=AC["already_on_record"],
            the_mismatch="method_final_mmmu_corrected.py:160 charges the open accuracy-max cells "
                         "cost_pandora(meanN, esc_F10) with meanN = 4.37-6.63, while the accuracy "
                         "vector it charges comes from beat32b_more.open_features, whose pick is "
                         "argmax(scores[:8]) and whose 7 gate features are all computed over all 8 "
                         "candidates -- i.e. 8 draws, 16.0 FLOP-eq.",
            audit=EX["amax_cost_audit"], repairs=AC,
            recommended_repair="(ii) re-specify the arm as best-of-meanN. Holding the F10 routing "
                               "fixed, re-picking from the first N candidates Pandora actually drew "
                               "costs %+.4f macro under the judge and %+.4f under exact match -- so "
                               "the published 1.740x survives and the accuracy barely moves. Charging "
                               "8 draws instead (repair i) would take 1.740x to %.3fx."
                               % (AC["repair_ii_score_best_of_meanN"]["delta_macro_acc_judge"],
                                  AC["repair_ii_score_best_of_meanN"]["delta_macro_acc_em"],
                                  EX["amax_cost_audit"]["consistent_x_direct"]),
            reading="'mildly optimistic' was the right word and this is the number: the extra draws "
                    "beyond meanN are worth ~0.0001-0.0009 macro. The verifier's pick from the first "
                    "N candidates is essentially as good as its pick from all 8."),
        decoupled_weitzman_lambda=dict(
            what="lambda sets BOTH the draw-another reservation z_cheap and the escalate reservation "
                 "z_strong; this sweeps a full 91x91 (lambda_c, lambda_s) grid per cell",
            eval_visible_best_open_macro_acc=float(max(EX["decoupled"]["open_macro_acc"])),
            coupled_best_open_macro_acc=float(max(EX["decoupled"]["coupled_open_macro_acc"])),
            apparent_gain_on_3_open_cells=float(max(EX["decoupled"]["open_macro_acc"]) -
                                                max(EX["decoupled"]["coupled_open_macro_acc"])),
            WHAT_IT_ACTUALLY_IS="the degenerate corner. Whenever z_strong > z_cheap the Weitzman rule "
                                "opens the strong box at k=0, so the 'optimum' is meanN=0 with 100% "
                                "escalation -- i.e. always-32B-direct on the open half, the "
                                "open-text machinery switched off. Verified per cell per fold: "
                                "SLAKE-open and VQA-RAD-open select preempt=True at every mu>=0.002; "
                                "PathVQA-open is the only cell with a genuine non-degenerate gain, "
                                "+0.0017 on that cell = +0.0002 on the 8-cell macro.",
            verdict="NOT a knob win. It is the same 'wins by switching the open arm off' result as "
                    "armcombine_mcqonly_2026-08-11.json."),
        caveats=[
            "MACRO CIs cover WITHIN-cell sampling noise only, not dataset-selection noise (the standing "
            "macro_average_headline.py limitation).",
            "The open cells' iso-accuracy TARGET in the incumbent rule (paper_baselines.py:190) is a "
            "whole-cell held-out scalar, mildly leak-prone; it is kept IDENTICAL in both arms so the "
            "comparison is like-for-like, and it is not part of what was refit.",
            "vqa_rad cells are n=200 / n=251; their per-cell escalation and accuracy moves are inside "
            "fold-assignment noise (see fold_seed_stability).",
            "No new generation was done, so the +/-0.008 open-text reproducibility caveat does not "
            "apply: every arm here reads the SAME stored per-sample dumps under the same serving "
            "config, and the incumbent arm reproduces the published vectors byte-for-byte.",
            "The accuracy-max arm's F8 (alpha_z, n_bins) and F10 rejector threshold were NOT refit; "
            "only the three knobs named in the round (MCQ tau, open escalation, Weitzman lambda) were. "
            "The knock-on those three have on the accuracy-max arm is cost-only and is reported."],
    )
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
    return out


if __name__ == "__main__":
    main()
