#!/usr/bin/env python3
"""weitzman_T04_report.py -- console reader for artifacts/weitzman_T04_2026-08-15.json.
No new numbers are computed here; every figure is printed from the artifact."""
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
P = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(ROOT, "results/cascade_methods/artifacts/weitzman_T04_2026-08-15.json")
O = json.load(open(P))
CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

print("=" * 126)
print(O["title"])
print("=" * 126)
print("NULL TESTS  passed =", O["null_test_passed"])
for k, v in O["null_test_max_abs_deviation"].items():
    print(f"    {k:22s} max abs deviation {v:.3e}")

print("\nPOOLS")
print(f"  {'tag':10s}{'tok':>7s}{'orc8_j':>9s}{'orc8_em':>9s}{'rand_j':>9s}{'bo8_j':>9s}{'bo8_em':>9s}")
for t, p in O["pools"].items():
    print(f"  {t:10s}{p['mean_gen_tokens_all_slots']:7.3f}{p['oracle8_judge']:9.4f}"
          f"{p['oracle8_em']:9.4f}{p['random_slot_judge']:9.4f}{p['bo8_selected_judge']:9.4f}"
          f"{p['bo8_selected_em']:9.4f}")

print("\nARMS -- open-3 macro (equal weight per open cell)")
print(f"  {'arm':48s}{'acc_j':>9s}{'acc_em':>9s}{'meanN':>8s}{'esc%':>8s}{'FLOPeq':>8s}"
      f"{'latseq':>9s}{'latpar':>9s}")
for k in sorted(O["ARMS"]):
    b = O["ARMS"][k]["open3_macro"]
    print(f"  {k:48s}{b['acc_judge']:9.5f}{b['acc_em']:9.5f}{b['meanN']:8.3f}{b['esc']*100:8.2f}"
          f"{b['flops_eq']:8.3f}{b['lat_seq_ms']:9.0f}{b['lat_par_ms']:9.0f}")

print("\nARMS -- per cell (objective O1 = the deployed iso-accuracy objective, resub selection)")
for arm in ("A_deployed_T07r", "B_stale_on_T04", "B2_lambdaStale_recalibrated_T04", "C_refit_T04"):
    print(f"  {arm}")
    for c in CELLS:
        d = O["ARMS"][f"O1|resub|{arm}"]["per_cell"][c]
        ss = d["seed_spread_judge"]["across_cv_seeds"]
        print(f"    {c:14s} n={d['n']:5d} acc_j={d['acc_judge']:.5f} acc_em={d['acc_em']:.5f} "
              f"meanN={d['meanN']:5.3f} esc={d['esc']*100:5.2f}% F={d['flops_eq']:6.3f} "
              f"lam[{d['lambda_min']:.5f},{d['lambda_max']:.5f}] med={d['lambda_median']:.5f} "
              f"| cvseed sd={ss['sd']:.5f} range=[{ss['min']:.5f},{ss['max']:.5f}] "
              f"| bar_j={d['bar_judge']:.4f}")

print("\nCONTRASTS -- open-3 macro (paired item bootstrap, nboot=%d)" % O["nboot"])
for k in sorted(O["ARM_CONTRASTS"]):
    j = O["ARM_CONTRASTS"][k]["open3_macro_judge"]; e = O["ARM_CONTRASTS"][k]["open3_macro_em"]
    print(f"  {k:56s}")
    print(f"      judge {j['delta']:+.5f} [{j['lo']:+.5f},{j['hi']:+.5f}] {j['verdict']:5s} "
          f"(macro-8 scaled {j['macro8_scaled_delta']:+.5f})")
    print(f"      em    {e['delta']:+.5f} [{e['lo']:+.5f},{e['hi']:+.5f}] {e['verdict']:5s} "
          f"| dFLOP-eq {O['ARM_CONTRASTS'][k]['open3_macro_d_flops_eq']:+.3f}")

fr = O["FRONTIER"]
print("\nFRONTIER -- open-3 macro at T=0.4 (cross-fit, nothing selected), thinned")
print(f"  {'lambda':>10s}{'meanN':>8s}{'esc%':>8s}{'FLOPeq':>8s}{'acc_j':>9s}{'acc_em':>9s}"
      f"{'latseq':>9s}{'latpar':>9s}")
rows = fr["open3_macro_T04"]
for r in rows[::12] + [rows[-1]]:
    print(f"  {r['lam']:10.5f}{r['meanN']:8.3f}{r['esc']*100:8.2f}{r['flops_eq']:8.3f}"
          f"{r['acc_judge']:9.5f}{r['acc_em']:9.5f}{r['lat_seq_ms']:9.0f}{r['lat_par_ms']:9.0f}")

op = O["OPERATING_POINTS"]
print(f"\nTHE BAR: always-32B-direct open-3 macro judge {op['open3_bar']['judge']:.5f} / "
      f"em {op['open3_bar']['em']:.5f} at 4.570 FLOP-eq")
print("MIN-COMPUTE POINT THAT REACHES THE BAR (open-3 macro)")
for k, v in op["selected"].items():
    if v is None:
        print(f"  {k:26s} NONE -- no point on the swept curve reaches the bar")
    else:
        knob = f"lam={v['lam']:.5f}" if "lam" in v else f"N={v['N']} tau={v['tau']:.2f}"
        print(f"  {k:26s} {knob:22s} acc_j={v['acc_judge']:.5f} acc_em={v['acc_em']:.5f} "
              f"F={v['flops_eq']:6.3f} ({v['flops_eq']/4.57:5.3f}x direct) "
              f"latseq={v['lat_seq_ms']:6.0f} latpar={v['lat_par_ms']:6.0f}")

print("\nPER-CELL MIN-COMPUTE MARKS AT THAT CELL'S OWN BAR (T=0.4)")
for c in CELLS:
    m = op["per_cell_marks_T04"][c]
    for lab in ("min_flops_at_bar_judge", "min_flops_at_bar_em"):
        v = m[lab]
        print(f"  {c:14s} {lab:24s} " + ("NONE" if v is None else
              f"lam={v['lam']:.5f} acc_j={v['acc_judge']:.5f} acc_em={v['acc_em']:.5f} "
              f"meanN={v['meanN']:.2f} esc={v['esc']*100:.1f}% F={v['flops_eq']:.3f}"))

print("\nMARKED POINT vs always-32B-direct (paired bootstrap)")
for k, v in op["marked_point_CIs_vs_always32b_direct"].items():
    if v is None:
        print(f"  {k:26s} NONE"); continue
    print(f"  {k:26s} open3 {v['open3_macro_delta_vs_bar']:+.5f} [{v['lo']:+.5f},{v['hi']:+.5f}] "
          f"{v['verdict']:5s}  macro-8 scaled {v['macro8_scaled_delta']:+.5f}")
    for c in CELLS:
        b = v["per_cell"][c]
        print(f"        {c:14s} {b['delta']:+.5f} [{b['lo']:+.5f},{b['hi']:+.5f}] {b['verdict']}")

print("\nADAPTIVE vs FIXED-N at equal or lower FLOP-eq (T=0.4)")
s = op["adaptive_vs_fixedN_T04"]
print(f"  Pareto points compared: {s['n_points']}   adaptive strictly better at "
      f"{s['n_where_adaptive_wins']}   median advantage {s['median_advantage']}   "
      f"max {s['max_advantage']}   min {s['min_advantage']}")
print(f"  {'lam':>9s}{'FLOPeq':>8s}{'adap_acc':>10s}{'fixed_acc':>10s}{'adv':>9s}   "
      f"{'fixed cfg':16s}{'adap latseq':>12s}{'fix latpar':>11s}")
for d in s["points"]:
    if d["best_fixed_acc_judge"] is None:
        continue
    print(f"  {d['lam']:9.5f}{d['flops_eq']:8.3f}{d['adaptive_acc_judge']:10.5f}"
          f"{d['best_fixed_acc_judge']:10.5f}{d['adaptive_advantage']:+9.5f}   "
          f"{d['best_fixed_config']:16s}{d['adaptive_lat_seq_ms']:12.0f}"
          f"{d['best_fixed_lat_par_ms']:11.0f}")

print("\nFIXED-N PARETO at T=0.4 (open-3 macro)")
print(f"  {'N':>3s}{'tau':>7s}{'esc%':>8s}{'FLOPeq':>8s}{'acc_j':>9s}{'acc_em':>9s}{'latpar':>9s}")
for r in O["FIXED_N"]["pareto_open3_macro_T04"]:
    print(f"  {r['N']:3d}{r['tau']:7.2f}{r['esc']*100:8.2f}{r['flops_eq']:8.3f}"
          f"{r['acc_judge']:9.5f}{r['acc_em']:9.5f}{r['lat_par_ms']:9.0f}")

print("\nMACRO-8 CONSEQUENCE")
for half in ("compute_lean", "accuracy_max"):
    print(f"  [{half} multiple-choice half; always-32B-direct 0.6567 @ 4.570; "
          f"shipped {O['OPERATING_POINTS']['MACRO_references']['shipped'][half]}]")
    for k, v in op["MACRO"][half].items():
        if v is None:
            print(f"    {k:44s} NONE"); continue
        print(f"    {k:44s} acc={v['macro_acc']:.5f} ({v['vs_always32b_direct_acc']:+.5f} vs bar) "
              f"F={v['macro_flops_eq']:6.3f} ({v['compute_x_vs_always32b_direct']:5.3f}x bar, "
              f"{v['compute_x_vs_shipped']:5.3f}x shipped)")

print("\nGUARDRAIL")
for k, blk in op["GUARDRAIL"].items():
    print(f"  {k}: flags vs T07r control {blk['FLAGS_vs_T07r_control']} | "
          f"flags vs always-32B-direct {blk['FLAGS_vs_always32b_direct']}")
    for c in CELLS:
        d = blk[c]
        print(f"      {c:14s} n={d['n']:5d} vs control judge {d['vs_T07r_control_judge']['delta']:+.5f}"
              f" [{d['vs_T07r_control_judge']['lo']:+.5f},{d['vs_T07r_control_judge']['hi']:+.5f}]"
              f" {d['vs_T07r_control_judge']['verdict']:5s} | vs bar judge "
              f"{d['vs_always32b_direct_judge']['delta']:+.5f} "
              f"[{d['vs_always32b_direct_judge']['lo']:+.5f},"
              f"{d['vs_always32b_direct_judge']['hi']:+.5f}] "
              f"{d['vs_always32b_direct_judge']['verdict']}")

print("\nPERMUTATION NULL")
pn = O["PERMUTATION_NULL"]
for k in ("S1_did_the_null_ever_reach_the_bar", "S2_min_flops_at_the_bar",
          "S3_best_accuracy_on_the_frontier", "S4_refit_minus_stale",
          "S5_T04_best_minus_T07r_best"):
    print(f"  {k}")
    print("   ", json.dumps({kk: vv for kk, vv in pn[k].items() if kk not in ("what", "read")},
                            default=float)[:600])
