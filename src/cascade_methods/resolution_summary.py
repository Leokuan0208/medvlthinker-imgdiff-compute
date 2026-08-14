#!/usr/bin/env python3
"""resolution_summary.py -- compact console readout of resolution_sweep_2026-08-13.json.

Prints only numbers that exist in the artifact; anything absent prints as "not measured".

    python3 src/cascade_methods/resolution_summary.py
"""
import json
import os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
A = os.path.join(ROOT, "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json")


def g(d, *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def main():
    d = json.load(open(A))
    print("=" * 88)
    print("NULL TESTS")
    nt = d.get("null_tests", {})
    print(f"  N1 frozen metric            max|dev| = {g(nt,'N1_frozen_metric','max_abs_deviation')}"
          f"   {g(nt,'N1_frozen_metric','verdict')}")
    for arm in ("lingshu7b", "lingshu32b"):
        print(f"  N2 published MCQ ({arm:10s}) max|dev| = "
              f"{g(nt,'N2_published_mcq_cells',arm,'max_abs_deviation')}   "
              f"{g(nt,'N2_published_mcq_cells',arm,'verdict')}")
    print(f"  N3 generation path          pooled answer agreement with the stored deployed "
          f"greedy dump = {g(nt,'N3_generation_path','pooled_agreement_rate')}")
    n4 = nt.get("N4_verifier_rescore")
    if isinstance(n4, dict):
        print(f"  N4 verifier batch1 vs stored max|dev| = "
              f"{g(n4,'batch1_vs_stored','max_abs_dev')}  "
              f"(n={g(n4,'batch1_vs_stored','n')})")
        if "batchN_vs_batch1" in n4:
            print(f"     verifier batch{g(n4,'batchN_vs_batch1','batch')} vs batch1  max|dev| = "
                  f"{g(n4,'batchN_vs_batch1','max_abs_dev')}")
    else:
        print(f"  N4 verifier re-score        {n4}")
    print(f"  N5 'greedy' in the open pool is modal-of-8 "
          f"{g(nt,'N5_what_greedy_means_in_the_published_open_pool','pooled','published_row_called_greedy_is_modal_of_8')}"
          f" vs true T=0 decode "
          f"{g(nt,'N5_what_greedy_means_in_the_published_open_pool','pooled','true_temperature_0_decode')}")

    print("=" * 88)
    print("MCQ HALF -- MedEvalKit track, full n, cap320 (250,880) vs the default (12,845,056)")
    for cell, v in d.get("mcq_cap320_vs_default_full_n_existing_dumps", {}).items():
        if isinstance(v, dict) and "delta_default_minus_cap320" in v:
            print(f"  {cell:16s} n={v['n_cell']:6d}  cap320 {v['acc_cap320_250880']:.6f}  "
                  f"default {v['acc_default_12845056']:.6f}  "
                  f"delta {v['delta_default_minus_cap320']:+.6f} "
                  f"{v['ci95']} {'SIG' if v['significant'] else 'n.s.'}")
        elif isinstance(v, dict) and "macro8_delta_default_minus_cap320" in v:
            print(f"  MACRO-8 cost of the cut, THESE 3 CELLS ONLY (weight 1/8 each): "
                  f"{v['macro8_delta_default_minus_cap320']:+.6f} {v['ci95']} "
                  f"{'SIG' if v['significant'] else 'n.s.'}   "
                  f"(project significance threshold {v['project_significance_threshold_on_macro8']})")
        elif isinstance(v, dict):
            print(f"  {cell:16s} {v.get('status')}")
    print("  headroom above the default:")
    for cell, v in g(d, "facts_image_geometry", "mcq_cells", default={}).items():
        if isinstance(v, dict) and "by_cap" in v:
            print(f"    {cell:16s} images above the 12,845,056 cap = "
                  f"{v['by_cap']['medevalkit_default']['frac_images_above_cap']}"
                  f"   mean vision tokens cap320/default = "
                  f"{v['by_cap']['cap320']['mean_vision_tokens']:.1f}/"
                  f"{v['by_cap']['medevalkit_default']['mean_vision_tokens']:.1f}")

    print("=" * 88)
    print("OPEN HALF -- generator resolution, judge-labelled endpoint (n=2,345)")
    orr = d.get("open_generator_resolution", {})
    if not orr:
        ls = d.get("labelling_stage_status", {})
        print("  NOT AVAILABLE -- the verifier-scoring / judge stage did not complete on this "
              "session's shared cards.")
        print(f"    verifier_score_cache_exists={ls.get('verifier_score_cache_exists')} "
              f"judge_cache_entries={ls.get('judge_cache_entries')}")
        print("    the open half is reported on the exact-match secondary endpoint below.")
    for cap, v in orr.get("by_cap", {}).items():
        ms = v["mean_sd"]
        print(f"  {cap:8s} px={v['max_pixels']:>9d} seeds={v['seeds']}")
        print(f"           greedy_t0 {g(v,'greedy_t0','all')}  pool_modal {ms['pool_modal']['mean']}"
              f"  oracle@8 {ms['oracle8']['mean']} (sd {ms['oracle8']['sd']})"
              f"  selected {ms['selected']['mean']}  sel_eff {ms['sel_eff']['mean']}")
    for cap, b in orr.get("vs_control", {}).items():
        print(f"  vs cap320 control -- {cap}:")
        for k, m in b.get("per_metric", {}).items():
            print(f"      {k:32s} d={m['delta_mean_over_seeds']:+.6f} per-seed "
                  f"{m['delta_per_seed']} allCIexcl0={m['all_seeds_ci_exclude_zero']}")
        mc = b.get("manipulation_check", {})
        print(f"      manipulation check: mean pool Jaccard vs control = "
              f"{mc.get('mean_pool_jaccard_vs_control')}")
        print(f"      guardrail: " + ", ".join(
            f"{ds}:{v['delta_selected_mean']:+.4f}" for ds, v in b.get("guardrail", {}).items()))
    print("  capture-recapture (Lincoln-Petersen over this cap's own seeds):")
    for cap, v in orr.get("capture_recapture", {}).items():
        print(f"    {cap:8s} macro-open3 LP ceiling {v['macro_open3_LP_ceiling']}  "
              f"(oracle@8 {v['macro_open3_oracle8']})")
    print("  laterality / length strata (arm-invariant masks):")
    for cap, v in orr.get("strata", {}).items():
        if not isinstance(v, dict) or cap.startswith("_"):
            continue
        for nm, s in v.items():
            print(f"    {cap:8s} {nm:30s} n={s['n_items']:5d} sel_eff {s['sel_eff_mean']}  "
                  f"oracle@8 {s['oracle8_mean']}")

    print("=" * 88)
    print("OPEN HALF -- exact-match secondary (no judge, no GPU)")
    em = d.get("open_generator_resolution_exact_match_secondary", {})
    for cap, v in em.get("by_cap", {}).items():
        print(f"  {cap:8s} seeds={v['seeds']} greedy_t0_em "
              f"{g(v,'greedy_t0_em','all')}  modal_em {v['modal_em']['mean']}  "
              f"oracle8_em {v['oracle8_em']['mean']} (sd {v['oracle8_em']['sd']})")
    for cap, b in em.get("vs_control", {}).items():
        for k in ("oracle8_em", "modal_em"):
            if k in b:
                print(f"  vs cap320 -- {cap:8s} {k:12s} d={b[k]['delta_mean_over_seeds']:+.6f} "
                      f"allCIexcl0={b[k]['all_seeds_ci_exclude_zero']}")
        if "greedy_t0_em" in b:
            print(f"  vs cap320 -- {cap:8s} greedy_t0_em d={b['greedy_t0_em']['delta']:+.6f} "
                  f"{b['greedy_t0_em']['ci95']} "
                  f"{'SIG' if b['greedy_t0_em']['significant'] else 'n.s.'}")

    print("=" * 88)
    print("COST")
    for cap, r in g(d, "frontier_open_half", "rows", default={}).items():
        print(f"  {cap:8s} px={r['max_pixels']:>9d} vis_tok={r['measured_mean_vision_tokens_open_pool']:7.1f}"
              f"  FLOPs gen {r.get('flops_rel_to_cap320_generator')}x  whole open arm "
              f"{r.get('flops_rel_to_cap320_whole_open_arm')}x"
              f"  VRAM(d) open arm {r.get('vram_open_arm_d_process_footprint_gib')} GiB")
    w = g(d, "cost", "where_the_open_arm_spends_its_flops", "at_the_deployed_operating_point", default={})
    if w:
        print(f"  verifier/generator FLOPs per candidate = {w.get('verifier_over_generator')}x; "
              f"verifier is {w.get('verifier_share_of_the_8gen_plus_8verify_arm')} of the arm")
    r32 = g(d, "cost", "R32_by_resolution", "by_cap", default={})
    for k, v in r32.items():
        print(f"  R32 at {k:20s} px={v['max_pixels']:>9d} -> {v['R32']}")
    print("=" * 88)


if __name__ == "__main__":
    main()
