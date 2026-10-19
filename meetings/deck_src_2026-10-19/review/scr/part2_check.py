#!/usr/bin/env python3
"""part2_check.py — print the artifact values the Part III/IV fixes cite (read-only)."""
import json
A = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/"
M = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/"
D = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/"


def j(p):
    for b in (A, M, D):
        try:
            return json.load(open(b + p))
        except FileNotFoundError:
            pass
    raise FileNotFoundError(p)


def dig(o, path):
    for k in path.split("."):
        o = o[k]
    return o


v = j("verifier_32b_gpu.json")
print("32B pooled deltas", json.dumps(dig(v, "pooled.deltas_vs_32B"))[:900])
fh = j("free_head_2026-08-16.json")
d3 = fh["DELIVERABLE_3_cost"]
print("arms_fixed_bestof8 keys", list(d3["arms_fixed_bestof8"].keys()))
for k in ("bo8_head_TODAY", "bo8_incumbent_LoRA", "bo8_generation_only"):
    print(k, json.dumps(d3["arms_fixed_bestof8"].get(k))[:400])
print("passes", json.dumps(d3.get("passes_per_question"))[:300])
print("per_forward_flopeq", json.dumps(d3.get("per_forward_flopeq"))[:400])
print("geometry", json.dumps(d3.get("geometry"))[:600])
tc = j("temp_crossfit_2026-10-15.json")["macro"]
print("temp_crossfit", {k: (round(v["macro"], 4) if isinstance(v, dict) and "macro" in v else v) for k, v in tc.items() if "fixed" in k})
ro = j("readout_variants_2026-10-15.json")["macro"]
for k in ("z_mean_minus_rank_em", "vote_prob_minus_rank_em", "rank_vote_minus_rank_j", "rank_vote_minus_rank_em", "z_mean_minus_rank_j", "z_mean_minus_rank_x"):
    print("readout", k, json.dumps(ro.get(k))[:200])
sp = j("select_policy_cap320_vs_fullres_T07.json")["macro"]
print("select_policy", {k: (v.get("macro") if isinstance(v, dict) else v) for k, v in sp.items()})
ab = j("audit_stats_baselines_2026-09-20.json")["summary_by_currency"]
for cur in ab:
    for k, v in ab[cur].items():
        if isinstance(v, dict):
            print("baselines", cur, k, {kk: v.get(kk) for kk in ("macro", "benchmark_t_interval_df7")}, "gemex", v.get("per_benchmark", {}).get("gemex_open"))
hp = j("head_price_from_lobo_2026-08-30.json")
print("price summary", json.dumps(hp.get("summary"))[:500])
for c, cv in hp.get("cells", {}).items():
    print(" price cell", c, "crossover_k", cv.get("crossover_k"), json.dumps({k: cv[k] for k in cv if "frac" in k or "k100" in k})[:200])
bl = j("bestofn_latency_energy_2026-08-03.json")
print("bestofn_latency keys", list(bl.keys())[:20])
print(json.dumps(bl.get("measured", bl))[:1200])
we = j("worked_example_12231.json")
print("worked example keys", list(we.keys()))
