"""Question-weighted (micro) gains and the ratio of macro pick to macro oracle, from the shipped artifact.
Writes /home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/review/scr/r2_w1_micro_out.json."""
import json
A = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/final_eval_FINAL_cap320_T07.json"
d = json.load(open(A))
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
cells = {c: d["cells"][c]["all"] for c in OPEN8}
N = sum(cells[c]["n"] for c in OPEN8)
out = {}
for cur in ["judge", "xjudge", "em"]:
    v = sum(cells[c]["n"] * cells[c][f"pick_minus_greedy_{cur}"] for c in OPEN8) / N
    out[f"micro_pick_minus_greedy_{cur}"] = {"value": round(v, 4), "computed_by": __file__,
                                             "inputs": [A + " cells.<open8>.all.{n, pick_minus_greedy_" + cur + "}"],
                                             "note": f"sum(n_i * delta_i) / {N}; no CI"}
pk = sum(cells[c]["pick_judge"] for c in OPEN8) / 8
orc = sum(cells[c]["oracle_judge"] for c in OPEN8) / 8
per = [cells[c]["pick_judge"] / cells[c]["oracle_judge"] for c in OPEN8]
out["ratio_of_macro_pick_to_macro_oracle_judge"] = {"value": round(pk / orc, 4), "computed_by": __file__,
                                                    "inputs": [A + " cells.<open8>.all.{pick_judge, oracle_judge}"]}
out["mean_of_per_benchmark_pick_over_oracle_judge"] = {"value": round(sum(per) / 8, 4), "computed_by": __file__,
                                                       "inputs": [A + " cells.<open8>.all.{pick_judge, oracle_judge}"],
                                                       "per_benchmark": {c: round(p, 3) for c, p in zip(OPEN8, per)}}
import statistics as st
T975_DF7 = 2.364624  # Student t, 97.5th percentile, 7 degrees of freedom
for cur in ["judge", "xjudge", "em"]:
    xs = [cells[c][f"pick_minus_greedy_{cur}"] for c in OPEN8]
    m, s = st.mean(xs), st.stdev(xs)
    h = T975_DF7 * s / 8 ** 0.5
    out[f"benchmark_as_unit_t_ci_{cur}"] = {"value": [round(m - h, 4), round(m + h, 4)], "mean": round(m, 4),
                                            "computed_by": __file__,
                                            "inputs": [A + f" cells.<open8>.all.pick_minus_greedy_{cur}"],
                                            "note": "one-sample t interval over the 8 per-benchmark deltas (df = 7): treats benchmarks as the sampling unit"}
out["n_questions"] ={"value": N, "computed_by": __file__, "inputs": [A]}
json.dump(out, open("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/review/scr/r2_w1_micro_out.json", "w"), indent=1)
print(json.dumps(out, indent=1))
