"""Mean four_domain / lobo / pooled minus greedy over the 4 benchmarks with no training split in the 4-domain set,
and over the 4 same-source ones (head_lobo_pooled_2026-08-25.json, August-era probe, judge). Appends to numbers_review_part3.json."""
import json
A = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/head_lobo_pooled_2026-08-25.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part3.json"
c = json.load(open(A))["cells"]
groups = {"no_training_split": ["radimagenet_open", "omnimed_open", "vqamed_open", "gemex_open"],
          "same_source": ["pathvqa_open", "slake_open", "vqa_rad_open", "kvasir_x1_open"]}
res = {g: {k: round(sum(c[x][f"{k}_minus_greedy"] for x in cs) / len(cs), 4) for k in ("four_domain", "lobo", "pooled")}
       for g, cs in groups.items()}
print(res)
d = json.load(open(OUT))
d["lobo.four_benchmark_means"] = {"value": res, "computed_by": "tmp/deck/review/scr/r3_lobo4.py",
                                  "inputs": [A + " cells.*.{four_domain,lobo,pooled}_minus_greedy"],
                                  "note": "unweighted mean over the 4 cells in each group; August-era pooled probe, judge"}
json.dump(d, open(OUT, "w"), indent=1)
