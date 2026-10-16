#!/usr/bin/env python3
"""VQA-Med 2019 official test (500): ImageCLEF-style strict accuracy (re-implemented, see final_method_eval.py)
and judge accuracy per category, greedy vs the final method (Tmap policy, cross-fitted records)."""
import collections, json

R = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_Tmap_records.json"))["vqamed2019_test"]
cat = {r["idx"]: r["category"] for r in json.load(open("/data/dan/dataset/comparability_cells/vqamed2019_test.json"))}
s = collections.defaultdict(lambda: [0.0, 0.0, 0, 0.0, 0.0])
for r in R:
    c = cat[r["idx"]]
    s[c][0] += r["greedy"]["imageclef"]; s[c][1] += r["pick"]["imageclef"]; s[c][2] += 1
    s[c][3] += r["greedy"]["judge"]; s[c][4] += r["pick"]["judge"]
out = {}
for c, (g, p, n, gj, pj) in sorted(s.items()):
    out[c] = {"n": n, "imageclef_greedy": g / n, "imageclef_pick": p / n, "judge_greedy": gj / n, "judge_pick": pj / n}
    print(f"{c:12} n {n} strict acc greedy {g / n:.3f} pick {p / n:.3f} | judge greedy {gj / n:.3f} pick {pj / n:.3f}")
n = len(R)
out["overall"] = {"n": n, "imageclef_greedy": sum(r["greedy"]["imageclef"] for r in R) / n,
                  "imageclef_pick": sum(r["pick"]["imageclef"] for r in R) / n}
print("overall", out["overall"])
json.dump(out, open("/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts/"
                    "vqamed2019_by_category_2026-10-16.json", "w"), indent=1)
