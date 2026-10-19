"""Round-3 number audit: read-only recomputation from artifacts. Writes only review/scr/r3_checks_out.json."""
import json, math
ART = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/"
out = {}
a = json.load(open(ART + "audit_stats_baselines_2026-09-20.json"))
pb = {}
for c, v in a["per_benchmark"].items():
    r = {}
    for cur in ("judge", "lenientEM"):
        d = v[cur]
        pg, prg = d["probe_minus_greedy"], d["answer_prior_LOO_optimistic_minus_greedy"]
        r[cur] = dict(probe=round(pg, 4), prior=round(prg, 4), share=round(prg / pg, 4) if pg > 0 else None,
                      probe_minus_prior=round(d["probe_minus_prior"], 4), ci=[round(x, 4) for x in d["probe_minus_prior_ci"]])
    pb[c] = r
out["answer_prior"] = pb
out["answer_prior_summary_judge"] = a["summary_by_currency"]["judge"]
j = json.load(open(ART + "judge_2x2_2026-09-20.json"))
po, p1, p2 = j["train_label_agreement_AB"], j["train_label_pos_rate"]["A"], j["train_label_pos_rate"]["B"]
pe = p1 * p2 + (1 - p1) * (1 - p2)
out["kappa"] = dict(rows=j["train_rows"], agree=po, pA=p1, pB=p2, kappa=(po - pe) / (1 - pe))
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]
out["wilson_2_50"] = wilson(2, 50)
out["wilson_7_90"] = wilson(7, 90)
g = json.load(open(ART + "image_geometry_all8_2026-10-15.json"))
out["gemex_geometry"] = g["benchmarks"]["gemex_open"]
out["pooled_geometry"] = g.get("pooled_question_weighted")
json.dump(out, open("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/review/scr/r3_checks_out.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:6000])
