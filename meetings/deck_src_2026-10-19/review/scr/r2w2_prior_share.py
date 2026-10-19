"""Round-2, worker 2: per-benchmark share of the probe's gain that the answer-string prior recovers.
Reads audit_stats_baselines_2026-09-20.json (September pooled probe v2, image half A, answer prior = leave-one-out, optimistic).
Adds key 'answer_prior_share' to numbers_review_part2.json without touching other keys."""
import json
ART = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/audit_stats_baselines_2026-09-20.json"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part2.json"
a = json.load(open(ART))
res = {}
for c, v in a["per_benchmark"].items():
    row = {}
    for cur in ("judge", "lenientEM"):
        d = v[cur]
        pg, prg = d["probe_minus_greedy"], d["answer_prior_LOO_optimistic_minus_greedy"]
        row[cur] = {"probe_minus_greedy": round(pg, 4), "prior_minus_greedy": round(prg, 4),
                    "prior_share_of_probe_gain": round(prg / pg, 2) if pg > 0 else None,
                    "probe_minus_prior": round(d["probe_minus_prior"], 4),
                    "probe_minus_prior_ci": [round(x, 4) for x in d["probe_minus_prior_ci"]]}
    res[c] = row
o = json.load(open(OUT))
o["answer_prior_share"] = {"value": res, "computed_by": __file__,
                           "inputs": [ART + " per_benchmark.*.{judge,lenientEM}"],
                           "note": "September pooled probe v2, image half A; prior = answer_prior_LOO_optimistic (optimistic)"}
json.dump(o, open(OUT, "w"), indent=1)
for c, r in res.items():
    print(f"{c:18} judge share {r['judge']['prior_share_of_probe_gain']}  probe-prior {r['judge']['probe_minus_prior']} {r['judge']['probe_minus_prior_ci']} | "
          f"EM share {r['lenientEM']['prior_share_of_probe_gain']}  probe-prior {r['lenientEM']['probe_minus_prior']} {r['lenientEM']['probe_minus_prior_ci']}")
