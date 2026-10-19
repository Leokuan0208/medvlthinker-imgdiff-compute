#!/usr/bin/env python3
"""r2_part3_extra.py — round-2 numbers for deck_part3 slides; ADDS keys to numbers_review_part3.json (never removes).

1. answer-prior share of the probe's judge gain per benchmark (audit_stats_baselines_2026-09-20.json per_benchmark.*.judge;
   September probe, image half A; prior = answer_prior_LOO_optimistic).
2. Wilson 95% interval for 2 wrong of 50 hand-checked judge decisions (pathvqa_judge_audit.json direct_accept + direct_reject).
3. the 7 graders of the WED head-to-head (wed_vs_probe_2026-10-16.json keys) and how many each benchmark wins.
"""
import json, math
A = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts/"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part3.json"
ME = "review/scr/r2_part3_extra.py"

a = json.load(open(A + "audit_stats_baselines_2026-09-20.json"))
share = {}
for c, v in a["per_benchmark"].items():
    j = v["judge"]
    pg = j["probe"] - j["greedy"]
    ag = j["answer_prior_LOO_optimistic"] - j["greedy"]
    share[c] = {"greedy": j["greedy"], "prior": j["answer_prior_LOO_optimistic"], "probe": j["probe"],
                "prior_minus_greedy": ag, "probe_minus_greedy": pg, "prior_share_of_probe_gain": (ag / pg) if pg > 0 else None}

# Wilson interval, z = 1.96
k, n, z = 2, 50, 1.96
p = k / n
den = 1 + z * z / n
mid = (p + z * z / (2 * n)) / den
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
wil = [mid - half, mid + half]

w = json.load(open(A + "wed_vs_probe_2026-10-16.json"))["cells"]
grs = ["judge", "xjudge", "em", "wed_em", "f1", "wed_f1", "llava_recall"]
wins = {}
for c in ["pathvqa_open", "slake_open", "vqa_rad_open"]:
    res = {}
    for g in grs:
        e = w[c][f"probe_minus_wed_{g}"]
        lo, hi = e["ci"]
        res[g] = "win" if lo > 0 else ("loss" if hi < 0 else "tie")
    wins[c] = res

d = json.load(open(OUT))
d["judge.answer_prior_share_per_benchmark_sept_halfA"] = {"value": share, "computed_by": ME,
    "inputs": [A + "audit_stats_baselines_2026-09-20.json"]}
d["judge_audit.wilson95_2_of_50"] = {"value": wil, "computed_by": ME, "inputs": [A + "pathvqa_judge_audit.json (direct_accept 1/25 + direct_reject 1/25)"]}
d["wed.verdict_per_grader"] = {"value": {"graders": grs, "verdicts": wins}, "computed_by": ME, "inputs": [A + "wed_vs_probe_2026-10-16.json"]}
json.dump(d, open(OUT, "w"), indent=1)
for c, v in share.items():
    print(c, {k_: (round(x, 4) if isinstance(x, float) else x) for k_, x in v.items()})
print("wilson 2/50", [round(x, 4) for x in wil])
print(json.dumps(wins))
