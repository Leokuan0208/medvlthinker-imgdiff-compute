#!/usr/bin/env python3
"""s09: judge-vs-EM leniency per generator (same-family self-preference check), from s06's
full-benchmark dump table. READ-ONLY, no new computation beyond aggregation."""
import json, numpy as np
D = json.load(open("/data/dan/audit_2026-09-18/tmp/replication/s06_dump_table.json"))
BENCH = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
print("%-10s %-17s %8s %8s %8s %9s %9s %9s %9s" % (
    "gen", "bench", "g_judge", "g_em", "gap", "P(J|EM0)", "P(J|F1=0)", "poolJ|EM0", "poolJ|F1=0"))
agg = {}
for g in ("lingshu", "qwen", "medgemma"):
    C = D[g]["cells"]
    rows = []
    for b in BENCH:
        o = C[b]
        pd, gd = o["pool_diagnostics"], o["greedy_diagnostics"]
        rows.append((o["greedy_judge"], o["greedy_em"], o["greedy_judge"] - o["greedy_em"],
                     gd["P_judge1_given_em0"], gd["P_judge1_given_f1_zero"],
                     pd["P_judge1_given_em0"], pd["P_judge1_given_f1_zero"]))
        print("%-10s %-17s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %9.4f" % ((g, b) + rows[-1]))
    a = np.array(rows)
    agg[g] = {"macro_greedy_judge": float(a[:, 0].mean()), "macro_greedy_em": float(a[:, 1].mean()),
              "macro_judge_minus_em": float(a[:, 2].mean()),
              "macro_P_judge1_given_em0_greedy": float(a[:, 3].mean()),
              "macro_P_judge1_given_f1zero_greedy": float(a[:, 4].mean()),
              "macro_P_judge1_given_em0_pool": float(a[:, 5].mean()),
              "macro_P_judge1_given_f1zero_pool": float(a[:, 6].mean())}
    print("%-10s %-17s %8.4f %8.4f %8.4f %9.4f %9.4f %9.4f %9.4f" % (
        (g, "MACRO") + tuple(a.mean(0))))
json.dump(agg, open("/data/dan/audit_2026-09-18/tmp/replication/s09_judgefamily.json", "w"),
          indent=1)
print(json.dumps(agg, indent=1))
