#!/usr/bin/env python3
"""Per-dataset breakdown of the finished cascade run for the weekly report.
CPU only, safe on the completed JSONL. Prints per benchmark + overall + excl-MedXpert."""
import json, sys, argparse
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="ckpts/rt_cascade_cap320.jsonl")
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
if not rows: sys.exit("no rows in " + args.jsonl)

s = rows[0]
def pick(c):
    for k in c:
        if k in s: return k
    return None
F_ds  = pick(["dataset","benchmark","ds"])
F_g   = pick(["gold","answer","label","gt"])
F_p7  = pick(["pred7","pred_7b","pred7b","cheap_pred","p7"])
F_p32 = pick(["pred32","pred_32b","pred32b","strong_pred","p32"])
F_esc = pick(["escalated","escalate","esc","routed_to_32b"])
F_lat = pick(["latency_s","latency","lat_s","total_latency"])
F_en  = pick(["energy_j","energy","query_energy_j","energy_query_j"])
need = [n for n,v in [("dataset",F_ds),("gold",F_g),("pred7",F_p7),
        ("pred32",F_p32),("escalated",F_esc)] if v is None]
if need:
    print("!! missing fields:", need)
    print("!! keys in first record:", list(s.keys()))
    sys.exit("Paste me one JSONL line and I'll fix the field names.")

def ok(p,g): return str(p).strip().upper()[:1] == str(g).strip().upper()[:1]
MEDX = {"MedXpert","MedXpert-Reasoning","MedXpert-Understanding","MedXpertQA","MedX-M"}
by = defaultdict(list)
for r in rows: by[r[F_ds]].append(r)

def blk(name, R):
    n = len(R)
    casc = sum(ok(r[F_p32] if r.get(F_esc) else r[F_p7], r[F_g]) for r in R)/n
    a7   = sum(ok(r[F_p7], r[F_g]) for r in R)/n
    esc  = [r for r in R if r.get(F_esc)]
    er   = len(esc)/n
    rs = sum(1 for r in esc if not ok(r[F_p7],r[F_g]) and ok(r[F_p32],r[F_g]))
    bk = sum(1 for r in esc if ok(r[F_p7],r[F_g]) and not ok(r[F_p32],r[F_g]))
    wa = sum(1 for r in esc if not ok(r[F_p7],r[F_g]) and not ok(r[F_p32],r[F_g]))
    rd = sum(1 for r in esc if ok(r[F_p7],r[F_g]) and ok(r[F_p32],r[F_g]))
    lat = [r[F_lat] for r in R if F_lat and r.get(F_lat) is not None]
    en  = [r[F_en]  for r in R if F_en  and r.get(F_en)  is not None]
    print(f"\n=== {name}  (n={n}) ===")
    print(f"  cascade {casc:.4f} | always-7B {a7:.4f} | esc {er*100:.1f}%")
    if lat: print(f"  latency mean {np.mean(lat):.2f}s p95 {np.percentile(lat,95):.2f}s", end="")
    if en:  print(f" | energy/query {np.mean(en):.0f} J", end="")
    print(f"\n  rescued {rs} | broken {bk} | wasted {wa} | redundant {rd} | net {rs-bk:+d}")

for ds in sorted(by): blk(ds, by[ds])
blk("OVERALL (all 6)", rows)
blk("EXCL MedXpert (5)", [r for r in rows if r[F_ds] not in MEDX])
