#!/usr/bin/env python3
"""Per-dataset time/energy saved vs always-32B, from the finished cascade JSONL. No GPU.
always-32B per-query cost = measured cost of an escalated query (the 32B fired);
cascade per-query cost = measured mean over all queries; saved = 1 - cascade/always-32B.
Matches the paper's 0.639x energy accounting on the same two-GPU box."""
import json, argparse
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", default="ckpts/rt_cascade_cap320.jsonl")  # now in ckpts/
args = ap.parse_args()
rows = [json.loads(l) for l in open(args.jsonl) if l.strip()]
MEDX = {"MedXpert-Reasoning", "MedXpert-Understanding"}
by = defaultdict(list)
for r in rows: by[r["dataset"]].append(r)

def show(name, R):
    n   = len(R)
    esc = [r for r in R if r.get("escalate")]
    if not esc:
        print(f"\n=== {name} (n={n}) === no escalations, skipping"); return
    er       = len(esc) / n
    base_lat = np.mean([r["latency_s"] for r in esc])   # always-32B
    base_en  = np.mean([r["energy_j"]  for r in esc])
    casc_lat = np.mean([r["latency_s"] for r in R])      # cascade
    casc_en  = np.mean([r["energy_j"]  for r in R])
    print(f"\n=== {name}  (n={n}, esc {er*100:.1f}%) ===")
    print(f"  LATENCY  always-32B {base_lat:6.2f}s | cascade {casc_lat:6.2f}s | saved {(1-casc_lat/base_lat)*100:4.1f}%")
    print(f"  ENERGY   always-32B {base_en:7.0f}J | cascade {casc_en:7.0f}J | saved {(1-casc_en/base_en)*100:4.1f}%")

for ds in sorted(by): show(ds, by[ds])
show("OVERALL (all 6)", rows)
show("EXCL MedXpert (5)", [r for r in rows if r["dataset"] not in MEDX])
