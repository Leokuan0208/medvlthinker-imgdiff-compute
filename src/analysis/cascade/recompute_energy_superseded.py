#!/usr/bin/env python3
"""
recompute_energy.py  --  honest per-dataset time/energy saved from the live cascade run.

WHY THIS EXISTS
---------------
The deck's "energy saved" column tracks "time saved" within ~0.1 pp on every row.
That is suspicious: the live run measured very different power on the two cards
(GPU0/7B ~84 W, GPU1/32B ~245 W). If energy were truly read off the power counters,
the cheap 7B leg would be *even cheaper in energy than in wall-clock share*, so
energy-saved should sit a bit ABOVE time-saved -- not glued to it. Numbers that glued
usually mean energy was set proportional to time with one flat power figure.

This script ignores any earlier flat-power shortcut and recomputes everything from the
ACTUAL per-query measured fields in rt_cascade_cap320.jsonl. It prints three things side
by side so you can SEE whether energy really diverges from time:

  (T)  time saved        -- baseline = measured 32B-leg latency; cascade = measured total
  (Ea) energy saved, ACTIVE/marginal  -- only the energy each card draws while computing
  (En) energy saved, NODE total       -- the whole 2-GPU board incl. the idle second card

It also prints the measured mean power of each leg, which is the real diagnostic: if
P(7B) and P(32B) are far apart, (T) and (Ea) MUST differ.

ACCOUNTING (same baseline logic the deck already uses)
------------------------------------------------------
"always-32B" = the 32B runs on every query (the 7B card idle). We never ran that, so per
the standing convention always-32B per-query cost is the measured cost of an escalated
query's 32B leg, applied to all queries. Cascade cost is the measured mean over ALL
queries. saved = 1 - cascade / always-32B.

  time:           base = mean(lat32 | escalated)
                  casc = mean(total_lat | all)   [logged field if present, else lat7 + esc*lat32]
  energy ACTIVE:  base = mean(e32_active | escalated)
                  casc = mean(e7_active | all) + esc_rate * mean(e32_active | escalated)
  energy NODE:    base = mean(e32_active | escalated)   [GPU1-only; see caveat]
                  casc = mean(energy_total | all)        [both boards over the window]

CAVEAT on NODE: the cascade's logged total energy counts BOTH cards (incl. the idle 80 GB
card), while the always-32B baseline here is one card's active energy. That penalises the
cascade for hardware it isn't using for compute, so (En) is a pessimistic floor, not the
headline. (Ea) is the fair "energy of the inference" and lines up with the FLOPs argument.

Usage (run from the repo root):
  python3 src/analysis/cascade/recompute_energy_superseded.py
  python3 src/analysis/cascade/recompute_energy_superseded.py --jsonl /path/to/rt_cascade_cap320.jsonl
NOTE: superseded by recompute_energy.py (same folder), which corrects the per-leg energy accounting.
"""

import argparse, json, os, sys
from collections import defaultdict

# Field auto-detection: first candidate that's actually present wins.
CANDIDATES = {
    "dataset":  ["dataset", "benchmark", "ds", "data", "subset", "source", "task"],
    "escalate": ["escalate", "escalated", "esc", "routed", "went_32b", "use_32b"],
    "gold":     ["gold", "answer", "label", "gt", "target", "correct_answer"],
    "pred7":    ["pred7", "pred_7b", "p7", "ans7", "pred_7", "y7"],
    "pred32":   ["pred32", "pred_32b", "p32", "ans32", "pred_32", "y32"],
    "predf":    ["pred", "final_pred", "prediction", "pred_final", "ans"],          # optional
    "lat7":     ["lat7_s", "lat_7b_s", "lat7", "latency7_s", "lat_7b", "t7_s"],
    "lat32":    ["lat32_s", "lat_32b_s", "lat32", "latency32_s", "lat_32b", "t32_s"],
    "lat_tot":  ["lat_s", "latency_s", "total_lat_s", "lat_total_s", "query_lat_s", "wall_s"],
    "e7":       ["gpu7_energy_j", "gpu0_energy_j", "energy7_j", "e7_j", "energy_7b_j", "gpu0_e_j"],
    "e32":      ["gpu32_energy_j", "gpu1_energy_j", "energy32_j", "e32_j", "energy_32b_j", "gpu1_e_j"],
    "e_tot":    ["energy_j", "total_energy_j", "energy", "e_total_j", "node_energy_j"],
}

def detect(sample):
    return {logical: next((k for k in opts if k in sample), None)
            for logical, opts in CANDIDATES.items()}

def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def first_letter_match(pred, gold):
    return str(pred).strip().upper()[:1] == str(gold).strip().upper()[:1]

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def norm(name):
    return str(name).lower().replace("-", "").replace("_", "").replace(" ", "")

MEDX_KEYS = {norm(x) for x in
             ["MedXpert", "MedXpert-Reasoning", "MedXpert-Understanding",
              "MedXpertQA", "MedX-M", "MedXpert-MM"]}
FOUR_KEYS = {norm(x) for x in ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default=os.path.expanduser(
                        "~/medvlthinker-imgdiff-compute/ckpts/rt_cascade_cap320.jsonl"),
                    help="finished live-cascade JSONL (in the PROJECT folder)")
    args = ap.parse_args()

    if not os.path.exists(args.jsonl):
        sys.exit(f"!! file not found: {args.jsonl}")
    rows = load(args.jsonl)
    if not rows:
        sys.exit("!! file is empty")

    F = detect(rows[0])

    print("=" * 74)
    print(f"FILE: {args.jsonl}   ({len(rows)} records)")
    print("=" * 74)
    print("DETECTED FIELDS (logical -> actual key):")
    for k in ["dataset", "escalate", "gold", "pred7", "pred32", "predf",
              "lat7", "lat32", "lat_tot", "e7", "e32", "e_tot"]:
        print(f"   {k:9s} -> {F[k]}")
    print("\nONE SAMPLE RECORD:\n  ", json.dumps(rows[0])[:400], "\n")

    need = ["dataset", "escalate", "gold", "pred7", "pred32", "lat32", "e32"]
    missing = [k for k in need if F[k] is None]
    if missing:
        print(f"!! cannot find required fields: {missing}")
        print(f"!! keys present: {sorted(rows[0].keys())}")
        sys.exit("Paste me head -n 1 of the file and I'll fix the names.")

    have_e7      = F["e7"]      is not None
    have_e_tot   = F["e_tot"]   is not None
    have_lat7    = F["lat7"]    is not None
    have_lat_tot = F["lat_tot"] is not None

    by = defaultdict(list)
    for r in rows:
        by[r[F["dataset"]]].append(r)

    def cascade_pred(r):
        if F["predf"]:
            return r[F["predf"]]
        return r[F["pred32"]] if r.get(F["escalate"]) else r[F["pred7"]]

    def block(name, R):
        n = len(R)
        esc = [r for r in R if r.get(F["escalate"])]
        er = len(esc) / n
        casc_acc = mean([1.0 if first_letter_match(cascade_pred(r), r[F["gold"]]) else 0.0 for r in R])
        a7_acc   = mean([1.0 if first_letter_match(r[F["pred7"]],    r[F["gold"]]) else 0.0 for r in R])

        e32_mean   = mean([r[F["e32"]]   for r in esc])
        lat32_mean = mean([r[F["lat32"]] for r in esc])
        p32 = e32_mean / lat32_mean if lat32_mean else float("nan")
        if have_e7 and have_lat7:
            e7_mean   = mean([r[F["e7"]]   for r in R])
            lat7_mean = mean([r[F["lat7"]] for r in R])
            p7 = e7_mean / lat7_mean if lat7_mean else float("nan")
        else:
            e7_mean = lat7_mean = p7 = float("nan")

        base_lat = lat32_mean
        if have_lat_tot:
            casc_lat = mean([r[F["lat_tot"]] for r in R])
        elif have_lat7:
            casc_lat = lat7_mean + er * lat32_mean
        else:
            casc_lat = float("nan")
        time_saved = 1 - casc_lat / base_lat if base_lat else float("nan")

        base_e = e32_mean
        if have_e7:
            casc_e_active = e7_mean + er * e32_mean
            e_saved_active = 1 - casc_e_active / base_e if base_e else float("nan")
        else:
            e_saved_active = float("nan")
        if have_e_tot:
            casc_e_node = mean([r[F["e_tot"]] for r in R])
            e_saved_node = 1 - casc_e_node / base_e if base_e else float("nan")
        else:
            e_saved_node = float("nan")

        def pct(x): return f"{x*100:5.1f}%" if x == x else "   -- "
        print(f"--- {name}  (n={n}) ---")
        print(f"    escalation {er*100:5.1f}%   cascade acc {casc_acc:.4f}   always-7B acc {a7_acc:.4f}")
        if p7 == p7 and p32 == p32:
            print(f"    measured leg power:  7B {p7:6.1f} W   |   32B {p32:6.1f} W   (ratio {p7/p32:.2f})")
        else:
            print(f"    measured 32B leg power: {p32:6.1f} W   (no GPU0 energy field -> 7B power unknown)")
        print(f"    SAVED vs always-32B :  time {pct(time_saved)}   "
              f"energy[active] {pct(e_saved_active)}   energy[node] {pct(e_saved_node)}")
        if e_saved_active == e_saved_active:
            print(f"    energy[active] ratio (cascade / always-32B): {1 - e_saved_active:.3f}x")
        print()

    print("#" * 74 + "\n# PER-DATASET\n" + "#" * 74 + "\n")
    for ds in sorted(by):
        block(ds, by[ds])
    print("#" * 74 + "\n# GROUPED\n" + "#" * 74 + "\n")
    block("OVERALL (all 6)", rows)
    block("EXCL MedXpert (5)", [r for r in rows if norm(r[F["dataset"]]) not in MEDX_KEYS])
    block("FOUR COMPETENT (PMC/SLAKE/VQA-RAD/PathVQA)",
          [r for r in rows if norm(r[F["dataset"]]) in FOUR_KEYS])

    print("=" * 74)
    print("READING IT: if 7B and 32B leg power are far apart, time and energy[active]")
    print("MUST differ (energy a touch higher). energy[active] is the headline; energy[node]")
    print("is a pessimistic floor that charges the cascade for the idle second card.")

if __name__ == "__main__":
    main()
