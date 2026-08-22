#!/usr/bin/env python3
"""temperature_report.py -- the sampling penalty and the oracle, as functions of temperature.

WHY.  The eight-cell decomposition (2026-08-22) splits the result into
    head - greedy = (head - one sample) - (greedy - one sample)
and the second term, the SAMPLING PENALTY, is what sinks the cells the head loses on: vqa_rad has
+0.0436 of real selection skill and still loses because its penalty is 0.0686.  Temperature is the
one knob that acts directly on that term, so this maps it.

The tension is that lowering T cuts the penalty AND the pool diversity, and diversity is what puts
a correct answer in the pool at all (oracle@8).  Selection can only ever recover what coverage
provides.  So the quantity to watch is not the penalty alone but the HEADROOM ABOVE GREEDY,
oracle@8 - greedy: if that collapses faster than the penalty falls, a colder pool is worse even
though its average sample is better.

Needs only gen + judge dumps -- no hidden-state extraction -- so it can be run over many
temperatures cheaply, and features extracted afterwards only at whichever T is worth measuring the
head on.

  python3 src/cascade_methods/temperature_report.py
"""
import glob, json, os, re
import numpy as np
from collections import defaultdict

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CK = os.path.join(ROOT, "ckpts/openvqa/cheap_lingshu7b")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/temperature_curve_2026-08-22.json")
# tag suffix -> nominal temperature
TAGS = {"lingshu7b": 0.7, "lingshu7bT02": 0.2, "lingshu7bT04": 0.4, "lingshu7bT10": 1.0}


def main():
    art = {"title": "Sampling penalty and coverage vs generation temperature",
           "date": "2026-08-22", "no_fabricated_numbers": True,
           "note": "greedy is T=0 and identical across every row for a cell, so the comparison "
                   "across temperatures is exact",
           "cells": defaultdict(dict)}
    cells = sorted({re.match(r"ckpt_(.+?)_lingshu7b", os.path.basename(p)).group(1)
                    for p in glob.glob(f"{CK}/ckpt_*_lingshu7b*_sc8.jsonl")
                    if re.match(r"ckpt_(.+?)_lingshu7b", os.path.basename(p))})
    for cell in cells:
        gjp = f"{CK}/ckpt_{cell}_lingshu7b.judge.jsonl"
        if not os.path.exists(gjp):
            continue
        gok = {}
        for l in open(gjp):
            if l.strip():
                d = json.loads(l); gok[d["idx"]] = int(d["judge_ok"])
        for tag, T in sorted(TAGS.items(), key=lambda kv: kv[1]):
            ex = f"{CK}/ckpt_{cell}_{tag}_sc8_scexploded.judge.jsonl"
            if not os.path.exists(ex) or os.path.getsize(ex) == 0:
                continue
            per = defaultdict(list)
            for l in open(ex):
                if l.strip():
                    d = json.loads(l)
                    per[str(d["idx"]).rsplit("#", 1)[0]].append(int(d["judge_ok"]))
            # only questions greedy also covers, so every arm is on the same denominator
            qs = [q for q in per if (q in gok) or (q.isdigit() and int(q) in gok)]
            if len(qs) < 50:
                continue
            def g(q):
                return gok[q] if q in gok else gok[int(q)]
            one = float(np.mean([np.mean(per[q]) for q in qs]))
            orc = float(np.mean([max(per[q]) for q in qs]))
            gr = float(np.mean([g(q) for q in qs]))
            nd = float(np.mean([len(set(per[q])) for q in qs]))
            art["cells"][cell][str(T)] = {
                "n_questions": len(qs), "greedy": gr, "one_sample": one, "oracle_at_8": orc,
                "sampling_penalty": gr - one, "headroom_above_greedy": orc - gr,
                "mean_pool_size": float(np.mean([len(per[q]) for q in qs])),
                "mean_distinct_labels_in_pool": nd}
            r = art["cells"][cell][str(T)]
            print(f"  {cell:17} T={T:<4} n{len(qs):6}  greedy {gr:.4f}  1samp {one:.4f}  "
                  f"penalty {r['sampling_penalty']:+.4f}  oracle {orc:.4f}  "
                  f"headroom {r['headroom_above_greedy']:+.4f}", flush=True)
    art["cells"] = dict(art["cells"])
    # summarise only cells measured at more than one temperature
    multi = {c: v for c, v in art["cells"].items() if len(v) > 1}
    art["temperature_comparison"] = {}
    for c, v in multi.items():
        Ts = sorted(float(t) for t in v)
        art["temperature_comparison"][c] = {
            "temperatures": Ts,
            "penalty": [v[str(t)]["sampling_penalty"] for t in Ts],
            "headroom_above_greedy": [v[str(t)]["headroom_above_greedy"] for t in Ts],
            "oracle": [v[str(t)]["oracle_at_8"] for t in Ts]}
        print(f"\n  {c}: T {Ts}")
        print(f"     penalty  {[round(x,4) for x in art['temperature_comparison'][c]['penalty']]}")
        print(f"     headroom {[round(x,4) for x in art['temperature_comparison'][c]['headroom_above_greedy']]}")
    json.dump(art, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
