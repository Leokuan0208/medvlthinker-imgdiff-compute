#!/usr/bin/env python3
"""
think_rescue_mechanism.py - WHY does rescue@think work? Among the think-CANDIDATES (small-nt
disagrees with big-nt -> ACC-v2 would fire the 28s think pass), split by whether the SMALL model's
answer is RESOLUTION-STABLE. If stable candidates benefit LESS from think (big-think acc - big-nt
acc is ~0 or negative) while unstable ones benefit, then skipping think on the stable ones is sound
(not just blind think-reduction). MedVLThinker, offline.
"""
import os, json
import numpy as np
PRUNE = "ckpts/gate_7b_prune"; BIGNT = "ckpts/gate_32b_modes/nothink_cap320"; BIGTH = "ckpts/gate_32b"
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]; ALL5 = COMP4 + ["MMMU"]
EXTRA = ["cap80", "cap160", "cap640"]
def load(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def main():
    rows = []
    for ds in ALL5:
        c320 = load(f"{PRUNE}/cap320/ckpt_{ds}_nothink_norag.jsonl")
        cex = {c: load(f"{PRUNE}/{c}/ckpt_{ds}_nothink_norag.jsonl") for c in EXTRA}
        bnt = load(f"{BIGNT}/ckpt_{ds}_nothink_norag.jsonl"); bth = load(f"{BIGTH}/ckpt_{ds}_think_norag.jsonl")
        idx = set(c320) & set(bnt) & set(bth)
        for c in EXTRA: idx &= set(cex[c])
        for i in sorted(idx):
            p = c320[i]["pred"]
            rows.append(dict(ds=ds, disagree=(p != bnt[i]["pred"]),
                             stable=all(cex[c][i]["pred"] == p for c in EXTRA),
                             ok_bignt=bnt[i]["ok"], ok_bigthink=bth[i]["ok"]))
    R = rows
    cand = [r for r in R if r["disagree"]]          # think candidates (ACC-v2 would think here)
    st = [r for r in cand if r["stable"]]; un = [r for r in cand if not r["stable"]]
    def benefit(g): return (np.mean([r["ok_bigthink"] for r in g]) - np.mean([r["ok_bignt"] for r in g])) if g else float("nan")
    print("=" * 78)
    print("rescue@think MECHANISM (ALL-5): among think-candidates (small-nt != big-nt)")
    print("=" * 78)
    print(f"  think-candidates: {len(cand)}  | resolution-STABLE: {len(st)}  UNSTABLE: {len(un)}")
    print(f"  THINK BENEFIT (big-think acc - big-nt acc) on this set:")
    print(f"    stable candidates   : {benefit(st):+.4f}   (skip think here -> ~free if ~0/negative)")
    print(f"    unstable candidates : {benefit(un):+.4f}   (keep think here)")
    print(f"    all candidates      : {benefit(cand):+.4f}")
    # per-benchmark
    print(f"\n  per-benchmark think-benefit (stable | unstable):")
    for ds in ALL5:
        cs = [r for r in cand if r["ds"] == ds]; s = [r for r in cs if r["stable"]]; u = [r for r in cs if not r["stable"]]
        print(f"    {ds:<22} {benefit(s):+.4f} (n={len(s):4d})  |  {benefit(u):+.4f} (n={len(u):4d})")
    print("\n  READ: if stable<=unstable benefit, resolution-stability genuinely selects think-calls")
    print("  that don't help -> rescue@think is real signal, not blind think reduction.")

if __name__ == "__main__":
    main()
