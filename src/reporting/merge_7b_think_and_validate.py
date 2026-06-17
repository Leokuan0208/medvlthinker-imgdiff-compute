#!/usr/bin/env python3
"""
analyze_7b_think.py - merge the two 7B-think shards and validate the harness.
The deployed cascade's cheap leg is NO_THINK; this 7B-THINK run exists only to (a) confirm
our harness reproduces the paper's 7B-RL row on the benchmarks with the biggest no_think>paper
gaps (SLAKE +7.5, VQA-RAD +11.4), and (b) give the paper its think-7B baseline column.
"""
import json, glob, os, re
import numpy as np
from collections import defaultdict

PAPER_7B = {"PMC-VQA":50.67, "MMMU":56.86, "MedXpert-MM":24.43,
            "PathVQA":66.83, "SLAKE":65.79, "VQA-RAD":64.71}
THINK_DIR   = "ckpts/gate_7b_think"
NOTHINK_DIR = "ckpts/gate_7b_vllm"
ORDER = ["PMC-VQA","SLAKE","VQA-RAD","PathVQA","MMMU","MedXpert-Reasoning","MedXpert-Understanding"]

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            l = l.strip()
            if not l: continue
            try: r = json.loads(l)
            except Exception: continue
            if "idx" in r: d[m.group(1)][r["idx"]] = r
    return d

def stats(rows):
    if not rows: return None
    ok = np.array([r["ok"] for r in rows.values()], float)
    pk = np.array([r.get("parse_ok", 1) for r in rows.values()], float)
    return len(ok), 100*ok.mean(), 100*pk.mean()

def run():
    repo = os.path.expanduser("~/medvlthinker-imgdiff-compute")
    think   = load_arm(os.path.join(repo, THINK_DIR),   "think_norag")
    nothink = load_arm(os.path.join(repo, NOTHINK_DIR), "nothink_norag")

    print(f"merged 7B-think shards from {THINK_DIR}")
    for ds in ORDER:
        n = len(think.get(ds, {}))
        if n: print(f"   {ds:<24} {n} think rows")
    print()
    hdr = (f"{'benchmark':<13}{'n':>6}{'think_acc':>10}{'paper':>8}{'Δpaper':>8}"
           f"{'parse%':>8}   {'nothink_acc':>12}{'Δ(nt-think)':>12}")
    print(hdr); print("-"*len(hdr))

    for ds in ["PMC-VQA","SLAKE","VQA-RAD","PathVQA","MMMU"]:
        st = stats(think.get(ds, {}))
        if not st:
            print(f"{ds:<13}  (no think rows)"); continue
        n, acc, pk = st
        pap = PAPER_7B[ds]; dp = acc - pap
        nt = stats(nothink.get(ds, {}))
        nt_acc = nt[1] if nt else float("nan")
        print(f"{ds:<13}{n:>6}{acc:>10.2f}{pap:>8.2f}{dp:>+8.2f}{pk:>8.1f}   "
              f"{nt_acc:>12.2f}{nt_acc-acc:>+12.2f}")

    def pooled(arm):
        tot_ok = tot_n = 0
        for ds in ["MedXpert-Reasoning","MedXpert-Understanding"]:
            r = arm.get(ds, {})
            tot_ok += sum(x["ok"] for x in r.values()); tot_n += len(r)
        return (100*tot_ok/tot_n, tot_n) if tot_n else (float("nan"), 0)
    t_acc, t_n = pooled(think); nt_acc, _ = pooled(nothink)
    pap = PAPER_7B["MedXpert-MM"]
    print(f"{'MedXpert-MM':<13}{t_n:>6}{t_acc:>10.2f}{pap:>8.2f}{t_acc-pap:>+8.2f}{'':>8}   "
          f"{nt_acc:>12.2f}{nt_acc-t_acc:>+12.2f}   (Reasoning+Understanding pooled)")

    print("\nREAD: Δpaper near 0 => harness reproduces the paper's 7B think row (esp. SLAKE/VQA-RAD,")
    print("the two with the biggest no_think>paper gaps -- this is the airtight check). Δ(nt-think) > 0")
    print("on perception sets = the no_think cheap leg genuinely beats think there; ~0 on MedXpert.")

if __name__ == "__main__":
    run()
