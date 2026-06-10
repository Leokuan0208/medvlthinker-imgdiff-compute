#!/usr/bin/env python3
"""
check_7b_think_vs_paper.py - validate the eval harness in THINK mode using the
7B think data we already have (ckpts/gate_7b_v2), paired on idx against our 7B
no_think (gate_7b_vllm) and the 32B (gate_32b), and compared to the paper row.
All three accuracies are computed on the SAME samples (the think subset limits n).
CPU only, read-only.
"""
import json, glob, os, re, argparse
import numpy as np
from collections import defaultdict

DIR_7B_THINK   = "ckpts/gate_7b_v2"      # 7B think_norag (+ some nothink), sharded
DIR_7B_NOTHINK = "ckpts/gate_7b_vllm"    # 7B nothink_norag, full eval
DIR_32B        = "ckpts/gate_32b"        # 32B think_norag

# paper MedVLThinker RL m23k (7B, 32B); MedX-M = MedXpert-MM combined
PAPER = {"PMC-VQA": (50.67, 54.37), "MedX-M": (24.43, 34.60)}

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}_s\d+of\d+\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if "idx" in r: d[m.group(1)][r["idx"]] = r
    return d

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A = ap.parse_args(); repo = A.repo
    t7  = load_arm(os.path.join(repo, DIR_7B_THINK),   "think_norag")
    n7  = load_arm(os.path.join(repo, DIR_7B_NOTHINK), "nothink_norag")
    r32 = load_arm(os.path.join(repo, DIR_32B),        "think_norag")

    have = sorted(t7.keys())
    print("7B-think datasets found:", have, "\n")
    def common(ds): return sorted(set(t7.get(ds,{})) & set(n7.get(ds,{})) & set(r32.get(ds,{})))
    def acc(arm, ds, idx): return 100*np.mean([arm[ds][i]["ok"] for i in idx]) if idx else float("nan")

    hdr = f"{'dataset':<22}{'n':>6}{'7B think':>10}{'7B nothk':>10}{'32B think':>11}{'paper7B':>10}{'paper32B':>10}"
    print(hdr); print("-"*len(hdr))
    for ds in have:
        idx = common(ds)
        p = PAPER.get(ds, ("", ""))
        ps  = f"{p[0]:>10}" if p[0] != "" else f"{'--':>10}"
        ps2 = f"{p[1]:>10}" if p[1] != "" else f"{'--':>10}"
        print(f"{ds:<22}{len(idx):>6}{acc(t7,ds,idx):>10.2f}{acc(n7,ds,idx):>10.2f}{acc(r32,ds,idx):>11.2f}{ps}{ps2}")

    mx = [d for d in have if d.startswith("MedXpert")]
    if len(mx) == 2:
        idxs = {d: common(d) for d in mx}
        def comb(arm):
            oks = []
            for d in mx: oks += [arm[d][i]["ok"] for i in idxs[d]]
            return 100*np.mean(oks)
        n = sum(len(idxs[d]) for d in mx)
        p = PAPER["MedX-M"]
        print("-"*len(hdr))
        print(f"{'MedX-M (R+U)':<22}{n:>6}{comb(t7):>10.2f}{comb(n7):>10.2f}{comb(r32):>11.2f}{p[0]:>10}{p[1]:>10}")

    print("\nRead: 32B-think should sit on paper-32B (harness OK on these files). If 7B-think")
    print("also lands on paper-7B while 7B-nothink differs, the gap is confirmed as the MODE.")

if __name__ == "__main__":
    run()
