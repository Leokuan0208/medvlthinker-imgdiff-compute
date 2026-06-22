#!/usr/bin/env python3
"""
extract_paper_numbers.py -- builds Table 1 (accuracy, four competent benchmarks).
best-of per model: 7B = max(HF cap320 no-think, vLLM fullres no-think); 32B = vLLM think (full).
Prints per-benchmark + four-benchmark macro-average + deltas vs MedVLThinker published.
"""
import argparse, json, glob, os, re
import numpy as np
from collections import defaultdict

FOUR = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
PUB = {  # MedVLThinker RL(m23k), Table 2 of arXiv:2508.02669 (x100)
    "7B":  {"PMC-VQA": 50.67, "SLAKE": 65.79, "VQA-RAD": 64.71, "PathVQA": 66.83},
    "32B": {"PMC-VQA": 54.37, "SLAKE": 73.96, "VQA-RAD": 76.96, "PathVQA": 68.82},
}
def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
FOUR_N = {norm(d): d for d in FOUR}
def load_jsonl(p): return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
def load_arm(ckdir, cell):
    ok, dsn = {}, {}
    if not os.path.isdir(ckdir): return ok, dsn
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$")
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        ds = m.group(1)
        for l in open(f):
            if not l.strip(): continue
            try: r = json.loads(l)
            except Exception: continue
            o = r.get("ok")
            if o is None and "pred" in r and "gold" in r: o = (r["pred"] == r["gold"])
            ok[r["idx"]] = bool(o); dsn[r["idx"]] = ds
    return ok, dsn
def by_ds(ok_map, dsn_map):
    acc = {}
    for d in FOUR:
        vals = [ok_map[i] for i in ok_map if norm(dsn_map.get(i, "")) == norm(d)]
        acc[d] = 100.0 * np.mean(vals) if vals else float("nan")
    return acc
def macro(acc):
    vals = [acc[d] for d in FOUR if acc[d] == acc[d]]
    return float(np.mean(vals)) if len(vals) == len(FOUR) else float("nan")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cascade", default="ckpts/rt_cascade_cap320.jsonl")
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A = ap.parse_args(); R = lambda s: os.path.join(A.repo, s)
    casc = load_jsonl(A.cascade)
    hf7, casc_ok, hf32_esc = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in casc:
        if norm(r["dataset"]) not in FOUR_N: continue
        dd = FOUR_N[norm(r["dataset"])]
        hf7[dd].append(r["pred7"] == r["gold"]); casc_ok[dd].append(bool(r["ok"]))
        if r["escalate"] and r.get("pred32", "") != "": hf32_esc[dd].append(r["pred32"] == r["gold"])
    ours_7b_hf   = {d: 100.0*np.mean(hf7[d])      if hf7[d]      else float("nan") for d in FOUR}
    cascade      = {d: 100.0*np.mean(casc_ok[d])  if casc_ok[d]  else float("nan") for d in FOUR}
    ours_32b_hfE = {d: 100.0*np.mean(hf32_esc[d]) if hf32_esc[d] else float("nan") for d in FOUR}
    ours_7b_vllm  = by_ds(*load_arm(R("ckpts/gate_7b_vllm"),  "nothink_norag"))
    ours_7b_think = by_ds(*load_arm(R("ckpts/gate_7b_think"), "think_norag"))
    ours_32b_vllm = by_ds(*load_arm(R("ckpts/gate_32b"),      "think_norag"))
    ours_7b_best  = {d: np.nanmax([ours_7b_hf[d], ours_7b_vllm[d]]) for d in FOUR}
    ours_32b_best = dict(ours_32b_vllm)
    def row(name, acc, comp=""):
        cells = "".join(f"{acc[d]:8.2f}" if acc[d]==acc[d] else f"{'n/a':>8}" for d in FOUR)
        print(f"  {name:<34}{cells}{macro(acc):8.2f}   {comp}")
    print("\n==== Table 1 numbers : accuracy (%) on the four competent benchmarks ====")
    print(f"  {'method':<34}" + "".join(f"{d:>8}" for d in FOUR) + f"{'Avg4':>8}")
    row("MedVLThinker-7B  [published]",  PUB["7B"]); row("MedVLThinker-32B [published]", PUB["32B"])
    print("  " + "-"*92)
    row("ours 7B  HF cap320 no-think",   ours_7b_hf); row("ours 7B  vLLM fullres no-think", ours_7b_vllm)
    row("ours 7B  BEST-of {HF,vLLM}",    ours_7b_best, "<- use for the 7B row")
    row("ours 7B  vLLM fullres THINK",   ours_7b_think, "(validation vs published 7B)")
    print("  " + "-"*92)
    row("ours 32B vLLM think (full)",    ours_32b_vllm, "<- use for the 32B row")
    row("ours 32B HF think (ESC-only)",  ours_32b_hfE, "(partial; gap G1)")
    print("  " + "-"*92)
    row("ours CASCADE (margin gate)",    cascade, "<- headline row")
    print("\n  deltas (pp):")
    print(f"    our 32B(vLLM) vs published 32B : " + "  ".join(f"{d} {ours_32b_best[d]-PUB['32B'][d]:+.2f}" for d in FOUR) + f"   | macro {macro(ours_32b_best)-macro(PUB['32B']):+.2f}  (validation)")
    print(f"    our 7B(think) vs published 7B  : " + "  ".join(f"{d} {ours_7b_think[d]-PUB['7B'][d]:+.2f}" for d in FOUR if ours_7b_think[d]==ours_7b_think[d]))
    print(f"    cascade       vs our 32B(best) : " + "  ".join(f"{d} {cascade[d]-ours_32b_best[d]:+.2f}" for d in FOUR) + f"   | macro {macro(cascade)-macro(ours_32b_best):+.2f}")
    print(f"    cascade       vs always-7B(HF) : " + "  ".join(f"{d} {cascade[d]-ours_7b_hf[d]:+.2f}" for d in FOUR) + f"   | macro {macro(cascade)-macro(ours_7b_hf):+.2f}  (never-worse: want all >= 0)")
if __name__ == "__main__": main()
