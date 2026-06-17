#!/usr/bin/env python3
"""Inspect timing/energy fields in the saved checkpoints + the cascade JSONL.
CPU only; reads one record from each. Tells us if per-dataset always-32B
time/energy can be pulled from existing data or needs a fresh run."""
import json, glob, os

def show(label, path, jsonl):
    print("\n" + "="*70); print(label, "->", path); print("="*70)
    if jsonl:
        if not os.path.exists(path): print("  (not found)"); return
        line = open(path).readline()
        if not line.strip(): print("  (empty)"); return
        rec = json.loads(line)
        print("  keys:", sorted(rec.keys()))
        hits = {k: rec[k] for k in rec if any(t in k.lower()
                for t in ["lat","time","sec","energy","joul","power","token","gen","watt","dur"])}
        print("  timing/energy fields:", hits or "(none)")
    else:
        fs = sorted(glob.glob(os.path.join(path, "*.json"))) or \
             sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
        if not fs: print("  (no .json here)"); return
        rec = json.load(open(fs[0])); print("  file:", fs[0])
        val = next(iter(rec.values())) if isinstance(rec, dict) and rec else (rec[0] if isinstance(rec, list) and rec else rec)
        if isinstance(val, dict):
            print("  per-item keys:", sorted(val.keys()))
            hits = {k: val[k] for k in val if any(t in k.lower()
                    for t in ["lat","time","sec","energy","joul","power","token","gen","watt","dur"])}
            print("  timing/energy fields:", hits or "(none)")
        else:
            print("  item type:", type(val).__name__)

# EDIT paths to match your repo (jsonl is in the project root, per your note):
show("rt_cascade live run (HF)",      os.path.expanduser("~/medvlthinker-imgdiff-compute/rt_cascade_cap320.jsonl"), True)
show("always-32B vLLM eval (gate_32b)", os.path.expanduser("~/medvlthinker-imgdiff-compute/ckpts/gate_32b"), False)
show("always-7B vLLM eval (gate_7b)",   os.path.expanduser("~/medvlthinker-imgdiff-compute/ckpts/gate_7b_vllm"), False)
