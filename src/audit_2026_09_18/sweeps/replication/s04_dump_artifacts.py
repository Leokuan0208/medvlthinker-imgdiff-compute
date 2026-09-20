#!/usr/bin/env python3
"""Dump the head_final_stack artifacts for all generators, side by side. READ-ONLY."""
import json, os, sys

A = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
FILES = ["head_final_stack_PVFIXED_2026-09-13.json",
         "head_final_stack_2026-08-24.json",
         "head_final_stack_qwen_2026-09-13.json",
         "head_final_stack_medgemma_ALL8_2026-09-16.json",
         "head_final_stack_medgemma_2026-09-13.json",
         "head_final_stack_medgemma_ABSLAYERS_2026-09-13.json",
         "head_final_stack_qwen_matched_2026-09-13.json",
         "head_final_stack_lingshu_matched_2026-09-13.json"]
for f in FILES:
    p = os.path.join(A, f)
    if not os.path.exists(p):
        print("MISSING", f); continue
    a = json.load(open(p))
    print("=" * 100)
    print("###", f, " mtime", os.path.getmtime(p))
    for k, v in a.items():
        if k == "cells":
            continue
        print("   ", k, "=", json.dumps(v)[:600])
    print("    cells:")
    for c, d in a.get("cells", {}).items():
        print("      %-18s %s" % (c, json.dumps(d)))
