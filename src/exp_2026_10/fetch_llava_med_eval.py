#!/usr/bin/env python3
"""Fetch LLaVA-Med v1.0.0's VQA evaluation code (pinned tag) so its open-ended 'recall' and closed
accuracy can be reused VERBATIM. Saved to /data/dan/repos/llava_med_v1.0.0_eval/ (outside the repo)."""
import os, urllib.request

BASE = "https://raw.githubusercontent.com/microsoft/LLaVA-Med/v1.0.0/"
FILES = ["llava/eval/run_eval.py", "llava/eval/eval_metrics/evaluate_metrics.py",
         "llava/eval/eval_metrics/utils.py", "llava/eval/eval_metrics/glossary.py"]
OUT = "/data/dan/repos/llava_med_v1.0.0_eval"
os.makedirs(OUT, exist_ok=True)
for f in FILES:
    try:
        data = urllib.request.urlopen(BASE + f, timeout=60).read()
        open(os.path.join(OUT, os.path.basename(f)), "wb").write(data)
        print("ok", f, len(data), "bytes")
    except Exception as e:
        print("FAIL", f, e)
