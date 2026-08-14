#!/usr/bin/env python3
"""vision_diversity_explode.py -- turn the (idx, view) generation dump into judge inputs.

One row per UNIQUE (question, normalized answer) over ALL views AND the iid control, so the judge
is called once per distinct string and both arms are labelled by the SAME judge call whenever they
produced the same answer (no arm can be advantaged by judge noise).

Composite idx is '<origidx>||<j>' where j indexes the question's unique normalized answers, in
first-appearance order. Field names match run_judge.py's expectations (idx, question, gold,
modal_pred).

  python3 src/cascade_methods/vision_diversity_explode.py ckpts/openvqa/visdiv/gen_slake_open.jsonl
"""
import json
import os
import sys
from collections import OrderedDict

src = sys.argv[1]
out = src.replace(".jsonl", "_scexploded.jsonl")
byq = OrderedDict()
meta = {}
for l in open(src):
    if not l.strip():
        continue
    r = json.loads(l)
    k = str(r["idx"])
    meta.setdefault(k, (r.get("question", ""), r.get("gold", "")))
    d = byq.setdefault(k, OrderedDict())
    for a in r.get("preds", []):
        d.setdefault(str(a).strip().lower(), a)   # keep the first surface form
n_out = 0
with open(out, "w") as fh:
    for k, d in byq.items():
        q, gold = meta[k]
        for j, (na, surf) in enumerate(d.items()):
            fh.write(json.dumps({"idx": f"{k}||{j}", "na": na, "question": q, "gold": gold,
                                 "modal_pred": surf}) + "\n")
            n_out += 1
print(f"{os.path.basename(src)}: {len(byq)} questions -> {n_out} unique (idx,answer) judge rows "
      f"({n_out/max(len(byq),1):.1f}/q) -> {os.path.basename(out)}")
