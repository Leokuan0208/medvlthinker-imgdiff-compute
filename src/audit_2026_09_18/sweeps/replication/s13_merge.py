#!/usr/bin/env python3
"""s13: merge every number this audit computed into replication_currency_2026-09-20.json."""
import json, os
import numpy as np
T = "/data/dan/audit_2026-09-18/tmp"
P = f"{T}/replication"
A = "/home/jamesyang/medvlthinker-imgdiff-compute/results/cascade_methods/artifacts"
B = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
     "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]
R = json.load(open(f"{P}/replication_currency_2026-09-20.json"))
R["_meta"]["scripts"] = ["s03_refit_worker.py (predecessor)", "s06_dump_table.py", "s07_currency.py",
                         "s09_judgefamily.py", "s10_tables.py", "s11_length.py", "s12_degenerate.py"]
R["_meta"]["outputs"] = ["s06_dump_table.json", "s09_judgefamily.json", "s11_length.json",
                         "s12_degenerate.json", "s10_tables.out", "s07_currency.out"]
R["full_benchmark_dump_table_s06"] = json.load(open(f"{P}/s06_dump_table.json"))
R["judge_family_s09"] = json.load(open(f"{P}/s09_judgefamily.json"))
R["length_control_s11"] = json.load(open(f"{P}/s11_length.json"))
R["degeneration_s12"] = json.load(open(f"{P}/s12_degenerate.json"))
R["lingshu_reference_em_rescore"] = json.load(open(f"{T}/em-rescore/em_rescore_pooled_probe_2026-09-18.json"))["macro"]

# artifact facts
af = {}
for f in sorted(os.listdir(A)):
    if not (f.startswith("head_final_stack") or f.startswith("repro_")):
        continue
    a = json.load(open(os.path.join(A, f)))
    if "macro" not in a:
        continue
    af[f] = {"generator": a.get("generator", "lingshu"), "macro": a["macro"],
             "beats_greedy": a.get("beats_greedy"), "VERDICT": a.get("VERDICT"),
             "pooled_rows": a.get("pooled_rows"), "seeds": a.get("seeds"),
             "has_provenance": "provenance" in a,
             "provenance": a.get("provenance"),
             "best_arm_by_macro": max(a["macro"], key=a["macro"].get),
             "n_cells": len(a.get("cells", {}))}
R["artifact_facts"] = af

# generation-config audit (read back from the printed run: recompute here so it is in the JSON)
CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
gc = {}
for tag in ("lingshu7b", "qwen25vl7b", "medgemma4b"):
    gc[tag] = {}
    for b in B:
        for suf, lab in (("", "greedy"), ("_sc8", "sc8")):
            gt = []
            with open(f"{CK}/ckpt_{b}_{tag}{suf}.jsonl") as fh:
                for l in fh:
                    if l.strip():
                        d = json.loads(l)
                        gt += (d.get("gen_tokens_all") or [d.get("gen_tokens")])
            gt = np.array([x for x in gt if x is not None])
            gc[tag].setdefault(b, {})[lab] = {
                "n": int(len(gt)), "mean_gen_tokens": float(gt.mean()),
                "max_gen_tokens": int(gt.max()),
                "frac_at_max": float((gt == gt.max()).mean()),
                "frac_at_64": float((gt == 64).mean())}
R["generation_config_audit"] = gc
json.dump(R, open(f"{P}/replication_currency_2026-09-20.json", "w"), indent=1)
print("merged; top-level keys:", list(R))
print("medgemma sc8 frac_at_64 per bench:",
      {b: round(gc["medgemma4b"][b]["sc8"]["frac_at_64"], 4) for b in B})
print("qwen sc8 frac_at_64 per bench:",
      {b: round(gc["qwen25vl7b"][b]["sc8"]["frac_at_64"], 4) for b in B})
print("lingshu sc8 frac_at_64 per bench:",
      {b: round(gc["lingshu7b"][b]["sc8"]["frac_at_64"], 4) for b in B})
