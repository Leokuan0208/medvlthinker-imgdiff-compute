#!/usr/bin/env python3
"""resolution_arm_provenance.py -- SWEEP 2: which session generated each open-text arm.

The +-0.008 caveat says a fresh generation may only be differenced against a control generated in
the SAME serving configuration. This round spans two sessions:

  2026-08-13  every cap320 arm (s0,s1,s2,t0), cap80 s0/s1, native s0 and native t0,
              via runners/run_resolution_open_gen.sh at --gpu_mem 0.30
  2026-08-14  native s1 and s2, generated directly at --gpu_mem 0.60 to use an empty card

So the seed-0 pair (native s0 vs cap320 s0) and the temperature-0 pair are WITHIN-session and are
the primary result; a seed-1 or seed-2 native-vs-control pair crosses sessions AND changes
gpu_memory_utilization, which changes the KV-cache size and therefore vLLM's batch composition.
The round measured what that is worth directly: re-running the SAME cap320 configuration in a
different serving config reproduced only 96.887% of answer strings (null test N3).

This script writes a per-arm provenance map from file mtimes and flags every cap/seed pair that
crosses the session boundary, so no reader can take a cross-session delta for a matched one.

    python3 src/cascade_methods/resolution_arm_provenance.py
"""
import datetime
import glob
import json
import os

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres", "native"]
CONTROL = "cap320"
NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}

SESSION_CFG = {
    "2026-08-13": {"driver": "runners/run_resolution_open_gen.sh", "gpu_mem": 0.30,
                   "vllm": "0.10.1.1+381074ae.nv25.09", "tp": 1},
    "2026-08-14": {"driver": "src/cascade_methods/resolution_open_generate.py invoked directly",
                   "gpu_mem": 0.60, "vllm": "0.10.1.1+381074ae.nv25.09", "tp": 1},
}


def main():
    arms = {}
    for cap in CAPS:
        for tag in ["t0", "s0", "s1", "s2"]:
            days, nrows, complete = set(), {}, True
            for ds in DS:
                p = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{tag}.jsonl")
                if not os.path.exists(p):
                    complete = False
                    continue
                n = sum(1 for l in open(p) if l.strip())
                nrows[ds] = n
                if n < NEXP[ds]:
                    complete = False
                days.add(datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d"))
            if not nrows:
                continue
            arms[f"{cap}_{tag}"] = {
                "cap": cap, "seed_tag": tag, "complete": complete, "rows": nrows,
                "generated_on": sorted(days),
                "session_config": {d: SESSION_CFG.get(d) for d in sorted(days)},
            }

    pairs = {}
    for cap in CAPS:
        if cap == CONTROL:
            continue
        for tag in ["t0", "s0", "s1", "s2"]:
            a, b = arms.get(f"{CONTROL}_{tag}"), arms.get(f"{cap}_{tag}")
            if not a or not b or not a["complete"] or not b["complete"]:
                continue
            same = a["generated_on"] == b["generated_on"]
            pairs[f"{cap}_vs_{CONTROL}_{tag}"] = {
                "control_generated_on": a["generated_on"],
                "treatment_generated_on": b["generated_on"],
                "WITHIN_SESSION_MATCHED": bool(same),
                "gpu_mem_control": [SESSION_CFG[d]["gpu_mem"] for d in a["generated_on"]],
                "gpu_mem_treatment": [SESSION_CFG[d]["gpu_mem"] for d in b["generated_on"]],
                "_status": ("PRIMARY -- matched control, same session, same serving config"
                            if same else
                            "SECONDARY -- crosses the session boundary AND changes "
                            "gpu_memory_utilization (0.30 -> 0.60), which resizes the KV cache and "
                            "changes vLLM's batch composition. Report per seed, never merged into "
                            "the matched delta, and never as the headline."),
            }

    res = {
        "_what": "which session generated each open-text arm, and which cap-vs-control pairs are "
                 "therefore matched. Derived from file mtimes in ckpts/openvqa/resolution_sweep/.",
        "_why": "the project's +-0.008 reproducibility caveat: regenerating an arm under a "
                "different serving configuration moves cells by ~+-0.008, which is larger than the "
                "entire published vs-direct delta. Null test N3 measured this round's own version "
                "of it: re-running the SAME cap320 config in a different serving config reproduced "
                "96.887% of answer strings with NO experimental variable changed.",
        "arms": arms,
        "pairs": pairs,
        "_headline_rule": "the primary native-vs-cap320 result in this artifact is the SEED-0 "
                          "sampled pair plus the deterministic TEMPERATURE-0 pair, both generated "
                          "2026-08-13 in one session at gpu_mem 0.30. Native s1/s2 exist to show "
                          "seed-to-seed spread at native resolution; they are not merged into the "
                          "primary delta.",
    }
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "arm_provenance.json"), "w"), indent=1)
    for k, v in pairs.items():
        print(f"{k:28} matched={v['WITHIN_SESSION_MATCHED']}  "
              f"ctl={v['control_generated_on']} trt={v['treatment_generated_on']}")
    print("wrote", os.path.join(OUT, "arm_provenance.json"))


if __name__ == "__main__":
    main()
