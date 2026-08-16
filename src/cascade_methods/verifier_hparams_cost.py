#!/usr/bin/env python3
"""verifier_hparams_cost.py -- KNOB 3: what the verifier's scoring resolution costs.

The FLOP model is the project's own: flop_ratio_derivation.forward_flops (prefill-inclusive --
vision tower + merger + LM prefill + LM decode + lm_head), with parameter counts read from the
safetensors headers on disk.  Nothing is a name-plate ratio.

The GEOMETRY IS MEASURED IN THIS ROUND, on all 8,965 scored triples per rung rather than on a
120-triple sample: verifier_hparams_score.py records `in_tok` (prompt tokens) and `patch`
(pre-merge patch rows of pixel_values; vision tokens = patch/4) for every forward it does.

THE ARM.  The deployed open arm is 8 generator samples + 8 verifier forwards per question.  The
generator is FROZEN at cap320 in this round, so its term is a constant read from the previous
round's measured geometry (resolution_sweep_2026-08-13.json, open_half_per_candidate.cap320:
5.6927e12 FLOPs/candidate at 274.13 vision / 318.99 prompt / 5.404 generated tokens).

    python3 src/cascade_methods/verifier_hparams_cost.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from flop_ratio_derivation import forward_flops, param_counts   # noqa: E402

SCOREDIR = os.path.join(ROOT, "ckpts/openvqa/verifier_hparams")
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
PRIOR = os.path.join(ROOT, "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json")
CONTROL_PX = 1003520
GEN_PX = 250880


def main():
    os.makedirs(PARTS, exist_ok=True)
    pc7 = param_counts("Lingshu-7B")
    prior = json.load(open(PRIOR))
    gcap = prior["cost"]["open_half_per_candidate"]["cap320"]
    gen_flops = float(gcap["flops_per_candidate"])

    res = {"_meta": {
        "flop_model": "src/cascade_methods/flop_ratio_derivation.forward_flops -- prefill-"
                      "inclusive; parameter counts from the safetensors headers on disk.",
        "params_7b": pc7["total_params"],
        "verifier_geometry": "MEASURED on all 8,965 scored triples per rung by "
                             "verifier_hparams_score.py (in_tok, patch). vision tokens = "
                             "pixel_values rows / 4 (Qwen2.5-VL 2x2 spatial merge).",
        "verifier_gen_tokens": 1.0,
        "verifier_gen_tokens_note": "the verifier does ONE forward and reads the Yes/No logits at "
                                    "the last position -- it generates nothing. G=1 is the "
                                    "convention of resolution_cost.verifier_flops, kept identical "
                                    "so the two rounds' verifier terms are comparable.",
        "generator_term_held_fixed": {
            "source": "results/cascade_methods/artifacts/resolution_sweep_2026-08-13.json "
                      "cost.open_half_per_candidate.cap320",
            "max_pixels": gcap["max_pixels"],
            "measured_mean_vision_tokens": gcap["measured_mean_vision_tokens"],
            "measured_mean_prompt_tokens": gcap["measured_mean_prompt_tokens"],
            "measured_mean_gen_tokens": gcap["measured_mean_gen_tokens"],
            "flops_per_candidate": gen_flops},
        "arm_definition": "8 generator samples + 8 verifier forwards per question (the frozen "
                          "best-of-8 endpoint). The DEPLOYED arm draws an adaptive N (Weitzman); "
                          "N=8 is the frozen metric's convention, and the ratios below are "
                          "invariant to N as long as generator and verifier draw the same N.",
    }, "by_max_pixels": {}}

    pxs = sorted(int(os.path.basename(f)[len("scores_px"):-len(".jsonl")])
                 for f in glob.glob(os.path.join(SCOREDIR, "scores_px*.jsonl")))
    for px in pxs:
        rows = []
        for l in open(os.path.join(SCOREDIR, f"scores_px{px}.jsonl")):
            if l.strip():
                r = json.loads(l)
                if r.get("p") is not None:
                    rows.append((r["in_tok"], r["patch"], r["wall_s"]))
        if len(rows) < 8965:
            print(f"  px{px}: incomplete ({len(rows)}/8965) -- skipped")
            continue
        a = np.array(rows, float)
        vis, prm = float(a[:, 1].mean() / 4.0), float(a[:, 0].mean())
        f = forward_flops(pc7, vis, prm, 1.0)
        res["by_max_pixels"][str(px)] = {
            "max_pixels": px, "n_forwards_measured": len(rows),
            "measured_mean_vision_tokens": round(vis, 3),
            "measured_max_vision_tokens": float(a[:, 1].max() / 4.0),
            "measured_mean_prompt_tokens": round(prm, 3),
            "flops_per_verifier_forward": f["TOTAL"],
            "parts": {k: v for k, v in f.items()},
            "in_run_mean_wall_s_batch1": float(a[:, 2].mean()),
            "in_run_median_wall_s_batch1": float(np.median(a[:, 2])),
            "_wall_s_caveat": "measured inside the scoring run with the OTHER A100 also busy with "
                              "this round's own arms; treat as indicative. The clean timing is "
                              "verifier_hparams_vram.py latency_batch1_s.",
            "flops_open_arm_8gen_plus_8verify": 8 * gen_flops + 8 * f["TOTAL"],
            "verifier_share_of_open_arm_flops": f["TOTAL"] / (f["TOTAL"] + gen_flops),
            "verifier_flops_rel_to_generator": f["TOTAL"] / gen_flops,
        }
    if str(CONTROL_PX) in res["by_max_pixels"]:
        b = res["by_max_pixels"][str(CONTROL_PX)]
        for k, r in res["by_max_pixels"].items():
            r["flops_verifier_rel_to_deployed"] = r["flops_per_verifier_forward"] / \
                b["flops_per_verifier_forward"]
            r["flops_open_arm_rel_to_deployed"] = r["flops_open_arm_8gen_plus_8verify"] / \
                b["flops_open_arm_8gen_plus_8verify"]
            r["open_arm_flops_saved_pct"] = 100.0 * (1 - r["flops_open_arm_rel_to_deployed"])
    json.dump(res, open(os.path.join(PARTS, "cost.json"), "w"), indent=1, default=float)
    print(json.dumps({k: {kk: v[kk] for kk in
                          ["measured_mean_vision_tokens", "flops_per_verifier_forward",
                           "verifier_share_of_open_arm_flops", "flops_open_arm_rel_to_deployed",
                           "open_arm_flops_saved_pct"] if kk in v}
                      for k, v in res["by_max_pixels"].items()}, indent=1))
    print(f"\nwrote {PARTS}/cost.json")


if __name__ == "__main__":
    main()
