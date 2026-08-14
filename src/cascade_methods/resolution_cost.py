#!/usr/bin/env python3
"""resolution_cost.py -- SWEEP 2: the COST axis of the resolution frontier.

FLOP-eq per resolution for BOTH halves, computed with the project's own analytic forward-pass
model (src/cascade_methods/flop_ratio_derivation.forward_flops, exact parameter counts read from
the safetensors headers) evaluated on the TOKEN GEOMETRY THIS SWEEP ACTUALLY MEASURED -- vision
tokens, prompt tokens and generated tokens per item per cap, recorded by
resolution_open_generate.py for the open half and by the MedEvalKit dumps for the MCQ half.

No number here is a name-plate ratio: the geometry is measured, the parameter counts are read off
disk, and the per-cap FLOP totals are the model evaluated on those.

VRAM is NOT re-measured here.  results/cascade_methods/artifacts/vram_levers_2026-08-12.json
already carries, for the same six-cap ladder and with the same four conventions
(a weights_resident / b peak_allocated / c peak_reserved / d process_footprint), the 7B MCQ leg
and the open-text best-of-8 arm under HF transformers.  Those rows are cited verbatim and marked
as a prior session's measurement; nothing here re-labels them.

    python3 src/cascade_methods/resolution_cost.py
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_resolution_parts")
os.makedirs(OUT, exist_ok=True)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
CAPS = [("cap80", 62720), ("cap160", 125440), ("cap320", 250880), ("cap640", 501760),
        ("fullres", 1003520), ("native", 12845056)]

from src.cascade_methods.flop_ratio_derivation import forward_flops, param_counts  # noqa: E402


NEXP = {"slake_open": 645, "vqa_rad_open": 200, "pathvqa_open": 1500}


def open_geometry(cap, seeds=("s0", "s1", "s2", "t0")):
    """mean per-item (vision tokens, prompt tokens, generated tokens per sample) at this cap.

    Reads the first COMPLETE arm at this cap. The vision/prompt geometry is a property of the cap
    and the images, not of the sampling seed, so any complete arm gives the same answer; the
    generated-token mean is the one quantity that is arm-specific and it is taken from the same arm
    and named in the output.
    """
    seed = next((s for s in seeds
                 if all(os.path.exists(os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{s}.jsonl"))
                        and sum(1 for l in open(os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{s}.jsonl"))
                                if l.strip()) >= NEXP[ds] for ds in DS)), None)
    if seed is None:
        return None
    v, p, g, n = [], [], [], 0
    for ds in DS:
        f = os.path.join(SWEEP, f"ckpt_{ds}_{cap}_{seed}.jsonl")
        if not os.path.exists(f):
            return None
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            v.append(r.get("vision_px_tokens", np.nan))
            p.append(r.get("prompt_tokens", np.nan))
            g.append(float(np.mean(r["gen_tokens_all"])))
            n += 1
    return dict(n=n, seed_used=seed, vision=float(np.nanmean(v)), prompt=float(np.nanmean(p)),
                gen=float(np.nanmean(g)))


def mcq_geometry():
    """MedEvalKit-track generated tokens per cap, read from this session's ladder dumps."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "MedEvalKit", "eval_results_res7b_px*"))):
        px = int(os.path.basename(d).split("px")[-1])
        cells = {}
        for cell in ["SLAKE", "VQA_RAD", "PATH_VQA", "MedXpertQA-MM", "PMC_VQA"]:
            p = os.path.join(d, "{}", cell, "results.json")
            if not os.path.exists(p):
                continue
            rs = json.load(open(p))
            cells[cell] = {"n": len(rs),
                           "mean_gen_tokens": round(float(np.mean([r["gen_toks"] for r in rs])), 3),
                           "mean_recorded_latency_s": round(
                               float(np.mean([r["latency_s"] for r in rs])), 5),
                           "acc": round(float(np.mean([bool(r["correct"]) for r in rs])), 6)}
        out[px] = cells
    return out


_VERIF_GEOM = {}


def verifier_flops(pc7, n_per_ds=40):
    """One verifier forward, at the verifier's OWN deployed configuration.

    The token geometry is MEASURED: the exact verifier prompt of
    src/training_methods/verifier_transfer_eval.py is rebuilt for a sample of real (image,
    question, candidate) triples with the Lingshu-7B processor at max_pixels 1,003,520, and its
    tokens are counted. Processor only -- no GPU, no model.
    """
    if not _VERIF_GEOM:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from src.cascade_methods.resolution_verifier_score import (MAXPX, MINPX, SYS, imgs_for)
        proc = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")
        vis, tot = [], []
        dumps = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
        for ds, nm in [("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"),
                       ("pathvqa_open", "pathvqa")]:
            d = json.load(open(os.path.join(dumps, f"transfer_dump_{nm}_open_lingshu7b.json")))
            IMG = imgs_for(ds)
            for r in d[:n_per_ds]:
                if r["idx"] not in IMG:
                    continue
                q, img = IMG[r["idx"]]
                m = [{"role": "system", "content": SYS},
                     {"role": "user", "content": [
                         {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                         {"type": "text", "text": f"Question: {q}\nProposed answer: {r['preds'][0]}"
                                                  f"\nIs the proposed answer correct? Answer Yes "
                                                  f"or No."}]}]
                text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                igs, vids = process_vision_info(m)
                enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt")
                tot.append(int(enc["input_ids"].shape[1]))
                vis.append(int(igs[0].size[0] * igs[0].size[1] / (28 * 28)))
        _VERIF_GEOM["vision"] = float(np.mean(vis))
        _VERIF_GEOM["prompt"] = float(np.mean(tot))
        _VERIF_GEOM["n"] = len(tot)
    f = forward_flops(pc7, _VERIF_GEOM["vision"], _VERIF_GEOM["prompt"], 1.0)
    f["_geometry"] = dict(_VERIF_GEOM)
    return f


def main():
    pc7 = param_counts("Lingshu-7B")
    pc32 = param_counts("Lingshu-32B")
    res = {"_meta": {
        "flop_model": "src/cascade_methods/flop_ratio_derivation.forward_flops -- prefill-inclusive, "
                      "vision tower + merger + LM prefill + LM decode + lm_head; parameter counts "
                      "read from the safetensors headers on disk (no weights loaded).",
        "geometry": "MEASURED in this sweep: vision/prompt/generated tokens per item per cap, "
                    "recorded by resolution_open_generate.py (open half) and by MedEvalKit's own "
                    "per-item dumps (MCQ half). Nothing is a name-plate ratio.",
        "params_7b": pc7["total_params"], "params_32b": pc32["total_params"]},
        "open_half_per_candidate": {}, "open_half_arm": {}, "mcq_half": {}}

    base = None
    for cap, px in CAPS:
        g = open_geometry(cap)
        if g is None:
            continue
        f = forward_flops(pc7, g["vision"], g["prompt"], g["gen"])
        # the deployed open arm = 8 sampled candidates (generator at THIS cap) + 8 verifier
        # forwards (verifier held at its own 1,003,520 -- see resolution_verifier_score.py), so
        # the verifier term is CONSTANT across caps by construction.
        fv = verifier_flops(pc7)
        row = {"max_pixels": px, "vision_token_budget": px // (28 * 28),
               "measured_mean_vision_tokens": round(g["vision"], 2),
               "measured_mean_prompt_tokens": round(g["prompt"], 2),
               "measured_mean_gen_tokens": round(g["gen"], 3),
               "geometry_read_from_arm": g["seed_used"],
               "flops_per_candidate": f["TOTAL"], "parts": {k: v for k, v in f.items()}}
        if fv:
            row["flops_arm_8gen_plus_8verify"] = 8 * f["TOTAL"] + 8 * fv["TOTAL"]
            row["flops_verifier_per_candidate_at_1003520"] = fv["TOTAL"]
            row["verifier_measured_geometry"] = fv["_geometry"]
        res["open_half_per_candidate"][cap] = row
        if cap == "cap320":
            base = row
    if base:
        for cap in res["open_half_per_candidate"]:
            r = res["open_half_per_candidate"][cap]
            r["flops_rel_to_cap320_generator"] = round(
                r["flops_per_candidate"] / base["flops_per_candidate"], 5)
            if "flops_arm_8gen_plus_8verify" in r and "flops_arm_8gen_plus_8verify" in base:
                r["flops_rel_to_cap320_whole_arm"] = round(
                    r["flops_arm_8gen_plus_8verify"] / base["flops_arm_8gen_plus_8verify"], 5)

    res["mcq_half"] = {"measured_this_session": mcq_geometry()}

    # ---- the 32B/7B FLOP-equivalence ratio AT THE RESOLUTION THE MCQ CELLS ACTUALLY RAN AT ----
    # flop_ratio_derivation_2026-08-03.json derives R32 = 3.816 from token geometry measured at
    # cap320 (image_tok_mean 280.48) and states a sensitivity band [3.734, 3.859] over three
    # operating points -- none of which is max_pixels 12,845,056, the resolution 62.5% of the
    # macro weight (the 5 MCQ cells, 7B AND the 32B-direct bar) is actually evaluated at.
    # The MCQ token geometry at that cap was MEASURED on 2026-08-12
    # (vram_levers_2026-08-12.json flops_by_cap.medevalkit_default).
    try:
        vl2 = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts",
                                          "vram_levers_2026-08-12.json")))
        fb = vl2["flops_by_cap"]
        rr = {}
        for capname, row in fb.items():
            if not isinstance(row, dict) or "measured_merged_vision_tokens_mean" not in row:
                continue
            M = row["measured_merged_vision_tokens_mean"]
            T = row["measured_input_tokens_mean"]
            Gt = row["measured_gen_tokens_mean"]
            f7 = forward_flops(pc7, M, T, Gt)["TOTAL"]
            f32 = forward_flops(pc32, M, T, Gt)["TOTAL"]
            rr[capname] = {"max_pixels": row["max_pixels"],
                           "measured_vision_tokens_mean": M, "measured_input_tokens_mean": T,
                           "flops_7b": f7, "flops_32b": f32, "R32": round(f32 / f7, 4)}
        res["R32_by_resolution"] = {
            "_what": "the 32B/7B whole-forward FLOP ratio recomputed at each cap's MEASURED MCQ "
                     "token geometry, with the project's own forward_flops and its own exact "
                     "parameter counts.",
            "_why": "flop_ratio_derivation_2026-08-03.json fixes R32 = 3.816 from cap320 geometry "
                    "and gives a sensitivity band [3.734, 3.859] whose three operating points are "
                    "cap320-open, cap320-MCQ and 1,003,520-MCQ. The 5 MCQ cells -- 62.5% of the "
                    "macro weight, for BOTH the method's cheap leg and the always-32B-direct bar "
                    "-- ran at 12,845,056, which is not among them.",
            "by_cap": rr,
            "project_value": 3.816, "project_band": [3.734, 3.859],
            "_read": "this is arithmetic on two already-measured inputs, not a new experiment, and "
                     "it is reported as a flag for a re-costing, not as a re-costing."}
    except Exception as e:
        res["R32_by_resolution"] = {"error": f"{type(e).__name__}: {e}"}
    res["vram_cited_not_remeasured"] = {
        "_source": "results/cascade_methods/artifacts/vram_levers_2026-08-12.json "
                   "(resolution_frontier / strong_leg_by_cap), measured 2026-08-12 under HF "
                   "transformers, bf16, flash_attention_2, tp=1, batch 1, with the four "
                   "conventions of vram_testtime_2026-08-11.json. Cited, not re-measured: the "
                   "instrument, the caps and the conventions are identical, and re-running it "
                   "would burn GPU for a number already measured.",
        "_caveat": "those open-arm rows were driven at max_new_tokens=512 while the deployed "
                   "generator uses 64, so their KV component is an upper bound; the cap-vs-cap "
                   "DELTAS are unaffected because the setting is constant across caps."}
    try:
        vl = json.load(open(os.path.join(ROOT, "results/cascade_methods/artifacts",
                                         "vram_levers_2026-08-12.json")))
        rows = {}
        for r in vl["resolution_frontier"]:
            rows[r["cap"]] = {
                "max_pixels": r["max_pixels"],
                "mcq_7b_b_peak_allocated_gib": r["vram_7b_mcq"]["b_peak_allocated_gib"],
                "mcq_7b_c_peak_reserved_gib": r["vram_7b_mcq"]["c_peak_reserved_gib"],
                "mcq_7b_d_process_footprint_gib": r["vram_7b_mcq"]["d_process_footprint_gib"],
                "open_arm_b_peak_allocated_gib": r["vram_opentext_bestof8_arm"]["b_peak_allocated_gib"],
                "open_arm_c_peak_reserved_gib": r["vram_opentext_bestof8_arm"]["c_peak_reserved_gib"],
                "open_arm_d_process_footprint_gib": r["vram_opentext_bestof8_arm"]["d_process_footprint_gib"]}
        res["vram_cited_not_remeasured"]["by_cap"] = rows
        res["vram_cited_not_remeasured"]["a_weights_resident_gib_7b"] = 15.4937
        res["vram_cited_not_remeasured"]["strong_leg_32b_by_cap"] = vl["strong_leg_by_cap"]["by_cap"]
    except Exception as e:
        res["vram_cited_not_remeasured"]["error"] = f"{type(e).__name__}: {e}"

    # ---- where the open arm's FLOPs actually are --------------------------------------------
    try:
        c320 = res["open_half_per_candidate"]["cap320"]
        gen = c320["flops_per_candidate"]
        ver = c320["flops_verifier_per_candidate_at_1003520"]
        # the same verifier prompt geometry with the image capped at the generator's 250,880:
        # vision tokens fall to the open pool's measured cap320 mean, text tokens are unchanged.
        vg = c320["verifier_measured_geometry"]
        text_tok = vg["prompt"] - vg["vision"]
        v320 = forward_flops(pc7, c320["measured_mean_vision_tokens"],
                             c320["measured_mean_vision_tokens"] + text_tok, 1.0)["TOTAL"]
        res["where_the_open_arm_spends_its_flops"] = {
            "at_the_deployed_operating_point": {
                "generator_per_candidate_at_250880": gen,
                "verifier_per_candidate_at_1003520": ver,
                "verifier_over_generator": round(ver / gen, 4),
                "verifier_share_of_the_8gen_plus_8verify_arm": round(ver / (gen + ver), 4)},
            "_read": "the deployed open arm spends more compute VERIFYING than GENERATING, because "
                     "the verifier runs at 4x the generator's resolution on the same image. That "
                     "caps what a generator-side resolution cut can buy: taking the generator to "
                     "cap80 (a 4x token cut) removes only "
                     f"{round(100 * (1 - res['open_half_per_candidate'].get('cap80', {}).get('flops_rel_to_cap320_whole_arm', float('nan'))), 1)}% "
                     "of the arm.",
            "counterfactual_verifier_at_the_generators_250880": {
                "verifier_per_candidate": v320,
                "arm_flops_rel_to_deployed": round((gen + v320) / (gen + ver), 4),
                "_accuracy_already_measured_elsewhere":
                    "results/cascade_methods/artifacts/vram_levers_2026-08-12.json "
                    "open_half_levers arm bf16_cap320: d_sel_eff -0.014528 "
                    "[-0.036320, +0.007264] n.s., d_selected_acc -0.010000 [-0.025000, +0.005000] "
                    "n.s., on a seeded 200-per-set subsample (n=600). NOT re-measured here.",
                "_this_is_a_cost_counterfactual": "the FLOP number is this round's arithmetic on "
                                                  "this round's measured geometry; the accuracy "
                                                  "number is the prior round's and carries its "
                                                  "subsample and its CI."}}
    except Exception as e:
        res["where_the_open_arm_spends_its_flops"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- wall-clock per arm, from the generation log. CONTEXT ONLY. -------------------------
    import re
    wall = {}
    lg = os.path.join(ROOT, "logs/resolution_open_gen_2026-08-13.log")
    if os.path.exists(lg):
        for line in open(lg, errors="ignore"):
            m = re.match(r"^DONE (\S+) (\S+)_(s\d|t0) in ([\d.]+) min", line.strip())
            if m:
                ds, cap, tag, mins = m.group(1), m.group(2), m.group(3), float(m.group(4))
                wall.setdefault(cap, {}).setdefault(tag, {})[ds] = mins
    res["wall_clock_per_arm_minutes_CONTEXT_ONLY"] = {
        "_what": "wall time to generate one whole arm (all 2,345 items; 8 samples each for s0/s1/s2, "
                 "1 for t0) at each cap, read from logs/resolution_open_gen_2026-08-13.log.",
        "_why_it_is_not_a_latency_result": "vLLM, batched, tp=1, on a card that carried another "
                                           "tenant's jobs for the entire session (two of this "
                                           "round's own loads were OOM-killed by that tenant). It "
                                           "shows the DIRECTION and rough size of the throughput "
                                           "cost of resolution, nothing more. Batch-1 latency for "
                                           "the open arm was NOT measured this round.",
        "by_cap": wall}

    json.dump(res, open(os.path.join(OUT, "cost_by_resolution.json"), "w"), indent=1)
    for c, r in res["open_half_per_candidate"].items():
        print(f"{c:10s} px={r['max_pixels']:>9d} vis={r['measured_mean_vision_tokens']:8.1f} "
              f"flops/cand={r['flops_per_candidate']:.3e} "
              f"rel_gen={r.get('flops_rel_to_cap320_generator')} "
              f"rel_arm={r.get('flops_rel_to_cap320_whole_arm')}")
    print("wrote", os.path.join(OUT, "cost_by_resolution.json"))


if __name__ == "__main__":
    main()
