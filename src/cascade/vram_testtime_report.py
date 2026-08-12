#!/usr/bin/env python3
"""vram_testtime_report.py -- assemble the per-scenario parts written by
src/cascade/measure_testtime_vram.py into the single artifact
results/cascade_methods/artifacts/vram_testtime_2026-08-11.json.

Adds nothing measured: it only aggregates, adds the conventions block, the per-source (driver)
breakdown, the reconciliation against the circulating vLLM numbers -- every one of which is read
verbatim from a named log line -- and the not_measured list.

  python3 src/cascade/vram_testtime_report.py
"""
import glob, json, os, subprocess, time

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
PARTS = os.path.join(ROOT, "results/cascade_methods/artifacts/_vram_parts")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json")
GIB_PER_GB = 1e9 / 1024 ** 3          # decimal GB -> GiB
A100_TOTAL_BYTES = 81920 * 1024 ** 2  # 85,899,345,920 B = 85.899 GB decimal = 80.0 GiB


def load_parts():
    out = {}
    for f in sorted(glob.glob(os.path.join(PARTS, "vram_*.json"))):
        for k, v in json.load(open(f)).items():
            v["_part_file"] = os.path.relpath(f, ROOT)
            if k in out:      # a re-run supersedes; the older one is KEPT, renamed, as evidence
                old = out[k]
                old["_status"] = ("SUPERSEDED / ABORTED -- retained deliberately as the record of a "
                                  "failed token audit; see its meta.token_audit.verdict. DO NOT QUOTE.")
                out[k + "_ABORTED_prompt_conflict"] = old
            out[k] = v
    return out


def agg(rows, key):
    v = [r[key] for r in rows if key in r and "error" not in r]
    if not v:
        return None
    return dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                min=round(float(np.min(v)), 4), n=len(v))


def contamination_audit(sc):
    """(d) is board `used`, so a foreign process landing on the card inflates it. The CUDA context is
    a CONSTANT offset above (c), so delta = d_board - c_peak_reserved must be flat across items. Any
    upward step in delta is foreign memory, not ours.

    The clean footprint is therefore  (c) + min(delta)  -- min because contamination only ADDS.
    Reported alongside the raw board reading, never silently in place of it."""
    rows = [r for r in sc.get("rows", []) if "error" not in r and "d_board_used_gib" in r]
    if not rows:
        return None, None
    dl = [round(r["d_board_used_gib"] - r["c_peak_reserved_gib"], 4) for r in rows]
    ctx, spread = min(dl), round(max(dl) - min(dl), 4)
    clean = [round(r["c_peak_reserved_gib"] + ctx, 4) for r in rows]
    dirty = [i for i, x in enumerate(dl) if x > ctx + 0.01]
    aud = dict(
        cuda_context_offset_gib=ctx,
        offset_spread_gib=spread,
        verdict=("CLEAN -- (d_board - c_peak_reserved) is constant to <=0.01 GiB across every item, "
                 "which a competing process could not leave intact"
                 if spread <= 0.01 else
                 f"FOREIGN MEMORY DETECTED on {len(dirty)} of {len(rows)} items (first: item "
                 f"{dirty[0]+1}), up to +{spread} GiB above the {ctx} GiB CUDA-context offset that "
                 f"every other item shows. A second tenant shared this GPU during the run "
                 f"(independently corroborated by nvidia-smi: foreign residue was observed on GPU 0 "
                 f"after our processes exited, and GPU occupancy rose to 43-48 GB with none of our "
                 f"processes running). (b) and (c) are torch-internal and therefore UNAFFECTED; only "
                 f"the raw board reading is. d_clean_peak_gib is the corrected figure and is the one "
                 f"quoted in deployer_guidance."),
        n_items_affected=len(dirty),
        affected_item_positions=[i + 1 for i in dirty],
        d_raw_board_peak_gib=round(max(r["d_board_used_gib"] for r in rows), 4),
        d_clean_peak_gib=max(clean),
        d_clean_method=("c_peak_reserved + min(d_board - c_peak_reserved) over the scenario's items; "
                        "identical to the raw board reading when the verdict is CLEAN"))
    return aud, clean


def per_source(sc):
    out = {}
    for r in sc.get("rows", []):
        if "error" in r:
            continue
        out.setdefault(r["src"], []).append(r)
    return {k: dict(n=len(v),
                    b_peak_allocated_gib=agg(v, "b_peak_allocated_gib"),
                    c_peak_reserved_gib=agg(v, "c_peak_reserved_gib"),
                    d_process_footprint_gib=agg(v, "d_process_footprint_clean_gib"),
                    image_pixels=agg(v, "image_pixels"), vision_tokens=agg(v, "vision_tokens"),
                    input_tokens=agg(v, "input_tokens"), gen_tokens=agg(v, "gen_tokens"))
            for k, v in out.items()}


P = load_parts()

# ------------------------------------------------------------------ conventions (the whole point)
CONV = {
    "units": ("ALL VRAM values in this file are GiB = bytes / 1024**3, unless a key literally says "
              "_gb. The circulating vLLM numbers (71.45 / 152.82 / 154.08) are DECIMAL GB "
              "(bytes/1e9) because src/labeling/nvml_power.py:29 divides by 1e9 -- so they are not "
              "even in the same unit as (a)-(d). Both units are given in the reconciliation block."),
    "a_weights_resident": {
        "what": "torch.cuda.memory_allocated() immediately after from_pretrained().to('cuda') + synchronize.",
        "is": "the model parameters, and nothing else. No activations, no KV cache, no CUDA context.",
        "is_not": "what you provision. It is a floor."},
    "b_peak_allocated": {
        "what": "torch.cuda.max_memory_allocated(), torch.cuda.reset_peak_memory_stats() called immediately before EVERY item.",
        "is": "peak LIVE tensor bytes during the item: weights + activations + KV cache.",
        "is_not": "what the allocator took from the driver (that is (c))."},
    "c_peak_reserved": {
        "what": "torch.cuda.max_memory_reserved(), reset before every item.",
        "is": "(b) plus caching-allocator fragmentation -- what torch held from the driver.",
        "note": ("torch.cuda.empty_cache() + reset_peak_memory_stats() is called BETWEEN scenarios so a "
                 "later scenario's (c) is not inflated by an earlier one's high-water mark; each "
                 "scenario's meta.scenario_reset records the post-release reserved/allocated as proof.")},
    "d_process_footprint": {
        "what": ("whole-process GPU memory, max over a 20 ms NVML sampler running during the item. "
                 "= (c) + CUDA context + cuBLAS workspaces + everything outside the torch allocator."),
        "is": "WHAT A DEPLOYER PROVISIONS. This is the number to size a GPU against.",
        "method_actually_used": ("nvml_board_used_minus_prerun_baseline. NVML per-process "
                                 "usedGpuMemory was attempted first but returns 0 in this container: "
                                 "NVML reports HOST pids, which never match os.getpid(). Every row "
                                 "carries d_method, plus the raw d_board_used_gib and "
                                 "d_nvml_per_process_gib, so the substitution is auditable."),
        "why_the_substitute_is_sound_here": [
            "both A100s were verified idle at 13 MiB immediately before launch (logs/vram_testtime_pre_run_nvidia_smi.txt)",
            "exactly ONE of our processes was pinned per GPU via CUDA_VISIBLE_DEVICES",
            "(d_board_used - c_peak_reserved) is the CUDA-context offset and must be CONSTANT across "
            "items; every scenario is tested for that and the result is recorded in its "
            "contamination_audit block"],
        "a_second_tenant_DID_appear_partway_through": (
            "S1, S2 and S3 are CLEAN (offset constant at 1.3835-1.3855 GiB across every item). S4 and "
            "S5 are NOT: a foreign process shared the card for part of those runs (+0.59 GiB on 8 of "
            "12 S4 items, +16.93 GiB on 1 of 15 S5 items), corroborated by nvidia-smi showing 43-48 GB "
            "occupied on GPU 0 with none of our processes running. (a), (b) and (c) are torch-internal "
            "and are UNAFFECTED. For (d) the contaminated board readings are discarded and the "
            "footprint is reconstructed as (c) + the CUDA-context offset from that scenario's own "
            "clean items. The reconstruction is validated on S5: it returns 72.696 GiB, which is "
            "exactly the board value directly measured on the uncontaminated peak-driving item "
            "(MM-1561). Raw board readings are retained in every row so the correction is auditable.")},
    "what_is_NOT_here": ("no vLLM number in this file is a measurement OF THIS METHOD. vLLM reserves a "
                         "memory pool sized by --gpu_mem and reports the POOL. See reconciliation.")}

# ------------------------------------------------------------ reconciliation (verbatim log sources)
RECON = {
    "why_three_numbers_disagree": (
        "They measure three different things. (1) The vLLM RESERVATION (71.45 / 152.82 / 154.08 'GB') "
        "is NVML board `used` while vLLM held a pre-allocated pool whose size is the --gpu_mem flag "
        "times the card, summed over every visible GPU -- it is a property of the FLAG and the number "
        "of GPUs, not of the model, which is why three different 7B models all report exactly 71.45 "
        "and three different 32B models exactly 152.82. (2) vLLM's own 'model weights take X GiB' line "
        "is honest but is PER TENSOR-PARALLEL WORKER and covers weights ONLY -- at tp=2 you must "
        "double it, and it excludes activations, KV cache and the CUDA context. (3) The HF numbers "
        "below are per-quantity and per-item: (a) matches vLLM's weights line to within 0.5%, which "
        "cross-validates both; (b)/(c)/(d) then add the activations, KV cache and CUDA context that "
        "(2) omits and that (1) buries inside an arbitrary pool. Rule of thumb from these "
        "measurements: provision (d). It is ~1.4 GiB above (c) and, at the 32B tier, roughly HALF the "
        "reservation that has been circulating."),
    "tiers": {}}


def add_recon(tier, hf_a, vllm_res_gb, res_src, res_cfg, w_gib, w_src, w_cfg, w_tp, extra=None):
    tot = w_gib * w_tp
    r = dict(
        hf_measured_a_weights_resident_gib=hf_a,
        hf_measured_source="THIS artifact, scenarios below (torch.cuda.memory_allocated after load, tp=1, HF, bf16)",
        vllm_reservation_gb_decimal=vllm_res_gb,
        vllm_reservation_gib=round(vllm_res_gb * GIB_PER_GB, 3),
        vllm_reservation_config=res_cfg,
        vllm_reservation_source=res_src,
        vllm_reservation_status="NOT A FOOTPRINT -- a memory-pool reservation set by --gpu_mem, summed over visible GPUs",
        vllm_honest_weights_line_gib_per_worker=w_gib,
        vllm_honest_weights_tp=w_tp,
        vllm_honest_weights_total_gib=round(tot, 3),
        vllm_honest_weights_config=w_cfg,
        vllm_honest_weights_source=w_src,
        hf_vs_vllm_weights_delta_gib=round(hf_a - tot, 3),
        hf_vs_vllm_weights_delta_pct=round(100.0 * (hf_a - tot) / tot, 2))
    if extra:
        r.update(extra)
    RECON["tiers"][tier] = r


s1 = P.get("S1_lingshu7b_direct_mcq", {})
s2 = P.get("S2_lingshu32b_direct_mcq", {})
a7 = s1.get("meta", {}).get("load", {}).get("a_weights_resident_gib")
a32 = s2.get("meta", {}).get("load", {}).get("a_weights_resident_gib")

add_recon(
    "Lingshu-7B", a7, 71.45,
    "logs/lat_lingshu_s.log 'PEAK_VRAM_GB=71.45' (scraped into results/cascade_methods/docs/archive_mcq/MASTER_TABLES.md '## Peak VRAM (GB, batch-1)')",
    "vLLM tp=1, gpu_memory_utilization=0.85, max_model_len 8192, batch-1, 1 GPU",
    15.57,
    "logs/medevalkit_smoke.log 'model weights take 15.57GiB' (model='lingshu-medical-mllm/Lingshu-7B', tensor_parallel_size=1)",
    "vLLM tp=1, bfloat16", 1,
    extra=dict(flag_arithmetic=("0.85 x 85.899 GB (A100 80GB total) = 73.0 GB, vs the 71.45 GB reported: "
                                "the reservation tracks the FLAG, not the 15.57 GiB of weights it holds."),
               identical_value_across_families=("MASTER_TABLES.md reports the SAME 71.45 for "
                                                "MedVLThinker-7B, Lingshu-7B and QoQ-Med-7B -- the proof it is not a model property.")))

add_recon(
    "Lingshu-32B", a32, 152.82,
    "logs/lat_lingshu_bnt.log 'PEAK_VRAM_GB=152.82' (scraped into MASTER_TABLES.md '## Peak VRAM (GB, batch-1)')",
    "vLLM tp=2, gpu_memory_utilization = vLLM default 0.90 (not passed in that run's non-default args), max_model_len 8192, batch-1, SUMMED OVER BOTH GPUs",
    31.28,
    "logs/medeval_mmmu_repro.log / logs/omnimed_rerun.log / logs/mmmu_perm_32b.log 'model weights take 31.28GiB' (model='lingshu-medical-mllm/Lingshu-32B', tensor_parallel_size=2)",
    "vLLM tp=2, bfloat16 -- PER WORKER, so the whole model is 2 x 31.28", 2,
    extra=dict(flag_arithmetic=("2 x 0.90 x 85.899 GB = 154.6 GB, vs the 152.82 GB reported (and 154.08 "
                                "for the 'think' tier): the reservation tracks the FLAG and the GPU COUNT. "
                                "'think adds 1.26 GB of VRAM' is an artifact of that, not a KV-cache "
                                "measurement -- see S5 in this artifact for the real reasoning delta."),
               independent_tp1_confirmation=("logs/omni32b_tp1_probe.log:41 loaded Lingshu-32B at tp=1 and "
                                             "reported 'model weights take 62.43GiB; non_torch_memory takes "
                                             "0.09GiB; PyTorch activation peak memory takes 6.90GiB; the rest "
                                             "of the memory reserved for KV Cache is 0.22GiB.' 62.43 GiB vs "
                                             "2 x 31.28 = 62.56 GiB vs this artifact's HF 62.3125 GiB -- three "
                                             "independent routes to the same weights figure."),
               identical_value_across_families=("MASTER_TABLES.md reports the SAME 152.82 / 154.08 for "
                                                "MedVLThinker-32B, Lingshu-32B and QoQ-Med-32B.")))

# --------------------------------------------------------------------------- deployer guidance
def d_peak(sid):
    """The contamination-corrected (d) peak -- what a deployer provisions."""
    sc = P.get(sid)
    if not sc:
        return None
    a = sc.get("contamination_audit")
    return a["d_clean_peak_gib"] if a else None


def card(gib):
    """Map a measured (d) peak onto the smallest GPU that holds it.
    GPU VRAM marketing 'GB' is GiB (an 'A100 80GB' is 81920 MiB = 80.0 GiB exactly), so the
    comparison is done in GiB. Decimal GB is also printed, labelled, to prevent the unit confusion
    that produced the 71.45/152.82 'GB' figures in the first place."""
    if gib is None:
        return None
    cards = [("24 GB (RTX 4090 / L4)", 24.0), ("32 GB (V100 32GB)", 32.0),
             ("40 GB (A100 40GB)", 40.0), ("48 GB (L40S / RTX A6000)", 48.0),
             ("80 GB (A100 80GB / H100 80GB)", 80.0)]
    fits5 = next((n for n, c in cards if gib <= 0.95 * c), None)
    fits0 = next((n for n, c in cards if gib <= c), None)
    return dict(measured_peak_gib=round(gib, 4),
                measured_peak_gb_decimal=round(gib * 1024 ** 3 / 1e9, 2),
                smallest_card_with_5pct_headroom=fits5 or "does not fit an 80 GiB card with 5% headroom",
                smallest_card_that_holds_it_at_all=fits0 or "does not fit an 80 GiB card",
                pct_of_one_A100_80GB=round(100.0 * gib / 80.0, 1),
                headroom_on_A100_80GB_gib=round(80.0 - gib, 2))


P_final = {}
for k, v in P.items():
    aud, clean = contamination_audit(v)
    if aud:
        v["contamination_audit"] = aud
        rows = [r for r in v.get("rows", []) if "error" not in r and "d_board_used_gib" in r]
        for r, c in zip(rows, clean):
            r["d_process_footprint_clean_gib"] = c
        # The raw per-item d_process_footprint_gib is 0.0 for every scenario measured BEFORE the
        # nvml-pid fallback was added (S1-S4 and the aborted S5): NVML per-process returned 0. Move
        # that dead field out of the way so nobody can quote a zero, and publish the clean figure.
        v["d_nvml_per_process_gib_UNAVAILABLE"] = v.pop("d_process_footprint_gib", None)
        v["d_process_footprint_gib"] = agg(rows, "d_process_footprint_clean_gib")
        v["d_process_footprint_note"] = (
            "= (c) + the CUDA-context offset measured on this scenario's own uncontaminated items. "
            "NVML per-process usedGpuMemory is unavailable in this container (host-pid mismatch), so "
            "the board reading is used and corrected -- see contamination_audit.")
    v["per_source_breakdown"] = per_source(v)
    P_final[k] = v

meta_env = dict(
    date="2026-08-11",
    host="dual A100 80GB PCIe, driver 550.54.15, CUDA 13.0",
    gpu_state_before_run="both GPUs 13 MiB used, 0% util -- logs/vram_testtime_pre_run_nvidia_smi.txt",
    gpu_exclusivity=("one measured process pinned per GPU (CUDA_VISIBLE_DEVICES). Both cards were "
                     "idle (13 MiB) at launch, but a SECOND TENANT appeared partway through and "
                     "touched S4 and S5; S1/S2/S3 are clean. (a)/(b)/(c) are torch-internal and "
                     "unaffected; (d) is corrected per scenario. Full account: "
                     "conventions.d_process_footprint.a_second_tenant_DID_appear_partway_through and "
                     "each scenario's contamination_audit."),
    framework="HuggingFace transformers (NEVER vLLM)",
    torch="2.9.0a0+50eac811a6.nv25.09", transformers="4.55.2", peft="0.14.0",
    dtype="bfloat16", attn_implementation="flash_attention_2 (flash_attn 2.7.4.post1)",
    tensor_parallel=1, batch_size=1,
    code="src/cascade/measure_testtime_vram.py (+ this aggregator src/cascade/vram_testtime_report.py)",
    logs=["logs/vram_testtime_small_2026-08-11.log", "logs/vram_testtime_big_2026-08-11.log",
          "logs/vram_testtime_s5rerun_2026-08-11.log"],
    relation_to_existing_code=("the memory instrument of src/cascade/measure_single_leg.py:112-119, "
                               "generalized to the Lingshu era + LoRA verifier + best-of-8 and extended "
                               "with peak RESERVED and the whole-process footprint. measure_single_leg.py "
                               "is untouched (it is the June reproducibility anchor)."),
    scope=("an ESTIMATION, as asked: 5 scenarios, 12-15 representative items each, ~1 h of GPU time. "
           "NOT a sweep. Items were chosen to bracket the driver space (image pixels 7,659 -> 9,107,712; "
           "1 -> 6 images per item; question length 20 -> 1,309 chars), because VRAM peaks are set by "
           "the largest image and the longest sequence, not by the mean."))

NOT_MEASURED = [
    {"item": "batch sizes > 1", "reason": "out of scope: the deployed cascade is batch-1 (single-query serving). Peaks scale roughly linearly in batch for activations and KV; do not extrapolate from this file."},
    {"item": "tensor-parallel (tp=2) HF footprint", "reason": "tp=1 answers the deployer question 'does it fit on one card'. The repo's own 32B runs used vLLM tp=2 for throughput; per-GPU footprint at tp=2 is roughly half the weights plus full activations, but it was NOT measured here."},
    {"item": "the 32B leg inside the OPEN-TEXT arm (escalation target)", "reason": "cut for time. Its footprint is S2's (same model, same batch-1 decode); only the prompt differs. Stated as an inference, not a measurement."},
    {"item": "co-residency of the 7B and 32B legs on ONE card", "reason": "cut for time. Additive from (d): S1 peak + S2 peak. The June cascade ran them on SEPARATE cards (logs/rt_cascade.log)."},
    {"item": "INT4 / AWQ quantized strong leg", "reason": "the AWQ shards were never downloaded (pre-existing project gap, method_inventory_2026-08-11.json). Still unmeasured."},
    {"item": "resolution-cap ablation of VRAM", "reason": "only the DEPLOYED caps were measured (MedEvalKit full-res 12,845,056 px for the MCQ arm; cap320 250,880 px for the open generator; 1,003,520 px for the verifier). The per-source breakdown does show the resolution effect indirectly, since PMC-VQA images are ~0.01-0.76 MP and MedXpert up to 9.1 MP."},
    {"item": "energy / latency / power", "reason": "this task was VRAM only. src/cascade/measure_single_leg.py owns latency and energy."},
    {"item": "sustained multi-hour footprint / allocator drift", "reason": "each scenario is 12-15 items. Fragmentation over a long serving run was not characterised."},
]

s1p = P_final["S1_lingshu7b_direct_mcq"]["per_source_breakdown"]
s2p = P_final["S2_lingshu32b_direct_mcq"]["per_source_breakdown"]
s5p = P_final["S5_lingshu32b_genuine_reasoning"]["per_source_breakdown"]


def pk(bd, src):
    return bd[src]["d_process_footprint_gib"]["peak"]


KEY_FINDINGS = {
    "1_reasoning_is_almost_FREE_in_VRAM": {
        "claim": ("Turning on genuine reasoning costs +0.09 GiB of footprint. VRAM is NOT the reason "
                  "reasoning is expensive -- latency and energy are."),
        "evidence": (f"S2 (direct, 3 generated tokens) vs S5 (genuine reasoning, mean 259.1 generated "
                     f"tokens) on the SAME 15 items at the SAME resolution: "
                     f"(d) peak {pk(s2p,'medxpert_mm_test')} -> {pk(s5p,'medxpert_mm_test')} GiB on "
                     f"MedXpert and {pk(s2p,'pmc_vqa_test_2')} -> {pk(s5p,'pmc_vqa_test_2')} GiB on "
                     f"PMC-VQA; (b) peak 67.868 -> 67.8736 GiB, i.e. +0.006 GiB."),
        "why": ("the peak is set by the PREFILL of a long multimodal prompt (up to 46,816 vision tokens "
                "/ 11,944 input tokens), not by the decode. ~255 extra KV entries on a 32B model is "
                "tens of MB against a ~5 GiB prefill activation peak."),
        "consequence_for_the_repo": ("the circulating 'think adds 1.26 GB' (152.82 -> 154.08) is NOT a "
                                     "KV-cache measurement -- both figures are --gpu_mem pool "
                                     "reservations. The real number is +0.09 GiB.")},
    "2_the_driver_is_the_IMAGE_not_the_model_and_not_the_answer": {
        "claim": "peak VRAM is set by vision-token count. Within a tier it moves ~6 GiB across the suite.",
        "evidence": (f"7B: PMC-VQA (<=3,968 vision tokens) peaks at {pk(s1p,'pmc_vqa_test_2')} GiB, "
                     f"MedXpert (up to 46,816) at {pk(s1p,'medxpert_mm_test')} GiB. "
                     f"32B: {pk(s2p,'pmc_vqa_test_2')} -> {pk(s2p,'medxpert_mm_test')} GiB. "
                     "The MCQ arm runs at MedEvalKit's full resolution (12,845,056 px cap), so a 9.1 MP "
                     "MedXpert image is not down-sampled."),
        "consequence": ("provision for the tail, or cap max_pixels. A resolution cap is the direct VRAM "
                        "lever and it was never characterised -- see not_measured.")},
    "3_the_verifier_is_nearly_free_and_the_open_arm_costs_ONE_7B": {
        "claim": ("the clean LoRA verifier adds 0.1961 GiB of weights to Lingshu-7B, and the whole "
                  "best-of-8 open-text arm -- generator AND verifier -- fits in 18.76 GiB."),
        "evidence": ("S3: PeftModel.from_pretrained on the already-loaded base moves "
                     "torch.cuda.memory_allocated by +0.1961 GiB (584 LoRA params, 192 of them on "
                     "visual.*). S4 runs 8-sample generation then 8 verifier scoring passes in ONE "
                     "process and peaks at (b) 16.6581 / (c) 17.3789 / (d) 18.7644 GiB."),
        "why_it_matters": ("the verifier is a LoRA adapter ON the generator's own base, so generator and "
                           "verifier SHARE one copy of the weights. The open-text arm costs one 7B plus "
                           "0.2 GiB, not two 7Bs. It is the CHEAPEST configuration measured here -- "
                           "cheaper than the 7B MCQ leg, because the open arm runs cap320 (250,880 px) "
                           "while the MCQ arm runs uncapped."),
        "which_phase_drives_the_S4_peak": ("the GENERATOR phase on 10 of 12 items, the verifier on 2 "
                                           "(per-row b_peak_allocated_generator_phase_gib vs "
                                           "b_peak_allocated_gib). Generating 8 sequences at once "
                                           "costs more than scoring them one at a time, even though "
                                           "the verifier runs at 4x the generator's pixel cap."),
        "the_192_visual_modules": ("independently confirms the repo's standing warning: the adapter "
                                   "really does carry 192 visual.* LoRA modules, which vLLM 0.9.0.1 "
                                   "silently drops. HF is mandatory here, and this run is HF.")},
    "4_the_32B_fits_on_one_80GB_card_but_only_just": {
        "claim": f"always-32B-direct peaks at {d_peak('S2_lingshu32b_direct_mcq')} GiB = 90.8% of one A100 80GB.",
        "evidence": "S2, tp=1, HF, bf16, batch-1, measured on a single A100 80GB.",
        "caveat": ("7.4 GiB of headroom on the worst item in the pool. Batch >1, a larger image than "
                   "the 9.1 MP seen here, or any co-tenant will not fit. The repo's own vLLM runs used "
                   "tp=2, which is a throughput choice, not a capacity requirement.")}}

art = dict(
    _meta=dict(
        title="Test-time VRAM for the deployed Lingshu cascade legs, measured under HF transformers",
        created="2026-08-11",
        why=("VRAM was essentially unmeasured in this project: 160 of 229 method rows in "
             "results/cascade_methods/artifacts/method_inventory_2026-08-11.json carry no VRAM, the IEEE "
             "paper contains none, and the nine figures that circulate are vLLM memory-pool RESERVATIONS "
             "set by a --gpu_mem flag. This artifact is the first per-quantity, per-item measurement of "
             "the current-era legs."),
        supersedes_nothing=("no existing number is retracted here. The vLLM reservations remain correct AS "
                            "RESERVATIONS; this artifact says what they are and adds the footprint they "
                            "were being mistaken for."),
        environment=meta_env),
    conventions=CONV,
    scenarios=P_final,
    reconciliation=RECON,
    deployer_guidance=dict(
        rule="provision (d) process_footprint peak, not (a), (b) or (c), and not the vLLM reservation.",
        cheap_leg_alone=card(d_peak("S1_lingshu7b_direct_mcq")),
        full_opentext_arm=card(d_peak("S4_opentext_bestof8_full_arm")),
        always_32b_direct=card(d_peak("S2_lingshu32b_direct_mcq")),
        thirtytwob_with_reasoning=card(d_peak("S5_lingshu32b_genuine_reasoning")),
        caveat=("these peaks are set by the WORST item in a 15-item pool spanning the whole suite "
                "(MedXpert, up to 9.1 MP and 6 images per question). See per_source_breakdown in each "
                "scenario: on PMC-VQA-only traffic -- 79.2% of the paper's sample-weighted pool -- the "
                "peak is materially lower.")),
    key_findings=KEY_FINDINGS,
    not_measured=NOT_MEASURED)

json.dump(art, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
for k, v in P_final.items():
    if not v.get("b_peak_allocated_gib"):
        continue
    print(f"  {k:<42} n={v['n']:>2}  a={v['meta'].get('load',{}).get('a_weights_resident_gib')}  "
          f"b={v['b_peak_allocated_gib']['peak']}  c={v['c_peak_reserved_gib']['peak']}  "
          f"d={v['d_process_footprint_gib']['peak'] if v.get('d_process_footprint_gib') else None}")
