#!/usr/bin/env python3
"""
shrink_strong_leg_assemble.py -- ATTACK 3 final artifact.

Assembles the three parts into results/cascade_methods/artifacts/shrink_strong_leg_2026-08-12.json.
Re-runnable: every GPU stage is resumable and writes its own file into _shrink_parts/, so
re-running this picks up whatever has landed since.  Anything not yet measured is emitted as
"not measured" -- never estimated, never interpolated (CRITICAL RULE 7).

    python3 src/cascade_methods/shrink_strong_leg_assemble.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "results/cascade_methods/artifacts/_shrink_parts")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/shrink_strong_leg_2026-08-12.json")
VEC = os.path.join(ROOT, "results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz")

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
GIB = 1024.0 ** 3
NOT_MEASURED = "not measured"


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    foot = jload(os.path.join(P, "footprint.json"))
    inv = jload(os.path.join(P, "invert_constraint.json"))
    cost = jload(os.path.join(P, "strong_leg_cost.json"))          # part 4 (PRIMARY objective)
    paired = jload(os.path.join(P, "quant_acc_paired.json"))       # paired accuracy analysis
    vram = {os.path.basename(f)[5:-5]: jload(f) for f in sorted(glob.glob(os.path.join(P, "vram_*.json")))}
    accs = {os.path.basename(f)[4:-5]: jload(f) for f in sorted(glob.glob(os.path.join(P, "acc_*.json")))}

    z = np.load(VEC)
    per_cell_32b = {c: float(z["%s|always_32b_direct" % c].mean()) for c in CELLS}
    per_cell_7b = {c: float(z["%s|always_7b" % c].mean()) for c in CELLS}

    R = {k: v["R_vs_lingshu_7b"] for k, v in foot["per_pass_flops"].items()}
    R32_derived = R["lingshu_32b"]
    R32_as_charged = 4.57

    # ---- footprint + cost table, both R32 conventions -------------------------------------
    # AS-CHARGED is a convention of LITERALS, not a formula: the project fixes Lingshu-7B at 1.0
    # and Lingshu-32B at 4.57.  A candidate with no published literal (Lingshu-I-8B) is charged
    # by its DERIVED ratio to the 7B, since the convention pins the 7B at 1.0.  Dividing the
    # derived 32B ratio by 4.57 would wrongly make the baseline 0.835x itself.
    as_charged = {"lingshu_7b": 1.0, "lingshu_32b": R32_as_charged,
                  "qwen25vl_32b_awq": R32_as_charged, "lingshu_i8b": R["lingshu_i8b"]}

    table = {}
    for k, f in foot["footprint"].items():
        r = R[k]
        table[k] = dict(
            architecture=f["architectures"][0],
            logical_params=f["logical_params"],
            params_B=round(f["logical_params"] / 1e9, 3),
            weight_bytes=f["weight_bytes"],
            weight_gib=f["weight_gib"],
            stored_precision=("AWQ 4-bit (group 128, visual kept bf16)"
                              if f.get("quantization_config") else "bf16"),
            per_pass_gflops=foot["per_pass_flops"][k]["gflops"],
            R_vs_lingshu_7b=r,
            # Both conventions, as the round requires.  DERIVED: the analytic FLOP model, which
            # reproduces the project's published R32=3.816 exactly (see null_tests below).
            # AS-CHARGED: the project's paper literal, which fixes 7B=1.0 and 32B=4.57; a
            # candidate's as-charged share is its derived ratio to the 7B divided by 4.57.
            x_of_always_32b_direct_DERIVED_R32_3p816=round(r / R32_derived, 4),
            as_charged_cost=as_charged[k],
            x_of_always_32b_direct_AS_CHARGED_R32_4p57=round(as_charged[k] / R32_as_charged, 4),
        )

    # ---- measured VRAM ---------------------------------------------------------------------
    vram_tbl = {}
    for k, v in vram.items():
        if not v:
            continue
        vram_tbl[k] = dict(
            label=v["label"],
            resident_mib_during_generation=v["vram_mib"]["torch_peak_alloc_during_gen"],
            resident_gib_during_generation=round(
                v["vram_mib"]["torch_peak_alloc_during_gen"] / 1024.0, 3),
            reserved_mib_during_generation=v["vram_mib"]["torch_peak_reserved_during_gen"],
            load_transient_peak_mib=v["vram_mib"]["torch_alloc_after_load"],
            latency_ms_batch1_mean=v["latency_ms_batch1"]["mean"],
            latency_ms_batch1_median=v["latency_ms_batch1"]["median"],
            gen_tokens_mean=v["gen_tokens_mean"],
            load_seconds=v["load_seconds"],
            n_timed=v["latency_ms_batch1"]["n"],
        )
    for want in ["bf16", "int8", "nf4", "bf16_7b"]:
        if want not in vram_tbl:
            vram_tbl[want] = (NOT_MEASURED + " -- queued behind two other rounds' GPU jobs; the "
                              "run waits for free VRAM rather than oversubscribing, and will "
                              "write _shrink_parts/vram_%s.json when a card frees" % want)
    # ---- an anomaly I can see but cannot yet explain: flag it, do not dress it up -----------
    # UPDATED 2026-08-12 after the bf16 Lingshu-32B control landed: the anomaly is NOT a
    # property of "the bf16 path".  It is confined to the 7B arm.
    anomaly = None
    b7 = vram_tbl.get("bf16_7b") if isinstance(vram_tbl.get("bf16_7b"), dict) else None
    b32 = vram_tbl.get("bf16") if isinstance(vram_tbl.get("bf16"), dict) else None
    if b7:
        expect_mib = foot["footprint"]["lingshu_7b"]["weight_bytes"] / 2 ** 20
        e32 = foot["footprint"]["lingshu_32b"]["weight_bytes"] / 2 ** 20
        anomaly = dict(
            what="the bf16 Lingshu-7B VRAM reading is ~2x that model's own weight bytes.  It is "
                 "CONFINED TO THE 7B ARM: the bf16 Lingshu-32B control reproduces its "
                 "safetensors header almost exactly.",
            lingshu_7b=dict(measured_resident_mib=b7["resident_mib_during_generation"],
                            expected_weight_mib=round(expect_mib, 1),
                            ratio=round(b7["resident_mib_during_generation"] / expect_mib, 3)),
            lingshu_32b=(dict(measured_resident_mib=b32["resident_mib_during_generation"],
                              expected_weight_mib=round(e32, 1),
                              ratio=round(b32["resident_mib_during_generation"] / e32, 4))
                         if b32 else NOT_MEASURED),
            checks_done=[
                "torch_dtype=torch.bfloat16 IS honoured: loading the same checkpoint with "
                "device_map='cpu' gives 729/729 parameters in torch.bfloat16 and exactly "
                "16,584,333,312 bytes = 15.445 GiB, matching the safetensors headers.",
                "batch-1 activations cannot account for it: the probe uses one 320x320 image "
                "(~121 Qwen2.5-VL image tokens) and generates 2-3 tokens.",
            ],
            hypothesis_FALSIFIED=(
                "The earlier leading hypothesis -- 'the bf16 path passes device_map=<device "
                "string> while the quantised path passes device_map={\"\": idx}, and only the "
                "string path inflates' -- is now REFUTED.  build_model takes the SAME string "
                "branch for bf16_7b and for bf16 (Lingshu-32B), and the 32B lands at ratio "
                "%s, not ~2x.  Whatever inflates the 7B arm is specific to that checkpoint or "
                "that probe invocation, not to the device_map form."
                % (round(b32["resident_mib_during_generation"] / e32, 4) if b32 else "?")),
            independent_cross_check=(
                "artifacts/vram_testtime_2026-08-11.json, a separate and more careful "
                "instrument, measures Lingshu-7B under HF at 17.4616-23.4206 GiB PEAK PROCESS "
                "FOOTPRINT across the suite -- i.e. nowhere near 30.5 GiB of live weights.  "
                "That artifact, not this probe, is the authority on the cheap leg's VRAM."),
            consequence="NO CONCLUSION IN THIS ARTIFACT DEPENDS ON IT.  The FLOP-invariance "
                        "result and every params/GiB figure come from safetensors headers, not "
                        "from the GPU.  The two figures this attack actually quotes -- bf16 "
                        "Lingshu-32B and NF4 Lingshu-32B -- are both independently corroborated: "
                        "bf16 reproduces its header to %s, and NF4's 19.53 GiB matches an "
                        "analytic 4-bit-LM-linears + bf16-embed/lm_head/vision prediction "
                        "(~18.7 GiB) and the AWQ checkpoint's own header (19.2844 GiB)."
                        % (round(b32["resident_mib_during_generation"] / e32, 4) if b32 else "?"),
            status="OPEN, and scoped: do not quote the bf16_7b resident figure.  The bf16 32B "
                   "and NF4 32B figures are sound.")

    nvml_note = ("NVML PER-PROCESS attribution is IMPOSSIBLE in this container: nvidia-smi "
                 "reports PIDs from a different PID namespace (e.g. 3964324) than the job's own "
                 "(e.g. 1472818), so no process footprint can be matched.  Total-GPU memory is "
                 "also unusable because two other rounds share both cards.  The numbers above "
                 "are therefore the TORCH ALLOCATOR's, which excludes the ~300-500 MiB CUDA "
                 "context -- stated, not silently omitted.")

    # ---- verdicts ---------------------------------------------------------------------------
    bar = inv["L2b_error_correlation_bracket"]["bar"]
    base_macro = inv["baseline"]["always_32b_direct_macro"]

    # cost figures pulled from part 4 so the verdict text and the table can never drift apart
    def _cx(conv, leg, arm):
        try:
            r = cost["macro_cost_when_strong_leg_is_swapped"][conv][leg]
            return (r[arm]["macro_x_of_always_32b_direct"] if arm != "alone"
                    else r["always_this_leg_alone"]["macro_x_of_always_32b_direct"])
        except Exception:
            return float("nan")

    ac_accmax = _cx("as_charged_R32_4p57", "lingshu_32b", "method_accuracy_max_veto")
    ac_i8b_alone = _cx("as_charged_R32_4p57", "lingshu_i8b", "alone")
    de_i8b_alone = _cx("derived_R32_3p816", "lingshu_i8b", "alone")

    quant_measured = {k: v for k, v in vram_tbl.items() if isinstance(v, dict)}
    nf4 = quant_measured.get("nf4")
    bf16 = quant_measured.get("bf16")

    v1 = dict(
        lever="1. QUANTISE THE 32B",
        flop_verdict=("NO EFFECT ON THE PRIMARY OBJECTIVE, and this is structural, not empirical. "
                      "The AWQ checkpoint's logical parameter count is IDENTICAL to Lingshu-32B's "
                      "(33,452,718,336 both, read from the safetensors headers of each), so a "
                      "forward pass performs exactly the same multiply-accumulates: R stays "
                      "%.4f.  Nor is there a throughput credit: the A100 DOES expose INT4 "
                      "tensor-core ops, but reaching them needs BOTH operands quantised (W4A4), "
                      "whereas AWQ and bitsandbytes-NF4 are W4A16 -- weights are dequantised to "
                      "bf16/fp16 and multiplied by 16-bit activations on the same 16-bit tensor "
                      "cores.  The 4-bit storage buys bandwidth, not arithmetic." % R32_derived),
        memory_verdict=(
            "REAL ON THE WEIGHTS, AND NOW QUANTIFIED FOR THE FIRST TIME.  Lingshu-32B bf16 weights are %.4f GiB "
            "(%d B, index total_size).  Measured resident during generation: NF4 "
            "%s GiB, INT8 %s GiB." % (
                foot["footprint"]["lingshu_32b"]["weight_gib"],
                foot["footprint"]["lingshu_32b"]["weight_bytes"],
                nf4["resident_gib_during_generation"] if nf4 else NOT_MEASURED,
                quant_measured["int8"]["resident_gib_during_generation"]
                if "int8" in quant_measured else NOT_MEASURED)),
        latency_verdict=(
            "QUANTISATION MAKES THE STRONG LEG SLOWER, MEASURED AGAINST ITS OWN MATCHED bf16 "
            "CONTROL.  Same probe, same 25 fixed prompts, same batch-1 greedy decoding, same "
            "process shape; only the weight representation differs.  Mean batch-1 latency: "
            "bf16 %s ms, NF4 %s ms (%sx SLOWER), INT8 %s ms (%sx SLOWER).  This CONTRADICTS "
            "artifacts/quantized_strong_leg.json's projected int4_latency_ratio of 0.8763 "
            "(a claimed 12%% speed-UP): the measured NF4 ratio is %sx in the opposite "
            "direction.  Mechanism: W4A16 dequantises every weight tile to bf16 before the "
            "GEMM, so a prefill-bound pass pays the dequant on top of unchanged arithmetic; "
            "bandwidth relief cannot pay for it when the pass is compute-bound, and these "
            "passes are (mean generated tokens %s)."
            % (bf16["latency_ms_batch1_mean"] if bf16 else NOT_MEASURED,
               nf4["latency_ms_batch1_mean"] if nf4 else NOT_MEASURED,
               round(nf4["latency_ms_batch1_mean"] / bf16["latency_ms_batch1_mean"], 2)
               if (nf4 and bf16) else "?",
               quant_measured["int8"]["latency_ms_batch1_mean"] if "int8" in quant_measured
               else NOT_MEASURED,
               round(quant_measured["int8"]["latency_ms_batch1_mean"]
                     / bf16["latency_ms_batch1_mean"], 1)
               if ("int8" in quant_measured and bf16) else "?",
               round(nf4["latency_ms_batch1_mean"] / bf16["latency_ms_batch1_mean"], 2)
               if (nf4 and bf16) else "?",
               bf16["gen_tokens_mean"] if bf16 else "?")),
        accuracy_verdict=(accs.get("nf4") and "see quantised_accuracy below")
                         or (NOT_MEASURED + " -- queued behind the VRAM stage"),
        corrects_prior_claim=(
            "artifacts/quantized_strong_leg.json (2026-07-07) called an INT4 strong leg 'a real "
            "VRAM win' with ZERO measured numbers (its own tractability_note records that the "
            "AWQ shards never downloaded).  The AWQ shards are now complete on disk and the "
            "memory claim is CONFIRMED and quantified; its companion latency projection "
            "(int4_latency_ratio 0.8763, i.e. 12% FASTER) is CONTRADICTED by measurement."),
    )

    v2 = dict(
        lever="2. A SMALLER STRONG LEG (Lingshu-I-8B)",
        exists=("CONFIRMED ON DISK, and it is NOT what the literature note implied.  "
                "Lingshu-I-8B is InternVLForConditionalGeneration (model_type 'internvl'), NOT "
                "the Qwen2.5-VL architecture every other Lingshu here uses: InternViT-300M vision "
                "tower + a Qwen2 text backbone.  Consequence: vLLM 0.9.0.1 cannot serve it at "
                "all, so it cannot be scored on the stack every published cell used."),
        footprint=("7,944,373,760 params / 14.7975 GiB bf16 -- SMALLER than the deployed cheap "
                   "leg (Lingshu-7B: 8,292,166,656 / 15.4454 GiB), and 4.21x smaller than "
                   "Lingshu-32B (33,452,718,336 / 62.3105 GiB)."),
        surprise=("Its language model is byte-identically sized to Lingshu-7B's -- lm_body is "
                  "6,525,621,760 parameters in BOTH.  The entire difference is the vision tower "
                  "(InternViT 304,012,288 vs Qwen2.5-VL 631,975,680) and the merger/projector. "
                  "So it is not 'a bigger 8B'; it is the same 7B language model behind a "
                  "cheaper, differently-trained visual front end."),
        cost=("R = 0.8179 vs Lingshu-7B, i.e. CHEAPER PER PASS THAN THE CURRENT CHEAP LEG "
              "(fixed 256 image tokens from crop_to_patches=false, and a 2.1x smaller tower). "
              "As always-I-8B that is 0.2143x always-32B-direct (derived R32) / 0.1790x "
              "(as-charged R32)."),
        accuracy=(paired or {}).get("C_partial_macro_i8b_vs_32b", NOT_MEASURED),
        accuracy_summary=(
            "MEASURED, on 4 of the 8 reporting cells, by re-analysing the concurrent round's "
            "(ATTACK A) per-item outputs against always-32B-direct's stored per-item outputs.  "
            "I did not duplicate that GPU work.  Per cell, I-8B minus always-32B-direct: "
            "SLAKE_closed +0.0658 [+0.0443,+0.0873] WIN, PATH_VQA_closed +0.0425 "
            "[+0.0324,+0.0526] WIN, VQA_RAD_closed -0.0478 [-0.0996,+0.0040] n.s., "
            "MedXpertQA-MM -0.0190 [-0.0425,+0.0045] n.s.  4-cell partial macro +0.0104 "
            "[-0.0050,+0.0251] -- A TIE, NOT A WIN.  Item alignment asserted row by row on "
            "(question, gold) for every cell: 0 mismatches.  "
            "THE OTHER 4 CELLS ARE NOT MEASURED AND ARE NOT FILLED IN: PMC_VQA is VOID under "
            "the HF driver (all 33,430 responses blank, mean gen_tokens 0.0 -- which also voids "
            "the PMC_VQA row of artifacts/lingshu_i8b_cheapleg_2026-08-11.json), and the three "
            "OPEN cells were scored without the LLM judge, so both arms read exactly 0.0000 "
            "there and the rows carry no information."),
        prior_expectation=(
            "The vendor table [V*, LITERATURE_UPDATE_2026-08-11.md 5.1] puts Lingshu-I-8B's "
            "7-benchmark average at 64.5 vs Lingshu-32B's 66.6 -- 2.1 points BELOW the model it "
            "would have to replace, with PathVQA +13.0 and SLAKE +8.5 but MMMU-Med -4.9.  That "
            "is a DIFFERENT harness from our 8-cell macro and must not be cross-multiplied; it "
            "is quoted only as a prior on plausibility.  Against the requirement derived below "
            "(a replacement must land within -0.0149..+0.0009 macro of the 32B), a 2.1-point "
            "deficit on the vendor's own scale is a strong prior that I-8B does NOT hold the tie "
            "as a REPLACEMENT strong leg -- while remaining a very promising CHEAP leg, which is "
            "exactly what the concurrent round is testing."),
    )

    v3 = dict(
        lever="3. INVERT THE CONSTRAINT",
        question="What accuracy must a strong leg reach, at a given size, to hold the tie?",
        answer_bracket=bar,
        lambda_star=inv["L2_requirement_curve"]["lambda_star"],
        interpretation=(
            "Deployed ALONE, a replacement strong leg must recover %.1f%% of every error "
            "always-32B-direct fixes over Lingshu-7B before the CI lower bound clears -0.0029. "
            "Recovering 90%% is NOT enough (macro 0.6504, delta -0.0062 [-0.0104,-0.0024], "
            "FAILS).  In accuracy terms the bar is 8-cell macro >= %.4f if the candidate's "
            "errors are highly correlated with the 32B's, rising to >= %.4f if they are "
            "independent of it -- and %.4f is ABOVE the baseline's own %.4f."
            % (100 * inv["L2_requirement_curve"]["lambda_star"],
               bar["correlated_bar"], bar["independent_bar"],
               bar["independent_bar"], base_macro)),
        the_structural_consequence=(
            "THIS IS THE MOST USEFUL RESULT OF THE ATTACK.  Because the test is PAIRED, the "
            "width of the CI depends on how the candidate's errors correlate with the "
            "baseline's.  A near-clone of the 32B gets a narrow CI and needs only to match it; "
            "an INDEPENDENT model of the same accuracy gets a wider CI and must BEAT it by "
            "+0.0149 to be certified 'not worse'.  So the pre-registered constraint admits "
            "essentially two shapes: (a) something that IS the 32B computationally -- a "
            "quantised or distilled copy, which saves memory but no FLOPs; or (b) a genuinely "
            "better model, which is a capability result, not a compute result.  A merely-"
            "comparable smaller model of different lineage cannot pass, however cheap it is."),
        per_cell_targets={c: dict(
            always_32b_direct=round(per_cell_32b[c], 4),
            always_7b=round(per_cell_7b[c], 4),
            gap_the_replacement_must_close=round(per_cell_32b[c] - per_cell_7b[c], 4))
            for c in CELLS},
        caveat=("The M(lambda) family is a SIMULATED interpolation of the two MEASURED arms' "
                "per-item outcomes; lambda=0 and lambda=1 reproduce always-7B and "
                "always-32B-direct exactly (asserted in code).  It is a REQUIREMENT CURVE, not "
                "a measurement of any real model.  It also inverts the constraint for the "
                "ALWAYS-S deployment, which is the cheapest possible one; a 7B->S cascade needs "
                "S to be at least as good, and per-item escalation masks are not stored in "
                "vec_disjoint.npz so the cascade variant could not be recomposed here."),
    )

    out = dict(
        title="ATTACK 3 -- SHRINK THE STRONG LEG: quantisation, a smaller strong leg, and the "
              "inverted non-inferiority constraint",
        date="2026-08-12",
        objective="MINIMISE macro FLOP-eq SUBJECT TO paired-bootstrap CI lower bound of "
                  "(policy - always-32B-direct) on the 8-cell macro >= -0.0029",
        reproduce=[
            "python3 src/cascade_methods/shrink_strong_leg_footprint.py",
            "python3 src/cascade_methods/shrink_invert_constraint.py",
            "python3 src/cascade_methods/shrink_quantised_strong_leg.py --stage vram "
            "--configs nf4,int8,bf16_7b,bf16",
            "runners/run_shrink_quant_acc.sh",
            "python3 src/cascade_methods/shrink_strong_leg_assemble.py",
        ],
        no_fabricated_numbers=True,
        null_tests=dict(
            N1_macro=inv["null_test"],
            N2_published_tie=inv["published_tie_reproduced"],
            N3_flop_model=dict(
                statement="the analytic FLOP model rebuilt here reproduces the project's "
                          "published DERIVED R32 exactly",
                rebuilt=R32_derived, published=3.816,
                abs_dev=round(abs(R32_derived - 3.816), 6),
                source="artifacts/flop_ratio_derivation_2026-08-03.json:derived_ratio.R32_derived",
                verdict="PASS"),
            N4_serving_stack=(paired or {}).get("N4_serving_stack_null_test", NOT_MEASURED),
            N4b_serving_stack_on_lingshu_7b=(paired or {}).get(
                "N4b_serving_stack_measured_on_lingshu_7b", NOT_MEASURED),
            N4b_verdict=(
                "FAIL, AND THE FAILURE IS INFORMATIVE.  The HF driver does NOT reproduce the "
                "vLLM stack every published cell came from: on identical Lingshu-7B weights it "
                "scores -0.0708 [-0.0830,-0.0589] on PATH_VQA_closed (significant) and voids "
                "PMC_VQA entirely (33,430 blank responses).  It DOES agree on VQA_RAD_closed "
                "(-0.0040, n.s.) and MedXpertQA-MM (+0.0015, n.s.).  Consequence: every "
                "cross-stack number in this artifact is labelled as such, and the direction of "
                "the confound is worked out in paired_accuracy -> D_cross_stack_sensitivity "
                "(it runs AGAINST Lingshu-I-8B, so the +0.0104 tie is not inflated by it)."),
            N5_cost_decomposition=(cost or {}).get(
                "null_test_N5_reproduces_published_costs", NOT_MEASURED),
            N6_item_alignment=dict(
                statement="every paired comparison asserts, row by row, that the two arms' "
                          "items agree on (question, gold answer) before any delta is taken",
                result="0 mismatches in every compared cell (see paired_accuracy -> each cell's "
                       "alignment_verdict)",
                verdict="PASS"),
        ),
        numerics_pins=dict(OMP_NUM_THREADS="1 (footprint) / 4 (bootstrap)", PYTHONHASHSEED="0",
                           tf32="not applicable -- numpy on stored 0/1 vectors; the GPU stages "
                                "measure memory and wall-clock, not accuracy-critical arithmetic",
                           bootstrap="paired item-level multinomial counts, ONE shared stream "
                                     "reused by every candidate and the baseline, nboot=10000, "
                                     "seed=20260812"),
        PRIMARY_OBJECTIVE_macro_flop_eq=cost or (
            NOT_MEASURED + " -- run src/cascade_methods/shrink_strong_leg_cost.py"),
        paired_accuracy=paired or (NOT_MEASURED),
        footprint_and_cost=table,
        measured_vram_and_latency=dict(
            table=vram_tbl,
            which_number_to_quote=(
                "TWO DIFFERENT NUMBERS, DO NOT CONFLATE THEM.  "
                "resident_*_during_generation is torch.cuda.max_memory_allocated -- LIVE tensor "
                "bytes (weights + activations + KV) -- and it is the honest measure of how much "
                "memory the model NEEDS.  reserved_*_during_generation is "
                "torch.cuda.max_memory_reserved -- what the caching allocator held from the "
                "driver -- and it is what a co-tenant sees.  They agree for bf16 (63,904 vs "
                "63,930 MiB) and for INT8, but NOT for NF4: 20,000.3 MiB live against 32,036 "
                "MiB reserved.  That 12 GiB gap is LOAD-TIME TRANSIENT bf16 buffers that "
                "bitsandbytes allocates while quantising and that the allocator then caches "
                "instead of returning; the probe never calls torch.cuda.empty_cache() after "
                "load.  So NF4's 3.19x saving is real as a memory REQUIREMENT (19.53 GiB, "
                "independently corroborated by the AWQ checkpoint's own header at 19.2844 GiB "
                "and by an analytic 4-bit-LM-linears + bf16-embed/lm_head/vision prediction of "
                "~18.7 GiB), but a deployer who does not empty the cache after load will see "
                "~31 GiB held.  Quote 19.53 GiB as the requirement and say so; quote 31.3 GiB "
                "if the question is what the process occupies as loaded by this code path."),
            nvml_limitation=nvml_note,
                                       bf16_resident_anomaly=anomaly or "n/a (bf16 arms not "
                                       "measured yet)",
                                       workload="25 fixed synthetic 320x320 images + one fixed "
                                                "VQA-RAD-style question, batch 1, greedy, "
                                                "3 warmup passes excluded"),
        quantised_accuracy=dict(
            status=NOT_MEASURED,
            parts=accs or {},
            what_is_running=(
                "runners/run_shrink_quant_acc2.sh.  The NF4 arm IS running (GPU1, VQA_RAD -> "
                "SLAKE -> PATH_VQA, batch 4, ~50 items/min => ~4-5 h for the three datasets).  "
                "The matched bf16 control needs ~62 GiB of weights and is parked on the "
                "driver's own wait_for_vram guard because two other rounds hold both A100s; it "
                "polls every 60 s and loads only when a card has 70 GB free.  Neither job was "
                "allowed to displace another round's process."),
            resume=("both arms are resumable per dataset (metrics.json) and per item "
                    "(ckpts/shrink_quant/<arm>/<DS>/gen.jsonl).  When they finish they write "
                    "_shrink_parts/acc_<arm>.json and per-item results.json; re-running "
                    "src/cascade_methods/shrink_quant_acc_analyze.py then "
                    "src/cascade_methods/shrink_strong_leg_assemble.py fills section A "
                    "(the paired NF4-minus-bf16 delta) and null test N4 with no other change."),
            why_this_does_not_block_the_verdict=(
                "lever 1 is rejected on the PRIMARY objective by a structural argument that "
                "does not involve accuracy at all -- an AWQ/NF4 Lingshu-32B has the identical "
                "logical parameter count and therefore the identical MAC count, so macro "
                "FLOP-eq is unchanged to every decimal -- and on the TERTIARY objective by a "
                "latency measurement that DOES have its matched control (the vram stage ran "
                "bf16, NF4 and INT8 through the same probe).  A quantised strong leg that held "
                "accuracy perfectly would still not save a single FLOP."),
            what_it_would_settle=(
                "only the sub-question 'does a quantised strong leg hold the accuracy tie', "
                "which matters if the project ever wants the 32B to fit on a 24 GB card.  "
                "That is a deployability question, not a compute one."),
        ),
        inverted_constraint=v3,
        levers=dict(lever_1_quantise=v1, lever_2_smaller_strong_leg=v2, lever_3_invert=v3),
        corrections_to_the_brief=dict(
            lingshu_32b_footprint=(
                "The round brief states always-32B-direct weighs '~32.8B params / ~31.5 GiB "
                "bf16'.  MEASURED from the safetensors headers: %d params and %d bytes = %.4f "
                "GiB.  The GiB figure in the brief is too small "
                "by ~2x -- that many parameters at 2 bytes each cannot be 31.5 GiB.  The brief's "
                "Lingshu-7B figures (8.29B / 15.45 GiB, index total_size 16.59e9 B) ARE correct "
                "and reproduce exactly (measured %d params / %d bytes / %.4f GiB), so only the "
                "32B row is affected.  Every footprint comparison in this round should use "
                "%.4f GiB."
                % (foot["footprint"]["lingshu_32b"]["logical_params"],
                   foot["footprint"]["lingshu_32b"]["weight_bytes"],
                   foot["footprint"]["lingshu_32b"]["weight_gib"],
                   foot["footprint"]["lingshu_7b"]["logical_params"],
                   foot["footprint"]["lingshu_7b"]["weight_bytes"],
                   foot["footprint"]["lingshu_7b"]["weight_gib"],
                   foot["footprint"]["lingshu_32b"]["weight_gib"])),
        ),
        VERDICT=dict(
            one_line=(
                "SHRINKING THE STRONG LEG IS THE ONLY ONE OF THE TWO LEVERS THAT TOUCHES THE "
                "PRIMARY OBJECTIVE, AND NEITHER LEVER PRODUCES A CERTIFIED WIN.  Quantisation "
                "changes macro FLOP-eq by EXACTLY ZERO (1.7398x -> 1.7398x as-charged), buys a "
                "3.19x weight-footprint reduction, and COSTS 2.27x batch-1 latency against its "
                "own matched control.  Lingshu-I-8B is the real candidate -- 7.944B params / "
                "14.7975 GiB, R=0.8179, so always-I-8B costs 0.1790x (as-charged) / 0.2143x "
                "(derived) of always-32B-direct and 4.21x less resident weight -- and on the 4 "
                "of 8 cells that could be measured it is +0.0104 [-0.0050,+0.0251] versus "
                "always-32B-direct, i.e. A TIE AT ~1/5 THE COMPUTE.  It is NOT certified: 4 "
                "cells are unmeasured, and the inverted constraint says an independently-erring "
                "replacement must BEAT the 32B by +0.0144 macro, which even a generous "
                "projection (0.6619) does not reach.  Finishing those 4 cells is the highest-"
                "value next measurement this project has."),
            headline=(
                "QUANTISATION IS A FOOTPRINT LEVER THAT COSTS LATENCY AND SAVES NO COMPUTE; A "
                "SMALLER MODEL IS THE ONLY REAL COMPUTE LEVER, AND ONE EXISTS.  "
                "(1) Quantising Lingshu-32B cannot reduce FLOP-eq at all: the AWQ checkpoint's "
                "logical parameter count is IDENTICAL to the bf16 one (33,452,718,336 in both "
                "safetensors headers), so a forward pass performs the same multiply-accumulates "
                "and accuracy-max stays at %.4fx always-32B-direct as-charged, unchanged to "
                "every decimal.  There is no throughput credit either: A100 sm80 has INT4 "
                "tensor cores but they need W4A4, and AWQ/NF4 are W4A16 -- weights are "
                "dequantised to bf16 before the same 16-bit GEMM.  What quantisation DOES buy, "
                "measured here for the first time in this project, is memory: %.4f GiB of "
                "weights -> %s GiB resident (NF4), a %sx reduction, corroborated by the AWQ "
                "checkpoint's own header (19.2844 GiB).  It costs latency: against its own "
                "matched bf16 control on the same probe, NF4 is %sx SLOWER and INT8 %sx SLOWER. "
                "(2) A SMALLER STRONG LEG is the only lever that moves the primary objective, "
                "and Lingshu-I-8B is a serious candidate on BOTH axes: 7,944,373,760 params / "
                "14.7975 GiB (4.21x less resident weight than the 32B, and less than the "
                "CURRENT CHEAP LEG), R=0.8179, so always-I-8B costs %.4fx as-charged / %.4fx "
                "derived.  Measured on the 4 of 8 cells that are usable, it is +0.0104 "
                "[-0.0050,+0.0251] versus always-32B-direct: a TIE at roughly one fifth of the "
                "compute.  Under the round's own rule (cheaper AND tying is a WIN) that is the "
                "most promising result available -- but it is NOT certified, because the "
                "constraint lives on 8 cells and 4 are unmeasured. "
                "(3) The INVERSION is the transferable result and it explains the difficulty: "
                "because the test is PAIRED, a candidate whose errors are correlated with the "
                "32B's need only match it (macro >= %.4f, i.e. -%.4f), while an "
                "INDEPENDENTLY-erring candidate inherits a wider CI and must BEAT it by "
                "+%.4f (macro >= %.4f).  So the constraint as written admits quantised or "
                "distilled COPIES of the 32B -- which save memory but no FLOPs -- and genuinely "
                "BETTER models, which is a capability result; a merely-comparable smaller model "
                "of different lineage cannot pass however cheap it is.  Lingshu-I-8B is exactly "
                "that third kind, which is why a +0.0104 tie on 4 cells is not enough."
                % (ac_accmax, foot["footprint"]["lingshu_32b"]["weight_gib"],
                   nf4["resident_gib_during_generation"] if nf4 else "?",
                   round(foot["footprint"]["lingshu_32b"]["weight_gib"]
                         / nf4["resident_gib_during_generation"], 2) if nf4 else "?",
                   round(nf4["latency_ms_batch1_mean"] / bf16["latency_ms_batch1_mean"], 2)
                   if (nf4 and bf16) else "?",
                   round(quant_measured["int8"]["latency_ms_batch1_mean"]
                         / bf16["latency_ms_batch1_mean"], 1)
                   if ("int8" in quant_measured and bf16) else "?",
                   ac_i8b_alone, de_i8b_alone,
                   bar["correlated_bar"], abs(bar["correlated_bar"] - base_macro),
                   bar["independent_bar"] - base_macro, bar["independent_bar"])),
            best_operating_point_from_this_attack=(
                "always-Lingshu-I-8B: 0.1790x macro FLOP-eq (as-charged R32=4.57) / 0.2143x "
                "(derived R32=3.816) of always-32B-direct, 7,944,373,760 params, 14.7975 GiB of "
                "bf16 weights -- 4.21x less resident weight than the 32B and even less than the "
                "CURRENT CHEAP LEG (Lingshu-7B, 15.4454 GiB).  Measured accuracy on the 4 "
                "usable cells: +0.0104 [-0.0050,+0.0251] versus always-32B-direct, a TIE.  "
                "NOT CERTIFIED -- the constraint is defined on all 8 cells and 4 of them are "
                "unmeasured, so this is the best CANDIDATE, not a validated operating point.  "
                "The runner-up, if the 8-cell macro turns out not to hold, is accuracy-max with "
                "an I-8B strong leg: 1.0771x / 1.2899x, i.e. still below always-32B-direct "
                "as-charged and a 1.62x compute reduction against the shipped 1.7398x."),
            what_would_change_this=(
                "(a) MEASURING THE 4 MISSING CELLS for Lingshu-I-8B -- PMC_VQA needs a re-run "
                "(the HF driver produced 33,430 blank responses) and the three OPEN cells need "
                "the LLM judge that every published open-cell number used.  That is a bounded "
                "job and it is the single highest-value measurement available: it decides "
                "whether a 4.2x smaller model ties the 32B at ~1/5 the compute.  "
                "(b) Relaxing or re-deriving the non-inferiority margin -- at eps = -0.0029 it "
                "is the PAIRED-CI GEOMETRY, not the model, that rejects cheap alternatives, "
                "because an independently-erring candidate inherits a wider CI and must beat "
                "the baseline outright.  That is a decision for the project, not a measurement.  "
                "(c) Hardware with a real INT4 tensor-core path (W4A4), which would convert the "
                "measured memory win into a throughput win; on A100 sm80 it cannot."),
            what_this_attack_RETIRES=[
                "artifacts/quantized_strong_leg.json's projected int4_latency_ratio = 0.8763 "
                "(a claimed 12% speed-UP from INT4).  MEASURED against a matched bf16 control "
                "on the same probe: NF4 is 2.27x SLOWER (936.4 ms vs 411.9 ms) and INT8 27.3x "
                "SLOWER (11,241.9 ms).  The direction of the projection was wrong.",
                "The same file's claim that an INT4 strong leg is 'a real VRAM win' is "
                "CONFIRMED and now quantified for the first time (62.31 -> 19.53 GiB, 3.19x) -- "
                "but it was never a compute win and must stop being cited as one.",
                "The PMC_VQA row of artifacts/lingshu_i8b_cheapleg_2026-08-11.json (0.1323 for "
                "BOTH the I-8B and the base-7B arm, byte-identically) is VOID: every one of the "
                "33,430 responses is blank and mean generated tokens is 0.0.  A blank-response "
                "cell scored by MedEvalKit yields whatever its extraction fallback matches; it "
                "is not an accuracy.",
            ],
        ),
        limitations=[
            "QUANTISED ACCURACY IS " + NOT_MEASURED + ".  The NF4 arm and its matched bf16 "
            "control were re-launched after a driver bug (QuantHFVLM never set crop_to_patches, "
            "so the 2026-08-12 03:29 run wrote 2,545 error rows per arm; fixed, and the "
            "poisoned checkpoints MOVED to ckpts/_failed_shrink_quant_20260812/ rather than "
            "deleted).  The bf16 control needs ~62 GiB of weights and both A100s were taken by "
            "two other rounds, so it is parked on a wait-for-VRAM guard rather than "
            "oversubscribing anyone.  CONSEQUENCE, STATED PLAINLY: this artifact does NOT answer "
            "'does a quantised strong leg hold the accuracy tie'.  It does not need to in order "
            "to reject quantisation as a COMPUTE lever -- that rejection is structural (identical "
            "logical parameter count => identical MACs) and is independent of accuracy.  An "
            "unmatched NF4 accuracy would be uninterpretable anyway: the HF driver is known to "
            "deviate from the vLLM stack every published cell used (see next item), so only a "
            "same-driver bf16 control can isolate the quantisation effect.",
            "SERVING-STACK DEVIATION IS REAL AND IS NOT YET BOUNDED FOR THE 32B.  The concurrent "
            "round's HF Lingshu-7B arm scores 0.7662 on PATH_VQA_closed where the STORED vLLM "
            "Lingshu-7B scores 0.8370 (-0.0708), and its PMC_VQA cell is void.  So the "
            "Lingshu-I-8B-vs-always-32B-direct comparison in this artifact is CROSS-STACK and "
            "its N4 null test (my bf16 32B under the same HF driver) is exactly the measurement "
            "that would bound it -- and that is the measurement blocked above.  Read the "
            "+0.0104 partial-macro tie as PROVISIONAL for this reason, not merely because 4 "
            "cells are missing.",
            "NVML per-process VRAM attribution is impossible in this container (PID namespace "
            "mismatch); torch-allocator figures are reported instead and exclude the CUDA "
            "context.",
            "The latency/VRAM probe uses synthetic 320x320 images so it carries no dataset "
            "dependency; it is NOT an accuracy measurement and is not used as one.",
            "The M(lambda) requirement curve inverts the constraint for the always-S deployment "
            "only; per-item escalation masks are not stored, so the 7B->S cascade variant could "
            "not be recomposed.",
            "Lingshu-I-8B cannot be served by vLLM 0.9.0.1, so any number for it comes from a "
            "different serving stack than every published cell -- the +/-0.008 open-text "
            "reproducibility caveat applies and a matched control is mandatory.",
        ],
    )

    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(dict(footprint_and_cost=table,
                          vram=vram_tbl,
                          bar=bar,
                          verdict=out["VERDICT"]["headline"]), indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
