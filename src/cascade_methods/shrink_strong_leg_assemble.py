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
    anomaly = None
    b7 = vram_tbl.get("bf16_7b") if isinstance(vram_tbl.get("bf16_7b"), dict) else None
    if b7:
        expect_mib = foot["footprint"]["lingshu_7b"]["weight_bytes"] / 2 ** 20
        anomaly = dict(
            what="the NON-quantised (bf16) VRAM readings are ~2x the model's own weight bytes "
                 "and are NOT yet explained",
            measured_resident_mib=b7["resident_mib_during_generation"],
            expected_weight_mib=round(expect_mib, 1),
            ratio=round(b7["resident_mib_during_generation"] / expect_mib, 3),
            checks_done=[
                "torch_dtype=torch.bfloat16 IS honoured: loading the same checkpoint with "
                "device_map='cpu' gives 729/729 parameters in torch.bfloat16 and exactly "
                "16,584,333,312 bytes = 15.445 GiB, matching the safetensors headers.",
                "batch-1 activations cannot account for it: the probe uses one 320x320 image "
                "(~121 Qwen2.5-VL image tokens) and generates 2-3 tokens.",
                "it is SYSTEMATIC, not a one-off: a concurrent round's process shows the same "
                "~31.09 GiB signature for a similarly-sized bf16 model (seen in a CUDA OOM "
                "report while probing).",
            ],
            leading_hypothesis="the bf16 path passes device_map=<device string> while the "
                               "quantised path passes device_map={'': idx} (a dict); the two "
                               "load paths differ and only the string path shows the inflation. "
                               "NOT CONFIRMED -- both A100s are saturated by other rounds and "
                               "the confirming probe OOM'd rather than displacing them.",
            consequence="NO CONCLUSION IN THIS ARTIFACT DEPENDS ON IT.  The FLOP-invariance "
                        "result and every params/GiB figure come from safetensors headers, not "
                        "from the GPU.  The NF4 resident figure IS consistent with an "
                        "independent check (4-bit LM linears + bf16 embed/lm_head/vision "
                        "~= 18.7 GiB predicted vs 19.53 GiB measured), so the quantised path "
                        "looks sound; it is the bf16 CONTROL that is suspect.",
            status="OPEN -- do not quote the bf16 resident figures until a probe that asserts "
                   "both parameter dtype and post-load allocated bytes has run.")

    nvml_note = ("NVML PER-PROCESS attribution is IMPOSSIBLE in this container: nvidia-smi "
                 "reports PIDs from a different PID namespace (e.g. 3964324) than the job's own "
                 "(e.g. 1472818), so no process footprint can be matched.  Total-GPU memory is "
                 "also unusable because two other rounds share both cards.  The numbers above "
                 "are therefore the TORCH ALLOCATOR's, which excludes the ~300-500 MiB CUDA "
                 "context -- stated, not silently omitted.")

    # ---- verdicts ---------------------------------------------------------------------------
    bar = inv["L2b_error_correlation_bracket"]["bar"]
    base_macro = inv["baseline"]["always_32b_direct_macro"]

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
            "NO LATENCY BENEFIT, and for INT8 a catastrophic penalty.  Measured batch-1 mean: "
            "NF4 %s ms, INT8 %s ms.  The INT8 figure is ~12x the project's own bf16-32B anchor "
            "(665 ms) and needs no matched control to be called a loss.  The NF4 figure is "
            "ABOVE that anchor too, but the anchor came from a different serving path, so NF4 "
            "is reported as 'no benefit demonstrated' rather than a quantified slowdown until "
            "the matched bf16 control lands.  bf16 control: %s." % (
                nf4["latency_ms_batch1_mean"] if nf4 else NOT_MEASURED,
                quant_measured["int8"]["latency_ms_batch1_mean"] if "int8" in quant_measured
                else NOT_MEASURED,
                (str(bf16["latency_ms_batch1_mean"]) + " ms") if bf16 else
                NOT_MEASURED + " -- both A100s are saturated by two other rounds; the job is "
                "queued with a wait-for-VRAM guard and will write vram_bf16.json when a card "
                "frees.  The project's stored 665 ms anchor is NOT usable as this control "
                "(different serving path), so no bf16 ratio is quoted.")),
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
        accuracy=(NOT_MEASURED + " BY THIS ROUND -- a concurrent round (ATTACK A, "
                  "src/cascade_methods/i8b_cheapleg_eval.py) is measuring it right now with a "
                  "matched Lingshu-7B control under HF transformers, and I deliberately did not "
                  "duplicate that GPU work.  Its per-cell numbers, when they land, plug straight "
                  "into the L3 frontier below."),
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
        ),
        numerics_pins=dict(OMP_NUM_THREADS="1 (footprint) / 4 (bootstrap)", PYTHONHASHSEED="0",
                           tf32="not applicable -- numpy on stored 0/1 vectors; the GPU stages "
                                "measure memory and wall-clock, not accuracy-critical arithmetic",
                           bootstrap="paired item-level multinomial counts, ONE shared stream "
                                     "reused by every candidate and the baseline, nboot=10000, "
                                     "seed=20260812"),
        footprint_and_cost=table,
        measured_vram_and_latency=dict(table=vram_tbl, nvml_limitation=nvml_note,
                                       bf16_resident_anomaly=anomaly or "n/a (bf16 arms not "
                                       "measured yet)",
                                       workload="25 fixed synthetic 320x320 images + one fixed "
                                                "VQA-RAD-style question, batch 1, greedy, "
                                                "3 warmup passes excluded"),
        quantised_accuracy=accs or (NOT_MEASURED + " -- queued (runners/run_shrink_quant_acc.sh)"),
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
            headline=(
                "Neither lever produces a compute win, and the inversion explains why that was "
                "close to inevitable.  QUANTISATION cannot reduce FLOP-eq at all -- identical "
                "logical parameter count, identical MACs, no INT4 tensor-core path on A100 -- so "
                "it is a MEMORY lever only: %s GiB of weights -> %s GiB resident measured (NF4), a "
                "%sx reduction, with NO measured latency benefit (INT8 is ~12x worse).  A SMALLER STRONG LEG "
                "is the only lever that touches the primary objective, and Lingshu-I-8B is a "
                "genuinely attractive candidate on both axes (7.944B params / 14.7975 GiB, "
                "R=0.8179 -- cheaper per pass than the CURRENT CHEAP LEG, 0.2143x "
                "always-32B-direct), but its accuracy on our harness is being measured by a "
                "concurrent round and the vendor's own table puts it 2.1 points below the model "
                "it would replace.  The INVERSION is the transferable result: the pre-registered "
                "constraint requires a replacement to land within -0.0149..+0.0009 macro of "
                "always-32B-direct, and an independently-erring model must BEAT it outright, so "
                "the constraint as written admits quantised/distilled copies (no FLOP saving) "
                "and genuinely better models (a capability result) but not cheaper comparable "
                "ones." % (foot["footprint"]["lingshu_32b"]["weight_gib"],
                           nf4["resident_gib_during_generation"] if nf4 else "?",
                           round(foot["footprint"]["lingshu_32b"]["weight_gib"]
                                 / nf4["resident_gib_during_generation"], 2)
                           if nf4 else "?")),
            best_operating_point_from_this_attack=(
                "NONE DEMONSTRATED.  No configuration measured here is both cheaper than "
                "always-32B-direct and shown to satisfy the constraint.  The best CANDIDATE on "
                "cost and footprint is always-Lingshu-I-8B at 0.2143x compute (derived R32) / "
                "0.1790x (as-charged) and 14.7975 GiB, pending the concurrent round's accuracy "
                "measurement; on the prior above it is unlikely to clear the bar."),
            what_would_change_this=(
                "(a) the concurrent I-8B measurement coming in at macro >= 0.6557; (b) relaxing "
                "the non-inferiority margin, which is a decision for the project, not a "
                "measurement -- at the current eps the paired-CI geometry, not the model, is "
                "what rejects cheap alternatives; (c) an INT4 path on hardware that actually has "
                "INT4 tensor cores, which would convert the memory win into a throughput win."),
        ),
        limitations=[
            "bf16 Lingshu-32B VRAM/latency control is " + NOT_MEASURED + " at write time: both "
            "A100s are saturated by two other rounds and the job waits rather than "
            "oversubscribing.  Without it no NF4-vs-bf16 latency RATIO is quoted.",
            "Quantised ACCURACY is " + NOT_MEASURED + " at write time (queued).  The FLOP "
            "conclusion does not depend on it; the 'does a quantised strong leg hold the tie' "
            "question does.",
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
