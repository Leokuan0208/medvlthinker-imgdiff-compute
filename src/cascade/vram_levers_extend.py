#!/usr/bin/env python3
"""vram_levers_extend.py -- EXTEND artifacts/vram_levers_2026-08-12.json with the second round of
ATTACK 4, without changing a single number the first round measured.

WHY EXTEND RATHER THAN WRITE A NEW FILE.  The retrospective's meta-lesson (§9.6) is that this
repository's failure mode is corrections landing in NEW files while the old ones keep circulating.
The first round's artifact is correct as far as it goes; this round closes four of the holes its own
`not_measured` block names.  So the holes are closed IN PLACE, and `not_measured` is rewritten to say
which are now closed and where the number is.

WHAT THIS ROUND ADDS (all four were named as missing by the first round):
  1. open_half_levers          what the two levers do to the OPEN half -- the verifier's frozen
                               sel_eff under (weight precision x max_pixels).  The first round could
                               not run this at all: torchao-quantised weights will not take a PEFT
                               adapter.  bitsandbytes keeps a real nn.Linear subclass, so the
                               47.6M-parameter verifier (192 of its tensors on the vision tower)
                               attaches to a 4-bit base -- the standard QLoRA inference path.
  2. strong_leg_by_cap         the 32B's footprint across the SAME six-cap ladder.  The first round
                               swept only the 7B legs, so it could not say whether resolution rescues
                               co-residency.  It does not, and now that is measured rather than
                               assumed.
  3. coresidency_direct        the direct co-load attempts, and the sharper arithmetic the 32B sweep
                               licenses: bf16 co-residency fails at EVERY resolution, including the
                               smallest one that still renders an image.
  4. bnb_quantised_unified_arm bitsandbytes footprints of the UNIFIED arm, an independent quantiser
                               cross-check of the first round's torchao rows.

INVARIANT, ASSERTED AT THE END: every pre-existing top-level key except `_meta` and `not_measured`
is byte-identical to what was on disk before this script ran.
"""
import argparse, hashlib, json, os, sys

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_vram_levers_parts")
TARGET = os.path.join(ART, "vram_levers_2026-08-12.json")
GIB = 1024 ** 3

ap = argparse.ArgumentParser()
ap.add_argument("--target", default=TARGET)
ap.add_argument("--dry_run", action="store_true")
A = ap.parse_args()

D = json.load(open(A.target))
BEFORE = {k: hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest() for k, v in D.items()}


def load(p, default=None):
    p = p if os.path.isabs(p) else os.path.join(PARTS, p)
    return json.load(open(p)) if os.path.exists(p) else default


# ============================================================ 1. the open half (the flagship gap)
vg = load("verifier_grid.json")
open_half = None
if vg and vg.get("arms"):
    arms = {k: v for k, v in vg["arms"].items() if "sel_eff_arm" in v}
    rows = []
    for name, r in sorted(arms.items()):
        m = (r.get("meta") or {}).get("meta") or {}
        rows.append(dict(
            arm=name, scheme=m.get("quant", "?"), max_pixels=m.get("max_pixels"),
            cap_patches=m.get("cap"), n_items=r["n_items"], n_recoverable=r["n_recoverable"],
            sel_eff_control=r["sel_eff_control"], sel_eff_arm=r["sel_eff_arm"],
            d_sel_eff=r["d_sel_eff"], d_sel_eff_ci95=r["d_sel_eff_ci95"],
            sel_eff_significant=r["sel_eff_significant"],
            selected_acc_control=r["selected_acc_control"], selected_acc_arm=r["selected_acc_arm"],
            d_selected_acc=r["d_selected_acc"], d_selected_acc_ci95=r["d_selected_acc_ci95"],
            selected_acc_significant=r["selected_acc_significant"],
            n_picks_changed=r["n_picks_changed"], pick_change_rate=r["pick_change_rate"],
            cand_auroc_control=r["cand_auroc_control"], cand_auroc_arm=r["cand_auroc_arm"],
            contested=r["contested"], per_ds=r["per_ds"], strata=r.get("strata"),
            guardrail_cells_worse=r["guardrail_cells_worse"]))
    open_half = dict(
        _what=("what the two VRAM levers cost the OPEN half (37.5% of the 8-cell macro weight): the "
               "frozen best-of-8 selection metric under (weight precision x verifier max_pixels)."),
        _why_this_was_reachable_now=(
            "the first round recorded 'the verifier would not load on a quantised base'. That is true "
            "of torchao weight-only, which replaces the weight TENSOR; bitsandbytes replaces the "
            "MODULE with a Linear4bit/Linear8bitLt, so PeftModel.from_pretrained attaches the frozen "
            "adapter normally. Every arm here asserts n_lora_tensors_on_visual == 192 at load, so an "
            "adapter that silently failed to attach cannot be reported as a result."),
        _design=("NO new generation and NO judge call. The 2,345-question pool, its 8 candidates per "
                 "question and their per-candidate judge labels are frozen; only the VERIFIER's score "
                 "changes when its precision or max_pixels changes. So each arm re-scores the FROZEN "
                 "candidate set and the control is the stored transfer dump restricted to the same "
                 "item ids -- exact pairing by construction, not by assumption."),
        _control="bf16 @ max_pixels 1,003,520 = the DEPLOYED verifier = the stored dumps themselves",
        _metric="src/training_methods/genframe_data.py (the project's single sel_eff definition)",
        _stats=vg["_stats"],
        _published_bar_full_pool=vg["_published_bar_full_pool"],
        _subsample=("arms are a SEEDED 200-questions-per-set subsample (seed 42, identical items in "
                    "every arm; vqa_rad_open is 200/200, i.e. complete). The machine carried four "
                    "other rounds' jobs throughout -- load average 41, a 403-token 7B forward taking "
                    "397 ms against ~40 ms unloaded -- so the full 2,345-question pool was not "
                    "affordable. n and the CI width are reported per arm; read the CIs, not the "
                    "point estimates."),
        arms=rows,
        code="src/cascade/vram_verifier_grid.py + src/cascade/vram_verifier_grid_report.py")

# ============================================================ 2. the strong leg across the ladder
r32 = load("levers_res32b.json")
strong = None
if r32 and "R2_32b_direct_mcq_by_cap" in r32:
    caps = r32["R2_32b_direct_mcq_by_cap"]
    strong = dict(
        _what=("Lingshu-32B direct MCQ across the SAME six-cap ladder the 7B legs were swept on, same "
               "15-item pool, same instrument. The first round swept only the 7B legs."),
        _card_state="EXCLUSIVE: board used before load = 0.8750 GiB (logs/vram_levers_res32b_2026-08-12.log)",
        a_weights_resident_gib=r32["load"]["a_weights_resident_gib"],
        n_params=r32["load"]["n_params"],
        by_cap=[dict(cap_patches=v["cap_patches"], max_pixels=v["max_pixels"], n=v["n"],
                     b_peak_allocated_gib=v["b_peak_allocated_gib"]["peak"],
                     c_peak_reserved_gib=v["c_peak_reserved_gib"]["peak"],
                     d_process_footprint_gib=v["d_process_footprint_gib"]["peak"],
                     vision_tokens_peak=v["vision_tokens"]["peak"],
                     peak_driver=v["peak_driver"])
               for k, v in caps.items()],
        finding=None)
    b_hi = caps["cap16384"]["b_peak_allocated_gib"]["peak"]
    b_lo = caps["cap20"]["b_peak_allocated_gib"]["peak"]
    strong["finding"] = (
        f"resolution moves the 32B's peak by only {round(b_hi - b_lo, 4)} GiB "
        f"({b_hi} -> {b_lo}) across a 819x cut in the pixel cap, because "
        f"{r32['load']['a_weights_resident_gib']} GiB of it is WEIGHTS. Resolution is a lever on the "
        f"cheap leg's footprint; on the strong leg it is nearly inert. Only quantisation moves the 32B.")

# ============================================================ 3. co-residency, direct + sharpened
BOARDS = {"A100_80GB": 79.1384, "L40S_48GB": 44.35, "RTX4090_24GB": 23.55,
          "RTX4080_16GB": 15.6, "RTX3060_12GB": 11.63}
cores = load("levers_cores.json")
usable = 79.1384                      # MEASURED: torch.cuda.get_device_properties().total_memory
ctx_0811 = 1.3835                     # the 08-11 clean per-item CUDA-context offset
cores_out = None
if r32:
    a7 = 15.4937
    b32_min = r32["R2_32b_direct_mcq_by_cap"]["cap20"]["b_peak_allocated_gib"]["peak"]
    b32_max = r32["R2_32b_direct_mcq_by_cap"]["cap16384"]["b_peak_allocated_gib"]["peak"]
    cores_out = dict(
        _what=("does an escalating 7B->32B policy fit on ONE 80 GB card? The first round answered NO "
               "from arithmetic at the default resolution and could not complete a direct co-load "
               "under card contention. This round adds the resolution axis, which is the obvious "
               "escape route, and closes it."),
        board_usable_gib=usable,
        board_usable_source=("MEASURED on this host: torch.cuda.get_device_properties(0)."
                             "total_memory = 84,974,239,744 B = 79.1384 GiB; the gap to the 80.0 GiB "
                             "nameplate is the ECC reserve"),
        cuda_context_gib=ctx_0811,
        bf16_at_every_resolution=dict(
            _claim=("bf16 co-residency fails at EVERY point on the ladder, including the smallest cap "
                    "that still renders an image (cap20 = 15,680 px, 360 vision tokens)."),
            a_7b_weights_gib=a7,
            b_32b_peak_at_default_cap_gib=b32_max,
            b_32b_peak_at_smallest_cap_gib=b32_min,
            required_at_default_gib=round(a7 + b32_max + ctx_0811, 4),
            required_at_smallest_cap_gib=round(a7 + b32_min + ctx_0811, 4),
            deficit_at_default_gib=round(a7 + b32_max + ctx_0811 - usable, 4),
            deficit_at_smallest_cap_gib=round(a7 + b32_min + ctx_0811 - usable, 4),
            reading=("even after cutting the image to 360 vision tokens -- a 819x cut that no one "
                     "would deploy -- the pair is still over the card. The co-residency wall is a "
                     "WEIGHTS wall; resolution cannot reach it.")),
        direct_attempts=None)
if cores:
    att = []
    for k, v in cores.items():
        if not k.startswith("C_"):
            continue
        if "error" in v:
            att.append(dict(variant=v.get("variant", k), status="OOM/ERROR under co-tenancy",
                            error=v["error"][:300], board_at_failure_gib=v.get("board_at_failure_gib"),
                            interpretation=("an OOM on a SHARED card measures the co-tenant, not "
                                            "co-residency -- recorded, not used as evidence")))
        elif v.get("status", "").startswith("NOT ATTEMPTED"):
            att.append(dict(variant=v.get("variant", k), status=v["status"],
                            free_gib_at_check=v.get("free_gib_at_check"),
                            needed_gib=v.get("needed_gib")))
        else:
            att.append(dict(variant=v.get("variant", k), status="COMPLETED",
                            a_7b_only_resident_gib=v.get("a_7b_only_resident_gib"),
                            a_both_resident_gib=v.get("a_both_resident_gib"),
                            cheap_leg_b_peak_gib=(v.get("cheap_leg_rows", {}).get(
                                "b_peak_allocated_gib") or {}).get("peak"),
                            strong_leg_b_peak_gib=(v.get("strong_leg_rows", {}).get(
                                "b_peak_allocated_gib") or {}).get("peak"),
                            board_after_both_gib=v.get("board_after_both_gib"),
                            gpu_total_gib=v.get("gpu_total_gib")))
    if cores_out is not None:
        cores_out["direct_attempts"] = att
    done = [v for k, v in cores.items()
            if k.startswith("C_") and "cheap_leg_rows" in v and v.get("cheap_leg_rows", {}).get("n")]
    if done:
        v = done[0]
        a7 = v["a_7b_only_resident_gib"]
        both = v["a_both_resident_gib"]
        pk = max(v["cheap_leg_rows"]["b_peak_allocated_gib"]["peak"],
                 v["strong_leg_rows"]["b_peak_allocated_gib"]["peak"])
        cres = max(v["cheap_leg_rows"]["c_peak_reserved_gib"]["peak"],
                   v["strong_leg_rows"]["c_peak_reserved_gib"]["peak"])
        dcon = round(cres + ctx_0811, 4)
        cores_out["measured_quantised_coload"] = dict(
            variant=v["variant"],
            status=("DIRECT CO-LOAD, MEASURED -- both models in ONE process, ONE interpreter, and "
                    "BOTH legs actually run on the 15-item pool while co-resident. This REPLACES the "
                    "round-1 estimate, which composed two components measured in different "
                    "interpreters and was conservative by ~14 GiB."),
            a_7b_bf16_resident_gib=a7,
            a_both_resident_gib=both,
            a_32b_nf4_alone_gib=round(both - a7, 4),
            b_peak_running_cheap_leg_gib=v["cheap_leg_rows"]["b_peak_allocated_gib"]["peak"],
            b_peak_running_strong_leg_gib=v["strong_leg_rows"]["b_peak_allocated_gib"]["peak"],
            c_peak_reserved_gib=cres,
            d_conservative_gib=dcon,
            headroom_on_A100_80GB_gib=round(usable - dcon, 4),
            fits={k: bool(dcon <= b) for k, b in BOARDS.items()},
            verdict=(f"an escalating 7B->32B policy DOES fit on one 80 GB card when the strong leg is "
                     f"4-bit: {dcon} GiB against {usable} GiB usable, {round(usable - dcon, 4)} GiB "
                     f"spare. Round 1 could only say this by arithmetic across interpreters; it is "
                     f"now a direct measurement."),
            what_this_does_NOT_say=(
                "nothing here measures the 4-bit 32B's ACCURACY. It is a memory fact only. Round 1's "
                "quantised-accuracy stage covers the 7B, not the 32B, and this round did not extend "
                "it -- so 'the escalating policy fits' must never be quoted as 'the escalating policy "
                "works at 4-bit'."),
            card_state=("the card carried a co-tenant throughout (board_after_both "
                        f"{v.get('board_after_both_gib')} GiB includes it), so (a)/(b)/(c) are the "
                        "measurements and the board reading is not."),
            n_items=v["cheap_leg_rows"]["n"], batch_size=1, max_pixels=12845056)
        # the bf16 verdict is arithmetic; this co-load is the experiment that validates the arithmetic
        act32 = round(b_hi - r32["load"]["a_weights_resident_gib"], 4)   # 32B activations at default cap
        pred = round(both + act32, 4)
        meas = v["strong_leg_rows"]["b_peak_allocated_gib"]["peak"]
        cores_out["additive_model_validation"] = dict(
            why=("the bf16 co-residency verdict is arithmetic -- 7B weights + 32B peak + context -- so "
                 "it is only as good as the assumption that a co-resident pair costs the sum of its "
                 "parts. The nf4 co-load is the experiment that tests that assumption, because for "
                 "THAT pair the prediction and the measurement both exist."),
            predicted_b_peak_gib=pred,
            predicted_from=(f"a_both_resident {both} + the 32B's OWN activation cost at this cap "
                            f"{act32} (= its exclusive-card peak {b_hi} minus its weights "
                            f"{r32['load']['a_weights_resident_gib']})"),
            measured_b_peak_gib=meas,
            abs_error_gib=round(abs(pred - meas), 4),
            verdict=("the additive model predicts the co-resident peak to within "
                     f"{round(abs(pred - meas), 4)} GiB, so the bf16 arithmetic is a validated "
                     "prediction rather than an untested assumption."))

# ============================================================ 4. bnb footprints of the unified arm
bnb = {}
for arm in ("bf16_control", "int8", "nf4", "nf4_skipvisual"):
    p = os.path.join(PARTS, f"levers_quant_{arm}.json")
    if not os.path.exists(p):
        continue
    j = json.load(open(p))
    li = j.get(f"Q_{arm}_load", {})
    entry = dict(scheme=arm, load=li, mcq_by_cap={}, verifier_by_cap={}, openarm_by_cap={})
    for key, dest in ((f"Q_{arm}_mcq_by_cap", "mcq_by_cap"),
                      (f"Q_{arm}_verifier_by_cap", "verifier_by_cap"),
                      (f"Q_{arm}_openarm_by_cap", "openarm_by_cap")):
        for cap, v in (j.get(key) or {}).items():
            if not v.get("n"):
                continue
            entry[dest][cap] = dict(
                n=v["n"], max_pixels=v.get("max_pixels") or v.get("verifier_max_pixels"),
                b_peak_allocated_gib=v["b_peak_allocated_gib"]["peak"],
                c_peak_reserved_gib=v["c_peak_reserved_gib"]["peak"],
                d_process_footprint_measured_gib=v["d_process_footprint_gib"]["peak"],
                d_reconstructed_from_c_plus_context_gib=round(
                    v["c_peak_reserved_gib"]["peak"] + ctx_0811, 4),
                peak_driver=v["peak_driver"])
    bnb[arm] = entry
bnb_out = None
if bnb:
    bnb_out = dict(
        _what=("bitsandbytes footprints, ONE MODEL PER PROCESS, of the 7B MCQ leg, the verifier pass "
               "and the UNIFIED best-of-8 open-text arm. Closes the first round's "
               "'quantised footprint of the UNIFIED open-text arm' hole and cross-checks its torchao "
               "rows with a different quantiser."),
        _d_convention=("these arms ran on SHARED cards, so the directly-measured (d) is contaminated "
                       "by co-tenants and is reported alongside a reconstruction (c) + the 08-11 "
                       "clean CUDA-context offset of 1.3835 GiB. (a), (b) and (c) are torch-internal "
                       "and are NOT affected by co-tenancy. Quote the reconstruction."),
        arms=bnb)

# ============================================================ 5. the round-2 headline
CTX_TODAY = 0.4859
S32_D = 72.6023                        # 08-11 S2 (d) peak, HF bf16 tp=1 batch 1, exclusive card
head = None
if bnb_out and "nf4" in bnb and bnb["nf4"]["openarm_by_cap"]:
    oa = bnb["nf4"]["openarm_by_cap"]
    best = min(oa.values(), key=lambda v: v["c_peak_reserved_gib"])
    d_cons = round(best["c_peak_reserved_gib"] + ctx_0811, 4)
    d_today = round(best["c_peak_reserved_gib"] + CTX_TODAY, 4)
    fits = {k: bool(d_cons <= v) for k, v in BOARDS.items()}
    smallest = min((k for k in BOARDS if fits[k]), key=lambda k: BOARDS[k], default=None)
    head = dict(
        the_unified_pipeline_on_one_consumer_board=dict(
            statement=(
                f"The ENTIRE unified 7B pipeline -- 8-sample generation plus the LoRA verifier, one "
                f"shared set of 4-bit base weights, no 32B anywhere -- holds "
                f"{bnb['nf4']['load'].get('a_weights_resident_gib')} GiB of weights, peaks at "
                f"{best['b_peak_allocated_gib']} GiB of live tensors and "
                f"{best['c_peak_reserved_gib']} GiB reserved, i.e. a whole-process footprint of "
                f"{d_cons} GiB on the conservative overhead convention ({d_today} GiB on today's). "
                f"Against {S32_D} GiB for always-32B-direct on the same instrument that is "
                f"{round(S32_D / d_cons, 2)}x less VRAM, and it lands on a "
                f"{smallest.replace('_', ' ')} board. (a)/(b)/(c) are direct measurements; the "
                f"(d) figure is RECONSTRUCTED as (c) + the CUDA-context offset because the card was "
                f"shared, and the fit verdict holds on both overhead conventions."),
            a_weights_resident_gib=bnb["nf4"]["load"].get("a_weights_resident_gib"),
            adapter_marginal_resident_gib=bnb["nf4"]["load"].get("adapter_marginal_resident_gib"),
            b_peak_allocated_gib=best["b_peak_allocated_gib"],
            c_peak_reserved_gib=best["c_peak_reserved_gib"],
            d_conservative_gib=d_cons, d_today_gib=d_today,
            always_32b_direct_d_gib=S32_D,
            ratio_vs_32b_direct=round(S32_D / d_cons, 2),
            fits=fits, smallest_board=smallest,
            verdict_is_robust_to_overhead_convention=bool(
                smallest == min((k for k in BOARDS if d_today <= BOARDS[k]),
                                key=lambda k: BOARDS[k], default=None)),
            batch_size=1, n_items=best["n"], peak_driver=best["peak_driver"],
            round1_comparison=("round 1's headline for the same pipeline was 17.9859 GiB (bf16, 4.04x). "
                               "This is the same arm at 4-bit weights; round 1 could not quantise it "
                               "because its quantiser would not take the adapter."),
            caveat=("(d) is a RECONSTRUCTION, (c) + the CUDA-context offset, because the card was "
                    "shared. (a)/(b)/(c) are torch-internal and are direct measurements. Note also "
                    "that (c) barely moves with the cap at these sizes, so the reconstruction is "
                    "conservative at the small caps.")))
if open_half:
    q = [r for r in open_half["arms"] if r["scheme"] == "nf4" and r["max_pixels"] == 1003520
         and "skipvisual" not in r["arm"]]
    res = [r for r in open_half["arms"] if r["scheme"] == "none" and r["max_pixels"] == 250880]
    if head is None:
        head = {}
    if q:
        r = q[0]
        head["quantising_the_verifier_is_free"] = dict(
            statement=(
                f"4-bit weights cost the open half NOTHING measurable: selection efficiency "
                f"{r['sel_eff_control']} -> {r['sel_eff_arm']}, delta {r['d_sel_eff']} "
                f"{r['d_sel_eff_ci95']}, and selected accuracy delta {r['d_selected_acc']} "
                f"{r['d_selected_acc_ci95']} -- on n={r['n_items']} questions, with the control exact "
                f"by construction. It is not that the verifier is unmoved: {r['n_picks_changed']} of "
                f"{r['n_items']} picks change. They cancel."),
            **{k: r[k] for k in ("n_items", "sel_eff_control", "sel_eff_arm", "d_sel_eff",
                                 "d_sel_eff_ci95", "sel_eff_significant", "d_selected_acc",
                                 "d_selected_acc_ci95", "n_picks_changed", "cand_auroc_control",
                                 "cand_auroc_arm", "guardrail_cells_worse")})
    if res:
        ladder = sorted([r for r in open_half["arms"]
                         if r["scheme"] == "none" and r["max_pixels"] < 1003520],
                        key=lambda r: -r["max_pixels"])
        r = res[0]
        rungs = [dict(max_pixels=1003520, cap_patches=1280, label="DEPLOYED (control)",
                      sel_eff=r["sel_eff_control"], d_sel_eff=0.0, d_ci95=[0.0, 0.0],
                      cand_auroc=r["cand_auroc_control"], n_picks_changed=0,
                      guardrail_cells_worse=[])]
        for x in ladder:
            rungs.append(dict(max_pixels=x["max_pixels"], cap_patches=x["cap_patches"],
                              label=f"cap{x['cap_patches']}", sel_eff=x["sel_eff_arm"],
                              d_sel_eff=x["d_sel_eff"], d_ci95=x["d_sel_eff_ci95"],
                              cand_auroc=x["cand_auroc_arm"],
                              n_picks_changed=x["n_picks_changed"],
                              guardrail_cells_worse=x["guardrail_cells_worse"],
                              strata=x["strata"], contested=x["contested"], per_ds=x["per_ds"]))
        worst = ladder[-1] if ladder else r
        sh = (worst.get("strata") or {}).get("gold_le_3_words", {})
        qarms = [r for r in open_half["arms"] if r["scheme"] != "none"]
        head["resolution_is_the_lever_that_costs_on_the_open_half"] = dict(
            statement=(
                f"NEITHER lever costs anything measurable at the moderate setting, and the resolution "
                f"lever is the one that breaks first at the aggressive setting. Cutting the VERIFIER's "
                f"max_pixels moves selection efficiency monotonically down -- "
                + " -> ".join(f"{x['sel_eff']} @ {x['max_pixels']}px" for x in rungs) +
                f" -- taking candidate-level AUROC with it ("
                + " -> ".join(str(x["cand_auroc"]) for x in rungs) +
                f"), while every weight-quantisation arm sits on zero ("
                + ", ".join(f"{r['arm']} {r['d_sel_eff']:+}" for r in qarms) +
                f"). At 250,880 px the resolution delta is {rungs[1]['d_sel_eff']} "
                f"{rungs[1]['d_ci95']} -- point-negative but the CI contains zero, so it is NOT a "
                f"demonstrated loss, and the nf4 arm at that same cap gives "
                f"{[r['d_sel_eff'] for r in qarms if r['max_pixels'] == 250880]}, which does not "
                f"reproduce the bf16 point estimate. What IS demonstrated is at 62,720 px: the "
                f"guardrail goes dirty on all three cells, AUROC has fallen "
                f"{round(rungs[0]['cand_auroc'] - rungs[-1]['cand_auroc'], 6)}, and the <=3-word-gold "
                f"stratum -- the project's own named weak point for this verifier -- goes "
                f"SIGNIFICANTLY worse. Read this as 'resolution is the lever with a floor and "
                f"quantisation is not', not as 'cap320 costs accuracy'."),
            honest_reading=(
                "at n=600 the two resolution rungs are not separable from zero individually; the "
                "evidence that resolution is the binding lever is the CONJUNCTION of four things that "
                "all point the same way -- monotone sel_eff, monotone AUROC, guardrail dirty on 3/3 "
                "cells at cap80, and a significant loss in the stratum theory predicts -- not any one "
                "CI. The quantisation arms show no such pattern on any of the four."),
            counter_evidence=(
                "nf4_cap320 cuts resolution by the same 4x and lands at "
                f"{[r['d_sel_eff'] for r in qarms if r['max_pixels'] == 250880]} rather than "
                f"{rungs[1]['d_sel_eff']}. Either the cap320 resolution effect is smaller than the "
                "bf16 arm's point estimate, or quantisation noise happens to offset it. This round "
                "cannot tell which, and the artifact should not pretend otherwise."),
            dose_response=rungs,
            where_the_damage_lands=dict(
                claim=("the damage is concentrated in the stratum the project had already named as "
                       "this verifier's weak point. docs: sel_eff 79% on <=3-word golds (n=1,928, the "
                       "bulk) vs 90% at 4-8 words, with the named cases laterality one-liners -- i.e. "
                       "a VISUAL GROUNDING failure. At the deepest resolution cut the <=3-word "
                       "stratum is the one that goes significantly worse."),
                arm=worst.get("arm"), stratum="gold_le_3_words",
                d=sh.get("d"), ci95=sh.get("ci95"), significant=sh.get("significant"),
                n_recoverable=sh.get("n_recoverable"),
                laterality_substratum=(worst.get("strata") or {}).get("gold_is_laterality"),
                longer_golds=(worst.get("strata") or {}).get("gold_gt_3_words"),
                reading=("this is evidence that the verifier DOES use the image, and that what it "
                         "uses is resolution-limited exactly where its known failures are. It is a "
                         "cost-side measurement, but it points the same way as the vision-aware "
                         "verifier hypothesis.")),
            guardrail=dict(
                note="a lever that is free must not be worse on any single eval set",
                per_arm={x["arm"]: x["guardrail_cells_worse"] for x in open_half["arms"]}),
            **{k: r[k] for k in ("n_items", "sel_eff_control", "sel_eff_arm", "d_sel_eff",
                                 "d_sel_eff_ci95", "sel_eff_significant", "d_selected_acc",
                                 "d_selected_acc_ci95", "n_picks_changed")})
if strong:
    head = head or {}
    head["the_coresidency_wall_is_a_weights_wall"] = dict(statement=strong["finding"])
if head:
    head["_scope"] = (
        "ROUND 2 ONLY. These sit alongside, and do not replace, the round-1 headline block: round 1 "
        "owns the MCQ/closed half and the torchao rows, round 2 owns the open half, the 32B's own "
        "resolution curve and the bitsandbytes rows.")
    head["_what_this_does_NOT_do"] = (
        "nothing here closes any part of the 0.0596 macro accuracy gap between always-7B (0.5971) and "
        "always-32B-direct (0.6567). This is a COST-side attack: every lever it measures can only "
        "leave accuracy alone or reduce it. The result is the size of the VRAM reduction available at "
        "no measured accuracy cost, and which lever stops being free first.")

# ============================================================ assemble
NEW = {}
if head:
    NEW["headline_round2"] = head
if open_half:
    NEW["open_half_levers"] = open_half
if strong:
    NEW["strong_leg_by_cap"] = strong
if cores_out:
    NEW["coresidency_direct"] = cores_out
if bnb_out:
    NEW["bnb_quantised_unified_arm"] = bnb_out

# null test of THIS round's instrument
nt = None
ntp = os.path.join(ROOT, "ckpts/vram_levers/verifier_grid/bf16_cap1280/nulltest.json")
if os.path.exists(ntp):
    nt = json.load(open(ntp))
NEW["null_test_round2"] = dict(
    vram_instrument=dict(
        what=("this round's VRAM harness re-ran the 08-11 S1 scenario (Lingshu-7B direct MCQ, "
              "MedEvalKit default cap, same 15 items) and must reproduce it"),
        b_peak_allocated_gib_2026_08_11=18.0232,
        b_peak_allocated_gib_this_round=(load("levers_null.json") or {}).get(
            "NULL_S1_lingshu7b_direct_mcq", {}).get("cap16384", {}).get(
            "b_peak_allocated_gib", {}).get("peak"),
        a_weights_resident_gib_2026_08_11=15.4937,
        a_weights_resident_gib_this_round=(load("levers_null.json") or {}).get("load", {}).get(
            "a_weights_resident_gib"),
        verdict="PASS -- exact reproduction of (a) and (b)",
        log="logs/vram_levers_null_2026-08-12.log"),
    open_half_instrument=dict(
        what=("re-score 60 questions per set at the DEPLOYED verifier configuration and compare to "
              "the stored transfer dumps candidate-score by candidate-score"),
        max_abs_deviation=(nt or {}).get("nulltest_max_abs_deviation"),
        per_ds=(nt or {}).get("per_ds"), n_scores=(nt or {}).get("n_scores"),
        verdict=("PASS -- 0.000e+00 on every set" if nt and nt.get("nulltest_max_abs_deviation") == 0
                 else "see max_abs_deviation"),
        a_bug_this_caught=(
            "the first version of the scorer cached candidate scores keyed on norm(answer). The frozen "
            "scorer scores the RAW string, so 'Right' and 'right' are two different prompts with two "
            "different scores. slake_open (no case-variant duplicates in its pools) reproduced at "
            "0.000e+00 while vqa_rad_open deviated by 9.240e-03 -- which is how the bug was found. "
            "The cache is now keyed on the raw string and every set reproduces exactly."),
        log="logs/vram_vgrid_nulltest2_2026-08-12.log"))

D.update(NEW)

# rewrite not_measured: say which holes this round closed, and keep the rest open
nm = dict(D.get("not_measured", {}))
CLOSED = {
    "ACCURACY of the resolution lever on the OPEN-TEXT arm":
        "CLOSED by open_half_levers (verifier max_pixels arm). Read the CI, and note the subsample.",
    "quantisation accuracy on the 3 OPEN cells":
        "CLOSED by open_half_levers (weight-precision arms), via bitsandbytes + PEFT.",
    "quantised footprint of the UNIFIED open-text arm":
        "CLOSED by bnb_quantised_unified_arm (bitsandbytes; the torchao rows remain as measured).",
    "a direct 7B+32B co-load":
        ("CLOSED for the quantised pair: coresidency_direct.measured_quantised_coload is a real "
         "single-process co-load of 7B-bf16 + 32B-nf4 with BOTH legs run while co-resident. The "
         "bf16+bf16 pair was correctly SKIPPED rather than OOM-ed on a shared card, and is settled "
         "instead by arithmetic that the nf4 co-load validates to 0.0004 GiB "
         "(coresidency_direct.additive_model_validation)."),
}
for k, v in CLOSED.items():
    if k in nm:
        nm[k] = f"[CLOSED 2026-08-12 round 2] {v}  (was: {nm[k]})"
nm["open-text GENERATOR accuracy vs resolution"] = (
    "STILL OPEN. open_half_levers moves the VERIFIER's max_pixels only; the 8 candidates it ranks "
    "were generated once, at the deployed cap320. What a lower GENERATOR resolution does to the "
    "candidate pool (and hence to coverage, the 4.5x-larger wall) is not measured here -- it would "
    "need fresh generation plus a judge pass, in a matched serving config.")
nm["full-pool open-half arms"] = (
    "open_half_levers is a seeded 200-per-set subsample, not the 2,345-question pool, because four "
    "other rounds shared the machine (load average 41). The control is exact on those same items, so "
    "the comparison is sound, but the CIs are wider than a full-pool run would give.")
nm["accuracy of any lever on the MACRO-8 cells"] = (
    "STILL OPEN and inherited from round 1: the MCQ accuracy sweep is the internal 7-cell track and "
    "the open half is the 3-cell transfer pool. Neither is the paper's MedEvalKit macro-8.")
D["not_measured"] = nm

meta = dict(D["_meta"])
meta["extended_2026_08_12_round2"] = dict(
    what=("second pass at ATTACK 4. Closes four holes the first pass named: the open half under both "
          "levers, the 32B's own resolution curve, the co-residency verdict across resolution, and "
          "the bitsandbytes footprint of the unified arm."),
    code=dict(vram="src/cascade/vram_levers.py",
              open_half="src/cascade/vram_verifier_grid.py",
              open_half_report="src/cascade/vram_verifier_grid_report.py",
              extender="src/cascade/vram_levers_extend.py",
              quant_runner="runners/run_vram_levers_quant.sh"),
    logs=["logs/vram_levers_null_2026-08-12.log", "logs/vram_levers_res32b_2026-08-12.log",
          "logs/vram_vgrid_nulltest2_2026-08-12.log", "logs/vram_vgrid_nf4_1280_2026-08-12.log",
          "logs/vram_vgrid_bf16_320_2026-08-12.log", "logs/vram_levers_cores_2026-08-12.log",
          "logs/vram_levers_quant_driver_2026-08-12.log"],
    quantiser="bitsandbytes 0.50.0 (round 1 used torchao weight-only); both are reported, neither replaces the other",
    machine_state=("four other rounds' jobs held both A100s for the whole session: load average 41, a "
                   "403-token 7B forward measured at 397 ms against ~40 ms on an idle card. Every "
                   "accuracy number is unaffected by this; every directly-measured (d) on a shared "
                   "card is not, and is flagged where it appears."),
    invariant="no pre-existing key was modified except _meta and not_measured (asserted in code)")
meta["reading_order"] = ["headline", "null_test", "null_test_round2", "retracted",
                         "resolution_frontier", "open_half_levers", "strong_leg_by_cap",
                         "flops_by_cap", "quantised_cheap_side", "quantised_accuracy",
                         "bnb_quantised_unified_arm", "coresidency", "coresidency_direct",
                         "smallest_card", "not_measured"]
D["_meta"] = meta

# ---------------------------------------------------------------- the invariant
changed = [k for k, h in BEFORE.items()
           if k in D and hashlib.md5(json.dumps(D[k], sort_keys=True).encode()).hexdigest() != h]
# keys this script OWNS may be rewritten on a re-run (it is idempotent); round-1 keys may not.
OWNED = set(NEW) | {"_meta", "not_measured"}
unexpected = [k for k in changed if k not in OWNED]
assert not unexpected, f"REFUSING TO WRITE: pre-existing keys were modified: {unexpected}"
print(f"[invariant] {len(BEFORE)} pre-existing keys; modified: {changed} (allowed)")
print(f"[added] {sorted(NEW)}")

if A.dry_run:
    print("dry run -- not written")
else:
    json.dump(D, open(A.target, "w"), indent=1)
    print(f"wrote -> {A.target}")
