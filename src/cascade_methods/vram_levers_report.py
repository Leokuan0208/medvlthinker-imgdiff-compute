#!/usr/bin/env python3
"""
vram_levers_report.py -- ATTACK 4 aggregator: build artifacts/vram_levers_2026-08-12.json.

Joins four measured parts into one accuracy-vs-VRAM-vs-FLOPs frontier:
  * _vram_levers_parts/R1_resolution_7b_mcq.json        VRAM, 7B direct MCQ, 6 resolution caps
  * _vram_levers_parts/R2_resolution_opentext_arm.json  VRAM, full open-text best-of-8 arm
  * _vram_levers_parts/Q_quantised_cheap_side.json      VRAM, torchao int8wo / int4wo cheap side
  * _vram_levers_parts/C_coresidency.json               7B + 32B on ONE card
  * _vram_levers_parts/accuracy_resolution_sweep.json   ACCURACY cost of each cap (paired, CIs)

and adds two derived blocks that need no GPU:
  * FLOPs per cap, from src/cascade_methods/flop_ratio_derivation.py:forward_flops evaluated on the
    MEASURED token geometry of each cap (not an assumed geometry).  Reported SEPARATELY from memory
    and never conflated: resolution cuts BOTH; weight-only quantisation cuts memory and NOTHING else.
  * the smallest-card analysis: for each configuration, the measured (d) process footprint against
    80 / 48 / 24 / 16 / 12 GB boards.

    python3 src/cascade_methods/vram_levers_report.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_vram_levers_parts")
OUT = os.path.join(ART, "vram_levers_2026-08-12.json")
GIB = 1024 ** 3

# the 08-11 rows this run must reproduce -- verbatim from artifacts/vram_testtime_2026-08-11.json
NULL_TARGETS = {
    "R1.medevalkit_default": dict(source="S1_lingshu7b_direct_mcq",
                                  a=15.4937, b_peak=18.0232, c_peak=22.0371, b_mean=16.2473),
    "R2.s4_deployed_null_test": dict(source="S4_opentext_bestof8_full_arm",
                                     a=15.4937, b_peak=16.6581, c_peak=17.3789, b_mean=16.3141),
}
BOARDS_GIB = {"A100_80GB": 79.14, "L40S_48GB": 44.35, "RTX4090_24GB": 23.55,
              "RTX4080_16GB": 15.60, "RTX3060_12GB": 11.63}
BOARD_NOTE = ("usable board capacity, not nameplate: an 80 GB A100 reports 79.14 GiB usable to "
              "torch (this run's own OOM message) and consumer cards lose ~2-4% to the same "
              "reserved regions plus the display. Values are the conventional usable figures; a "
              "configuration is called a FIT only if its measured (d) is below them.")


def load(name):
    p = os.path.join(PARTS, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def flops_block(r1):
    """FLOPs per cap on the MEASURED token geometry, via the repo's own analytic model."""
    sys.path.insert(0, os.path.join(REPO, "src", "cascade_methods"))
    import flop_ratio_derivation as F
    try:
        P7 = F.param_counts("Lingshu-7B")
        P32 = F.param_counts("Lingshu-32B")
    except Exception as e:
        return dict(error=f"param_counts failed: {type(e).__name__}: {e}")
    out = {}
    for cap, blk in (r1 or {}).items():
        if not isinstance(blk, dict) or "vision_tokens" not in blk or blk.get("n", 0) == 0:
            continue
        # this script's `vision_tokens` is image_grid_thw.prod() = PRE-merge patches P.
        # forward_flops takes M = merged tokens = P/4.
        M = blk["vision_tokens"]["mean"] / 4.0
        T = blk["input_tokens"]["mean"]
        G = max(blk["gen_tokens"]["mean"], 1.0)
        f7 = F.forward_flops(P7, M, T, G)
        out[cap] = dict(
            max_pixels=blk["meta"]["max_pixels"],
            vision_token_budget=blk["meta"]["vision_token_budget"],
            measured_merged_vision_tokens_mean=round(M, 1),
            measured_input_tokens_mean=round(T, 1),
            measured_gen_tokens_mean=round(G, 2),
            flops_7b_total=f7["TOTAL"],
            flops_7b_parts={k: v for k, v in f7.items() if k != "TOTAL"})
    if out:
        base = out.get("medevalkit_default") or list(out.values())[0]
        for cap, v in out.items():
            v["flops_relative_to_medevalkit_default"] = round(
                v["flops_7b_total"] / base["flops_7b_total"], 4)
        # one 32B-direct pass at the SAME geometry the paper charges it at, for scale
        b = base
        out["_reference_32b_direct_at_medevalkit_default"] = dict(
            flops_32b_total=F.forward_flops(P32, b["measured_merged_vision_tokens_mean"],
                                            b["measured_input_tokens_mean"],
                                            b["measured_gen_tokens_mean"])["TOTAL"])
    out["_convention"] = (
        "src/cascade_methods/flop_ratio_derivation.py:forward_flops, evaluated on THIS run's "
        "measured per-cap token geometry (merged vision tokens = image_grid_thw.prod()/4, prompt "
        "tokens, generated tokens). Analytic, not a hardware counter. FLOPs and VRAM are different "
        "quantities and are never summed or traded off against each other in this file.")
    return out


def null_test(r1, r2):
    res = {}
    devs = []
    for key, tgt in NULL_TARGETS.items():
        scen, cap = key.split(".")
        blk = (r1 if scen == "R1" else r2)
        blk = (blk or {}).get(cap)
        if not blk or blk.get("n", 0) == 0:
            res[key] = dict(status="NOT RUN")
            continue
        got = dict(b_peak=blk["b_peak_allocated_gib"]["peak"],
                   c_peak=blk["c_peak_reserved_gib"]["peak"],
                   b_mean=blk["b_peak_allocated_gib"]["mean"])
        d = {k: round(got[k] - tgt[k], 4) for k in got}
        devs += [abs(v) for v in d.values()]
        res[key] = dict(against=tgt["source"], expected={k: tgt[k] for k in got},
                        measured=got, deviation=d,
                        max_abs_deviation_gib=round(max(abs(v) for v in d.values()), 4))
    res["_summary"] = dict(
        max_abs_deviation_gib=round(max(devs), 4) if devs else None,
        quantities="(b) peak_allocated and (c) peak_reserved only. (a) is reported separately "
                   "because it is a property of the load, not of the item.",
        why_d_is_excluded="(d) is NOT null-tested against 08-11: that run measured (d) as board "
                          "used minus a pre-run baseline on an EXCLUSIVE card, while this run "
                          "measures the true per-process NVML reading on a SHARED card. The two "
                          "are different instruments; comparing them would be a category error.")
    return res


def frontier(r1, r2, acc):
    """The deliverable: accuracy cost and VRAM saving of each cap, side by side."""
    rows = []
    accmac = (acc or {}).get("internal_track_macro7", {})
    accguard = (acc or {}).get("guardrail", {})
    for cap, blk in (r1 or {}).items():
        if not isinstance(blk, dict) or blk.get("n", 0) == 0:
            continue
        a = accmac.get(cap)
        rows.append(dict(
            cap=cap, max_pixels=blk["meta"]["max_pixels"],
            vision_token_budget=blk["meta"]["vision_token_budget"],
            vram_7b_mcq=dict(
                b_peak_allocated_gib=blk["b_peak_allocated_gib"]["peak"],
                c_peak_reserved_gib=blk["c_peak_reserved_gib"]["peak"],
                d_process_footprint_gib=blk["d_process_footprint_gib"]["peak"],
                d_method=blk["rows"][0].get("d_method") if blk.get("rows") else None,
                mean_b_peak_allocated_gib=blk["b_peak_allocated_gib"]["mean"],
                peak_driver=blk.get("peak_driver")),
            accuracy_internal_track=(
                None if a is None else
                dict(macro7=a["macro7"], delta_vs_fullres=a["delta_vs_fullres"],
                     ci95=a["ci95"], significant=a["significant"],
                     n_cells_significantly_worse=accguard.get(cap, {})
                     .get("n_cells_significantly_worse"),
                     worst_cell=accguard.get(cap, {}).get("worst_cell_delta"))),
            accuracy_note=("NOT MEASURED at this cap on the internal accuracy sweep"
                           if a is None else None)))
    for cap, blk in (r2 or {}).items():
        if not isinstance(blk, dict) or blk.get("n", 0) == 0:
            continue
        for r in rows:
            if r["cap"] == cap:
                r["vram_opentext_bestof8_arm"] = dict(
                    b_peak_allocated_gib=blk["b_peak_allocated_gib"]["peak"],
                    c_peak_reserved_gib=blk["c_peak_reserved_gib"]["peak"],
                    d_process_footprint_gib=blk["d_process_footprint_gib"]["peak"],
                    mean_b_peak_allocated_gib=blk["b_peak_allocated_gib"]["mean"],
                    peak_driver=blk.get("peak_driver"))
                break
    rows.sort(key=lambda r: -r["max_pixels"])
    return rows


def smallest_card(rows, q):
    """Which physical GPU each configuration actually fits on, by measured (d)."""
    cfgs = []
    for r in rows:
        for arm, key in (("7B direct MCQ", "vram_7b_mcq"),
                         ("open-text best-of-8 + verifier", "vram_opentext_bestof8_arm")):
            v = r.get(key)
            if not v:
                continue
            cfgs.append(dict(config=f"bf16 {arm} @ {r['cap']}", precision="bf16",
                             cap=r["cap"], d_gib=v["d_process_footprint_gib"]))
    for key, blk in (q or {}).items():
        if key.startswith("_") or not isinstance(blk, dict) or blk.get("n", 0) == 0:
            continue
        scheme, cap = key.split("__")
        cfgs.append(dict(config=f"{scheme} 7B direct MCQ @ {cap}", precision=scheme, cap=cap,
                         d_gib=blk["d_process_footprint_gib"]["peak"]))
    for c in cfgs:
        c["fits"] = {b: bool(c["d_gib"] <= cap_gib) for b, cap_gib in BOARDS_GIB.items()}
        fit = [b for b, ok in c["fits"].items() if ok]
        c["smallest_board_that_fits"] = (min(fit, key=lambda b: BOARDS_GIB[b]) if fit
                                         else "NONE of the boards listed")
    cfgs.sort(key=lambda c: c["d_gib"])
    return dict(_boards_gib=BOARDS_GIB, _board_note=BOARD_NOTE,
                _criterion="measured (d) whole-process footprint at batch 1, PEAK over the item "
                           "pool (the pool is chosen to bracket the driver space, so the peak is "
                           "a worst-case not a mean). A deployer provisions (d).",
                configurations=cfgs)


def main():
    r1, r2 = load("R1_resolution_7b_mcq"), load("R2_resolution_opentext_arm")
    q, c = load("Q_quantised_cheap_side"), load("C_coresidency")
    acc = load("accuracy_resolution_sweep")
    rows = frontier(r1, r2, acc)

    out = {
        "_meta": dict(
            title="ATTACK 4 -- the VRAM levers of the 7B-side pipeline: resolution, quantisation, "
                  "co-residency, and the smallest card the pipeline fits on",
            created="2026-08-12",
            builds_on="results/cascade_methods/artifacts/vram_testtime_2026-08-11.json, which "
                      "established that the peak is PREFILL-bound and driven by vision-token "
                      "count (46,816 patch-grid units on the worst item), not by decode -- making "
                      "image resolution the direct VRAM lever, and listing 7B+32B co-residency "
                      "under `not_measured`.",
            code=dict(vram="src/cascade/measure_vram_levers.py",
                      accuracy="src/cascade_methods/vram_levers_accuracy.py",
                      aggregator="src/cascade_methods/vram_levers_report.py"),
            logs=["logs/vram_levers_R_2026-08-12.log", "logs/vram_levers_QC_2026-08-12.log"],
            environment=dict(
                host="dual A100 80GB PCIe, driver 550.54.15",
                framework="HuggingFace transformers (NEVER vLLM -- vLLM reserves a pool and hides "
                          "true allocation, and drops all 192 visual.* LoRA modules)",
                dtype="bfloat16", attn_implementation="flash_attention_2",
                tensor_parallel=1, batch_size=1,
                card_sharing="BOTH A100s carried other rounds' jobs throughout. Every load waits "
                             "for free VRAM and claims it atomically; no other process was ever "
                             "killed. The first attempt of the resolution sweep was itself OOM-"
                             "killed by a co-tenant mid-load (logs/vram_levers_R_2026-08-12.log) "
                             "-- that is why loads now reserve before loading."),
            conventions=dict(
                units="GiB = bytes/1024**3 everywhere, matching the 08-11 artifact.",
                a_weights_resident="torch.cuda.memory_allocated() after from_pretrained().to(cuda).",
                b_peak_allocated="torch.cuda.max_memory_allocated(), reset before EVERY item.",
                c_peak_reserved="torch.cuda.max_memory_reserved(), reset before every item.",
                d_process_footprint="whole-process GPU memory, max over a 20 ms NVML sampler. "
                                    "THIS RUN MEASURES IT DIRECTLY per process: NVML reports HOST "
                                    "pids that never match os.getpid(), so our host pid is "
                                    "IDENTIFIED by the memory jump across our own model load and "
                                    "then read directly. That is a strict improvement on 08-11, "
                                    "which could only reconstruct (d) as board-used minus a "
                                    "pre-run baseline and needed an exclusive card to do it.",
                every_row_states="batch size, max_pixels, n items, and the peak driver, per the "
                                 "08-11 conventions block."),
            reading_order=["null_test", "resolution_frontier", "flops_by_cap",
                           "quantised_cheap_side", "coresidency", "smallest_card",
                           "key_findings", "not_measured"]),
        "null_test": null_test(r1, r2),
        "resolution_frontier": rows,
        "flops_by_cap": flops_block(r1),
        "accuracy_sweep_full": acc,
        "quantised_cheap_side": q,
        "coresidency": c,
        "smallest_card": smallest_card(rows, q),
        "raw": dict(R1=r1, R2=r2),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print(json.dumps(out["null_test"], indent=1)[:2000])


if __name__ == "__main__":
    main()
