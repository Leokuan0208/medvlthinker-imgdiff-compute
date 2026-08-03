#!/usr/bin/env python3
"""
bestofn_latency_reconcile.py -- close ITEM 2 of COMPREHENSIVE_WRITEUP_2026-07-30.md section 11:
the physical contradiction between the open-text best-of-N arm's 522 ms latency and its 568.8 J energy.

Pure arithmetic over (a) the CLAIMED cost model, (b) the CANONICAL measured batch-1 constants, and
(c) the NEW batch-8 measurement produced by src/cascade_methods/bestofn_measure_batch8.py.
No GPU. Launch from repo root:
    python3 src/cascade_methods/bestofn_latency_reconcile.py \
        --rep results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json \
        --rep /tmp/bestofn_rep2.json
Writes the merged verdict back into the artifact (key "reconciliation").
"""
import argparse, json, os

ap = argparse.ArgumentParser()
ap.add_argument("--rep", action="append", required=True, help="one or more replicate JSONs (measure script output)")
ap.add_argument("--out", default="results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json")
A = ap.parse_args()

# ---------------------------------------------------------------- (b) CANONICAL measured constants
# source: logs/latency_opentext.jsonl <- src/cascade_methods/open_measure_latency_energy.py (2026-06-30),
# tabulated in results/cascade_methods/docs/current/UNIFIED_METHOD_EXPERIMENTS.md L228-234
GEN7_MS, GEN7_J = 347.1, 45.80          # Lingshu-7B one greedy open-text generation, batch-1
VER7_MS, VER7_J = 175.5, 25.28          # Lingshu-7B one verifier forward, batch-1
GEN32_MS, GEN32_J = 665.0, 126.89       # Lingshu-32B one generation (no-think), batch-1
GEN32T_MS = 10521.6                     # Lingshu-32B THINK, measured batch-1 (integrated_method.py::GEN32T)
CARD_LIMIT_W = 300.0                    # A100 80GB PCIe enforced limit, read from NVML this run
CARD_LIMIT_W_SXM = 400.0                # the bound the writeup assumed

# ---------------------------------------------------------------- (a) the CLAIMED (modelled) cost model
K = 8
BO8_MS_CLAIMED = 522.0                  # integrated_method.py::BO8 = "1 gen + 1 verify; N drops out"
BO8_MS_CLAIMED_EXACT = GEN7_MS + VER7_MS
BO8_J_CLAIMED = K * (GEN7_J + VER7_J)   # macro_headline_clean_verifier.py::BO8_J / latency_reexamination.py::energy
BO8_SEQ_MS_CLAIMED = K * (GEN7_MS + VER7_MS)

# ---------------------------------------------------------------- (c) NEW measurement, pooled over replicates
reps = [json.load(open(p)) for p in A.rep]


def pooled(key, field):
    num = sum(r["measured"][key]["n"] * r["measured"][key][field] for r in reps)
    den = sum(r["measured"][key]["n"] for r in reps)
    return num / den


M = {k: dict(n=sum(r["measured"][k]["n"] for r in reps),
             lat_ms_mean=round(pooled(k, "lat_ms_mean"), 1),
             lat_ms_median=round(pooled(k, "lat_ms_median"), 1),
             lat_ms_sd_within_rep=[r["measured"][k]["lat_ms_sd"] for r in reps],
             lat_ms_p10_p90=[[r["measured"][k]["lat_ms_p10"], r["measured"][k]["lat_ms_p90"]] for r in reps],
             energy_j_mean=round(pooled(k, "energy_j_mean"), 2),
             per_replicate_lat_ms_mean=[r["measured"][k]["lat_ms_mean"] for r in reps],
             per_replicate_energy_j_mean=[r["measured"][k]["energy_j_mean"] for r in reps])
     for k in ("gen1", "gen8", "verify1", "verify8", "bo8_total")}

BO8_MS_MEAS = M["bo8_total"]["lat_ms_mean"]
BO8_J_MEAS = M["bo8_total"]["energy_j_mean"]

# ---------------------------------------------------------------- the contradiction, stated arithmetically
contradiction = dict(
    claimed_latency_ms=BO8_MS_CLAIMED,
    claimed_latency_construction=f"GEN7 {GEN7_MS} + VER7 {VER7_MS} = {BO8_MS_CLAIMED_EXACT:.1f} ms (rounded to 522)",
    claimed_latency_status="MODELLED / ASSERTED — no batch-8 wall-clock was ever measured",
    claimed_energy_j=round(BO8_J_CLAIMED, 1),
    claimed_energy_construction=f"{K} x (GEN7_J {GEN7_J} + VER7_J {VER7_J}) = {BO8_J_CLAIMED:.1f} J",
    claimed_energy_status="MODELLED — assumes each of the 8 gens and 8 verifies costs its FULL batch-1 energy",
    implied_power_w=round(BO8_J_CLAIMED / (BO8_MS_CLAIMED / 1000.0), 1),
    measured_power_during_gen7_w=round(GEN7_J / (GEN7_MS / 1000.0), 1),
    card_enforced_limit_w=CARD_LIMIT_W,
    n_gpus_in_canonical_measurement=1,
    verdict=(f"{BO8_J_CLAIMED:.1f} J delivered in {BO8_MS_CLAIMED/1000:.4f} s requires "
             f"{BO8_J_CLAIMED/(BO8_MS_CLAIMED/1000.0):.0f} W on the ONE A100 80GB PCIe the constants were "
             f"measured on, whose NVML-enforced limit is {CARD_LIMIT_W:.0f} W "
             f"({BO8_J_CLAIMED/(BO8_MS_CLAIMED/1000.0)/CARD_LIMIT_W:.1f}x the limit) and which actually drew "
             f"{GEN7_J/(GEN7_MS/1000.0):.0f} W during GEN7. Physically impossible; the two figures cannot both "
             f"be right."),
    energy_consistent_lower_bounds_s=dict(
        at_400W_assumed_by_writeup=round(BO8_J_CLAIMED / CARD_LIMIT_W_SXM, 3),
        at_300W_actual_card_limit=round(BO8_J_CLAIMED / CARD_LIMIT_W, 3),
    ),
)

# ---------------------------------------------------------------- which figure was wrong
resolution = dict(
    latency=dict(claimed_ms=BO8_MS_CLAIMED, measured_ms=BO8_MS_MEAS,
                 factor_wrong=round(BO8_MS_MEAS / BO8_MS_CLAIMED, 2),
                 verdict="WRONG — understated by ~2.5x. Batching 8 does NOT make N drop out."),
    energy=dict(claimed_j=round(BO8_J_CLAIMED, 1), measured_j=BO8_J_MEAS,
                factor_wrong=round(BO8_J_MEAS / BO8_J_CLAIMED, 2),
                verdict="ALSO WRONG, in the other direction — overstated by ~1.8x. Batching does save "
                        "energy (shared prefill/weight reads), just not latency."),
    why=("The cost model applied PERFECT parallelism to latency (N drops out) and ZERO parallelism to "
         "energy (N multiplies). Reality is in between and much closer to the energy side: a batch-8 "
         "forward is far from free in wall-clock. Measured batch-8 speedup over 8 sequential calls is "
         "only ~{:.1f}x for generation and ~{:.1f}x for verification, not 8x.").format(
             (K * M["gen1"]["lat_ms_mean"]) / M["gen8"]["lat_ms_mean"],
             (K * M["verify1"]["lat_ms_mean"]) / M["verify8"]["lat_ms_mean"]),
    harness_reproduces_canonical=dict(
        gen1_measured_ms=M["gen1"]["lat_ms_mean"], gen1_canonical_ms=GEN7_MS,
        gen1_pct_diff=round(100 * (M["gen1"]["lat_ms_mean"] - GEN7_MS) / GEN7_MS, 1),
        verify1_measured_ms_median=M["verify1"]["lat_ms_median"], verify1_canonical_ms_median=173.0,
        note="gen1 reproduces GEN7 to within a few percent, so the batch-8 numbers are on the same footing "
             "as the canonical constants. verify1's MEAN is inflated by a few multi-hundred-ms outliers "
             "(LoRA adapter toggling + NVML sampling jitter); its MEDIAN is the comparable statistic."),
)

# ---------------------------------------------------------------- corrected arm-level numbers
def arm(label, esc, claimed_lat_ms, claimed_saved_pct, source):
    cl = claimed_lat_ms
    co = BO8_MS_MEAS + esc * GEN32_MS
    return dict(
        label=label, escalation_rate=esc, source=source,
        claimed_latency_ms=cl, corrected_latency_ms=round(co, 1),
        claimed_saved_vs_32b_think_pct=claimed_saved_pct,
        corrected_saved_vs_32b_think_pct=round(100 * (1 - co / GEN32T_MS), 1),
        claimed_x_of_one_32b_forward=round(cl / GEN32_MS, 2),
        corrected_x_of_one_32b_forward=round(co / GEN32_MS, 2),
        claimed_energy_j=round(BO8_J_CLAIMED + esc * GEN32_J, 1),
        corrected_energy_j=round(BO8_J_MEAS + esc * GEN32_J, 1),
        corrected_energy_x_of_one_32b_forward=round((BO8_J_MEAS + esc * GEN32_J) / GEN32_J, 2),
        flips_from_faster_to_slower_than_32b=(cl < GEN32_MS) and (co > GEN32_MS),
    )


arms = [
    arm("open-text best-of-8 + verifier gate (fixed-bo8 reference, esc=3.97%)", 0.0397, 548.4, 94.8,
        "method_final.json /modes/compute_lean/compute_lean_fixed_bo8_reference/pooled/open_only/"
        "sample_weighted; also integrated_pandora_opentext.json method_latency_ms=548.3"),
    arm("open-text best-of-8 + verifier gate (compute_lean / accuracy_max, esc=28.23%)", 0.2823, 709.7, 93.3,
        "method_final.json /modes/{compute_lean,accuracy_max}/pooled/open_only/sample_weighted"),
    arm("open-text best-of-8 + verifier gate (method_final_v2, esc=25.76%)", 0.2576, 693.3, 93.4,
        "method_final_v2.json /modes_v2/{compute_lean,accuracy_max}/pooled/open_only/sample_weighted"),
]

deployment = dict(
    question="Is batching 8 draws even the right model of batch-1 serving?",
    sequential_8_draws_measured_ms=round(K * (M["gen1"]["lat_ms_mean"] + M["verify1"]["lat_ms_median"]), 1),
    sequential_8_draws_claimed_ms=round(BO8_SEQ_MS_CLAIMED, 1),
    batched_8_measured_ms=BO8_MS_MEAS,
    answer=("Batching is a legitimate model ONLY for a dedicated single-request server that can hold 8 "
            "concurrent sequences; it converts idle 8-wide throughput into latency. It is measured here and "
            "IS ~{:.1f}x faster than drawing sequentially, so batching is the right choice IF you are doing "
            "best-of-8 at all. What is NOT legitimate is the claim that batching makes N free: measured "
            "batch-8 costs {:.0f} ms vs {:.0f} ms for a single greedy draw ({:.1f}x), and the whole best-of-8 "
            "round-trip is {:.0f} ms vs a {:.0f} ms single 32B forward. Under ANY serving assumption "
            "(sequential {:.0f} ms or batched {:.0f} ms) the open-text best-of-8 arm is SLOWER than simply "
            "calling the 32B once. In a real multi-tenant server the batched figure is also optimistic, "
            "because the 8-wide slots are not free — they displace other requests.").format(
                (K * (M["gen1"]["lat_ms_mean"] + M["verify1"]["lat_ms_median"])) / BO8_MS_MEAS,
                M["gen8"]["lat_ms_mean"], M["gen1"]["lat_ms_mean"],
                M["gen8"]["lat_ms_mean"] / M["gen1"]["lat_ms_mean"],
                BO8_MS_MEAS, GEN32_MS,
                K * (M["gen1"]["lat_ms_mean"] + M["verify1"]["lat_ms_median"]), BO8_MS_MEAS),
)

out = dict(
    item="COMPREHENSIVE_WRITEUP_2026-07-30.md section 11, item 2 — best-of-N latency/energy contradiction",
    date="2026-08-03",
    status="RESOLVED by direct measurement",
    canonical_constants=dict(GEN7_ms=GEN7_MS, GEN7_J=GEN7_J, VER7_ms=VER7_MS, VER7_J=VER7_J,
                             GEN32_nothink_ms=GEN32_MS, GEN32_nothink_J=GEN32_J, GEN32_think_ms=GEN32T_MS,
                             source="logs/latency_opentext.jsonl via src/cascade_methods/"
                                    "open_measure_latency_energy.py, 2026-06-30, 1 visible GPU, n=25"),
    contradiction=contradiction,
    measurement=dict(
        script="src/cascade_methods/bestofn_measure_batch8.py",
        replicates=[dict(file=p, n_kept=r["n_kept"], gpu=r["gpu_names"], limit_w=r["gpu_power_limit_w"],
                         idle_w_model_resident=r["idle_w_model_resident"],
                         prefill_tok_mean=r["prefill_tok_mean"], gen_tok_mean=r["gen_tok_mean"])
                    for p, r in zip(A.rep, reps)],
        pooled=M,
    ),
    resolution=resolution,
    corrected_arms=arms,
    deployment_model=deployment,
)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
base = json.load(open(A.out)) if os.path.exists(A.out) else {}
base["reconciliation"] = out
with open(A.out, "w") as fh:
    json.dump(base, fh, indent=2)
print(json.dumps(out, indent=2))
print("\nwrote " + A.out)
