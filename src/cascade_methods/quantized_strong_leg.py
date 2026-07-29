#!/usr/bin/env python3
"""
quantized_strong_leg.py -- G3: re-cost the integrated format-aware cascade with an AWQ/GPTQ-INT4
strong leg (32B escalation), to test whether INT4 makes the method faster on FLOPs (not just latency)
vs always-32B-no-think while holding the accuracy win.

NO fabricated numbers. Every field is tagged MEASURED or PROJECTED:
  * MEASURED  -- batch-1 vLLM latency of AWQ-INT4 vs FP16 on Qwen2.5-VL-32B (bench_int4_strong_leg.py).
                 Lingshu-32B IS a medical fine-tune of Qwen2.5-VL-32B, so batch-1 prefill/decode
                 *latency* (architecture+kernel dependent, weight-value independent) transfers exactly.
  * PROJECTED -- INT4 medical ACCURACY delta (literature <=1%); and the "throughput-effective FLOP"
                 figure the task hypothesizes (INT4 strong leg ~1.5x 7B). We have NO pre-quantized
                 Lingshu-32B and DID NOT quantize it ourselves (bounded); accuracy stays projected.

TWO FLOP ACCOUNTINGS (kept distinct on purpose):
  A) MAC-FLOP  == the repo's own unit ("7B-forward-equivalents", pandora_controller.py). This counts
     multiply-accumulates and is WEIGHT-PRECISION-INDEPENDENT: a 32B INT4 forward does the SAME MACs
     as a 32B FP16 forward = 4.57x the 7B. So under the repo's own FLOP unit, INT4 changes NOTHING.
  B) THROUGHPUT-EFFECTIVE FLOP == MAC-FLOP / (hardware speedup at that precision). This is the task's
     "~1.5x" framing. It is only ~1.5x when the op is DECODE-heavy (memory-bound, INT4 reads 4x fewer
     weight bytes). The method's strong leg is 32B-NO-THINK = PREFILL-bound (~2 gen tokens), where
     INT4/AWQ-Marlin dequantizes to FP16 for the matmul and gives little/no speedup -> so the honest
     effective figure for THIS leg is NOT ~1.5x. Both are reported.

Run from repo root:  python3 src/cascade_methods/quantized_strong_leg.py
"""
import json, os
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SRC = os.path.join(ROOT, "results/cascade_methods/artifacts/integrated_method_vs_think.json")
BENCH = os.path.join(ROOT, "results/cascade_methods/artifacts/int4_bench_raw.jsonl")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/quantized_strong_leg.json")

M = json.load(open(SRC))
CC = M["cost_constants"]
GEN7_MS = CC["GEN7"]["ms"]; BO8_MS = CC["BO8"]["ms"]
GEN32N_MS = CC["GEN32_nothink"]["ms"]; GEN32N_FLOP = CC["GEN32_nothink"]["flop"]   # 665ms / 4.57
GEN32T_MS = CC["GEN32_think"]["ms"]                                                # 10521.6ms
CHEAP_FLOP = {"MCQ": CC["GEN7"]["flop"], "open": CC["BO8"]["flop"]}                # 1.0 / 16.0


# ---------- 1. INT4 latency multiplier: MEASURED (same arch) or PROJECTED (literature) ----------
def load_bench():
    """Optional: if the AWQ vLLM benchmark ran (bench_int4_strong_leg.py), use its MEASURED ratio.
    (Not available this session: a HF-CDN outage blocked 2 of 6 AWQ shards; the script is ready.)"""
    if not os.path.exists(BENCH):
        return None
    rows = [json.loads(l) for l in open(BENCH) if l.strip()]
    by = {r["tag"]: r for r in rows}
    if "int4" not in by or "fp16" not in by:
        return None
    i4, f16 = by["int4"], by["fp16"]
    return dict(source="MEASURED (vLLM batch-1, Qwen2.5-VL-32B AWQ-INT4 vs FP16, same arch as Lingshu-32B)",
                strong_leg_latency_ratio=round(i4["regimes"]["4"]["lat_ms_med"] / f16["regimes"]["4"]["lat_ms_med"], 4),
                decode_per_tok_ratio=round(i4["decode_ms_per_tok"] / f16["decode_ms_per_tok"], 4),
                int4_raw=i4, fp16_raw=f16)


def latency_composition():
    """Prefill/decode split of the 32B (Qwen2.5-VL-32B arch) from the repo's OWN measured latency
    (results/.../latency_32b.jsonl, HF batch-1). Grounds the INT4 latency projection in real data:
    AWQ-INT4 accelerates only memory-bound DECODE (~2-3x on A100, literature), NOT compute-bound
    PREFILL (dequant to fp16) nor the FP16-kept vision tower. So the INT4 win on the NO-THINK strong
    leg (~2 gen tok) is bounded by its small decode fraction."""
    import statistics as st
    from collections import defaultdict
    p = os.path.join(ROOT, "results/cascade_methods/artifacts/latency_32b.jsonl")
    g = defaultdict(list)
    for l in open(p):
        if l.strip():
            r = json.loads(l); g[r["config"]].append(r)
    comp = {c: dict(lat=st.median([r["latency_s"] for r in g[c]]) * 1000,
                    gen=st.median([r["gen_tok"] for r in g[c]])) for c in g}
    nt, th = comp["nothink@cap320"], comp["think@cap320"]
    decode_ms_tok = (th["lat"] - nt["lat"]) / (th["gen"] - nt["gen"])       # ~68.6 ms/tok (32B, HF b1)
    return dict(decode_ms_per_tok=round(decode_ms_tok, 2),
                nothink_lat=round(nt["lat"], 1), nothink_gen=nt["gen"],
                think_lat=round(th["lat"], 1), think_gen=th["gen"],
                nothink_decode_frac=round(nt["gen"] * decode_ms_tok / nt["lat"], 3))


bench = load_bench()
COMP = latency_composition()

# INT4 decode speedup on A100 (AWQ-Marlin, memory-bound decode): literature bracket [2.0, 3.0], mid 2.5.
INT4_DECODE_SPEEDUP = [2.0, 2.5, 3.0]

# Decompose the method's 665 ms no-think strong leg into prefill + decode using the DIRECTLY-MEASURED
# 32B decode rate (68.6 ms/tok from latency_32b.jsonl, actual token counts) x the leg's ~2 gen tokens.
# INT4/AWQ speeds ONLY the memory-bound decode part; prefill (compute-bound, dequant-to-fp16) and the
# FP16-kept vision tower are INT4-invariant.
DECODE_MS_TOK = COMP["decode_ms_per_tok"]                    # ~68.6 ms/tok, MEASURED
GEN_NOTHINK_TOK = COMP["nothink_gen"]                        # ~2 tokens (letter/short answer)
NOTHINK_DECODE_MS = GEN_NOTHINK_TOK * DECODE_MS_TOK          # ~137 ms decode inside the 665 ms leg
PREFILL_MS = GEN32N_MS - NOTHINK_DECODE_MS                   # ~528 ms prefill+vision (INT4-invariant)
STRONG_DECODE_FRAC = NOTHINK_DECODE_MS / GEN32N_MS           # ~0.21 of the strong leg is decode
GEN_THINK_TOK = 300.0                                        # think baseline ~275-320 reason tokens (JSON note)


def int4_strong_ms(speedup):
    """no-think strong-leg latency with INT4: prefill unchanged + decode/speedup."""
    return PREFILL_MS + NOTHINK_DECODE_MS / speedup


if bench:
    R_STRONG_LAT = bench["strong_leg_latency_ratio"]
    INT4_STRONG_MS = GEN32N_MS * R_STRONG_LAT
    LAT_SRC = "MEASURED"
else:
    INT4_STRONG_MS = int4_strong_ms(INT4_DECODE_SPEEDUP[1])          # midpoint 2.5x decode speedup
    R_STRONG_LAT = INT4_STRONG_MS / GEN32N_MS
    LAT_SRC = "PROJECTED (composition-grounded: prefill-bound no-think leg; INT4 speeds decode only)"

# strong-leg FLOP under each accounting
STRONG_FLOP = {
    "A_mac_flop_repo_unit": GEN32N_FLOP,                    # 4.57 -- UNCHANGED by INT4 (weight-precision independent)
    "B_effective_literature": 1.5,                          # PROJECTED task hypothesis (decode-heavy assumption)
    "B_effective_prefillbound_projected": round(GEN32N_FLOP * R_STRONG_LAT, 3),  # throughput-eff for THIS prefill-bound leg
}


# ---------- 2. INT4 accuracy delta on the strong leg: PROJECTED ----------
# literature W4A16 AWQ/GPTQ on Qwen2.5-VL-class: <=1% full-benchmark accuracy loss. The strong leg only
# answers the ESCALATED subset, so method-acc drop = esc * delta. Bracketed 0.5% and 1.0%.
DELTA_STRONG = [0.005, 0.010]


# ---------- 3. re-cost every benchmark ----------
def recost(v, strong_flop_key, strong_ms, delta):
    """return method flops/latency/acc for one benchmark under an INT4 strong leg."""
    fmt = v["format"]; m = v["method"]; esc = m["esc"]
    if v.get("gate") == "none (keep 7B)" or esc == 0.0:
        # MMMU keep-7B (or 0% escalation): strong leg never fires -> identical
        return dict(flops=m["flops"], latency_ms=m["latency_ms"], acc=m["acc"], esc=esc, strong_fires=False)
    cheap_flop = CHEAP_FLOP[fmt]
    cheap_ms = m["cheap_leg_ms"]
    sf = STRONG_FLOP[strong_flop_key]
    sf = GEN32N_FLOP if not isinstance(sf, (int, float)) else sf
    flops = cheap_flop + esc * sf
    lat = cheap_ms + esc * strong_ms
    # accuracy: escalated samples answered by INT4 32B (delta worse); kept samples unchanged.
    # method_acc = base_method_acc - esc*delta   (the strong leg's share degrades by delta)
    acc = m["acc"] - esc * delta
    return dict(flops=round(flops, 3), latency_ms=round(lat, 1), acc=round(acc, 4), esc=esc, strong_fires=True)


per = {}
for name, v in M["per_benchmark"].items():
    if "method" not in v:
        continue
    row = dict(format=v["format"], n=v["n"], esc=v["method"]["esc"],
               fp16=dict(flops=v["method"]["flops"], latency_ms=v["method"]["latency_ms"], acc=v["method"]["acc"]),
               acc_32b_nothink=v["acc_32b_nothink"])
    # INT4 scenarios (all use the composition-grounded INT4 strong-leg latency; they differ in FLOP accounting)
    row["int4_macflop_reponunit"] = recost(v, "A_mac_flop_repo_unit", INT4_STRONG_MS, DELTA_STRONG[1])
    row["int4_effective_literature"] = recost(v, "B_effective_literature", INT4_STRONG_MS, DELTA_STRONG[1])
    row["int4_effective_prefillbound"] = recost(v, "B_effective_prefillbound_projected", INT4_STRONG_MS, DELTA_STRONG[1])
    per[name] = row


# ---------- 4. pool (sample-weighted), mirror integrated_method pooling ----------
def pool(keys, field_path):
    num = 0.0; den = 0
    for k in keys:
        v = per[k]; n = v["n"]
        x = v
        for p in field_path:
            x = x[p]
        num += x * n; den += n
    return num / den if den else float("nan")


MCQ = [k for k in per if per[k]["format"] == "MCQ"]
OPEN = [k for k in per if per[k]["format"] == "open"]
ALL = list(per.keys())

pooled = {}
for label, keys in [("full_suite", ALL), ("mcq_only", MCQ), ("open_only", OPEN)]:
    d = dict(n=sum(per[k]["n"] for k in keys),
             fp16_flops=round(pool(keys, ["fp16", "flops"]), 3),
             fp16_latency_ms=round(pool(keys, ["fp16", "latency_ms"]), 1),
             fp16_acc=round(pool(keys, ["fp16", "acc"]), 4),
             int4_macflop_flops=round(pool(keys, ["int4_macflop_reponunit", "flops"]), 3),
             int4_lit_flops=round(pool(keys, ["int4_effective_literature", "flops"]), 3),
             int4_lit_latency_ms=round(pool(keys, ["int4_effective_literature", "latency_ms"]), 1),
             int4_prefillbound_eff_flops=round(pool(keys, ["int4_effective_prefillbound", "flops"]), 3),
             int4_latency_ms=round(pool(keys, ["int4_macflop_reponunit", "latency_ms"]), 1),
             int4_acc=round(pool(keys, ["int4_macflop_reponunit", "acc"]), 4),
             always_32bnt_flops=GEN32N_FLOP,
             acc_32b_nothink=round(pool(keys, ["acc_32b_nothink"]), 4))
    pooled[label] = d


# ---------- 5. central-question verdict ----------
fs = pooled["full_suite"]; oo = pooled["open_only"]; mm = pooled["mcq_only"]
ESC_FS = M["pooled"]["full_suite"]["sample_weighted"]["esc"]   # 0.1548 pooled escalation rate
verdict = {
    "central_question": "Does an INT4 strong leg make the integrated method faster on FLOPs (not just "
                        "latency) vs always-32B-no-think, while holding the accuracy win?",
    "A_under_repo_MAC_FLOP_unit": (
        "NO. The repo's FLOP unit is MAC-count (7B-forward-equivalents); INT4 is a memory/precision "
        "optimization with ZERO MAC reduction, so a 32B-INT4 forward is still 4.57 FLOP-eq. Method FLOPs "
        "are literally unchanged: full-suite %.3f, mcq %.3f, open %.3f. INT4 helps LATENCY/energy/VRAM, "
        "not MAC-FLOPs." % (fs["int4_macflop_flops"], mm["int4_macflop_flops"], oo["int4_macflop_flops"])),
    "B_under_throughput_effective_FLOP": (
        "PARTIALLY, and only where escalation is FREQUENT. If you credit INT4 with the task's ~1.5 "
        "effective-FLOP-eq strong leg, the method's high-escalation MCQ cells drop below always-32B-nt "
        "(4.57): MedXpert 5.09->%.2f, VQA_RAD_closed 3.60->%.2f, PathVQA_closed 3.09->%.2f. But the "
        "OPEN-TEXT arm is UNCHANGED (16.0->~16.0): its FLOP is 97-98%% best-of-8 cheap 7B forwards, the "
        "strong leg is only 2-3%%, so quantizing the strong leg CANNOT close the open-text FLOP gap. "
        "Pooled full-suite (already 2.54<4.57 at FP16) -> %.2f." % (
            per["MedXpertQA-MM"]["int4_effective_literature"]["flops"],
            per["VQA_RAD_closed"]["int4_effective_literature"]["flops"],
            per["PATH_VQA_closed"]["int4_effective_literature"]["flops"],
            fs["int4_lit_flops"])),
    "latency": (("MEASURED strong-leg INT4/FP16 batch-1 ratio = %.3f (Qwen2.5-VL-32B AWQ, same arch)."
                 % R_STRONG_LAT) if bench else
                ("PROJECTED, composition-grounded (NOT a flat literature factor). KEY FINDING: the strong "
                 "leg is 32B-NO-THINK, which the repo's OWN measured 32B latency (latency_32b.jsonl, "
                 "decode=%.0f ms/tok) shows is PREFILL-DOMINATED (~2 gen tok; decode is only ~%.0f%% of "
                 "the 665 ms leg). AWQ-INT4 accelerates ONLY memory-bound DECODE (~%.1f-%.1fx on A100), "
                 "not compute-bound PREFILL (dequant to fp16) nor the FP16-kept vision tower. So the "
                 "strong leg drops only %d ms -> %d ms (ratio ~%.2f, a ~%.0f%% cut) -- NOT the ~2.5-3x "
                 "the task hypothesizes. The full INT4 decode win DOES apply to the always-32B-THINK "
                 "baseline (~300 decode tok): ~%d ms -> ~%d ms -- but the method already beats that "
                 "baseline by ~23x on latency, so quantizing it is moot for the comparison."
                 % (COMP["decode_ms_per_tok"], STRONG_DECODE_FRAC * 100,
                    INT4_DECODE_SPEEDUP[0], INT4_DECODE_SPEEDUP[-1],
                    GEN32N_MS, INT4_STRONG_MS, R_STRONG_LAT, (1 - R_STRONG_LAT) * 100,
                    GEN32T_MS, PREFILL_MS + (GEN32T_MS - PREFILL_MS) / INT4_DECODE_SPEEDUP[1]))),
    "accuracy": ("PROJECTED HOLD. INT4 medical accuracy loss <=1%% (literature, W4A16 AWQ/GPTQ on "
                 "Qwen2.5-VL-class). It only touches the ESCALATED subset -> method-acc drop = esc*delta = "
                 "full-suite -%.4f..-%.4f (esc=%.2f). Win vs 32B-THINK (+0.0118) is robustly held; the "
                 "razor-thin +0.0018 vs 32B-NO-THINK could erode to +%.4f..+%.4f. Open-text accuracy win "
                 "is FULLY preserved (its engine is the UNquantized 7B best-of-8, not the strong leg)."
                 % (DELTA_STRONG[0] * ESC_FS, DELTA_STRONG[1] * ESC_FS, ESC_FS,
                    0.0018 - DELTA_STRONG[1] * ESC_FS, 0.0018 - DELTA_STRONG[0] * ESC_FS)),
    "bottom_line": (
        f"The task's hypothesis ('INT4 strong leg ~1.5x FLOPs / ~2.5x faster, making the method faster on "
        f"FLOPs too') does NOT hold as stated, for two structural reasons this analysis surfaces: (1) FLOPs "
        f"-- under the repo's MAC-count unit INT4 gives ZERO reduction; only under a throughput-effective "
        f"accounting does the strong leg get cheaper, and even then it fixes only the HIGH-escalation MCQ "
        f"cells (MedXpert, VQA_RAD/PathVQA closed), NOT the open-text FLOP overage (which is 97-98pct "
        f"best-of-8 cheap-7B forwards -- the strong leg is only 2-3pct there, so strong-leg quantization "
        f"is the WRONG lever for it). (2) Latency -- the ~2.5x INT4 speedup is a DECODE-heavy number; the "
        f"method's strong leg is 32B-NO-THINK = PREFILL-bound (~2 gen tok, decode ~{STRONG_DECODE_FRAC*100:.0f}pct "
        f"of the call), so INT4 cuts it only ~{(1-R_STRONG_LAT)*100:.0f}pct (665->{INT4_STRONG_MS:.0f} ms), "
        f"not ~2.5x. INT4 IS a real energy/VRAM win and lets the escalation model fit tp=1 on one GPU "
        f"(fixing the OmniMed tp=2 hang). NET: INT4 strong leg is worth adopting for VRAM/energy/"
        f"deployability and cleans up the MCQ FLOP story under a throughput-effective view, but it does "
        f"NOT make the method uniformly FLOP-dominant vs always-32B-nt and does NOT materially change "
        f"latency (which the method already wins by ~23x). The open-text FLOP overage needs a different "
        f"lever: fewer best-of-N samples or a cheaper verifier."),
}

out = dict(
    title="G3: integrated cascade re-costed with an AWQ/GPTQ-INT4 strong leg "
          "(latency PROJECTED composition-grounded; accuracy & effective-FLOP PROJECTED)",
    baseline_method_source=SRC,
    tractability_note=(
        "Option 1 (pre-quantized Lingshu-32B for vLLM): NOT AVAILABLE. HF has only MLX (Apple-only), "
        "GGUF (llama.cpp, poor Qwen2.5-VL vision support), a bitsandbytes-NF4 *7B*, and dead/empty repos "
        "-- no AWQ/GPTQ-INT4 Lingshu-32B loadable in vLLM. Option 2 (quantize Lingshu-32B ourselves): "
        "NOT DONE (bounded) -- no quant libs installed (no autoawq/gptqmodel/llmcompressor), and "
        "calibrating+quantizing a 32B VLM is the rabbit hole the task forbids. Option 3 taken: re-cost "
        "analytically. Plan A was to MEASURE INT4-vs-FP16 batch-1 latency on the identical architecture "
        "(official Qwen/Qwen2.5-VL-32B-Instruct-AWQ, of which Lingshu-32B is a medical fine-tune; a ready "
        "vLLM benchmark is committed at src/cascade_methods/bench_int4_strong_leg.py) -- but a HF-CDN "
        "outage this session stalled 2 of 6 AWQ shards (xet/curl/hf_transfer all failed on those blobs). "
        "So latency is PROJECTED from the repo's OWN measured 32B prefill/decode composition; INT4 "
        "ACCURACY is projected from literature. Empirical AWQ latency + Lingshu-32B-INT4 accuracy = "
        "future work (re-run bench_int4_strong_leg.py when the CDN recovers; INT4-32B tp=1 ~20GB fits 1 "
        "GPU, memory-safe, avoids the OmniMed tp=2 hang)."),
    flop_accounting_note=(
        "A) MAC-FLOP = repo unit (7B-fwd-eq), weight-precision INDEPENDENT: INT4-32B = 4.57 (UNCHANGED; "
        "INT4 cuts bits/bytes, not multiply-accumulates). B) THROUGHPUT-EFFECTIVE FLOP = MAC/hardware-"
        "speedup: ~1.5x only for DECODE-heavy ops; the no-think strong leg is PREFILL-bound, so its "
        "throughput-effective figure is ~%.1f (near 4.57), far from 1.5. Note also: quantizing the strong "
        "leg but NOT the baseline is an apples-to-oranges comparison; a fair 'always-INT4-32B-nt' baseline "
        "scales identically, leaving the method/baseline FLOP RATIO ~invariant." % STRONG_FLOP["B_effective_prefillbound_projected"]),
    latency_composition_measured=COMP,
    int4_decode_speedup_bracket_literature=INT4_DECODE_SPEEDUP,
    strong_leg_latency=dict(fp16_ms=GEN32N_MS, int4_ms=round(INT4_STRONG_MS, 1), ratio=round(R_STRONG_LAT, 4),
                            prefill_ms=round(PREFILL_MS, 1), nothink_decode_ms=round(NOTHINK_DECODE_MS, 1),
                            source=LAT_SRC),
    int4_latency_ratio=dict(value=round(R_STRONG_LAT, 4), source=LAT_SRC,
                            detail=(bench if bench else "composition-grounded projection (see latency_composition_measured)")),
    strong_leg_flop_by_accounting=STRONG_FLOP,
    int4_strong_leg_ms=round(INT4_STRONG_MS, 1),
    accuracy_delta_projected=DELTA_STRONG,
    per_benchmark=per,
    pooled=pooled,
    verdict=verdict,
)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=2, default=str)

# ---------- console ----------
print("=" * 120)
print("G3  INT4 STRONG LEG  re-costing   (latency=%s ; accuracy/FLOP-effective=PROJECTED)" % LAT_SRC)
print("=" * 120)
print("32B latency composition (MEASURED, latency_32b.jsonl): decode=%.1f ms/tok; strong-leg (665ms) decode-frac=%.0f%%"
      % (COMP["decode_ms_per_tok"], STRONG_DECODE_FRAC * 100))
print("strong leg (no-think) INT4/FP16 latency ratio = %.3f  ->  %d ms -> %d ms  (prefill %d ms unchanged + decode %d ms /%.1fx)"
      % (R_STRONG_LAT, GEN32N_MS, INT4_STRONG_MS, PREFILL_MS, NOTHINK_DECODE_MS, INT4_DECODE_SPEEDUP[1]))
print("strong-leg FLOP by accounting:", STRONG_FLOP)
print("-" * 120)
h = f"{'benchmark':<18}{'esc%':>5}{'FP16 F':>8}{'MACeq F':>8}{'LIT F':>7}{'vs4.57':>8}{'FP16 lat':>9}{'INT4 lat':>9}"
print(h); print("-" * len(h))
for k, v in per.items():
    a = v["int4_macflop_reponunit"]; b = v["int4_effective_literature"]
    flag = "MORE" if b["flops"] > GEN32N_FLOP else "less"
    print(f"{k:<18}{v['esc']*100:>4.0f}%{v['fp16']['flops']:>8.2f}{a['flops']:>8.2f}{b['flops']:>7.2f}{flag:>8}"
          f"{v['fp16']['latency_ms']:>8.0f}m{a['latency_ms']:>8.0f}m")
print("-" * len(h))
for lab in ("full_suite", "mcq_only", "open_only"):
    p = pooled[lab]
    print(f"POOLED[{lab:<10}] n={p['n']:5d}  FP16_F={p['fp16_flops']:.2f}  MACeq_F(unchanged)={p['int4_macflop_flops']:.2f}  "
          f"LIT_eff_F={p['int4_lit_flops']:.2f}  prefillbound_eff_F={p['int4_prefillbound_eff_flops']:.2f}  "
          f"(always-32Bnt=4.57)  FP16_lat={p['fp16_latency_ms']:.0f}ms INT4_lat={p['int4_latency_ms']:.0f}ms")
print("\nVERDICT")
for kk, vv in verdict.items():
    print(f"  [{kk}]\n    {vv}\n")
print(f"wrote {OUT}")
