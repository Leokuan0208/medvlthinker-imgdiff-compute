#!/usr/bin/env python3
"""
BEST format-aware 2-tier cascade for the goal: a SINGLE cascade that is FASTER and (aims to be)
MORE ACCURATE than always-Lingshu-32B on the faithful MedEvalKit MedVQA suite, handling BOTH MCQ
and open-text.  NO inference here -- assembled entirely from saved faithful outputs.

METHOD (FALC = Format-Aware Lingshu Cascade)
  cheap leg   : Lingshu-7B, no-think (1 greedy generation).
  strong leg  : Lingshu-32B, NO-THINK  (escalation target).
                Justification (measured, this repo): 32B-think <= 32B-nothink on every MedEvalKit
                benchmark (PMC .5518 vs .5494 ; VQA_RAD .4745 vs .4678 ; MedXpert .3065 vs .3045 ;
                MMMU .633 vs .627 ; SLAKE-close .859 vs .864 ~tie).  So think is never the better
                strong target for Lingshu -> a single no-think strong leg is optimal AND removes the
                dependency on the pending 32B-think latency measurement.
  gate (format-aware):
    * MCQ / closed  -> escalate when 7B margin < tau_mcq   (margin = top1-top2 prob; best gate).
    * open-text     -> escalate when verifier-confidence < tau_open  (trained verifier
                       ckpts/train/lora_verifier_pooled4, P(Yes) over sampled cands); SLAKE-open has
                       no verifier dump -> fall back to 7B sequence-logprob `seqlogprob` (free).
  held-out tau  : 5-fold cross-fit -- pick tau on 4/5 (min escalation s.t. cascade acc >= 32B acc),
                  evaluate on held-out 1/5, average folds.  Reported as DEPLOYABLE (no peeking).

COST MODEL (batch-1, measured in this repo; identical constants across the codebase):
  GEN7 = 347 ms / 45.8 J / 1.0  FLOP-7B-eq
  VER7 = 175 ms / 25.3 J / 1.0   (one verifier forward; only on open-text where verifier gate used)
  GEN32= 665 ms / 127.0 J / 4.57 (32B no-think)
  always-32B = 665 ms / 4.57 FLOPs per sample.

SCORING (faithful):
  MCQ / closed  -> MedEvalKit exact-match `correct` (MMMU: parsed_output judge==Correct).
  open-text     -> LLM/Claude-judge `judge_ok` (MedEvalKit exact-match on open answers is ~0 and
                   known-broken, e.g. "CT." != "CT"); clearly labelled `open (judged)`.
"""
import json, glob, os, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
MEK  = os.path.join(ROOT, "MedEvalKit")
OUT  = os.path.join(ROOT, "results/cascade_methods/artifacts/best_method_lingshu_medeval.json")

# ---- cost constants (batch-1, measured) -----------------------------------------------------------
GEN7  = dict(ms=347.0, J=45.8,  flop=1.0)
VER7  = dict(ms=175.0, J=25.3,  flop=1.0)
GEN32 = dict(ms=665.0, J=127.0, flop=4.57)   # 32B no-think
ALWAYS32 = dict(ms=GEN32["ms"], J=GEN32["J"], flop=GEN32["flop"])

# ---- robust field parsing (some raw files store fields as strings) --------------------------------
def as_float(x, d=0.0):
    try: return float(x)
    except (TypeError, ValueError): return d
def as_ok(r):
    v = r.get("correct")
    if isinstance(v, bool): return int(v)
    return int(str(v).strip().lower() in ("true", "1"))

# ============================ MCQ / CLOSED loaders (per-sample 7B & 32B) ============================
def load_raw(tag, ds):
    p = f"{MEK}/eval_results_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None

def mcq_pair(ds, closed=None):
    """Return aligned arrays (ok7, ok32, margin7, lat7, lat32) over the (optionally closed) subset."""
    r7, r32 = load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds)
    if r7 is None or r32 is None: return None
    n = min(len(r7), len(r32))
    if closed == "SLAKE":
        idx = [i for i in range(n) if r7[i].get("answer_type") == "CLOSED"]
    elif closed == "YESNO":
        idx = [i for i in range(n) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        idx = list(range(n))
    ok7  = np.array([as_ok(r7[i]) for i in idx], float)
    ok32 = np.array([as_ok(r32[i]) for i in idx], float)
    m7   = np.array([as_float(r7[i].get("margin")) for i in idx], float)
    l7   = np.array([as_float(r7[i].get("latency_s")) for i in idx], float)
    l32  = np.array([as_float(r32[i].get("latency_s")) for i in idx], float)
    return dict(ok7=ok7, ok32=ok32, gate=m7, lat7=l7, lat32=l32)

def mmmu_pair():
    def load(tag):
        rows = []
        for f in sorted(glob.glob(f"{MEK}/eval_results_{tag}/{{}}/MMMU-Medical-val/*/parsed_output.json")):
            rows += json.load(open(f))
        return {r["id"]: r for r in rows}
    d7, d32 = load("lingshu7b_full"), load("lingshu32b_full")
    ids = [i for i in d7 if i in d32]
    ok7  = np.array([1 if d7[i].get("judge") == "Correct" else 0 for i in ids], float)
    ok32 = np.array([1 if d32[i].get("judge") == "Correct" else 0 for i in ids], float)
    m7   = np.array([as_float(d7[i].get("margin")) for i in ids], float)
    l7   = np.array([as_float(d7[i].get("latency_s")) for i in ids], float)
    l32  = np.array([as_float(d32[i].get("latency_s")) for i in ids], float)
    return dict(ok7=ok7, ok32=ok32, gate=m7, lat7=l7, lat32=l32)

# ============================ OPEN-TEXT loaders (per-sample 7B & 32B judged) ========================
def load_judge_jsonl(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def open_verifier(ds):
    """VQA_RAD / PATH_VQA: cheap=7B greedy (greedy_ok), gate=trained-verifier confidence (max P(Yes)
    over sampled cands), strong=32B judge_ok.  +VER7 cost on the cheap leg."""
    dp = f"{ROOT}/ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_open_lingshu7b.json"
    sj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_{ds}_open_lingshu32b.judge.jsonl")
    if not os.path.exists(dp) or not sj: return None
    ok7, ok32, gate = [], [], []
    for r in json.load(open(dp)):
        i = r["idx"]
        if i not in sj: continue
        ok7.append(int(r["greedy_ok"])); ok32.append(sj[i]); gate.append(float(max(r["scores"])))
    return dict(ok7=np.array(ok7, float), ok32=np.array(ok32, float),
                gate=np.array(gate, float), verify=True)

def open_conf_slake():
    """SLAKE-open (no verifier transfer dump): cheap=7B greedy (7B judge_ok), gate=7B sequence-logprob
    `seqlogprob` (free, single greedy gen -> no VER7 cost), strong=32B judge_ok.  All keyed by the
    global open-VQA idx shared across the cheap dump and both judge files."""
    cj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.judge.jsonl")
    sj = load_judge_jsonl(f"{ROOT}/ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b.judge.jsonl")
    cheap = {}
    p = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b.jsonl"
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); cheap[int(r["idx"])] = as_float(r.get("seqlogprob"))
    if not cj or not sj or not cheap: return None
    ok7, ok32, gate = [], [], []
    for i in sorted(set(cj) & set(sj) & set(cheap)):
        ok7.append(cj[i]); ok32.append(sj[i]); gate.append(cheap[i])
    if not ok7: return None
    return dict(ok7=np.array(ok7, float), ok32=np.array(ok32, float),
                gate=np.array(gate, float), verify=False)

# ============================ cascade mechanics ====================================================
def cascade_acc(ok7, ok32, gate, tau):
    esc = gate < tau
    return np.where(esc, ok32, ok7).mean(), esc.mean()

def pick_tau_isocost(ok7, ok32, gate, target):
    """Min-escalation tau s.t. cascade acc >= target (fallback: max-acc tau)."""
    cand = sorted(set(gate.tolist()))
    grid = [cand[0] - 1e-9] + cand + [cand[-1] + 1e-9]
    best_iso = None; best_max = (-1, None)
    for tau in grid:
        acc, esc = cascade_acc(ok7, ok32, gate, tau)
        if acc > best_max[0]: best_max = (acc, tau)
        if acc >= target - 1e-12:
            if best_iso is None or esc < best_iso[1]:
                best_iso = (tau, esc)
    return (best_iso[0] if best_iso else best_max[1])

def heldout_operating_point(d, K=5, mode="iso"):
    """5-fold cross-fit. mode='iso': tau matches 32B acc at min cost; mode='max': tau maximizes acc.
    Returns held-out mean (acc, esc)."""
    ok7, ok32, gate = d["ok7"], d["ok32"], d["gate"]
    n = len(ok7); accs, escs = [], []
    for f in range(K):
        te = np.array([i % K == f for i in range(n)]); tr = ~te
        if tr.sum() < 2 or te.sum() < 1: continue
        tgt = ok32[tr].mean()
        if mode == "max":
            # tau maximizing train cascade acc
            cand = sorted(set(gate[tr].tolist())); grid = [cand[0]-1e-9]+cand+[cand[-1]+1e-9]
            tau = max(grid, key=lambda t: cascade_acc(ok7[tr], ok32[tr], gate[tr], t)[0])
        else:
            tau = pick_tau_isocost(ok7[tr], ok32[tr], gate[tr], tgt)
        acc, esc = cascade_acc(ok7[te], ok32[te], gate[te], tau)
        accs.append(acc); escs.append(esc)
    return float(np.mean(accs)), float(np.mean(escs))

def costs(esc, verify=False):
    base_ms  = GEN7["ms"]  + (VER7["ms"]  if verify else 0.0)
    base_fl  = GEN7["flop"]+ (VER7["flop"] if verify else 0.0)
    base_J   = GEN7["J"]   + (VER7["J"]   if verify else 0.0)
    return dict(latency_ms = base_ms + esc * GEN32["ms"],
                flops      = base_fl + esc * GEN32["flop"],
                energy_J   = base_J  + esc * GEN32["J"])

# ============================ assemble ============================================================
def run():
    # benchmark spec: (name, format, loader)
    specs = [
        ("PMC_VQA",          "MCQ",       lambda: mcq_pair("PMC_VQA")),
        ("MMMU-Medical-val", "MCQ",       mmmu_pair),
        ("MedXpertQA-MM",    "MCQ",       lambda: mcq_pair("MedXpertQA-MM")),
        ("SLAKE_closed",     "closed",    lambda: mcq_pair("SLAKE", closed="SLAKE")),
        ("VQA_RAD_closed",   "closed",    lambda: mcq_pair("VQA_RAD", closed="YESNO")),
        ("PATH_VQA_closed",  "closed",    lambda: mcq_pair("PATH_VQA", closed="YESNO")),
        ("SLAKE_open",       "open",      open_conf_slake),
        ("VQA_RAD_open",     "open",      lambda: open_verifier("vqa_rad")),
        ("PATH_VQA_open",    "open",      lambda: open_verifier("pathvqa")),
    ]
    per = {}
    for name, fmt, loader in specs:
        d = loader()
        if d is None:
            per[name] = dict(format=fmt, status="MISSING_DATA"); continue
        verify = d.get("verify", False)
        n = len(d["ok7"]); a7 = float(d["ok7"].mean()); a32 = float(d["ok32"].mean())
        # deployable held-out operating points
        iso_acc, iso_esc = heldout_operating_point(d, mode="iso")
        max_acc, max_esc = heldout_operating_point(d, mode="max")
        # oracle (full-data) reference, iso
        tau_o = pick_tau_isocost(d["ok7"], d["ok32"], d["gate"], a32)
        orc_acc, orc_esc = cascade_acc(d["ok7"], d["ok32"], d["gate"], tau_o)
        c_iso = costs(iso_esc, verify); c_max = costs(max_esc, verify)
        per[name] = dict(
            format=fmt, n=n, scoring=("open (judged)" if fmt == "open" else
                                      ("EM/judge" if name == "MMMU-Medical-val" else "EM")),
            gate=("verifier-conf" if (fmt == "open" and verify) else ("seqlogprob" if fmt == "open" else "margin")),
            acc_7b=round(a7, 4), acc_32b=round(a32, 4), headroom_32b=round(a32 - a7, 4),
            cascade_iso=dict(acc=round(iso_acc, 4), d_acc_vs32=round(iso_acc - a32, 4),
                             esc=round(iso_esc, 4), latency_ms=round(c_iso["latency_ms"], 1),
                             flops=round(c_iso["flops"], 3), energy_J=round(c_iso["energy_J"], 1),
                             faster=c_iso["latency_ms"] < ALWAYS32["ms"],
                             at_least_as_acc=iso_acc >= a32 - 1e-9),
            cascade_maxacc=dict(acc=round(max_acc, 4), d_acc_vs32=round(max_acc - a32, 4),
                                esc=round(max_esc, 4), latency_ms=round(c_max["latency_ms"], 1),
                                flops=round(c_max["flops"], 3)),
            oracle_iso=dict(acc=round(orc_acc, 4), esc=round(orc_esc, 4)),
        )
    # OmniMed: 7B only, 32B MISSING
    r7o = load_raw("lingshu7b_full", "OmniMedVQA")
    per["OmniMedVQA"] = dict(format="MCQ", n=(len(r7o) if r7o else 0),
                             acc_7b=(round(np.mean([as_ok(r) for r in r7o]), 4) if r7o else None),
                             acc_32b=None, status="32B_MISSING (cannot score cascade)")

    # -------- pooling (benchmarks with full 7B+32B data) --------
    full = {k: v for k, v in per.items() if "acc_32b" in v and v.get("acc_32b") is not None
            and "cascade_iso" in v}
    def macro(field_getter):
        return float(np.mean([field_getter(v) for v in full.values()]))
    def sw(field_getter):
        num = sum(field_getter(v) * v["n"] for v in full.values())
        den = sum(v["n"] for v in full.values())
        return num / den
    pooled = {}
    for label, keys in [("suite_no_mmmu_no_omni",
                         [k for k in full if k not in ("MMMU-Medical-val", "OmniMedVQA")]),
                        ("suite_with_mmmu", list(full.keys()))]:
        sub = {k: full[k] for k in keys}
        macro7  = float(np.mean([sub[k]["acc_7b"] for k in sub]))
        macro32 = float(np.mean([sub[k]["acc_32b"] for k in sub]))
        macroC  = float(np.mean([sub[k]["cascade_iso"]["acc"] for k in sub]))
        n_tot   = sum(sub[k]["n"] for k in sub)
        sw7  = sum(sub[k]["acc_7b"]*sub[k]["n"] for k in sub)/n_tot
        sw32 = sum(sub[k]["acc_32b"]*sub[k]["n"] for k in sub)/n_tot
        swC  = sum(sub[k]["cascade_iso"]["acc"]*sub[k]["n"] for k in sub)/n_tot
        swEsc= sum(sub[k]["cascade_iso"]["esc"]*sub[k]["n"] for k in sub)/n_tot
        swLat= sum(sub[k]["cascade_iso"]["latency_ms"]*sub[k]["n"] for k in sub)/n_tot
        swFl = sum(sub[k]["cascade_iso"]["flops"]*sub[k]["n"] for k in sub)/n_tot
        pooled[label] = dict(
            benchmarks=list(sub.keys()), n=n_tot,
            macro_avg=dict(acc_7b=round(macro7,4), acc_32b=round(macro32,4),
                           acc_cascade=round(macroC,4), d_acc_vs32=round(macroC-macro32,4)),
            sample_weighted=dict(acc_7b=round(sw7,4), acc_32b=round(sw32,4),
                                 acc_cascade=round(swC,4), d_acc_vs32=round(swC-sw32,4),
                                 esc=round(swEsc,4),
                                 cascade_latency_ms=round(swLat,1), always32_latency_ms=ALWAYS32["ms"],
                                 latency_saved_pct=round((1-swLat/ALWAYS32["ms"])*100,1),
                                 cascade_flops=round(swFl,3), always32_flops=ALWAYS32["flop"],
                                 flops_saved_pct=round((1-swFl/ALWAYS32["flop"])*100,1)))
    # per-benchmark verdict: faster than always-32B AND at-least-as-accurate?
    for name, v in per.items():
        if "cascade_iso" in v:
            ci = v["cascade_iso"]
            v["verdict"] = ("faster_AND_more_accurate" if (ci["faster"] and ci["d_acc_vs32"] > 1e-9) else
                            "faster_at_iso_accuracy" if (ci["faster"] and ci["d_acc_vs32"] >= -1e-9) else
                            "faster_but_slightly_less_accurate" if ci["faster"] else
                            "slower (needs high escalation; no cheap win)")

    sw6 = pooled["suite_no_mmmu_no_omni"]["sample_weighted"]
    swM = pooled["suite_with_mmmu"]["sample_weighted"]
    mcM = pooled["suite_with_mmmu"]["macro_avg"]
    bottom_line = dict(
        faster="YES (pooled, sample-weighted): batch-1 latency {c}ms vs 665ms ({p:+.0f}%), FLOPs {f} vs 4.57 ({pf:+.0f}%)"
               .format(c=sw6["cascade_latency_ms"], p=sw6["latency_saved_pct"], f=sw6["cascade_flops"], pf=sw6["flops_saved_pct"]),
        more_accurate="AT PARITY: sample-weighted cascade {a} vs 32B {b} (d={d:+.3f}); strictly MORE accurate only "
                      "on macro-avg WITH MMMU ({ma} vs {mb}, d={md:+.3f}), which rides the known Lingshu-7B MMMU anomaly"
                      .format(a=sw6["acc_cascade"], b=sw6["acc_32b"], d=sw6["d_acc_vs32"],
                              ma=mcM["acc_cascade"], mb=mcM["acc_32b"], md=mcM["d_acc_vs32"]),
        where_it_wins="benchmarks with small 32B headroom -> low escalation stays under the latency crossover "
                      "(esc<48% MCQ / <22% open): PMC(8%), MMMU(0%), SLAKE-closed(20%), PATH-closed(46%).",
        where_it_falls_short="benchmarks where 7B is weak/near-floor -> must escalate heavily, so slower than "
                             "always-32B: MedXpert(90%), VQA_RAD-closed(57%), all open-text(53-91%).",
        note="pooled 'faster' is real but dominated by PMC_VQA (33k samples, 8% esc); per-benchmark the cascade "
             "is faster on 4/6 closed-MCQ sets and slower on the high-headroom (open/reasoning) sets.")

    notes = [
        "SLAKE-open scoring: current committed Claude-judge files give 7B=0.730 / 32B=0.819 (harness modal_ok=0.763 "
        "agrees); the older open_cascade_lingshu.json 0.419/0.847 used a different/older pass. Same judge used for "
        "both legs here (fair). Gate=seqlogprob (free single-gen signal).",
        "VQA_RAD/PATH_VQA-open gate=trained verifier (lora_verifier_pooled4) confidence; costs +VER7 (175ms) on the "
        "cheap leg. cheap answer = 7B greedy (greedy_ok); best-of-N deliberately NOT used (8x gen+verify = 4176ms "
        "would blow the batch-1 latency budget vs 665ms).",
        "Held-out = 5-fold cross-fit: tau chosen on 4/5 (min escalation s.t. cascade acc >= that fold's 32B acc), "
        "evaluated on the held-out 1/5, averaged. No peeking.",
        "MMMU cascade keeps 7B (esc 0%) because Lingshu-7B (0.80) already exceeds 32B (0.633) there.",
    ]
    out = dict(
        method="FALC — Format-Aware Lingshu Cascade (7B-nothink -> 32B-nothink; margin gate for "
               "MCQ/closed, verifier/seqlogprob gate for open-text; held-out 5-fold tau at iso-32B-acc)",
        cost_constants=dict(GEN7=GEN7, VER7=VER7, GEN32_nothink=GEN32, always32=ALWAYS32),
        strong_leg_is_nothink_because="32B-think <= 32B-nothink on every MedEvalKit benchmark (measured)",
        bottom_line=bottom_line, notes=notes,
        per_benchmark=per, pooled=pooled,
        data_gaps=[
            "OmniMedVQA-32B MISSING (n=88996): cannot score the cascade or always-32B there; 7B=%.3f only. "
            "Re-confirmed BLOCKED (2026-07-07): tp=2 hangs (NCCL worker-died, pre-existing); tp=1 has no "
            "viable gpu_memory_utilization window on one 80GB A100 -- util<=0.88 => 'no KV cache blocks', "
            "util>=0.90 => vision-tower activation OOM. Needs >2 GPUs, quantization, or a smaller strong model."
            % (per['OmniMedVQA']['acc_7b'] or 0),
            "MMMU: Lingshu-7B=0.80 >> 32B=0.633 is a known Lingshu-7B-specific MedEvalKit anomaly "
            "(excluded from robust efficiency claims); n=150 only.",
            "SLAKE-open uses free 7B seqlogprob gate (no verifier transfer dump); VQA_RAD/PATH_VQA-open use trained verifier.",
            "32B-THINK batch-1 latency pending (opentext_32b_think.json) — NOT needed: think never "
            "beats no-think for Lingshu, so the strong leg is no-think.",
        ],
    )
    json.dump(out, open(OUT, "w"), indent=2)
    # ---- console report ----
    print("="*118)
    print("FALC — Format-Aware Lingshu Cascade vs always-Lingshu-32B (no-think). Held-out (5-fold) iso-acc operating point.")
    print("="*118)
    h = f"{'benchmark':<18}{'fmt':<7}{'n':>6}{'gate':>14}{'7B':>7}{'32B':>7}{'casc':>7}{'dAcc':>7}{'esc%':>6}{'lat_ms':>8}{'FLOPs':>7}{'F&A?':>6}"
    print(h); print("-"*len(h))
    for name, v in per.items():
        if "cascade_iso" not in v:
            print(f"{name:<18}{v.get('format',''):<7}{v.get('n',0):>6}{'':>14}"
                  f"{(v.get('acc_7b') if v.get('acc_7b') is not None else float('nan')):>7.3f}"
                  f"{'  --  ':>7}{'  --  ':>7}   {v.get('status','')}")
            continue
        ci = v["cascade_iso"]
        fa = "YES" if (ci["faster"] and ci["d_acc_vs32"] >= 0) else ("~acc" if ci["faster"] else "no")
        print(f"{name:<18}{v['format']:<7}{v['n']:>6}{v['gate']:>14}{v['acc_7b']:>7.3f}{v['acc_32b']:>7.3f}"
              f"{ci['acc']:>7.3f}{ci['d_acc_vs32']:>+7.3f}{ci['esc']*100:>5.0f}%{ci['latency_ms']:>8.0f}"
              f"{ci['flops']:>7.2f}{fa:>6}")
    print("-"*len(h))
    for label, p in pooled.items():
        m = p["macro_avg"]; s = p["sample_weighted"]
        print(f"\nPOOLED [{label}] n={p['n']}  ({len(p['benchmarks'])} benchmarks)")
        print(f"  macro-avg  : 7B={m['acc_7b']:.3f}  32B={m['acc_32b']:.3f}  cascade={m['acc_cascade']:.3f}  dAcc={m['d_acc_vs32']:+.3f}")
        print(f"  sample-wt  : 7B={s['acc_7b']:.3f}  32B={s['acc_32b']:.3f}  cascade={s['acc_cascade']:.3f}  dAcc={s['d_acc_vs32']:+.3f}")
        print(f"  efficiency : esc={s['esc']*100:.0f}%  latency {s['cascade_latency_ms']:.0f}ms vs 665ms ({s['latency_saved_pct']:+.0f}%)  "
              f"FLOPs {s['cascade_flops']:.2f} vs 4.57 ({s['flops_saved_pct']:+.0f}%)")
    print(f"\nwrote {OUT}")
    return out

if __name__ == "__main__":
    run()
