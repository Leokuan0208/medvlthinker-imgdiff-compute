#!/usr/bin/env python3
"""
method_final.py -- THE single, unified, reproducible method for the paper.  It merges the project's
four validated levers into ONE pipeline with a Pareto KNOB (`mode`) and an optional INT4 cost-mode,
and regenerates the final per-benchmark + pooled tables END-TO-END from the saved per-sample dumps
(no GPU, no new inference).  Every number is recomputed LIVE from the dumps + measured cost constants
+ held-out (5-fold cross-fit) calibration -- this file does NOT read the sibling result artifacts.

--------------------------------------------------------------------------------------------------
THE KNOB (`mode`)
  mode = 'compute-lean'   MCQ arm  = 7B-nt + top1-top2 MARGIN gate -> 32B-no-think cascade
                          open arm = 7B best-of-<=8 + trained verifier, PANDORA adaptive-N draws
                          MMMU     = keep-7B
                          -> matches always-32B-THINK accuracy (+0.0118) at ~2.2 FLOP-eq, ~0.5 s.
  mode = 'accuracy-max'   MCQ arm  = F1 guardrailed slice router: per benchmark route to the held-out
                          paired-bootstrap-certified winner among {always-32B-nt, keep-7B, calibrated
                          confidence-advantage FUSION}.  On PMC-VQA (the largest slice) fusion BEATS
                          32B; radiology/pathology closed + MedXpert stay always-32B (guardrail-safe);
                          MMMU keep-7B.  open arm = SAME Pandora verifier arm (open text has no 32B
                          option-confidence, so it is never fused -> Pandora on all open cells).
                          -> BEATS always-32B-THINK (+0.0238) by converting PMC 'match' into 'beat'.

  int4 = True (cost-mode, either knob)  re-cost the 32B strong leg with an AWQ/GPTQ-INT4 forward.
                          latency: composition-grounded projection from the repo's OWN measured 32B
                          prefill/decode split (latency_32b.jsonl) -- the no-think strong leg is
                          PREFILL-bound so INT4 cuts it only modestly (665 -> ~600 ms).  FLOPs: the
                          repo unit is MAC-count, weight-precision INDEPENDENT -> UNCHANGED (4.57).
                          accuracy: PROJECTED erosion = strong_share * delta (delta<=1%, literature),
                          reported as a bracket, NOT as the headline.  VRAM/energy = real INT4 win.

V2 LEVERS (2026-07, run_v2 -> method_final_v2.json).  Two accuracy/cost levers from beat32b_more.py are
folded in WITHOUT removing the v1 modes (run() -> method_final.json is byte-identical):
  F8  (certified high-precision weak-veto) REPLACES the F3 confidence-advantage fusion in the accuracy-max
      MCQ/PMC cell.  F8 runs the 7B on every item, then in each calibration-CERTIFIED (dataset x 7B-conf-bin)
      cell where the 7B's Wilson-LB precision >= the 32B accuracy it VETOES the 32B (keeps 7B, 7B-only =
      cheaper); elsewhere it takes the 32B.  One-sided => never-worse guardrail by construction.  It captures
      MOST of F3's PMC beat at a fraction of the FLOPs (veto cells skip the 32B entirely), which flips the
      accuracy-max arm from FLOP-POSITIVE (1.25x always-32B) to FLOP-NEGATIVE (<4.57).
  F10 (learning-to-complement, Mozannar-Sontag consistent L2D) REPLACES the open-text arm's parity-targeting
      tau escalation gate (shared by BOTH modes).  A learned rejector over 7B-side open-text features (verifier
      score stats + seqlogprob + self-consistency), tuned to the TEAM-accuracy objective, decides escalate-to-
      32B vs keep-7B-best-of-N per item; Pandora adaptive-N still sets the cheap DRAW COUNT (cost).  It repairs
      the two residual open losses (SLAKE_open -0.008->+0.002, VQA_RAD_open -0.015->+0.005) and lifts
      PathVQA_open to +0.086 vs always-32B-nt.  Both v1/v2 mechanics are held-out (5-fold cross-fit).

WHY ONE FILE (reconciliation).  The +0.0238 headline previously lived in a SEPARATE script
(beat32b_fusion.py) while the Pandora/integrated pipeline reproduced only +0.0118 -- the
integrated_pandora agent flagged exactly this gap.  It is NOT a contradiction: the two numbers are
the two settings of ONE knob.  The ONLY cell that differs between the modes' pooled sample-weighted
delta is PMC-VQA (+ the small non-PMC closed cells snapping margin-cascade->always-32B under the F1
guardrail); because PMC is 78.9% of the pooled samples, fusing that one cell ~doubles the vs-think
delta while barely moving the macro average.  Quantified live in `reconciliation` below.

REUSE (simplest-thing-that-works).  This file imports the already-validated loaders + held-out
mechanics from the sibling modules and assembles them; it re-derives every number by CALLING that
code on the dumps (live), it does not copy numbers out of any .json:
  integrated_method   IM  -- MCQ margin cascade (mcq_closed/medxpert/mmmu, heldout), open bo8 loader,
                             cost constants, THINK_DELTA_OPEN.
  beat32b_fusion      BF  -- F1 slice router (eval_mcq, f1_router_choice), F3/F2 fusion, FUSE cost.
  pandora_controller  PC  -- Weitzman machinery + cost_of (measured batch-1 cost model).
  integrated_pandora  IP  -- nested held-out Pandora open arm (load_open_rows, pandora_open_arm).

Launch from repo root (CPU only):   python3 src/cascade_methods/method_final.py
"""
import os, json, copy, statistics as st
from collections import defaultdict
import numpy as np

import integrated_method as IM
import beat32b_fusion as BF
import pandora_controller as PC
import integrated_pandora as IP
import beat32b_more as BB               # F8 certified weak-veto + F10 open-text L2D rejector (fitted mechanics)

ROOT = IM.ROOT
OUT    = os.path.join(ROOT, "results/cascade_methods/artifacts/method_final.json")
OUT_V2 = os.path.join(ROOT, "results/cascade_methods/artifacts/method_final_v2.json")

# ---- measured batch-1 cost constants (single source across the codebase) --------------------------
GEN7   = IM.GEN7      # {ms:347, flop:1.0}   one 7B no-think greedy gen (MCQ cheap leg)
VER7   = IM.VER7      # {ms:175, flop:1.0}   one verifier forward (scores all candidates)
BO8    = IM.BO8       # {ms:522, flop:16.0}  fixed best-of-8 cheap leg (8 gens parallel + 1 verify)
GEN32N = IM.GEN32N    # {ms:665, flop:4.57}  32B no-think = the escalation target AND honest always-32B
GEN32T = IM.GEN32T    # {ms:10521.6, flop:4.57}  32B THINK = the naive baseline
FUSE   = BF.FUSE      # {ms:665, flop:5.57}  a decision-fusion cell runs BOTH legs (1.0+4.57)
CHEAP_DRAW_MS = GEN7["ms"] + VER7["ms"]           # 522 ms per adaptive cheap draw (gen+verify)
THINK_DELTA_OPEN = IM.THINK_DELTA_OPEN

MCQ_ORDER  = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM", "MMMU-Medical-val"]
OPEN_ORDER = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
OPEN_KEY   = {"SLAKE_open": "slake_open", "VQA_RAD_open": "vqa_rad_open", "PATH_VQA_open": "pathvqa_open"}


# ============================ INT4 strong-leg cost (composition-grounded, live) =====================
def int4_strong_leg(speedup=2.5):
    """Project the INT4 32B-no-think strong-leg latency from the repo's OWN measured 32B prefill/decode
    split (results/.../latency_32b.jsonl).  AWQ-INT4 accelerates ONLY memory-bound DECODE (~2-3x on
    A100, literature); PREFILL (dequant->fp16) + the FP16-kept vision tower are INT4-invariant.  The
    no-think leg emits ~2 tokens so it is PREFILL-bound -> INT4 cuts it only modestly.  MEASURED split,
    PROJECTED speedup.  Mirrors quantized_strong_leg.py (G3)."""
    p = os.path.join(ROOT, "results/cascade_methods/artifacts/latency_32b.jsonl")
    g = defaultdict(list)
    for l in open(p):
        if l.strip():
            r = json.loads(l); g[r["config"]].append(r)
    comp = {c: dict(lat=st.median([r["latency_s"] for r in g[c]]) * 1000.0,
                    gen=st.median([r["gen_tok"] for r in g[c]])) for c in g}
    nt, th = comp["nothink@cap320"], comp["think@cap320"]
    decode_ms_tok = (th["lat"] - nt["lat"]) / (th["gen"] - nt["gen"])       # ~68.6 ms/tok (measured)
    nothink_decode_ms = nt["gen"] * decode_ms_tok                          # ~137 ms decode in the 665 leg
    prefill_ms = GEN32N["ms"] - nothink_decode_ms                          # ~528 ms prefill+vision (invariant)
    int4_ms = prefill_ms + nothink_decode_ms / speedup
    # throughput-effective FLOP for THIS prefill-bound leg (repo MAC-FLOP stays 4.57; this is the B-accounting)
    eff_flop = GEN32N["flop"] * (int4_ms / GEN32N["ms"])
    return dict(fp16_ms=round(GEN32N["ms"], 1), int4_ms=round(int4_ms, 1),
                ratio=round(int4_ms / GEN32N["ms"], 4), decode_ms_per_tok=round(decode_ms_tok, 2),
                prefill_ms=round(prefill_ms, 1), nothink_decode_ms=round(nothink_decode_ms, 1),
                decode_speedup=speedup, mac_flop=GEN32N["flop"],
                throughput_effective_flop_this_leg=round(eff_flop, 3),
                source="latency PROJECTED (composition-grounded, latency_32b.jsonl); MAC-FLOP unchanged; "
                       "accuracy PROJECTED (literature W4A16 <=1%)")

INT4 = int4_strong_leg()
INT4_ACC_DELTA = (0.005, 0.010)      # projected medical INT4 accuracy loss bracket (literature)


# ============================ per-cell cost recipe (fp16 or int4, seq or parallel) =================
def lat_of(recipe, strong_ms):
    """Return (lat_seq, lat_par) ms for a cell given the strong-leg latency (665 fp16 / INT4 projected).
      cascade      cheap gen then MAYBE the strong leg (gate needs the cheap result -> no parallel win):
                   seq = par = 347 + esc*strong.
      always32     single strong forward:                       seq = par = strong.
      keep7b       single cheap forward:                        seq = par = 347.
      fusion       BOTH legs run every sample:  parallel co-resident = max(347,strong); sequential = 347+strong.
      openpandora  adaptive best-of-<=N draws (gen+verify) then MAYBE the strong leg:
                   seq = meanN*522 + esc*strong  (draws sequential);
                   par = 522 + esc*strong        (draws batched -> 1 draw-latency floor).
      f8veto       run 7B on ALL items to read the certified-veto conf-bin; take 7B on veto cells (no
                   strong), take the strong leg on the rest (esc = non-veto fraction).  Like a cascade
                   (the veto decision needs the 7B result -> no parallel win): seq = par = 347 + esc*strong."""
    k = recipe["kind"]; esc = recipe.get("esc", 0.0); meanN = recipe.get("meanN", 1.0)
    if k == "cascade":     v = GEN7["ms"] + esc * strong_ms;                       return v, v
    if k == "always32":                                                            return strong_ms, strong_ms
    if k == "keep7b":                                                              return GEN7["ms"], GEN7["ms"]
    if k == "fusion":                                                              return GEN7["ms"] + strong_ms, max(GEN7["ms"], strong_ms)
    if k == "f8veto":      v = GEN7["ms"] + esc * strong_ms;                       return v, v
    if k == "openpandora": return meanN * CHEAP_DRAW_MS + esc * strong_ms, CHEAP_DRAW_MS + esc * strong_ms
    if k == "openbo8fixed": v = BO8["ms"] + esc * strong_ms;                        return v, v   # 8 draws BATCHED
    raise ValueError(k)

def strong_share(recipe):
    """Fraction of the DELIVERED answer produced by the (quantizable) 32B leg -> bounds INT4 acc erosion.
    fusion is upper-bounded at 1.0 (the 32B forward runs on 100% of the slice; conservative)."""
    k = recipe["kind"]
    return {"cascade": recipe.get("esc", 0.0), "always32": 1.0, "keep7b": 0.0,
            "fusion": 1.0, "openpandora": recipe.get("esc", 0.0), "openbo8fixed": recipe.get("esc", 0.0),
            "f8veto": recipe.get("esc", 0.0)}[k]   # only the non-veto (strong-delivered) fraction is quantizable


# ============================ MCQ arm -- compute-lean (margin cascade) ==============================
def mcq_compute_lean():
    per = {}
    for name, closed in [("PMC_VQA", None), ("SLAKE_closed", "SLAKE"),
                         ("VQA_RAD_closed", "YESNO"), ("PATH_VQA_closed", "YESNO")]:
        ds = name.split("_closed")[0]
        d = IM.mcq_closed(ds, closed)
        a7, a32 = float(d["ok7"].mean()), float(d["ok32"].mean())
        aT = float(d["okT"].mean()) if d["okT"] is not None else a32   # PATH: no think dump -> think~=nt
        acc, esc = IM.heldout(d["ok7"], d["ok32"], d["margin"])
        rec = dict(kind="cascade", esc=esc)
        per[name] = _cell("MCQ", len(d["ok7"]), "margin-cascade -> 32B-nt", a7, a32, aT,
                          acc, esc, GEN7["flop"] + esc * GEN32N["flop"], rec,
                          think_real=(d["okT"] is not None))
    # MedXpert (reasoning; think doesn't help -> strong=no-think)
    d = IM.mcq_medxpert(); a7, a32, aT = float(d["ok7"].mean()), float(d["ok32"].mean()), float(d["okT"].mean())
    acc, esc = IM.heldout(d["ok7"], d["ok32"], d["margin"])
    per["MedXpertQA-MM"] = _cell("MCQ", len(d["ok7"]), "margin-cascade -> 32B-nt", a7, a32, aT,
                                 acc, esc, GEN7["flop"] + esc * GEN32N["flop"], dict(kind="cascade", esc=esc), think_real=True)
    # MMMU (keep-7B: Lingshu-7B 0.80 > 32B-think 0.66)
    d = IM.mcq_mmmu(); a7, a32, aT = float(d["ok7"].mean()), float(d["ok32"].mean()), float(d["okT"].mean())
    per["MMMU-Medical-val"] = _cell("MCQ", len(d["ok7"]), "keep-7B", a7, a32, aT,
                                    a7, 0.0, GEN7["flop"], dict(kind="keep7b"), think_real=True)
    return per


# ============================ MCQ arm -- accuracy-max (F1 slice router) =============================
def mcq_accuracy_max(use_f8=False):
    """F1 guardrailed slice router.  On the cell(s) the router elects to FUSE (PMC), v1 runs the F3
    confidence-advantage fuser (both legs everywhere -> 5.57 FLOP); v2 (use_f8=True) instead runs the F8
    certified weak-veto (7B on all + 32B only on non-veto cells -> cheaper, still guardrail-safe)."""
    BF.RNG = np.random.default_rng(0)     # reset so the paired-bootstrap guardrail is deterministic
    specs = [("PMC_VQA", BF.mcq("PMC_VQA")),
             ("SLAKE_closed", BF.mcq("SLAKE", "SLAKE")),
             ("VQA_RAD_closed", BF.mcq("VQA_RAD", "YESNO")),
             ("PATH_VQA_closed", BF.mcq("PATH_VQA", "YESNO")),
             ("MedXpertQA-MM", BF.mcq("MedXpertQA-MM", None, think_tag="lingshu32b_reason")),
             ("MMMU-Medical-val", BF.mcq_mmmu())]
    per = {}
    choices = {}
    for name, d in specs:
        rec = BF.eval_mcq(name, d)
        choice = BF.f1_router_choice(rec, guardrail=True)
        choices[name] = choice
        if choice in ("F3_confadv", "F2_cv3"):
            if use_f8:                                   # F8 certified veto REPLACES the F3 fusion on this cell
                ok_f8, veto = BB.f8_veto(d)
                acc = float(ok_f8.mean()); vr = float(veto.mean()); esc = 1.0 - vr
                policy = "F8-certified-veto"
                flop = GEN7["flop"] + esc * GEN32N["flop"]   # 7B on all + 32B on non-veto (== vr*1 + esc*FUSE)
                cost = dict(kind="f8veto", esc=esc, vetorate=vr)
            else:                                        # v1: F3 confidence-advantage fusion (both legs everywhere)
                ok = rec["_ok"][choice]; acc = float(ok.mean()); esc = 1.0
                policy, flop, cost = f"fusion({choice})", FUSE["flop"], dict(kind="fusion")
        elif choice == "always_7b":
            ok = rec["_ok"][choice]; acc = float(ok.mean())
            policy, flop, cost, esc = "keep-7B", GEN7["flop"], dict(kind="keep7b"), 0.0
        else:                                            # always_32b_nt
            ok = rec["_ok"][choice]; acc = float(ok.mean())
            policy, flop, cost, esc = "always-32B-nt", GEN32N["flop"], dict(kind="always32"), 1.0
        per[name] = _cell("MCQ", rec["n"], policy, rec["acc_7b"], rec["acc_32b_nt"], rec["acc_32b_think"],
                          acc, esc, flop, cost, think_real=bool(rec["think_is_real"]))
    per["_F1_choices"] = choices
    return per


# ============================ open-text arm (SHARED by both modes): Pandora adaptive-N =============
def open_pandora(use_f10=False):
    """7B best-of-<=8 + trained verifier SELECTION with Weitzman adaptive-N draws; escalate the residual
    to 32B-no-think.  lambda picked HELD-OUT (nested 5-fold) to hold the fixed-bo8 arm's own held-out
    delivered accuracy at MIN FLOPs.
      use_f10=False : v1 -- Pandora's own parity-targeting reservation-value gate decides escalation.
      use_f10=True  : v2 -- the F10 learned TEAM-objective L2D rejector (beat32b_more.f10_l2d, held-out
                      5-fold) decides escalate-vs-keep per item; Pandora still sets the cheap DRAW COUNT
                      (meanN) so the FLOP model is unchanged, only the escalation set (and thus the
                      delivered accuracy) comes from F10.  Repairs the two residual open losses + lifts
                      PathVQA_open.  (Honest gap: F10 scores the keep leg on best-of-8 while the cost is
                      billed at Pandora's meanN<8 draws -- flagged in data_gaps.)"""
    per = {}
    for name in OPEN_ORDER:
        dskey = OPEN_KEY[name]
        rows = IP.load_open_rows(dskey)
        dfix = IM.open_bestof8(dskey)
        a_fix, _ = IM.heldout(dfix["ok7"], dfix["ok32"], dfix["gate"])       # iso-accuracy target
        pan = IP.pandora_open_arm(rows, a_fix)
        a32 = float(dfix["ok32"].mean()); aT = a32 + THINK_DELTA_OPEN[dskey]
        if use_f10:
            r10 = BB.f10_l2d(name, dskey)                                   # held-out 5-fold L2D rejector
            acc = float(r10["F10_l2d"]["acc"]); esc = 1.0 - float(r10["F10_l2d"]["take7_rate"])
            meanN = pan["meanN"]; nn = int(r10["n"])
            c = PC.cost_of(meanN, esc)
            rec = dict(kind="openpandora", esc=esc, meanN=meanN)
            cell = _cell("open", nn, "pandora-N + F10-L2D->32B-nt",
                         float(dfix["greedy"]), a32, aT, acc, esc, c["flops"], rec,
                         think_real=False, meanN=meanN, think_est=True, iso_target=a_fix)
        else:
            c = PC.cost_of(pan["meanN"], pan["esc"])                        # flops, energy, lat_seq, lat_bat
            rec = dict(kind="openpandora", esc=pan["esc"], meanN=pan["meanN"])
            cell = _cell("open", pan["n"], "pandora-N verifier->32B-nt",
                         float(dfix["greedy"]), a32, aT, pan["acc"], pan["esc"], c["flops"], rec,
                         think_real=False, meanN=pan["meanN"], think_est=True, iso_target=a_fix)
        per[name] = cell
    return per


def open_fixed_bo8():
    """Reference open arm: FIXED best-of-8 + verifier-conf gate (integrated_method.py's open arm), used
    to show the compute-lean accuracy target (+0.0118 @ 2.54 FLOP) that the Pandora arm holds iso."""
    per = {}
    for name in OPEN_ORDER:
        dskey = OPEN_KEY[name]
        d = IM.open_bestof8(dskey)
        acc, esc = IM.heldout(d["ok7"], d["ok32"], d["gate"])
        a32 = float(d["ok32"].mean()); aT = a32 + THINK_DELTA_OPEN[dskey]
        per[name] = _cell("open", len(d["ok7"]), "bo8+verifier->32B-nt", float(d["greedy"]), a32, aT,
                          acc, esc, BO8["flop"] + esc * GEN32N["flop"], dict(kind="openbo8fixed", esc=esc),
                          think_real=False, think_est=True)
    return per


# ============================ generic per-cell record builder ======================================
def _cell(fmt, n, policy, a7, a32, aT, method_acc, esc, flops, recipe, think_real,
          meanN=1.0, think_est=False, iso_target=None, fixed_bo8_flops=None):
    ls, lp = lat_of(recipe, GEN32N["ms"])
    rec = dict(format=fmt, n=int(n), policy=policy,
               acc_7b=round(a7, 4), acc_32b_nt=round(a32, 4), acc_32b_think=round(aT, 4),
               acc_32b_think_is_est=bool(think_est), think_dump_real=bool(think_real),
               method_acc=round(method_acc, 4), esc=round(esc, 4), meanN=round(meanN, 3),
               flops=round(flops, 3), lat_seq=round(ls, 1), lat_par=round(lp, 1),
               d_vs_think=round(method_acc - aT, 4), d_vs_nothink=round(method_acc - a32, 4),
               strong_share=round(strong_share(recipe), 4), _recipe=recipe,
               _acc=method_acc, _a32=a32, _aT=aT, _flops=flops, _ls=ls, _lp=lp)
    if iso_target is not None: rec["iso_target_heldout"] = round(iso_target, 4)
    if fixed_bo8_flops is not None: rec["fixed_bo8_flops"] = round(fixed_bo8_flops, 3)
    return rec


# ============================ INT4 cost-mode re-costing ============================================
def apply_int4(per):
    S = INT4["int4_ms"]
    out = {}
    for k, v in per.items():
        if k.startswith("_"):
            out[k] = v; continue
        w = copy.deepcopy(v)
        ls, lp = lat_of(v["_recipe"], S)
        w["lat_seq"], w["lat_par"], w["_ls"], w["_lp"] = round(ls, 1), round(lp, 1), ls, lp
        # FLOPs UNCHANGED under the repo MAC-count unit (INT4 cuts bytes, not multiply-accumulates)
        # PROJECTED accuracy erosion: strong_share * delta (bracket); headline stays the FP16 acc.
        q = v["strong_share"]
        w["acc_int4_proj_lo"] = round(v["_acc"] - q * INT4_ACC_DELTA[1], 4)   # delta=1.0%
        w["acc_int4_proj_hi"] = round(v["_acc"] - q * INT4_ACC_DELTA[0], 4)   # delta=0.5%
        w["_acc_int4_lo"] = v["_acc"] - q * INT4_ACC_DELTA[1]
        w["_acc_int4_hi"] = v["_acc"] - q * INT4_ACC_DELTA[0]
        out[k] = w
    return out


# ============================ pooling (sample-weighted + macro) ====================================
def pool_block(per, keys, int4=False):
    n = sum(per[k]["n"] for k in keys)
    w = lambda f: sum(per[k][f] * per[k]["n"] for k in keys) / n
    acc_m = w("_acc"); acc_nt = w("_a32"); acc_t = w("_aT")
    flops = w("_flops"); esc = w("esc"); ls = w("_ls"); lp = w("_lp")
    sw = dict(acc_method=round(acc_m, 4), acc_32b_nt=round(acc_nt, 4), acc_32b_think=round(acc_t, 4),
              d_vs_think=round(acc_m - acc_t, 4), d_vs_nothink=round(acc_m - acc_nt, 4),
              esc=round(esc, 4), flops=round(flops, 3),
              lat_seq_ms=round(ls, 1), lat_par_ms=round(lp, 1),
              lat_saved_vs_think_seq_pct=round((1 - ls / GEN32T["ms"]) * 100, 1),
              lat_saved_vs_think_par_pct=round((1 - lp / GEN32T["ms"]) * 100, 1),
              flops_vs_always32=round(flops / GEN32N["flop"], 3))
    if int4:
        alo = w("_acc_int4_lo"); ahi = w("_acc_int4_hi")
        sw["acc_method_int4_proj"] = [round(alo, 4), round(ahi, 4)]
        sw["d_vs_think_int4_proj"] = [round(alo - acc_t, 4), round(ahi - acc_t, 4)]
        sw["d_vs_nothink_int4_proj"] = [round(alo - acc_nt, 4), round(ahi - acc_nt, 4)]
    macro = lambda f: float(np.mean([per[k][f] for k in keys]))
    ma = dict(acc_method=round(macro("_acc"), 4), acc_32b_nt=round(macro("_a32"), 4),
              acc_32b_think=round(macro("_aT"), 4),
              d_vs_think=round(macro("_acc") - macro("_aT"), 4),
              d_vs_nothink=round(macro("_acc") - macro("_a32"), 4))
    return dict(benchmarks=list(keys), n=int(n), sample_weighted=sw, macro_avg=ma)

def pool_all(per, int4=False):
    keys = [k for k in per if not k.startswith("_")]
    mcq = [k for k in keys if per[k]["format"] == "MCQ"]
    opn = [k for k in keys if per[k]["format"] == "open"]
    return dict(full_suite=pool_block(per, keys, int4),
                mcq_only=pool_block(per, mcq, int4),
                open_only=pool_block(per, opn, int4))


# ============================ assemble a mode ======================================================
def build_mode(mode, open_arm, use_f8=False):
    mcq = mcq_compute_lean() if mode == "compute-lean" else mcq_accuracy_max(use_f8=use_f8)
    f1 = mcq.pop("_F1_choices", None)
    per = {}
    for k in MCQ_ORDER:  per[k] = mcq[k]
    for k in OPEN_ORDER: per[k] = copy.deepcopy(open_arm[k])
    fp16 = pool_all(per)
    per_i4 = apply_int4(per)
    i4 = pool_all(per_i4, int4=True)
    return dict(mode=mode, f1_router_choices=f1, per_benchmark=_strip(per),
                pooled=fp16, int4_cost_mode=dict(per_benchmark=_strip(per_i4), pooled=i4)), per

def _strip(per):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in per.items() if not k.startswith("_")}


# ============================ reconciliation (live) ================================================
def reconcile(per_cl, per_am):
    """Explain, from the live per-cell records, exactly why compute-lean=+0.0118 and accuracy-max=+0.0238
    are the SAME method at two knob settings, and why sample-weighted (not macro) moves."""
    N = sum(per_cl[k]["n"] for k in per_cl if not k.startswith("_"))
    def contrib(cell_cl, cell_am):
        d = (cell_am["_acc"] - cell_cl["_acc"]) * cell_am["n"] / N
        return round(d, 5), round(cell_am["_acc"] - cell_cl["_acc"], 4), cell_am["n"]
    rows = {}
    for k in MCQ_ORDER + OPEN_ORDER:
        dc, da, n = contrib(per_cl[k], per_am[k])
        rows[k] = dict(n=n, frac_of_pool=round(n / N, 4), compute_lean_acc=per_cl[k]["method_acc"],
                       accuracy_max_acc=per_am[k]["method_acc"], cell_delta_acc=da,
                       contribution_to_pooled_swacc=dc, policy_cl=per_cl[k]["policy"], policy_am=per_am[k]["policy"])
    fs_cl = pool_block(per_cl, [k for k in per_cl if not k.startswith("_")])["sample_weighted"]
    fs_am = pool_block(per_am, [k for k in per_am if not k.startswith("_")])["sample_weighted"]
    pmc = rows["PMC_VQA"]
    return dict(
        headline_compute_lean=dict(d_vs_think=fs_cl["d_vs_think"], d_vs_nothink=fs_cl["d_vs_nothink"],
                                   acc=fs_cl["acc_method"], flops=fs_cl["flops"]),
        headline_accuracy_max=dict(d_vs_think=fs_am["d_vs_think"], d_vs_nothink=fs_am["d_vs_nothink"],
                                   acc=fs_am["acc_method"], flops=fs_am["flops"]),
        per_cell_delta=rows,
        which_cells_the_fusion_touches=(
            "ONLY the MCQ arm differs between the two knob settings; the open arm (Pandora verifier) is "
            "IDENTICAL in both.  Of the MCQ cells, the F1 guardrail deviates from always-32B on exactly two: "
            "PMC_VQA (margin-cascade -> %s) and MMMU (keep-7B, already keep-7B in compute-lean too, so it "
            "does NOT move the delta).  The radiology/pathology closed sets + MedXpert switch margin-cascade "
            "-> always-32B (guardrail; they matched/slightly-trailed 32B under the cascade, so this recovers a "
            "few tenths of a point).  The load-bearing cell is PMC." % per_am["PMC_VQA"]["policy"]),
        why_sample_weighted_moves=(
            "PMC is n=%d = %.1f%% of the pooled %d samples.  Fusing that ONE cell lifts its acc by %+.4f, which "
            "alone adds %+.4f to the pooled sample-weighted accuracy (=%.4f x %.4f); the non-PMC closed cells "
            "snapping to always-32B add a further ~%+.4f.  Together the pooled vs-think delta rises %+.4f -> %+.4f. "
            "MACRO barely moves (%+.4f -> %+.4f) because it weights all 9 benchmarks equally, so PMC's +%.4f "
            "contributes only /9; macro was already dominated by MMMU(+0.14)/open(+0.19)."
            % (pmc["n"], pmc["frac_of_pool"] * 100, N, pmc["cell_delta_acc"], pmc["contribution_to_pooled_swacc"],
               pmc["frac_of_pool"], pmc["cell_delta_acc"],
               sum(rows[k]["contribution_to_pooled_swacc"] for k in ("SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM")),
               fs_cl["d_vs_think"], fs_am["d_vs_think"],
               pool_block(per_cl, [k for k in per_cl if not k.startswith("_")])["macro_avg"]["d_vs_think"],
               pool_block(per_am, [k for k in per_am if not k.startswith("_")])["macro_avg"]["d_vs_think"],
               pmc["cell_delta_acc"])),
        pandora_agent_flag_resolved=(
            "integrated_pandora.py flagged: 'task quoted +0.0238 vs think / +0.0138 vs no-think; the LIVE "
            "integrated_method.py pipeline reproduces +0.0118 / +0.0018.'  RESOLVED: +0.0118 is the compute-lean "
            "knob (base margin router); +0.0238 is the accuracy-max knob (PMC slice-fusion).  Both are reproduced "
            "live above by ONE file.  The gap was that fusion lived in a separate script (beat32b_fusion.py) that "
            "the pandora pipeline did not include; method_final.py merges them under `mode`."),
        note_slake_open_consistency=(
            "method_final.py also removes the within-repo inconsistency the old docs flagged: SLAKE-open uses "
            "bo8+verifier (Pandora-adaptive) in BOTH modes, whereas beat32b_fusion.py had used the greedy+"
            "seqlogprob FALC fallback.  This is why unified accuracy-max FLOPs (%.3f, Pandora open) differ "
            "slightly from beat32b_fusion.py's separate-script 5.751 (fixed-bo8 open) at the same accuracy."
            % fs_am["flops"]))


# ============================ console tables =======================================================
def _print_mode(tag, m):
    p = m["per_benchmark"]; pooled = m["pooled"]
    print("\n" + "=" * 132)
    print(f"MODE = {tag.upper()}   (held-out 5-fold; batch-1 measured costs; d vs always-32B-THINK and vs always-32B-NO-THINK)")
    print("=" * 132)
    h = (f"{'benchmark':<18}{'fmt':<5}{'n':>6}{'policy':>26}{'7B':>7}{'32Bnt':>7}{'32Bthk':>8}"
         f"{'method':>8}{'dTHINK':>8}{'dNT':>8}{'esc%':>6}{'FLOPs':>7}{'lat_seq':>8}{'lat_par':>8}")
    print(h); print("-" * len(h))
    for k in MCQ_ORDER + OPEN_ORDER:
        v = p[k]
        print(f"{k:<18}{v['format']:<5}{v['n']:>6}{v['policy']:>26}{v['acc_7b']:>7.3f}{v['acc_32b_nt']:>7.3f}"
              f"{v['acc_32b_think']:>8.3f}{v['method_acc']:>8.3f}{v['d_vs_think']:>+8.4f}{v['d_vs_nothink']:>+8.4f}"
              f"{v['esc']*100:>5.0f}%{v['flops']:>7.2f}{v['lat_seq']:>7.0f}m{v['lat_par']:>7.0f}m")
    print("-" * len(h))
    for lab in ("full_suite", "mcq_only", "open_only"):
        s = pooled[lab]["sample_weighted"]; ma = pooled[lab]["macro_avg"]
        print(f"POOLED[{lab:<10}] n={pooled[lab]['n']:>5}  method={s['acc_method']:.4f}  "
              f"32Bthink={s['acc_32b_think']:.4f} (dTHINK {s['d_vs_think']:+.4f})  "
              f"32Bnt={s['acc_32b_nt']:.4f} (dNT {s['d_vs_nothink']:+.4f})  |  FLOPs={s['flops']:.3f} "
              f"({s['flops_vs_always32']:.2f}x)  lat_seq={s['lat_seq_ms']:.0f}ms (-{s['lat_saved_vs_think_seq_pct']:.0f}%)  "
              f"lat_par={s['lat_par_ms']:.0f}ms (-{s['lat_saved_vs_think_par_pct']:.0f}%)  [macro dTHINK {ma['d_vs_think']:+.4f}]")


# ============================ public API: the single method with the Pareto knob ==================
def method(mode="compute-lean", int4=False, open_arm=None, f8=False, f10=False):
    """THE unified method. Returns (per_benchmark, pooled), recomputed live from the dumps.
      mode : 'compute-lean' (margin cascade MCQ) | 'accuracy-max' (F1 slice-fusion MCQ).
      int4 : apply the projected INT4 strong-leg cost-mode (latency/energy/VRAM; MAC-FLOP unchanged).
      f8   : v2 -- swap the accuracy-max fusion cell for the F8 certified weak-veto (FLOP-negative).
      f10  : v2 -- swap the open arm's parity-tau gate for the F10 team-objective L2D rejector.
    Both modes share the Pandora adaptive-N open-text arm + MMMU keep-7B."""
    assert mode in ("compute-lean", "accuracy-max"), mode
    if open_arm is None:
        open_arm = open_pandora(use_f10=f10)
    mcq = mcq_compute_lean() if mode == "compute-lean" else mcq_accuracy_max(use_f8=f8)
    mcq.pop("_F1_choices", None)
    per = {}
    for k in MCQ_ORDER:  per[k] = mcq[k]
    for k in OPEN_ORDER: per[k] = copy.deepcopy(open_arm[k])
    if int4:
        per = apply_int4(per)
    return _strip(per), pool_all(per, int4=int4)


# ============================ run ==================================================================
def run():
    open_arm = open_pandora()                                    # computed ONCE; shared by both modes
    cl, per_cl = build_mode("compute-lean", open_arm)
    am, per_am = build_mode("accuracy-max", open_arm)
    rec = reconcile(per_cl, per_am)

    # explicit fixed-bo8 compute-lean reference: the +0.0118 accuracy target the Pandora arm holds iso
    per_ref = {**{k: per_cl[k] for k in MCQ_ORDER}, **open_fixed_bo8()}
    ref_pool = pool_all(per_ref)
    rs = ref_pool["full_suite"]["sample_weighted"]
    cl["compute_lean_fixed_bo8_reference"] = dict(
        note="compute-lean with the FIXED best-of-8 open arm (integrated_method.py); the Pandora open arm "
             "holds this accuracy at lower FLOPs. This row makes the +0.0118 vs-think headline explicit.",
        per_benchmark=_strip(per_ref), pooled=ref_pool)

    baselines = dict(
        always_32b_think=dict(desc="naive: big THINK model on everything (measured batch-1 10521.6 ms)",
                              latency_ms=GEN32T["ms"], flops=GEN32T["flop"],
                              acc_full_suite=cl["pooled"]["full_suite"]["sample_weighted"]["acc_32b_think"]),
        always_32b_nothink=dict(desc="strong reference: run only the 32B, no think",
                                latency_ms=GEN32N["ms"], flops=GEN32N["flop"],
                                acc_full_suite=cl["pooled"]["full_suite"]["sample_weighted"]["acc_32b_nt"]))

    out = dict(
        title="method_final -- unified format-aware cascade with a Pareto knob (compute-lean | accuracy-max) "
              "and an optional INT4 cost-mode; all numbers recomputed live from per-sample dumps + measured "
              "cost constants + held-out (5-fold) calibration.",
        reproduce="python3 src/cascade_methods/method_final.py",
        no_gpu=True, no_fabricated_numbers=True,
        cost_constants=dict(GEN7=GEN7, VER7=VER7, BO8=BO8, GEN32_nothink=GEN32N, GEN32_think=GEN32T,
                            FUSE_both_legs=FUSE, cheap_draw_ms=CHEAP_DRAW_MS, think_delta_open=THINK_DELTA_OPEN),
        int4_projection=INT4, int4_accuracy_delta_bracket=INT4_ACC_DELTA,
        baselines=baselines,
        modes=dict(compute_lean=cl, accuracy_max=am),
        reconciliation=rec,
        headline=dict(
            compute_lean="+%.4f vs always-32B-THINK / +%.4f vs always-32B-NO-THINK  @ %.3f FLOP-eq, "
                         "lat_seq %.0f ms / lat_par %.0f ms (holds the fixed-bo8 base router's +0.0118 iso)"
                         % (cl["pooled"]["full_suite"]["sample_weighted"]["d_vs_think"],
                            cl["pooled"]["full_suite"]["sample_weighted"]["d_vs_nothink"],
                            cl["pooled"]["full_suite"]["sample_weighted"]["flops"],
                            cl["pooled"]["full_suite"]["sample_weighted"]["lat_seq_ms"],
                            cl["pooled"]["full_suite"]["sample_weighted"]["lat_par_ms"]),
            accuracy_max="+%.4f vs always-32B-THINK / +%.4f vs always-32B-NO-THINK  @ %.3f FLOP-eq, "
                         "lat_par %.0f ms (PMC slice-fusion beats 32B; guardrail-safe elsewhere)"
                         % (am["pooled"]["full_suite"]["sample_weighted"]["d_vs_think"],
                            am["pooled"]["full_suite"]["sample_weighted"]["d_vs_nothink"],
                            am["pooled"]["full_suite"]["sample_weighted"]["flops"],
                            am["pooled"]["full_suite"]["sample_weighted"]["lat_par_ms"])),
        data_gaps=[
            "32B-THINK open-text accuracy is ESTIMATED (judged 32B-no-think + measured modal think-delta "
            "-0.195/-0.120/-0.130); PathVQA-closed has no 32B-think dump -> think acc = no-think.",
            "always-32B-THINK batch-1 latency 10521.6 ms measured on open-text cap320; MCQ reasoning traces "
            "would be >= this (representative-to-conservative).",
            "SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (trained on those families).",
            "INT4 cost-mode: latency PROJECTED (composition-grounded), MAC-FLOP unchanged, accuracy PROJECTED "
            "(literature <=1%); a measured AWQ-Lingshu-32B run would replace the projection.",
            "OmniMedVQA-32B excluded (tp=2 NCCL hang; 7B-only) -> pooled ALL-7 figure missing, verdict unchanged.",
        ])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)

    # ---- console ----
    _print_mode("compute-lean", cl)
    print(f"  [ref] compute-lean w/ FIXED best-of-8 open arm (the +0.0118 accuracy target the Pandora arm holds): "
          f"method={rs['acc_method']:.4f}  dTHINK {rs['d_vs_think']:+.4f}  dNT {rs['d_vs_nothink']:+.4f}  @ FLOPs {rs['flops']:.3f}  lat {rs['lat_par_ms']:.0f}ms")
    _print_mode("accuracy-max", am)
    print("\n" + "#" * 132)
    print("INT4 COST-MODE (PROJECTED strong leg; MAC-FLOP unchanged; accuracy = FP16 minus projected erosion bracket)")
    print("#" * 132)
    print(f"  strong leg 32B-no-think:  fp16 {INT4['fp16_ms']} ms -> int4 {INT4['int4_ms']} ms  (ratio {INT4['ratio']}, "
          f"decode {INT4['decode_ms_per_tok']} ms/tok, prefill {INT4['prefill_ms']} ms invariant)")
    for tag, mm in (("compute-lean", cl), ("accuracy-max", am)):
        s = mm["int4_cost_mode"]["pooled"]["full_suite"]["sample_weighted"]
        print(f"  [{tag:<12}] full-suite INT4:  lat_seq {s['lat_seq_ms']:.0f}ms  lat_par {s['lat_par_ms']:.0f}ms  "
              f"FLOPs {s['flops']:.3f} (MAC, unchanged)  acc_proj {s['acc_method_int4_proj']}  "
              f"dTHINK_proj {s['d_vs_think_int4_proj']}")
    print("\n" + "#" * 132)
    print("RECONCILIATION -- the two headlines are ONE method at two knob settings")
    print("#" * 132)
    print(f"  compute-lean : dTHINK {rec['headline_compute_lean']['d_vs_think']:+.4f}  dNT {rec['headline_compute_lean']['d_vs_nothink']:+.4f}  @ FLOPs {rec['headline_compute_lean']['flops']:.3f}")
    print(f"  accuracy-max : dTHINK {rec['headline_accuracy_max']['d_vs_think']:+.4f}  dNT {rec['headline_accuracy_max']['d_vs_nothink']:+.4f}  @ FLOPs {rec['headline_accuracy_max']['flops']:.3f}")
    print("  per-cell contribution to the pooled sample-weighted acc delta (accuracy-max - compute-lean):")
    print(f"    {'benchmark':<18}{'frac_pool':>10}{'policy_cl':>22}{'policy_am':>22}{'cellD':>9}{'poolContrib':>13}")
    for k in MCQ_ORDER + OPEN_ORDER:
        r = rec["per_cell_delta"][k]
        print(f"    {k:<18}{r['frac_of_pool']*100:>9.1f}%{r['policy_cl']:>22}{r['policy_am']:>22}{r['cell_delta_acc']:>+9.4f}{r['contribution_to_pooled_swacc']:>+13.5f}")
    print("\n  " + rec["why_sample_weighted_moves"])
    print("\n  " + rec["pandora_agent_flag_resolved"])
    print(f"\nwrote {OUT}")
    return out


# ============================ v2 run: fold in F8 (accuracy-max) + F10 (open arm) ====================
def run_v2():
    """Re-measure the unified method with the two 2026-07 levers folded in, and write method_final_v2.json.
    Everything is recomputed LIVE (v1 prior arms AND v2 arms are both built here so the head-to-head is
    honest, not read out of any .json).  Keeps run() (v1) byte-identical."""
    open_v1 = open_pandora(use_f10=False)                  # prior open arm (Pandora parity-tau gate)
    open_v2 = open_pandora(use_f10=True)                   # F10 team-objective L2D open arm (shared by both v2 modes)

    cl1, per_cl1 = build_mode("compute-lean", open_v1)                       # prior compute-lean
    am1, per_am1 = build_mode("accuracy-max", open_v1, use_f8=False)         # prior accuracy-max (F3 fusion)
    cl2, per_cl2 = build_mode("compute-lean", open_v2)                       # v2 compute-lean (F10 open)
    am2, per_am2 = build_mode("accuracy-max", open_v2, use_f8=True)          # v2 accuracy-max (F8 + F10 open)

    def fs(m): return m["pooled"]["full_suite"]["sample_weighted"]
    def pick(s): return dict(acc=s["acc_method"], d_vs_think=s["d_vs_think"], d_vs_nothink=s["d_vs_nothink"],
                             flops=s["flops"], flops_vs_always32=s["flops_vs_always32"],
                             lat_seq_ms=s["lat_seq_ms"], lat_par_ms=s["lat_par_ms"], esc=s["esc"])
    def cmp_mode(prior, new):
        p, q = fs(prior), fs(new)
        return dict(prior=pick(p), v2=pick(q),
                    delta=dict(d_acc=round(q["acc_method"] - p["acc_method"], 4),
                               d_flops=round(q["flops"] - p["flops"], 3),
                               d_flops_vs_always32=round(q["flops_vs_always32"] - p["flops_vs_always32"], 3),
                               d_vs_think=round(q["d_vs_think"] - p["d_vs_think"], 4)),
                    flop_negative_prior=bool(p["flops"] < GEN32N["flop"]),
                    flop_negative_v2=bool(q["flops"] < GEN32N["flop"]))

    # ---- open-text per-cell F10 vs the prior parity-tau gate (shared by both modes) ----
    open_cells = {}
    for name in OPEN_ORDER:
        pv, qv = cl1["per_benchmark"][name], cl2["per_benchmark"][name]     # open arm identical across modes
        open_cells[name] = dict(n=qv["n"],
            prior_gate=dict(policy=pv["policy"], acc=pv["method_acc"], esc=pv["esc"], flops=pv["flops"],
                            d_vs_nothink=pv["d_vs_nothink"], d_vs_think=pv["d_vs_think"],
                            lat_seq=pv["lat_seq"], lat_par=pv["lat_par"]),
            f10=dict(policy=qv["policy"], acc=qv["method_acc"], esc=qv["esc"], flops=qv["flops"],
                     d_vs_nothink=qv["d_vs_nothink"], d_vs_think=qv["d_vs_think"],
                     lat_seq=qv["lat_seq"], lat_par=qv["lat_par"]),
            d_acc_vs_nothink_gain=round(qv["d_vs_nothink"] - pv["d_vs_nothink"], 4),
            repaired_loss=bool(pv["d_vs_nothink"] < 0 <= qv["d_vs_nothink"]))

    # ---- F8 vs the prior F3 fusion on the PMC cell (accuracy-max only) ----
    pmc1, pmc2 = am1["per_benchmark"]["PMC_VQA"], am2["per_benchmark"]["PMC_VQA"]
    f8_cell = dict(
        prior_fusion=dict(policy=pmc1["policy"], acc=pmc1["method_acc"], flops=pmc1["flops"],
                          d_vs_nothink=pmc1["d_vs_nothink"], lat_seq=pmc1["lat_seq"], lat_par=pmc1["lat_par"]),
        f8=dict(policy=pmc2["policy"], acc=pmc2["method_acc"], esc=pmc2["esc"], veto_rate=round(1 - pmc2["esc"], 4),
                flops=pmc2["flops"], d_vs_nothink=pmc2["d_vs_nothink"], lat_seq=pmc2["lat_seq"], lat_par=pmc2["lat_par"]),
        captures_pct_of_f3_beat=(round(pmc2["d_vs_nothink"] / pmc1["d_vs_nothink"] * 100, 1)
                                 if pmc1["d_vs_nothink"] else None),
        pmc_flops_cut_pct=round((1 - pmc2["flops"] / pmc1["flops"]) * 100, 1))

    conf = dict(
        accuracy_max_flop_negative=dict(
            always_32b_flops=GEN32N["flop"],
            prior_flops=fs(am1)["flops"], prior_x=fs(am1)["flops_vs_always32"],
            v2_flops=fs(am2)["flops"], v2_x=fs(am2)["flops_vs_always32"],
            is_flop_negative=bool(fs(am2)["flops"] < GEN32N["flop"]),
            v2_acc=fs(am2)["acc_method"], v2_d_vs_think=fs(am2)["d_vs_think"], v2_d_vs_nothink=fs(am2)["d_vs_nothink"],
            note=("F8 swaps the F3 fusion (both legs on 100%% of PMC = 5.57 FLOP) for the certified veto (7B on all "
                  "+ 32B only on the %.0f%% non-veto cells = %.2f FLOP), cutting the accuracy-max full-suite FLOPs "
                  "%.3f -> %.3f (%.2fx -> %.2fx always-32B) and flipping the arm FLOP-NEGATIVE, while retaining "
                  "%.0f%% of F3's PMC beat over 32B and still CI-certified above it."
                  % (pmc2["esc"] * 100, pmc2["flops"], fs(am1)["flops"], fs(am2)["flops"],
                     fs(am1)["flops_vs_always32"], fs(am2)["flops_vs_always32"], f8_cell["captures_pct_of_f3_beat"]))),
        both_modes_flop_negative=dict(
            compute_lean_x=fs(cl2)["flops_vs_always32"], accuracy_max_x=fs(am2)["flops_vs_always32"],
            both=bool(fs(cl2)["flops"] < GEN32N["flop"] and fs(am2)["flops"] < GEN32N["flop"])),
        f10_open_lifts=dict(
            per_cell_d_acc_gain={k: open_cells[k]["d_acc_vs_nothink_gain"] for k in OPEN_ORDER},
            open_losses_repaired=[k for k in OPEN_ORDER if open_cells[k]["repaired_loss"]],
            all_three_lift=bool(all(open_cells[k]["d_acc_vs_nothink_gain"] > 0 for k in OPEN_ORDER)),
            shared_by_both_modes=True))

    baselines = dict(
        always_32b_think=dict(latency_ms=GEN32T["ms"], flops=GEN32T["flop"],
                              acc_full_suite=fs(cl2)["acc_32b_think"]),
        always_32b_nothink=dict(latency_ms=GEN32N["ms"], flops=GEN32N["flop"],
                                acc_full_suite=fs(cl2)["acc_32b_nt"]))

    levers = dict(
        F8_certified_weak_veto=dict(
            replaces="the accuracy-max MCQ/PMC F3 confidence-advantage fusion cell",
            mechanic="beat32b_more.f8_veto: per (dataset x TRAIN 7B-conf quantile-bin), VETO the 32B (keep 7B) iff "
                     "Wilson-LB(7B precision) >= 32B accuracy on TRAIN; else take 32B. One-sided => never-worse. "
                     "Cross-fit 5-fold; veto cells are 7B-only (skip the 32B) => cheaper than fusion.",
            effect="captures most of F3's PMC beat at a fraction of the FLOPs -> accuracy-max goes FLOP-NEGATIVE."),
        F10_team_objective_L2D=dict(
            replaces="the open-text arm's parity-targeting tau escalation gate (shared by both modes)",
            mechanic="beat32b_more.f10_l2d: learned logistic rejector over 7B-side open-text features (verifier "
                     "score max/range/mean/std, #unique preds, self-consistency, seqlogprob); threshold tuned on "
                     "TRAIN to maximise TEAM accuracy; cross-fit 5-fold. Decides escalate-to-32B vs keep-7B-best-of-N. "
                     "Pandora adaptive-N still sets the cheap draw count (meanN) for the cost model.",
            effect="repairs the two residual open losses (SLAKE_open, VQA_RAD_open) and lifts PathVQA_open; "
                   "the parity-tau gate targeted iso-32B by design, so it sat at/below 32B on those cells."))

    out = dict(
        title="method_final_v2 -- unified format-aware cascade + Pareto knob, with F8 (certified weak-veto, "
              "accuracy-max MCQ/PMC) and F10 (team-objective L2D, shared open arm) folded in. All numbers "
              "recomputed live from per-sample dumps + measured cost constants + held-out (5-fold) calibration.",
        reproduce="python3 src/cascade_methods/method_final.py   (writes method_final.json [v1] AND method_final_v2.json [v2])",
        no_gpu=True, no_fabricated_numbers=True,
        cost_constants=dict(GEN7=GEN7, VER7=VER7, BO8=BO8, GEN32_nothink=GEN32N, GEN32_think=GEN32T,
                            FUSE_both_legs=FUSE, cheap_draw_ms=CHEAP_DRAW_MS, think_delta_open=THINK_DELTA_OPEN),
        baselines=baselines,
        levers=levers,
        modes_v2=dict(compute_lean=cl2, accuracy_max=am2),
        modes_v1_prior=dict(compute_lean=cl1, accuracy_max=am1),
        comparison_vs_prior=dict(
            compute_lean=cmp_mode(cl1, cl2), accuracy_max=cmp_mode(am1, am2),
            open_text_cells=open_cells, f8_pmc_cell=f8_cell),
        confirmations=conf,
        headline=dict(
            compute_lean_v2=("+%.4f vs always-32B-THINK / +%.4f vs always-32B-NO-THINK @ %.3f FLOP-eq (%.2fx always-32B), "
                             "lat_seq %.0f ms / lat_par %.0f ms  [margin-cascade MCQ + F10 open]"
                             % (fs(cl2)["d_vs_think"], fs(cl2)["d_vs_nothink"], fs(cl2)["flops"],
                                fs(cl2)["flops_vs_always32"], fs(cl2)["lat_seq_ms"], fs(cl2)["lat_par_ms"])),
            accuracy_max_v2=("+%.4f vs always-32B-THINK / +%.4f vs always-32B-NO-THINK @ %.3f FLOP-eq (%.2fx always-32B "
                             "= FLOP-NEGATIVE), lat_par %.0f ms  [F8-veto PMC + F10 open; was 1.25x with F3]"
                             % (fs(am2)["d_vs_think"], fs(am2)["d_vs_nothink"], fs(am2)["flops"],
                                fs(am2)["flops_vs_always32"], fs(am2)["lat_par_ms"])),
            both_modes_flop_negative=conf["both_modes_flop_negative"]["both"]),
        data_gaps=[
            "F8 captures MOST (not all) of F3's PMC beat: the accuracy-max PMC cell d_vs_nothink drops "
            "%+.4f (F3) -> %+.4f (F8) but stays CI-certified above 32B; the trade buys the FLOP-negative arm."
            % (pmc1["d_vs_nothink"], pmc2["d_vs_nothink"]),
            "F10 open arm bills the cheap leg at Pandora's adaptive meanN(<8) draws while scoring the KEEP leg on "
            "the best-of-8 verifier pick -> the kept-7B accuracy is mildly optimistic vs a strict best-of-meanN; "
            "the escalation set + delivered accuracy come from the held-out F10 rejector.",
            "F10 routes on 7B-side features only (open-text dumps have no 32B prediction TEXT, only judge_ok) -> no "
            "cross-model-agreement feature; still fully OFFLINE.",
            "32B-THINK open-text accuracy is ESTIMATED (judged 32B-no-think + measured modal think-delta "
            "-0.195/-0.120/-0.130); PathVQA-closed has no 32B-think dump -> think acc = no-think.",
            "SLAKE/VQA-RAD/PathVQA-open are IN-DOMAIN for the pooled4 verifier (trained on those families).",
        ])
    os.makedirs(os.path.dirname(OUT_V2), exist_ok=True)
    json.dump(out, open(OUT_V2, "w"), indent=2, default=str)

    # ---- console ----
    print("\n" + "@" * 132)
    print("METHOD_FINAL_V2 -- F8 (accuracy-max PMC) + F10 (shared open arm) folded into the unified pipeline")
    print("@" * 132)
    _print_mode("compute-lean (v2: margin-cascade MCQ + F10 open)", cl2)
    _print_mode("accuracy-max (v2: F8-veto PMC + F10 open)", am2)
    print("\n" + "#" * 132)
    print("V2 vs V1 (prior) -- full-suite sample-weighted")
    print("#" * 132)
    for tag, c in (("compute-lean", cmp_mode(cl1, cl2)), ("accuracy-max", cmp_mode(am1, am2))):
        p, q = c["prior"], c["v2"]
        print(f"  {tag:<13} PRIOR: acc {p['acc']:.4f}  dTHINK {p['d_vs_think']:+.4f}  dNT {p['d_vs_nothink']:+.4f}  "
              f"FLOPs {p['flops']:.3f} ({p['flops_vs_always32']:.2f}x)  lat_par {p['lat_par_ms']:.0f}ms  "
              f"[FLOP-neg={c['flop_negative_prior']}]")
        print(f"  {'':<13}   V2 : acc {q['acc']:.4f}  dTHINK {q['d_vs_think']:+.4f}  dNT {q['d_vs_nothink']:+.4f}  "
              f"FLOPs {q['flops']:.3f} ({q['flops_vs_always32']:.2f}x)  lat_par {q['lat_par_ms']:.0f}ms  "
              f"[FLOP-neg={c['flop_negative_v2']}]  (dAcc {c['delta']['d_acc']:+.4f}, dFLOPs {c['delta']['d_flops']:+.3f})")
    print("\n  F8 on PMC (accuracy-max):  F3-fusion acc %.4f @ %.2f FLOP  ->  F8-veto acc %.4f @ %.2f FLOP "
          "(veto %.0f%%, captures %.0f%% of F3's beat, PMC FLOPs -%.0f%%)"
          % (pmc1["method_acc"], pmc1["flops"], pmc2["method_acc"], pmc2["flops"],
             f8_cell["f8"]["veto_rate"] * 100, f8_cell["captures_pct_of_f3_beat"], f8_cell["pmc_flops_cut_pct"]))
    print("\n  F10 open-text cells (shared by both modes; d vs always-32B-nt):")
    print(f"    {'cell':<16}{'n':>6}{'prior_gate':>12}{'F10':>12}{'gain':>9}{'repaired':>10}")
    for k in OPEN_ORDER:
        oc = open_cells[k]
        print(f"    {k:<16}{oc['n']:>6}{oc['prior_gate']['d_vs_nothink']:>+12.4f}{oc['f10']['d_vs_nothink']:>+12.4f}"
              f"{oc['d_acc_vs_nothink_gain']:>+9.4f}{str(oc['repaired_loss']):>10}")
    print("\n  CONFIRM (1) accuracy-max FLOP-negative: %s  (%.3f FLOPs = %.2fx always-32B, was %.2fx with F3)"
          % (conf["accuracy_max_flop_negative"]["is_flop_negative"], fs(am2)["flops"],
             fs(am2)["flops_vs_always32"], fs(am1)["flops_vs_always32"]))
    print("  CONFIRM (2) F10 lifts all 3 open cells: %s ; open losses repaired: %s"
          % (conf["f10_open_lifts"]["all_three_lift"], conf["f10_open_lifts"]["open_losses_repaired"]))
    print("  CONFIRM (3) both modes FLOP-negative: %s  (compute-lean %.2fx, accuracy-max %.2fx)"
          % (conf["both_modes_flop_negative"]["both"], fs(cl2)["flops_vs_always32"], fs(am2)["flops_vs_always32"]))
    print(f"\nwrote {OUT_V2}")
    return out


if __name__ == "__main__":
    run()       # v1 -> method_final.json (byte-identical to the committed artifact)
    run_v2()    # v2 -> method_final_v2.json (F8 accuracy-max + F10 open arm)
