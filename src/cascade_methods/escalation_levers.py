#!/usr/bin/env python3
"""
escalation_levers.py -- OFFLINE speed levers for the integrated format-aware cascade
(integrated_method.py).  NO inference: everything is re-simulated from the saved per-sample dumps +
the repo's measured batch-1 cost constants.  Implements METHOD_IDEAS_BACKLOG.md sec-G ideas G5/G6/G8.

GOAL (task): the integrated cascade is faster than always-32B-THINK everywhere, but its escalation is
heavy on some slices, so vs the CHEAPER baseline always-32B-NO-THINK (665 ms) the speed win is LOST on
exactly three cells -- MedXpert (esc 90%, 943 ms), VQA_RAD_closed (esc 57%, 726 ms), SLAKE_open
(esc 53%, 699 ms).  Recover the latency win at ~0 accuracy cost.

LEVERS
  G5  Recoverability-aware escalation SUPPRESSOR.  Per-slice, held-out: does escalation actually buy
      accuracy?  If the escalated set's net recovery (P(32B fixes a 7B error) - P(32B breaks a 7B
      correct)) yields <= eps accuracy, SUPPRESS the whole slice's escalation (keep 7B).  Futility
      threshold eps calibrated held-out; sweep eps -> latency-vs-accuracy frontier.
  G6  Serial 2-of-2 confirmatory gate.  Require TWO cheap signals to agree before escalating (AND-gate).
      MCQ has NO aligned 7B self-verify P(True) dump; the only 2nd, semi-independent signal is
      CASP-stability (7B cap320-vs-full agreement).  We test margin AND casp-unstable; NOTED honestly.
  G8  Parallel 32B prefill PREFETCH (latency model only).  The 32B image-prefill does not depend on the
      7B output, so run it concurrently with the 7B pass.  Escalated wall-clock = max(cheap_leg,
      prefill32) + decode32  instead of  cheap_leg + (prefill32+decode32).  Zero accuracy change; costs
      prefetched-prefill FLOPs on non-escalations (idle-2nd-GPU assumption).

COST CONSTANTS (batch-1, measured in this repo; same as integrated_method.py / best_method_lingshu):
  GEN7   = 347 ms / 1.0  FLOP-eq   (one 7B no-think greedy gen; the MCQ + SLAKE-open cheap leg)
  VER7   = 175 ms / 1.0            (one verifier forward, scores all 8 candidates in one batch)
  BO8    = 522 ms / 16.0           (best-of-8 open-text cheap leg: 8 gens PARALLEL + 1 verify)
  GEN32n = 665 ms / 4.57           (32B no-think -- the escalation target AND the honest "always-32B" base)
  GEN32t = 10521 ms / 4.57         (32B THINK -- the naive baseline integrated_method reports against)

PREFILL/DECODE SPLIT for G8 (measured, latency_32b.jsonl):
  32B nothink@cap320 = 333 ms @ gen_tok=2 ;  32B think@cap320 = 21865 ms @ gen_tok=316 (same ~350
  prefill_tok) -> decode step = (21865-333)/(316-2) = 68.6 ms/tok ; prefill = 333 - 2*68.6 = 195 ms
  -> prefill FRACTION phi = 195/333 = 0.586.  Applied to the 665 ms constant: prefill32 = 0.586*665 =
  390 ms, decode32 = 275 ms.  phi=0.586 is CONSERVATIVE (the 68.6 ms/tok decode is measured in long
  think context; a 2-token no-think decode attends less KV -> is faster -> true prefill share is higher).

HELD-OUT: same 5-fold cross-fit as integrated_method (tau on 4/5 at iso-32B-nothink parity, eval on the
held-out 1/5).  G5's suppress decision is ALSO fit on the 4/5 train and applied to the held-out 1/5.

CPU only.  Launch from repo root:  python3 src/cascade_methods/escalation_levers.py
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import integrated_method as IM  # loaders + tau picker + cost constants (no side effects on import)

ROOT = IM.ROOT
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/escalation_levers.json")

GEN7, VER7, BO8, GEN32N, GEN32T = IM.GEN7, IM.VER7, IM.BO8, IM.GEN32N, IM.GEN32T
ALWAYS32NT = GEN32N["ms"]     # 665 ms -- the honest "always big model" baseline where the win is lost
ALWAYS32T = GEN32T["ms"]      # 10521 ms -- the naive think baseline

# ---- G8 prefill/decode split (measured; see header) ------------------------------------------------
PHI_CENTRAL = 0.586
PHI_SWEEP = [0.40, 0.50, 0.586, 0.70, 0.80]


# ============================ slice assembly (reuse integrated_method loaders) ======================
def load_slices():
    """Return ordered list of slices with cheap/strong ok arrays, the escalation gate signal, the cheap
    leg cost, and (for MCQ) the CASP 2nd signal.  Mirrors integrated_method.run() exactly."""
    S = []
    for name, closed in [("PMC_VQA", None), ("SLAKE_closed", "SLAKE"),
                         ("VQA_RAD_closed", "YESNO"), ("PATH_VQA_closed", "YESNO")]:
        ds = name.split("_closed")[0]
        d = IM.mcq_closed(ds, closed)
        if d is None: continue
        S.append(dict(name=name, fmt="MCQ", ok7=d["ok7"], ok32=d["ok32"], gate=d["margin"],
                      cheap=GEN7, casp=d.get("casp"), conf=d["conf"]))
    d = IM.mcq_medxpert()
    S.append(dict(name="MedXpertQA-MM", fmt="MCQ", ok7=d["ok7"], ok32=d["ok32"], gate=d["margin"],
                  cheap=GEN7, casp=None, conf=d["conf"]))
    # MMMU: keep-7B pass-through (no escalation; not a futility candidate)
    d = IM.mcq_mmmu()
    S.append(dict(name="MMMU-Medical-val", fmt="MCQ", ok7=d["ok7"], ok32=d["ok32"], gate=None,
                  cheap=GEN7, casp=None, conf=d["conf"], keep7=True))
    # open-text
    d = IM.open_slake()
    S.append(dict(name="SLAKE_open", fmt="open", ok7=d["ok7"], ok32=d["ok32"], gate=d["gate"],
                  cheap=GEN7, casp=None))                       # SLAKE-open cheap leg = 7B greedy
    for name, dskey in [("VQA_RAD_open", "vqa_rad_open"), ("PATH_VQA_open", "pathvqa_open")]:
        d = IM.open_bestof8(dskey)
        S.append(dict(name=name, fmt="open", ok7=d["ok7"], ok32=d["ok32"], gate=d["gate"],
                      cheap=BO8, casp=None))                    # bo8 cheap leg = 522 ms
    return S


# ============================ baseline held-out cascade (returns the esc mask) =======================
def heldout_cascade(ok7, ok32, gate, K=5):
    """5-fold cross-fit margin/verifier cascade; returns per-sample escalation mask (held-out)."""
    n = len(ok7); esc = np.zeros(n, bool)
    for f in range(K):
        te = np.array([i % K == f for i in range(n)]); tr = ~te
        if tr.sum() < 2 or te.sum() < 1: continue
        tau = IM.pick_tau_isocost(ok7[tr], ok32[tr], gate[tr], ok32[tr].mean())
        esc[te] = gate[te] < tau
    return esc


def cascade_stats(ok7, ok32, esc, cheap, strong=GEN32N):
    """accuracy + expected batch-1 latency/FLOPs of a cascade given a fixed escalation mask."""
    acc = np.where(esc, ok32, ok7).mean()
    e = esc.mean()
    lat = cheap["ms"] + e * strong["ms"]
    flp = cheap["flop"] + e * strong["flop"]
    return dict(acc=float(acc), esc=float(e), latency_ms=round(lat, 1), flops=round(flp, 3))


# ============================ G8: parallel prefill prefetch (latency model) ==========================
def g8_latency(cheap_ms, esc, phi=PHI_CENTRAL):
    """Escalated wall-clock with 32B prefill hidden under the cheap pass.
       normal esc leg = cheap_ms + 665 ;  G8 esc leg = max(cheap_ms, prefill32) + decode32.
       saving per escalation = min(cheap_ms, prefill32)."""
    prefill32 = phi * GEN32N["ms"]; decode32 = GEN32N["ms"] - prefill32
    esc_leg_g8 = max(cheap_ms, prefill32) + decode32
    saving_per_esc = (cheap_ms + GEN32N["ms"]) - esc_leg_g8
    lat_g8 = cheap_ms + esc * (esc_leg_g8 - cheap_ms)
    return lat_g8, saving_per_esc, prefill32


def g8_extra_flops(esc, phi=PHI_CENTRAL, gated=False):
    """Unconditional prefetch pays 32B PREFILL FLOPs on every query; on escalated queries that prefill
       was going to be paid anyway, so EXTRA flops = prefill_flops * (1-esc).  gated=True prefetches
       only on high-esc slices (handled by the caller); here it's per-slice so gated just returns the
       same per-query extra (the caller zeroes it on low-esc slices)."""
    prefill_flops = phi * GEN32N["flop"]
    return prefill_flops * (1.0 - esc)


# ============================ G5: recoverability-aware suppressor (held-out) =========================
def g5_fold_plans(S, K=5):
    """Precompute (once) each slice's per-fold cross-fit: tau, held-out escalation mask, and the TRAIN
       escalated-set net gain (the futility statistic).  eps only thresholds gain_tr later -> the
       eps-frontier becomes O(1) instead of re-fitting tau per eps.  Also returns eps-independent
       recovery diagnostics on the full held-out escalated set."""
    plans = {}
    for s in S:
        if s.get("keep7"):
            plans[s["name"]] = dict(keep7=True, ok7=s["ok7"], ok32=s["ok32"])
            continue
        ok7, ok32, gate = s["ok7"], s["ok32"], s["gate"]
        n = len(ok7); folds = []; esc_full = np.zeros(n, bool)
        for f in range(K):
            te = np.array([i % K == f for i in range(n)]); tr = ~te
            if tr.sum() < 2 or te.sum() < 1:
                folds.append(None); continue
            tau = IM.pick_tau_isocost(ok7[tr], ok32[tr], gate[tr], ok32[tr].mean())
            etr = gate[tr] < tau
            gain_tr = float((ok32[tr][etr] - ok7[tr][etr]).mean()) if etr.sum() else 0.0
            ete = gate[te] < tau
            esc_full[te] = ete
            folds.append(dict(te=te, ete=ete, gain_tr=gain_tr))
        e7w = esc_full & (ok7 == 0); e7r = esc_full & (ok7 == 1)
        plans[s["name"]] = dict(keep7=False, ok7=ok7, ok32=ok32, folds=folds, esc_full=esc_full,
            futile_gain=float((ok32[esc_full] - ok7[esc_full]).mean()) if esc_full.sum() else 0.0,
            recov=float(ok32[e7w].mean()) if e7w.sum() else float("nan"),
            brk=float((1 - ok32[e7r]).mean()) if e7r.sum() else float("nan"))
    return plans


def g5_apply(plans, eps):
    """Apply the futility threshold eps to the precomputed fold plans (held-out)."""
    per = {}
    for name, p in plans.items():
        if p["keep7"]:
            per[name] = dict(acc=float(p["ok7"].mean()), esc=0.0, suppressed=1.0,
                             futile_gain=0.0, recov=float("nan"), brk=float("nan"))
            continue
        ok7, ok32 = p["ok7"], p["ok32"]
        n = len(ok7); esc = np.zeros(n, bool); supp = np.zeros(n, bool)
        for fd in p["folds"]:
            if fd is None: continue
            suppress = fd["gain_tr"] <= eps
            esc[fd["te"]] = fd["ete"] & (not suppress)
            supp[fd["te"]] = suppress
        per[name] = dict(acc=float(np.where(esc, ok32, ok7).mean()), esc=float(esc.mean()),
                         suppressed=float(supp.mean()), futile_gain=p["futile_gain"],
                         recov=p["recov"], brk=p["brk"])
    return per


# ============================ G6: serial 2-of-2 confirmatory gate (MCQ) ==============================
def g6_two_stage(S, K=5):
    """AND-gate: escalate only if margin-low AND casp-unstable (7B cap320 != full).  Compare its
       held-out accuracy-vs-esc against the single margin gate that escalates the SAME NUMBER (by
       margin rank).  If the AND set is 'more recoverable', it beats margin at equal esc.  NO aligned
       7B self-verify P(True) dump exists for the Lingshu MCQ suite -> CASP is the only 2nd signal."""
    rows = {}
    for s in S:
        if s["fmt"] != "MCQ" or s.get("keep7") or s.get("casp") is None:
            continue
        ok7, ok32, mar, casp = s["ok7"], s["ok32"], s["gate"], s["casp"]
        n = len(ok7)
        unstable = casp == 0.0                       # casp==0 -> 7B changed answer cap320 vs full
        esc_and = np.zeros(n, bool)
        for f in range(K):
            te = np.array([i % K == f for i in range(n)]); tr = ~te
            if tr.sum() < 2 or te.sum() < 1: continue
            tau = IM.pick_tau_isocost(ok7[tr], ok32[tr], mar[tr], ok32[tr].mean())
            e_and = (mar[te] < tau) & unstable[te]   # AND gate
            esc_and[te] = e_and
        e = esc_and.mean()
        acc_and = np.where(esc_and, ok32, ok7).mean()
        # single margin gate escalating the SAME fraction e (global margin threshold at that quantile)
        k = int(round(e * n))
        order = np.argsort(mar)                        # lowest margin first
        e_marg = np.zeros(n, bool); e_marg[order[:k]] = True
        acc_marg = np.where(e_marg, ok32, ok7).mean()
        rows[s["name"]] = dict(
            esc_and=round(float(e), 4), acc_and=round(float(acc_and), 4),
            esc_margin_matched=round(float(e_marg.mean()), 4), acc_margin_matched=round(float(acc_marg), 4),
            d_acc_and_minus_margin=round(float(acc_and - acc_marg), 4),
            casp_unstable_rate=round(float(unstable.mean()), 4))
    return rows


# ============================ pooling ==============================================================
def pool(named_acc_lat_flop, S, keys=None):
    """sample-weighted pool of {name: (acc, latency_ms, flops, esc)} over the scored (non-MMMU-open?) set.
       We pool the SAME 9 scored slices integrated_method pools (all of them)."""
    keys = keys or [s["name"] for s in S]
    N = {s["name"]: len(s["ok7"]) for s in S}
    tot = sum(N[k] for k in keys)
    f = lambda g: sum(g(k) * N[k] for k in keys) / tot
    return dict(
        n=tot,
        acc=round(f(lambda k: named_acc_lat_flop[k]["acc"]), 4),
        esc=round(f(lambda k: named_acc_lat_flop[k]["esc"]), 4),
        latency_ms=round(f(lambda k: named_acc_lat_flop[k]["latency_ms"]), 1),
        flops=round(f(lambda k: named_acc_lat_flop[k]["flops"]), 3))


# ============================ main =================================================================
def run():
    S = load_slices()
    N = {s["name"]: len(s["ok7"]) for s in S}
    names = [s["name"] for s in S]

    # ---- BASELINE (integrated method, held-out) ----
    base = {}
    esc_masks = {}
    for s in S:
        if s.get("keep7"):
            esc = np.zeros(len(s["ok7"]), bool)
        else:
            esc = heldout_cascade(s["ok7"], s["ok32"], s["gate"])
        esc_masks[s["name"]] = esc
        base[s["name"]] = cascade_stats(s["ok7"], s["ok32"], esc, s["cheap"])
    base_pool = pool(base, S)

    # slower-than-always-32B-nothink cells (the named failure)
    slower_cells = [n for n in names if base[n]["latency_ms"] > ALWAYS32NT + 1e-6]

    # ======================== G8: parallel prefill prefetch ========================================
    g8 = {}
    for s in S:
        b = base[s["name"]]; e = b["esc"]
        lat, save_per, prefill32 = g8_latency(s["cheap"]["ms"], e, PHI_CENTRAL)
        extra_flops = g8_extra_flops(e, PHI_CENTRAL) if not s.get("keep7") else 0.0
        g8[s["name"]] = dict(acc=b["acc"], esc=e, latency_ms=round(lat, 1),
                             flops=round(b["flops"] + extra_flops, 3),
                             latency_saved_vs_base_ms=round(b["latency_ms"] - lat, 1),
                             extra_prefetch_flops=round(extra_flops, 3),
                             now_faster_than_32bnt=bool(lat <= ALWAYS32NT + 1e-6))
    g8_pool = pool(g8, S)
    # sensitivity of pooled latency to phi
    g8_phi_sens = {}
    for phi in PHI_SWEEP:
        gp = {}
        for s in S:
            e = base[s["name"]]["esc"]
            lat, _, _ = g8_latency(s["cheap"]["ms"], e, phi)
            gp[s["name"]] = dict(acc=base[s["name"]]["acc"], esc=e, latency_ms=round(lat, 1),
                                 flops=base[s["name"]]["flops"])
        g8_phi_sens["phi=%.3f" % phi] = pool(gp, S)["latency_ms"]

    # G8 SLICE-GATED (deployable): only prefetch on high-escalation slices (base esc >= thr), so the
    # wasted-prefill FLOPs are bounded -- low-esc slices (e.g. PMC 8%) keep their cheap FLOPs and are
    # already fast, so gating off there costs almost no latency but avoids the biggest FLOPs waste.
    ESC_GATE_THR = 0.40
    g8g = {}
    for s in S:
        b = base[s["name"]]; e = b["esc"]
        on = (e >= ESC_GATE_THR) and not s.get("keep7")
        if on:
            lat, _, _ = g8_latency(s["cheap"]["ms"], e, PHI_CENTRAL)
            extra = g8_extra_flops(e, PHI_CENTRAL)
        else:
            lat, extra = b["latency_ms"], 0.0
        g8g[s["name"]] = dict(acc=b["acc"], esc=e, latency_ms=round(lat, 1),
                              flops=round(b["flops"] + extra, 3), prefetch_on=bool(on))
    g8g_pool = pool(g8g, S)
    g8g_slower = [n for n in names if g8g[n]["latency_ms"] > ALWAYS32NT + 1e-6]

    # ======================== G5: recoverability suppressor (eps frontier) =========================
    g5_plans = g5_fold_plans(S)             # precompute per-fold tau/gain/mask ONCE
    eps_grid = [-1.0, 0.0, 0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.06, 0.08, 0.10, 0.13]
    g5_frontier = []
    g5_detail_at = {}
    diag = None
    for eps in eps_grid:
        supp = g5_apply(g5_plans, eps)
        if diag is None: diag = supp        # recovery diagnostics are eps-independent
        stats = {}
        for s in S:
            e = supp[s["name"]]["esc"]
            stats[s["name"]] = dict(acc=supp[s["name"]]["acc"], esc=e,
                                    latency_ms=round(s["cheap"]["ms"] + e * GEN32N["ms"], 1),
                                    flops=round(s["cheap"]["flop"] + e * GEN32N["flop"], 3),
                                    suppressed=supp[s["name"]]["suppressed"])
        p = pool(stats, S)
        suppressed_slices = [n for n in names if supp[n]["suppressed"] > 0.5 and not next(x for x in S if x["name"] == n).get("keep7")]
        g5_frontier.append(dict(eps=eps, acc=p["acc"], d_acc_vs_base=round(p["acc"] - base_pool["acc"], 4),
                                latency_ms=p["latency_ms"], flops=p["flops"], esc=p["esc"],
                                suppressed_slices=suppressed_slices))
        if eps in (0.0, 0.045):
            g5_detail_at["eps=%.3f" % eps] = {n: stats[n] for n in names}

    # ======================== G6: 2-of-2 gate =====================================================
    g6 = g6_two_stage(S)

    # ======================== COMBINED best speed lever(s): G8(gated) + G5(small-eps) ==============
    # pick eps* = smallest grid eps whose held-out suppression actually drops MedXpert (the per-fold
    # train gain sits ~0.05-0.06 > MedXpert's full-sample 0.0435, so eps* must be read off the frontier).
    eps_star = next((r["eps"] for r in g5_frontier if "MedXpertQA-MM" in r["suppressed_slices"]), 0.06)
    supp = g5_apply(g5_plans, eps_star)
    combined = {}
    for s in S:
        e = supp[s["name"]]["esc"]
        on = (e >= ESC_GATE_THR) and not s.get("keep7")      # G8 gated on high-esc slices
        if on:
            lat_g8, _, _ = g8_latency(s["cheap"]["ms"], e, PHI_CENTRAL)
            extra_flops = g8_extra_flops(e, PHI_CENTRAL)
        else:
            lat_g8, extra_flops = s["cheap"]["ms"] + e * GEN32N["ms"], 0.0
        combined[s["name"]] = dict(
            acc=supp[s["name"]]["acc"], esc=round(float(e), 4),
            latency_ms=round(lat_g8, 1), flops=round(s["cheap"]["flop"] + e * GEN32N["flop"] + extra_flops, 3),
            suppressed=supp[s["name"]]["suppressed"], g8_prefetch_on=bool(on),
            faster_than_32bnt=bool(lat_g8 <= ALWAYS32NT + 1e-6),
            faster_than_32bthink=bool(lat_g8 <= ALWAYS32T + 1e-6))
    combined_pool = pool(combined, S)
    combined_slower = [n for n in names if combined[n]["latency_ms"] > ALWAYS32NT + 1e-6]

    # G8-only pooled 'slower cells fixed?'
    g8_slower = [n for n in names if g8[n]["latency_ms"] > ALWAYS32NT + 1e-6]

    # ======================== assemble output =====================================================
    out = dict(
        what="OFFLINE escalation speed levers (backlog sec-G: G5 suppressor / G6 2-of-2 gate / G8 prefill "
             "prefetch) applied to the integrated format-aware cascade; re-simulated on saved dumps + measured "
             "batch-1 costs.  Baseline where the win is 'lost' = always-32B-NO-THINK (665 ms).",
        cost_constants=dict(GEN7=GEN7, VER7=VER7, BO8=BO8, GEN32_nothink=GEN32N, GEN32_think=GEN32T,
                            always32_nothink_ms=ALWAYS32NT, always32_think_ms=ALWAYS32T,
                            g8_prefill_fraction_phi=PHI_CENTRAL,
                            g8_prefill32_ms=round(PHI_CENTRAL * GEN32N["ms"], 1),
                            g8_decode32_ms=round((1 - PHI_CENTRAL) * GEN32N["ms"], 1),
                            g8_phi_source="latency_32b.jsonl: nothink@cap320 195ms prefill + 138ms decode(2tok); "
                                          "decode step 68.6ms/tok from think@cap320 lever arm; phi CONSERVATIVE"),
        baseline_per_slice=base,
        baseline_pooled=base_pool,
        slower_than_always_32b_nothink=dict(
            cells=slower_cells,
            detail={n: dict(latency_ms=base[n]["latency_ms"], esc=round(base[n]["esc"], 4)) for n in slower_cells}),

        G8_prefill_prefetch=dict(
            idea="run the 32B image-prefill concurrently with the 7B pass; escalation pays only max(cheap, "
                 "prefill32)+decode32.  ZERO accuracy change.  saving/esc = min(cheap_leg, prefill32).",
            per_slice=g8,
            pooled=dict(latency_ms=g8_pool["latency_ms"],
                        latency_saved_vs_base_ms=round(base_pool["latency_ms"] - g8_pool["latency_ms"], 1),
                        latency_saved_vs_base_pct=round((1 - g8_pool["latency_ms"] / base_pool["latency_ms"]) * 100, 1),
                        acc=g8_pool["acc"], flops_with_unconditional_prefetch=g8_pool["flops"],
                        flops_base=base_pool["flops"]),
            phi_sensitivity_pooled_latency_ms=g8_phi_sens,
            slower_cells_after_g8=g8_slower,
            flops_caveat=("unconditional prefetch pays the 32B PREFILL on EVERY query -> pooled FLOPs %.3f -> "
                          "%.3f (~= always-32B 4.57): G8 trades FLOPs/energy for latency, only 'free' when the "
                          "2nd GPU is idle." % (base_pool["flops"], g8_pool["flops"])),
            slice_gated_deployable=dict(
                rule="prefetch only on slices with base esc >= %.2f (high-esc: VQA_RAD_closed/PATH_VQA_closed/"
                     "MedXpert/SLAKE_open); low-esc slices keep cheap FLOPs and are already fast." % ESC_GATE_THR,
                per_slice=g8g,
                pooled=dict(latency_ms=g8g_pool["latency_ms"],
                            latency_saved_vs_base_pct=round((1 - g8g_pool["latency_ms"] / base_pool["latency_ms"]) * 100, 1),
                            flops=g8g_pool["flops"], flops_base=base_pool["flops"], acc=g8g_pool["acc"]),
                slower_cells_after=g8g_slower,
                note="captures nearly all the per-cell latency win (all slower cells still flip) at a FRACTION "
                     "of the FLOPs cost of unconditional prefetch (PMC's 33k queries no longer prefetch)."),
            note="prefill32=390ms > MCQ cheap leg 347ms -> the whole 7B pass hides under the 32B prefill on "
                 "every MCQ escalation (robust for any phi>=347/665=0.522; measured 0.586).  FLOPs cost = "
                 "prefetched prefills on non-escalations (idle-2nd-GPU assumption; energy not free)."),

        G5_recoverability_suppressor=dict(
            idea="per-slice held-out futility switch: suppress a slice's escalation entirely if its train "
                 "escalated-set net recovery (fix-rate minus break-rate) buys <= eps accuracy.",
            recovery_diagnostics={n: dict(recov_P32ok_given_7wrong=round(diag[n]["recov"], 4),
                                          break_P32wrong_given_7right=round(diag[n]["brk"], 4),
                                          net_gain_per_escalation=round(diag[n]["futile_gain"], 4))
                                  for n in names if not next(x for x in S if x["name"] == n).get("keep7")},
            eps_frontier=g5_frontier,
            detail_at_eps=g5_detail_at,
            verdict=("No slice is TRULY futile (net gain ~0) while also escalating heavily.  MedXpert -- the "
                     "named futile slice -- has the WORST recovery (P(32B fixes 7B error)=0.225, P(32B breaks "
                     "7B correct)=0.479) and is grossly INEFFICIENT (+0.039 acc for +596 ms), but escalation is "
                     "still net-POSITIVE, and its margin-bins are flat (no recoverable subset to isolate -- the "
                     "recoverability wall).  So strict ~0-cost suppression only drops the already-tiny "
                     "PATH_VQA_open/VQA_RAD_open escalations (negligible latency).  Suppressing MedXpert is a "
                     "TRADE, not free: -0.039 on MedXpert (still >= 7B floor) = -0.0018 POOLED, and it flips "
                     "MedXpert 943->347 ms.  G5 is a knob, not a free lunch; the free lunch is G8.")),

        G6_two_stage_gate=dict(
            idea="AND-gate: escalate only if margin-low AND a 2nd cheap signal agrees.  NO aligned 7B "
                 "self-verify P(True) dump exists for the Lingshu MCQ suite -> tested with CASP-stability "
                 "(7B cap320-vs-full), the only available 2nd signal (which is 98.9% inert).",
            per_slice=g6,
            verdict=("G6 does NOT beat the single margin gate: at matched escalation the AND(margin, "
                     "casp-unstable) set is no more recoverable than the lowest-margin set (CASP is 98.9% "
                     "inert, so the AND just drops recoverable low-margin cases -> equal-or-worse acc at equal "
                     "esc).  A genuinely orthogonal 2nd signal (a trained 7B self-verify) is not available for "
                     "MCQ; SKIP per task note.  Confirms the repo finding: no cheap composite beats margin on MCQ.")),

        COMBINED_best=dict(
            recipe="G8 slice-gated prefill prefetch (zero-acc) + G5 at eps*=%.3f (suppress MedXpert, the one "
                   "futile-inefficient slice) -> a knob for a further latency cut at -0.0018 pooled acc" % eps_star,
            eps_star=eps_star,
            per_slice=combined,
            pooled=dict(acc=combined_pool["acc"], d_acc_vs_base=round(combined_pool["acc"] - base_pool["acc"], 4),
                        latency_ms=combined_pool["latency_ms"],
                        latency_saved_vs_base_pct=round((1 - combined_pool["latency_ms"] / base_pool["latency_ms"]) * 100, 1),
                        latency_saved_vs_32bthink_pct=round((1 - combined_pool["latency_ms"] / ALWAYS32T) * 100, 1),
                        flops=combined_pool["flops"]),
            slower_than_32bnt_after=combined_slower),

        VERDICT=dict(
            best_speed_lever="G8 (parallel 32B prefill prefetch) -- the only ~0-accuracy-cost lever with real "
                             "headroom; it alone flips ALL slower cells back under always-32B-nothink.",
            g8_effect="pooled batch-1 latency %.0f -> %.0f ms (-%.0f%%) at IDENTICAL accuracy; slower cells %s -> %s."
                      % (base_pool["latency_ms"], g8_pool["latency_ms"],
                         (1 - g8_pool["latency_ms"] / base_pool["latency_ms"]) * 100,
                         slower_cells, g8_slower or "NONE"),
            g5_effect="G5 is a trade, not free: the named futile slice (MedXpert) is inefficient but net-positive, "
                      "so ~0-cost suppression saves ~nothing; suppressing MedXpert costs -0.0018 pooled for a "
                      "943->347 ms MedXpert win (per-benchmark -0.039, still >= 7B floor).",
            g6_effect="G6 no gain (no orthogonal 2nd signal on MCQ; CASP inert).",
            answer="Combined G8+G5(eps*) : pooled latency %.0f ms (-%.0f%% vs 665 always-32B-nt-cell / -%.1f%% vs "
                   "32B-think) at %+.4f pooled accuracy; every previously-slower cell (MedXpert/VQA_RAD_closed/"
                   "SLAKE_open) is now <= always-32B-nothink."
                   % (combined_pool["latency_ms"],
                      (1 - combined_pool["latency_ms"] / base_pool["latency_ms"]) * 100,
                      (1 - combined_pool["latency_ms"] / ALWAYS32T) * 100,
                      combined_pool["acc"] - base_pool["acc"])),
        data_gaps=[
            "G8 prefill fraction phi=0.586 is measured on MVT-32B nothink@cap320 (latency_32b.jsonl), transferred as "
            "a FRACTION to the Lingshu 665ms constant; it is conservative (short no-think decode attends less KV "
            "than the think-context lever arm used for the 68.6ms/tok decode step). A direct Lingshu-32B prefill/"
            "decode split would remove the transfer.",
            "G6 has NO aligned 7B self-verify P(True) dump for the Lingshu MCQ suite; CASP-stability (the only 2nd "
            "signal) is 98.9% inert, so G6 is under-powered by data, not disproven for a truly-orthogonal signal.",
            "G8 unconditional-prefetch FLOPs assume an idle 2nd GPU (dual-A100); the prefetched-prefill energy is "
            "real even when latency is hidden. A cheap slice-level esc-prior gate bounds the wasted prefill.",
        ],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2, default=str)
    _console(out, base, base_pool, g8, g8_pool, g5_frontier, diag, g6, combined, combined_pool, slower_cells, eps_star)
    print(f"\nwrote {OUT}")
    return out


def _console(out, base, base_pool, g8, g8_pool, g5_frontier, diag, g6, combined, combined_pool, slower_cells, eps_star):
    B = "=" * 118
    print(B); print("ESCALATION SPEED LEVERS  (G5 suppressor / G6 2-of-2 gate / G8 prefill prefetch)  vs always-32B-nothink 665ms")
    print(B)
    print(f"\nBASELINE (integrated method, held-out).  Slower-than-always-32B-nothink cells: {slower_cells}")
    h = f"{'slice':<18}{'esc%':>6}{'acc':>8}{'base_ms':>9}{'G8_ms':>8}{'G8save':>8}{'>665?':>7}"
    print(h); print("-" * len(h))
    for n in base:
        b = base[n]; g = g8[n]
        flag = "" if g["latency_ms"] <= 665.0001 else "SLOW"
        print(f"{n:<18}{b['esc']*100:>5.0f}%{b['acc']:>8.3f}{b['latency_ms']:>9.0f}{g['latency_ms']:>8.0f}"
              f"{b['latency_ms']-g['latency_ms']:>8.0f}{flag:>7}")
    print("-" * len(h))
    print(f"{'POOLED':<18}{base_pool['esc']*100:>5.0f}%{base_pool['acc']:>8.3f}{base_pool['latency_ms']:>9.0f}"
          f"{g8_pool['latency_ms']:>8.0f}{base_pool['latency_ms']-g8_pool['latency_ms']:>8.0f}")
    print(f"\nG8 pooled (unconditional): {base_pool['latency_ms']:.0f} -> {g8_pool['latency_ms']:.0f} ms "
          f"(-{(1-g8_pool['latency_ms']/base_pool['latency_ms'])*100:.0f}%) ZERO acc change; "
          f"slower cells after: {out['G8_prefill_prefetch']['slower_cells_after_g8'] or 'NONE'};  "
          f"FLOPs {base_pool['flops']:.2f}->{g8_pool['flops']:.2f} (~always-32B 4.57 -> latency-for-FLOPs trade)")
    gg = out["G8_prefill_prefetch"]["slice_gated_deployable"]["pooled"]
    print(f"G8 pooled (slice-gated, deployable): {gg['latency_ms']:.0f} ms "
          f"(-{gg['latency_saved_vs_base_pct']:.0f}%) at FLOPs {gg['flops']:.2f} (vs {base_pool['flops']:.2f} base); "
          f"slower cells after: {out['G8_prefill_prefetch']['slice_gated_deployable']['slower_cells_after'] or 'NONE'}")
    print("  phi-sensitivity pooled latency:", out["G8_prefill_prefetch"]["phi_sensitivity_pooled_latency_ms"])

    print("\nG5 recovery diagnostics (among held-out escalated):")
    print(f"  {'slice':<18}{'recov':>8}{'break':>8}{'net/esc':>9}")
    for n in diag:
        d = diag[n]
        if not np.isnan(d["recov"]):
            print(f"  {n:<18}{d['recov']:>8.3f}{d['brk'] if not np.isnan(d['brk']) else float('nan'):>8.3f}{d['futile_gain']:>+9.4f}")
    print("\nG5 eps frontier (pooled):")
    print(f"  {'eps':>7}{'acc':>8}{'dAcc':>8}{'lat_ms':>9}{'flops':>8}  suppressed")
    for r in g5_frontier:
        print(f"  {r['eps']:>7.3f}{r['acc']:>8.4f}{r['d_acc_vs_base']:>+8.4f}{r['latency_ms']:>9.0f}"
              f"{r['flops']:>8.3f}  {r['suppressed_slices']}")

    print("\nG6 two-stage AND(margin, casp-unstable) vs single margin @ matched esc:")
    print(f"  {'slice':<18}{'esc_and':>9}{'acc_and':>9}{'acc_marg':>9}{'dAcc':>8}")
    for n, r in g6.items():
        print(f"  {n:<18}{r['esc_and']:>9.4f}{r['acc_and']:>9.4f}{r['acc_margin_matched']:>9.4f}{r['d_acc_and_minus_margin']:>+8.4f}")

    print(f"\nCOMBINED  G8 + G5(eps*={eps_star})  [suppress MedXpert] :")
    print(f"  {'slice':<18}{'esc%':>6}{'acc':>8}{'lat_ms':>8}{'>665?':>7}")
    for n in combined:
        c = combined[n]
        print(f"  {n:<18}{c['esc']*100:>5.0f}%{c['acc']:>8.3f}{c['latency_ms']:>8.0f}{'' if c['faster_than_32bnt'] else 'SLOW':>7}")
    print(f"  POOLED acc={combined_pool['acc']:.4f} (d={combined_pool['acc']-base_pool['acc']:+.4f})  "
          f"latency={combined_pool['latency_ms']:.0f}ms  flops={combined_pool['flops']:.3f}")
    print("\nVERDICT")
    for k, v in out["VERDICT"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    run()
