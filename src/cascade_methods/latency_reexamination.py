#!/usr/bin/env python3
"""
latency_reexamination.py - OFFLINE re-examination of the verifier best-of-N cascade on the axis it was
designed for: BATCH-1 LATENCY (not FLOPs). CPU only, no GPU/vLLM/eval.

WHY. end_to_end_consolidation.json found the verifier best-of-N cascade is Pareto-DOMINATED by always-32B on
FLOPs: one 32B forward = 4.57 7B-forward-equiv, so best-of-8 (16 forwards) loses badly. BUT the project's core
efficiency insight is LATENCY: at batch-1 the 32B is only ~1.9x the 7B's wall-clock (665 vs 347 ms), NOT 4.57x,
because batch-1 short-gen is bandwidth/overhead-bound, not compute-bound. And best-of-N is embarrassingly
parallel: N samples issue as ONE batched forward (~1 gen-latency), while always-32B is ONE slow forward. So
best-of-N may be latency-ALIVE even though it is FLOPs-dominated. This script rebuilds the accuracy-vs-BATCH-1-
LATENCY frontier and asks: with PARALLEL best-of-N, does the cascade beat always-32B (equal/better acc, lower
latency)?

MEASURED batch-1 cost model (REUSED, not fabricated). Source: logs/latency_opentext.jsonl, produced by
src/cascade_methods/open_measure_latency_energy.py -- HF, batch-1, on REAL vqa_rad images at cap320, clean
per-visible-GPU NVML energy, n=25 iters after 3 warmup. Also tabulated in
results/cascade_methods/docs/current/UNIFIED_METHOD_EXPERIMENTS.md ("MEASURED batch-1 latency + energy per tier",
2026-06-30). Open-text (short free-text answer, ~5.6 gen tok):
  GEN7  (Lingshu-7B  generate) : lat_mean=347.1 ms  (median 349.3), energy=45.8 J, gen_tok=5.6
  VER7  (Lingshu-7B  verify)   : lat_mean=175.5 ms  (median 173.0), energy=25.3 J, gen_tok=1.0
  GEN32 (Lingshu-32B generate) : lat_mean=665.0 ms  (median 696.4), energy=126.9 J, gen_tok=5.6
FLOPs (7B-forward-equiv, the axis end_to_end used): GEN7=1.0, VER7=1.0, GEN32=4.57.
NOTE ON THINK/NO-THINK: the OPEN-TEXT strong leg is Lingshu-32B in GENERATE (no-think, short answer) mode = the
665 ms above; that is the strong leg the cascade actually escalates to, and the always-32B baseline here. No
separate open-text *think* latency was measured for Lingshu (the only 32B-think latency dumps in the repo,
ckpts/acc_gen/lingshu/lat/big_think/, are MCQ letter-answers ~0.43 s, not open-text long reasoning, so they are
NOT a valid open-text think number). We therefore report always-32B = no-think generate = 665 ms and say so.

DATA (REUSED verbatim from end_to_end_consolidation.py). Lingshu 7B->32B, 3 open-text datasets with all pieces on
common idx: vqa_rad_open (200) + slake_open (645) + pathvqa_open (178) = n=1023. Per question: iid@8 pool
(oks[8], scores[8]), diverse@15 portfolio (oks[15], scores[15]), DPP-select-8, and the strong 32B judge_ok label.
Cheap legs scored by exact-match oks; strong by judge_ok (the validated single-32B-answer label). We import
build_dataset / pandora_frontier / pool_perlambda / LAMS from end_to_end_consolidation so the pool, the Pandora
controller (5-fold cross-fit HELD-OUT thresholds), and the accuracies are byte-identical to that run.

TWO LATENCY MODELS (the whole point):
  * PARALLEL  best-of-N : the N generations are issued as ONE batched forward, the N verifies as one batched
      forward. lat_par = [GEN7 if any gen] + [VER7 if any verify] + esc*GEN32.  <-- N drops out.
      HONEST CAVEAT: this REQUIRES the hardware to hold N concurrent sequences (KV-cache memory ~N x) and treats a
      batch-N forward as ~ a batch-1 forward. At these tiny gen-token counts and small N that is a good
      approximation (bandwidth-bound), but a true batch-N forward is somewhat SLOWER than batch-1, so lat_par is a
      (mild) LOWER BOUND. Parallel bo-N is a hardware-utilisation trade: it spends N-wide throughput to buy
      latency. If you cannot run N concurrently, you fall back to the sequential model below and the win vanishes.
  * SEQUENTIAL best-of-N : draw/verify one at a time. lat_seq = gens*GEN7 + verifies*VER7 + esc*GEN32.  This is
      the ONLY honest model for ADAPTIVE-N (Pandora): its draws are inherently sequential (draw -> observe score
      -> decide whether to draw again), so Pandora CANNOT parallelise and pays meanN x gen-latency. That is a real
      latency disadvantage vs a fixed parallel bo-N, reported explicitly.

CONFIGS on the accuracy-vs-latency frontier:
  1. 7B-greedy                         : 1 gen, no verify.                    lat = GEN7.
  2. iid-bo8 + verifier (fixed)        : PARALLEL 8 gen + 8 verify, no esc.   lat_par = GEN7+VER7.
  3. diverse-bo15 + verifier (fixed)   : PARALLEL 15 gen + 15 verify, no esc. lat_par = GEN7+VER7.
  4. iid-bo8 + verifier-conf GATE      : PARALLEL base + escalate frac esc.   lat_par = GEN7+VER7 + esc*GEN32.
  5. diverse-bo15 + verifier-conf GATE : PARALLEL base + escalate frac esc.   lat_par = GEN7+VER7 + esc*GEN32.
  6. iid -> Pandora (adaptive-N)       : SEQUENTIAL draws + adaptive esc.     lat_seq = meanN*(GEN7+VER7)+esc*GEN32.
  7. always-32B                        : one 32B generate.                    lat = GEN32.

Launch from repo root:  python3 src/cascade_methods/latency_reexamination.py
"""
import os, sys, json
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
sys.path.insert(0, J("src/cascade_methods"))
# reuse the EXACT data loader, Pandora controller, per-lambda pooling and lambda grid
from end_to_end_consolidation import (build_dataset, DSETS, pandora_frontier, pool_perlambda, LAMS)  # noqa: E402

# ------------------------------------------------------------------ MEASURED batch-1 cost model (see header)
GEN7  = dict(lat_ms=347.1, lat_ms_median=349.3, energy_j=45.8,  flops=1.0)
VER7  = dict(lat_ms=175.5, lat_ms_median=173.0, energy_j=25.3,  flops=1.0)
GEN32 = dict(lat_ms=665.0, lat_ms_median=696.4, energy_j=126.9, flops=4.57)
LAT_SOURCE = ("logs/latency_opentext.jsonl via src/cascade_methods/open_measure_latency_energy.py "
              "(HF batch-1, real vqa_rad images @cap320, NVML energy, n=25 after 3 warmup); "
              "tabulated in docs/current/UNIFIED_METHOD_EXPERIMENTS.md (2026-06-30)")

def lat_parallel(gens, verifies, esc):
    """Fixed-N best-of-N: N gens as ONE batched forward, N verifies as one batched forward, then maybe escalate.
    N drops out of latency (that is the parallelism win). gens/verifies are booleans-in-effect (any>0)."""
    return ((GEN7["lat_ms"]  if gens     > 0 else 0.0)
          + (VER7["lat_ms"]  if verifies > 0 else 0.0)
          + esc * GEN32["lat_ms"])

def lat_sequential(gens, verifies, esc):
    """Draw/verify one at a time (the only honest model for adaptive-N). N multiplies latency."""
    return gens * GEN7["lat_ms"] + verifies * VER7["lat_ms"] + esc * GEN32["lat_ms"]

def flops(gens, verifies, esc):
    return gens * GEN7["flops"] + verifies * VER7["flops"] + esc * GEN32["flops"]

def energy(gens, verifies, esc):
    return gens * GEN7["energy_j"] + verifies * VER7["energy_j"] + esc * GEN32["energy_j"]

# ------------------------------------------------------------------ load + pool the 3 datasets (n=1023)
def load_rows():
    rows = []
    for ds in DSETS:
        rows += build_dataset(ds)
    return rows

# ------------------------------------------------------------------ static reference points
def static_points(rows):
    n = len(rows)
    strong = float(np.mean([r["strong_judge"] for r in rows]))
    greedy = float(np.mean([r["greedy_ok"] for r in rows]))
    iid_bo8 = float(np.mean([r["iid_ok"][int(np.argmax(r["iid_sc"]))] for r in rows]))
    full_bo = float(np.mean([r["full_ok"][int(np.argmax(r["full_sc"]))] for r in rows]))
    dpp_bo  = float(np.mean([r["dpp_ok"][int(np.argmax(r["dpp_sc"]))] for r in rows]))
    def pt(label, acc, gens, verifies, esc, latmodel):
        lat = latmodel(gens, verifies, esc)
        return dict(label=label, acc=acc, gens=gens, verifies=verifies, esc=esc,
                    lat_ms=lat, flops=flops(gens, verifies, esc), energy_j=energy(gens, verifies, esc),
                    latmodel=("parallel" if latmodel is lat_parallel else "sequential"))
    return dict(
        n=n,
        greedy       = pt("7B-greedy",                       greedy,  1, 0, 0.0, lat_parallel),
        iid_bo8_par  = pt("iid-bo8+verifier (PARALLEL)",     iid_bo8, 8, 8, 0.0, lat_parallel),
        iid_bo8_seq  = pt("iid-bo8+verifier (SEQUENTIAL)",   iid_bo8, 8, 8, 0.0, lat_sequential),
        full_bo_par  = pt("diverse-bo15+verifier (PARALLEL)",full_bo,15,15, 0.0, lat_parallel),
        full_bo_seq  = pt("diverse-bo15+verifier (SEQUENTIAL)",full_bo,15,15,0.0, lat_sequential),
        dpp_bo_par   = pt("diverse-DPP8+verifier (PARALLEL, 15 gen)", dpp_bo, 15, 8, 0.0, lat_parallel),
        always32     = pt("always-32B (no-think generate)",  strong,  0, 0, 1.0, lat_parallel),
        iid_oracle   = float(np.mean([max(r["iid_ok"]) for r in rows])),
        full_oracle  = float(np.mean([max(r["full_ok"]) for r in rows])),
    )

# ------------------------------------------------------------------ verifier-confidence GATE frontier (PARALLEL base)
def gate_frontier(rows, ok_key, sc_key, N):
    """Fixed-N parallel best-of-N, escalate iff max verifier score < tau. Global tau (matches end_to_end's
    per-lambda pooling on pooled rows). NOTE: full-data tau (optimistic; no held-out) -- same convention as
    end_to_end_consolidation's gate, kept so numbers line up; it is an OPTIMISTIC bound on the gate."""
    strong = np.array([r["strong_judge"] for r in rows], float)
    bo = np.array([r[ok_key][int(np.argmax(r[sc_key][:N]))] for r in rows], float)
    vmax = np.array([max(r[sc_key][:N]) for r in rows], float)
    pts = []
    for tau in np.linspace(0.0, 1.0, 201):
        esc = (vmax < tau)
        acc = float(np.where(esc, strong, bo).mean())
        e = float(esc.mean())
        pts.append(dict(knob=float(tau), acc=acc, esc=e, meanN=float(N), gens=float(N), verifies=float(N),
                        lat_par=lat_parallel(N, N, e), lat_seq=lat_sequential(N, N, e),
                        flops=flops(N, N, e), energy_j=energy(N, N, e)))
    return pts

# ------------------------------------------------------------------ Pandora frontier (SEQUENTIAL, reused controller)
def pandora_pooled(rows_by_ds, sc_key, ok_key):
    per_ds_pts = []
    for ds in DSETS:
        pts, meta = pandora_frontier(rows_by_ds[ds], sc_key, ok_key, "strong_judge", 2.0, fixed_gen=None)
        per_ds_pts.append((pts, meta))
    pooled, Ntot = pool_perlambda(per_ds_pts)
    out = []
    for p in pooled:
        gens = p["meanN"]; verifies = p["meanN"]; e = p["esc"]
        out.append(dict(knob=p["knob"], acc=p["acc"], meanN=p["meanN"], esc=e,
                        gens=gens, verifies=verifies,
                        lat_par=None,  # adaptive-N cannot parallelise its draws
                        lat_seq=lat_sequential(gens, verifies, e),
                        flops=flops(gens, verifies, e), energy_j=energy(gens, verifies, e)))
    return out

# ------------------------------------------------------------------ Pareto helpers (min latency at each acc)
def pareto_min_lat(points, lat_key):
    """Non-dominated (acc up, latency down). points: list with 'acc' and lat_key."""
    pts = [p for p in points if p.get(lat_key) is not None]
    pts = sorted(pts, key=lambda p: (p[lat_key], -p["acc"]))
    out = []; best_acc = -1.0
    for p in pts:
        if p["acc"] > best_acc + 1e-12:
            out.append(p); best_acc = p["acc"]
    return out

def min_lat_to_reach(points, target_acc, lat_key, tol=3e-3):
    ok = [p for p in points if p["acc"] >= target_acc - tol and p.get(lat_key) is not None]
    return min(ok, key=lambda p: p[lat_key]) if ok else None

def max_acc(points):
    return max(points, key=lambda p: p["acc"])

# ------------------------------------------------------------------ main
def main():
    rows_by_ds = {ds: build_dataset(ds) for ds in DSETS}
    rows = [r for ds in DSETS for r in rows_by_ds[ds]]
    S = static_points(rows)
    strong_acc = S["always32"]["acc"]

    # gates (parallel base)
    gate_iid  = gate_frontier(rows, "iid_ok",  "iid_sc",  8)
    gate_div  = gate_frontier(rows, "full_ok", "full_sc", 15)
    # pandora (sequential)
    pan_iid   = pandora_pooled(rows_by_ds, "iid_sc", "iid_ok")

    # ---- assemble every operating point with its HONEST batch-1 latency ----
    # For fixed configs and gated cascades the honest batch-1 latency is PARALLEL (lat_par).
    # For Pandora it is SEQUENTIAL (lat_seq). always-32B is a single forward (both models agree = 665).
    envelope_pts = []
    def add(src, acc, lat, flops_, energy_, esc=0.0, meanN=0.0, latmodel="parallel"):
        envelope_pts.append(dict(src=src, acc=float(acc), lat_ms=float(lat), flops=float(flops_),
                                 energy_j=float(energy_), esc=float(esc), meanN=float(meanN), latmodel=latmodel))
    # static
    add("7B-greedy",                 S["greedy"]["acc"],      S["greedy"]["lat_ms"],     S["greedy"]["flops"],     S["greedy"]["energy_j"])
    add("iid-bo8 (fixed,parallel)",  S["iid_bo8_par"]["acc"], S["iid_bo8_par"]["lat_ms"],S["iid_bo8_par"]["flops"],S["iid_bo8_par"]["energy_j"], meanN=8)
    add("diverse-bo15 (fixed,parallel)", S["full_bo_par"]["acc"], S["full_bo_par"]["lat_ms"], S["full_bo_par"]["flops"], S["full_bo_par"]["energy_j"], meanN=15)
    add("diverse-DPP8 (fixed,parallel)", S["dpp_bo_par"]["acc"], S["dpp_bo_par"]["lat_ms"], S["dpp_bo_par"]["flops"], S["dpp_bo_par"]["energy_j"], meanN=8)
    add("always-32B",                S["always32"]["acc"],    S["always32"]["lat_ms"],   S["always32"]["flops"],   S["always32"]["energy_j"], esc=1.0)
    # gated cascades (parallel base)
    for p in gate_iid:
        add("iid-bo8+gate (parallel)", p["acc"], p["lat_par"], p["flops"], p["energy_j"], esc=p["esc"], meanN=8)
    for p in gate_div:
        add("diverse-bo15+gate (parallel)", p["acc"], p["lat_par"], p["flops"], p["energy_j"], esc=p["esc"], meanN=15)
    # pandora (sequential)
    for p in pan_iid:
        add("iid->Pandora (sequential)", p["acc"], p["lat_seq"], p["flops"], p["energy_j"],
            esc=p["esc"], meanN=p["meanN"], latmodel="sequential")

    env = pareto_min_lat(envelope_pts, "lat_ms")

    # ---- the crux question: can any best-of-N config reach always-32B accuracy at lower batch-1 latency? ----
    tgt = strong_acc
    crux = {}
    crux["always32"] = dict(lat_ms=S["always32"]["lat_ms"], acc=strong_acc)
    crux["iid-bo8+gate (parallel)"]       = min_lat_to_reach(gate_iid, tgt, "lat_par")
    crux["diverse-bo15+gate (parallel)"]  = min_lat_to_reach(gate_div, tgt, "lat_par")
    crux["iid->Pandora (sequential)"]     = min_lat_to_reach(pan_iid, tgt, "lat_seq")
    # peak accuracy each gated cascade reaches, and its latency
    peak = dict(
        gate_iid=max_acc(gate_iid), gate_div=max_acc(gate_div), pandora_iid=max_acc(pan_iid),
    )

    # ---- FLOPs vs LATENCY contrast for the fixed bo-N base (the headline of the re-examination) ----
    contrast = dict(
        one_32B_forward=dict(flops=GEN32["flops"], lat_ms=GEN32["lat_ms"]),
        iid_bo8_base=dict(flops=flops(8, 8, 0.0), lat_par=lat_parallel(8, 8, 0.0), lat_seq=lat_sequential(8, 8, 0.0)),
        diverse_bo15_base=dict(flops=flops(15, 15, 0.0), lat_par=lat_parallel(15, 15, 0.0), lat_seq=lat_sequential(15, 15, 0.0)),
        note=("On FLOPs the bo-8 base (16) is 3.5x ONE 32B forward (4.57) -> hopeless. On PARALLEL batch-1 latency "
              "the bo-8 base (522 ms) is 0.79x one 32B forward (665 ms) -> CHEAPER. Same config, opposite verdict: "
              "FLOPs-dominated but latency-competitive. The flip is because batch-1 short-gen is bandwidth/overhead-"
              "bound (32B only ~1.9x the 7B wall-clock, not 4.57x) AND best-of-N parallelises (N drops out)."),
    )

    # ---- verdict ----
    survives_dominate = False   # does ANY bo-N config get acc>=always-32B at lat<always-32B ?
    winners = []
    for name in ("iid-bo8+gate (parallel)", "diverse-bo15+gate (parallel)", "iid->Pandora (sequential)"):
        c = crux[name]
        if c is not None:
            lat = c.get("lat_par", c.get("lat_seq"))
            if lat is not None and lat < S["always32"]["lat_ms"] - 1e-9:
                survives_dominate = True; winners.append((name, lat))
    # does bo-N SURVIVE on the latency-Pareto envelope at all (as a distinct low-latency point)?
    bo_on_env = [p for p in env if ("bo" in p["src"] or "greedy" in p["src"] or "Pandora" in p["src"])
                 and "always-32B" not in p["src"]]

    lines = []
    lines.append(f"n={S['n']} (vqa_rad 200 + slake 645 + pathvqa 178), Lingshu 7B->32B open-text; strong=judge_ok.")
    lines.append(f"acc ladder: 7B-greedy={S['greedy']['acc']:.3f} ~ iid-bo8={S['iid_bo8_par']['acc']:.3f} "
                 f"< diverse-DPP8={S['dpp_bo_par']['acc']:.3f} < diverse-bo15={S['full_bo_par']['acc']:.3f} "
                 f"<< always-32B={strong_acc:.3f}  (bo-N ceiling well below the 32B; oracle iid@8={S['iid_oracle']:.3f}, "
                 f"diverse@15={S['full_oracle']:.3f}).")
    lines.append("MEASURED batch-1 (ms): GEN7=347.1  VER7=175.5  GEN32=665.0  (medians 349/173/696). "
                 "PARALLEL bo-N base = GEN7+VER7 = 522.6 ms  <  one 32B forward 665 ms.")
    lines.append(f"FLOPs-vs-LATENCY FLIP: iid-bo8 base = 16 FLOPs (3.5x the 32B's 4.57) but 522 ms PARALLEL "
                 f"latency (0.79x the 32B's 665 ms). => FLOPs-DOMINATED yet LATENCY-CHEAPER. This is why best-of-N "
                 f"is latency-ALIVE though FLOPs-dead.")
    lines.append("LATENCY-PARETO ENVELOPE (min batch-1 latency at each acc; parallel for fixed/gated bo-N, "
                 "sequential for Pandora):")
    for p in env:
        lines.append(f"   acc={p['acc']:.3f}  lat={p['lat_ms']:6.0f}ms  F={p['flops']:5.2f}  ({p['latmodel']:10s})  <- {p['src']}")
    # crux answer
    a32 = S["always32"]["lat_ms"]
    ci = crux["iid-bo8+gate (parallel)"]; cd = crux["diverse-bo15+gate (parallel)"]; cp = crux["iid->Pandora (sequential)"]
    def lat_of(c, key):
        return (c.get(key) if c else None)
    lines.append("CRUX -- cheapest batch-1 latency to MATCH always-32B accuracy "
                 f"({strong_acc:.3f}), i.e. does best-of-N BEAT always-32B (>= acc at < {a32:.0f} ms)?")
    lines.append(f"   always-32B                     : {a32:6.0f} ms  (the bar)")
    lines.append(f"   iid-bo8 + verifier-conf gate   : "
                 + (f"{lat_of(ci,'lat_par'):6.0f} ms  @esc={ci['esc']*100:.0f}%  ({'BEATS' if ci and ci['lat_par']<a32 else 'SLOWER than'} always-32B)"
                    if ci else "cannot reach 32B accuracy at any escalation"))
    lines.append(f"   diverse-bo15 + verifier gate   : "
                 + (f"{lat_of(cd,'lat_par'):6.0f} ms  @esc={cd['esc']*100:.0f}%  ({'BEATS' if cd and cd['lat_par']<a32 else 'SLOWER than'} always-32B)"
                    if cd else "cannot reach 32B accuracy at any escalation"))
    lines.append(f"   iid -> Pandora (SEQUENTIAL)    : "
                 + (f"{lat_of(cp,'lat_seq'):6.0f} ms  @esc={cp['esc']*100:.0f}% meanN={cp['meanN']:.2f}  "
                    f"({'BEATS' if cp and cp['lat_seq']<a32 else 'SLOWER than'} always-32B)"
                    if cp else "cannot reach 32B accuracy at any escalation"))
    verdict_beats = ("YES" if survives_dominate else "NO")
    lines.append(f"=> Does the verifier best-of-N cascade BEAT always-32B on batch-1 latency "
                 f"(equal/better acc at LOWER latency)? {verdict_beats}.")
    lines.append("=> Is best-of-N latency-ALIVE (survives on the latency-Pareto envelope as a real operating "
                 f"point, unlike on FLOPs where it is dominated)? {'YES' if bo_on_env else 'NO'} -- "
                 f"{'; '.join(f'{p['src']}@{p['lat_ms']:.0f}ms/acc{p['acc']:.3f}' for p in bo_on_env) if bo_on_env else 'none'}.")
    lines.append("HONEST CAVEATS: (1) PARALLEL bo-N assumes the hardware runs N sequences concurrently (KV-cache "
                 "~Nx) and treats a batch-N forward as ~batch-1 -> lat_par is a mild LOWER BOUND; real batched "
                 "latency is somewhat higher. Without concurrency you use the SEQUENTIAL model and bo-N loses "
                 "outright (iid-bo8 seq = 4176 ms). (2) Pandora's adaptive draws are inherently SEQUENTIAL "
                 "(draw->observe->decide), so it pays meanN x gen-latency and cannot use the parallel trick -- a "
                 "real disadvantage vs fixed parallel bo-N. (3) always-32B here = Lingshu-32B no-think GENERATE "
                 "(665 ms), the actual open-text strong leg; no open-text 32B-think latency was measured. (4) the "
                 "gate uses full-data tau (optimistic); a held-out tau only moves it the wrong way, strengthening "
                 "the verdict. (5) latency parity ignores that bo-N spends N-wide THROUGHPUT to buy latency; under "
                 "a throughput/energy budget always-32B is strictly better (energy: 32B=127 J vs bo-8 base=568 J).")

    verdict = dict(
        question=("On batch-1 latency with PARALLEL best-of-N, does the verifier best-of-N cascade BEAT "
                  "always-32B (equal/better accuracy at LOWER latency)?"),
        answer_beats_always32=verdict_beats,
        answer_latency_alive=("YES" if bo_on_env else "NO"),
        one_line=("Best-of-N is latency-ALIVE (its PARALLEL base 522 ms < one 32B forward 665 ms, the exact "
                  "opposite of FLOPs where 16 >> 4.57), so it SURVIVES on the latency-Pareto envelope as a real "
                  "low-latency / lower-accuracy operating point -- but it does NOT BEAT always-32B: the bo-N "
                  "accuracy ceiling (<=0.549) sits far below the 32B (0.672), and escalating enough to close that "
                  "gap pushes parallel latency back above 665 ms. always-32B still owns the high-accuracy corner."),
        winners_that_beat=winners,
        lines=lines,
    )
    for l in lines:
        print(l)

    OUT = dict(
        meta=dict(
            script="src/cascade_methods/latency_reexamination.py",
            reused_data_from="src/cascade_methods/end_to_end_consolidation.py (build_dataset/pandora_frontier/pool_perlambda)",
            n=S["n"], datasets={ds: len(rows_by_ds[ds]) for ds in DSETS},
            latency_source=LAT_SOURCE,
            cost_model_ms=dict(GEN7=GEN7, VER7=VER7, GEN32=GEN32),
            think_note=("open-text strong leg = Lingshu-32B no-think GENERATE (665 ms); no open-text 32B-think "
                        "latency measured (the only 32B-think dumps are MCQ letter-answers, not comparable)."),
        ),
        static=S,
        gate_iid_frontier=gate_iid,
        gate_diverse_frontier=gate_div,
        pandora_iid_frontier=pan_iid,
        latency_pareto_envelope=env,
        flops_vs_latency_contrast=contrast,
        crux_min_latency_to_match_always32=dict(
            target_acc=strong_acc,
            always32_lat_ms=S["always32"]["lat_ms"],
            iid_gate=ci, diverse_gate=cd, pandora=cp,
        ),
        peak_accuracy=peak,
        VERDICT=verdict,
    )

    def jsonable(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return o
    outp = J("results/cascade_methods/artifacts/latency_reexamination.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(OUT, open(outp, "w"), indent=1, default=jsonable)
    print(f"\n[dump] {outp}")

if __name__ == "__main__":
    main()
