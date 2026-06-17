#!/usr/bin/env python3
"""
rt_analyze.py - analyze a rt_cascade JSONL (partial or complete; safe to run mid-run).
Prints: pooled summary (now incl. an always-32B ceiling from the validated vLLM gate_32b labels),
a data-driven gate-discrimination verdict, escalation 2x2 (rescued/broken/wasted/redundant),
latency percentiles, energy/correct, a 32B-truncation proxy, a FAITHFULNESS check vs the vLLM 32B
eval labels, and a per-benchmark table that compares cascade vs always-7B vs always-32B side by
side. VRAM is not per-query (constant resident); read it from the run's startup 'resident:' line.
"""
import argparse, json, glob, os, re
import numpy as np
from collections import defaultdict

def load_jsonl(path): return [json.loads(l) for l in open(path) if l.strip()]
def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                try: r = json.loads(l); d[m.group(1)][r["idx"]] = r
                except Exception: pass
    return d
def pct(a, q): return float(np.percentile(a, q)) if len(a) else float("nan")

FOUR = {"PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"}  # competent benchmarks (excludes MMMU, MedXpert)

def run(path, repo, maxtok, four_only=False):
    rows = load_jsonl(path)
    if four_only:
        rows = [r for r in rows if r["dataset"] in FOUR]
        print(f"[--four-only] restricting analysis to {sorted(FOUR)}\n")
    n = len(rows)
    esc = np.array([r["escalate"] for r in rows]); ok = np.array([r["ok"] for r in rows])
    ok7 = np.array([r["pred7"] == r["gold"] for r in rows])
    lat = np.array([r["latency_s"] for r in rows]); en = np.array([r["energy_j"] for r in rows])

    # ---- Load the validated vLLM 32B eval labels ONCE (gate_32b). These ran the 32B on the FULL
    # ---- eval set, so they give an always-32B "ceiling" with full per-question coverage. The
    # ---- real-time run only has HF 32B answers on the ESCALATED rows, so HF can't supply an
    # ---- always-32B baseline (it never ran 32B on the kept rows) -- vLLM is the right source.
    r32v = load_arm(os.path.join(repo, "ckpts/gate_32b"), "think_norag")
    flat = {}
    for _, rs in r32v.items():
        for i, rr in rs.items(): flat[i] = rr
    always32 = np.array([flat[r["idx"]].get("ok") if r["idx"] in flat else None for r in rows], dtype=object)
    has32 = np.array([v is not None for v in always32])
    a32_acc = float(np.mean([v for v in always32[has32]])) if has32.any() else float("nan")
    cov32 = has32.mean()

    # Cascade score IF its 32B leg were the validated vLLM 32B (isolates the gate design from any
    # HF<->vLLM implementation drift): keep the 7B answer when not escalated, swap in the vLLM
    # outcome when escalated and labeled, else fall back to the actual cascade outcome.
    casc_vllm = []
    for r in rows:
        if r["escalate"] and r["idx"] in flat and flat[r["idx"]].get("ok") is not None:
            casc_vllm.append(1 if flat[r["idx"]].get("ok") else 0)
        else:
            casc_vllm.append(1 if r["ok"] else 0)
    casc_vllm = np.array(casc_vllm, dtype=float)

    print(f"==== {os.path.basename(path)}  (n={n}) ====\n")

    print("POOLED")
    print(f"  cascade acc      : {ok.mean():.3f}")
    print(f"  always-7B acc    : {ok7.mean():.3f}   (floor: cheap no_think leg on every query)")
    if has32.any():
        print(f"  always-32B acc   : {a32_acc:.3f}   (ceiling: validated vLLM 32B, same questions; "
              f"coverage {int(has32.sum())}/{n} = {cov32:.0%})")
    print(f"  escalation rate  : {esc.mean():.1%}   (compute proxy: share of queries that also pay the 32B cost)")
    print(f"  latency  mean    : {lat.mean():.2f}s    p50/p90/p95/p99/max: "
          f"{pct(lat,50):.1f}/{pct(lat,90):.1f}/{pct(lat,95):.1f}/{pct(lat,99):.1f}/{lat.max():.1f}s")
    print(f"  energy / query   : {en.mean():.0f} J    energy / correct: {en.sum()/max(ok.sum(),1):.0f} J")
    if has32.any():
        print(f"  --- three-way comparison ---")
        print(f"  cascade vs always-7B   : {ok.mean()-ok7.mean():+.3f}   (accuracy bought over the floor)")
        print(f"  cascade vs always-32B  : {ok[has32].mean()-a32_acc:+.3f}   "
              f"(gap to the ceiling on labeled rows; ~0 => cascade matches always-32B accuracy)")
        print(f"  cascade acc (vLLM-32B leg) : {casc_vllm.mean():.3f}   "
              f"(cascade design scored with the validated 32B; HF-cascade gap {casc_vllm.mean()-ok.mean():+.3f})")
    if ok.mean() < ok7.mean() - 1e-9:
        print(f"  ** WARNING: cascade ({ok.mean():.3f}) < always-7B ({ok7.mean():.3f}) -- escalation is net")
        print(f"     HURTING accuracy. The 32B leg or the gate is misbehaving; the 2x2 + faithfulness")
        print(f"     check below say which. **")
    print()

    print("GATE DISCRIMINATION  (is it escalating the questions the 7B gets wrong?)")
    if esc.any() and (~esc).any():
        a_esc = ok7[esc].mean(); a_keep = ok7[~esc].mean(); gap = a_keep - a_esc
        print(f"  7B acc | escalated     : {a_esc:.3f}   (want LOW -- the unsure ones)")
        print(f"  7B acc | NOT escalated : {a_keep:.3f}   (want HIGH -- kept as confident)")
        print(f"  discrimination gap     : {gap:+.3f}   (kept minus escalated; bigger = sharper gate)")
        if gap >= 0.10:
            print("  -> gate IS discriminating: it sends up the hard ones and keeps the easy ones.")
        elif gap >= 0.03:
            print("  -> weak discrimination: some signal, but the gate is only loosely sorting hard vs easy.")
        else:
            print("  -> ~equal => gate isn't discriminating -> margin-scale problem (HF margin != trained scale).")
    else:
        print("  (need both escalated and kept queries to assess discrimination)")
    print()

    print("ESCALATION OUTCOME  (where the 32B compute goes; escalated queries only)")
    e = [r for r in rows if r["escalate"]]
    if e:
        r7 = np.array([x["pred7"] == x["gold"] for x in e]); r32 = np.array([x["pred32"] == x["gold"] for x in e])
        rescued=int(((~r7)&( r32)).sum()); broken=int(( r7 &(~r32)).sum())
        wasted =int(((~r7)&(~r32)).sum()); redundant=int(( r7 &( r32)).sum()); tot=len(e)
        print(f"  rescued   (7B wrong, 32B right): {rescued:4d}  ({rescued/tot:.0%})   <- accuracy routing BOUGHT")
        print(f"  broken    (7B right, 32B wrong): {broken:4d}  ({broken/tot:.0%})   <- accuracy routing LOST")
        print(f"  wasted    (both wrong)         : {wasted:4d}  ({wasted/tot:.0%})   <- 32B compute, still wrong")
        print(f"  redundant (both right)         : {redundant:4d}  ({redundant/tot:.0%})   <- 32B compute, not needed")
        print(f"  NET accuracy from routing = rescued - broken = {rescued-broken:+d}  ({(rescued-broken)/tot:+.0%} of esc)")
        print(f"  32B acc on escalated set       = {r32.mean():.3f}   (this is the HARD TAIL by construction,")
        print(f"                                    so a low number is EXPECTED -- not proof the 32B is broken;")
        print(f"                                    the FAITHFULNESS check below is what decides that)")
    print()

    g32 = np.array([r.get("gen32",0) for r in rows if r["escalate"]])
    if len(g32):
        hit = (g32>=maxtok-1).mean()
        flag = "  <- OK, nothing truncated" if hit < 0.02 else "  <- HIGH: reasoning truncated, no answer to parse"
        print(f"32B GENERATION  mean gen_tokens={g32.mean():.0f}  max={int(g32.max())}  "
              f"hit cap({maxtok}): {hit:.0%}{flag}\n")

    matched = [(r, flat[r["idx"]]) for r in rows if r["escalate"] and r["idx"] in flat]
    print(f"FAITHFULNESS vs vLLM 32B eval  (escalated idx also in gate_32b, n={len(matched)})")
    if matched:
        agree = np.mean([hf["pred32"] == vl.get("pred") for hf, vl in matched])
        hf_ok = np.mean([hf["pred32"] == hf["gold"] for hf, vl in matched])
        vl_ok = np.mean([vl["ok"] for hf, vl in matched])
        acc_gap = vl_ok - hf_ok
        print(f"  HF-32B vs vLLM-32B prediction agreement : {agree:.3f}")
        print(f"  HF-32B acc {hf_ok:.3f}   vs   vLLM-32B acc {vl_ok:.3f}   (same questions; acc gap {acc_gap:+.3f})")
        # The bug signal is an ACCURACY gap, not mere prediction disagreement: two stochastic decodes
        # of the SAME model can disagree per-item yet match in accuracy. Key the verdict on the gap.
        if abs(acc_gap) > 0.05:
            print("  -> ACCURACY GAP: the HF 32B leg performs differently from the validated eval == a real bug.")
            print("     Check prompt template / answer parsing / image resolution on the HF path.")
        elif agree < 0.85:
            print("  -> accuracies MATCH but per-item agreement is sub-1.0: decoding stochasticity")
            print("     (temperature/top_p/seed), NOT a correctness bug. Set greedy + fixed seed to pin it.")
        else:
            print("  -> high agreement AND matched accuracy: HF 32B faithfully reproduces the eval.")
    else:
        print("  no overlap with gate_32b labels -- can't cross-check on this subsample.")
    print()

    print("PER-BENCHMARK  (casc vs always-7B vs always-32B; vs-columns are cascade minus that baseline)")
    print(f"  {'benchmark':<22}{'n':>5}{'casc':>8}{'7B':>8}{'32B':>8}"
          f"{'vs7B':>8}{'vs32B':>8}{'esc%':>6}{'lat':>8}{'prs7%':>7}")
    by = defaultdict(list)
    for r in rows: by[r["dataset"]].append(r)
    any_partial = False
    for dsn in sorted(by):
        rs = by[dsn]
        a = np.array([r["ok"] for r in rs]); a7 = np.array([r["pred7"]==r["gold"] for r in rs])
        e_ = np.array([r["escalate"] for r in rs]); l = np.array([r["latency_s"] for r in rs])
        par = np.array([r["pred7"] != "?" for r in rs])
        lab = [r for r in rs if r["idx"] in flat and flat[r["idx"]].get("ok") is not None]
        if lab:
            a32 = float(np.mean([flat[r["idx"]].get("ok") for r in lab]))
            am  = np.array([r["ok"] for r in lab])
            a7m = np.array([r["pred7"]==r["gold"] for r in lab])
            a32s = f"{a32:.3f}"; d7s = f"{am.mean()-a7m.mean():+.3f}"; d32s = f"{am.mean()-a32:+.3f}"
            if len(lab) < len(rs): any_partial = True
        else:
            a32s = d7s = d32s = "n/a"
        print(f"  {dsn:<22}{len(rs):>5}{a.mean():>8.3f}{a7.mean():>8.3f}{a32s:>8}"
              f"{d7s:>8}{d32s:>8}{100*e_.mean():>5.0f}%{l.mean():>8.2f}{100*par.mean():>6.0f}%")
    if any_partial:
        print("  (note: 32B/vs columns use only rows that have a vLLM label; some benchmarks are <100% covered)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    ap.add_argument("--maxtok", type=int, default=2048)
    ap.add_argument("--four-only", action="store_true",
                    help="restrict the whole analysis to PMC-VQA, SLAKE, VQA-RAD, PathVQA")
    A = ap.parse_args(); run(A.jsonl, A.repo, A.maxtok, A.four_only)