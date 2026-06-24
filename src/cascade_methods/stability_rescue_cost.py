#!/usr/bin/env python3
"""
stability_rescue_cost.py - prefill-inclusive FLOPs + accuracy + guardrail for the Visual-Stability
RESCUE gate, per benchmark and FLOPs-weighted, vs the deployed margin gate.

Method (PARAMETER-FREE rescue): keep the deployed margin gate (tau=0.4264) but DON'T escalate a
low-margin sample if its 7B-nt answer is invariant across an extra resolution ladder (visually
stable). The stability is computed with EXTRA cheap 7B passes that fire ONLY on rescue-eligible
(margin<tau) samples, so their cost is charged honestly.

FLOPs model identical to cascade_cost_prefill_flops.py: one model run = 2*N*(P+G); P = prompt
tokens incl. vision (from token_cache.json), G = generated tokens (from checkpoints).
  deployed(q) = run7@cap320 + [margin<tau] run32think
  rescue(q)   = run7@cap320 + [margin<tau]( run7@cap80 + run7@cap160 + run7@cap640 )
                            + [margin<tau AND not visually-stable] run32think
Reports: 32B-call rate, backbone% of always-32B, accuracy, per-benchmark guardrail (>= always-cheap).
token_cache positions are aligned to checkpoint idx by reproducing the pipeline's fixed_slice order.
"""
import os, re, json, glob, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
TC = json.load(open("ckpts/token_cache.json"))
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
def gate_escalate(m):  # DEPLOYED rule: logistic gate on margin, threshold the probability at tau
    return bool(GATE.predict_proba(np.array([[m]], dtype=np.float32))[0, 1] < TAU)
N7, N32 = 7.6e9, 33.0e9
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
EXTRA_SETS = {"3-cap (cap80,160,640)": ["cap80", "cap160", "cap640"],
              "2-cap (cap80,160)":     ["cap80", "cap160"],
              "2-cap (cap160,640)":    ["cap160", "cap640"]}

def load_jsonl(path):
    m = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l); m[r["idx"]] = r
    return m
def cap_file(cap, ds):
    return (os.path.join(FULLRES, f"ckpt_{ds}_nothink_norag.jsonl") if cap == "fullres"
            else os.path.join(PRUNE, cap, f"ckpt_{ds}_nothink_norag.jsonl"))
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

# token_cache is keyed by GLOBAL dataset idx == checkpoint idx (PMC 0-1999, PathVQA 2000-5361, ...)
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]
def P_at(ds, cap, idx):  # prompt tokens for sample idx at cap (token_cache [P, vis])
    return TC[ds][cap][str(idx)][0]

def build(ds):
    caps = {c: load_jsonl(cap_file(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    rows = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        if any(str(i) not in TC[ds][c] for c in CAPS): continue
        p320 = caps["cap320"][i]["pred"]
        rows.append(dict(
            ds=ds, idx=i,
            margin=margin(caps["cap320"][i]),
            ok320=caps["cap320"][i]["ok"], ok32=strong[i]["ok"],
            G7={c: (caps[c][i].get("gen_tokens") or 2) for c in CAPS},
            G32=(strong[i].get("gen_tokens") or 0),
            P={c: P_at(ds, c, i) for c in CAPS},
            agree={c: int(caps[c][i]["pred"] == p320) for c in CAPS if c != "cap320"}))
    return rows

REC = {ds: build(ds) for ds in COMP4}

def run7(r, cap): return 2 * N7 * (r["P"][cap] + r["G7"][cap])
def run32(r):     return 2 * N32 * (r["P"]["fullres"] + r["G32"])   # 32B think at fullres

def evaluate(rows, extra=None):
    """extra=None -> deployed margin gate. extra=[caps] -> param-free stability rescue over those caps."""
    esc = np.zeros(len(rows), bool); flops = np.zeros(len(rows)); ok = np.zeros(len(rows))
    always32 = np.array([run32(r) for r in rows])
    # DEPLOYED escalation decision (logistic gate on margin, threshold proba at tau) — vectorized
    mg = np.array([[r["margin"]] for r in rows], dtype=np.float32)
    esc_dep = GATE.predict_proba(mg)[:, 1] < TAU
    for j, r in enumerate(rows):
        c = run7(r, "cap320")                      # cheap leg always paid
        low = bool(esc_dep[j])
        if extra is not None and low:
            c += sum(run7(r, cap) for cap in extra)  # extra cheap passes (only on eligible)
            stable = all(r["agree"][cap] for cap in extra)
        else:
            stable = False
        do_esc = low and (extra is None or not stable)
        if do_esc: c += run32(r)
        esc[j] = do_esc; flops[j] = c; ok[j] = (r["ok32"] if do_esc else r["ok320"])
    return dict(esc=esc.mean(), acc=ok.mean(), backbone=flops.sum() / always32.sum(),
                cheap_acc=np.mean([r["ok320"] for r in rows]))

def main():
    allrows = [r for ds in COMP4 for r in REC[ds]]
    print("=" * 96)
    print("VISUAL-STABILITY RESCUE — prefill-inclusive cost + accuracy + guardrail (competent-4)")
    print("=" * 96)
    dep = evaluate(allrows, None)
    allstrong = 1.0  # backbone baseline is always-32B by construction
    print(f"\nPOOLED competent-4 (n={len(allrows)}):  always-cheap acc={dep['cheap_acc']:.4f}")
    print(f"  {'method':<26}{'32B-call%':>11}{'backbone%':>11}{'acc':>9}{'Δacc':>9}")
    print(f"  {'DEPLOYED margin gate':<26}{dep['esc']*100:>10.1f}%{dep['backbone']*100:>10.1f}%{dep['acc']:>9.4f}{0.0:>+9.4f}")
    summary = {"deployed": dep}
    for name, extra in EXTRA_SETS.items():
        e = evaluate(allrows, extra)
        print(f"  {('RESCUE '+name):<26}{e['esc']*100:>10.1f}%{e['backbone']*100:>10.1f}%{e['acc']:>9.4f}{e['acc']-dep['acc']:>+9.4f}")
        summary[name] = e

    # per-benchmark guardrail for the headline 3-cap variant
    extra = EXTRA_SETS["3-cap (cap80,160,640)"]
    print(f"\nPER-BENCHMARK (headline = RESCUE 3-cap):   guardrail = acc >= always-cheap on every set")
    print(f"  {'benchmark':<12}{'cheap':>8}{'dep esc%':>10}{'dep acc':>9}  |  {'resc esc%':>10}{'resc acc':>9}{'resc bb%':>10}{'guard':>7}")
    for ds in COMP4:
        d = evaluate(REC[ds], None); rsc = evaluate(REC[ds], extra)
        guard = "OK" if rsc["acc"] >= d["cheap_acc"] - 1e-9 else "FAIL"
        print(f"  {ds:<12}{d['cheap_acc']:>8.3f}{d['esc']*100:>9.1f}%{d['acc']:>9.4f}  |  "
              f"{rsc['esc']*100:>9.1f}%{rsc['acc']:>9.4f}{rsc['backbone']*100:>9.1f}%{guard:>7}")

    # ---- iso-accuracy frontier: is the rescue point BELOW the plain-margin frontier? ----
    print("\nISO-ACCURACY FRONTIER (sweep gate tau; metric of record = min backbone at parity).")
    parity = 0.6451  # always-strong on competent-4
    taus = np.linspace(0.28, 0.52, 49)
    mg = np.array([[r["margin"]] for r in allrows], dtype=np.float32)
    proba = GATE.predict_proba(mg)[:, 1]
    always32 = np.array([run32(r) for r in allrows]); base = always32.sum()
    run7_320 = np.array([run7(r, "cap320") for r in allrows])
    ok320 = np.array([r["ok320"] for r in allrows]); ok32 = np.array([r["ok32"] for r in allrows])
    extra3 = EXTRA_SETS["3-cap (cap80,160,640)"]
    stable3 = np.array([all(r["agree"][c] for c in extra3) for r in allrows])
    extra_cost = np.array([sum(run7(r, c) for c in extra3) for r in allrows])
    def frontier(rescue):
        pts = []
        for t in taus:
            elig = proba < t
            esc = elig & (~stable3) if rescue else elig
            flops = run7_320.sum() + (extra_cost[elig].sum() if rescue else 0.0) + always32[esc].sum()
            acc = np.where(esc, ok32, ok320).mean()
            pts.append((float(flops / base), float(acc), float(esc.mean())))
        return pts
    def min_bb_at(pts, target):
        ok = [(bb, a, e) for (bb, a, e) in pts if a >= target - 1e-9]
        return min(ok, default=(None, None, None), key=lambda x: x[0])
    fm, fr = frontier(False), frontier(True)
    for tag, target in [("always-strong parity (0.6451)", parity), ("rescue acc (0.6448)", 0.6448)]:
        bm, am, em = min_bb_at(fm, target); br, ar, er = min_bb_at(fr, target)
        sm = f"backbone={bm*100:.1f}% (esc {em*100:.0f}%)" if bm else "unreachable"
        sr = f"backbone={br*100:.1f}% (esc {er*100:.0f}%)" if br else "unreachable"
        print(f"  @ {tag:<32} margin: {sm:<28} rescue-3cap: {sr}")
    summary["frontier_margin"] = fm; summary["frontier_rescue3"] = fr

    os.makedirs("results/cascade_methods", exist_ok=True)
    out = {k: ({kk: (float(vv) if not isinstance(vv, np.ndarray) else None) for kk, vv in v.items()}
               if isinstance(v, dict) else v)
           for k, v in summary.items()}
    json.dump(out, open("results/cascade_methods/stability_rescue_cost.json", "w"), indent=1)
    print("\n-> results/cascade_methods/stability_rescue_cost.json")
    # headline deltas
    h = summary["3-cap (cap80,160,640)"]
    print(f"\nHEADLINE (3-cap, parameter-free, frozen tau): "
          f"32B-call {dep['esc']*100:.0f}%->{h['esc']*100:.0f}%, "
          f"backbone {dep['backbone']*100:.0f}%->{h['backbone']*100:.0f}% of always-32B, "
          f"acc {dep['acc']:.4f}->{h['acc']:.4f} (Δ{h['acc']-dep['acc']:+.4f}).")

if __name__ == "__main__":
    main()
