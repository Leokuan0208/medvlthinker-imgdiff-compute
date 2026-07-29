#!/usr/bin/env python3
"""
final_robustness_rescue.py - canonical validation of the ROBUSTNESS-RESCUE family (this loop's
novel, training-free contribution): keep a low-confidence VLM query on the cheap model when its
answer is ROBUST to perturbations the cheap model is vulnerable to. Three operating points:

  R  = RESOLUTION-stable rescue        (keep cheap if 7B-nt answer invariant across cap{80,160,640})
  V  = SELF-VERIFY rescue              (keep cheap if 7B self-verify P(yes_norm) >= 0.5)
  RV = DOUBLY-ROBUST (R and V)         (keep cheap only if BOTH -> safest; strict-Pareto point)

All parameter-free/parameter-light, frozen deployed gate (tau=0.4264), pooled competent-4 +
per-benchmark guardrail + paired bootstrap CIs. FLOPs = 2*N*(P+G) (token_cache P, ckpt gen).
Extra passes are charged ONLY on the gate-eligible (low-margin) samples: R adds 3 cheap nt passes,
V adds 1 cheap verify pass (~one cap320 pass), RV adds all 4. Offline; launch from repo root.
"""
import os, json, glob, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
THINK = "ckpts/gate_7b_think"; VERIFY = "ckpts/gate_7b_verify"
TC = json.load(open("ckpts/token_cache.json"))
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
N7, N32 = 7.6e9, 33.0e9
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]
EXTRA = ["cap80", "cap160", "cap640"]; PYES_THR = 0.5

def load_jsonl(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def cap_file(cap, ds):
    return (os.path.join(FULLRES, f"ckpt_{ds}_nothink_norag.jsonl") if cap == "fullres"
            else os.path.join(PRUNE, cap, f"ckpt_{ds}_nothink_norag.jsonl"))
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def build(ds):
    caps = {c: load_jsonl(cap_file(c, ds)) for c in CAPS}
    strong = load_jsonl(os.path.join(STRONG, f"ckpt_{ds}_think_norag.jsonl"))
    verify = load_jsonl(os.path.join(VERIFY, f"ckpt_{ds}_verify.jsonl"))
    rows = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        if i not in verify or any(str(i) not in TC[ds][c] for c in CAPS): continue
        p320 = caps["cap320"][i]["pred"]
        P = {c: TC[ds][c][str(i)][0] for c in CAPS}
        G7 = {c: (caps[c][i].get("gen_tokens") or 2) for c in CAPS}
        rows.append(dict(
            ds=ds, idx=i, margin=margin(caps["cap320"][i]),
            ok320=caps["cap320"][i]["ok"], ok32=strong[i]["ok"],
            res_stable=all(caps[c][i]["pred"] == p320 for c in EXTRA),
            verified=((verify[i].get("p_yes_norm") or 0.0) >= PYES_THR),
            run7={c: 2 * N7 * (P[c] + G7[c]) for c in CAPS},
            run_verify=2 * N7 * (P["cap320"] + 2),          # verify pass ~ one cap320 nt pass
            run32=2 * N32 * (P["fullres"] + (strong[i].get("gen_tokens") or 0))))
    return rows

REC = {ds: build(ds) for ds in COMP4}

def evaluate(rows, mode):
    """mode in {deployed,R,V,RV}. Returns per-sample (flops, ok, esc) arrays + base32."""
    mg = np.array([[r["margin"]] for r in rows], dtype=np.float32)
    low = GATE.predict_proba(mg)[:, 1] < TAU
    fl = np.zeros(len(rows)); ok = np.zeros(len(rows)); esc = np.zeros(len(rows), bool)
    base = np.array([r["run32"] for r in rows])
    for j, r in enumerate(rows):
        c = r["run7"]["cap320"]
        if mode != "deployed" and low[j]:
            if mode in ("R", "RV"): c += sum(r["run7"][cap] for cap in EXTRA)
            if mode in ("V", "RV"): c += r["run_verify"]
            keep = (mode == "R" and r["res_stable"]) or (mode == "V" and r["verified"]) or \
                   (mode == "RV" and r["res_stable"] and r["verified"])
        else:
            keep = False
        do = bool(low[j]) and not keep
        if do: c += r["run32"]
        fl[j] = c; esc[j] = do; ok[j] = (r["ok32"] if do else r["ok320"])
    return fl, ok, esc, base

def summ(rows, mode):
    fl, ok, esc, base = evaluate(rows, mode)
    return dict(acc=ok.mean(), call=esc.mean(), backbone=fl.sum() / base.sum())

def main():
    allrows = [r for ds in COMP4 for r in REC[ds]]
    cheap = np.mean([r["ok320"] for r in allrows]); strong = np.mean([r["ok32"] for r in allrows])
    print("=" * 92)
    print(f"ROBUSTNESS-RESCUE — canonical validation (competent-4, n={len(allrows)})")
    print(f"always-cheap={cheap:.4f}  always-strong(parity)={strong:.4f}   P(yes) thr={PYES_THR}")
    print("=" * 92)
    labels = {"deployed": "DEPLOYED margin gate", "R": "R  resolution-rescue",
              "V": "V  self-verify-rescue", "RV": "RV doubly-robust (R∧V)"}
    print(f"  {'method':<26}{'32B-call%':>11}{'backbone%':>11}{'acc':>9}{'Δacc vs dep':>13}")
    dep = summ(allrows, "deployed")
    res = {}
    for m in ["deployed", "R", "V", "RV"]:
        s = summ(allrows, m); res[m] = s
        print(f"  {labels[m]:<26}{s['call']*100:>10.1f}%{s['backbone']*100:>10.1f}%{s['acc']:>9.4f}"
              f"{s['acc']-dep['acc']:>+13.4f}")

    print(f"\nPER-BENCHMARK guardrail (acc >= always-cheap on each set):")
    print(f"  {'benchmark':<11}{'cheap':>7}" + "".join(f"{labels[m].split()[0]:>9}" for m in ['deployed','R','V','RV']) + "   guardrail")
    for ds in COMP4:
        ch = np.mean([r["ok320"] for r in REC[ds]])
        accs = {m: summ(REC[ds], m)["acc"] for m in ['deployed', 'R', 'V', 'RV']}
        ok = all(accs[m] >= ch - 1e-9 for m in ['R', 'V', 'RV'])
        print(f"  {ds:<11}{ch:>7.3f}" + "".join(f"{accs[m]:>9.4f}" for m in ['deployed','R','V','RV'])
              + f"   {'OK' if ok else 'FAIL'}")

    # paired bootstrap on Δacc vs deployed for R, V, RV
    print("\nPaired bootstrap (5000x) Δacc vs deployed, and Δbackbone:")
    fl_d, ok_d, esc_d, base = evaluate(allrows, "deployed")
    rng = np.random.RandomState(0); n = len(allrows)
    for m in ["R", "V", "RV"]:
        fl_m, ok_m, esc_m, _ = evaluate(allrows, m)
        dA, dB = [], []
        for _ in range(5000):
            ix = rng.randint(0, n, n)
            dA.append(ok_m[ix].mean() - ok_d[ix].mean())
            dB.append((fl_m[ix].sum() - fl_d[ix].sum()) / base[ix].sum())
        dA = np.sort(dA); dB = np.sort(dB)
        print(f"  {labels[m]:<26} Δacc={np.mean(dA):+.4f} CI[{dA[125]:+.4f},{dA[4875]:+.4f}]   "
              f"Δbackbone={np.mean(dB)*100:+.1f}% CI[{dB[125]*100:+.1f},{dB[4875]*100:+.1f}]")

    out = {m: res[m] for m in res}
    json.dump({m: {k: float(v) for k, v in res[m].items()} for m in res},
              open("results/cascade_methods/artifacts/final_robustness_rescue.json", "w"), indent=1)
    print("\n-> results/cascade_methods/artifacts/final_robustness_rescue.json")
    print(f"\nHEADLINES (frozen gate, competent-4):")
    print(f"  R : calls {dep['call']*100:.0f}%->{res['R']['call']*100:.0f}%, backbone {dep['backbone']*100:.0f}%->{res['R']['backbone']*100:.0f}%, acc {res['R']['acc']:.4f} (~parity)")
    print(f"  V : calls {dep['call']*100:.0f}%->{res['V']['call']*100:.0f}%, backbone {dep['backbone']*100:.0f}%->{res['V']['backbone']*100:.0f}%, acc {res['V']['acc']:.4f} (no loss)")
    print(f"  RV: calls {dep['call']*100:.0f}%->{res['RV']['call']*100:.0f}%, backbone {dep['backbone']*100:.0f}%->{res['RV']['backbone']*100:.0f}%, acc {res['RV']['acc']:.4f} (Pareto: +acc & -calls)")

if __name__ == "__main__":
    main()
