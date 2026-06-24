#!/usr/bin/env python3
"""
combined_rescue.py - do MULTIPLE orthogonal training-free robustness signals compound for the
deferral RESCUE? A low-margin sample is "settled" (keep cheap) if its 7B-nt answer is robust to:
  - RESOLUTION  : invariant across cap ladder           (extra cheap nt passes)
  - REASONING   : 7B-nt pred == 7B-think pred            (needs a think pass -- EXPENSIVE)
  - SELF-VERIFY : P(yes) the answer is correct is high   (1 cheap verify pass)
We test SIGNAL strength on the escalate-set first (cost only matters for winners). Offline.

For each robustness criterion we report: how many low-margin samples it keeps cheap, the cheap-acc
of the kept set (higher=safer), the 32B-fix-rate on the kept-wrong (lower=safer), and the resulting
pooled (32B-call%, acc) vs the resolution-only rescue and the deployed gate.
"""
import os, json, glob, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
THINK = "ckpts/gate_7b_think"; VERIFY = "ckpts/gate_7b_verify"
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]
EXTRA = ["cap80", "cap160", "cap640"]

def load_jsonl(p):
    m = {}
    for l in open(p):
        if l.strip(): r = json.loads(l); m[r["idx"]] = r
    return m
def load_sharded(ds):  # merge think shards by idx
    m = {}
    for f in glob.glob(os.path.join(THINK, f"ckpt_{ds}_think_norag_s*.jsonl")):
        for l in open(f):
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
    think = load_sharded(ds); verify = load_jsonl(os.path.join(VERIFY, f"ckpt_{ds}_verify.jsonl"))
    rows = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        if i not in think or i not in verify: continue
        p320 = caps["cap320"][i]["pred"]
        rows.append(dict(ds=ds, idx=i, margin=margin(caps["cap320"][i]),
                         ok320=caps["cap320"][i]["ok"], ok32=strong[i]["ok"],
                         res_stable=all(caps[c][i]["pred"] == p320 for c in EXTRA),
                         reason_stable=(think[i]["pred"] == p320),
                         pyes=verify[i].get("p_yes_norm")))
    return rows

ROWS = [r for ds in COMP4 for r in build(ds)]
print(f"aligned competent-4 with think+verify: n={len(ROWS)}")
mg = np.array([[r["margin"]] for r in ROWS], dtype=np.float32)
ESC = GATE.predict_proba(mg)[:, 1] < TAU
OK320 = np.array([r["ok320"] for r in ROWS]); OK32 = np.array([r["ok32"] for r in ROWS])
RES = np.array([r["res_stable"] for r in ROWS]); REA = np.array([r["reason_stable"] for r in ROWS])
PYES = np.array([(r["pyes"] if r["pyes"] is not None else 0.0) for r in ROWS])

def report(keep_cheap, name):
    """keep_cheap: bool mask of which ESCALATE-eligible samples to KEEP cheap instead."""
    esc = ESC & (~keep_cheap)
    acc = np.where(esc, OK32, OK320).mean()
    rescued = ESC & keep_cheap
    cheap_acc = OK320[rescued].mean() if rescued.any() else float("nan")
    fix = OK32[rescued & (OK320 == 0)].mean() if (rescued & (OK320 == 0)).any() else float("nan")
    print(f"  {name:<42} call%={esc.mean()*100:5.1f}  acc={acc:.4f}  "
          f"rescued={rescued.sum():4d} (cheap-acc={cheap_acc:.3f}, 32B-fix={fix:.3f})")
    return esc.mean(), acc

def main():
    print(f"deployed gate: call%={ESC.mean()*100:.1f}  acc={np.where(ESC,OK32,OK320).mean():.4f}")
    print(f"always-strong parity = {OK32.mean():.4f}\n")
    # pick a P(yes) keep-threshold by a fixed quantile of escalate-set (parameter-light)
    pq = np.quantile(PYES[ESC], 0.5)  # keep the upper-half-confident among escalate set
    print(f"signal robustness criteria (keep low-margin sample cheap if ...), P(yes) thr={pq:.3f}:")
    report(RES, "RESOLUTION-stable (our rescue)")
    report(REA, "REASONING-stable (nt==think)  [think cost]")
    report(PYES >= pq, "SELF-VERIFY P(yes)>=median")
    print("  -- unions (keep cheap if ANY robustness holds; rescues MORE, risks acc) --")
    report(RES | REA, "RESOLUTION or REASONING")
    report(RES | (PYES >= pq), "RESOLUTION or SELF-VERIFY")
    report(RES | REA | (PYES >= pq), "RESOLUTION or REASONING or VERIFY")
    print("  -- intersections (keep cheap only if BOTH; rescues FEWER, safer) --")
    report(RES & REA, "RESOLUTION and REASONING")
    report(RES & (PYES >= pq), "RESOLUTION and SELF-VERIFY")
    # gated combination: among resolution-stable, also require reasoning-stable to keep (precision)
    print("\nINTERPRETATION: a union rescues more (lower call%) but only helps if its rescued set keeps")
    print("high cheap-acc and low 32B-fix. Compare acc to always-strong parity and to RESOLUTION-only.")

if __name__ == "__main__":
    main()
