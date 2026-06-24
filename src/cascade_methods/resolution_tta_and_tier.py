#!/usr/bin/env python3
"""
resolution_tta_and_tier.py - two MORE training-free ideas from the multi-resolution 7B-nt data:

(1) MULTI-RESOLUTION VOTE (test-time augmentation as the cheap ANSWER): instead of trusting
    cap320 alone, take the majority answer across resolution caps. If the cheap leg gets more
    accurate for free, the whole cascade improves (fewer escalations needed, higher floor).

(2) RESOLUTION TIER (same-model intermediate tier): for low-margin cap320 samples, try the 7B at
    a HIGHER resolution (cap640/fullres) BEFORE escalating to the 32B. If more pixels on the same
    cheap model fixes many of them, we avoid the expensive 32B call. Premise test: among the
    samples the gate would escalate, how often does 7B@fullres already produce the right answer
    (and how does that compare to the 32B)?

Offline from existing ckpts. Pooled competent-4 + per-benchmark.
"""
import os, json, pickle
import numpy as np

PRUNE = "ckpts/gate_7b_prune"; FULLRES = "ckpts/gate_7b_vllm"; STRONG = "ckpts/gate_32b"
_R = pickle.load(open("ckpts/router_margin.pkl", "rb")); GATE, TAU = _R["gate"], _R["tau"]
COMP4 = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
CAPS = ["cap80", "cap160", "cap320", "cap640", "fullres"]

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
    rows = []
    for i in sorted(caps["cap320"]):
        if i not in strong or any(i not in caps[c] for c in CAPS): continue
        rows.append(dict(ds=ds, idx=i, gold=caps["cap320"][i]["gold"],
                         preds={c: caps[c][i]["pred"] for c in CAPS},
                         margin=margin(caps["cap320"][i]),
                         ok={c: caps[c][i]["ok"] for c in CAPS},
                         ok32=strong[i]["ok"]))
    return rows

REC = {ds: build(ds) for ds in COMP4}
def pool(names): return [r for ds in names for r in REC[ds]]

def vote_pred(r, capset):
    from collections import Counter
    c = Counter(r["preds"][k] for k in capset)
    top = c.most_common()
    # tie-break toward cap320's answer if present among the top
    best = top[0][1]
    cands = [k for k, v in top if v == best]
    return r["preds"]["cap320"] if r["preds"]["cap320"] in cands else cands[0]

def main():
    print("=" * 86)
    print("(1) MULTI-RESOLUTION VOTE — does TTA over caps beat cap320 as the cheap answer?")
    print("=" * 86)
    votesets = {"cap320 (baseline)": ["cap320"],
                "{160,320,640}": ["cap160", "cap320", "cap640"],
                "{80,160,320,640}": ["cap80", "cap160", "cap320", "cap640"],
                "all 5 caps": CAPS}
    print(f"  {'benchmark':<12}" + "".join(f"{k:>20}" for k in votesets))
    for ds in COMP4 + ["POOLED"]:
        rows = pool(COMP4) if ds == "POOLED" else REC[ds]
        accs = []
        for k, cs in votesets.items():
            acc = np.mean([int(vote_pred(r, cs) == r["gold"]) for r in rows])
            accs.append(acc)
        print(f"  {ds:<12}" + "".join(f"{a:>20.4f}" for a in accs))

    print("\n" + "=" * 86)
    print("(2) RESOLUTION TIER — for gate-ESCALATE samples, does 7B@higher-res fix them cheaply?")
    print("=" * 86)
    rows = pool(COMP4)
    mg = np.array([[r["margin"]] for r in rows], dtype=np.float32)
    esc = GATE.predict_proba(mg)[:, 1] < TAU
    elig = [r for r, e in zip(rows, esc) if e]
    print(f"  gate-escalate set: n={len(elig)} ({esc.mean()*100:.0f}% of {len(rows)})")
    for cap in ["cap320", "cap640", "fullres"]:
        a = np.mean([r["ok"][cap] for r in elig])
        print(f"    7B@{cap:<8} acc on escalate-set = {a:.4f}")
    a32 = np.mean([r["ok32"] for r in elig])
    print(f"    32B-think  acc on escalate-set = {a32:.4f}  <- what escalation currently buys")
    # of the cap320-WRONG escalate samples, how many does 7B@fullres fix vs 32B fix?
    wrong = [r for r in elig if r["ok"]["cap320"] == 0]
    print(f"\n  among cap320-WRONG in escalate-set (n={len(wrong)}):")
    print(f"    7B@fullres fixes : {np.mean([r['ok']['fullres'] for r in wrong]):.4f}")
    print(f"    7B@cap640 fixes  : {np.mean([r['ok']['cap640'] for r in wrong]):.4f}")
    print(f"    32B-think fixes  : {np.mean([r['ok32'] for r in wrong]):.4f}")
    # how many would a fullres-confidence intermediate tier resolve without the 32B?
    print("\n  INTERPRETATION: a 7B@fullres intermediate tier only helps if its fix-rate approaches")
    print("  the 32B's AND it is much cheaper. Compare the two fix-rates above.")

if __name__ == "__main__":
    main()
