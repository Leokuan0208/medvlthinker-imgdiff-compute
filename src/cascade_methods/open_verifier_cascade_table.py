#!/usr/bin/env python3
"""Compute the verifier-augmented cascade table for one family from saved artifacts (no inference):
  verifier best-of-8 (argmax trained-verifier P(Yes)) + verifier-confidence gate -> escalate to STRONG.
Reads:  <adapter_dir>/transfer_dump_{ds}_{TAG}.json  (per-q: sl[8], scores[8], pick, greedy_ok)
        <strong_dir>/ckpt_{ds}_{STRONG_TAG}_t0.judge.jsonl  (per-idx judge_ok of the strong model)
Reports per dataset + pooled: SC/greedy, verifier-bo8, STRONG, oracle@8, and the frontier-best cascade
accuracy with its escalation %. The cascade gate threshold is swept (frontier-best, oracle-tau) AND a
fixed transfer tau (median of cheap-correct scores) is reported for honesty.
Usage: open_cascade_table.py --adapter_dir ckpts/train/lora_verifier_pooled4 --tag iv3_8b \
         --strong_dir ckpts/openvqa/internvl3_38b --strong_tag iv3_38b \
         --datasets vqa_rad_open pathvqa_open kvasir_open radimagenet_open
"""
import argparse, os, json
import numpy as np
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--adapter_dir", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--strong_dir", required=True)
ap.add_argument("--strong_tag", required=True)
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--heldout", default="radimagenet_open", help="dataset name to mark as held-out OOD")
A = ap.parse_args()

def load_judge(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def frontier_best(g, strong, gate):
    """g=verifier-pick ok (array), strong=strong ok (array), gate=signal (array, higher=more confident).
    Escalate when gate < tau. Sweep tau; return best cascade acc + esc% at that tau."""
    g = np.array(g, float); strong = np.array(strong, float); gate = np.array(gate, float)
    cand = sorted(set(gate.tolist()))
    best = (-1, 0.0, None)  # acc, esc, tau
    # tau below min => escalate none (pure verifier); tau above max => escalate all (pure strong)
    for tau in [cand[0]-1e-9] + cand + [cand[-1]+1e-9]:
        esc = gate < tau
        acc = np.where(esc, strong, g).mean()
        if acc > best[0]:
            best = (acc, esc.mean(), tau)
    return best

def at_tau(g, strong, gate, tau):
    g = np.array(g, float); strong = np.array(strong, float); gate = np.array(gate, float)
    esc = gate < tau
    return np.where(esc, strong, g).mean(), esc.mean()

rows = []
pool = {"g":[], "v":[], "s":[], "o":[], "gate":[]}
for ds in A.datasets:
    dp = os.path.join(ROOT, A.adapter_dir, f"transfer_dump_{ds}_{A.tag}.json")
    if not os.path.exists(dp):
        print(f"  {ds:<16} MISSING dump {dp}"); continue
    dump = json.load(open(dp))
    sj = {}
    for cand in (f"ckpt_{ds}_{A.strong_tag}_t0.judge.jsonl", f"ckpt_{ds}_{A.strong_tag}.judge.jsonl"):
        sj = load_judge(os.path.join(ROOT, A.strong_dir, cand))
        if sj: break
    g=[]; v=[]; s=[]; o=[]; gate=[]
    n_no_strong = 0
    for r in dump:
        idx = r["idx"]
        if idx not in sj:           # need the strong label to score escalation fairly
            n_no_strong += 1; continue
        sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
        v.append(sl[r["pick"]])
        g.append(int(r["greedy_ok"]))
        o.append(max(sl))
        s.append(sj[idx])
        gate.append(max(r["scores"]))
    n = len(v)
    if n == 0:
        print(f"  {ds:<16} no overlap with strong judge"); continue
    acc_b, esc_b, tau_b = frontier_best(v, s, gate)
    tag = "  (HELD-OUT OOD)" if ds == A.heldout else ""
    print(f"  {ds:<16} n={n:>4} SC/greedy={np.mean(g):.3f}  verifier-bo8={np.mean(v):.3f}  "
          f"STRONG={np.mean(s):.3f}  oracle@8={np.mean(o):.3f}  || cascade-best={acc_b:.3f} @esc={esc_b*100:.0f}%"
          f"  {'BEATS-STRONG' if acc_b > np.mean(s)+1e-9 else ''}{tag}")
    if n_no_strong: print(f"      ({n_no_strong} q dropped: no strong judge label)")
    for k,arr in zip(("g","v","s","o","gate"),(g,v,s,o,gate)): pool[k]+=list(arr)
    rows.append((ds, n, np.mean(g), np.mean(v), np.mean(s), np.mean(o), acc_b, esc_b))

if pool["v"]:
    g,v,s,o,gate = pool["g"],pool["v"],pool["s"],pool["o"],pool["gate"]
    acc_b, esc_b, _ = frontier_best(v, s, gate)
    print(f"  {'POOLED':<16} n={len(v):>4} SC/greedy={np.mean(g):.3f}  verifier-bo8={np.mean(v):.3f}  "
          f"STRONG={np.mean(s):.3f}  oracle@8={np.mean(o):.3f}  || cascade-best={acc_b:.3f} @esc={esc_b*100:.0f}%"
          f"  {'BEATS-STRONG' if acc_b > np.mean(s)+1e-9 else ''}")
