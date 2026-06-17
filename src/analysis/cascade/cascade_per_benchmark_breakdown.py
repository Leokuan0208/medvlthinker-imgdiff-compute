#!/usr/bin/env python3
"""
cascade_breakdown.py - per-benchmark cap320 cascade behavior from the VALIDATED vLLM labels.
The grid pools all six benchmarks; this breaks the chosen operating point (fullres-trained gate,
cap320 inference) out per benchmark, so we can see whether the cascade wins everywhere or only on
some (e.g. PMC has ~no 7B->32B headroom). Frozen gate applied to the cap320 7B margins vs the 32B
eval. CPU only; validated vLLM data; no HF, no GPU.
"""
import json, glob, os, re, pickle
import numpy as np
from collections import defaultdict

SIX = ["PMC-VQA","SLAKE","VQA-RAD","PathVQA","MMMU","MedXpert-Reasoning","MedXpert-Understanding"]
EVAL_320 = "ckpts/gate_7b_prune/cap320"
DIR_32B  = "ckpts/gate_32b"

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
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0

def run(repo):
    J = lambda p: os.path.join(repo, p)
    ev = load_arm(J(EVAL_320), "nothink_norag"); r32 = load_arm(J(DIR_32B), "think_norag")
    R = pickle.load(open(J("ckpts/router_margin.pkl"), "rb")); gate, tau = R["gate"], R["tau"]
    print(f"per-benchmark cap320 cascade  (validated vLLM labels, frozen gate tau={tau:.3f})")
    print("7B = no_think@cap320, 32B = think@full-res, casc = routed; resc/brok = rescued/broken\n")
    hdr = (f"{'benchmark':<24}{'n':>6}{'7B':>7}{'32B':>7}{'casc':>7}{'vs7B':>7}{'esc%':>6}"
           f"{'resc':>6}{'brok':>6}{'net':>6}")
    print(hdr); print("-"*len(hdr))
    P7=[]; P32=[]; PC=[]; PE=[]
    for ds in SIX:
        if ds not in ev or ds not in r32: continue
        idx = sorted(set(ev[ds]) & set(r32[ds]))
        if not idx: continue
        mg = np.array([margin(ev[ds][i]) for i in idx])
        ok7 = np.array([ev[ds][i]["ok"] for i in idx], float)
        ok32 = np.array([r32[ds][i]["ok"] for i in idx], float)
        esc = gate.predict_proba(mg.reshape(-1,1))[:,1] < tau
        casc = np.where(esc, ok32, ok7)
        b7 = ok7.astype(bool); b32 = ok32.astype(bool)
        resc = int(((~b7)&b32&esc).sum()); brok = int((b7&(~b32)&esc).sum())
        print(f"{ds:<24}{len(idx):>6}{ok7.mean():>7.3f}{ok32.mean():>7.3f}{casc.mean():>7.3f}"
              f"{casc.mean()-ok7.mean():>+7.3f}{100*esc.mean():>5.0f}%{resc:>6}{brok:>6}{resc-brok:>+6}")
        P7.append(ok7); P32.append(ok32); PC.append(casc); PE.append(esc)
    print("-"*len(hdr))
    a7=np.concatenate(P7); a32=np.concatenate(P32); ac=np.concatenate(PC); ae=np.concatenate(PE)
    b7=a7.astype(bool); b32=a32.astype(bool)
    resc=int(((~b7)&b32&ae).sum()); brok=int((b7&(~b32)&ae).sum())
    print(f"{'POOLED (all six)':<24}{len(a7):>6}{a7.mean():>7.3f}{a32.mean():>7.3f}{ac.mean():>7.3f}"
          f"{ac.mean()-a7.mean():>+7.3f}{100*ae.mean():>5.0f}%{resc:>6}{brok:>6}{resc-brok:>+6}")
    print(f"\nREAD: 'vs7B' = cascade - always-7B per benchmark. Positive => escalation helps there;")
    print(f"negative => 32B-think loses to 7B-no_think on that benchmark's escalated questions. The")
    print(f"pooled 'vs7B' is the honest aggregate the smoke's PMC-only number could not show.")

if __name__ == "__main__":
    run(os.path.expanduser("~/medvlthinker-imgdiff-compute"))
