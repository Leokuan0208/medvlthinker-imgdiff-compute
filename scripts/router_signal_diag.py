#!/usr/bin/env python3
"""
router_signal_diag.py - the mechanism proof: WHY no margin threshold beats parity.

A cascade's accuracy gain over always-32B comes ONLY from the questions it KEEPS on
the 7B:  gain = (1/N) * sum over kept of (a7 - a32). Per question a7-a32 is
  +1  "break"   : 7B right, 32B wrong  -> should KEEP   (escalating breaks it)
  -1  "fixable" : 7B wrong, 32B right  -> should ESCALATE (keeping wastes the fix)
   0  both right / both wrong          -> accuracy-neutral (cost only)
A margin threshold raises accuracy IFF margin ranks BREAKS high (kept) and FIXABLE
low (escalated). This measures that separation, per dataset:
  - 2x2 cell rates,
  - where each cell sits in the margin distribution (median percentile),
  - AUROC(margin -> 7B-correct)            [the cost signal],
  - AUROC(margin -> break vs fixable)      [THE crux; ~0.5 => margin can't separate them],
  - BEST accuracy gain achievable by ANY margin threshold (label-oracle: max prefix sum
    of (a7-a32) ordered by descending margin) vs a SHUFFLE floor (same statistic, margin
    permuted). real ~ shuffle => the signal, not the method, sets the floor.
CPU-only; reads existing 7B+32B eval labels.
"""
import json, glob, os, re, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score

EVAL = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]; B_SHUF = 2000
rng = np.random.RandomState(42)

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{cell}_s\dof\d\.jsonl$"); d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip(): r = json.loads(l); d[m.group(1)][r["idx"]] = r
    return d
def margin(row):
    lp = row.get("opt_logprobs") or {}; v = sorted(lp.values(), reverse=True)
    return (v[0]-v[1]) if len(v) >= 2 else 0.0
def shuffle_floor(delta, B):
    n = len(delta); out = np.empty(B)
    for b in range(B):
        out[b] = np.cumsum(delta[rng.permutation(n)]).max()
    return out

def run(r7, r32):
    print("=" * 104)
    print("SIGNAL DIAGNOSTIC  -  can the margin separate 32B-FIXABLE (escalate) from BREAKS (keep)?")
    print("=" * 104)
    for name in EVAL:
        if name not in r7 or name not in r32: continue
        idx = sorted(set(r7[name]) & set(r32[name]))
        a7  = np.array([r7[name][i]["ok"] for i in idx]).astype(int)
        a32 = np.array([r32[name][i]["ok"] for i in idx]).astype(int)
        mg  = np.array([margin(r7[name][i]) for i in idx]); N = len(idx)
        pct = (np.argsort(np.argsort(mg)) + 0.5) / N
        fix = (a7 == 0) & (a32 == 1); brk = (a7 == 1) & (a32 == 0)
        bR  = (a7 == 1) & (a32 == 1); bW = (a7 == 0) & (a32 == 0)
        print(f"\n### {name}  (N={N})  a7={a7.mean():.3f} a32={a32.mean():.3f}")
        print(f"    cells:  both_right {bR.mean():.3f}   fixable(7Bx,32Bok) {fix.mean():.3f}   "
              f"break(7Bok,32Bx) {brk.mean():.3f}   both_wrong {bW.mean():.3f}")
        loc = lambda m: (f"med_margin={np.median(mg[m]):+.2f} med_pctile={np.median(pct[m])*100:.0f}%"
                         if m.sum() else "n/a")
        print(f"    margin loc:  fixable[{int(fix.sum())}] {loc(fix)}  |  "
              f"break[{int(brk.sum())}] {loc(brk)}  |  both_right {loc(bR)}")
        try: auc_c = roc_auc_score(a7, mg)
        except Exception: auc_c = float("nan")
        dis = fix | brk
        auc_bf = (roc_auc_score(brk[dis].astype(int), mg[dis])
                  if dis.sum() > 0 and 0 < brk[dis].sum() < dis.sum() else float("nan"))
        print(f"    AUROC(margin->7B correct)       = {auc_c:.3f}   [cost signal]")
        print(f"    AUROC(margin->break vs fixable) = {auc_bf:.3f}   [CRUX: ~0.5 => can't tell a 32B-fix from a 32B-break]")
        delta = (a7 - a32).astype(float); order = np.argsort(-mg)
        cs = np.cumsum(delta[order]); k = int(np.argmax(cs)); best = cs[k]
        floor = shuffle_floor(delta, B_SHUF); f95 = np.percentile(floor, 95)
        verdict = "SIGNAL (real > 95th shuffle)" if best > f95 else "NO SEPARATION (real <= 95th shuffle)"
        print(f"    BEST margin-threshold gain (oracle) = {best/N:+.3f}  at keep_top={k+1} (esc {100*(1-(k+1)/N):.0f}%)")
        print(f"    shuffle floor: mean {floor.mean()/N:+.3f}, 95th {f95/N:+.3f}   -> {verdict}")
    print("\n" + "=" * 104)
    print("READ: fixable and break at similar margin (AUROC_bf ~0.5) AND best oracle gain within the")
    print("shuffle floor => NO margin threshold beats always-32B on accuracy. The confidence SIGNAL,")
    print("not the routing method, sets the floor; the margin gate already realizes the parity it")
    print("allows. That is the deployable frontier and the paper's core mechanism result.")

if __name__ == "__main__":
    run(load_arm("ckpts/gate_7b_vllm", "nothink_norag"), load_arm("ckpts/gate_32b", "think_norag"))
