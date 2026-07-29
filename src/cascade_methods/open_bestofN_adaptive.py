#!/usr/bin/env python3
"""
open_bestofN_adaptive.py - OFFLINE test of the compute/accuracy trade-off of "letting the model talk more"
in the OPEN-text verifier cascade, and of ADAPTIVE best-of-N (sample more only when the verifier is unsure).

Mechanism under test (format-agnostic, no router): cheap 7B samples answers; a trained verifier scores each;
we KEEP argmax P(correct). "Talking more" = drawing more samples N (more compute, maybe higher accuracy).
Question 1: how fast does verifier-best-of-N accuracy saturate with N? (the scaling curve)
Question 2: can an ADAPTIVE-N policy (draw sequentially, stop once the running-max verifier score >= tau_stop,
            escalate to strong if still unsure after N_max) match best-of-8 accuracy at a lower MEAN sample count?

Reads ONLY existing artifacts (transfer_dump: per-question scores[8]+sl[8]+pick; strong judge jsonl).
CPU only. Cost accounting: FLOPs = 2*N_backbone*(P+G), P=420 prefill-bound; a cheap sample ~= 1 cheap pass,
a strong call ~= 1 strong pass (N32/N7 = 4.34x). Reports acc AND mean cheap-samples AND FLOPs vs always-strong.
Launch from repo root:  python3 src/cascade_methods/open_bestofN_adaptive.py
"""
import os, json
import numpy as np
from collections import defaultdict

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
N7, N32, PFIX = 7.6e9, 33.0e9, 420.0
ADAPTER = "ckpts/train/lora_verifier_pooled4"
OPEN_DSETS = ["kvasir_open", "vqa_rad_open", "pathvqa_open", "radimagenet_open"]
FAM = {
    "Lingshu":     dict(vtag="lingshu7b", sdir="ckpts/openvqa/strong_lingshu",
                        sj=["ckpt_{ds}_lingshu32b_t0.judge.jsonl", "ckpt_{ds}_lingshu32b.judge.jsonl"]),
    "MedVLThinker":dict(vtag="7b", sdir="ckpts/openvqa/strong",
                        sj=["ckpt_{ds}_32b_t0.judge.jsonl", "ckpt_{ds}_32b.judge.jsonl"]),
    "InternVL3":   dict(vtag="iv3_8b", sdir="ckpts/openvqa/internvl3_38b",
                        sj=["ckpt_{ds}_iv3_38b_t0.judge.jsonl", "ckpt_{ds}_iv3_38b.judge.jsonl"]),
}

def load_judge(p):
    m = {}
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); m[r["idx"]] = int(r["judge_ok"])
    return m

def load_family(cfg):
    rows = []
    for ds in OPEN_DSETS:
        dp = J(os.path.join(ADAPTER, f"transfer_dump_{ds}_{cfg['vtag']}.json"))
        if not os.path.exists(dp): continue
        dump = json.load(open(dp))
        sj = {}
        for t in cfg["sj"]:
            sj = load_judge(J(os.path.join(cfg["sdir"], t.format(ds=ds))))
            if sj: break
        for r in dump:
            i = r["idx"]
            if i not in sj: continue
            sl = [0 if x is None or x == -1 else int(x) for x in r["sl"]]
            sc = list(r["scores"])
            n = min(len(sl), len(sc))
            if n < 1: continue
            rows.append(dict(ds=ds, scores=sc[:n], sl=sl[:n], strong=int(sj[i])))
    return rows

def verifier_pick_ok(scores, sl, k):
    """best-of-first-k: pick argmax verifier score among first k samples; return its judge_ok."""
    k = min(k, len(scores))
    j = int(np.argmax(scores[:k]))
    return sl[j]

def main():
    print("=" * 100)
    print("OPEN best-of-N scaling + ADAPTIVE-N (verifier). acc vs #cheap-samples vs FLOPs-vs-always-strong")
    print("=" * 100)
    cost_cheap = 2 * N7 * PFIX        # one cheap sample ~ one cheap prefill (prefill-bound)
    cost_strong = 2 * N32 * PFIX
    summary = {}
    for name, cfg in FAM.items():
        rows = load_family(cfg)
        if not rows:
            print(f"\n### {name}: no data"); continue
        strong = np.mean([r["strong"] for r in rows])
        n = len(rows)
        print(f"\n############  {name}  (n={n})  always-strong acc={strong:.3f}  ############")

        # ---- (1) fixed best-of-N scaling curve ----
        print("  fixed best-of-N (verifier pick over first N):")
        print(f"    {'N':>3}{'acc':>8}{'FLOPs vs strong':>18}")
        for k in [1, 2, 3, 4, 6, 8]:
            if k > min(len(r["scores"]) for r in rows): continue
            acc = np.mean([verifier_pick_ok(r["scores"], r["sl"], k) for r in rows])
            flops = (k * cost_cheap * n) / (cost_strong * n)     # k cheap passes vs 1 strong pass, per item
            print(f"    {k:>3}{acc:>8.3f}{flops:>17.2f}x")

        # ---- (2) adaptive-N: sequential draw, stop when running-max score >= tau_stop; else escalate ----
        Nmax = min(8, min(len(r["scores"]) for r in rows))
        # For a fair "sequential" sim, sort each question's samples in DRAW order (as stored).
        print(f"  ADAPTIVE-N (stop early when running verifier-max >= tau_stop; escalate to strong if <tau_esc after N={Nmax}):")
        print(f"    {'tau_stop':>9}{'tau_esc':>9}{'acc':>8}{'meanN':>7}{'esc%':>7}{'FLOPs vs strong':>17}")
        best_match = None
        for tau_stop in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            for tau_esc in [0.0, 0.2, 0.3, 0.4]:
                accs = []; used = []; escs = []
                for r in rows:
                    sc, sl = r["scores"], r["sl"]
                    kk = Nmax; stop_k = Nmax
                    for k in range(1, Nmax + 1):
                        if max(sc[:k]) >= tau_stop:
                            stop_k = k; break
                    runmax = max(sc[:stop_k])
                    if runmax < tau_esc:
                        accs.append(r["strong"]); used.append(stop_k); escs.append(1)
                    else:
                        j = int(np.argmax(sc[:stop_k])); accs.append(sl[j]); used.append(stop_k); escs.append(0)
                acc = np.mean(accs); meanN = np.mean(used); esc = np.mean(escs)
                flops = (meanN * cost_cheap * n + esc * cost_strong * n) / (cost_strong * n)
                # track the cheapest policy that matches best-of-8 accuracy (within 0.3pt)
                bo8 = np.mean([verifier_pick_ok(r["scores"], r["sl"], Nmax) for r in rows])
                if acc >= bo8 - 0.003 and (best_match is None or flops < best_match[-1]):
                    best_match = (tau_stop, tau_esc, acc, meanN, esc, flops)
                if tau_esc == 0.0:   # print the no-escalation sweep row for readability
                    print(f"    {tau_stop:>9.2f}{tau_esc:>9.2f}{acc:>8.3f}{meanN:>7.2f}{esc*100:>6.0f}%{flops:>16.2f}x")
        bo8 = np.mean([verifier_pick_ok(r["scores"], r["sl"], Nmax) for r in rows])
        print(f"    -> best-of-{Nmax} (fixed) acc={bo8:.3f} at meanN={Nmax} (FLOPs={Nmax*cost_cheap/cost_strong:.2f}x strong)")
        if best_match:
            ts, te, ac, mn, es, fl = best_match
            print(f"    -> ADAPTIVE match: acc={ac:.3f} (>= bo{Nmax}-0.3pt) at meanN={mn:.2f} esc={es*100:.0f}%  "
                  f"FLOPs={fl:.2f}x strong  [tau_stop={ts}, tau_esc={te}]  => {(1-mn/Nmax)*100:.0f}% fewer cheap samples")
        summary[name] = dict(n=n, strong=float(strong), bo8=float(bo8),
                             adaptive=(None if not best_match else dict(acc=best_match[2], meanN=best_match[3],
                                       esc=best_match[4], flops=best_match[5])))
    os.makedirs(J("results/cascade_methods/artifacts"), exist_ok=True)
    json.dump(summary, open(J("results/cascade_methods/artifacts/open_bestofN_adaptive.json"), "w"), indent=1)
    print("\n-> results/cascade_methods/artifacts/open_bestofN_adaptive.json")

if __name__ == "__main__":
    main()
