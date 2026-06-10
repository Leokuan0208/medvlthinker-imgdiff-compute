#!/usr/bin/env python3
"""
cascade_heldout_frontier.py — HONEST test-time eval of the resolution x escalation cascade.

Fixes the test-set peeking in cascade_resolution_sweep.py. The escalation threshold tau is set on
HELD-OUT pmctrain margins (uncontaminated: m23k RL is text-only), then applied BLIND to eval.
For each resolution we sweep a target escalation rate t, set tau = the t-quantile of the
pmctrain-at-cap margins, apply it to eval-at-cap, and report the eval accuracy-vs-compute frontier.
Comparing whole frontiers (not one tuned point) is robust to per-dataset noise like the SLAKE swing.

Per resolution:
  held-out tau source : ckpts/gate_7b_pmctrain[/ _prune/<cap>]   (*nothink*.jsonl, margins only)
  eval 7B             : ckpts/gate_7b_vllm | ckpts/gate_7b_prune/<cap>   (nothink_norag)
  32B (fixed)         : ckpts/gate_32b                                    (think_norag)

Compute proxy = 7B-leg overhead + eval escalation, normalized to always-32B (= 100%).
"""
import json, glob, os, re, argparse
import numpy as np
from collections import defaultdict

np.random.seed(0)
COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
P7, P32, TXT_TOK, VIS_TOK_32B = 7.0e9, 32.0e9, 64, 640
N_BOOT = 1000
TGRID = np.linspace(0.0, 1.0, 41)          # target escalation rates set on pmctrain

# resolution -> (eval_dir, pmctrain_dir, vis_tok)
RES = [
    ("fullres", "ckpts/gate_7b_vllm",        "ckpts/gate_7b_pmctrain",            640),
    ("cap640",  "ckpts/gate_7b_prune/cap640", "ckpts/gate_7b_pmctrain_prune/cap640", 640),
    ("cap320",  "ckpts/gate_7b_prune/cap320", "ckpts/gate_7b_pmctrain_prune/cap320", 320),
    ("cap160",  "ckpts/gate_7b_prune/cap160", "ckpts/gate_7b_pmctrain_prune/cap160", 160),
    ("cap80",   "ckpts/gate_7b_prune/cap80",  "ckpts/gate_7b_pmctrain_prune/cap80",   80),
]
DIR_32B = "ckpts/gate_32b"

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}_s\d+of\d+\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m: continue
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            if "idx" in r: d[m.group(1)][r["idx"]] = r
    return d

def margin_of(row):
    lp = row.get("opt_logprobs") or {}
    v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def load_pmctrain_margins(dirpath):
    out = []
    for f in glob.glob(os.path.join(dirpath, "*nothink*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line)
            except Exception: continue
            out.append(margin_of(r))
    return np.array(out, dtype=float)

def mean_gen(arm):
    vals = [r.get("gen_tokens", 0) or 0 for ds in arm.values() for r in ds.values()]
    return float(np.mean(vals)) if vals else 0.0

def joined(eval_arm, r32, ds):
    a, b = eval_arm.get(ds, {}), r32.get(ds, {})
    idx = sorted(set(a) & set(b))
    ok7  = np.array([a[i]["ok"] for i in idx], float)
    ok32 = np.array([b[i]["ok"] for i in idx], float)
    marg = np.array([margin_of(a[i]) for i in idx], float)
    return ok7, ok32, marg

def boot_std(ok):
    if len(ok) == 0: return 0.0
    n = len(ok)
    return float(ok[np.random.randint(0, n, (N_BOOT, n))].mean(1).std())

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A = ap.parse_args(); repo = A.repo

    r32 = load_arm(os.path.join(repo, DIR_32B), "think_norag")
    g32 = mean_gen(r32)
    cost_32b = 2*P32*(VIS_TOK_32B + TXT_TOK) + 2*P32*g32

    # pooled 32B accuracy (parity target) over the joined competent samples
    ev0 = load_arm(os.path.join(repo, RES[0][1]), "nothink_norag")
    ok32_pool = np.concatenate([joined(ev0, r32, ds)[1] for ds in COMPETENT])
    acc32 = ok32_pool.mean(); tol = boot_std(ok32_pool)
    print(f"Parity target: always-32B pooled acc = {acc32:.4f}  (tol = 1 boot std = {tol:.4f})")
    print(f"mean_gen_32B = {g32:.0f} tokens\n")

    rows = []
    for label, evd, pmd, vtok in RES:
        ev = load_arm(os.path.join(repo, evd), "nothink_norag")
        pm = load_pmctrain_margins(os.path.join(repo, pmd))
        offset = (2*P7*(vtok + TXT_TOK)) / cost_32b      # 7B-leg overhead (decode negligible, nothink)
        DS = {ds: joined(ev, r32, ds) for ds in COMPETENT}
        ok7_pool  = np.concatenate([DS[ds][0] for ds in COMPETENT])
        marg_pool = np.concatenate([DS[ds][2] for ds in COMPETENT])
        ok32_p    = np.concatenate([DS[ds][1] for ds in COMPETENT])

        if pm.size == 0:
            print(f"--- {label}: NO pmctrain margins found at {pmd} (skipped) ---\n")
            rows.append((label, None)); continue

        # frontier: tau from held-out pmctrain quantile, applied blind to eval
        frontier = []
        for t in TGRID:
            tau = np.quantile(pm, t)
            esc = marg_pool < tau
            casc = np.where(esc, ok32_p, ok7_pool)
            frontier.append((t, float(tau), esc.mean(), casc.mean()))
        # parity point: smallest eval cost with pooled cascade acc >= acc32 - tol
        parity = None
        for t, tau, e_eval, acc in frontier:
            cost = offset + e_eval
            if acc >= acc32 - tol:
                parity = (t, tau, e_eval, acc, cost); break

        acc7_all = ok7_pool.mean()
        print(f"--- {label}  (7B overhead {100*offset:.1f}%) ---")
        print(f"    corner all-7B : acc={acc7_all:.4f}  cost={100*offset:.1f}%")
        print(f"    corner all-32B: acc={acc32:.4f}  cost~{100*(offset+1):.0f}%")
        if parity:
            t, tau, e_eval, acc, cost = parity
            # per-dataset accuracy at this held-out tau
            perds = []
            for ds in COMPETENT:
                o7, o32, mg = DS[ds]
                em = mg < tau
                perds.append(f"{ds}={np.where(em,o32,o7).mean():.3f}(esc{100*em.mean():.0f}%)")
            print(f"    PARITY @ held-out tau={tau:+.3f} (pmctrain target esc={100*t:.0f}%):")
            print(f"      eval escalation={100*e_eval:.1f}%  ->  COST={100*cost:.1f}% of always-32B")
            print(f"      pooled acc={acc:.4f} (vs 32B {acc32:.4f})")
            print(f"      {'  '.join(perds)}")
            rows.append((label, 100*cost))
        else:
            print("    never reaches 32B parity on the frontier (shouldn't happen)")
            rows.append((label, None))
        print()

    print("="*64)
    print("HONEST cost to match always-32B accuracy, per resolution (held-out tau):")
    print(f"{'resolution':<10}{'cost% of always-32B':>22}")
    print("-"*32)
    feas = [(l, c) for l, c in rows if c is not None]
    for l, c in rows:
        print(f"{l:<10}{(f'{c:.1f}%' if c is not None else 'n/a'):>22}")
    if feas:
        win = min(feas, key=lambda x: x[1])
        print(f"\n  >>> best: {win[0]} at {win[1]:.1f}% of always-32B compute (honest, held-out tau)")
    print("\n  NOTE: this is COMPUTE (FLOPs proxy). Wall-clock latency differs — escalated questions")
    print("  pay 7B + 32B sequentially, so per-question latency on those is higher than 32B alone.")

if __name__ == "__main__":
    run()
