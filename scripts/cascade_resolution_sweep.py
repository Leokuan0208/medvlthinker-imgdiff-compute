#!/usr/bin/env python3
"""
cascade_resolution_sweep.py — 2-D efficiency sweep over (resolution cap x escalation threshold).

WHAT THIS ANSWERS
  The 7B->32B cascade keeps the cheap 7B answer when the 7B is confident (margin >= tau)
  and escalates to the expensive 32B otherwise. Two knobs control compute:
    (1) resolution of the cheap 7B leg  -> sets the 7B prefill cost (vision tokens)
    (2) the escalation threshold tau    -> sets how often we pay for the 32B
  This script finds the (resolution, tau) pair that minimizes compute while holding cascade
  accuracy at the always-32B level on the competent benchmarks.

ARMS / FILES (CPU only, no GPU; reads the jsonls already on disk)
  full-res 7B : <repo>/ckpts/gate_7b_vllm            cell nothink_norag
  capped 7B   : <repo>/ckpts/gate_7b_prune/<cap>     cell nothink_norag   (cap640/320/160/80)
  32B         : <repo>/ckpts/gate_32b                cell think_norag
  Each row schema: {idx, gold, pred, ok, parse_ok, opt_logprobs{A..}, gen_tokens, ...}

MARGIN  = top1 - top2 of the 7B option log-probs (nats). Low margin => 7B unsure => escalate.

DECISION RULE (competent datasets only; MedXpert is near-chance and reported, not gated):
  A (resolution, tau) point is ACCEPTABLE iff, for EVERY competent dataset,
       cascade_acc >= always32_acc - 1 * bootstrap_std(always32_acc).
  Among acceptable points the SWEET SPOT has the lowest estimated compute.

COST MODEL (transparent FLOPs proxy; calibrate against yesterday's prefill-inclusive ~75%)
  per-question cost = 7B leg (paid always) + escalation_rate * 32B leg, normalized to always-32B.
  7B leg  = 2*P7 *(vis_tok + TXT_TOK) + 2*P7 *mean_gen_7b
  32B leg = 2*P32*(VIS_TOK_32B + TXT_TOK) + 2*P32*mean_gen_32b      (mean_gen measured from files)
  If the full-res cascade does NOT land near 0.75 at its min-acceptable tau, swap in the exact
  constants from router_cost_prefill.py.
"""
import json, glob, os, re, argparse
import numpy as np
from collections import defaultdict

np.random.seed(0)

# ----------------------------- config -----------------------------
COMPETENT = ["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"]
MEDXPERT  = ["MedXpert-Reasoning", "MedXpert-Understanding"]   # reported, excluded from the verdict

# resolution label -> (subdir, assumed vision-token count for the cost model)
# NOTE: full-res token count is unknown from the logs; the pmctrain anchor (0.460 ~ cap640 0.458)
# suggests the baseline runs near ~640 tokens. If Section A shows fullres acc >> cap640 acc, the
# baseline used higher resolution -> raise VIS_TOK_FULLRES accordingly.
VIS_TOK_FULLRES = 640
RES_SOURCES = [
    ("fullres", "ckpts/gate_7b_vllm",        VIS_TOK_FULLRES),
    ("cap640",  "ckpts/gate_7b_prune/cap640", 640),
    ("cap320",  "ckpts/gate_7b_prune/cap320", 320),
    ("cap160",  "ckpts/gate_7b_prune/cap160", 160),
    ("cap80",   "ckpts/gate_7b_prune/cap80",   80),
]
DIR_32B = "ckpts/gate_32b"

# cost-model constants
P7, P32      = 7.0e9, 32.0e9   # parameter counts
TXT_TOK      = 64              # mean prompt/text tokens (approx; refine if measured)
VIS_TOK_32B  = 640            # 32B runs at ~full res (same model family)
N_BOOT       = 1000           # bootstrap resamples for the accuracy std
N_TAU        = 240            # tau grid resolution

# ----------------------------- loaders ----------------------------
def load_arm(ckdir, cell):
    """idx-keyed rows per dataset, merging any shards. Mirrors router_escalate.py.
    The strict regex (leading underscore before the cell) means a 'think_norag' query does
    NOT accidentally pick up 'nothink_norag' files even though glob matches the substring."""
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}_s\d+of\d+\.jsonl$")
    d = defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m = pat.search(os.path.basename(f))
        if not m:
            continue
        ds = m.group(1)
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "idx" in r:
                d[ds][r["idx"]] = r
    return d

def margin(row):
    lp = row.get("opt_logprobs") or {}
    v = sorted(lp.values(), reverse=True)
    return (v[0] - v[1]) if len(v) >= 2 else 0.0

def mean_gen(arm):
    """mean generated-token count over all rows of an arm (for the decode cost term)."""
    vals = [r.get("gen_tokens", 0) or 0 for ds in arm.values() for r in ds.values()]
    return float(np.mean(vals)) if vals else 0.0

# ----------------------------- joins ------------------------------
def joined_arrays(r7res, r32, ds):
    """Per-dataset aligned arrays over the common idx of (7B-at-res, 32B)."""
    a, b = r7res.get(ds, {}), r32.get(ds, {})
    idx = sorted(set(a) & set(b))
    ok7  = np.array([a[i]["ok"] for i in idx], dtype=float)
    ok32 = np.array([b[i]["ok"] for i in idx], dtype=float)
    marg = np.array([margin(a[i]) for i in idx], dtype=float)
    return ok7, ok32, marg

def boot_std(ok, n_boot=N_BOOT):
    if len(ok) == 0:
        return 0.0
    n = len(ok)
    accs = ok[np.random.randint(0, n, size=(n_boot, n))].mean(axis=1)
    return float(accs.std())

# ----------------------------- core sweep -------------------------
def cascade_acc(ok7, ok32, marg, tau):
    """keep 7B where margin>=tau, else escalate to 32B. returns (cascade_acc, escalation_rate)."""
    esc = marg < tau
    out = np.where(esc, ok32, ok7)
    return out.mean(), esc.mean()

def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
    A = ap.parse_args()
    repo = A.repo

    # load the fixed arms once
    r32 = load_arm(os.path.join(repo, DIR_32B), "think_norag")
    g32 = mean_gen(r32)
    cost_32b_leg = 2*P32*(VIS_TOK_32B + TXT_TOK) + 2*P32*g32

    # load every resolution's 7B arm
    res_arms = {}
    for label, sub, vtok in RES_SOURCES:
        arm = load_arm(os.path.join(repo, sub), "nothink_norag")
        res_arms[label] = (arm, vtok)

    # ---------- sample-count sanity ----------
    print("="*88)
    print("SAMPLE COUNTS (common idx with 32B), per competent dataset")
    hdr = f"{'dataset':<24}" + "".join(f"{lab:>9}" for lab,_,_ in RES_SOURCES) + f"{'32B(n)':>9}"
    print(hdr); print("-"*len(hdr))
    for ds in COMPETENT + MEDXPERT:
        row = f"{ds:<24}"
        for lab,_,_ in RES_SOURCES:
            arm = res_arms[lab][0]
            n = len(set(arm.get(ds,{})) & set(r32.get(ds,{})))
            row += f"{n:>9}"
        row += f"{len(r32.get(ds,{})):>9}"
        print(row)

    # ---------- Section A: 7B accuracy by resolution (comparison vs baseline) ----------
    print("\n" + "="*88)
    print("SECTION A — 7B accuracy by resolution (e=0, all-7B) vs always-32B")
    hdr = f"{'dataset':<12}{'32B':>8}" + "".join(f"{lab:>9}" for lab,_,_ in RES_SOURCES)
    print(hdr); print("-"*len(hdr))
    acc7 = {}   # (res,ds)->acc ; acc32[ds]
    acc32, std32 = {}, {}
    for ds in COMPETENT:
        ok7_fr, ok32, _ = joined_arrays(res_arms["fullres"][0], r32, ds)
        acc32[ds] = ok32.mean() if len(ok32) else float("nan")
        std32[ds] = boot_std(ok32)
        row = f"{ds:<12}{acc32[ds]:>8.3f}"
        for lab,_,_ in RES_SOURCES:
            ok7, _, _ = joined_arrays(res_arms[lab][0], r32, ds)
            acc7[(lab,ds)] = ok7.mean() if len(ok7) else float("nan")
            row += f"{acc7[(lab,ds)]:>9.3f}"
        print(row)
    # pooled (micro over competent)
    def pooled_arrays(reslabel):
        oks7, oks32, margs = [], [], []
        for ds in COMPETENT:
            o7,o32,m = joined_arrays(res_arms[reslabel][0], r32, ds)
            oks7.append(o7); oks32.append(o32); margs.append(m)
        return np.concatenate(oks7), np.concatenate(oks32), np.concatenate(margs)
    row = f"{'POOLED':<12}"
    _, p32, _ = pooled_arrays("fullres")
    row += f"{p32.mean():>8.3f}"
    for lab,_,_ in RES_SOURCES:
        p7,_,_ = pooled_arrays(lab)
        row += f"{p7.mean():>9.3f}"
    print(row)
    print("\n  Read: if 'fullres' ~ 'cap640' the baseline already runs near 640 tokens, and the")
    print("  usable reductions are cap320/cap160/cap80. Degradation that only appears at cap80 is")
    print("  the recurring 'no visual cliff' pattern.")

    # ---------- Section B: per-resolution escalation sweep + sweet spot ----------
    print("\n" + "="*88)
    print("SECTION B — escalation sweep per resolution (global tau; competent sets)")
    print(f"  cost% = (7B-leg / 32B-leg) + pooled escalation rate ; always-32B = 100%")
    print(f"  ACCEPTABLE = every competent dataset cascade_acc >= always32_acc - 1*boot_std\n")

    # tau grid from pooled competent margins (shared across resolutions for comparability is fine,
    # but each resolution has its own margin distribution, so build per-resolution grids).
    summary = []   # (res, best_tau, pooled_esc, cost_pct, per-ds dict)
    for label, sub, vtok in RES_SOURCES:
        cost_7b_leg = 2*P7*(vtok + TXT_TOK) + 2*P7*mean_gen({k:v for k,v in [(d, res_arms[label][0].get(d,{})) for d in COMPETENT]})
        offset = cost_7b_leg / cost_32b_leg            # additive 7B cost in always-32B units

        # per-dataset arrays
        DS = {ds: joined_arrays(res_arms[label][0], r32, ds) for ds in COMPETENT}
        pooled_m = np.concatenate([DS[ds][2] for ds in COMPETENT])
        taus = np.unique(np.quantile(pooled_m, np.linspace(0, 1, N_TAU)))

        best = None  # (cost_pct, tau, pooled_esc, per_ds)
        for tau in taus:
            ok_all = True
            per_ds = {}
            esc_num = esc_den = 0
            for ds in COMPETENT:
                ok7, ok32, marg = DS[ds]
                ca, er = cascade_acc(ok7, ok32, marg, tau)
                per_ds[ds] = (ca, er)
                esc_num += er*len(ok7); esc_den += len(ok7)
                if ca < acc32[ds] - std32[ds]:
                    ok_all = False
            if not ok_all:
                continue
            pooled_esc = esc_num/esc_den if esc_den else 1.0
            cost_pct = 100.0*(offset + pooled_esc)
            if best is None or cost_pct < best[0]:
                best = (cost_pct, float(tau), pooled_esc, per_ds)

        # corners for context: all-7B (tau=-inf) and always-32B (tau=+inf)
        all7_acc = {ds: DS[ds][0].mean() for ds in COMPETENT}
        print(f"--- {label}  (7B-leg overhead = {100*offset:.1f}% of always-32B) ---")
        if best is None:
            print("    NO acceptable tau — even full escalation needed on some dataset "
                  "(low-res 7B too weak to keep any answers here).")
            summary.append((label, None, None, None, None))
        else:
            cost_pct, tau, pooled_esc, per_ds = best
            print(f"    sweet tau={tau:+.3f}  ->  pooled escalation={100*pooled_esc:.1f}%   "
                  f"cost={cost_pct:.1f}% of always-32B")
            for ds in COMPETENT:
                ca, er = per_ds[ds]
                flag = "" if ca >= acc32[ds]-std32[ds] else "  (<bar!)"
                print(f"      {ds:<11} cascade_acc={ca:.3f}  (32B={acc32[ds]:.3f}±{std32[ds]:.3f})  "
                      f"esc={100*er:4.1f}%{flag}")
            summary.append((label, tau, pooled_esc, cost_pct, per_ds))
        print()

    # ---------- Section C: cross-resolution sweet spot ----------
    print("="*88)
    print("SECTION C — best operating point per resolution (lower cost% is better)")
    hdr = f"{'resolution':<10}{'tau':>9}{'pooled_esc%':>13}{'cost%':>9}"
    print(hdr); print("-"*len(hdr))
    feasible = [s for s in summary if s[1] is not None]
    for label, tau, pe, cost_pct, _ in summary:
        if tau is None:
            print(f"{label:<10}{'--':>9}{'--':>13}{'infeasible':>9}")
        else:
            print(f"{label:<10}{tau:>+9.3f}{100*pe:>13.1f}{cost_pct:>9.1f}")
    if feasible:
        win = min(feasible, key=lambda s: s[3])
        print(f"\n  >>> SWEET SPOT: {win[0]} at tau={win[1]:+.3f} -> {win[3]:.1f}% of always-32B compute "
              f"(pooled escalation {100*win[2]:.1f}%)")
    print("\n  Cost-model assumptions:")
    print(f"    P7={P7:.0e} P32={P32:.0e}  TXT_TOK={TXT_TOK}  VIS_TOK_32B={VIS_TOK_32B}  "
          f"mean_gen_32B={g32:.0f}")
    print("    (If fullres cost% is far from ~75%, replace the cost terms with router_cost_prefill.py.)")

if __name__ == "__main__":
    run()
