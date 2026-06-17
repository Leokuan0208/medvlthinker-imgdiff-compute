#!/usr/bin/env python3
"""Per-question router analysis on merged n=500 gate records.
Reads archive/single-model-routing/gate_7b_rag_axes/*.jsonl, joins all cells per (dataset, idx),
and reports: confusion / flip-rate, oracle ceiling, per-axis decomposition,
and opt_logprobs confidence-predictability for the rescue decision."""
import json, glob, os, re, math
from collections import defaultdict

CKDIR = "archive/single-model-routing/gate_7b_rag_axes"
DATASETS = ["MedXpert-Reasoning", "MedXpert-Understanding", "PMC-VQA"]
CELLS = ["nothink_norag", "think_norag", "think_rag_StatPearls", "think_rag_Textbooks"]
BASELINE = "nothink_norag"   # cheapest cell; the policy a router "defaults" to

pat = re.compile(r"ckpt_(.+?)_(" + "|".join(CELLS) + r")(?:_s\dof\d)?\.jsonl$")

# rec[(ds, cell)][idx] = row
rec = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(CKDIR, "*.jsonl"))):
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    ds, cell = m.group(1), m.group(2)
    for l in open(f):
        if l.strip():
            r = json.loads(l)
            rec[(ds, cell)][r["idx"]] = r

def margin(row):
    """top1 - top2 option logprob = cheap confidence; higher = more confident."""
    lp = sorted(row["opt_logprobs"].values(), reverse=True)
    return lp[0] - lp[1] if len(lp) >= 2 else 0.0

def auroc(scores, labels):
    """rank-based AUROC; labels in {0,1}."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j+1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

print("=" * 78)
print("ROUTER ANALYSIS  (merged n=500, all cells joined per idx)")
print("=" * 78)

pooled = {"oracle": 0, "base": 0, "bestfixed_n": 0, "n": 0}
pooled_axis = defaultdict(lambda: [0, 0])  # axis -> [oracle_correct, base_correct]
pred_scores, pred_labels = [], []          # for retrieval/think rescue predictability

for ds in DATASETS:
    idxs = sorted(rec[(ds, BASELINE)].keys())
    n = len(idxs)
    # fixed-cell accuracies
    fixed_acc = {c: sum(rec[(ds, c)][i]["ok"] for i in idxs) / n for c in CELLS}
    best_fixed_cell = max(fixed_acc, key=fixed_acc.get)

    # per-question oracle (best of any cell)
    oracle_hits = sum(1 for i in idxs if any(rec[(ds, c)][i]["ok"] for c in CELLS))
    base_hits = sum(rec[(ds, BASELINE)][i]["ok"] for i in idxs)
    oracle_acc = oracle_hits / n

    # flip rate: questions where baseline is wrong but some other cell is right
    rescued = sum(1 for i in idxs
                  if not rec[(ds, BASELINE)][i]["ok"]
                  and any(rec[(ds, c)][i]["ok"] for c in CELLS if c != BASELINE))
    # partition
    allright = sum(1 for i in idxs if all(rec[(ds, c)][i]["ok"] for c in CELLS))
    allwrong = sum(1 for i in idxs if not any(rec[(ds, c)][i]["ok"] for c in CELLS))
    mixed = n - allright - allwrong

    print(f"\n### {ds}  (n={n})")
    for c in CELLS:
        star = "  <- best fixed" if c == best_fixed_cell else ""
        print(f"    {c:24s} acc={fixed_acc[c]:.3f}{star}")
    print(f"    {'ORACLE (any cell)':24s} acc={oracle_acc:.3f}")
    print(f"    headroom over best fixed = {oracle_acc - fixed_acc[best_fixed_cell]:+.3f}"
          f"   over baseline = {oracle_acc - fixed_acc[BASELINE]:+.3f}")
    print(f"    partition: all-right={allright}  all-wrong={allwrong}  mixed={mixed}"
          f"  ({mixed/n:.1%} routable)")
    print(f"    baseline-wrong rescued by another cell = {rescued}  ({rescued/n:.1%})")

    pooled["oracle"] += oracle_hits
    pooled["base"] += base_hits
    pooled["bestfixed_n"] += round(fixed_acc[best_fixed_cell] * n)
    pooled["n"] += n

    # ---- axis decomposition ----
    # reasoning axis: nothink_norag vs think_norag (retrieval held off)
    th_oracle = sum(1 for i in idxs
                    if rec[(ds,"nothink_norag")][i]["ok"] or rec[(ds,"think_norag")][i]["ok"])
    pooled_axis["reasoning(nothink|think)"][0] += th_oracle
    pooled_axis["reasoning(nothink|think)"][1] += base_hits
    # retrieval axis: think_norag vs best-rag (reasoning held on)
    best_rag = "think_rag_StatPearls" if fixed_acc["think_rag_StatPearls"] >= fixed_acc["think_rag_Textbooks"] else "think_rag_Textbooks"
    rag_oracle = sum(1 for i in idxs
                     if rec[(ds,"think_norag")][i]["ok"] or rec[(ds,best_rag)][i]["ok"])
    rag_base = sum(rec[(ds,"think_norag")][i]["ok"] for i in idxs)
    pooled_axis[f"retrieval(think_norag|{best_rag})"][0] += rag_oracle
    pooled_axis[f"retrieval(think_norag|{best_rag})"][1] += rag_base

    # ---- predictability: does baseline confidence predict need-to-route? ----
    # label=1 if baseline WRONG and some other cell RIGHT (rescuable);
    # score = baseline margin. A USEFUL signal => low margin predicts rescuable.
    for i in idxs:
        b = rec[(ds, BASELINE)][i]
        rescuable = (not b["ok"]) and any(rec[(ds,c)][i]["ok"] for c in CELLS if c != BASELINE)
        # only consider questions the baseline got wrong: among those, is margin informative?
        if not b["ok"]:
            pred_scores.append(-margin(b))   # negate: high score = low confidence = should route
            pred_labels.append(1 if rescuable else 0)

print("\n" + "=" * 78)
print("POOLED (all 3 datasets)")
print("=" * 78)
N = pooled["n"]
print(f"    baseline acc       = {pooled['base']/N:.3f}")
print(f"    best-fixed acc     = {pooled['bestfixed_n']/N:.3f}")
print(f"    ORACLE acc         = {pooled['oracle']/N:.3f}")
print(f"    headroom: oracle - best_fixed = {(pooled['oracle']-pooled['bestfixed_n'])/N:+.3f}")
print(f"             oracle - baseline    = {(pooled['oracle']-pooled['base'])/N:+.3f}")

print("\n    --- per-axis oracle (which axis carries the routable gap) ---")
for axis, (orc, base) in pooled_axis.items():
    print(f"    {axis:42s} oracle={orc/N:.3f}  base={base/N:.3f}  gap={(orc-base)/N:+.3f}")

print("\n" + "=" * 78)
print("PREDICTABILITY  (among baseline-WRONG questions: is low confidence == rescuable?)")
print("=" * 78)
n_wrong = len(pred_labels)
n_rescuable = sum(pred_labels)
print(f"    baseline-wrong questions = {n_wrong}   of which rescuable = {n_rescuable} ({n_rescuable/n_wrong:.1%})")
a = auroc(pred_scores, pred_labels)
print(f"    AUROC(low-confidence -> rescuable) = {a:.3f}")
print("    (0.5 = confidence tells us nothing about which wrong answers another cell fixes;")
print("     >0.65 = a cheap confidence gate could route usefully)")

# confidence-binned rescue rate
print("\n    rescue rate by baseline-confidence quartile (Q1=least confident):")
paired = sorted(zip(pred_scores, pred_labels), reverse=True)  # high -> least confident first
q = max(1, len(paired)//4)
for qi in range(4):
    chunk = paired[qi*q:(qi+1)*q] if qi < 3 else paired[qi*q:]
    if chunk:
        rr = sum(y for _, y in chunk)/len(chunk)
        print(f"      Q{qi+1}  n={len(chunk):3d}  rescue_rate={rr:.3f}")
