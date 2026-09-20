"""s06: from the em-rescore per-question cache (frozen 24 probes, held-out halves):
   (a) how often is the argmax a TIE (n_top>1) -> first-index (alphabetical) tie-break decides
   (b) vote multiplicity: BCE trains one row per DISTINCT candidate, so a candidate sampled 7/8
       times weighs the same as one sampled once. How skewed is that in practice, and does the
       probe's pick differ from the modal (self-consistency) candidate?"""
import json
import numpy as np
from collections import Counter

P = "/data/dan/audit_2026-09-18/tmp/em-rescore/em_rescore_per_question_cache.json"
d = json.load(open(P))
R = d["records"]

print("cell | Q | mean cands | P(n_top>1) | acc@pick | acc@tie-questions | "
      "P(pick votes==1) | mean votes of pick | P(pick==modal) | acc modal")
tot = []
for cell, recs in R.items():
    n = len(recs)
    ntop = np.array([r.get("n_top", 1) for r in recs])
    pick = np.array([r["pick"] for r in recs])
    accs = np.array([recs[i]["cands"][pick[i]]["y"] for i in range(n)])
    vpick = np.array([recs[i]["cands"][pick[i]]["votes"] for i in range(n)])
    ncand = np.array([len(r["cands"]) for r in recs])
    modal = np.array([int(np.argmax([c["votes"] for c in r["cands"]])) for r in recs])
    accm = np.array([recs[i]["cands"][modal[i]]["y"] for i in range(n)])
    tie = ntop > 1
    print(f"{cell:17s} {n:6d} {ncand.mean():6.2f}  {tie.mean():.4f}  {accs.mean():.4f}  "
          f"{(accs[tie].mean() if tie.any() else float('nan')):.4f}  "
          f"{(vpick == 1).mean():.4f}  {vpick.mean():.3f}  "
          f"{(pick == modal).mean():.4f}  {accm.mean():.4f}")
    tot.append((cell, n, tie.mean(), accs.mean(), (pick == modal).mean()))

print()
print("macro P(tie) =", np.mean([t[2] for t in tot]))
print("macro P(pick==modal) =", np.mean([t[4] for t in tot]))

# vote histogram over ALL held-out candidates
c = Counter()
for cell, recs in R.items():
    for r in recs:
        for x in r["cands"]:
            c[x["votes"]] += 1
tt = sum(c.values())
print("\nvote multiplicity over all held-out distinct candidates (n=%d):" % tt)
for k in sorted(c):
    print(f"  votes={k:2d}  {c[k]:8d}  {c[k]/tt:.4f}")
print("share of candidates sampled exactly once:", c[1] / tt)
