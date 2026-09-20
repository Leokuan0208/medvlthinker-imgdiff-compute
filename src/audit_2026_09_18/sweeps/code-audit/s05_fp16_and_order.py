"""s05: (a) float16 headroom / non-finite in the Lingshu h_span caches that feed the headline,
       (b) is cache ROW ORDER within a question independent of correctness?
           (rows are sorted by (ds, str(idx), na) in extract_generator_hidden.py:294, so the
           first-index argmax tie-break picks the ALPHABETICALLY FIRST distinct answer)."""
import json, os
import numpy as np
from collections import defaultdict

FE = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
ENS = [18, 20, 22]
CELLS = ["generator_eval_finelayer", "generator_eval_finelayer_pathvqa_open",
         "generator_eval_finelayer_radimagenet_open", "generator_eval_finelayer_vqamed_open",
         "generator_train_finelayer"]

print("=== (a) float16 range / finiteness of h_span at layers 18/20/22 (ceiling 65504) ===")
for st in CELLS:
    p = f"{FE}/{st}.npz"
    if not os.path.exists(p):
        print(f"{st}: MISSING"); continue
    z = np.load(p)
    lay = [int(x) for x in z["layers"]]
    H = z["h_span"]
    print(f"{st:45s} shape={H.shape} dtype={H.dtype} layers={lay}")
    for L in ENS:
        if L not in lay:
            print(f"   layer {L}: not in cache"); continue
        a = H[:, lay.index(L)]
        nf = int((~np.isfinite(a.astype(np.float32))).sum())
        z0 = int((np.abs(a.astype(np.float32)).max(1) == 0).sum())
        print(f"   L{L}: max|h| {float(np.abs(a.astype(np.float32)).max()):10.1f}  "
              f"non-finite {nf}  all-zero-rows {z0}")
    del z, H

print()
print("=== (b) within-question row order vs correctness (alphabetical by normalised answer) ===")
print("cell | questions | P(y=1 | first row) | P(y=1 | last row) | P(y=1 | random row) | n>=2 cands")
for st in CELLS:
    mp = f"{FE}/{st}.meta.json"
    if not os.path.exists(mp):
        continue
    rows = json.load(open(mp))["rows"]
    byq = defaultdict(list)
    for i, r in enumerate(rows):
        byq[(r["ds"], r["idx"])].append(i)
    multi = [v for v in byq.values() if len(v) >= 2]
    if not multi:
        continue
    y = np.array([r["y"] for r in rows], dtype=float)
    first = np.mean([y[v[0]] for v in multi])
    last = np.mean([y[v[-1]] for v in multi])
    rnd = np.mean([y[v].mean() for v in multi])
    # also: are rows actually sorted by na within a question?
    unsorted = sum(1 for v in byq.values()
                   if [rows[i]["na"] for i in v] != sorted(rows[i]["na"] for i in v))
    print(f"{st:45s} {len(multi):7d}  {first:.4f}  {last:.4f}  {rnd:.4f}   "
          f"unsorted-questions={unsorted}")
