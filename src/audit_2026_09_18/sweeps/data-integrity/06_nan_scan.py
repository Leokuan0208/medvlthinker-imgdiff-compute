#!/usr/bin/env python3
"""NaN/Inf scan of h_span for every feature cache head_final_stack.py actually reads
(Lingshu finelayer*, qwen, medgemma). Memory-mapped, chunked, read-only. No torch needed."""
import numpy as np, glob, os, sys, time

FEATS = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
STEMS = []
for pat in ["generator_eval_finelayer.npz", "generator_eval_finelayer_*.npz",
            "generator_eval_medgemma_*.npz", "generator_eval_qwen_*.npz",
            "generator_train_finelayer.npz", "generator_train_qwen_*.npz"]:
    STEMS += sorted(glob.glob(os.path.join(FEATS, pat)))
STEMS = sorted(set(STEMS))
print(f"scanning {len(STEMS)} files", flush=True)

CHUNK = 4000
bad = []
t0 = time.time()
for fp in STEMS:
    try:
        z = np.load(fp, mmap_mode="r")
    except Exception as e:
        bad.append((fp, "LOAD-FAIL", str(e)[:120]))
        print(f"  {os.path.basename(fp):55s} LOAD-FAIL {e}", flush=True)
        continue
    if "h_span" not in z.files:
        bad.append((fp, "NO-H-SPAN", str(z.files)))
        print(f"  {os.path.basename(fp):55s} NO h_span key; keys={z.files}", flush=True)
        continue
    arr = z["h_span"]
    n = arr.shape[0]
    n_nan = 0
    n_inf = 0
    for s in range(0, n, CHUNK):
        c = np.asarray(arr[s:s+CHUNK])  # materialize just this chunk
        n_nan += int(np.isnan(c).sum())
        n_inf += int(np.isinf(c).sum())
    status = "OK" if (n_nan == 0 and n_inf == 0) else "BAD"
    if status == "BAD":
        bad.append((fp, "NAN/INF", f"nan={n_nan} inf={n_inf}"))
    print(f"  {os.path.basename(fp):55s} shape={arr.shape} dtype={arr.dtype} nan={n_nan} inf={n_inf} [{status}]", flush=True)

print(f"\nDONE in {time.time()-t0:.1f}s. bad files: {len(bad)}")
for b in bad:
    print(" ", b)
