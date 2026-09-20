#!/usr/bin/env python3
"""Cross-benchmark pixel-level image overlap under the project's by-image split
half(img) = md5("nd"+img_md5)%2 (1=train half, 0=held-out half), replicating
src/training_methods/head_final_stack.py's half() and drop logic (verified by direct grep,
see report). Joins out/reference.json (idx/question/gold/img_key) with out/pixhash.json
(img_key -> [pixmd5, dhash, w, h]). Read-only, <=4 threads (pure python/stdlib, no torch).
"""
import json, hashlib, sys
from collections import defaultdict

OUT = "/data/dan/audit_2026-09-18/tmp/data-integrity/out"

def half(img_md5):
    return int(hashlib.md5(("nd" + str(img_md5)).encode()).hexdigest(), 16) % 2

ref = json.load(open(f"{OUT}/reference.json"))
pix = json.load(open(f"{OUT}/pixhash.json"))

TRAIN_DOMAINS = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
EVAL8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]

# Build joined per-dataset record list: idx, question, gold, img_md5, dhash, half
recs = {}
join_fail = {}
for ds, items in ref.items():
    if ds not in pix:
        continue
    pmap = pix[ds]
    out = []
    fails = 0
    for x in items:
        k = x["img_key"]
        v = pmap.get(k)
        if v is None or (isinstance(v[0], str) and v[0].startswith("ERR:")):
            fails += 1
            continue
        pmd5, dh, w, h = v
        out.append({"idx": x["idx"], "q": x["question"], "gold": x["gold"],
                     "img_key": k, "pmd5": pmd5, "dh": dh, "half": half(pmd5)})
    recs[ds] = out
    join_fail[ds] = fails

print("=== join summary (n joined / n ref / join failures) ===")
for ds in ref:
    print(f"  {ds:22s} joined={len(recs.get(ds,[])):6d} ref={len(ref[ds]):6d} fail={join_fail.get(ds,'N/A')}")

# For each eval-8 dataset, split by half; for train-domain datasets, ALL rows count as train
def dataset_split(ds):
    rr = recs.get(ds, [])
    if ds in TRAIN_DOMAINS:
        return rr, []  # (train, heldout)
    tr = [r for r in rr if r["half"] == 1]
    ho = [r for r in rr if r["half"] == 0]
    return tr, ho

splits = {ds: dataset_split(ds) for ds in list(ref.keys())}

print()
print("=== per-dataset train/held-out counts (own-half split; train-domains are 100% train) ===")
for ds, (tr, ho) in splits.items():
    print(f"  {ds:22s} train={len(tr):6d} heldout={len(ho):6d}")

# ---- exact pixmd5 overlap across dataset pairs ----
def pmd5_index(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["pmd5"]].append(r)
    return d

def dh_index(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["dh"]].append(r)
    return d

def report_pair(nameA, dsA_rows, nameB, dsB_rows, labelA, labelB):
    """dsA_rows/dsB_rows: list of records (already the relevant half-subset)."""
    idxA = pmd5_index(dsA_rows)
    idxB = pmd5_index(dsB_rows)
    common = set(idxA) & set(idxB)
    print(f"  [{labelA}] {nameA} (n={len(dsA_rows)}) vs [{labelB}] {nameB} (n={len(dsB_rows)}): "
          f"exact-pmd5 shared images = {len(common)}")
    if common and len(common) <= 5:
        for c in list(common)[:5]:
            print(f"      md5={c[:10]}... A_idx={[r['idx'] for r in idxA[c]]} B_idx={[r['idx'] for r in idxB[c]]}")
    # dHash exact match, excluding those already caught by exact pmd5
    dhA = dh_index(dsA_rows)
    dhB = dh_index(dsB_rows)
    commonDH = (set(dhA) & set(dhB))
    # count image PAIRS with matching dhash but different pmd5 (the incremental "md5 misses" set)
    incremental = 0
    for dh in commonDH:
        pmdA = {r["pmd5"] for r in dhA[dh]}
        pmdB = {r["pmd5"] for r in dhB[dh]}
        if pmdA - pmdB or pmdB - pmdA or (pmdA != pmdB):
            # any dhash bucket containing pmd5 values not fully identical across sides counts
            if not (pmdA <= (set(idxB.keys())) and pmdB <= (set(idxA.keys()))):
                incremental += 1
    print(f"      dHash-exact (64-bit) shared buckets = {len(commonDH)}, of which NOT already exact-pmd5 matched = {incremental}")
    return len(common), len(commonDH), incremental

print()
print("========== PRIORITY PAIR 1: SLAKE/VQA-RAD/RadImageNet/PathVQA (train-half) vs OmniMedVQA (held-out) ==========")
om_tr, om_ho = splits["omnimed_open"]
for src in ["slake_open", "vqa_rad_open", "radimagenet_open", "pathvqa_open"]:
    s_tr, s_ho = splits[src]
    report_pair(src, s_tr, "omnimed_open", om_ho, "train-half", "held-out")

print()
print("========== PRIORITY PAIR 1b: same, but ALL of SLAKE/VQA-RAD/RadImageNet/PathVQA vs ALL of OmniMedVQA (either half) ==========")
om_all = om_tr + om_ho
for src in ["slake_open", "vqa_rad_open", "radimagenet_open", "pathvqa_open"]:
    s_tr, s_ho = splits[src]
    s_all = s_tr + s_ho
    report_pair(src, s_all, "omnimed_open", om_all, "any", "any")

print()
print("========== PRIORITY PAIR 2: VQA-RAD vs VQA-Med ==========")
vr_tr, vr_ho = splits["vqa_rad_open"]
vm_tr, vm_ho = splits["vqamed_open"]
report_pair("vqa_rad_open", vr_tr + vr_ho, "vqamed_open", vm_tr + vm_ho, "any", "any")

print()
print("========== PRIORITY PAIR 3: Kvasir-x1 (held-out) vs kvasir_open TRAIN DOMAIN ==========")
kx_tr, kx_ho = splits["kvasir_x1_open"]
ko_tr, ko_ho = splits["kvasir_open"]  # all train (train domain)
report_pair("kvasir_x1_open", kx_ho, "kvasir_open(train-domain)", ko_tr, "held-out", "train(all)")
report_pair("kvasir_x1_open", kx_tr, "kvasir_open(train-domain)", ko_tr, "own-train-half", "train(all)")

print()
print("========== PRIORITY PAIR 4: four TRAIN DOMAINS vs EVERY held-out half (union) ==========")
evimgs_pmd5 = set()
evimgs_dh = set()
per_bench_ho = {}
for b in EVAL8:
    tr, ho = splits[b]
    per_bench_ho[b] = ho
    evimgs_pmd5 |= {r["pmd5"] for r in ho}
    evimgs_dh |= {r["dh"] for r in ho}

total_drop_md5 = 0
total_rows_train_domains = 0
for td in sorted(TRAIN_DOMAINS):
    tr, _ = splits[td]
    total_rows_train_domains += len(tr)
    n_leak_md5 = sum(1 for r in tr if r["pmd5"] in evimgs_pmd5)
    n_leak_dh_only = sum(1 for r in tr if r["pmd5"] not in evimgs_pmd5 and r["dh"] in evimgs_dh)
    total_drop_md5 += n_leak_md5
    print(f"  {td:22s} train_rows={len(tr):6d}  exact-md5 leak into held-out union = {n_leak_md5:5d}  "
          f"ADDITIONAL dHash-only leak (md5 differs) = {n_leak_dh_only:5d}")

print(f"\n  TOTAL exact-md5 leaking rows across 4 train domains = {total_drop_md5} "
      f"(compare to head_final_stack.py's own reported drop count / '19 MedPix images' comment)")

# also: own-half train rows of the 8 eval benchmarks leaking into the held-out union (the code
# computes drop over rows_all which includes BOTH train-domain rows AND own-half-train eval rows)
print()
print("  -- own-half-train rows of the 8 eval benchmarks vs held-out union (same evimgs) --")
for b in EVAL8:
    tr, ho = splits[b]
    other_evimgs = evimgs_pmd5 - {r["pmd5"] for r in ho}  # exclude own held-out (self-consistent by construction)
    n_leak = sum(1 for r in tr if r["pmd5"] in other_evimgs)
    print(f"  {b:22s} own_train_half={len(tr):6d}  leak into OTHER benchmarks' held-out = {n_leak}")

print("\nDONE")
