#!/usr/bin/env python3
"""decoding_sweep_prepare.py -- build the JUDGE inputs and the VERIFIER work list for the decoding sweep.

Both are content-addressed and incremental, so the marginal cost of a setting is only its NEW strings:
  * judge  : deduplicated by (ds, idx, norm(answer)) -- the convention the frozen pipeline uses
             (verifier_transfer_eval.py maps judge labels back through norm()).  Reuses the project's
             own judge, src/labeling/run_judge.py (MedVLThinker-32B, text-only, temp 0).
  * verify : deduplicated by (ds, idx, EXACT answer text) -- verifier_transfer_eval.py scores per SLOT
             with the raw surface string, so exact text is the right key (8965 vs 8943 rows on the
             deployed pool -- a 22-row difference, kept for exactness).
"""
import argparse, glob, json, os, sys
from collections import defaultdict
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G      # noqa: E402

SWEEP = os.path.join(ROOT, "ckpts/openvqa/decoding_sweep")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]

ap = argparse.ArgumentParser()
ap.add_argument("--include_deployed", action="store_true",
                help="also queue the stored deployed pool (transfer-dump preds) -- NULL TEST 2")
ap.add_argument("--verify_settings", default=None,
                help="comma-separated setting names (e.g. T07,T13,rp11) to queue for VERIFIER scoring. "
                     "The judge always covers every generated pool, so oracle@8 / distinct / the "
                     "capture-recapture ceiling are available for ALL settings; SELECTED and sel_eff "
                     "need the verifier and are therefore scoped when GPU time is short.")
ap.add_argument("--deployed_items", type=int, default=180,
                help="NULL TEST 2 sample size in ITEMS (all 8 slots of each), stratified by dataset. "
                     "Whole items, not loose slots, so sel_eff can be recomputed on the sample. "
                     "0 = the full 2345-item pool.")
A = ap.parse_args()

pools = defaultdict(list)          # ds -> list of (idx, question, gold, [texts])  -- for the JUDGE
vpools = defaultdict(list)         # same, restricted to --verify_settings          -- for the VERIFIER
VS = [t.strip() for t in A.verify_settings.split(",")] if A.verify_settings else None
for f in sorted(glob.glob(os.path.join(SWEEP, "ckpt_*.jsonl"))):
    base = os.path.basename(f)[len("ckpt_"):-len(".jsonl")]
    ds = next((d for d in DS if base.startswith(d)), None)
    if ds is None:
        continue
    tag = base[len(ds) + 1:]
    setting = tag.rsplit("_s", 1)[0]
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        rec = (r["idx"], r["question"], r["gold"], r["preds"])
        pools[ds].append(rec)
        if VS is None or setting in VS:
            vpools[ds].append(rec)
if A.include_deployed:
    import random
    Q = {}
    for ds in DS:
        for l in open(os.path.join(ROOT, f"ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl")):
            if l.strip():
                r = json.loads(l); Q[(ds, r["idx"])] = (r["question"], r["gold"])
    items = G.load_items()
    if A.deployed_items:
        byds = defaultdict(list)
        for i, it in enumerate(items):
            byds[it["ds"]].append(i)
        rng = random.Random(20260813)
        keep = []
        for ds in DS:                       # proportional, stratified, seeded
            k = max(1, round(A.deployed_items * len(byds[ds]) / len(items)))
            keep += rng.sample(byds[ds], min(k, len(byds[ds])))
        items = [items[i] for i in sorted(keep)]
        json.dump([[it["ds"], it["idx"]] for it in items],
                  open(os.path.join(SWEEP, "nulltest2_items.json"), "w"))
        print(f"NULL TEST 2 sample: {len(items)} items ({sum(len(it['preds']) for it in items)} slots)")
    for it in items:
        q, g = Q[(it["ds"], it["idx"])]
        rec = (it["idx"], q, g, list(it["preds"]))
        pools[it["ds"]].append(rec); vpools[it["ds"]].append(rec)

# ---------------- judge inputs (dedup by na, stable ids) ----------------
# Strings the project's judge has ALREADY labelled (validated preload) are free -- never re-judge them.
PRE = {}
for ds in DS:
    p = os.path.join(SWEEP, f"judgecache_preload_{ds}.jsonl")
    if os.path.exists(p):
        for l in open(p):
            if l.strip():
                r = json.loads(l); PRE[(ds, int(r["idx"]), r["na"])] = 1
print(f"preloaded judge labels available: {len(PRE)}")

n_new_judge = 0
for ds in DS:
    mp = os.path.join(SWEEP, f"judgemap_{ds}.json")
    jin = os.path.join(SWEEP, f"judgein_{ds}.jsonl")
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    ctr = defaultdict(int)
    for k, v in m.items():
        ctr[k.split("|")[0]] = max(ctr[k.split("|")[0]], int(v.split("#")[1]) + 1)
    with open(jin, "a") as fh:
        for idx, q, gold, texts in pools[ds]:
            for t in texts:
                na = G.norm(t)
                k = f"{idx}|{na}"
                if k in m or (ds, idx, na) in PRE:
                    continue
                jid = f"{idx}#{ctr[str(idx)]}"; ctr[str(idx)] += 1
                m[k] = jid; n_new_judge += 1
                fh.write(json.dumps({"idx": jid, "question": q, "gold": gold, "modal_pred": t, "na": na}) + "\n")
    json.dump(m, open(mp, "w"))
    print(f"{ds}: judge map {len(m)} entries -> {os.path.basename(jin)}", flush=True)

# ---------------- verifier work list (dedup by exact text) ----------------
have = set()
for f in glob.glob(os.path.join(SWEEP, "vscore_cache_shard*.jsonl")):
    for l in open(f):
        if l.strip():
            try:
                r = json.loads(l); have.add((r["ds"], r["idx"], r["ans"]))
            except Exception:
                pass
work, seen = [], set()
for ds in DS:
    for idx, q, gold, texts in vpools[ds]:
        for t in texts:
            k = (ds, idx, t)
            if k in seen or k in have:
                continue
            seen.add(k); work.append([ds, idx, t])
out = os.path.join(SWEEP, "verifier_work.json")
json.dump(work, open(out, "w"))
print(f"judge: {n_new_judge} NEW candidate strings queued")
print(f"verify: {len(work)} NEW (ds,idx,text) to score (cache already holds {len(have)}) -> {out}")
