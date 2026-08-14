#!/usr/bin/env python3
"""coadapt_T04_dryrun_composition.py -- what the trainer WILL draw from the T=0.4 pools, without
touching a GPU.

This replicates src/training_methods/run_lora_verifier_disjoint.py's example-assembly block VERBATIM
(the QREC build, `by_src`, the composition-matched draw, the radimagenet top-up and the pos_rate
computation) but stops before the model load.  Its only purpose is to surface the composition and the
achieved pos_rate BEFORE ~108 GPU-minutes are spent, and to prove the judge/exploded/sc8 key plumbing
lines up under the VERIF_CK / VERIF_TAG hooks.

Nothing here is a substitute for the real train_config.json -- that remains the artifact of record.

  VERIF_CK=ckpts/openvqa/cheap_lingshu7b_T04 VERIF_TAG=lingshu7bT04 \
    python3 src/training_methods/coadapt_T04_dryrun_composition.py --seed 0
"""
import argparse, json, os, random
from collections import defaultdict
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
CONTAMINATED_QUOTA = {"slake": 894, "pathvqa": 4973, "vqa_rad": 522, "kvasir": 3975}
FAMILY = {"slake_open_train": "slake", "pathvqa_open_train": "pathvqa",
          "vqa_rad_open_train": "vqa_rad", "kvasir_open": "kvasir", "radimagenet_open": "radimagenet"}

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_train", type=int, default=10364)
ap.add_argument("--level", choices=["L1", "L2"], default="L1")
ap.add_argument("--split", default="results/cascade_methods/artifacts/verifier_disjoint_split.json")
ap.add_argument("--idx_dir", default="data/disjoint_split")
A = ap.parse_args()

CK = os.path.join(ROOT, os.environ.get("VERIF_CK", "ckpts/openvqa/cheap_lingshu7b"))
TAG = os.environ.get("VERIF_TAG", "lingshu7b")


def loadj(p):
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} \
        if os.path.exists(p) else {}


def norm(s):
    return str(s).strip().lower()


SPL = json.load(open(os.path.join(ROOT, A.split)))
assert SPL["disjointness_assertion"]["image_pixel_hash_intersection"] == 0, "split is not image-disjoint"
DSETS = list(SPL["train"].keys())
pref = "idx_" if A.level == "L1" else "strict_idx_"
ALLOW = {ds: set(json.load(open(os.path.join(ROOT, A.idx_dir, f"{pref}{ds}.json")))) for ds in DSETS}

QREC = {}
for ds in DSETS:
    sc = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.jsonl")
    jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{TAG}_sc8_scexploded.judge.jsonl").items()}
    if not sc or not jud:
        print(f"  !! {ds}: missing sc8({len(sc)}) or judge({len(jud)}) -- SKIPPED", flush=True); continue
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]
            oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]
    n = 0
    for i in sc:
        if i not in ALLOW[ds] or i not in aj:
            continue
        QREC[(ds, i)] = {"preds": sc[i]["preds"], "slabels": aj[i]}
        n += 1
    print(f"  {ds:20s} sc8={len(sc):6d} exploded={len(exp):6d} judged={len(jud):6d} usable={n:6d}",
          flush=True)
print(f"{len(QREC)} training questions with labels "
      f"(NOTE: the real trainer also requires a resolvable IMAGE, which this dry run does not load)")

rng = random.Random(A.seed)
by_src = defaultdict(list)
for k in sorted(QREC, key=lambda t: (t[0], str(t[1]))):
    r = QREC[k]
    for na, lab in r["slabels"].items():
        by_src[k[0]].append((k[0], lab))
avail = {ds: len(v) for ds, v in by_src.items()}
print(f"[examples] available per source: {avail}")

train_ex, taken, short = [], {}, {}
for ds in DSETS:
    fam = FAMILY[ds]
    if fam == "radimagenet":
        continue
    want = CONTAMINATED_QUOTA[fam]
    pool = by_src.get(ds, []); rng.shuffle(pool)
    take = pool[:want]; train_ex += take; taken[ds] = len(take)
    if len(take) < want:
        short[ds] = want - len(take)
deficit = A.max_train - len(train_ex)
if deficit > 0:
    pool = by_src.get("radimagenet_open", []); rng.shuffle(pool)
    take = pool[:deficit]; train_ex += take; taken["radimagenet_open"] = len(take)
    print(f"[examples] would top up {len(take)} from radimagenet_open (shortfalls: {short})")
print(f"[examples] per-source taken: {taken}")
print(f"[examples] incumbent quota:  {CONTAMINATED_QUOTA} (total 10364)")
print(f"train examples={len(train_ex)} (target {A.max_train}); "
      f"pos_rate {np.mean([e[1] for e in train_ex]):.6f}  "
      f"(incumbent 0.19924739482825163)")
