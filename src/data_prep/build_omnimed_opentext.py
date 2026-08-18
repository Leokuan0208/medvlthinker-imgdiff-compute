#!/usr/bin/env python3
"""build_omnimed_opentext.py -- turn OmniMedVQA into OPEN-TEXT training data for the head.

WHY THIS IS THE RIGHT DATA.  head_lodo_2026-08-18.json showed the head's skill is domain-specific:
it captures ~35-41% of above-floor headroom in-domain and ~13% out of domain, with per-dataset
transfer penalties of +0.10 to +0.20 sel_eff -- six to eleven times larger than any objective or
architecture effect in the 74-config sweep.  The fix is therefore breadth of DOMAIN in training,
not more rows of PathVQA and not a bigger head (head_curve_bce_2026-08-18.json: capacity spans
0.0041 at full data, while the last data doubling is worth +0.024 to +0.036).

The head's four training domains are pathology, GI endoscopy, and two radiology sets.  OmniMedVQA's
open-access half is 88,996 samples over 42 source datasets and 9 modalities, of which FIVE are
absent from our pool entirely: ultrasound (10,991), dermoscopy (6,679), microscopy (5,680), fundus
photography (5,398) and OCT (4,646).

WHY IT FITS THE PIPELINE.  OmniMedVQA ships as multiple choice, but `gt_answer` is a free-text
STRING ("Fundus imaging"), not a letter, and averages 2.35 words -- the same profile as our own
open cells (slake_open gold 1.72, pathvqa_open gold 2.36).  So the options are simply dropped and
the question asked open-ended, which is also the intervention we independently validated as a
DEBIASING fix (output_bias_correct_2026-08-17.json: naming the answer space biases the model).
The 32B judge then scores free text against gt_answer exactly as it does for every other cell.

NOT AN EVAL CELL.  OmniMedVQA is not one of the project's eight reporting cells, so using it as
training data cannot contaminate a reported number.  --audit_overlap additionally checks its image
pixel hashes against the open eval pools and refuses to write if any collide.

  python3 src/data_prep/build_omnimed_opentext.py --per_modality 2500 --audit_overlap
"""
import argparse, hashlib, json, os, glob, random
import numpy as np
from collections import defaultdict, Counter

OMNI = "/data/dan/dataset/medevalkit/OmniMedVQA_unpacked/OmniMedVQA"
QA = os.path.join(OMNI, "QA_information", "Open-access")
OUT = "/data/dan/dataset/omnimed_opentext"
# the modalities our head has never seen; drawn first when balancing
NEW_MODALITIES = {"ultrasound", "Dermoscopy", "Microscopy Images", "Fundus Photography",
                  "OCT (Optical Coherence Tomography"}


def img_md5(p):
    try:
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def eval_image_hashes():
    """Pixel hashes of every image behind the three open eval cells, from the frozen feature cache."""
    hs = set()
    for sh in (0, 1):
        p = f"feats_hidden/generator_eval_s{sh}of2.meta.json"
        if os.path.exists(p):
            for r in json.load(open(p))["rows"]:
                hs.add(r["img_md5"])
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_modality", type=int, default=2500,
                    help="questions to draw per modality; balances breadth against volume")
    ap.add_argument("--per_dataset_cap", type=int, default=900,
                    help="cap per source dataset so one big set cannot dominate its modality")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--audit_overlap", action="store_true",
                    help="hash every drawn image against the eval pools and refuse to write on a hit")
    A = ap.parse_args()

    recs = []
    for f in sorted(glob.glob(os.path.join(QA, "*.json"))):
        for r in json.load(open(f)):
            ip = os.path.join(OMNI, r["image_path"])
            if not os.path.exists(ip):
                continue
            recs.append({"src": os.path.basename(f)[:-5], "modality": r.get("modality_type", "?"),
                         "qtype": r.get("question_type", "?"), "question": r["question"],
                         "answer": str(r["gt_answer"]), "img_path": ip,
                         "qid": r.get("question_id", "")})
    print(f"[load] {len(recs)} answerable records over "
          f"{len({r['src'] for r in recs})} datasets, {len({r['modality'] for r in recs})} modalities",
          flush=True)

    rng = random.Random(A.seed)
    by_mod = defaultdict(list)
    for r in recs:
        by_mod[r["modality"]].append(r)
    draw = []
    for mod, rs in sorted(by_mod.items(), key=lambda kv: (kv[0] not in NEW_MODALITIES, kv[0])):
        by_src = defaultdict(list)
        for r in rs:
            by_src[r["src"]].append(r)
        picked = []
        srcs = sorted(by_src)
        rng.shuffle(srcs)
        # round-robin over source datasets so a modality is not one dataset in disguise
        pools = {s: (rng.sample(by_src[s], len(by_src[s]))) for s in srcs}
        # BUGFIX 2026-08-18: the original loop spun forever when every source had hit
        # per_dataset_cap but the modality quota was still unmet (e.g. 2 datasets x 900 cap
        # < 2500 quota).  Track a per-source count and stop when a full pass adds nothing.
        taken = {s: 0 for s in srcs}
        while len(picked) < A.per_modality:
            progressed = False
            for s in srcs:
                if len(picked) >= A.per_modality:
                    break
                if not pools[s] or taken[s] >= A.per_dataset_cap:
                    continue
                picked.append(pools[s].pop()); taken[s] += 1; progressed = True
            if not progressed:
                break
        draw += picked
        print(f"  {mod:34} drew {len(picked):5} from {len({p['src'] for p in picked})} datasets"
              f"{'   <-- NEW MODALITY' if mod in NEW_MODALITIES else ''}", flush=True)

    if A.audit_overlap:
        ev = eval_image_hashes()
        print(f"[audit] hashing {len(draw)} drawn images against {len(ev)} eval images ...", flush=True)
        hits = 0
        for i, r in enumerate(draw):
            h = img_md5(r["img_path"])
            r["img_md5"] = h
            if h in ev:
                hits += 1
                print(f"  COLLISION: {r['img_path']}")
        if hits:
            raise SystemExit(f"ABORT: {hits} drawn images collide with the eval pools")
        print(f"[audit] 0 collisions with the eval pools", flush=True)

    os.makedirs(A.out, exist_ok=True)
    for i, r in enumerate(draw):
        r["idx"] = i
    jp = os.path.join(A.out, "omnimed_open.json")
    json.dump(draw, open(jp, "w"))
    ws = [len(r["answer"].split()) for r in draw]
    print(f"\nwrote {jp}\n  {len(draw)} questions over {len({r['src'] for r in draw})} datasets "
          f"and {len({r['modality'] for r in draw})} modalities")
    print(f"  answer words mean {np.mean(ws):.2f} median {np.median(ws):.0f} p90 {np.percentile(ws,90):.0f}")
    print("  modality mix:", dict(Counter(r["modality"] for r in draw)))
    print("  question types:", dict(Counter(r["qtype"] for r in draw)))


if __name__ == "__main__":
    main()
