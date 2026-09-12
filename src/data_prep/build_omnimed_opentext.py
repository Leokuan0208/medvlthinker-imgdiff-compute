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

RETRACTION 2026-08-19 -- THE ORIGINAL JUSTIFICATION HERE WAS WRONG, AND WRONG IN A WAY THIS FILE
HAD ALREADY BEEN WARNED ABOUT.  It argued that OmniMedVQA fits because `gt_answer` averages 2.35
words, "the same profile as our own open cells".  That is an ANSWER-LENGTH test -- the exact test
OPENTEXT_CELL_SURVEY_2026-08-18.md section 2 had already disowned when correcting the Quilt-VQA
rejection: "The disqualifying property is the KIND of answer, not the count."  The lesson was
written down and then not applied one day later.

Measured on the 15,831-row file this script actually produced:
  Disease Diagnosis          8,191 rows (51.7%) with only 224 distinct golds
  ... of those, 42.9% have a question string that maps to MORE THAN ONE gold
  ... the largest template, "What can be observed in this image?", is 436 rows over 65 golds,
      i.e. a 65-way blind classification once the options are gone
  binary-in-disguise         2,738 rows (17.3%) -- e.g. "Is the lesion depicted in this image
                             non-cancerous?" -> gold "Yes".  That is the property that got
                             ProbMed rejected, smuggled into "open-text" training data.
  golds sit at a taxonomy abstraction no model emits unprompted: "Benign epidermal.",
  "Chondral abnormality", "Arterial pathology".

WHAT SURVIVES.  Modality Recognition and Anatomy Identification DO constrain the answer space from
the question alone ("What imaging technique was employed?" -> "Fundus imaging"; "What organ is
shown?" -> "liver").  That is 5,670 of the 15,831 rows over 74 distinct golds, and it is what
--qtypes now defaults to.  Disease Diagnosis and Lesion Grading are excluded: their golds are only
meaningful as a choice among the options that were dropped.

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
    """DECODED-RGB-PIXEL md5 -- the project's contamination currency.

    BUGFIX 2026-08-19: this hashed raw FILE BYTES while the reference set it was compared against
    (extract_generator_hidden.py:70-74) stores decoded-pixel hashes.  Two disjoint hash spaces, so
    the collision guard was mathematically incapable of firing and its "0 collisions" line was not
    evidence of anything.
    """
    try:
        from PIL import Image
        with Image.open(p) as im:
            return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()
    except Exception:
        return None


EVAL_META = [
    # 2026-09-12: the TRAIN pool was never audited against. A new cell could silently share images
    # with the probe's own training data -- which is exactly what happened to vqamed (19 MedPix
    # images from vqa_rad_open_train). Measured 0 overlap for gemex and omnimed, so this closes a
    # latent gap for them and a realised one for vqamed.
    "feats_hidden/generator_train_s0of2.meta.json",
    "feats_hidden/generator_train_s1of2.meta.json","feats_hidden/generator_eval_s0of2.meta.json",
             "feats_hidden/generator_eval_s1of2.meta.json",
             "feats_hidden/generator_eval_radimagenet.meta.json"]


def eval_image_hashes():
    """Decoded-pixel hashes of every image behind EVERY open eval cell.

    BUGFIX 2026-08-19: this covered only generator_eval_s{0,1}of2 -- the three original open cells --
    and so never looked at radimagenet_open, which had meanwhile been adopted as a reporting cell.
    """
    hs = set()
    for p in EVAL_META:
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
    ap.add_argument("--qtypes", nargs="*",
                    default=["Modality Recognition", "Anatomy Identification"],
                    help="question types to keep. Default excludes Disease Diagnosis and Lesion "
                         "Grading, whose golds are only meaningful given the dropped options. "
                         "Pass ALL to disable the filter (and read the retraction above first).")
    ap.add_argument("--exclude_src", nargs="*", default=["RadImageNet"],
                    help="source datasets to drop entirely. Defaults to RadImageNet, which is the "
                         "source of the radimagenet_open EVAL cell.")
    ap.add_argument("--audit_overlap", action="store_true",
                    help="hash every drawn image against the eval pools and refuse to write on a hit")
    A = ap.parse_args()

    keep_qt = None if (len(A.qtypes) == 1 and A.qtypes[0] == "ALL") else set(A.qtypes)
    recs, dropped, skipped_src = [], 0, []
    for f in sorted(glob.glob(os.path.join(QA, "*.json"))):
        # EXCLUDED 2026-08-19: RadImageNet is one of OmniMedVQA's 42 sources AND the source of the
        # radimagenet_open eval cell.  It was 16.2% of the unfiltered draw, and the --qtypes filter
        # RAISES it to 23.3% because Modality Recognition / Anatomy Identification are exactly the
        # types it dominates (ultrasound and MR are 100% RadImageNet).  62 exact decoded-pixel
        # collisions with the 1,000-image eval cell, plus task and taxonomy overlap a pixel hash
        # cannot see.  You cannot keep both; the cell is the reported artefact, so the source goes.
        if os.path.basename(f)[:-5] in set(A.exclude_src):
            skipped_src.append(os.path.basename(f)[:-5])
            continue
        for r in json.load(open(f)):
            if keep_qt is not None and r.get("question_type") not in keep_qt:
                dropped += 1
                continue
            ip = os.path.join(OMNI, r["image_path"])
            if not os.path.exists(ip):
                continue
            recs.append({"src": os.path.basename(f)[:-5], "modality": r.get("modality_type", "?"),
                         "qtype": r.get("question_type", "?"), "question": r["question"],
                         "answer": str(r["gt_answer"]), "img_path": ip,
                         "qid": r.get("question_id", "")})
    if skipped_src:
        print(f"[filter] EXCLUDED source datasets: {sorted(set(skipped_src))}", flush=True)
    print(f"[filter] kept question types {sorted(keep_qt) if keep_qt else 'ALL'}; "
          f"dropped {dropped} rows whose gold needs the options", flush=True)
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
    # ALWAYS write the image key: without it downstream image-grouped CV cannot group the rows that
    # share an image (2,083 of the 15,831 in the retracted draw did), and would leak across folds.
    for r in draw:
        if "img_md5" not in r:
            r["img_md5"] = img_md5(r["img_path"])
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
