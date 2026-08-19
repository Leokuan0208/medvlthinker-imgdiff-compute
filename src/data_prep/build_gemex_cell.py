#!/usr/bin/env python3
"""build_gemex_cell.py -- a chest X-ray open-text cell from GEMeX, using only HF-hosted images.

PROVENANCE, AND A CORRECTION.  OPENTEXT_CELL_SURVEY_2026-08-18.md section 3 declared GEMeX "not
obtainable": BoKelvin/GEMeX-VQA holds 19 sample rows and BoKelvin/GEMeX 404s.  That was wrong -- I
checked two repo names and stopped.  The real release is under a THIRD name by the same author,
BoKelvin/GEMeX-ThinkVG (ungated, 83.6 MB): 202,384 rows over 21,994 images, of which 61,240 are
open_ended_questions.  Four other repos (ChiragXP, son13567, kangqinyao, szbz) are byte-identical
re-uploads of the 19-row sample, so the survey's conclusion was right about those and wrong overall.

CREDENTIAL CORRECTION 2026-08-19.  I reported that the PhysioNet credential in ~/.netrc was dead
because `curl --netrc` returned 403 identically with and without it.  That was a tool artifact:
curl does not send basic auth the way PhysioNet's redirect flow needs, while `wget` reading the SAME
~/.netrc fetches credentialed files fine (verified: LICENSE.txt 2,518 B, and a real CXR at
1,430,084 B decoding to 2544x3056).  Nothing was wrong with the account.  So --source physionet
pulls all 21,955 images and all 61,240 open-ended questions, and avoids the third-party re-hosting
licence problem entirely.  --source hf remains for offline/prototype use.

IMAGES WITHOUT PHYSIONET (the fallback).  GEMeX ships MIMIC-CXR-JPG paths, not pixels.  macrinalobo/
Fair-GEMeX-VQA-Large redistributes 3,693 of those JPGs on HuggingFace, ungated, in the MIMIC
directory layout.  Matching on the DICOM id (the annotations write a LEADING SLASH the image repo
does not -- comparing raw strings gives a spurious 0% overlap) those 3,693 images cover 34,390
ThinkVG rows, of which 10,336 are open-ended.  That is larger than SLAKE-open + VQA-RAD-open +
PathVQA-open combined.

LICENCE WARNING, CARRIED INTO THE ARTIFACT.  Those images are PhysioNet-controlled data that a third
party re-hosted; that re-hosting does not transfer the DUA to us.  This cell is fine for prototyping
and MUST be re-pulled through our own MIMIC-CXR-JPG credential before anything is published.

  python3 src/data_prep/build_gemex_cell.py --max_words 12 --audit_overlap
"""
import argparse, hashlib, json, os
import numpy as np
from collections import Counter

OUT = "/data/dan/dataset/gemex_cell"
EVAL_META = ["feats_hidden/generator_eval_s0of2.meta.json",
             "feats_hidden/generator_eval_s1of2.meta.json",
             "feats_hidden/generator_eval_radimagenet.meta.json"]


def dicom_id(s):
    return os.path.splitext(os.path.basename(str(s).replace("\\", "/")))[0]


def pixel_md5(p):
    from PIL import Image
    with Image.open(p) as im:
        return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--max_words", type=int, default=12,
                    help="drop verbose answers; the Quilt reject line is 20.4 mean words")
    ap.add_argument("--drop_hedged", action="store_true", default=True,
                    help="drop answers containing ' or ' -- a disjunction has no single gold")
    ap.add_argument("--drop_prior_study", action="store_true", default=True,
                    help="drop answers referencing a prior study; not single-image answerable")
    ap.add_argument("--source", choices=["physionet", "hf"], default="physionet",
                    help="physionet = all 21,955 images via ~/.netrc (wget); hf = the 3,693 "
                         "third-party re-hosted subset")
    ap.add_argument("--max_images", type=int, default=0, help="cap the physionet pull (0 = all)")
    ap.add_argument("--audit_overlap", action="store_true")
    A = ap.parse_args()

    from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
    import pandas as pd

    print("[1/4] GEMeX-ThinkVG annotations ...", flush=True)
    p = hf_hub_download("BoKelvin/GEMeX-ThinkVG", "data/train-00000-of-00001.parquet",
                        repo_type="dataset")
    df = pd.read_parquet(p)
    df = df[df["question_type"] == "open_ended_questions"].copy()
    print(f"   {len(df)} open-ended rows over {df['image_path'].nunique()} images", flush=True)

    df["did"] = df["image_path"].map(dicom_id)
    if A.source == "hf":
        print("[2/4] HF-hosted MIMIC images (fallback) ...", flush=True)
        files = [f for f in list_repo_files("macrinalobo/Fair-GEMeX-VQA-Large", repo_type="dataset")
                 if f.lower().endswith(".jpg")]
        have = {dicom_id(f): f for f in files}
        df = df[df["did"].isin(have)].copy()
        print(f"   {len(have)} images / {len(df)} rows answerable from them", flush=True)
    else:
        print("[2/4] PhysioNet source (all images) ...", flush=True)
        have = {dicom_id(x): str(x).lstrip("/") for x in df["image_path"].unique()}
        print(f"   {len(have)} images referenced", flush=True)

    # the answer sits inside <answer>...</answer>
    df["gold"] = df["response"].astype(str).str.extract(r"<answer>(.*?)</answer>", expand=False)
    df = df[df["gold"].notna()].copy()
    df["gold"] = df["gold"].str.strip()
    n0 = len(df)
    df["w"] = df["gold"].str.split().str.len()
    if A.max_words:
        df = df[df["w"] <= A.max_words]
    if A.drop_hedged:
        df = df[~df["gold"].str.contains(r"\bor\b", case=False, regex=True)]
    if A.drop_prior_study:
        df = df[~df["gold"].str.contains(r"prior|previous|compared|interval|unchanged",
                                         case=False, regex=True)]
    print(f"[3/4] filtered {n0} -> {len(df)} (definite, <= {A.max_words} words, single-image)",
          flush=True)

    root = os.path.join(A.out, "images")
    os.makedirs(root, exist_ok=True)
    if A.source == "hf":
        print("[4/4] downloading the HF image subset ...", flush=True)
        root = snapshot_download("macrinalobo/Fair-GEMeX-VQA-Large", repo_type="dataset",
                                 local_dir=root, allow_patterns=["*.jpg"], max_workers=8)
    else:
        import subprocess, concurrent.futures as cf
        need = sorted({have[d] for d in df["did"].unique()})
        if A.max_images:
            need = need[:A.max_images]
        print(f"[4/4] pulling {len(need)} images from PhysioNet via ~/.netrc ...", flush=True)
        BASE = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/"

        def pull(rel):
            dst = os.path.join(root, rel)
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                return True
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            r = subprocess.run(["wget", "-q", "-O", dst, "--timeout=60", "--tries=3", BASE + rel],
                               capture_output=True)
            if r.returncode != 0 or os.path.getsize(dst) < 1000:
                if os.path.exists(dst):
                    os.remove(dst)
                return False
            return True

        okn = 0
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            for i, got in enumerate(ex.map(pull, need), 1):
                okn += bool(got)
                if i % 1000 == 0:
                    print(f"   {i}/{len(need)}  ok={okn}", flush=True)
        print(f"   pulled {okn}/{len(need)}", flush=True)
        keep = {d for d in df["did"].unique()
                if os.path.exists(os.path.join(root, have[d]))}
        df = df[df["did"].isin(keep)].copy()
        print(f"   {len(df)} rows have a local image", flush=True)

    ev = set()
    if A.audit_overlap:
        for m in EVAL_META:
            if os.path.exists(m):
                for r in json.load(open(m))["rows"]:
                    ev.add(r["img_md5"])

    rows, hits, seen = [], 0, {}
    for i, r in enumerate(df.itertuples()):
        ip = os.path.join(root, have[r.did])
        if not os.path.exists(ip):
            continue
        if r.did not in seen:
            seen[r.did] = pixel_md5(ip) if A.audit_overlap else None
            if A.audit_overlap and seen[r.did] in ev:
                hits += 1
                print(f"   COLLISION with an eval image: {ip}")
        rows.append({"idx": len(rows), "question": str(r.question), "answer": str(r.gold),
                     "img_path": ip, "img_md5": seen[r.did], "dicom_id": r.did})
    if hits:
        raise SystemExit(f"ABORT: {hits} images collide with the eval pools")

    os.makedirs(A.out, exist_ok=True)
    json.dump(rows, open(os.path.join(A.out, "gemex_open.json"), "w"))
    ws = [len(x["answer"].split()) for x in rows]
    print(f"\nwrote {A.out}/gemex_open.json")
    print(f"  {len(rows)} questions / {len({x['dicom_id'] for x in rows})} images")
    print(f"  answer words mean {np.mean(ws):.2f} median {np.median(ws):.0f} p90 {np.percentile(ws,90):.0f}")
    print(f"  distinct golds {len({x['answer'].lower() for x in rows})}")
    print(f"  eval-pool pixel collisions: {hits}  (audited against {len(ev)} eval images)")
    print(f"  LICENCE: images re-hosted from PhysioNet by a third party -- prototype only, "
          f"re-pull via our own MIMIC-CXR-JPG credential before publishing.")


if __name__ == "__main__":
    main()
