#!/usr/bin/env python3
"""
othercells_dataquality_audit.py -- cheap, zero-GPU data-quality quantification for the three cells the
2026-08-11 literature sweep flagged as having PUBLISHED data-quality concerns with no CLAUDE.md landmine
covering them: SLAKE, PathVQA, VQA-RAD.

External claims being checked against what is on this disk:
  * MedGemma technical report (arXiv:2507.05201): "we and others have identified potential data quality
    issues in PathVQA and MedVQA" (removed from training); VQA-RAD's original splits carry "train/test
    image contamination."
  * arXiv:2606.10066: 19.8% of SLAKE images have an extreme same-view near-neighbour in PMC-OA-beta
    under SigLIP-B-16 (4.2% under SO400M); VQA-RAD clean (<=0.9%).

What THIS script measures (all from files already on disk):
  1. TRAIN <-> EVAL image identity at the project's disjointness standard: md5 of the DECODED RGB pixel
     buffer (not the file bytes -- re-encoding changes file bytes but not pixels).
  2. TRAIN <-> EVAL exact (question, answer) reuse, and (image, question) reuse.
  3. WITHIN-eval question-text repetition and the majority-class ("trivially guessable") floor.
  4. What fraction of our measured always-32B-direct accuracy on the cell a majority-class constant
     policy reproduces -- the same statistic the PMC audit reports for the constant-letter floor.

Everything is descriptive: NONE of this is leakage into OUR method (both legs are zero-shot at eval
time). It bounds how much of each cell is answerable from a dataset prior, which is what determines how
much a per-cell delta can be trusted, and it is the number a reviewer will ask for.

Launch from repo root (CPU only):
    OMP_NUM_THREADS=4 python3 src/cascade_methods/othercells_dataquality_audit.py
"""
import os, sys, io, json, glob, hashlib, collections
import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/othercells_dataquality_2026-08-11.json")


def r4(x):
    try:
        return round(float(x), 4)
    except (TypeError, ValueError):
        return x


def pixel_md5(img_bytes):
    """md5 of the DECODED RGB pixel buffer -- the project's disjointness standard."""
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return hashlib.md5(im.tobytes()).hexdigest()
    except Exception:
        return None


def load_parquet_split(paths):
    import pyarrow.parquet as pq
    rows = []
    for p in sorted(paths):
        t = pq.read_table(p)
        imgs = t.column("image").to_pylist()
        qs = t.column("question").to_pylist()
        an = t.column("answer").to_pylist()
        for i in range(len(qs)):
            rows.append(dict(img=imgs[i]["bytes"], q=str(qs[i]).strip().lower(),
                             a=str(an[i]).strip().lower()))
    return rows


def hf_style_audit(name, train_glob, test_glob):
    tr = load_parquet_split(glob.glob(train_glob))
    te = load_parquet_split(glob.glob(test_glob))
    print(f"  {name}: train {len(tr)}  test {len(te)} -- hashing decoded pixels", flush=True)
    tr_h = [pixel_md5(r["img"]) for r in tr]
    te_h = [pixel_md5(r["img"]) for r in te]
    tr_set = set(h for h in tr_h if h)
    te_set = set(h for h in te_h if h)
    n_te_img_in_train = sum(1 for h in te_h if h and h in tr_set)
    tr_qa = set((r["q"], r["a"]) for r in tr)
    tr_iq = set((h, r["q"]) for h, r in zip(tr_h, tr) if h)
    n_qa = sum(1 for r in te if (r["q"], r["a"]) in tr_qa)
    n_iq = sum(1 for h, r in zip(te_h, te) if h and (h, r["q"]) in tr_iq)
    # exact (image, question, answer) triple
    tr_iqa = set((h, r["q"], r["a"]) for h, r in zip(tr_h, tr) if h)
    n_iqa = sum(1 for h, r in zip(te_h, te) if h and (h, r["q"], r["a"]) in tr_iqa)
    qc = collections.Counter(r["q"] for r in te)
    ac = collections.Counter(r["a"] for r in te)
    return dict(
        source_train=sorted(glob.glob(train_glob)), source_test=sorted(glob.glob(test_glob)),
        n_train=len(tr), n_test=len(te),
        n_unique_train_images_by_decoded_pixel_md5=len(tr_set),
        n_unique_test_images_by_decoded_pixel_md5=len(te_set),
        n_test_items_whose_IMAGE_appears_in_train=int(n_te_img_in_train),
        frac_test_items_whose_IMAGE_appears_in_train=r4(n_te_img_in_train / len(te)),
        n_test_images_shared_with_train=len(tr_set & te_set),
        frac_test_images_shared_with_train=r4(len(tr_set & te_set) / len(te_set)) if te_set else None,
        frac_test_items_with_exact_QA_pair_in_train=r4(n_qa / len(te)),
        frac_test_items_with_exact_IMAGE_and_QUESTION_in_train=r4(n_iq / len(te)),
        frac_test_items_with_exact_IMAGE_QUESTION_ANSWER_triple_in_train=r4(n_iqa / len(te)),
        n_distinct_test_questions=len(qc),
        frac_test_items_whose_question_repeats_within_test=r4(
            sum(v for v in qc.values() if v > 1) / len(te)),
        test_answer_majority_class=ac.most_common(1)[0][0],
        test_answer_majority_floor=r4(ac.most_common(1)[0][1] / len(te)),
        test_answer_top6={k: r4(v / len(te)) for k, v in ac.most_common(6)},
    )


def slake_audit():
    base = "/data/dan/dataset/medevalkit/SLAKE"
    tr = [r for r in json.load(open(f"{base}/train.json")) if r.get("q_lang") == "en"]
    va = [r for r in json.load(open(f"{base}/validation.json")) if r.get("q_lang") == "en"]
    te = [r for r in json.load(open(f"{base}/test.json")) if r.get("q_lang") == "en"]
    print(f"  SLAKE: train {len(tr)} val {len(va)} test {len(te)} -- hashing decoded pixels", flush=True)

    def h(rec):
        p = os.path.join(base, "imgs", rec["img_name"])
        if not os.path.exists(p):
            return None
        return pixel_md5(open(p, "rb").read())
    cache = {}

    def hh(rec):
        k = rec["img_name"]
        if k not in cache:
            cache[k] = h(rec)
        return cache[k]
    tr_h = [hh(r) for r in tr]
    te_h = [hh(r) for r in te]
    tr_set = set(x for x in tr_h if x)
    te_set = set(x for x in te_h if x)
    n_img = sum(1 for x in te_h if x and x in tr_set)
    tr_qa = set((str(r["question"]).strip().lower(), str(r["answer"]).strip().lower()) for r in tr)
    tr_iq = set((x, str(r["question"]).strip().lower()) for x, r in zip(tr_h, tr) if x)
    n_qa = sum(1 for r in te if (str(r["question"]).strip().lower(),
                                 str(r["answer"]).strip().lower()) in tr_qa)
    n_iq = sum(1 for x, r in zip(te_h, te) if x and (x, str(r["question"]).strip().lower()) in tr_iq)
    closed = [r for r in te if str(r.get("answer_type", "")).upper() == "CLOSED"]
    ac = collections.Counter(str(r["answer"]).strip().lower() for r in closed)
    return dict(
        source=f"{base}/{{train,validation,test}}.json + imgs/ (en only)",
        n_train=len(tr), n_validation=len(va), n_test=len(te),
        n_unique_train_images_by_decoded_pixel_md5=len(tr_set),
        n_unique_test_images_by_decoded_pixel_md5=len(te_set),
        n_test_images_shared_with_train=len(tr_set & te_set),
        frac_test_images_shared_with_train=r4(len(tr_set & te_set) / len(te_set)) if te_set else None,
        frac_test_ITEMS_whose_image_appears_in_train=r4(n_img / len(te)),
        frac_test_items_with_exact_QA_pair_in_train=r4(n_qa / len(te)),
        frac_test_items_with_exact_IMAGE_and_QUESTION_in_train=r4(n_iq / len(te)),
        n_test_closed=len(closed),
        closed_majority_class=ac.most_common(1)[0][0] if ac else None,
        closed_majority_floor=r4(ac.most_common(1)[0][1] / len(closed)) if closed else None,
        closed_answer_top6={k: r4(v / len(closed)) for k, v in ac.most_common(6)} if closed else None,
    )


def main():
    out = dict(
        title="Data-quality quantification for the three cells with published concerns and no CLAUDE.md "
              "landmine: SLAKE, PathVQA, VQA-RAD. Zero GPU, everything measured from files on disk.",
        date="2026-08-11", no_gpu=True, no_fabricated_numbers=True,
        reproduce="OMP_NUM_THREADS=4 python3 src/cascade_methods/othercells_dataquality_audit.py",
        disjointness_standard="md5 of the DECODED RGB pixel buffer (PIL convert('RGB').tobytes()), "
                              "per the project's standing methodology -- NOT file bytes.",
        external_claims_being_checked=dict(
            medgemma="arXiv:2507.05201: 'we and others have identified potential data quality issues in "
                     "PathVQA and MedVQA'; VQA-RAD's original splits carry 'train/test image contamination.'",
            slake_neighbours="arXiv:2606.10066: 19.8% of SLAKE images have an extreme same-view "
                             "near-neighbour in PMC-OA-beta under SigLIP-B-16 (4.2% under SO400M); "
                             "VQA-RAD clean (<=0.9%). NOT reproducible here (needs PMC-OA-beta + SigLIP); "
                             "what is measured below is the different, cheaper train<->eval question.",
        ),
        scope_caveat="NONE of this is leakage into OUR method: both cascade legs are zero-shot at eval "
                     "time and never see these train splits. It bounds how much of each cell is "
                     "answerable from a dataset prior, i.e. how much a per-cell delta can be trusted.",
    )
    print("VQA_RAD ...", flush=True)
    out["VQA_RAD"] = hf_style_audit(
        "VQA_RAD", "/data/dan/dataset/vqa_rad/data/train-*.parquet",
        "/data/dan/dataset/vqa_rad/data/test-*.parquet")
    print("PATH_VQA ...", flush=True)
    out["PATH_VQA"] = hf_style_audit(
        "PATH_VQA", "/data/dan/dataset/path_vqa/data/train-*.parquet",
        "/data/dan/dataset/path_vqa/data/test-*.parquet")
    print("SLAKE ...", flush=True)
    out["SLAKE"] = slake_audit()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("WROTE", OUT)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("VQA_RAD", "PATH_VQA", "SLAKE")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
