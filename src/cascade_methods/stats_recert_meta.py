#!/usr/bin/env python3
"""
stats_recert_meta.py -- build per-item METADATA (image-cluster id + question text) for all 8
Variant-B reporting cells, in EXACTLY the per-item order used by
`paper_baselines.build_cells()` (and therefore by
`results/cascade_methods/artifacts/_selector_rerun_parts/vec_*.npz`).

WHY.  Two of the four ATTACK-C tasks need per-item side information that the eval vectors do not
carry:
  * cluster resampling (RouteGuard, arXiv:2608.07583) needs the IMAGE each question is about, so
    that whole images -- not individual questions -- are the resampling unit;
  * the pre-generation router (Bouchard, arXiv:2605.06350) needs the PROMPT text (and a cheap
    image descriptor) as its only inputs.

NO GPU, NO NEW INFERENCE.  Every field is read from the same on-disk dataset files that
MedEvalKit / run_openvqa.py read at generation time, in the same order.

ORDER PROVENANCE (asserted at build time, not assumed):
  PMC_VQA          MedEvalKit/utils/PMC_VQA/PMC_VQA.py reads datas/PMC-VQA/test_2.csv row by row.
                   Cluster = the csv's Figure_path.  Order check: the csv question text must
                   appear inside the results.json `prompt` for every row.
  SLAKE_closed     datas/SLAKE/test.json in file order, then MedEvalKit filters
                   answer_type == "CLOSED".  Cluster = img_name.  Order check: question strings
                   must match results.json exactly.
  SLAKE_open       run_openvqa.py keys SLAKE-open items by SLAKE's own `qid`; the cascade's open
                   arm iterates the transfer dump in file order.  Cluster = img_name via qid.
  VQA_RAD_closed   HF flaviagiammarino/vqa-rad test, positional; the local mirror
                   /data/dan/dataset/vqa_rad/data/test-*.parquet is byte-checked against the
                   results.json question strings before use.  Cluster = md5 of DECODED RGB pixels.
  VQA_RAD_open     transfer-dump order; idx = parquet row index.  Cluster = md5 decoded RGB.
  PATH_VQA_closed  /data/dan/dataset/path_vqa/data/test-*.parquet positional (this is what
                   MedEvalKit itself reads -- see utils/PATH_VQA/PATH_VQA.py), yes/no filter.
                   Cluster = md5 decoded RGB.
  PATH_VQA_open    transfer-dump order; idx = parquet row index.  Cluster = md5 decoded RGB.
  MedXpertQA-MM    HF TsinghuaC3I/MedXpertQA config MM, split test, positional.  Cluster = the
                   tuple of image file names (a MedXpert item may carry several).

Launch from the repo root:   python3 src/cascade_methods/stats_recert_meta.py
Writes: results/cascade_methods/artifacts/_stats_recert/meta_<cell>.json
"""
import csv
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import integrated_method as IM      # noqa: E402

ROOT = IM.ROOT
MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
OUTDIR = os.path.join(ART, "_stats_recert")
os.makedirs(OUTDIR, exist_ok=True)

MEKDATA = "/data/dan/dataset/medevalkit"
SLAKE_OPEN_ROOT = "/data/dan/dataset/slake"
VQARAD_PARQ = "/data/dan/dataset/vqa_rad/data"
PATHVQA_PARQ = "/data/dan/dataset/path_vqa/data"

DISJOINT = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
STRONG = os.path.join(ROOT, "ckpts/openvqa/strong_lingshu")

csv.field_size_limit(10 ** 9)


def md5_rgb(pil):
    """md5 of DECODED RGB pixels -- the project's standard image identity (never the file bytes,
    which differ for re-encoded copies of the same picture)."""
    return hashlib.md5(pil.convert("RGB").tobytes()).hexdigest()


def thumb_feats(pil, side=12):
    """Cheap CPU-only image descriptor for the pre-generation router: grayscale side x side
    thumbnail (mean-normalised) + log aspect + log area.  No model, no GPU."""
    g = pil.convert("L").resize((side, side))
    a = np.asarray(g, np.float32).ravel() / 255.0
    w, h = pil.size
    return np.concatenate([a, [np.log(max(w, 1) / max(h, 1)), np.log(max(w * h, 1))]]).astype(np.float32)


# -----------------------------------------------------------------------------------------------
def load_parquet_concat(base, split="test"):
    import pandas as pd
    fs = sorted(glob.glob(os.path.join(base, f"{split}-*.parquet")))
    assert fs, base
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True), fs


def cell_pmc():
    r7 = IM.load_raw("lingshu7b_full", "PMC_VQA")
    assert r7 is not None
    rows = list(csv.reader(open(os.path.join(MEKDATA, "PMC-VQA/test_2.csv"), "r", encoding="utf-8")))
    hdr, rows = rows[0], rows[1:]
    assert len(rows) >= len(r7), (len(rows), len(r7))
    fig, q = [], []
    bad = 0
    for i in range(len(r7)):
        index, figure_path, caption, question, cA, cB, cC, cD, answer, split = rows[i]
        fig.append(figure_path)
        q.append(question)
        if question.strip() and question.strip() not in str(r7[i].get("prompt", "")):
            bad += 1
    return dict(cluster=fig, question=q, order_check=dict(rows=len(r7), question_not_in_prompt=bad),
                image_path=[os.path.join(MEKDATA, "PMC-VQA/figures", f) for f in fig])


def _slake_test():
    return json.load(open(os.path.join(MEKDATA, "SLAKE/test.json"), "r", encoding="utf-8"))


def cell_slake_closed():
    r7 = IM.load_raw("lingshu7b_full", "SLAKE")
    d = _slake_test()
    assert len(d) == len(r7), (len(d), len(r7))
    keep = [i for i in range(len(r7)) if r7[i].get("answer_type") == "CLOSED"]
    bad = sum(1 for i in keep if str(d[i]["question"]).strip() != str(r7[i].get("question", "")).strip())
    return dict(cluster=[d[i]["img_name"] for i in keep], question=[d[i]["question"] for i in keep],
                order_check=dict(rows=len(keep), question_mismatch=bad),
                image_path=[os.path.join(MEKDATA, "SLAKE/imgs", d[i]["img_name"]) for i in keep])


def _open_order(dskey):
    """Item order of the cascade's open arm = transfer-dump order, filtered to items that have a
    32B judge label (integrated_pandora.load_open_rows)."""
    dp = json.load(open(os.path.join(DISJOINT, f"transfer_dump_{dskey}_lingshu7b.json")))
    sj = IM.load_judge_jsonl(os.path.join(STRONG, f"ckpt_{dskey}_lingshu32b.judge.jsonl"))
    return [r["idx"] for r in dp if r["idx"] in sj]


def cell_slake_open():
    idxs = _open_order("slake_open")
    d = json.load(open(os.path.join(SLAKE_OPEN_ROOT, "test.json"), "r", encoding="utf-8"))
    by = {x["qid"]: x for x in d}
    miss = sum(1 for i in idxs if i not in by)
    return dict(cluster=[by[i]["img_name"] for i in idxs], question=[by[i]["question"] for i in idxs],
                order_check=dict(rows=len(idxs), qid_missing=miss),
                image_path=[os.path.join(SLAKE_OPEN_ROOT, "imgs", by[i]["img_name"]) for i in idxs])


def _parquet_meta(df, keep_rows, want_thumb):
    from PIL import Image
    cl, th = [], []
    for i in keep_rows:
        b = df.iloc[i]["image"]["bytes"]
        im = Image.open(io.BytesIO(b))
        cl.append(md5_rgb(im))
        th.append(thumb_feats(im) if want_thumb else None)
    return cl, th


def cell_vqarad(kind):
    df, fs = load_parquet_concat(VQARAD_PARQ)
    if kind == "closed":
        r7 = IM.load_raw("lingshu7b_full", "VQA_RAD")
        assert len(df) == len(r7), (len(df), len(r7))
        keep = [i for i in range(len(r7)) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
        bad = sum(1 for i in keep if str(df.iloc[i]["question"]).strip() != str(r7[i].get("question", "")).strip())
        q = [str(df.iloc[i]["question"]) for i in keep]
        oc = dict(rows=len(keep), question_mismatch=bad, parquet_files=[os.path.basename(f) for f in fs])
    else:
        keep = _open_order("vqa_rad_open")
        bad = 0
        q = [str(df.iloc[i]["question"]) for i in keep]
        oc = dict(rows=len(keep), max_idx=int(max(keep)), n_parquet=len(df))
    cl, th = _parquet_meta(df, keep, True)
    return dict(cluster=cl, question=q, order_check=oc, thumb=[t.tolist() for t in th])


def cell_pathvqa(kind):
    df, fs = load_parquet_concat(PATHVQA_PARQ)
    if kind == "closed":
        r7 = IM.load_raw("lingshu7b_full", "PATH_VQA")
        assert len(df) == len(r7), (len(df), len(r7))
        keep = [i for i in range(len(r7)) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
        bad = sum(1 for i in keep if str(df.iloc[i]["question"]).strip() != str(r7[i].get("question", "")).strip())
        q = [str(df.iloc[i]["question"]) for i in keep]
        oc = dict(rows=len(keep), question_mismatch=bad, parquet_files=[os.path.basename(f) for f in fs])
    else:
        keep = _open_order("pathvqa_open")
        bad = 0
        q = [str(df.iloc[i]["question"]) for i in keep]
        oc = dict(rows=len(keep), max_idx=int(max(keep)), n_parquet=len(df))
    cl, th = _parquet_meta(df, keep, True)
    return dict(cluster=cl, question=q, order_check=oc, thumb=[t.tolist() for t in th])


def cell_medxpert():
    r7 = IM.load_raw("lingshu7b_full", "MedXpertQA-MM")
    from datasets import load_dataset
    os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", cache_dir=os.path.join(MEKDATA, "MedXpertQA"))["test"]
    n = len(r7)
    assert len(ds) >= n, (len(ds), n)
    cl, q, ip = [], [], []
    bad = 0
    for i in range(n):
        s = ds[i]
        imgs = s.get("images") or []
        cl.append("|".join(sorted(imgs)) if imgs else f"__noimg__{i}")
        q.append(s["question"])
        ip.append([os.path.join(MEKDATA, "MedXpertQA/images", x) for x in imgs])
        if str(s["question"]).strip()[:60] not in str(r7[i].get("prompt", "")):
            bad += 1
    return dict(cluster=cl, question=q, order_check=dict(rows=n, question_not_in_prompt=bad), image_path=ip)


BUILDERS = {
    "PMC_VQA": cell_pmc,
    "SLAKE_closed": cell_slake_closed,
    "VQA_RAD_closed": lambda: cell_vqarad("closed"),
    "PATH_VQA_closed": lambda: cell_pathvqa("closed"),
    "MedXpertQA-MM": cell_medxpert,
    "SLAKE_open": cell_slake_open,
    "VQA_RAD_open": lambda: cell_vqarad("open"),
    "PATH_VQA_open": lambda: cell_pathvqa("open"),
}

EXPECT_N = {"PMC_VQA": 33430, "SLAKE_closed": 836, "VQA_RAD_closed": 251, "PATH_VQA_closed": 3362,
            "MedXpertQA-MM": 2000, "SLAKE_open": 645, "VQA_RAD_open": 200, "PATH_VQA_open": 1500}


def main(only=None):
    for cell, fn in BUILDERS.items():
        if only and cell not in only:
            continue
        p = os.path.join(OUTDIR, f"meta_{cell}.json")
        if os.path.exists(p):
            print(f"SKIP(done) {cell}")
            continue
        print(f">> {cell}", flush=True)
        d = fn()
        assert len(d["cluster"]) == EXPECT_N[cell], (cell, len(d["cluster"]), EXPECT_N[cell])
        d["cell"] = cell
        d["n"] = len(d["cluster"])
        d["n_clusters"] = len(set(d["cluster"]))
        json.dump(d, open(p, "w"))
        print(f"   n={d['n']} clusters={d['n_clusters']} check={d['order_check']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
