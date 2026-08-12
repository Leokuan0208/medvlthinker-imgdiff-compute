#!/usr/bin/env python3
"""
pregen_router_features.py -- ATTACK 2 (PRE-GENERATION ROUTER), stage 1: build the PRE-GENERATION
feature cache for all 8 Variant-B reporting cells, item-aligned to
    results/cascade_methods/artifacts/_selector_rerun_parts/vec_disjoint.npz

EVERY feature in here is computable BEFORE any model is run: the question text, the option strings,
the prompt-only format flag (reused from unified_router.detect_format), and statistics of the DECODED
image.  Nothing derived from the 7B's output (response / margin / conf / cum_logprob / gen_toks /
latency) is read.  That is the whole point of the attack: a cascade pays the cheap model on every
question before it can decide; a pre-generation router must decide without it.

ALIGNMENT is asserted, not assumed: for every cell the 7B and 32B correctness vectors are rebuilt
from the raw dumps in the same order the features are emitted, and asserted identical to
vec_disjoint.npz.  If any cell drifts, the script dies.

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/pregen_router_features.py
Writes results/cascade_methods/artifacts/_pregen_router_parts/features_<cell>.npz  (+ texts_<cell>.json)
"""
import os, sys, io, re, json, csv, glob, hashlib
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(REPO, "src/cascade_methods"))
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
PARTS = os.path.join(ART, "_pregen_router_parts")
MEK = os.path.join(REPO, "MedEvalKit")
os.makedirs(PARTS, exist_ok=True)

from unified_router import detect_format          # prompt-only format detector, reused verbatim

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
         "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

Z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"))


# ===================================================================================================
# raw dump loaders (same code path as integrated_method.py, so the ordering is the published one)
# ===================================================================================================
def load_raw(tag, ds):
    p = f"{MEK}/eval_results_{tag}/{{}}/{ds}/results.json"
    return json.load(open(p)) if os.path.exists(p) else None


def as_ok(r):
    v = r.get("correct")
    return int(v is True or str(v).strip().lower() in ("true", "1"))


def load_judge(p):
    m = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            m[int(r["idx"])] = int(r["judge_ok"])
    return m


# ===================================================================================================
# IMAGE features -- decoded, 64x64 grayscale + a 96x96 RGB thumbnail for colour
# ===================================================================================================
IMG_NAMES = ["img_w", "img_h", "img_logarea", "img_aspect", "img_gray_mean", "img_gray_std",
             "img_entropy", "img_edge", "img_dark_frac", "img_bright_frac", "img_colorfulness",
             "img_ok"]


def img_feats(pil):
    """12 statistics of the decoded image.  img_ok=0 and everything else 0 if the image is missing."""
    if pil is None:
        return np.zeros(len(IMG_NAMES), np.float64)
    w, h = pil.size
    g = np.asarray(pil.convert("L").resize((64, 64), Image.BILINEAR), np.float64) / 255.0
    hist = np.bincount((g * 255).astype(np.int64).ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    gx = np.abs(np.diff(g, axis=1)).mean()
    gy = np.abs(np.diff(g, axis=0)).mean()
    rgb = np.asarray(pil.convert("RGB").resize((32, 32), Image.BILINEAR), np.float64) / 255.0
    colorf = float(rgb.std(axis=2).mean())
    return np.array([w, h, np.log(max(w * h, 1)), w / max(h, 1),
                     float(g.mean()), float(g.std()), ent, float(gx + gy),
                     float((g < 0.05).mean()), float((g > 0.95).mean()), colorf, 1.0], np.float64)


def open_img(path):
    try:
        return Image.open(path)
    except Exception:
        return None


# ===================================================================================================
# per-cell builders -> (texts, options_list, dense_extra, images_iterable, ok7, ok32)
# ===================================================================================================
def build_pmc():
    ds = "PMC_VQA"
    r7, r32 = load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds)
    n = min(len(r7), len(r32))
    base = "/data/dan/dataset/medevalkit/PMC-VQA"
    rows = list(csv.reader(open(os.path.join(base, "test_2.csv"), encoding="utf-8")))[1:]
    assert len(rows) == len(r7), (len(rows), len(r7))
    texts, opts, imgs = [], [], []
    for i in range(n):
        q = rows[i][3]
        texts.append(str(q))
        opts.append([str(x) for x in (r7[i].get("choices") or [])])
        p = os.path.join(base, "figures", rows[i][1])
        if not os.path.exists(p):
            p = os.path.join(base, "images", rows[i][1])
        imgs.append(p)
    ok7 = np.array([as_ok(r7[i]) for i in range(n)], float)
    ok32 = np.array([as_ok(r32[i]) for i in range(n)], float)
    return texts, opts, imgs, ok7, ok32


def build_slake(closed=True):
    """SLAKE_closed: results.json order filtered by answer_type==CLOSED, images from test.json order.
    SLAKE_open: the open-text arm's own item order (transfer dump idx == SLAKE qid)."""
    d = json.load(open("/data/dan/dataset/medevalkit/SLAKE/test.json"))
    root = "/data/dan/dataset/medevalkit/SLAKE/imgs"
    if closed:
        ds = "SLAKE"
        r7, r32 = load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds)
        n = min(len(r7), len(r32))
        assert len(d) == len(r7), (len(d), len(r7))
        idx = [i for i in range(n) if r7[i].get("answer_type") == "CLOSED"]
        texts = [str(r7[i].get("question", "")) for i in idx]
        imgs = [os.path.join(root, d[i]["img_name"]) for i in idx]
        ok7 = np.array([as_ok(r7[i]) for i in idx], float)
        ok32 = np.array([as_ok(r32[i]) for i in idx], float)
        return texts, [[] for _ in idx], imgs, ok7, ok32
    # ---- open ----
    by_qid = {x["qid"]: x for x in d}
    dp = os.path.join(REPO, "ckpts/train/lora_verifier_disjoint/transfer_dump_slake_open_lingshu7b.json")
    sj = load_judge(os.path.join(REPO, "ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b.judge.jsonl"))
    texts, imgs, ok7, ok32 = [], [], [], []
    for r in json.load(open(dp)):
        i = r["idx"]
        if i not in sj:
            continue
        x = by_qid[i]
        texts.append(str(x["question"]))
        imgs.append(os.path.join(root, x["img_name"]))
        ok7.append(int(r["greedy_ok"]))
        ok32.append(int(sj[i]))
    return texts, [[] for _ in texts], imgs, np.array(ok7, float), np.array(ok32, float)


_PARQ = {}


def parquet_test(name):
    """flaviagiammarino/{vqa-rad,path-vqa} test split, sorted shards, exactly as run_openvqa.py
    and MedEvalKit's PATH_VQA/VQA_RAD loaders read them."""
    if name in _PARQ:
        return _PARQ[name]
    import pandas as pd
    base = f"/data/dan/dataset/{name}/data"
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))]
    df = pd.concat(dfs, ignore_index=True)
    _PARQ[name] = df
    return df


def _pil_from_row(cellval):
    if isinstance(cellval, dict) and "bytes" in cellval:
        try:
            return Image.open(io.BytesIO(cellval["bytes"]))
        except Exception:
            return None
    return None


def build_parquet_closed(cell):
    ds = "VQA_RAD" if cell == "VQA_RAD_closed" else "PATH_VQA"
    name = "vqa_rad" if cell == "VQA_RAD_closed" else "path_vqa"
    r7, r32 = load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds)
    n = min(len(r7), len(r32))
    df = parquet_test(name)
    assert len(df) == len(r7), (cell, len(df), len(r7))
    idx = [i for i in range(n) if str(r7[i].get("answer", "")).strip().lower() in ("yes", "no")]
    texts = [str(r7[i].get("question", "")) for i in idx]
    imgs = [_pil_from_row(df.iloc[i]["image"]) for i in idx]
    ok7 = np.array([as_ok(r7[i]) for i in idx], float)
    ok32 = np.array([as_ok(r32[i]) for i in idx], float)
    return texts, [[] for _ in idx], imgs, ok7, ok32


def build_parquet_open(cell):
    name = "vqa_rad" if cell == "VQA_RAD_open" else "path_vqa"
    dskey = "vqa_rad_open" if cell == "VQA_RAD_open" else "pathvqa_open"
    df = parquet_test(name)
    dp = os.path.join(REPO, f"ckpts/train/lora_verifier_disjoint/transfer_dump_{dskey}_lingshu7b.json")
    sj = load_judge(os.path.join(REPO, f"ckpts/openvqa/strong_lingshu/ckpt_{dskey}_lingshu32b.judge.jsonl"))
    cheap = {}
    cp = os.path.join(REPO, f"ckpts/openvqa/cheap_lingshu7b/ckpt_{dskey}_lingshu7b.jsonl")
    for l in open(cp):
        if l.strip():
            r = json.loads(l)
            cheap[int(r["idx"])] = r.get("question", "")
    texts, imgs, ok7, ok32 = [], [], [], []
    for r in json.load(open(dp)):
        i = r["idx"]
        if i not in sj:
            continue
        row = df.iloc[i]
        q = row.get("question")
        if q is None and "conversations" in row:
            q = row["conversations"][0]["value"].replace("<image>", "").strip()
        if not q:
            q = cheap.get(i, "")
        texts.append(str(q))
        imgs.append(_pil_from_row(row["image"]))
        ok7.append(int(r["greedy_ok"]))
        ok32.append(int(sj[i]))
    return texts, [[] for _ in texts], imgs, np.array(ok7, float), np.array(ok32, float)


def build_medxpert():
    ds = "MedXpertQA-MM"
    r7, r32 = load_raw("lingshu7b_full", ds), load_raw("lingshu32b_full", ds)
    n = min(len(r7), len(r32))
    root = "/data/dan/dataset/medevalkit/MedXpertQA/images"
    # MedEvalKit names the files "<id>-a.<ext>", "<id>-b.<ext>", ... (MedXpertQA.py:59-63).
    # The first image is the one the router sees; multi-image items are flagged by t_nimg.
    texts, opts, imgs = [], [], []
    for i in range(n):
        texts.append(str(r7[i].get("question", "")))
        opts.append([str(x) for x in (r7[i].get("choices") or [])])
        g = sorted(glob.glob(os.path.join(root, str(r7[i].get("id")) + "-*")))
        imgs.append(g[0] if g else None)
    ok7 = np.array([as_ok(r7[i]) for i in range(n)], float)
    ok32 = np.array([as_ok(r32[i]) for i in range(n)], float)
    return texts, opts, imgs, ok7, ok32


BUILDERS = {
    "PMC_VQA": build_pmc,
    "SLAKE_closed": lambda: build_slake(True),
    "SLAKE_open": lambda: build_slake(False),
    "VQA_RAD_closed": lambda: build_parquet_closed("VQA_RAD_closed"),
    "PATH_VQA_closed": lambda: build_parquet_closed("PATH_VQA_closed"),
    "VQA_RAD_open": lambda: build_parquet_open("VQA_RAD_open"),
    "PATH_VQA_open": lambda: build_parquet_open("PATH_VQA_open"),
    "MedXpertQA-MM": build_medxpert,
}


# ===================================================================================================
# TEXT features (hand-built, dense) -- everything is a property of the QUESTION STRING
# ===================================================================================================
WH = ["what", "which", "where", "how", "why", "when", "who", "name", "describe", "identify"]
YN = ["is", "are", "does", "do", "did", "was", "were", "has", "have", "had", "can", "could",
      "should", "will", "would", "may", "might"]
KEY = ["modality", "organ", "plane", "abnormal", "normal", "present", "show", "sign", "diagnos",
       "stain", "artery", "vein", "left", "right", "lesion", "tumor", "cancer", "cell", "tissue",
       "ct", "mri", "x-ray", "xray", "ultrasound", "image", "patient", "year-old", "history",
       "treatment", "management", "next step", "most likely", "located", "location", "count",
       "number", "color", "size", "largest", "type", "function", "cause", "effect"]

TXT_NAMES = (["t_nchar", "t_nword", "t_meanwlen", "t_maxwlen", "t_fraclong", "t_ttr", "t_nqmark",
              "t_ndigit", "t_ncomma", "t_nsent", "t_nupper", "t_nparen", "t_is_mcq", "t_nopt",
              "t_optmean", "t_optmax", "t_optstd", "t_yn_lead"]
             + [f"t_wh_{w}" for w in WH] + [f"t_kw_{k.replace(' ', '_').replace('-', '_')}" for k in KEY])

_word = re.compile(r"[A-Za-z][A-Za-z\-']+")


def text_feats(q, options):
    ql = q.lower()
    ws = _word.findall(ql)
    wl = np.array([len(w) for w in ws], float) if ws else np.zeros(1)
    ol = np.array([len(o) for o in options], float) if options else np.zeros(1)
    first = ql.strip().split()[0] if ql.strip() else ""
    v = [len(q), len(ws), wl.mean(), wl.max(), float((wl > 8).mean()),
         (len(set(ws)) / max(len(ws), 1)), q.count("?"), sum(c.isdigit() for c in q),
         q.count(","), max(q.count(".") + q.count("?") + q.count("!"), 1),
         sum(c.isupper() for c in q), q.count("("),
         float(len(options) >= 2), float(len(options)),
         ol.mean(), ol.max(), ol.std(),
         float(first.strip("?,.") in YN)]
    v += [float(w in ws[:3]) if ws else 0.0 for w in WH]
    v += [float(k in ql) for k in KEY]
    return np.array(v, np.float64)


def main():
    only = sys.argv[1:]                       # optional: rebuild only these cells
    manifest = {}
    if os.path.exists(os.path.join(PARTS, "manifest.json")):
        manifest = json.load(open(os.path.join(PARTS, "manifest.json")))
    for cell in (only or CELLS):
        print(f"[{cell}] building ...", flush=True)
        texts, opts, imgs, ok7, ok32 = BUILDERS[cell]()
        v7 = Z[f"{cell}|always_7b"].astype(float)
        v32 = Z[f"{cell}|always_32b_direct"].astype(float)
        assert len(ok7) == len(v7), (cell, len(ok7), len(v7))
        d7 = float(np.abs(ok7 - v7).max())
        d32 = float(np.abs(ok32 - v32).max())
        assert d7 == 0.0 and d32 == 0.0, (cell, d7, d32)

        T = np.stack([text_feats(t, o) for t, o in zip(texts, opts)])
        fmt = np.array([1.0 if detect_format({"question": t, "choices": o}) == "mcq" else
                        (0.5 if detect_format({"question": t, "choices": o}) == "closed" else 0.0)
                        for t, o in zip(texts, opts)], np.float64)[:, None]

        I = np.empty((len(texts), len(IMG_NAMES)))
        nmiss = 0
        for i, im in enumerate(imgs):
            pil = open_img(im) if isinstance(im, str) else im
            if pil is None:
                nmiss += 1
            I[i] = img_feats(pil)
            if isinstance(im, str) and pil is not None:
                pil.close()
            if (i + 1) % 5000 == 0:
                print(f"    images {i+1}/{len(imgs)}", flush=True)

        X = np.hstack([T, fmt, I])
        names = TXT_NAMES + ["t_fmt_detect"] + IMG_NAMES
        assert X.shape[1] == len(names), (X.shape, len(names))
        np.savez_compressed(os.path.join(PARTS, f"features_{cell}.npz"),
                            X=X, ok7=ok7, ok32=ok32, names=np.array(names))
        json.dump(texts, open(os.path.join(PARTS, f"texts_{cell}.json"), "w"))
        manifest[cell] = dict(n=len(texts), n_features=int(X.shape[1]), images_missing=nmiss,
                              align_dev_ok7=d7, align_dev_ok32=d32,
                              acc_7b=round(float(ok7.mean()), 6), acc_32b=round(float(ok32.mean()), 6),
                              text_md5=hashlib.md5("\x00".join(texts).encode("utf8")).hexdigest())
        print(f"  n={len(texts)} feats={X.shape[1]} imgs_missing={nmiss} "
              f"a7={ok7.mean():.4f} a32={ok32.mean():.4f}", flush=True)
    json.dump(manifest, open(os.path.join(PARTS, "manifest.json"), "w"), indent=1)
    print("\nwrote", os.path.join(PARTS, "manifest.json"))


if __name__ == "__main__":
    main()
