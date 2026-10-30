#!/usr/bin/env python3
"""crossfit_data.py -- the FINAL method's data universe, for baselines that must be trained on exactly what the
probe was trained on (the LoRA verifier, generator SFT).

Per fold f (the questions scored are those whose image has ER.half(img_md5) == f):
  candidate rows   = the four original train domains (generator_train_finelayer meta rows) + the OTHER image half
                     of every open and comparability cell's T=0.7 cache, minus every image scored in fold f under
                     T07 or Tmap -- final_method_eval.py's fitting set, row for row (labels y = MedVLThinker-32B).
  question / image = run_openvqa.py's own item loader (predict_modality.load_items), by (dataset, idx).
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_method_eval as FME                     # noqa: E402
import predict_modality as PM                       # noqa: E402

ER = FME.ER
FEATS = FME.FEATS
TRAIN_DS = sorted(FME.TRAIN_DOMAINS)                # kvasir_open, pathvqa_open_train, slake_open_train, vqa_rad_open_train
CELLS = FME.OPEN8 + FME.COMP


def meta_rows(stem, dsf=None):
    m = json.load(open(f"{FEATS}/{stem}.meta.json"))
    return [r for r in m["rows"] if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]


def t07_rows(c):
    stem, dsf = FME.eval_stem(c, "lingshu7b_sc8", True)
    return meta_rows(stem, dsf)


def tmap_rows(c):
    tag = FME.pool_tag(c, "Tmap", True)
    stem, dsf = FME.eval_stem(c, tag, True)
    return meta_rows(stem, dsf)


def eval_images():
    ev = {0: set(), 1: set()}
    for c in CELLS:
        for r in t07_rows(c) + tmap_rows(c):
            ev[ER.half(r["img_md5"])].add(r["img_md5"])
    return ev


def fit_rows(f, ev=None):
    """[(source_dataset, row)] -- the probe's fitting set for fold f."""
    ev = ev or eval_images()
    out = [(r["ds"], r) for r in meta_rows("generator_train_finelayer", FME.TRAIN_DOMAINS) if r["img_md5"] not in ev[f]]
    for c in CELLS:
        out += [(c, r) for r in t07_rows(c) if ER.half(r["img_md5"]) == 1 - f and r["img_md5"] not in ev[f]]
    return out


_ITEMS = {}


def items(ds):
    """idx -> (question, gold, image path or PIL)"""
    if ds not in _ITEMS:
        _ITEMS[ds] = {it[0]: (it[1], it[2], it[3]) for it in PM.load_items(ds)}
    return _ITEMS[ds]
