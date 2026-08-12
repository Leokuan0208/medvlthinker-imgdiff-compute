#!/usr/bin/env python3
"""extract_generator_hidden_ablated.py -- re-extract the LANGUAGE-SIDE generator hidden states with
the IMAGE DESTROYED, so the language-side bar can be image-ablated exactly like the vision arms.

WHY THIS EXISTS.  Attack 1's premise is that the verifier "is not really looking at the image".
The vision arms test whether an EXPLICIT vision readout helps.  This script tests the prior
question, which turns out to be the decisive one: does the LANGUAGE-side vector -- the input to the
bar, and to ~20 previously tested verifier architectures -- already carry the image?

In Qwen2.5-VL the image precedes the question and the answer, so the answer tokens ATTEND to the
vision tokens.  h_span at layer 21 over the answer span is therefore a function of the image.  If
replacing the image with noise collapses the language-side head, the language side was never
"vision-blind" and the premise of the attack is wrong.  If it does not collapse it, the head is a
text-prior scorer and explicit vision injection has something to fix.

HOW IT AVOIDS DIVERGING FROM THE PUBLISHED CACHE.  It does not reimplement the extraction loop.  It
imports extract_generator_hidden, monkeypatches ONLY the row builders so that each row's "img" is
replaced by an ablated PIL image of IDENTICAL pixel dimensions (so the image-token count, the merged
grid and every sequence length are unchanged), and then calls that module's own main().  Readout
positions, span offsets, dtypes, memoisation and the meta schema are therefore identical by
construction; the only difference in the whole pipeline is the pixel content.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 -u \
    src/training_methods/extract_generator_hidden_ablated.py --ablate noise \
    --mode generator --split eval --shard 0 --nshard 2 --out feats_hidden_noise
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import extract_generator_hidden as EG  # noqa: E402


def ablate_image(im, kind, seed):
    """Same width/height, same resulting patch grid, no image information."""
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.convert("RGB")
    w, h = im.size
    if kind == "blank":
        return Image.new("RGB", (w, h), (128, 128, 128))
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")
    raise ValueError(kind)


def patch_rows(kind):
    """Wrap EG's row builders so every row carries an ablated image. The seed is derived from the
    ORIGINAL image's decoded-RGB md5, so the ablation is deterministic and per-image."""
    orig_eval, orig_train = EG.build_eval_rows, EG.build_train_rows

    def wrap(fn):
        def inner(*a, **kw):
            rows = fn(*a, **kw)
            cache = {}
            for r in rows:
                m = EG.img_md5(r["img"])
                if m not in cache:
                    cache[m] = ablate_image(r["img"], kind, seed=int(m[:8], 16))
                r["img"] = cache[m]
            print(f"[ablate] {kind}: {len(rows)} rows over {len(cache)} unique images", flush=True)
            return rows
        return inner

    EG.build_eval_rows = wrap(orig_eval)
    EG.build_train_rows = wrap(orig_train)


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ablate", choices=["blank", "noise"], required=True)
    known, rest = ap.parse_known_args()
    patch_rows(known.ablate)
    sys.argv = [sys.argv[0]] + rest
    EG.main()


if __name__ == "__main__":
    main()
