#!/usr/bin/env python3
"""build_sft_manifest.py -- generator fine-tuning (LoRA-SFT on gold answers) on the probe's OWN data universe.

Owed baseline from the 2026-10-19 deck ("32B and fine-tuning baselines"). The only earlier generator SFT
(+0.0142 greedy at 1x cost, ckpts/cheapleg/scores_*) used three benchmarks, one seed and a different split. This
builds, per cross-fitting fold f, the question set the probe's fitting rows come from -- the four original train
domains' questions (those in generator_train_finelayer) + every open and comparability cell's questions from the
OTHER image half -- minus every image scored in fold f (crossfit_data, verified against the FINAL run's row
counts), with each question's GOLD answer. Items use train_cheap_leg.py's "open_generic" source: run_openvqa's
prompt at cap320, image by path or by (parquet base, row).

  python3 src/exp_2026_10/build_sft_manifest.py   -> data/sft_crossfit_2026-10-22/fold{0,1}.json
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crossfit_data as CD                           # noqa: E402

PARQ = {"pathvqa_open": "/data/dan/dataset/path_vqa/data/test", "vqa_rad_open": "/data/dan/dataset/vqa_rad/data/test",
        "pathvqa_open_train": "/data/dan/dataset/path_vqa/data/train", "vqa_rad_open_train": "/data/dan/dataset/vqa_rad/data/train"}
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/data/sft_crossfit_2026-10-22"
os.makedirs(OUT, exist_ok=True)
ev = CD.eval_images()
for f in (0, 1):
    seen, items = set(), []
    for ds, r in CD.fit_rows(f, ev):
        if (ds, r["idx"]) in seen:
            continue
        seen.add((ds, r["idx"]))
        it = CD.items(ds).get(r["idx"])
        if it is None:
            continue
        q, gold, img = it
        e = {"source": "open_generic", "frame": "openvqa", "ds": ds, "idx": r["idx"], "imghash": r["img_md5"],
             "question": q, "answer": gold}
        if isinstance(img, str):
            e["img"] = img
        else:
            assert ds in PARQ, f"{ds}: PIL image without a known parquet base"; e["parquet"] = PARQ[ds]
        items.append(e)
    assert not any(e["imghash"] in ev[f] for e in items), "an evaluation image leaked into the training set"
    json.dump({"open_generic": items}, open(f"{OUT}/fold{f}.json", "w"))
    from collections import Counter
    print(f"fold {f}: {len(items):,} questions; by source {dict(Counter(e['ds'] for e in items).most_common(12))}")
