#!/usr/bin/env python3
"""part3_rows.py — append the training-row breakdown (from a re-run of review/rr_rows.py) to numbers_review_part3.json."""
import json
p = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part3.json"
d = json.load(open(p))
f0 = {'pathvqa_open': 8220, 'slake_open': 635, 'vqa_rad_open': 365, 'radimagenet_open': 4084, 'kvasir_x1_open': 23221, 'omnimed_open': 16963,
      'vqamed_open': 11991, 'gemex_open': 15852, 'vqa_rad_closed': 167, 'pathvqa_closed': 2246, 'slake_closed': 251, 'vqamed2019_test': 702}
f1 = {'pathvqa_open': 7899, 'slake_open': 678, 'vqa_rad_open': 344, 'radimagenet_open': 4071, 'kvasir_x1_open': 24397, 'omnimed_open': 17057,
      'vqamed_open': 11500, 'gemex_open': 15608, 'vqa_rad_closed': 180, 'pathvqa_closed': 2123, 'slake_closed': 273, 'vqamed2019_test': 749}
CL = ['vqa_rad_closed', 'pathvqa_closed', 'slake_closed', 'vqamed2019_test']


def summ(per, kept, dropped, total):
    return {"total": total, "extra_domains_kept": kept, "extra_dropped_image_overlap": dropped,
            "open8_other_half": sum(v for k, v in per.items() if k not in CL),
            "closed3_plus_vqamed2019_other_half": sum(per[k] for k in CL), "per_cell": per}


d["training_rows_by_source"] = {
    "value": {"extra_domains_rows_total": 31498,
              "extra_domains_by_source": {"pathvqa_open_train": 22390, "kvasir_open": 5562, "slake_open_train": 2207, "vqa_rad_open_train": 1339},
              "fold0": summ(f0, 31237, 261, 115934), "fold1": summ(f1, 31121, 377, 116000)},
    "computed_by": "review/rr_rows.py (re-run 2026-10-19 by deck worker 3; per-cell counts copied from its output)",
    "inputs": "feats_hidden/generator_{train,eval}_finelayer*.meta.json; totals equal final_eval_FINAL_cap320_T07.json folds.*.train_rows"}
for f in ("fold0", "fold1"):
    v = d["training_rows_by_source"]["value"][f]
    assert v["extra_domains_kept"] + v["open8_other_half"] + v["closed3_plus_vqamed2019_other_half"] == v["total"], f
json.dump(d, open(p, "w"), indent=1)
print({f: {k: v for k, v in d["training_rows_by_source"]["value"][f].items() if k != "per_cell"} for f in ("fold0", "fold1")})
