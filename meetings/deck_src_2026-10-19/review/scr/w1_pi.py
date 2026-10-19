"""Positive rate pi of each fold's probe-training rows and the BCE pos_weight (1-pi)/pi, reproducing the row selection
of final_method_eval.py (train = original-domain or opposite-half rows, minus every image of the scored half), following
review/rr_rows.py (which reproduces folds.*.train_rows 115,934 / 116,000). Appends to numbers_review_part1.json."""
import json, hashlib
F = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part1.json"


def half(i):
    return int(hashlib.md5(("nd" + str(i)).encode()).hexdigest(), 16) % 2


def rows(stem, dsf=None):
    m = json.load(open(f"{F}/{stem}.meta.json"))
    return [r for r in m["rows"] if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]


TD = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
tr = rows("generator_train_finelayer", TD)
cells = ['pathvqa_open', 'slake_open', 'vqa_rad_open', 'radimagenet_open', 'kvasir_x1_open', 'omnimed_open',
         'vqamed_open', 'gemex_open', 'vqa_rad_closed', 'pathvqa_closed', 'slake_closed', 'vqamed2019_test']
C = {c: (rows("generator_eval_finelayer", {c}) if c in {"slake_open", "vqa_rad_open"} else rows(f"generator_eval_finelayer_{c}"))
     for c in cells}
res = {}
for f in (0, 1):
    evimg = set()
    for c in cells:
        evimg |= {r['img_md5'] for r in C[c] if half(r['img_md5']) == f}
    keep = [r for r in tr if r['img_md5'] not in evimg]
    for c in cells:
        keep += [r for r in C[c] if half(r['img_md5']) == 1 - f and r['img_md5'] not in evimg]
    pi = sum(int(r['y']) for r in keep) / len(keep)
    res[f] = {"rows": len(keep), "pi": round(pi, 4), "pos_weight": round((1 - pi) / pi, 2)}
print(res)
out = json.load(open(OUT))
out["probe_training_positive_rate"] = {"value": {str(k): v for k, v in res.items()}, "computed_by": __file__,
                                       "inputs": [F + "/generator_train_finelayer.meta.json", F + "/generator_eval_finelayer*.meta.json",
                                                  "row filter as in src/exp_2026_10/final_method_eval.py:274"],
                                       "note": "pos_weight = (1-pi)/pi is what head_sweep.py:172-174 computes per fit"}
json.dump(out, open(OUT, "w"), indent=1)
