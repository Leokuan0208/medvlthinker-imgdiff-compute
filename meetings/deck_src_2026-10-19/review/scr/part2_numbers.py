#!/usr/bin/env python3
"""part2_numbers.py — recompute the training-set composition, class balance and pool statistics that the
deck's Part III/IV slides quote, from the feature-cache metas that final_method_eval.py reads (read-only).
Reproduces final_eval_FINAL_cap320_T07.json folds.{0,1}.train_rows as a check.
Writes ../../numbers_review_part2.json."""
import json, hashlib, collections, os
F = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
ART = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/results/cascade_methods/artifacts"
OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/numbers_review_part2.json"
SHARED = {"slake_open", "vqa_rad_open"}
TD = {"kvasir_open", "pathvqa_open_train", "slake_open_train", "vqa_rad_open_train"}
OPEN8 = ['pathvqa_open', 'slake_open', 'vqa_rad_open', 'radimagenet_open', 'kvasir_x1_open', 'omnimed_open',
         'vqamed_open', 'gemex_open']
COMP = ['vqa_rad_closed', 'pathvqa_closed', 'slake_closed', 'vqamed2019_test']


def half(i):
    return int(hashlib.md5(("nd" + str(i)).encode()).hexdigest(), 16) % 2


def rows(stem, dsf=None):
    m = json.load(open(f"{F}/{stem}.meta.json"))
    return [r for r in m["rows"] if r.get("n_tok", -1) > 0 and (dsf is None or r.get("ds") in dsf)]


tr = rows("generator_train_finelayer", TD)
C = {c: (rows("generator_eval_finelayer", {c}) if c in SHARED else rows(f"generator_eval_finelayer_{c}"))
     for c in OPEN8 + COMP}
out = {}
fin = json.load(open(f"{ART}/final_eval_FINAL_cap320_T07.json"))
for f in (0, 1):
    evimg = set()
    for c in OPEN8 + COMP:
        evimg |= {r["img_md5"] for r in C[c] if half(r["img_md5"]) == f}
    orig = [r for r in tr if r["img_md5"] not in evimg]
    per = {c: [r for r in C[c] if half(r["img_md5"]) == 1 - f and r["img_md5"] not in evimg] for c in OPEN8 + COMP}
    allr = orig + [r for c in per for r in per[c]]
    pos = sum(r["y"] for r in allr) / len(allr)
    out[f"fold{f}"] = {
        "train_rows_total": len(allr),
        "check_vs_artifact_train_rows": fin["folds"][str(f)]["train_rows"],
        "rows_extra_domains": len(orig),
        "rows_open8_other_half": sum(len(per[c]) for c in OPEN8),
        "rows_comparability_other_half": sum(len(per[c]) for c in COMP),
        "rows_per_comparability_cell": {c: len(per[c]) for c in COMP},
        "positive_rate_pi": round(pos, 4),
        "pos_weight_(1-pi)/pi": round((1 - pos) / pos, 2),
    }
qs = {(r["ds"], r["idx"]) for r in tr}
out["extra_domains"] = {"questions": len(qs), "rows": len(tr),
                        "rows_by_domain": dict(collections.Counter(r["ds"] for r in tr))}
# distinct answers per question on the 8 open benchmarks (final T=0.7 cap320 pools)
nq = nr = n1 = 0
per_b = {}
for c in OPEN8:
    byq = collections.Counter(r["idx"] for r in C[c])
    nq += len(byq); nr += len(C[c]); n1 += sum(1 for v in byq.values() if v == 1)
    per_b[c] = round(sum(1 for v in byq.values() if v == 1) / len(byq), 3)
out["open8_pools"] = {"questions": nq, "distinct_answer_rows": nr, "distinct_per_question": round(nr / nq, 2),
                      "share_questions_one_distinct": round(n1 / nq, 3),
                      "share_one_distinct_by_benchmark": per_b}
json.dump({k: {"value": v, "computed_by": "tmp/deck/review/scr/part2_numbers.py",
               "inputs": "feats_hidden/generator_{train,eval}_finelayer*.meta.json (rows with n_tok>0), "
                         "half = md5('nd'+img_md5) mod 2, filter as final_method_eval.py lines 270-273"}
           for k, v in out.items()}, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
