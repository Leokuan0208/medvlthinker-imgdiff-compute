import json, os, glob, sys
import numpy as np
from collections import Counter
FE = "/home/jamesyang/medvlthinker-imgdiff-compute/feats_hidden"
stems = (["generator_train_finelayer"]
         + [f"generator_eval_finelayer_{c}" for c in
            ["pathvqa_open", "radimagenet_open", "kvasir_x1_open", "omnimed_open",
             "vqamed_open", "gemex_open"]]
         + ["generator_eval_finelayer"])
print("stem | n_rows | n_tok<=0 | n_failed(meta) | empty-na | uniq ds")
tot_bad = 0
for st in stems:
    mp = f"{FE}/{st}.meta.json"
    if not os.path.exists(mp):
        print(f"{st}: MISSING")
        continue
    m = json.load(open(mp))
    rows = m["rows"]
    bad = [r for r in rows if r.get("n_tok", -1) <= 0]
    emp = [r for r in rows if not str(r.get("na", "")).strip()]
    ds = sorted({r.get("ds") for r in rows})
    print(f"{st:48s} {len(rows):7d} {len(bad):6d} {str(m.get('n_failed')):>6} {len(emp):6d}  {ds}")
    tot_bad += len(bad)
    if bad:
        print("     n_tok<=0 by ds:", Counter(r.get('ds') for r in bad))
        print("     first err:", str(bad[0].get("err"))[:160])
    if emp:
        print("     EMPTY na rows by ds:", Counter(r.get('ds') for r in emp))
print("TOTAL n_tok<=0 rows:", tot_bad)
