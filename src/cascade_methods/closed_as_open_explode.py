#!/usr/bin/env python3
"""closed_as_open_explode.py -- BUILD 3: turn every arm's generations into ONE judge worklist per cell.

One row per (item, DISTINCT normalised answer) pooled over ALL arms, so an answer string produced by
two different arms is judged ONCE and both arms inherit the same label -- no arm can be advantaged by
judge noise.  Field names match src/labeling/run_judge.py (idx, question, gold, modal_pred).

THE KEY IS CONTENT-ADDRESSED, NOT POSITIONAL: idx = "<item>||<md5(normalised answer)[:12]>".  A
positional "<item>||<j>" would silently RE-LABEL every answer the moment a later arm adds a new
distinct string to an item, because run_judge.py resumes by idx.  With a content hash, re-exploding
after more arms have finished is safe and the judge only has to label the genuinely new rows.

CPU only.  python3 src/cascade_methods/closed_as_open_explode.py
"""
import hashlib
import json
import os
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402


def main(cells=None):
    for cell in (cells or L.CELLS):
        byq, meta, arms_seen = OrderedDict(), {}, []
        for arm in L.ARMS:
            g = L.load_gen(cell, arm)
            if not g:
                continue
            arms_seen.append((arm, len(g)))
            for i in sorted(g):
                r = g[i]
                meta.setdefault(i, (r["question"], r["gold"]))
                d = byq.setdefault(i, OrderedDict())
                for a in r["preds"]:
                    d.setdefault(L.norm_text(a), a)      # keep the first surface form seen
        out = L.explode_path(cell)
        n = 0
        with open(out, "w", encoding="utf-8") as fh:
            for i, d in byq.items():
                q, gold = meta[i]
                for na, surf in d.items():
                    h = hashlib.md5(na.encode("utf-8")).hexdigest()[:12]
                    fh.write(json.dumps({"idx": f"{i}||{h}", "i": i, "na": na, "question": q,
                                         "gold": gold, "modal_pred": surf},
                                        ensure_ascii=False) + "\n")
                    n += 1
        print(f"{cell}: arms={arms_seen} items={len(byq)} -> {n} unique (item,answer) judge rows "
              f"({n/max(len(byq),1):.2f}/item) -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
