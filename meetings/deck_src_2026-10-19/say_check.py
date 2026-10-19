#!/usr/bin/env python3
"""say_check.py snapshot|check — guard for rewriting the spoken script.
snapshot: save every slide's `say` text to say_snapshot.json.
check:    compare the current `say` texts with the snapshot, slide by slide: every number must survive (same multiset),
          and every say must follow the titled-bullet format ("# label" groups, "- " bullets)."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from deck_lib import Deck
import deck_part1, deck_part2, deck_part3

NUM = re.compile(r"[−-]?\d[\d,]*(?:\.\d+)?")


def says():
    d = Deck()
    for p in (deck_part1, deck_part2, deck_part3):
        p.build(d)
    return {s["sid"] if "sid" in s else str(s["n"]): s["say"] or "" for s in d.script}


def nums(t):
    return sorted(n.replace("−", "-").replace(",", "") for n in NUM.findall(t))


if sys.argv[1] == "snapshot":
    json.dump(says(), open(f"{HERE}/say_snapshot.json", "w"), indent=1, ensure_ascii=False)
    print("snapshot saved")
else:
    old, new = json.load(open(f"{HERE}/say_snapshot.json")), says()
    bad = 0
    for k, t in new.items():
        o = old.get(k, "")
        # group labels copy on-screen text, so their numbers are not spoken claims: count bullets only
        body = "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("# "))
        a, b = nums(o), nums(body)
        lost = [x for x in set(a) if a.count(x) > b.count(x)]
        added = [x for x in set(b) if b.count(x) > a.count(x)]
        lines = [l for l in t.strip().split("\n") if l.strip()]
        fmt_ok = not t.strip() or (any(l.lstrip().startswith("# ") for l in lines)
                                   and all(l.lstrip().startswith(("# ", "- ")) for l in lines))
        if lost or added or not fmt_ok:
            bad += 1
            print(f"{k}: lost={lost} added={added} format_ok={fmt_ok}")
    print(f"{len(new)} slides checked, {bad} with problems")
