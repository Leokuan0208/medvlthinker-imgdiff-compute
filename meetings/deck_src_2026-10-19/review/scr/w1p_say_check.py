"""Read-only variant of say_check: strip '# label' lines on BOTH sides, and pool each paper's method + results slides
(sid and sid_res) so numbers that moved from a method slide to its results slide are not counted as lost."""
import json, os, re, sys
HERE = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19"
sys.path.insert(0, HERE)
from deck_lib import Deck
import deck_part1, deck_part2, deck_part3
NUM = re.compile(r"[−-]?\d[\d,]*(?:\.\d+)?")


def body(t):
    return "\n".join(l for l in t.split("\n") if not l.lstrip().startswith("# "))


def nums(t):
    return sorted(n.replace("−", "-").replace(",", "") for n in NUM.findall(body(t)))


d = Deck()
for p in (deck_part1, deck_part2, deck_part3):
    p.build(d)
new = {}
for s in d.script:
    k = s["sid"][:-4] if s["sid"].endswith("_res") else s["sid"]
    new[k] = new.get(k, "") + "\n" + (s["say"] or "")
old = json.load(open(f"{HERE}/say_snapshot.json"))
bad = 0
for k in new:
    a, b = nums(old.get(k, "")), nums(new[k])
    lost = [x for x in set(a) if a.count(x) > b.count(x)]
    added = [x for x in set(b) if b.count(x) > a.count(x)]
    if lost or added:
        bad += 1
        print(f"{k}: lost={sorted(lost)} added={sorted(added)}")
print(len(new), "slide groups,", bad, "with differences")
