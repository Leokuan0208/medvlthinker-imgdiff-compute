"""Read-only variant of say_check.py: compares numbers in `say` bodies only, ignoring '# <on-screen label>' lines
(labels must copy on-screen text exactly, so their digits are not spoken numbers). Prints problems for part-3 slides."""
import json, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HERE)
sys.argv = [sys.argv[0], "noop"]
import say_check as sc

old = json.load(open(f"{HERE}/say_snapshot.json"))
new = sc.says()
P3 = "p5 headline perbench fixharm helps memo repl p6 tiers wedh2h llavamed vqamed19 context p7 limits stats untested next summary pA".split()
for k in P3:
    body = "\n".join(l for l in new.get(k, "").split("\n") if not l.lstrip().startswith("# "))
    a, b = sc.nums(old.get(k, "")), sc.nums(body)
    lost = [x for x in set(a) if a.count(x) > b.count(x)]
    added = [x for x in set(b) if b.count(x) > a.count(x)]
    print(f"{k}: lost={lost} added={added}")
