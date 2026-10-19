import json, glob
want = "2112.09332 2207.05221 2210.10760 2211.14275 2212.03827 2304.13734 2305.20050 2310.09139 2406.15927 2410.01064 2410.02707 2502.18581 2603.22492 2605.28527".split()
found = {}
for f in glob.glob("/home/jamesyang/.claude/jobs/61c9eff9/tmp/lit/cards_*.json"):
    d = json.load(open(f))
    items = d if isinstance(d, list) else d.get("cards", list(d.values()))
    for c in items:
        s = json.dumps(c)
        for w in want:
            if w in s and w not in found and isinstance(c, dict):
                found[w] = {k: c[k] for k in c if k in ("title", "authors", "author", "year", "venue", "arxiv", "id", "cite", "short")}
for w in want:
    print(w, found.get(w))
