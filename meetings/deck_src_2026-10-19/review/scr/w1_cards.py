import json, glob
want = ["2305.20050", "2211.14275", "2207.05221", "2212.03827", "2304.13734", "2410.02707", "2406.15927", "2502.18581",
        "2310.09139", "2410.01064", "2112.09332", "2210.10760", "2306.00890", "2307.14334", "2406.19280", "2507.05201",
        "2302.08468", "2605.28527", "2603.22492"]
seen = set()
for f in sorted(glob.glob("/home/jamesyang/.claude/jobs/61c9eff9/tmp/lit/cards_*.json")):
    data = json.load(open(f))
    if isinstance(data, dict):
        data = data.get("cards", list(data.values()))
    for c in data:
        s = json.dumps(c, ensure_ascii=False)
        for w in want:
            if w in s and w not in seen:
                seen.add(w)
                print("==", f.split("/")[-1], w)
                print(s[:1400])
                print()
print("missing:", [w for w in want if w not in seen])
