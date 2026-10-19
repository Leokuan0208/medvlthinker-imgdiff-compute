import json, glob
want = {"2211.14275", "2305.20050", "2410.02707", "2406.15927", "2207.05221", "2112.09332", "2210.10760", "2410.01064", "2502.18581"}
for f in sorted(glob.glob("/home/jamesyang/.claude/jobs/61c9eff9/tmp/lit/cards_*.json")):
    data = json.load(open(f))
    if isinstance(data, dict):
        data = data.get("cards", [])
    for c in data:
        if isinstance(c, dict) and c.get("arxiv") in want:
            print("==", c["arxiv"], c.get("title"), "|", c.get("venue"))
            print("RESULTS:", c.get("results"))
            print("CONCL:", c.get("conclusions"))
            print()
