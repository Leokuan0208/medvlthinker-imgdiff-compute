import json, glob
keys, ids = [], set()
for f in sorted(glob.glob("/home/jamesyang/.claude/jobs/61c9eff9/tmp/lit/cards_*.json")):
    data = json.load(open(f))
    if isinstance(data, dict):
        data = data.get("cards", [])
    for c in data:
        if isinstance(c, dict) and ("title" in c):
            keys.append(c.get("key"))
            ids.add(c.get("arxiv") or c.get("doi") or c.get("title"))
print("cards", len(keys), "distinct keys", len(set(keys)), "distinct ids", len(ids))
