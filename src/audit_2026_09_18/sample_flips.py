import json, random, sys
T = "/data/dan/audit_2026-09-18/tmp"
R = json.load(open(f"{T}/em-rescore/em_rescore_per_question_cache.json"))["records"]
Q = {}
for l in open(f"{T}/me/xjudge/heldout_lingshu7b_medgemma27b.jsonl"):
    d = json.loads(l); c, i, _ = d["idx"].split("|"); Q[(c, i)] = d["question"]
random.seed(7)
for cell in sys.argv[2:]:
    ups = [r for r in R[cell] if r["cands"][r["pick"]]["y"] == 1 and r["greedy"]["y"] == 0
           and r["cands"][r["pick"]]["f1"] == 0]
    print(f"\n===== {cell}: {len(ups)} upward flips with ZERO token overlap; sample of {sys.argv[1]}")
    for r in random.sample(ups, min(int(sys.argv[1]), len(ups))):
        print(f"Q: {Q[(cell, str(r['idx']))][:150]}\n   GOLD: {r['gold'][:140]}\n   GREEDY(judge=0): {r['greedy']['ans'][:120]}\n   PICK  (judge=1): {r['cands'][r['pick']]['ans'][:120]}")
