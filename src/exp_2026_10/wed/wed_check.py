"""Sanity check of a WED picks file against its pool: fraction of picks equal to the modal answer, and the pool's
own `oks` (exact-match-style flag from run_openvqa.score) at the pick vs at the modal answer and oracle-of-8.
SANITY ONLY -- not an accuracy claim.  Usage: python3 wed_check.py --pool <pool.jsonl> --picks <picks.jsonl>"""
import argparse, json, re, string
from collections import Counter

ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--picks", required=True)
A = ap.parse_args()

def norm(s):  # = src/labeling/run_openvqa.py norm()
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

pool = {r["idx"]: r for r in map(json.loads, filter(str.strip, open(A.pool)))}
allp = [json.loads(l) for l in open(A.picks) if l.strip()]
print("n_cand distribution (all rows):", sorted(Counter(p.get("n_cand", 0) for p in allp).items()))
picks = [p for p in allp if p.get("pick_index") is not None]
if not picks: raise SystemExit("no picks (candidates_only run)")
n = len(picks); same_modal = same_field = em_pick = em_modal = oracle = fb = 0; cands = []; iters = []
for p in picks:
    r = pool[p["idx"]]; preds = r["preds"]
    c = Counter(norm(x) for x in preds); top = max(c.values())
    modal_keys = {k for k, v in c.items() if v == top}           # ties: any tied-modal answer counts as modal
    same_modal += norm(p["pick_text"]) in modal_keys
    same_field += norm(p["pick_text"]) == norm(r.get("modal_pred", ""))
    if "oks" in r:
        em_pick += r["oks"][p["pick_index"]]; em_modal += r.get("modal_ok", 0); oracle += max(r["oks"])
    fb += "fallback" in p; cands.append(p.get("n_cand", 0)); iters.append(p.get("n_iters", 0))
print(f"n={n}  fallbacks={fb}  mean_n_cand={sum(cands)/n:.2f}  mean_W-BDG_iters={sum(iters)/n:.2f}")
print(f"pick == a modal answer (norm, ties count): {same_modal}/{n} = {same_modal/n:.4f}")
print(f"pick == pool modal_pred field (norm):      {same_field}/{n} = {same_field/n:.4f}")
print(f"pool oks @WED pick {em_pick/n:.4f} | @modal_pred (modal_ok) {em_modal/n:.4f} | oracle-of-8 {oracle/n:.4f}")
