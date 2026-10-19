#!/usr/bin/env python3
"""Real worked example for the deck: SLAKE-open question 12231, scored by the frozen final probes of the
fold whose training half did NOT contain this image. Prints every probe's score and rank, the mean rank,
the vote counts and the pick. Output: worked_example_12231.json (all numbers on the slide come from here)."""
import json, sys
import numpy as np
import torch

WT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15"
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, WT + "/src/audit_2026_09_18")
sys.path.insert(0, WT + "/src/training_methods")
import em_rescore_pooled_probe as ER   # noqa: E402
from genframe_data import rank_avg     # noqa: E402

CELL, Q = "slake_open", 12231
z = np.load(f"{MAIN}/feats_hidden/generator_eval_finelayer.npz")
m = json.load(open(f"{MAIN}/feats_hidden/generator_eval_finelayer.meta.json"))
lay = [int(x) for x in z["layers"]]
ii = [i for i, r in enumerate(m["rows"]) if r.get("ds") == CELL and r["idx"] == Q and r.get("n_tok", -1) > 0]
rows = [m["rows"][i] for i in ii]
img = rows[0]["img_md5"]
fold = ER.half(img)
heads, stats, _ = ER.load_probes(f"{MAIN}/ckpts/train/final_probe_2026-10-16/fold{fold}")
S, names = [], []
with torch.no_grad():
    for L in (18, 20, 22):
        mu, sd = stats[L]
        X = z["h_span"][ii, lay.index(L)].astype(np.float32)
        Zs = torch.tensor((X - mu) / sd)
        for s, h in enumerate(heads[L]):
            h.eval(); S.append(h(Zs).numpy().reshape(-1)); names.append(f"L{L}/seed{s}")
S = np.stack(S)                                   # (24, n_distinct)
R = np.stack([rank_avg(s) for s in S])            # within-question ranks in [0, 1]
mean_rank = R.mean(0)
pool = {}
for l in open(f"{MAIN}/ckpts/openvqa/cheap_lingshu7b/ckpt_{CELL}_lingshu7b_sc8.jsonl"):
    d = json.loads(l)
    if d["idx"] == Q:
        pool = d
pn = [ER.row_norm(a) for a in pool["preds"]]
votes = [pn.count(r["na"]) for r in rows]
top = np.flatnonzero(mean_rank == mean_rank.max())
pick = int(top[np.argmax(np.array(votes)[top])])
out = {"cell": CELL, "idx": Q, "question": pool["question"], "gold": pool["gold"], "samples_T07": pool["preds"],
       "image_md5": img, "scored_by_fold_probes": fold, "candidates": [
           {"answer": r["ans"], "judge_label": int(r["y"]), "votes": v, "mean_rank": round(float(mr), 4),
            "n_probes_ranking_it_top": int((R[:, k] == R.max(0)[k]).sum() if False else (R[:, k] == 1.0).sum()),
            "probe_scores": [round(float(x), 3) for x in S[:, k]], "probe_ranks": [round(float(x), 3) for x in R[:, k]]}
           for k, (r, v, mr) in enumerate(zip(rows, votes, mean_rank))],
       "probe_names": names, "pick": rows[pick]["ans"], "majority_vote": rows[int(np.argmax(votes))]["ans"]}
json.dump(out, open("/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/worked_example_12231.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "candidates"}, indent=1))
for c in out["candidates"]:
    print(c["answer"], "judge", c["judge_label"], "votes", c["votes"], "mean_rank", c["mean_rank"],
          "top-ranked by", c["n_probes_ranking_it_top"], "of 24")
