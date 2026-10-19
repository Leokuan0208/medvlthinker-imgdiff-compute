# Judge accuracy of WED's four recorded variants (alt picks) on the same questions as wed_vs_probe; read-only.
import json, sys
from collections import defaultdict
import numpy as np
MAIN = "/home/jamesyang/medvlthinker-imgdiff-compute"
sys.path.insert(0, MAIN + "/.claude/worktrees/exp-2026-10-15/src/audit_2026_09_18")
import em_rescore_pooled_probe as ER
CK = MAIN + "/ckpts/openvqa/cheap_lingshu7b"
R = json.load(open("/data/dan/exp_2026_10/final_eval_FINAL_cap320_T07_records.json"))
for c in ["pathvqa_open", "slake_open", "vqa_rad_open"]:
    wed = {d["idx"]: d for d in map(json.loads, open(f"/data/dan/exp_2026_10/wed/wed_{c}_c320.jsonl"))}
    sc8 = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8.jsonl")
    exp = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.jsonl")
    jud = ER.loadj(f"{CK}/ckpt_{c}_lingshu7b_sc8_scexploded.judge.jsonl")
    jmap = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            q = cid.split("#")[0]; q = int(q) if q.lstrip("-").isdigit() else q
            jmap[q][ER.row_norm(r["modal_pred"])] = int(jud[cid]["judge_ok"])
    acc = defaultdict(list)
    for r in R[c]:
        q = r["idx"]; w = wed.get(q)
        if w is None or q not in sc8: continue
        preds = sc8[q]["preds"]
        if w.get("n_cand", 0) >= 2: acc["probe_nc2"].append(r["pick"]["judge"]); acc["greedy_nc2"].append(r["greedy"]["judge"]); acc["wed_nc2"].append(jmap[q].get(ER.row_norm(w["pick_text"])))
        acc["probe"].append(r["pick"]["judge"]); acc["greedy"].append(r["greedy"]["judge"])
        acc["wed_bdg_w"].append(jmap[q].get(ER.row_norm(w["pick_text"])))
        for m in ("bdg_classic", "verifier_D", "greedy_G"):
            i = w.get("alt", {}).get(m, w["pick_index"])
            acc["wed_" + m].append(jmap[q].get(ER.row_norm(preds[i])))
    print(c, len(acc["probe"]), {k: (round(float(np.mean([x for x in v if x is not None])), 4), sum(x is None for x in v)) for k, v in acc.items()})
