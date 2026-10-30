#!/usr/bin/env python3
"""build_xjudge_inputs.py -- MedGemma-27B-it work lists for the 2026-10-22 baselines, in run_judge.py's input format
(idx, question, gold, modal_pred), split in two so each GPU judges half (as runners/fr1015_chain5.sh does).

  python3 build_xjudge_inputs.py missing   -> /data/dan/exp_2026_10/xjudge_1022/lf_part{0,1}.jsonl
        the picks temp_labelfree.py could not score under MedGemma (xjudge_missing_temp_labelfree.jsonl)
  python3 build_xjudge_inputs.py g32       -> /data/dan/exp_2026_10/xjudge_1022/g32_part{0,1}.jsonl
        the fresh Lingshu-32B greedy answers (ckpts/openvqa/strong_lingshu32b_c320), idx "<cell>|<q>|g32"
  python3 build_xjudge_inputs.py picks     -> /data/dan/exp_2026_10/xjudge_1022/picks_part{0,1}.jsonl
        USC and LoRA-verifier picks with no MedGemma label yet (run after merge)
  python3 build_xjudge_inputs.py merge     -> joins every *.judge.jsonl part into /data/dan/exp_2026_10/xjudge_1022/merged.{jsonl,judge.jsonl}
"""
import glob, json, os, sys
D = "/data/dan/exp_2026_10/xjudge_1022"; os.makedirs(D, exist_ok=True)
OPEN8 = ["pathvqa_open", "slake_open", "vqa_rad_open", "radimagenet_open",
         "kvasir_x1_open", "omnimed_open", "vqamed_open", "gemex_open"]


def write_parts(tag, rows):
    for p in (0, 1):
        with open(f"{D}/{tag}_part{p}.jsonl", "w") as fh:
            for r in rows[p::2]:
                fh.write(json.dumps(r) + "\n")
    print(f"{tag}: {len(rows)} items -> {D}/{tag}_part{{0,1}}.jsonl")


mode = sys.argv[1]
if mode == "missing":
    rows = [json.loads(l) for l in open("/data/dan/exp_2026_10/xjudge_missing_temp_labelfree.jsonl") if l.strip()]
    write_parts("lf", [{"idx": f"{r['cell']}|{r['q']}|lf{i}", "question": r["question"], "gold": r["gold"],
                        "modal_pred": r["modal_pred"]} for i, r in enumerate(rows)])
elif mode == "g32":
    rows = []
    for c in OPEN8:
        p = f"/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/strong_lingshu32b_c320/ckpt_{c}_lingshu32b.jsonl"
        if not os.path.exists(p):
            print(f"  [skip] {c}: no 32B ckpt"); continue
        for l in open(p):
            if l.strip():
                d = json.loads(l)
                rows.append({"idx": f"{c}|{d['idx']}|g32", "question": d["question"], "gold": d["gold"], "modal_pred": d["modal_pred"]})
    write_parts("g32", rows)
elif mode == "new":
    # every answer produced by chain 2: SFT greedy (both folds) + WED-as-published picks. One file for the judge
    # of record (run_judge.py writes new_all.judge.jsonl) and two halves for MedGemma.
    M = "/home/jamesyang/medvlthinker-imgdiff-compute"; rows = []
    for c in OPEN8:
        for f in (0, 1):
            p = f"{M}/ckpts/openvqa/sft_crossfit_1022/ckpt_{c}_sft1022f{f}.jsonl"
            if os.path.exists(p):
                for l in open(p):
                    if l.strip():
                        d = json.loads(l)
                        rows.append({"idx": f"{c}|{d['idx']}|sft{f}", "question": d["question"], "gold": d["gold"], "modal_pred": d["modal_pred"]})
    pool = {}
    for c in ("pathvqa_open", "slake_open", "vqa_rad_open"):
        for l in open(f"{M}/ckpts/openvqa/cheap_lingshu7b/ckpt_{c}_lingshu7b_sc8.jsonl"):
            if l.strip():
                d = json.loads(l); pool[(c, d["idx"])] = d
    for fpath in sorted(glob.glob("/data/dan/exp_2026_10/wed_published/wed_pub_*.jsonl")):
        c = os.path.basename(fpath).replace("wed_pub_", "").replace(".jsonl", "").rsplit("_s", 1)[0] if "_s" in os.path.basename(fpath)[-9:] else os.path.basename(fpath).replace("wed_pub_", "").replace(".jsonl", "")
        for l in open(fpath):
            if l.strip():
                d = json.loads(l); q = pool[(c, d["idx"])]
                rows.append({"idx": f"{c}|{d['idx']}|wedpub", "question": q["question"], "gold": q["gold"], "modal_pred": d["pick_text"]})
    with open(f"{D}/new_all.jsonl", "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    write_parts("new", rows)
elif mode == "picks":
    # USC (both pools) and LoRA-verifier (both folds) picks that no MedGemma label covers yet -> picks_part{0,1}.
    # Run after `merge`. "Covered" is decided exactly as baselines_compare.py looks labels up (first labelled idx
    # per (cell, q, answer) across the two merged files), so every pick this sends is one that script would miss.
    M = "/home/jamesyang/medvlthinker-imgdiff-compute"; xlab, xj = {}, {}
    XJS = ["/data/dan/exp_2026_10/xjudge_c320/merged.jsonl", f"{D}/merged.jsonl"]
    for kind in ("judge", "lab"):                     # labels first: xlab keeps the first LABELLED idx per answer
        for X in XJS:
            p = X if kind == "lab" else X.replace(".jsonl", ".judge.jsonl")
            for l in (open(p) if os.path.exists(p) else []):
                if l.strip():
                    d = json.loads(l)
                    if kind == "judge": xj.setdefault(d["idx"], int(d["judge_ok"]))
                    elif d["idx"] in xj: xlab.setdefault(tuple(d["idx"].split("|")[:2]) + (d["modal_pred"],), d["idx"])
    meta = {}
    for c in OPEN8:
        for l in open(f"{M}/ckpts/openvqa/cheap_lingshu7b/ckpt_{c}_lingshu7b_sc8.jsonl"):
            if l.strip():
                d = json.loads(l); meta[(c, str(d["idx"]))] = (d["question"], d["gold"])
    want = {}
    srcs = [(f"/data/dan/exp_2026_10/usc_{p}.jsonl", "ans") for p in ("T07", "Tmap")] + \
           [(f"/data/dan/exp_2026_10/lora_verifier_crossfit_fold{f}.jsonl", "pick_ans") for f in (0, 1)]
    for p, key in srcs:
        for l in (open(p) if os.path.exists(p) else []):
            if l.strip():
                d = json.loads(l); k = (d["cell"], str(d["idx"]), d[key])
                if xj.get(xlab.get(k)) is None and (k[0], k[1]) in meta:
                    want.setdefault(k, None)
    write_parts("picks", [{"idx": f"{c}|{q}|pk{i}", "question": meta[(c, q)][0], "gold": meta[(c, q)][1], "modal_pred": a}
                          for i, (c, q, a) in enumerate(want)])
elif mode == "merge":
    for kind, pat in (("jsonl", "*_part[01].jsonl"), ("judge.jsonl", "*_part[01].judge.jsonl")):
        n = 0
        with open(f"{D}/merged.{kind}", "w") as out:
            for f in sorted(glob.glob(f"{D}/{pat}")):
                for l in open(f):
                    if l.strip(): out.write(l if l.endswith("\n") else l + "\n"); n += 1
        print(f"merged.{kind}: {n} lines")
