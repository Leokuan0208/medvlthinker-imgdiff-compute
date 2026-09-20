"""Independent spot-verification of the audit agents' most serious claims (read-only)."""
import json, glob, os, re
import numpy as np
ROOT = "/home/jamesyang/medvlthinker-imgdiff-compute"
CK = f"{ROOT}/ckpts/openvqa/cheap_lingshu7b"

print("== 1. MedGemma sampled pools: tokens generated per candidate vs greedy; garbage rate")
for tag in ("medgemma4b", "lingshu7b", "qwen25vl7b"):
    for cell in ("slake_open", "gemex_open"):
        p, g = f"{CK}/ckpt_{cell}_{tag}_sc8.jsonl", f"{CK}/ckpt_{cell}_{tag}.jsonl"
        if not (os.path.exists(p) and os.path.exists(g)):
            print("   missing", tag, cell); continue
        rows = [json.loads(l) for l in open(p) if l.strip()][:400]
        gr = [json.loads(l) for l in open(g) if l.strip()][:400]
        keys = sorted(rows[0].keys())
        gt = [r.get("gen_tokens") for r in rows if r.get("gen_tokens") is not None]
        ggt = [r.get("gen_tokens") for r in gr if r.get("gen_tokens") is not None]
        preds = [x for r in rows for x in r["preds"]]
        bad = [x for x in preds if re.search(r"\nmodel|\nimage|```|<start_of_turn>|<end_of_turn>", x)]
        wl = np.mean([len(x.split()) for x in preds])
        print(f"   {tag:11} {cell:11} sc8 gen_tokens mean {np.mean(gt) if gt else None} (n={len(gt)}) | greedy gen_tokens mean "
              f"{np.mean(ggt) if ggt else None} | sampled words/cand {wl:.1f} | template-garbage cands {len(bad)}/{len(preds)}")
        if tag == "medgemma4b" and cell == "slake_open":
            print("      keys:", keys); print("      example preds:", [x[:90] for x in rows[3]["preds"][:3]])

print("== 2. extraction resolution vs generation cap")
for f, pat in ((f"{ROOT}/src/training_methods/extract_generator_hidden.py", r"HIGH_PX\s*="),
               (f"{ROOT}/src/labeling/run_openvqa.py", r"cap320|--cap|max_pixels")):
    for i, l in enumerate(open(f), 1):
        if re.search(pat, l):
            print(f"   {os.path.basename(f)}:{i}: {l.strip()[:150]}")

print("== 3. generator LoRA-SFT ('cheapleg') baseline summaries")
for p in sorted(glob.glob(f"{ROOT}/ckpts/cheapleg/scores_*/score_summary.json")):
    d = json.load(open(p)); print("  ", p.replace(ROOT + "/", "")); print("     ", json.dumps(d)[:900])
