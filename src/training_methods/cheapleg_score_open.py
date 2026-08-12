#!/usr/bin/env python3
"""cheapleg_score_open.py -- score ONE open-text arm's 8-sample pools with the FROZEN incumbent
verifier and emit a transfer dump in the incumbent's exact format.

This is src/training_methods/verifier_transfer_eval.py's scoring path, unchanged in every detail that
can move a number (the pyes() prompt, SYS string, max_pixels/min_pixels, the argmax pick, the judge
lookup by NORMALIZED answer), with three differences that cannot:
  * the generation checkpoint dir and tag are arguments (so a NEW generator arm can be scored),
  * the dump is written to an arm-owned directory instead of into the verifier's checkpoint dir
    (ckpts/train/lora_verifier_disjoint is a frozen artifact of record and is not written to),
  * --nulltest re-scores an EXISTING dump's items and reports the max abs deviation from the stored
    scores, which is the only acceptable way to start using a new scoring harness here.

*** ADAPTERS ARE SCORED UNDER HF TRANSFORMERS, NEVER vLLM *** (vLLM 0.9.0.1 silently drops all 192
visual.* LoRA modules: the same adapter scores 0.775204 under HF and 0.702997 under vLLM).

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/cheapleg_score_open.py --gen_dir ckpts/openvqa/cheapleg_base7b \
    --tag base7b --out_dir ckpts/cheapleg/scores_base7b
"""
import argparse, glob, io, json, math, os, time
from collections import Counter, defaultdict

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
DEV = "cuda"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
       "the proposed answer is correct. Respond with only 'Yes' or 'No'.")

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--verifier", default="ckpts/train/lora_verifier_disjoint",
                help="THE FROZEN incumbent verifier. Never retrained here.")
ap.add_argument("--gen_dir", required=True, help="dir holding ckpt_<ds>_<tag>_sc8*.jsonl")
ap.add_argument("--tag", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--datasets", nargs="+", default=["slake_open", "vqa_rad_open", "pathvqa_open"])
ap.add_argument("--nulltest", type=int, default=0,
                help=">0: re-score this many items per dataset from the PUBLISHED pool "
                     "(ckpts/openvqa/cheap_lingshu7b, tag lingshu7b) and report max abs deviation "
                     "against the stored transfer dump. No new dump is written.")
A = ap.parse_args()
OUT = os.path.join(ROOT, A.out_dir)
os.makedirs(OUT, exist_ok=True)
print(f"[numerics] torch={torch.__version__} matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
      f"cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}", flush=True)


def loadj(p):
    return {r["idx"]: r for r in (json.loads(l) for l in open(p) if l.strip())} if os.path.exists(p) else {}


def norm(s):
    return str(s).strip().lower()


def imgs_for(ds):
    """VERBATIM from verifier_transfer_eval.py (incl. its slake_open fix)."""
    m = {}
    if ds == "slake_open":
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
                ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                if os.path.exists(ip):
                    m[x["qid"]] = (x["question"], ip)
    else:
        import pandas as pd
        base = "/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                       ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
print("loading base + FROZEN verifier adapter (HF, never vLLM)...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.verifier))
model.eval()


def pyes(q, img, ans):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": [
                {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                         f"Is the proposed answer correct? Answer Yes or No."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


# ------------------------------------------------------------------ null test
if A.nulltest:
    dev = 0.0
    n = 0
    for ds in A.datasets:
        stored = {r["idx"]: r for r in json.load(open(os.path.join(
            ROOT, A.verifier, f"transfer_dump_{ds}_lingshu7b.json")))}
        IMG = imgs_for(ds)
        for i in list(stored)[:A.nulltest]:
            q, img = IMG[i]
            sc = [pyes(q, img, a) for a in stored[i]["preds"]]
            d = max(abs(round(float(x), 5) - y) for x, y in zip(sc, stored[i]["scores"]))
            dev = max(dev, d); n += len(sc)
        print(f"  {ds}: running max abs deviation {dev:.3e} over {n} candidate scores", flush=True)
    print(f"\nNULL TEST: max abs deviation vs stored dump = {dev:.3e} over {n} candidate scores")
    json.dump({"nulltest_max_abs_deviation": dev, "n_scores": n, "verifier": A.verifier,
               "stored_dump": "ckpts/train/lora_verifier_disjoint/transfer_dump_*_lingshu7b.json"},
              open(os.path.join(OUT, "nulltest.json"), "w"), indent=1)
    raise SystemExit(0)

# ------------------------------------------------------------------ score an arm
CK = os.path.join(ROOT, A.gen_dir)
summary = {}
for ds in A.datasets:
    sc = loadj(f"{CK}/ckpt_{ds}_{A.tag}_sc8.jsonl")
    exp = loadj(f"{CK}/ckpt_{ds}_{A.tag}_sc8_scexploded.jsonl")
    jud = {k: v["judge_ok"] for k, v in loadj(f"{CK}/ckpt_{ds}_{A.tag}_sc8_scexploded.judge.jsonl").items()}
    if not sc or not jud:
        print(f"{ds}: missing sc8({len(sc)}) or judge({len(jud)}) -- SKIP", flush=True)
        continue
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]
            oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]
    IMG = imgs_for(ds)
    dump = []
    t0 = time.time()
    outp = os.path.join(OUT, f"transfer_dump_{ds}_lingshu7b.json")
    for c, i in enumerate(sc):
        if i not in aj or i not in IMG:
            continue
        q, img = IMG[i]
        preds = sc[i]["preds"]
        sl = [aj[i].get(norm(a)) for a in preds]
        if all(x is None for x in sl):
            continue
        try:
            scores = [pyes(q, img, a) for a in preds]
        except Exception as e:                                   # per-item error guard
            print(f"  score fail {ds}/{i}: {str(e)[:60]}", flush=True)
            continue
        k = int(np.argmax(scores))
        dump.append({"ds": ds, "idx": i, "sl": [(-1 if x is None else int(x)) for x in sl],
                     "scores": [round(float(x), 5) for x in scores], "pick": k,
                     "greedy_ok": int(aj[i].get(norm(sc[i]["modal_pred"]), 0)), "preds": preds})
        if (c + 1) % 200 == 0:
            print(f"  {ds} {c+1}/{len(sc)}  {(time.time()-t0)/60:.1f} min", flush=True)
    json.dump(dump, open(outp, "w"))
    g = float(np.mean([d["greedy_ok"] for d in dump]))
    sel = float(np.mean([1 if d["sl"][d["pick"]] == 1 else 0 for d in dump]))
    orc = float(np.mean([1 if 1 in d["sl"] else 0 for d in dump]))
    summary[ds] = dict(n=len(dump), greedy=g, selected=sel, oracle8=orc,
                       sel_eff=(sel - g) / (orc - g) if orc > g else None)
    print(f"  {ds:<14} n={len(dump):4d} greedy={g:.4f} selected={sel:.4f} oracle@8={orc:.4f}", flush=True)

json.dump(dict(gen_dir=A.gen_dir, tag=A.tag, verifier=A.verifier, per_ds=summary,
               note="sel_eff here is the crude pooled-per-ds form; THE frozen metric is "
                    "src/training_methods/genframe_data.sel_eff and is applied in the analysis step."),
          open(os.path.join(OUT, "score_summary.json"), "w"), indent=1)
print(f"\nwrote -> {A.out_dir}")
