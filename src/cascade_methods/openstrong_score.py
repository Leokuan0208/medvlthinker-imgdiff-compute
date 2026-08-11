#!/usr/bin/env python3
"""openstrong_score.py -- ATTACK 1: score a Lingshu-32B best-of-N pool with the FROZEN
CLEAN disjoint-trained LoRA verifier, and emit a transfer-dump in the incumbent's schema.

The verifier is NOT retrained, refit, or recalibrated.  The scoring path (system prompt,
resolution, dtype, attention impl, the P(Yes)/(P(Yes)+P(No)) readout) is copied verbatim from
src/training_methods/verifier_transfer_eval.py, which produced the published incumbent dumps.

HF TRANSFORMERS ONLY.  vLLM 0.9.0.1 silently drops visual.* LoRA modules (same adapter scores
0.775204 under HF and 0.702997 under vLLM), so a vLLM score of this adapter is not the adapter.

INPUT   ckpts/openvqa/strong_lingshu_bo/ckpt_{ds}_{tag}.jsonl            (N-sample pool)
        ckpts/openvqa/strong_lingshu_bo/ckpt_{ds}_{tag}_scexploded.judge.jsonl  (per-candidate judge)
        ckpts/openvqa/strong_lingshu_bo/ckpt_{ds}_l32_n1.judge.jsonl     (the N=1 greedy label)
OUTPUT  <outdir>/transfer_dump_{ds}_{tag}.json   -- {ds, idx, sl[N], scores[N], pick,
                                                     greedy_ok, preds[N]}
        plus a resumable per-(idx,candidate) score cache <outdir>/scorecache_{ds}_{tag}.jsonl

`greedy_ok` here is the judge label of the SEPARATE N=1 temperature-0 run (i.e. exactly
always-32B-direct on that item), NOT the pool's modal answer.  That differs from the incumbent
7B dumps, where greedy_ok is the pool modal -- stated because it changes what `greedy` means in
the frozen metric, and for this attack the 32B-direct baseline is precisely what we want there.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
  python3 src/cascade_methods/openstrong_score.py --tag l32_bo8_s0 \
      --datasets slake_open vqa_rad_open pathvqa_open
"""
import argparse
import glob
import io
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

torch.backends.cuda.matmul.allow_tf32 = False        # numerics pin
torch.backends.cudnn.allow_tf32 = False

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
ap.add_argument("--ckpt_dir", default="ckpts/openvqa/strong_lingshu_bo")
ap.add_argument("--tag", required=True)
ap.add_argument("--greedy_tag", default="l32_n1")
ap.add_argument("--outdir", default=None, help="default: <ckpt_dir>/verif_<adapter basename>")
ap.add_argument("--datasets", nargs="+",
                default=["slake_open", "vqa_rad_open", "pathvqa_open"])
A = ap.parse_args()
DEV = "cuda"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28           # verbatim: verifier scores at FULLRES
CK = os.path.join(ROOT, A.ckpt_dir)
OUT = A.outdir or os.path.join(CK, "verif_" + os.path.basename(A.adapter))
os.makedirs(OUT, exist_ok=True)

SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
       "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def norm(s):
    return str(s).strip().lower()


def imgs_for(ds):
    """Verbatim from verifier_transfer_eval.imgs_for (incl. its slake_open bug fix)."""
    m = {}
    if ds == "slake_open":
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
                ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                if os.path.exists(ip):
                    m[x["qid"]] = (x["question"], ip)
    else:
        import pandas as pd
        base = ("/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open"
                else "/data/dan/dataset/path_vqa/data")
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                       ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question")
            a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip()
                a = conv[1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


print("loading base + adapter (HF transformers, never vLLM) ...", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
model.eval()


def pyes(q, img, ans):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": [
                {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                {"type": "text",
                 "text": f"Question: {q}\nProposed answer: {ans}\n"
                         f"Is the proposed answer correct? Answer Yes or No."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item())
        pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


summary = {}
for ds in A.datasets:
    pool = {r["idx"]: r for r in jl(f"{CK}/ckpt_{ds}_{A.tag}.jsonl")}
    exp = {r["idx"]: r for r in jl(f"{CK}/ckpt_{ds}_{A.tag}_scexploded.jsonl")}
    jud = {r["idx"]: int(r["judge_ok"])
           for r in jl(f"{CK}/ckpt_{ds}_{A.tag}_scexploded.judge.jsonl")}
    gj = {r["idx"]: int(r["judge_ok"])
          for r in jl(f"{CK}/ckpt_{ds}_{A.greedy_tag}.judge.jsonl")}
    if not pool or not jud:
        print(f"{ds}: missing pool ({len(pool)}) or judge ({len(jud)}) -- skip", flush=True)
        continue

    # per-question {normalized answer -> judge label}, reconstructed from the exploded file
    aj = defaultdict(dict)
    for cid, r in exp.items():
        if cid in jud:
            oi = cid.split("#")[0]
            oi = int(oi) if oi.lstrip("-").isdigit() else oi
            aj[oi][norm(r["modal_pred"])] = jud[cid]

    IMG = imgs_for(ds)
    cache_p = os.path.join(OUT, f"scorecache_{ds}_{A.tag}.jsonl")
    cache = {}
    for r in jl(cache_p):
        cache[(r["idx"], r["na"])] = float(r["score"])
    print(f"{ds}: pool={len(pool)} judged_q={len(aj)} imgs={len(IMG)} cached={len(cache)}",
          flush=True)

    dump = []
    n_new = 0
    with open(cache_p, "a") as cf:
        for i, r in pool.items():
            if i not in aj or i not in IMG:
                continue
            q, img = IMG[i]
            preds = r["preds"]
            sl = [aj[i].get(norm(a)) for a in preds]
            if all(x is None for x in sl):
                continue
            scores = []
            for a in preds:                                    # per-candidate error guard
                na = norm(a)
                if (i, na) in cache:
                    scores.append(cache[(i, na)])
                    continue
                try:
                    v = float(pyes(q, img, a))
                except Exception as e:
                    print(f"  SCORE FAIL {ds} idx={i}: {type(e).__name__}: {e}", flush=True)
                    v = 0.5
                v = round(v, 5)
                cache[(i, na)] = v
                cf.write(json.dumps({"idx": i, "na": na, "score": v}) + "\n")
                n_new += 1
                if n_new % 500 == 0:
                    cf.flush()
                    print(f"   scored {n_new} new candidates", flush=True)
                scores.append(v)
            k = int(np.argmax(scores))
            dump.append({"ds": ds, "idx": i,
                         "sl": [(-1 if x is None else int(x)) for x in sl],
                         "scores": scores, "pick": k,
                         "greedy_ok": int(gj.get(i, 0)),
                         "greedy_present": int(i in gj),
                         "preds": preds})
    json.dump(dump, open(os.path.join(OUT, f"transfer_dump_{ds}_{A.tag}.json"), "w"))
    sl = np.array([[0 if x < 0 else x for x in d["sl"]] for d in dump])
    got = np.array([d["sl"][d["pick"]] == 1 for d in dump], float)
    rec = (sl.max(1) == 1)
    summary[ds] = {"n": len(dump), "greedy_n1": float(np.mean([d["greedy_ok"] for d in dump])),
                   "oracle": float(rec.mean()), "selected": float(got.mean()),
                   "sel_eff": float(got[rec].mean()) if rec.any() else float("nan"),
                   "n_new_scored": n_new}
    print(f"  {ds}: {json.dumps(summary[ds])}", flush=True)

json.dump(summary, open(os.path.join(OUT, f"score_summary_{A.tag}.json"), "w"), indent=1)
print("DONE", json.dumps(summary, indent=1), flush=True)
