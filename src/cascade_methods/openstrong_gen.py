#!/usr/bin/env python3
"""openstrong_gen.py -- ATTACK 1 (OPEN-STRONG) generation runner.

Generates N open-text answers per question from Lingshu-32B with vLLM `n=N` (SHARED PREFILL),
into a resumable per-item JSONL checkpoint.

WHY ITS OWN FILE AND NOT `src/labeling/run_openvqa.py`.  The brief's null test N3 requires
"regenerate the 32B open-text arm at N=1, temperature 0, WITH YOUR OWN RUNNER, and show it
reproduces the existing always-32B-direct per-cell accuracies on the same item ids".  A runner
that is byte-identical to the deployed one cannot fail that test for an interesting reason, so
this file exists as an INDEPENDENT re-implementation of the same decode path -- and N3 is what
certifies that "independent" did not become "different".

Every prompt/decode constant below is copied verbatim from src/labeling/run_openvqa.py as it
stands on 2026-08-10 and is asserted in the artifact:
    SYS               the deployed DIRECT system prompt (styled)
    cap320            MAXPX = 1280*28*28 // 4 = 250880 ; MINPX = 4*28*28 = 3136
    max_tokens 64, max_model_len 4096, dtype bfloat16, gpu_memory_utilization 0.90
    answer = raw generated text, stripped   (no <think> handling: this is a DIRECT arm)
The item loaders are likewise copied verbatim, so the item id universe is identical.

ADDITION over run_openvqa.py: `--seed`, passed to vLLM SamplingParams, so an 8-sample draw is
reproducible and independent draws can be labelled by seed rather than by luck.

Row schema (resumable; one line per question):
    idx, question, gold, preds[N], oks[N] (exact-match, DIAGNOSTIC ONLY -- the frozen currency
    is the LLM judge), gen_tokens_all[N], seqlogprob, margin, conf, lat_s, n_samples, temp, seed
Plus, persisted on EVERY row because this repo has been burned by unpersisted prompts
(CLAUDE.md Finding-1 lesson): `prompt_sha1` and, on the first row of a file, `prompt_example`.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/openstrong_gen.py \
     --dataset pathvqa_open --n_samples 8 --temp 0.7 --seed 0 --tag l32_bo8_s0 \
     --ckpt_dir ckpts/openvqa/strong_lingshu_bo --tp 1
"""
import argparse
import glob
import hashlib
import io
import json
import math
import os
import re
import string
import time
from collections import Counter

import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------- verbatim from run_openvqa.py
SYS = ("You are an expert medical image analyst. Answer the question with a short, specific "
       "phrase. Do not explain.")
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}

LINGSHU_32B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/snapshots/"
               "36b98277cacb60db86f34b75ce0540b1ea35183c/")

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default=LINGSHU_32B)
ap.add_argument("--tag", required=True)
ap.add_argument("--dataset", required=True,
                choices=["slake_open", "vqa_rad_open", "pathvqa_open"])
ap.add_argument("--n_samples", type=int, default=1)
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320")
ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--max_model_len", type=int, default=4096)
ap.add_argument("--max_tokens", type=int, default=64)
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
MAXPX = HIGH_PX // CAP_DIV[A.cap]


# ---------------------------------------------------------------- verbatim from run_openvqa.py
def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def score(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g):
        return 1
    return 0


def load_items(dataset, nmax):
    """Verbatim item universe of run_openvqa.py for the three open reporting cells."""
    items = []
    if dataset == "slake_open":
        d = json.load(open("/data/dan/dataset/slake/test.json"))
        root = "/data/dan/dataset/slake/imgs"
        for x in d:
            if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en":
                continue
            ip = os.path.join(root, x["img_name"])
            if not os.path.exists(ip):
                continue
            items.append((x["qid"], x["question"], str(x["answer"]), ip))
    else:
        import pandas as pd
        base = ("/data/dan/dataset/vqa_rad/data" if dataset == "vqa_rad_open"
                else "/data/dan/dataset/path_vqa/data")
        dfs = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))]
        df = pd.concat(dfs, ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question")
            a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip()
                a = conv[1]["value"]
            a = str(a).strip()
            if a.lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            else:
                continue
            items.append((int(i), str(q), a, pil))
    return items[:nmax]


items = load_items(A.dataset, A.n)
print(f"{A.dataset}: {len(items)} open-ended items | tag={A.tag} N={A.n_samples} "
      f"temp={A.temp} seed={A.seed} cap={A.cap} maxpx={MAXPX}", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)


def build(q, img):
    im = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX}]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if imgs:
        req["multi_modal_data"] = {"image": imgs}
    return req, text


llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
          limit_mm_per_prompt={"image": 4}, trust_remote_code=True)
sp = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, n=A.n_samples,
                    logprobs=5, seed=A.seed)

ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl")
done = set()
if os.path.exists(ckpt):
    for l in open(ckpt):
        if l.strip():
            try:
                done.add(json.loads(l)["idx"])
            except Exception:
                pass
todo = [it for it in items if it[0] not in done]
print(f"  {len(todo)} to run -> {ckpt}", flush=True)

t0 = time.time()
CH = 64
first = True
with open(ckpt, "a") as fh:
    for c0 in range(0, len(todo), CH):
        ch = todo[c0:c0 + CH]
        built = [build(q, im) for (_, q, _, im) in ch]
        reqs = [b[0] for b in built]
        _t0 = time.time()
        try:
            outs = llm.generate(reqs, sp)
        except Exception as e:                                    # chunk-level guard
            print(f"  CHUNK FAIL {c0}: {type(e).__name__}: {e}", flush=True)
            continue
        _blat = (time.time() - _t0) / max(1, len(ch))
        for (idx, q, gold, _), o, (_, ptext) in zip(ch, outs, built):
            try:
                preds = [c.text.strip() for c in o.outputs]
                oks = [score(p, gold) for p in preds]
                cnt = Counter(norm(p) for p in preds)
                modal_norm, modal_n = cnt.most_common(1)[0]
                modal_pred = next(p for p in preds if norm(p) == modal_norm)
                slp = None
                if o.outputs[0].logprobs:
                    slp = float(np.mean([next(iter(lp.values())).logprob
                                         for lp in o.outputs[0].logprobs if lp]))
                margin = conf = None
                try:
                    _ps = sorted([math.exp(v.logprob)
                                  for v in o.outputs[0].logprobs[0].values()], reverse=True)
                    conf = round(_ps[0], 4)
                    margin = round(_ps[0] - (_ps[1] if len(_ps) > 1 else 0.0), 4)
                except Exception:
                    pass
                row = {"idx": idx, "question": q, "gold": gold, "preds": preds, "oks": oks,
                       "modal_pred": modal_pred, "modal_ok": int(score(modal_pred, gold)),
                       "self_consistency": round(modal_n / len(preds), 4),
                       "n_distinct": len(cnt),
                       "seqlogprob": (round(slp, 4) if slp is not None else None),
                       "margin": margin, "conf": conf,
                       "gen_tokens": len(o.outputs[0].token_ids),
                       "gen_tokens_all": [len(c.token_ids) for c in o.outputs],
                       "lat_s": round(_blat, 4),
                       "n_samples": A.n_samples, "temp": A.temp, "seed": A.seed,
                       "cap": A.cap, "maxpx": MAXPX,
                       "prompt_sha1": hashlib.sha1(ptext.encode()).hexdigest()}
                if first:
                    row["prompt_example"] = ptext
                    first = False
                fh.write(json.dumps(row) + "\n")
            except Exception as e:                                # per-item guard
                print(f"  ITEM FAIL idx={idx}: {type(e).__name__}: {e}", flush=True)
        fh.flush()
        print(f"   [{min(c0 + CH, len(todo))}/{len(todo)}] "
              f"{(min(c0 + CH, len(todo))) / (time.time() - t0):.1f}/s", flush=True)
print(f"DONE {A.tag} {A.dataset}: {len(items)} in {(time.time() - t0) / 60:.1f} min", flush=True)
