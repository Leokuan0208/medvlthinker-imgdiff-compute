#!/usr/bin/env python3
"""
run_openvqa.py - GENERATIVE (open-ended) medical VQA runner for the open-ended cascade experiment.
Unlike the MCQ runners, there are NO options: the model produces a free-text answer. Supports
SELF-CONSISTENCY sampling (--n_samples K at --temp T) so we can measure semantic agreement among a
cheap model's own samples -- a signal that is degenerate for single-letter MCQ but rich here.

Datasets (open-ended only):
  slake_open   : /data/dan/dataset/slake/test.json  (answer_type=OPEN, q_lang=en), imgs/<img_name>
  vqa_rad_open : /data/dan/dataset/vqa_rad/data/test-*.parquet (open answers)
  pathvqa_open : /data/dan/dataset/path_vqa/data/test-*.parquet (open answers; drops yes/no)
Output JSONL per sample: idx, question, gold, preds(list), oks(list), modal_pred, modal_ok,
  self_consistency(max vote fraction), n_distinct, seqlogprob(of 1st sample), gen_tokens.
Scoring = normalized exact match + 'contains' fallback (short factual answers). vLLM. Launch from repo root.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa.py \
     --model_path /data/dan/weights/MedVLThinker-7B-RL_m23k --tag 7b --dataset slake_open \
     --n_samples 5 --temp 0.7 --ckpt_dir ckpts/openvqa/7b_sc --tp 1
"""
import argparse, re, json, os, glob, string
import numpy as np
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image

SYS = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts within "
             "<think> </think> tags before providing the answer. After </think>, give only the short final answer.")
# MATCHED reasoning prompt: identical to SYS in persona + answer-style constraints ("expert medical image
# analyst", "short, specific phrase", "Do not explain"); differs ONLY by the reason-first instruction. Used
# to de-confound the reasoning effect from the output-convention effect (--think_matched).
SYS_THINK_MATCHED = ("You are an expert medical image analyst. You should provide your thoughts within "
                     "<think> </think> tags before providing the answer. After </think>, answer the question "
                     "with a short, specific phrase. Do not explain.")
# MATCHED variant 2 (--think_matched2). SYS_THINK_MATCHED paraphrases the reasoning trigger and these models
# then SKIP the <think> trace most of the time (documented quirk: the trigger sentences must appear verbatim),
# which would silently turn reasoning OFF instead of matching the prompts. This variant keeps SYS_THINK's
# trigger sentences VERBATIM and replaces only its answer-style clause ("give only the short final answer")
# with the direct prompt's persona + answer-style + no-explanation constraints -- so reasoning stays ON and
# the ONLY difference from the direct arm is the reason-first instruction.
SYS_THINK_MATCHED2 = ("You will solve a problem/request. You should provide your thoughts within "
                      "<think> </think> tags before providing the answer. You are an expert medical image "
                      "analyst. After </think>, answer the question with a short, specific phrase. "
                      "Do not explain.")
# UNSTYLED DIRECT prompt (--direct_unstyled): SYS_THINK with the reasoning instruction REMOVED and nothing
# else changed -- no persona, no "specific phrase", no "Do not explain", same "give only the short final
# answer" convention. It completes the 2x2 (style x reasoning): pairing it with SYS_THINK isolates the
# reasoning effect at a FIXED output convention, which matched reasoning prompts cannot do, because telling
# this model "Do not explain" also suppresses its <think> trace.
SYS_DIRECT_UNSTYLED = "You will solve a problem/request. Give only the short final answer."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--dataset", required=True, choices=["slake_open", "vqa_rad_open", "pathvqa_open", "kvasir_open", "radimagenet_open",
                # *_train = the datasets' OFFICIAL TRAIN splits, same open-ended filter. Added for the
                # disjoint verifier retrain (src/training_methods/build_disjoint_verifier_split.py): the
                # verifier must be trained on items that share no question and no image with the eval sets.
                "slake_open_train", "vqa_rad_open_train", "pathvqa_open_train"])
ap.add_argument("--n_samples", type=int, default=1); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320"); ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88); ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=64)
ap.add_argument("--think", action="store_true", help="reasoning mode: think system prompt, answer = text after </think>")
ap.add_argument("--think_matched", action="store_true", help="reasoning mode with the MATCHED system prompt "
                "(SYS_THINK_MATCHED: same persona/answer-style constraints as direct, reason-first added); implies --think")
ap.add_argument("--think_matched2", action="store_true", help="reasoning mode with MATCHED variant 2 "
                "(SYS_THINK_MATCHED2: SYS_THINK's trigger sentences kept VERBATIM so the <think> trace still "
                "fires, + the direct prompt's persona/answer-style constraints); implies --think")
ap.add_argument("--direct_unstyled", action="store_true", help="DIRECT mode with SYS_DIRECT_UNSTYLED "
                "(SYS_THINK minus the reasoning instruction): completes the style x reasoning 2x2")
ap.add_argument("--save_raw", action="store_true", help="also store the raw un-extracted sample texts (key 'raw')")
ap.add_argument("--idx_file", default=None, help="optional JSON list of idx: restrict generation to this allowlist")
A = ap.parse_args(); os.makedirs(A.ckpt_dir, exist_ok=True); MAXPX = HIGH_PX // CAP_DIV[A.cap]
if A.think_matched or A.think_matched2: A.think = True
if A.think and A.max_tokens < 256: A.max_tokens = 512

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()
def score(pred, gold):
    p, g = norm(pred), norm(gold)
    if not p: return 0
    if p == g: return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g): return 1   # short-answer contains
    return 0

# ---- load open-ended items: (idx, question, gold, PIL image) ----
items = []
SPLIT = "train" if A.dataset.endswith("_train") else "test"   # *_open_train -> official train split
if A.dataset in ("slake_open", "slake_open_train"):
    d = json.load(open(f"/data/dan/dataset/slake/{SPLIT}.json")); root = "/data/dan/dataset/slake/imgs"
    seen = set()
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if not os.path.exists(ip): continue
        items.append((x["qid"], x["question"], str(x["answer"]), ip))
elif A.dataset == "radimagenet_open":
    d = json.load(open("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json"))
    for r in d:
        if os.path.exists(r["img_path"]):
            items.append((r["idx"], r["question"], r["answer"], r["img_path"]))
elif A.dataset == "kvasir_open":
    d = json.load(open(f"/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"))
    for r in d:
        if os.path.exists(r["img_path"]):
            items.append((r["idx"], r["question"], r["answer"], r["img_path"]))
else:
    import pandas as pd
    base = "/data/dan/dataset/vqa_rad/data" if A.dataset.startswith("vqa_rad_open") else "/data/dan/dataset/path_vqa/data"
    dfs = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, f"{SPLIT}-*.parquet")))]
    df = pd.concat(dfs, ignore_index=True)
    for i, r in df.iterrows():
        # parquet schema: 'question','answer','image'(dict bytes) OR 'conversations'
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:  # vqa_rad conversations format
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
        a = str(a).strip()
        if a.lower() in ("yes", "no"): continue                 # keep OPEN-ended only
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            import io; pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else: continue
        items.append((int(i), str(q), a, pil))
if A.idx_file:
    allow = set(json.load(open(A.idx_file)))
    items = [it for it in items if it[0] in allow]
items = items[:A.n]
print(f"{A.dataset}: {len(items)} open-ended items | tag={A.tag} n_samples={A.n_samples} temp={A.temp}"
      f"{' | idx_file allowlist='+str(len(json.load(open(A.idx_file)))) if A.idx_file else ''}", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
def extract(t):  # think mode: answer = text after the LAST </think>; else the raw text
    if A.think and "</think>" in t:
        tail = t.split("</think>")[-1].strip()
        return tail.splitlines()[-1].strip() if tail else t.strip()
    return t.strip()
def build(q, img):
    im = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX}]
    sys = ((SYS_THINK_MATCHED2 if A.think_matched2 else SYS_THINK_MATCHED if A.think_matched else SYS_THINK)
           if A.think else (SYS_DIRECT_UNSTYLED if A.direct_unstyled else SYS))
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 4}, trust_remote_code=True)
sp = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, n=A.n_samples,
                    logprobs=5)   # capture top-5 token logprobs -> margin/confidence (cascade gate signal)
ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl")
done = set()
if os.path.exists(ckpt):
    for l in open(ckpt):
        if l.strip():
            try: done.add(json.loads(l)["idx"])
            except Exception: pass
todo = [it for it in items if it[0] not in done]
print(f"  {len(todo)} to run -> {ckpt}", flush=True)
import time; t0 = time.time(); CH = 64
with open(ckpt, "a") as fh:
    for c0 in range(0, len(todo), CH):
        ch = todo[c0:c0+CH]; reqs = [build(q, im) for (_, q, _, im) in ch]
        import math, time as _t; _t0 = _t.time(); outs = llm.generate(reqs, sp); _blat = (_t.time()-_t0)/max(1, len(ch))
        for (idx, q, gold, _), o in zip(ch, outs):
            preds = [extract(c.text) for c in o.outputs]
            oks = [score(p, gold) for p in preds]
            from collections import Counter
            cnt = Counter(norm(p) for p in preds); modal_norm, modal_n = cnt.most_common(1)[0]
            modal_pred = next(p for p in preds if norm(p) == modal_norm)
            sc = modal_n / len(preds); ndist = len(cnt)
            slp = None
            if o.outputs[0].logprobs:
                slp = float(np.mean([next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs if lp]))
            margin = conf = None
            try:  # 1st-token top1-top2 prob -> confidence margin (cascade gate signal)
                _ps = sorted([math.exp(v.logprob) for v in o.outputs[0].logprobs[0].values()], reverse=True)
                conf = round(_ps[0], 4); margin = round(_ps[0] - (_ps[1] if len(_ps) > 1 else 0.0), 4)
            except Exception:
                pass
            fh.write(json.dumps({"idx": idx, "question": q, "gold": gold, "preds": preds, "oks": oks,
                "modal_pred": modal_pred, "modal_ok": int(score(modal_pred, gold)),
                "self_consistency": round(sc, 4), "n_distinct": ndist,
                "seqlogprob": (round(slp, 4) if slp is not None else None),
                "margin": margin, "conf": conf,
                "gen_tokens": len(o.outputs[0].token_ids),
                "gen_tokens_all": [len(c.token_ids) for c in o.outputs], "lat_s": round(_blat, 4),
                **({"raw": [c.text for c in o.outputs]} if A.save_raw else {})}) + "\n")
        fh.flush()
        acc = np.mean([score(json.loads(l)["modal_pred"], json.loads(l)["gold"]) for l in open(ckpt) if l.strip()])
        print(f"   [{min(c0+CH,len(todo))}/{len(todo)}] running modal-acc={acc:.3f} {(min(c0+CH,len(todo)))/(time.time()-t0):.1f}/s", flush=True)
print(f"DONE {A.tag} {A.dataset}: {len(items)} in {(time.time()-t0)/60:.1f} min", flush=True)
