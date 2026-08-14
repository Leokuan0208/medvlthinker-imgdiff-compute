#!/usr/bin/env python3
"""decoding_sweep_verify.py -- score candidate answers with the FROZEN incumbent verifier.

The scoring function `pyes` is VERBATIM the one that produced the published cells
(src/training_methods/verifier_transfer_eval.py): the CLEAN disjoint-trained LoRA on Lingshu-7B, grader
system prompt, max_pixels = 1280*28*28, Yes/No logit ratio at the last position, batch 1.

HF transformers ONLY -- vLLM silently drops visual.* LoRA modules (CLAUDE.md landmine; 0.775204 HF vs
0.702997 vLLM).  Note this adapter targets LM projections only, but the rule is obeyed regardless.

Writes a RESUMABLE, append-only score cache keyed by (ds, idx, exact answer text), so the marginal cost
of each new decoding setting is only its NEW candidate strings.
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/decoding_sweep_verify.py --work work.json --shard 0 --nshard 2
"""
import argparse, glob, io, json, math, os, sys, time
import numpy as np, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
DEV = "cuda"
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28          # verifier resolution, as deployed
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True, help="JSON list of [ds, idx, answer_text]")
ap.add_argument("--cache", default="ckpts/openvqa/decoding_sweep/vscore_cache")
ap.add_argument("--model_path", default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
                                        "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
A = ap.parse_args()

print(f"[numerics] tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
      f"tf32_cudnn={torch.backends.cudnn.allow_tf32} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','unset')} "
      f"torch={torch.__version__}", flush=True)


def imgs_for(ds):
    """Verbatim from verifier_transfer_eval.py (incl. its slake_open image-path fix)."""
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
print("loading base + adapter...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.model_path, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").to(DEV)
model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter)); model.eval()


def _msgs(q, img, ans):
    return [{"role": "system", "content": SYS}, {"role": "user", "content": [
        {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
        {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."}]}]


def pyes(q, img, ans):
    """VERBATIM the published scoring path (verifier_transfer_eval.py)."""
    msgs = _msgs(q, img, ans)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


def pyes_fast(q, img, ans, cache):
    """Same computation, but the image is resized/patchified ONCE per question.

    Only the answer substring changes between a question's candidates, so pixel_values and
    image_grid_thw are identical across them. Validated against pyes() below -- the run aborts unless
    the two agree BIT-EXACTLY on the first CHECK_N candidates.
    """
    msgs = _msgs(q, img, ans)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "igs" not in cache:
        cache["igs"], cache["vids"] = process_vision_info(msgs)
        e = proc(text=[text], images=cache["igs"], videos=cache["vids"], return_tensors="pt", padding=True)
        cache["pixel_values"] = e["pixel_values"].to(DEV)
        cache["image_grid_thw"] = e["image_grid_thw"].to(DEV)
        # The PROCESSOR (not the tokenizer) expands <|image_pad|> into one token per merged patch.
        # Replicate that expansion verbatim (Qwen2_5_VLProcessor.__call__) or the LM gets 1 image
        # token for 324 image features.
        cache["n_img_tok"] = int(e["image_grid_thw"][0].prod()) // (proc.image_processor.merge_size ** 2)
    IMTOK = proc.image_token if hasattr(proc, "image_token") else "<|image_pad|>"
    text = text.replace(IMTOK, IMTOK * cache["n_img_tok"], 1)
    tk = proc.tokenizer([text], return_tensors="pt", padding=True)
    enc = {"input_ids": tk["input_ids"].to(DEV), "attention_mask": tk["attention_mask"].to(DEV),
           "pixel_values": cache["pixel_values"], "image_grid_thw": cache["image_grid_thw"]}
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


work = json.load(open(A.work))
work = [w for i, w in enumerate(work) if i % A.nshard == A.shard]
CACHE = os.path.join(ROOT, A.cache + f"_shard{A.shard}of{A.nshard}.jsonl")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
done = set()
if os.path.exists(CACHE):
    for l in open(CACHE):
        if l.strip():
            try:
                r = json.loads(l); done.add((r["ds"], r["idx"], r["ans"]))
            except Exception:
                pass
todo = [w for w in work if (w[0], w[1], w[2]) not in done]
print(f"shard {A.shard}/{A.nshard}: {len(todo)}/{len(work)} to score -> {os.path.basename(CACHE)}", flush=True)

# group by question so each image is decoded/resized once
byq = {}
for ds, idx, ans in todo:
    byq.setdefault((ds, idx), []).append(ans)
IMG = {}
CHECK_N = 40           # candidates scored BOTH ways before trusting the per-question image cache
n_chk = [0]
t0 = time.time(); n = 0
with open(CACHE, "a") as fh:
    for (ds, idx), answers in byq.items():
        if ds not in IMG:
            print(f"  loading images for {ds}...", flush=True); IMG[ds] = imgs_for(ds)
        if idx not in IMG[ds]:
            print(f"  !! no image for {ds}/{idx}, skipping {len(answers)} candidates", flush=True)
            continue
        q, img = IMG[ds][idx]
        cache = {}
        for ans in answers:
            try:
                if n_chk[0] < CHECK_N:                      # exactness self-check on the fast path
                    s_slow = pyes(q, img, ans)
                    try:
                        s_fast = pyes_fast(q, img, ans, cache)
                    except Exception as e:                  # a broken fast path must ABORT, not be skipped
                        raise SystemExit(f"FAST PATH ERROR: {type(e).__name__}: {e} -- aborting")
                    if s_slow != s_fast:
                        raise SystemExit(f"FAST PATH MISMATCH slow={s_slow!r} fast={s_fast!r} -- aborting")
                    n_chk[0] += 1; s = s_slow
                    if n_chk[0] == CHECK_N:
                        print(f"  [self-check] fast path bit-exact on {CHECK_N}/{CHECK_N} candidates", flush=True)
                else:
                    s = pyes_fast(q, img, ans, cache)
                fh.write(json.dumps({"ds": ds, "idx": idx, "ans": ans, "pyes": round(float(s), 5)}) + "\n")
            except Exception as e:                                   # per-item error guard
                print(f"  !! {ds}/{idx} failed: {type(e).__name__}: {e}", flush=True)
            n += 1
            if n % 500 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  [{n}/{len(todo)}] {n/el:.1f}/s  eta {(len(todo)-n)/max(n/el,1e-9)/60:.1f} min", flush=True)
        fh.flush()
print(f"VERIFY_DONE shard {A.shard} n={n} in {(time.time()-t0)/60:.1f} min", flush=True)
