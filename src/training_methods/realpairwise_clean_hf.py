#!/usr/bin/env python3
"""realpairwise_clean_hf.py -- the real-pairwise arm run on the HUGGINGFACE stack.

WHY. The vLLM run (realpairwise_clean_gpu.py) is engine-handicapped: vLLM 0.9.0.1 applies a
LoRA to the language model only and DROPS every visual.* target module, which costs the
verifier a lot -- scored pointwise under vLLM the same clean adapter gets candidate AUROC
0.760242 and sel_eff 0.702997, versus 0.885592 / 0.775204 under HuggingFace + PeftModel.
Any pairwise-vs-pointwise comparison that crosses that boundary is confounded.

This script therefore runs the IDENTICAL pairwise prompt through the SAME stack that
produced the 0.775204 bar: AutoModelForImageTextToText + PeftModel, full adapter (vision
tower included), max_pixels 1003520, first-token logits over the A / B token ids, both
orders. Then the pairwise-vs-pointwise contrast is within one engine and one adapter, and
the only thing that differs is the prompt frame.

Resumable JSONL (one row per ORDERED forward), per-item error guard, batched.

  PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/training_methods/realpairwise_clean_hf.py --dataset pathvqa_open --shard 0 --nshard 2
"""
import argparse, os, sys, json, math, time, itertools

import numpy as np
import torch

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402
from src.training_methods.realpairwise_clean_gpu import (  # noqa: E402
    PAIR_SYS, HIGH_PX, MIN_PX, imgs_for, distinct_cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=G.EVAL_DS)
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--tag", default="hf")
    ap.add_argument("--out_dir", default="ckpts/pairwise_clean")
    ap.add_argument("--max_pixels", type=int, default=HIGH_PX)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--subsample", type=int, default=0,
                    help="PRE-REGISTERED question subsample size (0 = all). Drawn with "
                         "np.random.default_rng(--subsample_seed).choice(n, k, replace=False), sorted. "
                         "Registered for pathvqa_open at k=500, seed 0, BEFORE any HF pathvqa number "
                         "was computed, because the full HF round-robin costs ~3 GPU-hours.")
    ap.add_argument("--subsample_seed", type=int, default=0)
    A = ap.parse_args()
    if os.environ.get("PAIRWISE_GPU_OK") != "1":
        sys.exit("[REFUSED] set PAIRWISE_GPU_OK=1")

    items = [it for it in G.load_items() if it["ds"] == A.dataset]
    if A.subsample and A.subsample < len(items):
        sel = np.sort(np.random.default_rng(A.subsample_seed).choice(
            len(items), A.subsample, replace=False))
        items = [items[i] for i in sel]
        print(f"[subsample] PRE-REGISTERED {len(items)} of the {A.dataset} questions "
              f"(seed {A.subsample_seed})", flush=True)
    if A.nshard > 1:
        items = items[A.shard::A.nshard]
    img_map = imgs_for(A.dataset)
    sfx = "" if A.nshard <= 1 else f"_s{A.shard}of{A.nshard}"
    os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
    outp = os.path.join(ROOT, A.out_dir, f"ordered_{A.dataset}_{A.tag}{sfx}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                r = json.loads(l)
                done.add((str(r["idx"]), int(r["ai"]), int(r["bi"]), int(r["order"])))
    print(f"[hf-pairwise] {A.dataset} shard {A.shard}/{A.nshard}: {len(items)} items, "
          f"{len(done)} rows resumed -> {outp}", flush=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel
    DEV = "cuda"
    proc = AutoProcessor.from_pretrained(A.model_path)
    proc.tokenizer.padding_side = "left"

    def tok_ids(w):
        out = []
        for v in (w, " " + w):
            e = proc.tokenizer.encode(v, add_special_tokens=False)
            if len(e) == 1:
                out.append(e[0])
        return out
    TA, TB = tok_ids("A"), tok_ids("B")
    print(f"[tok] A={TA} B={TB}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    nvis = sum(1 for n, _ in model.named_modules() if "visual" in n and "lora_A" in n)
    print(f"[adapter] FULL adapter loaded; visual lora_A modules present: {nvis}", flush=True)

    def build(q, img, ansA, ansB):
        body = (f"Question: {q}\nAnswer A: {ansA}\nAnswer B: {ansB}\n"
                f"Which candidate answer is more likely correct, A or B? Respond with only A or B.")
        return [{"role": "system", "content": PAIR_SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": A.max_pixels, "min_pixels": MIN_PX},
                    {"type": "text", "text": body}]}]

    fh = open(outp, "a"); t0 = time.time(); n = 0; nerr = 0
    buf_msgs, buf_meta = [], []

    def flush():
        nonlocal buf_msgs, buf_meta, n, nerr
        if not buf_msgs:
            return
        try:
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                     for m in buf_msgs]
            igs, vids = [], []
            for m in buf_msgs:
                a, b = process_vision_info(m)
                igs += (a or []); vids += (b or [])
            enc = proc(text=texts, images=igs or None, videos=vids or None,
                       return_tensors="pt", padding=True).to(DEV)
            with torch.no_grad():
                lg = model(**enc).logits[:, -1, :].float()
            pa = torch.stack([lg[:, t] for t in TA], 1).exp().max(1).values
            pb = torch.stack([lg[:, t] for t in TB], 1).exp().max(1).values
            p = (pa / (pa + pb)).cpu().numpy()
            p = np.where(np.isfinite(p), p, 0.5)
            for pv, m in zip(p, buf_meta):
                fh.write(json.dumps({**m, "p_first": round(float(pv), 6)}) + "\n")
                n += 1
        except Exception as e:                                    # batch-level guard
            print(f"  !! batch failed ({e}); writing errors and continuing", flush=True)
            for m in buf_meta:
                fh.write(json.dumps({**m, "p_first": None, "error": str(e)[:120]}) + "\n")
                nerr += 1
        fh.flush(); buf_msgs, buf_meta = [], []

    total = sum(len(set(G.norm(a) for a in it["preds"])) * (len(set(G.norm(a) for a in it["preds"])) - 1)
                for it in items)
    for qi, it in enumerate(items):
        na, slots, text = distinct_cands(it["preds"])
        q, img = img_map[it["idx"]]
        need = [(a, b, o) for (a, b) in itertools.combinations(range(len(na)), 2) for o in (0, 1)
                if (str(it["idx"]), a, b, o) not in done]
        for (a, b, o) in need:
            fa, fb = (na[a], na[b]) if o == 0 else (na[b], na[a])
            buf_msgs.append(build(q, img, text[fa], text[fb]))
            buf_meta.append({"ds": A.dataset, "idx": str(it["idx"]), "ai": a, "bi": b, "order": o})
            if len(buf_msgs) >= A.batch:
                flush()
        if (qi + 1) % 50 == 0:
            el = time.time() - t0
            print(f"   q{qi+1}/{len(items)} rows={n} err={nerr} {el/60:.1f} min "
                  f"{n/max(el,1e-9):.1f} row/s (target {total})", flush=True)
    flush(); fh.close()
    print(f"[hf-pairwise] DONE {A.dataset} shard {A.shard}: {n} ok / {nerr} err in "
          f"{(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
