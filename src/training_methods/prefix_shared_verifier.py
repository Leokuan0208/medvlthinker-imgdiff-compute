#!/usr/bin/env python3
"""prefix_shared_verifier.py -- BUILD 2: score a question's N candidates with ONE shared
(image + question) prefill instead of N.

THE DEFECT THIS REMOVES
-----------------------
src/training_methods/cheapleg_score_open.py:125 is
    sc = [pyes(q, img, a) for a in stored[i]["preds"]]
and pyes() re-runs the processor and a full `model(**enc)` per candidate.  The image and the
question are IDENTICAL across a question's candidates; only the answer string differs (mean 5.3
tokens).  The verifier's workload is 82.1% LM prefill / 16.7% vision tower / 1.2% decode
(artifacts/cost_decomposition_2026-08-12.json:N2.stage_shares.lingshu_7b), so the deployed loop
pays the whole prefill and the whole vision tower N times to score N short strings.

WHAT THIS DOES INSTEAD
----------------------
For each question:
  1. tokenize every candidate's FULL prompt (the exact string the deployed pyes() builds),
  2. take L = the longest COMMON TOKEN PREFIX over those sequences.  Splitting on token ids, not
     on characters, is what makes the refactor exact: BPE is not compositional, so a character
     split after "Proposed answer: " would retokenize the boundary and silently change the input.
  3. run ONE forward over ids[:L] with the image, keep its KV cache,
  4. per candidate, run ONLY ids[L:] (~20 tokens) against that cache, crop the cache back to L.
The concatenation of (prefix, tail) is bit-identical to the deployed full tokenization by
construction -- asserted per question, not assumed.

M-ROPE IS PASSED EXPLICITLY.  Qwen2.5-VL derives 3D mrope positions from `get_rope_index` on the
prefill and then extrapolates with a cached `rope_deltas`.  Relying on that would make the split
depend on internal state, so position_ids for BOTH passes are sliced out of get_rope_index run on
each candidate's FULL sequence -- exactly the tensor the deployed one-shot forward would build.

*** ADAPTERS ARE SCORED UNDER HF TRANSFORMERS, NEVER vLLM *** (vLLM 0.9.0.1 silently drops all
192 visual.* LoRA modules: same adapter 0.775204 HF vs 0.702997 vLLM).

NUMERICS.  TF32 is left at this container's torch DEFAULT (matmul.allow_tf32 == True), because
that is what produced the frozen transfer dumps.  This matters: a sibling script that pins TF32
OFF re-scores the same pairs at the same resolution with max abs deviation 6.03e-2
(ckpts/openvqa/verifier_hparams/null_test_rescore.json).  --tf32 off is available to reproduce
that, but the default here is `default`.

MODES
  --mode both    score every triple BOTH ways, interleaved in one process, so the deviation AND
                 the wall-clock ratio are matched-in-session (the only honest way to time two
                 implementations on a shared box).
  --mode prefix  prefix-shared only (bulk).
  --mode base    deployed loop only (control).

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
      src/training_methods/prefix_shared_verifier.py --mode both --max_pixels 1003520
"""
import argparse
import glob
import io
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
DUMPS = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
OUTDIR = os.path.join(ROOT, "ckpts/openvqa/prefix_shared")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
DUMPNAME = {"slake_open": "slake", "vqa_rad_open": "vqa_rad", "pathvqa_open": "pathvqa"}

# verbatim from src/training_methods/verifier_transfer_eval.py (the script that produced the
# deployed transfer dumps).
DEPLOYED_MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28          # 1,003,520 / 3,136
GEN_MAXPX = 320 * 28 * 28                                    # 250,880 -- what run_openvqa.py renders at
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
       "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")


def imgs_for(ds):
    """verbatim from verifier_transfer_eval.imgs_for -- same images, same filters, same order."""
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
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
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


def load_dump_items():
    """The 2345 items in genframe_data.DUMP_ORDER (slake, vqa_rad, pathvqa)."""
    items = []
    for nm in ["slake", "vqa_rad", "pathvqa"]:
        items.extend(json.load(open(os.path.join(DUMPS, f"transfer_dump_{nm}_open_lingshu7b.json"))))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
                            "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--max_pixels", type=int, default=DEPLOYED_MAXPX)
    ap.add_argument("--mode", choices=["both", "prefix", "base"], default="both")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tf32", choices=["default", "off"], default="default")
    ap.add_argument("--limit_items", type=int, default=0, help="probe: first N questions only")
    ap.add_argument("--flush_every", type=int, default=50)
    A = ap.parse_args()

    if A.tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    os.makedirs(OUTDIR, exist_ok=True)
    tag = A.tag or f"{A.mode}_px{A.max_pixels}" + ("" if A.tf32 == "default" else "_tf32off")
    ckpt = os.path.join(OUTDIR, f"scores_{tag}.jsonl")
    print(f"[numerics] torch={torch.__version__} matmul.allow_tf32="
          f"{torch.backends.cuda.matmul.allow_tf32} cudnn.allow_tf32="
          f"{torch.backends.cudnn.allow_tf32} threads={torch.get_num_threads()}", flush=True)

    items = load_dump_items()
    if A.limit_items:
        items = items[:A.limit_items]
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try:
                    r = json.loads(l)
                    done.add((r["ds"], r["idx"]))
                except Exception:
                    pass
    todo = [it for it in items if (it["ds"], it["idx"]) not in done]
    print(f"[{tag}] max_pixels={A.max_pixels} items={len(items)} done={len(done)} "
          f"todo={len(todo)}", flush=True)
    if not todo:
        return

    proc = AutoProcessor.from_pretrained(A.model_path)
    tok = proc.tokenizer
    YES = tok.encode("Yes", add_special_tokens=False)[0]
    NO = tok.encode("No", add_special_tokens=False)[0]
    IMTOK = proc.image_token if hasattr(proc, "image_token") else "<|image_pad|>"
    IMID = tok.convert_tokens_to_ids(IMTOK)
    MERGE = proc.image_processor.merge_size ** 2

    print("loading base + FROZEN verifier adapter (HF, never vLLM)...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to("cuda")
    if A.adapter.lower() != "none":
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
        nvis = sum(1 for n, _ in model.named_parameters() if "lora" in n and "visual." in n)
        print(f"  LoRA tensors on visual.*: {nvis}", flush=True)
    model.eval()

    # the Qwen2_5_VLModel that owns get_rope_index, through the PEFT wrapper
    core = model
    while not hasattr(core, "get_rope_index"):
        core = getattr(core, "model", None) or getattr(core, "base_model")
    print(f"  rope owner: {type(core).__name__}", flush=True)

    def msgs_for(q, img, ans, maxpx):
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": maxpx, "min_pixels": MINPX},
                    {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                             f"Is the proposed answer correct? Answer Yes or No."}]}]

    def pyes_base(q, img, ans, maxpx):
        """VERBATIM the deployed scoring path (cheapleg_score_open.pyes). batch 1 = deployed."""
        msgs = msgs_for(q, img, ans, maxpx)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt",
                   padding=True).to("cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            py = math.exp(lg[YES].item())
            pn = math.exp(lg[NO].item())
        torch.cuda.synchronize()
        w = time.time() - t0
        p = py / (py + pn) if (py + pn) > 0 else 0.5
        return p, int(enc["input_ids"].shape[1]), int(enc["pixel_values"].shape[0]), w

    ones = lambda n: torch.ones((1, n), dtype=torch.long, device="cuda")

    def score_question_prefix(q, img, answers, maxpx):
        """ONE prefill for the shared image+question prefix, one short tail per candidate.

        Returns (scores, geometry dict).  Raises on any structural violation -- a silently wrong
        split is exactly the failure this build must not ship.
        """
        # ---- image ONCE (identical across candidates by construction) --------------------
        msgs0 = msgs_for(q, img, answers[0], maxpx)
        text0 = proc.apply_chat_template(msgs0, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs0)
        e0 = proc(text=[text0], images=igs, videos=vids, return_tensors="pt", padding=True)
        pv = e0["pixel_values"].to("cuda")
        grid = e0["image_grid_thw"].to("cuda")
        n_img_tok = int(e0["image_grid_thw"][0].prod()) // MERGE

        # ---- token ids per candidate, from the SAME expanded string the processor builds ---
        ids = []
        for a in answers:
            t = proc.apply_chat_template(msgs_for(q, img, a, maxpx), tokenize=False,
                                         add_generation_prompt=True)
            t = t.replace(IMTOK, IMTOK * n_img_tok, 1)
            ids.append(tok([t], return_tensors="pt")["input_ids"][0].to("cuda"))
        # sanity: candidate 0 must reproduce the processor's own input_ids exactly
        if not torch.equal(ids[0].cpu(), e0["input_ids"][0]):
            raise RuntimeError("expanded-text tokenization != processor input_ids")

        # ---- longest common TOKEN prefix ------------------------------------------------
        Lmax = min(int(x.shape[0]) for x in ids)
        L = 0
        while L < Lmax and all(int(x[L]) == int(ids[0][L]) for x in ids):
            L += 1
        if L >= Lmax:                       # all candidates identical -> nothing to share
            L = Lmax - 1
        # every image token must live inside the shared prefix, or the tail would need pixels
        last_img = int((ids[0] == IMID).nonzero()[-1])
        if last_img >= L:
            raise RuntimeError(f"image token at {last_img} >= prefix length {L}")

        wall = {}
        torch.cuda.synchronize()
        t0 = time.time()
        pos0 = core.get_rope_index(ids[0][None, :], grid, None,
                                   attention_mask=ones(int(ids[0].shape[0])))[0]
        with torch.no_grad():
            out = model(input_ids=ids[0][None, :L], attention_mask=ones(L),
                        pixel_values=pv, image_grid_thw=grid,
                        position_ids=pos0[:, :, :L], use_cache=True)
        cache = out.past_key_values
        torch.cuda.synchronize()
        wall["prefix_s"] = time.time() - t0

        scores, tail_tok, tail_wall = [], [], []
        for k, idk in enumerate(ids):
            torch.cuda.synchronize()
            t1 = time.time()
            n = int(idk.shape[0])
            cache.crop(L)
            posk = core.get_rope_index(idk[None, :], grid, None, attention_mask=ones(n))[0]
            with torch.no_grad():
                o = model(input_ids=idk[None, L:], attention_mask=ones(n),
                          past_key_values=cache, position_ids=posk[:, :, L:],
                          cache_position=torch.arange(L, n, device="cuda"), use_cache=True)
                lg = o.logits[0, -1]
                py = math.exp(lg[YES].item())
                pn = math.exp(lg[NO].item())
            torch.cuda.synchronize()
            tail_wall.append(time.time() - t1)
            tail_tok.append(n - L)
            scores.append(py / (py + pn) if (py + pn) > 0 else 0.5)
        geo = dict(prefix_tok=L, n_img_tok=n_img_tok, n_patches=int(pv.shape[0]),
                   full_tok=[int(x.shape[0]) for x in ids], tail_tok=tail_tok,
                   prefix_s=wall["prefix_s"], tail_s=tail_wall)
        del cache
        return scores, geo

    IMG = {}
    t_start = time.time()
    n_q = 0
    with open(ckpt, "a") as fh:
        for it in todo:
            ds, idx = it["ds"], it["idx"]
            if ds not in IMG:
                print(f"  loading images for {ds}...", flush=True)
                IMG[ds] = imgs_for(ds)
            if idx not in IMG[ds]:
                print(f"  !! no image for {ds}/{idx}", flush=True)
                continue
            q, img = IMG[ds][idx]
            # DISTINCT RAW answers, first-occurrence order -- the deployed unit of work
            # (8,965 over 18,760 slots = the 3.823 passes/question the cost model charges).
            seen, answers, slot_of = set(), [], []
            for a in it["preds"]:
                if a not in seen:
                    seen.add(a)
                    answers.append(a)
                slot_of.append(answers.index(a))
            rec = dict(ds=ds, idx=idx, answers=answers, slot_of=slot_of,
                       max_pixels=A.max_pixels, stored=it["scores"])
            try:
                if A.mode in ("both", "prefix"):
                    sp, geo = score_question_prefix(q, img, answers, A.max_pixels)
                    rec["prefix"] = sp
                    rec["geo"] = geo
                if A.mode in ("both", "base"):
                    bs, bt, bp, bw = [], [], [], []
                    for a in answers:
                        p, ntok, npatch, w = pyes_base(q, img, a, A.max_pixels)
                        bs.append(p); bt.append(ntok); bp.append(npatch); bw.append(w)
                    rec["base"] = bs
                    rec["base_geo"] = dict(full_tok=bt, n_patches=bp[0], wall_s=bw)
            except Exception as e:                                   # per-item error guard
                print(f"  !! {ds}/{idx} failed: {type(e).__name__}: {e}", flush=True)
                rec["error"] = f"{type(e).__name__}: {e}"
            fh.write(json.dumps(rec) + "\n")
            n_q += 1
            if n_q % A.flush_every == 0:
                fh.flush()
                el = time.time() - t_start
                print(f"  [{n_q}/{len(todo)}] {el/60:.1f} min  "
                      f"eta {(len(todo)-n_q)*el/max(n_q,1)/60:.1f} min", flush=True)
    print(f"DONE {tag} n_questions={n_q} in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
