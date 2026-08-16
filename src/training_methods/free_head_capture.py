#!/usr/bin/env python3
"""free_head_capture.py -- BUILD 1, part B (GPU): capture the generator-frame features DURING
generation instead of recomputing them with a separate teacher-forced pass.

THE CLAIM UNDER TEST.  The frozen generator-frame head reads h_span at layer 21 of the UNMODIFIED
base Lingshu-7B for each candidate.  While the generator produced that candidate autoregressively it
computed exactly those hidden states; src/training_methods/extract_generator_hidden.py throws them
away and recomputes them in a separate teacher-forced forward pass -- 3.813646 passes/question
(= 8943 distinct normalised answers / 2345 questions).  If the captured states reproduce the
teacher-forced ones, the head becomes FREE.

WHAT THIS SCRIPT COMPUTES, per (ds, idx, normalised-answer) row, at a chosen image budget:

  TF  ("teacher-forced")  the EXACT path of extract_generator_hidden.py mode=generator:
      one forward over  chat_template(SYS_GEN, image, question) + answer , output_hidden_states,
      h_span = mean of layer-L states over the LAST n_ans_tok positions, h_last = last position.

  AR  ("captured during generation")  model.generate() from the PROMPT ONLY, with a LogitsProcessor
      that forces the candidate's token ids (then EOS).  This drives the real generation code path
      -- prefill, then one cached decode step per token -- and is numerically what a deployed
      capture would see, while guaranteeing the token sequence is the stored candidate rather than a
      fresh sample.  h(a_k) is read from generate()'s step-k hidden states; the span/pooling/layer
      are IDENTICAL to TF by construction (same function applied to the same positions).

  Both are written for the SAME row, so the comparison is exactly paired.

IMAGE BUDGET MATTERS AND IS THE SECOND HALF OF THE STORY.  The deployed pools were GENERATED at
cap320 (src/labeling/run_openvqa.py default --cap cap320 -> max_pixels = 1280*28*28//4), but the
feature cache the head was trained and evaluated on was EXTRACTED AT FULLRES (HIGH_PX = 1280*28*28).
A capture during the real generation therefore sees cap320 states, not fullres ones.  Run this
script at BOTH budgets:
  --cap fullres   AR-vs-TF in isolation, at the resolution the head was trained on; TF must
                  reproduce feats_hidden/generator_eval_s*of2.npz (harness null test)
  --cap cap320    what a capture during the DEPLOYED generation actually yields

RESUMABLE.  Features go to fp16 .npy memmaps of a fixed, deterministic row order; a per-row JSONL
records completion + metadata, and a restart skips rows already in it.  Per-row try/except.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/free_head_capture.py --cap cap320 --out feats_free
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessor
from qwen_vl_utils import process_vision_info

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import extract_generator_hidden as X   # noqa: E402  (build_eval_rows, SYS_GEN, img_md5 -- verbatim)

ROOT = X.ROOT
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8}


class ForceTokens(LogitsProcessor):
    """Force generate() to emit a known token sequence, so the AR path is the generation path but
    the tokens are the stored candidate."""

    def __init__(self, ids):
        self.ids = list(ids)
        self.i = 0

    def __call__(self, input_ids, scores):
        scores = torch.full_like(scores, -1e30)
        scores[:, self.ids[min(self.i, len(self.ids) - 1)]] = 0.0
        self.i += 1
        return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--out", default="feats_free")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    A = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False      # pinned; the project measured TF32 at -0.0089/+0.024
    torch.backends.cudnn.allow_tf32 = False
    DEV = "cuda"
    MAXPX = X.HIGH_PX // CAP_DIV[A.cap]
    outdir = os.path.join(ROOT, A.out)
    os.makedirs(outdir, exist_ok=True)
    stem = f"free_{A.cap}_L{A.layer}" + (f"_s{A.shard}of{A.nshard}" if A.nshard > 1 else "")
    print(f"[cfg] cap={A.cap} max_pixels={MAXPX} layer={A.layer} tf32={torch.backends.cuda.matmul.allow_tf32}",
          flush=True)

    # ---- rows: VERBATIM extract_generator_hidden.build_eval_rows, same global sort ---------------
    rows = X.build_eval_rows()
    rows.sort(key=lambda r: (r["ds"], str(r["idx"]), r["na"]))
    if A.limit:
        rows = rows[:A.limit]
    if A.nshard > 1:
        rows = rows[A.shard::A.nshard]
    n = len(rows)
    print(f"[build] {n} rows", flush=True)

    proc = AutoProcessor.from_pretrained(A.model_path)
    tok = proc.tokenizer
    EOS = tok.eos_token_id
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model.eval()
    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    MERGE = proc.image_processor.merge_size

    # ---- resumable storage ---------------------------------------------------------------------
    paths = {k: os.path.join(outdir, f"{stem}.{k}.npy") for k in
             ("h_span_tf", "h_last_tf", "h_span_ar", "h_last_ar")}
    arrs = {}
    for k, p in paths.items():
        arrs[k] = np.lib.format.open_memmap(p, mode=("r+" if os.path.exists(p) else "w+"),
                                            dtype=np.float16, shape=(n, H))
    logp = os.path.join(outdir, f"{stem}.rows.jsonl")
    done = {}
    if os.path.exists(logp):
        for l in open(logp):
            if l.strip():
                try:
                    r = json.loads(l)
                    if "err" in r:            # a failed row is RETRIED on resume, never skipped
                        done.pop(r["row"], None)
                    else:
                        done[r["row"]] = r
                except Exception:
                    pass
    print(f"[resume] {len(done)}/{n} rows already done", flush=True)

    memo = {"key": None}

    def image_inputs(r):
        k = (r["ds"], r["idx"])
        if memo["key"] != k:
            msgs_img = [{"role": "user", "content": [{"type": "image", "image": r["img"],
                                                      "max_pixels": MAXPX, "min_pixels": X.MIN_PX}]}]
            igs, _ = process_vision_info(msgs_img)
            ip = proc.image_processor(images=igs, return_tensors="pt")
            memo.update({"key": k, "pv": ip["pixel_values"], "grid": ip["image_grid_thw"],
                         "md5": X.img_md5(r["img"])})
        return memo["pv"], memo["grid"], memo["md5"]

    def encode(text, r):
        """extract_generator_hidden.encode, verbatim (image-pad expansion + tokenizer)."""
        pv, grid, md5 = image_inputs(r)
        npad = int(grid.prod().item()) // (MERGE ** 2)
        t = text.replace("<|image_pad|>", "<|placeholder|>" * npad).replace("<|placeholder|>", "<|image_pad|>")
        enc = proc.tokenizer([t], return_tensors="pt", padding=True)
        enc["pixel_values"] = pv
        enc["image_grid_thw"] = grid
        return enc, md5, npad

    t0 = time.time()
    fh = open(logp, "a")
    nbad = 0
    for ri, r in enumerate(rows):
        if ri in done:
            continue
        try:
            msgs = [{"role": "system", "content": X.SYS_GEN},
                    {"role": "user", "content": [{"type": "image", "image": r["img"],
                                                  "max_pixels": MAXPX, "min_pixels": X.MIN_PX},
                                                 {"type": "text", "text": r["q"]}]}]
            prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            n_ans_tok = len(tok.encode(r["ans"], add_special_tokens=False))
            full = prompt + r["ans"]

            # ---------------- TF: the extraction path, unchanged --------------------------------
            enc, md5, npad = encode(full, r)
            ids = enc["input_ids"][0]
            s0 = max(0, ids.shape[0] - n_ans_tok)
            enc_d = enc.to(DEV)
            with torch.no_grad():
                out = model(**enc_d, output_hidden_states=True)
            h = out.hidden_states[A.layer][0]
            h_span_tf = h[s0:].float().mean(0)
            h_last_tf = h[ids.shape[0] - 1].float()
            del out

            # ---------------- token-boundary check ----------------------------------------------
            enc_p, _, _ = encode(prompt, r)
            pids = enc_p["input_ids"][0]
            ans_ids_cat = ids[pids.shape[0]:].tolist()          # what the FULL string tokenizes to
            ans_ids_std = tok.encode(r["ans"], add_special_tokens=False)
            boundary_ok = (ids[:pids.shape[0]].tolist() == pids.tolist()
                           and ans_ids_cat == ans_ids_std)
            span_ok = (s0 == pids.shape[0])

            # ---------------- AR: the generation path, tokens forced ----------------------------
            force = list(ans_ids_cat) + [IM_END]
            fp = ForceTokens(force)
            enc_pd = enc_p.to(DEV)
            with torch.no_grad():
                g = model.generate(**enc_pd, max_new_tokens=len(force), do_sample=False,
                                   logits_processor=[fp], output_hidden_states=True,
                                   return_dict_in_generate=True, use_cache=True,
                                   pad_token_id=EOS)
            gen_ids = g.sequences[0, pids.shape[0]:].tolist()
            forced_ok = gen_ids[:len(ans_ids_cat)] == list(ans_ids_cat)
            nsteps = len(g.hidden_states)
            hs_ar = torch.stack([g.hidden_states[k][A.layer][0, -1].float()
                                 for k in range(1, len(ans_ids_cat) + 1)])
            h_span_ar = hs_ar.mean(0)
            h_last_ar = hs_ar[-1]
            del g

            arrs["h_span_tf"][ri] = h_span_tf.cpu().numpy().astype(np.float16)
            arrs["h_last_tf"][ri] = h_last_tf.cpu().numpy().astype(np.float16)
            arrs["h_span_ar"][ri] = h_span_ar.cpu().numpy().astype(np.float16)
            arrs["h_last_ar"][ri] = h_last_ar.cpu().numpy().astype(np.float16)

            d_span = float((h_span_tf - h_span_ar).abs().max().item())
            d_last = float((h_last_tf - h_last_ar).abs().max().item())
            fh.write(json.dumps({
                "row": ri, "ds": r["ds"], "idx": r["idx"], "na": r["na"], "ans": r["ans"],
                "y": r["y"], "img_md5": md5, "n_tok": int(ids.shape[0]),
                "n_prompt_tok": int(pids.shape[0]), "n_ans_tok": int(len(ans_ids_cat)),
                "n_ans_tok_standalone": int(len(ans_ids_std)), "n_img_tok": int(npad),
                "boundary_ok": bool(boundary_ok), "span_ok": bool(span_ok),
                "forced_ok": bool(forced_ok), "ar_steps": int(nsteps),
                "d_span_max": d_span, "d_last_max": d_last}) + "\n")
        except Exception as e:
            nbad += 1
            fh.write(json.dumps({"row": ri, "ds": r["ds"], "idx": r["idx"], "na": r["na"],
                                 "ans": r["ans"], "y": r["y"], "err": f"{type(e).__name__}: {str(e)[:160]}"}) + "\n")
            print(f"  !! row {ri} {r['ds']}/{r['idx']}: {type(e).__name__}: {str(e)[:140]}", flush=True)
        if (ri + 1) % 100 == 0:
            fh.flush()
            el = time.time() - t0
            nd = ri + 1 - len(done)
            print(f"  {ri+1}/{n}  {el/60:.1f}min  {el/max(nd,1):.3f}s/row  "
                  f"eta {(n-ri-1)*el/max(nd,1)/60:.1f}min  bad={nbad}", flush=True)
    fh.close()
    for k in arrs:
        arrs[k].flush()
    json.dump({"cap": A.cap, "max_pixels": MAXPX, "layer": A.layer, "n": n,
               "model": A.model_path, "sys_prompt": X.SYS_GEN,
               "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
               "attn": "flash_attention_2", "dtype": "bfloat16",
               "row_order": "global sort by (ds, str(idx), na) -- extract_generator_hidden's order",
               "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(outdir, f"{stem}.meta.json"), "w"), indent=1)
    print(f"DONE {stem}: {n} rows, {nbad} failed, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
