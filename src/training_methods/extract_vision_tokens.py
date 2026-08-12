#!/usr/bin/env python3
"""extract_vision_tokens.py -- cache the generator's hidden states at VISION-TOKEN positions.

WHY.  feats_hidden/ (extract_generator_hidden.py) caches LANGUAGE-side readouts only: the last
answer token and the mean over the answer span.  Every verifier architecture this project has
tested -- ~20 of them -- reads a language-side representation.  The verifier's documented failure
mode is short laterality / anatomy answers ("Right." vs "Left."), which is a VISUAL GROUNDING
failure.  This script produces the missing raw material: the representation of the IMAGE ITSELF,
inside the same frozen base Lingshu-7B, at the same prompt frame, so a head can be given explicit,
spatially-resolved access to the picture.

THE STRUCTURAL FACT THAT MAKES THIS CHEAP (and that constrains what a vision feature can do).
Qwen2.5-VL's LM is CAUSAL and the chat template orders the sequence
      system | user: <image> <question> | assistant: <answer>
so a vision token can only attend to the system prompt and to the image.  Vision-token hidden
states are therefore a function of (system prompt, image) ALONE -- identical for every question
that shares an image and for every candidate answer of a question.  Two consequences:
  1. extraction is per-IMAGE (3,457 train + 528 eval unique images), not per (question,candidate)
     row (31,498 + 8,943) -- ~10x cheaper;
  2. a vision feature used ADDITIVELY is a per-question CONSTANT and cannot change a within-pool
     argmax.  Any gain must come from an INTERACTION between the candidate and the image.
Both are asserted, not assumed: --verify_causal checks (1) numerically, and the head-fitting
script measures (2) as a degenerate control arm.

WHAT IS SAVED, per unique image:
  v_mean[L]   (3584,)          mean over the merged vision tokens at LM layer L
  v_grid[L]   (P*P, 3584)      the merged patch grid adaptively average-pooled to P x P, so the
                               spatial layout survives (left/right is a 2-column question)
  grid_hw     (Gh, Gw)         the merged grid shape;  n_vis = Gh*Gw
Vision tokens are laid out ROW-MAJOR over the merged grid (Qwen2VLImageProcessor reshapes to
(grid_t, grid_h//merge, grid_w//merge, merge, merge, ...) before flattening) -- asserted at load.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 \
    src/training_methods/extract_vision_tokens.py --split eval --out feats_vision
"""
import argparse, os, json, sys, time, hashlib
import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import extract_generator_hidden as EG  # noqa: E402  -- reuse its row builders VERBATIM

HIGH_PX, MIN_PX = EG.HIGH_PX, EG.MIN_PX
SYS_GEN = EG.SYS_GEN


def ablate(im, kind, seed):
    """Same size, same grid, no information. 'blank' = mid-grey, 'noise' = uniform RGB noise."""
    if kind == "none":
        return im
    if not isinstance(im, Image.Image):
        im = Image.open(im)
    im = im.convert("RGB")
    w, h = im.size
    if kind == "blank":
        return Image.new("RGB", (w, h), (128, 128, 128))
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--split", choices=["eval", "train"], required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21, 28])
    ap.add_argument("--grid_layers", type=int, nargs="+", default=[21, 28])
    ap.add_argument("--pool", type=int, default=6, help="P: patch grid is pooled to P x P")
    ap.add_argument("--ablate", choices=["none", "blank", "noise"], default="none")
    ap.add_argument("--out", default="feats_vision")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify_causal", type=int, default=0,
                    help="assert vision hidden states are unchanged by the text that FOLLOWS the image")
    A = ap.parse_args()
    DEV = "cuda"
    outdir = os.path.join(ROOT, A.out); os.makedirs(outdir, exist_ok=True)
    tag = "" if A.ablate == "none" else f"_{A.ablate}"
    stem = f"vis_{A.split}{tag}"

    print(f"[build] rows for split={A.split} ...", flush=True)
    rows = EG.build_eval_rows() if A.split == "eval" else EG.build_train_rows()
    rows.sort(key=lambda r: (r["ds"], str(r["idx"]), r["na"]))

    # ---- one representative row per UNIQUE DECODED-RGB IMAGE (the project's image identity)
    seen, uniq = {}, []
    for r in rows:
        m = EG.img_md5(r["img"])
        if m not in seen:
            seen[m] = len(uniq)
            uniq.append({"img_md5": m, "img": r["img"], "ds": r["ds"], "idx": r["idx"]})
    if A.limit:
        uniq = uniq[:A.limit]
    print(f"[build] {len(rows)} rows -> {len(uniq)} unique images", flush=True)

    proc = EG.AutoProcessor.from_pretrained(A.model_path)
    tok = proc.tokenizer
    IMG_TOK = tok.convert_tokens_to_ids("<|image_pad|>")
    print(f"loading base {A.model_path} (NO adapter) ...", flush=True)
    model = EG.AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model.eval()
    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    MERGE = proc.image_processor.merge_size
    P = A.pool
    gl_idx = [A.layers.index(L) for L in A.grid_layers]

    n = len(uniq)
    v_mean = np.zeros((n, len(A.layers), H), dtype=np.float16)
    v_grid = np.zeros((n, len(A.grid_layers), P * P, H), dtype=np.float16)
    grid_hw = np.zeros((n, 2), dtype=np.int32)
    n_vis = np.zeros((n,), dtype=np.int32)
    meta, bad = [], 0
    t0 = time.time()

    def encode(img, extra_text=""):
        """system + user(<image> [+ extra_text]).  The vision block is identical either way."""
        msgs = [{"role": "system", "content": SYS_GEN},
                {"role": "user", "content": [{"type": "image", "image": img,
                                              "max_pixels": HIGH_PX, "min_pixels": MIN_PX}]
                 + ([{"type": "text", "text": extra_text}] if extra_text else [])}]
        igs, _ = EG.process_vision_info(msgs)
        ip = proc.image_processor(images=igs, return_tensors="pt")
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        npad = int(ip["image_grid_thw"].prod().item()) // (MERGE ** 2)
        t = text.replace("<|image_pad|>", "<|placeholder|>" * npad).replace("<|placeholder|>", "<|image_pad|>")
        enc = proc.tokenizer([t], return_tensors="pt", padding=True)
        enc["pixel_values"] = ip["pixel_values"]; enc["image_grid_thw"] = ip["image_grid_thw"]
        return enc

    for i, u in enumerate(uniq):
        try:
            im = ablate(u["img"], A.ablate, seed=int(u["img_md5"][:8], 16))
            enc = encode(im)
            grid = enc["image_grid_thw"][0].tolist()
            enc = enc.to(DEV)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            ids = enc["input_ids"][0]
            pos = (ids == IMG_TOK).nonzero().flatten()
            t_, gh, gw = grid
            Gh, Gw = gh // MERGE, gw // MERGE
            assert t_ == 1 and len(pos) == Gh * Gw, f"vision token count {len(pos)} != {Gh}x{Gw}"
            assert int(pos[-1] - pos[0]) == len(pos) - 1, "vision tokens are not contiguous"
            grid_hw[i] = [Gh, Gw]; n_vis[i] = len(pos)
            for li, L in enumerate(A.layers):
                hv = out.hidden_states[L][0, pos].float()                      # (n_vis, H)
                v_mean[i, li] = hv.mean(0).cpu().numpy().astype(np.float16)
                if li in gl_idx:
                    g = hv.reshape(Gh, Gw, H).permute(2, 0, 1).unsqueeze(0)    # (1,H,Gh,Gw)
                    gp = torch.nn.functional.adaptive_avg_pool2d(g, (P, P))    # (1,H,P,P)
                    v_grid[i, gl_idx.index(li)] = gp.squeeze(0).permute(1, 2, 0).reshape(P * P, H) \
                        .cpu().numpy().astype(np.float16)
            if A.verify_causal and i < 8:
                enc2 = encode(im, extra_text="Is the lesion on the left or the right side?").to(DEV)
                with torch.no_grad():
                    out2 = model(**enc2, output_hidden_states=True)
                ids2 = enc2["input_ids"][0]
                pos2 = (ids2 == IMG_TOK).nonzero().flatten()
                dev = max(float((out.hidden_states[L][0, pos].float()
                                 - out2.hidden_states[L][0, pos2].float()).abs().max())
                          for L in A.layers)
                meta.append({"causal_check_max_abs_dev": dev})
                print(f"  [verify_causal] img {i}: max|Dh_vision| with vs without trailing "
                      f"question = {dev:.3e}", flush=True)
            meta.append({"img_md5": u["img_md5"], "ds": u["ds"], "idx": u["idx"],
                         "Gh": int(Gh), "Gw": int(Gw), "n_vis": int(len(pos))})
        except Exception as e:
            bad += 1
            meta.append({"img_md5": u["img_md5"], "ds": u["ds"], "idx": u["idx"], "err": str(e)[:160]})
            print(f"  skip {u['ds']}/{u['idx']}: {str(e)[:120]}", flush=True)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{n}  {el/60:.1f}min  eta {(n-i-1)*el/(i+1)/60:.1f}min", flush=True)

    np.savez(os.path.join(outdir, f"{stem}.npz"), v_mean=v_mean, v_grid=v_grid,
             grid_hw=grid_hw, n_vis=n_vis, layers=np.array(A.layers),
             grid_layers=np.array(A.grid_layers))
    json.dump({"split": A.split, "ablate": A.ablate, "layers": A.layers, "grid_layers": A.grid_layers,
               "pool": P, "n_images": n, "n_failed": bad, "model": A.model_path, "adapter": None,
               "max_pixels": HIGH_PX, "sys_prompt": SYS_GEN, "frame": "generator",
               "minutes": round((time.time() - t0) / 60, 1), "rows": meta},
              open(os.path.join(outdir, f"{stem}.meta.json"), "w"))
    print(f"saved {outdir}/{stem}.npz  ({n} images, {bad} failed, {(time.time()-t0)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
