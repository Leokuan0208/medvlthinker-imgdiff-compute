#!/usr/bin/env python3
"""vision_diversity_gen.py -- SWEEP 3: candidate diversity along the VISION axis.

WHAT IS DIFFERENT FROM THE KILLED DIVERSE ARM
---------------------------------------------
The 2026-08-10 arm (src/cascade_methods/diversity_generate_gpu.py,
artifacts/open_diverse_2026-08-10.json) varied the TEXT side: a 5-prompt portfolio x a
3-temperature ladder, i.e. it confounded prompt with temperature and never touched the image.
It was a CLEAN LOSS after decontamination.

Here the VARIED FACTOR IS THE IMAGE.  The system prompt, the user text, the temperature, top_p
and max_tokens are FROZEN at the incumbent pool's values (src/labeling/run_openvqa.py: SYS,
temp=0.7, max_tokens=64, top_p=1.0).  Only the rendered pixels and the vision-token budget move.

  resolution   the same frame at 4 token budgets (r160 / r320 / r640 / rfull)
  crop/tile    a central crop and an AnyRes-style 2x2 tiling -- for medical images a region crop
               is well motivated (the finding usually lives in a sub-region, and a tile is seen at
               HIGHER effective magnification than the downsampled full frame)
  photometric  gamma up / gamma down / autocontrast -- the software analogue of changing the
               window/level a radiologist reads at.

DELIBERATELY EXCLUDED, AND WHY (this is part of the result, not an omission)
---------------------------------------------------------------------------
  * horizontal / vertical FLIP and any rotation.  The weakest verifier stratum in this project is
    LATERALITY (sel_eff 0.613043 vs 0.817186 on short non-laterality items).  A mirrored medical
    image makes "left" the correct answer to a question whose gold is "right".  A flip augmentation
    would manufacture confident, fluent, WRONG laterality candidates -- it would destroy exactly the
    signal this attack is supposed to help, and it would inflate pool DIVERSITY while lowering pool
    QUALITY.  Not run.
  * hue / channel permutation / colour jitter.  PathVQA is H&E histology where stain COLOUR is
    diagnostic (haematoxylin blue-purple nuclei vs eosin pink cytoplasm); a hue shift is clinically
    meaningless.  Gamma/contrast are kept because they are monotone intensity remaps -- the same
    thing a viewer's window/level control does -- and preserve colour ordering.

CONTROL (mandatory).  The same script, in the SAME session and serving config, also draws an
iid pool at the BASE view (r320 = the incumbent's exact rendering convention).  Every comparison
is portfolio-vs-iid inside one process; nothing is compared to a stored number from another
config (the +-0.008 open-text reproducibility caveat).

OUTPUT SCHEMA (one line per (idx, view), resumable, per-item try/except):
  {ds, idx, view, question, gold, preds[k], vis_tokens, img_wh, err}

  DIVERSITY_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/vision_diversity_gen.py --dataset slake_open --k 2 --k_iid 24
"""
import argparse
import glob
import io
import json
import os
import sys
import time

from PIL import Image, ImageOps

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)

# ---------------------------------------------------------------- FROZEN generation config
# verbatim from src/labeling/run_openvqa.py (the script that produced the incumbent 8-sample pool)
SYS = ("You are an expert medical image analyst. Answer the question with a short, specific phrase. "
       "Do not explain.")
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
BASE_MAXPX = HIGH_PX // CAP_DIV["cap320"]          # 250880 -- the incumbent pool's budget
TEMP, MAX_TOKENS, TOP_P = 0.7, 64, 1.0             # frozen: identical to the incumbent pool

EVAL_DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


# ---------------------------------------------------------------- image views
def _crop_frac(im, l, t, r, b):
    W, H = im.size
    return im.crop((int(l * W), int(t * H), int(r * W), int(b * H)))


def _gamma(im, g):
    lut = [min(255, int(round(255.0 * ((i / 255.0) ** g)))) for i in range(256)]
    if im.mode == "RGB":
        return im.point(lut * 3)
    return im.point(lut)


def _fit_tokens(im, ntok):
    """Resize (up OR down, LANCZOS) so the frame realizes ~ntok vision tokens.

    WHY THIS IS NEEDED, and the honesty caveat.  A max_pixels cap only ever shrinks: Qwen2-VL's
    smart_resize leaves an image alone when it is already under the budget.  Measured native token
    counts (this session): SLAKE median 334, PathVQA 533, VQA-RAD 754.  So a plain cap640 / fullres
    view is BYTE-IDENTICAL to the cap320 view for most SLAKE images -- the 'resolution portfolio'
    would silently collapse into duplicates.  Forcing the budget makes each view genuinely distinct.
    CAVEAT, reported with the result: where the source is below the target, the extra tokens are
    INTERPOLATED -- they add patchification granularity, not information.
    """
    W, H = im.size
    s = (float(ntok) * 28 * 28 / max(W * H, 1)) ** 0.5
    return im.resize((max(28, int(round(W * s))), max(28, int(round(H * s)))), Image.LANCZOS)


#: name -> (pil transform, max_pixels).  min_pixels is MIN_PX everywhere (incumbent convention).
#: 'r320' is the BASE view: identity transform at the incumbent's exact max_pixels -- byte-identical
#: to how ckpts/openvqa/cheap_lingshu7b/*_sc8.jsonl was rendered, so the control is exact.
TILE = 0.60          # tile side as a fraction of the frame -> 2x2 tiles overlap by 2*0.60-1 = 20%
VIEWS = {
    # --- resolution: identity frame, four REALIZED vision-token budgets -----------------------
    "r160":   (lambda im: im, HIGH_PX // CAP_DIV["cap160"]),      # cap-only: always binds (down)
    "r320":   (lambda im: im, BASE_MAXPX),                        # BASE / control, incumbent-exact
    "up640":  (lambda im: _fit_tokens(im, 640), HIGH_PX // CAP_DIV["cap640"]),
    "up1280": (lambda im: _fit_tokens(im, 1280), HIGH_PX),
    # --- crop / tiling, each REFITTED to the base budget: AnyRes-style, so a tile is seen at
    #     higher effective magnification than the same region inside the downsampled full frame --
    "c_center": (lambda im: _fit_tokens(_crop_frac(im, 0.15, 0.15, 0.85, 0.85), 320), BASE_MAXPX),
    "t_tl": (lambda im: _fit_tokens(_crop_frac(im, 0.0, 0.0, TILE, TILE), 320), BASE_MAXPX),
    "t_tr": (lambda im: _fit_tokens(_crop_frac(im, 1 - TILE, 0.0, 1.0, TILE), 320), BASE_MAXPX),
    "t_bl": (lambda im: _fit_tokens(_crop_frac(im, 0.0, 1 - TILE, TILE, 1.0), 320), BASE_MAXPX),
    "t_br": (lambda im: _fit_tokens(_crop_frac(im, 1 - TILE, 1 - TILE, 1.0, 1.0), 320), BASE_MAXPX),
    # --- photometric (monotone intensity remaps only; no hue, no flip) ------------------------
    "p_gamma_lo": (lambda im: _gamma(im, 0.60), BASE_MAXPX),
    "p_gamma_hi": (lambda im: _gamma(im, 1.60), BASE_MAXPX),
    "p_autoc":    (lambda im: ImageOps.autocontrast(im, cutoff=1), BASE_MAXPX),
}
VIEW_GROUP = {"r160": "resolution", "r320": "base", "up640": "resolution", "up1280": "resolution",
              "c_center": "crop", "t_tl": "crop", "t_tr": "crop", "t_bl": "crop", "t_br": "crop",
              "p_gamma_lo": "photometric", "p_gamma_hi": "photometric", "p_autoc": "photometric"}

#: views EXPLICITLY REFUSED -- recorded in the artifact so the refusal is auditable, not silent.
REFUSED_VIEWS = {
    "hflip": "mirroring makes 'left' the correct answer to a 'right' gold -- it manufactures "
             "confident wrong LATERALITY candidates, and laterality is the verifier's weakest "
             "stratum (0.613043). Diversity bought by destroying the label is not coverage.",
    "vflip": "same laterality/orientation argument (superior/inferior).",
    "rot90": "same; also changes apparent anatomy orientation in CT/MR/CXR.",
    "hue_shift": "PathVQA is H&E histology -- stain colour IS the diagnostic signal; a hue shift "
                 "is clinically meaningless. Only monotone intensity remaps (gamma/autocontrast) "
                 "are kept.",
}


# ---------------------------------------------------------------- items (verbatim run_openvqa.py)
def load_items(dataset, limit=100000, idx_file=None):
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
        base = "/data/dan/dataset/vqa_rad/data" if dataset.startswith("vqa_rad_open") \
            else "/data/dan/dataset/path_vqa/data"
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                       ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            a = str(a).strip()
            if a.lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                items.append((int(i), str(q), a, Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
    if idx_file:
        allow = set(json.load(open(idx_file)))
        items = [it for it in items if it[0] in allow]
    return items[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
                            "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
    ap.add_argument("--dataset", required=True, choices=EVAL_DS)
    ap.add_argument("--ckpt_dir", default="ckpts/openvqa/visdiv")
    ap.add_argument("--k", type=int, default=2, help="samples per VIEW")
    ap.add_argument("--k_iid", type=int, default=24, help="iid samples at the BASE view (control)")
    ap.add_argument("--views", nargs="+", default=list(VIEWS))
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--idx_file", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    A = ap.parse_args()
    if os.environ.get("DIVERSITY_GPU_OK") != "1":
        sys.exit("[REFUSED] set DIVERSITY_GPU_OK=1 to use the GPU")

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def build(q, pil, maxpx):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [{"type": "image", "image": pil,
                                              "max_pixels": maxpx, "min_pixels": MIN_PX},
                                             {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        req = {"prompt": text}
        if imgs:
            req["multi_modal_data"] = {"image": imgs}
        # exact vision-token count: Qwen2-VL merges 2x2 patches of 14px -> one token per 28x28 px
        w, h = imgs[0].size if imgs else (0, 0)
        return req, int(w * h // (28 * 28)), (w, h)

    items = load_items(A.dataset, A.n, A.idx_file)
    print(f"[{A.dataset}] {len(items)} items | views={A.views} k={A.k} k_iid={A.k_iid} "
          f"| temp={TEMP} max_tokens={MAX_TOKENS} top_p={TOP_P} (FROZEN)", flush=True)

    os.makedirs(J(A.ckpt_dir), exist_ok=True)
    ckpt = J(f"{A.ckpt_dir}/gen_{A.dataset}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try:
                    r = json.loads(l)
                    done.add((str(r["idx"]), r["view"]))
                except Exception:
                    pass
    # work list: (view, maxpx, k) -- the iid control is view 'iid' at the BASE rendering
    plan = [(v, VIEWS[v][1], A.k) for v in A.views]
    if A.k_iid:
        plan.append(("iid", BASE_MAXPX, A.k_iid))

    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True, seed=A.seed)

    t0 = time.time(); ngen = 0; nfail = 0
    with open(ckpt, "a") as fh:
        for view, maxpx, k in plan:
            todo = [it for it in items if (str(it[0]), view) not in done]
            if not todo:
                print(f"  [{view}] already complete", flush=True)
                continue
            sp = SamplingParams(temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS, n=k, seed=None)
            tf = (lambda im: im) if view == "iid" else VIEWS[view][0]
            for c0 in range(0, len(todo), A.chunk):
                ch = todo[c0:c0 + A.chunk]
                reqs, meta, keep = [], [], []
                for (idx, q, gold, ref) in ch:
                    try:
                        pil = Image.open(ref).convert("RGB") if isinstance(ref, str) else ref
                        req, ntok, wh = build(q, tf(pil), maxpx)
                        reqs.append(req); meta.append((ntok, wh)); keep.append((idx, q, gold))
                    except Exception as e:                       # per-item error guard
                        nfail += 1
                        fh.write(json.dumps({"ds": A.dataset, "idx": idx, "view": view,
                                             "question": q, "gold": gold, "preds": [],
                                             "err": repr(e)[:200]}) + "\n")
                if not reqs:
                    continue
                try:
                    outs = llm.generate(reqs, sp)
                except Exception as e:                           # chunk-level guard
                    nfail += len(reqs)
                    for (idx, q, gold) in keep:
                        fh.write(json.dumps({"ds": A.dataset, "idx": idx, "view": view,
                                             "question": q, "gold": gold, "preds": [],
                                             "err": "chunk:" + repr(e)[:180]}) + "\n")
                    fh.flush()
                    continue
                for (idx, q, gold), (ntok, wh), o in zip(keep, meta, outs):
                    preds = [c.text.strip() for c in o.outputs]
                    ngen += len(preds)
                    fh.write(json.dumps({"ds": A.dataset, "idx": idx, "view": view, "question": q,
                                         "gold": gold, "preds": preds, "vis_tokens": ntok,
                                         "img_wh": list(wh)}) + "\n")
                fh.flush()
                el = time.time() - t0
                print(f"   [{view}] {min(c0 + A.chunk, len(todo))}/{len(todo)}  {el:.0f}s  "
                      f"{ngen / max(el, 1e-9):.1f} gen/s  fail={nfail}", flush=True)
    print(f"[{A.dataset}] GEN_DONE gens={ngen} fail={nfail} in {time.time() - t0:.0f}s -> {ckpt}",
          flush=True)


if __name__ == "__main__":
    main()
