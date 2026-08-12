#!/usr/bin/env python3
"""vram_verifier_grid.py -- ATTACK 4, the OPEN half of the VRAM levers.

WHAT GAP THIS CLOSES.  artifacts/vram_levers_2026-08-12.json:not_measured names two holes that
together cover the 37.5% of the macro weight carried by the three open-text cells:

  "ACCURACY of the resolution lever on the OPEN-TEXT arm ... NOTHING here measures what lowering the
   generator's or the verifier's max_pixels does to open-text accuracy or to the verifier's sel_eff.
   This matters more than usual: the project's own diagnosis is that the verifier's failures are
   VISUAL GROUNDING failures on short answers, which is exactly where cutting image resolution would
   be expected to hurt first."
  "quantisation accuracy on the 3 OPEN cells ... a quantised-verifier sel_eff run is NOT in this
   artifact."   (that round could not run one: torchao-quantised weights will not take a PEFT adapter)

Both are measurable with NO new generation and NO judge call, because the pool is frozen: the 2,345
questions, their 8 sampled candidates and their per-candidate judge labels already exist.  Only the
VERIFIER's score changes when the verifier's max_pixels or weight precision changes.  So this script
re-scores the FROZEN candidate set under a (scheme x cap) grid and reports the frozen metric.

  pool     ckpts/train/lora_verifier_disjoint/transfer_dump_{slake,vqa_rad,pathvqa}_open_lingshu7b.json
           2345 questions (slake_open 645 / vqa_rad_open 200 / pathvqa_open 1500)
  metric   src/training_methods/genframe_data.py -- the SINGLE definition of sel_eff in this project
  bar      incumbent sel_eff 0.775204 | selected 0.485288 | greedy 0.449467 | oracle@8 0.626013
           per set slake 0.850088 / vqa_rad 0.761905 / pathvqa 0.722581

NULL TEST (--nulltest K).  Re-score K questions per set at the DEPLOYED configuration
(bf16, max_pixels 1,003,520) and report the max abs deviation against the stored per-candidate
scores.  Nothing else in this file may be believed until that deviation is ~0.

QUANTISATION IS bitsandbytes, NOT torchao.  bitsandbytes keeps a real nn.Linear subclass, so
PeftModel.from_pretrained attaches the 47.6M-parameter verifier adapter (192 of whose tensors are on
the vision tower) to a 4-bit or 8-bit base -- the standard QLoRA inference path.  That is why the
open half is reachable here and was not reachable with torchao weight-only.

*** HF TRANSFORMERS, NEVER vLLM *** (vLLM 0.9.0.1 drops all 192 visual.* LoRA modules: the same
adapter scores 0.775204 under HF and 0.702997 under vLLM).

Usage (from the repo root):
  HF_HOME=/data/dan/hf_cache PYTHONPATH=/home/jamesyang/.pylibs_vram CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade/vram_verifier_grid.py --arm bf16_cap1280 --nulltest 40
  ... --arm bf16_cap320   --quant none --cap 320
  ... --arm nf4_cap1280   --quant nf4  --cap 1280
"""
import argparse, glob, io, json, math, os, sys, time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src"))
DEV = "cuda"
PATCH = 28 * 28
MINPX = 4 * 28 * 28
DEPLOYED_MAXPX = 1280 * 28 * 28            # cheapleg_score_open.py:31 -- the deployed verifier cap
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
       "the proposed answer is correct. Respond with only 'Yes' or 'No'.")
VERIFIER = "ckpts/train/lora_verifier_disjoint"
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--quant", default="none", choices=["none", "int8", "nf4"])
ap.add_argument("--skip_visual", action="store_true", help="keep the ViT in bf16 while quantising the LM")
ap.add_argument("--cap", type=int, default=1280, help="max_pixels = cap * 28 * 28 (deployed = 1280)")
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--nulltest", type=int, default=0)
ap.add_argument("--limit", type=int, default=0, help="questions per set (0 = all)")
ap.add_argument("--out_dir", default="ckpts/vram_levers/verifier_grid")
A = ap.parse_args()
MAXPX = A.cap * PATCH
OUT = os.path.join(ROOT, A.out_dir, A.arm)
os.makedirs(OUT, exist_ok=True)
print(f"[cfg] arm={A.arm} quant={A.quant} skip_visual={A.skip_visual} cap={A.cap} "
      f"max_pixels={MAXPX} (deployed={DEPLOYED_MAXPX})", flush=True)
print(f"[numerics] torch={torch.__version__} tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
      f"tf32_cudnn={torch.backends.cudnn.allow_tf32}", flush=True)

from peft import PeftModel                                                       # noqa: E402
from qwen_vl_utils import process_vision_info                                    # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText              # noqa: E402


def norm(s):
    return str(s).strip().lower()


def imgs_for(ds):
    """VERBATIM from src/training_methods/cheapleg_score_open.py:imgs_for."""
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
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                       ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]

kw = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
if A.quant != "none":
    from transformers import BitsAndBytesConfig
    skip = ["lm_head"] + (["visual"] if A.skip_visual else [])
    if A.quant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip,
                                                       llm_int8_threshold=6.0)
    else:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=skip)
    kw["device_map"] = {"": 0}
print("loading base + FROZEN verifier adapter (HF, never vLLM)...", flush=True)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(A.model_path, **kw)
if A.quant == "none":
    model = model.to(DEV)
base_alloc = torch.cuda.memory_allocated()
model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER))
model.eval()
torch.cuda.synchronize()
n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
n_lora_vis = sum(1 for n, _ in model.named_parameters() if "lora_" in n and "visual." in n)
nq = sum(1 for m in model.modules() if type(m).__name__ in ("Linear8bitLt", "Linear4bit"))
LOADINFO = dict(load_s=round(time.time() - t0, 1), quant=A.quant, skip_visual=A.skip_visual,
                a_base_resident_gib=round(base_alloc / 1024 ** 3, 4),
                a_with_adapter_resident_gib=round(torch.cuda.memory_allocated() / 1024 ** 3, 4),
                n_quantized_linear=nq, n_lora_tensors=n_lora, n_lora_tensors_on_visual=n_lora_vis)
print(f"[load] {LOADINFO}", flush=True)
assert n_lora_vis == 192, f"expected 192 visual LoRA tensors, got {n_lora_vis} -- adapter did not attach"


def pyes(q, img, ans):
    """VERBATIM from cheapleg_score_open.py:pyes except that max_pixels is the swept knob."""
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": [
                {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": min(MINPX, MAXPX)},
                {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                         f"Is the proposed answer correct? Answer Yes or No."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


def score_row(q, img, preds):
    """Score the 8 slots.  BYTE-IDENTICAL answer strings are scored once and reused -- same prompt,
       same eval-mode forward pass, so this is an exact identity.

       *** THE KEY MUST BE THE RAW STRING, NOT norm(). ***  cheapleg_score_open.py:pyes is called on
       the RAW pred, so 'Right' and 'right' are two different prompts with two different scores.
       Keying the cache on norm() collapsed them and the null test caught it immediately:
       slake_open (no case-variant duplicates) reproduced at 0.000e+00 while vqa_rad_open deviated by
       9.240e-03.  Keep this keyed on `a`."""
    cache, out = {}, []
    for a in preds:
        if a not in cache:
            cache[a] = pyes(q, img, a)
        out.append(cache[a])
    return out


def stored(ds):
    return json.load(open(os.path.join(ROOT, VERIFIER, f"transfer_dump_{ds}_lingshu7b.json")))


# ------------------------------------------------------------------------------------- null test
if A.nulltest:
    dev, n = 0.0, 0
    per = {}
    for ds in DS:
        IMG = imgs_for(ds)
        d = 0.0
        for r in stored(ds)[:A.nulltest]:
            q, img = IMG[r["idx"]]
            sc = score_row(q, img, r["preds"])
            d = max(d, max(abs(round(float(x), 5) - y) for x, y in zip(sc, r["scores"])))
            n += len(sc)
        per[ds] = d
        dev = max(dev, d)
        print(f"  {ds}: max abs deviation {d:.3e}", flush=True)
    res = dict(arm=A.arm, quant=A.quant, cap=A.cap, max_pixels=MAXPX,
               nulltest_max_abs_deviation=dev, per_ds=per, n_scores=n, load=LOADINFO,
               stored_dump=f"{VERIFIER}/transfer_dump_*_lingshu7b.json",
               tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32))
    json.dump(res, open(os.path.join(OUT, "nulltest.json"), "w"), indent=1)
    print(f"\nNULL TEST max abs deviation = {dev:.3e} over {n} candidate scores -> {OUT}/nulltest.json")
    raise SystemExit(0)

# ------------------------------------------------------------------------------- score the grid arm
t0 = time.time()
for ds in DS:
    outp = os.path.join(OUT, f"scores_{ds}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                done.add(json.loads(l)["idx"])
    rows = stored(ds)
    if A.limit and A.limit < len(rows):
        # SEEDED subsample, and the seed depends only on (ds, len(rows), A.limit) -- never on the
        # arm -- so every arm scores the SAME items and the comparison stays paired.  The control is
        # the stored dump restricted to these same indices.
        import random as _r
        keep = sorted(_r.Random(42).sample(range(len(rows)), A.limit))
        rows = [rows[i] for i in keep]
    todo = [r for r in rows if r["idx"] not in done]
    print(f"[{ds}] {len(todo)}/{len(rows)} to score -> {outp}", flush=True)
    if not todo:
        continue
    IMG = imgs_for(ds)
    with open(outp, "a") as fh:
        for c, r in enumerate(todo):
            try:
                q, img = IMG[r["idx"]]
                sc = score_row(q, img, r["preds"])
                fh.write(json.dumps({"ds": ds, "idx": r["idx"],
                                     "scores": [round(float(x), 5) for x in sc]}) + "\n")
                fh.flush()
            except Exception as e:                                    # per-item error guard
                print(f"  score fail {ds}/{r['idx']}: {type(e).__name__}: {str(e)[:90]}", flush=True)
                torch.cuda.empty_cache()
            if (c + 1) % 200 == 0:
                print(f"  {ds} {c+1}/{len(todo)}  {(time.time()-t0)/60:.1f} min", flush=True)
json.dump(dict(arm=A.arm, quant=A.quant, skip_visual=A.skip_visual, cap=A.cap, max_pixels=MAXPX,
               load=LOADINFO, wall_min=round((time.time() - t0) / 60, 2),
               tf32_matmul=bool(torch.backends.cuda.matmul.allow_tf32)),
          open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print(f"\nDONE arm={A.arm} in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)
