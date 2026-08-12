#!/usr/bin/env python3
"""train_cheap_leg.py -- ATTACK B: LoRA-adapt the GENERATOR (Lingshu-7B, the cascade's cheap leg) on
image-disjoint medical-VQA TRAIN splits.

WHY.  Every ceiling this project has measured -- p10 = P(7B right & 32B wrong), oracle@8, greedy,
sel_eff -- is a property of a FROZEN Lingshu-7B.  The verifier has been trained repeatedly (~20
approaches, all at sel_eff 0.80-0.81); the generator never has.  Training the cheap leg moves greedy,
oracle@8 and p10 at once, i.e. it moves the frontier instead of harvesting inside it.

RECIPE = the verifier's, deliberately (src/training_methods/run_lora_verifier_disjoint.py). This is a
PROBE, not a scaling study:
  base       lingshu-medical-mllm/Lingshu-7B, bf16, flash_attention_2
  LoRA       r=16, alpha=32, dropout=0.05, bias=none,
             target_modules = q,k,v,o,gate,up,down _proj
  optimizer  AdamW lr=1e-4, grad-clip 1.0, bs=2, accum=8, epochs=1
  objective  next-token CE on the ANSWER tokens only (the prompt is masked with -100)

DATA = data/cheapleg_split/train_manifest.json, built and PROVEN image-disjoint by
src/training_methods/build_cheapleg_train_split.py (md5 of decoded RGB pixels; the build asserts an
empty intersection with every image behind the eight Variant-B reporting cells and fails otherwise).

TWO PROMPT FRAMES, because the cascade's two arms are evaluated under two different prompts AND two
different image resolutions, and a resolution/prompt mismatch is precisely the defect that forced this
project's Finding-1 correction:
  medeval  MedEvalKit's own prompt strings (utils/question_formats.py), FULL resolution
           -> PMC_VQA (multiple choice), SLAKE_closed, VQA_RAD_closed, PATH_VQA_closed
  openvqa  run_openvqa.py's SYS prompt, cap320 (1280*28*28//4 px)
           -> SLAKE_open, VQA_RAD_open, PATH_VQA_open

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/training_methods/train_cheap_leg.py \
      --seed 0 --out_dir ckpts/train/lora_cheapleg_s0
"""
import argparse, glob, io, json, os, random, time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoModelForImageTextToText

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")

# ---- prompt strings, copied VERBATIM from the two eval harnesses (never paraphrased) --------------
# MedEvalKit/utils/question_formats.py (is_reasoning=False, lang=en)
MEK_CLOSE = "Answer the question using a single word or phrase."
MEK_OPEN = "Please answer the question concisely."
MEK_YESNO = "Please output 'yes' or 'no'(no extra output)."
MEK_MCQ = "Answer with the option's letter from the given choices directly."
# src/labeling/run_openvqa.py SYS
OPENVQA_SYS = ("You are an expert medical image analyst. Answer the question with a short, specific "
               "phrase. Do not explain.")

HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP320 = HIGH_PX // 4                      # run_openvqa.py default --cap cap320
FULLRES = 12845056                         # Lingshu-7B preprocessor_config.json max_pixels

# per-source example quotas. Chosen BEFORE any measurement, to (a) put most of the weight on the
# multiple-choice cell that carries 5/8 of the macro weight's only converting cell (PMC_VQA) and
# (b) cover every one of the 8 reporting cells that has a usable train split.
DEFAULT_QUOTA = {
    "pmc_vqa_train_mcq": 5000,
    "slake_train_closed": 800,
    "vqa_rad_train_closed": 700,
    "pathvqa_train_closed": 2000,
    "slake_train_open": 1200,
    "vqa_rad_train_open": 700,
    "pathvqa_train_open": 1600,
}

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--manifest", default="data/cheapleg_split/train_manifest.json")
ap.add_argument("--out_dir", default="ckpts/train/lora_cheapleg_s0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--train_maxpx", type=int, default=HIGH_PX,
                help="cap on image pixels for the MEDEVAL-frame training examples. Eval runs full "
                     "res; training at full res on PMC journal figures is ~4x slower for no probe "
                     "value, so it is capped and the cap is RECORDED as a known train/eval mismatch.")
ap.add_argument("--quota", default="", help="json dict overriding DEFAULT_QUOTA")
ap.add_argument("--deadline_s", type=float, default=0.0)
ap.add_argument("--supervise_eos", type=int, default=0,
                help="0 (default) = supervise the ANSWER TOKENS ONLY; the turn terminator is NOT in "
                     "the loss, so the adaptation cannot teach the model a new answer FORMAT or a "
                     "new stopping habit -- only content. This matters here: answer FORMAT is a "
                     "known confound in this project (Finding 1, CLAUDE.md sec 0), and supervising "
                     "<|im_end|> after a bare letter costs ~12 nats/token on the base model, which "
                     "would dominate the gradient and make the probe a format-training run.")
ap.add_argument("--debug_n", type=int, default=0,
                help=">0: print the decoded prompt tail / unmasked label span / base-model loss for "
                     "this many examples and exit (encoding sanity check, no training)")
A = ap.parse_args()
QUOTA = dict(DEFAULT_QUOTA)
if A.quota:
    QUOTA.update(json.loads(A.quota))
os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
DEV = "cuda"
T0 = time.time()

# ======================================================================== 1. examples
MAN = json.load(open(os.path.join(ROOT, A.manifest)))
rng = random.Random(A.seed)


def parquet_images(base, idxs):
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}-*.parquet"))],
                   ignore_index=True)
    want = set(int(i) for i in idxs)
    return {int(i): r["image"]["bytes"] for i, r in df.iterrows() if int(i) in want}


print("[data] selecting examples ...", flush=True)
sel = {}
for src, want in QUOTA.items():
    pool = list(MAN.get(src, []))
    rng.shuffle(pool)
    sel[src] = pool[:want]
    print(f"  {src:22s} available={len(pool):6d} taken={len(sel[src]):5d}"
          f"{'  *** SHORT ***' if len(sel[src]) < want else ''}", flush=True)

# resolve parquet image bytes once per family
PQ = {}
for src in ("vqa_rad_train_closed", "vqa_rad_train_open", "pathvqa_train_closed", "pathvqa_train_open"):
    rows = sel.get(src, [])
    if not rows:
        continue
    base = rows[0]["parquet"]
    PQ.setdefault(base, set()).update(int(r["idx"]) for r in rows)
PQIMG = {}
for base, idxs in PQ.items():
    print(f"  loading {len(idxs)} images from {base}-*.parquet", flush=True)
    PQIMG[base] = parquet_images(base, idxs)

PMC_FIG = "/data/dan/dataset/medevalkit/PMC-VQA/figures"


def surface(ans):
    """Put the gold answer into the BASE MODEL'S OWN surface convention before supervising it.

    Lingshu-7B answers these prompts as 'Yes.' / 'CT.' / 'Left side.' -- capitalised, terminated by a
    period -- while the parquet golds are bare lowercase ('yes', 'multiple circumscribed').  Every
    scorer in this project is case- and punctuation-insensitive (run_openvqa.norm, MedEvalKit
    judge_judgement/judge_open_end_vqa, the 32B judge), so this rewrite cannot change any measured
    correctness.  What it prevents is the adaptation silently learning a new ANSWER FORMAT (lowercase,
    unterminated) instead of new content -- a change that would also shift the surface strings the
    FROZEN verifier sees, confounding the whole probe.  Format is a known confound here (Finding 1).
    """
    a = str(ans).strip()
    if not a:
        return a
    a = a[0].upper() + a[1:]
    return a if a[-1] in ".?!" else a + "."


def build_example(r):
    """-> (system_or_None, user_text, image (PIL or path), max_pixels, target_text)"""
    src = r["source"]
    if src == "pmc_vqa_train_mcq":
        opts = "\n".join(str(c) for c in r["choices"])
        text = f"\nQuestion: {r['question']}\nOptions: \n{opts}\n{MEK_MCQ}"
        return None, text, os.path.join(PMC_FIG, r["fig"]), A.train_maxpx, str(r["answer"]).strip()
    if src == "slake_train_closed":
        return None, f"{r['question']}\n{MEK_CLOSE}", r["img"], A.train_maxpx, surface(r["answer"])
    if src == "slake_train_open":
        return OPENVQA_SYS, r["question"], r["img"], CAP320, surface(r["answer"])
    img = Image.open(io.BytesIO(PQIMG[r["parquet"]][int(r["idx"])]))
    if src.endswith("_closed"):
        return None, f"{r['question']}\n{MEK_YESNO}", img, A.train_maxpx, surface(r["answer"])
    return OPENVQA_SYS, r["question"], img, CAP320, surface(r["answer"])


train_ex = []
for src, rows in sel.items():
    for r in rows:
        try:
            train_ex.append((src,) + build_example(r))
        except Exception as e:                                    # per-item error guard
            print(f"  skip build {src}/{r.get('idx')}: {str(e)[:60]}", flush=True)
rng.shuffle(train_ex)
print(f"[data] {len(train_ex)} training examples", flush=True)
assert train_ex, "no training examples"

# ======================================================================== 2. model
proc = AutoProcessor.from_pretrained(A.model_path)
IM_END = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")


def encode(system, user_text, img, maxpx, target):
    im = {"type": "image", "image": img, "max_pixels": int(maxpx), "min_pixels": MIN_PX}
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": [im, {"type": "text", "text": user_text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    e_p = proc(text=[prompt], images=imgs, videos=vids, return_tensors="pt", padding=True)
    n_prompt = int(e_p["input_ids"].shape[1])
    e_f = proc(text=[prompt + target], images=imgs, videos=vids, return_tensors="pt", padding=True)
    ids = e_f["input_ids"]
    if A.supervise_eos:
        ids = torch.cat([ids, torch.tensor([[IM_END]])], dim=1)
        e_f["input_ids"] = ids
        e_f["attention_mask"] = torch.ones_like(ids)
    labels = ids.clone()
    labels[:, :n_prompt] = -100
    return e_f, labels, int(ids.shape[1] - n_prompt)


print("loading Lingshu-7B + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)

if A.debug_n:
    # ENCODING SANITY CHECK: what exactly is the model being trained to emit, and what does the
    # UNADAPTED base model already score on it?  A high base loss here means the target format
    # differs from what the base model naturally emits -- which is the point of the probe, but it
    # must be seen, not assumed.
    model.eval()
    for (src, system, ut, img, mpx, tgt) in train_ex[:A.debug_n]:
        enc, labels, ntok = encode(system, ut, img, mpx, tgt)
        ids = enc["input_ids"][0]
        keep = labels[0] != -100
        with torch.no_grad():
            out = model(**{k: v.to(DEV) for k, v in enc.items()}, labels=labels.to(DEV))
        print(f"\n--- {src}  target_tokens={ntok}  base_loss={float(out.loss):.4f}")
        print(f"    prompt tail : {proc.tokenizer.decode(ids[-(ntok+24):-ntok])!r}")
        print(f"    LABEL span  : {proc.tokenizer.decode(ids[keep])!r}")
        print(f"    label ids   : {ids[keep].tolist()}")
    raise SystemExit(0)
lcfg = LoraConfig(r=A.lora_r, lora_alpha=2 * A.lora_r, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)

# ======================================================================== 3. train
model.train()
t0 = time.time()
step = 0
nskip = 0
stop = False
losses = []
tgt_len = []
for ep in range(A.epochs):
    rng.shuffle(train_ex)
    for bi in range(0, len(train_ex), A.bs):
        for (src, system, ut, img, mpx, tgt) in train_ex[bi:bi + A.bs]:
            try:
                enc, labels, ntok = encode(system, ut, img, mpx, tgt)
                enc = {k: v.to(DEV) for k, v in enc.items()}
                out = model(**enc, labels=labels.to(DEV))
                (out.loss / (A.bs * A.accum)).backward()
                losses.append(float(out.loss.item()))
                tgt_len.append(ntok)
            except Exception as e:
                nskip += 1
                print(f"  skip: {str(e)[:80]}", flush=True)
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if step % 100 == 0:
            print(f"  ep{ep} step{step}/{len(train_ex)//A.bs} loss(last200)="
                  f"{np.mean(losses[-200:]):.4f} {(time.time()-t0)/60:.1f}min", flush=True)
        if A.deadline_s > 0 and (time.time() - T0) > A.deadline_s:
            print(f"  DEADLINE at step {step}; stopping cleanly", flush=True)
            stop = True
            break
    if stop:
        break

model.save_pretrained(os.path.join(ROOT, A.out_dir))
json.dump({"attack": "B -- train the cheap leg (generator LoRA)",
           "manifest": A.manifest, "quota": QUOTA,
           "taken_per_source": {k: len(v) for k, v in sel.items()},
           "n_train_examples": len(train_ex), "skipped": nskip, "steps": step,
           "epochs": A.epochs, "bs": A.bs, "accum": A.accum, "lr": A.lr,
           "lora_r": A.lora_r, "lora_alpha": 2 * A.lora_r, "lora_dropout": 0.05,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
           "base_model": A.model_path, "seed": A.seed,
           "train_maxpx_medeval_frame": A.train_maxpx,
           "eval_maxpx_medeval_frame": FULLRES,
           "known_train_eval_mismatch": "medeval-frame training images are capped at "
                                        f"{A.train_maxpx} px while MedEvalKit evaluates at full "
                                        f"resolution ({FULLRES} px cap). Recorded, not hidden.",
           "openvqa_frame_px": CAP320,
           "loss_first200": float(np.mean(losses[:200])) if losses else None,
           "loss_last200": float(np.mean(losses[-200:])) if losses else None,
           "mean_target_tokens": float(np.mean(tgt_len)) if tgt_len else None,
           "train_minutes": round((time.time() - t0) / 60, 1), "early_stopped": stop},
          open(os.path.join(ROOT, A.out_dir, "train_config.json"), "w"), indent=1)
print(f"saved -> {A.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)
