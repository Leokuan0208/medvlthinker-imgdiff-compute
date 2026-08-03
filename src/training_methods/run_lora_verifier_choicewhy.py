#!/usr/bin/env python3
"""run_lora_verifier_choicewhy.py -- train the MULTIPLE-CHOICE outcome verifier on (choice)(why)
candidates (Phase 2 of the choicewhy program).

ARCHITECTURE / OBJECTIVE / HYPERPARAMETERS ARE THOSE OF ckpts/train/lora_verifier_disjoint, verbatim
(src/training_methods/run_lora_verifier_disjoint.py; the checkpoint's own adapter_config.json and
train_config.json were read to confirm each value):

  base model  lingshu-medical-mllm/Lingshu-7B, bf16, flash_attention_2
  LoRA        r=16, alpha=2*r=32, dropout=0.05, bias=none,
              target_modules = q,k,v,o,gate,up,down _proj
  objective   next-token CE on the single "Yes"/"No" continuation token (labels[:, :-1] = -100)
  optimizer   AdamW lr=1e-4, grad-clip 1.0, bs=2, accum=8, epochs=1
  images      max_pixels = 1280*28*28 (cap_div 1, i.e. fullres), min_pixels = 4*28*28
  seed        0

THE ONLY CHANGES vs that reference are the ones the experiment is about:
  (1) the task is MCQ, so the verifier's user turn carries the OPTION BLOCK as well as the question
      (a proposed answer of "B" is uninterpretable without it) -- and it carries it IDENTICALLY in
      every arm, so the option block cannot be the variable;
  (2) the proposed answer is the model's own sampled candidate: a bare letter in arm A, a
      "<letter>. <one-sentence finding>" string in arm B2.  FORMAT is the variable under test.

The training examples come from src/training_methods/build_choicewhy_verifier_examples.py, which draws
identical per-source example counts for every arm from a pool proven disjoint from every evaluation
image (src/training_methods/build_choicewhy_mcq_split.py).  Nothing is split off for validation: every
evaluation item is held out BY CONSTRUCTION, and measurement happens afterwards over the Phase-1
evaluation candidates.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/run_lora_verifier_choicewhy.py --arm B2 \
    --out_dir ckpts/train/lora_verifier_choicewhy_B2
"""
import argparse, hashlib, json, os, random, sys, time
import numpy as np, torch
from collections import Counter
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src", "cascade_methods"))
from choicewhy_common import ARM_NAME, VERIF_SYS, verifier_body  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--arm", default="B2", choices=list(ARM_NAME))
ap.add_argument("--variant", default="", choices=["", "posmatched"],
                help="'posmatched' trains on the variant whose per-source positive/negative counts equal "
                     "the reference arm's, isolating discrimination from the label base rate")
ap.add_argument("--examples_dir", default="data/choicewhy_mcq_split")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=2)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--cap_div", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out_dir", default=None)
ap.add_argument("--deadline_s", type=float, default=0.0, help=">0: stop training cleanly at this wall time")
A = ap.parse_args()
AN = ARM_NAME[A.arm]
SUF = f"_{A.variant}" if A.variant else ""
A.out_dir = A.out_dir or f"ckpts/train/lora_verifier_choicewhy_{A.arm}{SUF}"
os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX // A.cap_div
DEV = "cuda"
T0 = time.time()

EX_PATH = os.path.join(ROOT, A.examples_dir, f"verifier_examples_{AN}{SUF}.jsonl")
train_ex = [json.loads(l) for l in open(EX_PATH) if l.strip()]
assert train_ex, f"no training examples in {EX_PATH}"
pos_rate = float(np.mean([e["label"] for e in train_ex]))
print(f"[data] {EX_PATH}", flush=True)
print(f"[data] {len(train_ex)} examples | {len({(e['src'],e['idx']) for e in train_ex})} questions | "
      f"{len({e['image_md5_rgb'] for e in train_ex})} images", flush=True)
print(f"[data] per-source {dict(Counter(e['src'] for e in train_ex))}", flush=True)
print(f"[data] POSITIVE-LABEL RATE {pos_rate:.4f}  (base-rate shifts move a verifier's operating point "
      f"independently of its discrimination -- recorded here and in train_config.json)", flush=True)

# every training image must still hash to the value the disjoint-split builder cleared
print("[assert] re-verifying staged training images against the proven decoded-RGB md5 ...", flush=True)
_checked = {}
for e in train_ex:
    if e["image_md5_rgb"] in _checked:
        continue
    h = hashlib.md5(Image.open(e["img_path"]).convert("RGB").tobytes()).hexdigest()
    assert h == e["image_md5_rgb"], f"IMAGE MISMATCH {e['img_path']}"
    _checked[e["image_md5_rgb"]] = 1
print(f"[assert] {len(_checked)} distinct training images verified", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]


def build_msgs(e):
    return [{"role": "system", "content": VERIF_SYS},
            {"role": "user", "content": [{"type": "image", "image": Image.open(e["img_path"]).convert("RGB"),
                                          "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                         {"type": "text", "text": verifier_body(e["question"], e["options"],
                                                                                e["candidate"])}]}]


def encode(e, label=None):
    msgs = build_msgs(e)
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if label is not None:
        text = text + ("Yes" if label == 1 else "No")
    imgs, vids = process_vision_info(msgs)
    return proc(text=[text], images=imgs, videos=vids, return_tensors="pt", padding=True)


print("loading Lingshu-7B + LoRA...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.model_path, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").to(DEV)
lcfg = LoraConfig(r=A.lora_r, lora_alpha=2 * A.lora_r, lora_dropout=0.05, bias="none",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=A.lr)
rng = random.Random(A.seed)
model.train()
t0 = time.time(); step = 0; nskip = 0; stop = False
for ep in range(A.epochs):
    rng.shuffle(train_ex)
    for bi in range(0, len(train_ex), A.bs):
        for e in train_ex[bi:bi + A.bs]:
            try:
                enc = encode(e, e["label"]).to(DEV)
                labels = enc["input_ids"].clone(); labels[:, :-1] = -100
                out = model(**enc, labels=labels)
                (out.loss / (A.bs * A.accum)).backward()
            except Exception as ex_:
                nskip += 1
                print(f"  skip: {str(ex_)[:70]}", flush=True)
        step += 1
        if step % A.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if step % 50 == 0:
            print(f"  ep{ep} step{step}/{len(train_ex)//A.bs} {(time.time()-t0)/60:.1f}min", flush=True)
        if A.deadline_s > 0 and (time.time() - T0) > A.deadline_s:
            print(f"  DEADLINE at step {step}; stopping cleanly", flush=True); stop = True; break
    if stop:
        break

model.save_pretrained(os.path.join(ROOT, A.out_dir))
json.dump({
    "program": "choicewhy Phase 2 -- MCQ (choice)(why) outcome verifier",
    "arm": AN, "variant": A.variant or "natural", "examples_file": os.path.relpath(EX_PATH, ROOT),
    "n_train_examples": len(train_ex),
    "n_train_questions": len({(e["src"], e["idx"]) for e in train_ex}),
    "n_train_images": len(_checked),
    "per_source_examples": dict(Counter(e["src"] for e in train_ex)),
    "pos_rate": round(pos_rate, 4),
    "pos_rate_per_source": {s: round(float(np.mean([e["label"] for e in train_ex if e["src"] == s])), 4)
                            for s in sorted({e["src"] for e in train_ex})},
    "reference_config": "ckpts/train/lora_verifier_disjoint (adapter_config.json + train_config.json)",
    "skipped_examples": nskip, "steps": step, "epochs": A.epochs, "bs": A.bs, "accum": A.accum,
    "lr": A.lr, "lora_r": A.lora_r, "lora_alpha": 2 * A.lora_r, "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "base_model": A.model_path, "cap_div": A.cap_div, "max_pixels": MAXPX, "seed": A.seed,
    "verifier_prompt": "src/cascade_methods/choicewhy_common.py::VERIF_SYS + verifier_body "
                       "(question + option block + proposed answer); identical in every arm",
    "train_minutes": round((time.time() - t0) / 60, 1), "early_stopped": stop,
}, open(os.path.join(ROOT, A.out_dir, "train_config.json"), "w"), indent=1)
print(f"saved -> {A.out_dir} in {(time.time()-t0)/60:.1f} min", flush=True)
