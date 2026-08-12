#!/usr/bin/env python3
"""merge_cheapleg_lora.py -- merge a cheap-leg LoRA into full Lingshu-7B weights and save a standalone
checkpoint.

WHY MERGE INSTEAD OF SERVING THE ADAPTER.  vLLM 0.9.0.1/0.10.x SILENTLY DROPS all 192 `visual.*` LoRA
modules when it loads a PEFT adapter (the same verifier adapter scores 0.775204 under HF and 0.702997
under vLLM -- standing rule 10 of this research loop).  A MERGED checkpoint is not an adapter: vLLM
loads it through the ordinary full-weights path, exactly as it loads the base model, so the adapter
bug cannot apply.  The merge itself is done under HF/peft, where the adapter is known to load
correctly, and this script VERIFIES that both language-model AND visual tensors actually changed.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/training_methods/merge_cheapleg_lora.py --adapter ckpts/train/lora_cheapleg_s0 \
    --out ckpts/train/merged_cheapleg_s0
"""
import argparse, json, os, shutil

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
BASE_SNAP = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
             "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=BASE_SNAP)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
A = ap.parse_args()
OUT = os.path.join(ROOT, A.out)
os.makedirs(OUT, exist_ok=True)

print("loading base ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(A.base, torch_dtype=torch.bfloat16)
before = {k: v.detach().clone() for k, v in model.state_dict().items()
          if ("visual" in k and "attn.qkv.weight" in k) or k.endswith("model.layers.0.self_attn.q_proj.weight")
          or k.endswith("model.layers.20.mlp.down_proj.weight")}

print("attaching adapter ...", flush=True)
model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
print("merging ...", flush=True)
model = model.merge_and_unload()

sd = model.state_dict()
changed = {}
for k, v0 in before.items():
    v1 = sd.get(k)
    if v1 is None:
        # peft renames nothing on merge_and_unload, but be explicit rather than silent
        cand = [kk for kk in sd if kk.endswith(k.split("model.")[-1])]
        v1 = sd[cand[0]] if cand else None
    changed[k] = None if v1 is None else float((v0.float() - v1.float()).abs().max())
print("\n[verify] max |delta| on probe tensors (0.0 => the LoRA did NOT reach this module):")
for k, d in changed.items():
    print(f"  {d!s:>22}  {k}")

n_vis = sum(1 for k, d in changed.items() if "visual" in k and d)
assert any(d for k, d in changed.items() if "visual" not in k and d), "language-model weights unchanged"

model.save_pretrained(OUT, safe_serialization=True)
proc = AutoProcessor.from_pretrained(A.base)
proc.save_pretrained(OUT)
for f in ("chat_template.json", "generation_config.json"):
    src = os.path.join(A.base, f)
    if os.path.exists(src) and not os.path.exists(os.path.join(OUT, f)):
        shutil.copy(src, OUT)
json.dump({"base": A.base, "adapter": A.adapter, "out": A.out,
           "probe_tensor_max_abs_delta": changed,
           "visual_modules_touched": bool(n_vis),
           "why_merged": "vLLM silently drops visual.* LoRA modules; a merged full checkpoint is "
                         "loaded through the ordinary weights path so that bug cannot apply."},
          open(os.path.join(OUT, "merge_report.json"), "w"), indent=1)
print(f"\nwrote merged checkpoint -> {A.out}")
