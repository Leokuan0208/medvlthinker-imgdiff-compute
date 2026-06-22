#!/usr/bin/env python3
"""
embed_siglip.py - frozen SigLIP image+text embeddings for the competent-4 eval samples, for the
zero-leg image-content router test (the scout's actual proposed method: route by modality/visual
content, not confidence). Saves feats_peer/siglip_<ds>.npz {idx, img_emb, txt_emb}. GPU, fast.
Set HF_HOME=/data/dan/hf_cache. Run from repo root.
"""
import os, json, re, random, argparse
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from PIL import Image

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
MID = "google/siglip-so400m-patch14-384"
ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"])
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--out", default="feats_peer")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
def subset(*keys):
    return [i for i, n in enumerate(data["dataset_name"]) if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
DSI = {"PMC-VQA": lambda: subset("pmcvqa", "pmc"), "SLAKE": lambda: subset("slake"),
       "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"), "PathVQA": lambda: subset("pathvqa", "path")}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); return s[:A.n]

dev = "cuda"
model = AutoModel.from_pretrained(MID, torch_dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(MID)
print(f"loaded {MID}", flush=True)

@torch.no_grad()
def embed(idxs):
    imgs, txts = [], []
    for i in idxs:
        ex = data[i]; ims = ex.get("images") or []
        im = ims[0] if ims else Image.new("RGB", (384, 384))
        if not isinstance(im, Image.Image):
            im = Image.open(im)
        imgs.append(im.convert("RGB"))
        txts.append((ex["question"] or "")[:300])
    ie = proc(images=imgs, return_tensors="pt").to(dev)
    img_emb = model.get_image_features(**ie).float().cpu().numpy()
    te = proc(text=txts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(dev)
    txt_emb = model.get_text_features(**te).float().cpu().numpy()
    return img_emb, txt_emb

for name in A.datasets:
    sel = fixed_slice(DSI[name]())
    out = os.path.join(A.out, f"siglip_{name}.npz")
    IE, TE = [], []
    for c in range(0, len(sel), 64):
        ie, te = embed(sel[c:c + 64]); IE.append(ie); TE.append(te)
        print(f"  {name}: {min(c+64,len(sel))}/{len(sel)}", flush=True)
    np.savez(out, idx=np.array(sel), img_emb=np.concatenate(IE), txt_emb=np.concatenate(TE))
    print(f">> saved {out}  ({len(sel)} samples, dim={IE[0].shape[1]})", flush=True)
print("DONE", flush=True)
