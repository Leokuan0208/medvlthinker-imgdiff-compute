#!/usr/bin/env python3
"""align_embed_all.py -- frozen contrastive image/text embeddings for every distinct image and every
distinct candidate string in data/align_cache/manifest.json, for three encoders:

  siglip      google/siglip-so400m-patch14-384          (general-domain, already on disk)
  pubmedclip  flaviagiammarino/pubmed-clip-vit-base-patch32   (CLIP ViT-B/32 fine-tuned on ROCO)
  biomedclip  microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224  (open_clip)

Text templates (all three embedded; the template is chosen on the TRAIN pools only):
  ans   "<answer>"
  qa    "<question> <answer>"
  decl  "medical image. <question> answer: <answer>"

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/verifier_arch/align_embed_all.py
  -> data/align_cache/emb_<encoder>.npz  {img_hash, img_emb, txt_key, txt_emb}
Run from the repo root.
"""
import os, json, argparse
import numpy as np, torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "data/align_cache")
IMGDIR = os.path.join(OUT, "img")

ap = argparse.ArgumentParser()
ap.add_argument("--encoders", nargs="+", default=["siglip", "pubmedclip", "biomedclip"])
ap.add_argument("--bs", type=int, default=64)
A = ap.parse_args()
DEV = "cuda"

man = json.load(open(os.path.join(OUT, "manifest.json")))
rows = man["eval"] + man["train"]
img_hashes = sorted({r["img"] for r in rows})

TEMPLATES = {
    "ans": lambda q, a: a,
    "qa": lambda q, a: f"{q} {a}",
    "decl": lambda q, a: f"medical image. {q} answer: {a}",
}
txt_keys = set()
for r in rows:
    for a in set(r["preds"]):
        for t, f in TEMPLATES.items():
            txt_keys.add(t + "\x00" + f(r["q"], a)[:400])
txt_keys = sorted(txt_keys)
print(f"{len(img_hashes)} distinct images, {len(txt_keys)} distinct (template,text) strings", flush=True)


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def run_siglip():
    from transformers import AutoModel, AutoProcessor
    MID = "google/siglip-so400m-patch14-384"
    m = AutoModel.from_pretrained(MID, torch_dtype=torch.bfloat16).to(DEV).eval()
    p = AutoProcessor.from_pretrained(MID)

    @torch.no_grad()
    def enc_img(paths):
        ims = [Image.open(x).convert("RGB") for x in paths]
        e = p(images=ims, return_tensors="pt").to(DEV)
        return m.get_image_features(**e).float().cpu().numpy()

    @torch.no_grad()
    def enc_txt(txts):
        e = p(text=txts, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(DEV)
        return m.get_text_features(**e).float().cpu().numpy()
    return enc_img, enc_txt, (lambda: (m.__class__, None))


def run_pubmedclip():
    from transformers import CLIPModel, CLIPProcessor
    MID = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    m = CLIPModel.from_pretrained(MID, torch_dtype=torch.float32).to(DEV).eval()
    p = CLIPProcessor.from_pretrained(MID)

    @torch.no_grad()
    def enc_img(paths):
        ims = [Image.open(x).convert("RGB") for x in paths]
        e = p(images=ims, return_tensors="pt").to(DEV)
        return m.get_image_features(**e).float().cpu().numpy()

    @torch.no_grad()
    def enc_txt(txts):
        e = p(text=txts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEV)
        return m.get_text_features(**e).float().cpu().numpy()
    return enc_img, enc_txt, None


def run_biomedclip():
    import open_clip
    MID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    m, prep = open_clip.create_model_from_pretrained(MID)
    tok = open_clip.get_tokenizer(MID)
    m = m.to(DEV).eval()

    @torch.no_grad()
    def enc_img(paths):
        ims = torch.stack([prep(Image.open(x).convert("RGB")) for x in paths]).to(DEV)
        return m.encode_image(ims).float().cpu().numpy()

    @torch.no_grad()
    def enc_txt(txts):
        t = tok(txts, context_length=256).to(DEV)
        return m.encode_text(t).float().cpu().numpy()
    return enc_img, enc_txt, None


BUILD = {"siglip": run_siglip, "pubmedclip": run_pubmedclip, "biomedclip": run_biomedclip}

for name in A.encoders:
    outp = os.path.join(OUT, f"emb_{name}.npz")
    if os.path.exists(outp):
        print(f"skip {name} (exists)", flush=True); continue
    enc_img, enc_txt, _ = BUILD[name]()
    IE = []
    for c in batched(img_hashes, A.bs):
        IE.append(enc_img([os.path.join(IMGDIR, h + ".png") for h in c]))
        if len(IE) % 10 == 0: print(f"  {name} img {len(IE)*A.bs}/{len(img_hashes)}", flush=True)
    TE = []
    for c in batched(txt_keys, 256):
        TE.append(enc_txt([k.split("\x00", 1)[1] for k in c]))
        if len(TE) % 20 == 0: print(f"  {name} txt {len(TE)*256}/{len(txt_keys)}", flush=True)
    np.savez(outp, img_hash=np.array(img_hashes), img_emb=np.concatenate(IE),
             txt_key=np.array(txt_keys), txt_emb=np.concatenate(TE))
    print(f">> {outp}  img={len(img_hashes)} txt={len(txt_keys)} dim={TE[0].shape[1]}", flush=True)
    del enc_img, enc_txt
    torch.cuda.empty_cache()
print("DONE", flush=True)
