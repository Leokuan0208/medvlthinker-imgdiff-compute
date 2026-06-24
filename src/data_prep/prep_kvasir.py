#!/usr/bin/env python3
"""
prep_kvasir.py - prepare a local open-ended Kvasir-VQA-x1 subset (GI endoscopy, free-text answers; a NEW
modality for the open-ended cascade generality test, §5.7). Downloads a fixed N-sample subset's images to
/data/dan/dataset/kvasir_vqa_x1/images and writes a jsonl (idx, question, answer, img_path). Images are
HF URLs; we fetch+cache by img_id. Run: HF_HOME=/data/dan/hf_cache python3 src/data_prep/prep_kvasir.py --n 1200
"""
import argparse, json, os, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1200); A = ap.parse_args()
from datasets import load_dataset
import random
OUT = "/data/dan/dataset/kvasir_vqa_x1"; IMG = os.path.join(OUT, "images"); os.makedirs(IMG, exist_ok=True)
from concurrent.futures import ThreadPoolExecutor
d = load_dataset("SimulaMet/Kvasir-VQA-x1", split="test")
idxs = list(range(len(d))); random.Random(42).shuffle(idxs); idxs = idxs[:A.n]
meta = [(i, d[i]["question"], str(d[i]["answer"]), str(d[i]["image"]), d[i]["img_id"]) for i in idxs]
def fetch(url, ip):
    if os.path.exists(ip) and os.path.getsize(ip) > 0: return True
    try:
        urllib.request.urlretrieve(url, ip); return True
    except Exception: return False
uniq = {img_id: url for (_, _, _, url, img_id) in meta}
print(f"downloading {len(uniq)} unique images (parallel)...", flush=True)
done = 0
with ThreadPoolExecutor(max_workers=24) as ex:
    futs = {ex.submit(fetch, url, os.path.join(IMG, f"{iid}.jpg")): iid for iid, url in uniq.items()}
    for f in futs:
        if f.result(): done += 1
print(f"  {done}/{len(uniq)} images ok", flush=True)
rows = [{"idx": i, "question": q, "answer": a, "img_path": os.path.join(IMG, f"{iid}.jpg")}
        for (i, q, a, url, iid) in meta if os.path.exists(os.path.join(IMG, f"{iid}.jpg")) and os.path.getsize(os.path.join(IMG, f"{iid}.jpg")) > 0]
json.dump(rows, open(os.path.join(OUT, f"kvasir_open_{A.n}.json"), "w"))
print(f"DONE: {len(rows)} items -> {OUT}/kvasir_open_{A.n}.json ; images in {IMG}", flush=True)
