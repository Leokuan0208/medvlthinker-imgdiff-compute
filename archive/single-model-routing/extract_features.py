#!/usr/bin/env python3
"""
extract_features.py - cache layer-L last-token + mean-pooled hidden states
for the router. ONE frozen forward pass per question (no generation).

Mirrors gate_router.py's prompt construction EXACTLY for the baseline
(nothink_norag) cell, so features correspond to what the model saw when
it produced the baseline outcome we're routing from. Same fixed_slice
(seed-42) so idx align 1:1 with the gate checkpoints.

Output: one .npz per dataset in feats/ with arrays:
  idx[N], h_last[N,3584], h_mean[N,3584], seq_len[N]
Resumable: skips datasets whose .npz already exists with all idx present.

Reuse for background gen: --datasets PMC-VQA --root <train_root> --out feats_train
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, re, json, random, time, numpy as np, torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL    = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
RETRY_CAPS = [HIGH_PX, HIGH_PX//4, HIGH_PX//16, HIGH_PX//64]   # same OOM guard as the gate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--datasets", nargs="+",
                default=["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"])
ap.add_argument("--root", default="/data/dan/dataset/MedVLThinker-Eval")
ap.add_argument("--layer", type=int, default=14)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--out", default="feats")
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"  # single shard -> no suffix
os.makedirs(A.out, exist_ok=True)

ds = load_dataset(A.root); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]
print(f"root={A.root} split={split} N={len(data)} layer={A.layer} shard={A.shard}", flush=True)

def subset(*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt = ""
        if qt.lower()==t: out.append(i)
    return out
DATASET_IDX = {
    "MedXpert-Reasoning":     lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type("understanding"),
    "MedXpert-MM":            lambda: subset("medxpert"),
    "PMC-VQA":                lambda: subset("pmcvqa","pmc"),
    "SLAKE":                  lambda: subset("slake"),
    "VQA-RAD":                lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                lambda: subset("pathvqa","path"),
    "MMMU":                   lambda: subset("mmmu"),
}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]
    return s[SHARD_K::SHARD_N]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL)

# locate the image-pad token id so we can mean-pool over TEXT tokens only
img_pad_id = getattr(model.config, "image_token_id", None)
print("image_token_id:", img_pad_id, flush=True)

def build_inputs(ex, cap):
    opts = parse_opts(ex["options"]); assert opts, "no options"
    q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k,v in opts.items())
    img = [{"type":"image","image":im,"max_pixels":cap,"min_pixels":MIN_PX}
           for im in (ex.get("images") or [])]
    msgs = [{"role":"system","content":SYS_NOTHINK},
            {"role":"user","content":img+[{"type":"text","text":q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs)
    return proc(text=[text], images=imgs, return_tensors="pt").to(model.device)

for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! unknown {name}", flush=True); continue
    outpath = os.path.join(A.out, f"feat_{name}_L{A.layer}{SHARD_TAG}.npz")
    pool = DATASET_IDX[name]()
    if not pool: print(f"!! empty {name}", flush=True); continue
    sel = fixed_slice(pool)
    done = set()
    if os.path.exists(outpath):
        prev = np.load(outpath); done = set(int(x) for x in prev["idx"])
        if done >= set(sel): print(f"[skip] {name}: all {len(sel)} present", flush=True); continue
    print(f"\n--- {name}: {len(sel)} samples -> {outpath} ---", flush=True)

    rows_idx, rows_last, rows_mean, rows_len = [], [], [], []
    t0 = time.time()
    for j,i in enumerate(sel):
        ex = data[i]; inp = None
        for cap in RETRY_CAPS:
            try:
                inp = build_inputs(ex, cap)
                with torch.no_grad():
                    out = model(**inp, output_hidden_states=True)
                break
            except torch.cuda.OutOfMemoryError:
                if inp is not None: del inp; inp=None
                torch.cuda.empty_cache()
                print(f"   [{j+1}] OOM at cap {cap//784} -> retry", flush=True)
        if inp is None:
            print(f"   [{j+1}] OOM-SKIP idx={i}", flush=True); continue
        H = out.hidden_states[A.layer][0]                 # [seq, 3584]
        ids = inp.input_ids[0]
        h_last = H[-1].float().cpu().numpy()
        if img_pad_id is not None:
            txt_mask = (ids != img_pad_id)
            h_mean = H[txt_mask].float().mean(0).cpu().numpy()
        else:
            h_mean = H.float().mean(0).cpu().numpy()
        rows_idx.append(i); rows_last.append(h_last)
        rows_mean.append(h_mean); rows_len.append(int(ids.shape[0]))
        del inp, out, H; torch.cuda.empty_cache()
        if (j+1) % 25 == 0 or j+1 == len(sel):
            print(f"   [{j+1}/{len(sel)}] {(time.time()-t0)/(j+1):.2f}s/it", flush=True)

    np.savez(outpath,
             idx=np.array(rows_idx),
             h_last=np.stack(rows_last),
             h_mean=np.stack(rows_mean),
             seq_len=np.array(rows_len))
    print(f">> saved {len(rows_idx)} feats to {outpath}", flush=True)
print("\nDONE", flush=True)
