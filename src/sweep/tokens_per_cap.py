#!/usr/bin/env python3
"""
tokens_per_cap.py - exact 7B prompt-token (text+vision) count at every resolution cap.
Cap label files store gen_tokens (decode) but NOT prompt length P; the cost model needs P per
cap for the cheap leg. Runs the Qwen processor (NO model weights) over build() at each
max_pixels, mirroring run_7b_vllm.build()/run_7b_prune_sweep.build(). CPU only.
Output: ckpts/token_cache.json = {benchmark:{cap:{idx:[P,vis]}}}
"""
import argparse, json, glob, os, re, time
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_7B = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT     = "/data/dan/dataset/MedVLThinker-Eval"
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAPS = {"fullres":1, "cap640":2, "cap320":4, "cap160":8, "cap80":16}   # max_pixels = HIGH_PX//div
MERGE = 2

ap = argparse.ArgumentParser()
ap.add_argument("--src_dir", default="ckpts/gate_32b", help="label dir whose idx define the evaluated set")
ap.add_argument("--cell", default="think_norag")
ap.add_argument("--out", default="ckpts/token_cache.json")
ap.add_argument("--max_per_ds", type=int, default=0)
A = ap.parse_args()

def load_arm(ckdir, cell):
    pat = re.compile(rf"ckpt_(.+?)_{re.escape(cell)}(?:_s\d+of\d+)?\.jsonl$"); d=defaultdict(dict)
    for f in glob.glob(os.path.join(ckdir, f"*{cell}*.jsonl")):
        m=pat.search(os.path.basename(f))
        if not m: continue
        for l in open(f):
            if l.strip():
                try: r=json.loads(l); d[m.group(1)][r["idx"]]=r
                except Exception: pass
    return d
def parse_opts(s):
    if isinstance(s,dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
proc = AutoProcessor.from_pretrained(MODEL_7B)
idx_by = {k: sorted(v.keys()) for k,v in load_arm(A.src_dir, A.cell).items()}
print(f"counting tokens for {sum(len(v) for v in idx_by.values())} samples x {len(CAPS)} caps", flush=True)

def count(ex, max_pixels):
    opts = parse_opts(ex["options"])
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    im = [{"type":"image","image":x,"max_pixels":max_pixels,"min_pixels":MIN_PX} for x in (ex.get("images") or [])]
    msgs=[{"role":"system","content":SYS_NOTHINK},{"role":"user","content":im+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    out=proc(text=[text], images=imgs if imgs else None, return_tensors="pt")
    P=int(out["input_ids"].shape[1])
    vis=int(out["image_grid_thw"].prod(dim=-1).sum().item())//(MERGE*MERGE) if "image_grid_thw" in out else 0
    return P, vis

cache={}; t0=time.time(); done=0
for ds_name, idxs in idx_by.items():
    if A.max_per_ds: idxs = idxs[:A.max_per_ds]
    cache[ds_name]={c:{} for c in CAPS}
    for i in idxs:
        for cap,div in CAPS.items():
            try:
                P,vis = count(data[i], HIGH_PX//div)
                cache[ds_name][cap][str(i)] = [P, vis]
            except Exception as e:
                if done < 5: print(f"   skip {ds_name} idx={i} {cap}: {e}", flush=True)
        done+=1
        if done % 200 == 0: print(f"   {done} samples ({time.time()-t0:.0f}s)", flush=True)
    print(f">> {ds_name}: {len(idxs)} cached", flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(cache, open(A.out,"w"))
print(f"\nwrote {A.out}", flush=True)
