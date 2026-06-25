#!/usr/bin/env python3
"""
run_ground_slake.py - medical visual GROUNDING on SLAKE (zero-download: uses per-image detection.json
gold boxes). Lingshu-7B outputs a bounding box for a named organ/abnormality; we generate 1 greedy box +
N sampled boxes (temp). Structured/verifiable output (IoU) -> tests whether SELECTION/self-consistency
escapes the free-text luck floor. Writes {idx, label, gold[x1y1x2y2], W,H, greedy_raw, sample_raws[N]}.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_ground_slake.py --n 8 --out ckpts/ground/slake_lingshu7b.jsonl
"""
import argparse, json, os, glob
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--gpu_mem", type=float, default=0.85)
A = ap.parse_args(); os.makedirs(os.path.dirname(A.out), exist_ok=True)
ROOTI = "/data/dan/dataset/slake/imgs"
ABN = {"cancer","tumor","tumour","nodule","pneumonia","edema","effusion","atelectasis","pneumothorax",
       "consolidation","mass","lesion","fracture","opacity","infiltrat"}
items = []
for d in sorted(glob.glob(f"{ROOTI}/xmlab*/")):
    dj, sp = os.path.join(d,"detection.json"), os.path.join(d,"source.jpg")
    if not (os.path.exists(dj) and os.path.exists(sp)): continue
    try: det = json.load(open(dj))
    except: continue
    W,H = Image.open(sp).size
    for obj in det:
        for label, box in obj.items():
            x,y,w,h = box
            kind = "abn" if any(a in label.lower() for a in ABN) else "organ"
            items.append({"idx": f"{os.path.basename(d.rstrip('/'))}|{label}", "img": sp, "label": label,
                          "gold": [x,y,x+w,y+h], "W":W, "H":H, "kind":kind})
print(f"{len(items)} grounding targets ({sum(1 for it in items if it['kind']=='abn')} abnormalities)", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
def build(path, label):
    msgs=[{"role":"user","content":[{"type":"image","image":path,"max_pixels":1280*28*28,"min_pixels":4*28*28},
        {"type":"text","text":f"Locate the {label} in the image. Provide its bounding box as [x1, y1, x2, y2]."}]}]
    text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_=process_vision_info(msgs); req={"prompt":text}
    if imgs: req["multi_modal_data"]={"image":imgs}
    return req
done=set()
if os.path.exists(A.out):
    for l in open(A.out):
        if l.strip(): done.add(json.loads(l)["idx"])
todo=[it for it in items if it["idx"] not in done]
print(f"{len(todo)} to do", flush=True)
llm=LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
        max_model_len=4096, limit_mm_per_prompt={"image":1}, trust_remote_code=True)
sp_greedy=SamplingParams(temperature=0.0, max_tokens=64)
sp_samp=SamplingParams(temperature=A.temp, max_tokens=64, n=A.n)
with open(A.out,"a") as fh:
    for c0 in range(0,len(todo),64):
        chunk=todo[c0:c0+64]; reqs=[build(it["img"],it["label"]) for it in chunk]
        g=llm.generate(reqs, sp_greedy); s=llm.generate(reqs, sp_samp)
        for it,go,so in zip(chunk,g,s):
            it["greedy_raw"]=go.outputs[0].text.strip()
            it["sample_raws"]=[o.text.strip() for o in so.outputs]
            fh.write(json.dumps(it)+"\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE ground", flush=True)
