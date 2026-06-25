#!/usr/bin/env python3
"""
run_ground_mscxr.py - phrase grounding on the REAL MS-CXR benchmark (PhysioNet; 1448 boxes / 1047 chest
X-rays / 8 pathology categories). For each (image, clinical phrase) the model outputs a box; 1 greedy + N
sampled. Gold from MS_CXR_Local_Alignment CSV (x,y,w,h in original-image coords). Same schema as
run_ground_slake.py so ground_analyze.py / the box-verifier reuse it. Baseline: MedGround-R1 (arXiv 2507.02994).
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_ground_mscxr.py --n 8 \
    --out ckpts/ground/mscxr_qwenvl7b.jsonl
"""
import argparse, json, os
import pandas as pd
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
ap.add_argument("--csv", default="/data/dan/dataset/ms_cxr/MS_CXR_v1.1.0.csv")
ap.add_argument("--imgdir", default="/data/dan/dataset/ms_cxr/images")
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--gpu_mem", type=float, default=0.85)
A = ap.parse_args(); os.makedirs(os.path.dirname(A.out), exist_ok=True)
df = pd.read_csv(A.csv)
items = []
for i, r in df.iterrows():
    ip = os.path.join(A.imgdir, f"{r['dicom_id']}.jpg")
    if not (os.path.exists(ip) and os.path.getsize(ip) > 1000): continue
    x, y, w, h = r["x"], r["y"], r["w"], r["h"]
    items.append({"idx": f"{r['dicom_id']}|{i}", "img": ip, "label": str(r["label_text"]),
                  "category": str(r["category_name"]), "gold": [float(x), float(y), float(x+w), float(y+h)],
                  "W": int(r["image_width"]), "H": int(r["image_height"]), "kind": "abn"})
print(f"{len(items)} MS-CXR grounding targets (of {len(df)} boxes; images present)", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
def build(path, label):
    msgs=[{"role":"user","content":[{"type":"image","image":path,"max_pixels":1280*28*28,"min_pixels":4*28*28},
        {"type":"text","text":f"Locate the finding '{label}' in this chest X-ray. Provide its bounding box as [x1, y1, x2, y2]."}]}]
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
sp_g=SamplingParams(temperature=0.0, max_tokens=64); sp_s=SamplingParams(temperature=A.temp, max_tokens=64, n=A.n)
with open(A.out,"a") as fh:
    for c0 in range(0,len(todo),64):
        chunk=todo[c0:c0+64]; reqs=[build(it["img"],it["label"]) for it in chunk]
        g=llm.generate(reqs, sp_g); s=llm.generate(reqs, sp_s)
        for it,go,so in zip(chunk,g,s):
            it["greedy_raw"]=go.outputs[0].text.strip(); it["sample_raws"]=[o.text.strip() for o in so.outputs]
            fh.write(json.dumps(it)+"\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE ground mscxr", flush=True)
