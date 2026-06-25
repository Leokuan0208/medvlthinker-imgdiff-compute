#!/usr/bin/env python3
"""
run_openvqa_select_listwise.py - LISTWISE verifier-guided selection over an sc8 ckpt. For each question,
show the verifier model (image + question + the UNIQUE candidate answers, numbered) and ask it to pick
the number of the correct candidate. This lets the verifier COMPARE candidates (vs independent pointwise
P(Yes)). Returns the picked answer per question. Verifier set by --model_path/--tp (e.g. Lingshu-32B).
Writes {idx, preds[8], cand[unique], pick_idx_in_cand, picked_answer}.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa_select_listwise.py \
    --model_path lingshu-medical-mllm/Lingshu-32B --dataset slake_open --cap fullres --tp 2 \
    --sc8 ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b_sc8.jsonl \
    --out ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b_sel_listwise32b.jsonl
"""
import argparse, json, os, glob, re, io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image
SYS = ("You are a careful medical expert. You are given a medical image, a question, and several candidate "
       "answers. Exactly one candidate best answers the question for this image. Reply with ONLY the number "
       "of the best candidate.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV = {"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--dataset", required=True, choices=["slake_open","vqa_rad_open","pathvqa_open","kvasir_open"])
ap.add_argument("--sc8", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", default="fullres", choices=list(CAP_DIV)); ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.90); ap.add_argument("--max_model_len", type=int, default=4096)
A = ap.parse_args(); MAXPX = HIGH_PX//CAP_DIV[A.cap]; os.makedirs(os.path.dirname(A.out), exist_ok=True)
SC = {}
for l in open(A.sc8):
    if l.strip(): r = json.loads(l); SC[r["idx"]] = r["preds"]
IMG = {}
if A.dataset == "slake_open":
    d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if os.path.exists(ip) and x["qid"] in SC: IMG[x["qid"]] = (x["question"], ip)
elif A.dataset == "kvasir_open":
    d = json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"))
    for r in d:
        if os.path.exists(r["img_path"]) and r["idx"] in SC: IMG[r["idx"]] = (r["question"], r["img_path"])
else:
    import pandas as pd
    base = "/data/dan/dataset/vqa_rad/data" if A.dataset == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
    for i, r in df.iterrows():
        q = r.get("question"); a = r.get("answer")
        if q is None and "conversations" in r:
            conv = r["conversations"]; q = conv[0]["value"].replace("<image>","").strip(); a = conv[1]["value"]
        if str(a).strip().lower() in ("yes","no") or int(i) not in SC: continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            IMG[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
idxs = [i for i in SC if i in IMG]
print(f"{A.dataset}: {len(idxs)} questions", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
def uniq_cands(preds):
    seen, out = set(), []
    for a in preds:
        k = str(a).strip().lower()
        if k and k not in seen: seen.add(k); out.append(str(a).strip())
    return out
def build(q, img, cands):
    lines = "\n".join(f"{j+1}. {c}" for j, c in enumerate(cands))
    body = f"Question: {q}\nCandidate answers:\n{lines}\nReply with only the number of the best candidate."
    msgs = [{"role":"system","content":SYS},
            {"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},
                                       {"type":"text","text":body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        if l.strip(): done.add(json.loads(l)["idx"])
todo = [i for i in idxs if i not in done]
print(f"{len(todo)} to do", flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image":1}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=8)
def parse(t, ncand):
    m = re.search(r"\d+", t)
    if not m: return 0
    v = int(m.group()) - 1
    return v if 0 <= v < ncand else 0
with open(A.out, "a") as fh:
    for c0 in range(0, len(todo), 64):
        chunk = todo[c0:c0+64]; reqs, meta = [], []
        for i in chunk:
            q, img = IMG[i]; cands = uniq_cands(SC[i]); meta.append((i, cands))
            reqs.append(build(q, img, cands))
        outs = llm.generate(reqs, sp)
        for (i, cands), o in zip(meta, outs):
            pk = parse(o.outputs[0].text, len(cands))
            fh.write(json.dumps({"idx": i, "preds": SC[i], "cand": cands, "pick": pk,
                                 "picked_answer": cands[pk], "raw": o.outputs[0].text.strip()}) + "\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE listwise", flush=True)
