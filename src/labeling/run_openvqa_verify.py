#!/usr/bin/env python3
"""
run_openvqa_verify.py - open-ended SELF-VERIFICATION (AutoMix / P(True)) for the gate hunt. The cheap
model judges its own temp-0 answer: given image+question+proposed answer, "Is the proposed answer
correct? Yes/No" -> normalized P(Yes). The one classic routing signal not yet tested in open-ended;
somewhat orthogonal to confidence, so a candidate to push a fusion gate above the confidence bar.
Reads proposed answers from a t0 ckpt (--pred_dir/--pred_tag). Writes p_yes_norm per sample.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa_verify.py \
    --model_path <L7> --dataset slake_open --pred_dir ckpts/openvqa/cheap_lingshu7b --pred_tag lingshu7b \
    --ckpt_dir ckpts/openvqa/cheap_lingshu7b --tag lingshu7b_verify --tp 1
"""
import argparse, re, json, os, glob, math
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image
SYS = "You are a careful medical exam grader. Given a question and a proposed answer, decide whether the proposed answer is correct. Respond with only 'Yes' or 'No'."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV = {"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True); ap.add_argument("--dataset", required=True, choices=["slake_open","vqa_rad_open"])
ap.add_argument("--pred_dir", required=True); ap.add_argument("--pred_tag", required=True)
ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV)); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88); ap.add_argument("--max_model_len", type=int, default=4096)
A = ap.parse_args(); os.makedirs(A.ckpt_dir, exist_ok=True); MAXPX = HIGH_PX//CAP_DIV[A.cap]
preds = {}
for l in open(os.path.join(A.pred_dir, f"ckpt_{A.dataset}_{A.pred_tag}.jsonl")):
    if l.strip(): r = json.loads(l); preds[r["idx"]] = r["modal_pred"]
# build items (idx, question, gold, image, proposed)
items = []
if A.dataset == "slake_open":
    d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if not os.path.exists(ip) or x["qid"] not in preds: continue
        items.append((x["qid"], x["question"], str(x["answer"]), ip, preds[x["qid"]]))
else:
    import pandas as pd, io
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/data/dan/dataset/vqa_rad/data/test-*.parquet"))], ignore_index=True)
    for i, r in df.iterrows():
        a = str(r.get("answer")).strip()
        if a.lower() in ("yes","no") or int(i) not in preds: continue
        img = r["image"]
        if not (isinstance(img, dict) and "bytes" in img): continue
        items.append((int(i), str(r.get("question")), a, Image.open(io.BytesIO(img["bytes"])).convert("RGB"), preds[int(i)]))
print(f"{A.dataset}: {len(items)} verify items", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
def tok_ids(words):
    ids = {}
    for w in words:
        for v in (w, " "+w):
            e = proc.tokenizer.encode(v, add_special_tokens=False)
            if len(e) == 1: ids[e[0]] = w
    return ids
YES, NO = tok_ids(["Yes","yes","YES"]), tok_ids(["No","no","NO"])
def build(q, img, proposed):
    body = f"Question: {q}\nProposed answer: {proposed}\nIs the proposed answer correct? Answer Yes or No."
    msgs = [{"role":"system","content":SYS}, {"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},{"type":"text","text":body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image":4}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)
ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl")
done = set()
if os.path.exists(ckpt):
    for l in open(ckpt):
        if l.strip(): done.add(json.loads(l)["idx"])
todo = [it for it in items if it[0] not in done]
print(f"  {len(todo)} to run -> {ckpt}", flush=True)
with open(ckpt, "a") as fh:
    for c0 in range(0, len(todo), 128):
        ch = todo[c0:c0+128]; outs = llm.generate([build(q,im,p) for (_,q,_,im,p) in ch], sp)
        for (idx,_,_,_,_), o in zip(ch, outs):
            lps = (o.outputs[0].logprobs or [{}])[0]
            py = max((math.exp(v.logprob) for t,v in lps.items() if t in YES), default=0.0)
            pn = max((math.exp(v.logprob) for t,v in lps.items() if t in NO), default=0.0)
            norm = py/(py+pn) if (py+pn) > 0 else None
            fh.write(json.dumps({"idx": idx, "p_yes_norm": (round(norm,5) if norm is not None else None)}) + "\n")
        fh.flush(); print(f"   [{min(c0+128,len(todo))}/{len(todo)}]", flush=True)
print("DONE verify", flush=True)
