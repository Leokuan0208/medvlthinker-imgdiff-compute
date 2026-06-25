#!/usr/bin/env python3
"""
run_openvqa_synth.py - candidate-CONDITIONED generation: the strong model answers the question using
a cheap model's diverse candidate answers as NOISY HINTS (not authority), with the image as primary
evidence. Tests whether priming with the cheap model's sample diversity (whose oracle exceeds the
strong model's single pass) lifts the strong model ABOVE its own free generation. Free-form output.
Writes {idx, question, gold, modal_pred=synth_answer, cand}.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa_synth.py \
    --model_path lingshu-medical-mllm/Lingshu-32B --dataset slake_open --cap fullres --tp 2 \
    --sc8 ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b_sc8.jsonl \
    --out ckpts/openvqa/strong_lingshu/ckpt_slake_open_lingshu32b_synth.jsonl
"""
import argparse, json, os, glob, io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image
SYS = ("You are an expert medical AI answering a visual question. A smaller, less reliable model proposed "
       "several candidate answers; treat them only as hints. Rely primarily on the image. Give your single "
       "best, concise final answer.")
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
    if l.strip(): r = json.loads(l); SC[r["idx"]] = r
IMG = {}   # idx -> (question, gold, image)
if A.dataset == "slake_open":
    d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if os.path.exists(ip) and x["qid"] in SC: IMG[x["qid"]] = (x["question"], str(x["answer"]), ip)
elif A.dataset == "kvasir_open":
    d = json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"))
    for r in d:
        if os.path.exists(r["img_path"]) and r["idx"] in SC: IMG[r["idx"]] = (r["question"], r["answer"], r["img_path"])
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
            IMG[int(i)] = (str(q), str(a).strip(), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
idxs = [i for i in SC if i in IMG]
print(f"{A.dataset}: {len(idxs)} questions", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path)
def uniq(preds):
    seen, out = set(), []
    for a in preds:
        k = str(a).strip().lower()
        if k and k not in seen: seen.add(k); out.append(str(a).strip())
    return out
def build(q, img, cands):
    lines = "; ".join(cands)
    body = f"Question: {q}\nCandidate hints (may be wrong): {lines}\nYour best final answer:"
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
sp = SamplingParams(temperature=0.0, max_tokens=48)
with open(A.out, "a") as fh:
    for c0 in range(0, len(todo), 64):
        chunk = todo[c0:c0+64]; reqs, meta = [], []
        for i in chunk:
            q, gold, img = IMG[i]; cands = uniq(SC[i]["preds"]); meta.append((i, q, gold, cands))
            reqs.append(build(q, img, cands))
        outs = llm.generate(reqs, sp)
        for (i, q, gold, cands), o in zip(meta, outs):
            ans = o.outputs[0].text.strip().split("\n")[0].strip()
            fh.write(json.dumps({"idx": i, "question": q, "gold": gold, "modal_pred": ans, "cand": cands}) + "\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE synth", flush=True)
