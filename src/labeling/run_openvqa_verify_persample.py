#!/usr/bin/env python3
"""
run_openvqa_verify_persample.py - PER-SAMPLE self/cross verification for verifier-guided SELECTION over
an sc8 open-ended ckpt. For each question and EACH of its 8 sampled answers, ask the verifier model
(image + question + proposed answer): "Is the proposed answer correct? Yes/No" -> normalized P(Yes).
Writes {idx, p_yes:[8 floats], preds:[8 answers]} so an offline selector can pick argmax-P(Yes).
Verifier can be the SAME cheap model (self-verify) or the 32B (cross-verify): just set --model_path/--tp.
Dedups identical answer strings per question (1 verify call each), maps the score back to every slot.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_openvqa_verify_persample.py \
    --model_path /data/dan/weights/Lingshu-7B --dataset slake_open \
    --sc8 ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b_sc8.jsonl \
    --out ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_lingshu7b_pv_self.jsonl --tp 1
"""
import argparse, json, os, glob, math, io
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
       "the proposed answer is correct. Respond with only 'Yes' or 'No'.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV = {"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--dataset", required=True, choices=["slake_open","vqa_rad_open","pathvqa_open","kvasir_open","radimagenet_open"])
ap.add_argument("--sc8", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV)); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88); ap.add_argument("--max_model_len", type=int, default=4096)
A = ap.parse_args(); MAXPX = HIGH_PX//CAP_DIV[A.cap]
os.makedirs(os.path.dirname(A.out), exist_ok=True)

# ---- proposed answers from sc8 ----
SC = {}
for l in open(A.sc8):
    if l.strip(): r = json.loads(l); SC[r["idx"]] = r["preds"]

# ---- images: idx -> (question, image[path or PIL]) ----
IMG = {}
if A.dataset == "slake_open":
    d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
    for x in d:
        if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
        ip = os.path.join(root, x["img_name"])
        if os.path.exists(ip) and x["qid"] in SC: IMG[x["qid"]] = (x["question"], ip)
elif A.dataset in ("kvasir_open", "radimagenet_open"):
    jp = "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json" if A.dataset=="kvasir_open" else "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json"
    d = json.load(open(jp))
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
        if str(a).strip().lower() in ("yes","no"): continue
        if int(i) not in SC: continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            IMG[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))

idxs = [i for i in SC if i in IMG]
print(f"{A.dataset}: {len(idxs)} questions with images (of {len(SC)} sc8)", flush=True)
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
    msgs = [{"role":"system","content":SYS},
            {"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},
                                       {"type":"text","text":body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

# resume
done = set()
if os.path.exists(A.out):
    for l in open(A.out):
        if l.strip(): done.add(json.loads(l)["idx"])
todo = [i for i in idxs if i not in done]
print(f"{len(todo)} to do ({len(done)} done)", flush=True)

llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image":1}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

def p_yes(o):
    lps = (o.outputs[0].logprobs or [{}])[0]
    py = max((math.exp(v.logprob) for t,v in lps.items() if t in YES), default=0.0)
    pn = max((math.exp(v.logprob) for t,v in lps.items() if t in NO), default=0.0)
    return py/(py+pn) if (py+pn) > 0 else 0.0

with open(A.out, "a") as fh:
    for c0 in range(0, len(todo), 64):
        chunk = todo[c0:c0+64]
        reqs, meta = [], []     # meta: (idx, uniq_answer)
        per_idx_uniq = {}
        for i in chunk:
            q, img = IMG[i]; preds = SC[i]
            uniq = {}
            for ans in preds:
                k = str(ans).strip().lower()
                if k not in uniq: uniq[k] = ans
            per_idx_uniq[i] = uniq
            for k, ans in uniq.items():
                reqs.append(build(q, img, ans)); meta.append((i, k))
        outs = llm.generate(reqs, sp)
        score = {}
        for (i,k), o in zip(meta, outs): score[(i,k)] = p_yes(o)
        for i in chunk:
            preds = SC[i]
            pys = [score[(i, str(a).strip().lower())] for a in preds]
            fh.write(json.dumps({"idx": i, "preds": preds, "p_yes": pys}) + "\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE verify_persample", flush=True)
