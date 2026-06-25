#!/usr/bin/env python3
"""
run_openvqa_fewshot.py - few-shot in-context exemplars for open-ended med-VQA, to align ANSWER STYLE
(targets the 'plausible-but-wrong-style' failure mode). Prepends k text-only train Q->A pairs before the
test question; the test IMAGE is attached as usual. Greedy (temp 0). Writes {idx, question, gold, modal_pred}.
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0,1 python3 src/labeling/run_openvqa_fewshot.py \
    --model_path lingshu-medical-mllm/Lingshu-7B --dataset pathvqa_open --k 5 --tp 2 \
    --out ckpts/openvqa/cheap_lingshu7b/ckpt_pathvqa_open_lingshu7b_fs5.jsonl
"""
import argparse, json, os, glob, io, random
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image
SYS = "You are an expert medical AI. Answer the question about the image with a single concise answer."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV={"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--dataset", required=True, choices=["slake_open","pathvqa_open","vqa_rad_open"])
ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=5)
ap.add_argument("--cap", default="fullres", choices=list(CAP_DIV)); ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.90); ap.add_argument("--max_model_len", type=int, default=4096)
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args(); MAXPX=HIGH_PX//CAP_DIV[A.cap]; os.makedirs(os.path.dirname(A.out), exist_ok=True)
rng=random.Random(A.seed)

def load_split_qa(ds, split):
    """text-only Q->A exemplar pool from a train split."""
    out=[]
    if ds=="slake_open":
        d=json.load(open(f"/data/dan/dataset/slake/{split}.json"))
        for x in d:
            if x.get("answer_type")=="OPEN" and x.get("q_lang")=="en":
                out.append((x["question"], str(x["answer"])))
    else:
        import pandas as pd
        base="/data/dan/dataset/path_vqa/data" if ds=="pathvqa_open" else "/data/dan/dataset/vqa_rad/data"
        for f in sorted(glob.glob(os.path.join(base, f"{split}-*.parquet"))):
            df=pd.read_parquet(f, columns=None)
            for _,r in df.iterrows():
                q=r.get("question"); a=r.get("answer")
                if q is None and "conversations" in r:
                    conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
                a=str(a).strip()
                if a.lower() in ("yes","no"): continue
                out.append((str(q), a))
    return out
exemplar_pool = load_split_qa(A.dataset, "train")
print(f"exemplar pool: {len(exemplar_pool)} train Q->A", flush=True)

# test items (idx, question, gold, image) — same loaders as run_openvqa
items=[]
if A.dataset=="slake_open":
    d=json.load(open("/data/dan/dataset/slake/test.json")); root="/data/dan/dataset/slake/imgs"
    for x in d:
        if x.get("answer_type")!="OPEN" or x.get("q_lang")!="en": continue
        ip=os.path.join(root,x["img_name"])
        if os.path.exists(ip): items.append((x["qid"], x["question"], str(x["answer"]), ip))
else:
    import pandas as pd
    base="/data/dan/dataset/path_vqa/data" if A.dataset=="pathvqa_open" else "/data/dan/dataset/vqa_rad/data"
    df=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base,"test-*.parquet")))], ignore_index=True)
    for i,r in df.iterrows():
        q=r.get("question"); a=r.get("answer")
        if q is None and "conversations" in r:
            conv=r["conversations"]; q=conv[0]["value"].replace("<image>","").strip(); a=conv[1]["value"]
        a=str(a).strip()
        if a.lower() in ("yes","no"): continue
        img=r["image"]
        if isinstance(img,dict) and "bytes" in img:
            items.append((int(i), str(q), a, Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
print(f"{A.dataset}: {len(items)} test items, k={A.k}", flush=True)
proc=AutoProcessor.from_pretrained(A.model_path)
def fewshot_text(q):
    ex=rng.sample(exemplar_pool, min(A.k, len(exemplar_pool)))
    lines="\n".join(f"Q: {eq}\nA: {ea}" for eq,ea in ex)
    return f"Here are examples of the expected answer style:\n{lines}\n\nNow answer this question about the image.\nQ: {q}\nA:"
def build(q,img):
    msgs=[{"role":"system","content":SYS},
          {"role":"user","content":[{"type":"image","image":img,"max_pixels":MAXPX,"min_pixels":MIN_PX},
                                     {"type":"text","text":fewshot_text(q)}]}]
    text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_=process_vision_info(msgs); req={"prompt":text}
    if imgs: req["multi_modal_data"]={"image":imgs}
    return req
done=set()
if os.path.exists(A.out):
    for l in open(A.out):
        if l.strip(): done.add(json.loads(l)["idx"])
todo=[it for it in items if it[0] not in done]
print(f"{len(todo)} to do", flush=True)
llm=LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
        max_model_len=A.max_model_len, limit_mm_per_prompt={"image":1}, trust_remote_code=True)
sp=SamplingParams(temperature=0.0, max_tokens=48)
with open(A.out,"a") as fh:
    for c0 in range(0,len(todo),64):
        chunk=todo[c0:c0+64]; outs=llm.generate([build(q,img) for _,q,_,img in chunk], sp)
        for (i,q,g,_),o in zip(chunk,outs):
            ans=o.outputs[0].text.strip().split("\n")[0].strip()
            fh.write(json.dumps({"idx":i,"question":q,"gold":g,"modal_pred":ans})+"\n")
        fh.flush(); print(f"  [{min(c0+64,len(todo))}/{len(todo)}]", flush=True)
print("DONE fewshot", flush=True)
