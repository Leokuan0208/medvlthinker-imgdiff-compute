"""Definitive gate: per-case difficulty for MedVLThinker-3B (reasoning model).
Samples G traces per case, parses <answer>, difficulty=1-frac_correct.
Shardable across VMs: --num_shards 2 --shard 0  (and --shard 1 on the other VM).
"""
import csv, math, collections, re, argparse
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/data/models/MedVLThinker-3B-RL_m23k"
SUBSET_CSV = "subset.csv"
SYS_PROMPT = ("You will solve a problem/request. You should provide your thoughts within "
              "<think> </think> tags before providing the answer. Put the final answer within "
              "<answer> </answer> tags.")

def parse_yes_no(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S|re.I)
    seg = m.group(1) if m else text
    t = re.sub(r"^[^a-z]+","",seg.strip().lower())
    first = t.split()[0] if t.split() else ""
    if first in ("yes","yeah","yep","correct","true"): return "yes"
    if first in ("no","nope","false","incorrect"): return "no"
    if re.search(r"\bno\b",t) or "not " in t or "n't" in t: return "no"
    if re.search(r"\byes\b",t): return "yes"
    return ""

def load_model():
    kw=dict(torch_dtype=torch.bfloat16, device_map="auto")
    try:
        m=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, attn_implementation="flash_attention_2", **kw)
        print("[ok] flash_attention_2")
    except Exception as e:
        print(f"[warn] FA2 unavailable ({e}); using sdpa"); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_PATH, attn_implementation="sdpa", **kw)
    m.eval(); return m, AutoProcessor.from_pretrained(MODEL_PATH)

def sample(model, proc, image_path, question, G, mnt, temp, top_p):
    msgs=[{"role":"system","content":SYS_PROMPT},
          {"role":"user","content":[{"type":"image","image":image_path},
           {"type":"text","text":question.strip()+"\nAnswer yes or no."}]}]
    text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    inp=proc(text=[text], images=imgs, videos=None, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out=model.generate(**inp, do_sample=True, temperature=temp, top_p=top_p,
                           num_return_sequences=G, max_new_tokens=mnt)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)

def entropy(preds):
    c=collections.Counter(preds); n=sum(c.values())
    return -sum((v/n)*math.log(v/n+1e-12) for v in c.values())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--limit",type=int,default=0); ap.add_argument("--G",type=int,default=8)
    ap.add_argument("--max_new_tokens",type=int,default=512)
    ap.add_argument("--temperature",type=float,default=0.7); ap.add_argument("--top_p",type=float,default=0.95)
    ap.add_argument("--num_shards",type=int,default=1); ap.add_argument("--shard",type=int,default=0)
    a=ap.parse_args()
    out_csv = f"difficulty_shard{a.shard}of{a.num_shards}.csv" if a.num_shards>1 else "difficulty.csv"

    model,proc=load_model()
    rows=list(csv.DictReader(open(SUBSET_CSV)))
    if a.limit: rows=rows[:a.limit]
    rows=[r for i,r in enumerate(rows) if i % a.num_shards == a.shard]   # this VM's slice
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} rows -> {out_csv}")

    fields=["qid","image_id","image_path","question_type","modality","n_samples","frac_correct","difficulty","answer_entropy"]
    written=0
    with open(out_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,r in enumerate(rows):
            try:
                raws=sample(model,proc,r["image_path"],r["question"],a.G,a.max_new_tokens,a.temperature,a.top_p)
                preds=[parse_yes_no(x) for x in raws]; gold=str(r["gold"]).strip().lower()
                correct=sum(1 for p in preds if p==gold); frac=correct/len(preds)
                w.writerow({"qid":r["qid"],"image_id":r["image_id"],"image_path":r["image_path"],
                    "question_type":r["question_type"],"modality":r["modality"],"n_samples":len(preds),
                    "frac_correct":round(frac,4),"difficulty":round(1-frac,4),"answer_entropy":round(entropy(preds),4)})
                written+=1
            except Exception as e:
                print(f"  [skip] row {i} ({r.get('image_id')}): {e}")
            if (i+1)%25==0: f.flush(); print(f"  scored {i+1}/{len(rows)} (written={written})")
    print(f"wrote {out_csv}: {written} rows")
if __name__=="__main__": main()
