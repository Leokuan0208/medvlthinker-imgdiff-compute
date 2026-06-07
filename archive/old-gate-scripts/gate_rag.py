import argparse, re, json, random, os, torch
from collections import Counter
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
RETR  = "/data/dan/retrieval_kb/retrieved_medxpert_n100.jsonl"
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts "
             "within <think> </think> tags before providing the answer.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CKPT_DIR = "gate_ckpts_7b"  

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--max_snip_chars", type=int, default=500)   # truncate each snippet to bound prompt length
A = ap.parse_args()
os.makedirs(CKPT_DIR, exist_ok=True)

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]

# retrieved snippets, keyed by dataset index
retr = {}
for line in open(RETR):
    r = json.loads(line); retr[r["idx"]] = r["snippets"]
print(f"loaded {len(retr)} retrieved rows from {RETR}", flush=True)

# identical seed-42 MedXpert slice
mx = [i for i,n in enumerate(data["dataset_name"]) if "medxpert" in n.lower()]
rng = random.Random(42); sel = mx[:]; rng.shuffle(sel); sel = sel[:A.n]
assert all(i in retr for i in sel), "slice/retrieval mismatch — regenerate retrieve.py with same --n"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")

def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]
def pred(t):
    tail = t.split("</think>")[-1]; m = LET.search(tail) or LET.search(t)
    return m.group(1) if m else "?"

def build(ex, idx, use_rag):
    opts = parse_opts(ex["options"]); assert opts
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    if use_rag:
        snips = [s[:A.max_snip_chars] for s in retr.get(idx, [])]
        ctx = "\n\n".join(f"[{n+1}] {s}" for n,s in enumerate(snips))
        q = ("Relevant medical references:\n" + ctx +
             "\n\nUsing the references above where helpful, answer the question:\n" + q)
    img=[{"type":"image","image":im,"max_pixels":HIGH_PX,"min_pixels":MIN_PX} for im in (ex.get("images") or [])]
    msgs=[{"role":"system","content":SYS_THINK},{"role":"user","content":img+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs); return text, imgs

def run(tag, use_rag):
    ckpt=os.path.join(CKPT_DIR,f"ckpt_rag_MedXpert_{tag}.jsonl"); done={}
    if os.path.exists(ckpt):
        for ln in open(ckpt):
            ln=ln.strip()
            if ln:
                try: r=json.loads(ln); done[r["idx"]]=int(r["ok"])
                except: pass
    print(f"--- {tag} (use_rag={use_rag}) | resuming {sum(1 for i in sel if i in done)}/{len(sel)} ---", flush=True)
    correct=0
    with open(ckpt,"a") as fh:
        for j,i in enumerate(sel):
            if i in done:
                correct+=done[i]; print(f"   [{j+1}/{len(sel)}] idx={i} CACHED ok={done[i]} acc={correct/(j+1):.3f}", flush=True); continue
            ex=data[i]; text,imgs=build(ex,i,use_rag)
            inp=proc(text=[text],images=imgs,return_tensors="pt").to(model.device)
            with torch.no_grad():
                o=model.generate(**inp,max_new_tokens=512,do_sample=False)
            gen=proc.batch_decode(o[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]
            g,p=gold(ex),pred(gen); ok=int(g==p); correct+=ok
            del inp,o; torch.cuda.empty_cache()
            fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":ok})+"\n"); fh.flush()
            tl=gen.split("</think>")[-1].strip().replace("\n"," ")[:45]
            print(f"   [{j+1}/{len(sel)}] gold={g} pred={p} {'OK ' if ok else 'XX '} acc={correct/(j+1):.3f} | {tl!r}", flush=True)
    acc=correct/len(sel); print(f">> {tag} n={len(sel)} ACC={acc:.3f}", flush=True); return acc

print("="*64,"\nRETRIEVAL AXIS GATE (both arms use think)",flush=True)
norag=run("norag", False)
rag  =run("rag",   True)
print(f"\n### MedXpert RAG-axis: rag={rag:.3f} norag={norag:.3f} DELTA={rag-norag:+.3f}",flush=True)
