import argparse, re, json, random, os, torch
from collections import Counter
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts "
             "within <think> </think> tags before providing the answer.")
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, LOW_PX, MIN_PX = 1280*28*28, 256*28*28, 4*28*28   # full ceiling / visual-low / floor
CKPT_DIR = "gate_ckpts_7b"  

ap = argparse.ArgumentParser()
ap.add_argument("--axis", required=True, choices=["reasoning","visual"])
ap.add_argument("--n", type=int, default=100)
A = ap.parse_args()
os.makedirs(CKPT_DIR, exist_ok=True)

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]
print("SPLIT:", split, "| N:", len(data), "| counts:", dict(Counter(data["dataset_name"])), flush=True)

def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

def subset(*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]

def mx_by_type(t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt = ""
        if qt.lower()==t: out.append(i)
    return out

if A.axis=="reasoning":
    SUBSETS = {"MedXpert-ALL": subset("medxpert"),
               "MedXpert-Reasoning": mx_by_type("reasoning"),
               "MedXpert-Understanding": mx_by_type("understanding"),
               "PMC-VQA": subset("pmcvqa","pmc")}
else:
    SUBSETS = {"MedXpert-MM": subset("medxpert"), "PMC-VQA": subset("pmcvqa","pmc")}
for k,v in SUBSETS.items(): print(f"subset {k}: {len(v)}", flush=True)
assert all(len(v)>0 for v in SUBSETS.values()), "empty subset"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")

def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]
def pred(t):
    tail = t.split("</think>")[-1]
    m = LET.search(tail) or LET.search(t)
    return m.group(1) if m else "?"

def build(ex, sys_prompt, mp):
    cap = mp if mp else HIGH_PX            # every image bounded; mp set only on the visual-low arm
    opts = parse_opts(ex["options"]); assert opts, "no options parsed"
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    img=[{"type":"image","image":im,"max_pixels":cap,"min_pixels":MIN_PX}
         for im in (ex.get("images") or [])]
    msgs=[{"role":"system","content":sys_prompt},
          {"role":"user","content":img+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    return text, imgs

def run(name, idxs, mode, mp=None, tag=None):
    tag = tag or mode
    rng=random.Random(42); sel=idxs[:]; rng.shuffle(sel); sel=sel[:A.n]
    sys_prompt = SYS_NOTHINK if mode=="nothink" else SYS_THINK
    maxtok = 8 if mode=="nothink" else 512
    ckpt = os.path.join(CKPT_DIR, f"ckpt_{A.axis}_{name}_{tag}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        for line in open(ckpt):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line); done[r["idx"]]=int(r["ok"])
            except Exception: continue
    print(f"   [{tag}] checkpoint {ckpt} | resuming {sum(1 for i in sel if i in done)}/{len(sel)} cached", flush=True)
    correct=0
    with open(ckpt,"a") as fh:
        for j,i in enumerate(sel):
            if i in done:
                correct+=done[i]
                print(f"   [{j+1}/{len(sel)}] idx={i} CACHED ok={done[i]} acc={correct/(j+1):.3f}", flush=True)
                continue
            ex=data[i]; text,imgs=build(ex,sys_prompt,mp)
            inp=proc(text=[text],images=imgs,return_tensors="pt").to(model.device)
            with torch.no_grad():
                o=model.generate(**inp,max_new_tokens=maxtok,do_sample=False)
            gen=proc.batch_decode(o[:,inp.input_ids.shape[1]:],skip_special_tokens=True)[0]
            g,p=gold(ex),pred(gen); is_ok=int(g==p); correct+=is_ok
            del inp,o; torch.cuda.empty_cache()
            fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":is_ok})+"\n"); fh.flush()
            tail=gen.split("</think>")[-1].strip().replace("\n"," ")[:45]
            print(f"   [{j+1}/{len(sel)}] gold={g} pred={p} {'OK ' if is_ok else 'XX '} acc={correct/(j+1):.3f} | {tail!r}", flush=True)
    acc=correct/len(sel); print(f">> {name} [{tag}] n={len(sel)} ACC={acc:.3f}", flush=True); return acc

print("="*64,"\nAXIS:",A.axis,flush=True)
for name,idxs in SUBSETS.items():
    print(f"\n--- {name} ({len(idxs)}) ---",flush=True)
    if A.axis=="reasoning":
        t=run(name,idxs,"think",tag="think"); nt=run(name,idxs,"nothink",tag="nothink")
        print(f"### {name}: think={t:.3f} nothink={nt:.3f} DELTA={t-nt:+.3f}",flush=True)
    else:
        f=run(name,idxs,"think",mp=None,tag="visfull"); l=run(name,idxs,"think",mp=LOW_PX,tag="vislow")
        print(f"### {name}: vis_full={f:.3f} vis_low={l:.3f} DELTA={f-l:+.3f}",flush=True)
