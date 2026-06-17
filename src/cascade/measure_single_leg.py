#!/usr/bin/env python3
"""
rt_measure.py - real-time SINGLE-QUERY (batch-1) latency / power / VRAM for ONE model leg.
Simulates a user submitting one image+text pair at a time (NO batching), via HF transformers
generate() -- HF, not vLLM, because vLLM pre-allocates the KV pool and hides true VRAM. Measures
per-query wall-clock, GPU energy (NVML power integrated over the query), peak VRAM. Run once
--model 7b and once --model 32b; rt_report.py joins them through the gate. Writes rt_<model>.jsonl.
Pin ONE GPU with CUDA_VISIBLE_DEVICES.
"""
import argparse, json, os, re, time, random, threading
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import pynvml

WEIGHTS = {"7b":"/data/dan/weights/MedVLThinker-7B-RL_m23k",
           "32b":"/data/dan/weights/MedVLThinker-32B-RL_m23k"}
ROOT = "/data/dan/dataset/MedVLThinker-Eval"
SYS  = {"nothink":"Answer with only the correct option letter (e.g. 'A'). Do not explain.",
        "think":"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer."}
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["7b","32b"], required=True)
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA"])
ap.add_argument("--n", type=int, default=200, help="queries per dataset (batch-1, sequential)")
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--out", default="")
A = ap.parse_args()
MODE   = "think" if A.model=="32b" else "nothink"
MAXTOK = 2048 if A.model=="32b" else 16
OUT    = A.out or f"rt_{A.model}.jsonl"

class PowerSampler(threading.Thread):
    def __init__(self, idx=0, interval=0.025):
        super().__init__(daemon=True)
        self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        self.interval = interval; self.samples = []; self.flag = True
    def run(self):
        while self.flag:
            self.samples.append((time.time(), pynvml.nvmlDeviceGetPowerUsage(self.h)/1000.0))
            time.sleep(self.interval)
    def stop(self): self.flag = False
    def window(self, t0, t1):
        s = [(t,w) for (t,w) in self.samples if t0 <= t <= t1]
        if len(s) < 2:
            w = pynvml.nvmlDeviceGetPowerUsage(self.h)/1000.0
            return w, w*(t1-t0)
        ts = [x[0] for x in s]; ws = [x[1] for x in s]
        energy = sum((ws[k]+ws[k+1])/2.0*(ts[k+1]-ts[k]) for k in range(len(s)-1))  # trapezoid J
        return float(sum(ws)/len(ws)), float(energy)         # mean W, energy J

def subset(data,*keys):
    return [i for i,n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(data,t):
    out=[]
    for i,n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc=data[i].get("misc")
        try: qt=json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt=""
        if qt.lower()==t: out.append(i)
    return out
def parse_opts(s):
    if isinstance(s,dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

pynvml.nvmlInit()
NVML_IDX = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])  # read the GPU torch actually uses
ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
DATASET_IDX = {
    "MedXpert-Reasoning": lambda: mx_by_type(data,"reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type(data,"understanding"),
    "PMC-VQA": lambda: subset(data,"pmcvqa","pmc"), "SLAKE": lambda: subset(data,"slake"),
    "VQA-RAD": lambda: subset(data,"vqarad","vqa_rad","rad"), "PathVQA": lambda: subset(data,"pathvqa","path"),
    "MMMU": lambda: subset(data,"mmmu"),
}
def fixed_slice(idxs):
    rng=random.Random(42); s=idxs[:]; rng.shuffle(s); return s[:A.n]

proc = AutoProcessor.from_pretrained(WEIGHTS[A.model])
LET  = re.compile(r"\b([A-J])\b")
def pred_from_text(t):
    tail=t.split("</think>")[-1]; m=LET.search(tail) or LET.search(t)
    return (m.group(1),1) if m else ("?",0)
def build_inputs(ex):
    opts=parse_opts(ex["options"]); q=ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    im=[{"type":"image","image":x,"max_pixels":HIGH_PX,"min_pixels":MIN_PX} for x in (ex.get("images") or [])]
    msgs=[{"role":"system","content":SYS[MODE]},{"role":"user","content":im+[{"type":"text","text":q}]}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    imgs,_=process_vision_info(msgs)
    inp=proc(text=[text], images=imgs if imgs else None, return_tensors="pt")
    return {k:(v.to("cuda") if hasattr(v,"to") else v) for k,v in inp.items()}

def load_model():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except Exception:
        from transformers import AutoModelForImageTextToText as Cls
    return Cls.from_pretrained(WEIGHTS[A.model], torch_dtype=torch.bfloat16,
                               device_map="cuda", trust_remote_code=True).eval()

model = load_model(); torch.cuda.synchronize()
resident = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(NVML_IDX)).used/1e6
print(f"loaded {A.model} (mode={MODE}, max_tok={MAXTOK}); resident VRAM ~{resident:.0f} MB", flush=True)

def gen_one(inputs):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0=time.time()
    with torch.no_grad():
        out=model.generate(**inputs, max_new_tokens=MAXTOK, do_sample=False)
    torch.cuda.synchronize(); t1=time.time()
    new=out[0][inputs["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0]), t0, t1, torch.cuda.max_memory_allocated()/1e6

query_idx=[]
for nm in A.datasets:
    query_idx += [(nm,i) for i in fixed_slice(DATASET_IDX[nm]())]
print(f"{len(query_idx)} queries (batch-1). warming up {A.warmup}...", flush=True)
for _,i in query_idx[:A.warmup]:
    gen_one(build_inputs(data[i]))

sampler=PowerSampler(idx=NVML_IDX); sampler.start()
t_run=time.time(); nc=0
with open(OUT,"w") as fh:
    for k,(nm,i) in enumerate(query_idx):
        txt,ntok,t0,t1,peak = gen_one(build_inputs(data[i]))
        mw,ej = sampler.window(t0,t1)
        p,pk = pred_from_text(txt); g=gold(data[i]); ok=int(g==p); nc+=ok
        fh.write(json.dumps({"idx":i,"dataset":nm,"ok":ok,"pred":p,"gen_tokens":ntok,
            "latency_s":round(t1-t0,4),"energy_j":round(ej,2),"mean_power_w":round(mw,1),
            "peak_vram_mb":round(peak,0),"resident_vram_mb":round(resident,0)})+"\n"); fh.flush()
        if (k+1)%25==0:
            print(f"   [{k+1}/{len(query_idx)}] acc={nc/(k+1):.3f} last_lat={t1-t0:.2f}s "
                  f"last_pow={mw:.0f}W ({(time.time()-t_run)/60:.1f}min)", flush=True)
sampler.stop()
print(f"\nDONE {A.model}: {len(query_idx)} queries, acc={nc/len(query_idx):.3f} -> {OUT}", flush=True)
