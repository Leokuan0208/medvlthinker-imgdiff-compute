#!/usr/bin/env python3
"""
measure_config.py - real-time (batch-1, HF) latency/energy for MULTIPLE (arm x resolution-cap)
configs of ONE model, loaded once. Extends measure_single_leg.py to the configs the Adaptive-Compute
Cascade routes over (e.g. 32B nothink@cap320, nothink@fullres, think@fullres, think@cap320). Writes
per-query {config, dataset, idx, prefill_tok, gen_tok, latency_s, energy_j} so a latency model
lat = f(prefill_tok, gen_tok) can be fit per model. Pin ONE GPU via CUDA_VISIBLE_DEVICES.
"""
import argparse, json, os, re, time, random, threading
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import pynvml

WEIGHTS = {"7b": "/data/dan/weights/MedVLThinker-7B-RL_m23k", "32b": "/data/dan/weights/MedVLThinker-32B-RL_m23k"}
ROOT = "/data/dan/dataset/MedVLThinker-Eval"
SYS = {"nothink": "Answer with only the correct option letter (e.g. 'A'). Do not explain.",
       "think": "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer."}
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["7b", "32b"], required=True)
ap.add_argument("--configs", required=True, help="comma list of arm:cap e.g. nothink:cap320,nothink:fullres,think:fullres")
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU", "MedXpert-Reasoning"])
ap.add_argument("--n", type=int, default=20, help="queries per dataset (batch-1)")
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--out", default="")
A = ap.parse_args()
CONFIGS = [tuple(c.split(":")) for c in A.configs.split(",")]
OUT = A.out or f"results/cascade_methods/artifacts/latency_{A.model}.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

class PowerSampler(threading.Thread):
    def __init__(self, idx=0, interval=0.025):
        super().__init__(daemon=True)
        self.h = pynvml.nvmlDeviceGetHandleByIndex(idx); self.interval = interval; self.samples = []; self.flag = True
    def run(self):
        while self.flag:
            self.samples.append((time.time(), pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0)); time.sleep(self.interval)
    def stop(self): self.flag = False
    def energy(self, t0, t1):
        s = [(t, w) for (t, w) in self.samples if t0 <= t <= t1]
        if len(s) < 2: return pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0 * (t1 - t0)
        ts = [x[0] for x in s]; ws = [x[1] for x in s]
        return float(sum((ws[k] + ws[k+1]) / 2.0 * (ts[k+1] - ts[k]) for k in range(len(s) - 1)))

def subset(data, *keys):
    return [i for i, n in enumerate(data["dataset_name"]) if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
def mx_by_type(data, t):
    out = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: out.append(i)
    return out
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

pynvml.nvmlInit()
NVML_IDX = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
DSI = {"MedXpert-Reasoning": lambda: mx_by_type(data, "reasoning"), "MedXpert-Understanding": lambda: mx_by_type(data, "understanding"),
       "PMC-VQA": lambda: subset(data, "pmcvqa", "pmc"), "SLAKE": lambda: subset(data, "slake"),
       "VQA-RAD": lambda: subset(data, "vqarad", "vqa_rad", "rad"), "PathVQA": lambda: subset(data, "pathvqa", "path"),
       "MMMU": lambda: subset(data, "mmmu")}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); return s[:A.n]

proc = AutoProcessor.from_pretrained(WEIGHTS[A.model])
def build_inputs(ex, arm, cap):
    opts = parse_opts(ex["options"]); q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    mp = HIGH_PX // CAP_DIV[cap]
    im = [{"type": "image", "image": x, "max_pixels": mp, "min_pixels": MIN_PX} for x in (ex.get("images") or [])]
    msgs = [{"role": "system", "content": SYS[arm]}, {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    inp = proc(text=[text], images=imgs if imgs else None, return_tensors="pt")
    return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}

def load_model():
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except Exception:
        from transformers import AutoModelForImageTextToText as Cls
    return Cls.from_pretrained(WEIGHTS[A.model], torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()

model = load_model(); torch.cuda.synchronize()
print(f"loaded {A.model}; configs={CONFIGS}", flush=True)
def gen_one(inputs, maxtok):
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=maxtok, do_sample=False)
    torch.cuda.synchronize(); t1 = time.time()
    new = out[0][inputs["input_ids"].shape[1]:]
    return int(new.shape[0]), t0, t1

query_idx = []
for nm in A.datasets:
    if nm in DSI: query_idx += [(nm, i) for i in fixed_slice(DSI[nm]())]
sampler = PowerSampler(idx=NVML_IDX); sampler.start()
with open(OUT, "w") as fh:
    for arm, cap in CONFIGS:
        maxtok = 2048 if arm == "think" else 16
        # warmup
        for _, i in query_idx[:A.warmup]:
            try: gen_one(build_inputs(data[i], arm, cap), maxtok)
            except Exception: pass
        n = 0; t_cfg = time.time()
        for nm, i in query_idx:
            try:
                inp = build_inputs(data[i], arm, cap); P = int(inp["input_ids"].shape[1])
                g, t0, t1 = gen_one(inp, maxtok)
                fh.write(json.dumps({"config": f"{arm}@{cap}", "dataset": nm, "idx": i, "prefill_tok": P,
                                     "gen_tok": g, "latency_s": round(t1 - t0, 4), "energy_j": round(sampler.energy(t0, t1), 1)}) + "\n")
                fh.flush(); n += 1
            except Exception as e:
                print(f"  skip {nm}/{i} {arm}@{cap}: {e}", flush=True)
        print(f">> {arm}@{cap}: {n} queries in {(time.time()-t_cfg)/60:.1f} min", flush=True)
sampler.stop()
print(f"DONE -> {OUT}", flush=True)
