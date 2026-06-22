#!/usr/bin/env python3
"""
rt_cascade.py - LIVE co-resident cascade, full eval suite, real-time single-query routing.

ONE process, batch-1. 7B (cheap, no_think, at --cap) on GPU0 and 32B (strong, think, full-res) on
GPU1 are BOTH resident. Per query the 7B answers, the frozen margin gate decides live, and ONLY on
escalation does the 32B run -- final answer comes from whichever leg. Captures BOTH GPUs' power
(NVML), true VRAM (HF allocates on demand; vLLM pre-grabs the pool and hides it), per-query
latency, live routed accuracy. Checkpoint-resumes from --out. Both GPUs must be visible; run when
the GPUs are otherwise idle (NVML 'used' reports the whole board).

Operating point (Week 5 decision): --cap cap320, deployed gate (tau=0.426). Grid: cap320 holds
exact parity (0.572 == always-32B) at 74% of always-32B compute.
"""
import argparse, json, os, re, time, random, threading, pickle
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import pynvml

W7  = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
W32 = "/data/dan/weights/MedVLThinker-32B-RL_m23k"
ROOT = "/data/dan/dataset/MedVLThinker-Eval"
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
SYS_THINK   = "You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
DIVS = {"fullres":1, "cap640":2, "cap320":4, "cap160":8, "cap80":16}
ALL6 = ["PMC-VQA","SLAKE","VQA-RAD","PathVQA","MMMU","MedXpert-Reasoning","MedXpert-Understanding"]

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", default=ALL6, help="default = full six-benchmark suite")
ap.add_argument("--n", type=int, default=0, help="queries per dataset; 0 = ALL (full eval)")
ap.add_argument("--cap", choices=list(DIVS), default="cap320", help="resolution cap for the 7B cheap leg")
ap.add_argument("--gate", default="ckpts/router_margin.pkl", help="gate+tau artifact")
ap.add_argument("--gpu7", type=int, default=0); ap.add_argument("--gpu32", type=int, default=1)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--repo", default=os.path.expanduser("~/medvlthinker-imgdiff-compute"))
ap.add_argument("--out", default="ckpts/rt_cascade_cap320.jsonl")
A = ap.parse_args()
DEV7, DEV32 = f"cuda:{A.gpu7}", f"cuda:{A.gpu32}"
PX7 = HIGH_PX // DIVS[A.cap]                      # cheap-leg pixel budget; 32B always uses HIGH_PX

class DualPowerSampler(threading.Thread):
    """Polls BOTH GPUs' board power in the background; integrate over a query window for energy."""
    def __init__(self, idxs, interval=0.025):
        super().__init__(daemon=True)
        self.handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in idxs}
        self.interval = interval; self.samples = []; self.flag = True
    def run(self):
        while self.flag:
            self.samples.append((time.time(), {i: pynvml.nvmlDeviceGetPowerUsage(h)/1000.0
                                               for i, h in self.handles.items()}))
            time.sleep(self.interval)
    def stop(self): self.flag = False
    def window(self, t0, t1):
        s = [(t, d) for (t, d) in self.samples if t0 <= t <= t1]
        if len(s) < 2:
            d = {i: pynvml.nvmlDeviceGetPowerUsage(h)/1000.0 for i, h in self.handles.items()}
            return d, {i: d[i]*(t1-t0) for i in d}
        mean = {}; energy = {}; ts = [t for (t, _) in s]
        for i in self.handles:
            ws = [d[i] for (_, d) in s]
            energy[i] = sum((ws[k]+ws[k+1])/2.0*(ts[k+1]-ts[k]) for k in range(len(s)-1))  # trapezoid J
            mean[i] = sum(ws)/len(ws)
        return mean, energy

def subset(data, *keys):
    return [i for i, n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-","").replace("_","") for k in keys)]
def mx_by_type(data, t):
    out = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type","") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: out.append(i)
    return out
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

pynvml.nvmlInit()
ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
DATASET_IDX = {
    "MedXpert-Reasoning": lambda: mx_by_type(data,"reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type(data,"understanding"),
    "PMC-VQA": lambda: subset(data,"pmcvqa","pmc"), "SLAKE": lambda: subset(data,"slake"),
    "VQA-RAD": lambda: subset(data,"vqarad","vqa_rad","rad"), "PathVQA": lambda: subset(data,"pathvqa","path"),
    "MMMU": lambda: subset(data,"mmmu"),
}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); return s if A.n <= 0 else s[:A.n]

def load(weights, gpu):
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    except Exception:
        from transformers import AutoModelForImageTextToText as Cls
    return Cls.from_pretrained(weights, torch_dtype=torch.bfloat16,
                               device_map={"": gpu}, trust_remote_code=True).eval()

proc = AutoProcessor.from_pretrained(W7)
print(f"operating point: cap={A.cap} (7B max_pixels=HIGH_PX//{DIVS[A.cap]}), 32B=full-res", flush=True)
print(f"loading 7B on {DEV7} ...", flush=True);  m7  = load(W7,  A.gpu7)
print(f"loading 32B on {DEV32} ...", flush=True); m32 = load(W32, A.gpu32)
for _m in (m7, m32):
    _m.generation_config.do_sample = False
    _m.generation_config.temperature = None
    _m.generation_config.top_p = None
    _m.generation_config.top_k = None
torch.cuda.synchronize(A.gpu7); torch.cuda.synchronize(A.gpu32)
def vram(i): return pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).used/1e6
VR7, VR32 = vram(A.gpu7), vram(A.gpu32)
print(f"resident: GPU{A.gpu7}(7B)={VR7:.0f}MB  GPU{A.gpu32}(32B)={VR32:.0f}MB  total={VR7+VR32:.0f}MB", flush=True)

gp = A.gate if os.path.isabs(A.gate) else os.path.join(A.repo, A.gate)
R = pickle.load(open(gp, "rb")); GATE, TAU = R["gate"], R["tau"]
print(f"gate: {gp}  (tau={TAU:.3f})", flush=True)

LET = re.compile(r"\b([A-J])\b")
def build(ex, sysmsg, max_pixels):
    opts = parse_opts(ex["options"]); q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    im = [{"type":"image","image":x,"max_pixels":max_pixels,"min_pixels":MIN_PX} for x in (ex.get("images") or [])]
    msgs = [{"role":"system","content":sysmsg},{"role":"user","content":im+[{"type":"text","text":q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    return proc(text=[text], images=imgs if imgs else None, return_tensors="pt"), list(opts.keys())
_LID = {}
def letter_id(L):
    if L not in _LID:
        ids = proc.tokenizer.encode(L, add_special_tokens=False); _LID[L] = ids[0] if ids else None
    return _LID[L]

def run_7b(ex):
    inp, opts = build(ex, SYS_NOTHINK, PX7); inp = {k:(v.to(DEV7) if hasattr(v,"to") else v) for k,v in inp.items()}
    torch.cuda.synchronize(A.gpu7); t0 = time.time()
    with torch.no_grad():
        out = m7.generate(**inp, max_new_tokens=16, do_sample=False,
                          output_scores=True, return_dict_in_generate=True)
    torch.cuda.synchronize(A.gpu7); t1 = time.time()
    seq = out.sequences[0][inp["input_ids"].shape[1]:]
    txt = proc.tokenizer.decode(seq, skip_special_tokens=True)
    logp = torch.log_softmax(out.scores[0][0].float(), dim=-1)
    lp = {L: logp[letter_id(L)].item() for L in opts if letter_id(L) is not None}
    v = sorted(lp.values(), reverse=True); mg = (v[0]-v[1]) if len(v) >= 2 else 0.0
    m = LET.search(txt); return (m.group(1) if m else "?"), mg, t1-t0, int(seq.shape[0])

def run_32b(ex):
    inp, _ = build(ex, SYS_THINK, HIGH_PX); inp = {k:(v.to(DEV32) if hasattr(v,"to") else v) for k,v in inp.items()}
    torch.cuda.synchronize(A.gpu32); t0 = time.time()
    with torch.no_grad():
        out = m32.generate(**inp, max_new_tokens=2048, do_sample=False)
    torch.cuda.synchronize(A.gpu32); t1 = time.time()
    seq = out[0][inp["input_ids"].shape[1]:]
    txt = proc.tokenizer.decode(seq, skip_special_tokens=True)
    tail = txt.split("</think>")[-1]; m = LET.search(tail) or LET.search(txt)
    return (m.group(1) if m else "?"), t1-t0, int(seq.shape[0])

queries = []
for nm in A.datasets: queries += [(nm, i) for i in fixed_slice(DATASET_IDX[nm]())]

done = set(); n_done = nc = esc_n = 0
if os.path.exists(A.out):
    for l in open(A.out):
        if l.strip():
            r = json.loads(l); done.add((r["dataset"], r["idx"])); n_done += 1
            nc += r["ok"]; esc_n += int(r["escalate"])
    print(f"resuming: {n_done} queries already in {A.out}", flush=True)
todo = [(nm, i) for (nm, i) in queries if (nm, i) not in done]
print(f"{len(queries)} total, {len(todo)} remaining. batch-1 LIVE cascade. warmup {A.warmup}...", flush=True)
for _, i in queries[:A.warmup]:
    run_7b(data[i]); run_32b(data[i])

sampler = DualPowerSampler([A.gpu7, A.gpu32]); sampler.start()
t_run = time.time()
fh = open(A.out, "a")
for (nm, i) in todo:
    tq0 = time.time()
    pred7, mg, lat7, g7 = run_7b(data[i])
    escalate = bool(GATE.predict_proba([[mg]])[:, 1][0] < TAU)        # the router, live
    if escalate:
        pred32, lat32, g32 = run_32b(data[i]); final = pred32
    else:
        pred32, lat32, g32 = "", 0.0, 0; final = pred7
    tq1 = time.time()
    mean_p, energy = sampler.window(tq0, tq1)
    g = gold(data[i]); ok = int(final == g)
    fh.write(json.dumps({"idx":i,"dataset":nm,"escalate":escalate,"ok":ok,"final":final,
        "pred7":pred7,"pred32":pred32,"gold":g,"margin":round(mg,4),
        "latency_s":round(tq1-tq0,4),"lat7_s":round(lat7,4),"lat32_s":round(lat32,4),
        "gpu7_energy_j":round(energy[A.gpu7],2),"gpu32_energy_j":round(energy[A.gpu32],2),
        "energy_j":round(energy[A.gpu7]+energy[A.gpu32],2),
        "gpu7_power_w":round(mean_p[A.gpu7],1),"gpu32_power_w":round(mean_p[A.gpu32],1),
        "gen7":g7,"gen32":g32})+"\n"); fh.flush()
    n_done += 1; nc += ok; esc_n += int(escalate)
    if n_done % 20 == 0:
        print(f"   [{n_done}/{len(queries)}] acc={nc/n_done:.3f} esc={esc_n/n_done:.0%} "
              f"last_lat={tq1-tq0:.2f}s ({(time.time()-t_run)/60:.1f}min this session)", flush=True)
sampler.stop(); fh.close()

rows = [json.loads(l) for l in open(A.out)]
lat = np.array([r["latency_s"] for r in rows]); en = np.array([r["energy_j"] for r in rows])
esc = np.array([r["escalate"] for r in rows]); ok = np.array([r["ok"] for r in rows])
lat7 = np.array([r["lat7_s"] for r in rows]); lat32 = np.array([r["lat32_s"] for r in rows])
p7 = np.array([r["gpu7_power_w"] for r in rows]); p32 = np.array([r["gpu32_power_w"] for r in rows])
acc7 = np.array([r["pred7"] == r["gold"] for r in rows]); est32 = lat32[esc].mean() if esc.any() else 0.0
print(f"\n===== LIVE CASCADE  cap={A.cap}  ({len(rows)} queries, batch-1, real escalation) =====")
print(f"accuracy (live routed) : {ok.mean():.3f}")
print(f"escalation rate        : {esc.mean():.1%}   (tau={TAU:.3f})")
print(f"latency mean/p50/p95   : {lat.mean():.2f} / {np.percentile(lat,50):.2f} / {np.percentile(lat,95):.2f} s")
print(f"  7B leg (all queries) : {lat7.mean():.2f}s    32B leg (escalated)  : {est32:.2f}s")
print(f"energy / query         : {en.mean():.1f} J  (GPU{A.gpu7}+GPU{A.gpu32} summed over the query)")
print(f"power while active     : GPU{A.gpu7}(7B) {p7.mean():.0f}W   GPU{A.gpu32}(32B, escalated) {p32[esc].mean() if esc.any() else 0:.0f}W")
print(f"VRAM resident          : GPU{A.gpu7}(7B) {VR7:.0f} + GPU{A.gpu32}(32B) {VR32:.0f} = {VR7+VR32:.0f} MB total")
print(f"\nreference points (same queries):")
print(f"  always-7B  : acc={acc7.mean():.3f}  lat={lat7.mean():.2f}s/q  VRAM={VR7:.0f}MB")
print(f"  always-32B : lat~{est32:.2f}s/q (est. from the escalated 32B calls)  VRAM={VR32:.0f}MB")
print(f"  cascade    : acc={ok.mean():.3f}  lat={lat.mean():.2f}s/q  VRAM={VR7+VR32:.0f}MB"
      f"  -> {(est32/lat.mean()) if lat.mean()>0 else 0:.2f}x faster than always-32B (real-time, batch-1)")
