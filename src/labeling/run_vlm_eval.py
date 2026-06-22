#!/usr/bin/env python3
"""
run_vlm_eval.py - GENERAL Qwen2.5-VL-family eval runner for the research loop. Runs ANY local/HF
Qwen2.5-VL model (--model_path) over the MedVLThinker-Eval benchmarks (or a subset) and writes
per-sample predictions in the SAME schema as the rest of the repo (idx,gold,pred,ok,parse_ok,
opt_logprobs,gen_tokens,raw_output), so the existing harness/analysis reads them unchanged.
Generalizes run_32b_modes_vllm.py by exposing --model_path/--system/--cap. vLLM, TP configurable.
Set HF_HOME=/data/dan/hf_cache so any download stays off the main drive. Launch from repo root.

Examples:
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_vlm_eval.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct --tag qwen25vl7b --cap fullres --tp 1
  (downloads to /data cache if not local; writes ckpts/peer/qwen25vl7b/ckpt_<ds>_<tag>.jsonl)
"""
import argparse, re, json, random, time, os
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts within "
             "<think> </think> tags before providing the answer.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
CHUNK = 128

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True, help="local path or HF id (Qwen2.5-VL arch)")
ap.add_argument("--tag", required=True, help="short name used in output filenames")
ap.add_argument("--arm", choices=["think", "nothink"], default="nothink")
ap.add_argument("--system", default="", help="override system prompt; default by arm")
ap.add_argument("--cap", choices=list(CAP_DIV), default="fullres")
ap.add_argument("--datasets", nargs="+",
                default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
                         "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=0)
ap.add_argument("--max_images", type=int, default=8)
ap.add_argument("--blank", action="store_true", help="replace each image with mid-gray (vision-sensitivity counterfactual)")
ap.add_argument("--batch1", action="store_true", help="measure real batch-1 latency: one request at a time, record latency_s")
ap.add_argument("--user_instr", default="", help="append a NATIVE instruction to the user turn (model-specific reasoning trigger)")
ap.add_argument("--no_system", action="store_true", help="omit the system message (models that reason with an empty system prompt, e.g. Lingshu)")
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
if A.batch1: CHUNK = 1
SYS = A.system or (SYS_THINK if A.arm == "think" else SYS_NOTHINK)
MAXTOK = A.max_tokens or (2048 if A.arm == "think" else 16)
MAXPX = HIGH_PX // CAP_DIV[A.cap]

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]; data = ds[split]
def subset(*keys):
    return [i for i, n in enumerate(data["dataset_name"]) if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
def mx_by_type(t):
    out = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: out.append(i)
    return out
DSI = {"MedXpert-Reasoning": lambda: mx_by_type("reasoning"), "MedXpert-Understanding": lambda: mx_by_type("understanding"),
       "PMC-VQA": lambda: subset("pmcvqa", "pmc"), "SLAKE": lambda: subset("slake"),
       "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"), "PathVQA": lambda: subset("pathvqa", "path"),
       "MMMU": lambda: subset("mmmu")}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); return s[:A.n]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

proc = AutoProcessor.from_pretrained(A.model_path)
LET = re.compile(r"\b([A-J])\b"); LETTER_SET = set(chr(ord('A') + k) for k in range(10))
def _lid(L):
    for v in (L, " " + L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: return e[0]
    return None
LID = {L: _lid(L) for L in LETTER_SET}; ID2LET = {v: k for k, v in LID.items() if v is not None}
ANS_MARK = re.compile(r"(?:answer|option|choice|correct|select)\s*(?:is|:|=)?\s*\**\(?\s*([A-J])\b", re.I)
BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-J])", re.I)
def pred_from_text(t):
    if A.arm == "think":  # CoT: \boxed{X} > final 'Answer: X' marker > last standalone letter
        bx = BOXED.findall(t)
        if bx: return (bx[-1].upper(), 1)
        tail = t.split("</think>")[-1] if "</think>" in t else t
        ms = list(ANS_MARK.finditer(tail))
        if ms: return (ms[-1].group(1).upper(), 1)
        ls = LET.findall(tail)
        return (ls[-1], 1) if ls else ("?", 0)
    m = LET.search(t); return (m.group(1), 1) if m else ("?", 0)
def opt_lp(tok, lps):
    start = 0
    if A.arm == "think":
        seen = ""
        for step, tk in enumerate(tok):
            seen += proc.tokenizer.decode([tk])
            if "</think>" in seen: start = step + 1; break
    for step in range(start, len(tok)):
        d = proc.tokenizer.decode([tok[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d) == 1 or d[1] in ").: "):
            lp = lps[step] if step < len(lps) else None
            if not lp: return {}
            return {ID2LET[t]: round(float(o.logprob), 4) for t, o in lp.items() if t in ID2LET}
    return {}
def _blank_like(x):
    sz = x.size if isinstance(x, Image.Image) else (336, 336)
    return Image.new("RGB", sz, (127, 127, 127))
def build(ex):
    opts = parse_opts(ex["options"]); q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    if A.user_instr: q = q + "\n" + A.user_instr
    srcs = ex.get("images") or []
    if A.blank: srcs = [_blank_like(x) for x in srcs]
    im = [{"type": "image", "image": x, "max_pixels": MAXPX, "min_pixels": MIN_PX} for x in srcs]
    msgs = ([] if A.no_system else [{"role": "system", "content": SYS}]) + [{"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

print(f"loading {A.model_path} (tag={A.tag}, arm={A.arm}, cap={A.cap}, tp={A.tp})", flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": A.max_images}, trust_remote_code=True,
          max_num_seqs=(1 if A.batch1 else 256))
sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, logprobs=20)
PWR = None
if A.batch1:
    import sys as _sys; _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nvml_power import PowerMonitor, gpu_indices
    PWR = PowerMonitor(gpu_indices(A.tp)); PWR.start()
t0 = time.time(); tn = 0
for name in A.datasets:
    if name not in DSI: continue
    sel = fixed_slice(DSI[name]())
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{name}_{A.tag}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    todo = [i for i in sel if i not in done]
    print(f"\n--- {name}: {len(sel)} total, {len(todo)} to run -> {ckpt} ---", flush=True)
    with open(ckpt, "a") as fh:
        for c0 in range(0, len(todo), CHUNK):
            ch = todo[c0:c0 + CHUNK]; reqs = [build(data[i]) for i in ch]
            _tc = time.time()
            try: outs = llm.generate(reqs, sp)
            except Exception as e:
                print(f"   chunk failed ({e}); one-by-one", flush=True); outs = []
                for r in reqs:
                    try: outs.append(llm.generate([r], sp)[0])
                    except Exception as e2: print(f"     skip: {e2}", flush=True); outs.append(None)
            _te = time.time(); _lat = (_te - _tc) / max(len(ch), 1); _ej = None
            if PWR and PWR.ok:
                _e = PWR.energy_between(_tc, _te)
                if _e is not None: _ej = _e / max(len(ch), 1)
                PWR.trim(_te - 5)
            nc = nd = 0
            for i, o in zip(ch, outs):
                if o is None: continue
                gen = o.outputs[0].text; tk = list(o.outputs[0].token_ids); lps = o.outputs[0].logprobs or []
                g = gold(data[i]); p, pk = pred_from_text(gen); ok = int(g == p)
                fh.write(json.dumps({"idx": i, "gold": g, "pred": p, "ok": ok, "parse_ok": pk,
                    "opt_logprobs": opt_lp(tk, lps), "gen_tokens": len(tk), "latency_s": (round(_lat, 4) if A.batch1 else None),
                    "energy_j": (round(_ej, 3) if (A.batch1 and _ej is not None) else None), "raw_output": gen}) + "\n")
                nc += ok; nd += 1; tn += 1
            fh.flush(); print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} | {tn/(time.time()-t0):.1f}/s", flush=True)
    print(f">> {name} done", flush=True)
if PWR: PWR.stop(); print(f"PEAK_VRAM_GB={PWR.peak_mem_gb():.2f}", flush=True)
print(f"\nDONE {A.tag}: {tn} samples in {(time.time()-t0)/60:.1f} min", flush=True)
