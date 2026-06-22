#!/usr/bin/env python3
"""
run_peer_eval.py - MODEL-AGNOSTIC vLLM eval runner for CROSS-FAMILY peer VLMs (InternVL, Phi-3.5-V,
Llama-Vision, etc.). Uses vLLM's llm.chat() so each model's own chat template + image-token handling
is applied automatically; images are passed as base64 data-URIs (resized to --max_side). Writes the
SAME per-sample schema as the rest of the repo (idx,gold,pred,ok,parse_ok,opt_logprobs,gen_tokens,
raw_output) so the existing harness reads peer labels unchanged. Set HF_HOME=/data/dan/hf_cache to
keep downloads off the main drive. Launch from repo root.

Example (cross-family peer, premise test on competent-4):
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_peer_eval.py \
    --model_path OpenGVLab/InternVL2_5-8B --tag internvl25_8b --tp 1 \
    --datasets PMC-VQA SLAKE VQA-RAD PathVQA --ckpt_dir ckpts/peer/internvl25_8b
"""
import argparse, re, json, random, time, os, io, base64
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
INSTR = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
THINK_INSTR = "Reason step by step about the image and the question, then end with a line 'Answer: X' where X is the correct option letter."
CHUNK = 64
LETTER_SET = set(chr(ord('A') + k) for k in range(10))
LET = re.compile(r"\b([A-J])\b")
ANS_MARK = re.compile(r"(?:answer|option|choice|correct|select)\s*(?:is|:|=)?\s*\**\(?\s*([A-J])\b", re.I)
BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-J])", re.I)
FINAL = re.compile(r"final answer\s*(?:is)?\s*[:=]?\s*\**\(?\s*([A-J])\b", re.I)  # "### The final answer is: X" / "Final Answer: X"

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--datasets", nargs="+", default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA"])
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_side", type=int, default=896, help="resize so longest image side <= this (cost control)")
ap.add_argument("--max_images", type=int, default=6)
ap.add_argument("--think", action="store_true", help="CoT reasoning then final 'Answer: X'")
ap.add_argument("--max_tokens", type=int, default=0)
ap.add_argument("--verify", action="store_true", help="AutoMix self-verify: judge own nt-answer Yes/No, record p_yes_norm")
ap.add_argument("--pred_dir", default="", help="dir with this model's nt predictions (required for --verify)")
ap.add_argument("--batch1", action="store_true", help="measure real batch-1 latency: one request at a time, record latency_s")
ap.add_argument("--think_instr", default="", help="override the appended think instruction with a model's NATIVE one")
ap.add_argument("--system", default="", help="prepend a system message (for models whose native reasoning is a system prompt)")
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
if A.batch1: CHUNK = 1

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

def to_data_uri(img):
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img)) if isinstance(img, (bytes, bytearray)) else Image.open(img)
    img = img.convert("RGB")
    w, h = img.size; m = max(w, h)
    if m > A.max_side:
        r = A.max_side / m; img = img.resize((max(1, int(w * r)), max(1, int(h * r))))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def conv(ex):
    opts = parse_opts(ex["options"]); q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    imgs = (ex.get("images") or [])[:A.max_images]
    content = [{"type": "image_url", "image_url": {"url": to_data_uri(im)}} for im in imgs]
    instr = (A.think_instr or THINK_INSTR) if A.think else INSTR
    content.append({"type": "text", "text": q + "\n" + instr})
    msgs = []
    if A.system: msgs.append({"role": "system", "content": A.system})
    msgs.append({"role": "user", "content": content})
    return msgs

tok = AutoTokenizer.from_pretrained(A.model_path, trust_remote_code=True)
def _ids(L):
    out = []
    for v in (L, " " + L):
        e = tok.encode(v, add_special_tokens=False)
        if e: out.append(e[-1])
    return out
ID2LET = {}
for L in LETTER_SET:
    for tid in _ids(L): ID2LET.setdefault(tid, L)
def pred_and_lp(text, tokids, lps):
    if A.think:  # CoT: final answer = \boxed{X}, else last 'Answer: X' marker, else last standalone letter
        bx = BOXED.findall(text); fa = FINAL.findall(text); ms = list(ANS_MARK.finditer(text)); ls = LET.findall(text)
        pred = (bx[-1].upper() if bx else (fa[-1].upper() if fa else (ms[-1].group(1).upper() if ms else (ls[-1] if ls else "?"))))
        pk = int(bool(bx or fa or ms or ls))
    else:
        m = LET.search(text); pred = (m.group(1) if m else "?"); pk = int(bool(m))
    olp = {}
    for step in range(len(tokids)):
        d = tok.decode([tokids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d) == 1 or d[1] in ").: "):
            lp = lps[step] if step < len(lps) else None
            if lp: olp = {ID2LET[t]: round(float(o.logprob), 4) for t, o in lp.items() if t in ID2LET}
            break
    return pred, pk, olp

import math, glob as _glob
def _yn_ids(words):
    ids = {}
    for w in words:
        for v in (w, " " + w):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1: ids[e[0]] = w
    return ids
YES_IDS = _yn_ids(["Yes", "yes", "YES"]); NO_IDS = _yn_ids(["No", "no", "NO"])
def load_preds(d):
    m = {}
    for f in _glob.glob(os.path.join(d, "*.jsonl")):
        mm = re.match(r"ckpt_(.+?)_", os.path.basename(f)); dsn = mm.group(1) if mm else None
        for l in open(f):
            if l.strip():
                r = json.loads(l); m[(dsn, r["idx"])] = r.get("pred")
    return m
PREDS = load_preds(A.pred_dir) if A.verify else {}
def conv_verify(ex, proposed):
    opts = parse_opts(ex["options"])
    ptxt = f"{proposed}) {opts.get(proposed, '')}" if proposed and proposed != "?" else str(proposed)
    q = (ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
         + f"\n\nProposed answer: {ptxt}\nIs the proposed answer correct? Answer only Yes or No.")
    imgs = (ex.get("images") or [])[:A.max_images]
    content = [{"type": "image_url", "image_url": {"url": to_data_uri(im)}} for im in imgs]
    content.append({"type": "text", "text": q})
    return [{"role": "user", "content": content}]
def yn(lps):
    if not lps: return None, None
    lp0 = lps[0]
    py = max((math.exp(o.logprob) for t, o in lp0.items() if t in YES_IDS), default=0.0)
    pn = max((math.exp(o.logprob) for t, o in lp0.items() if t in NO_IDS), default=0.0)
    return py, pn

print(f"loading {A.model_path} (tag={A.tag}, tp={A.tp}, max_side={A.max_side}, verify={A.verify})", flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": A.max_images}, trust_remote_code=True,
          max_num_seqs=(1 if A.batch1 else 256))
sp = SamplingParams(temperature=0.0, max_tokens=(2 if A.verify else (A.max_tokens or (1024 if A.think else 12))), logprobs=20)
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
    todo = [i for i in sel if i not in done and (not A.verify or (name, i) in PREDS)]
    print(f"\n--- {name}: {len(sel)} total, {len(todo)} to run -> {ckpt} ---", flush=True)
    with open(ckpt, "a") as fh:
        for c0 in range(0, len(todo), CHUNK):
            ch = todo[c0:c0 + CHUNK]
            convs = [(conv_verify(data[i], PREDS.get((name, i))) if A.verify else conv(data[i])) for i in ch]
            _tc = time.time()
            try: outs = llm.chat(convs, sp, use_tqdm=False)
            except Exception as e:
                print(f"   chunk failed ({str(e)[:120]}); one-by-one", flush=True); outs = []
                for cv in convs:
                    try: outs.append(llm.chat([cv], sp, use_tqdm=False)[0])
                    except Exception as e2: print(f"     skip: {str(e2)[:80]}", flush=True); outs.append(None)
            _te = time.time(); _lat = (_te - _tc) / max(len(ch), 1); _ej = None
            if PWR and PWR.ok:
                _e = PWR.energy_between(_tc, _te)
                if _e is not None: _ej = _e / max(len(ch), 1)
                PWR.trim(_te - 5)
            nc = nd = 0
            for i, o in zip(ch, outs):
                if o is None: continue
                gen = o.outputs[0].text; tkids = list(o.outputs[0].token_ids); lps = o.outputs[0].logprobs or []
                g = gold(data[i])
                if A.verify:
                    proposed = PREDS.get((name, i)); py, pn = yn(lps)
                    norm = (py / (py + pn)) if (py is not None and (py + pn) > 0) else None
                    fh.write(json.dumps({"idx": i, "gold": g, "pred": proposed,
                        "p_yes": round(py, 5) if py is not None else None, "p_no": round(pn, 5) if pn is not None else None,
                        "p_yes_norm": round(norm, 5) if norm is not None else None,
                        "ok": int(g == proposed), "raw_output": gen}) + "\n")
                    nc += int(g == proposed); nd += 1; tn += 1
                else:
                    p, pk, olp = pred_and_lp(gen, tkids, lps); ok = int(g == p)
                    fh.write(json.dumps({"idx": i, "gold": g, "pred": p, "ok": ok, "parse_ok": pk,
                        "opt_logprobs": olp, "gen_tokens": len(tkids), "latency_s": (round(_lat, 4) if A.batch1 else None),
                        "energy_j": (round(_ej, 3) if (A.batch1 and _ej is not None) else None), "raw_output": gen}) + "\n")
                    nc += ok; nd += 1; tn += 1
            fh.flush(); print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} | {tn/(time.time()-t0):.1f}/s", flush=True)
    print(f">> {name} done", flush=True)
if PWR: PWR.stop(); print(f"PEAK_VRAM_GB={PWR.peak_mem_gb():.2f}", flush=True)
print(f"\nDONE {A.tag}: {tn} samples in {(time.time()-t0)/60:.1f} min", flush=True)
