#!/usr/bin/env python3
"""
run_32b_modes_vllm.py - strong-leg ABLATION runner for the 32B: vary the escalation TARGET's
think/no-think mode and image-resolution cap, to test a CHEAPER strong leg. The deployed cascade
escalates to 32B-think@fullres (~477 decode tokens + full-res prefill = the dominant cascade cost).
This runner produces 32B-{think,nothink}@{fullres,cap320,...} on the SAME eval idx (identical
fixed_slice seed-42 selection + parsing as run_32b_vllm.py) so per-sample accuracy/cost compare
directly to ckpts/gate_32b. If a cheaper 32B config retains accuracy, the cascade's per-escalation
cost drops sharply regardless of the gate.

  --arm think|nothink   think: SYS_THINK, max_tokens 2048 ; nothink: SYS_NOTHINK, max_tokens 16
  --cap fullres|cap640|cap320|cap160|cap80   max_pixels = HIGH_PX // {1,2,4,8,16}
Output: <ckpt_dir>/ckpt_<ds>_<arm>_norag.jsonl   (schema identical to gate_32b)
"""
import argparse, re, json, random, time, os
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

MODEL = "/data/dan/weights/MedVLThinker-32B-RL_m23k"
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
SYS_THINK = ("You will solve a problem/request. You should provide your thoughts "
             "within <think> </think> tags before providing the answer.")
SYS_NOTHINK = "Answer with only the correct option letter (e.g. 'A'). Do not explain."
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
CHUNK = 128

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=4000)            # >= largest benchmark -> covers all idx
ap.add_argument("--datasets", nargs="+",
                default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
                         "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--arm", choices=["think", "nothink"], required=True)
ap.add_argument("--cap", choices=list(CAP_DIV), default="fullres")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=0)      # 0 -> auto by arm
ap.add_argument("--max_images", type=int, default=8)
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"
os.makedirs(A.ckpt_dir, exist_ok=True)
SYS = SYS_THINK if A.arm == "think" else SYS_NOTHINK
MAXTOK = A.max_tokens or (2048 if A.arm == "think" else 16)
MAXPX = HIGH_PX // CAP_DIV[A.cap]

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]

def subset(*keys):
    return [i for i, n in enumerate(data["dataset_name"])
            if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
def mx_by_type(t):
    out = []
    for i, n in enumerate(data["dataset_name"]):
        if "medxpert" not in n.lower(): continue
        mc = data[i].get("misc")
        try: qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception: qt = ""
        if qt.lower() == t: out.append(i)
    return out
DATASET_IDX = {
    "MedXpert-Reasoning": lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding": lambda: mx_by_type("understanding"),
    "PMC-VQA": lambda: subset("pmcvqa", "pmc"),
    "SLAKE": lambda: subset("slake"),
    "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"),
    "PathVQA": lambda: subset("pathvqa", "path"),
    "MMMU": lambda: subset("mmmu"),
}
def fixed_slice(idxs):
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]
    return s[SHARD_K::SHARD_N]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")
LETTER_SET = set(chr(ord('A') + k) for k in range(10))
def _lid(L):
    for v in (L, " " + L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: return e[0]
    return None
LID = {L: _lid(L) for L in LETTER_SET}; ID2LET = {v: k for k, v in LID.items() if v is not None}

def pred_from_text(t):
    if A.arm == "think":
        tail = t.split("</think>")[-1]; m = LET.search(tail) or LET.search(t)
    else:
        m = LET.search(t)
    return (m.group(1), 1) if m else ("?", 0)

def opt_logprob_vllm(token_ids, logprobs_list):
    seen = ""; start = 0
    if A.arm == "think":
        for step, tok in enumerate(token_ids):
            seen += proc.tokenizer.decode([tok])
            if "</think>" in seen: start = step + 1; break
    for step in range(start, len(token_ids)):
        d = proc.tokenizer.decode([token_ids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d) == 1 or d[1] in ").: "):
            lp = logprobs_list[step] if step < len(logprobs_list) else None
            if not lp: return {}
            return {ID2LET[tid]: round(float(o.logprob), 4) for tid, o in lp.items() if tid in ID2LET}
    return {}

def build_prompt(ex):
    opts = parse_opts(ex["options"]); assert opts
    q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    imgs_meta = [{"type": "image", "image": im, "max_pixels": MAXPX, "min_pixels": MIN_PX}
                 for im in (ex.get("images") or [])]
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": imgs_meta + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if image_inputs: req["multi_modal_data"] = {"image": image_inputs}
    return req

print(f"loading 32B vLLM TP={A.tp} | arm={A.arm} cap={A.cap} (max_pixels={MAXPX}) max_tokens={MAXTOK}", flush=True)
llm = LLM(model=MODEL, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": A.max_images}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, logprobs=20)

def write_result(fh, i, o):
    gen = o.outputs[0].text; tok_ids = list(o.outputs[0].token_ids); lps = o.outputs[0].logprobs or []
    g = gold(data[i]); p, parse_ok = pred_from_text(gen); lp = opt_logprob_vllm(tok_ids, lps); ok = int(g == p)
    fh.write(json.dumps({"idx": i, "gold": g, "pred": p, "ok": ok, "parse_ok": parse_ok,
        "opt_logprobs": lp, "gen_tokens": len(tok_ids), "latency_s": None, "raw_output": gen}) + "\n")
    return ok, len(tok_ids)

t0 = time.time(); tg = tn = 0
for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! unknown {name}", flush=True); continue
    sel = fixed_slice(DATASET_IDX[name]())
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{name}_{A.arm}_norag{SHARD_TAG}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    todo = [i for i in sel if i not in done]
    print(f"\n--- {name}: {len(sel)} total, {len(done)} done, {len(todo)} to run -> {ckpt} ---", flush=True)
    with open(ckpt, "a") as fh:
        for c0 in range(0, len(todo), CHUNK):
            chunk = todo[c0:c0 + CHUNK]; reqs = [build_prompt(data[i]) for i in chunk]
            try:
                outs = llm.generate(reqs, sp)
            except Exception as e:
                print(f"   chunk failed ({e}); one-by-one", flush=True); outs = []
                for r in reqs:
                    try: outs.append(llm.generate([r], sp)[0])
                    except Exception as e2: print(f"     skip: {e2}", flush=True); outs.append(None)
            nc = nd = 0
            for i, o in zip(chunk, outs):
                if o is None: continue
                ok, ntok = write_result(fh, i, o); nc += ok; nd += 1; tg += ntok; tn += 1
            fh.flush(); el = time.time() - t0
            print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] acc={nc/nd if nd else 0:.3f} "
                  f"| {tn/el:.2f} samp/s, {tg/el:.0f} tok/s", flush=True)
    print(f">> {name} done", flush=True)
print(f"\nDONE arm={A.arm} cap={A.cap}: {tn} samples in {(time.time()-t0)/60:.1f} min", flush=True)
