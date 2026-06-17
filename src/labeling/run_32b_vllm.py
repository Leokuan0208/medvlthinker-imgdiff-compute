#!/usr/bin/env python3
"""
run_32b_vllm.py - vLLM tensor-parallel (2 GPU) runner for MedVLThinker-32B.
Both GPUs compute each forward pass together (TP=2) AND vLLM continuous-batches
many questions at once -> large throughput win over HF pipeline-shard batch-1.

Mirrors gate_router.py EXACTLY where it matters for 7B<->32B comparability:
  same SYS_THINK prompt, same question+options format, same seed-42 fixed_slice,
  same answer-letter parsing, same max_pixels image cap, greedy decoding.
Output schema + filenames match run_32b.py so cascade_check.py reads it unchanged.

max_tokens=2048: the 32B writes long reasoning; 512 truncated ~80% of traces.
limit_mm_per_prompt image=8: MedXpert bundles up to 6 figure panels per question;
  vLLM (unlike HF generate) hard-rejects prompts above the declared image limit.
Per-chunk try/except falls back to one-by-one so a single bad sample can't kill
the run (cascade_check joins on idx, so a few skips are harmless).

Honest differences from HF run (negligible for the complementarity check):
  - opt_logprobs from vLLM TOP-20 logprobs at the answer token, not full vocab.
    pred/gold/ok come from TEXT parsing exactly like HF, so correctness identical.
  - latency_s = null (meaningless under continuous batching); throughput printed.
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
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28      # SAME cap as gate_router.py
CHUNK = 128

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--datasets", nargs="+",
                default=["MedXpert-Reasoning","MedXpert-Understanding","PMC-VQA"])
ap.add_argument("--shard", default="0/1")
ap.add_argument("--ckpt_dir", default="ckpts/gate_32b")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=2048)
ap.add_argument("--max_images", type=int, default=8)
A = ap.parse_args()
SHARD_K, SHARD_N = (int(x) for x in A.shard.split("/"))
SHARD_TAG = "" if SHARD_N == 1 else f"_s{SHARD_K}of{SHARD_N}"  # single shard -> no suffix
os.makedirs(A.ckpt_dir, exist_ok=True)

ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]

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
DATASET_IDX = {
    "MedXpert-Reasoning":               lambda: mx_by_type("reasoning"),
    "MedXpert-Understanding":           lambda: mx_by_type("understanding"),
    "PMC-VQA":                          lambda: subset("pmcvqa","pmc"),
    "SLAKE":                            lambda: subset("slake"),
    "VQA-RAD":                          lambda: subset("vqarad","vqa_rad","rad"),
    "PathVQA":                          lambda: subset("pathvqa","path"),
    "MMMU":                             lambda: subset("mmmu"),
}
def fixed_slice(idxs):                              # IDENTICAL to gate_router.py
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]
    return s[SHARD_K::SHARD_N]
def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))
def gold(ex): return str(ex["answer_label"]).strip().upper()[:1]

proc = AutoProcessor.from_pretrained(MODEL)
LET = re.compile(r"\b([A-J])\b")
LETTER_SET = set(chr(ord('A')+k) for k in range(10))
def _lid(L):
    for v in (L, " "+L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e: return e[0]
    return None
LID = {L:_lid(L) for L in LETTER_SET}
ID2LET = {v:k for k,v in LID.items() if v is not None}

def pred_from_text(t):
    tail = t.split("</think>")[-1]; m = LET.search(tail) or LET.search(t)
    return (m.group(1), 1) if m else ("?", 0)

def opt_logprob_vllm(token_ids, logprobs_list):
    seen=""; start=0
    for step,tok in enumerate(token_ids):
        seen += proc.tokenizer.decode([tok])
        if "</think>" in seen: start=step+1; break
    for step in range(start, len(token_ids)):
        d = proc.tokenizer.decode([token_ids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d)==1 or d[1] in ").: "):
            lp = logprobs_list[step] if step < len(logprobs_list) else None
            if not lp: return {}
            out={}
            for tid, lpobj in lp.items():
                if tid in ID2LET: out[ID2LET[tid]] = round(float(lpobj.logprob), 4)
            return out
    return {}

def build_prompt(ex):
    opts = parse_opts(ex["options"]); assert opts
    q = ex["question"]+"\n"+"\n".join(f"{k}) {v}" for k,v in opts.items())
    imgs_meta = [{"type":"image","image":im,"max_pixels":HIGH_PX,"min_pixels":MIN_PX}
                 for im in (ex.get("images") or [])]
    msgs = [{"role":"system","content":SYS_THINK},
            {"role":"user","content":imgs_meta+[{"type":"text","text":q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if image_inputs: req["multi_modal_data"] = {"image": image_inputs}
    return req

print(f"loading 32B in vLLM (TP={A.tp}, gpu_mem={A.gpu_mem}, max_len={A.max_model_len}, "
      f"max_tokens={A.max_tokens}, max_images={A.max_images})...", flush=True)
llm = LLM(model=MODEL, tensor_parallel_size=A.tp, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
          limit_mm_per_prompt={"image": A.max_images}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=A.max_tokens, logprobs=20)

def write_result(fh, i, o):
    gen = o.outputs[0].text
    tok_ids = list(o.outputs[0].token_ids)
    lps = o.outputs[0].logprobs or []
    g = gold(data[i]); p, parse_ok = pred_from_text(gen)
    lp = opt_logprob_vllm(tok_ids, lps); ok = int(g==p)
    fh.write(json.dumps({"idx":i,"gold":g,"pred":p,"ok":ok,"parse_ok":parse_ok,
        "opt_logprobs":lp,"gen_tokens":len(tok_ids),
        "latency_s":None,"raw_output":gen})+"\n")
    return ok, len(tok_ids)

t_start = time.time(); total_gen = 0; total_n = 0
for name in A.datasets:
    if name not in DATASET_IDX: print(f"!! unknown {name}", flush=True); continue
    sel = fixed_slice(DATASET_IDX[name]())
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{name}_think_norag{SHARD_TAG}.jsonl")
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
            chunk = todo[c0:c0+CHUNK]
            reqs = [build_prompt(data[i]) for i in chunk]
            # resilient generate: if a chunk fails, retry one-by-one and skip bad samples
            try:
                outs = llm.generate(reqs, sp)
            except Exception as e:
                print(f"   chunk generate failed ({e}); retrying one-by-one", flush=True)
                outs = []
                for r in reqs:
                    try:
                        outs.append(llm.generate([r], sp)[0])
                    except Exception as e2:
                        print(f"     skipping one sample: {e2}", flush=True)
                        outs.append(None)
            ncorr = ndone = 0
            for i, o in zip(chunk, outs):
                if o is None:                       # sample that failed even individually
                    continue
                ok, ntok = write_result(fh, i, o)
                ncorr += ok; ndone += 1; total_gen += ntok; total_n += 1
            fh.flush()
            el = time.time()-t_start
            acc = ncorr/ndone if ndone else 0.0
            print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] chunk_acc={acc:.3f} (n={ndone}) "
                  f"| {total_n/el:.2f} samp/s, {total_gen/el:.0f} tok/s", flush=True)
    print(f">> {name} done", flush=True)

el = time.time()-t_start
print(f"\nDONE  {total_n} samples in {el/60:.1f} min "
      f"({total_n/max(el,1):.2f} samp/s, {total_gen/max(el,1):.0f} tok/s)", flush=True)
