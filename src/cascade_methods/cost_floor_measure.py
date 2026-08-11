#!/usr/bin/env python3
"""
cost_floor_measure.py -- ATTACK 3 (COST-FLOOR), rule-2 corroboration measurement.

WHY.  The deployed cost model charges open-text best-of-N as  N generations + N verifier
forwards, EACH AT FULL BATCH-1 COST (`BO8 = dict(ms=522.0, flop=16.0)`,
src/cascade_methods/integrated_method.py:55, reused by every headline script).  Under the
repo's own derived FLOP model (artifacts/flop_ratio_derivation_2026-08-03.json) a 7B forward
on this workload is 98.85% prefill (vision tower + LM prefill) and 1.16% decode+head, so a
best-of-N that SHARES the prefill should cost ~1 + (N-1)*0.0116 forwards, not N.
`src/labeling/run_openvqa.py:154` really does generate with vLLM `SamplingParams(n=N)`, i.e.
the deployed generation path is a shared-prefill path.

Attack 3's pre-registration rule 2 says the re-costing must be CORROBORATED BY MEASUREMENT:
if measured energy for BoN@8 exceeds the re-costed model by more than 30%, the re-costing is
REJECTED.  This script produces that measurement.

WHAT IT MEASURES (batch-1 *request*, NVML energy integrated over the timed call across visible
GPUs only, real vqa_rad open-text items, cap320 = max_pixels 1280*28*28//4 -- identical recipe
to src/cascade_methods/open_measure_latency_energy.py and bestofn_measure_batch8.py):

  phase=vllm   (Lingshu-7B, vLLM, tp=1; --prefix_caching on|off)
    gen_n1 / gen_n4 / gen_n8   : one request, SamplingParams(n=N, T=0.7) -- the DEPLOYED
                                 generation path.  Records `num_cached_tokens` so prefill
                                 sharing is PROVEN, not assumed.
    ver_n1 / ver_n8            : the verifier PROMPT GEOMETRY -- 1 vs 8 candidate answers on
                                 the same (image, question), max_tokens=1.  Run on the BASE
                                 model with NO LoRA: this is a COST measurement of the prompt
                                 geometry, not a scoring run, so the documented
                                 "vLLM 0.9.0.1 silently drops visual.* LoRA modules" scoring
                                 ban does not apply (LoRA rank 32 moves FLOPs <1%).  NEVER use
                                 this path to score candidates.

  phase=hf     (Lingshu-7B, HF transformers, flash_attention_2, bf16)
    gen1                       : reproduces the canonical GEN7 = 347.1 ms / 45.8 J anchor
    gen8_hf                    : num_return_sequences=8 -- the NON-shared-prefill path that
                                 bestofn_latency_energy_2026-08-03.json measured (control)
    ver1 / ver8_batch          : one verifier forward / one batched forward over 8 candidates
                                 (with the deployed LoRA, --adapter)

Every leg writes a resumable per-item JSONL row with a per-item error guard.  Launch from the
repo root, one GPU pinned.

  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/cost_floor_measure.py --phase vllm \
      --prefix_caching on  --n 20 --warmup 4 --rep 1
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/cost_floor_measure.py --phase hf \
      --adapter ckpts/train/lora_verifier_disjoint --n 20 --warmup 4 --rep 1
"""
import argparse, os, io, glob, time, json, threading, traceback
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
MODEL = "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"

ap = argparse.ArgumentParser()
ap.add_argument("--phase", required=True, choices=["vllm", "hf"])
ap.add_argument("--model_path", default=MODEL)
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
ap.add_argument("--prefix_caching", default="on", choices=["on", "off"])
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--warmup", type=int, default=4)
ap.add_argument("--max_new", type=int, default=32)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--rep", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.85)
ap.add_argument("--outdir", default="results/cascade_methods/artifacts/_cost_floor_measure")
A = ap.parse_args()

# ---- numerics pins (protocol rule 8) -------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(1)

MAXPX, MINPX = 1280 * 28 * 28 // 4, 4 * 28 * 28          # cap320
SYS_GEN = ("You are an expert medical image analyst. Answer the question with a short, specific phrase. "
           "Do not explain.")
SYS_VER = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
           "proposed answer is correct. Respond with only 'Yes' or 'No'.")
# 8 fixed candidate strings, so ver_n8 has the same suffix geometry every iteration.
CANDS = ["pneumonia", "left lower lobe", "no acute abnormality", "cardiomegaly",
         "pleural effusion", "the liver", "yes, mild atelectasis", "chest x-ray"]

# ---- NVML power sampler (identical to open_measure_latency_energy.py) -----------------------
import pynvml
pynvml.nvmlInit()
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
_toks = [x.strip() for x in _vis.split(",") if x.strip() != ""]
_idxs = [int(t) for t in _toks] if (_toks and all(t.isdigit() for t in _toks)) \
        else list(range(pynvml.nvmlDeviceGetCount()))
HS = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in _idxs]
GPU_NAMES = [pynvml.nvmlDeviceGetName(h) for h in HS]
GPU_NAMES = [g.decode() if isinstance(g, bytes) else g for g in GPU_NAMES]
GPU_LIMIT_W = [pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0 for h in HS]


def watts():
    return sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in HS) / 1000.0


class E:
    def __enter__(s):
        s.j = 0.0; s.go = True
        def loop():
            last = time.time()
            while s.go:
                time.sleep(0.01); now = time.time(); s.j += watts() * (now - last); last = now
        s.th = threading.Thread(target=loop); s.th.start(); return s
    def __exit__(s, *a):
        s.go = False; s.th.join()


def idle_w(sec=2.0):
    t0 = time.time(); acc = 0.0; last = t0
    while time.time() - t0 < sec:
        time.sleep(0.02); now = time.time(); acc += watts() * (now - last); last = now
    return acc / (last - t0)


# ---- real items (identical selection rule to open_measure_latency_energy.py) ----------------
from PIL import Image
import pandas as pd
df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("/data/dan/dataset/vqa_rad/data/test-*.parquet"))],
               ignore_index=True)
ITEMS = []
for _, r in df.iterrows():
    q, a = r.get("question"), r.get("answer")
    if q is None and "conversations" in r:
        conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
    if str(a).strip().lower() in ("yes", "no"):
        continue
    img = r["image"]
    if isinstance(img, dict) and "bytes" in img:
        ITEMS.append((str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
    if len(ITEMS) >= A.n + A.warmup:
        break
NEED = A.n + A.warmup
assert len(ITEMS) >= NEED, f"only {len(ITEMS)} items, need {NEED}"
print(f"[items] {len(ITEMS)} real vqa_rad open items | GPUs {GPU_NAMES} limits {GPU_LIMIT_W} W", flush=True)

os.makedirs(os.path.join(REPO, A.outdir), exist_ok=True)
TAG = f"{A.phase}_{A.prefix_caching if A.phase == 'vllm' else 'hf'}_rep{A.rep}"
JL = os.path.join(REPO, A.outdir, f"{TAG}.jsonl")
done = set()
if os.path.exists(JL):
    for line in open(JL):
        try:
            r = json.loads(line); done.add((r["leg"], r["i"]))
        except Exception:
            pass
FH = open(JL, "a")


def emit(rec):
    FH.write(json.dumps(rec) + "\n"); FH.flush()


def run_leg(name, fn, nitems):
    """fn(i) -> dict(extra fields).  Times + NVML-integrates each call; per-item error guard."""
    print(f"[leg] {name}", flush=True)
    for i in range(nitems):
        if (name, i) in done:
            continue
        try:
            torch.cuda.synchronize()
            with E() as e:
                t0 = time.time()
                extra = fn(i)
                torch.cuda.synchronize()
                dt = time.time() - t0
            emit(dict(leg=name, i=i, warm=bool(i < A.warmup), lat_ms=dt * 1000.0,
                      energy_j=e.j, rep=A.rep, phase=A.phase, pc=A.prefix_caching, **extra))
            if i % 8 == 0:
                print(f"   [{name} {i}] {dt*1000:.0f} ms  {e.j:.1f} J  {extra}", flush=True)
        except Exception:
            emit(dict(leg=name, i=i, warm=bool(i < A.warmup), error=traceback.format_exc()[-800:]))
            print(f"   [{name} {i}] ERROR", flush=True)


# =================================================================================================
if A.phase == "vllm":
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def req(sys, body, img):
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": [{"type": "image", "image": img,
                                              "max_pixels": MAXPX, "min_pixels": MINPX},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        r = {"prompt": text}
        if imgs:
            r["multi_modal_data"] = {"image": imgs}
        return r

    llm = LLM(model=A.model_path, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=8192,
              limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
              enable_prefix_caching=(A.prefix_caching == "on"), seed=0)
    IDLE = idle_w()
    print(f"[idle-with-model-resident] {IDLE:.1f} W", flush=True)

    def _stats(outs):
        gen = sum(len(o.token_ids) for out in outs for o in out.outputs)
        cached = sum(int(getattr(out, "num_cached_tokens", 0) or 0) for out in outs)
        ptok = sum(len(out.prompt_token_ids) for out in outs)
        return dict(gen_tok=gen, cached_tok=cached, prompt_tok=ptok, n_req=len(outs))

    def gen_fn(N):
        sp = SamplingParams(temperature=A.temp, max_tokens=A.max_new, n=N, seed=0)
        def f(i):
            q, img = ITEMS[i]
            return _stats(llm.generate([req(SYS_GEN, f"Question: {q}", img)], sp, use_tqdm=False))
        return f

    def ver_fn(K):
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)
        def f(i):
            q, img = ITEMS[i]
            rs = [req(SYS_VER, f"Question: {q}\nProposed answer: {c}\n"
                               "Is the proposed answer correct? Answer Yes or No.", img)
                  for c in CANDS[:K]]
            return _stats(llm.generate(rs, sp, use_tqdm=False))
        return f

    for N in (1, 4, 8):
        run_leg(f"gen_n{N}", gen_fn(N), NEED)
    for K in (1, 8):
        run_leg(f"ver_n{K}", ver_fn(K), NEED)
    meta = dict(phase="vllm", prefix_caching=A.prefix_caching, idle_w=IDLE,
                gpu_names=GPU_NAMES, gpu_limit_w=GPU_LIMIT_W, model=A.model_path,
                temp=A.temp, max_new=A.max_new, n=A.n, warmup=A.warmup, rep=A.rep,
                vllm=__import__("vllm").__version__)

# =================================================================================================
else:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2").to("cuda").eval()
    HAS_LORA = False
    if A.adapter:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, os.path.join(REPO, A.adapter))
            model.eval(); HAS_LORA = True
        except Exception:
            print("[warn] adapter load failed:\n" + traceback.format_exc()[-600:], flush=True)

    def enc(sys, body, img, n_copies=1):
        msgs = [{"role": "system", "content": sys},
                {"role": "user", "content": [{"type": "image", "image": img,
                                              "max_pixels": MAXPX, "min_pixels": MINPX},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        e = proc(text=[text] * n_copies, images=igs * n_copies if igs else None,
                 videos=vids, return_tensors="pt", padding=True)
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in e.items()}

    def enc_multi(sys, bodies, img):
        msgs = [[{"role": "system", "content": sys},
                 {"role": "user", "content": [{"type": "image", "image": img,
                                               "max_pixels": MAXPX, "min_pixels": MINPX},
                                              {"type": "text", "text": b}]}] for b in bodies]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        igs, vids = [], None
        for m in msgs:
            a, b = process_vision_info(m); igs += a
        proc.tokenizer.padding_side = "left"
        e = proc(text=texts, images=igs, videos=vids, return_tensors="pt", padding=True)
        return {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in e.items()}

    IDLE = idle_w()
    print(f"[idle-with-model-resident] {IDLE:.1f} W  lora={HAS_LORA}", flush=True)

    def g1(i):
        q, img = ITEMS[i]; e = enc(SYS_GEN, f"Question: {q}", img)
        p = e["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=A.max_new, do_sample=False)
        return dict(prompt_tok=p, gen_tok=int(o.shape[1] - p), n_req=1)

    def g8(i):
        q, img = ITEMS[i]; e = enc(SYS_GEN, f"Question: {q}", img)
        p = e["input_ids"].shape[1]
        with torch.no_grad():
            o = model.generate(**e, max_new_tokens=A.max_new, do_sample=True,
                               temperature=A.temp, top_p=1.0, num_return_sequences=8)
        return dict(prompt_tok=p, gen_tok=int((o.shape[1] - p) * 8), n_req=8)

    def vk(K):
        def f(i):
            q, img = ITEMS[i]
            bodies = [f"Question: {q}\nProposed answer: {c}\n"
                      "Is the proposed answer correct? Answer Yes or No." for c in CANDS[:K]]
            e = enc_multi(SYS_VER, bodies, img) if K > 1 else enc(SYS_VER, bodies[0], img)
            with torch.no_grad():
                _ = model(**e).logits[:, -1]
            return dict(prompt_tok=int(e["input_ids"].shape[1]), gen_tok=K, n_req=K)
        return f

    run_leg("gen1", g1, NEED)
    run_leg("gen8_hf", g8, NEED)
    run_leg("ver1", vk(1), NEED)
    run_leg("ver8_batch", vk(8), NEED)
    meta = dict(phase="hf", idle_w=IDLE, gpu_names=GPU_NAMES, gpu_limit_w=GPU_LIMIT_W,
                model=A.model_path, adapter=A.adapter, lora_loaded=HAS_LORA,
                temp=A.temp, max_new=A.max_new, n=A.n, warmup=A.warmup, rep=A.rep,
                transformers=__import__("transformers").__version__)

FH.close()
json.dump(meta, open(os.path.join(REPO, A.outdir, f"{TAG}.meta.json"), "w"), indent=1)
print("DONE -> " + JL, flush=True)
