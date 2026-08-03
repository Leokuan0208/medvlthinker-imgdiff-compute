#!/usr/bin/env python3
"""
bestofn_measure_batch8.py -- MEASURE the real batch-1-serving cost of PARALLEL best-of-8.

WHY (the contradiction this closes).
  The open-text best-of-N arm is costed with TWO MUTUALLY INCONSISTENT models:
    * LATENCY  (src/cascade_methods/latency_reexamination.py::lat_parallel, integrated_method.py::BO8):
        "the 8 generations issue as ONE batched forward, the 8 verifies as one batched forward, so N
         drops out"  ->  BO8 = GEN7 + VER7 = 347.1 + 175.5 = 522.6 ms.   ASSERTED, never measured.
    * ENERGY   (src/cascade_methods/latency_reexamination.py::energy,
                macro_headline_clean_verifier.py::BO8_J = 8*(GEN7_J+VER7_J)):
        "8 gens + 8 verifies each cost their full batch-1 energy" -> 8*(45.8+25.3) = 568.8 J.
  568.8 J in 0.5226 s = 1088.5 W on ONE A100 80GB PCIe whose power limit is 300 W. Impossible.
  Exactly one of the two is wrong (or both). This script measures the truth.

WHAT IT MEASURES (same harness as the canonical constants: src/cascade_methods/open_measure_latency_energy.py
  -- HF transformers, batch-1 request, cap320 (max_pixels = 1280*28*28//4), REAL vqa_rad test images,
  NVML power integrated over the timed call across VISIBLE GPUs only, warmup iters discarded):
    gen1     : one greedy generation                        (reproduces GEN7 = 347.1 ms / 45.8 J)
    gen8     : 8 samples generated as ONE batch             (the "parallel best-of-N" the claim describes)
               num_return_sequences=8, do_sample=True, temperature=0.7 -- the K=8 recipe the project used.
    verify1  : one verifier forward on one candidate        (reproduces VER7 = 175.5 ms / 25.3 J)
    verify8  : ONE batched verifier forward over 8 candidates (the best-of-8 selection cost)
  gen8 and verify8 are timed on the SAME item back-to-back, and verify8 scores the 8 candidates that
  gen8 actually produced -> bo8_total per item = gen8 + verify8 is a true end-to-end wall clock.

MODEL: Lingshu-7B (the open-text cheap leg). Verifier = same weights + the trained LoRA
  ckpts/train/lora_verifier_pooled4 (--adapter), matching how VER7 was measured.

Launch from repo root, one GPU pinned:
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade_methods/bestofn_measure_batch8.py \
      --model_path lingshu-medical-mllm/Lingshu-7B --adapter ckpts/train/lora_verifier_pooled4 \
      --n 20 --warmup 3 --out results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json
NOTE: --adapter is applied ONLY to the verify passes (PeftModel enable/disable), so gen and verify use
  exactly the weights each is supposed to use, in one process, on one warm GPU.
"""
import argparse, os, io, glob, time, json, threading
import numpy as np, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from PIL import Image
import pynvml

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--adapter", default="ckpts/train/lora_verifier_pooled4")
ap.add_argument("--tag", default="lingshu7b")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max_new", type=int, default=32)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--out", default="results/cascade_methods/artifacts/bestofn_latency_energy_2026-08-03.json")
A = ap.parse_args()

MAXPX, MINPX = 1280 * 28 * 28 // 4, 4 * 28 * 28          # cap320, identical to open_measure_latency_energy.py
DEV = "cuda"
SYS_GEN = ("You are an expert medical image analyst. Answer the question with a short, specific phrase. "
           "Do not explain.")
SYS_VER = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
           "proposed answer is correct. Respond with only 'Yes' or 'No'.")

# ---------------- NVML power sampler (verbatim policy from open_measure_latency_energy.py) ----------------
pynvml.nvmlInit()
_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
_toks = [x.strip() for x in _vis.split(",") if x.strip() != ""]
_idxs = [int(t) for t in _toks] if (_toks and all(t.isdigit() for t in _toks)) \
        else list(range(pynvml.nvmlDeviceGetCount()))
HS = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in _idxs]
NG = len(HS)
GPU_NAMES = [pynvml.nvmlDeviceGetName(h) for h in HS]
GPU_NAMES = [g.decode() if isinstance(g, bytes) else g for g in GPU_NAMES]
GPU_LIMIT_W = [pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0 for h in HS]


_W_LAST = [0.0]


def watts():
    """Sum of instantaneous draw over visible GPUs. NVML can transiently return NotSupported on
    A100 PCIe; hold the last good reading rather than letting the sampler thread die (which would
    silently under-count energy for that call)."""
    try:
        w = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in HS) / 1000.0
        _W_LAST[0] = w
        return w
    except pynvml.NVMLError:
        return _W_LAST[0]


class E:
    """Integrate W over the timed call (10 ms sampling), same as the canonical harness."""
    def __init__(s): s.go = False; s.j = 0.0
    def __enter__(s):
        s.j = 0.0; s.go = True
        def loop():
            last = time.time()
            while s.go:
                time.sleep(0.01); now = time.time(); s.j += watts() * (now - last); last = now
        s.th = threading.Thread(target=loop); s.th.start(); return s
    def __exit__(s, *a): s.go = False; s.th.join()


def idle_watts(sec=3.0):
    t0 = time.time(); acc = 0.0; last = t0
    while time.time() - t0 < sec:
        time.sleep(0.02); now = time.time(); acc += watts() * (now - last); last = now
    return acc / (last - t0)


# ---------------- real vqa_rad items (identical selection rule to the canonical harness) ----------------
import pandas as pd
df_files = sorted(glob.glob("/data/dan/dataset/vqa_rad/data/test-*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in df_files], ignore_index=True)
items = []
for _, r in df.iterrows():
    q = r.get("question"); a = r.get("answer")
    if q is None and "conversations" in r:
        conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
    if str(a).strip().lower() in ("yes", "no"):          # open-ended items only
        continue
    img = r["image"]
    if isinstance(img, dict) and "bytes" in img:
        items.append((str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"), str(a)))
    if len(items) >= A.n + A.warmup:
        break
print(f"{A.tag}: {len(items)} items, K={A.k}, visible GPUs={NG} {GPU_NAMES} limit={GPU_LIMIT_W} W", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    attn_implementation="flash_attention_2").to(DEV)
model.eval()
HAS_ADAPTER = False
if A.adapter and os.path.isdir(os.path.expanduser(A.adapter)):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.expanduser(A.adapter))
    model.eval(); HAS_ADAPTER = True
    print(f"verifier LoRA loaded: {A.adapter}", flush=True)
else:
    print(f"WARNING: adapter not found at {A.adapter}; verify measured on BASE weights", flush=True)

if proc.tokenizer.padding_side != "left":
    proc.tokenizer.padding_side = "left"                  # required for batched decoder-only generation


def enc_gen(q, img):
    msgs = [{"role": "system", "content": SYS_GEN},
            {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                                         {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    e = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True)
    return {k: (v.to(DEV) if hasattr(v, 'to') else v) for k, v in e.items()}


def enc_ver(q, img, answers):
    """One batch of len(answers) verifier prompts over the SAME image."""
    texts, all_igs = [], []
    for ans in answers:
        body = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        msgs = [{"role": "system", "content": SYS_VER},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                                             {"type": "text", "text": body}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        igs, _ = process_vision_info(msgs); all_igs += igs
    e = proc(text=texts, images=all_igs, videos=None, return_tensors="pt", padding=True)
    return {k: (v.to(DEV) if hasattr(v, 'to') else v) for k, v in e.items()}


YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]


def set_adapter(on):
    if not HAS_ADAPTER:
        return
    (model.enable_adapter_layers if on else model.disable_adapter_layers)()


def timed(fn):
    torch.cuda.synchronize()
    with E() as e:
        t0 = time.time(); out = fn(); torch.cuda.synchronize(); dt = time.time() - t0
    return dt, e.j, out


REC = {k: [] for k in ("gen1", "gen8", "verify1", "verify8", "bo8_total")}
GT = {k: [] for k in ("gen1", "gen8")}
PT = []

W_IDLE = idle_watts(3.0)
print(f"idle draw over visible GPUs (model resident): {W_IDLE:.1f} W", flush=True)

for i, (q, img, gold) in enumerate(items):
    keep = i >= A.warmup
    eg = enc_gen(q, img); ptok = eg["input_ids"].shape[1]

    # ---- (a) single greedy generation ------------------------------------------------------------
    set_adapter(False)
    dt1, j1, o1 = timed(lambda: model.generate(**eg, max_new_tokens=A.max_new, do_sample=False))
    g1 = o1.shape[1] - ptok
    greedy_txt = proc.tokenizer.decode(o1[0, ptok:], skip_special_tokens=True).strip()

    # ---- (b) 8 samples as ONE batch (parallel best-of-8 generation) --------------------------------
    dt8, j8, o8 = timed(lambda: model.generate(**eg, max_new_tokens=A.max_new, do_sample=True,
                                               temperature=A.temp, top_p=0.95,
                                               num_return_sequences=A.k))
    g8 = o8.shape[1] - ptok
    cands = [proc.tokenizer.decode(o8[r, ptok:], skip_special_tokens=True).strip() for r in range(o8.shape[0])]
    cands = [c if c else greedy_txt for c in cands]

    # ---- (c) verifier: one candidate, and all 8 in ONE batched forward -----------------------------
    set_adapter(True)
    ev1 = enc_ver(q, img, cands[:1])
    dtv1, jv1, _ = timed(lambda: model(**ev1).logits[:, -1, [YES, NO]])
    ev8 = enc_ver(q, img, cands)
    dtv8, jv8, _ = timed(lambda: model(**ev8).logits[:, -1, [YES, NO]])

    if keep:
        REC["gen1"].append((dt1, j1)); GT["gen1"].append(g1)
        REC["gen8"].append((dt8, j8)); GT["gen8"].append(g8)
        REC["verify1"].append((dtv1, jv1))
        REC["verify8"].append((dtv8, jv8))
        REC["bo8_total"].append((dt8 + dtv8, j8 + jv8))
        PT.append(ptok)
    print(f"  [{i}]{'' if keep else ' (warmup)'} gen1={dt1*1000:.0f}ms/{j1:.1f}J  "
          f"gen8={dt8*1000:.0f}ms/{j8:.1f}J  ver1={dtv1*1000:.0f}ms/{jv1:.1f}J  "
          f"ver8={dtv8*1000:.0f}ms/{jv8:.1f}J  ptok={ptok} gtok1={g1} gtok8={g8}", flush=True)
    # free between items (OUTSIDE every timed region, so it cannot inflate a measurement)
    del eg, ev1, ev8, o1, o8
    torch.cuda.empty_cache()


def summ(key):
    lat = np.array([r[0] for r in REC[key]]) * 1000.0
    enj = np.array([r[1] for r in REC[key]])
    return dict(n=int(lat.size),
                lat_ms_mean=round(float(lat.mean()), 1), lat_ms_median=round(float(np.median(lat)), 1),
                lat_ms_sd=round(float(lat.std(ddof=1)), 1),
                lat_ms_p10=round(float(np.percentile(lat, 10)), 1),
                lat_ms_p90=round(float(np.percentile(lat, 90)), 1),
                lat_ms_min=round(float(lat.min()), 1), lat_ms_max=round(float(lat.max()), 1),
                energy_j_mean=round(float(enj.mean()), 2), energy_j_median=round(float(np.median(enj)), 2),
                energy_j_sd=round(float(enj.std(ddof=1)), 2),
                mean_power_w=round(float(enj.sum() / (lat.sum() / 1000.0)), 1))


out = dict(
    tag=A.tag, model=A.model_path, adapter=(A.adapter if HAS_ADAPTER else None), k=A.k,
    harness=("HF batch-1 request, cap320 (max_pixels=1280*28*28//4), real vqa_rad open items, "
             "NVML integrated over visible GPUs, flash_attention_2, bf16; same recipe as "
             "src/cascade_methods/open_measure_latency_energy.py"),
    gpus=NG, gpu_names=GPU_NAMES, gpu_power_limit_w=GPU_LIMIT_W,
    idle_w_model_resident=round(W_IDLE, 1),
    n_kept=len(PT), warmup=A.warmup, max_new=A.max_new, temperature=A.temp,
    prefill_tok_mean=round(float(np.mean(PT)), 1),
    gen_tok_mean={k: round(float(np.mean(v)), 2) for k, v in GT.items()},
    measured={k: summ(k) for k in REC},
    date="2026-08-03",
)
print("RESULT " + json.dumps(out["measured"]), flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
with open(A.out, "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote " + A.out, flush=True)
