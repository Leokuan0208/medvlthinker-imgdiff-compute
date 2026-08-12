#!/usr/bin/env python3
"""measure_quant_footprint.py -- ATTACK 4, REPAIR of the quantised-cheap-side VRAM rows.

WHY THIS EXISTS.  src/cascade/measure_vram_levers.py scenario Q loaded all six
(scheme x cap) configurations in ONE process, one after another.  Its own numbers show that is
not safe: for a FIXED scheme the second configuration always reported MORE resident memory than
the first (int8wo 9.3942 -> 15.4865 GiB; int4wo 19.0781 -> 22.6715 GiB), and int4wo reported MORE
than the bf16 control (15.4464 GiB), which is impossible for a weight-only 4-bit scheme.  The
ordering is monotone in load order, i.e. the reading is contaminated by allocator state carried
across model loads, not a property of the scheme.  Those six rows are therefore RETRACTED and
replaced by this script, which loads exactly ONE model per PROCESS.

Same four quantities, same names, same units (GiB = bytes/1024**3), same deterministic item pool
(seed 42), same env as artifacts/vram_testtime_2026-08-11.json, so rows stay comparable:
  (a) weights_resident, (b) peak_allocated, (c) peak_reserved, (d) process footprint.

(a) IS READ TWICE for a quantised arm: once immediately after from_pretrained (the bf16 model),
and once after quantize_() followed by gc.collect() + empty_cache(), because torchao REPLACES the
weight tensors and the originals are only freed when Python drops the last reference.  Reading (a)
without that collection is exactly the contamination this script repairs.

*** THE STANDING FLOP CAVEAT. ***  int8/int4 WEIGHT-ONLY quantisation performs ZERO fewer
multiply-accumulates on sm80.  The weight is dequantised (or fed to a tinygemm kernel accumulating
in bf16) and the same bf16 GEMM runs.  MEMORY lever only, never a FLOP lever; reported separately.

  python3 src/cascade/measure_quant_footprint.py --scheme bf16
  python3 src/cascade/measure_quant_footprint.py --scheme int8wo
  python3 src/cascade/measure_quant_footprint.py --scheme int4wo
"""
import argparse, gc, glob, json, os, random, subprocess, threading, time

import numpy as np
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
GIB = 1024 ** 3
M7 = "lingshu-medical-mllm/Lingshu-7B"
BASE_PX = 1280 * 28 * 28
CAPS = [("medevalkit_default", 12845056, 16384), ("cap320", BASE_PX // 4, 320)]
MIN_PX = 4 * 28 * 28
MEDEVAL_MCQ_TAIL = "Answer with the option's letter from the given choices directly."

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["bf16", "int8wo", "int4wo"])
ap.add_argument("--arm", default="mcq", choices=["mcq", "open"],
                help="mcq = 7B direct MCQ. open = THE UNIFIED ARM: 8-sample generation + the LoRA "
                     "verifier scoring all 8, co-resident in one process (one set of base weights).")
ap.add_argument("--n_open", type=int, default=12)
ap.add_argument("--n_mcq", type=int, default=15)
ap.add_argument("--max_new", type=int, default=8)
ap.add_argument("--wait_mb", type=int, default=26000)
ap.add_argument("--wait_timeout_s", type=int, default=10800)
ap.add_argument("--out", default="results/cascade_methods/artifacts/_vram_levers_parts")
A = ap.parse_args()
OUT = os.path.join(ROOT, A.out)
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def free_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()
    return [int(t) - int(u) for u, t in (l.split(",") for l in out)]


t0 = time.time()
while True:                                          # GPU etiquette: wait, never kill
    f = free_mb()
    dev = max(range(len(f)), key=lambda i: f[i])
    if f[dev] >= A.wait_mb:
        print(f"[wait] GPU {dev} has {f[dev]} MB free (need {A.wait_mb}) -- proceeding", flush=True)
        break
    if time.time() - t0 > A.wait_timeout_s:
        raise SystemExit(f"no GPU with {A.wait_mb} MB free (free={f})")
    print(f"[wait] free={f}; sleeping 90 s", flush=True)
    time.sleep(90)
os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)

import pynvml                                                                  # noqa: E402
import torch                                                                   # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText            # noqa: E402
from qwen_vl_utils import process_vision_info                                  # noqa: E402

pynvml.nvmlInit()
H = pynvml.nvmlDeviceGetHandleByIndex(dev)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def board():
    return int(pynvml.nvmlDeviceGetMemoryInfo(H).used)


def gb(x):
    return round(x / GIB, 4)


# CUDA context, measured in THIS process before any model load: (d) = (c) + this.  The card is
# SHARED, so board-minus-baseline would charge another round's allocation to us.
b0 = board()
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
CTX = max(0, board() - b0 - torch.cuda.memory_reserved())
print(f"[ctx] cuda context = {gb(CTX)} GiB", flush=True)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.on, self.mx = True, 0

    def run(self):
        while self.on:
            self.mx = max(self.mx, board())
            time.sleep(0.02)

    def stop(self):
        self.on = False
        self.join(timeout=2.0)
        return max(self.mx, board())


def pool_mcq(n):
    """VERBATIM from src/cascade/measure_testtime_vram.py:pool_mcq (seed 42) so the rows compare."""
    import pandas as pd
    items = []
    df = pd.read_csv("/data/dan/dataset/medevalkit/PMC-VQA/test_2.csv")
    rng = random.Random(42)
    cand = []
    for i in rng.sample(range(len(df)), 300):
        r = df.iloc[i]
        for d in ("images", "figures"):
            p = f"/data/dan/dataset/medevalkit/PMC-VQA/{d}/{r['Figure_path']}"
            if os.path.exists(p):
                try:
                    w, h = Image.open(p).size
                except Exception:
                    break
                cand.append(dict(src="pmc_vqa_test_2", idx=int(r["index"]), imgs=[p], px=w * h,
                                 question=str(r["Question"]).strip(),
                                 options=[str(r[f"Choice {c}"]).strip() for c in "ABCD"]))
                break
    cand.sort(key=lambda x: x["px"])
    n_pmc = max(2, n // 2)
    k = n_pmc // 2
    items += cand[:k] + cand[-(n_pmc - k):]
    from datasets import Dataset
    arw = glob.glob("/data/dan/dataset/medevalkit/MedXpertQA/TsinghuaC3I___med_xpert_qa/MM/"
                    "**/med_xpert_qa-test.arrow", recursive=True)[0]
    mx = Dataset.from_file(arw)
    IMD = "/data/dan/dataset/medevalkit/MedXpertQA/images"
    mcand = []
    for j in range(len(mx)):
        r = mx[j]
        ps = [os.path.join(IMD, f) for f in (r["images"] or [])]
        ps = [p for p in ps if os.path.exists(p)]
        if not ps:
            continue
        try:
            px = sum(Image.open(p).size[0] * Image.open(p).size[1] for p in ps)
        except Exception:
            continue
        mcand.append(dict(src="medxpert_mm_test", idx=r["id"], imgs=ps, px=px,
                          question=str(r["question"]).strip(),
                          options=[f"{k2}: {v}" for k2, v in r["options"].items()]))
    mcand.sort(key=lambda x: x["px"])
    n_mx = n - len(items)
    m = len(mcand) // 2
    items += mcand[m:m + (n_mx - n_mx // 2)] + mcand[-(n_mx // 2):]
    return items[:n]


def pool_open(n):
    """VERBATIM from src/cascade/measure_testtime_vram.py:pool_open -- the three DEPLOYED open
    cells, smallest and largest image per cell, loaded the way cheapleg_score_open.py loads them."""
    import io
    import pandas as pd
    per = max(2, n // 3)
    out, sl = [], []
    for x in json.load(open("/data/dan/dataset/slake/test.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                try:
                    w, h = Image.open(ip).size
                except Exception:
                    continue
                sl.append(dict(src="slake_open", idx=x["qid"], img=ip, px=w * h,
                               question=x["question"], answer=str(x.get("answer", ""))))
    sl.sort(key=lambda z: z["px"])
    out += sl[:per // 2] + sl[-(per - per // 2):]
    for name, base in (("vqa_rad_open", "/data/dan/dataset/vqa_rad/data"),
                       ("pathvqa_open", "/data/dan/dataset/path_vqa/data")):
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(base + "/test-*.parquet"))],
                       ignore_index=True)
        c = []
        for i, r in df.iterrows():
            if str(r.get("answer")).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if not (isinstance(img, dict) and "bytes" in img):
                continue
            try:
                im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            except Exception:
                continue
            c.append(dict(src=name, idx=int(i), img=im, px=im.size[0] * im.size[1],
                          question=str(r.get("question")), answer=str(r.get("answer"))))
            if name == "pathvqa_open" and i >= 1500:
                break
        c.sort(key=lambda z: z["px"])
        out += c[:per // 2] + c[-(per - per // 2):]
    return out[:n]


# deployed open-text constants, VERBATIM from src/cascade/measure_testtime_vram.py
OPEN_GEN_MAXPX, OPEN_GEN_MINPX = (1280 * 28 * 28) // 4, 4 * 28 * 28
VERIF_MAXPX, VERIF_MINPX = 1280 * 28 * 28, 4 * 28 * 28
VERIFIER = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
SYS_OPEN = ("You are an expert medical image analyst. Answer the question with a short, specific "
            "phrase. Do not explain.")
SYS_VERIF = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
             "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

items = pool_mcq(A.n_mcq) if A.arm == "mcq" else pool_open(A.n_open)
proc = AutoProcessor.from_pretrained(M7)
tl = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    M7, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
torch.cuda.synchronize()
a_bf16 = torch.cuda.memory_allocated()
qinfo = dict(scheme=A.scheme, library=None, applied_to=None)
if A.scheme != "bf16":
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    import torchao
    target = model.model.language_model if hasattr(model.model, "language_model") else model.model
    n_lin = sum(1 for m in target.modules() if m.__class__.__name__ == "Linear")
    quantize_(target, Int8WeightOnlyConfig() if A.scheme == "int8wo" else Int4WeightOnlyConfig())
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    qinfo = dict(scheme=A.scheme, library=f"torchao {torchao.__version__}",
                 applied_to="language_model only (vision tower left bf16)",
                 n_linear_modules_targeted=int(n_lin),
                 flop_note="weight-only quantisation: ZERO MAC reduction on sm80 (A100). "
                           "MEMORY lever only; FLOPs are unchanged BY CONSTRUCTION.")
adapter_info = None
if A.arm == "open":
    # THE UNIFIED ARM: attach the LoRA verifier to the SAME (already quantised) base weights, so
    # generator and verifier share one set of weights.  Order matters and is deliberate: quantise
    # FIRST, attach the adapter SECOND, so the small LoRA tensors stay bf16 and are not quantised.
    try:
        from peft import PeftModel
        pre = torch.cuda.memory_allocated()
        model = PeftModel.from_pretrained(model, VERIFIER)
        model.eval()
        torch.cuda.synchronize()
        n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
        n_vis = sum(1 for n, _ in model.named_parameters() if "lora_" in n and "visual." in n)
        adapter_info = dict(adapter=VERIFIER, ok=True,
                            marginal_resident_gib=gb(torch.cuda.memory_allocated() - pre),
                            n_lora_params=n_lora, n_lora_params_on_visual=n_vis,
                            note="generator and verifier SHARE one base -- the arm costs one 7B, "
                                 "not two. HF only: vLLM 0.9.0.1 drops all 192 visual.* modules.")
    except Exception as e:
        adapter_info = dict(adapter=VERIFIER, ok=False,
                            error=f"{type(e).__name__}: {str(e)[:300]}",
                            consequence="the unified arm could NOT be built at this precision; "
                                        "its footprint is NOT measured for this scheme.")
        print(f"[adapter] FAILED: {adapter_info['error']}", flush=True)
torch.cuda.synchronize()
a_final = torch.cuda.memory_allocated()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
load_info = dict(model=M7, scheme=A.scheme, load_s=round(time.time() - tl, 1), dtype="bfloat16",
                 attn_implementation="flash_attention_2", quantisation=qinfo,
                 a_weights_resident_gib=gb(a_final),
                 a_before_quantisation_gib=gb(a_bf16),
                 a_reading_protocol=("memory_allocated() after gc.collect()+empty_cache() so the "
                                     "replaced bf16 tensors are not still counted"),
                 n_params_logical=int(sum(p.numel() for p in model.parameters())),
                 n_params_note=("logical parameter count is unchanged by weight-only quantisation "
                                "-- only the STORAGE changes, which is why (a) is the number to "
                                "read and n_params is not"),
                 cuda_context_gib=gb(CTX), gpu_physical_index=dev, arm=A.arm,
                 lora_verifier=adapter_info)
print(f"[load] {A.scheme}: (a)={load_info['a_weights_resident_gib']} GiB "
      f"(bf16 before quantisation {load_info['a_before_quantisation_gib']})", flush=True)


def run_cap(cap, maxpx, budget):
    rows = []
    for it in items:
        try:
            msgs = [{"role": "user", "content":
                     [{"type": "image", "image": p, "max_pixels": maxpx, "min_pixels": MIN_PX}
                      for p in it["imgs"]]
                     + [{"type": "text", "text": "\nQuestion: " + it["question"] + "\nOptions: \n"
                         + "\n".join(it["options"]) + "\n" + MEDEVAL_MCQ_TAIL}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ims, vids = process_vision_info(msgs)
            enc = proc(text=[text], images=ims, videos=vids, return_tensors="pt",
                       padding=True).to("cuda")
            nin = int(enc["input_ids"].shape[1])
            nvis = int(enc["image_grid_thw"].prod(dim=-1).sum().item()) \
                if enc.get("image_grid_thw") is not None else 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            s = Sampler(); s.start(); t1 = time.time()
            with torch.inference_mode():
                o = model.generate(**enc, max_new_tokens=A.max_new, do_sample=False)
            torch.cuda.synchronize()
            dt = time.time() - t1
            mb = s.stop()
            c = torch.cuda.max_memory_reserved()
            rows.append(dict(b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                             c_peak_reserved_gib=gb(c),
                             d_process_footprint_gib=gb(c + CTX),
                             d_method="c_peak_reserved + measured cuda context (card is SHARED)",
                             d_board_used_gib=gb(mb),
                             wall_s=round(dt, 3), src=it["src"], idx=it["idx"],
                             image_pixels=it["px"], n_images=len(it["imgs"]),
                             input_tokens=nin, vision_tokens=nvis,
                             gen_tokens=int(o.shape[1] - nin)))
            del enc, o
        except Exception as e:                                    # per-item error guard
            rows.append(dict(error=f"{type(e).__name__}: {str(e)[:200]}",
                             src=it["src"], idx=it["idx"]))
            torch.cuda.empty_cache()
    ok = [r for r in rows if "error" not in r]

    def agg(k):
        v = [r[k] for r in ok]
        return dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                    min=round(float(np.min(v)), 4)) if v else None
    drv = max(ok, key=lambda r: r["b_peak_allocated_gib"]) if ok else {}
    return dict(meta=dict(scenario=f"Lingshu-7B [{A.scheme}] direct MCQ, batch 1, cap={cap}",
                          quant_scheme=A.scheme, max_pixels=maxpx, vision_token_budget=budget,
                          batch_size=1, tp=1, max_new_tokens=A.max_new, load=load_info),
                n=len(ok), n_failed=len(rows) - len(ok),
                b_peak_allocated_gib=agg("b_peak_allocated_gib"),
                c_peak_reserved_gib=agg("c_peak_reserved_gib"),
                d_process_footprint_gib=agg("d_process_footprint_gib"),
                d_board_used_gib=agg("d_board_used_gib"),
                input_tokens=agg("input_tokens"), vision_tokens=agg("vision_tokens"),
                gen_tokens=agg("gen_tokens"), wall_s=agg("wall_s"),
                peak_driver=dict(source=drv.get("src"), idx=drv.get("idx"),
                                 image_pixels=drv.get("image_pixels"),
                                 vision_tokens=drv.get("vision_tokens"),
                                 b_peak_allocated_gib=drv.get("b_peak_allocated_gib")),
                rows=rows)


def run_open(gen_px, verif_px, tag):
    """THE UNIFIED ARM end to end in one process: generate 8 samples with the adapter DISABLED
    (base behaviour, as deployed), then score all 8 with the adapter ENABLED.  Peak is over the
    WHOLE pipeline -- exactly scenario S4 of artifacts/vram_testtime_2026-08-11.json."""
    rows = []
    for it in items:
        try:
            gmsg = [{"role": "system", "content": SYS_OPEN},
                    {"role": "user", "content": [
                        {"type": "image", "image": it["img"], "max_pixels": gen_px,
                         "min_pixels": OPEN_GEN_MINPX},
                        {"type": "text", "text": it["question"]}]}]
            text = proc.apply_chat_template(gmsg, tokenize=False, add_generation_prompt=True)
            ims, vids = process_vision_info(gmsg)
            enc = proc(text=[text], images=ims, videos=vids, return_tensors="pt",
                       padding=True).to("cuda")
            nin = int(enc["input_ids"].shape[1])
            nvis = int(enc["image_grid_thw"].prod(dim=-1).sum().item()) \
                if enc.get("image_grid_thw") is not None else 0
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            s = Sampler(); s.start(); t1 = time.time()
            with model.disable_adapter():
                with torch.inference_mode():
                    o = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.7,
                                       top_p=1.0, num_return_sequences=8)
            torch.cuda.synchronize()
            gen_peak = torch.cuda.max_memory_allocated()
            cands = [proc.tokenizer.decode(x[nin:], skip_special_tokens=True).strip() for x in o]
            gtok = int(o.shape[1] - nin)
            del enc, o
            for cand in cands:
                vmsg = [{"role": "system", "content": SYS_VERIF},
                        {"role": "user", "content": [
                            {"type": "image", "image": it["img"], "max_pixels": verif_px,
                             "min_pixels": VERIF_MINPX},
                            {"type": "text", "text":
                             f"Question: {it['question']}\nProposed answer: {cand}\n"
                             f"Is the proposed answer correct? Answer Yes or No."}]}]
                t2 = proc.apply_chat_template(vmsg, tokenize=False, add_generation_prompt=True)
                i2, v2 = process_vision_info(vmsg)
                e2 = proc(text=[t2], images=i2, videos=v2, return_tensors="pt",
                          padding=True).to("cuda")
                with torch.inference_mode():
                    _ = model(**e2).logits[0, -1]
                del e2
            torch.cuda.synchronize()
            dt = time.time() - t1
            mb = s.stop()
            c = torch.cuda.max_memory_reserved()
            rows.append(dict(b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                             b_peak_allocated_generator_phase_gib=gb(gen_peak),
                             c_peak_reserved_gib=gb(c),
                             d_process_footprint_gib=gb(c + CTX),
                             d_method="c_peak_reserved + measured cuda context (card is SHARED)",
                             d_board_used_gib=gb(mb), wall_s=round(dt, 3),
                             src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=1,
                             input_tokens=nin, vision_tokens=nvis, gen_tokens=gtok,
                             n_samples=8, n_verifier_passes=len(cands)))
        except Exception as e:                                    # per-item error guard
            rows.append(dict(error=f"{type(e).__name__}: {str(e)[:200]}",
                             src=it["src"], idx=it["idx"]))
            torch.cuda.empty_cache()
    ok = [r for r in rows if "error" not in r]

    def agg(k):
        v = [r[k] for r in ok if k in r]
        return dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                    min=round(float(np.min(v)), 4)) if v else None
    drv = max(ok, key=lambda r: r["b_peak_allocated_gib"]) if ok else {}
    return dict(meta=dict(
        scenario=f"THE UNIFIED OPEN-TEXT ARM [{A.scheme}] -- Lingshu-7B generates 8 samples "
                 f"(temp 0.7, max_new_tokens 64, adapter DISABLED) then the LoRA verifier scores "
                 f"all 8 (adapter ENABLED), one process, {tag}",
        quant_scheme=A.scheme, generator_max_pixels=gen_px, verifier_max_pixels=verif_px,
        batch_size="1 question, num_return_sequences=8", tp=1, n_samples=8, load=load_info,
        comparable_to="scenario S4 of artifacts/vram_testtime_2026-08-11.json and the R2 block"),
        n=len(ok), n_failed=len(rows) - len(ok),
        b_peak_allocated_gib=agg("b_peak_allocated_gib"),
        b_peak_allocated_generator_phase_gib=agg("b_peak_allocated_generator_phase_gib"),
        c_peak_reserved_gib=agg("c_peak_reserved_gib"),
        d_process_footprint_gib=agg("d_process_footprint_gib"),
        d_board_used_gib=agg("d_board_used_gib"),
        input_tokens=agg("input_tokens"), vision_tokens=agg("vision_tokens"),
        gen_tokens=agg("gen_tokens"), wall_s=agg("wall_s"),
        peak_driver=dict(source=drv.get("src"), idx=drv.get("idx"),
                         image_pixels=drv.get("image_pixels"),
                         vision_tokens=drv.get("vision_tokens"),
                         b_peak_allocated_gib=drv.get("b_peak_allocated_gib")),
        rows=rows)


res = {}
suffix = "" if A.arm == "mcq" else "_open"
if A.arm == "mcq":
    for cap, maxpx, budget in CAPS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        res[f"{A.scheme}__{cap}"] = run_cap(cap, maxpx, budget)
        r = res[f"{A.scheme}__{cap}"]
        print(f"[{A.scheme} {cap}] n={r['n']} (a)={load_info['a_weights_resident_gib']} "
              f"(b)peak={r['b_peak_allocated_gib']['peak']} "
              f"(c)peak={r['c_peak_reserved_gib']['peak']} "
              f"(d)peak={r['d_process_footprint_gib']['peak']}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"Q2_{A.scheme}.json"), "w"), indent=1)
elif adapter_info and adapter_info.get("ok"):
    # deployed geometry (generator cap320 / verifier 1,003,520) and an all-cap320 variant
    for tag, gpx, vpx in (("deployed_gen_cap320_verif_1003520", OPEN_GEN_MAXPX, VERIF_MAXPX),
                          ("both_cap320", OPEN_GEN_MAXPX, OPEN_GEN_MAXPX)):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        res[f"{A.scheme}__open__{tag}"] = run_open(gpx, vpx, tag)
        r = res[f"{A.scheme}__open__{tag}"]
        print(f"[{A.scheme} open {tag}] n={r['n']} (a)={load_info['a_weights_resident_gib']} "
              f"(b)peak={r['b_peak_allocated_gib']['peak']} "
              f"(c)peak={r['c_peak_reserved_gib']['peak']} "
              f"(d)peak={r['d_process_footprint_gib']['peak']}", flush=True)
        json.dump(res, open(os.path.join(OUT, f"Q2_{A.scheme}_open.json"), "w"), indent=1)
else:
    res["_adapter_failed"] = dict(load=load_info)
    json.dump(res, open(os.path.join(OUT, f"Q2_{A.scheme}_open.json"), "w"), indent=1)
print("wrote", os.path.join(OUT, f"Q2_{A.scheme}{suffix}.json"), flush=True)
