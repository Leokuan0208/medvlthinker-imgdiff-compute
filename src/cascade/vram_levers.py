#!/usr/bin/env python3
"""vram_levers.py -- ATTACK 4: the two VRAM levers (image RESOLUTION and WEIGHT QUANTIZATION),
plus the two footprints that artifacts/vram_testtime_2026-08-11.json explicitly left unmeasured
(7B+32B CO-RESIDENCY, and the smallest configuration the 7B-side pipeline runs in at all).

WHY THIS EXISTS.  vram_testtime_2026-08-11.json measured only the DEPLOYED operating points and its
key_findings[2] says outright: "peak VRAM is set by vision-token count ... A resolution cap is the
direct VRAM lever and it was never characterised".  Its not_measured list names four gaps: the
resolution-cap ablation, INT4/quantized weights, 7B+32B co-residency, and batch>1.  This file closes
the first three.  Batch>1 stays out of scope (the deployed cascade is batch-1 serving).

*** EVERY CONVENTION IS INHERITED VERBATIM FROM THE 2026-08-11 ARTIFACT SO THE ROWS ARE COMPARABLE ***
  (a) weights_resident   torch.cuda.memory_allocated() after load + synchronize.  Parameters only.
  (b) peak_allocated     torch.cuda.max_memory_allocated(), reset before EVERY item.  Live tensors.
  (c) peak_reserved      torch.cuda.max_memory_reserved(), reset before every item.  (b)+fragmentation.
  (d) process_footprint  max over a 20 ms NVML sampler during the item.  WHAT A DEPLOYER PROVISIONS.
                         NVML per-process returns 0 in this container (HOST pids), so the substitute
                         is `board used - pre-run baseline`; every row keeps the raw board reading.
  units: GiB = bytes/1024**3.  batch_size 1, tp 1, bfloat16 unless a row says otherwise, HF
  transformers with flash_attention_2 -- NEVER vLLM (it reserves a pool, and it drops visual LoRA).

THE ITEM POOLS ARE COPIED VERBATIM from src/cascade/measure_testtime_vram.py (pool_mcq / pool_open,
seed 42) *on purpose*: identical items are the only way a row here is comparable to a row there, and
that comparison is this file's NULL TEST (--part null).  measure_testtime_vram.py is the 2026-08-11
artifact of record and is not modified.

CAP LADDER.  "capN" follows the repo's own naming: N * 28 * 28 pixels, i.e. N pre-merge vision
patches, i.e. N/4 LLM tokens after the 2x2 merge.  The three DEPLOYED points are on the ladder:
  cap16384 = 12,845,056  MedEvalKit default (the MCQ arm as evaluated)  [qwen_vl_utils default]
  cap1280  =  1,003,520  the deployed open-text VERIFIER   (cheapleg_score_open.py:31)
  cap320   =    250,880  the deployed open-text GENERATOR  (run_openvqa.py --cap cap320)

Usage (run from the repo root, ONE GPU pinned per process):
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/cascade/vram_levers.py --part res7b
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/cascade/vram_levers.py --part res32b
  ... --part quant | --part cores | --part smallest
"""
import argparse, glob, io, json, os, random, sys, threading, time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
GIB = 1024 ** 3

M7 = "lingshu-medical-mllm/Lingshu-7B"
M32 = "lingshu-medical-mllm/Lingshu-32B"
VERIFIER = "ckpts/train/lora_verifier_disjoint"

MEDEVAL_MAXPX, MEDEVAL_MINPX = 12845056, 3136
OPEN_GEN_MAXPX, OPEN_GEN_MINPX = (1280 * 28 * 28) // 4, 4 * 28 * 28
VERIF_MAXPX, VERIF_MINPX = 1280 * 28 * 28, 4 * 28 * 28

MEDEVAL_MCQ_TAIL = "Answer with the option's letter from the given choices directly."
SYS_OPEN = ("You are an expert medical image analyst. Answer the question with a short, specific "
            "phrase. Do not explain.")
SYS_VERIF = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
             "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

PATCH = 28 * 28
# the cap ladder, in PATCHES (N).  N*784 = max_pixels.  Deployed points marked in the artifact.
CAPS = [16384, 4096, 1280, 320, 80, 20]

ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True,
                choices=["null", "res7b", "res32b", "resopen", "quant", "cores", "smallest"])
ap.add_argument("--n_mcq", type=int, default=15)
ap.add_argument("--n_open", type=int, default=12)
ap.add_argument("--mcq_max_new", type=int, default=256)
ap.add_argument("--caps", default="")
ap.add_argument("--quant_arm", default="",
                choices=["", "bf16_control", "int8", "nf4", "nf4_skipvisual"],
                help="--part quant: which single arm to run in THIS process (one model per process)")
ap.add_argument("--out", default="results/cascade_methods/artifacts/_vram_levers_parts")
ap.add_argument("--suffix", default="")
A = ap.parse_args()
if A.caps:
    CAPS = [int(x) for x in A.caps.split(",")]
OUT = os.path.join(ROOT, A.out)
os.makedirs(OUT, exist_ok=True)

import pynvml
pynvml.nvmlInit()
DEV_IDX = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
H = pynvml.nvmlDeviceGetHandleByIndex(0)          # after CUDA_VISIBLE_DEVICES the handle index is 0
                                                  # only if the var is a single id; assert below.
_vis = (os.environ.get("CUDA_VISIBLE_DEVICES") or "")
if _vis and "," not in _vis:
    H = pynvml.nvmlDeviceGetHandleByIndex(int(_vis))
PID = os.getpid()
TOTAL_BYTES = int(pynvml.nvmlDeviceGetMemoryInfo(H).total)


def nvml_proc_bytes():
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(H):
            if p.pid == PID:
                return int(p.usedGpuMemory or 0)
    except Exception:
        pass
    return 0


def nvml_board_bytes():
    return int(pynvml.nvmlDeviceGetMemoryInfo(H).used)


BASELINE_BOARD = nvml_board_bytes()
print(f"[baseline] board used before load = {BASELINE_BOARD/GIB:.4f} GiB on cuda:{_vis or 0}", flush=True)


def d_footprint(max_proc, max_board):
    if max_proc > 0:
        return max_proc, "nvml_per_process_usedGpuMemory"
    return max(0, max_board - BASELINE_BOARD), "nvml_board_used_minus_prerun_baseline"


class MemSampler(threading.Thread):
    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval, self.on = interval, True
        self.max_proc = self.max_board = 0

    def run(self):
        while self.on:
            self.max_proc = max(self.max_proc, nvml_proc_bytes())
            self.max_board = max(self.max_board, nvml_board_bytes())
            time.sleep(self.interval)

    def stop(self):
        self.on = False
        self.join(timeout=2.0)
        self.max_proc = max(self.max_proc, nvml_proc_bytes())
        self.max_board = max(self.max_board, nvml_board_bytes())
        return self.max_proc, self.max_board


def gb(x):
    return round(x / GIB, 4)


# ------------------------------------------------------- item pools (VERBATIM from the 08-11 script)
def pool_mcq(n):
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
                q = str(r["Question"]).strip()
                opts = [str(r[f"Choice {c}"]).strip() for c in "ABCD"]
                cand.append(dict(src="pmc_vqa_test_2", idx=int(r["index"]), imgs=[p], px=w * h,
                                 question=q, options=opts))
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
    import pandas as pd
    per = max(2, n // 3)
    out = []
    sl = []
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
            a = r.get("answer")
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if not (isinstance(img, dict) and "bytes" in img):
                continue
            try:
                im = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            except Exception:
                continue
            c.append(dict(src=name, idx=int(i), img=im, px=im.size[0] * im.size[1],
                          question=str(r.get("question")), answer=str(a)))
            if name == "pathvqa_open" and i >= 1500:
                break
        c.sort(key=lambda z: z["px"])
        out += c[:per // 2] + c[-(per - per // 2):]
    return out[:n]


# ------------------------------------------------------------------------------ model / measurement
from transformers import AutoProcessor, AutoModelForImageTextToText            # noqa: E402
from qwen_vl_utils import process_vision_info                                  # noqa: E402


def load(path, tag, quant=None, skip_visual=False):
    """quant in {None,'int8','nf4'}.  skip_visual keeps the ViT in bf16 (llm_int8_skip_modules)."""
    t0 = time.time()
    pre_board = nvml_board_bytes()
    kw = dict(torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    qinfo = None
    if quant:
        from transformers import BitsAndBytesConfig
        skip = ["lm_head"] + (["visual"] if skip_visual else [])
        if quant == "int8":
            qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip,
                                    llm_int8_threshold=6.0)
        else:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True,
                                    llm_int8_skip_modules=skip)
        kw["quantization_config"] = qc
        kw["device_map"] = {"": 0}
        qinfo = dict(scheme=quant, skip_modules=skip, double_quant=(quant == "nf4"),
                     compute_dtype="bfloat16", library="bitsandbytes 0.50.0")
        m = AutoModelForImageTextToText.from_pretrained(path, **kw).eval()
    else:
        m = AutoModelForImageTextToText.from_pretrained(path, **kw).to("cuda").eval()
    torch.cuda.synchronize()
    nq = nlin = 0
    for mod in m.modules():
        cn = type(mod).__name__
        if cn in ("Linear8bitLt", "Linear4bit", "LinearNF4"):
            nq += 1
        elif cn == "Linear":
            nlin += 1
    info = dict(model=path, tag=tag, load_s=round(time.time() - t0, 1),
                dtype="bfloat16", attn_implementation="flash_attention_2", quantization=qinfo,
                n_quantized_linear=nq, n_bf16_linear=nlin,
                a_weights_resident_gib=gb(torch.cuda.memory_allocated()),
                weights_reserved_after_load_gib=gb(torch.cuda.memory_reserved()),
                nvml_board_after_load_gib=gb(nvml_board_bytes()),
                nvml_board_before_load_gib=gb(pre_board),
                n_params=int(sum(p.numel() for p in m.parameters())))
    print(f"[load] {tag}: (a)={info['a_weights_resident_gib']} GiB  quantized_linear={nq} "
          f"bf16_linear={nlin}  ({info['load_s']}s)", flush=True)
    return m, info


def build_mcq(it, maxpx):
    q = "\nQuestion: " + it["question"] + "\nOptions: \n" + "\n".join(it["options"]) \
        + "\n" + MEDEVAL_MCQ_TAIL
    im = [{"type": "image", "image": p, "max_pixels": maxpx, "min_pixels": min(MEDEVAL_MINPX, maxpx)}
          for p in it["imgs"]]
    return [{"role": "user", "content": im + [{"type": "text", "text": q}]}]


def build_open_gen(it, maxpx):
    im = [{"type": "image", "image": it["img"], "max_pixels": maxpx,
           "min_pixels": min(OPEN_GEN_MINPX, maxpx)}]
    return [{"role": "system", "content": SYS_OPEN},
            {"role": "user", "content": im + [{"type": "text", "text": it["question"]}]}]


def build_verif(it, ans, maxpx):
    im = [{"type": "image", "image": it["img"], "max_pixels": maxpx,
           "min_pixels": min(VERIF_MINPX, maxpx)}]
    txt = (f"Question: {it['question']}\nProposed answer: {ans}\n"
           f"Is the proposed answer correct? Answer Yes or No.")
    return [{"role": "system", "content": SYS_VERIF},
            {"role": "user", "content": im + [{"type": "text", "text": txt}]}]


def encode(proc, msgs):
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    return proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to("cuda")


def vision_tokens(enc):
    g = enc.get("image_grid_thw")
    return 0 if g is None else int(g.prod(dim=-1).sum().item())


def measured(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    s = MemSampler(); s.start()
    t0 = time.time()
    try:
        res = fn()
    finally:
        torch.cuda.synchronize()
        dt = time.time() - t0
        mp, mb = s.stop()
    dv, dm = d_footprint(mp, mb)
    return res, dict(b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                     c_peak_reserved_gib=gb(torch.cuda.max_memory_reserved()),
                     d_process_footprint_gib=gb(dv), d_method=dm,
                     d_board_used_gib=gb(mb), d_nvml_per_process_gib=gb(mp),
                     wall_s=round(dt, 3))


def scenario_reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1.0)
    return dict(reserved_after_empty_cache_gib=gb(torch.cuda.memory_reserved()),
                allocated_after_empty_cache_gib=gb(torch.cuda.memory_allocated()),
                board_after_empty_cache_gib=gb(nvml_board_bytes()))


RESULTS = {}
OUTF = os.path.join(OUT, f"levers_{A.part}{A.suffix}.json")


def save():
    json.dump(RESULTS, open(OUTF, "w"), indent=1)


def agg_rows(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return dict(n=0, n_failed=len(rows), rows=rows, note="ALL ITEMS FAILED")

    def agg(k):
        v = [r[k] for r in ok if k in r]
        return (dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                     min=round(float(np.min(v)), 4)) if v else None)
    drv = max(ok, key=lambda r: r["b_peak_allocated_gib"])
    return dict(n=len(ok), n_failed=len(rows) - len(ok),
                b_peak_allocated_gib=agg("b_peak_allocated_gib"),
                c_peak_reserved_gib=agg("c_peak_reserved_gib"),
                d_process_footprint_gib=agg("d_process_footprint_gib"),
                d_board_used_gib=agg("d_board_used_gib"),
                input_tokens=agg("input_tokens"), vision_tokens=agg("vision_tokens"),
                gen_tokens=agg("gen_tokens"), wall_s=agg("wall_s"),
                peak_driver=dict(source=drv.get("src"), idx=drv.get("idx"),
                                 image_pixels=drv.get("image_pixels"), n_images=drv.get("n_images"),
                                 input_tokens=drv.get("input_tokens"),
                                 vision_tokens=drv.get("vision_tokens"),
                                 gen_tokens=drv.get("gen_tokens"),
                                 b_peak_allocated_gib=drv["b_peak_allocated_gib"]),
                rows=rows)


def sweep_mcq(model, proc, items, caps, label, max_new):
    """MCQ direct generation across the cap ladder.  One entry per cap."""
    outd = {}
    for N in caps:
        maxpx = N * PATCH
        pre = scenario_reset()
        rows = []
        for k, it in enumerate(items):
            try:
                enc = encode(proc, build_mcq(it, maxpx))
                nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)

                def run():
                    with torch.no_grad():
                        return model.generate(**enc, max_new_tokens=max_new, do_sample=False)
                out, m = measured(run)
                m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=len(it["imgs"]),
                         input_tokens=nin, vision_tokens=nvis, gen_tokens=int(out.shape[1] - nin))
                rows.append(m)
                del enc, out
            except Exception as e:
                rows.append(dict(src=it["src"], idx=it["idx"], error=f"{type(e).__name__}: {str(e)[:180]}"))
                print(f"  {label} cap{N} fail {it['idx']}: {str(e)[:110]}", flush=True)
                torch.cuda.empty_cache()
        d = agg_rows(rows)
        d["cap_patches"], d["max_pixels"], d["scenario_reset"] = N, maxpx, pre
        outd[f"cap{N}"] = d
        pk = d.get("b_peak_allocated_gib", {}) or {}
        print(f"[{label}] cap{N} (max_pixels={maxpx}) n={d['n']} (b)peak={pk.get('peak')} "
              f"(d)peak={(d.get('d_process_footprint_gib') or {}).get('peak')} "
              f"vis_tok_peak={(d.get('vision_tokens') or {}).get('peak')}", flush=True)
        RESULTS[label] = outd
        save()
    return outd


def sweep_verifier(model, proc, items, caps, label):
    outd = {}
    for N in caps:
        maxpx = N * PATCH
        pre = scenario_reset()
        rows = []
        for it in items:
            try:
                enc = encode(proc, build_verif(it, it["answer"], maxpx))
                nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)

                def run():
                    with torch.no_grad():
                        return model(**enc).logits[0, -1]
                _, m = measured(run)
                m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=1,
                         input_tokens=nin, vision_tokens=nvis, gen_tokens=0)
                rows.append(m)
                del enc
            except Exception as e:
                rows.append(dict(src=it["src"], idx=it["idx"], error=f"{type(e).__name__}: {str(e)[:180]}"))
                torch.cuda.empty_cache()
        d = agg_rows(rows)
        d["cap_patches"], d["max_pixels"], d["scenario_reset"] = N, maxpx, pre
        outd[f"cap{N}"] = d
        print(f"[{label}] cap{N} n={d['n']} (b)peak={(d.get('b_peak_allocated_gib') or {}).get('peak')}",
              flush=True)
        RESULTS[label] = outd
        save()
    return outd


def sweep_openarm(model, proc, items, caps, label, gen_ratio=0.25):
    """The FULL open-text arm (generate 8 @ cap*gen_ratio, then verifier-score 8 @ cap), per cap.
       gen_ratio 0.25 preserves the deployed relationship: generator cap320 = verifier cap1280 / 4."""
    outd = {}
    for N in caps:
        vpx, gpx = N * PATCH, max(PATCH, int(N * gen_ratio) * PATCH)
        pre = scenario_reset()
        rows = []
        for it in items:
            try:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                s = MemSampler(); s.start(); t0 = time.time()
                enc = encode(proc, build_open_gen(it, gpx))
                nin_g, nvis_g = int(enc["input_ids"].shape[1]), vision_tokens(enc)
                with model.disable_adapter():
                    with torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=64, do_sample=True,
                                             temperature=0.7, top_p=1.0, num_return_sequences=8)
                torch.cuda.synchronize()
                gen_peak_b = torch.cuda.max_memory_allocated()
                cands = [proc.tokenizer.decode(o[nin_g:], skip_special_tokens=True).strip() for o in out]
                gen_tok = int(out.shape[1] - nin_g)
                del enc, out
                nvis_v = 0
                for c in cands:
                    e2 = encode(proc, build_verif(it, c, vpx))
                    nvis_v = vision_tokens(e2)
                    with torch.no_grad():
                        _ = model(**e2).logits[0, -1]
                    del e2
                torch.cuda.synchronize()
                dt = time.time() - t0
                mp, mb = s.stop()
                dv, dm = d_footprint(mp, mb)
                rows.append(dict(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=1,
                                 input_tokens=nin_g, vision_tokens=nvis_g,
                                 verifier_vision_tokens=nvis_v, gen_tokens=gen_tok, n_samples=8,
                                 b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                                 b_peak_allocated_generator_phase_gib=gb(gen_peak_b),
                                 c_peak_reserved_gib=gb(torch.cuda.max_memory_reserved()),
                                 d_process_footprint_gib=gb(dv), d_method=dm,
                                 d_board_used_gib=gb(mb), wall_s=round(dt, 3)))
            except Exception as e:
                rows.append(dict(src=it["src"], idx=it["idx"], error=f"{type(e).__name__}: {str(e)[:180]}"))
                torch.cuda.empty_cache()
        d = agg_rows(rows)
        d.update(cap_patches=N, verifier_max_pixels=vpx, generator_max_pixels=gpx,
                 scenario_reset=pre)
        outd[f"cap{N}"] = d
        print(f"[{label}] cap{N} verif_px={vpx} gen_px={gpx} n={d['n']} "
              f"(b)peak={(d.get('b_peak_allocated_gib') or {}).get('peak')}", flush=True)
        RESULTS[label] = outd
        save()
    return outd


# ============================================================================================ PARTS
if A.part in ("null", "res7b", "resopen", "quant", "smallest"):
    mcq = pool_mcq(A.n_mcq)
    opn = pool_open(A.n_open)
    print(f"pools: mcq={len(mcq)} open={len(opn)}", flush=True)

if A.part == "null":
    # NULL TEST: reproduce S1 of vram_testtime_2026-08-11.json with THIS harness, same items, same cap.
    proc = AutoProcessor.from_pretrained(M7)
    model, li = load(M7, "Lingshu-7B")
    RESULTS["load"] = li
    sweep_mcq(model, proc, mcq, [16384], "NULL_S1_lingshu7b_direct_mcq", A.mcq_max_new)

elif A.part == "res7b":
    proc = AutoProcessor.from_pretrained(M7)
    model, li = load(M7, "Lingshu-7B")
    RESULTS["load"] = li
    sweep_mcq(model, proc, mcq, CAPS, "R1_7b_direct_mcq_by_cap", A.mcq_max_new)

elif A.part == "res32b":
    mcq = pool_mcq(A.n_mcq)
    proc = AutoProcessor.from_pretrained(M32)
    model, li = load(M32, "Lingshu-32B")
    RESULTS["load"] = li
    sweep_mcq(model, proc, mcq, CAPS, "R2_32b_direct_mcq_by_cap", A.mcq_max_new)

elif A.part == "resopen":
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(M7)
    model, li = load(M7, "Lingshu-7B")
    base_resident = torch.cuda.memory_allocated()
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER)).eval()
    torch.cuda.synchronize()
    li["adapter"] = VERIFIER
    li["adapter_marginal_resident_gib"] = gb(torch.cuda.memory_allocated() - base_resident)
    li["co_resident_total_resident_gib"] = gb(torch.cuda.memory_allocated())
    li["n_lora_params"] = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    li["n_lora_params_on_visual"] = sum(1 for n, _ in model.named_parameters()
                                        if "lora_" in n and "visual." in n)
    RESULTS["load"] = li
    sweep_verifier(model, proc, opn, CAPS, "R3_7b_plus_verifier_by_cap")
    sweep_openarm(model, proc, opn, CAPS, "R4_opentext_bestof8_arm_by_cap")

elif A.part == "quant":
    # ONE MODEL PER PROCESS.  vram_levers_2026-08-12.json:retracted withdrew six rows because a
    # second load in the SAME process read a contaminated (a); its stated lesson is "a VRAM
    # instrument must load one model per process".  --quant_arm selects exactly one.
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(M7)
    arms = [("bf16_control", None, False), ("int8", "int8", False),
            ("nf4", "nf4", False), ("nf4_skipvisual", "nf4", True)]
    if A.quant_arm:
        arms = [a for a in arms if a[0] == A.quant_arm]
        assert arms, f"unknown --quant_arm {A.quant_arm}"
    assert len(arms) == 1, ("refusing to load more than one model per process -- pass --quant_arm "
                            "and run one process per arm (see the 08-12 retraction)")
    for name, q, sv in arms:
        try:
            model, li = load(M7, f"Lingshu-7B[{name}]", quant=q, skip_visual=sv)
        except Exception as e:
            RESULTS[f"Q_{name}"] = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
            print(f"[quant] {name} LOAD FAILED: {e}", flush=True); save(); continue
        base_resident = torch.cuda.memory_allocated()
        RESULTS[f"Q_{name}_load"] = li
        save()
        sweep_mcq(model, proc, mcq, [16384, 1280, 320], f"Q_{name}_mcq_by_cap", A.mcq_max_new)
        try:
            model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER)).eval()
            torch.cuda.synchronize()
            RESULTS[f"Q_{name}_load"]["adapter_marginal_resident_gib"] = \
                gb(torch.cuda.memory_allocated() - base_resident)
            RESULTS[f"Q_{name}_load"]["co_resident_total_resident_gib"] = gb(torch.cuda.memory_allocated())
            sweep_verifier(model, proc, opn, [1280, 320], f"Q_{name}_verifier_by_cap")
            sweep_openarm(model, proc, opn, [1280, 320], f"Q_{name}_openarm_by_cap")
        except Exception as e:
            RESULTS[f"Q_{name}_adapter_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            print(f"[quant] {name} adapter phase failed: {e}", flush=True)
        del model
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        time.sleep(3)
        RESULTS[f"Q_{name}_teardown_board_gib"] = gb(nvml_board_bytes())
        save()

elif A.part == "cores":
    # 7B and 32B CO-RESIDENT on ONE card: does an escalating policy fit on a single 80 GB GPU?
    mcq = pool_mcq(A.n_mcq)
    proc7 = AutoProcessor.from_pretrained(M7)
    variants = [("both_bf16", None), ("32b_nf4", "nf4"), ("32b_int8", "int8")]
    m7 = m32 = None
    for vname, q32 in variants:
        # a FAILED variant must not leave its 7B resident, or the next variant's (a) reads two 7Bs
        # (observed once: (a)=30.9883 GiB for a "bare" 7B).  Free before every attempt.
        m7 = m32 = None
        import gc
        gc.collect()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); time.sleep(2)
        free_b = int(pynvml.nvmlDeviceGetMemoryInfo(H).total) - nvml_board_bytes()
        need = 82.0 if q32 is None else 55.0
        print(f"[cores {vname}] free={gb(free_b)} GiB, need ~{need} GiB exclusive", flush=True)
        if gb(free_b) < need:
            RESULTS[f"C_{vname}"] = dict(
                variant=vname, status="NOT ATTEMPTED -- insufficient free VRAM on a SHARED card",
                free_gib_at_check=gb(free_b), needed_gib=need, gpu_total_gib=gb(TOTAL_BYTES),
                note=("an OOM under co-tenancy would measure the CO-TENANT, not co-residency, so the "
                      "attempt is skipped rather than recorded as a failure to fit."))
            print(f"[cores {vname}] SKIPPED (need {need}, free {gb(free_b)})", flush=True)
            save(); continue
        try:
            m7, li7 = load(M7, "Lingshu-7B")
            after7 = gb(torch.cuda.memory_allocated())
            board7 = gb(nvml_board_bytes())
            m32, li32 = load(M32, f"Lingshu-32B[{q32 or 'bf16'}]", quant=q32)
            both = gb(torch.cuda.memory_allocated())
            entry = dict(variant=vname, load_7b=li7, load_32b=li32,
                         a_7b_only_resident_gib=after7, a_both_resident_gib=both,
                         board_after_7b_gib=board7, board_after_both_gib=gb(nvml_board_bytes()),
                         gpu_total_gib=gb(TOTAL_BYTES))
            rows7, rows32 = [], []
            for it in mcq:
                for tag, mdl, rows in (("7b", m7, rows7), ("32b", m32, rows32)):
                    try:
                        enc = encode(proc7, build_mcq(it, MEDEVAL_MAXPX))
                        nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)

                        def run():
                            with torch.no_grad():
                                return mdl.generate(**enc, max_new_tokens=A.mcq_max_new, do_sample=False)
                        out, m = measured(run)
                        m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"],
                                 n_images=len(it["imgs"]), input_tokens=nin, vision_tokens=nvis,
                                 gen_tokens=int(out.shape[1] - nin), leg=tag)
                        rows.append(m)
                        del enc, out
                    except Exception as e:
                        rows.append(dict(src=it["src"], idx=it["idx"], leg=tag,
                                         error=f"{type(e).__name__}: {str(e)[:180]}"))
                        print(f"  cores[{vname}] {tag} fail {it['idx']}: {str(e)[:110]}", flush=True)
                        torch.cuda.empty_cache()
            entry["cheap_leg_rows"] = agg_rows(rows7)
            entry["strong_leg_rows"] = agg_rows(rows32)
            RESULTS[f"C_{vname}"] = entry
            print(f"[cores {vname}] both resident (a)={both} GiB; "
                  f"7b peak={(entry['cheap_leg_rows'].get('b_peak_allocated_gib') or {}).get('peak')} "
                  f"32b peak={(entry['strong_leg_rows'].get('b_peak_allocated_gib') or {}).get('peak')}",
                  flush=True)
            del m7, m32
        except Exception as e:
            RESULTS[f"C_{vname}"] = dict(variant=vname, error=f"{type(e).__name__}: {str(e)[:400]}",
                                         gpu_total_gib=gb(TOTAL_BYTES),
                                         board_at_failure_gib=gb(nvml_board_bytes()))
            print(f"[cores {vname}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        time.sleep(3)
        save()

elif A.part == "smallest":
    # the minimum footprint the 7B-side pipeline runs in at all: nf4 weights x the low cap rungs.
    from peft import PeftModel
    proc = AutoProcessor.from_pretrained(M7)
    for qname, q in (("nf4", "nf4"), ("nf4_skipvisual", "nf4")):
        sv = qname.endswith("skipvisual")
        model, li = load(M7, f"Lingshu-7B[{qname}]", quant=q, skip_visual=sv)
        base_resident = torch.cuda.memory_allocated()
        model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER)).eval()
        torch.cuda.synchronize()
        li["adapter_marginal_resident_gib"] = gb(torch.cuda.memory_allocated() - base_resident)
        RESULTS[f"S_{qname}_load"] = li
        sweep_mcq(model, proc, mcq, [1280, 320, 80, 20], f"S_{qname}_mcq_by_cap", A.mcq_max_new)
        sweep_openarm(model, proc, opn, [1280, 320, 80], f"S_{qname}_openarm_by_cap")
        del model
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); time.sleep(3)
        save()

save()
print(f"\nDONE part={A.part} -> {OUTF}", flush=True)
