#!/usr/bin/env python3
"""measure_testtime_vram.py -- TEST-TIME VRAM for the deployed Lingshu cascade legs, under HF.

WHY THIS EXISTS.  The nine VRAM numbers that circulate in this project (71.45 / 152.82 / 154.08 GB
and friends) are vLLM MEMORY-POOL RESERVATIONS: NVML board `used` sampled while vLLM held a pool
sized by the --gpu_mem flag.  The tell is that three different 7B models all report exactly 71.45
and three different 32B models exactly 152.82 -- they measure the FLAG, not the model.  The only
genuine method-level measurement in the repo is logs/rt_cascade.log (18,014 MB 7B + 68,321 MB 32B,
June MedVLThinker co-resident cascade) and that is a resident-after-load snapshot, not a peak.
See results/cascade_methods/artifacts/method_inventory_2026-08-11.json:conventions.vram_convention.

RELATION TO src/cascade/measure_single_leg.py.  That script owns latency/power/VRAM for the JUNE
MedVLThinker era; its memory instrument is a single line (`torch.cuda.max_memory_allocated`, line
119) and its outputs rt_7b.jsonl / rt_32b.jsonl are not on disk.  This script is the MEMORY half of
that instrument, (i) generalized to the CURRENT Lingshu era + the LoRA verifier + best-of-8, and
(ii) extended with the two quantities measure_single_leg.py does not record: peak RESERVED and the
whole-PROCESS footprint at peak.  measure_single_leg.py is left untouched (it is the June
reproducibility anchor).

*** HF TRANSFORMERS, NEVER vLLM ***  vLLM reserves a pool and hides true allocation -- that is the
entire reason this data was missing.  It is also mandatory for the verifier: vLLM 0.9.0.1 silently
drops all 192 visual.* LoRA modules.

CONVENTIONS -- four separate quantities, never to be conflated:
  (a) weights_resident   torch.cuda.memory_allocated() after from_pretrained() + synchronize.
  (b) peak_allocated     torch.cuda.max_memory_allocated(), reset before EVERY item.  Live tensors
                         only -- what the model actually needs.
  (c) peak_reserved      torch.cuda.max_memory_reserved(), reset before every item.  What the torch
                         caching allocator holds from the driver: (b) + intra-pool fragmentation.
  (d) process_footprint  whole-process GPU memory, max over a 20 ms NVML sampler running during the
                         item.  = (c) + the CUDA context + cuBLAS workspaces + anything outside the
                         torch allocator.  THIS is what a deployer provisions.
                         METHOD: NVML per-process usedGpuMemory is preferred, but in this container
                         NVML reports HOST pids that never match os.getpid(), so it returns 0 and the
                         run falls back to `board used - pre-run baseline`.  That substitute is exact
                         here: the baseline is recorded at import, exactly ONE of our processes is
                         pinned per GPU, and exclusivity is asserted before and after.  Every row
                         carries d_method naming which of the two produced it, plus the raw
                         d_board_used_gib and d_nvml_per_process_gib.

Usage (run from repo root, one GPU pinned per process):
  CUDA_VISIBLE_DEVICES=0 python3 src/cascade/measure_testtime_vram.py --group small --out ...
  CUDA_VISIBLE_DEVICES=1 python3 src/cascade/measure_testtime_vram.py --group big   --out ...
"""
import argparse, glob, io, json, os, random, sys, threading, time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
GIB = 1024 ** 3

# ---------------------------------------------------------------- deployed configuration constants
M7 = "lingshu-medical-mllm/Lingshu-7B"
M32 = "lingshu-medical-mllm/Lingshu-32B"
VERIFIER = "ckpts/train/lora_verifier_disjoint"          # the clean (decontaminated) verifier

# harness A (MedEvalKit, the paper's MCQ suite): CAP_MAX_PIXELS unset => qwen_vl_utils defaults.
MEDEVAL_MAXPX, MEDEVAL_MINPX = 12845056, 3136            # qwen_vl_utils.vision_process defaults
# open-text GENERATOR as deployed (runners/run_cheapleg_open_gen.sh -> run_openvqa.py --cap cap320)
OPEN_GEN_MAXPX, OPEN_GEN_MINPX = (1280 * 28 * 28) // 4, 4 * 28 * 28      # 250,880 / 3,136
# open-text VERIFIER as deployed (src/training_methods/cheapleg_score_open.py:31)
VERIF_MAXPX, VERIF_MINPX = 1280 * 28 * 28, 4 * 28 * 28                   # 1,003,520 / 3,136

# MedEvalKit's own MCQ instruction, verbatim from MedEvalKit/utils/question_formats.py
# get_multiple_choice_prompt (is_reasoning=False).  MedEvalKit/ is READ ONLY and unmodified.
MEDEVAL_MCQ_TAIL = "Answer with the option's letter from the given choices directly."
# genuine-reasoning system prompt, verbatim from runners/run_lingshu_32b_phase2.sh:6 ($TS) -- the
# prompt that produced ckpts/acc_gen/lingshu32b/think_fullres (150-259 generated tokens).
SYS_REASON = ("You are an expert medical AI. Reason step by step about the image and the question, "
              "then end your response with a line 'Answer: X' where X is the correct option letter.")
# open-text generator system prompt, verbatim from src/labeling/run_openvqa.py:26
SYS_OPEN = ("You are an expert medical image analyst. Answer the question with a short, specific "
            "phrase. Do not explain.")
# verifier system prompt, verbatim from src/training_methods/cheapleg_score_open.py:32
SYS_VERIF = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
             "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

ap = argparse.ArgumentParser()
ap.add_argument("--group", choices=["small", "big"], required=True,
                help="small = S1/S3/S4 on Lingshu-7B (one load); big = S2/S5 on Lingshu-32B (one load)")
ap.add_argument("--n_mcq", type=int, default=15)
ap.add_argument("--n_open", type=int, default=12)
ap.add_argument("--n_reason", type=int, default=10)
ap.add_argument("--mcq_max_new", type=int, default=256)
ap.add_argument("--reason_max_new", type=int, default=1536)   # deployed value (phase2 runner)
ap.add_argument("--out", default="results/cascade_methods/artifacts/_vram_parts")
ap.add_argument("--pools_only", action="store_true", help="dry run: build+print the item pools, load no model")
ap.add_argument("--only", default="", help="comma-separated scenario ids to run (e.g. S5). Default: all in the group.")
ap.add_argument("--suffix", default="", help="appended to the output filename, so a re-run does not clobber")
A = ap.parse_args()
ONLY = set(x.strip() for x in A.only.split(",") if x.strip())


def want(sid):
    return (not ONLY) or sid in ONLY
OUT = os.path.join(ROOT, A.out)
os.makedirs(OUT, exist_ok=True)

import pynvml
pynvml.nvmlInit()
DEV_IDX = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
H = pynvml.nvmlDeviceGetHandleByIndex(DEV_IDX)
PID = os.getpid()
TOTAL_BYTES = int(pynvml.nvmlDeviceGetMemoryInfo(H).total)


def nvml_proc_bytes():
    """Per-process GPU memory for THIS pid (bytes). 0 if the driver will not report it."""
    try:
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(H):
            if p.pid == PID:
                return int(p.usedGpuMemory or 0)
    except Exception:
        pass
    return 0


def nvml_board_bytes():
    return int(pynvml.nvmlDeviceGetMemoryInfo(H).used)


# (d) is meant to be per-PROCESS NVML. In this container NVML reports HOST pids, which never match
# os.getpid(), so nvml_proc_bytes() returns 0. The substitute is board `used` minus the pre-run
# baseline, which is exact here because (i) the baseline is recorded below, (ii) exactly ONE of our
# processes is pinned per GPU, and (iii) exclusivity is asserted before and after the run.
BASELINE_BOARD = nvml_board_bytes()


def d_footprint(max_proc, max_board):
    """Return ((d) bytes, method name). Prefer the true per-process reading when the driver gives it."""
    if max_proc > 0:
        return max_proc, "nvml_per_process_usedGpuMemory"
    return max(0, max_board - BASELINE_BOARD), "nvml_board_used_minus_prerun_baseline"


class MemSampler(threading.Thread):
    """(d): poll NVML per-process + board memory at 20 ms during an item; keep the max."""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval = interval
        self.on = True
        self.max_proc = 0
        self.max_board = 0

    def run(self):
        while self.on:
            self.max_proc = max(self.max_proc, nvml_proc_bytes())
            self.max_board = max(self.max_board, nvml_board_bytes())
            time.sleep(self.interval)

    def stop(self):
        self.on = False
        self.join(timeout=2.0)
        # one final synchronous read, in case the item was shorter than the sample interval
        self.max_proc = max(self.max_proc, nvml_proc_bytes())
        self.max_board = max(self.max_board, nvml_board_bytes())
        return self.max_proc, self.max_board


def gb(x):
    return round(x / GIB, 4)


# ---------------------------------------------------------------------------- item pools (FIXED)
def pool_mcq(n):
    """MedEvalKit-suite MCQ items spanning the two extremes of the driver space:
       PMC-VQA test_2.csv  = tiny images (<0.18 MP), short prompts, 4 options, 79.2% of the pool;
       MedXpertQA-MM       = the largest images in the suite (up to 5.9 MP), MULTI-image items,
                             long clinical stems, 10 options.  Deterministic (seed 42)."""
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
    """The three DEPLOYED open-text cells (slake_open / vqa_rad_open / pathvqa_open), loaded exactly
       the way src/training_methods/cheapleg_score_open.py:imgs_for() loads them.  Smallest and
       largest image per cell, so the pool brackets the driver."""
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
            if name == "pathvqa_open" and i >= 1500:      # published PathVQA-open sub-pool
                break
        c.sort(key=lambda z: z["px"])
        out += c[:per // 2] + c[-(per - per // 2):]
    return out[:n]


if A.pools_only:
    m = pool_mcq(A.n_mcq)
    for x in m:
        print(f"  MCQ {x['src']:<18} idx={str(x['idx']):<10} px={x['px']:>9} n_img={len(x['imgs'])} "
              f"len_q={len(x['question'])} n_opt={len(x['options'])}")
    o = pool_open(A.n_open)
    for x in o:
        print(f"  OPEN {x['src']:<14} idx={str(x['idx']):<8} px={x['px']:>9} "
              f"len_q={len(x['question'])} ans={x['answer'][:30]!r}")
    print(f"pools built: mcq={len(m)} open={len(o)}")
    raise SystemExit(0)

# ---------------------------------------------------------------------------- model / measurement
from transformers import AutoProcessor, AutoModelForImageTextToText           # noqa: E402
from qwen_vl_utils import process_vision_info                                 # noqa: E402


def load(path, tag):
    t0 = time.time()
    pre_board = nvml_board_bytes()
    m = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    torch.cuda.synchronize()
    info = dict(model=path, tag=tag, load_s=round(time.time() - t0, 1),
                dtype="bfloat16", attn_implementation="flash_attention_2",
                a_weights_resident_gib=gb(torch.cuda.memory_allocated()),
                weights_reserved_after_load_gib=gb(torch.cuda.memory_reserved()),
                nvml_process_after_load_gib=gb(nvml_proc_bytes()),
                nvml_board_after_load_gib=gb(nvml_board_bytes()),
                nvml_board_before_load_gib=gb(pre_board),
                n_params=int(sum(p.numel() for p in m.parameters())))
    print(f"[load] {tag}: (a) weights_resident={info['a_weights_resident_gib']} GiB  "
          f"process={info['nvml_process_after_load_gib']} GiB  ({info['load_s']}s)", flush=True)
    return m, info


def build_mcq(proc, it, reasoning):
    """DIRECT arm  = MedEvalKit's own MCQ prompt (question_formats.get_multiple_choice_prompt,
                     is_reasoning=False), no system prompt. Harness A, the paper's MCQ suite.
       REASONING arm = src/labeling/run_vlm_eval.py build(): bare 'A) text' options and NO
                     instruction tail, with $TS as the system prompt -- the exact construction that
                     produced ckpts/acc_gen/lingshu32b/think_fullres.
       *** The tail matters and is not cosmetic. *** Pairing $TS with MedEvalKit's tail (which ends
       '...from the given choices directly.') yields 2.6 generated tokens: the 'directly' instruction
       sits closer to the generation point and overrides the reason-step-by-step system prompt. That
       run is recorded as S5_ABORTED_prompt_conflict -- it is exactly the 3-token pseudo-reasoning
       arm CLAUDE.md warns about, caught by this script's own token audit."""
    if reasoning:
        def as_paren(o):                      # normalize "A: text" / "A) text" -> "A) text"
            o = o.strip()
            return o[0] + ") " + o[2:].strip() if len(o) > 2 and o[1] in ":)" else o
        q = it["question"] + "\n" + "\n".join(as_paren(o) for o in it["options"])
        sysmsg = [{"role": "system", "content": SYS_REASON}]
    else:
        q = "\nQuestion: " + it["question"] + "\nOptions: \n" + "\n".join(it["options"]) \
            + "\n" + MEDEVAL_MCQ_TAIL
        sysmsg = []
    im = [{"type": "image", "image": p, "max_pixels": MEDEVAL_MAXPX, "min_pixels": MEDEVAL_MINPX}
          for p in it["imgs"]]
    return sysmsg + [{"role": "user", "content": im + [{"type": "text", "text": q}]}]


def build_open_gen(proc, it):
    im = [{"type": "image", "image": it["img"], "max_pixels": OPEN_GEN_MAXPX,
           "min_pixels": OPEN_GEN_MINPX}]
    return [{"role": "system", "content": SYS_OPEN},
            {"role": "user", "content": im + [{"type": "text", "text": it["question"]}]}]


def build_verif(proc, it, ans):
    im = [{"type": "image", "image": it["img"], "max_pixels": VERIF_MAXPX, "min_pixels": VERIF_MINPX}]
    txt = (f"Question: {it['question']}\nProposed answer: {ans}\n"
           f"Is the proposed answer correct? Answer Yes or No.")
    return [{"role": "system", "content": SYS_VERIF},
            {"role": "user", "content": im + [{"type": "text", "text": txt}]}]


def encode(proc, msgs):
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to("cuda")
    return enc


def vision_tokens(enc):
    g = enc.get("image_grid_thw")
    if g is None:
        return 0
    return int(g.prod(dim=-1).sum().item())


def measured(fn):
    """Run fn() with the four quantities captured.  (b)/(c) reset immediately before."""
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
    """Between scenarios: release cached blocks so a later scenario's (c)/(d) are not inflated by an
       earlier one's high-water mark.  The returned value proves the release happened."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1.0)
    return dict(reserved_after_empty_cache_gib=gb(torch.cuda.memory_reserved()),
                allocated_after_empty_cache_gib=gb(torch.cuda.memory_allocated()),
                nvml_process_after_empty_cache_gib=gb(nvml_proc_bytes()))


RESULTS = {}


def finish(name, rows, meta):
    if not want(name.split("_")[0]):
        return
    ok = [r for r in rows if "error" not in r]
    if not ok:
        RESULTS[name] = dict(meta=meta, n=0, rows=rows, note="ALL ITEMS FAILED")
        return
    def agg(k):
        v = [r[k] for r in ok if k in r]
        return dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                    min=round(float(np.min(v)), 4)) if v else None
    drv = max(ok, key=lambda r: r["b_peak_allocated_gib"])
    RESULTS[name] = dict(
        meta=meta, n=len(ok), n_failed=len(rows) - len(ok),
        b_peak_allocated_gib=agg("b_peak_allocated_gib"),
        c_peak_reserved_gib=agg("c_peak_reserved_gib"),
        d_process_footprint_gib=agg("d_process_footprint_gib"),
        d_board_used_gib=agg("d_board_used_gib"),
        input_tokens=agg("input_tokens"), vision_tokens=agg("vision_tokens"),
        gen_tokens=agg("gen_tokens"), wall_s=agg("wall_s"),
        peak_driver=dict(source=drv.get("src"), idx=drv.get("idx"),
                         image_pixels=drv.get("image_pixels"), n_images=drv.get("n_images"),
                         input_tokens=drv.get("input_tokens"), vision_tokens=drv.get("vision_tokens"),
                         gen_tokens=drv.get("gen_tokens"),
                         b_peak_allocated_gib=drv["b_peak_allocated_gib"]),
        rows=rows)
    print(f"[{name}] n={len(ok)}  (b) peak={agg('b_peak_allocated_gib')['peak']} GiB  "
          f"(c) peak={agg('c_peak_reserved_gib')['peak']} GiB  "
          f"(d) peak={agg('d_process_footprint_gib')['peak']} GiB", flush=True)
    json.dump(RESULTS, open(os.path.join(OUT, f"vram_{A.group}{A.suffix}.json"), "w"), indent=1)


# =================================================================================== GROUP: small
if A.group == "small":
    mcq = pool_mcq(A.n_mcq)
    opn = pool_open(A.n_open)
    print(f"pools: mcq={len(mcq)} open={len(opn)}", flush=True)
    proc = AutoProcessor.from_pretrained(M7)
    model, li = load(M7, "Lingshu-7B")
    base_resident = torch.cuda.memory_allocated()

    # ---- S1: Lingshu-7B direct, batch 1, MedEvalKit resolution policy (the cheap leg)
    pre = scenario_reset()
    rows = []
    for k, it in enumerate(mcq if want("S1") else []):
        try:
            enc = encode(proc, build_mcq(proc, it, reasoning=False))
            nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
            def run():
                with torch.no_grad():
                    return model.generate(**enc, max_new_tokens=A.mcq_max_new, do_sample=False)
            out, m = measured(run)
            m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=len(it["imgs"]),
                     input_tokens=nin, vision_tokens=nvis,
                     gen_tokens=int(out.shape[1] - nin))
            rows.append(m)
            del enc, out
        except Exception as e:                                            # per-item error guard
            rows.append(dict(src=it["src"], idx=it["idx"], error=str(e)[:200]))
            print(f"  S1 fail {it['idx']}: {str(e)[:120]}", flush=True)
        if (k + 1) % 5 == 0:
            print(f"  S1 {k+1}/{len(mcq)}", flush=True)
    finish("S1_lingshu7b_direct_mcq", rows, dict(
        scenario="Lingshu-7B direct, batch 1, MedEvalKit (harness A) resolution policy -- THE CHEAP LEG",
        model=M7, tp=1, batch_size=1, adapter=None,
        max_pixels=MEDEVAL_MAXPX, min_pixels=MEDEVAL_MINPX,
        resolution_note="CAP_MAX_PIXELS unset at eval => qwen_vl_utils defaults; verified 12,845,056",
        max_new_tokens=A.mcq_max_new,
        max_new_tokens_note=("deployed MedEvalKit cap is 8192 (MedEvalKit/eval.sh:23); capped at "
                             f"{A.mcq_max_new} here for runtime. Under HF the KV cache is allocated "
                             "per GENERATED token, not per cap, so the cap is not binding on VRAM as "
                             "long as observed gen_tokens stays well below it -- see gen_tokens."),
        prompt="MedEvalKit/utils/question_formats.py get_multiple_choice_prompt(is_reasoning=False), no system prompt",
        load=li))

    # ---- S3: + the clean LoRA verifier (marginal VRAM over the bare 7B, and co-resident total)
    from peft import PeftModel
    pre_adapter_alloc = torch.cuda.memory_allocated()
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER))
    model.eval()
    torch.cuda.synchronize()
    post_adapter_alloc = torch.cuda.memory_allocated()
    adapter_bytes = post_adapter_alloc - pre_adapter_alloc
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    n_lora_visual = sum(1 for n, _ in model.named_parameters() if "lora_" in n and "visual." in n)
    print(f"[adapter] marginal resident = {gb(adapter_bytes)} GiB over the bare 7B; "
          f"{n_lora} lora params ({n_lora_visual} on visual.*)", flush=True)

    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]

    pre3 = scenario_reset()
    rows = []
    for k, it in enumerate(opn if want("S3") else []):
        try:
            enc = encode(proc, build_verif(proc, it, it["answer"]))
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
            rows.append(dict(src=it["src"], idx=it["idx"], error=str(e)[:200]))
            print(f"  S3 fail {it['idx']}: {str(e)[:120]}", flush=True)
    finish("S3_lingshu7b_plus_lora_verifier", rows, dict(
        scenario=("Lingshu-7B + the CLEAN LoRA verifier (ckpts/train/lora_verifier_disjoint), batch 1, "
                  "ONE scoring forward pass (no generation) -- as deployed by "
                  "src/training_methods/cheapleg_score_open.py"),
        model=M7, adapter=VERIFIER, tp=1, batch_size=1,
        max_pixels=VERIF_MAXPX, min_pixels=VERIF_MINPX,
        max_new_tokens=0, forward_only=True,
        marginal_adapter_resident_gib=gb(adapter_bytes),
        marginal_adapter_note=("(a)-style: torch.cuda.memory_allocated() delta across "
                               "PeftModel.from_pretrained on the ALREADY-loaded base. The verifier is "
                               "a LoRA adapter ON Lingshu-7B, so generator and verifier SHARE one set "
                               "of base weights -- 'co-resident total' = base + this delta, NOT 2x7B."),
        co_resident_total_resident_gib=gb(post_adapter_alloc),
        bare_base_resident_gib=gb(base_resident),
        n_lora_params=n_lora, n_lora_params_on_visual=n_lora_visual,
        lora_note="HF only. vLLM 0.9.0.1 silently drops all 192 visual.* LoRA modules.",
        prompt="src/training_methods/cheapleg_score_open.py pyes()", load=li,
        scenario_reset=pre3))

    # ---- S4: the open-text best-of-8 arm end to end (generate 8, then score 8), one process
    pre4 = scenario_reset()
    rows = []
    for k, it in enumerate(opn if want("S4") else []):
        try:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            s = MemSampler(); s.start(); t0 = time.time()
            # phase 1 -- generator: 8 samples, temp 0.7, adapter OFF (base behaviour, as deployed)
            enc = encode(proc, build_open_gen(proc, it))
            nin_g, nvis_g = int(enc["input_ids"].shape[1]), vision_tokens(enc)
            with model.disable_adapter():
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.7,
                                         top_p=1.0, num_return_sequences=8)
            torch.cuda.synchronize()
            gen_peak_b = torch.cuda.max_memory_allocated()
            cands = [proc.tokenizer.decode(o[nin_g:], skip_special_tokens=True).strip()
                     for o in out]
            gen_tok = int(out.shape[1] - nin_g)
            del enc, out
            # phase 2 -- verifier: score all 8 candidates, adapter ON
            for c in cands:
                e2 = encode(proc, build_verif(proc, it, c))
                with torch.no_grad():
                    _ = model(**e2).logits[0, -1]
                del e2
            torch.cuda.synchronize()
            dt = time.time() - t0
            mp, mb = s.stop()
            rows.append(dict(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=1,
                             input_tokens=nin_g, vision_tokens=nvis_g, gen_tokens=gen_tok,
                             n_samples=8, n_verifier_passes=len(cands),
                             b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                             b_peak_allocated_generator_phase_gib=gb(gen_peak_b),
                             c_peak_reserved_gib=gb(torch.cuda.max_memory_reserved()),
                             d_process_footprint_gib=gb(mp), d_board_used_gib=gb(mb),
                             wall_s=round(dt, 3)))
        except Exception as e:
            rows.append(dict(src=it["src"], idx=it["idx"], error=str(e)[:200]))
            print(f"  S4 fail {it['idx']}: {str(e)[:120]}", flush=True)
        if (k + 1) % 4 == 0:
            print(f"  S4 {k+1}/{len(opn)}", flush=True)
    finish("S4_opentext_bestof8_full_arm", rows, dict(
        scenario=("THE OPEN-TEXT ARM END TO END in ONE process: Lingshu-7B generates 8 samples "
                  "(temp 0.7, max_new_tokens 64, cap320, adapter DISABLED), then the clean LoRA "
                  "verifier scores all 8 (adapter ENABLED, max_pixels 1,003,520). Peak is over the "
                  "WHOLE pipeline."),
        model=M7, adapter=VERIFIER, tp=1, batch_size="1 question, num_return_sequences=8",
        generator_max_pixels=OPEN_GEN_MAXPX, generator_min_pixels=OPEN_GEN_MINPX,
        generator_max_new_tokens=64, generator_temperature=0.7, n_samples=8,
        verifier_max_pixels=VERIF_MAXPX, verifier_min_pixels=VERIF_MINPX,
        deployment_note=("As RUN in this repo the two phases are separate processes (vLLM generation "
                         "then HF scoring). Measured here co-resident in ONE process because the "
                         "verifier is a LoRA adapter on the SAME base -- which is the configuration a "
                         "deployer should actually use, and it is why the arm costs one 7B, not two."),
        resolution_note=("the deployed generator runs cap320 (250,880 px) while the deployed verifier "
                         "runs 1,003,520 px -- both reproduced verbatim, so the two phases really do "
                         "differ in vision-token count."),
        load=li, scenario_reset=pre4))

# =================================================================================== GROUP: big
else:
    mcq = pool_mcq(A.n_mcq)
    print(f"pools: mcq={len(mcq)}", flush=True)
    proc = AutoProcessor.from_pretrained(M32)
    model, li = load(M32, "Lingshu-32B")

    # ---- S2: Lingshu-32B direct, batch 1, same items + resolution as S1 (the strong leg / paper bar)
    pre = scenario_reset()
    rows = []
    for k, it in enumerate(mcq if want("S2") else []):
        try:
            enc = encode(proc, build_mcq(proc, it, reasoning=False))
            nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
            def run():
                with torch.no_grad():
                    return model.generate(**enc, max_new_tokens=A.mcq_max_new, do_sample=False)
            out, m = measured(run)
            m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=len(it["imgs"]),
                     input_tokens=nin, vision_tokens=nvis, gen_tokens=int(out.shape[1] - nin))
            rows.append(m)
            del enc, out
        except Exception as e:
            rows.append(dict(src=it["src"], idx=it["idx"], error=str(e)[:200]))
            print(f"  S2 fail {it['idx']}: {str(e)[:120]}", flush=True)
        if (k + 1) % 5 == 0:
            print(f"  S2 {k+1}/{len(mcq)}", flush=True)
    finish("S2_lingshu32b_direct_mcq", rows, dict(
        scenario="Lingshu-32B direct, batch 1, SAME items and resolution as S1 -- THE STRONG LEG / the paper's bar",
        model=M32, tp=1, batch_size=1, adapter=None,
        tp_note=("tp=1 on ONE A100 80GB. The repo's own 32B runs used vLLM tp=2 for throughput; "
                 "tp=1 here answers the deployer's question 'does it fit on one card'."),
        max_pixels=MEDEVAL_MAXPX, min_pixels=MEDEVAL_MINPX, max_new_tokens=A.mcq_max_new,
        prompt="MedEvalKit/utils/question_formats.py get_multiple_choice_prompt(is_reasoning=False), no system prompt",
        load=li))

    # ---- S5: Lingshu-32B with GENUINE reasoning (long generations) -- token-audited
    pre5 = scenario_reset()
    rows = []
    for k, it in enumerate((mcq[:A.n_reason] if A.n_reason else mcq) if want("S5") else []):
        try:
            enc = encode(proc, build_mcq(proc, it, reasoning=True))
            nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
            def run():
                with torch.no_grad():
                    return model.generate(**enc, max_new_tokens=A.reason_max_new, do_sample=False)
            out, m = measured(run)
            ntok = int(out.shape[1] - nin)
            m.update(src=it["src"], idx=it["idx"], image_pixels=it["px"], n_images=len(it["imgs"]),
                     input_tokens=nin, vision_tokens=nvis, gen_tokens=ntok,
                     text_head=proc.tokenizer.decode(out[0][nin:], skip_special_tokens=True)[:180])
            rows.append(m)
            del enc, out
            print(f"  S5 {k+1}/{min(A.n_reason,len(mcq))} gen_tokens={ntok} "
                  f"peak_b={m['b_peak_allocated_gib']}", flush=True)
        except Exception as e:
            rows.append(dict(src=it["src"], idx=it["idx"], error=str(e)[:200]))
            print(f"  S5 fail {it['idx']}: {str(e)[:120]}", flush=True)
    ok = [r for r in rows if "error" not in r]
    audit = dict(mean_gen_tokens=round(float(np.mean([r["gen_tokens"] for r in ok])), 1) if ok else None,
                 min_gen_tokens=int(min(r["gen_tokens"] for r in ok)) if ok else None,
                 max_gen_tokens=int(max(r["gen_tokens"] for r in ok)) if ok else None,
                 reference_arm="ckpts/acc_gen/lingshu32b/think_fullres reports 150-259 generated tokens",
                 verdict=None)
    if ok:
        audit["verdict"] = ("GENUINE REASONING (mean gen_tokens >> 3)" if audit["mean_gen_tokens"] > 40
                            else "NOT A REASONING ARM -- mean gen_tokens too low; DO NOT LABEL AS REASONING")
    finish("S5_lingshu32b_genuine_reasoning", rows, dict(
        scenario="Lingshu-32B with GENUINE reasoning, batch 1 -- long generations, the KV-cache stress case",
        model=M32, tp=1, batch_size=1, adapter=None,
        system_prompt_source="runners/run_lingshu_32b_phase2.sh:6 ($TS) -- the prompt behind ckpts/acc_gen/lingshu32b/think_fullres",
        system_prompt=SYS_REASON, max_new_tokens=A.reason_max_new,
        max_pixels=MEDEVAL_MAXPX, min_pixels=MEDEVAL_MINPX,
        resolution_note=("held at S2's resolution policy ON PURPOSE so the S5-S2 delta isolates KV "
                         "growth from generated length. The repo's own think_fullres arm ran at a "
                         "LOWER cap (1,003,520 px), so this measurement upper-bounds it on the vision "
                         "side."),
        token_audit=audit, load=li, scenario_reset=pre5))

json.dump(RESULTS, open(os.path.join(OUT, f"vram_{A.group}{A.suffix}.json"), "w"), indent=1)
print(f"\nDONE group={A.group} -> {OUT}/vram_{A.group}.json", flush=True)
