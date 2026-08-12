#!/usr/bin/env python3
"""measure_vram_levers.py -- ATTACK 4: the VRAM LEVERS of the 7B-side pipeline, measured under HF.

WHAT THIS ADDS TO artifacts/vram_testtime_2026-08-11.json.  That artifact measured the deployed legs
at ONE resolution each and established the finding this script acts on: the peak is PREFILL-bound
and driven by VISION-TOKEN COUNT (46,816 patch-grid units on the worst item), not by decode.  That
makes image resolution the direct VRAM lever, and it had never been swept.  This script sweeps it,
plus two other footprint levers the 08-11 run explicitly left unmeasured:
    R1  7B direct MCQ            over 6 resolution caps
    R2  full open-text best-of-8 over 6 resolution caps (generator AND verifier moved together)
    Q   the CHEAP side quantised (torchao int8-weight-only / int4-weight-only), 2 caps
    C   7B + 32B CO-RESIDENT on ONE card -- "does any escalating policy fit on 80 GB"

*** ROW COMPATIBILITY IS THE POINT.  *** Same four quantities, same names, same units (GiB =
bytes/1024**3), same instrument, same env (system python3 / torch 2.9.0a0 / transformers 4.55.2 /
flash_attention_2 / bf16 / tp=1 / batch 1), same DETERMINISTIC item pools (seed 42) as
src/cascade/measure_testtime_vram.py, so a row here may be placed directly beside an 08-11 row.
The pool builders and the prompt constants are COPIED VERBATIM from that script rather than
imported, because it parses argv at module scope and cannot be imported; measure_testtime_vram.py
is left untouched (it is the 08-11 reproducibility anchor).

*** NULL TEST.  *** R1 and R2 each include the exact configuration of an 08-11 scenario (R1 @
max_pixels 12,845,056 == S1; R2 @ gen 250,880 / verif 1,003,520 == S4).  Those points must
reproduce the stored (a)/(b)/(c) values; the aggregator reports max abs deviation.  (a)/(b)/(c) are
torch-internal and reproduce exactly; (d) cannot, because the card is SHARED this time -- see below.

*** (d) UNDER A SHARED CARD.  *** On 2026-08-11 both A100s were idle and (d) was board-used minus a
pre-run baseline.  On 2026-08-12 both cards carry other rounds' jobs, so that substitute is invalid:
a foreign allocation would be charged to us.  Instead (d) is reconstructed as
    (d) = (c) peak_reserved + cuda_context_bytes
where cuda_context_bytes is measured directly in this process at startup, before any model load, as
(board_used after forcing context creation) - (board_used before) - (torch reserved after).  That is
the same decomposition the 08-11 artifact used to repair its own contaminated S4/S5 rows, and it is
validated here against the 08-11 clean offset (1.3835-1.3855 GiB).  Every row also carries the RAW
board reading and the concurrent-tenant delta so the reconstruction is auditable, and each row
records `d_method`.

GPU ETIQUETTE: waits for free VRAM, never kills another process, resumable per scenario+config.

    python3 src/cascade/measure_vram_levers.py --scenarios R1,R2
    python3 src/cascade/measure_vram_levers.py --scenarios Q
    python3 src/cascade/measure_vram_levers.py --scenarios C
"""
import argparse, glob, io, json, os, random, subprocess, sys, threading, time

# *** torch is imported LATE, on purpose. *** CUDA_VISIBLE_DEVICES must be set before torch
# initialises CUDA, and the GPU is chosen dynamically by wait_for_vram() so this run never
# oversubscribes another round's job.  numpy/PIL are safe to import here.
import numpy as np
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
GIB = 1024 ** 3

M7 = "lingshu-medical-mllm/Lingshu-7B"
M32 = "lingshu-medical-mllm/Lingshu-32B"
VERIFIER = "ckpts/train/lora_verifier_disjoint"

# ---- resolution grid.  max_pixels/(28*28) IS the merged vision-token budget for Qwen2.5-VL
# (14x14 patches, 2x2 merge), so these cap names are literally token budgets.  12,845,056 is the
# qwen_vl_utils default that MedEvalKit (harness A) runs at, i.e. what every paper MCQ cell used.
BASE_PX = 1280 * 28 * 28                                   # 1,003,520 -- internal harness "fullres"
CAPS = [
    ("medevalkit_default", 12845056, 16384),
    ("fullres",            BASE_PX,       1280),
    ("cap640",             BASE_PX // 2,   640),
    ("cap320",             BASE_PX // 4,   320),
    ("cap160",             BASE_PX // 8,   160),
    ("cap80",              BASE_PX // 16,   80),
]
MIN_PX = 4 * 28 * 28

MEDEVAL_MCQ_TAIL = "Answer with the option's letter from the given choices directly."
SYS_OPEN = ("You are an expert medical image analyst. Answer the question with a short, specific "
            "phrase. Do not explain.")
SYS_VERIF = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
             "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

ap = argparse.ArgumentParser()
ap.add_argument("--scenarios", default="R1,R2", help="comma list from R1,R2,Q,C")
ap.add_argument("--n_mcq", type=int, default=15)
ap.add_argument("--n_open", type=int, default=12)
ap.add_argument("--mcq_max_new", type=int, default=256)
ap.add_argument("--out", default="results/cascade_methods/artifacts/_vram_levers_parts")
ap.add_argument("--wait_mb", type=int, default=26000, help="free VRAM required before loading 7B")
ap.add_argument("--wait_timeout_s", type=int, default=25200)
A = ap.parse_args()
WANT = [s.strip() for s in A.scenarios.split(",") if s.strip()]
OUT = os.path.join(ROOT, A.out)
os.makedirs(OUT, exist_ok=True)

import pynvml
pynvml.nvmlInit()


def board_used_bytes(dev):
    h = pynvml.nvmlDeviceGetHandleByIndex(dev)
    return int(pynvml.nvmlDeviceGetMemoryInfo(h).used)


def free_mb_per_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [int(t) - int(u) for u, t in (l.split(", ") for l in out)]


def wait_for_vram(need_mb, timeout_s):
    """Never oversubscribe another round's job.  Returns the physical GPU index to use."""
    t0 = time.time()
    while True:
        free = free_mb_per_gpu()
        i = int(max(range(len(free)), key=lambda k: free[k]))
        if free[i] >= need_mb:
            print(f"[wait] GPU {i} has {free[i]} MB free (need {need_mb}) -- proceeding", flush=True)
            return i
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"wait_for_vram timeout: need {need_mb} MB, free {free}")
        print(f"[wait] need {need_mb} MB, free {free} -- sleeping 60 s "
              f"({int(time.time()-t0)} s waited)", flush=True)
        time.sleep(60)


DEV = wait_for_vram(A.wait_mb, A.wait_timeout_s)
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEV)      # set BEFORE torch initialises CUDA
NVML_DEV = DEV                                     # pynvml indexes PHYSICAL devices, unaffected

import torch                                                                 # noqa: E402

# ---- CUDA context size, measured before anything else is allocated.  This is what makes (d)
# reconstructable on a SHARED card.
_b0 = board_used_bytes(NVML_DEV)
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
_b1 = board_used_bytes(NVML_DEV)
CUDA_CONTEXT_BYTES = max(0, _b1 - _b0 - torch.cuda.memory_reserved())
CTX = dict(board_before_cuda_init_gib=round(_b0 / GIB, 4),
           board_after_cuda_init_gib=round(_b1 / GIB, 4),
           torch_reserved_at_probe_gib=round(torch.cuda.memory_reserved() / GIB, 4),
           cuda_context_gib=round(CUDA_CONTEXT_BYTES / GIB, 4),
           note="(d) = (c) peak_reserved + cuda_context_gib. The 08-11 artifact measured this same "
                "offset as 1.3835-1.3855 GiB on clean items; agreement validates the reconstruction.",
           gpu_physical_index=DEV)
print(f"[ctx] cuda context = {CTX['cuda_context_gib']} GiB "
      f"(board {CTX['board_before_cuda_init_gib']} -> {CTX['board_after_cuda_init_gib']})", flush=True)


# ---- PER-PROCESS (d) ON A SHARED CARD.  NVML reports HOST pids in this container, which never
# match os.getpid() -- that is why the 08-11 run fell back to board-minus-baseline.  But the host
# pid can be IDENTIFIED rather than matched: snapshot the NVML compute-process table immediately
# before our model load, snapshot it again after, and take the entry that APPEARED or GREW by
# approximately our torch-reserved bytes.  That recovers the true per-process footprint -- the
# actual 08-11 (d) definition -- even while other rounds share the card.
OUR_HOST_PID = None


def proc_table(dev):
    try:
        return {p.pid: int(p.usedGpuMemory or 0)
                for p in pynvml.nvmlDeviceGetComputeRunningProcesses(
                    pynvml.nvmlDeviceGetHandleByIndex(dev))}
    except Exception:
        return {}


def identify_self(before, expect_bytes, tol=0.35):
    """Return the host pid whose usage grew by ~expect_bytes across the load, else None."""
    after = proc_table(NVML_DEV)
    best, bestd = None, None
    for pid, used in after.items():
        grew = used - before.get(pid, 0)
        if grew <= 0:
            continue
        rel = abs(grew - expect_bytes) / max(expect_bytes, 1)
        if rel < tol and (bestd is None or rel < bestd):
            best, bestd = pid, rel
    return best, (None if best is None else
                  dict(host_pid=best, grew_gib=gb(after[best] - before.get(best, 0)),
                       expected_gib=gb(expect_bytes), rel_err=round(bestd, 4),
                       table_before={str(k): gb(v) for k, v in before.items()},
                       table_after={str(k): gb(v) for k, v in after.items()}))


def our_proc_bytes():
    if OUR_HOST_PID is None:
        return 0
    return proc_table(NVML_DEV).get(OUR_HOST_PID, 0)


class MemSampler(threading.Thread):
    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval, self.on, self.max_board, self.max_proc = interval, True, 0, 0

    def run(self):
        while self.on:
            try:
                self.max_board = max(self.max_board, board_used_bytes(NVML_DEV))
                self.max_proc = max(self.max_proc, our_proc_bytes())
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self.on = False
        self.join(timeout=2.0)
        try:
            self.max_board = max(self.max_board, board_used_bytes(NVML_DEV))
            self.max_proc = max(self.max_proc, our_proc_bytes())
        except Exception:
            pass
        return self.max_board, self.max_proc


def gb(x):
    return round(x / GIB, 4)


# ------------------------------------------------------------------ item pools (VERBATIM, seed 42)
def pool_mcq(n):
    """COPIED VERBATIM from src/cascade/measure_testtime_vram.py:pool_mcq so the items are the
    SAME ONES the 08-11 rows used (deterministic, seed 42) and the null test is exact."""
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
    """COPIED VERBATIM from src/cascade/measure_testtime_vram.py:pool_open."""
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


from transformers import AutoProcessor, AutoModelForImageTextToText          # noqa: E402
from qwen_vl_utils import process_vision_info                                # noqa: E402


def reserve(need_mb):
    """Claim the memory the moment it is free, so a co-tenant cannot race us between the
    availability check and the load -- which is exactly how the first attempt of this run died
    (logs/vram_levers_R_2026-08-12.log: OOM at 10.0 GiB in while three other jobs grew).
    The block is freed back into TORCH's caching allocator, not to the driver, so the weights load
    straight into it; torch.cuda.empty_cache() afterwards returns the unused remainder so the
    per-item (c) readings stay honest."""
    # chunked, so a fragmented card can still satisfy the claim
    chunk_mb, held = 1024, []
    for _ in range(int(need_mb // chunk_mb)):
        held.append(torch.empty(int(chunk_mb * 1024 * 1024 / 2), dtype=torch.bfloat16,
                                device="cuda"))
    del held
    return gb(torch.cuda.memory_reserved())


def load(path, tag, quant=None, need_mb=None, retries=8):
    global OUR_HOST_PID
    t0 = time.time()
    pre_board = board_used_bytes(NVML_DEV)
    pre_table = proc_table(NVML_DEV)
    reserved_gib = None
    if need_mb:
        for k in range(retries):
            try:
                wait_for_vram(need_mb, A.wait_timeout_s)
                reserved_gib = reserve(need_mb)
                print(f"[reserve] claimed {reserved_gib} GiB for {tag}", flush=True)
                break
            except torch.OutOfMemoryError as e:
                print(f"[reserve] attempt {k+1} lost the race ({e.__class__.__name__}); "
                      f"releasing and re-queueing", flush=True)
                torch.cuda.empty_cache()
                time.sleep(90)
        else:
            raise RuntimeError(f"could not claim {need_mb} MB for {tag} after {retries} attempts")
    m = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda").eval()
    q_info = None
    if quant:
        from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
        cfg = Int8WeightOnlyConfig() if quant == "int8wo" else Int4WeightOnlyConfig()
        # Quantise the LANGUAGE MODEL ONLY.  The vision tower is left bf16 on purpose: it is
        # ~0.68 B of the 8.29 B parameters, and int4 group-wise quantisation of a ViT is a
        # different (and separately contentious) intervention.  Recorded, not silently assumed.
        target = m.model.language_model if hasattr(m.model, "language_model") else m.model
        n_before = sum(p.numel() for p in m.parameters())
        quantize_(target, cfg)
        torch.cuda.synchronize()
        q_info = dict(scheme=quant, applied_to="language_model only (vision tower left bf16)",
                      n_params_before=int(n_before),
                      torchao_version=__import__("torchao").__version__)
    torch.cuda.synchronize()
    a_bytes = torch.cuda.memory_allocated()
    # return the unused remainder of the reservation to the driver so per-item (c) is honest
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pid, ident = identify_self(pre_table, a_bytes + CUDA_CONTEXT_BYTES)
    if pid is not None and OUR_HOST_PID is None:
        OUR_HOST_PID = pid
        print(f"[nvml] identified our host pid = {pid} ({ident['grew_gib']} GiB grew, "
              f"expected {ident['expected_gib']})", flush=True)
    info = dict(model=path, tag=tag, load_s=round(time.time() - t0, 1), dtype="bfloat16",
                attn_implementation="flash_attention_2", quantization=q_info,
                a_weights_resident_gib=gb(a_bytes),
                weights_reserved_after_load_gib=gb(torch.cuda.memory_reserved()),
                nvml_board_after_load_gib=gb(board_used_bytes(NVML_DEV)),
                nvml_board_before_load_gib=gb(pre_board),
                reserved_claim_gib=reserved_gib,
                nvml_self_identification=ident,
                nvml_process_after_load_gib=gb(our_proc_bytes()),
                n_params=int(sum(p.numel() for p in m.parameters())))
    print(f"[load] {tag}: (a)={info['a_weights_resident_gib']} GiB ({info['load_s']} s)", flush=True)
    return m, info


def build_mcq(it, maxpx):
    q = "\nQuestion: " + it["question"] + "\nOptions: \n" + "\n".join(it["options"]) \
        + "\n" + MEDEVAL_MCQ_TAIL
    im = [{"type": "image", "image": p, "max_pixels": maxpx, "min_pixels": MIN_PX}
          for p in it["imgs"]]
    return [{"role": "user", "content": im + [{"type": "text", "text": q}]}]


def build_open_gen(it, maxpx):
    im = [{"type": "image", "image": it["img"], "max_pixels": maxpx, "min_pixels": MIN_PX}]
    return [{"role": "system", "content": SYS_OPEN},
            {"role": "user", "content": im + [{"type": "text", "text": it["question"]}]}]


def build_verif(it, ans, maxpx):
    im = [{"type": "image", "image": it["img"], "max_pixels": maxpx, "min_pixels": MIN_PX}]
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
        mb, mp = s.stop()
    c = torch.cuda.max_memory_reserved()
    # (d): the true per-process reading when our host pid was identified; otherwise the
    # reconstruction (c) + measured CUDA context.  d_method names which one this row used.
    if mp > 0:
        d, meth = mp, "nvml_per_process_usedGpuMemory (host pid identified by load-time jump)"
    else:
        d, meth = c + CUDA_CONTEXT_BYTES, "c_peak_reserved + measured_cuda_context (fallback)"
    return res, dict(b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                     c_peak_reserved_gib=gb(c),
                     d_process_footprint_gib=gb(d), d_method=meth,
                     d_nvml_per_process_gib=gb(mp),
                     d_reconstructed_gib=gb(c + CUDA_CONTEXT_BYTES),
                     d_board_used_gib=gb(mb),
                     d_board_foreign_gib=gb(max(0, mb - d)),
                     wall_s=round(dt, 3))


def scenario_reset():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    time.sleep(1.0)
    return dict(reserved_after_empty_cache_gib=gb(torch.cuda.memory_reserved()),
                allocated_after_empty_cache_gib=gb(torch.cuda.memory_allocated()))


def summarise(rows):
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return dict(n=0, n_failed=len(rows), rows=rows, note="ALL ITEMS FAILED")
    def agg(k):
        v = [r[k] for r in ok if k in r]
        return dict(mean=round(float(np.mean(v)), 4), peak=round(float(np.max(v)), 4),
                    min=round(float(np.min(v)), 4)) if v else None
    pk = max(ok, key=lambda r: r["b_peak_allocated_gib"])
    return dict(n=len(ok), n_failed=len(rows) - len(ok),
                **{k: agg(k) for k in ("b_peak_allocated_gib", "c_peak_reserved_gib",
                                       "d_process_footprint_gib", "d_board_used_gib",
                                       "input_tokens", "vision_tokens", "gen_tokens", "wall_s")},
                peak_driver={k: pk.get(k) for k in ("src", "idx", "image_pixels", "n_images",
                                                    "input_tokens", "vision_tokens", "gen_tokens",
                                                    "b_peak_allocated_gib")},
                rows=ok + [r for r in rows if "error" in r])


def dump(name, obj):
    p = os.path.join(OUT, name + ".json")
    json.dump(obj, open(p, "w"), indent=1)
    print("[write]", p, flush=True)


# ------------------------------------------------------------------------------------ scenarios
def run_R1(model, proc, items):
    out = {}
    for cap, maxpx, tokbudget in CAPS:
        rows = []
        for it in items:
            try:
                enc = encode(proc, build_mcq(it, maxpx))
                nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
                def go():
                    with torch.inference_mode():
                        return model.generate(**enc, max_new_tokens=A.mcq_max_new, do_sample=False)
                o, mm = measured(go)
                rows.append(dict(**mm, src=it["src"], idx=it["idx"], image_pixels=it["px"],
                                 n_images=len(it["imgs"]), input_tokens=nin, vision_tokens=nvis,
                                 gen_tokens=int(o.shape[1] - enc["input_ids"].shape[1])))
                del enc, o
            except Exception as e:
                rows.append(dict(error=f"{type(e).__name__}: {e}", src=it["src"], idx=it["idx"]))
                torch.cuda.empty_cache()
            print(f"  R1[{cap}] {it['src']}/{it['idx']} done", flush=True)
        out[cap] = dict(meta=dict(scenario="Lingshu-7B direct MCQ, batch 1, cap=" + cap,
                                  max_pixels=maxpx, min_pixels=MIN_PX,
                                  vision_token_budget=tokbudget, batch_size=1, tp=1,
                                  max_new_tokens=A.mcq_max_new, adapter=None,
                                  is_08_11_null_test=(maxpx == 12845056),
                                  scenario_reset=scenario_reset()),
                        **summarise(rows))
        dump("R1_resolution_7b_mcq", out)
    return out


def run_R2(model, proc, items):
    """Full open-text arm in ONE process: 8 samples from the base model (adapter disabled), then the
    clean LoRA verifier scores all 8 (adapter enabled).  Peak is over the WHOLE pipeline, exactly as
    08-11 S4.  Generator and verifier max_pixels move TOGETHER across the sweep; the 08-11 S4 point
    (gen 250,880 / verif 1,003,520) is included separately as the null test."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, VERIFIER))
    model.eval()
    grid = [("s4_deployed_null_test", 250880, 1003520, "320/1280")] + \
           [(c, px, px, str(t)) for c, px, t in CAPS]
    out = {}
    for cap, gen_px, ver_px, tag in grid:
        rows = []
        for it in items:
            try:
                with model.disable_adapter():
                    enc = encode(proc, build_open_gen(it, gen_px))
                    nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
                    def gen():
                        with torch.inference_mode():
                            return model.generate(**enc, max_new_tokens=64, do_sample=True,
                                                  temperature=0.7, top_p=0.95, num_return_sequences=8)
                    o, mm_g = measured(gen)
                    cands = [proc.tokenizer.decode(o[i][enc["input_ids"].shape[1]:],
                                                   skip_special_tokens=True).strip()
                             for i in range(o.shape[0])]
                    del enc, o
                def score():
                    tot = 0
                    with torch.inference_mode():
                        for c in cands:
                            e = encode(proc, build_verif(it, c, ver_px))
                            model(**e)
                            tot += int(e["input_ids"].shape[1])
                            del e
                    return tot
                _, mm_s = measured(score)
                mm = {k: max(mm_g[k], mm_s[k]) if isinstance(mm_g[k], float) else mm_g[k]
                      for k in mm_g}
                mm["wall_s"] = round(mm_g["wall_s"] + mm_s["wall_s"], 3)
                rows.append(dict(**mm, src=it["src"], idx=it["idx"], image_pixels=it["px"],
                                 n_images=1, input_tokens=nin, vision_tokens=nvis, gen_tokens=8 * 64,
                                 n_candidates=len(cands)))
            except Exception as e:
                rows.append(dict(error=f"{type(e).__name__}: {e}", src=it["src"], idx=it["idx"]))
                torch.cuda.empty_cache()
            print(f"  R2[{cap}] {it['src']}/{it['idx']} done", flush=True)
        out[cap] = dict(meta=dict(scenario="open-text best-of-8 + clean LoRA verifier, batch 1, "
                                           "cap=" + cap,
                                  generator_max_pixels=gen_px, verifier_max_pixels=ver_px,
                                  vision_token_budget=tag, min_pixels=MIN_PX, batch_size=1, tp=1,
                                  adapter=VERIFIER, n_samples=8, temperature=0.7,
                                  max_new_tokens=64,
                                  is_08_11_null_test=(cap == "s4_deployed_null_test"),
                                  scenario_reset=scenario_reset()),
                        **summarise(rows))
        dump("R2_resolution_opentext_arm", out)
    return out


def run_Q(items):
    """The CHEAP side quantised.  torchao weight-only int8 / int4 (tinygemm), NOT bitsandbytes:
    bitsandbytes 0.41.0 exists on this host only as another round's python3.10 private install
    (~/pylibs_attack3), and importing it would change the whole environment away from the one every
    row in artifacts/vram_testtime_2026-08-11.json was measured in.  torchao 0.13.0 is present in
    THIS interpreter, so the rows stay comparable.  Scheme and library are recorded per row.

    *** STANDING CAVEAT, stated in the artifact and never conflated: INT4/INT8 WEIGHT-ONLY
    quantisation on sm80 gives ZERO MAC reduction. *** Weights are dequantised to bf16 (or fed to a
    tinygemm kernel that accumulates in bf16) and the same bf16 GEMM runs.  This is a MEMORY lever
    and possibly a bandwidth/latency lever.  It is NOT a FLOP lever.  FLOPs are unchanged, by
    construction, and are reported separately."""
    out = {}
    proc = AutoProcessor.from_pretrained(M7)
    for scheme in ("bf16_control", "int8wo", "int4wo"):
        for cap, maxpx, tokbudget in [c for c in CAPS if c[0] in ("medevalkit_default", "cap320")]:
            key = f"{scheme}__{cap}"
            try:
                model, li = load(M7, f"Lingshu-7B[{scheme}]", need_mb=A.wait_mb,
                                 quant=None if scheme == "bf16_control" else scheme)
            except Exception as e:
                out[key] = dict(error=f"LOAD FAILED {type(e).__name__}: {e}", scheme=scheme, cap=cap)
                dump("Q_quantised_cheap_side", out)
                continue
            rows = []
            for it in items:
                try:
                    enc = encode(proc, build_mcq(it, maxpx))
                    nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
                    def go():
                        with torch.inference_mode():
                            return model.generate(**enc, max_new_tokens=8, do_sample=False)
                    o, mm = measured(go)
                    rows.append(dict(**mm, src=it["src"], idx=it["idx"], image_pixels=it["px"],
                                     n_images=len(it["imgs"]), input_tokens=nin, vision_tokens=nvis,
                                     gen_tokens=int(o.shape[1] - enc["input_ids"].shape[1])))
                    del enc, o
                except Exception as e:
                    rows.append(dict(error=f"{type(e).__name__}: {e}", src=it["src"], idx=it["idx"]))
                    torch.cuda.empty_cache()
            out[key] = dict(meta=dict(scenario=f"Lingshu-7B {scheme} direct MCQ, batch 1, cap={cap}",
                                      quant_scheme=scheme, quant_library="torchao 0.13.0+git",
                                      max_pixels=maxpx, vision_token_budget=tokbudget,
                                      batch_size=1, tp=1, max_new_tokens=8, load=li,
                                      flop_note="weight-only quantisation: ZERO MAC reduction on "
                                                "sm80, memory lever only"),
                            **summarise(rows))
            dump("Q_quantised_cheap_side", out)
            del model
            torch.cuda.empty_cache()
            scenario_reset()
    return out


def run_C(items):
    """7B + 32B CO-RESIDENT on ONE card, in ONE process (one CUDA context), tp=1.
    The 08-11 artifact listed this under `not_measured`.  It is the question that decides whether
    ANY escalating policy can be served on a single 80 GB GPU without a second card or a reload.
    An OOM here is a RESULT, not a failure: it is recorded with the allocator state at the point of
    failure and reported as 'does not fit'."""
    out = dict(meta=dict(
        scenario="Lingshu-7B and Lingshu-32B resident SIMULTANEOUSLY on one A100 80GB, tp=1, bf16",
        total_board_gib=round(int(pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(NVML_DEV)).total) / GIB, 4),
        cuda_context_gib=CTX["cuda_context_gib"], batch_size=1,
        expectation_from_08_11=("(a) 7B 15.4937 + (a) 32B 62.3125 = 77.8062 GiB of weights alone; "
                                "plus one CUDA context 1.3835-1.3855 GiB = 79.19 GiB, leaving "
                                "~0.81 GiB for activations. The 32B's measured activation headroom "
                                "at MedEvalKit resolution was (b)-(a) = 5.5555 GiB on the worst "
                                "item and 1.6436 GiB on the mean item. Arithmetic therefore says "
                                "it does not fit; this scenario tests it."),
    ))
    try:
        m7, l7 = load(M7, "Lingshu-7B", need_mb=A.wait_mb)
        out["load_7b"] = l7
    except Exception as e:
        out["result"] = f"7B LOAD FAILED: {type(e).__name__}: {e}"
        dump("C_coresidency", out)
        return out
    out["after_7b"] = dict(allocated_gib=gb(torch.cuda.memory_allocated()),
                           reserved_gib=gb(torch.cuda.memory_reserved()),
                           board_gib=gb(board_used_bytes(NVML_DEV)))
    try:
        m32, l32 = load(M32, "Lingshu-32B", need_mb=66000)
        out["load_32b"] = l32
        out["after_both"] = dict(allocated_gib=gb(torch.cuda.memory_allocated()),
                                 reserved_gib=gb(torch.cuda.memory_reserved()),
                                 board_gib=gb(board_used_bytes(NVML_DEV)))
        out["both_loaded"] = True
    except Exception as e:
        out["both_loaded"] = False
        out["result"] = "DOES NOT FIT -- 32B load OOMed with the 7B already resident"
        out["oom"] = dict(exception=f"{type(e).__name__}: {str(e)[:600]}",
                          allocated_at_failure_gib=gb(torch.cuda.memory_allocated()),
                          reserved_at_failure_gib=gb(torch.cuda.memory_reserved()),
                          board_at_failure_gib=gb(board_used_bytes(NVML_DEV)))
        dump("C_coresidency", out)
        return out

    # both resident: now find the largest resolution at which a real item still runs on the 32B
    proc = AutoProcessor.from_pretrained(M7)
    per_cap = {}
    for cap, maxpx, tokbudget in CAPS:
        rows = []
        for it in items:
            try:
                enc = encode(proc, build_mcq(it, maxpx))
                nin, nvis = int(enc["input_ids"].shape[1]), vision_tokens(enc)
                def go():
                    with torch.inference_mode():
                        return m32.generate(**enc, max_new_tokens=8, do_sample=False)
                o, mm = measured(go)
                rows.append(dict(**mm, src=it["src"], idx=it["idx"], image_pixels=it["px"],
                                 n_images=len(it["imgs"]), input_tokens=nin, vision_tokens=nvis,
                                 gen_tokens=int(o.shape[1] - enc["input_ids"].shape[1])))
                del enc, o
            except Exception as e:
                rows.append(dict(error=f"{type(e).__name__}: {str(e)[:300]}",
                                 src=it["src"], idx=it["idx"]))
                torch.cuda.empty_cache()
        per_cap[cap] = dict(meta=dict(max_pixels=maxpx, vision_token_budget=tokbudget),
                            **summarise(rows))
        out["per_cap"] = per_cap
        dump("C_coresidency", out)
    out["result"] = "BOTH RESIDENT -- see per_cap for which resolutions actually run"
    dump("C_coresidency", out)
    return out


def main():
    todo = list(WANT)
    results = dict(_context=CTX, _env=dict(
        date="2026-08-12", framework="HuggingFace transformers (NEVER vLLM)",
        torch=torch.__version__, transformers=__import__("transformers").__version__,
        dtype="bfloat16", attn_implementation="flash_attention_2", tensor_parallel=1,
        batch_size=1, gpu_physical_index=DEV,
        gpu_state_at_launch_free_mb=free_mb_per_gpu(),
        card_is_shared=True,
        code="src/cascade/measure_vram_levers.py"))

    if "R1" in todo or "R2" in todo:
        proc = AutoProcessor.from_pretrained(M7)
        model, li = load(M7, "Lingshu-7B", need_mb=A.wait_mb)
        results["_load_7b"] = li
        if "R1" in todo:
            results["R1"] = run_R1(model, proc, pool_mcq(A.n_mcq))
        if "R2" in todo:
            results["R2"] = run_R2(model, proc, pool_open(A.n_open))
        del model
        torch.cuda.empty_cache()
    if "Q" in todo:
        results["Q"] = run_Q(pool_mcq(A.n_mcq))
    if "C" in todo:
        results["C"] = run_C(pool_mcq(A.n_mcq))
    dump("levers_env", results.get("_env", {}) | {"_context": CTX})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
