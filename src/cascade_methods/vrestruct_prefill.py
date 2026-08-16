#!/usr/bin/env python3
"""vrestruct_prefill.py -- SETTLE whether vLLM's SamplingParams(n=N) shares the generation prefill.

WHY THIS EXISTS.  The project's cost model charges 1.0 FLOP-eq per generated candidate (8.0 for
N=8).  Convention B of artifacts/cost_floor_2026-08-10.json instead assumes vLLM forks N sequences
from ONE prefill, giving G(N) = prefill_share + N*decode_share = 0.988456 + N*0.011544 (so
G(8) = 1.0808, not 8.0).  That is an ~8x swing in the headline and it was NEVER measured: the three
_cost_floor_measure/vllm_*.jsonl files are 0 bytes and num_cached_tokens was never captured.

INSTRUMENTS (all on the SAME requests):

  I1  DIRECT TOKEN ACCOUNTING (ground truth).  A forward pre-hook on the LM's first decoder layer
      records the HIDDEN-STATE tensor's leading dimension = the number of token positions in that
      forward; a pre-hook on the vision tower records patches.  Summed over one generate() call
      this is exactly the work the GPU did, independent of what vLLM reports.
        lm_positions_total = prefill positions + decode positions
        decode positions   = gen_tok_total  (vLLM emits exactly one position per sequence per step)
        => prefill_positions = lm_positions_total - gen_tok_total
      Requires the engine core IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0) and, crucially,
      enforce_eager=True -- CUDA-graph-captured decode steps do NOT fire Python forward hooks.

  I2  vLLM's OWN ACCOUNTING: RequestOutput.num_cached_tokens (present in 0.9.0.1).

  I3  WALL CLOCK vs N, measured SEPARATELY with CUDA graphs ON (the deployed configuration), since
      enforce_eager distorts decode latency.

TWO TRAPS THAT INVALIDATED THE FIRST ATTEMPT AND ARE FIXED HERE
  (a) `positions` in a Qwen2.5-VL vLLM decoder layer is M-ROPE shaped (3, num_tokens), so grabbing
      "the first tensor" counted 3 per forward.  We now select the 2-D tensor whose trailing dim is
      the LM hidden size.
  (b) Re-using the SAME 24 prompts for every (apc, N, rep) cell let automatic prefix caching serve
      the whole prompt from the previous cell -- 7584/7816 tokens cached, i.e. the experiment
      measured its own warm cache.  Every cell now gets a DISJOINT slice of items and the prefix
      cache is explicitly reset before each cell.

Workload = the DEPLOYED one: Lingshu-7B, cap320 (max_pixels 250880), SYS direct prompt, real eval
images, max_tokens 64, temperature 0.7 -- prompt construction copied from run_openvqa.py.

    HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      /data/dan/medeval_venv/bin/python src/cascade_methods/vrestruct_prefill.py
"""
from __future__ import annotations

import argparse
import json
import os
import time

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_vrestruct_parts")

MODEL = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
         "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
SYS = ("You are an expert medical image analyst. Answer the question with a short, specific phrase. "
       "Do not explain.")
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
MAXPX = HIGH_PX // 4          # cap320, the deployed operating point
HIDDEN = 3584                 # Lingshu-7B LM hidden size


# ---------------------------------------------------------------- instrumentation
class Counter:
    def __init__(self):
        self.reset()
        self.handles = []
        self.attached = {"lm": None, "vit": None, "lm_path": None}

    def reset(self):
        self.lm_positions = 0
        self.lm_calls = 0
        self.lm_shapes = []
        self.vit_patches = 0
        self.vit_calls = 0

    def attach(self, model):
        import torch

        lm_layer, lm_path, vit = _find_modules(model)

        if lm_layer is not None:
            def lm_hook(mod, args, kwargs):
                best = None
                for a in list(args) + list(kwargs.values()):
                    if torch.is_tensor(a) and a.dim() == 2 and a.shape[-1] == HIDDEN:
                        best = a
                        break
                if best is None:                       # fall back: largest 2-D tensor
                    for a in list(args) + list(kwargs.values()):
                        if torch.is_tensor(a) and a.dim() == 2:
                            if best is None or a.numel() > best.numel():
                                best = a
                if best is not None:
                    self.lm_positions += int(best.shape[0])
                    self.lm_calls += 1
                    if len(self.lm_shapes) < 200:
                        self.lm_shapes.append(int(best.shape[0]))
            self.handles.append(lm_layer.register_forward_pre_hook(lm_hook, with_kwargs=True))
            self.attached["lm"] = type(lm_layer).__name__
            self.attached["lm_path"] = lm_path

        if vit is not None:
            def vit_hook(mod, args, kwargs):
                t = args[0] if args else kwargs.get("pixel_values", kwargs.get("x"))
                if torch.is_tensor(t) and t.dim() >= 1:
                    self.vit_patches += int(t.shape[0])
                    self.vit_calls += 1
            self.handles.append(vit.register_forward_pre_hook(vit_hook, with_kwargs=True))
            self.attached["vit"] = type(vit).__name__
        return self.attached

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def _find_modules(model):
    lm_layer = lm_path = vit = None
    for path in ("language_model.model.layers", "model.language_model.layers", "model.layers",
                 "language_model.layers"):
        o = model
        try:
            for p in path.split("."):
                o = getattr(o, p)
            if len(o) > 0:
                lm_layer, lm_path = o[0], path
                break
        except Exception:
            continue
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        for name in ("visual", "vision_tower", "vision_model"):
            if hasattr(holder, name):
                vit = getattr(holder, name)
                break
        if vit is not None:
            break
    return lm_layer, lm_path, vit


def _get_model(llm):
    e = llm.llm_engine
    for path in (
        "model_executor.driver_worker.model_runner.model",
        "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
        "engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model",
        "engine_core.model_executor.driver_worker.model_runner.model",
    ):
        o, ok = e, True
        for p in path.split("."):
            if not hasattr(o, p):
                ok = False
                break
            o = getattr(o, p)
        if ok:
            return path, o
    return None, None


def _reset_prefix_cache(llm):
    for target in (llm, getattr(llm, "llm_engine", None)):
        if target is not None and hasattr(target, "reset_prefix_cache"):
            try:
                target.reset_prefix_cache()
                return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------- data
def load_items(n_items, seed=20260816):
    import glob
    import io
    import random

    import pandas as pd
    from PIL import Image

    out = []
    d = json.load(open("/data/dan/dataset/slake/test.json"))
    for x in d:
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                out.append(("slake_open", x["qid"], x["question"], ip))
    dfs = [pd.read_parquet(f) for f in
           sorted(glob.glob("/data/dan/dataset/path_vqa/data/test-*.parquet"))]
    df = pd.concat(dfs, ignore_index=True)
    for i, r in df.iterrows():
        a = str(r.get("answer")).strip()
        if a.lower() in ("yes", "no"):
            continue
        img = r["image"]
        if isinstance(img, dict) and "bytes" in img:
            out.append(("pathvqa_open", int(i), str(r["question"]),
                        Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
        if len(out) > n_items * 3:
            break
    random.Random(seed).shuffle(out)
    return out[:n_items]


def build_reqs(items, proc):
    from qwen_vl_utils import process_vision_info
    reqs = []
    for (_ds, _idx, q, img) in items:
        im = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX}]
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": im + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        r = {"prompt": text}
        if imgs:
            r["multi_modal_data"] = {"image": imgs}
        reqs.append(r)
    return reqs


def _gpu_mem_util(reserve_mib=22000):
    """The GPU is SHARED with a concurrent round whose footprint moves between 19 and 36 GB.

    vLLM's gpu_memory_utilization is a fraction of TOTAL memory and it subtracts whatever other
    processes already hold, so a fixed fraction fails whenever the neighbour grows.  Size it from
    the ACTUAL free memory immediately before the load and reserve only what we need.
    """
    import subprocess
    o = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True, timeout=10).strip().splitlines()
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    used, total = (float(x) for x in o[int(dev)].split(","))
    if total - used < reserve_mib + 2000:
        raise RuntimeError(f"only {total-used:.0f} MiB free on GPU {dev}; need "
                           f"{reserve_mib+2000}. QUEUEING rather than oversubscribing.")
    util = (used + reserve_mib) / total
    print(f"GPU{dev}: {used:.0f}/{total:.0f} MiB used by others -> "
          f"gpu_memory_utilization={util:.4f} (reserving {reserve_mib} MiB)", flush=True)
    return min(util, 0.95)


def _power_w():
    import subprocess
    try:
        o = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        return sum(float(x) for x in o.strip().splitlines() if x.strip())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cell", type=int, default=16, help="DISJOINT items per (apc,N,rep) cell")
    ap.add_argument("--reserve_mib", type=int, default=22000)
    ap.add_argument("--only_phase", default=None, choices=[None, "count", "time"])
    ap.add_argument("--only_apc", default=None)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--apcs", default="default,on,off",
                    help="'default' does NOT pass enable_prefix_caching at all -- that is what "
                         "src/labeling/run_openvqa.py:152 does, so it is the arm that describes "
                         "every generation this project has actually run")
    A = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    sink = os.path.join(OUT, f"prefill{A.tag}.jsonl")
    done = set()
    if os.path.exists(sink):
        for l in open(sink):
            if l.strip():
                try:
                    r = json.loads(l)
                    done.add((r["phase"], r["apc"], r["N"], r["rep"]))
                except Exception:
                    pass

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    import vllm

    NS = (1, 2, 4, 8)
    PHASES = (("count", True), ("time", False))       # (phase, enforce_eager)
    APCS = tuple(x.strip() for x in A.apcs.split(",") if x.strip())
    cells = [(ph, apc, N, rep) for ph, _ in PHASES for apc in APCS for N in NS
             for rep in range(1, A.reps + 1)]
    need = A.per_cell * len(cells) + 8
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    pool = load_items(need)
    print(f"pool {len(pool)} items for {len(cells)} disjoint cells x {A.per_cell}", flush=True)
    reqs_all = build_reqs(pool, proc)
    warm = reqs_all[:8]
    body = reqs_all[8:]
    slices = {c: body[i * A.per_cell:(i + 1) * A.per_cell] for i, c in enumerate(cells)}

    meta = {"model": MODEL, "cap": "cap320", "max_pixels": MAXPX, "per_cell": A.per_cell,
            "max_tokens": A.max_tokens, "temp": A.temp, "vllm_version": vllm.__version__,
            "vllm_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "design": "every (phase,apc,N,rep) cell gets a DISJOINT slice of items and the prefix "
                      "cache is reset before it, so no cell can be served from another cell's cache",
            "phases": {"count": "enforce_eager=True -- hooks see every forward; wall clock NOT "
                                "representative", "time": "CUDA graphs on (deployed config) -- wall "
                                "clock representative; hooks miss graph-captured decode"}}

    for phase, eager in PHASES:
        if A.only_phase and phase != A.only_phase:
            continue
        for apc in APCS:
            if A.only_apc and apc != A.only_apc:
                continue
            todo = [c for c in cells if c[0] == phase and c[1] == apc and c not in done]
            if not todo:
                continue
            kw = dict(model=MODEL, tensor_parallel_size=1, dtype="bfloat16",
                      gpu_memory_utilization=_gpu_mem_util(A.reserve_mib), max_model_len=8192,
                      limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
                      enforce_eager=eager)
            if apc != "default":                 # 'default' = do not pass the flag at all
                kw["enable_prefix_caching"] = (apc == "on")
            llm = LLM(**kw)
            try:
                eff_apc = llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
            except Exception:
                eff_apc = None
            path, model = _get_model(llm)
            cnt = Counter()
            attached = cnt.attach(model) if model is not None else {"lm": None, "vit": None}
            meta["model_path_in_engine"] = path
            meta["hooks"] = attached
            print(f"[{phase}/apc={apc}] hooks={attached}", flush=True)
            llm.generate(warm, SamplingParams(temperature=A.temp, max_tokens=8, n=2))

            for c in todo:
                _, _, N, rep = c
                reset_ok = _reset_prefix_cache(llm)
                sp = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, n=N,
                                    logprobs=5, seed=20260816 + rep)
                cnt.reset()
                p0 = _power_w()
                t0 = time.time()
                outs = llm.generate(slices[c], sp)
                dt = time.time() - t0
                p1 = _power_w()
                gen_tok = sum(len(o_.token_ids) for o in outs for o_ in o.outputs)
                prompt_tok = sum(len(o.prompt_token_ids) for o in outs)
                cached = [getattr(o, "num_cached_tokens", None) for o in outs]
                cached_sum = sum(x for x in cached if isinstance(x, int))
                rec = dict(phase=phase, apc=apc, effective_enable_prefix_caching=eff_apc,
                           engine=os.environ.get("VLLM_USE_V1", "1"),
                           N=N, rep=rep, n_items=len(slices[c]),
                           enforce_eager=eager, prefix_cache_reset=reset_ok,
                           wall_s=dt, mean_power_w=(p0 + p1) / 2.0,
                           energy_j_est=dt * (p0 + p1) / 2.0,
                           prompt_tok_total=prompt_tok, gen_tok_total=gen_tok,
                           num_cached_tokens_total=cached_sum,
                           lm_positions=cnt.lm_positions, lm_calls=cnt.lm_calls,
                           lm_shapes_head=cnt.lm_shapes[:40],
                           vit_patches=cnt.vit_patches, vit_calls=cnt.vit_calls,
                           lm_prefill_positions=(cnt.lm_positions - gen_tok
                                                 if cnt.lm_calls > 0 else None),
                           ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
                with open(sink, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(f"  {phase} apc={apc} N={N} rep={rep}  wall={dt:.2f}s "
                      f"lm_pos={cnt.lm_positions} (calls {cnt.lm_calls}) vit={cnt.vit_patches} "
                      f"prompt={prompt_tok} gen={gen_tok} cached={cached_sum}", flush=True)

            cnt.detach()
            del llm
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

    json.dump(meta, open(os.path.join(OUT, f"prefill{A.tag}.meta.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
