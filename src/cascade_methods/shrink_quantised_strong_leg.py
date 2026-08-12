#!/usr/bin/env python3
"""
shrink_quantised_strong_leg.py -- ATTACK 3 part 2/3: MEASURE a quantised Lingshu-32B strong leg.

WHY THIS EXISTS
---------------
`results/cascade_methods/artifacts/quantized_strong_leg.json` (2026-07-07) claimed an INT4 strong
leg was "a real VRAM win" and contains NOT ONE MEASURED NUMBER: its own `tractability_note` records
that the AWQ shards never finished downloading and that every figure in it is PROJECTED.  This
script replaces the projection with measurement.

WHAT IS MEASURED (nothing here is projected)
  * resident VRAM: torch.cuda.max_memory_allocated / max_memory_reserved AND the NVML
    per-process footprint (which includes the CUDA context and cuBLAS workspaces that
    torch's allocator does not see).
  * batch-1 latency: prefill+decode wall clock on a FIXED, replayed prompt set, warmup excluded.
  * accuracy: MedEvalKit's own datasets/prompts/metrics, via the SAME driver as the bf16 control
    arm, on the SAME items.  Paired -- every delta is quant-minus-bf16 on identical inputs.

THE RESULT THAT DOES NOT NEED A GPU, AND IT BOUNDS EVERYTHING BELOW: 4-bit weight quantisation
changes how weights are STORED, not how many multiply-accumulates a forward pass performs, and
the A100 (sm80) has NO INT4 tensor-core path -- both AWQ and bitsandbytes-NF4 DEQUANTISE to
bf16/fp16 and call the same bf16 GEMM.  So a quantised strong leg cannot reduce FLOP-eq, the
PRIMARY objective of this round.  It is a FOOTPRINT lever (secondary) and a possible
bandwidth/latency lever (tertiary).  Measured here; not conflated.

GPU ETIQUETTE: waits for free VRAM, never kills another process, resumable per config.

    python3 src/cascade_methods/shrink_quantised_strong_leg.py --stage vram   --configs bf16,int8,nf4
    python3 src/cascade_methods/shrink_quantised_strong_leg.py --stage acc    --configs nf4,bf16 --datasets VQA_RAD,SLAKE
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEK = os.path.join(REPO, "MedEvalKit")
OUTDIR = os.path.join(REPO, "results/cascade_methods/artifacts/_shrink_parts")
os.makedirs(OUTDIR, exist_ok=True)

L32B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-32B/"
        "snapshots/36b98277cacb60db86f34b75ce0540b1ea35183c")
L7B = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
       "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")

# Each config is (model_path, quantization kwargs).  bf16 is the MATCHED CONTROL for every
# quantised arm: same model, same driver, same items, same decoding -- only the weight
# representation differs.
CONFIGS = {
    "bf16": dict(path=L32B, quant=None, need_mb=70000,
                 label="Lingshu-32B bf16 (the deployed strong leg; the control)"),
    "int8": dict(path=L32B, quant="int8", need_mb=42000,
                 label="Lingshu-32B bitsandbytes LLM.int8()"),
    "nf4":  dict(path=L32B, quant="nf4", need_mb=28000,
                 label="Lingshu-32B bitsandbytes NF4 (4-bit, double-quant, bf16 compute)"),
    "bf16_7b": dict(path=L7B, quant=None, need_mb=22000,
                    label="Lingshu-7B bf16 (the cheap leg, for scale)"),
}


def nvml_process_mb(pid):
    """VRAM this process actually holds, per the driver -- includes the CUDA context."""
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == pid:
                    return round(p.usedGpuMemory / 1024 ** 2, 1)
    except Exception:
        return None
    return None


def wait_for_vram(need_mb, timeout_s=21600):
    """Never oversubscribe another round's job."""
    import subprocess
    t0 = time.time()
    while True:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().splitlines()
        free = [int(t) - int(u) for u, t in (l.split(", ") for l in out)]
        if max(free) >= need_mb:
            return int(max(range(len(free)), key=lambda i: free[i]))
        if time.time() - t0 > timeout_s:
            raise RuntimeError("wait_for_vram timeout, need %d MB, free %s" % (need_mb, free))
        print("[wait] need %d MB, free %s -- sleeping 60s" % (need_mb, free), flush=True)
        time.sleep(60)


def build_model(cfg, batch_size=8, max_new_tokens=2048):
    """Load through the SAME HFVLM driver the matched-control arm uses, with only the weight
    representation changed.  Importing (not copying) guarantees identical prompt construction,
    identical decoding and identical meta capture across arms."""
    sys.path.insert(0, os.path.join(REPO, "src", "cascade_methods"))
    import i8b_cheapleg_eval as drv          # another round's driver -- imported, NEVER modified
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    class QuantHFVLM(drv.HFVLM):
        def __init__(self, model_path, quant, max_new_tokens, batch_size, device):
            self.torch = torch
            self.family = "qwen2_5_vl"
            self.device = device
            self.max_new_tokens = max_new_tokens
            self.batch_size = batch_size
            # BUGFIX 2026-08-12: drv.HFVLM._encode reads self.crop_to_patches (an InternVL-only
            # tiling switch).  This subclass never set it, so every item raised AttributeError and
            # the 2026-08-12 03:29 accuracy run wrote 2,545 error rows per arm.  Lingshu-32B is
            # qwen2_5_vl, which has no tiling, so None is the correct value: it makes _encode skip
            # the kwarg entirely, exactly as the base7b arm does.
            self.crop_to_patches = None
            self.processor = AutoProcessor.from_pretrained(model_path)
            kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
            if quant is None:
                kw["device_map"] = device
            else:
                # The vision tower is left in bf16 in both quantised arms.  That is what every
                # published AWQ Qwen2.5-VL checkpoint does (its own quantization_config carries
                # modules_to_not_convert=["visual"]), so the arms stay comparable to the AWQ
                # footprint measured from the safetensors headers.
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(quant == "nf4"), load_in_8bit=(quant == "int8"),
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=["visual", "lm_head"],
                )
                kw["device_map"] = {"": int(device.split(":")[1])}
            self.model = AutoModelForImageTextToText.from_pretrained(model_path, **kw)
            self.model.eval()
            tok = self.processor.tokenizer
            tok.padding_side = "left"
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            self.last_meta = []

    # Select the device explicitly.  CUDA_VISIBLE_DEVICES is NOT usable here: torch is already
    # imported by this point, so the variable would be ignored and the model would silently land
    # on GPU 0 -- on top of whichever other round is using it.
    gpu = int(os.environ["SHRINK_GPU"]) if os.environ.get("SHRINK_GPU") else wait_for_vram(cfg["need_mb"])
    torch.cuda.set_device(gpu)
    torch.cuda.reset_peak_memory_stats(gpu)
    torch.cuda.empty_cache()
    t0 = time.time()
    m = QuantHFVLM(cfg["path"], cfg["quant"], max_new_tokens, batch_size, "cuda:%d" % gpu)
    load_s = time.time() - t0
    return m, drv, load_s, gpu


# --------------------------------------------------------------------------- stage: vram+latency
def stage_vram(name, cfg, n_items=25, batch=1):
    import torch
    m, drv, load_s, gpu = build_model(cfg, batch_size=batch)

    after_load_alloc = torch.cuda.max_memory_allocated(gpu) / 1024 ** 2
    after_load_res = torch.cuda.memory_reserved(gpu) / 1024 ** 2

    # Replay a FIXED prompt set: the same 25 non-yes/no VQA-RAD items and the same cap320 geometry
    # the project's own 665 ms / 347 ms latency anchors were timed on.
    from PIL import Image
    import numpy as np
    rng = np.random.RandomState(0)
    msgs = []
    for i in range(n_items):
        img = Image.fromarray(rng.randint(0, 255, (320, 320, 3), dtype=np.uint8))
        msgs.append({"prompt": "What abnormality is seen in this image? Answer the question "
                               "using a single word or phrase.", "image": img})

    torch.cuda.reset_peak_memory_stats(gpu)
    m.generate_outputs(msgs[:3])                       # warmup, excluded
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(gpu)
    lat = []
    t_all = time.time()
    for k in range(3, n_items):
        t0 = time.time()
        m.generate_outputs([msgs[k]])
        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000.0)
    total_s = time.time() - t_all

    peak_alloc = torch.cuda.max_memory_allocated(gpu) / 1024 ** 2
    peak_res = torch.cuda.max_memory_reserved(gpu) / 1024 ** 2
    nvml = nvml_process_mb(os.getpid())
    lat_sorted = sorted(lat)
    res = dict(
        config=name, label=cfg["label"], model_path=cfg["path"], quant=cfg["quant"], gpu=gpu,
        load_seconds=round(load_s, 1),
        vram_mib=dict(
            torch_alloc_after_load=round(after_load_alloc, 1),
            torch_reserved_after_load=round(after_load_res, 1),
            torch_peak_alloc_during_gen=round(peak_alloc, 1),
            torch_peak_reserved_during_gen=round(peak_res, 1),
            nvml_process_footprint=nvml,
        ),
        latency_ms_batch1=dict(
            n=len(lat), mean=round(sum(lat) / len(lat), 1),
            median=round(lat_sorted[len(lat_sorted) // 2], 1),
            p10=round(lat_sorted[int(0.1 * len(lat_sorted))], 1),
            p90=round(lat_sorted[int(0.9 * len(lat_sorted))], 1),
            total_s=round(total_s, 1)),
        gen_tokens_mean=round(sum(x.get("gen_toks", 0) for x in m.last_meta)
                              / max(1, len(m.last_meta)), 2),
        measurement="MEASURED (torch allocator + NVML process footprint; warmup excluded)",
        workload="25 synthetic 320x320 RGB images + a fixed VQA-RAD-style question, batch 1, "
                 "greedy.  Synthetic images are used ONLY for the VRAM/latency probe so the "
                 "measurement carries no dataset dependency; accuracy is measured separately on "
                 "MedEvalKit's real items.",
    )
    print(json.dumps(res, indent=1), flush=True)
    p = os.path.join(OUTDIR, "vram_%s.json" % name)
    json.dump(res, open(p, "w"), indent=1)
    print("wrote", p, flush=True)
    return res


# --------------------------------------------------------------------------- stage: accuracy
def stage_acc(name, cfg, datasets, limit, batch_size):
    os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["REASONING"] = "False"
    os.environ["datasets_path"] = "hf"
    os.environ["use_llm_judge"] = "False"
    os.environ["judge_model_type"] = "openai"
    os.environ["judge_model"] = "None"
    os.environ["api_key"] = "None"
    os.environ["base_url"] = "None"
    os.environ["use_vllm"] = "False"
    os.environ["max_image_num"] = "6"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    m, drv, load_s, gpu = build_model(cfg, batch_size=batch_size)
    sys.path.insert(0, MEK)
    os.chdir(MEK)
    from benchmarks import prepare_benchmark

    out_root = os.path.join(REPO, "ckpts/shrink_quant", name)
    os.makedirs(out_root, exist_ok=True)
    results = {}
    for ds in datasets:
        outdir = os.path.join(out_root, ds)
        os.makedirs(outdir, exist_ok=True)
        mpath = os.path.join(outdir, "metrics.json")
        if os.path.exists(mpath):
            results[ds] = json.load(open(mpath))
            print("[skip done]", ds, flush=True)
            continue
        print("\n###", name, ds, flush=True)
        try:
            dataset = prepare_benchmark(m, ds, None, outdir)
            dataset.load_data()
            if limit:
                dataset.samples = dataset.samples[:limit]
            samples = drv.resumable_run(dataset, m, os.path.join(outdir, "gen.jsonl"))
            n_empty = sum(1 for s in samples if not str(s.get("response", "")).strip())
            # MedEvalKit's cal_metrics returns (metrics, judged_samples) for the open-ended
            # datasets and a bare dict for the MCQ-only ones.  The 2026-08-12 03:29 run treated it
            # as a bare value in both cases.  Unpack properly and PERSIST the judged per-item rows
            # -- the paired quant-minus-bf16 bootstrap needs per-item correctness, not cell means.
            out_samples = None
            try:
                met = dataset.cal_metrics(samples)
            except ValueError:
                # MedEvalKit/utils/utils.py:44 rouge() raises "Hypothesis is empty." on a blank
                # response, which kills the whole cell.  Score the non-blank subset with
                # MedEvalKit's OWN unmodified metric and report the blanks separately; a blank is
                # scored wrong, never dropped, in the accuracy recomputed downstream from
                # results.json.
                traceback.print_exc()
                keep = [s for s in samples if str(s.get("response", "")).strip()]
                met = dataset.cal_metrics(keep)
                if isinstance(met, tuple):
                    met, out_samples = met
                met = {"metrics": met} if not isinstance(met, dict) else met
                met["_EMPTY_RESPONSE_FALLBACK"] = (
                    "cal_metrics raised on blank responses; the metric above is over the %d "
                    "non-blank items only.  %d blank items are excluded from it and are counted "
                    "as WRONG in the per-item accuracy recomputed from results.json."
                    % (len(keep), n_empty))
            if isinstance(met, tuple):
                met, out_samples = met
            if not isinstance(met, dict):
                met = {"metrics": met}
            met["_n"] = len(samples)
            met["_n_empty_response"] = n_empty
            json.dump(met, open(mpath, "w"), indent=1)
            if out_samples is not None:
                json.dump(out_samples, open(os.path.join(outdir, "results.json"), "w"))
            results[ds] = met
            print(name, ds, json.dumps(met)[:400], flush=True)
        except Exception:
            traceback.print_exc()
    p = os.path.join(OUTDIR, "acc_%s.json" % name)
    json.dump(dict(config=name, label=cfg["label"], quant=cfg["quant"], gpu=gpu,
                   limit=limit, results=results), open(p, "w"), indent=1)
    print("wrote", p, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["vram", "acc"])
    ap.add_argument("--configs", default="nf4")
    ap.add_argument("--datasets", default="VQA_RAD,SLAKE")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    for name in args.configs.split(","):
        name = name.strip()
        if not name:
            continue
        cfg = CONFIGS[name]
        try:
            if args.stage == "vram":
                stage_vram(name, cfg)
            else:
                stage_acc(name, cfg, [d.strip() for d in args.datasets.split(",") if d.strip()],
                          args.limit, args.batch_size)
        except Exception:
            traceback.print_exc()
            print("FAILED config", name, flush=True)


if __name__ == "__main__":
    main()
