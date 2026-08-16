#!/usr/bin/env python3
"""verifier_hparams_vram.py -- KNOB 3: VRAM + clean batch-1 latency for the verifier at each rung
of the scoring-resolution ladder.

CONVENTIONS -- the four of results/cascade_methods/artifacts/vram_testtime_2026-08-11.json, so the
rows are directly comparable to that artifact's S3 (which IS the 1,003,520 rung):

  (a) a_weights_resident   torch.cuda.memory_allocated() right after load + adapter (a FLOOR)
  (b) b_peak_allocated     torch.cuda.max_memory_allocated(), reset before EVERY item
  (c) c_peak_reserved      torch.cuda.max_memory_reserved(), reset before EVERY item
  (d) d_process_footprint  max board `used` over a 20 ms NVML sampler, minus the pre-run baseline
                           (NVML reports HOST pids in this container, so per-process is 0)
  units: GiB = bytes / 1024**3

ITEMS: the SAME 12 items as S3, in the same order -- chosen there to bracket the driver space
(57,600 -> 1,341,440 image pixels). The candidate string scored is each item's slot-0 answer from
the deployed transfer dump, so the prompt is a real deployed verifier prompt.

Requires an IDLE card: (d) is board-minus-baseline and is meaningless with a co-tenant. The script
refuses to start if the target GPU is not essentially empty, and re-asserts exclusivity at the end.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
      src/cascade_methods/verifier_hparams_vram.py
"""
import argparse
import json
import os
import threading
import time

import numpy as np
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
import sys                                                              # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
from verifier_hparams_score import (MINPX, SYS, imgs_for, load_dump_items)   # noqa: E402

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_verifier_hparams_parts")
GIB = 1024 ** 3
LADDER = [62720, 125440, 250880, 376320, 501760, 1003520, 12845056]
# 376,320 (cap480) is the EXPLORATORY knee rung added after the pre-registered ladder showed a
# STEP between 250,880 and 501,760; it is not part of the pre-registered set.  All seven rungs
# are measured in ONE session against ONE pre-run baseline so the rows stay comparable, and the
# pre-registered six-rung session is preserved at
# _verifier_hparams_parts/_prereg_6rung_backup/vram_latency.json for a session-to-session check.
# the S3 bracket, verbatim from vram_testtime_2026-08-11.json scenarios.S3...rows
S3_ITEMS = [("slake_open", 12615), ("slake_open", 12618), ("slake_open", 12532),
            ("slake_open", 12533), ("vqa_rad_open", 133), ("vqa_rad_open", 134),
            ("vqa_rad_open", 394), ("vqa_rad_open", 395), ("pathvqa_open", 424),
            ("pathvqa_open", 324), ("pathvqa_open", 7), ("pathvqa_open", 8)]


def gb(x):
    return round(x / GIB, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--reps", type=int, default=5, help="timed repeats per item (median reported)")
    ap.add_argument("--allow_busy_gib", type=float, default=1.0)
    A = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    os.makedirs(OUT, exist_ok=True)

    import pynvml
    pynvml.nvmlInit()
    dev = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
    H = pynvml.nvmlDeviceGetHandleByIndex(dev)

    def board():
        return int(pynvml.nvmlDeviceGetMemoryInfo(H).used)

    BASELINE = board()
    print(f"pre-run board used on GPU{dev}: {gb(BASELINE)} GiB", flush=True)
    if BASELINE > A.allow_busy_gib * GIB:
        print("REFUSING: target GPU is not idle; (d) would be contaminated. "
              "Wait for the card and re-run.", flush=True)
        return

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
            self.mx = max(self.mx, board())
            return self.mx

    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to("cuda")
    torch.cuda.synchronize()
    bare = torch.cuda.memory_allocated()
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    torch.cuda.synchronize()
    co_res = torch.cuda.memory_allocated()
    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    n_vis = sum(1 for n, _ in model.named_parameters() if "lora_" in n and "visual." in n)
    print(f"bare 7B resident {gb(bare)} GiB | +adapter {gb(co_res)} GiB "
          f"(marginal {gb(co_res-bare)}) | lora {n_lora} ({n_vis} visual)", flush=True)

    # the 12 bracket items + their slot-0 candidate string
    ans0 = {(it["ds"], it["idx"]): it["preds"][0] for it in load_dump_items()}
    IMG = {}
    for ds in {d for d, _ in S3_ITEMS}:
        IMG[ds] = imgs_for(ds)

    def build(ds, idx, maxpx):
        q, img = IMG[ds][idx]
        m = [{"role": "system", "content": SYS},
             {"role": "user", "content": [
                 {"type": "image", "image": img, "max_pixels": maxpx, "min_pixels": MINPX},
                 {"type": "text", "text": f"Question: {q}\nProposed answer: {ans0[(ds, idx)]}\n"
                                          f"Is the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(m)
        return proc(text=[text], images=igs, videos=vids, return_tensors="pt",
                    padding=True).to("cuda")

    res = {"_meta": {
        "title": "verifier scoring-resolution ladder: VRAM (4 conventions) + clean batch-1 latency",
        "conventions": "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json "
                       "(a)-(d), units GiB = bytes/1024**3",
        "comparable_row_in_that_artifact": "S3_lingshu7b_plus_lora_verifier IS the 1,003,520 rung",
        "items": "the same 12 bracket items as S3, same order",
        "framework": "HuggingFace transformers, bf16, flash_attention_2, tp=1, batch 1, "
                     "TF32 OFF, min_pixels 3,136. NEVER vLLM.",
        "torch": torch.__version__,
        "gpu": pynvml.nvmlDeviceGetName(H),
        "pre_run_board_baseline_gib": gb(BASELINE),
        "a_weights_resident_gib_bare_7b": gb(bare),
        "a_weights_resident_gib_with_adapter": gb(co_res),
        "marginal_adapter_resident_gib": gb(co_res - bare),
        "n_lora_params": n_lora, "n_lora_params_on_visual": n_vis,
        "reps_per_item": A.reps,
        "_latency_note": "wall time of the forward only (encode excluded, CUDA-synchronized), "
                         "median over reps then aggregated over items. Batch 1 = the deployed "
                         "verifier. This is a per-CANDIDATE cost; the deployed open arm runs 8.",
    }, "by_max_pixels": {}}

    for px in LADDER:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        rows = []
        for ds, idx in S3_ITEMS:
            try:
                enc = build(ds, idx, px)
                nin = int(enc["input_ids"].shape[1])
                npatch = int(enc["pixel_values"].shape[0])
                torch.cuda.reset_peak_memory_stats()
                s = Sampler()
                s.start()
                lat = []
                for _ in range(A.reps):
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.no_grad():
                        lg = model(**enc).logits[0, -1]
                        _ = (float(lg[YES]), float(lg[NO]))
                    torch.cuda.synchronize()
                    lat.append(time.time() - t0)
                mb = s.stop()
                rows.append(dict(ds=ds, idx=idx, input_tokens=nin,
                                 vision_tokens=npatch // 4, premerge_patches=npatch,
                                 b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                                 c_peak_reserved_gib=gb(torch.cuda.max_memory_reserved()),
                                 d_board_used_gib=gb(mb),
                                 d_process_footprint_gib=gb(max(0, mb - BASELINE)),
                                 lat_median_s=float(np.median(lat)),
                                 lat_min_s=float(np.min(lat))))
                del enc
            except Exception as e:                      # per-item error guard
                rows.append(dict(ds=ds, idx=idx, error=f"{type(e).__name__}: {str(e)[:180]}"))
                print(f"  FAIL {px} {ds} {idx}: {type(e).__name__} {e}", flush=True)
        ok = [r for r in rows if "error" not in r]

        def agg(k):
            v = np.array([r[k] for r in ok], float)
            return {"mean": float(v.mean()), "peak": float(v.max()), "min": float(v.min())}
        res["by_max_pixels"][str(px)] = {
            "max_pixels": px, "n": len(ok), "n_failed": len(rows) - len(ok),
            "a_weights_resident_gib": gb(co_res),
            "b_peak_allocated_gib": agg("b_peak_allocated_gib"),
            "c_peak_reserved_gib": agg("c_peak_reserved_gib"),
            "d_process_footprint_gib": agg("d_process_footprint_gib"),
            "d_board_used_gib": agg("d_board_used_gib"),
            "vision_tokens": agg("vision_tokens"),
            "input_tokens": agg("input_tokens"),
            "latency_batch1_s": agg("lat_median_s"),
            "rows": rows}
        a = res["by_max_pixels"][str(px)]
        print(f"  px{px:>9}  vis_tok {a['vision_tokens']['mean']:7.1f}  "
              f"b {a['b_peak_allocated_gib']['mean']:.3f}  c {a['c_peak_reserved_gib']['mean']:.3f}  "
              f"d {a['d_process_footprint_gib']['mean']:.3f} GiB  "
              f"lat {a['latency_batch1_s']['mean']*1000:.1f} ms", flush=True)

    post = board()
    res["_meta"]["post_run_board_used_gib"] = gb(post)
    res["_meta"]["exclusivity_asserted"] = "pre-run baseline and post-run board reading both " \
                                           "recorded; (d) = board max minus pre-run baseline"
    json.dump(res, open(os.path.join(OUT, "vram_latency.json"), "w"), indent=1, default=float)
    print(f"\nwrote {OUT}/vram_latency.json")


if __name__ == "__main__":
    main()
