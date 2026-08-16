#!/usr/bin/env python3
"""prefix_shared_vram.py -- BUILD 2: VRAM (4 conventions) + clean batch-1 latency for the
DEPLOYED per-candidate scoring loop and the PREFIX-SHARED replacement, measured side by side in
one process on an IDLE card.

The unit is ONE QUESTION with its real pool of distinct candidate answers (the deployed unit of
work: 8,965 distinct raw answers over 2,345 questions = 3.823 verifier passes/question, the charge
in artifacts/cost_decomposition_2026-08-12.json).  Measuring one candidate would hide the whole
effect, which is exactly the N-fold repetition of the prefill.

CONVENTIONS -- the four of results/cascade_methods/artifacts/vram_testtime_2026-08-11.json, and
the SAME 12 bracket items as its S3 scenario, so the rows line up with
artifacts/_verifier_hparams_parts/vram_latency.json.
  (a) a_weights_resident  torch.cuda.memory_allocated() after load + adapter (a FLOOR)
  (b) b_peak_allocated    torch.cuda.max_memory_allocated(), reset before EVERY question
  (c) c_peak_reserved     torch.cuda.max_memory_reserved(), reset before EVERY question
  (d) d_process_footprint max board `used` over a 20 ms NVML sampler minus the pre-run baseline
  units GiB = bytes / 1024**3

TF32 is left at this container's torch DEFAULT (True), which is what produced the frozen transfer
dumps.  vram_latency.json pinned TF32 OFF, so its latencies are NOT directly comparable to these;
the deployed-loop arm measured here is the in-session control for the prefix arm.

Requires an idle card -- (d) is board-minus-baseline and is meaningless with a co-tenant.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
      src/training_methods/prefix_shared_vram.py
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods.prefix_shared_verifier import (   # noqa: E402
    MINPX, SYS, imgs_for, load_dump_items)

OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/_prefix_shared_parts")
GIB = 1024 ** 3
S3_ITEMS = [("slake_open", 12615), ("slake_open", 12618), ("slake_open", 12532),
            ("slake_open", 12533), ("vqa_rad_open", 133), ("vqa_rad_open", 134),
            ("vqa_rad_open", 394), ("vqa_rad_open", 395), ("pathvqa_open", 424),
            ("pathvqa_open", 324), ("pathvqa_open", 7), ("pathvqa_open", 8)]


def gb(x):
    return round(x / GIB, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
                            "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--pixels", type=int, nargs="+", default=[250880, 1003520])
    ap.add_argument("--allow_busy_gib", type=float, default=1.0)
    ap.add_argument("--allow_contaminated", action="store_true",
                    help="run on a shared card anyway. (a)/(b)/(c) are torch-internal and are "
                         "unaffected by a co-tenant; (d) is board-minus-baseline and becomes "
                         "UNUSABLE -- it is emitted but flagged.")
    A = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    os.makedirs(OUT, exist_ok=True)

    import pynvml
    pynvml.nvmlInit()
    dev = int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0])
    H = pynvml.nvmlDeviceGetHandleByIndex(dev)

    def board():
        return int(pynvml.nvmlDeviceGetMemoryInfo(H).used)

    BASELINE = board()
    CONTAMINATED = BASELINE > A.allow_busy_gib * GIB
    print(f"pre-run board used on GPU{dev}: {gb(BASELINE)} GiB "
          f"({'CO-TENANT PRESENT' if CONTAMINATED else 'idle'})", flush=True)
    if CONTAMINATED and not A.allow_contaminated:
        print("REFUSING: target GPU is not idle; (d) would be contaminated. Re-run with "
              "--allow_contaminated to take (a)/(b)/(c) anyway.", flush=True)
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
    tok = proc.tokenizer
    YES = tok.encode("Yes", add_special_tokens=False)[0]
    NO = tok.encode("No", add_special_tokens=False)[0]
    IMTOK = proc.image_token
    IMID = tok.convert_tokens_to_ids(IMTOK)
    MERGE = proc.image_processor.merge_size ** 2

    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").to("cuda")
    torch.cuda.synchronize()
    bare = torch.cuda.memory_allocated()
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    torch.cuda.synchronize()
    co_res = torch.cuda.memory_allocated()
    core = model
    while not hasattr(core, "get_rope_index"):
        core = getattr(core, "model", None) or getattr(core, "base_model")
    print(f"bare {gb(bare)} GiB | +adapter {gb(co_res)} GiB", flush=True)

    POOL = {(it["ds"], it["idx"]): list(dict.fromkeys(it["preds"])) for it in load_dump_items()}
    IMG = {ds: imgs_for(ds) for ds in {d for d, _ in S3_ITEMS}}
    ones = lambda n: torch.ones((1, n), dtype=torch.long, device="cuda")

    def msgs(q, img, a, px):
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": px, "min_pixels": MINPX},
                    {"type": "text", "text": f"Question: {q}\nProposed answer: {a}\n"
                                             f"Is the proposed answer correct? Answer Yes or No."}]}]

    def prep(q, img, answers, px):
        """CPU-side preparation, excluded from every timed region (as in vram_latency.json)."""
        m0 = msgs(q, img, answers[0], px)
        t0 = proc.apply_chat_template(m0, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(m0)
        e0 = proc(text=[t0], images=igs, videos=vids, return_tensors="pt", padding=True)
        pv = e0["pixel_values"].to("cuda")
        grid = e0["image_grid_thw"].to("cuda")
        nimg = int(e0["image_grid_thw"][0].prod()) // MERGE
        encs, ids = [], []
        for a in answers:
            mm = msgs(q, img, a, px)
            t = proc.apply_chat_template(mm, tokenize=False, add_generation_prompt=True)
            gg, vv = process_vision_info(mm)
            encs.append(proc(text=[t], images=gg, videos=vv, return_tensors="pt",
                             padding=True).to("cuda"))
            ids.append(tok([t.replace(IMTOK, IMTOK * nimg, 1)],
                           return_tensors="pt")["input_ids"][0].to("cuda"))
        Lmax = min(int(x.shape[0]) for x in ids)
        L = 0
        while L < Lmax and all(int(x[L]) == int(ids[0][L]) for x in ids):
            L += 1
        if L >= Lmax:
            L = Lmax - 1
        pos = [core.get_rope_index(x[None], grid, None,
                                   attention_mask=ones(int(x.shape[0])))[0] for x in ids]
        return dict(pv=pv, grid=grid, encs=encs, ids=ids, L=L, pos=pos,
                    npatch=int(pv.shape[0]), nimg=nimg)

    def run_deployed(P):
        """N full forwards -- src/training_methods/cheapleg_score_open.py:125."""
        out = []
        with torch.no_grad():
            for enc in P["encs"]:
                lg = model(**enc).logits[0, -1]
                out.append(math.exp(lg[YES].item()) /
                           (math.exp(lg[YES].item()) + math.exp(lg[NO].item())))
        return out

    def run_prefix(P):
        """1 shared prefill + N short tails."""
        L, ids, pos = P["L"], P["ids"], P["pos"]
        out = []
        with torch.no_grad():
            o = model(input_ids=ids[0][None, :L], attention_mask=ones(L), pixel_values=P["pv"],
                      image_grid_thw=P["grid"], position_ids=pos[0][:, :, :L], use_cache=True)
            cache = o.past_key_values
            for k, idk in enumerate(ids):
                n = int(idk.shape[0])
                cache.crop(L)
                oo = model(input_ids=idk[None, L:], attention_mask=ones(n),
                           past_key_values=cache, position_ids=pos[k][:, :, L:],
                           cache_position=torch.arange(L, n, device="cuda"), use_cache=True)
                lg = oo.logits[0, -1]
                out.append(math.exp(lg[YES].item()) /
                           (math.exp(lg[YES].item()) + math.exp(lg[NO].item())))
        del cache
        return out

    PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def run_prefix_batched(P):
        """1 shared prefill + ONE batched forward over all N tails.

        Why this arm exists: at batch 1 a ~20-token tail is LAUNCH-LATENCY bound, not FLOP bound
        (28 layers of tiny kernels), so the FLOP saving of prefix sharing does not turn into a
        wall-clock saving. Batching the tails collapses N launches into 1. Tails are RIGHT-padded:
        under causal attention a row's true last token precedes every pad, so the pad content
        cannot reach it. Correctness is asserted against run_prefix(), not assumed.
        """
        L, ids, pos = P["L"], P["ids"], P["pos"]
        N = len(ids)
        tails = [x[L:] for x in ids]
        Tm = max(int(t.shape[0]) for t in tails)
        inp = torch.full((N, Tm), PAD, dtype=torch.long, device="cuda")
        am = torch.zeros((N, L + Tm), dtype=torch.long, device="cuda")
        pid = torch.zeros((3, N, Tm), dtype=pos[0].dtype, device="cuda")
        last = []
        for k, t in enumerate(tails):
            n = int(t.shape[0])
            inp[k, :n] = t
            am[k, :L + n] = 1
            pid[:, k, :n] = pos[k][:, 0, L:]
            if n < Tm:                      # pad positions continue the sequence (masked anyway)
                pid[:, k, n:] = pos[k][:, 0, -1:] + torch.arange(
                    1, Tm - n + 1, device="cuda", dtype=pid.dtype)
            last.append(n - 1)
        out = []
        with torch.no_grad():
            o = model(input_ids=ids[0][None, :L], attention_mask=ones(L), pixel_values=P["pv"],
                      image_grid_thw=P["grid"], position_ids=pos[0][:, :, :L], use_cache=True)
            cache = o.past_key_values
            cache.batch_repeat_interleave(N)
            oo = model(input_ids=inp, attention_mask=am, past_key_values=cache,
                       position_ids=pid,
                       cache_position=torch.arange(L, L + Tm, device="cuda"), use_cache=True)
            for k in range(N):
                lg = oo.logits[k, last[k]]
                out.append(math.exp(lg[YES].item()) /
                           (math.exp(lg[YES].item()) + math.exp(lg[NO].item())))
        del cache
        return out

    def timed(fn, P):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn(P)                                       # warm-up, not timed, not measured
        torch.cuda.reset_peak_memory_stats()
        s = Sampler()
        s.start()
        lat = []
        for _ in range(A.reps):
            torch.cuda.synchronize()
            t0 = time.time()
            fn(P)
            torch.cuda.synchronize()
            lat.append(time.time() - t0)
        mb = s.stop()
        return dict(lat_median_s=float(np.median(lat)), lat_min_s=float(np.min(lat)),
                    b_peak_allocated_gib=gb(torch.cuda.max_memory_allocated()),
                    c_peak_reserved_gib=gb(torch.cuda.max_memory_reserved()),
                    d_board_used_gib=gb(mb),
                    d_process_footprint_gib=gb(max(0, mb - BASELINE)))

    res = {"_meta": {
        "title": "BUILD 2 -- VRAM (4 conventions) + clean batch-1 latency PER QUESTION for the "
                 "deployed per-candidate verifier loop vs the prefix-shared replacement",
        "conventions": "results/cascade_methods/artifacts/vram_testtime_2026-08-11.json (a)-(d), "
                       "GiB = bytes/1024**3",
        "items": "the same 12 bracket items as that artifact's S3, same order; each scored on its "
                 "OWN pool of distinct raw candidate answers (the deployed unit of work)",
        "framework": "HuggingFace transformers, bf16, flash_attention_2, tp=1, batch 1. NEVER vLLM.",
        "tf32": "torch DEFAULT (matmul.allow_tf32=%s) -- the setting that produced the frozen "
                "transfer dumps. artifacts/_verifier_hparams_parts/vram_latency.json pinned TF32 "
                "OFF, so its latencies are not directly comparable; the deployed arm here is the "
                "in-session control." % torch.backends.cuda.matmul.allow_tf32,
        "torch": torch.__version__, "gpu": pynvml.nvmlDeviceGetName(H),
        "pre_run_board_baseline_gib": gb(BASELINE),
        "card_exclusivity": ("IDLE CARD -- (d) is valid" if not CONTAMINATED else
                             "CO-TENANT PRESENT -- convention (d) is board-minus-baseline and is "
                             "CONTAMINATED here; it is reported but must NOT be quoted. (a), (b) "
                             "and (c) are torch-internal and are unaffected."),
        "d_is_usable": not CONTAMINATED,
        "a_weights_resident_gib_bare_7b": gb(bare),
        "a_weights_resident_gib_with_adapter": gb(co_res),
        "marginal_adapter_resident_gib": gb(co_res - bare),
        "reps_per_item": A.reps,
        "_latency_note": "GPU wall time of the forwards for ONE QUESTION (all its distinct "
                         "candidates), CUDA-synchronized, CPU-side processor/tokenizer work "
                         "excluded from the timed region for BOTH arms. Median over reps."},
        "by_max_pixels": {}}

    for px in A.pixels:
        rows = []
        for ds, idx in S3_ITEMS:
            try:
                q, img = IMG[ds][idx]
                answers = POOL[(ds, idx)]
                P = prep(q, img, answers, px)
                dep = timed(run_deployed, P)
                pre = timed(run_prefix, P)
                sd, sp = run_deployed(P), run_prefix(P)
                try:                    # the batched-tail arm is EXPLORATORY -- never fatal
                    bat = timed(run_prefix_batched, P)
                    sb = run_prefix_batched(P)
                except Exception as e:
                    print(f"  batched-tail arm failed on {ds}/{idx}: {type(e).__name__}: "
                          f"{str(e)[:160]}", flush=True)
                    bat = {k: float("nan") for k in
                           ("lat_median_s", "lat_min_s", "b_peak_allocated_gib",
                            "c_peak_reserved_gib", "d_board_used_gib", "d_process_footprint_gib")}
                    bat["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    sb = sp
                rows.append(dict(
                    ds=ds, idx=idx, n_distinct_candidates=len(answers),
                    prefix_tok=P["L"], full_tok=[int(x.shape[0]) for x in P["ids"]],
                    tail_tok=[int(x.shape[0]) - P["L"] for x in P["ids"]],
                    vision_tokens=P["npatch"] // 4, premerge_patches=P["npatch"],
                    deployed=dep, prefix_shared=pre, prefix_shared_batched_tails=bat,
                    speedup_median=dep["lat_median_s"] / pre["lat_median_s"],
                    speedup_median_batched=dep["lat_median_s"] / bat["lat_median_s"],
                    max_abs_score_dev=float(max(abs(a - b) for a, b in zip(sd, sp))),
                    max_abs_score_dev_batched_vs_prefix=float(
                        max(abs(a - b) for a, b in zip(sp, sb))),
                    argmax_same=int(int(np.argmax(sd)) == int(np.argmax(sp))),
                    argmax_same_batched=int(int(np.argmax(sp)) == int(np.argmax(sb)))))
                del P
            except Exception as e:                        # per-item error guard
                rows.append(dict(ds=ds, idx=idx, error=f"{type(e).__name__}: {str(e)[:180]}"))
                print(f"  FAIL px{px} {ds} {idx}: {e}", flush=True)
        ok = [r for r in rows if "error" not in r]

        def agg(sel):
            v = np.array([sel(r) for r in ok], float)
            return {"mean": float(v.mean()), "peak": float(v.max()), "min": float(v.min())}
        res["by_max_pixels"][str(px)] = {
            "max_pixels": px, "n": len(ok), "n_failed": len(rows) - len(ok),
            "a_weights_resident_gib": gb(co_res),
            "mean_distinct_candidates": float(np.mean([r["n_distinct_candidates"] for r in ok])),
            "mean_prefix_tok": float(np.mean([r["prefix_tok"] for r in ok])),
            "mean_full_tok": float(np.mean([np.mean(r["full_tok"]) for r in ok])),
            "mean_tail_tok": float(np.mean([np.mean(r["tail_tok"]) for r in ok])),
            "mean_vision_tokens": float(np.mean([r["vision_tokens"] for r in ok])),
            "deployed": {k: agg(lambda r, k=k: r["deployed"][k]) for k in
                         ("lat_median_s", "b_peak_allocated_gib", "c_peak_reserved_gib",
                          "d_process_footprint_gib")},
            "prefix_shared": {k: agg(lambda r, k=k: r["prefix_shared"][k]) for k in
                              ("lat_median_s", "b_peak_allocated_gib", "c_peak_reserved_gib",
                               "d_process_footprint_gib")},
            "prefix_shared_batched_tails": {
                k: agg(lambda r, k=k: r["prefix_shared_batched_tails"][k]) for k in
                ("lat_median_s", "b_peak_allocated_gib", "c_peak_reserved_gib",
                 "d_process_footprint_gib")},
            "latency_per_question_speedup": agg(lambda r: r["speedup_median"]),
            "pooled_latency_speedup": float(sum(r["deployed"]["lat_median_s"] for r in ok) /
                                            sum(r["prefix_shared"]["lat_median_s"] for r in ok)),
            "pooled_latency_speedup_batched_tails": float(
                sum(r["deployed"]["lat_median_s"] for r in ok) /
                sum(r["prefix_shared_batched_tails"]["lat_median_s"] for r in ok)),
            "max_abs_score_dev": float(max(r["max_abs_score_dev"] for r in ok)),
            "max_abs_score_dev_batched_vs_prefix":
                float(max(r["max_abs_score_dev_batched_vs_prefix"] for r in ok)),
            "argmax_agreement": f"{sum(r['argmax_same'] for r in ok)}/{len(ok)}",
            "argmax_agreement_batched_vs_prefix":
                f"{sum(r['argmax_same_batched'] for r in ok)}/{len(ok)}",
            "rows": rows}
        r = res["by_max_pixels"][str(px)]
        print(f"px {px:>9}  dep {r['deployed']['lat_median_s']['mean']*1000:7.1f} ms/q  "
              f"pre {r['prefix_shared']['lat_median_s']['mean']*1000:7.1f} ms/q  "
              f"bat {r['prefix_shared_batched_tails']['lat_median_s']['mean']*1000:7.1f} ms/q  "
              f"x{r['pooled_latency_speedup']:.2f}/x"
              f"{r['pooled_latency_speedup_batched_tails']:.2f}  "
              f"d_dep {r['deployed']['d_process_footprint_gib']['mean']:.3f} GiB  "
              f"d_pre {r['prefix_shared']['d_process_footprint_gib']['mean']:.3f} GiB", flush=True)

    res["_meta"]["post_run_board_used_gib"] = gb(board())
    p = os.path.join(OUT, "vram_latency_prefix.json")
    json.dump(res, open(p, "w"), indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
