#!/usr/bin/env python3
"""closed_as_open_gen.py -- BUILD 3 GPU half A: generate every arm of the three closed cells.

ONE vLLM engine load, many (arm x cell) runs, resumable per item, per-chunk error guard, engine
death is FATAL (exit 17) so the runner restarts and RESUMES rather than writing an empty file.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
      src/cascade_methods/closed_as_open_gen.py --arms openPRJ_s8 openPRJ_g ...

Launch from the repo root (checkpoint paths are relative).  nohup, never tmux.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arms", nargs="+", default=list(L.ARMS))
ap.add_argument("--cells", nargs="+", default=L.CELLS)
ap.add_argument("--model_path", default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--"
                                        "Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ap.add_argument("--gpu_mem", type=float, default=0.42)
ap.add_argument("--max_model_len", type=int, default=4096)
ap.add_argument("--chunk", type=int, default=128)
A = ap.parse_args()

os.makedirs(L.CKPT, exist_ok=True)
LOCKS = os.path.join(L.CKPT, ".locks")
os.makedirs(LOCKS, exist_ok=True)

print("[items] loading ...", flush=True)
ITEMS = L.build_items()
for c in A.cells:
    print(f"[items] {c}: {len(ITEMS[c])} == frozen EXPECT_N", flush=True)


def todo_for(cell, arm):
    have = L.load_gen(cell, arm)
    return [it for it in ITEMS[cell] if it["i"] not in have], len(have)


WORK = []
for arm in A.arms:
    for cell in A.cells:
        td, have = todo_for(cell, arm)
        if td:
            WORK.append((arm, cell, len(td)))
        else:
            print(f"[skip] {arm} {cell}: complete ({have})", flush=True)
if not WORK:
    print("CLOSED_AS_OPEN_GEN_DONE (nothing to do)", flush=True)
    sys.exit(0)
print(f"[plan] {len(WORK)} (arm,cell) jobs: " + ", ".join(f"{a}/{c}:{n}" for a, c, n in WORK), flush=True)

from PIL import Image                                                     # noqa: E402
from transformers import AutoProcessor                                    # noqa: E402
from qwen_vl_utils import process_vision_info                             # noqa: E402
from vllm import LLM, SamplingParams                                      # noqa: E402

proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)


def load_image(item):
    out = []
    for p in item["images"]:
        if item["img_kind"] == "path":
            out.append(p)                              # qwen_vl_utils opens the path itself
        elif item["img_kind"] == "raw":
            out.append(Image.open(p))                  # VQA_RAD: HF-decoded PIL, no .convert
        elif item["img_kind"] == "rawrgb":
            out.append(Image.open(p).convert("RGB"))   # PATH_VQA: loader converts
        else:
            raise ValueError(item["img_kind"])
    return out


def build_req(item, prompt, cap, sysmsg):
    maxpx = L.HIGH_PX // L.CAP_DIV[cap]
    imds = []
    for img in load_image(item):
        imds.append({"type": "image", "image": img, "max_pixels": maxpx, "min_pixels": L.MIN_PX})
    if len(imds) == 1:
        content = [imds[0], {"type": "text", "text": prompt}]
    else:                                              # multi-image shape, verbatim from the wrapper
        content = []
        for k, d in enumerate(imds):
            content += [{"type": "text", "text": f"<image_{k+1}>: "}, d]
        content.append({"type": "text", "text": prompt})
    msgs = []
    if sysmsg:
        msgs.append({"role": "system", "content": sysmsg})
    msgs.append({"role": "user", "content": content})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if imgs:
        req["multi_modal_data"] = {"image": imgs}
    return req


def claim(arm, cell):
    p = os.path.join(LOCKS, f"{arm}__{cell}")
    try:
        os.mkdir(p)
    except FileExistsError:
        try:
            owner = int(open(os.path.join(p, "pid")).read().strip())
            os.kill(owner, 0)
            return False
        except Exception:
            pass
    with open(os.path.join(p, "pid"), "w") as f:
        f.write(str(os.getpid()))
    return True


llm = LLM(model=A.model_path, tensor_parallel_size=1, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
          limit_mm_per_prompt={"image": 4}, trust_remote_code=True, enforce_eager=True)

for arm in A.arms:
    cfg = L.ARMS[arm]
    sp = SamplingParams(temperature=cfg["temp"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                        min_p=cfg["min_p"], repetition_penalty=cfg["rep_pen"],
                        max_tokens=L.MAX_TOKENS, n=cfg["n"], seed=cfg["seed"])
    for cell in A.cells:
        todo, have = todo_for(cell, arm)
        if not todo:
            print(f"[skip] {arm} {cell}: complete ({have})", flush=True)
            continue
        if not claim(arm, cell):
            print(f"[busy] {arm} {cell}: claimed by another worker", flush=True)
            continue
        ck = L.gen_path(cell, arm)
        print(f"[run ] {arm} {cell}: {len(todo)} to go -> {os.path.basename(ck)}", flush=True)
        t0 = time.time()
        with open(ck, "a", encoding="utf-8") as fh:
            for c0 in range(0, len(todo), A.chunk):
                ch = todo[c0:c0 + A.chunk]
                try:
                    reqs = [build_req(it, L.build_prompt(cfg["prompt"], cell, it["question"],
                                                         it["lang"]), cfg["cap"], cfg["sys"])
                            for it in ch]
                    outs = llm.generate(reqs, sp)
                except Exception as e:
                    if "EngineDead" in type(e).__name__ or "EngineCore" in str(e):
                        print(f"   !!! FATAL engine death at chunk {c0}: {type(e).__name__}: {e}",
                              flush=True)
                        fh.flush()
                        sys.exit(17)
                    print(f"   !! chunk {c0} failed: {type(e).__name__}: {e}", flush=True)
                    continue
                for it, o in zip(ch, outs):
                    try:
                        preds = [c.text.strip() for c in o.outputs]
                        fh.write(json.dumps({
                            "i": it["i"], "src": it["src"], "cell": cell, "arm": arm,
                            "lang": it["lang"], "question": it["question"], "gold": it["answer"],
                            "prompt": L.build_prompt(cfg["prompt"], cell, it["question"], it["lang"]),
                            "preds": preds,
                            "gen_tokens": [len(c.token_ids) for c in o.outputs],
                        }, ensure_ascii=False) + "\n")
                    except Exception as e:
                        print(f"   !! item {it['i']} failed: {type(e).__name__}: {e}", flush=True)
                fh.flush()
                n = min(c0 + A.chunk, len(todo))
                print(f"   [{n}/{len(todo)}] {n/(time.time()-t0):.1f} it/s", flush=True)
        got = len(L.load_gen(cell, arm))
        if got != L.EXPECT_N[cell]:
            print(f"   !!! INCOMPLETE {arm} {cell}: {got}/{L.EXPECT_N[cell]} -- exiting to resume",
                  flush=True)
            sys.exit(18)
        print(f"[done] {arm} {cell} {got}/{L.EXPECT_N[cell]} in {(time.time()-t0)/60:.2f} min",
              flush=True)
print("CLOSED_AS_OPEN_GEN_DONE", flush=True)
