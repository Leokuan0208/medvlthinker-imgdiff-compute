#!/usr/bin/env python3
"""run_choicewhy_trainpool.py -- generate N=8 sampled MCQ candidates over the STRICTLY DISJOINT
training pool built by src/training_methods/build_choicewhy_mcq_split.py.

Same model, same sampling, same prompt construction, same max_tokens as the Phase-1 evaluation-side
best-of-N probe (src/labeling/run_choicewhy_pilot.py --n_samples 8 --temp 0.7 --seed 1234), so the
verifier's training distribution and its evaluation distribution differ ONLY in which items they cover:

  model        lingshu-medical-mllm/Lingshu-7B, vLLM tp=1, bfloat16, fullres (max_pixels=1280*28*28)
  sampling     n=8, temperature 0.7, seed 1234, max_tokens=320 (identical in every arm)
  user turn    images + question + "K) option" lines -- byte-identical to
               src/labeling/run_choicewhy_pilot.py::build_prompt (itself byte-identical to
               src/labeling/run_32b_modes_vllm.py::build_prompt)
  system turn  the ONLY thing that differs between arms (src/cascade_methods/choicewhy_common.py::SYS)

Every image is loaded from the staged PNG and its decoded-RGB md5 is ASSERTED against the manifest
value that the split builder proved disjoint from every evaluation image -- so the pixels fed to the
model here are provably the pixels that were checked.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/labeling/run_choicewhy_trainpool.py \
      --arms A B2 --n_samples 8 --temp 0.7 --seed 1234 --ckpt_dir ckpts/choicewhy_train
"""
import argparse, hashlib, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cascade_methods"))
from choicewhy_common import SYS, ARM_NAME  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--arms", nargs="+", default=["A", "B2"])
ap.add_argument("--manifest", default="data/choicewhy_mcq_split/train_items.jsonl")
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_train")
ap.add_argument("--cap", default="fullres", choices=["fullres", "cap640", "cap320", "cap160", "cap80"])
ap.add_argument("--max_tokens", type=int, default=320)
ap.add_argument("--n_samples", type=int, default=8)
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--srcs", nargs="+", default=None)
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)

HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
MAXPX = HIGH_PX // CAP_DIV[A.cap]

ITEMS = [json.loads(l) for l in open(A.manifest) if l.strip()]
SRCS = A.srcs or sorted({r["src"] for r in ITEMS})
print(f"train pool: {len(ITEMS)} questions across {SRCS}", flush=True)

from PIL import Image                          # noqa: E402
from transformers import AutoProcessor          # noqa: E402
from qwen_vl_utils import process_vision_info   # noqa: E402
from vllm import LLM, SamplingParams            # noqa: E402

proc = AutoProcessor.from_pretrained(A.model_path)


def load_image(rec):
    """Load the staged PNG and PROVE it is the image whose hash the split builder cleared."""
    im = Image.open(rec["img_path"]).convert("RGB")
    h = hashlib.md5(im.tobytes()).hexdigest()
    assert h == rec["image_md5_rgb"], f"IMAGE MISMATCH for {rec['src']}/{rec['idx']}: staged pixels differ"
    return im


def build_prompt(rec, arm, im):
    """Byte-identical in shape to run_choicewhy_pilot.py::build_prompt; only SYS[arm] differs."""
    opts = rec["options"]
    q = rec["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    msgs = [{"role": "system", "content": SYS[arm]},
            {"role": "user", "content": [{"type": "image", "image": im,
                                          "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                         {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if image_inputs:
        req["multi_modal_data"] = {"image": image_inputs}
    return req


print(f"loading {A.model_path} (tp={A.tp}, cap={A.cap}, n={A.n_samples}, temp={A.temp}, seed={A.seed})",
      flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
          limit_mm_per_prompt={"image": 8}, trust_remote_code=True)
sp = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, n=A.n_samples,
                    seed=(A.seed if A.n_samples > 1 else None))

t0 = time.time(); tot = 0
for arm in A.arms:
    for src in SRCS:
        sel = [r for r in ITEMS if r["src"] == src]
        ck = os.path.join(A.ckpt_dir, f"ckpt_{src}_{ARM_NAME[arm]}_sc{A.n_samples}.jsonl")
        done = set()
        if os.path.exists(ck):
            for l in open(ck):
                if l.strip():
                    try:
                        done.add(json.loads(l)["idx"])
                    except Exception:
                        pass
        todo = [r for r in sel if r["idx"] not in done]
        print(f"\n--- arm {arm} / {src}: {len(sel)} total, {len(todo)} to run -> {ck} ---", flush=True)
        with open(ck, "a") as fh:
            for c0 in range(0, len(todo), A.chunk):
                ch = todo[c0:c0 + A.chunk]
                ims = [load_image(r) for r in ch]
                reqs = [build_prompt(r, arm, im) for r, im in zip(ch, ims)]
                try:
                    outs = llm.generate(reqs, sp)
                except Exception as e:
                    print(f"   chunk failed ({e}); one-by-one", flush=True)
                    outs = []
                    for r in reqs:
                        try:
                            outs.append(llm.generate([r], sp)[0])
                        except Exception as e2:
                            print(f"     skip: {e2}", flush=True); outs.append(None)
                for r, o in zip(ch, outs):
                    if o is None:
                        continue
                    fh.write(json.dumps({
                        "idx": r["idx"], "src": src, "family": r["family"], "arm": ARM_NAME[arm],
                        "gold": r["gold"], "question": r["question"], "options": r["options"],
                        "image_md5_rgb": r["image_md5_rgb"], "L2_strict": r["L2_strict"],
                        "n_samples": A.n_samples, "temp": A.temp,
                        "gen_tokens_all": [len(c.token_ids) for c in o.outputs],
                        "raw_outputs": [c.text for c in o.outputs]}) + "\n")
                    tot += 1
                fh.flush()
                print(f"   [{min(c0+A.chunk,len(todo))}/{len(todo)}] {tot/(time.time()-t0):.1f} items/s",
                      flush=True)
        print(f">> arm {arm} / {src} done", flush=True)
print(f"\nDONE: {tot} items in {(time.time()-t0)/60:.1f} min", flush=True)
