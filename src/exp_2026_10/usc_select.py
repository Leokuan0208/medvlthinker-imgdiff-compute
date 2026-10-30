#!/usr/bin/env python3
"""usc_select.py -- Universal Self-Consistency (Chen et al. 2023, arXiv:2311.17311) as a baseline SELECTOR over
the final method's own candidate pools.

Owed since the 2026-10-19 deck ("baselines owed: USC"). USC is the standard training-free way to pick one of N
free-text samples: show the SAME model all N and ask which is most consistent. It is the natural rival to a
trained probe, because it needs no labels at all -- only a second (longer) prompt per question.

Prompt: the paper's USC template, verbatim, with the question; the image is included because this is VQA (the
paper is text-only). Candidates: the 8 stored samples of the pool, in sample order, duplicates kept (USC relies
on seeing repeated answers). Model: Lingshu-7B, the generator itself, greedy, run_openvqa.py's system prompt
and image handling at the generation cap (cap320). The reply is parsed for "Response X"; when it cannot be
parsed the pool's modal answer is used and flagged (`fallback`) -- the method always answers.
Pools: T07 (the deck's headline) and Tmap (FINAL_METHOD's method). usc_compare.py scores the picks.

  CUDA_VISIBLE_DEVICES=0 python3 src/exp_2026_10/usc_select.py --policy T07   (vLLM)
"""
import argparse, json, os, re, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict_modality as PM                     # run_openvqa's verbatim loader + constants

CK = "/home/jamesyang/medvlthinker-imgdiff-compute/ckpts/openvqa/cheap_lingshu7b"
TMAP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                   "results/cascade_methods/artifacts/frozen_temperature_map_2026-10-15.json")))["frozen_T"]
TAG = {0.2: "lingshu7bT02_sc8", 0.4: "lingshu7bT04_sc8", 0.7: "lingshu7b_sc8", 1.0: "lingshu7bT10_sc8"}
USC = ("I have generated the following responses to the question: {q}\n\n{resp}\n\n"
       "Evaluate these responses.\nSelect the most consistent response based on majority consensus.\n"
       "Start your answer with \"The most consistent response is Response X\" (without quotes).")


def pool_tag(cell, policy):
    return TAG[0.7] if policy == "T07" else TAG[TMAP.get(cell, 0.7)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--policy", choices=["T07", "Tmap"], required=True)
    A = ap.parse_args()
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    out = f"/data/dan/exp_2026_10/usc_{A.policy}.jsonl"
    done = {(json.loads(l)["cell"], json.loads(l)["idx"]) for l in open(out) if l.strip()} if os.path.exists(out) else set()
    proc = AutoProcessor.from_pretrained(PM.MP, trust_remote_code=True)
    llm = LLM(model=PM.MP, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=8192,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=24)
    t0 = time.time()
    with open(out, "a") as fh:
        for cell in PM.OPEN8:
            tag = pool_tag(cell, A.policy)
            pool = {}
            for l in open(f"{CK}/ckpt_{cell}_{tag}.jsonl"):
                if l.strip():
                    d = json.loads(l); pool[d["idx"]] = d
            items = [it for it in PM.load_items(cell) if it[0] in pool and (cell, it[0]) not in done]
            for c0 in range(0, len(items), 128):
                ch = items[c0:c0 + 128]; reqs = []
                for idx, q, gold, im in ch:
                    resp = "\n".join(f"Response {i}: {p}" for i, p in enumerate(pool[idx]["preds"]))
                    msgs = [{"role": "system", "content": PM.SYS}, {"role": "user", "content": [
                        {"type": "image", "image": im, "max_pixels": PM.MAXPX, "min_pixels": PM.MIN_PX},
                        {"type": "text", "text": USC.format(q=q, resp=resp)}]}]
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    imgs, _ = process_vision_info(msgs)
                    reqs.append({"prompt": text, "multi_modal_data": {"image": imgs}})
                for (idx, q, gold, im), o in zip(ch, llm.generate(reqs, sp)):
                    raw = o.outputs[0].text.strip(); preds = pool[idx]["preds"]
                    m = re.search(r"[Rr]esponse\s*(\d+)", raw)
                    k = int(m.group(1)) if m else None
                    fb = k is None or not (0 <= k < len(preds))
                    ans = pool[idx]["modal_pred"] if fb else preds[k]
                    fh.write(json.dumps({"cell": cell, "idx": idx, "pool": tag, "raw": raw, "choice": None if fb else k,
                                         "ans": ans, "fallback": fb}) + "\n")
                fh.flush()
            print(f"[{time.time() - t0:5.0f}s] {cell} [{A.policy} {tag}]: {len(items)} questions", flush=True)


if __name__ == "__main__":
    main()
