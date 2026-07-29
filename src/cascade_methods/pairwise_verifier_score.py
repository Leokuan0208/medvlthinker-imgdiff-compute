#!/usr/bin/env python3
"""
pairwise_verifier_score.py -- EXPERIMENT 2 GPU pass: REAL pairwise verifier verdicts.

The offline C9 (active_comparison_verifier.py) could not beat pointwise because pairwise preferences were
SIMULATED from the pointwise score (P(i>j)=sigmoid(logit s_i - logit s_j)), whose selection ceiling equals
pointwise-argmax by construction. This script produces REAL pairwise verdicts so we can test whether a
pairwise/active-comparison verifier actually beats the pointwise verifier on SELECTION ACCURACY.

Mechanism-clean choice: use the SAME weights as the pointwise verifier (Lingshu-7B + pooled4 LoRA) but in a
PAIRWISE prompt ("given image+question, which answer is better, A or B?"). Holding the model fixed isolates
the pointwise-vs-pairwise MECHANISM. Position bias is controlled by scoring BOTH orders (i as A, then i as B)
and averaging: p_i_gt_j = 0.5*(P(A | i=A,j=B) + P(B | i=B,j=A)). Identical answer strings -> p=0.5 (no GPU call).

Reads the iid@8 candidate dumps (ckpts/mcq_gen_verify/lingshu7b/ckpt_<DS>_..._content_sc8.jsonl): per q,
preds[8] + scores[8] (pointwise) + oks[8]. Re-loads images by idx via diversity_generate_gpu.load_items.
Writes verdicts jsonl: {ds, idx, i, j, p_i_gt_j, same_str} for all within-question distinct-answer pairs.

tp=1, GPU0, Lingshu-7B. Launch from repo root (guarded by PAIRWISE_GPU_OK=1):
  PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/pairwise_verifier_score.py --dataset vqa_rad_open --n 200
"""
import argparse, os, sys, json, math, itertools, importlib.util
import numpy as np

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
_spec = importlib.util.spec_from_file_location("dgg", J("src/cascade_methods/diversity_generate_gpu.py"))
_dgg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dgg)

MCQ_DUMP = {
    "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
    "slake_open":   "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
    "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl",
}
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV = {"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
PAIR_SYS = ("You are a careful medical exam grader. Given a medical image, a question, and two candidate "
            "answers (A and B), decide which candidate answer is more likely correct. Respond with only 'A' or 'B'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(MCQ_DUMP))
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--verifier_lora", default="ckpts/train/lora_verifier_pooled4",
                    help="pointwise verifier LoRA, reused in pairwise prompt mode (mechanism-clean). '' -> base model.")
    ap.add_argument("--tag", default="lingshu7b")
    ap.add_argument("--out_dir", default="ckpts/pairwise")
    ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV))
    ap.add_argument("--n", type=int, default=200, help="subsample: max questions from the dump (first-n)")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096); ap.add_argument("--chunk", type=int, default=128)
    A = ap.parse_args()
    if os.environ.get("PAIRWISE_GPU_OK") != "1":
        sys.exit("[REFUSED] set PAIRWISE_GPU_OK=1 to run the pairwise GPU pass.")
    MAXPX = HIGH_PX // CAP_DIV[A.cap]

    # candidate dump (preds by idx) + images by idx
    dump = {str(json.loads(l)["idx"]): json.loads(l) for l in open(J(MCQ_DUMP[A.dataset])) if l.strip()}
    items, _ = _dgg.load_items(A.dataset, A.cap, 100000)
    img_by_idx = {str(it[0]): it[3] for it in items}
    q_by_idx   = {str(it[0]): it[1] for it in items}
    ids = [k for k in dump if k in img_by_idx][:A.n]
    print(f"[pairwise] {A.dataset}: {len(ids)} questions (subsample n={A.n})", flush=True)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def tok_ids(words):
        ids_ = {}
        for w in words:
            for v in (w, " " + w):
                e = proc.tokenizer.encode(v, add_special_tokens=False)
                if len(e) == 1: ids_[e[0]] = w
        return ids_
    TA, TB = tok_ids(["A"]), tok_ids(["B"])

    def build(q, img, ansA, ansB):
        body = (f"Question: {q}\nAnswer A: {ansA}\nAnswer B: {ansB}\n"
                f"Which candidate answer is more likely correct, A or B? Respond with only A or B.")
        msgs = [{"role": "system", "content": PAIR_SYS},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs); req = {"prompt": text}
        if imgs: req["multi_modal_data"] = {"image": imgs}
        return req

    use_lora = bool(A.verifier_lora)
    lora_kw = dict(enable_lora=True, max_lora_rank=32) if use_lora else {}
    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 4}, trust_remote_code=True, **lora_kw)
    lora_req = None
    if use_lora:
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("verifier", 1, os.path.expanduser(os.path.join(REPO, A.verifier_lora)))
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    def pA(o):
        lps = (o.outputs[0].logprobs or [{}])[0]
        pa = max((math.exp(v.logprob) for t, v in lps.items() if t in TA), default=0.0)
        pb = max((math.exp(v.logprob) for t, v in lps.items() if t in TB), default=0.0)
        return (pa / (pa + pb)) if (pa + pb) > 0 else 0.5

    os.makedirs(J(A.out_dir), exist_ok=True)
    outp = J(os.path.join(A.out_dir, f"verdicts_{A.dataset}_{A.tag}.jsonl"))
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                r = json.loads(l); done.add((r["idx"], r["i"], r["j"]))

    # build ALL (idx, i<j) pairs; identical strings -> p=0.5 (no GPU); distinct -> 2 ordered GPU reqs
    reqs = []; meta = []   # meta: (idx, i, j, order) order 0: i=A,j=B ; 1: i=B,j=A
    same_rows = []
    for k in ids:
        preds = dump[k]["preds"]; N = len(preds); img = img_by_idx[k]; q = q_by_idx[k]
        for i, j in itertools.combinations(range(N), 2):
            if (k, i, j) in done: continue
            if str(preds[i]).strip().lower() == str(preds[j]).strip().lower():
                same_rows.append({"ds": A.dataset, "idx": k, "i": i, "j": j, "p_i_gt_j": 0.5, "same_str": 1})
                continue
            reqs.append(build(q, img, preds[i], preds[j])); meta.append((k, i, j, 0))
            reqs.append(build(q, img, preds[j], preds[i])); meta.append((k, i, j, 1))
    print(f"  {len(reqs)} ordered pairwise reqs (+{len(same_rows)} identical-string p=0.5) -> {outp}", flush=True)

    # accumulate both orders then write debiased verdict
    acc = {}
    fh = open(outp, "a")
    for sr in same_rows: fh.write(json.dumps(sr) + "\n")
    fh.flush()
    CH = A.chunk
    for c0 in range(0, len(reqs), CH):
        outs = llm.generate(reqs[c0:c0+CH], sp, lora_request=lora_req)
        for o, (k, i, j, order) in zip(outs, meta[c0:c0+CH]):
            p = pA(o)                       # P(the 'A' slot is better)
            key = (k, i, j)
            d = acc.setdefault(key, {})
            d[order] = p if order == 0 else (1.0 - p)   # order1: i was B, so P(i>j)=P(B)=1-P(A)
            if 0 in d and 1 in d:
                pij = 0.5 * (d[0] + d[1])
                fh.write(json.dumps({"ds": A.dataset, "idx": k, "i": i, "j": j,
                                     "p_i_gt_j": round(pij, 5), "same_str": 0}) + "\n")
        fh.flush()
        print(f"   [{min(c0+CH,len(reqs))}/{len(reqs)}]", flush=True)
    fh.close()
    print(f"[pairwise] DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
