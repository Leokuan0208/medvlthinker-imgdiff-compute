#!/usr/bin/env python3
"""
pairwise_verifier_diverse.py -- COMPOUNDING experiment GPU pass.

Scores the DIVERSE-GENERATION candidate pool (ckpts/openvqa/diverse/ckpt_<DS>_lingshu7b_div.jsonl,
~15 candidates/q, portfolio-prompted + temperature-laddered, with pooled4 POINTWISE scores) with the
SAME REAL pairwise verifier as pairwise_verifier_score.py (Lingshu-7B + pooled4 LoRA, PAIRWISE prompt,
both orders averaged for position debias). This is the second lever of the compounding hypothesis:
pairwise-verifier SELECTION over DIVERSE candidates, to test whether pairwise converts the diverse-gen
oracle lift that the pointwise verifier could not (distractor injection; esp. PMC +0.11 oracle).

Mechanism-identical to pairwise_verifier_score.py:
  - same weights, same PAIR_SYS prompt, same cap (cap320), same both-orders position debias
  - identical answer strings (.strip().lower()) -> p=0.5 (no GPU call)
GPU-cost optimization: many of the 15 slots repeat the same answer string, so we score each UNIQUE
distinct answer-STRING-PAIR once and reuse it for all slot pairs sharing those strings (the analyzer
expands back to slot level). Verdicts are keyed by the normalized string pair.

Reads diverse dumps (preds[15], oks[15], scores[15]); re-loads images by idx via
diversity_generate_gpu.load_items (supports pmc_content). Writes verdicts jsonl:
  {ds, idx, ki, kj, p_ki_gt_kj}  for every within-question distinct-answer-string pair.

tp=1, GPU0, Lingshu-7B. Launch from repo root (guarded by PAIRWISE_GPU_OK=1):
  PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/pairwise_verifier_diverse.py --dataset pmc_content
"""
import argparse, os, sys, json, math, itertools, importlib.util

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
_spec = importlib.util.spec_from_file_location("dgg", J("src/cascade_methods/diversity_generate_gpu.py"))
_dgg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dgg)

DIVERSE_DUMP = {
    "vqa_rad_open": "ckpts/openvqa/diverse/ckpt_vqa_rad_open_lingshu7b_div.jsonl",
    "slake_open":   "ckpts/openvqa/diverse/ckpt_slake_open_lingshu7b_div.jsonl",
    "pathvqa_open": "ckpts/openvqa/diverse/ckpt_pathvqa_open_lingshu7b_div.jsonl",
    "pmc_content":  "ckpts/openvqa/diverse/ckpt_pmc_content_lingshu7b_div.jsonl",
}
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28; CAP_DIV = {"fullres":1,"cap640":2,"cap320":4,"cap160":8,"cap80":16}
PAIR_SYS = ("You are a careful medical exam grader. Given a medical image, a question, and two candidate "
            "answers (A and B), decide which candidate answer is more likely correct. Respond with only 'A' or 'B'.")


def norm(s):
    return str(s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DIVERSE_DUMP))
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--verifier_lora", default="ckpts/train/lora_verifier_pooled4",
                    help="pointwise verifier LoRA, reused in pairwise prompt mode (mechanism-clean). '' -> base model.")
    ap.add_argument("--tag", default="lingshu7b_div")
    ap.add_argument("--out_dir", default="ckpts/pairwise_diverse")
    ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV))
    ap.add_argument("--n", type=int, default=100000, help="max questions from the dump (first-n)")
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096); ap.add_argument("--chunk", type=int, default=256)
    A = ap.parse_args()
    if os.environ.get("PAIRWISE_GPU_OK") != "1":
        sys.exit("[REFUSED] set PAIRWISE_GPU_OK=1 to run the pairwise GPU pass.")
    MAXPX = HIGH_PX // CAP_DIV[A.cap]

    dump = {str(json.loads(l)["idx"]): json.loads(l) for l in open(J(DIVERSE_DUMP[A.dataset])) if l.strip()}
    items, _ = _dgg.load_items(A.dataset, A.cap, 100000)
    img_by_idx = {str(it[0]): it[3] for it in items}
    q_by_idx   = {str(it[0]): it[1] for it in items}
    ids = [k for k in dump if k in img_by_idx][:A.n]
    print(f"[pairwise-div] {A.dataset}: {len(ids)}/{len(dump)} questions have images (n={A.n})", flush=True)

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
                r = json.loads(l); done.add((str(r["idx"]), r["ki"], r["kj"]))
    print(f"  resume: {len(done)} verdict rows already present", flush=True)

    # build UNIQUE distinct answer-string pairs per question (both orders); identical strings never enter.
    reqs = []; meta = []   # meta: (idx, ki, kj, order) order 0: ki=A,kj=B ; 1: ki=B,kj=A
    for k in ids:
        preds = dump[k]["preds"]
        keys = [norm(p) for p in preds]
        rep = {}
        for kk, raw in zip(keys, preds):
            if kk not in rep: rep[kk] = raw     # first raw representative for the prompt
        uniq = list(rep.keys()); img = img_by_idx[k]; q = q_by_idx[k]
        for ki, kj in itertools.combinations(uniq, 2):
            if (k, ki, kj) in done: continue
            reqs.append(build(q, img, rep[ki], rep[kj])); meta.append((k, ki, kj, 0))
            reqs.append(build(q, img, rep[kj], rep[ki])); meta.append((k, ki, kj, 1))
    print(f"  {len(reqs)} ordered pairwise reqs (unique string-pairs x2 orders) -> {outp}", flush=True)

    acc = {}
    fh = open(outp, "a")
    CH = A.chunk
    for c0 in range(0, len(reqs), CH):
        outs = llm.generate(reqs[c0:c0+CH], sp, lora_request=lora_req)
        for o, (k, ki, kj, order) in zip(outs, meta[c0:c0+CH]):
            p = pA(o)                       # P(the 'A' slot is better)
            key = (k, ki, kj)
            d = acc.setdefault(key, {})
            d[order] = p if order == 0 else (1.0 - p)   # order1: ki was B, so P(ki>kj)=P(B)=1-P(A)
            if 0 in d and 1 in d:
                pij = 0.5 * (d[0] + d[1])
                fh.write(json.dumps({"ds": A.dataset, "idx": k, "ki": ki, "kj": kj,
                                     "p_ki_gt_kj": round(pij, 5)}) + "\n")
        fh.flush()
        print(f"   [{min(c0+CH,len(reqs))}/{len(reqs)}]", flush=True)
    fh.close()
    print(f"[pairwise-div] DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
