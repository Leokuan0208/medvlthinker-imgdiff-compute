#!/usr/bin/env python3
"""
verifier_32b_gpu.py -- Does a STRONGER (zero-shot) verifier break the selectability wall?

BACKGROUND. The binding limit of the open-ended best-of-N cascade is the SELECTABILITY WALL:
the trained Lingshu-7B pointwise verifier (pooled4 LoRA) converts only a fraction of the
oracle@N coverage into realized selection accuracy (~74-82% selection ceiling in prior runs).
The obvious attack is CAPACITY: use a much bigger verifier. This script uses Lingshu-32B
(tp=2) as a ZERO-SHOT pointwise verifier and asks whether it converts oracle->selection
better than the trained 7B verifier.

DESIGN (mechanism-clean). Candidate sets, prompt, format, and correctness labels are held
IDENTICAL to how the trained-7B `scores` in the diverse dumps were produced (VERIFY_SYS +
build_verify from diversity_generate_gpu.py). We vary ONLY the verifier model:
  * 7B-trained  : the `scores` already in ckpts/openvqa/diverse/*.jsonl (Lingshu-7B + pooled4 LoRA).
  * 7B-zeroshot : base Lingshu-7B, SAME prompt, no LoRA  (controls the trained-vs-zeroshot confound).
  * 32B-zeroshot: base Lingshu-32B, SAME prompt          (the capacity attack).
Selection is argmax over DISTINCT answers per question (dedup by norm_score; representative =
modal raw string in the group). oracle = any distinct answer correct. Labels = the dump's
exact-match/substring `oks` (map_correct) -- a conservative correctness proxy applied
IDENTICALLY to every verifier and to the oracle, so relative deltas are apples-to-apples.

CONFOUND (honest): 7B-trained is trained (LoRA), 32B is zero-shot. The pure CAPACITY read is
7B-zeroshot vs 32B-zeroshot (both zero-shot); the headline read is 32B-zeroshot vs 7B-trained.

This script writes RAW per-question verdicts (one file per model/dataset). The CPU measure step
(verifier_32b_measure.py) reads them + the dump and writes the final JSON.

GUARDED: refuses to launch unless VERIFIER_GPU_OK=1. Runs ONE dataset per invocation (so an
external `timeout` can bound each dataset). Resumable per-question.

Launch (32B, tp=2, GPU0+1):
  VERIFIER_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0,1 \
    python3 src/cascade_methods/verifier_32b_gpu.py --dataset vqa_rad_open \
      --model_path lingshu-medical-mllm/Lingshu-32B --tag lingshu32b --tp 2 --n 200
Launch (7B zero-shot control, tp=1, GPU0):
  VERIFIER_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/verifier_32b_gpu.py --dataset vqa_rad_open \
      --model_path lingshu-medical-mllm/Lingshu-7B --tag lingshu7b_zs --tp 1 --n 200
"""
import argparse, os, sys, json, math, importlib.util
from collections import Counter

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)
_spec = importlib.util.spec_from_file_location("dgg", J("src/cascade_methods/diversity_generate_gpu.py"))
_dgg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dgg)
norm_score = _dgg.norm_score

DIV_DUMP = {
    "vqa_rad_open": "ckpts/openvqa/diverse/ckpt_vqa_rad_open_lingshu7b_div.jsonl",
    "slake_open":   "ckpts/openvqa/diverse/ckpt_slake_open_lingshu7b_div.jsonl",
    "pmc_content":  "ckpts/openvqa/diverse/ckpt_pmc_content_lingshu7b_div.jsonl",
}
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
# EXACT same grader prompt used to produce the trained-7B `scores` (diversity_generate_gpu.VERIFY_SYS)
VERIFY_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
              "the proposed answer is correct. Respond with only 'Yes' or 'No'.")


def group_distinct(row):
    """Dedup a question's pool into DISTINCT answers (by norm_score). Returns list of dicts:
       {ans (representative raw str), ok (max over members), s7b (max dump score over members)}.
       Order = first appearance of each normalized answer in the pool (stable across models)."""
    preds, oks = row["preds"], row["oks"]
    scores = row.get("scores") or [0.0] * len(preds)
    order, members = [], {}
    for k, p in enumerate(preds):
        nk = norm_score(p)
        if nk not in members:
            members[nk] = []; order.append(nk)
        members[nk].append(k)
    groups = []
    for nk in order:
        idxs = members[nk]
        raw_counts = Counter(preds[k] for k in idxs)
        rep = raw_counts.most_common(1)[0][0]  # modal raw string within the group
        groups.append({"ans": rep,
                       "ok": int(max(oks[k] for k in idxs)),
                       "s7b": float(max(scores[k] for k in idxs))})
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DIV_DUMP))
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-32B")
    ap.add_argument("--tag", default="lingshu32b")
    ap.add_argument("--out_dir", default="ckpts/openvqa/verifier32b")
    ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV))
    ap.add_argument("--n", type=int, default=200, help="subsample: first-n questions from the dump")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    A = ap.parse_args()
    if os.environ.get("VERIFIER_GPU_OK") != "1":
        sys.exit("[REFUSED] set VERIFIER_GPU_OK=1 to run the verifier GPU pass.")
    MAXPX = HIGH_PX // CAP_DIV[A.cap]

    rows = [json.loads(l) for l in open(J(DIV_DUMP[A.dataset])) if l.strip()]
    items, _ = _dgg.load_items(A.dataset, A.cap, 100000)
    img_by_idx = {str(it[0]): it[3] for it in items}
    q_by_idx = {str(it[0]): it[1] for it in items}
    rows = [r for r in rows if str(r["idx"]) in img_by_idx][:A.n]
    print(f"[verifier:{A.tag}] {A.dataset}: {len(rows)} questions (subsample n={A.n})", flush=True)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def tok_ids(words):
        d = {}
        for w in words:
            for v in (w, " " + w):
                e = proc.tokenizer.encode(v, add_special_tokens=False)
                if len(e) == 1: d[e[0]] = w
        return d
    YES, NO = tok_ids(["Yes", "yes", "YES"]), tok_ids(["No", "no", "NO"])

    def build_verify(q, ans, img):
        body = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        msgs = [{"role": "system", "content": VERIFY_SYS},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs); req = {"prompt": text}
        if imgs: req["multi_modal_data"] = {"image": imgs}
        return req

    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    def p_yes(o):
        lps = (o.outputs[0].logprobs or [{}])[0]
        py = max((math.exp(v.logprob) for t, v in lps.items() if t in YES), default=0.0)
        pn = max((math.exp(v.logprob) for t, v in lps.items() if t in NO), default=0.0)
        return round(py / (py + pn), 6) if (py + pn) > 0 else 0.0

    os.makedirs(J(A.out_dir), exist_ok=True)
    outp = J(os.path.join(A.out_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl"))
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                try: done.add(str(json.loads(l)["idx"]))
                except Exception: pass
    todo = [r for r in rows if str(r["idx"]) not in done]
    print(f"  {len(todo)} questions to verify (resume: {len(done)} done) -> {outp}", flush=True)

    # flatten all (question, distinct-group) verify requests, keep spans to regroup
    reqs, span, meta = [], [], []
    for r in todo:
        k = str(r["idx"]); img = img_by_idx[k]; q = q_by_idx[k]
        groups = group_distinct(r)
        for g in groups:
            reqs.append(build_verify(q, g["ans"], img))
        span.append((r, groups, len(groups)));
    # dedup identical (represented) strings within a question is already handled by grouping.
    print(f"  {len(reqs)} verify calls over {len(todo)} questions", flush=True)

    # run in chunks, but write per-question only after all its groups are scored
    p_all = []
    CH = A.chunk
    for c0 in range(0, len(reqs), CH):
        outs = llm.generate(reqs[c0:c0 + CH], sp)
        p_all.extend(p_yes(o) for o in outs)
        print(f"   [{min(c0 + CH, len(reqs))}/{len(reqs)}]", flush=True)

    fh = open(outp, "a")
    pos = 0
    for (r, groups, ng) in span:
        ps = p_all[pos:pos + ng]; pos += ng
        for g, pv in zip(groups, ps): g["p_yes"] = pv
        # reference baselines carried from the dump for self-containedness
        modal_ok = int(r.get("modal_ok", 0))
        prov = r.get("provenance") or []
        greedy_ok = int(r["oks"][prov.index("base@t0.7")]) if "base@t0.7" in prov else int(r["oks"][0])
        oracle = int(any(g["ok"] for g in groups))
        fh.write(json.dumps({"idx": r["idx"], "gold": r.get("gold"), "n_distinct": len(groups),
                             "pool_size": len(r["preds"]), "modal_ok": modal_ok, "greedy_ok": greedy_ok,
                             "oracle": oracle, "groups": groups}) + "\n")
    fh.flush(); fh.close()
    print(f"[verifier:{A.tag}] DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
