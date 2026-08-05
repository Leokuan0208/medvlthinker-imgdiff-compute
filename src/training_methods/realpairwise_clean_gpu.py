#!/usr/bin/env python3
"""realpairwise_clean_gpu.py -- CLEAN GPU replication of the REAL pairwise verifier.

WHY. The project's only prior selection win (pointwise 0.783 -> knockout 0.849 -> round-robin
0.859 sel_eff) was measured with the CONTAMINATED ckpts/train/lora_verifier_pooled4 adapter,
on n=578, on the ckpts/mcq_gen_verify/... candidate pool. This script re-runs the SAME
mechanism with the CLEAN, disjoint-trained adapter (ckpts/train/lora_verifier_disjoint) on the
CURRENT 2345-question eval pool whose incumbent bar is sel_eff 0.775204.

WHAT IS HELD VERBATIM from src/cascade_methods/pairwise_verifier_score.py (so this is a
DECONTAMINATION test and not a confounded re-design):
  * system prompt PAIR_SYS                                    -- byte-identical
  * user body "Question / Answer A / Answer B / Which ..."    -- byte-identical
  * P(A) read from the first generated token's logprobs over the {A, ' A'} / {B, ' B'} id sets
  * both orders scored; p_i_gt_j = 0.5*(P(A | i=A) + P(B | i=B))
  * identical normalized answer strings -> p = 0.5, no GPU call

WHAT DELIBERATELY CHANGES, and why (stated so nothing is silently confounded):
  * adapter    pooled4 (contaminated)  -> lora_verifier_disjoint (clean)      <- THE TEST
  * pool       mcq_gen_verify n=578    -> transfer-dump pool n=2345           <- the current bar
  * max_pixels cap320 (250880)         -> 1003520 (fullres)
        The clean adapter was TRAINED at max_pixels=1003520 (train_config.json) and the
        incumbent pointwise bar 0.775204 was MEASURED at 1003520
        (verifier_transfer_eval.py: MAXPX = 1280*28*28). Running the pairwise arm at cap320
        would compare a fullres pointwise arm against a cap320 pairwise arm. Matching the
        incumbent's own conditions is the only way the pairwise-vs-pointwise delta on THIS
        pool isolates the prompt frame.

COST SHAPE. Pairs are enumerated over DISTINCT normalized answers (a question contributes
1..8 distinct strings, mean 3.81). Full round-robin over distinct candidates is 19,952
unordered pairs = 39,904 ordered forward passes, and KNOCKOUT is a strict subset of those
pairs, so one full round-robin pass yields BOTH aggregators offline at no extra GPU cost.

Resumable: one JSONL line per ORDERED request, keyed (idx, ai, bi, order). Per-chunk error
guard; a failed chunk is retried once one request at a time and then skipped with a logged
error row, so a single bad image cannot kill a run.

  PAIRWISE_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    /data/dan/medeval_venv/bin/python src/training_methods/realpairwise_clean_gpu.py \
      --dataset slake_open --out_dir ckpts/pairwise_clean
"""
import argparse, os, sys, json, math, io, glob, itertools, time
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G  # noqa: E402

# ------------------------------------------------------------------ verbatim prompt
PAIR_SYS = ("You are a careful medical exam grader. Given a medical image, a question, and two candidate "
            "answers (A and B), decide which candidate answer is more likely correct. Respond with only 'A' or 'B'.")
# VERBATIM from verifier_transfer_eval.py -- the pointwise prompt that produced the 0.775204 bar.
# Used here only for the ENGINE-MATCHED pointwise control (see --mode pointwise below).
POINT_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
             "proposed answer is correct. Respond with only 'Yes' or 'No'.")
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28


def imgs_for(ds):
    """VERBATIM from src/training_methods/verifier_transfer_eval.py -- the loader that
    produced the transfer dumps this pool is defined by."""
    m = {}
    if ds in ("kvasir_open", "radimagenet_open"):
        jp = ("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json" if ds == "kvasir_open"
              else "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")
        for r in json.load(open(jp)):
            if os.path.exists(r["img_path"]):
                m[r["idx"]] = (r["question"], r["img_path"])
    elif ds == "slake_open":
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
                ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                if os.path.exists(ip):
                    m[x["qid"]] = (x["question"], ip)
    else:
        import pandas as pd
        base = "/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))],
                       ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


def distinct_cands(preds):
    """First-occurrence-ordered distinct normalized answers + the slot list of each."""
    order, slots = [], defaultdict(list)
    for k, a in enumerate(preds):
        na = G.norm(a)
        if na not in slots:
            order.append(na)
        slots[na].append(k)
    # representative surface text = the first slot's raw string (what the old code passed)
    text = {na: preds[slots[na][0]] for na in order}
    return order, slots, text


def plan(ds, items, img_map):
    """Per-question pair plan. Returns list of dicts and coverage stats."""
    qs, missing = [], []
    for it in items:
        if it["ds"] != ds:
            continue
        if it["idx"] not in img_map:
            missing.append(it["idx"]); continue
        q, img = img_map[it["idx"]]
        order, slots, text = distinct_cands(it["preds"])
        pairs = list(itertools.combinations(range(len(order)), 2))
        qs.append({"idx": it["idx"], "q": q, "img": img, "na": order, "slots": slots,
                   "text": text, "pairs": pairs})
    return qs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=G.EVAL_DS)
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint",
                    help="CLEAN disjoint-trained verifier LoRA, reused in the pairwise prompt frame")
    ap.add_argument("--tag", default="disjoint")
    ap.add_argument("--out_dir", default="ckpts/pairwise_clean")
    ap.add_argument("--max_pixels", type=int, default=HIGH_PX)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--plan_only", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1,
                    help="split the QUESTION list across processes; each shard writes its own jsonl")
    A = ap.parse_args()
    if not A.plan_only and os.environ.get("PAIRWISE_GPU_OK") != "1":
        sys.exit("[REFUSED] set PAIRWISE_GPU_OK=1 to run the pairwise GPU pass.")

    items = G.load_items()
    print(f"[pool] {len(items)} eval items total (canonical order)", flush=True)
    img_map = imgs_for(A.dataset)
    qs, missing = plan(A.dataset, items, img_map)
    if A.nshard > 1:
        qs = qs[A.shard::A.nshard]
    npairs = sum(len(x["pairs"]) for x in qs)
    print(f"[plan] {A.dataset}: {len(qs)} questions ({len(missing)} missing image), "
          f"{npairs} unordered distinct pairs -> {2*npairs} ordered GPU requests", flush=True)
    if missing:
        print(f"  !! missing idx sample: {missing[:10]}", flush=True)
    if A.plan_only:
        return

    os.makedirs(os.path.join(ROOT, A.out_dir), exist_ok=True)
    sfx = "" if A.nshard <= 1 else f"_s{A.shard}of{A.nshard}"
    outp = os.path.join(ROOT, A.out_dir, f"ordered_{A.dataset}_{A.tag}{sfx}.jsonl")
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            done.add((str(r["idx"]), int(r["ai"]), int(r["bi"]), int(r["order"])))
    print(f"[resume] {len(done)} ordered rows already on disk -> {outp}", flush=True)

    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def tok_ids(words):
        ids_ = {}
        for w in words:
            for v in (w, " " + w):
                e = proc.tokenizer.encode(v, add_special_tokens=False)
                if len(e) == 1:
                    ids_[e[0]] = w
        return ids_
    TA, TB = tok_ids(["A"]), tok_ids(["B"])
    print(f"[tok] A ids={sorted(TA)}  B ids={sorted(TB)}", flush=True)

    # one resized PIL per question -> vLLM's mm preprocessor cache hits across that
    # question's ~17 requests instead of re-encoding the image every time.
    def prep_img(img):
        msgs = [{"role": "user", "content": [{"type": "image", "image": img,
                                              "max_pixels": A.max_pixels, "min_pixels": MIN_PX}]}]
        igs, _ = process_vision_info(msgs)
        return igs[0]

    def build(q, pil, ansA, ansB):
        body = (f"Question: {q}\nAnswer A: {ansA}\nAnswer B: {ansB}\n"
                f"Which candidate answer is more likely correct, A or B? Respond with only A or B.")
        msgs = [{"role": "system", "content": PAIR_SYS},
                {"role": "user", "content": [{"type": "image", "image": pil},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return {"prompt": text, "multi_modal_data": {"image": pil}}

    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True,
              enable_lora=True, max_lora_rank=32)
    lora_req = LoRARequest("verifier", 1, os.path.join(ROOT, A.adapter))
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    def pA(o):
        lps = (o.outputs[0].logprobs or [{}])[0]
        pa = max((math.exp(v.logprob) for t, v in lps.items() if t in TA), default=0.0)
        pb = max((math.exp(v.logprob) for t, v in lps.items() if t in TB), default=0.0)
        return (pa / (pa + pb)) if (pa + pb) > 0 else 0.5, pa, pb

    fh = open(outp, "a")
    t0 = time.time()
    reqs, meta = [], []
    n_done = 0
    n_err = 0

    def flush_chunk():
        nonlocal reqs, meta, n_done, n_err
        if not reqs:
            return
        try:
            outs = llm.generate(reqs, sp, lora_request=lora_req)
        except Exception as e:                                   # chunk-level guard
            print(f"  !! chunk failed ({e}); retrying one at a time", flush=True)
            outs = []
            for r in reqs:
                try:
                    outs.append(llm.generate([r], sp, lora_request=lora_req)[0])
                except Exception as e2:
                    outs.append(None)
                    print(f"  !! request failed: {e2}", flush=True)
        for o, m in zip(outs, meta):
            if o is None:
                n_err += 1
                fh.write(json.dumps({**m, "p_first": None, "error": "generate_failed"}) + "\n")
                continue
            p, pa, pb = pA(o)
            fh.write(json.dumps({**m, "p_first": round(float(p), 6),
                                 "pa": round(float(pa), 6), "pb": round(float(pb), 6),
                                 "tok": (o.outputs[0].text or "")[:4]}) + "\n")
            n_done += 1
        fh.flush()
        reqs, meta = [], []

    total = 2 * npairs - len(done)
    seen = 0
    for qi, x in enumerate(qs):
        need = [(ai, bi, order) for (ai, bi) in x["pairs"] for order in (0, 1)
                if (str(x["idx"]), ai, bi, order) not in done]
        if not need:
            continue
        try:
            pil = prep_img(x["img"])
        except Exception as e:                                   # per-item guard
            print(f"  !! image prep failed idx={x['idx']}: {e}", flush=True)
            for (ai, bi, order) in need:
                fh.write(json.dumps({"ds": A.dataset, "idx": str(x["idx"]), "ai": ai, "bi": bi,
                                     "order": order, "p_first": None, "error": "image_prep"}) + "\n")
                n_err += 1
            fh.flush()
            continue
        for (ai, bi, order) in need:
            na_first, na_second = (x["na"][ai], x["na"][bi]) if order == 0 else (x["na"][bi], x["na"][ai])
            reqs.append(build(x["q"], pil, x["text"][na_first], x["text"][na_second]))
            meta.append({"ds": A.dataset, "idx": str(x["idx"]), "ai": ai, "bi": bi, "order": order})
            seen += 1
            if len(reqs) >= A.chunk:
                flush_chunk()
                el = time.time() - t0
                print(f"   [{seen}/{total}] q{qi+1}/{len(qs)}  {el/60:.1f} min  "
                      f"{seen/max(el,1e-9):.1f} req/s  err={n_err}", flush=True)
    flush_chunk()
    fh.close()
    print(f"[pairwise-clean] DONE {A.dataset}: wrote {n_done} ok / {n_err} err in "
          f"{(time.time()-t0)/60:.1f} min -> {outp}", flush=True)


if __name__ == "__main__":
    main()
