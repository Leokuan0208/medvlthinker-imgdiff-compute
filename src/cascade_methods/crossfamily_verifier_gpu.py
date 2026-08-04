#!/usr/bin/env python3
"""crossfamily_verifier_gpu.py -- score ONE fixed candidate pool with an ARBITRARY-FAMILY
zero-shot pointwise verifier.

WHY. The selection deficit has been shown to be an INFORMATION deficit, not a format deficit:
a 7B verifier is no better at picking the right candidate than the 7B generator that produced
them (choicewhy_measure_2026-08-03.json: -0.0024 for adding justifications; verifier_32b_gpu.json:
a 4.5x larger SAME-FAMILY verifier only ties). The hypothesis under test here is that a verifier
from a DIFFERENT MODEL FAMILY holds information the generator does not, and should therefore
select better than a same-family verifier of equal or larger size.

DESIGN (mechanism-clean, everything held fixed except the verifier weights):
  * POOL      : ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl -- the SAME 8-sample
                Lingshu-7B pools the published comparator (verifier_disjoint_retrain_2026-07-30.json)
                is measured on. Never regenerated.
  * LABELS    : ckpt_{ds}_lingshu7b_sc8_scexploded.judge.jsonl -- the SAME 32B LLM judge as the headline.
  * PROMPT    : the EXACT grader prompt used to produce the trained-7B scores
                (diversity_generate_gpu.VERIFY_SYS + build_verify body), verbatim.
  * IMAGE     : every family gets the same picture at the same budget -- the PIL image is downscaled
                to <= --max_pixels (default 250880 px = the cap320 budget the pools were generated
                at) and passed as a base64 data URI through vLLM's llm.chat(), so each model's own
                chat template / image-token handling is applied automatically. This is the only way
                to hold the *content* identical across families whose processors differ.
  * SCORE     : p_yes = P(Yes) / (P(Yes) + P(No)) from the first generated token's top-20 logprobs.
  * DEDUP     : identical normalized answers inside a question are scored ONCE (a deterministic
                verifier gives byte-identical prompts the same score); the score is broadcast back
                to every slot at measure time, so candidate-level AUROC is still over all 8 slots.

Writes RAW per-question verdicts; the CPU step (crossfamily_verifier_measure.py) does all analysis.
GUARDED: refuses to launch unless CROSSFAM_GPU_OK=1. One dataset per invocation so an external
`timeout` bounds each. Resumable per question.

  CROSSFAM_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/crossfamily_verifier_gpu.py --dataset slake_open \
      --model_path <path> --tag <tag> --tp 1
"""
import argparse, base64, io, json, math, os, sys, importlib.util

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(REPO, p)
_spec = importlib.util.spec_from_file_location("dgg", J("src/cascade_methods/diversity_generate_gpu.py"))
_dgg = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_dgg)

DSETS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
POOL = "ckpts/openvqa/cheap_lingshu7b/ckpt_{ds}_lingshu7b_sc8.jsonl"
# EXACT grader prompt that produced the trained-7B `scores` (diversity_generate_gpu.VERIFY_SYS)
VERIFY_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
              "the proposed answer is correct. Respond with only 'Yes' or 'No'.")
CAP320_PX = 1280 * 28 * 28 // 4          # 250880 -- the budget the pools were generated at

norm = lambda s: str(s).strip().lower()


def to_data_uri(img, max_pixels):
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        r = (max_pixels / (w * h)) ** 0.5
        img = img.resize((max(28, int(w * r)), max(28, int(h * r))))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=DSETS)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="ckpts/openvqa/crossfam_verifier")
    ap.add_argument("--max_pixels", type=int, default=CAP320_PX)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--no_system", action="store_true",
                    help="family has no system role: prepend VERIFY_SYS to the user text instead")
    ap.add_argument("--content_format", default="auto", choices=["auto", "string", "openai"],
                    help="vLLM chat_template_content_format. InternVL-family templates concatenate "
                         "content as a plain string and MUST use 'string'; Qwen-VL works with 'auto'.")
    A = ap.parse_args()
    if os.environ.get("CROSSFAM_GPU_OK") != "1":
        sys.exit("[REFUSED] set CROSSFAM_GPU_OK=1 to run the cross-family verifier GPU pass.")

    rows = [json.loads(l) for l in open(J(POOL.format(ds=A.dataset))) if l.strip()]
    items, _ = _dgg.load_items(A.dataset, "cap320", 100000)
    img_by_idx = {str(it[0]): it[3] for it in items}
    rows = [r for r in rows if str(r["idx"]) in img_by_idx]
    print(f"[{A.tag}] {A.dataset}: {len(rows)} questions", flush=True)

    os.makedirs(J(A.out_dir), exist_ok=True)
    outp = J(os.path.join(A.out_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl"))
    done = set()
    if os.path.exists(outp):
        for l in open(outp):
            if l.strip():
                try: done.add(str(json.loads(l)["idx"]))
                except Exception: pass
    todo = [r for r in rows if str(r["idx"]) not in done]
    print(f"  {len(todo)} to do (resume {len(done)}) -> {outp}", flush=True)
    if not todo:
        print(f"[{A.tag}] {A.dataset} ALREADY COMPLETE"); return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(A.model_path, trust_remote_code=True)

    def yn_ids(words):
        d = {}
        for w in words:
            for v in (w, " " + w):
                e = tok.encode(v, add_special_tokens=False)
                if len(e) == 1: d[e[0]] = w
        return d
    YES, NO = yn_ids(["Yes", "yes", "YES"]), yn_ids(["No", "no", "NO"])
    if not YES or not NO:
        sys.exit(f"[ABORT] {A.tag}: no single-token Yes/No ids (YES={YES} NO={NO})")

    def build(q, ans, uri):
        body = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        if A.no_system:
            return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                                 {"type": "text", "text": VERIFY_SYS + "\n\n" + body}]}]
        return [{"role": "system", "content": VERIFY_SYS},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                             {"type": "text", "text": body}]}]

    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    def p_yes(o):
        lps = (o.outputs[0].logprobs or [{}])[0]
        py = max((math.exp(v.logprob) for t, v in lps.items() if t in YES), default=0.0)
        pn = max((math.exp(v.logprob) for t, v in lps.items() if t in NO), default=0.0)
        return (round(py / (py + pn), 6) if (py + pn) > 0 else None)

    kw = {} if A.content_format == "auto" else {"chat_template_content_format": A.content_format}
    fh = open(outp, "a")
    B = A.chunk
    n_scored = n_none = 0
    for c0 in range(0, len(todo), B):
        block = todo[c0:c0 + B]
        convs, span = [], []
        for r in block:
            uri = to_data_uri(img_by_idx[str(r["idx"])], A.max_pixels)
            uniq = []
            for p in r["preds"]:
                if norm(p) not in [u[0] for u in uniq]: uniq.append((norm(p), p))
            for _, raw in uniq:
                convs.append(build(r["question"], raw, uri))
            span.append((r, [u[0] for u in uniq]))
        try:
            outs = llm.chat(convs, sp, use_tqdm=False, **kw)
        except Exception as e:
            print(f"   chunk failed ({str(e)[:160]}); one-by-one", flush=True)
            outs = []
            for cv in convs:
                try: outs.append(llm.chat([cv], sp, use_tqdm=False, **kw)[0])
                except Exception as e2:
                    print(f"     skip: {str(e2)[:90]}", flush=True); outs.append(None)
        pos = 0
        for (r, keys) in span:
            ps = outs[pos:pos + len(keys)]; pos += len(keys)
            scores = {k: (p_yes(o) if o is not None else None) for k, o in zip(keys, ps)}
            n_scored += len(scores); n_none += sum(1 for v in scores.values() if v is None)
            fh.write(json.dumps({"idx": r["idx"], "n_distinct": len(keys),
                                 "pool_size": len(r["preds"]), "scores_by_answer": scores}) + "\n")
        fh.flush()
        print(f"   [{min(c0 + B, len(todo))}/{len(todo)} questions] null_scores={n_none}/{n_scored}", flush=True)
        # fail loudly on a silently-broken chat template rather than writing a file of Nones
        if n_none == n_scored and n_scored > 0:
            fh.close()
            sys.exit(f"[ABORT] {A.tag}/{A.dataset}: every score is null after the first chunk "
                     f"(broken chat template / Yes-No ids?). Try --content_format string or --no_system.")
    fh.close()
    print(f"[{A.tag}] {A.dataset} DONE -> {outp}", flush=True)


if __name__ == "__main__":
    main()
