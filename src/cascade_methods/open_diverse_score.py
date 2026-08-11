#!/usr/bin/env python3
"""open_diverse_score.py -- ATTACK 4 (OPEN-DIVERSE), Phase 0b scoring pass.

Score every UNIQUE (question, normalized-answer) pair of the M=15 diverse portfolio pools with a
pointwise LoRA verifier, using **HF transformers** -- never vLLM.

WHY THIS EXISTS.  The published diverse-generation result
(results/cascade_methods/artifacts/diverse_generation_gpu.json) carries TWO defects that make its
sel_eff numbers unusable as a gate:
  (1) its verifier is the CONTAMINATED ckpts/train/lora_verifier_pooled4;
  (2) its scores were produced *inside* src/cascade_methods/diversity_generate_gpu.py, which scores
      with vLLM `LoRARequest` (see its phase_generate: enable_lora=True + LoRARequest).  vLLM 0.9.0.1
      SILENTLY DROPS ALL 192 visual.* LoRA modules -- the same adapter scores sel_eff 0.775204 under
      HF and 0.702997 under vLLM.  So the published diverse scores are from a *visually blind* verifier.
This script re-scores the same pools under HF with a chosen adapter so the gate can be decided in the
frozen currency (judge labels + HF-scored disjoint verifier).

Scoring function is byte-identical to src/training_methods/verifier_transfer_eval.py::pyes -- same SYS
string, same max_pixels/min_pixels, same Yes/No logit readout, same 5-dp rounding at write time.

Resumable: appends one JSON line per (idx, na) to the checkpoint; restarting skips completed pairs.
Per-item try/except guard; failures are recorded with score=null and counted.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 src/cascade_methods/open_diverse_score.py \
      --adapter ckpts/train/lora_verifier_disjoint --tag disjoint
"""
import argparse, os, glob, io, json, math, time
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
J = lambda p: os.path.join(ROOT, p)

SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28          # verifier_transfer_eval.py convention (fullres)
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]


def norm(s):
    return str(s).strip().lower()


def imgs_for(ds):
    """(idx -> (question, image)).  Verbatim from verifier_transfer_eval.imgs_for."""
    m = {}
    if ds == "slake_open":
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path",
                    default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/"
                            "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--tag", default="disjoint")
    ap.add_argument("--datasets", nargs="+", default=DS)
    ap.add_argument("--div_dir", default="ckpts/openvqa/diverse")
    ap.add_argument("--out_dir", default="ckpts/openvqa/diverse/scores")
    ap.add_argument("--pool", choices=["diverse", "iid_mcq"], default="diverse",
                    help="'diverse' = the M=15 portfolio pool; 'iid_mcq' = the mcq_gen_verify iid-8 pool "
                         "the published diverse arm was matched to (restricted to the diverse idx set). "
                         "Scoring BOTH under the same adapter removes the generation-run confound from "
                         "the iid-vs-portfolio contrast.")
    A = ap.parse_args()

    MCQ_IID = {"slake_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_SLAKE_lingshu7b_SLAKE_content_content_sc8.jsonl",
               "vqa_rad_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_VQA_RAD_lingshu7b_VQA_RAD_content_content_sc8.jsonl",
               "pathvqa_open": "ckpts/mcq_gen_verify/lingshu7b/ckpt_PATH_VQA_lingshu7b_content_sc8.jsonl"}

    # ---- numerics pins (protocol rule 8) -------------------------------------------------
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    os.makedirs(J(A.out_dir), exist_ok=True)
    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    print("loading base + adapter...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    model = PeftModel.from_pretrained(model, J(A.adapter))
    model.eval()

    def pyes(q, img, ans):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                    {"type": "text",
                     "text": f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    for ds in A.datasets:
        div = [json.loads(l) for l in open(J(f"{A.div_dir}/ckpt_{ds}_lingshu7b_div.jsonl")) if l.strip()]
        if A.pool == "iid_mcq":
            keep = {str(r["idx"]) for r in div}
            div = [json.loads(l) for l in open(J(MCQ_IID[ds])) if l.strip()]
            div = [r for r in div if str(r["idx"]) in keep]
        IMG = imgs_for(ds)
        suffix = "" if A.pool == "diverse" else "_iidmcq"
        ckpt = J(f"{A.out_dir}/scores_{ds}_{A.tag}{suffix}.jsonl")
        done = set()
        if os.path.exists(ckpt):
            for l in open(ckpt):
                if l.strip():
                    r = json.loads(l); done.add((str(r["idx"]), r["na"]))
        # unique (idx, na) pairs, deterministic order
        todo = []
        for r in div:
            seen = OrderedDict()
            for a in r["preds"]:
                seen.setdefault(norm(a), a)
            for na, surf in seen.items():
                if (str(r["idx"]), na) not in done:
                    todo.append((r["idx"], r["question"], na, surf))
        print(f"[{ds}] {len(div)} questions | {len(todo)} unique (idx,answer) pairs to score "
              f"({len(done)} already done) -> {os.path.basename(ckpt)}", flush=True)
        t0 = time.time(); nfail = 0
        with open(ckpt, "a") as fh:
            for k, (idx, q, na, surf) in enumerate(todo):
                try:
                    if idx not in IMG:
                        raise KeyError(f"no image for idx {idx}")
                    _q, img = IMG[idx]
                    s = pyes(_q, img, surf)
                    fh.write(json.dumps({"idx": idx, "na": na, "score": round(float(s), 5)}) + "\n")
                except Exception as e:                      # per-item error guard
                    nfail += 1
                    fh.write(json.dumps({"idx": idx, "na": na, "score": None, "err": repr(e)[:200]}) + "\n")
                if (k + 1) % 200 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"   [{k+1}/{len(todo)}] {el:.0f}s  {(k+1)/max(el,1e-9):.2f} it/s  fail={nfail}", flush=True)
        print(f"[{ds}] DONE fail={nfail} in {time.time()-t0:.0f}s", flush=True)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
