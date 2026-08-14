#!/usr/bin/env python3
"""vision_diversity_score.py -- SWEEP 3 scoring pass.

Score every UNIQUE (question, normalized-answer) pair produced by vision_diversity_gen.py with the
CLEAN disjoint LoRA verifier under **HF transformers** -- never vLLM (vLLM 0.9.0.1 silently drops all
192 visual.* LoRA modules: the same adapter scores sel_eff 0.775204 under HF and 0.702997 under vLLM).

THE SCORER IS HELD FIXED.  It always sees the ORIGINAL, untransformed image at the incumbent's
convention (max_pixels = 1280*28*28, min_pixels = 4*28*28), i.e. byte-identical to
src/training_methods/verifier_transfer_eval.py::pyes.  The varied factor in this sweep is the
GENERATOR's view; letting the verifier's view move too would confound the two.

Resumable: one JSON line per (idx, na); restarting skips completed pairs. Per-item try/except guard;
failures are recorded with score=null and counted.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 \
    src/cascade_methods/vision_diversity_score.py --datasets slake_open vqa_rad_open
"""
import argparse
import glob
import io
import json
import math
import os
import time
from collections import OrderedDict

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
    """(idx -> (question, image)). Verbatim from verifier_transfer_eval.imgs_for."""
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
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
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
    ap.add_argument("--gen_dir", default="ckpts/openvqa/visdiv")
    ap.add_argument("--out_dir", default="ckpts/openvqa/visdiv/scores")
    # Sharding is a THROUGHPUT lever only, never a numerics lever: each shard runs the identical
    # batch-1 forward pass, so scores are bit-identical to an unsharded run. Batching the answers
    # of one image would be far faster but would change padding/attention and therefore the
    # scores -- the incumbent's 0.775204 was measured at batch 1, so batch 1 it stays.
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    A = ap.parse_args()
    assert 0 <= A.shard < A.nshard

    # ---- numerics pins (protocol rule 8): TF32 OFF, seed 0 -- identical to open_diverse_score.py
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
        rows = [json.loads(l) for l in open(J(f"{A.gen_dir}/gen_{ds}.jsonl")) if l.strip()]
        IMG = imgs_for(ds)
        # repo shard-tag convention: no suffix when N==1; tagged only when genuinely sharded
        sfx = "" if A.nshard == 1 else f"_s{A.shard}of{A.nshard}"
        ckpt = J(f"{A.out_dir}/scores_{ds}_{A.tag}{sfx}.jsonl")
        # `done` is read from EVERY shard file of this dataset, not just our own: a pair scored by
        # any shard never needs scoring again (the score is a deterministic function of the pair).
        done = set()
        for dp in glob.glob(J(f"{A.out_dir}/scores_{ds}_{A.tag}.jsonl")) + \
                glob.glob(J(f"{A.out_dir}/scores_{ds}_{A.tag}_s*of*.jsonl")):
            for l in open(dp):
                if l.strip():
                    r = json.loads(l); done.add((str(r["idx"]), r["na"]))
        # unique (idx, na) across ALL views + the iid control, deterministic order
        seen = OrderedDict()
        for r in rows:
            for a in r.get("preds", []):
                seen.setdefault((str(r["idx"]), norm(a)), (r["idx"], r["question"], a))
        # PARTITION FIRST, THEN drop what is done. Doing it the other way round makes the shard
        # slice depend on resume state, so a restarted shard silently takes over another shard's
        # items -- duplicated work and no guarantee the union still covers everything.
        work = list(seen.items())
        if A.nshard > 1:
            work = work[A.shard::A.nshard]
        todo = [v for k, v in work if k not in done]
        print(f"[{ds}] {len(rows)} (idx,view) rows | {len(seen)} unique (idx,answer) pairs | "
              f"{len(todo)} to score ({len(done)} done) -> {os.path.basename(ckpt)}", flush=True)
        t0 = time.time(); nfail = 0
        with open(ckpt, "a") as fh:
            for k, (idx, q, surf) in enumerate(todo):
                try:
                    if idx not in IMG:
                        raise KeyError(f"no image for idx {idx}")
                    _q, img = IMG[idx]
                    s = pyes(_q, img, surf)
                    fh.write(json.dumps({"idx": idx, "na": norm(surf), "score": round(float(s), 5)}) + "\n")
                except Exception as e:                      # per-item error guard
                    nfail += 1
                    fh.write(json.dumps({"idx": idx, "na": norm(surf), "score": None,
                                         "err": repr(e)[:200]}) + "\n")
                if (k + 1) % 500 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"   [{k+1}/{len(todo)}] {el:.0f}s  {(k+1)/max(el,1e-9):.2f} it/s  fail={nfail}",
                          flush=True)
        print(f"[{ds}] SCORE_DONE fail={nfail} in {time.time()-t0:.0f}s", flush=True)
    print("ALL_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
