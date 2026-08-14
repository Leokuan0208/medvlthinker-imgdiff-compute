#!/usr/bin/env python3
"""resolution_verifier_score.py -- SWEEP 2: score the swept candidate pools with the DEPLOYED
clean disjoint LoRA verifier, at the verifier's OWN deployed resolution.

DESIGN.  This sweep moves the GENERATOR's max_pixels.  The verifier is held at its deployed
configuration -- ckpts/train/lora_verifier_disjoint, bf16, max_pixels 1,003,520, min_pixels 3,136,
batch 1, HF transformers (NEVER vLLM: vLLM 0.9.0.1 silently drops all 192 visual.* LoRA modules,
0.775204 HF vs 0.702997 vLLM) -- because the verifier-resolution arm was already measured on
2026-08-12 (vram_levers_2026-08-12.json, open_half_levers).  Holding it fixed is what makes this
a single-variable sweep.

BECAUSE THE VERIFIER IS HELD FIXED, ITS SCORE IS A PURE FUNCTION OF (image, question, answer
string).  Two arms that emit the same answer string for the same item get the same score by
construction, so scores are cached by (ds, idx, RAW answer string) -- raw, not normalized,
because the raw string is what is fed to the model.  The cache is seeded from the stored deployed
transfer dumps, which makes the deployed pool's scores byte-identical to the published ones and
turns the overlap between arms into saved GPU time rather than into noise.

NULL TEST: --null_test re-scores n stored (idx, pred) pairs and prints the max abs deviation from
the stored score.  Run it before trusting any new number.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
     src/cascade_methods/resolution_verifier_score.py --null_test 200
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
     src/cascade_methods/resolution_verifier_score.py --run
"""
import argparse
import glob
import io
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
SWEEP = os.path.join(ROOT, "ckpts/openvqa/resolution_sweep")
DUMPS = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")
CACHE = os.path.join(SWEEP, "verifier_score_cache.json")
DS = ["slake_open", "vqa_rad_open", "pathvqa_open"]
DUMPNAME = {"slake_open": "slake", "vqa_rad_open": "vqa_rad", "pathvqa_open": "pathvqa"}

# verbatim from src/training_methods/verifier_transfer_eval.py (the script that produced the
# deployed transfer dumps)
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
       "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")


def imgs_for(ds):
    """verbatim from verifier_transfer_eval.imgs_for -- same images, same order, same filters."""
    m = {}
    if ds == "slake_open":
        for x in json.load(open("/data/dan/dataset/slake/test.json")):
            if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en":
                ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
                if os.path.exists(ip):
                    m[x["qid"]] = (x["question"], ip)
    else:
        import pandas as pd
        base = ("/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open"
                else "/data/dan/dataset/path_vqa/data")
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question")
            a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]
                q = conv[0]["value"].replace("<image>", "").strip()
                a = conv[1]["value"]
            if str(a).strip().lower() in ("yes", "no"):
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                m[int(i)] = (str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB"))
    return m


def seed_cache():
    c = {}
    for ds in DS:
        p = os.path.join(DUMPS, f"transfer_dump_{DUMPNAME[ds]}_open_lingshu7b.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            for a, s in zip(r["preds"], r["scores"]):
                c[f"{ds}|{r['idx']}|{a}"] = float(s)
    return c


def subsample_ids(k, seed=42):
    """A seeded k-per-set item subsample, identical for every arm (drawn from the canonical
    endpoint order, not from any arm's contents). Used only when the full pool cannot be scored
    under card contention; the artifact then reports n and the CI, never the point estimate alone."""
    keep = set()
    for ds, nm in [("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"),
                   ("pathvqa_open", "pathvqa")]:
        p = os.path.join(DUMPS, f"transfer_dump_{nm}_open_lingshu7b.json")
        ids = [r["idx"] for r in json.load(open(p))]
        rng = np.random.default_rng(seed)
        sel = ids if len(ids) <= k else [ids[i] for i in
                                         sorted(rng.choice(len(ids), size=k, replace=False))]
        keep |= {f"{ds}|{i}" for i in sel}
    return keep


def needed(keep=None):
    out = {}
    for f in sorted(glob.glob(os.path.join(SWEEP, "ckpt_*.jsonl"))):
        b = os.path.basename(f)
        ds = next((d for d in DS if b.startswith(f"ckpt_{d}_")), None)
        if ds is None:
            continue
        for l in open(f):
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if keep is not None and f"{ds}|{r['idx']}" not in keep:
                continue
            for a in r.get("preds", []):
                out[f"{ds}|{r['idx']}|{a}"] = (ds, r["idx"], a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--null_test", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--flush_every", type=int, default=250)
    ap.add_argument("--batch", type=int, default=1,
                    help="score this many candidates per forward (LEFT-padded). The deployed "
                         "verifier is batch 1; --batch >1 is a throughput measure whose effect is "
                         "quantified by --null_test before it is used for any number.")
    ap.add_argument("--subsample", type=int, default=0,
                    help="if >0, score only a seeded <k>-per-set item subsample (fallback for "
                         "card contention; the analysis then reports n and CI, not a bare point)")
    A = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False      # pinned; TF32 is worth -0.0089/+0.024 here
    torch.backends.cudnn.allow_tf32 = False

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else seed_cache()
    print(f"cache seeded/loaded: {len(cache)} scores", flush=True)
    keep = subsample_ids(A.subsample) if A.subsample else None
    need = needed(keep)
    todo = [v for k, v in need.items() if k not in cache]
    print(f"needed={len(need)}  to score={len(todo)}", flush=True)
    if not A.null_test and not todo:
        json.dump(cache, open(CACHE, "w"))
        return

    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to("cuda")
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    nvis = sum(1 for n, _ in model.named_parameters() if "lora" in n and "visual." in n)
    print(f"LoRA tensors on visual.*: {nvis}", flush=True)

    IMG = {}

    def _msgs(ds, idx, ans):
        if ds not in IMG:
            IMG[ds] = imgs_for(ds)
        q, img = IMG[ds][idx]
        return [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                    {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                             f"Is the proposed answer correct? Answer Yes or No."}]}]

    def pyes(ds, idx, ans):
        msgs = _msgs(ds, idx, ans)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
            py = math.exp(lg[YES].item())
            pn = math.exp(lg[NO].item())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    def pyes_batch(triples):
        """LEFT-padded batch; logits[:, -1] is the last real token of every row."""
        texts, all_imgs = [], []
        for ds, idx, ans in triples:
            m = _msgs(ds, idx, ans)
            texts.append(proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True))
            igs, _v = process_vision_info(m)
            all_imgs.extend(igs)
        old = proc.tokenizer.padding_side
        proc.tokenizer.padding_side = "left"
        try:
            enc = proc(text=texts, images=all_imgs, videos=None, return_tensors="pt",
                       padding=True).to("cuda")
            with torch.no_grad():
                lg = model(**enc).logits[:, -1]
            py = torch.exp(lg[:, YES].float())
            pn = torch.exp(lg[:, NO].float())
            out = (py / (py + pn)).cpu().numpy()
        finally:
            proc.tokenizer.padding_side = old
        return [float(x) if np.isfinite(x) else 0.5 for x in out]

    if A.null_test:
        rng = np.random.default_rng(0)
        stored = seed_cache()
        keys = sorted(stored)
        pick = [keys[i] for i in rng.choice(len(keys), size=min(A.null_test, len(keys)), replace=False)]
        trip = []
        for k in pick:
            ds, idx, a = k.split("|", 2)
            trip.append((ds, int(idx) if idx.lstrip("-").isdigit() else idx, a))
        got1 = [pyes(*t) for t in trip]
        dev = np.array([abs(g - stored[k]) for g, k in zip(got1, pick)])
        rep = {"batch1_vs_stored": {
            "n": len(dev), "max_abs_dev": float(dev.max()), "mean_abs_dev": float(dev.mean()),
            "n_above_1e_3": int((dev > 1e-3).sum()),
            "_what": "batch-1 re-score of stored (item, candidate) pairs vs the score in the "
                     "deployed transfer dump. Same code path as verifier_transfer_eval.py."}}
        print(f"NULL TEST batch1-vs-stored n={len(dev)} max_abs_dev={dev.max():.3e} "
              f"mean={dev.mean():.3e} n>1e-3={int((dev > 1e-3).sum())}", flush=True)
        if A.batch > 1:
            gotB = []
            for i in range(0, len(trip), A.batch):
                gotB.extend(pyes_batch(trip[i:i + A.batch]))
            db = np.abs(np.array(gotB) - np.array(got1))
            rep["batchN_vs_batch1"] = {
                "batch": A.batch, "n": len(db), "max_abs_dev": float(db.max()),
                "mean_abs_dev": float(db.mean()), "n_above_1e_3": int((db > 1e-3).sum()),
                "_what": "the LEFT-padded batched scorer against the batch-1 scorer on the same "
                         "pairs, in the same process. Batching is a throughput measure only and is "
                         "used for production scoring only if this deviation is negligible."}
            print(f"NULL TEST batch{A.batch}-vs-batch1 max_abs_dev={db.max():.3e} "
                  f"mean={db.mean():.3e} n>1e-3={int((db > 1e-3).sum())}", flush=True)
        json.dump(rep, open(os.path.join(SWEEP, "verifier_null_test.json"), "w"), indent=1)
        if not A.run:
            return

    t0 = time.time()
    # PRIORITY ORDER: the arms that answer the decisive question first, so a run that is cut short
    # by card contention still yields the cap320-vs-native comparison rather than a partial sweep.
    # Within that, group by item so a batch shares one image (less padding, fewer image encodes).
    prio = set()
    for tag in os.environ.get("VERIF_PRIORITY_ARMS",
                              "cap320_s0,native_s0,cap320_t0,native_t0").split(","):
        for ds in DS:
            f = os.path.join(SWEEP, f"ckpt_{ds}_{tag.strip()}.jsonl")
            if not os.path.exists(f):
                continue
            for l in open(f):
                if l.strip():
                    r = json.loads(l)
                    for a in r.get("preds", []):
                        prio.add(f"{ds}|{r['idx']}|{a}")
    todo.sort(key=lambda t: (f"{t[0]}|{t[1]}|{t[2]}" not in prio, t[0], str(t[1])))
    print(f"  priority arms cover {sum(1 for t in todo if f'{t[0]}|{t[1]}|{t[2]}' in prio)} "
          f"of {len(todo)} to-score", flush=True)
    done_n = 0
    for i in range(0, len(todo), A.batch):
        chunk = todo[i:i + A.batch]
        try:
            vals = pyes_batch(chunk) if A.batch > 1 else [pyes(*chunk[0])]
            for (ds, idx, a), v in zip(chunk, vals):
                cache[f"{ds}|{idx}|{a}"] = v
        except Exception as e:          # per-chunk guard, then per-item retry at batch 1
            print(f"  CHUNK FAIL {i}: {type(e).__name__} {e}", flush=True)
            for ds, idx, a in chunk:
                try:
                    cache[f"{ds}|{idx}|{a}"] = pyes(ds, idx, a)
                except Exception as e2:
                    print(f"  ITEM FAIL {ds} {idx}: {type(e2).__name__} {e2}", flush=True)
        done_n += len(chunk)
        if done_n % max(A.flush_every, A.batch) < A.batch:
            json.dump(cache, open(CACHE, "w"))
            print(f"  [{done_n}/{len(todo)}] {done_n / (time.time() - t0):.2f}/s", flush=True)
    json.dump(cache, open(CACHE, "w"))
    print(f"DONE {len(todo)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
