#!/usr/bin/env python3
"""verifier_vision_capacity_ablation.py -- ATTACK 1(c), done by ABLATION instead of by retraining.

THE QUESTION 1(c) ASKS.  The deployed clean verifier (ckpts/train/lora_verifier_disjoint) spends
7,219,200 of its 47,589,376 LoRA parameters (15.17%) on the vision tower, purely because
target_modules names {q,k,v,o,up,gate,down}_proj match blocks in both the LM and the ViT.  1(c)
proposed retraining with MORE vision capacity to see whether that allocation is under-served.

WHY ABLATION IS THE BETTER EXPERIMENT HERE.  A retrain costs ~108 min/seed plus a full scoring pass,
and this project has documented a seed-to-seed spread of ~0.021 sel_eff that EXCEEDS every
architectural effect ever measured on this endpoint (docs/current/COMPARATIVE_VERIFIER_2026-08-05.md).
One or two seeds of a re-trained LoRA would therefore be uninterpretable -- exactly the error this
project has been burned by.  The ablation instead asks the SAME question with ZERO training
variance: take the deployed adapter and switch its vision-tower capacity OFF (and, separately, its
language capacity off), and measure what each half was contributing.  It is deterministic: the
adapter is fixed, so there are no seeds to average.

  full            the deployed adapter, untouched                 -- FIDELITY GATE
  no_visual_lora  zero lora_B on all 96 visual.* modules           -- what the vision capacity bought
  visual_only     zero lora_B on all 196 language_model.* modules  -- what vision capacity ALONE buys

A FACT WORTH RECORDING (verified from the adapter's own tensor names): all 192 visual tensors are
mlp.{down,gate,up}_proj on the 32 ViT blocks.  The ViT's ATTENTION is never adapted, because
Qwen2.5-VL names it attn.qkv / attn.proj, which match none of the target_modules.  So the
"incidental 15.2% on vision" is feed-forward only, and the spatial-mixing part of the vision tower
-- the part a laterality question would need -- carries no adapter capacity at all.

Zeroing lora_B is exact: the LoRA update is B @ A * scaling, so B = 0 removes the module's delta
while leaving every other weight, the base model and all shapes untouched.

RESUMABLE: one JSONL line per (arm, ds, idx) with the 8 candidate scores; a rerun skips completed
keys.  Per-item try/except so one bad image cannot lose the pass.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 -u \
    src/training_methods/verifier_vision_capacity_ablation.py --arms full no_visual_lora
"""
import argparse, glob, io, json, math, os, sys, time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import genframe_data as G  # noqa: E402

SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
       "proposed answer is correct. Respond with only 'Yes' or 'No'.")
MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
CKDIR = os.path.join(ROOT, "results/cascade_methods/artifacts/_visverif_parts")


def imgs_for(ds):
    """Identical to verifier_transfer_eval.imgs_for (including its explicit slake_open branch --
    the else-branch bug that would have served PathVQA images to SLAKE is avoided here too)."""
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
        df = pd.concat([pd.read_parquet(f) for f in
                        sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
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


def lora_modules(model):
    """(name, LoraLayer) for every adapted module."""
    out = []
    for name, mod in model.named_modules():
        lb = getattr(mod, "lora_B", None)
        if lb is not None and hasattr(lb, "keys") and len(lb) > 0:
            out.append((name, mod))
    return out


def apply_arm(model, arm, saved):
    """Restore all lora_B, then zero the half this arm switches off. Returns a count report."""
    mods = lora_modules(model)
    for name, mod in mods:
        for k, lin in mod.lora_B.items():
            lin.weight.data.copy_(saved[(name, k)])
    if arm == "full":
        return {"zeroed": 0, "kept": len(mods)}
    z = 0
    for name, mod in mods:
        is_vis = "visual" in name
        kill = is_vis if arm == "no_visual_lora" else (not is_vis)
        if kill:
            for k, lin in mod.lora_B.items():
                lin.weight.data.zero_()
            z += 1
    return {"zeroed": z, "kept": len(mods) - z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--arms", nargs="+", default=["full", "no_visual_lora", "visual_only"])
    ap.add_argument("--datasets", nargs="+", default=["slake_open", "vqa_rad_open", "pathvqa_open"])
    ap.add_argument("--fidelity_n", type=int, default=250,
                    help="how many items of the 'full' arm to score for the fidelity gate (0 = all)")
    ap.add_argument("--mixed_only", type=int, default=1,
                    help="score only items whose pool contains BOTH a correct and an incorrect "
                         "candidate. On every other item the pick cannot change the outcome, so "
                         "those forwards carry no information about a SELECTION difference. "
                         "n=852 of 2345 (slake 204 / vqa_rad 73 / pathvqa 575).")
    ap.add_argument("--limit_items", type=int, default=0,
                    help="cap items per (arm, dataset) after the mixed filter; 0 = no cap. The "
                         "subsample is a FIXED shuffle (seed 20260812) so every arm scores the "
                         "SAME items and the comparison stays paired.")
    ap.add_argument("--ckpt", default=os.path.join(CKDIR, "vision_capacity_scores.jsonl"))
    ap.add_argument("--deadline_min", type=float, default=600.0)
    A = ap.parse_args()
    DEV = "cuda"
    os.makedirs(CKDIR, exist_ok=True)

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    items = G.load_items()
    if A.mixed_only:
        items = [it for it in items
                 if any(x == 1 for x in it["sl"]) and any(x == 0 for x in it["sl"])]
        print(f"[filter] mixed-only: {len(items)} items whose pick can change the outcome",
              flush=True)
    by_ds = defaultdict(list)
    for it in items:
        by_ds[it["ds"]].append(it)
    if A.limit_items:
        rs = np.random.default_rng(20260812)
        for ds in by_ds:
            order = rs.permutation(len(by_ds[ds]))
            by_ds[ds] = [by_ds[ds][i] for i in order[:A.limit_items]]
        print(f"[filter] capped to {A.limit_items}/dataset: "
              + ", ".join(f"{d}={len(v)}" for d, v in by_ds.items()), flush=True)

    done = set()
    if os.path.exists(A.ckpt):
        for line in open(A.ckpt):
            try:
                r = json.loads(line)
                done.add((r["arm"], r["ds"], str(r["idx"])))
            except Exception:
                pass
    print(f"[resume] {len(done)} (arm,item) already scored", flush=True)

    proc = AutoProcessor.from_pretrained(A.model_path)
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    print("loading base + adapter ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()

    mods = lora_modules(model)
    saved = {(n, k): m.lora_B[k].weight.data.clone() for n, m in mods for k in m.lora_B}
    nvis = sum(1 for n, _ in mods if "visual" in n)
    print(f"[adapter] {len(mods)} adapted modules; {nvis} visual, {len(mods)-nvis} language",
          flush=True)

    def pyes(q, img, ans):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": [
                    {"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MINPX},
                    {"type": "text", "text": f"Question: {q}\nProposed answer: {ans}\n"
                                             f"Is the proposed answer correct? Answer Yes or No."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        igs, vids = process_vision_info(msgs)
        enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            lg = model(**enc).logits[0, -1]
        py = math.exp(lg[YES].item()); pn = math.exp(lg[NO].item())
        return py / (py + pn) if (py + pn) > 0 else 0.5

    t0 = time.time()
    fh = open(A.ckpt, "a")
    IMGC = {}
    for arm in A.arms:
        rep = apply_arm(model, arm, saved)
        print(f"[arm {arm}] zeroed lora_B on {rep['zeroed']} modules, kept {rep['kept']}", flush=True)
        for ds in A.datasets:
            if ds not in IMGC:
                IMGC[ds] = imgs_for(ds)
            IMG = IMGC[ds]
            todo = [it for it in by_ds[ds] if (arm, ds, str(it["idx"])) not in done]
            if arm == "full" and A.fidelity_n:
                todo = todo[:A.fidelity_n]
            for j, it in enumerate(todo):
                if (time.time() - t0) / 60 > A.deadline_min:
                    print("[deadline] stopping cleanly", flush=True); fh.close(); return
                try:
                    qi = IMG.get(it["idx"])
                    if qi is None:
                        continue
                    q, img = qi
                    sc = [pyes(q, img, a) for a in it["preds"]]
                    fh.write(json.dumps({"arm": arm, "ds": ds, "idx": it["idx"],
                                         "scores": [round(float(x), 6) for x in sc]}) + "\n")
                    fh.flush()
                except Exception as e:
                    print(f"  skip {arm}/{ds}/{it['idx']}: {str(e)[:110]}", flush=True)
                if (j + 1) % 100 == 0:
                    el = (time.time() - t0) / 60
                    print(f"  [{arm}/{ds}] {j+1}/{len(todo)}  {el:.1f}min", flush=True)
    fh.close()
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
