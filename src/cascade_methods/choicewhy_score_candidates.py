#!/usr/bin/env python3
"""choicewhy_score_candidates.py -- PHASE 3 scoring pass.

Score every N=8 evaluation candidate with a trained MCQ outcome verifier and dump P(Yes) per
candidate, so the offline analysis (choicewhy_measure.py) can compute selection efficiency.

The verifier prompt is src/cascade_methods/choicewhy_common.py::VERIF_SYS + verifier_body -- the
IDENTICAL prompt the verifier was trained with, and identical across arms; only the `candidate`
string differs (a bare letter in arm A, "<letter>. <one-sentence finding>" in arm B2).  FORMAT is
the only variable.

Candidates that are byte-identical within a question are scored ONCE and the score is copied to
every slot (the verifier is deterministic given its input, so this is exact, not an approximation).

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 \
    src/cascade_methods/choicewhy_score_candidates.py \
      --arm A --adapter ckpts/train/lora_verifier_choicewhy_A \
      --out ckpts/choicewhy_pilot/scores_A_by_verifA.jsonl
"""
import argparse, json, math, os, random, re, sys, time
from collections import Counter

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from peft import PeftModel

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src", "cascade_methods"))
from choicewhy_common import ARM_NAME, VERIF_SYS, verifier_body, extract, parse_opts  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--arm", default="A", choices=["A", "B2"], help="which CANDIDATE pool to score")
ap.add_argument("--adapter", required=True, help="LoRA verifier to score with")
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--suffix", default="_sc8")
ap.add_argument("--out", required=True)
ap.add_argument("--benches", nargs="+",
                default=["SLAKE", "VQA-RAD", "PMC-VQA", "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--candidate_mode", default="verbatim", choices=["verbatim", "letter_prefix"],
                help="'letter_prefix' TRUNCATES each candidate to its leading '<letter><delimiter>' and "
                     "drops the justification. Used as the ABLATION control: it keeps the (choice)(why) "
                     "POOL (same sampled letters, same distribution) but removes the TEXT, so the "
                     "difference against the verbatim scoring is attributable to the justification alone.")
A = ap.parse_args()
AN = ARM_NAME[A.arm]
DEV = "cuda"
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28      # cap_div 1 == the verifier's training resolution
OUT = os.path.join(ROOT, A.out)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# ------------------------------------------------------------------ item set (same loader as Phase 1)
from datasets import load_dataset  # noqa: E402
dset = load_dataset("/data/dan/dataset/MedVLThinker-Eval")
split = "test" if "test" in dset else list(dset.keys())[0]
data = dset[split]

MANIFEST = os.path.join(ROOT, A.ckpt_dir, "items.jsonl")
ITEMS = {int(r["idx"]): r for r in (json.loads(l) for l in open(MANIFEST) if l.strip())}
print(f"[items] manifest {len(ITEMS)}", flush=True)

# ------------------------------------------------------------------ candidate pools
POOL = []       # (bench, idx, gold, raw_outputs)
for b in A.benches:
    p = os.path.join(ROOT, A.ckpt_dir, f"ckpt_{b}_{AN}{A.suffix}.jsonl")
    assert os.path.exists(p), f"missing candidate dump {p}"
    for l in open(p):
        if not l.strip():
            continue
        r = json.loads(l)
        assert r["arm"] == AN and r["n_samples"] == 8, r["idx"]
        POOL.append((b, int(r["idx"]), r["gold"], r["raw_outputs"]))
print(f"[pool] {len(POOL)} items x 8 candidates = {8*len(POOL)} candidate slots", flush=True)

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        if l.strip():
            done.add(int(json.loads(l)["idx"]))
    print(f"[resume] {len(done)} items already scored", flush=True)

# ------------------------------------------------------------------ model
proc = AutoProcessor.from_pretrained(A.model_path)
YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
print(f"loading base + adapter {A.adapter} ...", flush=True)
model = AutoModelForImageTextToText.from_pretrained(
    A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
model.eval()


def p_yes(images, question, options, candidate):
    msgs = [{"role": "system", "content": VERIF_SYS},
            {"role": "user", "content": [{"type": "image", "image": im,
                                          "max_pixels": HIGH_PX, "min_pixels": MIN_PX} for im in images]
                                        + [{"type": "text",
                                            "text": verifier_body(question, options, candidate)}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    igs, vids = process_vision_info(msgs)
    enc = proc(text=[text], images=igs, videos=vids, return_tensors="pt", padding=True).to(DEV)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1]
        py, pn = math.exp(lg[YES].item()), math.exp(lg[NO].item())
    return py / (py + pn) if (py + pn) > 0 else 0.5


LEADCUT = re.compile(r"^(\s*[*\"'(\[]*\s*[A-J]\s*[).:,;\-—\]]?)")


def to_candidate(raw):
    """What the verifier is shown. 'verbatim' = the model's own candidate. 'letter_prefix' = only the
    leading '<letter><delimiter>' -- arm A's surface form -- with the justification removed."""
    if A.candidate_mode == "verbatim":
        return raw
    mm = LEADCUT.match(raw.strip())
    return mm.group(1).strip() if mm else extract(raw, AN)[0]


fh = open(OUT, "a")
t0 = time.time()
nfwd = 0
for k, (bench, idx, goldl, raws) in enumerate(POOL):
    if idx in done:
        continue
    try:
        ex = data[idx]
        opts = parse_opts(ex["options"])
        imgs = [im.convert("RGB") for im in (ex.get("images") or [])]
        q = ex["question"]
        assert str(ex["answer_label"]).strip().upper()[:1] == goldl, f"gold mismatch at {idx}"

        cands = [to_candidate(c) for c in raws]
        uniq = {}
        for c in cands:
            if c not in uniq:
                uniq[c] = p_yes(imgs, q, opts, c)
                nfwd += 1
        scores = [uniq[c] for c in cands]
        lets = [extract(c, AN) for c in raws]
        labels = [int(l[0] == goldl) for l in lets]
        fh.write(json.dumps({
            "idx": idx, "bench": bench, "arm": AN, "gold": goldl,
            "adapter": A.adapter, "candidate_mode": A.candidate_mode,
            "raw_outputs": raws,
            "candidates_shown": (None if A.candidate_mode == "verbatim" else cands),
            "letters": [l[0] for l in lets],
            "parse_ok": [int(l[1]) for l in lets],
            "labels": labels,
            "scores": [round(float(s), 6) for s in scores],
            "n_unique_strings": len(uniq),
        }) + "\n")
        fh.flush()
    except Exception as e:
        print(f"  SKIP idx={idx}: {str(e)[:160]}", flush=True)
    if (k + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {k+1}/{len(POOL)} items | {nfwd} forwards | {el/60:.1f} min "
              f"| {nfwd/max(el,1e-9):.2f} fwd/s", flush=True)
fh.close()
print(f"DONE {A.out}: {nfwd} verifier forwards in {(time.time()-t0)/60:.1f} min", flush=True)
