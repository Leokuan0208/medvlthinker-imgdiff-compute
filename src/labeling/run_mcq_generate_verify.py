#!/usr/bin/env python3
"""
run_mcq_generate_verify.py  --  [GPU EXPERIMENT — QUEUE AFTER OmniMed; do NOT run while GPUs are busy]

Purpose: unblock the UNIFIED GENERATIVE VERIFIER (UGV) on the MCQ half.  Today MCQ is answered by a single
letter-logprob pass (margin gate); open-text is answered by best-of-8 free-text + a trained verifier.  To make
ONE format-agnostic mechanism, we must run MCQ *as generation* and score each generation with the SAME trained
verifier that open-text uses.  This script produces exactly that per-sample data so the offline unified analysis
(src/cascade_methods/unified_router.py / open_bestofN_adaptive.py) can be re-run treating MCQ and open identically.

For each MCQ item it:
  (1) samples N free-text answers from the CHEAP model (temp>0), in two prompting modes:
        --mode content : options HIDDEN -> the model must generate the answer text (fully open, most unified)
        --mode letter  : options SHOWN  -> the model answers with the option (best-of-N over letters)
  (2) scores each sampled answer for correctness (map to nearest option; correct == gold option), and
  (3) scores each sampled answer with the TRAINED VERIFIER  P(correct | image, question, answer)  (same head
      + same "Yes/No" grader prompt as run_openvqa_verify.py / run_lora_verifier_open.py).
Output JSONL per sample (schema matches the open transfer_dump so downstream code is unchanged):
  idx, question, gold, preds[N], oks[N], verifier_scores[N], pick(argmax score), pick_ok, greedy_ok,
  self_consistency, n_distinct, seqlogprob, margin, conf, gen_tokens_all, lat_s

Reuses the exact model-load / prompt / max_pixels conventions of run_openvqa.py and the verifier Yes/No
logit read-out of run_openvqa_verify.py.  vLLM.  Launch from repo root, ONE model on ONE GPU.

Example (cheap Lingshu-7B, sample+verify PMC_VQA, content mode, N=8, verifier LoRA = pooled4):
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_mcq_generate_verify.py \
    --model_path lingshu-medical-mllm/Lingshu-7B --verifier_lora ckpts/train/lora_verifier_pooled4 \
    --dataset PMC_VQA --tag lingshu7b --mode content --n_samples 8 --temp 0.7 \
    --ckpt_dir ckpts/mcq_gen_verify/lingshu7b --tp 1

Datasets: PMC_VQA, MedXpertQA-MM, SLAKE, VQA_RAD, PATH_VQA (+MMMU-Medical when available).
NOTE on data paths: MCQ eval items (image + options + gold) are loaded from the MedEvalKit eval set.  Confirm
EVAL_ROOT below points at your copy (/data/dan/dataset/MedVLThinker-Eval); the per-benchmark reader mirrors the
fields used in MedEvalKit/eval_results_*/{}/<BENCH>/results.json (prompt/question, choices, answer).
"""
import argparse, re, json, os, glob, math, string, io
import numpy as np
from collections import Counter
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
from PIL import Image

EVAL_ROOT = "/data/dan/dataset/MedVLThinker-Eval"     # <-- confirm this path
SYS_GEN = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
SYS_MCQ = "You are an expert medical image analyst. Answer the multiple-choice question."
SYS_VERIFY = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
              "proposed answer is correct. Respond with only 'Yes' or 'No'.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", required=True)
ap.add_argument("--verifier_lora", required=True, help="LoRA adapter dir for the trained verifier (e.g. ckpts/train/lora_verifier_pooled4)")
ap.add_argument("--dataset", required=True, choices=["PMC_VQA", "MedXpertQA-MM", "SLAKE", "VQA_RAD", "PATH_VQA", "MMMU-Medical-val"])
ap.add_argument("--tag", required=True)
ap.add_argument("--mode", choices=["content", "letter"], default="content")
ap.add_argument("--n_samples", type=int, default=8); ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320")
ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88); ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_tokens", type=int, default=64); ap.add_argument("--n", type=int, default=100000)
ap.add_argument("--chunk", type=int, default=16, help="items per llm.generate call; keep small (memory cgroup)")
A = ap.parse_args(); os.makedirs(A.ckpt_dir, exist_ok=True); MAXPX = HIGH_PX // CAP_DIV[A.cap]

# --- MedEvalKit source paths (mirror MedEvalKit/utils/{PMC_VQA,MedXpertQA} loaders) ---
PMC_ROOT = "/data/dan/dataset/medevalkit/PMC-VQA"          # test_2.csv + figures/ (fallback images/)
MEDXPERT_ROOT = "/data/dan/dataset/medevalkit/MedXpertQA"  # HF-cache arrow (MM split) + images/

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation)); return re.sub(r"\s+", " ", s).strip()

# ------------------------------------------------------------------ dataset loaders
# Each yields (idx, question_text, options(list[str] or None), gold_text, gold_letter, PIL image).
def _gold_from_letter(ch, gl):
    """Return the (letter-prefix-stripped) option text for gold letter gl, else gl."""
    gl = str(gl).strip()
    if ch and gl[:1] in "ABCDE":
        gi = "ABCDE".index(gl[:1])
        if gi < len(ch):
            return re.sub(r"^\s*[A-E][:.\)]\s*", "", str(ch[gi])).strip()
    return gl

def load_items():
    """Load eval items, one dataset at a time, capped at A.n (cap BEFORE decoding images -> bounded memory).
    Each item = (idx, question, options(list|None), gold_text, gold_letter|None, img_ref).
    img_ref is a PIL image, a local path str, or a list of path strs (multi-image) -- all handled downstream.
    PMC_VQA / MedXpertQA-MM mirror MedEvalKit/utils/{PMC_VQA,MedXpertQA}: same csv/arrow + question/options/answer/image fields."""
    ds = A.dataset
    items = []
    if ds == "SLAKE":
        d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
        for x in d:
            if len(items) >= A.n: break
            if x.get("q_lang") != "en": continue
            ip = os.path.join(root, x["img_name"])
            if not os.path.exists(ip): continue
            items.append((x["qid"], x["question"], None, str(x["answer"]), None, ip))
    elif ds in ("VQA_RAD", "PATH_VQA"):
        import pandas as pd
        base = "/data/dan/dataset/vqa_rad/data" if ds == "VQA_RAD" else "/data/dan/dataset/path_vqa/data"
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
        for i, r in df.iterrows():
            if len(items) >= A.n: break            # cap early -> don't decode the whole split
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            img = r["image"]
            if not (isinstance(img, dict) and "bytes" in img): continue
            pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
            items.append((int(i), str(q), None, str(a).strip(), None, pil))
    elif ds == "PMC_VQA":
        # MedEvalKit PMC_VQA: test_2.csv columns = index,Figure_path,Caption,Question,ChoiceA..D,Answer(letter),split
        import csv
        csv_path = os.path.join(PMC_ROOT, "test_2.csv")
        reader = csv.reader(open(csv_path, encoding="utf-8")); next(reader)  # header
        for row in reader:
            if len(items) >= A.n: break
            if len(row) < 10: continue
            index, figpath, _cap, question, cA, cB, cC, cD, answer, _split = row[:10]
            ch = [cA, cB, cC, cD]
            ip = os.path.join(PMC_ROOT, "figures", figpath)
            if not os.path.exists(ip): ip = os.path.join(PMC_ROOT, "images", figpath)
            if not os.path.exists(ip): continue
            gl = str(answer).strip()
            items.append((index, str(question).strip(), ch, _gold_from_letter(ch, gl), gl, ip))
    elif ds == "MedXpertQA-MM":
        # MedEvalKit MedXpertQA (MM split): HF-cache arrow; fields question, options{A..E}, label(letter), images[list]
        from datasets import Dataset
        cand = sorted(glob.glob(os.path.join(MEDXPERT_ROOT, "**", "*med_xpert_qa-test.arrow"), recursive=True))
        assert cand, f"MedXpertQA test arrow not found under {MEDXPERT_ROOT} (expected .../MM/.../med_xpert_qa-test.arrow)"
        dset = Dataset.from_file(cand[0]); imgdir = os.path.join(MEDXPERT_ROOT, "images")
        for i, r in enumerate(dset):
            if len(items) >= A.n: break
            opts = r["options"]; ch = [str(opts[k]) for k in sorted(opts)]  # A,B,C,D,E order
            gl = str(r["label"]).strip()
            paths = [os.path.join(imgdir, m) for m in (r.get("images") or [])][:4]
            paths = [p for p in paths if os.path.exists(p)]
            if not paths: continue
            imref = paths if len(paths) > 1 else paths[0]
            items.append((r.get("id", i), str(r["question"]).strip(), ch, _gold_from_letter(ch, gl), gl, imref))
    else:
        raise ValueError(f"no loader for dataset {ds}")
    return items

def map_correct(pred, options, gold_text, gold_letter):
    """Correct iff the free-text pred maps to the gold option (content mode) or states the gold letter (letter mode)."""
    if options and gold_letter:
        # nearest option by normalized containment/overlap
        pn = norm(pred); best, bestj = -1, -1
        for j, o in enumerate(options):
            on = norm(re.sub(r"^\s*[A-E][:.\)]\s*", "", str(o)))
            sc = len(set(pn.split()) & set(on.split())) + (2 if on and on in pn else 0) + (3 if pn == on else 0)
            # also credit an explicit letter answer
            if re.match(rf"^\s*\(?{'ABCDE'[j]}\b", pred.strip(), re.I): sc += 3
            if sc > best: best, bestj = sc, j
        return int("ABCDE"[bestj] == gold_letter[:1])
    gn, pn = norm(gold_text), norm(pred)          # gn can be "" (gold that is all stopwords/punct) -> not creditable
    return int(bool(gn) and (pn == gn or gn in pn))

# ------------------------------------------------------------------ build model, prompts
items = load_items()
print(f"{A.dataset}: {len(items)} items | mode={A.mode} N={A.n_samples} temp={A.temp}", flush=True)
proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

def opt_block(options):
    return "\n".join(f" {'ABCDE'[j]}: {re.sub(r'^\\s*[A-E][:.)]\\s*','',str(o))}" for j, o in enumerate(options))

def _img_blocks(img):
    refs = img if isinstance(img, list) else [img]
    return [{"type": "image", "image": r, "max_pixels": MAXPX, "min_pixels": MIN_PX} for r in refs]

def build_gen(q, options, img):
    if A.mode == "letter" and options:
        body = f"Question: {q}\nOptions:\n{opt_block(options)}\nAnswer with the option's letter."
        sys = SYS_MCQ
    else:
        body = f"Question: {q}"; sys = SYS_GEN
    msgs = [{"role": "system", "content": sys},
            {"role": "user", "content": _img_blocks(img) + [{"type": "text", "text": body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

def build_verify(q, answer, img):
    body = f"Question: {q}\nProposed answer: {answer}\nIs the proposed answer correct? Answer Yes or No."
    msgs = [{"role": "system", "content": SYS_VERIFY},
            {"role": "user", "content": _img_blocks(img) + [{"type": "text", "text": body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

def tok_ids(words):
    ids = {}
    for w in words:
        for v in (w, " " + w):
            e = proc.tokenizer.encode(v, add_special_tokens=False)
            if len(e) == 1: ids[e[0]] = w
    return ids
YES, NO = tok_ids(["Yes", "yes", "YES"]), tok_ids(["No", "no", "NO"])

# Load base model once; the verifier is the SAME base + the trained LoRA adapter (loaded via vLLM LoRA).
from vllm.lora.request import LoRARequest
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
          enable_lora=True, max_lora_rank=32)
VLORA = LoRARequest("verifier", 1, os.path.expanduser(os.path.join("~/medvlthinker-imgdiff-compute", A.verifier_lora)))
sp_gen = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, n=A.n_samples, logprobs=5)
sp_ver = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}_{A.mode}_sc{A.n_samples}.jsonl")
done = set()
if os.path.exists(ckpt):
    for l in open(ckpt):
        if l.strip(): done.add(json.loads(l)["idx"])
todo = [it for it in items if it[0] not in done]
print(f"  {len(todo)} to run -> {ckpt}", flush=True)
import time; t0 = time.time(); CH = A.chunk
with open(ckpt, "a") as fh:
    for c0 in range(0, len(todo), CH):
        ch = todo[c0:c0+CH]
        _t = time.time()
        gouts = llm.generate([build_gen(q, opt, im) for (_, q, opt, _, _, im) in ch], sp_gen)
        glat = (time.time() - _t) / max(1, len(ch))
        # flatten all (item, sample) verify requests, score with the LoRA verifier
        vreqs = []; span = []
        allpreds = []
        for (idx, q, opt, gt, gl, im), o in zip(ch, gouts):
            preds = [c.text.strip() for c in o.outputs]
            allpreds.append(preds)
            for p in preds: vreqs.append(build_verify(q, p, im))
            span.append(len(preds))
        vouts = llm.generate(vreqs, sp_ver, lora_request=VLORA)
        vi = 0
        for (idx, q, opt, gt, gl, im), o, preds in zip(ch, gouts, allpreds):
            scores = []
            for _ in preds:
                lps = (vouts[vi].outputs[0].logprobs or [{}])[0]; vi += 1
                py = max((math.exp(v.logprob) for t, v in lps.items() if t in YES), default=0.0)
                pn = max((math.exp(v.logprob) for t, v in lps.items() if t in NO), default=0.0)
                scores.append(round(py / (py + pn), 5) if (py + pn) > 0 else 0.0)
            oks = [map_correct(p, opt, gt, gl) for p in preds]
            pick = int(np.argmax(scores)); cnt = Counter(norm(p) for p in preds)
            modal_norm = cnt.most_common(1)[0][0]
            slp = None
            if o.outputs[0].logprobs:
                slp = float(np.mean([next(iter(lp.values())).logprob for lp in o.outputs[0].logprobs if lp]))
            fh.write(json.dumps({"idx": idx, "question": q, "gold": gt, "preds": preds, "oks": oks,
                "sl": oks, "scores": scores, "pick": pick, "pick_ok": oks[pick],
                "greedy_ok": oks[0], "modal_pred": next(p for p in preds if norm(p) == modal_norm),
                "self_consistency": round(cnt.most_common(1)[0][1] / len(preds), 4), "n_distinct": len(cnt),
                "seqlogprob": (round(slp, 4) if slp is not None else None),
                "gen_tokens_all": [len(c.token_ids) for c in o.outputs], "lat_s": round(glat, 4)}) + "\n")
        fh.flush()
        acc = np.mean([json.loads(l)["pick_ok"] for l in open(ckpt) if l.strip()])
        print(f"   [{min(c0+CH,len(todo))}/{len(todo)}] verifier-pick-acc={acc:.3f}", flush=True)
print(f"DONE {A.tag} {A.dataset} {A.mode}: {len(items)} in {(time.time()-t0)/60:.1f} min", flush=True)
