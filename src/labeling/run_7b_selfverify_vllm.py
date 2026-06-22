#!/usr/bin/env python3
"""
run_7b_selfverify_vllm.py - AutoMix/P(True)-style SELF-VERIFICATION signal for the cheap 7B leg.
The 7B already produced a first-pass answer (nothink@cap320). Here it does ONE extra cheap pass:
given the question + its own proposed answer, it judges "Is the proposed answer correct? Yes/No"
and we read the normalized P(Yes). This is a NEW cascade signal (a second reasoning pass), not
derivable from the first-pass logprobs, and the basis for a verification-aware gate.

Modes:
  --source eval   : MedVLThinker-Eval, first-pass preds from ckpts/gate_7b_prune/cap320
  --source calib  : pmc_vqa_train sample, first-pass preds from ckpts/gate_7b_pmctrain_prune/cap320
Output: <ckpt_dir>/ckpt_<ds>_verify.jsonl (eval) or ckpt_verify.jsonl (calib) with
  {idx, gold, pred(first-pass), p_yes, p_no, p_yes_norm, ok}   (ok = first-pass correctness, for ref)
Runs 7B@cap320 (max_pixels=HIGH_PX//4), 1 token, logprobs. Launch from repo root.
"""
import argparse, re, json, random, time, os
from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

MODEL = "/data/dan/weights/MedVLThinker-7B-RL_m23k"
ROOT  = "/data/dan/dataset/MedVLThinker-Eval"
SAMPLE = "/data/dan/dataset/pmc_vqa_train/train_sample_3000.jsonl"
SYS_VERIFY = ("You are a careful medical exam grader. Given a question, the options, and a proposed "
              "answer, decide whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")
HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
CHUNK = 256

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default=MODEL)
ap.add_argument("--source", choices=["eval", "calib"], required=True)
ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320")
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--datasets", nargs="+",
                default=["PMC-VQA", "SLAKE", "VQA-RAD", "PathVQA", "MMMU",
                         "MedXpert-Reasoning", "MedXpert-Understanding"])
ap.add_argument("--pred_dir_eval", default="ckpts/gate_7b_prune/cap320")
ap.add_argument("--pred_dir_calib", default="ckpts/gate_7b_pmctrain_prune/cap320")
ap.add_argument("--ckpt_dir", required=True)
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--max_images", type=int, default=8)
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)
MAXPX = HIGH_PX // CAP_DIV[A.cap]

proc = AutoProcessor.from_pretrained(A.model_path)
def tok_ids(words):
    ids = {}
    for w in words:
        for v in (w, " " + w):
            e = proc.tokenizer.encode(v, add_special_tokens=False)
            if len(e) == 1:
                ids[e[0]] = w
    return ids
YES_IDS = tok_ids(["Yes", "yes", "YES"]); NO_IDS = tok_ids(["No", "no", "NO"])

def parse_opts(s):
    if isinstance(s, dict): return s
    try: return json.loads(s)
    except Exception: return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))

def load_preds(d):
    m = {}
    import glob
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        ds = None
        b = os.path.basename(f)
        mm = re.match(r"ckpt_(.+?)_", b)
        ds = mm.group(1) if mm else "calib"
        for l in open(f):
            if l.strip():
                r = json.loads(l); m[(ds, r["idx"])] = r.get("pred")
    return m

# ---- assemble work items: (ds, idx, question, options, images, gold, proposed) ----
def verify_prompt(q_text, opts, proposed, images):
    proposed_txt = f"{proposed}) {opts.get(proposed, '')}" if proposed and proposed != "?" else str(proposed)
    body = (q_text + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
            + f"\n\nProposed answer: {proposed_txt}\nIs the proposed answer correct? Answer Yes or No.")
    im = [{"type": "image", "image": x, "max_pixels": MAXPX, "min_pixels": MIN_PX} for x in (images or [])]
    msgs = [{"role": "system", "content": SYS_VERIFY},
            {"role": "user", "content": im + [{"type": "text", "text": body}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if imgs: req["multi_modal_data"] = {"image": imgs}
    return req

items = []  # (ds, idx, req, gold, proposed)
if A.source == "eval":
    ds_all = load_dataset(ROOT); split = "test" if "test" in ds_all else list(ds_all.keys())[0]
    data = ds_all[split]
    preds = load_preds(A.pred_dir_eval)
    def subset(*keys):
        return [i for i, n in enumerate(data["dataset_name"])
                if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]
    def mx_by_type(t):
        out = []
        for i, n in enumerate(data["dataset_name"]):
            if "medxpert" not in n.lower(): continue
            mc = data[i].get("misc")
            try: qt = json.loads(mc).get("question_type", "") if mc else ""
            except Exception: qt = ""
            if qt.lower() == t: out.append(i)
        return out
    DSI = {"MedXpert-Reasoning": lambda: mx_by_type("reasoning"),
           "MedXpert-Understanding": lambda: mx_by_type("understanding"),
           "PMC-VQA": lambda: subset("pmcvqa", "pmc"), "SLAKE": lambda: subset("slake"),
           "VQA-RAD": lambda: subset("vqarad", "vqa_rad", "rad"),
           "PathVQA": lambda: subset("pathvqa", "path"), "MMMU": lambda: subset("mmmu")}
    def fixed_slice(idxs):
        rng = random.Random(42); s = idxs[:]; rng.shuffle(s); s = s[:A.n]; return s
    for name in A.datasets:
        if name not in DSI: continue
        for i in fixed_slice(DSI[name]()):
            ex = data[i]; opts = parse_opts(ex["options"]); proposed = preds.get((name, i))
            if proposed is None: continue
            items.append((name, i, verify_prompt(ex["question"], opts, proposed, ex.get("images")),
                          str(ex["answer_label"]).strip().upper()[:1], proposed))
else:
    rows = [json.loads(l) for l in open(SAMPLE) if l.strip()]
    preds = load_preds(A.pred_dir_calib)
    for r in rows:
        proposed = preds.get(("calib", r["idx"]))
        if proposed is None: continue
        from PIL import Image
        items.append(("calib", r["idx"], verify_prompt(r["question"], r["options"], proposed, [r["image_path"]]),
                      r["answer_label"], proposed))

print(f"source={A.source} cap={A.cap}: {len(items)} verification items; YES={list(YES_IDS.values())} NO={list(NO_IDS.values())}", flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": A.max_images}, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

import math
def yn(lps):
    if not lps: return None, None
    lp0 = lps[0]
    py = max((math.exp(o.logprob) for t, o in lp0.items() if t in YES_IDS), default=0.0)
    pn = max((math.exp(o.logprob) for t, o in lp0.items() if t in NO_IDS), default=0.0)
    return py, pn

# group by ds for separate ckpt files
from collections import defaultdict
by_ds = defaultdict(list)
for it in items: by_ds[it[0]].append(it)

t0 = time.time(); tot = 0
for name, group in by_ds.items():
    tag = "verify" if A.source == "eval" else "verify"
    ckpt = os.path.join(A.ckpt_dir, (f"ckpt_{name}_verify.jsonl" if A.source == "eval" else "ckpt_verify.jsonl"))
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    todo = [it for it in group if it[1] not in done]
    print(f"\n--- {name}: {len(group)} total, {len(todo)} to run -> {ckpt} ---", flush=True)
    with open(ckpt, "a") as fh:
        for c0 in range(0, len(todo), CHUNK):
            ch = todo[c0:c0 + CHUNK]; reqs = [it[2] for it in ch]
            try: outs = llm.generate(reqs, sp)
            except Exception as e:
                print(f"   chunk failed ({e}); one-by-one", flush=True); outs = []
                for r in reqs:
                    try: outs.append(llm.generate([r], sp)[0])
                    except Exception as e2: print(f"     skip: {e2}", flush=True); outs.append(None)
            for it, o in zip(ch, outs):
                if o is None: continue
                lps = o.outputs[0].logprobs or []
                py, pn = yn(lps)
                norm = (py / (py + pn)) if (py is not None and (py + pn) > 0) else None
                fh.write(json.dumps({"idx": it[1], "gold": it[3], "pred": it[4],
                    "p_yes": round(py, 5) if py is not None else None,
                    "p_no": round(pn, 5) if pn is not None else None,
                    "p_yes_norm": round(norm, 5) if norm is not None else None,
                    "ok": int(it[3] == it[4]), "raw": o.outputs[0].text}) + "\n")
                tot += 1
            fh.flush()
            print(f"   [{min(c0+CHUNK,len(todo))}/{len(todo)}] {tot/(time.time()-t0):.1f}/s", flush=True)
    print(f">> {name} done", flush=True)
print(f"\nDONE self-verify {tot} items in {(time.time()-t0)/60:.1f} min", flush=True)
