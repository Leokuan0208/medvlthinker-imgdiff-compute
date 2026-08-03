#!/usr/bin/env python3
"""
run_choicewhy_pilot.py -- PHASE 1 GATE for the "(choice)(why)" program.

QUESTION
--------
On MULTIPLE CHOICE, the trained outcome verifier has essentially no signal because the answer is a
single letter: best-of-N degenerates.  HYPOTHESIS: if the model answers as (choice)(why) -- the option
letter FOLLOWED BY a short justification -- the verifier gains text to grade.  Before training any
verifier we must know whether the (choice)(why) FORMAT ITSELF costs multiple-choice accuracy.

THREE ARMS, differing ONLY in the system-message format instruction (the user turn -- images, question,
option block -- is byte-identical across arms, and byte-identical to the repo's existing Lingshu MCQ
dumps in ckpts/gate_lingshu7b_mcq/, so arm A is a NULL TEST against a known cell):

  A  letter_only    the repo's deployed baseline instruction (verbatim)
  B  answer_first   letter first, then a one-or-two-sentence justification
  C  reason_first   justification first, then the letter  (= conventional CoT; the control that
                    tests whether the answer-first ORDERING is what does the work)

Everything else is held fixed: same model (greedy, temperature 0), same max_tokens (so no arm is
truncated where another is not), same image resolution, same chunking, same item set.

ITEM SET (identical across arms; the repo's fixed_slice(seed=42) selection, so every idx is a subset
of the already-dumped ckpts/gate_lingshu7b_mcq/ idx sets):
  SLAKE-closed     416  (full)
  VQA-RAD-closed   272  (full)
  PMC-VQA          500  (seed-42 subsample of 2000; the SAME 500 as the existing dump)
  MedXpert-MM      300  (150 Reasoning + 150 Understanding; prefixes of the existing 500/500 slices)

USAGE (from repo root, one GPU, tp=1):
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 src/labeling/run_choicewhy_pilot.py \
      --arms A B C --ckpt_dir ckpts/choicewhy_pilot
  # item manifest only (no model load), incl. md5 of DECODED RGB pixels for later disjoint-split work:
  python3 src/labeling/run_choicewhy_pilot.py --manifest_only --ckpt_dir ckpts/choicewhy_pilot
"""
import argparse, hashlib, json, os, random, re, time

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
ap.add_argument("--arms", nargs="+", default=["A", "B", "C"])
ap.add_argument("--ckpt_dir", default="ckpts/choicewhy_pilot")
ap.add_argument("--cap", default="fullres", choices=["fullres", "cap640", "cap320", "cap160", "cap80"])
ap.add_argument("--max_tokens", type=int, default=320, help="IDENTICAL for every arm (no truncation confound)")
ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--max_model_len", type=int, default=8192)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--manifest_only", action="store_true")
ap.add_argument("--print_prompts", action="store_true")
ap.add_argument("--n_samples", type=int, default=1, help=">1 -> best-of-N probe (temperature>0)")
ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--benches", nargs="+", default=None, help="restrict to these benchmarks")
ap.add_argument("--suffix", default="", help="appended to the ckpt filename")
A = ap.parse_args()
os.makedirs(A.ckpt_dir, exist_ok=True)

ROOT = "/data/dan/dataset/MedVLThinker-Eval"
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
MAXPX = HIGH_PX // CAP_DIV[A.cap]

# ----------------------------------------------------------------------------- THE THREE INSTRUCTIONS
# These are the ONLY thing that differs between arms.  A is verbatim the repo's deployed
# SYS_NOTHINK (src/labeling/run_32b_modes_vllm.py:25, src/labeling/run_vlm_eval.py:22).
SYS = {
    "A": "Answer with only the correct option letter (e.g. 'A'). Do not explain.",
    "B": ("Answer with the correct option letter first (e.g. 'A'), then explain in one or two short "
          "sentences why that option is correct. Example: \"A. The mass is in the left lower lobe.\""),
    "C": ("Explain in one or two short sentences why an option is correct, then answer with the "
          "correct option letter last (e.g. 'A'). Example: \"The mass is in the left lower lobe. A.\""),
    # --- second pass (2026-08-03), added after the token audit showed arm C never justified anything
    # and arm B omitted a justification on 47% of items.  B2/C2 FORCE the justification and are exact
    # mirrors of each other, so B2-vs-C2 is the ordering control that arm C failed to provide.
    "B2": ("Answer with the correct option letter first (e.g. 'A'), then, in exactly one sentence, state "
           "the specific finding in the image that makes that option correct. Always give the sentence, "
           "even when the answer is obvious. Example: \"A. The mass is in the left lower lobe.\""),
    "C2": ("First, in exactly one sentence, state the specific finding in the image that makes an option "
           "correct, then answer with that option letter last (e.g. 'A'). Always give the sentence, "
           "even when the answer is obvious. Example: \"The mass is in the left lower lobe. A.\""),
}
ARM_NAME = {"A": "A_letter_only", "B": "B_answer_first", "C": "C_reason_first",
            "B2": "B2_answer_first_forced", "C2": "C2_reason_first_forced"}

# ----------------------------------------------------------------------------- item selection
from datasets import load_dataset  # noqa: E402
ds = load_dataset(ROOT); split = "test" if "test" in ds else list(ds.keys())[0]
data = ds[split]
names = data["dataset_name"]


def subset(*keys):
    return [i for i, n in enumerate(names)
            if any(k in n.lower().replace("-", "").replace("_", "") for k in keys)]


def mx_by_type(t):
    out = []
    for i, n in enumerate(names):
        if "medxpert" not in n.lower():
            continue
        mc = data[i].get("misc")
        try:
            qt = json.loads(mc).get("question_type", "") if mc else ""
        except Exception:
            qt = ""
        if qt.lower() == t:
            out.append(i)
    return out


def fixed_slice(idxs, n):
    """The repo's canonical selection: Random(42).shuffle then take the first n."""
    rng = random.Random(42); s = idxs[:]; rng.shuffle(s); return s[:n]


SLICES = [
    ("SLAKE",                   lambda: fixed_slice(subset("slake"), 4000)),
    ("VQA-RAD",                 lambda: fixed_slice(subset("vqarad", "vqa_rad", "rad"), 4000)),
    ("PMC-VQA",                 lambda: fixed_slice(subset("pmcvqa", "pmc"), 500)),
    ("MedXpert-Reasoning",      lambda: fixed_slice(mx_by_type("reasoning"), 150)),
    ("MedXpert-Understanding",  lambda: fixed_slice(mx_by_type("understanding"), 150)),
]


def parse_opts(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))


def gold(ex):
    return str(ex["answer_label"]).strip().upper()[:1]


# ----------------------------------------------------------------------------- manifest
MANIFEST = os.path.join(A.ckpt_dir, "items.jsonl")
if not os.path.exists(MANIFEST):
    with open(MANIFEST, "w") as fh:
        for name, fn in SLICES:
            for i in fn():
                ex = data[i]
                imgs = ex.get("images") or []
                md5 = [hashlib.md5(im.convert("RGB").tobytes()).hexdigest() for im in imgs]
                fh.write(json.dumps({"idx": i, "bench": name, "gold": gold(ex),
                                     "n_options": len(parse_opts(ex["options"])),
                                     "n_images": len(imgs), "image_md5_rgb": md5}) + "\n")
    print(f"wrote manifest {MANIFEST}", flush=True)
ITEMS = [json.loads(l) for l in open(MANIFEST) if l.strip()]
print(f"pilot item set: {len(ITEMS)} items", flush=True)
for name, _ in SLICES:
    print(f"   {name}: {sum(1 for r in ITEMS if r['bench'] == name)}")
if A.manifest_only:
    raise SystemExit(0)

# ----------------------------------------------------------------------------- model
from transformers import AutoProcessor          # noqa: E402
from qwen_vl_utils import process_vision_info   # noqa: E402
from vllm import LLM, SamplingParams            # noqa: E402

proc = AutoProcessor.from_pretrained(A.model_path)


def build_prompt(ex, arm):
    """Byte-identical to src/labeling/run_32b_modes_vllm.py::build_prompt except for SYS[arm]."""
    opts = parse_opts(ex["options"]); assert opts
    q = ex["question"] + "\n" + "\n".join(f"{k}) {v}" for k, v in opts.items())
    imgs_meta = [{"type": "image", "image": im, "max_pixels": MAXPX, "min_pixels": MIN_PX}
                 for im in (ex.get("images") or [])]
    msgs = [{"role": "system", "content": SYS[arm]},
            {"role": "user", "content": imgs_meta + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(msgs)
    req = {"prompt": text}
    if image_inputs:
        req["multi_modal_data"] = {"image": image_inputs}
    return req, text


if A.print_prompts:
    ex = data[ITEMS[0]["idx"]]
    for arm in ["A", "B", "C"]:
        _, t = build_prompt(ex, arm)
        print("=" * 90); print(f"ARM {arm} ({ARM_NAME[arm]})"); print(t)
    raise SystemExit(0)

print(f"loading {A.model_path} (tp={A.tp}, cap={A.cap}, max_tokens={A.max_tokens}, greedy)", flush=True)
llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16",
          gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
          limit_mm_per_prompt={"image": 8}, trust_remote_code=True)
sp = SamplingParams(temperature=A.temp, max_tokens=A.max_tokens, logprobs=20,
                    n=A.n_samples, seed=(A.seed if A.n_samples > 1 else None))

LETTER_SET = set(chr(ord('A') + k) for k in range(10))


def _lid(L):
    for v in (L, " " + L):
        e = proc.tokenizer.encode(v, add_special_tokens=False)
        if e:
            return e[0]
    return None


LID = {L: _lid(L) for L in LETTER_SET}
ID2LET = {v: k for k, v in LID.items() if v is not None}


def opt_logprob_at_first_letter(token_ids, logprobs_list):
    """Repo-standard: the option-letter logprob dict at the first letter-looking decode step."""
    for step in range(len(token_ids)):
        d = proc.tokenizer.decode([token_ids[step]]).strip()
        if d and d[0] in LETTER_SET and (len(d) == 1 or d[1] in ").: "):
            lp = logprobs_list[step] if step < len(logprobs_list) else None
            if not lp:
                return {}
            return {ID2LET[t]: round(float(o.logprob), 4) for t, o in lp.items() if t in ID2LET}
    return {}


t0 = time.time(); tot = 0
for arm in A.arms:
    for name, fn in SLICES:
        if A.benches and name not in A.benches: continue
        sel = [r["idx"] for r in ITEMS if r["bench"] == name]
        ck = os.path.join(A.ckpt_dir, f"ckpt_{name}_{ARM_NAME[arm]}{A.suffix}.jsonl")
        done = set()
        if os.path.exists(ck):
            for l in open(ck):
                if l.strip():
                    try:
                        done.add(json.loads(l)["idx"])
                    except Exception:
                        pass
        todo = [i for i in sel if i not in done]
        print(f"\n--- arm {arm} / {name}: {len(sel)} total, {len(todo)} to run -> {ck} ---", flush=True)
        with open(ck, "a") as fh:
            for c0 in range(0, len(todo), A.chunk):
                ch = todo[c0:c0 + A.chunk]
                reqs = [build_prompt(data[i], arm)[0] for i in ch]
                try:
                    outs = llm.generate(reqs, sp)
                except Exception as e:
                    print(f"   chunk failed ({e}); one-by-one", flush=True)
                    outs = []
                    for r in reqs:
                        try:
                            outs.append(llm.generate([r], sp)[0])
                        except Exception as e2:
                            print(f"     skip: {e2}", flush=True); outs.append(None)
                for i, o in zip(ch, outs):
                    if o is None:
                        continue
                    if A.n_samples > 1:
                        fh.write(json.dumps({
                            "idx": i, "bench": name, "arm": ARM_NAME[arm], "gold": gold(data[i]),
                            "n_samples": A.n_samples, "temp": A.temp,
                            "gen_tokens_all": [len(c.token_ids) for c in o.outputs],
                            "raw_outputs": [c.text for c in o.outputs]}) + "\n")
                        continue
                    gen = o.outputs[0].text
                    tk = list(o.outputs[0].token_ids)
                    lps = o.outputs[0].logprobs or []
                    fh.write(json.dumps({
                        "idx": i, "bench": name, "arm": ARM_NAME[arm], "gold": gold(data[i]),
                        "gen_tokens": len(tk), "finish": o.outputs[0].finish_reason,
                        "opt_logprobs": opt_logprob_at_first_letter(tk, lps),
                        "raw_output": gen}) + "\n")
                    tot += 1
                fh.flush()
                print(f"   [{min(c0+A.chunk,len(todo))}/{len(todo)}] {tot/(time.time()-t0):.1f}/s", flush=True)
        print(f">> arm {arm} / {name} done", flush=True)
print(f"\nDONE: {tot} generations in {(time.time()-t0)/60:.1f} min", flush=True)
