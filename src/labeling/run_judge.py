#!/usr/bin/env python3
"""
run_judge.py - LLM-judge scorer for open-ended VQA (robustness check vs exact-match). A strong NEUTRAL
text LLM (default MedVLThinker-32B, Qwen2.5-32B backbone -- NOT the model scored in the Lingshu cascade)
judges, per sample, whether the model's free-text answer is correct given the reference answer. Lenient
on phrasing/synonyms, strict on clinical meaning. Text-only (no image): equivalence of short factual
answers given the question. Reads prediction jsonls (idx, question, gold, modal_pred) and writes
{idx, judge_ok} per file. Judges MANY files in one model load.
  HF_HOME=/data/dan/hf_cache python3 src/labeling/run_judge.py --preds f1.jsonl f2.jsonl --tp 2
  -> writes alongside each input as <name>.judge.jsonl
"""
import argparse, json, os, math
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
JUDGE_SYS = ("You are a strict medical exam grader. Given a question, a reference (gold) answer, and a "
             "model answer, decide if the model answer is CORRECT: it must match the clinical meaning of "
             "the reference. Be lenient about phrasing, synonyms, and abbreviations (e.g. 'CT' = 'computed "
             "tomography'), but mark wrong answers, missing key findings, or different conclusions as No. "
             "Respond with only 'Yes' or 'No'.")
ap = argparse.ArgumentParser()
ap.add_argument("--judge_model", default="/data/dan/weights/MedVLThinker-32B-RL_m23k")
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--max_model_len", type=int, default=2048)
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.judge_model)
def yn_ids():
    ids = {}
    for w in ["Yes","yes","YES","No","no","NO"]:
        for v in (w, " "+w):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1: ids[e[0]] = w
    return {k:("Yes" if v.lower()=="yes" else "No") for k,v in ids.items()}
YN = yn_ids()
def prompt(q, gold, pred):
    body = f"Question: {q}\nReference answer: {gold}\nModel answer: {pred}\nIs the model answer correct? Answer Yes or No."
    msgs = [{"role":"system","content":JUDGE_SYS},{"role":"user","content":body}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
llm = LLM(model=A.judge_model, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)
for pf in A.preds:
    out = pf.replace(".jsonl", ".judge.jsonl")
    rows = [json.loads(l) for l in open(pf) if l.strip()]
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip(): done.add(json.loads(l)["idx"])
    todo = [r for r in rows if r["idx"] not in done]
    print(f"{os.path.basename(pf)}: {len(todo)}/{len(rows)} to judge -> {os.path.basename(out)}", flush=True)
    with open(out, "a") as fh:
        for c0 in range(0, len(todo), 256):
            ch = todo[c0:c0+256]
            outs = llm.generate([prompt(r["question"], r["gold"], r["modal_pred"]) for r in ch], sp)
            for r, o in zip(ch, outs):
                lps = (o.outputs[0].logprobs or [{}])[0]
                py = max((math.exp(v.logprob) for t,v in lps.items() if YN.get(t)=="Yes"), default=0.0)
                pn = max((math.exp(v.logprob) for t,v in lps.items() if YN.get(t)=="No"), default=0.0)
                ok = int(py >= pn) if (py+pn) > 0 else int(o.outputs[0].text.strip().lower().startswith("yes"))
                fh.write(json.dumps({"idx": r["idx"], "judge_ok": ok}) + "\n")
            fh.flush(); print(f"   [{min(c0+256,len(todo))}/{len(todo)}]", flush=True)

    # 2026-09-13. `out` is opened in APPEND mode so a resumed run keeps what it already judged --
    # but that also means anything writing here twice leaves duplicate idx behind silently. It
    # happened: ckpt_pathvqa_open_medgemma4b_sc8_scexploded.judge.jsonl came out of a single
    # REPORTED run holding 26,822 rows for 22,499 inputs, with duplicates spaced exactly one
    # 256-prompt chunk apart. The loop above provably emits at most len(todo) rows, so a second
    # writer is implied; I did not pin which, and this pass makes the state recoverable regardless.
    #
    # Of those 4,323 duplicated rows, 18 -- 0.42% -- carried a DIFFERENT judge_ok for the same
    # (question, answer) pair. The judge runs at temperature 0.0, so that is not sampling: it is
    # vLLM batch composition moving the numerics enough to flip an argmax on a near-tie. Take 0.42%
    # as this judge's label-noise floor on re-judging, and never assume a re-judge reproduces.
    raw = [json.loads(l) for l in open(out) if l.strip()]
    seen, order = {}, []
    for d in raw:
        if d["idx"] in seen:
            continue                      # FIRST occurrence wins: deterministic and order-stable
        seen[d["idx"]] = d; order.append(d["idx"])
    if len(order) != len(raw):
        print(f"  [dedup] {os.path.basename(out)}: {len(raw)} rows -> {len(order)} unique "
              f"({len(raw)-len(order)} duplicates dropped)", flush=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as g:
            for i in order:
                g.write(json.dumps(seen[i]) + "\n")
        os.replace(tmp, out)
print("DONE judge", flush=True)
