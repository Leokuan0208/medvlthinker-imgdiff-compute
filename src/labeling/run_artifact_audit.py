#!/usr/bin/env python3
"""
run_artifact_audit.py - audit an open-ended med-VQA set for ANSWERABLE vs ARTIFACT questions. A neutral
text LLM (MedVLThinker-32B) sees only (question, reference answer) and decides whether the item is a
well-formed answerable question or a decontextualized caption-extraction / under-specified / fragment-gold
ARTIFACT. Text-only by design: an artifact is identifiable from the Q+gold alone (presumes a figure
caption, refers to unspecified 'the process/figure', or gold is a sentence fragment not a direct answer).
Reads a pred jsonl (idx, question, gold/modal_pred not needed) -> writes {idx, label} (ANSWERABLE|ARTIFACT).
  HF_HOME=/data/dan/hf_cache python3 src/labeling/run_artifact_audit.py --preds f1.jsonl f2.jsonl --tp 2
"""
import argparse, json, os, math
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
SYS = ("You are auditing a medical visual-question-answering benchmark for question quality. You see only "
       "a question and its reference answer (not the image). Decide:\n"
       "ANSWERABLE: a well-formed, self-contained question with a clear specific answer a competent "
       "physician could give by looking at the image (e.g. 'What modality is this?' -> 'CT'; 'What is the "
       "main organ?' -> 'Lung').\n"
       "ARTIFACT: a decontextualized or under-specified item — it presumes a specific figure/caption, refers "
       "to unspecified 'the process/the figure/this' with no antecedent, is grammatically broken, OR the "
       "reference answer is a sentence fragment / continuation rather than a direct answer (e.g. Q 'what does "
       "process begin as?' ref 'a focus of microabscess in a vascular loop in the marrow'; Q 'where does the "
       "bone show?' ref 'at the margins').\n"
       "Reply with only one word: ANSWERABLE or ARTIFACT.")
ap = argparse.ArgumentParser()
ap.add_argument("--judge_model", default="/data/dan/weights/MedVLThinker-32B-RL_m23k")
ap.add_argument("--preds", nargs="+", required=True)
ap.add_argument("--tp", type=int, default=2); ap.add_argument("--gpu_mem", type=float, default=0.90)
ap.add_argument("--max_model_len", type=int, default=2048)
A = ap.parse_args()
tok = AutoTokenizer.from_pretrained(A.judge_model)
def first_ids(words):
    ids = {}
    for w in words:
        for v in (w, " "+w):
            e = tok.encode(v, add_special_tokens=False)
            if e: ids.setdefault(e[0], w)
    return ids
ANS = first_ids(["ANSWERABLE","Answerable","answerable"]); ART = first_ids(["ARTIFACT","Artifact","artifact"])
def prompt(q, gold):
    body = f"Question: {q}\nReference answer: {gold}\nClassify: ANSWERABLE or ARTIFACT."
    return tok.apply_chat_template([{"role":"system","content":SYS},{"role":"user","content":body}],
                                   tokenize=False, add_generation_prompt=True)
llm = LLM(model=A.judge_model, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, trust_remote_code=True)
sp = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)
for pf in A.preds:
    out = pf.replace(".jsonl", ".audit.jsonl")
    rows = [json.loads(l) for l in open(pf) if l.strip()]
    # dedup by idx (use first occurrence's q/gold)
    seen_idx = {}
    for r in rows:
        if r["idx"] not in seen_idx: seen_idx[r["idx"]] = r
    rows = list(seen_idx.values())
    done = set()
    if os.path.exists(out):
        for l in open(out):
            if l.strip(): done.add(json.loads(l)["idx"])
    todo = [r for r in rows if r["idx"] not in done]
    print(f"{os.path.basename(pf)}: {len(todo)}/{len(rows)} to audit", flush=True)
    with open(out, "a") as fh:
        for c0 in range(0, len(todo), 256):
            ch = todo[c0:c0+256]
            outs = llm.generate([prompt(r["question"], r.get("gold")) for r in ch], sp)
            for r, o in zip(ch, outs):
                lps = (o.outputs[0].logprobs or [{}])[0]
                pa = max((math.exp(v.logprob) for t,v in lps.items() if t in ANS), default=0.0)
                pr = max((math.exp(v.logprob) for t,v in lps.items() if t in ART), default=0.0)
                lab = "ARTIFACT" if pr > pa else "ANSWERABLE"
                fh.write(json.dumps({"idx": r["idx"], "label": lab, "p_artifact": pr/(pa+pr) if (pa+pr)>0 else 0.5}) + "\n")
            fh.flush(); print(f"   [{min(c0+256,len(todo))}/{len(todo)}]", flush=True)
print("DONE audit", flush=True)
