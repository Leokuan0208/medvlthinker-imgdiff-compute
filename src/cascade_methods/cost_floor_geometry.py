#!/usr/bin/env python3
"""cost_floor_geometry.py -- verifier PROMPT GEOMETRY for Attack 3's convention C.

Measures, with the real Lingshu-7B tokenizer and the real chat template, how many tokens of a
verifier prompt are SHARED across the 8 candidates of one question (image + system + question +
"Proposed answer: ") and how many are recomputed PER CANDIDATE (the answer text + the trailing
instruction + the assistant turn opener).  Real candidate answers are taken from the CLEAN
disjoint verifier's own transfer dumps.  Image-token count is the measured cap320 mean from
flop_ratio_derivation_2026-08-03.json (M = 280.48), not an assumption of this script.

CPU only.  python3 src/cascade_methods/cost_floor_geometry.py
"""
import json, os, numpy as np
from transformers import AutoTokenizer

REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute")
ART = os.path.join(REPO, "results/cascade_methods/artifacts")
MP = ("/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
      "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")
SYS_VER = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
           "proposed answer is correct. Respond with only 'Yes' or 'No'.")
SPLIT = "\nProposed answer: "
TAIL = "\nIs the proposed answer correct? Answer Yes or No."

tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True)
M = json.load(open(os.path.join(ART, "flop_ratio_derivation_2026-08-03.json")))["token_geometry"]["image_tok_mean"]

QC = []
for ds, fn in (("slake_open", "slake"), ("vqa_rad_open", "vqa_rad"), ("pathvqa_open", "pathvqa")):
    d = json.load(open(os.path.join(REPO, f"ckpts/train/lora_verifier_disjoint/transfer_dump_{fn}_open_lingshu7b.json")))
    for r in d:
        for p in r["preds"]:
            QC.append((ds, str(p)))

# a representative question stub is not available in the dumps; the QUESTION text is part of the SHARED
# prefix, so it cancels out of the marginal cost. We therefore measure the marginal (per-candidate) part
# exactly, and take the shared prefix length from the measured cap320 prompt geometry.
pre_t, suf_t = [], []
for ds, cand in QC:
    msgs = [{"role": "system", "content": SYS_VER},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Q" + SPLIT + cand + TAIL}]}]
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    cut = full.index(SPLIT) + len(SPLIT)
    pre_t.append(len(tok.encode(full[:cut], add_special_tokens=False)))
    suf_t.append(len(tok.encode(full, add_special_tokens=False)) - pre_t[-1])

TG = json.load(open(os.path.join(ART, "flop_ratio_derivation_2026-08-03.json")))["token_geometry"]
# shared prefix = measured gen prompt (image + question + gen system) minus the gen system prompt,
# plus the verifier system prompt, plus "\nProposed answer: "
gen_sys = len(tok.encode("You are an expert medical image analyst. Answer the question with a short, "
                         "specific phrase. Do not explain.", add_special_tokens=False))
ver_sys = len(tok.encode(SYS_VER, add_special_tokens=False))
split_t = len(tok.encode(SPLIT, add_special_tokens=False))
asst = len(tok.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False))
shared = TG["prompt_tok_mean"] - gen_sys + ver_sys + split_t - asst

out = dict(
    title="Verifier prompt geometry for Attack 3 convention C (verifier prefix sharing)",
    provenance="measured (tokenizer) + measured (cap320 prompt geometry from flop_ratio_derivation)",
    n_real_candidates=len(QC),
    image_tok_mean_cap320=M,
    gen_prompt_tok_mean=TG["prompt_tok_mean"],
    gen_system_prompt_tok=gen_sys, verifier_system_prompt_tok=ver_sys,
    proposed_answer_marker_tok=split_t, assistant_opener_tok=asst,
    candidate_answer_tok_mean=round(float(np.mean([len(tok.encode(c, add_special_tokens=False)) for _, c in QC])), 3),
    per_candidate_suffix_tok=round(float(np.mean(suf_t)), 3),
    per_candidate_suffix_tok_sd=round(float(np.std(suf_t)), 3),
    shared_prefix_tok=round(float(shared), 3),
    full_verifier_prompt_tok=round(float(shared + np.mean(suf_t)), 3),
    definition=("shared_prefix = everything up to and including 'Proposed answer: ' (image tokens + verifier "
                "system prompt + question + marker).  per_candidate_suffix = the candidate answer text + "
                "'Is the proposed answer correct? Answer Yes or No.' + the assistant turn opener -- the only "
                "part a prefix-caching verifier has to recompute for candidate k>1."))
json.dump(out, open(os.path.join(ART, "_cost_floor_verifier_geometry.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
