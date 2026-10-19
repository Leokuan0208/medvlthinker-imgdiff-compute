#!/usr/bin/env python3
"""prefix_tokens.py — count the tokens before the image and between question and answer in the feature pass's
chat template (Lingshu-7B tokenizer, SYS_GEN prompt as in extract_generator_hidden.py)."""
from transformers import AutoProcessor
P = "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
SYS = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
proc = AutoProcessor.from_pretrained(P)
msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What organ is the gray part on the left of the image?"}]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(repr(text))
tok = proc.tokenizer
pre = text.split("<|vision_start|>")[0]
print("tokens before <|vision_start|>:", len(tok(pre)["input_ids"]))
print("system text alone:", len(tok(SYS)["input_ids"]))
post = text.split("<|vision_end|>")[1]
q = "What organ is the gray part on the left of the image?"
print("after image (question + template):", len(tok(post)["input_ids"]), "question alone:", len(tok(q)["input_ids"]))
print("template between question and answer:", len(tok(post.split(q)[1])["input_ids"]), repr(post.split(q)[1]))
print("answer 'Large Bowel.':", len(tok("Large Bowel.")["input_ids"]))
