#!/usr/bin/env python3
"""verify_vision_causal_length.py -- DISCRIMINATE the two explanations of the non-zero deviation
found by verify_vision_causal.py.

WHAT THE FIRST PASS FOUND (artifacts/_visverif_parts/causal_null.json).  Re-running the SAME prompt
twice is bit-exact (max|dh| = 0.0 everywhere).  Appending text AFTER the image is not: raw max|dh|
= 12.75, v_mean 1.075, min cosine 0.998693.  Strictly, that fails "vision states are text
independent", so the per-image cache would not be exact.

BUT the three contrasts B ("Is the lesion on the left?"), D (+"Right.") and E (+"Left.") returned
deviations identical to NINE decimal places, while C (a much longer question) returned a larger one.
Different CONTENT, identical deviation; different LENGTH, different deviation.  That is the exact
signature of FlashAttention-2 tiling: the number of key/value blocks changes with sequence length,
which changes the accumulation order and therefore the bf16 rounding -- while a causal mask still
forbids any information flowing backwards from the trailing text to the vision tokens.

This script separates the two hypotheses instead of inferring them:

  H_length   deviation is a function of the TRAILING TOKEN COUNT alone
  H_content  deviation depends on WHAT the trailing text says (i.e. real backwards leakage)

Design: for each of several trailing token COUNTS, several semantically very different texts are
built and padded/truncated to EXACTLY that count.  H_length predicts every text at a given count is
bit-identical to every other at that count (max|dh| between them = 0.0).  H_content predicts they
differ, and in particular that "left" vs "right" texts differ.

The decisive statistic is therefore NOT the deviation from the no-text baseline -- it is the
deviation BETWEEN two equal-length texts of opposite meaning.  Under a causal mask that must be
exactly 0.

Run:
  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 python3 -u \
    src/training_methods/verify_vision_causal_length.py --n_images 12
"""
import argparse, json, os, sys
import numpy as np
import torch

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import extract_generator_hidden as EG  # noqa: E402

# Semantically maximally different trailing texts. Within a length bucket they are forced to the
# SAME token count, so any difference between them can only be content.
TEXTS = {
    "left":    "Is the abnormality on the left side of this image",
    "right":   "Is the abnormality on the right side of this image",
    "nonsense": "Purple bicycle helium seventeen marmalade orbit trombone quietly",
    "empty_ish": "the the the the the the the the",
}


def stats(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = np.abs(a - b)
    ca, cb = a.reshape(-1), b.reshape(-1)
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cosine": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--n_images", type=int, default=12)
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21, 28])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--pool", type=int, default=6)
    ap.add_argument("--out", default="results/cascade_methods/artifacts/_visverif_parts/causal_null_length.json")
    A = ap.parse_args()
    DEV = "cuda"

    rows = EG.build_eval_rows()
    rows.sort(key=lambda r: (r["ds"], str(r["idx"]), r["na"]))
    seen, uniq = set(), []
    for r in rows:
        m = EG.img_md5(r["img"])
        if m not in seen:
            seen.add(m); uniq.append(r)
        if len(uniq) >= A.n_images:
            break

    proc = EG.AutoProcessor.from_pretrained(A.model_path)
    tok = proc.tokenizer
    IMG = tok.convert_tokens_to_ids("<|image_pad|>")
    model = EG.AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model.eval()
    MERGE = proc.image_processor.merge_size
    P = A.pool

    def fixed_len(text, n):
        """Force `text` to EXACTLY n tokens (truncate, or pad by repeating its own words)."""
        ids = tok(text, add_special_tokens=False)["input_ids"]
        while len(ids) < n:
            ids = ids + ids
        return tok.decode(ids[:n])

    def feats(img, trailing):
        content = [{"type": "image", "image": img, "max_pixels": EG.HIGH_PX, "min_pixels": EG.MIN_PX}]
        if trailing:
            content.append({"type": "text", "text": trailing})
        msgs = [{"role": "system", "content": EG.SYS_GEN}, {"role": "user", "content": content}]
        igs, _ = EG.process_vision_info(msgs)
        ip = proc.image_processor(images=igs, return_tensors="pt")
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        npad = int(ip["image_grid_thw"].prod().item()) // (MERGE ** 2)
        t = text.replace("<|image_pad|>", "<|placeholder|>" * npad).replace("<|placeholder|>", "<|image_pad|>")
        enc = proc.tokenizer([t], return_tensors="pt", padding=True)
        enc["pixel_values"] = ip["pixel_values"]; enc["image_grid_thw"] = ip["image_grid_thw"]
        grid = ip["image_grid_thw"][0].tolist()
        enc = enc.to(DEV)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        ids = enc["input_ids"][0]
        pos = (ids == IMG).nonzero().flatten()
        Gh, Gw = grid[1] // MERGE, grid[2] // MERGE
        raw, vm = {}, {}
        for L in A.layers:
            hv = out.hidden_states[L][0, pos].float()
            raw[L] = hv.cpu().numpy()
            vm[L] = hv.mean(0).cpu().numpy()
        return raw, vm, int(ids.shape[0])

    # within-length (content) and across-length (tiling) aggregates
    same_len, across_len, seqlens = [], [], {}
    names = list(TEXTS)
    for i, r in enumerate(uniq):
        per_len = {}
        for n in A.lengths:
            per_len[n] = {}
            for nm in names:
                raw, vm, nt = feats(r["img"], fixed_len(TEXTS[nm], n))
                per_len[n][nm] = (raw, vm)
                seqlens.setdefault(n, set()).add(nt)
            # H_content test: every PAIR of equal-length, different-meaning texts
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    ra, va = per_len[n][names[a]]
                    rb, vb = per_len[n][names[b]]
                    s = max((stats(ra[L], rb[L]) for L in A.layers), key=lambda x: x["max_abs"])
                    sv = max((stats(va[L], vb[L]) for L in A.layers), key=lambda x: x["max_abs"])
                    same_len.append({"n_tok": n, "pair": f"{names[a]}|{names[b]}",
                                     "raw_max_abs": s["max_abs"], "v_mean_max_abs": sv["max_abs"],
                                     "raw_cos": s["cosine"]})
        # H_length test: SAME text, different lengths
        for a in range(len(A.lengths)):
            for b in range(a + 1, len(A.lengths)):
                na, nb = A.lengths[a], A.lengths[b]
                ra, va = per_len[na]["left"]; rb, vb = per_len[nb]["left"]
                s = max((stats(ra[L], rb[L]) for L in A.layers), key=lambda x: x["max_abs"])
                sv = max((stats(va[L], vb[L]) for L in A.layers), key=lambda x: x["max_abs"])
                across_len.append({"n_tok_a": na, "n_tok_b": nb, "raw_max_abs": s["max_abs"],
                                   "v_mean_max_abs": sv["max_abs"], "raw_cos": s["cosine"]})
        print(f"  {i+1}/{len(uniq)}", flush=True)

    def red(lst, keys=("raw_max_abs", "v_mean_max_abs")):
        return {k: {"max": float(max(x[k] for x in lst)),
                    "mean": float(np.mean([x[k] for x in lst]))} for k in keys}

    content_max = max(x["raw_max_abs"] for x in same_len)
    content_max_vm = max(x["v_mean_max_abs"] for x in same_len)
    length_max = max(x["raw_max_abs"] for x in across_len)
    lr_pairs = [x for x in same_len if set(x["pair"].split("|")) == {"left", "right"}]

    rep = {
        "what": "DISCRIMINATE tiling-noise from backwards text leakage in the vision-token cache.",
        "date": "2026-08-12", "n_images": len(uniq), "layers": A.layers,
        "trailing_token_counts": A.lengths, "texts": TEXTS,
        "model": A.model_path, "dtype": "bfloat16", "attn": "flash_attention_2",
        "design": "for each trailing TOKEN COUNT, four semantically very different texts are forced "
                  "to exactly that count. H_content predicts equal-length texts differ; H_length "
                  "predicts they are bit-identical and only the COUNT matters.",
        "H_content_equal_length_different_meaning": {
            "n_pairs": len(same_len), **red(same_len),
            "DECISIVE_max_abs_raw": content_max, "DECISIVE_max_abs_v_mean": content_max_vm,
            "left_vs_right_only": {"n_pairs": len(lr_pairs),
                                   "max_abs_raw": float(max(x["raw_max_abs"] for x in lr_pairs)),
                                   "max_abs_v_mean": float(max(x["v_mean_max_abs"] for x in lr_pairs))}},
        "H_length_same_text_different_length": {"n_pairs": len(across_len), **red(across_len),
                                                "max_abs_raw": length_max},
        "total_sequence_lengths_seen_per_bucket": {str(k): sorted(v) for k, v in seqlens.items()},
    }
    rep["verdict"] = {
        "equal_length_different_meaning_is_bit_identical": bool(content_max == 0.0),
        "left_vs_right_max_abs": rep["H_content_equal_length_different_meaning"]["left_vs_right_only"]["max_abs_raw"],
        "length_change_max_abs": length_max,
        "conclusion": ("CAUSAL MASK HOLDS: vision-token states are bit-identical across trailing "
                       "texts of opposite meaning at equal token count; the only thing that moves "
                       "them is the total sequence LENGTH, i.e. FlashAttention tiling / bf16 "
                       "rounding. The per-image cache is therefore semantically exact, and a vision "
                       "feature IS a per-question constant."
                       if content_max == 0.0 else
                       "NOT EXPLAINED BY TILING: equal-length texts of different meaning move the "
                       "vision states. The per-image cache is an approximation and the structural "
                       "argument must be weakened accordingly."),
        "consequence_for_this_round": ("a vision feature used ADDITIVELY cannot change a within-pool "
                                       "argmax; only a candidate x image INTERACTION can.")}
    os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, A.out), "w"), indent=1)
    print(json.dumps(rep["verdict"], indent=1), flush=True)


if __name__ == "__main__":
    main()
