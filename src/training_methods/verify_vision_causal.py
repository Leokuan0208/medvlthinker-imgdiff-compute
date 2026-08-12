#!/usr/bin/env python3
"""verify_vision_causal.py -- NULL TEST for the vision-token cache.

The whole per-IMAGE caching design rests on one claim: in Qwen2.5-VL the LM is causal and the
chat template puts the image BEFORE the question and the answer, so a vision token's hidden state
cannot depend on the question or on the candidate answer.

A first pass of --verify_causal inside the extractor reported max|dh| = 0.000e+00 on 7 of 8 images
and 1.275e+01 on one, which is exactly what a bf16 FlashAttention kernel does when the total
sequence length changes the tiling and one of Qwen's huge outlier activation dimensions is hit.
This script separates the two hypotheses instead of assuming:

  A. SAME prompt, run twice        -> the pure kernel non-determinism floor
  B. image + SHORT trailing text   -> vs
  C. image + LONG trailing text    -> the "does downstream text leak backwards" contrast
  D. image + a DIFFERENT candidate ANSWER appended (the quantity that actually matters)

If B/C/D deviate no more than A does, the states are text-independent up to kernel noise, and the
per-image cache is valid.  Reported for the RAW hidden states and, decisively, for the two pooled
quantities the round actually consumes (v_mean and the 6x6 v_grid): max abs, max relative, and
cosine similarity.
"""
import argparse, json, os, sys, math
import numpy as np
import torch

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))
import extract_generator_hidden as EG  # noqa: E402


def stats(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    d = np.abs(a - b)
    den = np.maximum(np.abs(a), np.abs(b))
    rel = np.where(den > 1e-6, d / np.maximum(den, 1e-6), 0.0)
    ca = a.reshape(-1); cb = b.reshape(-1)
    cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
    return {"max_abs": float(d.max()), "max_rel": float(rel.max()),
            "mean_abs": float(d.mean()), "cosine": cos,
            "max_abs_val": float(np.abs(a).max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--n_images", type=int, default=24)
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 14, 21, 28])
    ap.add_argument("--pool", type=int, default=6)
    ap.add_argument("--out", default="results/cascade_methods/artifacts/_visverif_parts/causal_null.json")
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

    def feats(img, trailing, answer=None):
        content = [{"type": "image", "image": img, "max_pixels": EG.HIGH_PX, "min_pixels": EG.MIN_PX}]
        if trailing:
            content.append({"type": "text", "text": trailing})
        msgs = [{"role": "system", "content": EG.SYS_GEN}, {"role": "user", "content": content}]
        igs, _ = EG.process_vision_info(msgs)
        ip = proc.image_processor(images=igs, return_tensors="pt")
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if answer:
            text = text + answer
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
        raw, vm, vg = {}, {}, {}
        for L in A.layers:
            hv = out.hidden_states[L][0, pos].float()
            raw[L] = hv.cpu().numpy()
            vm[L] = hv.mean(0).cpu().numpy()
            g = hv.reshape(Gh, Gw, -1).permute(2, 0, 1).unsqueeze(0)
            vg[L] = torch.nn.functional.adaptive_avg_pool2d(g, (P, P)) \
                .squeeze(0).permute(1, 2, 0).reshape(P * P, -1).cpu().numpy()
        return raw, vm, vg, int(len(pos)), int(ids.shape[0])

    CONTRASTS = {
        "A_same_prompt_twice": dict(trailing="", answer=None),
        "B_short_question": dict(trailing="Is the lesion on the left?", answer=None),
        "C_long_question": dict(trailing="Describe in full detail the anatomical structures visible "
                                         "in this medical image, including laterality, and explain "
                                         "which side of the organ border is obscured and why.",
                                answer=None),
        "D_candidate_answer_appended": dict(trailing="Which side is affected?", answer="Right."),
        "E_different_candidate_answer": dict(trailing="Which side is affected?", answer="Left."),
    }
    agg = {k: {"raw": [], "v_mean": [], "v_grid": []} for k in CONTRASTS}
    n_tok = []
    for i, r in enumerate(uniq):
        base_raw, base_vm, base_vg, nv, nt0 = feats(r["img"], "")
        for name, kw in CONTRASTS.items():
            raw, vm, vg, nv2, nt = feats(r["img"], **kw)
            assert nv == nv2, f"vision token count changed {nv} -> {nv2}"
            n_tok.append([nt0, nt])
            for key, a, b in (("raw", base_raw, raw), ("v_mean", base_vm, vm), ("v_grid", base_vg, vg)):
                worst = max((stats(a[L], b[L]) for L in A.layers), key=lambda s: s["max_abs"])
                cosmin = min(stats(a[L], b[L])["cosine"] for L in A.layers)
                worst["cosine_min_over_layers"] = cosmin
                agg[name][key].append(worst)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(uniq)}", flush=True)

    def red(lst):
        return {"max_abs": max(s["max_abs"] for s in lst),
                "max_rel": max(s["max_rel"] for s in lst),
                "mean_abs": float(np.mean([s["mean_abs"] for s in lst])),
                "min_cosine": min(s["cosine_min_over_layers"] for s in lst),
                "max_abs_activation_seen": max(s["max_abs_val"] for s in lst)}

    rep = {"what": "NULL TEST: are vision-token hidden states independent of the text that FOLLOWS "
                   "the image (question and candidate answer)?",
           "date": "2026-08-12", "n_images": len(uniq), "layers": A.layers, "pool": P,
           "model": A.model_path, "dtype": "bfloat16", "attn": "flash_attention_2",
           "reference_arm": "A_same_prompt_twice IS the kernel non-determinism floor -- every other "
                            "contrast must be judged against it, not against zero",
           "contrasts": {k: {kk: red(v[kk]) for kk in ("raw", "v_mean", "v_grid")}
                         for k, v in agg.items()}}
    fl = rep["contrasts"]["A_same_prompt_twice"]["v_mean"]["max_abs"]
    worst = max(rep["contrasts"][k]["v_mean"]["max_abs"] for k in CONTRASTS if k != "A_same_prompt_twice")
    rep["verdict"] = {
        "kernel_floor_v_mean_max_abs": fl,
        "worst_text_contrast_v_mean_max_abs": worst,
        "min_cosine_over_all_text_contrasts_v_mean":
            min(rep["contrasts"][k]["v_mean"]["min_cosine"] for k in CONTRASTS),
        "pass": bool(min(rep["contrasts"][k]["v_mean"]["min_cosine"] for k in CONTRASTS) > 0.999999),
        "reading": "PASS means the pooled vision features are text-independent to within bf16 kernel "
                   "noise, so caching them per IMAGE is exact, AND -- the load-bearing consequence -- "
                   "a vision feature is a per-question CONSTANT that cannot by itself change a "
                   "within-pool argmax."}
    os.makedirs(os.path.dirname(os.path.join(ROOT, A.out)), exist_ok=True)
    json.dump(rep, open(os.path.join(ROOT, A.out), "w"), indent=1)
    print(json.dumps(rep["verdict"], indent=1), flush=True)
    for k in CONTRASTS:
        print(k, json.dumps(rep["contrasts"][k]["v_mean"]), flush=True)


if __name__ == "__main__":
    main()
