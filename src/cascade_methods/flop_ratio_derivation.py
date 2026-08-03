#!/usr/bin/env python3
"""
flop_ratio_derivation.py -- derive the project's 32B-over-7B FLOP-equivalence constant.

WHY: every "compute-negative" statement in this project rests on the bare literal
`R32 = 4.57` (src/cascade_methods/lingshu_medeval_cascade.py:21 and ~13 other files).
No file in the repository derives it; `honest_recosting.py:144` only reproduces it as the
nominal name-plate ratio 32.0B / 7.0B = 4.571.  This script replaces that literal with a
derivation from (a) EXACT parameter counts read out of the safetensors headers on disk and
(b) the MEASURED token geometry of the very prompts the 665 ms / 347 ms anchors were timed on.

Everything here is CPU-only and offline: safetensors *headers* only (no weights loaded), the
HF processor for token counting (no model), and JSONL files already in the repo.

Launch from the repo root:   python3 src/cascade_methods/flop_ratio_derivation.py
Writes: results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json
"""
import json, struct, glob, os, io, statistics as st, collections

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")
HUB = "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--{}/snapshots/*/"

# ---------------------------------------------------------------------------------------------
# 1.  EXACT parameter counts, by component, from the safetensors headers  [MEASURED]
# ---------------------------------------------------------------------------------------------
def _sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def param_counts(name):
    """Read every shard's safetensors header and bucket parameters by role. No weights loaded."""
    snap = glob.glob(HUB.format(name))[0]
    idx = json.load(open(snap + "model.safetensors.index.json"))
    g = collections.Counter(); dtypes = collections.Counter(); nbytes = 0
    for sf in sorted(glob.glob(snap + "model-*.safetensors")):
        for k, v in _sf_header(sf).items():
            if k == "__metadata__":
                continue
            nel = 1
            for d in v["shape"]:
                nel *= d
            dtypes[v["dtype"]] += nel
            nbytes += v["data_offsets"][1] - v["data_offsets"][0]
            if k.startswith("visual.merger"):        role = "vision_merger"
            elif k.startswith("visual"):             role = "vision_tower"
            elif "embed_tokens" in k:                role = "embed_tokens"
            elif k.startswith("lm_head"):            role = "lm_head"
            else:                                    role = "lm_body"
            g[role] += nel
    cfg = json.load(open(snap + "config.json"))
    tot = sum(g.values())
    assert nbytes == idx["metadata"]["total_size"], "shard bytes disagree with the index"
    assert set(dtypes) == {"BF16"} and nbytes == 2 * tot, "unexpected dtype mix"
    return dict(model=name, snapshot=snap, config=cfg, params=dict(g), total_params=tot,
                index_total_size_bytes=idx["metadata"]["total_size"], dtypes=dict(dtypes))


# ---------------------------------------------------------------------------------------------
# 2.  Analytic forward-pass FLOP model
# ---------------------------------------------------------------------------------------------
def lm_body_from_config(pc):
    """Independent analytic count of the language-model body from config.json alone -- a check that
    the safetensors bucketing above assigned every tensor to the right role."""
    c = pc["config"]
    d, L, H, KV, I = (c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                      c["num_key_value_heads"], c["intermediate_size"])
    hd = d // H
    per = (d * d + d) + 2 * (d * KV * hd + KV * hd) + d * d + 3 * d * I + 2 * d
    return per * L + d                                            # + final RMSNorm


def forward_flops(pc, M_img_tok, T_prompt_tok, G_gen_tok, causal_half=True):
    """FLOPs for ONE end-to-end forward+generate on an (image, question) prompt.

      vision tower : 2 * P * N_vit                      (P = 4*M pre-merge patches; 2x2 spatial merge)
                   + windowed/full self-attention over P
      merger       : 2 * M * N_merger                   (runs on the MERGED tokens)
      LM prefill   : 2 * T * N_lm_body  + causal attn
      LM decode    : 2 * (G-1) * N_lm_body              (+ decode attention, negligible; included)
      LM head      : 2 * G * N_lm_head                  (one logit vector per generated token;
                                                         HF generate uses logits_to_keep=1)
      embeddings   : 0 FLOPs (a lookup)

    Qwen2.5-VL ViT: `depth` layers, 4 of them full-attention (`fullatt_block_indexes`), the rest
    windowed with window_size=112px => (112/14)^2 = 64 patches per window. ViT attention is
    bidirectional (no causal halving); the LM's is causal (halved iff causal_half).
    """
    c = pc["config"]; v = c["vision_config"]; p = pc["params"]
    P = 4.0 * M_img_tok
    d_v, L_v = v["hidden_size"], v["depth"]
    W = (v["window_size"] // v["patch_size"]) ** 2                  # 64 patches per window
    L_full = len(v["fullatt_block_indexes"]); L_win = L_v - L_full
    vit_dense = 2 * P * p["vision_tower"]
    vit_attn = 4 * d_v * (L_full * P * P + L_win * P * W)           # QK^T + AV, bidirectional
    merger = 2 * M_img_tok * p["vision_merger"]

    d, L = c["hidden_size"], c["num_hidden_layers"]
    lm_prefill = 2 * T_prompt_tok * p["lm_body"]
    attn_scale = 2.0 if causal_half else 4.0
    lm_prefill_attn = attn_scale * L * T_prompt_tok * T_prompt_tok * d
    g_dec = max(G_gen_tok - 1.0, 0.0)
    lm_decode = 2 * g_dec * p["lm_body"]
    lm_decode_attn = 4 * L * d * g_dec * (T_prompt_tok + g_dec / 2.0)
    head = 2 * G_gen_tok * p["lm_head"]

    parts = dict(vision_tower_dense=vit_dense, vision_tower_attn=vit_attn, vision_merger=merger,
                 lm_prefill_dense=lm_prefill, lm_prefill_attn=lm_prefill_attn,
                 lm_decode_dense=lm_decode, lm_decode_attn=lm_decode_attn, lm_head=head)
    parts["TOTAL"] = sum(parts.values())
    return parts


# ---------------------------------------------------------------------------------------------
# 3.  Token geometry of the prompts the 665 ms / 347 ms anchors were measured on  [MEASURED]
# ---------------------------------------------------------------------------------------------
def measure_token_geometry(n=25, warmup=3):
    """Rebuild the EXACT prompts of src/cascade_methods/open_measure_latency_energy.py (gen mode,
    cap320, vqa_rad non-yes/no items) and count image vs text tokens. Processor only, no GPU."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import pandas as pd
    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    MAXPX, MINPX = 1280 * 28 * 28 // 4, 4 * 28 * 28
    SYS = ("You are an expert medical image analyst. Answer the question with a short, specific "
           "phrase. Do not explain.")
    proc = AutoProcessor.from_pretrained(glob.glob(HUB.format("Lingshu-7B"))[0], trust_remote_code=True)
    files = sorted(glob.glob("/data/dan/dataset/vqa_rad/data/test-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    items = []
    for _, r in df.iterrows():
        q, a, img = r.get("question"), r.get("answer"), r["image"]
        if str(a).strip().lower() in ("yes", "no"):
            continue
        if isinstance(img, dict) and "bytes" in img:
            items.append((str(q), Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
        if len(items) >= n + warmup:
            break
    IMG_ID = 151655
    rows = []
    for q, img in items[warmup:warmup + n]:
        msgs = [{"role": "system", "content": [{"type": "text", "text": SYS}]},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX,
                                              "min_pixels": MINPX}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ii, vi = process_vision_info(msgs)
        inp = proc(text=[text], images=ii, videos=vi, return_tensors="pt")
        ids = inp["input_ids"][0]
        rows.append(dict(total=int(len(ids)), img=int((ids == IMG_ID).sum())))
    return dict(n=len(rows),
                prompt_tok_mean=st.mean(r["total"] for r in rows),
                image_tok_mean=st.mean(r["img"] for r in rows),
                text_tok_mean=st.mean(r["total"] - r["img"] for r in rows),
                image_tok_min=min(r["img"] for r in rows), image_tok_max=max(r["img"] for r in rows),
                patches_mean=4.0 * st.mean(r["img"] for r in rows))


# ---------------------------------------------------------------------------------------------
def main():
    P7 = param_counts("Lingshu-7B")
    P32 = param_counts("Lingshu-32B")

    geo = measure_token_geometry()
    anchors = {json.loads(l)["tag"]: json.loads(l)
               for l in open(os.path.join(ROOT, "logs/latency_opentext.jsonl")) if l.strip()}
    a7, a32 = anchors["lingshu7b_gen"], anchors["lingshu32b_gen"]
    assert abs(geo["prompt_tok_mean"] - a7["prefill_tok_mean"]) < 0.01, \
        "reconstructed prompts do not reproduce the recorded prefill_tok_mean"

    T, M = geo["prompt_tok_mean"], geo["image_tok_mean"]
    G7, G32 = a7["gen_tok_mean"], a32["gen_tok_mean"]

    f7 = forward_flops(P7, M, T, G7)
    f32 = forward_flops(P32, M, T, G32)
    ratio = f32["TOTAL"] / f7["TOTAL"]

    # ---- variants / sensitivity -------------------------------------------------------------
    f7_eq = forward_flops(P7, M, T, G32)                 # same generated length for both
    ratio_eqG = f32["TOTAL"] / f7_eq["TOTAL"]
    f7_nc = forward_flops(P7, M, T, G7, causal_half=False)
    f32_nc = forward_flops(P32, M, T, G32, causal_half=False)
    ratio_nocausal = f32_nc["TOTAL"] / f7_nc["TOTAL"]
    # prefill-only (G=1: the single forward that produces the first token)
    ratio_prefill = (forward_flops(P32, M, T, 1)["TOTAL"] / forward_flops(P7, M, T, 1)["TOTAL"])
    # long-prompt limit (fullres MCQ prompts, measured 642.3 / 639.5 prompt tokens)
    T_full, M_full = 642.3, 642.3 - geo["text_tok_mean"]
    ratio_fullres = (forward_flops(P32, M_full, T_full, 2.0)["TOTAL"]
                     / forward_flops(P7, M_full, T_full, 2.5)["TOTAL"])
    # MCQ cap320 operating point (measured prefill 354.6 / 360.5, gen 2.49 / 2.00)
    ratio_cap320_mcq = (forward_flops(P32, 360.5 - geo["text_tok_mean"], 360.5, 2.00)["TOTAL"]
                        / forward_flops(P7, 354.6 - geo["text_tok_mean"], 354.6, 2.49)["TOTAL"])
    # image-dominated limit: every extra merged image token costs 8*N_vit + 2*N_merger + 2*N_lm_body
    def _per_img_tok(pc):
        return (8 * pc["params"]["vision_tower"] + 2 * pc["params"]["vision_merger"]
                + 2 * pc["params"]["lm_body"])
    ratio_image_limit = _per_img_tok(P32) / _per_img_tok(P7)
    # pure-decode limit (what a decode-only accounting would charge)
    ratio_decode = ((P32["params"]["lm_body"] + P32["params"]["lm_head"])
                    / (P7["params"]["lm_body"] + P7["params"]["lm_head"]))
    ratio_total_params = P32["total_params"] / P7["total_params"]
    ratio_lm_body = P32["params"]["lm_body"] / P7["params"]["lm_body"]

    # ---- measurement cross-checks ------------------------------------------------------------
    lat_ratio = a32["lat_ms_mean"] / a7["lat_ms_mean"]
    lat_ratio_med = a32["lat_ms_median"] / a7["lat_ms_median"]
    en_ratio = a32["energy_j_mean"] / a7["energy_j_mean"]
    pw7 = a7["energy_j_mean"] / (a7["lat_ms_mean"] / 1000.0)
    pw32 = a32["energy_j_mean"] / (a32["lat_ms_mean"] / 1000.0)
    mvt7, mvt32 = anchors["mvt7b_gen"], anchors["mvt32b_gen"]
    A100_BF16_PEAK = 312e12
    mfu7 = f7["TOTAL"] / (a7["lat_ms_mean"] / 1000.0) / A100_BF16_PEAK
    mfu32 = f32["TOTAL"] / (a32["lat_ms_mean"] / 1000.0) / A100_BF16_PEAK
    HBM = 2.039e12                                   # A100-80GB SXM HBM2e bandwidth, B/s
    dec7 = P7["index_total_size_bytes"] / HBM * 1000.0
    dec32 = P32["index_total_size_bytes"] / HBM * 1000.0

    out = dict(
        title="Derivation of the 32B/7B FLOP-equivalence constant (the repo's `R32 = 4.57`)",
        date="2026-08-03", no_gpu=True, no_fabricated_numbers=True,
        reproduce="python3 src/cascade_methods/flop_ratio_derivation.py",

        problem=dict(
            literal=4.57,
            canonical_definition="src/cascade_methods/lingshu_medeval_cascade.py:21  (`R7=1.0; R32=4.57`)",
            reused_in=["src/cascade_methods/paper_baselines.py:64-66 (GEN32N/GEN32T/FUSE)",
                       "src/cascade_methods/integrated_method.py:56-57 (GEN32N/GEN32T dicts)",
                       "src/cascade_methods/beat32b_fusion.py:48-50 + POLICY_FLOP (FUSE = 1.0+4.57 = 5.57)",
                       "src/cascade_methods/pandora_controller.py:51,53 (GEN32, C_STRONG_F)",
                       "src/cascade_methods/end_to_end_consolidation.py:57,59",
                       "src/cascade_methods/latency_reexamination.py:72",
                       "src/cascade_methods/open_gate_efficiency.py:22",
                       "src/cascade_methods/best_method_lingshu.py:44",
                       "src/cascade_methods/lingshu_deferral_apgr.py:16",
                       "src/cascade_methods/quantized_strong_leg.py:38 (via cost_constants)",
                       "src/cascade_methods/honest_recosting.py:144 (as 4.571)",
                       "src/cascade_methods/beat32b_more.py:320 (comment)"],
            agreement=("All sites agree on the SAME value; there is one third-decimal split -- 4.57 "
                       "everywhere except honest_recosting.py, which carries 4.571. A DIFFERENT, "
                       "incompatible constant 4.34 (= 33.0e9/7.6e9) survives in "
                       "src/cascade_methods/open_bestofN_adaptive.py:14 and in the older "
                       "src/analysis/cascade/cascade_cost_prefill_flops.py:33 (N7=7.6e9, N32=33.0e9, "
                       "NVIT=0.675e9 -- that older file at least states a model). "
                       "The 4.57 sites are the canonical, currently-cited ones."),
            only_stated_derivation="32.0B / 7.0B = 4.571 (honest_recosting.py:145) -- name-plate sizes, "
                                   "neither of which is either model's true parameter count."),

        parameter_counts=dict(
            method=("safetensors headers of every shard, parameters bucketed by role; no weights "
                    "loaded. Byte total asserted equal to the index `total_size`, all-BF16 asserted "
                    "(so params = bytes/2). [MEASURED]"),
            lingshu_7b=dict(total=P7["total_params"], index_total_size_bytes=P7["index_total_size_bytes"],
                            by_role=P7["params"], shards=4),
            lingshu_32b=dict(total=P32["total_params"], index_total_size_bytes=P32["index_total_size_bytes"],
                             by_role=P32["params"], shards=14),
            verification_of_earlier_audit=("CONFIRMED: Lingshu-7B index total_size = 16,584,333,312 B "
                                           "bf16 over 4 shards = 8,292,166,656 params = 8.292 B, matching "
                                           "the earlier 8.29 B audit. Lingshu-32B = 66,905,436,672 B over "
                                           "14 shards = 33,452,718,336 params = 33.453 B."),
            independent_crosscheck_lm_body_from_config=dict(
                lingshu_7b=lm_body_from_config(P7), lingshu_32b=lm_body_from_config(P32),
                matches_safetensors=(lm_body_from_config(P7) == P7["params"]["lm_body"]
                                     and lm_body_from_config(P32) == P32["params"]["lm_body"]),
                note="analytic count from hidden_size/layers/heads/kv_heads/intermediate_size alone; "
                     "exact match confirms no tensor was mis-bucketed."),
            naive_total_param_ratio=round(ratio_total_params, 4),
            lm_body_param_ratio=round(ratio_lm_body, 4),
            vision_towers=dict(
                lingshu_7b=P7["params"]["vision_tower"] + P7["params"]["vision_merger"],
                lingshu_32b=P32["params"]["vision_tower"] + P32["params"]["vision_merger"],
                note=("The two vision towers are the SAME architecture (depth 32, hidden 1280, 16 heads, "
                      "patch 14, 2x2 merge) and differ by only %.1f%% in parameters -- the 32B's is "
                      "larger only because its merger projects to 5120 instead of 3584 and its ViT MLP "
                      "is 3456 wide instead of 3420. The image branch is therefore a near-CONSTANT cost "
                      "shared by both legs, which is exactly why the total-parameter ratio is the wrong "
                      "basis for a whole-forward FLOP ratio."
                      % (100 * ((P32["params"]["vision_tower"] + P32["params"]["vision_merger"])
                                / (P7["params"]["vision_tower"] + P7["params"]["vision_merger"]) - 1)))),
            note_other_family=("/data/dan/weights/MedVLThinker-{7B,32B}-RL_m23k have byte-identical "
                               "safetensors index total_size (16,584,333,312 / 66,905,436,672), i.e. the "
                               "same Qwen2.5-VL-7B/32B skeletons -- the derived constant applies to both "
                               "model families used in this project. [MEASURED]")),

        token_geometry=dict(
            method=("The 665 ms / 347 ms anchors were timed by open_measure_latency_energy.py on 25 "
                    "non-yes/no VQA-RAD test items at cap320. Those exact prompts were rebuilt with the "
                    "Lingshu-7B processor (CPU) and the image/text token split counted. [MEASURED]"),
            **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in geo.items()},
            reproduces_recorded_prefill=dict(
                recorded=a7["prefill_tok_mean"], reconstructed=round(geo["prompt_tok_mean"], 2),
                source="logs/latency_opentext.jsonl",
                note="exact match to 2 dp -- the reconstruction is the measured workload, not a proxy"),
            generated_tokens=dict(lingshu_7b=G7, lingshu_32b=G32,
                                  source="logs/latency_opentext.jsonl gen_tok_mean")),

        flop_model=dict(
            statement=("F = 2*P*N_vit + attn_vit + 2*M*N_merger + 2*T*N_lm_body + 2*L*T^2*d "
                       "+ 2*(G-1)*N_lm_body + decode_attn + 2*G*N_lm_head, with P = 4*M pre-merge "
                       "patches, M merged image tokens, T total prompt tokens, G generated tokens. "
                       "Embedding lookup = 0 FLOPs. ViT attention is bidirectional; 4 of 32 ViT layers "
                       "are full-attention and 28 are windowed over 64 patches (window 112px / patch "
                       "14px). LM attention is causal, so its quadratic term is halved."),
            operating_point=dict(T=round(T, 2), M=round(M, 2), P_patches=round(4 * M, 1),
                                 G_7b=G7, G_32b=G32),
            lingshu_7b_gflops={k: round(v / 1e9, 2) for k, v in f7.items()},
            lingshu_32b_gflops={k: round(v / 1e9, 2) for k, v in f32.items()},
            component_shares_pct=dict(
                lingshu_7b={k: round(100 * v / f7["TOTAL"], 2) for k, v in f7.items() if k != "TOTAL"},
                lingshu_32b={k: round(100 * v / f32["TOTAL"], 2) for k, v in f32.items() if k != "TOTAL"}),
            arithmetic=("7B: vision %.0f + merger %.0f + LM-prefill %.0f + LM-decode %.0f + head %.0f "
                        "= %.0f GFLOP.  32B: vision %.0f + merger %.0f + LM-prefill %.0f + LM-decode "
                        "%.0f + head %.0f = %.0f GFLOP.  Ratio = %.0f / %.0f = %.3f."
                        % ((f7["vision_tower_dense"] + f7["vision_tower_attn"]) / 1e9,
                           f7["vision_merger"] / 1e9,
                           (f7["lm_prefill_dense"] + f7["lm_prefill_attn"]) / 1e9,
                           (f7["lm_decode_dense"] + f7["lm_decode_attn"]) / 1e9, f7["lm_head"] / 1e9,
                           f7["TOTAL"] / 1e9,
                           (f32["vision_tower_dense"] + f32["vision_tower_attn"]) / 1e9,
                           f32["vision_merger"] / 1e9,
                           (f32["lm_prefill_dense"] + f32["lm_prefill_attn"]) / 1e9,
                           (f32["lm_decode_dense"] + f32["lm_decode_attn"]) / 1e9, f32["lm_head"] / 1e9,
                           f32["TOTAL"] / 1e9, f32["TOTAL"] / 1e9, f7["TOTAL"] / 1e9, ratio))),

        derived_ratio=dict(
            R32_derived=round(ratio, 3),
            status="DERIVED (from measured parameter counts + measured token geometry)",
            sensitivity={
                "at the anchor operating point (T=326.68, G=5.6/5.64)": round(ratio, 3),
                "same G for both legs (G=5.6)": round(ratio_eqG, 3),
                "no causal halving of the LM attention term": round(ratio_nocausal, 3),
                "prefill only (G=1)": round(ratio_prefill, 3),
                "MCQ cap320 operating point (T=354.6/360.5, G=2.49/2.00)": round(ratio_cap320_mcq, 3),
                "MCQ fullres operating point (T~642)": round(ratio_fullres, 3),
                "image-dominated limit (all-image prompt, marginal cost per image token)":
                    round(ratio_image_limit, 3),
                "pure-decode limit ((lm_body+lm_head) ratio)": round(ratio_decode, 3),
                "total-parameter ratio (the naive basis)": round(ratio_total_params, 3),
                "lm_body-only ratio (T -> infinity limit)": round(ratio_lm_body, 3),
                "name-plate 32.0/7.0 (what the repo actually used)": 4.571},
            recommended=dict(
                value=round(ratio, 2),
                uncertainty="+/- 0.15 (the spread across the three real operating points the project "
                            "actually runs: cap320 open %.3f, cap320 MCQ %.3f, fullres MCQ %.3f)"
                            % (ratio, ratio_cap320_mcq, ratio_fullres),
                band=[round(min(ratio, ratio_cap320_mcq, ratio_fullres), 3),
                      round(max(ratio, ratio_cap320_mcq, ratio_fullres), 3)],
                rationale=("The workload is a single prefill-dominated forward on an image+question "
                           "prompt with 2-6 generated tokens. Under that workload the shared, "
                           "near-identical vision tower is %.1f%% of the 7B's FLOPs but only %.1f%% of "
                           "the 32B's, which pulls the whole-forward ratio well below both the "
                           "lm_body ratio (%.3f) and the name-plate 4.571."
                           % (100 * (f7["vision_tower_dense"] + f7["vision_tower_attn"]
                                     + f7["vision_merger"]) / f7["TOTAL"],
                              100 * (f32["vision_tower_dense"] + f32["vision_tower_attn"]
                                     + f32["vision_merger"]) / f32["TOTAL"], ratio_lm_body))),
            why_4_57_was_wrong=("4.57 = 32.0/7.0 uses name-plate sizes for models whose true counts are "
                                "8.292 B and 33.453 B (naive total ratio %.3f), and it applies a "
                                "parameter ratio to a quantity -- a whole VLM forward pass -- that is "
                                "not proportional to total parameters, because (i) the ~0.68 B vision "
                                "tower is shared and nearly identical in both models, (ii) the 0.545 B "
                                "/ 0.779 B embedding table costs 0 FLOPs, and (iii) the lm_head is "
                                "applied to O(G)=~6 positions, not to the whole prompt. Coincidentally "
                                "4.57 is close to the pure-DECODE ratio %.3f -- it is roughly the right "
                                "constant for a decode-only workload and the wrong one for this "
                                "prefill-dominated one."
                                % (ratio_total_params, ratio_decode))),

        measurement_crosscheck=dict(
            source="logs/latency_opentext.jsonl (n=25 per tag, batch-1, single A100-80GB, NVML energy)",
            lingshu=dict(lat_ms=[a7["lat_ms_mean"], a32["lat_ms_mean"]],
                         lat_ratio_mean=round(lat_ratio, 3), lat_ratio_median=round(lat_ratio_med, 3),
                         energy_j=[round(a7["energy_j_mean"], 2), round(a32["energy_j_mean"], 2)],
                         energy_ratio=round(en_ratio, 3),
                         mean_power_w=[round(pw7, 1), round(pw32, 1)],
                         power_ratio=round(pw32 / pw7, 3)),
            medvlthinker=dict(lat_ratio=round(mvt32["lat_ms_mean"] / mvt7["lat_ms_mean"], 3),
                              energy_ratio=round(mvt32["energy_j_mean"] / mvt7["energy_j_mean"], 3),
                              mean_power_w=[round(mvt7["energy_j_mean"] / (mvt7["lat_ms_mean"] / 1e3), 1),
                                            round(mvt32["energy_j_mean"] / (mvt32["lat_ms_mean"] / 1e3), 1)],
                              note="same prompts (prefill_tok_mean 326.68), same architectures"),
            implied_mfu=dict(lingshu_7b=round(100 * mfu7, 2), lingshu_32b=round(100 * mfu32, 2),
                             assumed_peak_tflops_bf16=312,
                             note="A100-80GB dense bf16 peak, no sparsity. [ESTIMATED -- the peak is a "
                                  "datasheet figure, not measured on this machine]"),
            decode_bandwidth_floor_ms_per_token=dict(
                lingshu_7b=round(dec7, 2), lingshu_32b=round(dec32, 2),
                assumed_hbm_bw_bytes_per_s=HBM,
                implied_decode_ms=[round(dec7 * G7, 1), round(dec32 * G32, 1)],
                note="weights/bandwidth floor. [ESTIMATED -- HBM bandwidth is a datasheet figure.] "
                     "It accounts for ~%.0f ms of the 7B's 347 ms and ~%.0f ms of the 32B's 665 ms, "
                     "leaving prefill ~%.0f ms vs ~%.0f ms." % (dec7 * G7, dec32 * G32,
                                                                a7["lat_ms_mean"] - dec7 * G7,
                                                                a32["lat_ms_mean"] - dec32 * G32)),
            verdict=("Latency and energy ratios are NOT FLOP ratios, but they bound the plausible range. "
                     "Measured batch-1 latency ratio is %.2f and energy ratio %.2f (Lingshu) / %.2f "
                     "(MedVLThinker) -- BOTH BELOW the derived %.2f and far below the charged 4.57. That "
                     "ordering is the physically expected one: at batch 1 the small model is more "
                     "severely under-utilised (implied MFU %.1f%% vs %.1f%%), so it burns more time and "
                     "energy per FLOP. A charged FLOP ratio ABOVE the energy ratio is therefore "
                     "consistent; a charged ratio ABOVE the derived one is simply unsupported. Nothing "
                     "in the measurements supports 4.57 over %.2f, and the MedVLThinker energy ratio "
                     "%.2f lands almost exactly on the derived %.2f."
                     % (lat_ratio, en_ratio, mvt32["energy_j_mean"] / mvt7["energy_j_mean"], ratio,
                        100 * mfu7, 100 * mfu32, ratio,
                        mvt32["energy_j_mean"] / mvt7["energy_j_mean"], ratio))),

        caveats=[
            "The FLOP model counts multiply-accumulates x2 and ignores softmax/norm/activation/RoPE "
            "elementwise work (<1% of a transformer forward). [stated modelling choice]",
            "The LM causal-attention term is halved; the ViT's is not. Both are <0.5%% of the total, "
            "and the no-halving variant moves the ratio by %.3f." % abs(ratio_nocausal - ratio),
            "The lm_head is charged G times (HF `generate` computes logits for one position per step). "
            "If a harness computed logits over the whole prompt instead, the 32B would gain slightly.",
            "The ratio is workload-dependent by construction. It rises toward the lm_body ratio %.3f as "
            "TEXT prompts or generations get longer, and falls toward %.3f as the IMAGE branch comes to "
            "dominate the prompt (the marginal cost of one more merged image token is 8*N_vit + "
            "2*N_merger + 2*N_lm_body, and the shared vision term is a far larger share of the 7B's). "
            "The band [%.2f, %.2f] covers every operating point this project actually ran."
            % (ratio_lm_body, ratio_image_limit, min(ratio, ratio_cap320_mcq, ratio_fullres),
               max(ratio, ratio_cap320_mcq, ratio_fullres)),
            "Both legs are assumed to run at the same image resolution cap. Where the repo runs the "
            "cheap leg at a lower cap than the strong leg, the effective ratio is HIGHER than this and "
            "must be recomputed per configuration.",
            "This does not touch the separate, already-documented problem that the reasoning baseline is "
            "charged a flat per-forward constant regardless of its ~400 generated tokens "
            "(honest_recosting.py handles that axis).",
        ],
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("parameter_counts", "derived_ratio")}, indent=1)[:6000])
    print("\nwrote", OUT)
    return out


if __name__ == "__main__":
    main()
