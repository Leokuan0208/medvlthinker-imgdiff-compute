#!/usr/bin/env python3
"""
shrink_strong_leg_footprint.py -- ATTACK 3 part 1/3: EXACT resident-weight footprint and
analytic per-pass FLOP-eq for every candidate strong leg.

CPU-ONLY, OFFLINE.  Reads safetensors *headers* (the 8-byte length prefix + JSON) -- no weights
are ever loaded -- plus config.json.  Nothing here is estimated: every params/bytes figure is a
sum over tensor headers, and every FLOP figure is the analytic model already committed in
src/cascade_methods/flop_ratio_derivation.py, re-used verbatim for the Qwen2.5-VL family and
extended (new code, documented below) to the InternVL family that Lingshu-I-8B uses.

Launch from the repo root:
    python3 src/cascade_methods/shrink_strong_leg_footprint.py
Writes: results/cascade_methods/artifacts/_shrink_parts/footprint.json
"""
import collections
import glob
import json
import os
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "results/cascade_methods/artifacts/_shrink_parts")
os.makedirs(OUTDIR, exist_ok=True)

HUB = "/data/dan/hf_cache/hub/models--{}/snapshots/*/"

MODELS = {
    "lingshu_7b":  "lingshu-medical-mllm--Lingshu-7B",
    "lingshu_i8b": "lingshu-medical-mllm--Lingshu-I-8B",
    "lingshu_32b": "lingshu-medical-mllm--Lingshu-32B",
    "qwen25vl_32b_awq": "Qwen--Qwen2.5-VL-32B-Instruct-AWQ",
}

GIB = 1024.0 ** 3


def sf_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


# ---------------------------------------------------------------- exact footprint (MEASURED)
DTYPE_BITS = {"BF16": 16, "F16": 16, "F32": 32, "I32": 32, "I8": 8, "U8": 8, "F8_E4M3": 8}


def footprint(key):
    """Exact resident-weight footprint from safetensors headers.

    `stored_elements` counts tensor ELEMENTS as stored on disk.  For AWQ checkpoints the packed
    `qweight`/`qzeros` int32 tensors each hold 8 four-bit values, so stored_elements UNDERCOUNTS
    the logical parameter count; `logical_params` corrects for that and is the number to compare
    against a bf16 checkpoint's parameter count.
    """
    snap = glob.glob(HUB.format(MODELS[key]))[0]
    idx = json.load(open(snap + "model.safetensors.index.json"))
    cfg = json.load(open(snap + "config.json"))
    qcfg = cfg.get("quantization_config")

    role_params = collections.Counter()
    dtypes = collections.Counter()
    nbytes = 0
    logical = 0
    for sf in sorted(glob.glob(snap + "model-*.safetensors")):
        for k, v in sf_header(sf).items():
            if k == "__metadata__":
                continue
            nel = 1
            for d in v["shape"]:
                nel *= d
            b = v["data_offsets"][1] - v["data_offsets"][0]
            nbytes += b
            dtypes[v["dtype"]] += nel

            # logical parameter count (unpack 4-bit AWQ containers)
            if qcfg and qcfg.get("quant_method") == "awq" and k.endswith(".qweight"):
                logical += nel * (32 // qcfg["bits"])          # int32 container -> 8 x 4-bit
            elif qcfg and qcfg.get("quant_method") == "awq" and k.endswith(".qzeros"):
                logical += 0                                   # zero-points are not parameters
            elif qcfg and qcfg.get("quant_method") == "awq" and k.endswith(".scales"):
                logical += 0                                   # scales are not parameters
            else:
                logical += nel

            # Role bucketing must cover BOTH families:
            #   Qwen2.5-VL : visual.* / visual.merger.* / lm_head.weight / model.embed_tokens.weight
            #   InternVL   : vision_tower.* / multi_modal_projector.* / language_model.lm_head.weight
            if k.startswith("visual.merger") or ".merger." in k or "multi_modal_projector" in k:
                role = "vision_merger"
            elif k.startswith("visual") or k.startswith("vision_tower") or k.startswith("model.vision_tower"):
                role = "vision_tower"
            elif "embed_tokens" in k:
                role = "embed_tokens"
            elif k.endswith("lm_head.weight") or k.startswith("lm_head"):
                role = "lm_head"
            else:
                role = "lm_body"
            role_params[role] += nel

    assert nbytes == idx["metadata"]["total_size"], (
        "shard bytes disagree with the index for %s" % key)

    return dict(
        model=key,
        hf_repo=MODELS[key],
        snapshot=snap,
        architectures=cfg["architectures"],
        model_type=cfg.get("model_type"),
        quantization_config=qcfg,
        dtypes_by_element_count={k: int(v) for k, v in dtypes.items()},
        stored_elements=int(sum(dtypes.values())),
        logical_params=int(logical),
        weight_bytes=int(nbytes),
        weight_gib=round(nbytes / GIB, 4),
        index_total_size_bytes=idx["metadata"]["total_size"],
        role_stored_elements={k: int(v) for k, v in role_params.items()},
        provenance="safetensors headers + model.safetensors.index.json, read-only, no weights loaded",
    )


# ---------------------------------------------------------------- analytic forward FLOPs
def _qwen_lm_body_from_config(c):
    d, L, H, KV, I = (c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"],
                      c["num_key_value_heads"], c["intermediate_size"])
    hd = d // H
    per = (d * d + d) + 2 * (d * KV * hd + KV * hd) + d * d + 3 * d * I + 2 * d
    return per * L + d


def qwen_forward_flops(fp, cfg, M_img_tok, T_prompt_tok, G_gen_tok):
    """Qwen2.5-VL forward FLOPs -- identical model to flop_ratio_derivation.forward_flops."""
    v = cfg["vision_config"]
    p = fp["role_stored_elements"]
    P = 4.0 * M_img_tok                                   # 2x2 spatial merge
    d_v, L_v = v["hidden_size"], v["depth"]
    W = (v["window_size"] // v["patch_size"]) ** 2
    L_full = len(v["fullatt_block_indexes"])
    L_win = L_v - L_full
    vit_dense = 2 * P * p["vision_tower"]
    vit_attn = 4 * d_v * (L_full * P * P + L_win * P * W)
    merger = 2 * M_img_tok * p["vision_merger"]

    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    lm_prefill = 2 * T_prompt_tok * p["lm_body"]
    lm_prefill_attn = 2.0 * L * T_prompt_tok * T_prompt_tok * d      # causal -> halved
    g = max(G_gen_tok - 1.0, 0.0)
    lm_decode = 2 * g * p["lm_body"]
    lm_decode_attn = 4 * L * d * g * (T_prompt_tok + g / 2.0)
    head = 2 * G_gen_tok * p["lm_head"]
    parts = dict(vision_tower_dense=vit_dense, vision_tower_attn=vit_attn, vision_merger=merger,
                 lm_prefill_dense=lm_prefill, lm_prefill_attn=lm_prefill_attn,
                 lm_decode_dense=lm_decode, lm_decode_attn=lm_decode_attn, lm_head=head)
    parts["TOTAL"] = sum(parts.values())
    return parts


def internvl_forward_flops(fp, cfg, n_tiles, T_prompt_tok, G_gen_tok):
    """InternVL forward FLOPs (NEW CODE -- Lingshu-I-8B is InternVLForConditionalGeneration).

    Differences from the Qwen2.5-VL model, all of them forced by the architecture:
      * The vision tower is an InternViT run on FIXED 448x448 tiles, patch 14 -> 32x32 = 1024
        patches per tile, FULL bidirectional attention in every one of its 24 layers (no
        windowing, no fullatt_block_indexes).
      * `downsample_ratio` 0.5 is a pixel-shuffle applied AFTER the tower, so the tower runs on
        1024 patches/tile and the LM receives `image_seq_length` = 256 tokens/tile.
      * The projector ("multi_modal_projector") runs on the merged 256 tokens/tile.
    """
    v = cfg["vision_config"]
    p = fp["role_stored_elements"]
    ips = int(v["image_size"][0]) // int(v["patch_size"][0])
    P_per_tile = ips * ips                                # 1024 patches per 448px tile
    P = float(P_per_tile * n_tiles)
    M_img_tok = float(cfg["image_seq_length"] * n_tiles)  # 256 tokens per tile after pixel shuffle
    d_v, L_v = v["hidden_size"], v["num_hidden_layers"]

    vit_dense = 2 * P * p["vision_tower"]
    vit_attn = 4 * d_v * L_v * (P_per_tile * P_per_tile) * n_tiles   # per-tile full attention
    merger = 2 * M_img_tok * p["vision_merger"]

    c = cfg["text_config"]
    d, L = c["hidden_size"], c["num_hidden_layers"]
    lm_prefill = 2 * T_prompt_tok * p["lm_body"]
    lm_prefill_attn = 2.0 * L * T_prompt_tok * T_prompt_tok * d
    g = max(G_gen_tok - 1.0, 0.0)
    lm_decode = 2 * g * p["lm_body"]
    lm_decode_attn = 4 * L * d * g * (T_prompt_tok + g / 2.0)
    head = 2 * G_gen_tok * p["lm_head"]
    parts = dict(vision_tower_dense=vit_dense, vision_tower_attn=vit_attn, vision_merger=merger,
                 lm_prefill_dense=lm_prefill, lm_prefill_attn=lm_prefill_attn,
                 lm_decode_dense=lm_decode, lm_decode_attn=lm_decode_attn, lm_head=head)
    parts["TOTAL"] = sum(parts.values())
    return parts


def main():
    out = {
        "title": "ATTACK 3 part 1 -- exact resident-weight footprint and analytic per-pass FLOP-eq "
                 "for every candidate strong leg",
        "date": "2026-08-12",
        "cpu_only": True,
        "method": "safetensors headers only (no weights loaded) + the committed analytic FLOP model",
        "footprint": {},
    }
    for k in MODELS:
        out["footprint"][k] = footprint(k)

    # ---- token geometry: reuse the MEASURED geometry the project's own R32 derivation used ----
    geo_src = os.path.join(ROOT,
                           "results/cascade_methods/artifacts/flop_ratio_derivation_2026-08-03.json")
    geo = json.load(open(geo_src))
    tg = geo["token_geometry"]
    out["token_geometry_source"] = geo_src
    out["token_geometry"] = tg

    TEXT_TOK = tg["text_tok_mean"]            # 46.2  [MEASURED]
    QWEN_IMG_TOK = tg["image_tok_mean"]       # 280.48 [MEASURED, cap320]
    QWEN_T = tg["prompt_tok_mean"]            # 326.68 [MEASURED]
    G = tg["generated_tokens"]["lingshu_32b"]  # 5.6   [MEASURED]; same G for both legs

    cfgs = {k: json.load(open(out["footprint"][k]["snapshot"] + "config.json")) for k in MODELS}

    # Lingshu-I-8B preprocessor: crop_to_patches=false, size 448x448 => ONE tile, and
    # image_seq_length=256 tokens reach the LM.  Stated, not assumed silently.
    I8B_TILES = 1
    i8b_img_tok = cfgs["lingshu_i8b"]["image_seq_length"] * I8B_TILES
    i8b_T = i8b_img_tok + TEXT_TOK

    flops = {}
    flops["lingshu_7b"] = qwen_forward_flops(out["footprint"]["lingshu_7b"], cfgs["lingshu_7b"],
                                             QWEN_IMG_TOK, QWEN_T, G)
    flops["lingshu_32b"] = qwen_forward_flops(out["footprint"]["lingshu_32b"], cfgs["lingshu_32b"],
                                              QWEN_IMG_TOK, QWEN_T, G)
    flops["lingshu_i8b"] = internvl_forward_flops(out["footprint"]["lingshu_i8b"],
                                                  cfgs["lingshu_i8b"], I8B_TILES, i8b_T, G)
    # The AWQ 32B has IDENTICAL architecture/shapes to Lingshu-32B, so its MAC count is identical.
    flops["qwen25vl_32b_awq"] = dict(flops["lingshu_32b"])

    base = flops["lingshu_7b"]["TOTAL"]
    out["per_pass_flops"] = {
        k: dict(gflops=round(v["TOTAL"] / 1e9, 2),
                R_vs_lingshu_7b=round(v["TOTAL"] / base, 4),
                parts_gflops={p: round(x / 1e9, 2) for p, x in v.items() if p != "TOTAL"})
        for k, v in flops.items()
    }
    out["operating_point"] = dict(
        qwen_image_tokens=QWEN_IMG_TOK, qwen_prompt_tokens=QWEN_T,
        i8b_tiles=I8B_TILES, i8b_image_tokens=i8b_img_tok, i8b_prompt_tokens=round(i8b_T, 2),
        text_tokens=TEXT_TOK, generated_tokens=G,
        note=("cap320 open-text anchor, the same operating point the project's R32=3.816 was "
              "derived at.  I-8B's image-token count (256) is set by its own processor config "
              "(crop_to_patches=false, 448x448, image_seq_length=256), not chosen by us; it lands "
              "close to Qwen2.5-VL's measured 280.48 at cap320, so the two workloads are "
              "comparable but NOT identical -- stated, not hidden."))
    out["quantisation_flop_invariance"] = (
        "AWQ/GPTQ/NF4 change how weights are STORED, not how many multiply-accumulates a forward "
        "pass performs.  qwen25vl_32b_awq therefore carries EXACTLY the same MAC-FLOP count as "
        "lingshu_32b (R=%.4f).  Quantisation is a MEMORY (and possibly bandwidth/latency) lever, "
        "NOT a FLOP lever, and on A100 there is no INT4 tensor-core path, so it is not a "
        "throughput lever either." % out["per_pass_flops"]["lingshu_32b"]["R_vs_lingshu_7b"])

    print(json.dumps({k: dict(logical_params=v["logical_params"],
                              weight_gib=v["weight_gib"],
                              arch=v["architectures"][0],
                              roles=v["role_stored_elements"],
                              R_vs_7b=out["per_pass_flops"][k]["R_vs_lingshu_7b"],
                              gflops=out["per_pass_flops"][k]["gflops"])
                      for k, v in out["footprint"].items()}, indent=1))

    with open(os.path.join(OUTDIR, "footprint.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", os.path.join(OUTDIR, "footprint.json"))


if __name__ == "__main__":
    main()
