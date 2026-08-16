#!/usr/bin/env python3
"""vrestruct_prefill_analyze.py -- turn the measured token counts into the answer to QUESTION 2.

Reads _vrestruct_parts/prefill.jsonl (written by vrestruct_prefill.py) and reports, per
(engine-config, N), how much prefill work the GPU ACTUALLY did relative to what one sample would
cost -- i.e. whether SamplingParams(n=N) shares the prefill.

THE REFERENCE.  Each (phase, apc, N, rep) cell ran a DISJOINT slice of items, so cells cannot be
compared by raw totals.  This script rebuilds every slice's exact geometry on CPU with the same
processor and the same deterministic slicing, giving each cell its own one-encode reference:
    prompt_tok_ref  = sum over the slice of the prompt's token count
    patches_ref     = sum over the slice of the image's pre-merge patch count
Then
    lm_prefill_sharing_ratio = (lm_positions - gen_tok) / prompt_tok_ref
    vision_sharing_ratio     = vit_patches / patches_ref
A perfectly shared prefill gives ~1.0 at every N; a fully unshared one gives ~N.

FLOP-eq(N) is then rebuilt with the repo's own analytic model
(flop_ratio_derivation.forward_flops components), charging the MEASURED positions and patches.

    OMP_NUM_THREADS=4 python3 src/cascade_methods/vrestruct_prefill_analyze.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, os.path.join(ROOT, "src/cascade_methods"))
sys.path.insert(0, os.path.join(ROOT, "src/training_methods"))

import vrestruct_lib as V              # noqa: E402
import vrestruct_prefill as PF         # noqa: E402

PARTS = V.PARTS
SINK = os.path.join(PARTS, "prefill.jsonl")
GEO = os.path.join(PARTS, "prefill_geometry.json")


def build_geometry(per_cell, reps, apcs, phases=("count", "time"), NS=(1, 2, 4, 8)):
    """Exact per-cell reference geometry, rebuilt on CPU with the same deterministic slicing."""
    if os.path.exists(GEO):
        return json.load(open(GEO))
    os.environ.setdefault("HF_HOME", "/data/dan/hf_cache")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    cells = [(ph, apc, N, rep) for ph in phases for apc in apcs for N in NS
             for rep in range(1, reps + 1)]
    need = per_cell * len(cells) + 8
    proc = AutoProcessor.from_pretrained(PF.MODEL, trust_remote_code=True)
    tok = proc.tokenizer
    pool = PF.load_items(need)
    print(f"rebuilding geometry for {len(pool)} items ...", flush=True)

    rows = []
    for k, (ds, idx, q, img) in enumerate(pool):
        im = [{"type": "image", "image": img, "max_pixels": PF.MAXPX, "min_pixels": PF.MIN_PX}]
        msgs = [{"role": "system", "content": PF.SYS},
                {"role": "user", "content": im + [{"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        out = proc(text=[text], images=imgs, return_tensors="pt")
        n_prompt = int(out["input_ids"].shape[1])
        n_patch = int(out["pixel_values"].shape[0])
        rows.append(dict(k=k, ds=ds, idx=str(idx), prompt_tok=n_prompt, patches=n_patch,
                         vision_tok=n_patch / 4.0))
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(pool)}", flush=True)

    body = rows[8:]
    geo = {"per_cell": per_cell, "reps": reps, "apcs": list(apcs), "cells": {}}
    for i, c in enumerate(cells):
        sl = body[i * per_cell:(i + 1) * per_cell]
        geo["cells"]["|".join(map(str, c))] = dict(
            n=len(sl), prompt_tok_ref=int(sum(r["prompt_tok"] for r in sl)),
            patches_ref=int(sum(r["patches"] for r in sl)),
            vision_tok_ref=float(sum(r["vision_tok"] for r in sl)),
            text_tok_ref=float(sum(r["prompt_tok"] - r["vision_tok"] for r in sl)))
    json.dump(geo, open(GEO, "w"), indent=1)
    return geo


def flops_of(patches, lm_positions, gen_tok, n_seq_prefill, mean_prompt_tok):
    """Rebuild the repo's analytic FLOP components from MEASURED counts.

    Dense terms are exactly linear in positions/patches, so the measured totals give them exactly.
    The two attention terms are quadratic and need per-sequence lengths, which the hooks do not
    give; they are reconstructed from the mean prompt length and together account for 0.38% of a
    7B forward at this operating point (flop_ratio_derivation component_shares_pct), so the
    approximation cannot move any conclusion here.
    """
    import flop_ratio_derivation as F
    pc = V._param_counts()
    p, c = pc["params"], pc["config"]
    v = c["vision_config"]
    d_v, L_v = v["hidden_size"], v["depth"]
    Wp = (v["window_size"] // v["patch_size"]) ** 2
    L_full = len(v["fullatt_block_indexes"])
    L_win = L_v - L_full
    P = float(patches)
    vit_dense = 2 * P * p["vision_tower"]
    vit_attn = 4 * d_v * (L_full * P * P / max(n_seq_prefill, 1) + L_win * P * Wp)
    merger = 2 * (P / 4.0) * p["vision_merger"]
    d, L = c["hidden_size"], c["num_hidden_layers"]
    lm_dense = 2 * float(lm_positions) * p["lm_body"]
    T = float(mean_prompt_tok)
    lm_attn = 2.0 * L * T * T * d * max(n_seq_prefill, 1)
    head = 2 * float(gen_tok) * p["lm_head"]
    tot = vit_dense + vit_attn + merger + lm_dense + lm_attn + head
    return dict(vision_tower_dense=vit_dense, vision_tower_attn=vit_attn, vision_merger=merger,
                lm_dense=lm_dense, lm_attn=lm_attn, lm_head=head, TOTAL=tot)


def main():
    if not os.path.exists(SINK):
        raise SystemExit(f"no measurement yet at {SINK}")
    recs = [json.loads(l) for l in open(SINK) if l.strip()]
    meta = {}
    mp = os.path.join(PARTS, "prefill.meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    per_cell = recs[0]["n_items"]
    apcs = sorted({r["apc"] for r in recs}, key=lambda x: {"default": 0, "on": 1, "off": 2}.get(x, 9))
    reps = max(r["rep"] for r in recs)
    geo = build_geometry(per_cell, max(reps, 3), ("default", "on", "off"))

    c = V.cost_constants()
    unit = c["unit_tflop"] * 1e12

    by = defaultdict(list)
    for r in recs:
        key = "|".join([r["phase"], r["apc"], str(r["N"]), str(r["rep"])])
        g = geo["cells"].get(key)
        if g is None:
            continue
        pre = r["lm_positions"] - r["gen_tok_total"] if r["lm_calls"] else None
        row = dict(r)
        row["prompt_tok_ref"] = g["prompt_tok_ref"]
        row["patches_ref"] = g["patches_ref"]
        row["lm_prefill_positions"] = pre
        row["lm_prefill_sharing_ratio"] = (pre / g["prompt_tok_ref"]) if pre is not None else None
        row["vision_sharing_ratio"] = (r["vit_patches"] / g["patches_ref"]
                                       if g["patches_ref"] else None)
        row["gen_tok_per_item"] = r["gen_tok_total"] / r["n_items"]
        if r["lm_calls"]:
            f = flops_of(r["vit_patches"], r["lm_positions"], r["gen_tok_total"],
                         r["n_items"], g["prompt_tok_ref"] / g["n"])
            row["flops_total"] = f["TOTAL"]
            row["flopeq_per_question"] = f["TOTAL"] / r["n_items"] / unit
            row["flops_parts"] = f
        by[(r["phase"], r["apc"], r["N"])].append(row)

    table = {}
    for (ph, apc, N), rows in sorted(by.items()):
        def agg(k):
            xs = [x[k] for x in rows if x.get(k) is not None]
            return (dict(mean=float(np.mean(xs)), sd=float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0,
                         n=len(xs), values=[float(x) for x in xs]) if xs else None)
        table[f"{ph}|{apc}|N{N}"] = dict(
            phase=ph, apc=apc, N=N, n_reps=len(rows),
            effective_enable_prefix_caching=rows[0].get("effective_enable_prefix_caching"),
            lm_prefill_sharing_ratio=agg("lm_prefill_sharing_ratio"),
            vision_sharing_ratio=agg("vision_sharing_ratio"),
            flopeq_per_question=agg("flopeq_per_question"),
            wall_s=agg("wall_s"), gen_tok_per_item=agg("gen_tok_per_item"),
            num_cached_tokens_total=agg("num_cached_tokens_total"))

    # ---- the verdict: FLOP-eq(N) normalised so that N=1 in the SAME config is 1.0 -------------
    verdict = {}
    for ph, apc in sorted({(r["phase"], r["apc"]) for r in recs}):
        base = table.get(f"{ph}|{apc}|N1", {}).get("flopeq_per_question")
        wbase = table.get(f"{ph}|{apc}|N1", {}).get("wall_s")
        row = {}
        for N in (1, 2, 4, 8):
            t = table.get(f"{ph}|{apc}|N{N}")
            if not t:
                continue
            row[f"N{N}"] = dict(
                flopeq_per_question=t["flopeq_per_question"]["mean"] if t["flopeq_per_question"] else None,
                flopeq_rel_to_N1=(t["flopeq_per_question"]["mean"] / base["mean"]
                                  if (t["flopeq_per_question"] and base) else None),
                lm_prefill_sharing_ratio=(t["lm_prefill_sharing_ratio"]["mean"]
                                          if t["lm_prefill_sharing_ratio"] else None),
                vision_sharing_ratio=(t["vision_sharing_ratio"]["mean"]
                                      if t["vision_sharing_ratio"] else None),
                wall_rel_to_N1=(t["wall_s"]["mean"] / wbase["mean"] if wbase else None))
        # the two competing predictions
        row["_predictions"] = dict(
            as_charged_N8=8.0,
            shared_prefill_G8=V.G_of_N(8, c),
            note="as-charged = the project's cost model (1.0 FLOP-eq per sample); "
                 "shared_prefill = cost_floor convention B, G(N)=prefill_share+N*decode_share")
        verdict[f"{ph}|{apc}"] = row

    out = dict(
        title="QUESTION 2 -- does vLLM SamplingParams(n=N) share the generation prefill?",
        date="2026-08-16", meta=meta, per_cell_items=per_cell,
        instruments=dict(
            I1="direct token accounting via forward pre-hooks on the LM's first decoder layer and "
               "the vision tower (phase 'count', enforce_eager=True)",
            I2="RequestOutput.num_cached_tokens",
            I3="wall clock vs N with CUDA graphs on (phase 'time')"),
        design=meta.get("design"),
        cells=table, verdict=verdict, cost_constants_used=dict(
            unit_tflop=c["unit_tflop"], prefill_share_7b=c["prefill_share_7b"],
            decode_share_7b=c["decode_share_7b"]))
    json.dump(out, open(os.path.join(PARTS, "prefill_analysis.json"), "w"), indent=1, default=float)

    print("\n=== QUESTION 2: prefill sharing ===")
    for k, row in verdict.items():
        print(f"\n[{k}]")
        for N in (1, 2, 4, 8):
            r = row.get(f"N{N}")
            if not r:
                continue
            def f(x):
                return "   n/a" if x is None else f"{x:6.3f}"
            print(f"   N={N}: LMprefill x{f(r['lm_prefill_sharing_ratio'])}  "
                  f"vision x{f(r['vision_sharing_ratio'])}  "
                  f"FLOPeq/q {f(r['flopeq_per_question'])}  "
                  f"rel-to-N1 {f(r['flopeq_rel_to_N1'])}  wall rel {f(r['wall_rel_to_N1'])}")
    print(f"\n   competing predictions for N=8: as-charged 8.000 | shared-prefill "
          f"{V.G_of_N(8, c):.4f}")


if __name__ == "__main__":
    main()
