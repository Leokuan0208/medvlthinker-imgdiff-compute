#!/usr/bin/env python3
"""
unified_pipeline_score.py -- ATTACK 2 GPU half.  Score EVERY candidate of every option-branch item
with ONE verifier and write one resumable JSONL row per (cell, item, candidate).

The scoring function is byte-identical in shape to the incumbent's
src/training_methods/verifier_transfer_eval.py:pyes() -- same system prompt, same body template, same
max_pixels/min_pixels, same P(Yes)/(P(Yes)+P(No)) readout on the last position.  THE ONLY CHANGE is
that the candidate list comes from unified_pipeline.answer_space() (the prompt's own answer space)
instead of an 8-sample pool, and that K candidates of one item are scored in one padded batch.
Null test N4 (--nulltest_batch) measures batch-vs-batch1 deviation on a fixed sample and must be
reported; if it is not ~0 the batching is invalid and --batch_tokens 0 (batch=1) must be used.

  ! HF transformers ONLY.  vLLM silently drops all 192 visual.* LoRA modules
    (sel_eff 0.775204 HF vs 0.702997 vLLM).  This script never imports vLLM.

  HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=1 python3 \
      src/cascade_methods/unified_pipeline_score.py \
      --adapter ckpts/train/lora_verifier_disjoint --tag zeroshot \
      --cells VQA_RAD_closed,PATH_VQA_closed,MedXpertQA-MM,PMC_VQA

Resumable: already-written (i, c) keys are skipped.  Per-item error guard writes {"err": ...} and
moves on.  Launch from the repo root.  nohup, never tmux.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import unified_pipeline as U  # noqa: E402

ROOT = U.ROOT
CKPT = U.CKPT


def load_images(item):
    """Materialise image references exactly the way the deployed loader did."""
    from PIL import Image
    out = []
    for p in item["images"]:
        if item["img_kind"] == "path":
            out.append(p)                       # qwen_vl_utils opens the path itself
        elif item["img_kind"] == "raw":
            out.append(Image.open(p))           # VQA_RAD: HF-decoded PIL, no .convert
        elif item["img_kind"] == "rawrgb":
            out.append(Image.open(p).convert("RGB"))
        else:
            raise ValueError(item["img_kind"])
    return out


def build_msgs(item, cand):
    imgs = load_images(item)
    content = [{"type": "image", "image": im, "max_pixels": U.MAXPX, "min_pixels": U.MINPX}
               for im in imgs]
    body = (f"Question: {item['question']}\nProposed answer: {cand}\n"
            "Is the proposed answer correct? Answer Yes or No.")
    content.append({"type": "text", "text": body})
    return [{"role": "system", "content": U.SYS_VERIF},
            {"role": "user", "content": content}]


def est_vis_tokens(item):
    """Upper bound on visual tokens for one candidate of this item (max_pixels // 28**2)."""
    from PIL import Image
    tot = 0
    for p in item["images"]:
        try:
            with Image.open(p) as im:
                px = im.size[0] * im.size[1]
        except Exception:
            px = U.MAXPX
        tot += min(px, U.MAXPX) // (28 * 28)
    return max(tot, 64)


def done_keys(path):
    seen = set()
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                seen.add((int(r["i"]), int(r["c"])))
            except Exception:
                pass
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="ckpts/train/lora_verifier_disjoint")
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--cells", default=",".join(U.OPTION_CELLS))
    ap.add_argument("--model_path", default="lingshu-medical-mllm/Lingshu-7B")
    ap.add_argument("--batch_tokens", type=int, default=16000,
                    help="visual-token budget per padded batch; 0 forces batch=1")
    ap.add_argument("--max_batch", type=int, default=8)
    ap.add_argument("--nulltest_batch", type=int, default=0,
                    help=">0: score this many (item,candidate) pairs BOTH at batch=1 and batched, "
                         "report max abs deviation, write nothing else")
    ap.add_argument("--min_free_gib", type=float, default=26.0)
    ap.add_argument("--wait_s", type=float, default=0.0,
                    help=">0: poll until a GPU has --min_free_gib free, up to this many seconds")
    ap.add_argument("--deadline_s", type=float, default=0.0)
    ap.add_argument("--vram_probe", type=int, default=0,
                    help=">0: measure peak VRAM of the OPTION BRANCH on this many items per cell "
                         "(batch=1, the deployed serving shape) and write a part file; score nothing")
    A = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)

    # ---- queue politely: never oversubscribe a shared GPU -------------------------------------
    if A.wait_s > 0:
        t0 = time.time()
        while time.time() - t0 < A.wait_s:
            import subprocess
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True)
            free = []
            for line in q.stdout.strip().splitlines():
                idx, used, tot = [int(x.strip()) for x in line.split(",")]
                free.append((idx, (tot - used) / 1024.0))
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            if vis is not None and vis != "":
                keep = {int(x) for x in vis.split(",")}
                free = [f for f in free if f[0] in keep]
            if free and max(f[1] for f in free) >= A.min_free_gib:
                print(f"[queue] GPU free {free} -- starting", flush=True)
                break
            print(f"[queue] waiting, free={free} need {A.min_free_gib} GiB", flush=True)
            time.sleep(120)

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_utils import process_vision_info
    from peft import PeftModel

    DEV = "cuda"
    proc = AutoProcessor.from_pretrained(A.model_path)
    proc.tokenizer.padding_side = "left"        # so logits[:, -1] is the true last token in a batch
    YES = proc.tokenizer.encode("Yes", add_special_tokens=False)[0]
    NO = proc.tokenizer.encode("No", add_special_tokens=False)[0]
    print("[load] base + adapter ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        A.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(DEV)
    model = PeftModel.from_pretrained(model, os.path.join(ROOT, A.adapter))
    model.eval()
    nvis = sum(1 for n, _ in model.named_parameters() if "lora" in n and "visual" in n)
    print(f"[load] adapter={A.adapter}  lora tensors on the vision tower = {nvis}", flush=True)

    def score_batch(pairs):
        """pairs = [(item, cand_text)]  ->  [P(Yes)/(P(Yes)+P(No))]"""
        msgs = [build_msgs(it, c) for it, c in pairs]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs, vids = [], []
        for m in msgs:
            a, b = process_vision_info(m)
            imgs += (a or [])
            if b:
                vids += b
        enc = proc(text=texts, images=imgs or None, videos=vids or None,
                   return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        py = torch.exp(lg[:, YES]); pn = torch.exp(lg[:, NO])
        s = (py / (py + pn)).clamp(0, 1)
        return [float(x) for x in s.cpu().numpy()]

    work = U.build_worklist()

    # ---- VRAM probe: what does the OPTION BRANCH actually cost to serve? -----------------------
    # Convention identical to artifacts/vram_testtime_2026-08-11.json: (b) torch peak allocated,
    # (c) torch peak reserved, (d) NVML board used.  batch=1, one process, base + LoRA adapter,
    # i.e. the deployed shape -- generator and verifier SHARE the one copy of the 7B weights.
    if A.vram_probe > 0:
        import subprocess
        def board_gib():
            q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            vals = [int(x.strip()) for x in q.stdout.strip().splitlines()]
            vis = os.environ.get("CUDA_VISIBLE_DEVICES")
            k = int(vis.split(",")[0]) if vis else 0
            return vals[k] / 1024.0
        w_alloc = torch.cuda.memory_allocated() / 2**30
        rows_out, rng = [], np.random.default_rng(U.SEED_BOOT)
        for cell in A.cells.split(","):
            rows = work[cell]
            ids = rng.choice(len(rows), size=min(len(rows), A.vram_probe), replace=False)
            for j in ids:
                r = rows[int(j)]
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                b0 = board_gib()
                try:
                    for c in range(len(r["cands"])):        # batch=1, one candidate at a time
                        score_batch([(r, r["cands"][c])])
                except Exception as e:
                    rows_out.append({"cell": cell, "i": r["i"], "err": str(e)[:160]}); continue
                rows_out.append({
                    "cell": cell, "i": r["i"], "k_candidates": len(r["cands"]),
                    "est_vision_tokens": est_vis_tokens(r),
                    "b_peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 4),
                    "c_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 4),
                    "d_board_used_gib_after": round(board_gib(), 4),
                    "d_board_used_gib_before": round(b0, 4)})
        ok = [r for r in rows_out if "err" not in r]
        out = {"scenario": "unified option branch, batch=1, Lingshu-7B + LoRA verifier, "
                           f"max_pixels={U.MAXPX} min_pixels={U.MINPX}",
               "adapter": A.adapter, "n": len(ok), "n_failed": len(rows_out) - len(ok),
               "a_weights_resident_gib": round(w_alloc, 4),
               "b_peak_allocated_gib": {"mean": float(np.mean([r["b_peak_allocated_gib"] for r in ok])),
                                        "peak": float(np.max([r["b_peak_allocated_gib"] for r in ok]))},
               "c_peak_reserved_gib": {"mean": float(np.mean([r["c_peak_reserved_gib"] for r in ok])),
                                       "peak": float(np.max([r["c_peak_reserved_gib"] for r in ok]))},
               "d_board_used_gib": {"mean": float(np.mean([r["d_board_used_gib_after"] for r in ok])),
                                    "peak": float(np.max([r["d_board_used_gib_after"] for r in ok]))},
               "caveat": "d (board) includes any OTHER process sharing the card; b and c are this "
                         "process only. Three other rounds shared these GPUs during this measurement, "
                         "so read b/c, not d, unless d_board_used_gib_before is ~0.",
               "rows": rows_out}
        os.makedirs(U.PARTS, exist_ok=True)
        p = os.path.join(U.PARTS, f"vram_option_branch_{A.tag}.json")
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)
        return

    # ---- N4: batch-vs-batch1 numerics ---------------------------------------------------------
    if A.nulltest_batch > 0:
        rng = np.random.default_rng(U.SEED_BOOT)
        sample = []
        for cell in A.cells.split(","):
            rows = work[cell]
            ids = rng.choice(len(rows), size=min(len(rows), A.nulltest_batch // 4), replace=False)
            for j in ids:
                r = rows[int(j)]
                for c in range(len(r["cands"])):
                    sample.append((r, c))
        sample = sample[:A.nulltest_batch]
        s1 = [score_batch([(r, r["cands"][c])])[0] for r, c in sample]
        sb = []
        i = 0
        while i < len(sample):
            chunk = sample[i:i + 4]
            sb += score_batch([(r, r["cands"][c]) for r, c in chunk])
            i += 4
        d = np.abs(np.array(s1) - np.array(sb))
        out = {"n_pairs": len(sample), "max_abs_dev": float(d.max()),
               "mean_abs_dev": float(d.mean()), "p99_abs_dev": float(np.percentile(d, 99)),
               "argmax_flips": int(0)}
        # argmax flips at the item level
        flips = 0
        k = 0
        per_item = {}
        for (r, c), a, b in zip(sample, s1, sb):
            per_item.setdefault((r["cell"], r["i"]), {})[c] = (a, b)
        for key, d2 in per_item.items():
            if len(d2) < 2:
                continue
            cs = sorted(d2)
            if int(np.argmax([d2[c][0] for c in cs])) != int(np.argmax([d2[c][1] for c in cs])):
                flips += 1
            k += 1
        out["argmax_flips"] = flips
        out["n_full_items"] = k
        p = os.path.join(U.PARTS, f"n4_batch_numerics_{A.tag}.json")
        os.makedirs(U.PARTS, exist_ok=True)
        json.dump(out, open(p, "w"), indent=1)
        print(json.dumps(out, indent=1), flush=True)
        return

    T0 = time.time()
    for cell in A.cells.split(","):
        rows = work[cell]
        path = os.path.join(CKPT, f"{A.tag}_{cell}.jsonl")
        seen = done_keys(path)
        todo = [(r, c) for r in rows for c in range(len(r["cands"])) if (r["i"], c) not in seen]
        print(f"[{cell}] {len(rows)} items, {len(todo)} pair(s) to do "
              f"({len(seen)} resumed)", flush=True)
        f = open(path, "a")
        i = 0
        t_cell = time.time()
        while i < len(todo):
            # dynamic batch: same item's candidates go together, respecting the token budget
            batch = [todo[i]]
            vt = est_vis_tokens(todo[i][0])
            j = i + 1
            if A.batch_tokens > 0:
                while (j < len(todo) and len(batch) < A.max_batch
                       and vt + est_vis_tokens(todo[j][0]) <= A.batch_tokens):
                    vt += est_vis_tokens(todo[j][0])
                    batch.append(todo[j]); j += 1
            try:
                ss = score_batch([(r, r["cands"][c]) for r, c in batch])
                for (r, c), s in zip(batch, ss):
                    f.write(json.dumps({"cell": cell, "i": r["i"], "c": c,
                                        "s": round(float(s), 6)}) + "\n")
            except Exception as e:                       # per-batch error guard
                import traceback
                traceback.print_exc()
                for r, c in batch:
                    try:
                        s = score_batch([(r, r["cands"][c])])[0]
                        f.write(json.dumps({"cell": cell, "i": r["i"], "c": c,
                                            "s": round(float(s), 6)}) + "\n")
                    except Exception as e2:
                        f.write(json.dumps({"cell": cell, "i": r["i"], "c": c,
                                            "err": str(e2)[:200]}) + "\n")
                import torch as _t
                _t.cuda.empty_cache()
            i = j if A.batch_tokens > 0 else i + 1
            if (i // 200) != ((i - len(batch)) // 200):
                f.flush()
                el = time.time() - t_cell
                print(f"  [{cell}] {i}/{len(todo)}  {el:.0f}s  "
                      f"eta {el / max(1, i) * (len(todo) - i):.0f}s", flush=True)
            if A.deadline_s > 0 and time.time() - T0 > A.deadline_s:
                print("[deadline] stopping cleanly", flush=True)
                f.flush(); f.close()
                return
        f.flush(); f.close()
        print(f"[{cell}] done in {time.time() - t_cell:.0f}s", flush=True)
    print(f"[all] {time.time() - T0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
