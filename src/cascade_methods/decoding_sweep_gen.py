#!/usr/bin/env python3
"""decoding_sweep_gen.py -- SWEEP 1: 7B open-text candidate generation, ONE DECODING VARIABLE AT A TIME.

The prompt is HELD FIXED (SYS, verbatim from src/labeling/run_openvqa.py) and the image cap is HELD FIXED
(cap320, the published open-arm setting).  This is the crucial difference from the killed diverse-generation
arm (artifacts/open_diverse_2026-08-10.json), which varied prompt AND temperature together.

Item loaders are replicated VERBATIM from src/labeling/run_openvqa.py and then ASSERTED against the eval
pool idx read from the frozen transfer dumps -- if the loader ever drifts, the assert fires.

One vLLM load, many (setting x dataset) runs, resumable per item.
  python3 src/cascade_methods/decoding_sweep_gen.py --settings cfg.json --out ckpts/openvqa/decoding_sweep
"""
import argparse, glob, io, json, os, re, string, sys, time
from collections import Counter
import numpy as np
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
sys.path.insert(0, ROOT)
from src.training_methods import genframe_data as G          # noqa: E402  (frozen pool identity)

# ---- verbatim from src/labeling/run_openvqa.py -------------------------------------------------
SYS = "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain."
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}


def norm_em(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def score_em(pred, gold):
    p, g = norm_em(pred), norm_em(gold)
    if not p:
        return 0
    if p == g:
        return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g):
        return 1
    return 0


def load_items(ds, n_limit):
    """Verbatim replication of run_openvqa.py's loaders for the three open eval sets."""
    items = []
    if ds == "slake_open":
        d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
        for x in d:
            if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en":
                continue
            ip = os.path.join(root, x["img_name"])
            if not os.path.exists(ip):
                continue
            items.append((x["qid"], x["question"], str(x["answer"]), ip))
    else:
        import pandas as pd
        base = "/data/dan/dataset/vqa_rad/data" if ds == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
        dfs = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))]
        df = pd.concat(dfs, ignore_index=True)
        # Same filters, same order as run_openvqa.py -- but the JPEG decode is DEFERRED until after the
        # [:n_limit] slice, because every engine restart re-runs this loader and decoding all 6,719
        # PathVQA images costs minutes. None of the filters look at pixels, so the item set is identical
        # (and the caller asserts the resulting idx set against the frozen transfer dumps).
        raw = []
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            a = str(a).strip()
            if a.lower() in ("yes", "no"):
                continue
            img = r["image"]
            if not (isinstance(img, dict) and "bytes" in img):
                continue
            raw.append((int(i), str(q), a, img["bytes"]))
        for idx, q, a, b in raw[:n_limit]:
            items.append((idx, q, a, Image.open(io.BytesIO(b)).convert("RGB")))
        return items
    return items[:n_limit]


ap = argparse.ArgumentParser()
ap.add_argument("--settings", required=True, help="JSON list of {tag,temp,top_p,top_k,min_p,rep_pen,seed}")
ap.add_argument("--out", default="ckpts/openvqa/decoding_sweep")
ap.add_argument("--model_path", default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/"
                                        "snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
ap.add_argument("--cap", default="cap320", choices=list(CAP_DIV))
ap.add_argument("--n_samples", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=64)
ap.add_argument("--max_model_len", type=int, default=4096)
ap.add_argument("--gpu_mem", type=float, default=0.88)
ap.add_argument("--enforce_eager", action="store_true",
                help="skip CUDA-graph capture: ~0.5 GiB less VRAM and faster startup, for a SHARED GPU")
ap.add_argument("--datasets", nargs="+", default=["slake_open", "vqa_rad_open", "pathvqa_open"])
A = ap.parse_args()
MAXPX = HIGH_PX // CAP_DIV[A.cap]
OUT = os.path.join(ROOT, A.out); os.makedirs(OUT, exist_ok=True)
SETTINGS = json.load(open(A.settings))

# ---- the frozen eval pool: idx allowlist straight out of the transfer dumps --------------------
POOL = {}
for it in G.load_items():
    POOL.setdefault(it["ds"], set()).add(it["idx"])
N_LIMIT = {"slake_open": 100000, "vqa_rad_open": 100000, "pathvqa_open": 1500}   # --n 1500 in the runner

ITEMS = {}
for ds in A.datasets:
    its = load_items(ds, N_LIMIT[ds])
    got = set(i[0] for i in its)
    assert got == POOL[ds], (f"POOL DRIFT {ds}: loader {len(got)} vs frozen dump {len(POOL[ds])}; "
                             f"missing={len(POOL[ds]-got)} extra={len(got-POOL[ds])}")
    ITEMS[ds] = its
    print(f"[pool ok] {ds}: {len(its)} items == frozen transfer-dump idx set", flush=True)

proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)


def build(q, img):
    im = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX}]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": im + [{"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs); req = {"prompt": text}
    if imgs:
        req["multi_modal_data"] = {"image": imgs}
    return req


llm = LLM(model=A.model_path, tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
          max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 4}, trust_remote_code=True,
          enforce_eager=A.enforce_eager)

LOCKS = os.path.join(OUT, ".locks"); os.makedirs(LOCKS, exist_ok=True)


def claim(tag, ds):
    """Atomically claim (tag, ds) so two workers on two GPUs can drain ONE shared priority queue
    without colliding on the same output file. Steals locks held by dead pids."""
    p = os.path.join(LOCKS, f"{tag}__{ds}")
    try:
        os.mkdir(p)
    except FileExistsError:
        try:
            owner = int(open(os.path.join(p, "pid")).read().strip())
            os.kill(owner, 0)                       # alive -> genuinely claimed
            return False
        except Exception:
            pass                                    # stale (dead owner or unreadable) -> steal it
    with open(os.path.join(p, "pid"), "w") as f:
        f.write(str(os.getpid()))
    return True


for st in SETTINGS:
    sp = SamplingParams(temperature=st["temp"], top_p=st.get("top_p", 1.0), top_k=st.get("top_k", -1),
                        min_p=st.get("min_p", 0.0), repetition_penalty=st.get("rep_pen", 1.0),
                        max_tokens=A.max_tokens, n=A.n_samples, seed=st["seed"])
    for ds in A.datasets:
        ck = os.path.join(OUT, f"ckpt_{ds}_{st['tag']}.jsonl")
        n_have = sum(1 for l in open(ck) if l.strip()) if os.path.exists(ck) else 0
        if n_have == len(ITEMS[ds]):
            print(f"[skip] {st['tag']} {ds}: complete ({n_have})", flush=True)
            continue
        if not claim(st["tag"], ds):
            print(f"[busy] {st['tag']} {ds}: claimed by another worker", flush=True)
            continue
        done = set()
        if os.path.exists(ck):
            for l in open(ck):
                if l.strip():
                    try:
                        done.add(json.loads(l)["idx"])
                    except Exception:
                        pass
        todo = [it for it in ITEMS[ds] if it[0] not in done]
        if not todo:
            print(f"[skip] {st['tag']} {ds}: complete ({len(done)})", flush=True)
            continue
        print(f"[run ] {st['tag']} {ds}: {len(todo)} to go -> {os.path.basename(ck)}", flush=True)
        t0 = time.time(); CH = 128
        with open(ck, "a") as fh:
            for c0 in range(0, len(todo), CH):
                ch = todo[c0:c0 + CH]
                try:
                    reqs = [build(q, im) for (_, q, _, im) in ch]
                    outs = llm.generate(reqs, sp)
                except Exception as e:                                    # per-chunk error guard
                    # A DEAD ENGINE IS FATAL, NEVER SWALLOWED: swallowing it makes every later chunk
                    # "succeed" with zero rows and the run print [done] on an EMPTY file (observed
                    # 2026-08-13). Exit non-zero so the runner restarts a fresh engine and RESUMES.
                    if "EngineDead" in type(e).__name__ or "EngineCore" in str(e):
                        print(f"   !!! FATAL engine death at chunk {c0}: {type(e).__name__}: {e}", flush=True)
                        fh.flush(); sys.exit(17)
                    print(f"   !! chunk {c0} failed: {type(e).__name__}: {e}", flush=True)
                    continue
                for (idx, q, gold, _), o in zip(ch, outs):
                    try:
                        preds = [c.text.strip() for c in o.outputs]
                        cnt = Counter(norm_em(p) for p in preds)
                        modal_norm = cnt.most_common(1)[0][0]
                        modal_pred = next(p for p in preds if norm_em(p) == modal_norm)
                        fh.write(json.dumps({
                            "idx": idx, "question": q, "gold": gold, "preds": preds,
                            "oks_em": [score_em(p, gold) for p in preds],
                            "modal_pred": modal_pred, "n_distinct_em": len(cnt),
                            "gen_tokens_all": [len(c.token_ids) for c in o.outputs]}) + "\n")
                    except Exception as e:                                # per-item error guard
                        print(f"   !! item {idx} failed: {type(e).__name__}: {e}", flush=True)
                fh.flush()
                print(f"   [{min(c0+CH,len(todo))}/{len(todo)}] {(min(c0+CH,len(todo)))/(time.time()-t0):.1f} it/s", flush=True)
        # COMPLETENESS ASSERTION: never print [done] on a short file.
        got = sum(1 for l in open(ck) if l.strip())
        if got != len(ITEMS[ds]):
            print(f"   !!! INCOMPLETE {st['tag']} {ds}: {got}/{len(ITEMS[ds])} rows -- exiting to resume", flush=True)
            sys.exit(18)
        print(f"[done] {st['tag']} {ds} {got}/{len(ITEMS[ds])} rows in {(time.time()-t0)/60:.2f} min", flush=True)
print("DECODING_SWEEP_GEN_DONE", flush=True)
