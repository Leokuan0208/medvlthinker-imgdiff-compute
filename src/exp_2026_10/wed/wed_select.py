"""WED (Wasserstein Equilibrium Decoding / Wasserstein-BDG) as a BASELINE SELECTOR over our best-of-8 pools.

Paper : Hagen et al., "Wasserstein Equilibrium Decoding for Reliable Medical Visual Question Answering",
        arXiv 2605.18313 (v2, 2026-06-13).
Code  : github.com/luca-hagen/Wasserstein-BDG-medical-VQA (branch main), fetched by fetch_wed.py into
        /data/dan/repos/Wasserstein-BDG-medical-VQA-main.  NO LICENSE in that repo (README: "A license will be
        added upon public release") -- so we IMPORT the authors' files, we do not copy them into this repo.

What is used VERBATIM (imported, unmodified):
  config.ExperimentConfig (all defaults; use_wasserstein_update=False as in main.py), the prompt templates,
  utils.{canonicalize_answer, is_bad_candidate, semantic_dedup, cluster_and_select, compute_ground_metric,
  SapBERTCLSEmbedder, wasserstein_1 (POT)}, BDG.{generator_init, verifier_init, bdg_wasserstein},
  QwenBackend.QwenBackend.{logprob_completions, logprob_prompts}.
What is ADAPTED (and why):
  * Candidate set: the authors SAMPLE their own candidates (QwenBackend.build_candidate_set). We instead feed our
    8 pool samples through the post-sampling half of build_candidate_set, re-written here line-for-line
    (canonicalize -> is_bad filter -> lowercase dedup -> SapBERT-embed "question\nAnswer: ans" ->
    semantic_dedup(0.99) -> n_target=min(8,n), made even -> cluster_and_select -> D).  Not applicable to a
    fixed pool: their truncation filter, their sampling prompts/temperatures (0.5,1.0), N_max=16.
  * Model class: QwenBackend hard-codes Qwen3VLForConditionalGeneration (absent from transformers 4.55.2).
    Our generator is Lingshu-7B (Qwen2.5-VL architecture), so the name is aliased to
    Qwen2_5_VLForConditionalGeneration (one import line rewritten at exec time). Everything else runs as-is.
  * config.py overwrites HF_HOME/HF_TOKEN at import with the authors' paths; we restore ours right after.
  * Image resolution: the processor's max_pixels is set to the pool's generation cap (pool field `cap`, else
    --cap, default fullres = 1280*28*28) instead of the processor default (12.8 MP).
  * Always answers: if WED's filter leaves no candidate (all filtered: e.g. bare-number answers, >10 words) we
    fall back to the pool's modal answer and flag it (`fallback`).  The authors' main.py SKIPS such questions.
Pool log-probs (cumlogprob_all / seqlogprob_all) are NOT used: WED's generator/verifier initial strategies are
teacher-forced log-probs under WED's OWN prompts (generator "Answer:" prompt; verifier C/I token), which must be
recomputed with the VLM + image.  Hence WED needs a VLM forward pass for every pool, old or new.
"""
import argparse, glob, io, json, os, re, sys, time
from collections import Counter

WED_REPO = "/data/dan/repos/Wasserstein-BDG-medical-VQA-main"
LINGSHU7B = "/data/dan/hf_cache/hub/models--lingshu-medical-mllm--Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
SAPBERT = "/data/dan/hf_cache/hub/models--cambridgeltl--SapBERT-from-PubMedBERT-fulltext/snapshots/090663c3ae57bf35ffe4d0d468a2a88d03051a4d"
HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28                                   # = src/labeling/run_openvqa.py
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--gpu", action="store_true", help="run the VLM on cuda:0 (of CUDA_VISIBLE_DEVICES); default CPU")
ap.add_argument("--model", default=LINGSHU7B, help="generator==verifier VLM (WED default: same model for both)")
ap.add_argument("--dtype", default=None, help="float16|bfloat16|float32; default float16 on GPU (as WED main.py), float32 on CPU")
ap.add_argument("--cap", default=None, choices=list(CAP_DIV), help="override image cap (default: pool's `cap` field, else fullres)")
ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke tests)")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--candidates_only", action="store_true", help="no VLM: only build WED's candidate sets (diagnostic); "
                "questions WED actually has to decide (>=2 candidates) get pick_index=null")
A = ap.parse_args()
if not A.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ---- import the authors' code; undo config.py's environment side effects BEFORE transformers is imported ----
sys.path.insert(0, "/data/dan/repos/wed_pylib"); sys.path.insert(0, WED_REPO)
import config as wed_config  # noqa: E402
for k in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "XDG_CACHE_HOME", "HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
    os.environ.pop(k, None)
os.environ.update(HF_HOME="/data/dan/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
import numpy as np, torch, transformers  # noqa: E402
from PIL import Image  # noqa: E402
torch.set_num_threads(A.threads)
import utils as wed_utils  # noqa: E402
import BDG as wed_bdg  # noqa: E402
# QwenBackend.py, executed from the authors' file with ONE line changed: the Qwen3-VL class import is rebound to
# Qwen2.5-VL (Lingshu-7B's architecture).  transformers' lazy module refuses a plain attribute alias.
import types  # noqa: E402
_src = open(os.path.join(WED_REPO, "QwenBackend.py")).read()
_old = "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor"
assert _src.count(_old) == 1
_mod = types.ModuleType("QwenBackend"); _mod.__file__ = os.path.join(WED_REPO, "QwenBackend.py"); sys.modules["QwenBackend"] = _mod
exec(compile(_src.replace(_old, "from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration, AutoProcessor"),
             _mod.__file__, "exec"), _mod.__dict__)
QwenBackend = _mod.QwenBackend
assert wed_utils.HAS_POT, "POT missing: pip install --target /data/dan/repos/wed_pylib pot"

# ---- pool + images -------------------------------------------------------------------------------------------
rows = [json.loads(l) for l in open(A.pool) if l.strip()]
if A.limit: rows = rows[:A.limit]
m = re.match(r"ckpt_(.+?_open(?:_train)?)_", os.path.basename(A.pool))
cell = m.group(1)
JSON_CELLS = {"radimagenet_open": "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json",
              "gemex_open": "/data/dan/dataset/gemex_cell/gemex_open.json",
              "vqamed_open": "/data/dan/dataset/vqamed_cell/vqamed_open.json",
              "quilt_open": "/data/dan/dataset/quilt_cell/quilt_open.json",
              "omnimed_open": "/data/dan/dataset/omnimed_opentext/omnimed_open.json",
              "kvasir_x1_open": "/data/dan/dataset/kvasir_x1_cell/kvasir_x1_open.json",
              "kvasir_open": "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"}

def load_items(cell):
    """idx -> (question, image source) exactly as src/labeling/run_openvqa.py builds them."""
    split = "train" if cell.endswith("_train") else "test"
    if cell in JSON_CELLS:
        return {r["idx"]: (r["question"], r["img_path"]) for r in json.load(open(JSON_CELLS[cell]))}
    if cell.startswith("slake_open"):
        return {x["qid"]: (x["question"], os.path.join("/data/dan/dataset/slake/imgs", x["img_name"]))
                for x in json.load(open(f"/data/dan/dataset/slake/{split}.json"))
                if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en"}
    import pandas as pd
    base = "/data/dan/dataset/vqa_rad/data" if cell.startswith("vqa_rad_open") else "/data/dan/dataset/path_vqa/data"
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, f"{split}-*.parquet")))], ignore_index=True)
    out = {}
    for i, r in df.iterrows():
        q = r.get("question")
        if q is None and "conversations" in r:
            q = r["conversations"][0]["value"].replace("<image>", "").strip()
        out[int(i)] = (str(q), r["image"]["bytes"])
    return out

items = load_items(cell)
def get_image(src):
    return Image.open(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src).convert("RGB")

cap = A.cap or rows[0].get("cap") or "fullres"
cfg = wed_config.ExperimentConfig(use_wasserstein_update=False)                   # = main.py
cfg_classic = wed_config.ExperimentConfig(**{**cfg.__dict__, "use_wasserstein": False, "use_wasserstein_update": False})
dev = "cuda:0" if A.gpu else "cpu"
dtype = getattr(torch, A.dtype) if A.dtype else (torch.float16 if A.gpu else torch.float32)
print(f"[wed] cell={cell} n={len(rows)} cap={cap} device={dev} dtype={dtype} model={A.model}", flush=True)
t_load = time.time()
if not A.candidates_only:
    backend = QwenBackend(A.model, device=dev, dtype=dtype)
    ip = backend.processor.image_processor
    ip.max_pixels, ip.min_pixels = HIGH_PX // CAP_DIV[cap], MIN_PX
    if isinstance(getattr(ip, "size", None), dict):
        ip.size = {"longest_edge": ip.max_pixels, "shortest_edge": ip.min_pixels}
embedder = wed_utils.SapBERTCLSEmbedder(SAPBERT, device=dev)
print(f"[wed] models loaded in {time.time() - t_load:.1f}s", flush=True)

def canon_key(s):
    return wed_utils.canonicalize_answer(s).lower().strip()

def build_Y(question, preds):
    """Post-sampling half of QwenBackend.build_candidate_set, applied to a fixed pool (line-for-line)."""
    pool, seen, semantic_pool = [], set(), []
    for p in preds:
        if len(pool) >= cfg.n_oversample: break
        ans = wed_utils.canonicalize_answer(p)
        if wed_utils.is_bad_candidate(ans, cfg): continue
        key = ans.lower().strip()
        if key in seen: continue
        pool.append(ans); seen.add(key); semantic_pool.append(question + "\nAnswer: " + ans)
    if not pool: return None, None
    sem_emb = embedder.encode(semantic_pool, normalize_embeddings=True)
    pool, sem_emb = wed_utils.semantic_dedup(pool, sem_emb, threshold=0.99)
    n_target = min(cfg.n_candidates, len(pool))
    if n_target > 1 and n_target % 2 != 0: n_target -= 1
    sel_texts, sel_emb, _ = wed_utils.cluster_and_select(pool, sem_emb, n_target, cfg.cluster_seed)
    return sel_texts, wed_utils.compute_ground_metric(sel_emb, cfg.ground_metric)

def modal_index(preds):
    c = Counter(canon_key(p) for p in preds)
    top = max(c.values())
    return next(i for i, p in enumerate(preds) if c[canon_key(p)] == top)

os.makedirs(os.path.dirname(os.path.abspath(A.out)) or ".", exist_ok=True)
t0 = time.time(); n_fb = 0
with open(A.out, "w") as fo:
    for k, r in enumerate(rows):
        q_item, src = items[r["idx"]]
        assert q_item.strip() == r["question"].strip(), f"question mismatch at idx {r['idx']} (stale pool?)"
        question, preds = r["question"], r["preds"]
        tq = time.time(); Y, D = build_Y(question, preds)
        rec = {"idx": r["idx"]}
        if Y is None:                                    # authors skip; we must answer -> modal pool answer
            mi = modal_index(preds); n_fb += 1
            rec.update(pick_text=preds[mi], pick_index=mi, fallback="no_valid_candidate", n_cand=0)
        else:
            if len(Y) < 2:
                ans = {"bdg_wasserstein": Y[0], "bdg_classic": Y[0], "verifier_D": Y[0], "greedy_G": Y[0]}; it = 0
            elif A.candidates_only:                      # no VLM: record the candidate set only, no pick
                rec.update(pick_text=None, pick_index=None, n_cand=len(Y), candidates=Y)
                fo.write(json.dumps(rec) + "\n"); continue
            else:
                image = get_image(src)
                G_c, G_i, scd = wed_bdg.generator_init(question, Y, backend, image, cfg)
                V = wed_bdg.verifier_init(question, Y, backend, image, cfg)
                res_w = wed_bdg.bdg_wasserstein(Y, G_c, G_i, V, D, cfg)
                res_c = wed_bdg.bdg_wasserstein(Y, G_c, G_i, V, D, cfg_classic)
                ans = {"bdg_wasserstein": res_w.answer, "bdg_classic": res_c.answer,
                       "verifier_D": Y[int(np.argmax(V[:, 0]))], "greedy_G": Y[int(np.argmax(G_c))]}
                it = res_w.n_iters
                rec.update(converged=bool(res_w.converged), converge_reason=res_w.converge_reason)
            pick = ans["bdg_wasserstein"]
            pi = next(i for i, p in enumerate(preds) if canon_key(p) == pick.lower().strip())
            rec.update(pick_text=preds[pi], pick_index=pi, wed_answer=pick, n_cand=len(Y), n_iters=it,
                       alt={m: next(i for i, p in enumerate(preds) if canon_key(p) == a.lower().strip())
                            for m, a in ans.items() if m != "bdg_wasserstein"})
        rec["sec"] = round(time.time() - tq, 2); fo.write(json.dumps(rec) + "\n"); fo.flush()
        if (k + 1) % 10 == 0:
            print(f"[wed] {k + 1}/{len(rows)}  {(time.time() - t0) / (k + 1):.2f}s/q", flush=True)
dt = time.time() - t0
print(f"[wed] done: {len(rows)} questions in {dt:.1f}s = {dt / len(rows):.3f}s/q = {1000 * dt / len(rows) / 3600:.2f} h per 1,000 q "
      f"(excl. model load); fallbacks={n_fb}", flush=True)
