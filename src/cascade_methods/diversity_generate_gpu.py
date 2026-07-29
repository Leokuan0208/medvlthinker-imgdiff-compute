#!/usr/bin/env python3
"""
diversity_generate_gpu.py - QUEUED GPU experiment for idea A1 (Diversity-maximized candidate set).

>>> DO NOT RUN NOW: GPUs are busy with the OmniMed reproduction. This script is a SPECIFICATION +
>>> runnable harness for later. It refuses to launch unless env DIVERSITY_GPU_OK=1 is set, so it
>>> cannot accidentally contend for the GPU.

WHY THIS EXISTS. The OFFLINE analysis (src/cascade_methods/diversity_candidates.py) shows the existing
8 iid temperature samples are ~42% redundant (only ~4.7 distinct answers among 8), and that diversity-
AWARE SELECTION over those SAME 8 reaches the correct answer in ~15% fewer samples -- but it CANNOT
raise oracle@8, because reordering a fixed set never adds a new answer. Raising the oracle CEILING (the
project's #1 binding limit) requires diverse GENERATION: candidates drawn to COVER the answer space
instead of piling on near-duplicates. This script generates that diverse pool and re-measures oracle@N.

WHAT IT DOES (3 phases):
  --phase generate : draw a DIVERSE pool of M candidates per question using a PROMPT PORTFOLIO x a
                     TEMPERATURE LADDER (instead of M iid samples at one prompt/temp), writing a ckpt
                     jsonl in the SAME schema as run_openvqa.py (idx, question, gold, preds[list], oks[list], ...)
                     so the EXISTING explode->judge pipeline applies unchanged.
  --phase select   : (offline, no GPU) DPP/MMR down-select N-of-M from the pool by answer-string
                     distance (token-Jaccard here; swap in sentence-transformer embeddings if desired).
  --phase measure  : (offline, no GPU) recompute oracle@N for  iid@8 (baseline dump)  vs  diverse-pool@M
                     vs  DPP-selected-N,  per dataset + pooled. Writes diversity_generate_result.json.

HYPOTHESIS (to be tested, NOT a result): the diverse pool RAISES oracle@N at fixed N (the accuracy
ceiling the verifier picks from), and DPP-select-N matches the pool's oracle@M with N<M (fewer, non-
redundant samples => both-axes-plausible). RISK: diverse prompts can inject confident-but-wrong
distractors that fool the verifier -- so --phase measure reports BOTH oracle (coverage) AND
verifier-bo-N selection accuracy, per the A1 backlog caveat.

Downstream after --phase generate (unchanged existing tools):
  python3 src/cascade_methods/explode_sc_for_judge.py --ckpt <this_ckpt>     # explode preds -> per-sample judge inputs
  HF_HOME=/data/dan/hf_cache python3 src/labeling/run_judge.py --preds <exploded>.jsonl --tp 2
  python3 src/cascade_methods/diversity_generate_gpu.py --phase measure ...  # recompute oracle@N vs iid@8

Launch (LATER, when GPUs free):
  DIVERSITY_GPU_OK=1 HF_HOME=/data/dan/hf_cache CUDA_VISIBLE_DEVICES=0 \
    python3 src/cascade_methods/diversity_generate_gpu.py --phase generate \
      --model_path /data/dan/weights/Lingshu-7B --tag lingshu7b_div --dataset vqa_rad_open \
      --ckpt_dir ckpts/openvqa/diverse --per_combo 1
"""
import argparse, os, sys, json, re, glob, string, math
from collections import Counter
import numpy as np

# --------------------------------------------------------------------------- DIVERSE GENERATION KNOBS
# Prompt portfolio: label-blind "perspectives" that push the model toward DIFFERENT parts of the answer
# space (coverage), not toward the same mode. Each is a (name, system, user_prefix) triple.
PROMPT_PORTFOLIO = [
    ("base",     "You are an expert medical image analyst. Answer the question with a short, specific phrase. Do not explain.", ""),
    ("anatomy",  "You are an expert medical image analyst. First fix the relevant ANATOMY, then answer with a short, specific phrase. Do not explain.", ""),
    ("modality", "You are an expert medical image analyst. First identify the imaging MODALITY and region, then answer with a short, specific phrase. Do not explain.", ""),
    ("differential", "You are an expert medical image analyst. Consider the most likely alternative before deciding. Answer with a short, specific phrase. Do not explain.", ""),
    ("concise",  "You are an expert radiologist. Give only the single most precise term or phrase. No hedging.", ""),
]
TEMP_LADDER = [0.7, 1.0, 1.3]   # a ladder, not a single temp -> spreads the sampling distribution
# M = len(PROMPT_PORTFOLIO) * len(TEMP_LADDER) * per_combo   (default 5*3*1 = 15 diverse candidates)

HIGH_PX, MIN_PX = 1280*28*28, 4*28*28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4, "cap160": 8, "cap80": 16}
REPO = os.path.expanduser("~/medvlthinker-imgdiff-compute"); J = lambda p: os.path.join(REPO, p)

# --------------------------------------------------------------------------- scoring (mirrors run_openvqa.py)
def norm_score(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(the|a|an|is|are|of|in|on|at|this|image|picture)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()
def score(pred, gold):
    p, g = norm_score(pred), norm_score(gold)
    if not p: return 0
    if p == g: return 1
    if g and (g in p.split() or p in g.split() or g in p or p in g): return 1
    return 0

# CANONICAL correctness scorer -- IDENTICAL to run_mcq_generate_verify.map_correct so the diverse
# pool's `oks` are directly comparable to the iid@8 baseline dump's `oks`/`sl` (no re-scoring drift).
def map_correct(pred, options, gold_text, gold_letter):
    if options and gold_letter:
        pn = norm_score(pred); best, bestj = -1, -1
        for j, o in enumerate(options):
            on = norm_score(re.sub(r"^\s*[A-E][:.\)]\s*", "", str(o)))
            sc = len(set(pn.split()) & set(on.split())) + (2 if on and on in pn else 0) + (3 if pn == on else 0)
            if re.match(rf"^\s*\(?{'ABCDE'[j]}\b", pred.strip(), re.I): sc += 3
            if sc > best: best, bestj = sc, j
        return int("ABCDE"[bestj] == gold_letter[:1])
    gn, pn = norm_score(gold_text), norm_score(pred)
    return int(bool(gn) and (pn == gn or gn in pn))

# --------------------------------------------------------------------------- answer-string distance (for DPP/MMR)
_PUNCT = re.compile(r"[^a-z0-9 ]+")
def toks(s): return set(_PUNCT.sub(" ", (s or "").lower()).split())
def jacc(a, b):
    a, b = toks(a), toks(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / len(a | b)
def farthest_first_select(preds, N):
    """DPP-MAP proxy: greedy farthest-first pick of N diverse answers from the pool (token-Jaccard
    distance). Seed = medoid (most central). Returns the selected INDICES. (Swap token-Jaccard for a
    sentence-transformer cosine here to use embeddings.)"""
    m = len(preds)
    if N >= m: return list(range(m))
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(i+1, m):
            D[i, j] = D[j, i] = 1.0 - jacc(preds[i], preds[j])
    sel = [int(np.argmin(D.sum(1)))]
    while len(sel) < N:
        rem = [i for i in range(m) if i not in sel]
        sel.append(max(rem, key=lambda i: min(D[i, j] for j in sel)))
    return sel

# --------------------------------------------------------------------------- item loading (mirrors run_openvqa.py)
def _gold_from_letter(ch, gl):
    gl = str(gl).strip()
    if ch and gl[:1] in "ABCDE":
        gi = "ABCDE".index(gl[:1])
        if gi < len(ch): return re.sub(r"^\s*[A-E][:.\)]\s*", "", str(ch[gi])).strip()
    return gl

def load_items(dataset, cap, limit):
    """Returns (items, metas). items = list of (idx, question, gold, img_ref). metas = {idx: {options, gold_letter}}
    populated only for MCQ-as-generation datasets (pmc_content); empty for the pure open sets."""
    from PIL import Image
    items = []; metas = {}
    if dataset == "slake_open":
        d = json.load(open("/data/dan/dataset/slake/test.json")); root = "/data/dan/dataset/slake/imgs"
        for x in d:
            if x.get("answer_type") != "OPEN" or x.get("q_lang") != "en": continue
            ip = os.path.join(root, x["img_name"])
            if os.path.exists(ip): items.append((x["qid"], x["question"], str(x["answer"]), ip))
    elif dataset == "radimagenet_open":
        for r in json.load(open("/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")):
            if os.path.exists(r["img_path"]): items.append((r["idx"], r["question"], r["answer"], r["img_path"]))
    elif dataset == "kvasir_open":
        for r in json.load(open("/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json")):
            if os.path.exists(r["img_path"]): items.append((r["idx"], r["question"], r["answer"], r["img_path"]))
    elif dataset == "pmc_content":
        # MedEvalKit PMC_VQA (mirror run_mcq_generate_verify PMC loader): options HIDDEN -> generate answer text
        import csv
        PMC_ROOT = "/data/dan/dataset/medevalkit/PMC-VQA"
        reader = csv.reader(open(os.path.join(PMC_ROOT, "test_2.csv"), encoding="utf-8")); next(reader)
        for row in reader:
            if len(row) < 10: continue
            index, figpath, _cap, question, cA, cB, cC, cD, answer, _split = row[:10]
            ch = [cA, cB, cC, cD]
            ip = os.path.join(PMC_ROOT, "figures", figpath)
            if not os.path.exists(ip): ip = os.path.join(PMC_ROOT, "images", figpath)
            if not os.path.exists(ip): continue
            gl = str(answer).strip()
            items.append((index, str(question).strip(), _gold_from_letter(ch, gl), ip))
            metas[index] = {"options": ch, "gold_letter": gl}
    else:
        import pandas as pd, io
        base = "/data/dan/dataset/vqa_rad/data" if dataset == "vqa_rad_open" else "/data/dan/dataset/path_vqa/data"
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(base, "test-*.parquet")))], ignore_index=True)
        for i, r in df.iterrows():
            q = r.get("question"); a = r.get("answer")
            if q is None and "conversations" in r:
                conv = r["conversations"]; q = conv[0]["value"].replace("<image>", "").strip(); a = conv[1]["value"]
            a = str(a).strip()
            if a.lower() in ("yes", "no"): continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                items.append((int(i), str(q), a, Image.open(io.BytesIO(img["bytes"])).convert("RGB")))
    return items[:limit], metas

# =========================================================================== PHASE: generate (GPU)
def phase_generate(A):
    if os.environ.get("DIVERSITY_GPU_OK") != "1":
        sys.exit("[REFUSED] GPUs are reserved. Set DIVERSITY_GPU_OK=1 to run diverse generation.")
    import math
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams
    MAXPX = HIGH_PX // CAP_DIV[A.cap]
    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)
    VERIFY_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether "
                  "the proposed answer is correct. Respond with only 'Yes' or 'No'.")

    def build(q, img, sysprompt, uprefix):
        im = [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX}]
        msgs = [{"role": "system", "content": sysprompt},
                {"role": "user", "content": im + [{"type": "text", "text": (uprefix + q) if uprefix else q}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs); req = {"prompt": text}
        if imgs: req["multi_modal_data"] = {"image": imgs}
        return req

    def build_verify(q, ans, img):
        body = f"Question: {q}\nProposed answer: {ans}\nIs the proposed answer correct? Answer Yes or No."
        msgs = [{"role": "system", "content": VERIFY_SYS},
                {"role": "user", "content": [{"type": "image", "image": img, "max_pixels": MAXPX, "min_pixels": MIN_PX},
                                             {"type": "text", "text": body}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs); req = {"prompt": text}
        if imgs: req["multi_modal_data"] = {"image": imgs}
        return req

    def tok_ids(words):
        ids = {}
        for w in words:
            for v in (w, " " + w):
                e = proc.tokenizer.encode(v, add_special_tokens=False)
                if len(e) == 1: ids[e[0]] = w
        return ids
    YES, NO = tok_ids(["Yes", "yes", "YES"]), tok_ids(["No", "no", "NO"])

    items, metas = load_items(A.dataset, A.cap, A.n)
    if A.restrict_dump and os.path.exists(A.restrict_dump):
        keep = set(l.strip() for l in open(A.restrict_dump) if l.strip())
        items = [it for it in items if str(it[0]) in keep]
        print(f"[generate] restrict to {len(keep)} idx from {A.restrict_dump} -> {len(items)} matched items", flush=True)
    use_verifier = bool(A.verifier_lora)
    M = len(PROMPT_PORTFOLIO) * len(TEMP_LADDER) * A.per_combo
    print(f"[generate] {A.dataset}: {len(items)} items | portfolio={len(PROMPT_PORTFOLIO)} x temps={TEMP_LADDER} "
          f"x per_combo={A.per_combo} => M={M}/q | verifier={'ON' if use_verifier else 'off'}", flush=True)

    lora_kw = dict(enable_lora=True, max_lora_rank=32) if use_verifier else {}
    llm = LLM(model=A.model_path, tensor_parallel_size=A.tp, dtype="bfloat16", gpu_memory_utilization=A.gpu_mem,
              max_model_len=A.max_model_len, limit_mm_per_prompt={"image": 4}, trust_remote_code=True, **lora_kw)
    VLORA = None
    if use_verifier:
        from vllm.lora.request import LoRARequest
        VLORA = LoRARequest("verifier", 1, os.path.expanduser(os.path.join(REPO, A.verifier_lora)))
    sp_ver = SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)

    os.makedirs(A.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for l in open(ckpt):
            if l.strip():
                try: done.add(json.loads(l)["idx"])
                except Exception: pass
    todo = [it for it in items if it[0] not in done]
    print(f"  {len(todo)} to run -> {ckpt}", flush=True)
    with open(ckpt, "a") as fh:
        CH = A.chunk
        for c0 in range(0, len(todo), CH):
            ch = todo[c0:c0+CH]
            # one (prompt x temp) combo per generate() call so each keeps its own SamplingParams;
            # accumulate the pool per question across combos
            pool = {idx: {"preds": [], "provenance": []} for (idx, _, _, _) in ch}
            for (pname, sysp, upre) in PROMPT_PORTFOLIO:
                reqs = [build(q, im, sysp, upre) for (_, q, _, im) in ch]
                for temp in TEMP_LADDER:
                    sp = SamplingParams(temperature=temp, max_tokens=A.max_tokens, n=A.per_combo)
                    outs = llm.generate(reqs, sp)
                    for (idx, q, gold, _), o in zip(ch, outs):
                        for cand in o.outputs:
                            pool[idx]["preds"].append(cand.text.strip())
                            pool[idx]["provenance"].append(f"{pname}@t{temp}")
            # pointwise verifier pass over ALL diverse candidates (mirror run_mcq_generate_verify)
            scores_by_idx = {idx: None for (idx, _, _, _) in ch}
            if use_verifier:
                vreqs = []; span = []
                for (idx, q, gold, im) in ch:
                    for p in pool[idx]["preds"]: vreqs.append(build_verify(q, p, im))
                    span.append((idx, len(pool[idx]["preds"])))
                vouts = llm.generate(vreqs, sp_ver, lora_request=VLORA)
                vi = 0
                for (idx, n_c) in span:
                    sc = []
                    for _ in range(n_c):
                        lps = (vouts[vi].outputs[0].logprobs or [{}])[0]; vi += 1
                        py = max((math.exp(v.logprob) for t, v in lps.items() if t in YES), default=0.0)
                        pn = max((math.exp(v.logprob) for t, v in lps.items() if t in NO), default=0.0)
                        sc.append(round(py / (py + pn), 5) if (py + pn) > 0 else 0.0)
                    scores_by_idx[idx] = sc
            for (idx, q, gold, _) in ch:
                preds = pool[idx]["preds"]
                mopt = metas.get(idx, {}).get("options"); mgl = metas.get(idx, {}).get("gold_letter")
                oks = [map_correct(p, mopt, gold, mgl) for p in preds]
                cnt = Counter(norm_score(p) for p in preds); modal_norm, _ = cnt.most_common(1)[0]
                modal_pred = next(p for p in preds if norm_score(p) == modal_norm)
                row = {"idx": idx, "question": q, "gold": gold, "preds": preds, "oks": oks,
                       "scores": scores_by_idx[idx], "provenance": pool[idx]["provenance"],
                       "modal_pred": modal_pred, "modal_ok": int(map_correct(modal_pred, mopt, gold, mgl)),
                       "n_distinct": len(cnt), "pool_size": len(preds), "diverse": True}
                if mopt is not None: row["options"] = mopt; row["gold_letter"] = mgl
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"   [{min(c0+CH,len(todo))}/{len(todo)}] running-oracle="
                  f"{np.mean([1 if any(json.loads(l)['oks']) else 0 for l in open(ckpt) if l.strip()]):.3f}", flush=True)
    print(f"[generate] DONE -> {ckpt}", flush=True)

# =========================================================================== PHASE: select (offline)
def phase_select(A):
    """Down-select N diverse candidates from the M-pool by answer-string distance (DPP-MAP proxy)."""
    ckpt = os.path.join(A.ckpt_dir, f"ckpt_{A.dataset}_{A.tag}.jsonl")
    rows = [json.loads(l) for l in open(ckpt) if l.strip()]
    out = ckpt.replace(".jsonl", f".dppN{A.select_n}.jsonl")
    with open(out, "w") as fh:
        for r in rows:
            sel = farthest_first_select(r["preds"], A.select_n)
            fh.write(json.dumps({**r, "preds": [r["preds"][i] for i in sel],
                                 "oks": [r["oks"][i] for i in sel],
                                 "selected_idx": sel}) + "\n")
    print(f"[select] DPP-{A.select_n} of M -> {out}")

# =========================================================================== PHASE: measure (offline)
def phase_measure(A):
    """Recompute oracle@N: diverse-pool@M vs DPP-selected-N vs the iid@8 baseline dump. Needs the
    per-sample judge labels (sl) from run_judge on the exploded diverse ckpt; falls back to exact-match
    'oks' if judge labels are absent (clearly flagged)."""
    def oracle_curve(list_of_ok):
        maxn = max(len(o) for o in list_of_ok)
        cur = []
        for k in range(1, maxn+1):
            cur.append(float(np.mean([1 if any(o[:k]) else 0 for o in list_of_ok if len(o) >= 1])))
        return cur
    res = {}
    for ds in A.datasets:
        div_ckpt = os.path.join(A.ckpt_dir, f"ckpt_{ds}_{A.tag}.jsonl")
        if not os.path.exists(div_ckpt):
            res[ds] = {"status": "diverse ckpt missing (run --phase generate first)"}; continue
        rows = [json.loads(l) for l in open(div_ckpt) if l.strip()]
        div_ok = [r["oks"] for r in rows]          # exact-match fallback; replace with judge sl when available
        # iid@8 baseline oracle from the existing transfer dump
        base_p = J(f"ckpts/train/lora_verifier_pooled4/transfer_dump_{ds}_{A.base_tag}.json")
        base_ok = None
        if os.path.exists(base_p):
            base = json.load(open(base_p)); base_ok = [[0 if x in (None, -1) else int(x) for x in r["sl"]] for r in base]
        res[ds] = {
            "n_diverse": len(rows), "pool_size": int(np.mean([len(o) for o in div_ok])),
            "oracle_diverse_pool": oracle_curve(div_ok),
            "oracle_iid8_baseline": (oracle_curve(base_ok) if base_ok else "baseline dump missing"),
            "note": "diverse oracle uses EXACT-MATCH oks; rerun after run_judge.py for judge-scored sl.",
        }
        print(f"  {ds:<16} diverse-pool oracle@M[-1]="
              f"{res[ds]['oracle_diverse_pool'][-1]:.3f}  vs iid@8 oracle@8="
              f"{(res[ds]['oracle_iid8_baseline'][-1] if base_ok else float('nan')):.3f}")
    outp = J("results/cascade_methods/artifacts/diversity_generate_result.json")
    json.dump(res, open(outp, "w"), indent=1)
    print(f"-> {outp}")

# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["generate", "select", "measure"])
    ap.add_argument("--model_path"); ap.add_argument("--tag", default="diverse")
    ap.add_argument("--base_tag", default="lingshu7b", help="iid baseline transfer_dump tag to compare against")
    ap.add_argument("--dataset", choices=["slake_open", "vqa_rad_open", "pathvqa_open", "kvasir_open", "radimagenet_open", "pmc_content"])
    ap.add_argument("--datasets", nargs="+", default=["vqa_rad_open", "pathvqa_open", "kvasir_open", "radimagenet_open"])
    ap.add_argument("--ckpt_dir", default="ckpts/openvqa/diverse")
    ap.add_argument("--per_combo", type=int, default=1, help="samples per (prompt x temp) combo")
    ap.add_argument("--select_n", type=int, default=8)
    ap.add_argument("--cap", choices=list(CAP_DIV), default="cap320"); ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_mem", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=8192); ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16, help="items per llm.generate call (memory cgroup -> keep small)")
    ap.add_argument("--verifier_lora", default=None, help="LoRA adapter dir for the pointwise verifier; when set, "
                    "each diverse candidate is ALSO scored P(correct) with it (mirrors run_mcq_generate_verify)")
    ap.add_argument("--restrict_dump", default=None, help="path to an iid ckpt jsonl; restrict generation to ITS idx "
                    "set so the diverse pool is on EXACTLY the same questions as the iid@8 baseline (matched comparison)")
    A = ap.parse_args()
    if A.phase == "generate": phase_generate(A)
    elif A.phase == "select": phase_select(A)
    else: phase_measure(A)
