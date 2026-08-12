#!/usr/bin/env python3
"""
unified_pipeline.py -- ATTACK 2 (2026-08-12): ONE pipeline for BOTH answer formats, without the
best-of-N luck floor.

THE MECHANISM (never tried in this repo).  Keep ONE trained scorer, ONE feature set and ONE decision
rule.  Let only the CANDIDATE SET differ, and derive it from THE PROMPT ITSELF, not from a
hand-written format branch:

    candidates(item) = ANSWER_SPACE(prompt)      if the prompt supplies a complete answer space
                     = SAMPLE_N(7B, item)        otherwise
    pick             = argmax_c  s(image, question, c)      # s = the single trained verifier
    answer           = pick                                  # ALWAYS returns an answer (rule 6 ok)

ANSWER_SPACE is read off the deployed MedEvalKit prompt string:
  * a lettered option list ("Options:\n A: ... B: ...")          -> the option BODIES
  * "Please output 'yes' or 'no'(no extra output)."              -> ["yes", "no"]
  * anything else (get_close_ended_prompt / open-ended)          -> no answer space -> sampled branch

WHY THIS AND NOT THE TWO OBVIOUS UNIFICATIONS (both already measured, both dead):
  (a) (choice)(why) -- MCQ as constrained open-text so the verifier sees a justification:
      sel_eff 0.7751 vs 0.7977 letter-only, -0.0226 [-0.0433,-0.0024] SIGNIFICANT LOSS
      [artifacts/choicewhy_measure_2026-08-03.json].  Not re-run here.
  (b) "sample 8 and verify" applied to MCQ: on PMC the verifier's pick (0.4325) is BELOW greedy
      (0.5060) and MedXpert's oracle@8 (0.5365) is BELOW its own random-gold luck floor (0.6808).
      Not re-run here.
  The candidate-set-from-the-prompt rule sidesteps both: for an MCQ item the candidate set is FIXED,
  COMPLETE and identical across arms, so "coverage" carries no information at all -- the gold answer
  is in the set with probability 1 and the random-pick floor is exactly 1/K.  Nothing can be won by
  luck; every point above 1/K is scorer signal.

WHAT IS BEING TESTED, in order of how much it decides:
  Q1 (decisive, 50% of the macro weight): on the four cells whose prompt supplies an answer space
     (PMC_VQA, MedXpertQA-MM, VQA_RAD_closed, PATH_VQA_closed), does argmax-verifier-over-options
     BEAT the 7B's own greedy argmax?  If it does not, the unified pipeline has no MCQ headroom and
     the honest answer is a quantified shortfall.
  Q2: does unifying COST anything on the open-text cells, i.e. does one scorer trained on both
     formats lose to the format-specific open-text verifier on its own format?
  Q3: what does the unified 7B-only pipeline reach on the 8-cell macro, against always-32B-direct
     0.6567, and what is the minimum strong-leg usage that closes the remainder?

STATISTICS / PROTOCOL (pre-registered in --prereg BEFORE any GPU forward pass):
  * null tests N1 (open-text incumbent sel_eff) and N2 (8-cell macro baselines) first, deviations
    stated;
  * pixel-md5 disjointness on DECODED RGB between every verifier-training image and every eval image
    of every cell scored -- reported per cell, NOT assumed;
  * luck floor: random-gold permutation control (NLUCK=1000) per cell, plus the analytic 1/K;
  * paired item bootstrap nboot=10000, common random numbers, macro CI = resample items within each
    cell and recompute the macro;
  * per-cell guardrail vs always-7B;
  * numerics pinned: TF32 OFF, OMP_NUM_THREADS=1, PYTHONHASHSEED=0, sorted item order, HF
    transformers only (NEVER vLLM for a visual LoRA: 0.775204 HF vs 0.702997 vLLM).

CPU only.  Launch from the repo root:
    python3 src/cascade_methods/unified_pipeline.py --prereg
    python3 src/cascade_methods/unified_pipeline.py --nulltest
    python3 src/cascade_methods/unified_pipeline.py --disjoint
    python3 src/cascade_methods/unified_pipeline.py --analyse
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
CKPT = os.path.join(ROOT, "ckpts/unified_pipeline")
DATA = os.path.join(ROOT, "data/unified_pipeline")
PARTS = os.path.join(ART, "_unified_pipeline_parts")

DATE = "2026-08-12"
PREREG = os.path.join(ART, f"unified_pipeline_{DATE}_preregistration.json")
OUT = os.path.join(ART, f"unified_pipeline_{DATE}.json")

# ---------------------------------------------------------------------------------------------
# pre-registered constants -- fixed BEFORE the run
# ---------------------------------------------------------------------------------------------
SEED_SUBSAMPLE = 20260810     # reuse mcq_tta's ALREADY pre-registered PMC subsample (same 6,000 ids)
PMC_SUBSAMPLE_N = 6000
SEED_BOOT = 20260812
SEED_LUCK = 20260812
SEED_TRAIN = 20260812
NBOOT = 10000
NLUCK = 1000

MACRO8 = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
          "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
#: cells whose DEPLOYED prompt supplies a complete answer space -> the options branch
OPTION_CELLS = ["PMC_VQA", "MedXpertQA-MM", "VQA_RAD_closed", "PATH_VQA_closed"]
#: cells whose prompt supplies none -> the sampled branch
SAMPLED_CELLS = ["SLAKE_closed", "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

VEC_NPZ = os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz")
DUMP_DIR_CLEAN = os.path.join(ROOT, "ckpts/train/lora_verifier_disjoint")

# published bars that the null tests must reproduce
PUBLISHED_MACRO = {"always_7b": 0.5971, "always_32b_direct": 0.6567,
                   "always_32b_reasoning": 0.5974, "oracle_mode_32b": 0.6573}
PUBLISHED_SELEFF = 0.775204

MAXPX, MINPX = 1280 * 28 * 28, 4 * 28 * 28
SYS_VERIF = ("You are a careful medical exam grader. Given a question and a proposed answer, decide "
             "whether the proposed answer is correct. Respond with only 'Yes' or 'No'.")

CHOICE_RE = re.compile(r"^(\s*)([A-Za-z])(\s*[:.)]\s*)(.*)$", re.S)


def _norm(s):
    return str(s).strip().lower()


# ===============================================================================================
# 1. the ONE candidate-set rule
# ===============================================================================================
def answer_space(item):
    """THE unification rule.  Return (candidate_texts, gold_index) if the item's DEPLOYED prompt
    supplies a complete answer space, else (None, None).

    There is no per-cell branch here -- the decision is a property of the prompt template that
    MedEvalKit already used for that item, which is carried on the item as `fmt`:
        fmt == 'mcq'    -> lettered option list  -> option bodies
        fmt == 'judge'  -> "Please output 'yes' or 'no'" -> ['yes','no']
        fmt == 'close'  -> single word/phrase, no answer space -> sampled branch
    """
    if item["fmt"] == "mcq":
        bodies, letters = [], []
        for c in item["choices"]:
            m = CHOICE_RE.match(str(c))
            assert m is not None, f"unparsed choice {c!r}"
            letters.append(m.group(2).upper())
            bodies.append(m.group(4).strip())
        gold = str(item["answer"]).strip().upper()
        gi = letters.index(gold) if gold in letters else None
        return bodies, gi
    if item["fmt"] == "judge":
        cands = ["yes", "no"]
        gi = cands.index(_norm(item["answer"])) if _norm(item["answer"]) in cands else None
        return cands, gi
    return None, None


# ===============================================================================================
# 2. work list
# ===============================================================================================
def pmc_subsample_ids():
    rng = np.random.default_rng(SEED_SUBSAMPLE)
    return sorted(rng.choice(33430, size=PMC_SUBSAMPLE_N, replace=False).tolist())


def build_worklist():
    """{cell: [ (i, question, [cand,...], gold_index, [img_paths], img_kind) ]} for the option cells."""
    cache = os.path.join(DATA, "worklist.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    import mcq_tta as M
    items = M.build_items()
    sub = set(pmc_subsample_ids())
    work = {}
    for cell in OPTION_CELLS:
        rows = []
        for it in items[cell]:
            if cell == "PMC_VQA" and it["i"] not in sub:
                continue
            cands, gi = answer_space(it)
            assert cands is not None, (cell, it["i"])
            rows.append(dict(cell=cell, i=it["i"], src=it["src"], question=it["question"],
                             cands=cands, gold=gi, images=it["images"], img_kind=it["img_kind"]))
        work[cell] = rows
    os.makedirs(DATA, exist_ok=True)
    json.dump(work, open(cache, "w"))
    return work


# ===============================================================================================
# 3. null tests
# ===============================================================================================
def n1_seleff():
    """N1 -- reproduce the open-text incumbent sel_eff 0.775204 from the clean transfer dumps."""
    from training_methods import genframe_data as G
    G.assert_disjoint()
    ev = G.load_candidates("eval", layers=[], pooling=())
    r = G.sel_eff({(q.ds, q.idx): q.inc_scores for q in ev.questions})
    got = float(r["sel_eff"])
    return {"published": PUBLISHED_SELEFF, "got": got, "max_abs_dev": abs(got - PUBLISHED_SELEFF),
            "n": int(r["n"]), "n_recoverable": int(r["n_recoverable"]),
            "per_ds": {k: float(v["sel_eff"]) for k, v in r["per_ds"].items()}}


def n2_macro():
    """N2 -- reproduce every published 8-cell macro baseline from the per-sample vectors."""
    z = np.load(VEC_NPZ, allow_pickle=True)
    out, dev = {}, 0.0
    for s, pub in PUBLISHED_MACRO.items():
        m = float(np.mean([z[f"{c}|{s}"].mean() for c in MACRO8]))
        out[s] = {"published": pub, "got": m, "abs_dev": abs(m - pub)}
        dev = max(dev, abs(m - pub))
    return {"per_system": out, "max_abs_dev": dev,
            "cell_n": {c: int(len(z[f"{c}|always_7b"])) for c in MACRO8}}


def n3_gold_recovery(work):
    """N3 -- the candidate set must CONTAIN the gold answer on 100% of option-cell items, and the
    7B's graded `correct` must equal (7B greedy letter == gold letter).  If the candidate set did
    not always contain the gold, 'coverage' would carry information and the luck floor would move."""
    z = np.load(VEC_NPZ, allow_pickle=True)
    out = {}
    for cell, rows in work.items():
        ok7 = z[f"{cell}|always_7b"]
        miss = sum(1 for r in rows if r["gold"] is None)
        out[cell] = {"n": len(rows), "gold_not_in_candidate_set": miss,
                     "coverage": 1.0 - miss / max(1, len(rows)),
                     "k_candidates": sorted({len(r["cands"]) for r in rows}),
                     "always_7b_on_scored_subset": float(np.mean([ok7[r["i"]] for r in rows]))}
    return out


# ===============================================================================================
# 4. disjointness (pixel md5 of DECODED RGB)
# ===============================================================================================
def pixhash(img):
    from PIL import Image
    if isinstance(img, str):
        img = Image.open(img)
    img = img.convert("RGB")
    h = hashlib.md5()
    h.update(f"{img.size[0]}x{img.size[1]}|".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def _hash_paths(paths, nproc=8):
    from multiprocessing import Pool
    with Pool(nproc) as p:
        return list(p.map(_safe_hash, paths, chunksize=32))


def _safe_hash(p):
    try:
        return pixhash(p)
    except Exception:
        return None


def train_image_hashes():
    """md5 of decoded RGB for every image the CLEAN verifier (lora_verifier_disjoint) could have
    seen, i.e. the L1 allowlists of its five training pools."""
    import glob
    import io
    from PIL import Image
    cache = os.path.join(DATA, "train_imghash.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    os.makedirs(DATA, exist_ok=True)
    allow = {ds: set(json.load(open(os.path.join(ROOT, "data/disjoint_split", f"idx_{ds}.json"))))
             for ds in ["slake_open_train", "vqa_rad_open_train", "pathvqa_open_train",
                        "kvasir_open", "radimagenet_open"]}
    out = {}
    # slake train
    paths = []
    for x in json.load(open("/data/dan/dataset/slake/train.json")):
        if x.get("answer_type") == "OPEN" and x.get("q_lang") == "en" and x["qid"] in allow["slake_open_train"]:
            ip = os.path.join("/data/dan/dataset/slake/imgs", x["img_name"])
            if os.path.exists(ip):
                paths.append(ip)
    out["slake_open_train"] = sorted(set(h for h in _hash_paths(sorted(set(paths))) if h))
    # parquet families
    import pandas as pd
    for ds, base in [("vqa_rad_open_train", "/data/dan/dataset/vqa_rad/data"),
                     ("pathvqa_open_train", "/data/dan/dataset/path_vqa/data")]:
        df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/train-*.parquet"))],
                       ignore_index=True)
        hs = set()
        for i, r in df.iterrows():
            if int(i) not in allow[ds]:
                continue
            img = r["image"]
            if isinstance(img, dict) and "bytes" in img:
                try:
                    hs.add(pixhash(Image.open(io.BytesIO(img["bytes"]))))
                except Exception:
                    pass
        out[ds] = sorted(hs)
    # json families
    for ds, jp in [("kvasir_open", "/data/dan/dataset/kvasir_vqa_x1/kvasir_open_1200.json"),
                   ("radimagenet_open", "/data/dan/dataset/radimagenet_vqa/radimagenet_open_2000.json")]:
        paths = sorted({r["img_path"] for r in json.load(open(jp))
                        if r["idx"] in allow[ds] and os.path.exists(r["img_path"])})
        out[ds] = sorted(set(h for h in _hash_paths(paths) if h))
    json.dump(out, open(cache, "w"))
    return out


def eval_image_hashes(work):
    """md5 of decoded RGB for every image actually scored, per cell."""
    from PIL import Image
    cache = os.path.join(DATA, "eval_imghash_mcq.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    os.makedirs(DATA, exist_ok=True)
    out = {}
    for cell, rows in work.items():
        hs = set()
        for r in rows:
            for p in r["images"]:
                try:
                    im = Image.open(p)
                    if r["img_kind"] == "rawrgb":
                        im = im.convert("RGB")
                    hs.add(pixhash(im))
                except Exception:
                    pass
        out[cell] = sorted(hs)
    json.dump(out, open(cache, "w"))
    return out


def disjointness_report(work=None):
    work = work or build_worklist()
    tr = train_image_hashes()
    ev = eval_image_hashes(work)
    trall = set().union(*[set(v) for v in tr.values()])
    rep = {"n_train_images": len(trall),
           "train_pool_sizes": {k: len(v) for k, v in tr.items()},
           "method": "md5 of DECODED RGB pixels (WxH + raw bytes); catches re-encoded copies",
           "per_cell": {}}
    for cell, hs in ev.items():
        inter = set(hs) & trall
        rep["per_cell"][cell] = {
            "n_eval_images": len(hs), "intersection": len(inter),
            "intersection_frac_of_eval_images": len(inter) / max(1, len(hs)),
            "by_train_pool": {k: len(set(hs) & set(v)) for k, v in tr.items() if set(hs) & set(v)},
            "clean_for_zero_shot_arm": len(inter) == 0}
    return rep


# ===============================================================================================
# 5. scoring / metrics
# ===============================================================================================
def load_scores(cell, tag):
    """Read the resumable per-(item,candidate) score JSONL written by unified_pipeline_score.py.
    Returns {i: np.array(scores)} in candidate order."""
    p = os.path.join(CKPT, f"{tag}_{cell}.jsonl")
    per = {}
    if not os.path.exists(p):
        return per
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("err"):
            continue
        per.setdefault(int(r["i"]), {})[int(r["c"])] = float(r["s"])
    return {i: d for i, d in per.items()}


def pick_vector(rows, scores, k_of):
    """Per-item delivered-ok for argmax-over-candidates.  Items with no score fall back to the
    7B's own greedy answer (the pipeline ALWAYS returns an answer)."""
    ok, picked, covered = [], [], []
    for r in rows:
        d = scores.get(r["i"])
        if d is None or len(d) != len(r["cands"]):
            ok.append(None); picked.append(None); covered.append(0); continue
        v = np.array([d[c] for c in range(len(r["cands"]))], float)
        p = int(np.argmax(v))
        ok.append(int(p == r["gold"])); picked.append(p); covered.append(1)
    return ok, picked, covered


def luck_floor(rows, scores, nluck=NLUCK, seed=SEED_LUCK):
    """RANDOM-GOLD control: re-draw each item's gold uniformly among its own candidates, keeping the
    verifier's scores fixed.  Under this null the pick accuracy is exactly 1/K in expectation, so it
    proves option COVERAGE carries no information in this mechanism."""
    rng = np.random.default_rng(seed)
    idx, ks = [], []
    for r in rows:
        d = scores.get(r["i"])
        if d is None or len(d) != len(r["cands"]):
            continue
        v = np.array([d[c] for c in range(len(r["cands"]))], float)
        idx.append(int(np.argmax(v))); ks.append(len(r["cands"]))
    if not idx:
        return None
    idx = np.array(idx); ks = np.array(ks)
    accs = np.empty(nluck)
    for b in range(nluck):
        g = (rng.random(len(ks)) * ks).astype(int)
        accs[b] = float((g == idx).mean())
    return {"n": int(len(idx)), "analytic_1_over_K": float(np.mean(1.0 / ks)),
            "permutation_mean": float(accs.mean()), "permutation_sd": float(accs.std(ddof=1)),
            "permutation_p95": float(np.percentile(accs, 95)),
            "permutation_max": float(accs.max())}


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    P = score[y == 1]; N = score[y == 0]
    if len(P) == 0 or len(N) == 0:
        return float("nan")
    a = np.concatenate([P, N]); o = a.argsort(); rk = np.empty(len(a)); rk[o] = np.arange(1, len(a) + 1)
    _, inv, c = np.unique(a, return_inverse=True, return_counts=True)
    ss = np.zeros(len(c)); np.add.at(ss, inv, rk); rk = (ss / c)[inv]
    return float((rk[:len(P)].sum() - len(P) * (len(P) + 1) / 2) / (len(P) * len(N)))


def paired_boot(a, b, nboot=NBOOT, seed=SEED_BOOT):
    """Paired item bootstrap on delta = mean(a) - mean(b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a); rng = np.random.default_rng(seed)
    d = a - b
    reps = np.empty(nboot)
    for i in range(nboot):
        s = rng.integers(0, n, n)
        reps[i] = d[s].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "sig": bool(lo > 0 or hi < 0)}


def macro_boot(cells_a, cells_b, nboot=NBOOT, seed=SEED_BOOT):
    """Macro CI: resample items WITHIN each cell (common random numbers across systems), recompute
    the equal-weight macro on the replicate."""
    rng = np.random.default_rng(seed)
    keys = list(cells_a)
    reps = np.empty(nboot)
    A = {k: np.asarray(cells_a[k], float) for k in keys}
    B = {k: np.asarray(cells_b[k], float) for k in keys}
    for i in range(nboot):
        acc = 0.0
        for k in keys:
            n = len(A[k]); s = rng.integers(0, n, n)
            acc += A[k][s].mean() - B[k][s].mean()
        reps[i] = acc / len(keys)
    d = float(np.mean([A[k].mean() - B[k].mean() for k in keys]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"delta": d, "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


# ===============================================================================================
# 5b. the deployed 7B's own pick, and the full analysis
# ===============================================================================================
LETTER_RE = re.compile(r"([A-Za-z])")


def sevenb_pick(cell, rows):
    """The DEPLOYED 7B greedy answer as a candidate INDEX, read from the MedEvalKit dump the
    published per-sample vector was built from.  Also returns the consistency check N5:
    dump 'correct' must equal (parsed pick == gold) on every item."""
    dsname = {"PMC_VQA": "PMC_VQA", "MedXpertQA-MM": "MedXpertQA-MM",
              "VQA_RAD_closed": "VQA_RAD", "PATH_VQA_closed": "PATH_VQA"}[cell]
    raw = json.load(open(f"{MEK}/eval_results_lingshu7b_full/{{}}/{dsname}/results.json"))
    picks, agree, n = {}, 0, 0
    for r in rows:
        d = raw[r["src"]]
        resp = str(d.get("response", "")).strip()
        if len(r["cands"]) == 2:                      # yes/no
            t = re.sub(r"[^a-z]", "", resp.lower())
            p = 0 if t.startswith("yes") else (1 if t.startswith("no") else None)
        else:
            m = LETTER_RE.search(resp)
            p = (ord(m.group(1).upper()) - 65) if m else None
            if p is not None and not (0 <= p < len(r["cands"])):
                p = None
        picks[r["i"]] = p
        gotok = int(bool(d.get("correct") is True or str(d.get("correct")).lower() == "true"))
        if p is not None:
            agree += int(gotok == int(p == r["gold"]))
            n += 1
    return picks, {"n_parsed": n, "n_items": len(rows),
                   "dump_correct_equals_parsed_pick_is_gold": agree / max(1, n)}


def cost_flopeq(cell, k_mean, esc=0.0):
    """FLOP-eq per query in units of ONE 7B forward (paper_baselines constants: 7B gen 1.0,
    verifier forward 1.0, 32B direct 4.57).  The option branch needs NO generation at all --
    the candidates come from the prompt -- so its cost is exactly k verifier forwards."""
    return k_mean * 1.0 + esc * 4.57


def analyse(tag="zeroshot"):
    z = np.load(VEC_NPZ, allow_pickle=True)
    work = build_worklist()
    res = {"tag": tag, "date": DATE, "cells": {}}

    macro_pipe, macro_7b, macro_32d = {}, {}, {}
    gate_conf, gate_ok, gate_32 = {}, {}, {}

    for cell in OPTION_CELLS:
        rows = work[cell]
        sc = load_scores(cell, tag)
        ok, picked, cov = pick_vector(rows, sc)
        picks7, n5 = sevenb_pick(cell, rows)
        ok7v = z[f"{cell}|always_7b"]
        ok32v = z[f"{cell}|always_32b_direct"]
        keep = [j for j in range(len(rows)) if cov[j]]
        if not keep:
            res["cells"][cell] = {"n_scored": 0, "status": "not measured -- no scores on disk"}
            continue
        idx = [rows[j]["i"] for j in keep]
        a_pipe = np.array([ok[j] for j in keep], float)
        a_7b = np.array([ok7v[i] for i in idx], float)
        a_32 = np.array([ok32v[i] for i in idx], float)
        # candidate-level AUROC of the verifier score (gold vs distractors)
        s_all, y_all, conf = [], [], []
        for j in keep:
            r = rows[j]; d = sc[r["i"]]
            v = np.array([d[c] for c in range(len(r["cands"]))], float)
            s_all += list(v); y_all += [int(c == r["gold"]) for c in range(len(r["cands"]))]
            sv = np.sort(v)[::-1]
            conf.append(float(sv[0] - sv[1]))
        agree = np.array([int(picks7[rows[j]["i"]] == picked[j]) for j in keep], float)
        res["cells"][cell] = {
            "n_items_in_cell": int(len(ok7v)),
            "n_scored": len(keep),
            "scored_is_full_cell": len(keep) == len(ok7v),
            "k_candidates_mean": float(np.mean([len(rows[j]["cands"]) for j in keep])),
            "acc_unified_pick": float(a_pipe.mean()),
            "acc_7b_greedy_same_items": float(a_7b.mean()),
            "acc_32b_direct_same_items": float(a_32.mean()),
            "delta_vs_7b": paired_boot(a_pipe, a_7b),
            "delta_vs_32b_direct": paired_boot(a_pipe, a_32),
            "guardrail_never_worse_than_7b": bool(a_pipe.mean() >= a_7b.mean()),
            "luck_floor_random_gold": luck_floor(rows, sc),
            "candidate_auroc_gold_vs_distractor": auroc(s_all, y_all),
            "agreement_with_7b_greedy": float(agree.mean()),
            "disagreement_stratum": {
                "n": int((1 - agree).sum()),
                "acc_unified": float(a_pipe[agree == 0].mean()) if (agree == 0).any() else None,
                "acc_7b": float(a_7b[agree == 0].mean()) if (agree == 0).any() else None,
                "acc_32b_direct": float(a_32[agree == 0].mean()) if (agree == 0).any() else None},
            "n5_dump_consistency": n5,
            "flopeq_per_query_vs_32b_direct_4p57": cost_flopeq(
                cell, float(np.mean([len(rows[j]["cands"]) for j in keep]))),
        }
        macro_pipe[cell] = a_pipe; macro_7b[cell] = a_7b; macro_32d[cell] = a_32
        gate_conf[cell] = np.array(conf, float); gate_ok[cell] = a_pipe; gate_32[cell] = a_32

    # ---- the other four cells: the SAME scorer, unchanged (open) or degenerate (SLAKE_closed) --
    for cell in SAMPLED_CELLS:
        v7 = z[f"{cell}|always_7b"].astype(float)
        v32 = z[f"{cell}|always_32b_direct"].astype(float)
        if cell == "SLAKE_closed":
            # its prompt supplies no answer space and no 8-sample pool exists -> the candidate set
            # DEGENERATES to {greedy}, i.e. the pipeline returns the 7B's own answer.  Not a win.
            vp = v7.copy()
            note = ("no candidate pool generated; 1-candidate set == 7B greedy. NOT a win, and the "
                    "cell is carried at the 7B floor.")
            cf = np.zeros(len(vp))
        else:
            dp = os.path.join(DUMP_DIR_CLEAN,
                              f"transfer_dump_{ {'SLAKE_open':'slake','VQA_RAD_open':'vqa_rad','PATH_VQA_open':'pathvqa'}[cell] }_open_lingshu7b.json")
            rows_o = json.load(open(dp))
            ok_o, cf_o = [], []
            for r in rows_o:
                s = np.array(r["scores"][:8], float)
                sl = [0 if x in (None, -1) else int(x) for x in r["sl"][:8]]
                p = int(np.argmax(s))
                ok_o.append(sl[p])
                sv = np.sort(s)[::-1]
                cf_o.append(float(sv[0] - sv[1]) if len(sv) > 1 else 0.0)
            vp = np.array(ok_o, float); cf = np.array(cf_o, float)
            note = ("the SAME adapter, the SAME argmax rule; only the candidate set is sampled "
                    "instead of prompt-given (incumbent best-of-8 arm)")
            assert len(vp) == len(v7), (cell, len(vp), len(v7))
        res["cells"][cell] = {
            "n_items_in_cell": int(len(v7)), "n_scored": int(len(vp)), "branch": "sampled",
            "acc_unified_pick": float(vp.mean()), "acc_7b_greedy_same_items": float(v7.mean()),
            "acc_32b_direct_same_items": float(v32.mean()),
            "delta_vs_7b": paired_boot(vp, v7), "delta_vs_32b_direct": paired_boot(vp, v32),
            "guardrail_never_worse_than_7b": bool(vp.mean() >= v7.mean()), "note": note}
        macro_pipe[cell] = vp; macro_7b[cell] = v7; macro_32d[cell] = v32
        gate_conf[cell] = cf; gate_ok[cell] = vp; gate_32[cell] = v32

    # ---- macro ------------------------------------------------------------------------------
    if len(macro_pipe) == 8:
        res["macro"] = {
            "unified_pipeline_7b_only": float(np.mean([macro_pipe[c].mean() for c in MACRO8])),
            "always_7b_on_same_items": float(np.mean([macro_7b[c].mean() for c in MACRO8])),
            "always_32b_direct_on_same_items": float(np.mean([macro_32d[c].mean() for c in MACRO8])),
            "vs_7b": macro_boot(macro_pipe, macro_7b),
            "vs_32b_direct": macro_boot(macro_pipe, macro_32d),
            "published_bars": {"always_7b": 0.5971, "always_32b_direct": 0.6567,
                               "gap_to_close": 0.0596},
            "per_cell_delta_vs_7b": {c: float(macro_pipe[c].mean() - macro_7b[c].mean())
                                     for c in MACRO8},
            "leave_one_cell_out_vs_32b_direct": {
                c: float(np.mean([macro_pipe[k].mean() - macro_32d[k].mean()
                                  for k in MACRO8 if k != c])) for c in MACRO8},
        }
        # ---- minimum strong-leg usage that closes the remainder --------------------------
        res["min_strong_leg"] = min_strong_leg(gate_conf, gate_ok, gate_32)
    else:
        res["macro"] = {"status": "incomplete -- cells missing: "
                                  + ",".join(c for c in MACRO8 if c not in macro_pipe)}
    return res


def min_strong_leg(conf, ok_pipe, ok32, grid=None):
    """The honest 'how much 32B do you still need' curve.  A SINGLE global escalation budget e is
    applied per cell by the pipeline's OWN confidence rank (top-1 minus top-2 candidate score);
    escalated items are served by 32B-direct.  Reported next to a RANDOM-gate control, because a
    confidence gate that does no better than random is not a gate."""
    grid = grid or [round(x, 3) for x in np.arange(0.0, 1.0001, 0.02)]
    rng = np.random.default_rng(SEED_BOOT)
    out = {"grid": [], "target_macro_always_32b_direct":
           float(np.mean([ok32[c].mean() for c in MACRO8]))}
    rank = {}
    for c in MACRO8:
        v = conf[c]
        rank[c] = np.argsort(np.argsort(v)) / max(1, len(v) - 1)   # 0 = least confident
    perm = {c: rng.permutation(len(conf[c])) / max(1, len(conf[c]) - 1) for c in MACRO8}
    for e in grid:
        accs, accr, cost = [], [], []
        for c in MACRO8:
            m = rank[c] < e
            accs.append(float(np.where(m, ok32[c], ok_pipe[c]).mean()))
            mr = perm[c] < e
            accr.append(float(np.where(mr, ok32[c], ok_pipe[c]).mean()))
        out["grid"].append({"escalation": e,
                            "macro_confidence_gate": float(np.mean(accs)),
                            "macro_random_gate": float(np.mean(accr))})
    tgt = out["target_macro_always_32b_direct"]
    hit = [g for g in out["grid"] if g["macro_confidence_gate"] >= tgt]
    out["min_escalation_to_match_always_32b_direct"] = hit[0]["escalation"] if hit else None
    hitr = [g for g in out["grid"] if g["macro_random_gate"] >= tgt]
    out["min_escalation_random_gate"] = hitr[0]["escalation"] if hitr else None
    return out


# ===============================================================================================
# 6. pre-registration
# ===============================================================================================
def write_prereg():
    os.makedirs(ART, exist_ok=True)
    p = {
        "date": DATE,
        "attack": "ATTACK 2 -- one pipeline for both formats, candidate set read off the prompt",
        "written_before_any_gpu_forward_pass": True,
        "mechanism": {
            "rule": "candidates = ANSWER_SPACE(prompt) if the prompt supplies one else SAMPLE_N(7B); "
                    "pick = argmax_c verifier(image, question, c); the scorer, its features and the "
                    "decision rule are IDENTICAL in both branches",
            "no_32B_at_test_time": True,
            "always_returns_an_answer": True,
            "not_re_run_because_already_measured_and_dead": [
                "(choice)(why) MCQ-as-constrained-open-text: -0.0226 sel_eff SIGNIFICANT LOSS "
                "[artifacts/choicewhy_measure_2026-08-03.json]",
                "sample-8-and-verify on MCQ: PMC verifier pick 0.4325 < greedy 0.5060; MedXpert "
                "oracle@8 0.5365 < its own luck floor 0.6808"],
        },
        "cells": {
            "option_branch": OPTION_CELLS,
            "sampled_branch": SAMPLED_CELLS,
            "PMC_VQA_subsample": {"n": PMC_SUBSAMPLE_N, "seed": SEED_SUBSAMPLE,
                                  "note": "the SAME 6,000 ids mcq_tta pre-registered on 2026-08-10"},
            "SLAKE_closed_handling": "its deployed prompt (get_close_ended_prompt) supplies NO answer "
                                     "space, so under the rule it falls in the sampled branch. No "
                                     "8-sample pool exists for it. If none is generated the unified "
                                     "pipeline DEGENERATES to a 1-candidate set there = 7B greedy "
                                     "(0.8254), reported as such, never as a win.",
        },
        "arms": {
            "A_zero_shot": "the EXISTING clean open-text verifier ckpts/train/lora_verifier_disjoint "
                           "scored over the option candidates -- no training, measures transfer",
            "B_unified_trained": "ONE LoRA verifier trained on BOTH branches' candidate sets from "
                                 "images disjoint from all 8 eval cells (pixel-md5 asserted); MCQ "
                                 "training labels are FREE (option == gold), no judge needed",
        },
        "primary_endpoint": "per-cell accuracy of argmax-verifier-over-candidates vs always-7B "
                            "greedy on the four option cells, paired item bootstrap nboot=10000",
        "secondary_endpoints": [
            "8-cell macro of the unified 7B-only pipeline vs always-32B-direct 0.6567",
            "does the unified scorer lose to the format-specific open-text verifier on open text "
            "(sel_eff 0.775204 bar)",
            "minimum strong-leg usage that closes the remaining macro gap",
        ],
        "controls": {
            "null_tests": ["N1 open-text incumbent sel_eff == 0.775204",
                           "N2 8-cell macro baselines == 0.5971/0.6567/0.5974/0.6573",
                           "N3 gold is in the candidate set on 100% of option-cell items"],
            "luck_floor": {"kind": "random-gold permutation, gold re-drawn uniformly among the item's "
                                   "own candidates with the verifier scores held fixed",
                           "nluck": NLUCK, "seed": SEED_LUCK,
                           "analytic_expectation": "exactly 1/K per item"},
            "disjointness": "pixel-md5 of DECODED RGB between every verifier-training image and every "
                            "scored eval image, reported PER CELL, never assumed",
            "guardrail": "per cell, never worse than always-7B; flags reported with n",
        },
        "numerics_pins": {"tf32": False, "OMP_NUM_THREADS": 1, "PYTHONHASHSEED": 0,
                          "serving": "HF transformers only -- vLLM drops all 192 visual.* LoRA "
                                     "modules (0.775204 HF vs 0.702997 vLLM)",
                          "max_pixels": MAXPX, "min_pixels": MINPX,
                          "item_order": "sorted, fixed"},
        "seeds": {"boot": SEED_BOOT, "luck": SEED_LUCK, "subsample": SEED_SUBSAMPLE,
                  "train": SEED_TRAIN},
        "nboot": NBOOT,
        "bars": {"always_7b_macro": 0.5971, "always_32b_direct_macro": 0.6567,
                 "gap_to_close_with_no_32B": 0.0596,
                 "per_cell_7b": {"PMC_VQA": 0.5427, "SLAKE_closed": 0.8254, "VQA_RAD_closed": 0.7809,
                                 "PATH_VQA_closed": 0.8409, "MedXpertQA-MM": 0.2615,
                                 "SLAKE_open": 0.7364, "VQA_RAD_open": 0.4650,
                                 "PATH_VQA_open": 0.3240},
                 "per_cell_32b_direct": {"PMC_VQA": 0.5518, "SLAKE_closed": 0.8589,
                                         "VQA_RAD_closed": 0.8526, "PATH_VQA_closed": 0.8891,
                                         "MedXpertQA-MM": 0.3065, "SLAKE_open": 0.8186,
                                         "VQA_RAD_open": 0.6000, "PATH_VQA_open": 0.3760}},
        "declared_failure_modes": [
            "if argmax-verifier-over-options is BELOW 7B greedy on the option cells, the unified "
            "pipeline has no MCQ headroom and that shortfall IS the reported result",
            "a per-cell win inside the random-gold permutation p95 is NOT a win",
            "any cell whose eval images intersect the verifier's training images is reported as "
            "CONTAMINATED and excluded from the headline",
        ],
    }
    json.dump(p, open(PREREG, "w"), indent=1)
    print(f"wrote {PREREG}")
    return p


# ===============================================================================================
# 7. CLI
# ===============================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--nulltest", action="store_true")
    ap.add_argument("--disjoint", action="store_true")
    ap.add_argument("--worklist", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--tag", default="zeroshot")
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)
    if a.prereg:
        write_prereg()
    if a.nulltest:
        r = {"N1_open_text_seleff": n1_seleff(), "N2_macro_baselines": n2_macro()}
        if a.worklist or True:
            w = build_worklist()
            r["N3_candidate_coverage"] = n3_gold_recovery(w)
        r["max_abs_deviation_overall"] = max(r["N1_open_text_seleff"]["max_abs_dev"],
                                             r["N2_macro_baselines"]["max_abs_dev"])
        json.dump(r, open(os.path.join(PARTS, "nulltests.json"), "w"), indent=1)
        print(json.dumps(r, indent=1)[:4000])
    if a.disjoint:
        rep = disjointness_report()
        json.dump(rep, open(os.path.join(PARTS, "disjointness.json"), "w"), indent=1)
        print(json.dumps(rep, indent=1))
    if a.analyse:
        r = analyse(a.tag)
        p = os.path.join(PARTS, f"analysis_{a.tag}.json")
        json.dump(r, open(p, "w"), indent=1)
        print(json.dumps(r, indent=1))
        print(f"\nwrote {p}")
    if a.worklist and not a.nulltest:
        w = build_worklist()
        for c, rows in w.items():
            print(c, len(rows), "forwards", sum(len(r["cands"]) for r in rows))


if __name__ == "__main__":
    main()
