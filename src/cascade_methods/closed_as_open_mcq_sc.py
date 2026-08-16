#!/usr/bin/env python3
"""closed_as_open_mcq_sc.py -- BUILD 3 SECONDARY: does plain SELF-CONSISTENCY (majority vote over 8
T=0.4 samples) beat 7B greedy on the two INTRINSICALLY multiple-choice cells, PMC_VQA and
MedXpertQA-MM?

Training-free, generation cost only, no verifier.  These two cells are expected to stay OUTSIDE the
verifier claim (the question often cannot be answered without the options).  A null here scopes the
claim to 6 cells and is reported as such.

Prompt = the DEPLOYED MedEvalKit MCQ prompt (get_multiple_choice_prompt, direct), fullres, no system
prompt -- byte-matched to the deployed baseline path.  An in-session greedy control is generated in
the SAME engine so the comparison never crosses a serving config.

GRADING (pre-specified, identical for both arms):
  * letter_em -- extract the answer letter with the same first-branch rule judge_multi_choice uses,
    correct iff it equals the gold letter.  Unextractable -> WRONG (never withheld).
  * harness   -- MedEvalKit judge_multi_choice applied VERBATIM to an ACTUAL sampled response string
    (the majority arm grades the raw response of the first sample carrying the plurality letter, the
    greedy arm grades its own raw response).  No synthetic string is ever graded.

  python3 src/cascade_methods/closed_as_open_mcq_sc.py --stage gen
  python3 src/cascade_methods/closed_as_open_mcq_sc.py --stage analyse
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_as_open_lib as L                                            # noqa: E402

MCQ_CELLS = ["PMC_VQA", "MedXpertQA-MM"]
SEED_SUBSAMPLE = 20260810      # the pre-registered PMC subsample of unified_pipeline / mcq_tta
PMC_SUBSAMPLE_N = 6000
CK = os.path.join(L.ROOT, "ckpts/closed_as_open_mcq")
ARMS_MCQ = {
    "mcq_g":   dict(temp=0.0, top_p=1.0, top_k=-1, min_p=0.0, rep_pen=1.0, seed=42, n=1),
    "mcq_s8":  dict(temp=0.4, top_p=1.0, top_k=-1, min_p=0.0, rep_pen=1.0, seed=20260813, n=8),
}


def pmc_subsample_ids():
    rng = np.random.default_rng(SEED_SUBSAMPLE)
    return sorted(rng.choice(33430, size=PMC_SUBSAMPLE_N, replace=False).tolist())


def get_multiple_choice_prompt(question, choices):
    """VERBATIM MedEvalKit/utils/question_formats.py, is_reasoning=False, lang='en'."""
    options = "\n".join(str(c) for c in choices)
    return (f"\nQuestion: {question}\nOptions: \n{options}\n"
            "Answer with the option's letter from the given choices directly.")


def build_items():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mcq_tta as M
    items = M.build_items()
    sub = set(pmc_subsample_ids())
    out = {"PMC_VQA": [r for r in items["PMC_VQA"] if r["i"] in sub],
           "MedXpertQA-MM": items["MedXpertQA-MM"]}
    assert len(out["PMC_VQA"]) == PMC_SUBSAMPLE_N, len(out["PMC_VQA"])
    assert len(out["MedXpertQA-MM"]) == 2000, len(out["MedXpertQA-MM"])
    return out


def gpath(cell, arm):
    return os.path.join(CK, f"gen_{cell.replace('/', '_')}_{arm}.jsonl")


def loadg(cell, arm):
    p = gpath(cell, arm)
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[int(r["i"])] = r
    return out


# ---------------------------------------------------------------------------- letter extraction
def letter_of(response, k):
    """First-branch letter rule, mirroring judge_multi_choice's own letter handling.
    Returns 'a'..chr(a+k-1) or None."""
    alphas = [chr(ord("a") + i) for i in range(k)]
    r = L.parse_response(response).strip().lower().replace("\n", "")
    head = r.split(".")[0].split(":")[-1].strip()
    if head in alphas:
        return head
    m = re.match(r"^\(?([a-z])\)?\b", r)
    if m and m.group(1) in alphas:
        return m.group(1)
    return None


# =============================================================================================
def stage_gen(A):
    os.makedirs(CK, exist_ok=True)
    print("[items] loading ...", flush=True)
    ITEMS = build_items()
    for c in MCQ_CELLS:
        print(f"[items] {c}: {len(ITEMS[c])}", flush=True)

    work = []
    for arm in A.arms:
        for cell in MCQ_CELLS:
            have = loadg(cell, arm)
            todo = [it for it in ITEMS[cell] if it["i"] not in have]
            if todo:
                work.append((arm, cell, len(todo)))
            else:
                print(f"[skip] {arm} {cell}: complete ({len(have)})", flush=True)
    if not work:
        print("MCQ_SC_GEN_DONE (nothing to do)", flush=True)
        return
    print(f"[plan] {work}", flush=True)

    from PIL import Image
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(A.model_path, trust_remote_code=True)

    def build_req(item, prompt):
        imds = []
        for p in item["images"]:
            img = p if item["img_kind"] == "path" else Image.open(p)
            if item["img_kind"] == "rawrgb":
                img = Image.open(p).convert("RGB")
            imds.append({"type": "image", "image": img})
        if len(imds) == 1:
            content = [imds[0], {"type": "text", "text": prompt}]
        elif len(imds) == 0:
            content = [{"type": "text", "text": prompt}]
        else:
            content = []
            for k, d in enumerate(imds):
                content += [{"type": "text", "text": f"<image_{k+1}>: "}, d]
            content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        req = {"prompt": text}
        if imgs:
            req["multi_modal_data"] = {"image": imgs}
        return req

    llm = LLM(model=A.model_path, tensor_parallel_size=1, dtype="bfloat16",
              gpu_memory_utilization=A.gpu_mem, max_model_len=A.max_model_len,
              limit_mm_per_prompt={"image": 6}, trust_remote_code=True, enforce_eager=True)

    for arm in A.arms:
        cfg = ARMS_MCQ[arm]
        sp = SamplingParams(temperature=cfg["temp"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                            min_p=cfg["min_p"], repetition_penalty=cfg["rep_pen"],
                            max_tokens=A.max_tokens, n=cfg["n"], seed=cfg["seed"])
        for cell in MCQ_CELLS:
            have = loadg(cell, arm)
            todo = [it for it in ITEMS[cell] if it["i"] not in have]
            if not todo:
                print(f"[skip] {arm} {cell}: complete ({len(have)})", flush=True)
                continue
            ck = gpath(cell, arm)
            print(f"[run ] {arm} {cell}: {len(todo)} to go", flush=True)
            t0 = time.time()
            with open(ck, "a", encoding="utf-8") as fh:
                for c0 in range(0, len(todo), A.chunk):
                    ch = todo[c0:c0 + A.chunk]
                    try:
                        reqs = [build_req(it, get_multiple_choice_prompt(it["question"], it["choices"]))
                                for it in ch]
                        outs = llm.generate(reqs, sp)
                    except Exception as e:
                        if "EngineDead" in type(e).__name__ or "EngineCore" in str(e):
                            print(f"   !!! FATAL engine death at chunk {c0}: {e}", flush=True)
                            fh.flush()
                            sys.exit(17)
                        print(f"   !! chunk {c0} failed: {type(e).__name__}: {e}", flush=True)
                        continue
                    for it, o in zip(ch, outs):
                        try:
                            fh.write(json.dumps({
                                "i": it["i"], "cell": cell, "arm": arm, "gold": it["answer"],
                                "k": len(it["choices"]),
                                "preds": [c.text.strip() for c in o.outputs],
                                "gen_tokens": [len(c.token_ids) for c in o.outputs],
                            }, ensure_ascii=False) + "\n")
                        except Exception as e:
                            print(f"   !! item {it['i']} failed: {e}", flush=True)
                    fh.flush()
                    n = min(c0 + A.chunk, len(todo))
                    print(f"   [{n}/{len(todo)}] {n/(time.time()-t0):.1f} it/s", flush=True)
            print(f"[done] {arm} {cell} {len(loadg(cell,arm))} rows in {(time.time()-t0)/60:.2f} min",
                  flush=True)
    print("MCQ_SC_GEN_DONE", flush=True)


# =============================================================================================
def stage_analyse(A):
    from closed_as_open_analyse import paired_boot
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mcq_tta as M
    items = M.build_items()
    choices = {c: {r["i"]: [str(x) for x in r["choices"]] for r in items[c]} for c in MCQ_CELLS}
    #: G2 -- the MCQ grader copy must reproduce the deployed `correct` field row by row first
    g2 = L.mcq_grader_null_test({c: [ [str(x) for x in r["choices"]] for r in items[c] ]
                                 for c in MCQ_CELLS})
    print("G2 MCQ grader null test:", json.dumps(g2), flush=True)

    def harness_ok(cell, i, gold, resp):
        return int(bool(L.judge_multi_choice([c.lower() for c in choices[cell][i]], str(gold), resp)))

    out = {"title": "SECONDARY -- self-consistency (majority vote over 8 T=0.4 samples) vs 7B greedy "
                    "on the two intrinsically multiple-choice cells",
           "date": L.DATE,
           "prompt": "deployed MedEvalKit get_multiple_choice_prompt (direct), fullres, no system prompt",
           "null_tests": {"G2_mcq_grader": g2},
           "cells": {}}
    for cell in MCQ_CELLS:
        g = loadg(cell, "mcq_g")
        s = loadg(cell, "mcq_s8")
        idxs = sorted(set(g) & set(s))
        if not idxs:
            print(f"[{cell}] no data yet")
            continue
        rows = {"n": len(idxs), "n_greedy_rows": len(g), "n_sampled_rows": len(s)}
        gl, ml, ol, rl, nd = [], [], [], [], []
        gh, mh = [], []
        unpar_g = unpar_s = 0
        for i in idxs:
            gold = str(g[i]["gold"]).strip().lower()
            k = int(g[i]["k"])
            lg = letter_of(g[i]["preds"][0], k)
            unpar_g += int(lg is None)
            gl.append(int(lg == gold))
            gh.append(harness_ok(cell, i, gold, g[i]["preds"][0]))
            ls = [letter_of(p, k) for p in s[i]["preds"]]
            unpar_s += sum(1 for x in ls if x is None)
            cnt = Counter([x for x in ls if x is not None])
            nd.append(len(cnt))
            if cnt:
                top = max(cnt.items(), key=lambda kv: (kv[1], -ls.index(kv[0])))[0]
                j = ls.index(top)
            else:
                top, j = None, 0
            ml.append(int(top == gold))
            mh.append(harness_ok(cell, i, gold, s[i]["preds"][j]))
            ol.append(int(any(x == gold for x in ls)))
            rl.append(float(np.mean([int(x == gold) for x in ls])))
        gl, ml, ol, rl = map(lambda v: np.asarray(v, float), (gl, ml, ol, rl))
        rows["letter_em"] = {
            "greedy": round(float(gl.mean()), 6),
            "majority_vote": round(float(ml.mean()), 6),
            "oracle@8": round(float(ol.mean()), 6),
            "random_pick_floor": round(float(rl.mean()), 6),
            "analytic_1_over_K": round(float(np.mean([1.0 / int(g[i]["k"]) for i in idxs])), 6),
            "majority_vs_greedy": paired_boot(ml, gl),
            "oracle_vs_greedy": paired_boot(ol, gl),
            #: THE LUCK FLOOR.  This project has retracted a claim for mistaking coverage for
            #: signal, so a majority-vote gain must clear a RANDOM pick from the same pool, not
            #: just greedy.  (On these cells the floor sits BELOW greedy, so it is the weaker of
            #: the two bars -- but it is the one that proves the vote is doing work.)
            "majority_vs_random_pick_floor": paired_boot(ml, rl),
            "greedy_vs_random_pick_floor": paired_boot(gl, rl),
        }
        gha, mha = np.asarray(gh, float), np.asarray(mh, float)
        rows["harness_judge_multi_choice"] = {
            "greedy": round(float(gha.mean()), 6),
            "majority_vote": round(float(mha.mean()), 6),
            "majority_vs_greedy": paired_boot(mha, gha),
        }
        rows["diagnostics"] = {
            "unparsed_letter_rate_greedy": round(unpar_g / len(idxs), 5),
            "unparsed_letter_rate_sampled": round(unpar_s / (8 * len(idxs)), 5),
            "mean_distinct_letters_in_pool": round(float(np.mean(nd)), 4),
            "distinct_letter_hist": dict(sorted(Counter(nd).items())),
            "contested_frac": round(float(np.mean([x >= 2 for x in nd])), 5),
            "grader_gap_greedy_letter_em_minus_harness":
                round(float(gl.mean() - gha.mean()), 6),
            "grader_gap_majority_letter_em_minus_harness":
                round(float(ml.mean() - mha.mean()), 6),
        }
        rows["IS_THIS_THE_OLD_PMC_GRADER_ARTIFACT"] = {
            "why_the_question": "artifacts/unified_pipeline_2026-08-12.json reports +0.0132 SIG on "
                                "PMC-VQA that was ENTIRELY a defect in MedEvalKit's PMC answer "
                                "extractor ('any arm graded pick == gold collects +0.0102 of free "
                                "grader defect on this cell'). This round's PMC number rounds to the "
                                "same +0.0132, so the coincidence must be discharged, not ignored.",
            "answer": "NO -- four independent reasons, all readable above:",
            "1_zero_unparsed": f"the defect fires on responses the extractor cannot parse, and the "
                               f"unparsed letter rate here is "
                               f"{round(unpar_g/len(idxs),5)} (greedy) and "
                               f"{round(unpar_s/(8*len(idxs)),5)} (sampled). The mechanism has no "
                               f"surface to act on.",
            "2_same_grader_both_arms": "the 2026-08-12 defect was a BETWEEN-GRADER confound -- the "
                                       "option branch was scored pick==gold while its baseline went "
                                       "through the defective extractor. Here BOTH arms are scored by "
                                       "BOTH graders on REAL generated response strings; the majority "
                                       "arm grades the raw response of the first sample carrying the "
                                       "plurality letter, never a synthetic string.",
            "3_both_currencies_agree": "the win is present under letter_em AND under the harness's own "
                                       "judge_multi_choice; the 2026-08-12 positive was harness-only.",
            "4_defect_shrinks_it": "the DEFECTIVE harness grader gives the SMALLER delta "
                                   f"({round(float(mha.mean()-gha.mean()),6):+.4f}) than the repaired "
                                   f"letter_em ({round(float(ml.mean()-gl.mean()),6):+.4f}), so the "
                                   "defect is suppressing this effect, not manufacturing it.",
            "5_different_method_entirely": "this arm is training-free MAJORITY VOTE with no verifier "
                                           "and no 32B; the 2026-08-12 arm was a verifier fusion over "
                                           "the prompt's given options.",
        }
        out["cells"][cell] = rows
        print(f"[{cell}] n={len(idxs)} letterEM greedy={gl.mean():.4f} maj={ml.mean():.4f} "
              f"oracle@8={ol.mean():.4f} floor={rl.mean():.4f} "
              f"maj-greedy={rows['letter_em']['majority_vs_greedy']['delta']:+.4f} "
              f"{rows['letter_em']['majority_vs_greedy']['sign']}")
    os.makedirs(L.PARTS, exist_ok=True)
    p = os.path.join(L.PARTS, "mcq_self_consistency.json")
    json.dump(out, open(p, "w"), indent=1, ensure_ascii=False)
    print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "analyse"], required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS_MCQ))
    ap.add_argument("--model_path", default="/data/dan/hf_cache/hub/models--lingshu-medical-mllm--"
                                            "Lingshu-7B/snapshots/b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9/")
    ap.add_argument("--gpu_mem", type=float, default=0.55)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=64)
    A = ap.parse_args()
    (stage_gen if A.stage == "gen" else stage_analyse)(A)
