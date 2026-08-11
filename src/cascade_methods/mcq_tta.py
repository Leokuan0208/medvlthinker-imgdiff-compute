#!/usr/bin/env python3
"""
mcq_tta.py -- ATTACK 2 of ROUND 5: spend the second forward pass on the 32B, not on the 7B.

THE CLAIM UNDER TEST.  On 4 of the 5 MCQ reporting cells the deployed method IS literally
always-32B-direct (it contributes exactly 0.0000).  Those 5 cells are 62.5% of the macro weight.
Every prior MCQ attempt tried to CHOOSE BETWEEN the 7B and the 32B and every one is dead.  This
attack instead asks whether the 32B's own MCQ answer is improvable by a second, INPUT-PERTURBED
forward pass -- answer-order debiasing (cyclic option permutation) and prompt-form / image-view
ensembling -- which no experiment in this repo has ever run on a reporting cell.

WHAT THIS MODULE IS.  The CPU half: item construction, prompt building (a byte-exact
re-implementation of MedEvalKit's own prompt templates -- MedEvalKit itself is read ONLY, never
imported for generation and never modified), the permutation algebra, the null tests N1/N2, the
aggregation rules, the cross-fit gate, the paired bootstrap and the macro integration.

    python3 src/cascade_methods/mcq_tta.py --prereg     # write the pre-registration (BEFORE the run)
    python3 src/cascade_methods/mcq_tta.py --nulltest   # N1 + N2 (no GPU)
    python3 src/cascade_methods/mcq_tta.py --analyse    # N3 + N4 + endpoints (after generation)

The GPU half is src/cascade_methods/mcq_tta_generate.py (vLLM, Lingshu-32B, tp=2).

NUMERICS PINS (stated because in this project they are larger than most real effects):
  TF32 off, OMP_NUM_THREADS=1, PYTHONHASHSEED=0, sorted item order everywhere, rank_avg never used
  here (no rank fusion in this attack).  vLLM decode path is byte-matched to the deployed baseline:
  temperature 0, top_p 0.0001, repetition_penalty 1, max_new_tokens 2048, seed 42, enforce_eager,
  limit_mm_per_prompt={"image":6}; the ONLY change is logprobs 5 -> 20 (needed for the per-option
  posterior; greedy argmax is unaffected, and N3 measures whether it was).
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
CKPT = os.path.join(ROOT, "ckpts/mcq_tta")
PREREG = os.path.join(ART, "mcq_tta_2026-08-10_preregistration.json")
OUT = os.path.join(ART, "mcq_tta_2026-08-10.json")

# ---------------------------------------------------------------------------------------------
# pre-registered constants -- fixed BEFORE the run, never touched afterwards
# ---------------------------------------------------------------------------------------------
DATE = "2026-08-10"
SEED_SUBSAMPLE = 20260810          # PMC_VQA subsample RNG
SEED_BOOT = 20260810               # paired item-level bootstrap
SEED_LUCK = 20260810               # N4 permutation-label shuffle
NBOOT = 10000
NLUCK = 1000
PMC_SUBSAMPLE_N = 6000
K = 4                              # views per item, every cell
CAP320_PX = 320 * 28 * 28          # 250880  (src/cascade_methods/pairwise_verifier_score.py:35)
CAP640_PX = 640 * 28 * 28          # 501760
MIN_PX = 4 * 28 * 28

# the bar and the sensitivity, from the pre-registered plan (verified in --nulltest)
BAR_MACRO_DELTA = 0.0029           # macro delta needed for a CI-clean win
BAR_SUM_MCQ = 0.0235               # equivalently, summed per-cell gain over the 5 MCQ cells

CELLS = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM"]
PERMUTABLE = {"PMC_VQA", "MedXpertQA-MM"}          # cells with an explicit A:/B:/... option list
MACRO8 = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
          "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]

# ---- the EXACT view definitions, frozen here before the run ----------------------------------
# PMC_VQA / MedXpertQA-MM : K=4 cyclic option rotations, shift k in {0,1,2,3}.  Rotation k puts the
#   option that was at slot j into slot (j+k) mod n_opt and RELABELS it with the letter of its new
#   slot; the gold letter moves with it.  k=0 is the identity view (the N3 control).
# VQA_RAD_closed / PATH_VQA_closed (yes/no, get_judgement_prompt) : no option list, so no
#   permutation.  K=4 = {identity, instruction-order swap, identity@cap320, identity@cap640}.
# SLAKE_closed (get_close_ended_prompt) : K=4 = {identity, paraphrase#2, identity@cap320,
#   identity@cap640}, with the zh templates for the 420 zh items.
VIEWS = {
    "PMC_VQA":         [{"kind": "rot", "shift": k, "cap": 0} for k in range(4)],
    "MedXpertQA-MM":   [{"kind": "rot", "shift": k, "cap": 0} for k in range(4)],
    # AMENDMENT 3 (2026-08-10, BEFORE any GPU forward pass).  The pre-registered absolute caps were
    # DEGENERATE: measured native resolutions (150-item stratified sample per cell) put 96.7% of
    # PATH_VQA and 72.7% of SLAKE items ALREADY BELOW cap640, so the cap640 view would have been a
    # byte-identical duplicate of the identity view on most of those cells -- pure cost, zero
    # information, and it would have inflated the "K=4" cost while giving a 2-view ensemble.  The
    # two image views are therefore defined RELATIVE to each item's own native pixel count (0.50x
    # and 0.25x, floored at MIN_PX), which is guaranteed non-degenerate on 100% of items.  The
    # instruction axis is unchanged.  The per-item resolved pixel budget is written to every
    # checkpoint row.
    "VQA_RAD_closed":  [{"kind": "id", "cap": 0}, {"kind": "swap", "cap": 0},
                        {"kind": "id", "scale": 0.50}, {"kind": "id", "scale": 0.25}],
    "PATH_VQA_closed": [{"kind": "id", "cap": 0}, {"kind": "swap", "cap": 0},
                        {"kind": "id", "scale": 0.50}, {"kind": "id", "scale": 0.25}],
    "SLAKE_closed":    [{"kind": "id", "cap": 0}, {"kind": "para2", "cap": 0},
                        {"kind": "id", "scale": 0.50}, {"kind": "id", "scale": 0.25}],
}


def resolve_cap(item, view):
    """Absolute max_pixels for this (item, view).  0 = no cap (native resolution).  A relative
    `scale` view is resolved against the item's OWN native pixel count, so it can never collapse
    into the identity view the way an absolute cap does on an already-small image."""
    if not view.get("scale"):
        return int(view.get("cap", 0) or 0)
    from PIL import Image
    px = 0
    for p in item["images"]:
        try:
            with Image.open(p) as im:
                px = max(px, im.size[0] * im.size[1])
        except Exception:
            pass
    if px <= 0:
        return 0
    return max(MIN_PX, int(px * float(view["scale"])))

# frozen instruction strings (identity strings are verbatim MedEvalKit; the alternates are ours)
INSTR = {
    "mcq":            "Answer with the option's letter from the given choices directly.",
    "judge_id":       "Please output 'yes' or 'no'(no extra output).",
    "judge_swap":     "Please output 'no' or 'yes'(no extra output).",
    "close_en_id":    "Answer the question using a single word or phrase.",
    "close_en_para2": "Respond using a single word or phrase.",
    "close_zh_id":    "\u8bf7\u7528\u4e00\u4e2a\u5355\u8bcd\u6216\u8005\u77ed\u8bed\u56de\u7b54\u8be5\u95ee\u9898\u3002",
    "close_zh_para2": "\u8bf7\u4ee5\u4e00\u4e2a\u5355\u8bcd\u6216\u8005\u77ed\u8bed\u4f5c\u7b54\u3002",
}

CHOICE_RE = re.compile(r"^(\s*)([A-Za-z])(\s*[:.)]\s*)(.*)$", re.S)


# ===============================================================================================
# 1. prompt builders -- byte-exact re-implementations of MedEvalKit/utils/question_formats.py
#    (verified against the stored `prompt` strings by N2; MedEvalKit is read-only)
# ===============================================================================================
def p_mcq(question, choices):
    options = "\n".join([str(c) for c in choices])
    # NOTE: "Options: " carries a TRAILING SPACE in MedEvalKit's template; N2 asserts byte-equality.
    return f"""
Question: {question}
Options: 
{options}""" + "\n" + INSTR["mcq"]


def p_judge(question, swap=False):
    return question + "\n" + (INSTR["judge_swap"] if swap else INSTR["judge_id"])


def p_close(question, lang="en", para2=False):
    key = f"close_{lang}_{'para2' if para2 else 'id'}"
    return question + "\n" + INSTR[key]


def rotate_choices(choices, shift):
    """Cyclic rotation by `shift`.  Returns (new_choice_strings, orig_of_slot).
    orig_of_slot[s] = index in the ORIGINAL option list of the option now occupying slot s."""
    n = len(choices)
    parts = []
    for c in choices:
        m = CHOICE_RE.match(str(c))
        assert m is not None and "".join(m.groups()) == str(c), f"unparsed choice {c!r}"
        parts.append(list(m.groups()))            # lead, letter, sep, body
    orig_of_slot = [(s - shift) % n for s in range(n)]
    out = []
    for s in range(n):
        lead, letter, sep, body = parts[orig_of_slot[s]]
        newletter = chr(ord("A") + s) if letter.isupper() else chr(ord("a") + s)
        out.append(lead + newletter + sep + body)
    return out, orig_of_slot


# ===============================================================================================
# 2. item construction (pure data; images referenced by path, materialised by the GPU driver)
# ===============================================================================================
def load_stored(cell_file):
    p = f"{MEK}/eval_results_lingshu32b_full/{{}}/{cell_file}/results.json"
    return json.load(open(p))


def build_items():
    """Return {cell: [item, ...]} in the EXACT index order the deployed vectors use.
    item = dict(i=row index within the cell's per-sample vector, src=row index in the dump,
                question, choices|None, answer, lang, images=[paths], img_kind)"""
    items = {}

    # ---- PMC_VQA : test_2.csv, all 33,430 rows, dump order == csv order -----------------------
    base = "/data/dan/dataset/medevalkit/PMC-VQA"
    rows = list(csv.reader(open(os.path.join(base, "test_2.csv"), encoding="utf-8")))[1:]
    it = []
    for i, r in enumerate(rows):
        _, fig, _cap, q, cA, cB, cC, cD, ans, _split = r
        it.append(dict(i=i, src=i, question=q, choices=[cA, cB, cC, cD], answer=ans, lang="en",
                       images=[os.path.join(base, "figures", fig)], img_kind="path", fmt="mcq"))
    items["PMC_VQA"] = it

    # ---- MedXpertQA-MM : HF dataset, all 2,000 rows ------------------------------------------
    from datasets import load_dataset
    dp = os.path.join(MEK, "datas/MedXpertQA")
    ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", cache_dir=dp)["test"]
    it = []
    for i, s in enumerate(ds):
        ch = [f"{o}. {l}" for o, l in s["options"].items()]
        imgs = [os.path.join(dp, "images", im) for im in (s.get("images") or [])]
        it.append(dict(i=i, src=i, question=s["question"], choices=ch, answer=s["label"], lang="en",
                       images=imgs, img_kind="path", fmt="mcq", id=s["id"]))
    items["MedXpertQA-MM"] = it

    # ---- SLAKE_closed : test.json rows with answer_type == CLOSED ------------------------------
    sb = os.path.join(MEK, "datas/SLAKE")
    datas = json.load(open(os.path.join(sb, "test.json")))
    it = []
    k = 0
    for src, d in enumerate(datas):
        if d["answer_type"] != "CLOSED":
            continue
        it.append(dict(i=k, src=src, question=d["question"], choices=None, answer=d["answer"],
                       lang=d["q_lang"], images=[os.path.join(sb, "imgs", d["img_name"])],
                       img_kind="path", fmt="close"))
        k += 1
    items["SLAKE_closed"] = it

    # ---- VQA_RAD_closed : HF rows with answer.lower() in {yes,no} ------------------------------
    cache = os.path.join(CKPT, "img/VQA_RAD")
    os.makedirs(cache, exist_ok=True)
    from datasets import load_dataset as _ld, Image as HFImage
    dsr = _ld("flaviagiammarino/vqa-rad", split="test").cast_column("image", HFImage(decode=False))
    it = []
    k = 0
    for src, s in enumerate(dsr):
        a = s["answer"].lower()
        if a not in ("yes", "no"):
            continue
        pth = os.path.join(cache, f"{src}.bin")
        if not os.path.exists(pth):
            with open(pth, "wb") as f:
                f.write(s["image"]["bytes"])
        it.append(dict(i=k, src=src, question=s["question"], choices=None, answer=a, lang="en",
                       images=[pth], img_kind="raw", fmt="judge"))
        k += 1
    items["VQA_RAD_closed"] = it

    # ---- PATH_VQA_closed : parquet rows with answer in {yes,no} --------------------------------
    import pandas as pd
    cache = os.path.join(CKPT, "img/PATH_VQA")
    os.makedirs(cache, exist_ok=True)
    it = []
    k = 0
    src = 0
    for sp in sorted(glob.glob("/data/dan/dataset/path_vqa/data/test-*.parquet")):
        df = pd.read_parquet(sp)
        for _, row in df.iterrows():
            if row["answer"] in ("yes", "no"):
                pth = os.path.join(cache, f"{src}.bin")
                if not os.path.exists(pth):
                    with open(pth, "wb") as f:
                        f.write(row["image"]["bytes"])
                it.append(dict(i=k, src=src, question=row["question"], choices=None,
                               answer=row["answer"], lang="en", images=[pth], img_kind="rawrgb",
                               fmt="judge"))
                k += 1
            src += 1
    items["PATH_VQA_closed"] = it

    for cell in PERMUTABLE:
        for r in items[cell]:
            r["malformed"] = is_malformed(r)
            # AMENDMENT 1: does the question text itself carry the lettered option list?
            _q, ok = rotate_question(r["question"], r["choices"], 0)
            r["inline_options"] = bool(ok) and (_q == r["question"])
            if ok and _q != r["question"]:
                r["malformed"] = True          # identity rewrite not byte-exact -> K=1, never silent
            # a question that carries the list but whose span we cannot parse must not be permuted
            if (not ok) and inline_span(r["choices"]) in r["question"]:
                r["malformed"] = True
    for cell in CELLS:
        for r in items[cell]:
            r.setdefault("malformed", False)
            r.setdefault("inline_options", False)
    return items


def inline_span(choices):
    """The canonical '(A) body1 (B) body2 ...' span MedXpertQA-MM appends to every question."""
    parts = []
    for c in choices:
        m = CHOICE_RE.match(str(c))
        parts.append(f"({m.group(2)}) {m.group(4)}")
    return " ".join(parts)


def rotate_question(question, choices, shift):
    """PRE-REGISTRATION AMENDMENT 1 (2026-08-10, made BEFORE any GPU forward pass).

    MedXpertQA-MM embeds the FULL lettered option list inside the question text itself -- verified
    on disk: 2,000/2,000 questions end with the exact canonical span '(A) b1 (B) b2 ... (E) b5'.
    Permuting only the `Options:` block would therefore produce a SELF-CONTRADICTORY prompt on
    100% of that cell (the question would still letter the options the old way), and any measured
    effect would be an artifact of the contradiction, not answer-position debiasing.  The rotation
    is applied CONSISTENTLY to both places.  Returns (new_question, ok) -- ok is False if the
    canonical span is not present, in which case the caller must fall back to K=1 for that item.
    Identity (shift=0) is byte-identical to the input, which N2 asserts.
    PMC_VQA does not need this (6/33,430 = 0.018% of its questions contain all option bodies)."""
    span = inline_span(choices)
    if not question.endswith(span):
        return question, False
    rot, _ = rotate_choices(choices, shift)
    return question[:-len(span)] + inline_span(rot), True


def is_malformed(item):
    """True iff the SOURCE data's own option labels are not A,B,C,... in order (5 rows of PMC-VQA
    test_2.csv carry duplicated / out-of-order letters, e.g. row 6237 labels its options A,B,C,C).
    For such rows the identity rotation is not the identity string, so a permuted view would not be
    comparable to the deployed baseline.  PRE-REGISTERED HANDLING, fixed before generation: these
    items get K=1 (the identity view only) in every arm, i.e. the ensemble decision IS the baseline
    decision.  This can only remove headroom, never manufacture it, and it is 5/33,430 = 0.015%."""
    if item.get("choices") is None:
        return False
    ch, _ = rotate_choices(item["choices"], 0)
    return ch != [str(c) for c in item["choices"]]


def view_prompt(item, view):
    """Return (prompt_string, orig_of_slot|None) for `item` under `view`."""
    if view["kind"] == "rot" and item.get("malformed"):
        # malformed source labels: every view is the VERBATIM original prompt (K=1 in effect), so
        # the identity view stays byte-exact against the deployed baseline on 100% of rows.
        n = len(item["choices"])
        return p_mcq(item["question"], item["choices"]), list(range(n))
    if view["kind"] == "rot":
        ch, oos = rotate_choices(item["choices"], view["shift"])
        q = item["question"]
        if item.get("inline_options"):
            q, ok = rotate_question(q, item["choices"], view["shift"])
            assert ok, f"inline span vanished on item {item.get('i')}"
        return p_mcq(q, ch), oos
    if item["fmt"] == "judge":
        return p_judge(item["question"], swap=(view["kind"] == "swap")), None
    if item["fmt"] == "close":
        return p_close(item["question"], item["lang"], para2=(view["kind"] == "para2")), None
    if item["fmt"] == "mcq":
        return p_mcq(item["question"], item["choices"]), list(range(len(item["choices"])))
    raise ValueError(item["fmt"])


def pmc_subsample_ids():
    rng = np.random.default_rng(SEED_SUBSAMPLE)
    return sorted(rng.choice(33430, size=PMC_SUBSAMPLE_N, replace=False).tolist())


# ===============================================================================================
# 3. null tests
# ===============================================================================================
def n1_macro():
    """N1: reproduce the 8-cell macro from vec_disjoint.npz vs the published JSON.  Must be 0.0."""
    z = np.load(os.path.join(ART, "_selector_rerun_parts/vec_disjoint.npz"), allow_pickle=True)
    d = json.load(open(os.path.join(ART, "cascade_selector_rerun_2026-08-05.json")))
    pa = d["per_arm"]["disjoint"]
    arms = list(pa["macro_acc"].keys())
    dev = 0.0
    rec = {}
    for a in arms:
        per = [float(z[f"{c}|{a}"].mean()) for c in MACRO8]
        macro = float(np.mean(per))
        dev = max(dev, abs(round(macro, 4) - pa["macro_acc"][a]))
        for c, p in zip(MACRO8, per):
            dev = max(dev, abs(round(p, 4) - pa["per_cell_acc"][c][a]))
        rec[a] = dict(macro_recomputed=macro, macro_published=pa["macro_acc"][a],
                      per_cell={c: p for c, p in zip(MACRO8, per)})
    return dict(max_abs_deviation=dev, passed=(dev == 0.0), arms=rec,
                always_32b_direct_macro=float(np.mean([z[f"{c}|always_32b_direct"].mean()
                                                       for c in MACRO8])))


def n2_prompt_reconstruction():
    """N2: rebuild the identity prompt with OUR builder and assert byte-identical equality
    against the stored `prompt` on 100% of rows, for every cell that stores one.  MedXpertQA-MM's
    dump does NOT store `prompt` (schema verified on disk) -- for that cell we assert byte-exact
    equality of (id, question, choices, answer), which fully determines the prompt through the
    unmodified template, and we say so instead of claiming a prompt match we cannot make."""
    res = {}
    items = build_items()

    # PMC_VQA
    st = load_stored("PMC_VQA")
    ok = sum(1 for r, s in zip(items["PMC_VQA"], st)
             if p_mcq(r["question"], r["choices"]) == s["prompt"]
             and r["choices"] == s["choices"] and r["answer"] == s["answer"])
    res["PMC_VQA"] = dict(n=len(st), match=ok, rate=ok / len(st), kind="prompt string byte-equality")

    # SLAKE (closed subset; the dump holds all 2,094 rows)
    st = load_stored("SLAKE")
    stc = [s for s in st if s["answer_type"] == "CLOSED"]
    ok = sum(1 for r, s in zip(items["SLAKE_closed"], stc)
             if p_close(r["question"], r["lang"]) == s["prompt"] and r["answer"] == s["answer"]
             and r["lang"] == s["lang"])
    res["SLAKE_closed"] = dict(n=len(stc), match=ok, rate=ok / len(stc),
                               kind="prompt string byte-equality")

    # VQA_RAD / PATH_VQA: the dumps do NOT store `prompt`; assert question+answer byte-equality,
    # which determines the prompt through the unmodified get_judgement_prompt template.
    for cell, tag, pred in [("VQA_RAD_closed", "VQA_RAD", lambda s: s["answer"].lower() in ("yes", "no")),
                            ("PATH_VQA_closed", "PATH_VQA", lambda s: s["answer"] in ("yes", "no"))]:
        st = [s for s in load_stored(tag) if pred(s)]
        ok = sum(1 for r, s in zip(items[cell], st)
                 if r["question"] == s["question"] and r["answer"] == str(s["answer"]).lower())
        res[cell] = dict(n=len(st), match=ok, rate=ok / len(st),
                         kind="(question,answer) byte-equality; dump stores no prompt string")

    # MedXpertQA-MM
    st = load_stored("MedXpertQA-MM")
    ok = sum(1 for r, s in zip(items["MedXpertQA-MM"], st)
             if r["id"] == s["id"] and r["question"] == s["question"]
             and r["choices"] == s["choices"] and r["answer"] == s["answer"])
    res["MedXpertQA-MM"] = dict(n=len(st), match=ok, rate=ok / len(st),
                                kind="(id,question,choices,answer) byte-equality; dump stores no prompt string")

    # rotation algebra: identity rotation must reproduce every choice string byte-for-byte, EXCEPT
    # on rows whose SOURCE option labels are themselves malformed (see is_malformed).
    bad = {}
    tot = 0
    for cell in PERMUTABLE:
        bad[cell] = [r["i"] for r in items[cell] if r["malformed"]]
        tot += len(items[cell])
    res["rotation_identity_is_identity"] = dict(
        items_checked=tot,
        malformed_source_rows={c: v for c, v in bad.items()},
        n_malformed=sum(len(v) for v in bad.values()),
        handling="malformed rows get K=1 (identity only) in every arm; pre-registered before "
                 "generation; 5/33430 = 0.015% of PMC_VQA, 0 elsewhere",
        identity_is_identity_on_wellformed=True)
    res["all_passed"] = all(res[c]["rate"] == 1.0 for c in CELLS)
    return res


# ===============================================================================================
# 4. pre-registration writer
# ===============================================================================================
def write_prereg():
    ids = pmc_subsample_ids()
    doc = dict(
        title="ATTACK 2 (MCQ-TTA) -- PRE-REGISTRATION, written before any generation",
        date=DATE,
        attack="Answer-order / prompt-form / image-view test-time augmentation on Lingshu-32B, on the "
               "five MCQ/closed reporting cells (62.5% of the macro weight), gated adaptively.",
        target="MACRO over 8 cells, Variant B, vs always-32B-direct = 0.6567 "
               "(artifacts/cascade_selector_rerun_2026-08-05.json, arm 'disjoint'). "
               f"A CI-clean win needs macro delta >= {BAR_MACRO_DELTA}, i.e. summed per-cell gain "
               f"over the 5 MCQ cells >= {BAR_SUM_MCQ}.",
        prompt_geometry_verified_on_disk={
            "PMC_VQA": "get_multiple_choice_prompt, 4 options -> FULL cyclic permutation applies",
            "MedXpertQA-MM": "get_multiple_choice_prompt, 5 options (2000/2000 rows) -> cyclic permutation applies",
            "VQA_RAD_closed": "get_judgement_prompt (yes/no, NO option list) -> permutation does NOT apply",
            "PATH_VQA_closed": "get_judgement_prompt (yes/no, NO option list) -> permutation does NOT apply",
            "SLAKE_closed": "get_close_ended_prompt (single word/phrase; 420/836 zh) -> permutation does NOT apply",
            "reach_note": "the strong (permutation) lever reaches 2/8 = 25% of macro weight at full "
                          "strength; the other 3/8 get the weaker template/view-only lever. This is "
                          "stated up front so '62.5% of weight' is never read as 'the strong lever "
                          "reaches all of it'.",
        },
        views=VIEWS, instruction_strings=INSTR,
        cap_pixels=dict(cap320=CAP320_PX, cap640=CAP640_PX, min_pixels=MIN_PX,
                        provenance="src/cascade_methods/pairwise_verifier_score.py:35"),
        aggregation=dict(
            primary="mean over the K views of the un-permuted per-option FIRST-TOKEN log-probs "
                    "(renormalised over the option set within each view), argmax. Applies to "
                    "PMC_VQA / MedXpertQA-MM (letters) and to VQA_RAD_closed / PATH_VQA_closed "
                    "({yes,no}).",
            slake_closed="SLAKE_closed has no fixed option set (free single word/phrase, 2 languages), "
                         "so its PRE-REGISTERED primary is MAJORITY VOTE over normalised generated "
                         "strings, ties broken by highest mean cumulative logprob.",
            secondary="majority vote over un-permuted decisions, ties broken by mean confidence.",
            missing_letter_floor="if a letter is absent from the top-20 first-token logprobs, it is "
                                 "assigned min(observed logprob) - 1.0; the coverage rate is reported.",
            grading="every arm is graded by MedEvalKit's own grader, called read-only: "
                    "judge_multi_choice / judge_judgement / judge_close_end_vqa. The ensemble's "
                    "decision is rendered as the response string the grader would see for a clean "
                    "answer, so baseline and ensemble go through the identical function.",
        ),
        arms=["always_32b_direct (K=1, the deployed baseline and the N3 identity control)",
              "TTA always-K (K=4, every item) -- EVAL-VISIBLE UPPER BOUND, labelled DIAGNOSTIC",
              "TTA gated (K>1 only where the 32B's stored margin is below a 5-fold cross-fit "
              "threshold, fold labels held out) -- THE DEPLOYABLE POLICY",
              "always_7b (from vec_disjoint.npz)",
              "deployed method accuracy-max (from vec_disjoint.npz)",
              "equal-K temperature-sampling arm at T=0.7, same K, same cost -- STAGE B"],
        staging=dict(
            stage_A="the K=4 view arms (identity view = the N3 control).",
            stage_B="the equal-K temperature control. PRE-REGISTERED TRIGGER: Stage B is run if and "
                    "only if Stage A's always-K (eval-visible upper-bound) summed MCQ gain is "
                    f">= {BAR_SUM_MCQ}. If it is below, KILL(ii) has already fired and the "
                    "temperature control cannot change the verdict.",
            pmc_extension="PMC_VQA runs a pre-registered random subsample of n=6000 (ids below). "
                          "It is extended to the full 33,430 if and only if the 6,000-item delta "
                          "point estimate is >= +0.01. Two-stage design, not a re-cut.",
        ),
        null_tests=dict(
            N1="reproduce the 8-cell macro from _selector_rerun_parts/vec_disjoint.npz against "
               "cascade_selector_rerun_2026-08-05.json. Max abs deviation must be 0.0.",
            N2="rebuild the identity prompt with our own builder and assert byte-identical equality "
               "on 100% of rows for every cell that stores a prompt string; for the two cells whose "
               "dumps store no prompt, assert byte-equality of the fields that determine it. "
               "Below 100% anywhere, STOP.",
            N3="re-run the 32B at the identity view with our own runner and reproduce the stored "
               "per-cell accuracies (.5518 / .8589 / .8526 / .8891 / .3065) on the same ids. "
               "ABORT above 0.005 abs deviation on any cell.",
            N4="LUCK-FLOOR CONTROL: permutation-label shuffle -- recombine the K arms after randomly "
               "re-assigning which un-permutation map is applied, 1000 draws. The real ensemble must "
               "beat this null with a CI excluding zero. Applies to PMC_VQA and MedXpertQA-MM only "
               "(the other three cells have no permutation map; for them the discriminating control "
               "is the equal-K temperature arm).",
        ),
        endpoint_primary="macro over 8 cells vs always-32B-direct = 0.6567; paired item-level "
                         f"bootstrap, nboot={NBOOT}, seed={SEED_BOOT}.",
        endpoint_secondary=["per-cell delta + guardrail (never worse on any single benchmark)",
                            "the K-vs-accuracy curve",
                            "measured FLOP-eq / latency / energy of the K-view policy, including the "
                            "measured prefix-cache saving, batch-1 NVML, n>=20 items x 2 replicates"],
        success="summed MCQ-cell gain >= +0.0235 (=> macro +0.0029) with CI excluding zero, from the "
                "GATED CROSS-FIT policy (not the always-K oracle-tuned one), AND N4 rejected, AND "
                "the equal-K temperature arm does not match it.",
        kill=["(i) N4 not rejected",
              "(ii) the summed gain of the always-K (eval-visible upper bound) policy is < +0.0235 "
              "-- the gated policy cannot reach it either, so stop",
              "(iii) measured cost of K=4 exceeds 2.0x a single 32B pass -- report 'accuracy is "
              "buyable on MCQ but only at >=2x the strong model'"],
        not_a_rediscovery_of=[
            "artifacts/logit_fusion.json -- fused TWO MODELS' posteriors on internal-track n=170-500 "
            "subsamples at CI +-0.03; this is SINGLE-MODEL TTA on the MedEvalKit-track cells at full n",
            "single-model routing (killed ~29 sigma below the random-allocation floor) -- that negative "
            "is about a ROUTER's inability to PICK; an AVERAGE over configurations needs no router. "
            "The N4 luck-floor control is run anyway.",
            "MCQ best-of-N / temperature self-consistency (killed by the luck floor) -- permutation TTA "
            "changes the INPUT, not the sampling seed. The equal-K temperature arm tests exactly this.",
            "method_final_mmmu_corrected.py's cyclic permutation -- an n=150 contamination AUDIT on "
            "MMMU, the EXCLUDED cell; never a method, never a reporting cell",
            "the certified veto (beat32b_more.f8_veto) -- veto rate 0.0000 on 4 of 5 MCQ cells",
        ],
        forbidden="CRITICAL RULE 6: nothing abstention-shaped. Every arm returns an answer on every "
                  "item; the gate chooses HOW MANY forward passes to spend, never whether to answer.",
        numerics_pins=dict(tf32=False, omp_num_threads=1, pythonhashseed=0, item_order="sorted",
                           vllm="temperature 0, top_p 0.0001, repetition_penalty 1, max_new_tokens 2048, "
                                "seed 42, enforce_eager=True, limit_mm_per_prompt image=6, tp=2; "
                                "logprobs 5 -> 20 is the ONLY change vs the deployed baseline path",
                           prompt_persistence="the FULL prompt string is written on every checkpoint row "
                                              "(this repo's Finding-1 failure was caused by prompts living "
                                              "only in shell variables)"),
        seeds=dict(subsample=SEED_SUBSAMPLE, bootstrap=SEED_BOOT, luck_floor=SEED_LUCK,
                   vllm=42, nboot=NBOOT, nluck=NLUCK),
        pmc_subsample_n=PMC_SUBSAMPLE_N,
        pmc_subsample_ids=ids,
        amendments=[
            dict(id=1, when="2026-08-10, BEFORE any GPU forward pass, during the CPU smoke test",
                 finding="MedXpertQA-MM embeds the FULL lettered option list inside the question "
                         "text: 2,000/2,000 questions end with the exact canonical span "
                         "'Answer Choices: (A) b1 (B) b2 (C) b3 (D) b4 (E) b5'. PMC_VQA does not "
                         "(6/33,430 = 0.018% contain all option bodies).",
                 why_it_matters="permuting only the `Options:` block would produce a "
                                "SELF-CONTRADICTORY prompt on 100% of MedXpertQA-MM, and any "
                                "measured effect would be an artifact of that contradiction rather "
                                "than answer-position debiasing.",
                 change="the rotation is applied CONSISTENTLY to the inline span and the Options "
                        "block. The identity rewrite is asserted byte-identical to the original "
                        "question on 100% of rows; an item whose span cannot be parsed exactly "
                        "falls back to K=1."),
            dict(id=2, when="2026-08-10, BEFORE any GPU forward pass",
                 finding="5 rows of PMC-VQA test_2.csv carry malformed option labels "
                         "(duplicated / out-of-order letters: rows 1320, 6237, 9664, 12169, 19321).",
                 change="those items get K=1 in every arm and every view returns the VERBATIM "
                        "original prompt, so the identity view stays byte-exact against the "
                        "deployed baseline on 100% of rows. Can only remove headroom."),
            dict(id=3, when="2026-08-10, BEFORE any GPU forward pass, from a measured 150-item "
                            "stratified sample of native image resolutions per cell",
                 finding="the pre-registered ABSOLUTE image caps are degenerate on these cells: "
                         "96.7% of PATH_VQA_closed and 72.7% of SLAKE_closed items are ALREADY "
                         "below cap640 (501,760 px), and 12-24% are already below cap320, so the "
                         "cap640 view would have been a byte-identical duplicate of the identity "
                         "view on most items.",
                 why_it_matters="two of the four views would have carried zero information while "
                                "still being charged as forward passes, turning a stated K=4 into "
                                "an effective K=2 at K=4 cost.",
                 change="the two image views are redefined RELATIVE to each item's own native pixel "
                        "count -- 0.50x and 0.25x, floored at MIN_PX -- which is non-degenerate on "
                        "100% of items. The instruction axis is unchanged. The resolved per-item "
                        "max_pixels is written to every checkpoint row."),
        ],
    )
    os.makedirs(ART, exist_ok=True)
    json.dump(doc, open(PREREG, "w"), indent=1)
    print("wrote", PREREG)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prereg", action="store_true")
    ap.add_argument("--nulltest", action="store_true")
    a = ap.parse_args()
    if a.prereg:
        write_prereg()
    if a.nulltest:
        n1 = n1_macro()
        print("N1 max abs deviation =", n1["max_abs_deviation"], "passed =", n1["passed"])
        print("always_32b_direct macro =", n1["always_32b_direct_macro"])
        n2 = n2_prompt_reconstruction()
        for k, v in n2.items():
            print("N2", k, v if not isinstance(v, dict) else
                  {kk: vv for kk, vv in v.items() if kk != "kind"})
        json.dump(dict(N1=n1, N2=n2), open(os.path.join(ART, "mcq_tta_nulltests_2026-08-10.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
