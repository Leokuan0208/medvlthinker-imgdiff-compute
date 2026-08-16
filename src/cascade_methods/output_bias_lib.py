#!/usr/bin/env python3
"""output_bias_lib.py -- ATTACK 1 (2026-08-17): the OUTPUT-BIAS AUDIT AND CORRECTION.

THE HYPOTHESIS.  Medical VLMs carry systematic, measurable, FORMAT-INDUCED output biases -- a
yes-bias on binary questions and an answer-letter/position bias on multiple choice -- and removing
them at test time is nearly free.  The evidence that motivated it:
  * PATH_VQA_closed +0.0419 judge / +0.0416 EM at matched fullres purely from replacing
    MedEvalKit's "Please output 'yes' or 'no'(no extra output)." with an open instruction
    [artifacts/closed_as_open_2026-08-16.json, arms openMEK_g_full vs closedD_g_full].
  * PMC-VQA gold letters are A 13.2 / B 35.8 / C 37.8 / D 13.1 -- B+C = 73.6%
    [artifacts/pmcvqa_answer_bias_audit_2026-08-11.json].
  * 7B precision varies ~0.39 across predicted letters while its confidence varies ~0.10
    [artifacts/pmcvqa_answer_bias_audit_2026-08-11.json T11, via pmcvqa_answer_bias_verdict.py].

TWO CORRECTION FAMILIES, BOTH NEARLY FREE:
  (a) PROMPT-SIDE  -- change the instruction so the bias is never induced.  MedEvalKit is READ ONLY;
      every alternate prompt is built HERE, in our code, and the harness is never modified.
  (b) OUTPUT-SIDE  -- leave the prompt alone and recalibrate the first-token option posterior by a
      prior estimated OFF THE EVAL LABELS.  This is the "calibrate before use" / contextual
      calibration family and it costs ZERO extra forward passes at inference when the prior is
      global (the logprobs come out of the same forward pass that already produced the answer).

LEAKAGE RULE (pre-specified).  A correction fitted on the eval set's own labels is not a method.
Every prior here comes from one of exactly three label-free sources, and each is tagged:
    TRAIN        -- Lingshu-7B's own predictions on PMC-VQA train_2.csv + the TRAIN gold marginal.
    CONTENT_FREE -- forward passes on a content-free input (gray image, question "N/A").
    TRANSDUCTIVE -- the eval set's own PREDICTIONS (never its labels) matched to the TRAIN gold
                    marginal, 5-fold cross-fit so no item's own prediction sets its own shift.

NOT ABSTENTION.  Every arm returns an answer for every item; a corrected argmax is still an answer.

CPU only, no GPU, no file writes on import.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
CKPT = os.path.join(ROOT, "ckpts/output_bias")
DATE = "2026-08-17"

# ---------------------------------------------------------------------------------------------
# pre-registered constants
# ---------------------------------------------------------------------------------------------
SEED_TRAIN = 20260817          # PMC train subsample RNG
SEED_BOOT = 20260817           # paired item bootstrap
SEED_PERM = 20260817           # permutation null
NBOOT = 10000
NPERM = 1000
N_TRAIN = 6000                 # PMC-VQA train_2.csv items used to fit the TRAIN prior
NFOLD = 5                      # cross-fit folds for the TRANSDUCTIVE variant
BAR_MACRO_DELTA = 0.0029       # macro delta needed for a CI-clean win (CLAUDE.md sec 0)

MACRO8 = ["PMC_VQA", "SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed", "MedXpertQA-MM",
          "SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
#: published always-7B greedy per cell (artifacts/cascade_selector_rerun_2026-08-05.json)
PUBLISHED_7B = {"PMC_VQA": 0.5427, "SLAKE_closed": 0.8254, "VQA_RAD_closed": 0.7809,
                "PATH_VQA_closed": 0.8409, "MedXpertQA-MM": 0.2615, "SLAKE_open": 0.7364,
                "VQA_RAD_open": 0.4650, "PATH_VQA_open": 0.3240}
EXPECT_N = {"PMC_VQA": 33430, "SLAKE_closed": 836, "VQA_RAD_closed": 251,
            "PATH_VQA_closed": 3362, "MedXpertQA-MM": 2000,
            "SLAKE_open": 645, "VQA_RAD_open": 200, "PATH_VQA_open": 1500}

MCQ_CELLS = ["PMC_VQA", "MedXpertQA-MM"]
BINARY_CELLS = ["SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed"]
OPEN_CELLS = ["SLAKE_open", "VQA_RAD_open", "PATH_VQA_open"]
GEN_CELLS = MCQ_CELLS + BINARY_CELLS + ["PMC_TRAIN"]

#: the deployed dump every published cell is read from
DEPLOYED_TAG = "lingshu7b_full"
DEPLOYED_DS = {"PMC_VQA": "PMC_VQA", "MedXpertQA-MM": "MedXpertQA-MM", "SLAKE_closed": "SLAKE",
               "VQA_RAD_closed": "VQA_RAD", "PATH_VQA_closed": "PATH_VQA"}

MIN_PX = 4 * 28 * 28
GRAY_SIZE = (448, 448)         # content-free probe image; 448*448/784/4 = 64 visual tokens

# ---------------------------------------------------------------------------------------------
# THE ARMS.  Every arm is greedy and uses MedEvalKit's own decode settings for the cell; the ONLY
# deliberate change vs the deployed path is logprobs 5 -> 20, which is what makes the option
# posterior readable.  mcq_tta's N3 measured that this leaves the greedy argmax alone (per-item
# agreement 0.986-0.995 on the 32B) -- N3 below re-measures it for the 7B.
# ---------------------------------------------------------------------------------------------
ARMS = {
    # real input, real prompt: THE deployed pass.  Zero extra cost -- these logprobs already exist
    # inside the forward pass that produced the published answer.
    "id":       dict(question="real", image="real",  options="real",  cells=MCQ_CELLS + BINARY_CELLS),
    # content-free probe keeping the item's own option list: per-item contextual calibration
    # (Zhao et al. 2021).  Costs ONE extra forward pass per item on a 64-visual-token gray image.
    "cf_na":    dict(question="na",   image="gray",  options="real",  cells=MCQ_CELLS + BINARY_CELLS),
    # fully content-free: one request per option count -> a GLOBAL positional prior, cost O(1).
    "cf_blank": dict(question="na",   image="gray",  options="blank", cells=MCQ_CELLS + BINARY_CELLS),
    # TRAIN prior: the model's own letter marginal on PMC-VQA train_2.csv.  Offline fitting cost,
    # zero inference cost.
    "train":    dict(question="real", image="real",  options="real",  cells=["PMC_TRAIN"]),
    # ---- EXPLORATORY, ADDED AFTER THE PRIMARY ENDPOINT WAS SPECIFIED -------------------------
    # PROMPT-SIDE minimal edit: keep the closed answer space but swap the ORDER the two options are
    # named in ("no' or 'yes" instead of "yes' or 'no"), and for SLAKE_closed paraphrase its
    # close-ended instruction.  This separates "the answer space is given at all" (which is what
    # the 2026-08-16 open-instruction arm removed, worth +0.0419 on PATH_VQA_closed) from "the
    # order the options are listed in".  Same token budget, so it cannot move the grader by length.
    "swap":     dict(question="real", image="real",  options="real",  cells=BINARY_CELLS,
                     instruction="swap"),
    # IMAGE-free probe: the REAL question, a gray image.  On a binary cell the fully content-free
    # probe is degenerate -- with the question replaced by "N/A" and no option list to vary, every
    # item collapses to ONE prompt, so cf_na and cf_blank coincide there.  This arm instead removes
    # only the IMAGE, which measures the model's LANGUAGE-ONLY yes-prior for each question: the
    # blind prior a contextual calibration on these cells actually needs.
    "cf_img":   dict(question="real", image="gray",  options="real",  cells=BINARY_CELLS),
}
POST_HOC_ARMS = ("swap", "cf_img")
PRIMARY_ARM = "id"

#: PRE-SPECIFIED PRIMARY CORRECTION (chosen before any 7B number was read): pm_train -- the global
#: logit shift fitted by marginal matching on the TRAIN arm.  It is the only zero-inference-cost
#: correction whose prior touches neither the eval inputs nor the eval labels.
PRIMARY_CORRECTION = "pm_train"
CORRECTIONS = ["none", "pm_train", "pm_transductive_cv", "cc_cf_na", "cc_cf_blank", "cc_cf_img_blind_language_prior"]

# =============================================================================================
# 1. prompts -- byte-exact re-implementations of MedEvalKit/utils/question_formats.py
#    (MedEvalKit is read ONLY; N2 asserts byte-equality against its stored prompt strings)
# =============================================================================================
INSTR_MCQ = "Answer with the option's letter from the given choices directly."
INSTR_JUDGE = "Please output 'yes' or 'no'(no extra output)."
INSTR_JUDGE_SWAP = "Please output 'no' or 'yes'(no extra output)."
INSTR_CLOSE_EN = "Answer the question using a single word or phrase."
INSTR_CLOSE_ZH = "\u8bf7\u7528\u4e00\u4e2a\u5355\u8bcd\u6216\u8005\u77ed\u8bed\u56de\u7b54\u8be5\u95ee\u9898\u3002"


def p_mcq(question, choices):
    """VERBATIM MedEvalKit/utils/question_formats.py get_multiple_choice_prompt.
    NOTE the TRAILING SPACE after 'Options:' -- byte-checked by N2."""
    options = "\n".join([str(c) for c in choices])
    return "\nQuestion: " + str(question) + "\nOptions: \n" + options + "\n" + INSTR_MCQ


def p_judge(question, swap=False):
    return question + "\n" + (INSTR_JUDGE_SWAP if swap else INSTR_JUDGE)


def p_close(question, lang="en"):
    return question + "\n" + (INSTR_CLOSE_ZH if lang == "zh" else INSTR_CLOSE_EN)


CHOICE_RE = re.compile(r"^(\s*)([A-Za-z])(\s*[:.)]\s*)(.*)$", re.S)


def blank_choices(choices):
    """Same letters, same separators, body replaced by 'N/A' -- the fully content-free option list."""
    out = []
    for c in choices:
        m = CHOICE_RE.match(str(c))
        assert m is not None, f"unparsed choice {c!r}"
        lead, letter, sep, _body = m.groups()
        out.append(lead + letter + sep + "N/A")
    return out


def deployed_prompt(cell, item):
    if cell in ("PMC_VQA", "MedXpertQA-MM", "PMC_TRAIN"):
        return p_mcq(item["question"], item["choices"])
    if cell == "SLAKE_closed":
        return p_close(item["question"], item["lang"])
    return p_judge(item["question"])


#: paraphrases of SLAKE's close-ended instruction (ours, not MedEvalKit's) -- same token budget
INSTR_CLOSE_EN_PARA = "Respond using a single word or phrase."
INSTR_CLOSE_ZH_PARA = "请以一个单词或者短语作答。"


def arm_prompt(cell, item, arm):
    """The prompt for (cell, item, arm).  `real` reproduces MedEvalKit byte for byte."""
    cfg = ARMS[arm]
    q = item["question"] if cfg["question"] == "real" else "N/A"
    swap = cfg.get("instruction") == "swap"
    if cell in ("PMC_VQA", "MedXpertQA-MM", "PMC_TRAIN"):
        ch = item["choices"] if cfg["options"] == "real" else blank_choices(item["choices"])
        return p_mcq(q, ch)
    if cell == "SLAKE_closed":
        if swap:
            return q + "\n" + (INSTR_CLOSE_ZH_PARA if item["lang"] == "zh"
                               else INSTR_CLOSE_EN_PARA)
        return p_close(q, item["lang"])
    return p_judge(q, swap=swap)


# =============================================================================================
# 2. items
# =============================================================================================
def build_items(cells=None):
    """{cell: [item,...]} in the EXACT index order the deployed per-sample vectors use.
    PMC_VQA / MedXpertQA-MM / the three closed cells come from mcq_tta.build_items(), the loader
    whose order every deployed vector in this project is keyed to.  PMC_TRAIN is new here."""
    cells = list(cells or GEN_CELLS)
    out = {}
    need_mek = [c for c in cells if c != "PMC_TRAIN"]
    if need_mek:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mcq_tta as M
        it = M.build_items()
        for c in need_mek:
            rows = it[c]
            assert len(rows) == EXPECT_N[c], (c, len(rows), EXPECT_N[c])
            out[c] = rows
    if "PMC_TRAIN" in cells:
        out["PMC_TRAIN"] = build_pmc_train()
    return out


def build_pmc_train():
    """A fixed random N_TRAIN-item sample of PMC-VQA train_2.csv.  The TRAIN prior source.
    Disjoint from test_2.csv by construction (different split file); the disjointness is
    re-verified on Figure_path in the audit."""
    base = "/data/dan/dataset/medevalkit/PMC-VQA"
    rows = list(csv.reader(open(os.path.join(base, "train_2.csv"), encoding="utf-8")))[1:]
    rng = np.random.default_rng(SEED_TRAIN)
    order = rng.permutation(len(rows))[:N_TRAIN]
    it = []
    for k, j in enumerate(sorted(int(x) for x in order)):
        r = rows[j]
        _, fig, _cap, q, cA, cB, cC, cD, ans, _split = r
        it.append(dict(i=k, src=int(j), question=q, choices=[cA, cB, cC, cD], answer=ans,
                       lang="en", images=[os.path.join(base, "figures", fig)], img_kind="path",
                       fmt="mcq", figure=fig, malformed=False))
    return it


# =============================================================================================
# 3. GRADERS -- verbatim MedEvalKit/utils/utils.py (re-implemented, never imported for scoring)
# =============================================================================================
try:
    from mathruler.grader import extract_boxed_content as _boxed
except Exception:                                                    # pragma: no cover
    def _boxed(s):
        m = re.search(r"\\boxed\{([^}]*)\}", s)
        return m.group(1) if m else s


def _extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1) if m else text


def parse_response(response):
    """VERBATIM MedEvalKit/utils/utils.py parse_response."""
    response = str(response).lower()
    if "boxed" in response:
        response = _boxed(response)
    elif "<answer>" in response:
        response = _extract_tag(response, "answer")
    for pat in ["**answer**:", "**answer**", "*answer*:", "**answer:**", "answer is", "answer:",
                "\u7b54\u6848:", "final answer", "final answer is"]:
        if pat in response:
            response = response.split(pat)[-1]
    return response


def str_similarity(str1, str2):
    import difflib
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def find_most_similar_index(str_list, target_str):
    most_similar_index, highest_similarity = 0, 0
    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        if similarity > highest_similarity:
            most_similar_index, highest_similarity = i, similarity
    return most_similar_index


def judge_multi_choice(choices, answer, response):
    """VERBATIM MedEvalKit/utils/utils.py judge_multi_choice.  The known defect (an unparsed
    response falls through to find_most_similar_index against the option BODIES) is kept, because
    it is what defines the published cell; the letter_em currency below makes it visible."""
    response = str(response).lower()
    alph0 = [chr(ord('a') + i) for i in range(len(choices))]
    if response.split("\n\n")[0] in alph0:
        response = response.split("\n\n")[0]
    elif response.split("\n\n")[-1].split(".")[0] in alph0:
        response = response.split("\n\n")[-1].split(".")[0]
    response = parse_response(response)
    alphas = [chr(ord('a') + i) for i in range(len(choices))]
    choices = [str(c).lower() for c in choices]
    flag = False
    response = response.strip().lower().replace("\n", "")
    split_response = response.split(".")[0]
    split_response = split_response.split(":")[-1]
    answer = str(answer).strip().lower()
    if len(split_response) > 300:
        flag = False
    if split_response == answer:
        flag = True
    elif split_response in alphas:
        if choices[ord(split_response) - ord("a")] == answer:
            flag = True
    elif split_response in choices:
        if answer in alphas and split_response == choices[ord(answer) - ord("a")]:
            flag = True
    else:
        index = find_most_similar_index(choices, response)
        if alphas[index] == answer or choices[index] == answer:
            flag = True
    return flag


def judge_close_end_vqa(answer, response):
    """VERBATIM MedEvalKit/utils/utils.py judge_close_end_vqa (SLAKE_closed's grader)."""
    answer = str(answer).lower()
    response = parse_response(response)
    response = response.replace("\n", "").replace(".", "")
    return response == answer


def judge_judgement(answer, response):
    """VERBATIM MedEvalKit/utils/utils.py judge_judgement (VQA_RAD/PATH_VQA closed grader)."""
    answer = str(answer).lower()
    response = parse_response(response)
    response = response.replace("\n", "").replace(".", "")
    if ("yes" in response) ^ ("no" in response):
        if answer in response:
            return True
    return False


def em_harness(cell, item, response):
    """The cell's OWN deployed MedEvalKit grader, applied to a response string."""
    if cell in ("PMC_VQA", "MedXpertQA-MM", "PMC_TRAIN"):
        return int(bool(judge_multi_choice([str(c).lower() for c in item["choices"]],
                                           str(item["answer"]), response)))
    if cell == "SLAKE_closed":
        return int(bool(judge_close_end_vqa(item["answer"], response)))
    return int(bool(judge_judgement(item["answer"], response)))


# ---- length-neutral repaired grader for the binary cells (verbatim from closed_as_open_lib) ----
AFF_WORDS = {"yes", "yeah", "yep", "correct", "true", "present", "positive", "affirmative"}
NEG_WORDS = {"no", "not", "none", "false", "incorrect", "absent", "negative", "nope"}


def norm_text(s):
    s = str(s).strip().lower()
    s = s.strip("\"'\u201c\u201d\u2018\u2019 \t\n")
    s = s.rstrip("\u3002.")
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", norm_text(s)) if t]


def polarity(response):
    for t in _tokens(response):
        if t in AFF_WORDS:
            return "yes"
        if t in NEG_WORDS:
            return "no"
    return None


# =============================================================================================
# 4. first-token option posteriors
# =============================================================================================
#: first-token surface WORDS that vote YES / NO.  Frozen BEFORE any accuracy was read.  Matching is
#: done on the token with its leading word-boundary marker stripped ("\u0120" or " ") and lowercased,
#: because the vLLM top-20 contains both "No" and " No" for the same word.  The audit reports the
#: residual first-token mass that falls outside both sets, so the choice is auditable.
AFF_TOK = {"yes", "yeah", "yep", "y", "true", "correct"}
NEG_TOK = {"no", "not", "none", "nope", "n", "false", "incorrect"}
FLOOR_LOGPROB = -30.0          # value used for an option letter absent from the top-20


def _strip_marker(tok):
    t = str(tok)
    if t.startswith("\u0120") or t.startswith(" "):
        t = t[1:]
    return t


def letter_logits(row, n_choices):
    """Logprob of each option letter as the FIRST generated token, max over the bare and
    word-boundary-marked surface forms.

    A letter absent from the returned top-20 is floored at the SMALLEST logprob that IS in this
    row's top-20 -- a valid upper bound on the missing mass, and a bounded one.  A fixed constant
    like -30 would be arbitrary and, once it appears inside a contextual-calibration subtraction,
    would hand that option an enormous artificial boost."""
    lp = row.get("first_logprobs") or {}
    best = {}
    for tok, v in lp.items():
        s = _strip_marker(tok)
        if len(s) == 1 and "A" <= s <= "Z":
            best[s] = max(float(v), best.get(s, -1e9))
    floor = min([float(v) for v in lp.values()], default=FLOOR_LOGPROB) if lp else FLOOR_LOGPROB
    return np.array([best.get(chr(65 + i), floor) for i in range(n_choices)])


def letter_coverage(row, n_choices):
    """How many of the n option letters were actually present in this row's returned top-k."""
    lp = row.get("first_logprobs") or {}
    have = {_strip_marker(t) for t in lp}
    return sum(1 for i in range(n_choices) if chr(65 + i) in have)


def yesno_logits(row):
    """(yes_logit, no_logit) as the max over the frozen surface-form sets."""
    lp = row.get("first_logprobs") or {}
    a = n = FLOOR_LOGPROB
    for tok, v in lp.items():
        s = _strip_marker(tok).lower()
        if s in AFF_TOK:
            a = max(a, float(v))
        elif s in NEG_TOK:
            n = max(n, float(v))
    return a, n


# =============================================================================================
# 5. prior fitting -- marginal matching (all label-free w.r.t. the EVAL set)
# =============================================================================================
def fit_shift_marginal(logits, target, iters=800, lr=0.3):
    """Global additive logit shift w (mean 0) such that argmax(logits - w) reproduces `target` as
    its marginal.  `logits` is (n, K); `target` is (K,).  Uses NO labels of the rows in `logits` --
    only their predicted scores and an externally supplied target marginal."""
    n, K = logits.shape
    w = np.zeros(K)
    for _ in range(iters):
        pred = (logits - w).argmax(1)
        cur = np.bincount(pred, minlength=K) / n
        w = w + lr * np.log(np.maximum(cur, 1e-4) / np.maximum(target, 1e-4))
        w -= w.mean()
    return w


def fit_thresh_marginal(diff, target_yes_rate):
    """Threshold t on a yes-minus-no logit difference such that mean(diff > t) == target_yes_rate."""
    return float(np.quantile(diff, 1.0 - float(target_yes_rate)))


# =============================================================================================
# 6. statistics
# =============================================================================================
def boot_delta(a, b, nboot=NBOOT, seed=SEED_BOOT):
    """Paired item bootstrap on mean(a) - mean(b)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, size=(nboot, n))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    delta = float(d.mean())
    return dict(delta=delta, ci=[float(lo), float(hi)],
                significant=bool(lo > 0 or hi < 0),
                sign=("WIN" if lo > 0 else "LOSS" if hi < 0 else "TIE"),
                n=int(n), nboot=int(nboot), seed=int(seed))


def r6(x):
    return None if x is None else round(float(x), 6)


# =============================================================================================
# 7. generation-dump IO
# =============================================================================================
def gen_path(cell, arm, shard=None, nshard=1):
    tag = "" if (nshard or 1) <= 1 else f"_s{shard}of{nshard}"
    return os.path.join(CKPT, f"gen_{cell}_{arm}{tag}.jsonl")


def load_gen(cell, arm):
    """Merge every shard of (cell, arm) by item index."""
    import glob as _glob
    out = {}
    for p in sorted(_glob.glob(os.path.join(CKPT, f"gen_{cell}_{arm}.jsonl"))
                    + _glob.glob(os.path.join(CKPT, f"gen_{cell}_{arm}_s*of*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out[int(r["i"])] = r
    return out


def deployed_rows(cell, tag=DEPLOYED_TAG):
    p = os.path.join(MEK, f"eval_results_{tag}", "{}", DEPLOYED_DS[cell], "results.json")
    d = json.load(open(p, encoding="utf-8"))
    if cell == "SLAKE_closed":
        rows = [r for r in d if r.get("answer_type") == "CLOSED"]
    elif cell in ("VQA_RAD_closed", "PATH_VQA_closed"):
        rows = [r for r in d if str(r.get("answer", "")).strip().lower() in ("yes", "no")]
    else:
        rows = d
    assert len(rows) == EXPECT_N[cell], (cell, len(rows), EXPECT_N[cell])
    return rows


if __name__ == "__main__":
    print(json.dumps({"cells": GEN_CELLS, "arms": list(ARMS), "corrections": CORRECTIONS,
                      "primary_correction": PRIMARY_CORRECTION, "date": DATE}, indent=1))
