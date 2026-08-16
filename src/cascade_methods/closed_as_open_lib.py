#!/usr/bin/env python3
"""closed_as_open_lib.py -- BUILD 3 (2026-08-16) shared loader, prompt builder and GRADERS for the
"re-ask the closed cells as open text" experiment.

THE QUESTION.  The verifier works on the generator's OWN SAMPLES (2.18x the random floor) and fails
on the prompt's GIVEN answer space (below 1/K on 3 of 4 cells).  SLAKE_closed / VQA_RAD_closed /
PATH_VQA_closed are graded as closed cells but the underlying datasets are open VQA.  If asking them
open-ended moves the candidates from GIVEN to SAMPLED, the verifier claim goes from 3 cells to 6.

WHAT THE HARNESS ACTUALLY DOES (read off MedEvalKit, not assumed -- this is load-bearing and it
partly contradicts the premise):
    VQA_RAD_closed   MedEvalKit/utils/VQA_RAD/VQA_RAD.py:45   get_judgement_prompt
                     -> "<q>\\nPlease output 'yes' or 'no'(no extra output)."   ANSWER SPACE GIVEN, K=2
    PATH_VQA_closed  MedEvalKit/utils/PATH_VQA/PATH_VQA.py:60  get_judgement_prompt   same
    SLAKE_closed     MedEvalKit/utils/SLAKE/SLAKE.py:56        get_close_ended_prompt
                     -> "<q>\\nAnswer the question using a single word or phrase."  NO ANSWER SPACE
So SLAKE_closed is ALREADY asked open-ended; only its GRADER is closed (strict string equality).
The reformat is a real change of the answer space on 2 of the 3 cells, not 3.

GRADING IS PRE-SPECIFIED HERE, BEFORE ANY ACCURACY IS READ (protocol requirement).  Two EM rules,
both applied byte-identically to every arm so no arm can be advantaged by its grader:

  EM_harness  -- MedEvalKit's own function for the cell, re-implemented verbatim below and
                 NULL-TESTED against the deployed results.json `correct` field (grader_null_test()).
                 SLAKE_closed -> judge_close_end_vqa (strict equality)
                 VQA_RAD_closed / PATH_VQA_closed -> judge_judgement (xor of yes/no, then gold in it)
                 This is the grader that DEFINES the published 0.8254 / 0.7809 / 0.8409 cells.

  EM_repaired -- LENGTH-NEUTRAL by construction, so a verbose open-form answer can never gain:
                 * gold in {yes, no}  -> POLARITY rule: scan the normalised response left to right;
                   the FIRST token in AFF wins -> "yes"; the FIRST token in NEG wins -> "no";
                   neither -> UNPARSED, scored 0 (never an abstention: the answer is still returned,
                   it is simply graded wrong).
                 * otherwise -> STRICT normalised equality (lowercase, strip, drop trailing ASCII and
                   full-width period, collapse whitespace, strip surrounding quotes).  NO substring
                   containment anywhere, deliberately: containment is the mechanism by which a longer
                   answer scores higher for free, and this project has already had two apparent wins
                   turn out to be grader effects.

  The 32B judge (MedVLThinker-32B, src/labeling/run_judge.py) is the SECOND currency and is run
  per DISTINCT normalised answer string, so identical strings produced by different arms get the
  same judge call and no arm can be advantaged by judge noise.

CPU only, no GPU, no file writes on import.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MEK = os.path.join(ROOT, "MedEvalKit")
ART = os.path.join(ROOT, "results/cascade_methods/artifacts")
CKPT = os.path.join(ROOT, "ckpts/closed_as_open")
PARTS = os.path.join(ART, "_closed_as_open_parts")
DATE = "2026-08-16"

CELLS = ["SLAKE_closed", "VQA_RAD_closed", "PATH_VQA_closed"]
EXPECT_N = {"SLAKE_closed": 836, "VQA_RAD_closed": 251, "PATH_VQA_closed": 3362}
#: the deployed MedEvalKit run these cells are read from (fullres, no system prompt, greedy)
DEPLOYED_TAG = "lingshu7b_full"
DEPLOYED_DS = {"SLAKE_closed": "SLAKE", "VQA_RAD_closed": "VQA_RAD", "PATH_VQA_closed": "PATH_VQA"}
#: published always-7B cells (artifacts/cascade_selector_rerun_2026-08-05.json, per_cell_acc)
PUBLISHED_ALWAYS_7B = {"SLAKE_closed": 0.8254, "VQA_RAD_closed": 0.7809, "PATH_VQA_closed": 0.8409}

HIGH_PX, MIN_PX = 1280 * 28 * 28, 4 * 28 * 28
CAP_DIV = {"fullres": 1, "cap640": 2, "cap320": 4}

#: verbatim from src/labeling/run_openvqa.py -- the system prompt the DEPLOYED open arm generated
#: its 8-sample pools with, and therefore the regime the frozen verifier was built for.
SYS_OPEN = ("You are an expert medical image analyst. Answer the question with a short, specific "
            "phrase. Do not explain.")

#: T = 0.4 is the settled sampling optimum [artifacts/decoding_ladder_cold_2026-08-14.json];
#: seed / top_p / top_k / min_p / rep_pen copied from _decoding_sweep_settings_ladder.json.
T04 = dict(temp=0.4, top_p=1.0, top_k=-1, min_p=0.0, rep_pen=1.0, seed=20260813)
GREEDY = dict(temp=0.0, top_p=1.0, top_k=-1, min_p=0.0, rep_pen=1.0, seed=42)
N_SAMPLES = 8
MAX_TOKENS = 64          # the deployed open arm's value (run_openvqa.py --max_tokens 64)

# ---------------------------------------------------------------------------------------------
# THE ARMS.  All at cap320 (the deployed open-arm resolution, and the resolution the T=0.4 result
# was measured at) except closedD_g_full, which exists only to quantify the resolution half of the
# gap to the published cell.
# ---------------------------------------------------------------------------------------------
ARMS = {
    "closedD_g":      dict(prompt="deployed", sys=None,     cap="cap320",  **GREEDY, n=1),
    "closedD_g_full": dict(prompt="deployed", sys=None,     cap="fullres", **GREEDY, n=1),
    "closedD_s8":     dict(prompt="deployed", sys=None,     cap="cap320",  **T04, n=N_SAMPLES),
    "openMEK_g":      dict(prompt="open_mek", sys=None,     cap="cap320",  **GREEDY, n=1),
    "openMEK_s8":     dict(prompt="open_mek", sys=None,     cap="cap320",  **T04, n=N_SAMPLES),
    "openPRJ_g":      dict(prompt="bare",     sys=SYS_OPEN, cap="cap320",  **GREEDY, n=1),
    "openPRJ_s8":     dict(prompt="bare",     sys=SYS_OPEN, cap="cap320",  **T04, n=N_SAMPLES),
}
#: PRE-SPECIFIED PRIMARY ENDPOINT: SELECTED (frozen verifier over openPRJ_s8) vs GREEDY (openPRJ_g),
#: per cell, in BOTH currencies.  openPRJ is primary because its generation recipe is byte-identical
#: to the deployed open arm (SYS_OPEN + bare question + cap320 + max_tokens 64), i.e. it is the
#: honest "does the shipped method extend to these cells" test.
PRIMARY = dict(sampled="openPRJ_s8", greedy="openPRJ_g")


# =============================================================================================
# 1. prompts -- verbatim MedEvalKit/utils/question_formats.py
# =============================================================================================
def get_close_ended_prompt(question, lang="en"):
    if lang == "zh":
        return question + "\n" + "请用一个单词或者短语回答该问题。"
    return question + "\n" + "Answer the question using a single word or phrase."


def get_judgement_prompt(question, lang="en"):
    if lang == "zh":
        return question + "\n" + "请输出'是'或'否'(不要有任何其它输出)。"
    return question + "\n" + "Please output 'yes' or 'no'(no extra output)."


def get_open_ended_prompt(question, lang="en"):
    if lang == "zh":
        return question + "\n" + "请简要回答该问题。"
    return question + "\n" + "Please answer the question concisely."


def deployed_prompt(cell, question, lang):
    """The prompt MedEvalKit actually used for this cell."""
    if cell == "SLAKE_closed":
        return get_close_ended_prompt(question, lang)
    return get_judgement_prompt(question, "en")


def build_prompt(kind, cell, question, lang):
    if kind == "deployed":
        return deployed_prompt(cell, question, lang)
    if kind == "open_mek":
        return get_open_ended_prompt(question, lang)
    if kind == "bare":
        return question
    raise ValueError(kind)


def has_answer_space(kind, cell):
    """Does this (arm, cell) prompt hand the model a complete answer space?"""
    if kind != "deployed":
        return False
    return cell in ("VQA_RAD_closed", "PATH_VQA_closed")     # get_judgement_prompt -> {yes,no}


# =============================================================================================
# 2. items -- reuse mcq_tta.build_items(), the loader whose order the deployed vectors use
# =============================================================================================
def build_items():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mcq_tta as M
    it = M.build_items()
    out = {}
    for c in CELLS:
        rows = it[c]
        assert len(rows) == EXPECT_N[c], (c, len(rows), EXPECT_N[c])
        out[c] = rows
    return out


# =============================================================================================
# 3. GRADERS
# =============================================================================================
try:
    from mathruler.grader import extract_boxed_content as _boxed
except Exception:                                                    # pragma: no cover
    def _boxed(s):
        m = re.search(r"\\boxed\{([^}]*)\}", s)
        return m.group(1) if m else s


def _extract_tag(text, tag):
    """MedEvalKit/utils/utils.py extract(): content between <tag> and </tag>."""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1) if m else text


def parse_response(response):
    """VERBATIM re-implementation of MedEvalKit/utils/utils.py:138 parse_response."""
    response = str(response).lower()
    if "boxed" in response:
        response = _boxed(response)
    elif "<answer>" in response:
        response = _extract_tag(response, "answer")
    for pat in ["**answer**:", "**answer**", "*answer*:", "**answer:**", "answer is", "answer:",
                "答案:", "final answer", "final answer is"]:
        if pat in response:
            response = response.split(pat)[-1]
    return response


def judge_close_end_vqa(answer, response):
    """VERBATIM MedEvalKit/utils/utils.py:162."""
    answer = str(answer).lower()
    response = parse_response(response)
    response = response.replace("\n", "").replace(".", "")
    return response == answer


def judge_judgement(answer, response):
    """VERBATIM MedEvalKit/utils/utils.py:171."""
    answer = str(answer).lower()
    response = parse_response(response)
    response = response.replace("\n", "").replace(".", "")
    if ("yes" in response) ^ ("no" in response):
        if answer in response:
            return True
    return False


def em_harness(cell, gold, response):
    """EM currency 1 -- the cell's OWN deployed MedEvalKit grader."""
    if cell == "SLAKE_closed":
        return int(bool(judge_close_end_vqa(gold, response)))
    return int(bool(judge_judgement(gold, response)))


# ---- the MCQ grader, for the SECONDARY self-consistency cells --------------------------------
def str_similarity(str1, str2):
    """VERBATIM MedEvalKit/utils/utils.py:71."""
    import difflib
    return difflib.SequenceMatcher(None, str1, str2).ratio()


def find_most_similar_index(str_list, target_str):
    """VERBATIM MedEvalKit/utils/utils.py:75."""
    most_similar_index = 0
    highest_similarity = 0
    for i, s in enumerate(str_list):
        similarity = str_similarity(s, target_str)
        if similarity > highest_similarity:
            most_similar_index = i
            highest_similarity = similarity
    return most_similar_index


def judge_multi_choice(choices, answer, response):
    """VERBATIM MedEvalKit/utils/utils.py:98 (alphas argument is unused upstream too).

    NOTE the known defect: an unparsed response falls through to find_most_similar_index, i.e. it is
    silently mapped to the nearest option body -- the same defect that manufactured the +0.0132 PMC
    'win' in artifacts/unified_pipeline_2026-08-12.json.  Kept verbatim because it defines the
    deployed cell; the letter_em currency exists precisely so the defect is visible.
    """
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


# ---- EM currency 2: the length-neutral repaired grader (PRE-SPECIFIED) -----------------------
AFF = {"yes", "yeah", "yep", "correct", "true", "present", "positive", "affirmative"}
NEG = {"no", "not", "none", "false", "incorrect", "absent", "negative", "nope", "no."}
_PUNCT = "。．.,!?;:'\"()[]{}，、；：！？“”‘’"


def norm_text(s):
    """lowercase, strip surrounding quotes/space, drop trailing ASCII/full-width period,
    collapse internal whitespace.  Deliberately does NOT delete words."""
    s = str(s).strip().lower()
    s = s.strip("\"'“”‘’ \t\n")
    s = s.rstrip("。.")
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s):
    return [t for t in re.split(r"[^a-z0-9]+", norm_text(s)) if t]


def polarity(response):
    """First-token-wins polarity extraction.  Returns 'yes' / 'no' / None (unparsed)."""
    for t in _tokens(response):
        if t in AFF:
            return "yes"
        if t in NEG:
            return "no"
    return None


def em_repaired(cell, gold, response):
    """EM currency 2 -- length-neutral. Returns (ok, unparsed_flag)."""
    g = norm_text(gold)
    if g in ("yes", "no"):
        p = polarity(response)
        if p is None:
            return 0, 1
        return int(p == g), 0
    return int(norm_text(response) == g), 0


# =============================================================================================
# 4. deployed dumps + the grader NULL TEST
# =============================================================================================
def deployed_rows(cell, tag=DEPLOYED_TAG):
    p = os.path.join(MEK, f"eval_results_{tag}", "{}", DEPLOYED_DS[cell], "results.json")
    d = json.load(open(p, encoding="utf-8"))
    if cell == "SLAKE_closed":
        rows = [r for r in d if r.get("answer_type") == "CLOSED"]
    else:
        rows = [r for r in d if str(r.get("answer", "")).strip().lower() in ("yes", "no")]
    assert len(rows) == EXPECT_N[cell], (cell, len(rows), EXPECT_N[cell])
    return rows


def grader_null_test():
    """G1 -- re-grade the DEPLOYED responses with this module's EM_harness copy and require it to
    reproduce (a) MedEvalKit's own stored `correct` field row by row and (b) the published cell."""
    out = {}
    worst = 0.0
    for c in CELLS:
        rows = deployed_rows(c)
        mine = [em_harness(c, r["answer"], r["response"]) for r in rows]
        theirs = [int(r.get("correct") is True) for r in rows]
        dis = sum(1 for a, b in zip(mine, theirs) if a != b)
        acc = sum(mine) / len(mine)
        dev = abs(acc - PUBLISHED_ALWAYS_7B[c])
        worst = max(worst, dev, dis)
        out[c] = {"n": len(rows), "row_disagreements_with_harness_correct_field": dis,
                  "acc_regraded_by_this_module": round(acc, 6),
                  "published_always_7b": PUBLISHED_ALWAYS_7B[c],
                  "abs_dev_vs_published": round(dev, 6)}
    out["max_abs_deviation"] = round(float(worst), 6)
    out["pass"] = bool(worst <= 1e-4)
    out["note"] = ("published cells are 4-dp rounded, so a deviation at 1e-5 is rounding. Any "
                   "non-zero row disagreement is a grader defect and fails the test.")
    return out


MCQ_DEPLOYED_DS = {"PMC_VQA": "PMC_VQA", "MedXpertQA-MM": "MedXpertQA-MM"}
#: published always-7B MCQ cells (artifacts/cascade_selector_rerun_2026-08-05.json, per_cell_acc)
PUBLISHED_ALWAYS_7B_MCQ = {"PMC_VQA": 0.5427, "MedXpertQA-MM": 0.2615}


def mcq_grader_null_test(choices_by_cell):
    """G2 -- re-grade the DEPLOYED MCQ responses with this module's judge_multi_choice copy and
    require it to reproduce MedEvalKit's own stored `correct` field row by row."""
    out = {}
    worst = 0.0
    for cell, ds in MCQ_DEPLOYED_DS.items():
        p = os.path.join(MEK, f"eval_results_{DEPLOYED_TAG}", "{}", ds, "results.json")
        if not os.path.exists(p):
            out[cell] = "NOT MEASURED -- no deployed dump"
            continue
        rows = json.load(open(p, encoding="utf-8"))
        ch = choices_by_cell[cell]
        mine, theirs = [], []
        for i, r in enumerate(rows):
            mine.append(int(bool(judge_multi_choice([c.lower() for c in ch[i]],
                                                    str(r["answer"]), r["response"]))))
            theirs.append(int(r.get("correct") is True))
        dis = sum(1 for a, b in zip(mine, theirs) if a != b)
        acc = sum(mine) / len(mine)
        dev = abs(acc - PUBLISHED_ALWAYS_7B_MCQ[cell])
        worst = max(worst, dev, dis)
        out[cell] = {"n": len(rows), "row_disagreements_with_harness_correct_field": dis,
                     "acc_regraded_by_this_module": round(acc, 6),
                     "published_always_7b": PUBLISHED_ALWAYS_7B_MCQ[cell],
                     "abs_dev_vs_published": round(dev, 6)}
    out["max_abs_deviation"] = round(float(worst), 6)
    out["pass"] = bool(worst <= 1e-4)
    return out


# =============================================================================================
# 5. generation-dump IO
# =============================================================================================
def gen_path(cell, arm):
    return os.path.join(CKPT, f"gen_{cell}_{arm}.jsonl")


def load_gen(cell, arm):
    p = gen_path(cell, arm)
    if not os.path.exists(p):
        return {}
    out = {}
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


def explode_path(cell):
    return os.path.join(CKPT, f"judge_{cell}.jsonl")


def judge_map(cell):
    """{(i, normalised answer) -> judge_ok} for one cell, across every arm."""
    p = explode_path(cell).replace(".jsonl", ".judge.jsonl")
    if not os.path.exists(p):
        return {}
    keys = {}
    for line in open(explode_path(cell), encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            keys[r["idx"]] = (int(r["i"]), r["na"])
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["idx"] in keys:
                out[keys[r["idx"]]] = int(r["judge_ok"])
    return out


def scores_path(cell, arm):
    return os.path.join(CKPT, f"vscores_{cell}_{arm}.jsonl")


def load_scores(cell, arm):
    p = scores_path(cell, arm)
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[int(r["i"])] = r["scores"]
    return out


if __name__ == "__main__":
    print(json.dumps(grader_null_test(), indent=1))
