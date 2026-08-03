#!/usr/bin/env python3
"""choicewhy_common.py -- shared definitions for the (choice)(why) program (Phase 2).

Everything here is copied VERBATIM from the Phase-1 code so that generation, labelling and training
cannot drift apart:
  SYS            -> src/labeling/run_choicewhy_pilot.py (the arm system messages)
  extract/       -> src/cascade_methods/choicewhy_pilot_analyze.py (the letter extractor + rationale
  strip_letter      stripper used for every Phase-1 number)
The Phase-1 files themselves are left untouched; this module is the single import point for new code.
"""
import json, re, string

# --------------------------------------------------------------- arm system messages (verbatim)
SYS = {
    "A": "Answer with only the correct option letter (e.g. 'A'). Do not explain.",
    "B": ("Answer with the correct option letter first (e.g. 'A'), then explain in one or two short "
          "sentences why that option is correct. Example: \"A. The mass is in the left lower lobe.\""),
    "C": ("Explain in one or two short sentences why an option is correct, then answer with the "
          "correct option letter last (e.g. 'A'). Example: \"The mass is in the left lower lobe. A.\""),
    "B2": ("Answer with the correct option letter first (e.g. 'A'), then, in exactly one sentence, state "
           "the specific finding in the image that makes that option correct. Always give the sentence, "
           "even when the answer is obvious. Example: \"A. The mass is in the left lower lobe.\""),
    "C2": ("First, in exactly one sentence, state the specific finding in the image that makes an option "
           "correct, then answer with that option letter last (e.g. 'A'). Always give the sentence, "
           "even when the answer is obvious. Example: \"The mass is in the left lower lobe. A.\""),
}
ARM_NAME = {"A": "A_letter_only", "B": "B_answer_first", "C": "C_reason_first",
            "B2": "B2_answer_first_forced", "C2": "C2_reason_first_forced"}

# --------------------------------------------------------------- letter extraction (verbatim)
LET_FIRST = re.compile(r"\b([A-J])\b")
LEAD = re.compile(r"^\s*[*\"'(\[]*\s*([A-J])\s*(?=[).:,;\-\u2014\]]|$|\n)")
BOXED = re.compile(r"\\boxed\{\s*\(?\s*([A-J])")
MARK = re.compile(r"(?:answer|option|choice|correct|select)(?:\s+is)?\s*(?:letter)?\s*[:=]?\s*[*\"'(\[]*\s*([A-J])\b", re.I)


def extract(text, arm):
    """Return (letter, parse_ok, rule). Arm-appropriate rules; every rule hit is counted and reported."""
    t = text.strip()
    if "letter_only" in arm:
        m = LET_FIRST.search(t)
        return (m.group(1), 1, "first_standalone") if m else ("?", 0, "none")
    if "answer_first" in arm:
        m = LEAD.match(t)
        if m:
            return m.group(1), 1, "lead"
        b = BOXED.findall(t)
        if b:
            return b[-1].upper(), 1, "boxed"
        k = list(MARK.finditer(t))
        if k:
            return k[-1].group(1).upper(), 1, "marker"
        m = LET_FIRST.search(t)
        return (m.group(1), 1, "first_standalone") if m else ("?", 0, "none")
    b = BOXED.findall(t)
    if b:
        return b[-1].upper(), 1, "boxed"
    k = list(MARK.finditer(t))
    if k:
        return k[-1].group(1).upper(), 1, "marker"
    ls = LET_FIRST.findall(t)
    if ls:
        return ls[-1].upper(), 1, "last_standalone"
    return "?", 0, "none"


def strip_letter(text, arm):
    """Remove the answer token so the rationale text is graded on its own."""
    t = text.strip()
    if "answer_first" in arm:
        t = LEAD.sub("", t, count=1)
    else:
        t = re.sub(r"[\s.,;:*\"'()\[\]]*\b[A-J]\b[\s.)*\"'\]]*$", "", t)
    t = re.sub(r"\\boxed\{[^}]*\}", " ", t)
    return t.strip(" .,:;-\n\t")


def norm(s):
    s = str(s).lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()


def parse_opts(s):
    if isinstance(s, dict):
        return s
    try:
        return json.loads(s)
    except Exception:
        return dict(re.findall(r'"([A-J])"\s*:\s*"((?:[^"\\]|\\.)*)"', s))


# --------------------------------------------------------------- the MCQ verifier prompt
# Mirrors the open-text verifier's prompt (src/training_methods/run_lora_verifier_disjoint.py) with the
# option block added, because on MCQ the proposed answer is only interpretable next to the options.
VERIF_SYS = ("You are a careful medical exam grader. Given a question and a proposed answer, decide whether the "
             "proposed answer is correct. Respond with only 'Yes' or 'No'.")


def verifier_body(question, options, candidate):
    """The text half of the verifier's user turn. IDENTICAL for every arm -- only `candidate` differs
    (a bare letter in arm A, '<letter>. <one-sentence finding>' in arm B2), so FORMAT is the only variable."""
    opts = "\n".join(f"{k}) {v}" for k, v in options.items())
    return (f"Question: {question}\n{opts}\nProposed answer: {candidate}\n"
            f"Is the proposed answer correct? Answer Yes or No.")
