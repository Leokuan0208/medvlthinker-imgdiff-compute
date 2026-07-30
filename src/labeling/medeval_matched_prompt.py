#!/usr/bin/env python3
"""
Matched-prompt wrapper around MedEvalKit's eval.py  ("Option A").

WHY THIS EXISTS
---------------
MedEvalKit's multiple-choice prompt has two arms selected by `--reasoning` / $REASONING:

  upstream reasoning arm : 'Answer with the option's letter from the given choices and put the letter in one "\\boxed{}".'
  upstream direct arm    : "Answer with the option's letter from the given choices directly."

Upstream's "reasoning" arm contains NO reasoning trigger -- it only switches the output
format to \\boxed{}. That is why the pre-edit `*_think` dumps generated 2.6-3.2 tokens.

A local hand-edit to MedEvalKit (since reverted) replaced the reasoning tail with a real
reasoning trigger, which DID make the model reason -- but left the two arms differing in
THREE ways at once (reasoning trigger, the word "directly", and \\boxed{} vs bare letter),
so reasoning-vs-direct was not a matched experiment.

This wrapper restores the reasoning trigger AND removes the two confounds, from OUR repo,
by monkeypatching MedEvalKit's prompt construction at import time. MedEvalKit therefore
stays byte-identical to upstream on disk (it is a PROTECTED dependency -- do not edit it).

THE TWO MATCHED STRINGS (differ by ONLY the leading reasoning clause):

  reasoning : First reason step by step about the question and each option, then put the
              final answer letter from the given choices in one "\\boxed{}".
  direct    : Put the final answer letter from the given choices in one "\\boxed{}".

The reasoning string is byte-identical to the pre-revert local edit (asserted below against
`src/labeling/medevalkit_prompt_ref/`), so the EXISTING `eval_results_*_reason/` dumps stay a
valid comparator and do not need re-running.

ENV GATE
--------
The patch is applied only when MEDEVAL_MATCHED_PROMPT=1 (default: OFF).  With it unset, this
wrapper is a transparent pass-through and an unpatched MedEvalKit run is completely unaffected.

WHAT IS PATCHED
---------------
  1. utils.question_formats.get_multiple_choice_prompt   (MedXpertQA and every other MCQ set)
  2. utils.MMMU.data_utils.construct_prompt              (MMMU builds its prompt separately)
Both are patched by WRAPPING the original and replacing only the instruction tail, with an
assertion that the tail was found -- so the question/options body cannot drift from upstream.
Because MedEvalKit uses `from X import f` (which binds the name in the importer's namespace),
every already-imported reference is rebound too (see `_rebind_everywhere`).

SCOPE / RESIDUAL CAVEAT
-----------------------
Only the MULTIPLE-CHOICE branch is patched. MMMU-Medical-val contains 5 `open` items out of
150 (3.3%); those keep upstream's open-question strings, which remain format-unmatched between
arms. Report the 145-item multiple-choice subset for a fully matched comparison.

USAGE
-----
  # verify the patch on CPU, no model load, prints both arms for one MMMU + one MedXpert item
  MEDEVAL_MATCHED_PROMPT=1 python3 src/labeling/medeval_matched_prompt.py --verify

  # run an eval; every arg after `--` is forwarded verbatim to MedEvalKit's eval.py
  MEDEVAL_MATCHED_PROMPT=1 python3 src/labeling/medeval_matched_prompt.py -- \
      --eval_datasets MMMU-Medical-val --datasets_path hf --reasoning False ...
"""

import argparse
import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEDEVALKIT = os.path.join(REPO_ROOT, "MedEvalKit")
PROMPT_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medevalkit_prompt_ref")

# ----------------------------------------------------------------------------------------
# The strings
# ----------------------------------------------------------------------------------------

# What we install.  These two differ by EXACTLY the reasoning clause (asserted below).
REASON_CLAUSE = "First reason step by step about the question and each option, then "
MCQ_DIRECT_TAIL = 'Put the final answer letter from the given choices in one "\\boxed{}".'
MCQ_REASON_TAIL = (
    'First reason step by step about the question and each option, then put the final '
    'answer letter from the given choices in one "\\boxed{}".'
)

# What upstream MedEvalKit currently has on disk (the tails we replace).
UPSTREAM_REASON_TAIL = (
    'Answer with the option\'s letter from the given choices and put the letter in one "\\boxed{}".'
)
UPSTREAM_DIRECT_TAIL = "Answer with the option's letter from the given choices directly."


# ----------------------------------------------------------------------------------------
# Assertions -- run unconditionally at import, cheap, no MedEvalKit import needed
# ----------------------------------------------------------------------------------------

def _string_constants(path, func_name):
    """Every string literal appearing inside `func_name` of the python file at `path`."""
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [n.value for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    raise AssertionError(f"function {func_name} not found in {path}")


def assert_strings(verbose=False):
    """
    Three independent checks:
      A. the two installed tails differ by EXACTLY the reasoning clause;
      B. our reasoning tail is byte-identical to the pre-revert local edit (the backup), in
         BOTH patched files -- so the existing *_reason dumps remain a valid comparator;
      C. the tails we intend to replace really are what MedEvalKit has on disk right now
         (i.e. the revert to upstream is in place and the wrapper will actually bite).
    """
    out = {}

    # --- A. matched-by-construction -------------------------------------------------
    rebuilt = REASON_CLAUSE + MCQ_DIRECT_TAIL[0].lower() + MCQ_DIRECT_TAIL[1:]
    assert MCQ_REASON_TAIL == rebuilt, (
        "reasoning and direct tails do not differ by exactly the reasoning clause:\n"
        f"  reason  = {MCQ_REASON_TAIL!r}\n  rebuilt = {rebuilt!r}"
    )
    out["A_differ_by_exactly_reasoning_clause"] = True

    # --- B. byte-identical to the pre-revert local edit -----------------------------
    ref_qf = os.path.join(PROMPT_REF_DIR, "question_formats.py.ref")
    ref_du = os.path.join(PROMPT_REF_DIR, "data_utils.py.ref")
    override = os.environ.get("MEDEVAL_PROMPT_BACKUP_DIR")
    if override:
        ref_qf = os.path.join(override, "question_formats.py")
        ref_du = os.path.join(override, "data_utils.py")
    for path, func in ((ref_qf, "get_multiple_choice_prompt"), (ref_du, "construct_prompt")):
        assert os.path.exists(path), f"prompt backup reference missing: {path}"
        consts = _string_constants(path, func)
        assert MCQ_REASON_TAIL in consts, (
            f"our reasoning tail is NOT byte-identical to the backup {func} in {path}\n"
            f"  ours   = {MCQ_REASON_TAIL!r}\n"
            f"  backup = {[c for c in consts if 'boxed' in c]!r}"
        )
    out["B_reason_tail_byte_identical_to_backup"] = True
    out["B_backup_files"] = [ref_qf, ref_du]

    # --- C. upstream tails present on disk ------------------------------------------
    live_qf = os.path.join(MEDEVALKIT, "utils", "question_formats.py")
    live_du = os.path.join(MEDEVALKIT, "utils", "MMMU", "data_utils.py")
    for path, func in ((live_qf, "get_multiple_choice_prompt"), (live_du, "construct_prompt")):
        consts = _string_constants(path, func)
        assert UPSTREAM_REASON_TAIL in consts, f"upstream reasoning tail not found in {path}::{func}"
        assert UPSTREAM_DIRECT_TAIL in consts, f"upstream direct tail not found in {path}::{func}"
        assert MCQ_REASON_TAIL not in consts, (
            f"{path}::{func} still carries the LOCAL EDIT -- MedEvalKit was expected to be "
            "reverted to upstream; refusing to double-patch"
        )
    out["C_medevalkit_is_upstream"] = True
    out["C_live_files"] = [live_qf, live_du]

    if verbose:
        print("prompt-string assertions: PASS")
        for k, v in out.items():
            print(f"  {k}: {v}")
    return out


assert_strings()


# ----------------------------------------------------------------------------------------
# The patch
# ----------------------------------------------------------------------------------------

def _enabled():
    return os.environ.get("MEDEVAL_MATCHED_PROMPT", "0") == "1"


def _log_prompt(text, where):
    """
    Optional provenance: when MEDEVAL_PROMPT_LOG is set, append the first few prompts actually
    produced during a real run, so the dumps can be shown to have used the matched prompt.
    """
    path = os.environ.get("MEDEVAL_PROMPT_LOG")
    if not path:
        return
    cap = int(os.environ.get("MEDEVAL_PROMPT_LOG_MAX", "3"))
    seen = _PROMPT_LOG_COUNT.get(where, 0)
    if seen >= cap:
        return
    _PROMPT_LOG_COUNT[where] = seen + 1
    import json
    rec = {"where": where, "n": seen, "reasoning_env": os.environ.get("REASONING"),
           "prompt": text}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_PROMPT_LOG_COUNT = {}


def _swap_tail(text, is_reasoning, where):
    """Replace upstream's instruction tail with ours. Asserts the tail was actually there."""
    old = UPSTREAM_REASON_TAIL if is_reasoning else UPSTREAM_DIRECT_TAIL
    new = MCQ_REASON_TAIL if is_reasoning else MCQ_DIRECT_TAIL
    assert text.endswith(old), (
        f"{where}: expected prompt to end with upstream tail\n  expected: {old!r}\n"
        f"  got tail: {text[-len(old) - 20:]!r}"
    )
    return text[: len(text) - len(old)] + new


def _rebind_everywhere(old_obj, new_obj):
    """
    MedEvalKit uses `from X import f`, so the callable is bound in each importer's module
    namespace. Rebind every module attribute that IS the original object.
    """
    hits = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        try:
            members = list(vars(mod).items())
        except TypeError:
            continue
        for attr, val in members:
            if val is old_obj:
                setattr(mod, attr, new_obj)
                hits.append(f"{mod_name}.{attr}")
    return hits


_PATCH_STATE = {"applied": False, "rebound": []}

# eval.py sets these in main() BEFORE `from benchmarks import prepare_benchmark`, because
# MedEvalKit's utils/utils.py builds an LLM-judge client at import time and reads them.
# We import `benchmarks` earlier than eval.py does (to have every `from`-import site exist
# before rebinding), so we must seed the same values first -- taken from the forwarded argv
# so the judge client is constructed exactly as it would have been.
_JUDGE_ENV_DEFAULTS = {
    "use_llm_judge": "False",
    "judge_model_type": "openai",
    "judge_model": "None",
    "api_key": "None",
    "base_url": "None",
}


def _seed_judge_env(argv=()):
    argv = list(argv)
    for key, default in _JUDGE_ENV_DEFAULTS.items():
        val = default
        flag = "--" + key
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                val = argv[i + 1]
        else:
            for a in argv:
                if a.startswith(flag + "="):
                    val = a.split("=", 1)[1]
                    break
        os.environ.setdefault(key, val)


def apply_patch(verbose=True):
    """
    Import MedEvalKit's prompt modules (and `benchmarks`, which pulls in every dataset module
    so all `from`-import sites already exist), then swap both prompt builders and rebind.
    """
    if _PATCH_STATE["applied"]:
        return _PATCH_STATE
    if not _enabled():
        if verbose:
            print("MEDEVAL_MATCHED_PROMPT != 1 -> matched-prompt patch NOT applied "
                  "(MedEvalKit runs exactly as upstream)")
        return _PATCH_STATE

    if MEDEVALKIT not in sys.path:
        sys.path.insert(0, MEDEVALKIT)

    # NOTE: `utils/__init__.py` rebinds `utils.MMMU` / `utils.MedXpertQA` from the submodule to
    # the class of the same name, so `import utils.MMMU.data_utils as x` fails on the attribute
    # lookup. importlib.import_module resolves via sys.modules and is unaffected.
    import importlib
    importlib.import_module("benchmarks")  # forces every dataset module to be imported first
    qf = importlib.import_module("utils.question_formats")
    du = importlib.import_module("utils.MMMU.data_utils")

    orig_gmcp = qf.get_multiple_choice_prompt
    orig_cp = du.construct_prompt

    def patched_get_multiple_choice_prompt(question, choices, is_reasoning=False, lang="en"):
        out = orig_gmcp(question, choices, is_reasoning, lang)
        if lang != "en":
            return out  # zh unchanged -- not used by our benchmarks
        out = _swap_tail(out, is_reasoning, "get_multiple_choice_prompt")
        _log_prompt(out, "get_multiple_choice_prompt")
        return out

    def patched_construct_prompt(sample):
        res = orig_cp(sample)
        # Only the multiple-choice branch is matched; `open` items keep upstream's strings.
        if sample.get("question_type") != "multiple-choice":
            return res
        is_reasoning = os.environ.get("REASONING", "False") == "True"
        prompt = _swap_tail(res["empty_prompt"], is_reasoning, "MMMU construct_prompt")
        res["empty_prompt"] = prompt
        res["final_input_prompt"] = prompt
        _log_prompt(prompt, "MMMU_construct_prompt")
        return res

    patched_get_multiple_choice_prompt.__name__ = "get_multiple_choice_prompt"
    patched_construct_prompt.__name__ = "construct_prompt"

    qf.get_multiple_choice_prompt = patched_get_multiple_choice_prompt
    du.construct_prompt = patched_construct_prompt
    rebound = (_rebind_everywhere(orig_gmcp, patched_get_multiple_choice_prompt)
               + _rebind_everywhere(orig_cp, patched_construct_prompt))

    _PATCH_STATE.update(applied=True, rebound=rebound,
                        originals={"get_multiple_choice_prompt": orig_gmcp,
                                   "construct_prompt": orig_cp})
    if verbose:
        print("MATCHED-PROMPT PATCH APPLIED")
        print(f"  reasoning tail: {MCQ_REASON_TAIL}")
        print(f"  direct tail   : {MCQ_DIRECT_TAIL}")
        print(f"  rebound {len(rebound)} reference(s): {rebound}")
    return _PATCH_STATE


# ----------------------------------------------------------------------------------------
# --verify : print the exact prompt both arms produce, no model load
# ----------------------------------------------------------------------------------------

def verify(mmmu_subject="Basic_Medical_Science", mmmu_idx=0, medxpert_idx=0):
    os.environ["MEDEVAL_MATCHED_PROMPT"] = "1"
    os.chdir(MEDEVALKIT)
    if MEDEVALKIT not in sys.path:
        sys.path.insert(0, MEDEVALKIT)

    assert_strings(verbose=True)
    print()
    _seed_judge_env()
    apply_patch(verbose=True)
    print()

    import importlib
    from datasets import load_dataset
    process_single_sample = importlib.import_module("utils.MMMU.data_utils").process_single_sample
    ev = importlib.import_module("utils.MMMU.eval_val")          # the real MMMU call site
    mxq = importlib.import_module("utils.MedXpertQA.MedXpertQA")  # the real MedXpert call site

    report = {"assertions": {}, "items": {}}

    # ---- MMMU: use eval_val's OWN binding, i.e. the real call site -------------------
    ds = load_dataset("MMMU/MMMU", mmmu_subject, split="validation")
    raw = process_single_sample(dict(ds[mmmu_idx]))
    mmmu_prompts = {}
    for arm, flag in (("reasoning", "True"), ("direct", "False")):
        os.environ["REASONING"] = flag
        s = ev.construct_prompt(dict(raw))
        mmmu_prompts[arm] = s["final_input_prompt"]
    report["items"]["MMMU-Medical-val"] = {"id": raw["id"], "subject": mmmu_subject,
                                           "question_type": raw["question_type"],
                                           "prompts": mmmu_prompts}

    # ---- MedXpertQA: use MedXpertQA's OWN binding -----------------------------------
    mx = load_dataset("TsinghuaC3I/MedXpertQA", "MM", cache_dir="./datas/MedXpertQA")["test"]
    m = dict(mx[medxpert_idx])
    choices = [f"{k}. {v}" for k, v in m["options"].items()]
    mx_prompts = {}
    for arm, flag in (("reasoning", True), ("direct", False)):
        mx_prompts[arm] = mxq.get_multiple_choice_prompt(m["question"], choices, flag)
    report["items"]["MedXpertQA-MM"] = {"id": m["id"], "prompts": mx_prompts}

    # ---- print + assert the two arms differ by exactly the reasoning clause ---------
    for bench, info in report["items"].items():
        print("=" * 100)
        print(f"{bench}   item id = {info['id']}"
              + (f"   question_type = {info.get('question_type')}" if info.get("question_type") else ""))
        for arm in ("reasoning", "direct"):
            print("-" * 100)
            print(f"--- {bench} / {arm.upper()} ARM ---")
            print(info["prompts"][arm])
        print()

    print("=" * 100)
    print("MATCHED-PAIR ASSERTIONS")
    ok = True
    for bench, info in report["items"].items():
        r, d = info["prompts"]["reasoning"], info["prompts"]["direct"]
        # reasoning == direct with the clause spliced in before the final tail
        expect_r = d[: len(d) - len(MCQ_DIRECT_TAIL)] + MCQ_REASON_TAIL
        same_body = r[: len(r) - len(MCQ_REASON_TAIL)] == d[: len(d) - len(MCQ_DIRECT_TAIL)]
        r_tail_ok = r.endswith(MCQ_REASON_TAIL)
        d_tail_ok = d.endswith(MCQ_DIRECT_TAIL)
        exact = (r == expect_r)
        diff_is_clause = (len(r) - len(d) == len(REASON_CLAUSE))
        for name, val in (("bodies identical", same_body), ("reasoning tail exact", r_tail_ok),
                          ("direct tail exact", d_tail_ok),
                          ("reasoning == direct + clause", exact),
                          (f"length delta == len(clause)=={len(REASON_CLAUSE)}", diff_is_clause)):
            print(f"  [{'PASS' if val else 'FAIL'}] {bench}: {name}")
            ok = ok and val
        report["assertions"][bench] = {"bodies_identical": same_body,
                                       "reasoning_tail_exact": r_tail_ok,
                                       "direct_tail_exact": d_tail_ok,
                                       "reasoning_equals_direct_plus_clause": exact,
                                       "length_delta_equals_clause_len": diff_is_clause}
        assert exact, f"{bench}: arms differ by more than the reasoning clause"
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    report["overall_pass"] = ok
    return report


# ----------------------------------------------------------------------------------------
# main : delegate to MedEvalKit's eval.py
# ----------------------------------------------------------------------------------------

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--verify" in argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--verify", action="store_true")
        ap.add_argument("--mmmu_subject", default="Basic_Medical_Science")
        ap.add_argument("--mmmu_idx", type=int, default=0)
        ap.add_argument("--medxpert_idx", type=int, default=0)
        a = ap.parse_args(argv)
        rep = verify(a.mmmu_subject, a.mmmu_idx, a.medxpert_idx)
        return 0 if rep["overall_pass"] else 1

    if argv and argv[0] == "--":
        argv = argv[1:]

    # MedEvalKit resolves ./datas/... relative to CWD and imports LLMs/benchmarks by name
    os.chdir(MEDEVALKIT)
    if MEDEVALKIT not in sys.path:
        sys.path.insert(0, MEDEVALKIT)

    _seed_judge_env(argv)
    apply_patch(verbose=True)

    import importlib
    medeval_eval = importlib.import_module("eval")
    sys.argv = ["eval.py"] + argv
    medeval_eval.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
