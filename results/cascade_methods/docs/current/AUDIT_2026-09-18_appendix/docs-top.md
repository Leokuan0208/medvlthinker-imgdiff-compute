# Audit report -- entry-point docs of medvlthinker-imgdiff-compute
Agent: docs-top. Started 2026-09-20 15:07 UTC. In progress -- appending as findings are confirmed.
Reused predecessor outputs in this dir: structure_check.py/.out, structure_missing_py.txt,
structure_missing_sh.txt, structure_mentions_missing.txt, results_readme_unindexed_artifacts.txt,
results_readme_sept_artifacts.txt, wt_status.out/.py, main_lstree.txt, mergetree_old.txt, read_sept.py.

## Status: SETUP DONE, gathering findings now.

(sections filled below as confirmed)

## Item 1: CLAUDE.md §0 canonical-numbers table vs artifacts/cascade_selector_rerun_2026-08-05.json

Verified via `disjoint` arm (CLEAN, disjoint-trained verifier) of cascade_selector_rerun_2026-08-05.json
(python dump, see above). Table row-by-row:

| CLAUDE.md row | claimed | artifact key | artifact value | match? |
|---|---|---|---|---|
| always-7B | 0.5971 | per_arm.disjoint.macro_acc.always_7b | 0.5971 | EXACT |
| always-32B-reasoning (unmatched) | 0.5974 | ...always_32b_reasoning | 0.5974 | EXACT |
| always-32B-reasoning (prompt-matched) | 0.6250 | **NOT IN THIS FILE AT ALL** | -- | **NO SOURCE in the cited artifact** |
| always-32B-direct | 0.6567 | ...always_32b_direct | 0.6567 | EXACT |
| oracle-mode-32B | 0.6573 (+0.0006) | ...oracle_mode_32b | 0.6573 | EXACT (delta is plain arithmetic, no CI in artifact for this row, consistent w/ no CI shown) |
| accuracy-max (clean, canonical) | 0.6575 | ...method_accuracy_max_veto | 0.6575 | EXACT |
|   vs reasoning | +0.0601 [+0.0499,+0.0700] WIN | deltas.method_accuracy_max_veto.always_32b_reasoning | delta=0.0601 lo=0.0498 hi=0.0703 | delta EXACT; **lo/hi off by 0.0001/0.0003** |
|   vs direct | +0.0008 [-0.0022,+0.0037] TIE | deltas...always_32b_direct | delta=0.0008 lo=-0.0022 hi=0.0037 | EXACT |
| accuracy-max + frozen 8-seed selector | 0.6590 | per_arm.**ens8_scaled**.macro_acc.method_accuracy_max_veto | 0.659 | EXACT (identifies which arm "frozen 8-seed selector" means -- ens8_scaled, not ens8_rank which gives 0.6571) |
|   vs reasoning | +0.0615 [+0.0514,+0.0715] WIN | ens8_scaled.deltas...reasoning | delta=0.0615 lo=**0.0516** hi=0.0715 | delta/hi EXACT; **lo off by 0.0002** |
|   vs direct | +0.0023 [-0.0010,+0.0054] TIE | ens8_scaled.deltas...direct | delta=0.0023 lo=-0.001 hi=0.0054 | EXACT |
| compute-lean (clean) | 0.6443 | disjoint.macro_acc.method_compute_lean | 0.6443 | EXACT |
|   vs reasoning | +0.0469 WIN (no CI given) | disjoint.deltas...reasoning | delta=**0.0468** | off by 0.0001 |
|   vs direct | -0.0124 [-0.0191,-0.0062] LOSS | disjoint.deltas...direct | delta=-0.0124 lo=**-0.0188** hi=**-0.006** | delta EXACT; **lo off by 0.0003, hi off by 0.0002** |

**FINDING A (MED):** every CI *point delta* in the table is exact-to-4dp against the artifact, but
**5 of 6 CI bounds checked are off by 0.0001-0.0003** (never large, never flips a verdict). Consistent
with the table having been transcribed from an earlier bootstrap draw of the same script/seed rather
than the exact file now on disk (the file's own `seed: 20260805` is fixed, so a byte-identical rerun
should reproduce these bounds exactly -- it currently does not, to the 4th decimal). Not fabrication,
not verdict-changing, but the CIs printed in CLAUDE.md do not currently reproduce from the cited file.

**FINDING B (HIGH): "always-32B-reasoning (prompt-matched) 0.6250" has NO source in
cascade_selector_rerun_2026-08-05.json** -- the file the whole table is attributed to via the single
"Source:" line. The number is real and traceable (found in
`results/cascade_methods/docs/current/COMPREHENSIVE_WRITEUP_2026-08-03.md:106,1641,2679-2681,2712,3805`
and `CHEAP_VERIFIER_ON_7B_2026-08-16.md:275`, both dated *before* the 08-05 rerun), but CLAUDE.md's
attribution line is misleading for this one row: a reader who opens only the named artifact cannot find
this number. This is exactly the "labelling" failure CRITICAL RULE 7 was promoted to stop.

## Item 1 (Ceilings paragraph): "+0.0301 / +0.0091 / +0.0661 / 1.3% / sel_eff identity 5.6e-17"

**None of these five numbers are in cascade_selector_rerun_2026-08-05.json either** (grepped, absent).
Real sources, found by search:
- **+0.0301** (perfect selection over current 8-pool): `stats_recertification_2026-08-11.json`
  HEADLINE.part1_oracle_noise -- "moving the perfect-selection ceiling from +0.0301 to +0.0276" --
  **and that file's own text says the +0.0276 correction is NOT established** (per-cell differences
  inside the ±0.008 serving-config noise band), so +0.0301 (not +0.0276) is correctly the live number.
- **+0.0091** (perfect coverage / infinite sampling): `results/cascade_methods/docs/current/
  BEAT32B_ROUND_2026-08-10.md:186,644` ("open arm at iid sampling ceiling ... +0.00915" / "+0.0091 macro
  ... unreachable on 2 of 3 open cells at any N"). Doc-level, traced back informally to per-cell
  capture-recapture numbers in that doc; no single top-level JSON key found holding this exact figure.
- **+0.0661 / 1.3% converted** (perfect 7B-vs-32B routing): `BEAT32B_ROUND_2026-08-10.md:231,631`
  ("Σ p10 = 0.529 ⇒ macro +0.0661 ... Realised: +0.00083 ... 1.3%"), derived from per-cell p10 values,
  presumably in `headroom_percell_2026-08-10.json` (not individually confirmed to 4dp in the time
  available -- **NOT FULLY VERIFIED**, flagged below).
- **sel_eff identity, err 5.6e-17**: confirmed in THREE artifacts --
  `coverage_diagnosis_2026-08-10.json:19,1709`, `resolution_sweep_2026-08-13.json:4450`,
  `method_inventory_2026-08-11.json:4237`, all say "exact identity, max |err| 5.6e-17" and all warn
  against the additive form `greedy + sel_eff*(oracle-greedy))` (over-predicts by +0.09 to +0.11) --
  this matches CLAUDE.md's own warning verbatim.

**FINDING C (MED):** the Ceilings paragraph is numerically real (nothing fabricated) but its 5 numbers
come from **at least 4 different files, none of which is cascade_selector_rerun_2026-08-05.json**, and
CLAUDE.md gives this paragraph no inline citation at all (the "Source:" line two paragraphs up only
covers the table). A reader following CLAUDE.md's own rule ("recompute or say not measured, name the
file") cannot currently name the file for +0.0301/+0.0091/+0.0661/sel_eff from §0 text alone.

## Item 1 (armcombine_mcqonly paragraph): VERIFIED CLEAN, EXACT MATCH

`armcombine_mcqonly_2026-08-11.json`, `fixed_mcq_arm_policies.method_accuracy_max_veto`:
delta=0.00119(~0.0012) lo=0.0009 hi=0.00148(~0.0015) x_direct_as_charged=0.9773(~0.977x)
guardrail_flags=[] macro_leave_one_out.per_dropped_cell.PMC_VQA=0.0.
`...method_accuracy_max_fusion`: delta=0.00169(~0.0017) lo=0.00126(~0.0013) hi=0.00212(~0.0021)
x_direct_as_charged=1.0274(~1.027x) guardrail_flags=[] PMC_VQA-dropped=0.0.
**Every number in CLAUDE.md's "one CI-clean win" paragraph matches this artifact exactly to 4dp,
including the "PMC_VQA is 100% of the effect, drops to exactly 0.000" claim.** No finding here --
this is the one CLAUDE.md passage that is a model of correct sourcing.

## Item 2: CLAUDE.md internal contradictions vs real git/tree state

Ran from the audit-2026-09-18 worktree (git refs/objects are shared across worktrees with MAIN, so
`git log`/`git branch -vv`/`git ls-files` here are equivalent to running on MAIN's `main` branch;
confirmed both `main` and `audit-2026-09-18` are checked out at the same commit `e2aa6f0`).

- **`git branch -vv`**: `main` -> `e2aa6f0 [origin/main: ahead 78]`. **Confirms the briefing's expected
  78 unpushed commits exactly.**
- **`git log origin/main..main --oneline | wc -l` = 78.** CONFIRMED.
- **`git ls-files | wc -l` = 1916**, matches predecessor's `main_lstree.txt` (1916 lines) exactly --
  main has not moved since the predecessor snapshotted it.
- **`git log --oneline main | wc -l` = 291 total commits.** `8cdefef` is commit #129 of 291 counting
  from HEAD, dated 2026-07-02; HEAD (`e2aa6f0`) is dated 2026-09-17. **128 commits happened after
  8cdefef, the most recent 2 days before this audit.**

**FINDING D (CRITICAL): CLAUDE.md §7 is flatly false and self-contradicting.**
§7 states: *"The July/Lingshu work is not in git. Last commit `8cdefef` (2026-07-02)... The method, its
inputs and its outputs currently exist on one disk."* This is contradicted by:
  (a) §0's own line two paragraphs above it: *"Preservation: ✅ committed, ✅ pushed..."*
  (b) reality: HEAD is `e2aa6f0`, 128 commits and 77 days after `8cdefef`; predecessor's blob-hash
      diff (`wt_status.out`) found **zero modified, zero missing tracked files** -- the working tree
      matches the last commit exactly, i.e. everything IS committed (though 78 commits are unpushed,
      matching §0's separate "44 untracked .py files" framing only partially -- see below).
§7's file is simply not updated since roughly early July; it predates ~5 sub-eras of the project
(the whole probe-verifier era postdates it entirely). **Proposed fix: delete the "July/Lingshu work is
not in git" paragraph from §7 outright** (it is not just stale, it actively misleads a reader into
thinking work could be lost) and replace with a one-line pointer to `git log` / `git status` as the
live source of truth, per the project's own "regenerate, don't duplicate" numbers doctrine.

**FINDING E (HIGH): the "44 untracked .py files" claim in §7 is also stale/unverifiable as stated** --
predecessor's `wt_status.out` (live untracked scan against the current tree) found **163 untracked
files under `src/`** (mostly `src/graphify-out/` cache junk -- 152 of those are
`src/graphify-out/cache/ast/*.json` code-graph cache, not real work) plus **162 untracked files under
`results/cascade_methods/artifacts/`** (all `_..._parts/*.npz` intermediate caches) and 1 untracked
root file (`env_backup_2026-06-05.txt`). None of these breaks down to "44" cleanly; the "44" figure is
from early July and the untracked set has completely turned over since (different files, different
count). **Not independently re-derivable from the current tree; flag as unverifiable-as-written.**

**FINDING F (CRITICAL): CLAUDE.md §4.1 "`results/cascade_methods/artifacts/` holds ~107 numeric `.json`
outputs (gitignored, regeneratable)" is factually wrong on BOTH claims, contradicted by `.gitignore`
itself:**
  - `.gitignore:15-19`: `results/*` is ignored, but line 19 explicitly un-ignores it:
    `!results/cascade_methods/artifacts/`. **Artifacts are NOT gitignored -- they are the one
    deliberately-tracked exception.**
  - Count: **761 tracked files** live under `results/cascade_methods/artifacts/` per `main_lstree.txt`
    (`grep -c "results/cascade_methods/artifacts/" main_lstree.txt` = 761); **353 top-level `.json`
    files** currently on disk at that path (`structure_check.out`, predecessor's run) -- not ~107 by
    either count. §0's own Preservation line ("`results/` has 269 tracked files so the numbers travel
    with a push") is *also* wrong given the true count of 761+ (269 may be a much older snapshot).
  Note artifact-count-by-month from `structure_check.out`: 177 Aug-dated, 135 undated, 30 Sep-dated,
  11 Jul-dated .json files at top level -- none of this supports "~107" under any slicing tried.

**FINDING G (MED): §5 "runners/ 38 shell launchers" is stale.** Real count: **147 `.sh` files**, 318
total directory entries (`ls runners/*.sh | wc -l` = 147). Off by ~4x.

**FINDING H (MED): §5 "progress/ 13 dated daily diaries (June 17 -> July 8)" is stale.** Real count:
**24 files**, spanning June 17 through **August 17** (`progress_August_17.md` exists), not July 8.
11 diaries postdate the claimed end of the range (July 29, July 30, and 9 August entries).

## Item 2 (cont'd): cap320 / verifier glossary / thread-count / backup

**FINDING I (HIGH): §3 glossary "cap320 is the chosen operating point" is superseded by the project's
own newest planning doc, three days before this audit.** `git show
worktree-lit-domain-package:results/cascade_methods/docs/current/OPEN_EXPERIMENTS_2026-09-17.md`
line 5: *"§1.0 -- regenerate the open-text pools at full resolution -- is DECIDED (Leo, 2026-09-17)"*;
line 212-219: *"`cap320` is an inherited default from the visual-token-pruning era... Nobody is choosing
it now. `src/labeling/run_openvqa.py:65` sets `--cap default=\"cap320\"`"*. This doc lives only on the
unmerged `worktree-lit-domain-package` branch (merge-base analysis: item 7 below), so it has not
reached CLAUDE.md, but the underlying DECISION is dated and real, not speculative.

**FINDING J (HIGH): §3 glossary "Verifier -- a small LoRA-fine-tuned model... `lora_verifier_pooled4`"
describes the June/August verifier only.** The live (mid-Aug onward) verifier is an **MLP probe on
frozen hidden states** (not LoRA-fine-tuned at all -- no adapter, no backprop through the generator),
`ckpts/train/genframe_head_pooled_ens_v2/`, headline `head_final_stack_PVFIXED_2026-09-13.json`. §3
does not mention this at all; a reader relying on the glossary would misunderstand the mechanism of the
live method entirely (see item 3 below).

**FINDING K (LOW/informational): "CPU thread count (+0.0048)" landmine (§0) is a different,
older measurement than the one now on disk.** Source of +0.0048:
`results/cascade_methods/docs/current/PROGRESS_REPORT_2026-07-06_to_08-14.md:398` (Aug-era). The
project has SINCE done a much more rigorous characterization for the current probe-training pipeline:
`repro_threading_2026-09-13.json` -- `thread_sweep_spread = 0.003279` across thread counts {1,2,4,8}
on the `pooled_ens` arm, **deterministic given (code, thread count)** (`RETRACTION` field: an earlier
2-run test wrongly concluded nondeterminism; 5 further bitwise-identical replicates at 4 threads
closed it -- `VERDICT: CLOSED. ... What moves the number is the THREAD COUNT`). `head_final_stack.py`
now stamps thread count into every artifact as a direct consequence. CLAUDE.md's landmine bullet does
not mention this newer, sharper finding or its artifact/commit (`41175f1 CLOSED: the refactor was
innocent; it was the thread count nobody wrote down`).

**FINDING L (CRITICAL, data-loss risk): the §0 backup claim is now materially stale.**
`/data/dan/backups/medvlthinker-imgdiff-compute/2026-08-10/` exists (`ls -la`, confirmed) and contains
exactly two things: `ckpts_train/{genframe_head_ens8,lora_verifier_disjoint}` and `feats_hidden/`
(dated Aug 4). It does **NOT** contain:
  - `ckpts/train/genframe_head_pooled_ens_v2/` (**85M**, `du -sh` -- the live headline probe-verifier
    checkpoint the briefing names as the current deployed artifact) -- absent from the backup's
    `ckpts_train/` listing (`ls` shows only the two Aug dirs above).
  - `ckpts/openvqa/cheap_lingshu7b/` (**728M**, `du -sh`) -- the backup has no `ckpts/openvqa/`
    top-level category at all.
  - `feats_hidden/` currently totals **65G** on disk (`du -sh`) with **256 entries dated after
    2026-09-01** (`find -newermt`); the backup's `feats_hidden/` snapshot predates all September work
    (dated Aug 4 on the backup filesystem). Per the memory note, 181 of the *live* feats_hidden files
    are intentional symlinks to `/data/dan/archive/`, but the 256 September-dated entries include real
    (non-archived) local files not verified to be covered elsewhere.
CLAUDE.md's line **"Preservation: ✅ committed, ✅ pushed, ✅ inputs backed up (2026-08-10)"** was true
on 2026-08-10 and is presented in the present tense as if still current; it predates the entire live
probe-verifier era (mid-Aug onward) by design, so it silently no longer covers what the project itself
calls "the live work." Combined with 78 unpushed git commits (Finding D data), **the September
probe-verifier checkpoints and generation dumps currently exist on exactly one disk with no backup.**

## Item 3 part A: the two NEW audit facts, independently confirmed

**FINDING M (CRITICAL): the judge that produced every open-text label is MedVLThinker-32B, NOT
Lingshu-32B, and no entry doc says so.**
- `src/labeling/run_judge.py:4` (module docstring): *"text LLM (default MedVLThinker-32B, Qwen2.5-32B
  backbone -- NOT the model scored in the Lingshu cascade)"* -- the script's own author-comment already
  flags this as a gotcha.
- `src/labeling/run_judge.py:21`: `ap.add_argument("--judge_model",
  default="/data/dan/weights/MedVLThinker-32B-RL_m23k")`.
- Checked all 21 `runners/*.sh` that invoke `run_judge.py` for a `--judge_model` override: **zero**
  pass one (verified by grep for `--judge_model ` -- distinct from the unrelated
  `--judge_model_type openai --judge_model None` flags used by MedEvalKit's *own* harness invocations
  in 15 other runners, which is a different script/flag namespace and not a counter-example).
  **Every judge label in the live probe-verifier headline (`head_final_stack_PVFIXED_2026-09-13.json`,
  +0.0737 macro) was produced by MedVLThinker-32B**, a model from the project's OWN prior (June) era,
  not the Lingshu-32B family the project switched to as its "faithful, publicly-anchored" judge in
  §0 CRITICAL RULE 7 / §1 arc-step 7. None of CLAUDE.md, README.md, RESULTS.md, PROJECT_OVERVIEW.md,
  READING_GUIDE.md, STRUCTURE.md, or `results/cascade_methods/README.md` mentions this. The memory
  file `domain-guide-package.md` gets it actively wrong (see item 6 below: "same-family Lingshu-32B
  judge").

**FINDING N (confirmed, not new but independently re-derived): headline dual-currency numbers.**
`/home/jamesyang/.claude/jobs/37d73e6f/tmp/em-rescore/em_rescore_pooled_probe_2026-09-18.json`,
`macro` block (18,452 held-out questions, 8 benchmarks, frozen 24-probe ensemble):
  - `delta_judge.macro = 0.07372` (reproduces the `head_final_stack_PVFIXED_2026-09-13.json` headline
    +0.0737 to `deviation_vs_recipe_macro = 0.000113`, i.e. matches to 1e-4);
    `ci_image_clustered_within_benchmark = [0.0608, 0.0864]`, `verdict WIN` both benchmark- and
    image-clustered levels.
  - `delta_em.macro = 0.00471` (**+0.0047**), `ci_image_clustered = [-0.00774, +0.01711]`
    (**[-0.0077,+0.0171]**), `verdict TIE` both levels. **3 WIN / 3 TIE / 2 LOSS** at the per-benchmark
    level is not independently re-derived here (time) but the top-line numbers match the briefing
    exactly.
  - `delta_strict_em.macro = 0.01032` (+0.0103), verdict TIE. `delta_token_f1.macro = 0.01565`
    (+0.0156), `ci_image_clustered = [0.00439, 0.02652]` (**[+0.0044,+0.0265]**).
  - Benchmark-level (n=8) CI for the judge macro: `ci_benchmark_level_n8 = [0.0260, 0.1165]` --
    matches briefing exactly.
**No entry doc mentions the EM-currency result at all; every entry doc that quotes a probe-verifier
number (only PROJECT_OVERVIEW.md and STRUCTURE.md do, in passing -- see below) quotes ONLY the judge
number, which per this project's own "mixed-currency" failure mode (BRIEFING.md "Project's own known
failure modes") is exactly the kind of single-currency claim the project warns against making.**

## Item 3 part B: what the entry docs simply never mention

Grepped all 6 non-CLAUDE.md entry docs + `results/cascade_methods/README.md` for
`2026-09|probe|genframe|head_final_stack`:
- **README.md, RESULTS.md, READING_GUIDE.md, INCONSISTENCIES.md: ZERO mentions.** The live method
  (probe verifier) and every one of its artifacts, docs (`AUDIT_2026-09-12.md`,
  `PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`, `OPENTEXT_FULL_RUNDOWN_2026-09-04.md`) is absent.
- **PROJECT_OVERVIEW.md: 1 hit** (line 409, a backward-looking list of killed probe-adjacent ideas --
  not a forward reference to the live method).
- **STRUCTURE.md: 3 hits**, all incidental (a `feats/` gitignore comment, a script one-liner, a
  section header) -- **no entry for `head_final_stack.py`, `extract_generator_hidden.py`, or
  `genframe_head_pooled_ens_v2` despite `structure_check.py` showing 110 `training_methods/*.py` files
  unmentioned in STRUCTURE.md** (see item 4).
**Net: a new reader following any entry doc except CLAUDE.md's own §0 stub would not learn that the
project pivoted a second time in mid-August, would not find the +0.0737/+0.0047 headline, and would not
learn the judge-model identity finding.** CLAUDE.md §0 is itself dated 2026-08-11, i.e. also predates
the live era by 5-6 weeks, and does not mention it either -- confirming the BRIEFING's framing
("CLAUDE.md §0 is a month stale") from primary evidence.

## Item 3 part C: same "13 dated diaries" staleness duplicated in READING_GUIDE.md, plus new counts

`READING_GUIDE.md:3-4`: *"7 root docs, a 1,972-line retrospective, 14 current writeups, 20 archived
ones, **13 dated progress diaries**, an IEEE paper, 3 HTML decks, a 68-idea backlog, and ~199 Python
files."* Checked each clause against disk:

| claim | real | source | verdict |
|---|---|---|---|
| 7 root docs | 7 | `ls *.md` at repo root | MATCH |
| 1,972-line retrospective | **2,672 lines** | `wc -l PROJECT_RETROSPECTIVE_2026-07-29.md` | STALE (+700 lines, +35%) |
| 14 current writeups | **28** `.md` files | `find docs/current -maxdepth 1 -name '*.md'` | STALE (2x) |
| 20 archived ones | 20 | `ls docs/archive_mcq` | MATCH |
| 13 dated progress diaries | **24** | `ls progress/` | STALE (same bug as CLAUDE.md §5, Finding H) |
| ~199 Python files | **600** | `find src -name '*.py'` | STALE (3x) |

**FINDING O (MED): READING_GUIDE.md's orientation paragraph is stale on 3 of 6 checkable counts**,
independently reproducing the same "13 diaries" error found in CLAUDE.md §5 -- textbook instance of the
project's own documented failure mode ("corrections made in a new file but not propagated to older
copies... fix it everywhere it appears").

## Item 6: memory files -- staleness, contradictions, leaked tokens

- **`domain-guide-package.md` (modified 2026-09-18T09:52Z), line "the eight-benchmark probe has never
  been scored in exact-match currency, only under the same-family Lingshu-32B judge":**
  **FINDING P (HIGH): "same-family Lingshu-32B judge" is simply wrong** -- per Finding M the judge is
  MedVLThinker-32B (`run_judge.py` default, no runner override), not Lingshu-32B, so it is not
  "same-family" with the Lingshu-7B generator at all -- it's a leftover model from the project's PRIOR
  era. Additionally "has never been scored in exact-match currency" is now stale: the EM re-score
  (`em_rescore_pooled_probe_2026-09-18.json`) was completed the same day, timestamped ~13 minutes after
  this memory note (10:05 vs 09:52) -- an artifact of two parallel sessions racing, not a contradiction
  Leo could have caught, but the memory note now needs a one-line update either way.
- **`cascade-research-loop-state.md` (modified 2026-07-29T22:48Z, 70,348 bytes): confirmed silent on
  the probe-verifier era.** Grep for `head_final_stack|genframe_head_pooled|probe verifier|hidden.state
  probe` = **0 hits** in the entire file. Its own last entries (2026-06-29 "2-seed honesty correction")
  are two eras behind the live method; `MEMORY.md`'s index line for it ("autonomous training-free
  cascade research loop... key results & the cheaper-strong-leg lever") gives no staleness warning to a
  session that opens it expecting current findings.
- **`MEMORY.md` points to `literature/DOMAIN_GUIDE_2026-09-16.md`** with no caveat that this path
  **only exists on the unmerged `worktree-lit-domain-package` branch** (confirmed: not present in
  `main_lstree.txt`'s 1916 tracked paths on `main`). A session that `cat`s this path from MAIN's working
  tree (checked out at `main`) will get "file not found" with no explanation from the memory file
  itself.
- **Leaked-token grep (per BRIEFING item 6): CLEAN.** `git grep -n -I -E "hf_[A-Za-z0-9]{20,}"` over
  all 1916 tracked files at `main`/`e2aa6f0`: **0 hits**. Same pattern over the memory dir
  (`grep -rn -I -E "hf_[A-Za-z0-9]{20,}" memory/`): **0 hits**. Additionally checked (not required, done
  for diligence) `sk-[A-Za-z0-9]{20,}`, `ghp_[A-Za-z0-9]{20,}`, `AKIA[A-Z0-9]{16}` against both the
  tracked repo and the memory dir: **0 hits on all three, both locations.** No leaked tokens found.
- Other memory files (`gate-bakeoff-verdict.md`, `lingshu-baseline-eval-protocol.md`,
  `verify-foundational-facts.md`, `no-abstention-research.md`, `disk-usage-hard-limit.md`,
  `feats-hidden-archived-symlinks.md`) not individually contradicted by anything found in this audit;
  not exhaustively re-verified line-by-line (time budget) -- **listed as checked-but-not-adversarially-
  verified**, not "clean" in the strong sense used elsewhere in this report.

## Item 7: does merging `worktree-lit-domain-package` into `main` conflict?

Predecessor already ran the read-only `git merge-tree --write-tree main worktree-lit-domain-package`
equivalent and saved it to `mergetree_old.txt` (21,262 lines). Re-verified branch tips are unchanged
since that run (both `main` and `audit-2026-09-18` still at `e2aa6f0`; `worktree-lit-domain-package`
still at `cf0d1ac`), so the saved merge-tree result is current.

**Result: CLEAN, no conflicts.** `grep -c -i conflict mergetree_old.txt` = 0 (2 raw hits on the word
"conflict" are both inside the *prose content* of the added `OPEN_EXPERIMENTS_2026-09-17.md`, not
CONFLICT stanzas). The merge-tree output contains exactly:
  - **1 `merged` entry**: `.gitignore`, a clean 4-line auto-merge appending the `literature/papers`
    ignore-with-exception rule -- no conflict markers, both sides' 61-line prefix identical.
  - **7 `added in remote` entries** (files that exist only on the branch, added cleanly, nothing to
    conflict with on `main`): `literature/DOMAIN_GUIDE_2026-09-16.{docx,html,md}`,
    `literature/MANIFEST.md`, `literature/README.md`, `literature/references.bib`,
    `results/cascade_methods/docs/current/OPEN_EXPERIMENTS_2026-09-17.md`.
  - **0 `added in local`, 0 `removed in`, 0 `CONFLICT` stanzas.**
Merging `worktree-lit-domain-package` into `main` today would apply cleanly with no manual conflict
resolution required -- it is purely additive (7 new files + one clean `.gitignore` hunk). This does
**not** by itself mean it is *safe* to merge (e.g. the branch predates 128 of `main`'s current commits
per the merge-base `e01de8c`, 2026-08-17, so its prose references an older numbers-state and would need
a content review, not just a conflict check) -- only that **git itself sees no conflicting edits.**

## Item 4: STRUCTURE.md vs the real tree (predecessor's `structure_check.py`/`.out`, re-validated:
main tree unchanged since predecessor's run -- `git ls-files` = 1916 = `main_lstree.txt` line count)

- **`src/` .py files on disk: 600** across `analysis(25) cascade(13) cascade_methods(378) data_prep(11)
  gate(2) labeling(26) legacy_retrieval(1) reporting(10) sweep(5) training_methods(123)
  verifier_arch(6)`.
- **NOT mentioned anywhere in STRUCTURE.md: 401 of 600 (67%)** -- full list in
  `structure_missing_py.txt`. By dir: `cascade_methods 256, training_methods 110, analysis 9,
  cascade 8, data_prep 6, verifier_arch 6, labeling 3, reporting 3`. **`training_methods` is the
  September probe-verifier code** -- 110 of its 123 files (89%) are undocumented in the file index,
  consistent with item 3's finding that STRUCTURE.md predates the live method.
- **`runners/*.sh` NOT mentioned: 141 of 147 (96%)** -- full list in `structure_missing_sh.txt`.
- **STRUCTURE.md mentions 5 files/paths that no longer exist as named** (`structure_mentions_missing.txt`):
  `detection.json` (not found anywhere), `paper/cvgip2026_draft.md` and
  `paper/manuscript_final_2026-07.md` (both moved to `paper/archive/`, STRUCTURE.md's paths are stale
  by one directory level), `rt_cascade_cap320.json` (not found anywhere), `subset.csv` (not found
  anywhere). Minor (5 of 277 checkable tokens = 1.8% dead-link rate) but real.
- **FINDING Q (HIGH):** STRUCTURE.md, whose own header claims to be "the live per-file index", is
  missing two-thirds of the Python files and 96% of the shell runners actually on disk, overwhelmingly
  concentrated in exactly the parts of the tree (`training_methods/`, most of `cascade_methods/`) that
  carry the live method. This is the same "index doc lags the work" failure as items 3 and 5.

## Item 5: results/cascade_methods/README.md indexing gaps (predecessor's run, re-validated clean --
tree unchanged)

- **`docs/current/*`: 31 files on disk, 25 NOT indexed** in the README (list in
  `structure_check.out`, reproduced fully in the terminal above) -- includes **`AUDIT_2026-09-12.md`**
  and **`PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`**, the two docs BRIEFING.md specifically names as
  essential live-era reading, plus `OPENTEXT_FULL_RUNDOWN_2026-09-04.md`,
  `OPENTEXT_CORRECTIONS_2026-08-19.md`, `TERMINOLOGY_2026-08-24.md`, `HYPERPARAMETERS_2026-08-15.md`,
  and 19 others back to 2026-07-06.
- **Top-level `artifacts/*.json`: 353 on disk, 328 NOT indexed (93%)** (full list,
  `results_readme_unindexed_artifacts.txt`). By month: 177 Aug / 135 undated / 30 Sep / 11 Jul; of
  those, the README indexes only 20 undated + 5 Jul-dated files -- **0 of the 177 August files and 0
  of the 30 September files are indexed**, including every headline probe-verifier artifact
  (`head_final_stack_PVFIXED_2026-09-13.json`, `head_final_stack_qwen_2026-09-13.json`,
  `head_final_stack_medgemma_ALL8_2026-09-16.json`, `coverage_sc16_ci_ALL_2026-09-16.json`,
  `repro_threading_2026-09-13.json`, and 25 more -- full list `results_readme_sept_artifacts.txt`).
- **No dangling references the other direction**: every `.json`/`.md` filename the README itself
  names (24 `.json` tokens, 39 `.md` tokens) exists on disk -- the README does not point anyone at a
  missing file, it simply stopped being extended after roughly the August cutoff its own indexed set
  implies.
- **FINDING R (HIGH):** `results/cascade_methods/README.md`'s indexing has not been extended past
  (roughly) the earliest August artifacts. 100% of September work and effectively all of August's 177
  artifacts are invisible to a reader who trusts this README as the artifact index -- including the
  entire live-method evidence base.

## Findings table (summary, severity-ordered)

| id | sev | claim | evidence | proposed fix |
|---|---|---|---|---|
| D | CRITICAL | CLAUDE.md §7 "July/Lingshu work not in git, last commit 8cdefef" | `git log`: HEAD=e2aa6f0 (2026-09-17), 291 commits, 8cdefef is #129/291 (2026-07-02), 128 commits since | delete/replace §7 paragraph (text below) |
| F | CRITICAL | §4.1 "artifacts/ ~107 json, gitignored" | `.gitignore:19` un-ignores it; 761 tracked files, 353 top-level .json | replace line (text below) |
| L | CRITICAL | §0 "backed up (2026-08-10)" implies current | backup dir has only Aug-4-dated feats_hidden + 2 ckpt dirs; missing genframe_head_pooled_ens_v2 (85M), cheap_lingshu7b (728M), 256 Sept feats_hidden files, 65G total | add §0a warning (below) |
| M | CRITICAL | no entry doc names the judge model | `run_judge.py:21` default MedVLThinker-32B; 0/21 runners override | add to §0a |
| B | HIGH | §0 table row "0.6250" not in the cited source artifact | grep of cascade_selector_rerun_2026-08-05.json: absent; found in COMPREHENSIVE_WRITEUP_2026-08-03.md instead | amend Source: line (below) |
| I | HIGH | §3 "cap320 is the chosen operating point" | OPEN_EXPERIMENTS_2026-09-17.md §1.0: "DECIDED... nobody is choosing it now" | amend glossary line (below) |
| J | HIGH | §3 Verifier = LoRA, `lora_verifier_pooled4` | live verifier since mid-Aug is an MLP probe, not LoRA, `genframe_head_pooled_ens_v2` | amend glossary line (below) |
| P | HIGH | memory `domain-guide-package.md`: "same-family Lingshu-32B judge" | judge is MedVLThinker-32B (Finding M) | Leo/next session should edit the memory file |
| Q | HIGH | STRUCTURE.md claims to be the live file index | 401/600 .py (67%) and 141/147 .sh (96%) unmentioned, concentrated in training_methods/ (probe era) | needs a real refresh pass, not a one-line fix |
| R | HIGH | results/cascade_methods/README.md is the artifact index | 328/353 (93%) top-level artifacts unindexed; 0/30 Sept, 0/177 Aug | needs a real refresh pass |
| A | MED | §0 table CI bounds don't reproduce cascade_selector_rerun_2026-08-05.json to 4dp | 5/6 CI bounds checked off by 0.0001-0.0003 (deltas exact) | low priority, doesn't flip any verdict |
| C | MED | Ceilings paragraph unsourced, and not from the file §0's Source: line names | +0.0301 in stats_recertification_2026-08-11.json; +0.0091/+0.0661/1.3% in BEAT32B_ROUND_2026-08-10.md; sel_eff identity in 3 other files | add citation line (below) |
| G | MED | §5 "runners/ 38 shell launchers" | real: 147 | one-line fix (below) |
| H | MED | §5 "progress/ 13 diaries (June17->July8)" | real: 24, through August 17; same bug duplicated in READING_GUIDE.md | one-line fix (below) + separately fix READING_GUIDE.md |
| K | LOW | §0 "CPU thread count (+0.0048)" landmine is Aug-era, superseded in rigor (not value) by repro_threading_2026-09-13.json (spread 0.0033, now deterministic & attributed) | informational | optional footnote |
| O | MED | READING_GUIDE.md orientation stale on 3/6 counts (diaries, retrospective line count, current-writeup count) | 1972->2672 lines, 14->28 writeups, 13->24 diaries | needs its own refresh, not just CLAUDE.md |
| item7 | INFO | lit-domain-package merge-tree | 0 conflicts, 1 clean auto-merge (.gitignore), 7 additive files | safe to merge mechanically; content review still needed (branch predates 128 commits) |

## Verified-clean list

- `armcombine_mcqonly_2026-08-11.json` paragraph in CLAUDE.md §0: **every number exact to 4dp**
  (veto delta/lo/hi, fusion delta/lo/hi, both x_direct_as_charged multipliers, both guardrail_flags=[],
  both PMC_VQA-leave-one-out=0.0 claims).
- §0 table's macro *point* accuracies (always-7B, always-32B-direct, always-32B-reasoning-unmatched,
  oracle-mode-32B, accuracy-max clean, accuracy-max+8-seed, compute-lean) and their *delta* point
  values (not CI bounds): all exact matches to `cascade_selector_rerun_2026-08-05.json`'s `disjoint`
  and `ens8_scaled` arms.
- sel_eff exact identity (`selected = oracle@8 × sel_eff`, err 5.6e-17) and the "additive form
  over-predicts by +0.09 to +0.11" warning: independently confirmed in 3 separate artifacts
  (`coverage_diagnosis_2026-08-10.json`, `resolution_sweep_2026-08-13.json`,
  `method_inventory_2026-08-11.json`), all agree to the last digit.
- `git branch -vv` / `git log origin/main..main`: 78 unpushed commits, matches briefing exactly.
- No leaked HF/OpenAI/GitHub/AWS-style tokens in 1916 tracked files or the 9-file memory directory
  (4 patterns checked, 0 hits, see item 6).
- `worktree-lit-domain-package` merges into `main` with 0 conflicts (merge-tree, re-validated against
  unchanged branch tips).
- 7 root `.md` docs, 20 archived MCQ-era docs (`docs/archive_mcq/`): both counts in READING_GUIDE.md
  are exactly right.
- Em-rescore headline reproduction: `head_final_stack_PVFIXED_2026-09-13.json` macro.pooled_ens =
  0.073607 (+0.0736), self-summed per-cell n_questions = 18,452 exactly, 6/8 cells
  `pooled_ens_minus_greedy` positive (vqa_rad_open -0.041, vqamed_open -0.0017 negative) — all
  independently recomputed from the raw file, matching the briefing's "+0.0736... 6/8 benchmarks"
  description exactly.

## Could not verify (time-boxed out)

- Exact per-cell breakdown behind "+0.0661 routing headroom, Σ p10=0.529" (traced to
  `BEAT32B_ROUND_2026-08-10.md`'s own derivation from `headroom_percell_2026-08-10.json`'s per-cell
  `p10` fields, but did not sum the 8 per-cell values myself to confirm 0.529).
- Full per-benchmark 3-WIN/3-TIE/2-LOSS breakdown of the EM-currency re-score (top-line macro numbers
  independently confirmed; did not re-derive the per-benchmark verdict table).
- Line-by-line adversarial check of `gate-bakeoff-verdict.md`, `lingshu-baseline-eval-protocol.md`,
  `verify-foundational-facts.md`, `no-abstention-research.md`, `disk-usage-hard-limit.md`,
  `feats-hidden-archived-symlinks.md` (skimmed, nothing contradictory found, not exhaustively fact-checked).
- Whether the 152 `src/graphify-out/cache/ast/*.json` untracked files (found inside the `src/`
  untracked scan) represent a disk-usage or cleanliness issue -- out of scope, flagged only in passing.
- Did not re-verify `structure_mentions_missing.txt`'s 5 dead references by hand beyond trusting the
  predecessor script's file-existence check (script logic read and looks correct).

## Exact replacement text for CLAUDE.md (surgical, per the file's own "keep it short" rule)

### 1. §0's table "Source:" line -- fix Finding B/C (unsourced rows)
OLD:
```
**Convention: MACRO — equal weight per reporting cell, 8 cells, 1/8 each, Variant B (MMMU excluded),
CLEAN (disjoint) verifier.** Source: **`artifacts/cascade_selector_rerun_2026-08-05.json`**.
Never pair a macro accuracy with a sample-weighted cost, or vice versa.
```
NEW:
```
**Convention: MACRO — equal weight per reporting cell, 8 cells, 1/8 each, Variant B (MMMU excluded),
CLEAN (disjoint) verifier.** Source: **`artifacts/cascade_selector_rerun_2026-08-05.json`** for every
row except "always-32B-reasoning (prompt-matched)" 0.6250, which is from
`COMPREHENSIVE_WRITEUP_2026-08-03.md` (not in the JSON artifact). The Ceilings paragraph below is
sourced separately, see its own citation line.
Never pair a macro accuracy with a sample-weighted cost, or vice versa.
```

### 2. Ceilings paragraph -- fix Finding C (no citation at all)
OLD:
```
### Ceilings — measured free upper bounds on the 8-cell macro

perfect **selection** over the current 8-pool **+0.0301** · perfect **coverage** (infinite sampling)
**+0.0091** · perfect **7B-vs-32B routing +0.0661**, of which only **1.3%** is converted.
```
NEW:
```
### Ceilings — measured free upper bounds on the 8-cell macro
*(sources: +0.0301 `stats_recertification_2026-08-11.json`; +0.0091/+0.0661/1.3%
`docs/current/BEAT32B_ROUND_2026-08-10.md`; none of these four are in cascade_selector_rerun.)*

perfect **selection** over the current 8-pool **+0.0301** · perfect **coverage** (infinite sampling)
**+0.0091** · perfect **7B-vs-32B routing +0.0661**, of which only **1.3%** is converted.
```

### 3. §7 -- fix Finding D (flatly false, contradicts §0 itself)
OLD:
```
- **The July/Lingshu work is not in git.** Last commit `8cdefef` (2026-07-02). 44 untracked `.py`
  files under `src/` include the entire live headline chain; the IEEE paper, the July diaries and the
  2026-07-27 deck are untracked; `results/` and `MedEvalKit/` are gitignored. **The method, its inputs
  and its outputs currently exist on one disk.** Do not delete or relocate anything untracked, and
  treat "commit the working tree" as the standing top-priority chore.
```
NEW:
```
- **Git is committed but not pushed.** HEAD is well ahead of `origin/main` — check with
  `git log origin/main..main --oneline | wc -l` (78 as of 2026-09-20). Treat "push" as the standing
  top-priority chore. `results/cascade_methods/{docs,artifacts,README.md}` ARE tracked (§4.1);
  `MedEvalKit/`, `ckpts/`, `feats_hidden/`, `logs/`, `data/` are gitignored and their newest contents
  (the live probe-verifier era) are **not covered by the 2026-08-10 backup** — see §0a.
```

### 4. §4.1 -- fix Finding F (contradicts .gitignore)
OLD:
```
- **`results/cascade_methods/artifacts/`** holds ~107 numeric `.json` outputs (gitignored,
  regeneratable). The headline chain is `method_final.json`, `method_final_v2.json`,
```
NEW:
```
- **`results/cascade_methods/artifacts/`** holds 353+ top-level numeric `.json` outputs (761 tracked
  files total under the directory) — **tracked in git on purpose**
  (`.gitignore` explicitly un-ignores `results/cascade_methods/artifacts/`), so the numbers travel
  with `git push`, unlike `ckpts/`/`feats_hidden/`. The headline chain is `method_final.json`,
  `method_final_v2.json`,
```
(then continue the original sentence unchanged)

### 5. §5 tree diagram -- fix Findings G, H
OLD: `runners/            38 shell launchers (each cd's to the repo root)`
NEW: `runners/            147 shell launchers (each cd's to the repo root)`

OLD: `progress/           13 dated daily diaries (June 17 -> July 8) — the primary narrative record`
NEW: `progress/           24 dated daily diaries (June 17 -> August 17) — the primary narrative record`

### 6. §3 glossary -- fix Findings I, J
OLD:
```
- **Verifier** — a small LoRA-fine-tuned model scoring `P(correct | image, question, candidate)`; used
  to pick the best of N sampled open-text answers. `ckpts/train/lora_verifier_pooled4`.
```
NEW:
```
- **Verifier** — a small LoRA-fine-tuned model scoring `P(correct | image, question, candidate)`; used
  to pick the best of N sampled open-text answers. `ckpts/train/lora_verifier_pooled4`. **Superseded
  mid-Aug 2026 by an MLP probe on frozen hidden states (no LoRA, no backprop through the
  generator) — see §0a.**
```

OLD:
```
- **cap320 / cap640 / fullres** — image-resolution budgets (a cap on pixels via `max_pixels`).
  Lower cap = fewer image tokens = cheaper. "cap320" is the chosen operating point.
```
NEW:
```
- **cap320 / cap640 / fullres** — image-resolution budgets (a cap on pixels via `max_pixels`).
  Lower cap = fewer image tokens = cheaper. "cap320" was the chosen operating point through 2026-08;
  **regenerating at fullres was DECIDED 2026-09-17 — see §0a.**
```

## Proposed new "§0a Current status (2026-09-20)" block (insert right after §0, before §1)

```
## 0a. Current status (2026-09-20) — §0 above stops at 2026-08-11; read this first for anything newer

**Live work since mid-Aug: an MLP probe (1 hidden layer, width 256) on FROZEN Lingshu-7B hidden
states, best-of-N (N=8) verifier for open-text VQA.** Headline **+0.0736 macro** (judge currency) vs
greedy, 18,452 held-out questions, 6/8 benchmarks positive
(`artifacts/head_final_stack_PVFIXED_2026-09-13.json`: `macro.pooled_ens`=0.073607; per-cell
`n_questions` sums to 18,452; `pooled_ens_minus_greedy` negative only on vqa_rad_open/vqamed_open).
**In normalised exact-match currency on IDENTICAL picks: +0.0047 [-0.0077,+0.0171], a TIE**, not a win
(`em_rescore_pooled_probe_2026-09-18.json`: `macro.delta_em`). **Always report both currencies.**
**⚠️ The judge for every open-text label is MedVLThinker-32B, NOT Lingshu-32B**
(`src/labeling/run_judge.py:21` default; confirmed 0/21 calling runners override it). This is a
model from the project's *prior* era, not the "faithful Lingshu" family §0 describes.
Replications: Qwen2.5-VL-7B +0.0820 (`head_final_stack_qwen_2026-09-13.json`, 8/8), MedGemma-4b-it
+0.0481 (`head_final_stack_medgemma_ALL8_2026-09-16.json`, 8/8). Mechanism not novel — nearest prior
art ELHSR arXiv:2505.12225, HSRM arXiv:2608.30841 (`docs/current/PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`).
**cap320 is no longer the chosen generation resolution** — regenerating all 8 open-text pools at full
resolution was DECIDED 2026-09-17 (unmerged branch `worktree-lit-domain-package`,
`OPEN_EXPERIMENTS_2026-09-17.md` §1.0; merges into `main` with **zero git conflicts** as of this audit).
Key docs: `AUDIT_2026-09-12.md`, `PRIOR_ART_PROBE_VERIFIER_2026-09-14.md`,
`OPENTEXT_FULL_RUNDOWN_2026-09-04.md` — **not indexed in `results/cascade_methods/README.md` or
mentioned in README.md/RESULTS.md/READING_GUIDE.md/INCONSISTENCIES.md at all.**
Git: HEAD `e2aa6f0` (2026-09-17), **78 commits unpushed** to `origin/main`.
**⚠️ Not backed up anywhere**: `ckpts/train/genframe_head_pooled_ens_v2/` (85M, the live verifier),
`ckpts/openvqa/cheap_lingshu7b/` (728M), and most of `feats_hidden/`'s 65G (256 files newer than the
2026-08-10 backup) — this is currently the only copy on any disk.
```
(24 lines including the header and code fences; ≤25 satisfied.)
