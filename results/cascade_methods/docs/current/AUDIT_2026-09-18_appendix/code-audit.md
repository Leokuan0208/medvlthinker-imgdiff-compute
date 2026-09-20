## Q1 SPLIT — the by-image half, who uses it, where the standardizer is fit

**The split function.** `half(img) = int(md5(("nd"+str(img)).encode()).hexdigest(),16) % 2`, `==1` is
TRAIN, `==0` is held out. Defined identically (byte-for-byte same expression) in **13** live scripts:

| file:line |
|---|
| `src/training_methods/head_final_stack.py:133-134` (headline) |
| `src/training_methods/freeze_pooled_selector.py:61-62` (the frozen 24 probes) |
| `src/training_methods/head_lobo_pooled.py:55` |
| `src/training_methods/head_price_from_lobo.py:45` |
| `src/training_methods/pooled_selector.py:74` |
| `src/training_methods/head_ens_width.py:47` |
| `src/training_methods/head_visual_features.py:58` |
| `src/training_methods/head_pooled_alldomains.py:48` |
| `src/training_methods/head_second_generator.py:46` |
| `src/training_methods/_head_final_stack_prerefactor.py:46` |
| `src/cascade_methods/coverage_sc16_ci_all.py:154` |
| `src/cascade_methods/coverage_scaling.py:53` |
| `src/cascade_methods/head_temp_ensemble.py:40` |
| `src/cascade_methods/budget_conversion_sc32.py:39` |

`src/training_methods/head_newdomain_curve.py:116` also uses the `"nd"` salt but hashes
`r["img_md5"]` through a per-image dict — same value.

**Different salts** (`"rad"` head_domain_curve.py:115, `"omni"` head_omnimed_curve.py:99,
`"het"` head_donor_hetero.py:107). All three FIT THEIR OWN probe from scratch on their own donor
half and never load a frozen pooled probe — checked: none of them references
`genframe_head_pooled_ens*` or `FrozenSelector`. So a differently-salted half is not used to score
a probe that trained on the other salt's half. **CLEAN**, but fragile: the salt is a bare literal in
each file with no shared constant.

**Standardizer.** Fit on training rows only, everywhere it matters:
- `head_final_stack.py:266` `mu, sg = X[sub].mean(0), X[sub].std(0)+1e-6` where `X = Xpool` holds
  ONLY the four train domains + the `istr` (train) halves; eval features are a separate array
  (`head_final_stack.py:241` `{L: Xc[L][~istr] ...}`) and are standardized with the train mu/sg at
  `head_final_stack.py:313`. **CLEAN.**
- `freeze_pooled_selector.py:135` same, and `standardizer_L*.npz` is frozen to disk (line 140) and
  reloaded at inference rather than recomputed. **CLEAN.**

---

## FINDINGS TABLE (severity-ranked)

| id | sev | claim | evidence | correction |
|---|---|---|---|---|
| **A1** | **CRITICAL** | The probe's features come from a **separate teacher-forced replay pass at 4× the generation image resolution**, not from the generation pass. 84.35% of PathVQA candidate rows carry more image tokens than cap320 can physically produce. | see §Q3 below | retract "no extra forward pass" from 4 docs/decks; re-cost; or re-extract at cap320 |
| **A2** | **HIGH** | Docs/decks assert the probe "adds no forward pass" / "taps states the generator has already computed". The code does one full `model(...)` per (image, question, candidate). | `meetings/shipped_method_2026-09-12.html:203,243`; `meetings/opentext_progress_2026-09-14.html:355`; `PRIOR_ART_PROBE_VERIFIER_2026-09-14.md:15,42` vs `extract_generator_hidden.py:437-439` | the one honest copy already exists: `VERIFIER_ARCHITECTURES_2026-08-04.md:166-169` ("free-at-generation is an inference from the architecture, **not a measurement**"). Propagate it. |
| **A3** | **HIGH** | `freeze_pooled_selector._measured()` globs `head_final_stack*.json` and takes the **newest by mtime with no generator filter**. Today that is the **MedGemma** artifact. A refit would write MedGemma's +0.0447 into a Lingshu probe's `recipe.json`, or crash on `KeyError: 'deployed_4dom_L21ish'`. | `src/training_methods/freeze_pooled_selector.py:68-76`; mtimes in `out/` (medgemma_ALL8 2026-09-16 newest; its `macro` keys are `['pooled_singlelayer','pooled_ens','pooled_ens_sc']` — no `deployed_4dom_L21ish`) | filter the glob on the generator tag, or read an explicit `--measured_from` path |
| **A4** | **HIGH** | The three ensemble layers **18/20/22 were chosen after scoring each candidate layer end-to-end on the evaluation cells**, and the layer-set was then confirmed as the argmax of 8 sets on the **held-out halves**. There is no untouched test set anywhere in the pipeline. | `results/.../head_layer{18,19,20,21,22}_eval_2026-08-24.json` `macro_head_minus_greedy` = .0289/.0280/.0234/.0183/.0289; `head_ens_width_2026-08-25.json` scores 8 layer sets on held-out halves, `shipped_18_20_22` = argmax (+0.1051) | bound it honestly: the full spread over 8 layer sets is **0.1007–0.1051 = 0.0045 macro**, so layer choice can account for ≤ ~0.005 of +0.0737, not the effect. State that instead of "pre-specified". |
| **A5** | **HIGH** | `ckpts/train/genframe_head_pooled_ens_v2/README.md` on disk still says **+0.0802 / +0.0243** while its own `recipe.json` says **+0.0736 / +0.0182**. | `README.md` line 6 vs `recipe.json:measured_on_held_out_halves` | already fixed on branch `audit-2026-09-18` (`freeze_pooled_selector.py` now formats the line from the recipe); MAIN's on-disk README is still wrong and must be rewritten |
| **A6** | **MED** | `plan_next_wave.py` re-queues `temperature_report` and `head_temperature_sweep` on **every** wave because staleness is `any(*judge.jsonl or *.npz newer than the artifact)` over the WHOLE tree — including caches those two scripts never read (`*_sc16`, `*_sc32`, `qwen`, `medgemma`, `vis`, `oddlayer`, `finelayer`). Journal: 52 and 47 re-queues. `head_layer21_eval` is re-run 103 times the same way from per-wave runner files. | `src/reporting/plan_next_wave.py:126-138`; `logs/supervisor_journal.jsonl` counts | make the trigger depend on the inputs each script actually globs |
| **A7** | **MED** | Rewrite-every-wave **can** leave an internally inconsistent artifact: `head_temperature_sweep.py` writes the artifact **inside** the cell/temperature loop (`:146`) and only adds `best_temperature_per_cell`/`VERDICT` at the end (`:160-169`). A stall-kill leaves a partial cell set, size > 1 byte, mtime fresh → `supervisor.verify()` records **ok**. Same shape in `head_final_stack.py:325` and `head_lobo_pooled.py:158`. | `head_temperature_sweep.py:146,160-169`; `supervisor.py:74-96` (`expect_min_bytes` defaults to **1**; no queue entry for these analyses sets it) | adopt `coverage_sc16_ci_all.py`'s pattern: `COMPLETE:false` → atomic `os.replace` with `COMPLETE:true` (`coverage_sc16_ci_all.py:210,230-234`), and have `verify()` require it |
| **A8** | **MED** | `supervisor.verify()`'s freshness guard covers `expect` files only. `expect_grep` is matched against `logs/sv_<name>.log`, which is opened in **append** mode, so a marker printed by an **earlier attempt or earlier wave** still satisfies it. | `supervisor.py:97-103` vs `:116-118` | grep only the bytes written after `t0` (record the log offset before Popen) |
| **A9** | **MED** | `head_temp_ensemble_PVFIXED` was queued with `expect: .../head_temp_ensemble_2026-08-25.json` while the script writes `..._2026-08-30.json`. The supervisor caught it (`ok:false`), but the inverse — an `expect` path that some *other* job writes — would pass a no-op job silently. | journal row `{"name":"head_temp_ensemble_PVFIXED",...,"problems":["expected file missing: ...head_temp_ensemble_2026-08-25.json"],"ok":false}`; `src/cascade_methods/head_temp_ensemble.py:26` | derive `expect` from the script's `OUT` constant rather than re-typing it in each runner json |
| **A10** | **MED** | `decomposition_report.py`'s own provenance note is **false of the code**: it says `head_temperature_sweep.py` "reports a head REFITTED per temperature". `head_temperature_sweep.py:44` calls `FrozenSelector.load()` — the same frozen 4-domain incumbent, not a refit. | `decomposition_report.py:44` vs `head_temperature_sweep.py:44,106` | fix the note |
| **A11** | **MED** | `decomposition_2026-08-24.json` and `head_temperature_2026-08-22.json` report a **different verifier** (`genframe_head_ens8`: 4 domains, layer 21, BT, 8 seeds) on **full benchmarks**, macro head−greedy **+0.0130**. The headline +0.0736 is the pooled 24-probe ensemble on held-out halves. Both are internally legitimate; any document pairing the skill/penalty decomposition with +0.0736 is mixing two verifiers. | `decomposition_2026-08-24.json` cells (pathvqa n=3357 = FULL, omnimed n=8883, kvasir n=10121) and `summary_T07.mean_head_minus_greedy = 0.013032`; `head_temperature_sweep.py:44,79` (`LAYERS.index(21)`) | label both artifacts "incumbent 4-domain head, full benchmarks" in every doc that cites them |
| **A12** | **LOW-MED** | `mixed_temperature_pool.py` substitutes the **greedy** label when a temperature has no candidates for a question (`hd.append(gok[q])`), so the per-temperature "head" reference is partly a greedy arm — and `best_single_T_in_sample` is picked off it. | `src/cascade_methods/mixed_temperature_pool.py:128-130` | restrict to questions covered at every temperature, or report the coverage |
| **A13** | **LOW** | `head_final_stack.py:260` computes `sc` twice; the first call is dead code with a confusing `if False` guard (`rows_all.index(r)` would be O(n²) if it ever evaluated). | `head_final_stack.py:260-262` | delete line 260 |
| **A14** | **LOW** | No assertion that the pooled span is actually the answer. `s0 = len(ids) - n_ans_tok` with `n_ans_tok = len(tok.encode(r["ans"]))` computed **separately** from the concatenated `prompt + ans`; a BPE merge across the `assistant\n`/answer boundary shifts the span by a token with no error. | `extract_generator_hidden.py:432,451-452` | assert `tok.decode(ids[s0:]).strip() == r["ans"].strip()` (cheap, once per row) |
| **A15** | **LOW** | `head_domain_scaling.norm` (`strip().lower().rstrip(".")`) ≠ `extract_generator_hidden.norm` (`strip().lower()`) ≠ `run_openvqa.norm` (stopword+punctuation stripping). Three normalisers, and `head_final_stack.sc_of` mixes the first two. Affects only the `pooled_ens_sc` arm, not the shipped `pooled_ens`. | `head_domain_scaling.py:43-44`; `extract_generator_hidden.py:70-71`; `run_openvqa.py:83-87`; `head_final_stack.py:46,150-151` | one shared `norm` |
| **A16** | **LOW** | Cluster keys are near-degenerate on two cells, so "image-clustered" buys nothing there: omnimed 4,461 questions / **4,151** images, vqamed 1,807 / **1,807** (one question per image → identical to i.i.d.). Correctly *labelled*, but the label implies more than it delivers. | `coverage_sc16_ci_ALL_2026-09-16.json` `n_matched_questions` vs `n_images` | say "clustered where clusters exist; on vqamed the interval is i.i.d. by construction" |

---

## Q2 LABELS / DUPLICATES

**Training rows and eval rows are both DISTINCT candidates.** `extract_generator_hidden.build_eval_rows`
(`:155-165`) keeps a per-question `seen` dict keyed on `norm(a)` and emits one row per distinct
normalised answer; `build_train_rows` iterates `r["slabels"]` which is `{normalised_answer: judge_ok}`
(`:141`, built in `judged_answers`). The judge itself is run once per distinct string
(`explode_sc_for_judge.py:16-19` dedups on `str(ans).strip().lower()`).

**BCE therefore ignores vote counts, and it does so consistently on both sides.** `HS.fit(...,
objective="bce")` builds `X = torch.tensor(Xtr)` with one row per training example and
`pos_weight=(1-pos)/pos` (`head_sweep.py:169-175`) — no per-row weight is passed (`wtr=None` at
`head_final_stack.py:267`, `freeze_pooled_selector.py:138`, `head_lobo_pooled.py:125`,
`head_price_from_lobo.py:139`). Selection is `np.argmax` over the same distinct-candidate rows
(`head_final_stack.py:317-318`). **Train and selection agree. This is a design choice, not a bug** —
and `pooled_ens_sc` is the arm that re-injects frequency as an input feature.

Measured skew (`out/s06.txt`, 81,554 held-out distinct candidates, from
`em-rescore/em_rescore_per_question_cache.json`): **74.01% were sampled exactly once**, 3.71% were
sampled 8/8. So a unanimous answer and a one-off answer carry equal weight in the loss for ~3/4 of
the pool. The probe is **not** a self-consistency proxy: `P(pick == modal candidate) = 0.454` macro,
and the modal candidate is much worse than the probe's pick (kvasir 0.2715 vs 0.4051, radimagenet
0.3426 vs 0.4562).

**Tie-breaking is deterministic and carries no correctness prior — VERIFIED CLEAN.**
`extract_generator_hidden.py:294` sorts every row by `(ds, str(idx), na)`, so a question's rows are in
**alphabetical order of the normalised answer**, NOT frequency order. `np.argmax` takes the first
index. Empirically (`out/s05.txt`, 5 caches, 0 questions out of order):

| cache | questions ≥2 cands | P(y=1 \| first row) | P(y=1 \| last row) | P(y=1 \| random row) |
|---|---|---|---|---|
| finelayer (slake+vqa_rad+pathvqa-trunc) | 1,725 | 0.2191 | 0.2052 | 0.2137 |
| finelayer_pathvqa_open | 2,953 | 0.1473 | 0.1490 | 0.1527 |
| finelayer_radimagenet_open | 1,709 | 0.2270 | 0.2060 | 0.2225 |
| finelayer_vqamed_open | 3,638 | 0.0517 | 0.0465 | 0.0492 |
| train_finelayer | 5,440 | 0.1772 | 0.1640 | 0.1729 |

First-position advantage over a random row is ≤ +0.0055 absolute. And ties are rare:
**macro P(argmax tie) = 0.0058** (`out/s06.txt`, per cell 0.0000–0.0090). Worst-case contribution to
the headline ≈ 0.006 × 0.006 ≈ **3e-5**. Not a threat.

---

## Q3 EXTRACTION MISMATCH — **the CRITICAL finding**

**(a) It is a separate teacher-forced replay pass, one per candidate.** `extract_generator_hidden.py`
builds `text = proc.apply_chat_template(msgs, add_generation_prompt=True) + r["ans"]` (`:429-431`) and
runs `out = model(**enc, output_hidden_states=True)` (`:437-439`) on a **HuggingFace** model loaded at
`:331-333`. Generation ran under **vLLM** (`run_openvqa.py:178-180`). There is no path in this repo
that reads hidden states out of the generation pass.

**(b) At 4× the generation image resolution.**
- `run_openvqa.py:51-52,79`: `HIGH_PX = 1280*28*28 = 1003520`; `CAP_DIV["cap320"] = 4`;
  `MAXPX = HIGH_PX // CAP_DIV[A.cap]` with `--cap` defaulting to `cap320` → **250,880 px**.
  `grep -rn -- "--cap " runners/` returns **no `run_openvqa.py` invocation that passes `--cap`** — every
  Lingshu/Qwen/MedGemma pool in `runners/auto_gpu*.json`, `campaign*.json`, `run_kvasir.sh`,
  `run_openvqa_lingshu7b.sh` uses the default.
- `extract_generator_hidden.py:49`: `HIGH_PX, MIN_PX = 1280*28*28, 4*28*28` and it is passed as
  `"max_pixels": HIGH_PX` at `:351,417,424` — **1,003,520 px, i.e. fullres**.

**Empirically confirmed (`out/s08.txt`).** `generator_eval_finelayer_pathvqa_open.meta.json` records
`max_pixels = 1003520` (ratio to cap320 = **4.0**), and the saved `n_img_tok` (Qwen factor 28 →
tokens = px/784, ceiling 1280 at fullres vs **320** at cap320) is:

```
n_img_tok  n=16119  min 56  mean 485.4  median 532  p90 532  max 1225
   frac(n_img_tok >  320) = 0.8435      <- impossible under cap320
   frac(n_img_tok > 1280) = 0.0000      <- consistent with the fullres ceiling
mean image tokens at extraction 485.4 ; capped at 320 it would be 303.9
```

**84.35% of the rows the probe was trained and evaluated on provably come from an image encoding the
generator never executed.** Mean +181.5 image tokens per candidate.

**(c) What is pooled.** `mode=generator`: `s0 = max(0, ids.shape[0] - n_ans_tok)` with
`n_ans_tok = len(tok.encode(r["ans"], add_special_tokens=False))` (`:432`), then
`h_span[r_i, li] = hs[L][0][s0:].float().mean(0)` (`:452`). So the span is **the answer tokens only** —
no EOS, no `<|im_end|>`, no generation-prompt tokens (nothing is appended after the answer), and no
image or question tokens. `h_last` is the last answer token and is **not used** by the headline
(`head_final_stack.py:196` reads `z["h_span"]` only). Dtype: computed in bfloat16, cast
`.float()` → float32 → stored **float16** for `arch=qwen` (`:389`).
float16 is safe here — measured max |h_span| over the three shipped layers is **143.0** against the
65,504 ceiling, **0 non-finite, 0 all-zero rows** across 5 caches / 88,206 rows (`out/s05.txt`).

**(d) The cost claim.** The docs' claim is FALSE OF THE CODE AS RUN. The repo already contains the
correct statement, written 2026-08-04 and never propagated:
`VERIFIER_ARCHITECTURES_2026-08-04.md:166-169` — *"they could be cached during best-of-8 generation at
near-zero extra cost — but they were measured in a separate teacher-forced pass, so free-at-generation
is an inference from the architecture, not a measurement."* Four later documents state the inference
as fact (A2). `src/training_methods/finalize_hidden_head_artifact.py:160` also says "NOT MEASURED".

**Consequence beyond cost.** The mechanism claim ("the generator internally knows its answer is
wrong") is measured on states the generator did not produce: a different engine, a different image
encoding, and teacher-forcing instead of sampling. Caching at generation time would change the
features on 84% of rows, so **+0.0737 is not evidence that the free version works**.

---

## Q4 GREEDY and THE JUDGE

**Greedy is a separate T=0 decode of the same script, same prompt, same cap, judged by the same
judge.** e.g. `runners/auto_gpu0.json:12`:
`run_openvqa.py ... --tag lingshu7b --dataset omnimed_open --n_samples 1 --temp 0.0` (no `--cap` ->
cap320, identical to the `_sc8` pool at line 4 of the same file). System prompt is the same `SYS`
(`run_openvqa.py:26`, selected at `:170-171` because `--think` is absent). Output
`ckpt_<ds>_lingshu7b.jsonl` -> `ckpt_<ds>_lingshu7b.judge.jsonl`, read as the `greedy` baseline at
`head_final_stack.py:228,238-240,307`.

**The greedy answer is NOT in the candidate set - CONFIRMED.** Eval rows are built only from
`ckpt_<ds>_<tag>_sc8.jsonl` `preds` (`extract_generator_hidden.py:127-141,150-165`); the greedy
dump is never read by `build_eval_rows`, and `head_final_stack` uses `gok` solely for the baseline
mean. The greedy *string* coincidentally appears in the pool for 2,651/3,357 PathVQA questions, and
where it does the two judge labels disagree on only 0.11% (`out/s02.txt`, predecessor).

**The judge prompt (`src/labeling/run_judge.py:15-19,35-38`), verbatim:**

> system: `You are a strict medical exam grader. Given a question, a reference (gold) answer, and a model answer, decide if the model answer is CORRECT: it must match the clinical meaning of the reference. Be lenient about phrasing, synonyms, and abbreviations (e.g. 'CT' = 'computed tomography'), but mark wrong answers, missing key findings, or different conclusions as No. Respond with only 'Yes' or 'No'.`
>
> user: `Question: {q}\nReference answer: {gold}\nModel answer: {pred}\nIs the model answer correct? Answer Yes or No.`

- **The judge does NOT see the image.** It is a text-only vLLM run over `AutoTokenizer`
  (`run_judge.py:13-14,26,39-40`); no image ever enters the prompt. Its own docstring says so
  ("Text-only (no image)"). This is the mechanism behind the predecessor's finding that ~32% of the
  verifier's upward judge flips have zero token overlap with the gold.
- **The judge DOES see the gold.** It is a gold-vs-prediction equivalence call, not open grading.
- `judge_ok = int(P(Yes) >= P(No))` at temperature 0.0 with a 20-logprob readout
  (`run_judge.py:41,56-59`). **Ties go to Yes.**
- **[M1, MED] The judge is `MedVLThinker-32B-RL_m23k` (a Qwen2.5-32B backbone), NOT Lingshu-32B.**
  `run_judge.py:21` default, and **none of the 128 `run_judge.py` invocations under `runners/`
  passes `--judge_model`** (verified by grep). September docs and decks say only "32B judge", which
  reads as Lingshu-32B - the shared briefing for this audit made exactly that error. Name the model.
- Known label noise: the script's own comment (`:63-73`) measures **0.42%** label flips on
  re-judging the same (question, answer) pair at T=0, from vLLM batch composition.

---

## Q5 SELECTION ON THE EVALUATION DATA - the full list

| choice | value shipped | where it was decided | on what data |
|---|---|---|---|
| readout layers | **18 / 20 / 22** | `head_layer{18..22}_eval_2026-08-24.json` - every layer scored **end to end on the eval cells** (macro head-greedy .0289/.0280/.0234/.0183/.0289 for L18/19/20/21/22); then `head_ens_width_2026-08-25.json` scored 8 layer SETS and `shipped_18_20_22` is the **argmax** | **evaluation cells / held-out halves** |
| ensemble width | 3 layers | `head_ens_width_2026-08-25.json` (`sets` = 5 singletons + 3 sets) | held-out halves |
| (earlier) single layer 21 | superseded | `head_finelayer.py:68,79-104` - 5-fold image CV **on the train pool** + transfer | train pool (clean) |
| objective BCE vs BT | **BCE** | `freeze_pooled_selector.py:9-11` "settled 2026-08-19"; `head_eval_bce_2026-08-18.json` | **NOT VERIFIED** which split |
| width 256, epochs 30, wd 1e-2, lr 1e-3, bs 256 | fixed | inherited verbatim from the 2026-08-05 incumbent recipe (`genframe_selector.py:18-20`), hard-coded at `head_final_stack.py:267-268` and `freeze_pooled_selector.py:138` | inherited, not re-tuned |
| seeds | 8 per layer (frozen), 5 in the refit ladder | `freeze_pooled_selector.py:90`, `head_final_stack.py:158` | - |
| rank_avg vs argsort | **rank_avg** | `genframe_data.py:635-655` - chosen because it gave sel_eff **0.806540** vs 0.798365 | the July eval sets |
| T = 0.7 pool | 0.7 | `head_temperature_sweep.py` + `mixed_temperature_pool.py`; the latter states the problem itself (`:14`): *"picking it PER CELL means reading the eval labels, which is selection on the test set and not a deployable rule"* | eval labels |
| self-consistency feature | **excluded** | `freeze_pooled_selector.py:21-24`, on the held-out comparison (+0.0098 -> -0.0005) | held-out halves |
| pooled vs four-domain | pooled | `head_final_stack.py` arm ladder | held-out halves |

**Is there any untouched test set? NO.** Every live number is on the `md5("nd"+img_md5)%2 == 0`
half of the same eight benchmarks. That half has been scored by at least: 4 arms x 3 generators
(`head_final_stack`), 8 layer sets (`head_ens_width`), 3 arms x 8 targets (`head_lobo_pooled`),
7 k-values x 8 targets (`head_price_from_lobo`), the 8->16 budget arms (`coverage_sc16_ci_all`),
`head_visual_features`, `pooled_selector.verify`, and the EM re-score - well over 100
configurations on the same 18,452 questions.

**The honest bound, and it is reassuring:** the *entire* spread across the 8 layer sets in
`head_ens_width_2026-08-25.json` is **0.1007 -> 0.1051 = 0.0045 macro**. So the most eval-tuned
choice in the pipeline can account for at most ~0.005 of +0.0737. The tuning is real and must be
disclosed; it does not explain the effect. (Caveat: `head_ens_width`'s `pooled_rows` is **103,550**
vs the shipped **112,770** - it predates the PathVQA backfill, so its absolute +0.1051 is NOT
comparable to +0.0736 and must never be quoted beside it.)

---

## Q6 CONFIDENCE INTERVALS

- **The headline artifact carries no interval at all.** `head_final_stack_PVFIXED_2026-09-13.json`
  has keys `{title, date, no_fabricated_numbers, seeds, pooled_rows, original_rows, macro,
  beats_greedy, VERDICT, cells}` - no `ci` anywhere, and `head_final_stack.py` never bootstraps.
  The only interval on +0.0737 is the one the `em-rescore` agent computed on 2026-09-18
  (image-clustered [+0.0608, +0.0864]).
- **Image-clustered and correctly implemented:** `coverage_sc16_ci_all.py:108-120` (`boot()`
  resamples IMAGES, paired, nboot 10,000; `verdict()` at `:122-126`); `mixed_temperature_pool.py:48-58`
  (nboot 4,000, clusters = `q_img`), and that file documents at `:98-101` that 8 previously
  published WIN/LOSS verdicts had used a degenerate cluster key.
- **Remaining i.i.d.-labelled-as-clustered: none found in the live scripts.** But see A16: on
  `vqamed_open` the clustering is vacuous (1,807 questions / 1,807 images) and on `omnimed_open`
  nearly so (4,461 / 4,151), so `coverage_sc16_ci_ALL_2026-09-16.json`'s `"clustering": "...never
  i.i.d. over questions"` overstates what those two intervals are.
- `head_lobo_pooled.py`, `head_price_from_lobo.py`, `head_ens_width.py`, `head_temperature_sweep.py`
  and `decomposition_report.py` produce **no intervals at all** - breadth -0.0056, the price curve
  and best-T are point estimates only.

---

## Q7 SILENT-FAILURE CLASSES STILL LIVE IN plan_next_wave.py / supervisor.py

**Why the two analyses look stale every wave - `plan_next_wave.py:132-138`:**

    newer = [p for p in glob.glob(f"{CK}/*judge.jsonl") + glob.glob(f"{FEATS}/*.npz")
             if not os.path.exists(out) or os.path.getmtime(p) > os.path.getmtime(out)]

The trigger is **any** judge file or **any** .npz anywhere in the feature tree being newer than the
artifact. But `head_temperature_sweep.py` only reads `generator_eval_{cell}{_T0x}.npz` (the coarse
[7,14,21,28] caches, `:70,79`) and `ckpt_{cell}_{tag}*.judge.jsonl` (`:62,94`); it never touches
`generator_*_finelayer*`, `*_sc16`, `*_sc32`, `*_qwen_*`, `*_medgemma_*`, `*_vis`, `*_oddlayer` or
`generator_train_*`. Every wave produces some of those, so both jobs go stale every wave - 52 and 47
re-queues in `logs/supervisor_journal.jsonl`. `head_layer21_eval` (103 runs) is the same disease
from another source: `runners/auto_cpu_wave*.json` re-emits it per wave with a fixed `expect` path,
and `supervisor.main()` skips on `name`, which carries `_w{N}`.

**Can rewrite-every-wave make artifacts internally inconsistent? YES, two ways.**

1. **Partial rewrites pass verification.** `head_temperature_sweep.py:146` dumps the artifact inside
   the cell x temperature loop; `best_temperature_per_cell` and `VERDICT` are only added at
   `:160-169`. A stall-kill (this job runs with `stall_s: 3600`) leaves a file with a subset of
   cells, no verdict, size far above 1 byte and a fresh mtime - `supervisor.verify()` sees `expect`
   exists, size >= `expect_min_bytes` (**default 1**, and neither ANALYSES entry at `:137-138`
   sets it), mtime >= `t0`, so it records **`ok: True`**. Identical exposure at
   `head_final_stack.py:325` and `head_lobo_pooled.py:158`. The correct pattern is already in the
   repo: `coverage_sc16_ci_all.py:210` writes `COMPLETE:false` during the loop and `:230-234` does
   an atomic `os.replace` with `COMPLETE:true`.
2. **The artifact silently changes under a fixed name and date.** `head_temperature_2026-08-22.json`
   (also `temperature_curve_2026-08-22.json`, `head_layer21_eval_2026-08-24.json`) is overwritten
   dozens of times across a month; each rewrite covers whatever caches existed at that moment, while
   `art["date"]` is a hard-coded literal (`head_temperature_sweep.py:46`). A document that quoted
   the file in August cannot be checked against it today.

**Also still live:**

- `expect_grep` is matched against an **append-mode** log, so a marker printed by an earlier attempt
  or an earlier wave satisfies a later failed run (`supervisor.py:97-103` vs `:116-118`). The
  freshness guard added 2026-09-13 covers `expect` files only.
- **`head_temp_ensemble_PVFIXED`**: the journal row is
  `{"outcome":"exited","rc":0,"problems":["expected file missing: .../head_temp_ensemble_2026-08-25.json"],"ok":false}`.
  The supervisor **did** catch it - `ok:false`; the brief's "exited ok" is `rc:0`. The defect is
  that the queue's `expect` was typed by hand as `..._2026-08-25` while
  `src/cascade_methods/head_temp_ensemble.py:26` writes `..._2026-08-30.json` (the only such file on
  disk, mtime 2026-09-13). A rerun 20 minutes later passed. The generalisation: `expect` is a
  hand-copied literal in every runner json and drifts from the script's `OUT`. It fails loudly when
  the path is simply wrong, and **silently when the path happens to name a file some other job
  writes**.

---

## Q8 OTHER THINGS THAT COULD MOVE THE HEADLINE

- **`n_tok <= 0` rows dropped: ZERO.** All eight Lingshu finelayer caches have 0 rows with
  `n_tok <= 0`, `n_failed = 0` in every meta, and 0 rows with an empty normalised answer
  (`out/s04.txt`; 31,498 train + 170,306 eval rows). The `keep` filter at
  `head_final_stack.py:193-194`, `freeze_pooled_selector.py:82-83`, `head_lobo_pooled.py:60-62`,
  `head_price_from_lobo.py:50-52` and `coverage_sc16_ci_all.py:71-73` is therefore a no-op for
  Lingshu. **CLEAN.**
- **float16 overflow: does not occur for Lingshu.** max |h_span| over layers 18/20/22 is **143.0**
  against the 65,504 ceiling; 0 non-finite and 0 all-zero rows over 88,206 rows in 5 caches
  (`out/s05.txt`). The save-time guard at `extract_generator_hidden.py:466-470` would catch it
  anyway, and the MedGemma float32 switch at `:389` is correctly justified. **CLEAN.**
- **NaN handling.** A row with an empty answer would give `n_ans_tok = 0` -> `s0 = len(ids)` ->
  `h[s0:].mean(0)` over an empty slice -> NaN. No such row exists (above). Latent, not active.
- **idx-only joins.** `run_openvqa.py:191-211` has the stale-checkpoint guard (compares the question
  text at each idx and refuses to resume); `run_judge.py:74-87` dedups on first occurrence. The
  predecessor verified 0 orphans / 0 mismatches across 3 generators x 8 benchmarks (`out/s02.txt`).
  **CLEAN.**
- **Off-by-one in the pooled span (A14)** is the one unverified alignment: `n_ans_tok` is tokenised
  from the answer alone while the span is taken from the end of `prompt + answer`, so a BPE merge at
  the boundary would shift the span with no error. Not testable without loading the tokenizer.
- **`head_final_stack.py:260`** - dead duplicate `sc_of` call (A13).
- **`generator_train_finelayer` = 31,498 rows**; artifact reports `original_rows: 31,439`; recipe
  reports `leaking_rows_dropped: 59`. 31,498 - 59 = 31,439. **Consistent.**
- **Cross-check of two independent readouts of the same probe:** `head_final_stack` (refit, 5 seeds)
  macro **+0.073607** vs `coverage_sc16_ci_ALL` (frozen 24 heads, `at8_minus_greedy`) macro
  **+0.074622**, with byte-identical `greedy` and `n` on all 8 cells (`out/s03.txt`). The 0.001 gap
  is refit-vs-frozen plus readout order, exactly as the v2 recipe documents. **CLEAN.**

---

## VERIFIED CLEAN (checked, found correct)

1. Split function identical in **13** live scripts, `"nd"` salt, `==1` train (Q1 table).
2. Standardizer fit on train rows only in both the refit path and the frozen artifact; the frozen
   `standardizer_L*.npz` is reloaded, not recomputed (`pooled_selector.py:44-45,60-61`).
3. Train/eval image disjointness is asserted, not assumed: `freeze_pooled_selector.py:118-128`
   (59 rows dropped plus `assert not (train_imgs & evimgs)`), `head_final_stack.py:246-257`,
   `head_lobo_pooled.py:98-108`, `head_price_from_lobo.py:87-92`.
4. `coverage_sc16_ci_all.py:143-148` **asserts** that the pooled probe is never scored on all
   questions; `freeze_pooled_selector` records `MAY_NOT_BE_EVALUATED_ON` in the recipe.
5. Differently-salted splits (`"rad"`, `"omni"`, `"het"`) each fit their own probe from scratch and
   never load a frozen pooled probe - no cross-salt contamination.
6. Tie-break is alphabetical (not frequency-ordered), carries no correctness prior, and ties occur
   on 0.58% of questions.
7. Train and selection both operate on distinct candidates - consistent; vote counts ignored on both
   sides by design.
8. Qwen's four overlapping train caches ARE deduped on `(ds, idx, norm(na))`
   (`head_final_stack.py:200-218`; `duplicate_training_rows_dropped: 77,209` in the artifact) - the
   `head_second_generator.py` 4x-duplication bug is genuinely fixed in this path.
9. `run_openvqa.py`'s stale-checkpoint guard and `run_judge.py`'s duplicate dedup are real and each
   documents the incident that motivated it.
10. n_tok filter, float16 range, NaN exposure, dump alignment: all clean (Q8).
11. Image-clustered bootstrap correctly implemented wherever it is used.
12. All 10 Lingshu feature caches record the SAME `max_pixels` (1,003,520) - the extraction protocol
    is at least internally consistent, even though it does not match generation (`out/s09.txt`).

## COULD NOT VERIFY

- Whether the BCE-vs-BT decision (2026-08-19) was taken on the train pool or on held-out halves - I
  did not open the producing script for `head_eval_bce_2026-08-18.json`.
- Whether the last `n_ans_tok` tokens of `prompt + answer` really decode to the answer (A14) - needs
  a tokenizer load I did not run.
- Whether any deck or doc actually pairs `decomposition_2026-08-24.json`'s skill/penalty numbers with
  the +0.0736 headline (A11 is the code-side risk; the document sweep is another agent's scope).

## SCRIPTS AND OUTPUTS PRODUCED BY THIS AUDIT

- `scripts/s04_feats_audit.py` -> `out/s04.txt` (n_tok / n_failed / empty-answer census)
- `scripts/s05_fp16_and_order.py` -> `out/s05.txt` (float16 range; row order vs correctness)
- `scripts/s06_ties_votes.py` -> `out/s06.txt` (tie rate, vote multiplicity, pick vs modal)
- `scripts/s07_imgtokens.py` -> `out/s07.txt` (n_img_tok availability)
- `scripts/s08_imgtok_threshold.py` -> `out/s08.txt` (**the cap320-vs-fullres proof**)
- `scripts/s09_maxpixels.py` -> `out/s09.txt` (max_pixels of all 10 Lingshu caches)
- Predecessor, reused unchanged: `out/s01.txt`, `out/s02.txt`, `out/s03.txt`.

STATUS: COMPLETE (2026-09-20).
