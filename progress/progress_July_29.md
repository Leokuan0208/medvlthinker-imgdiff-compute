# Progress — July 29, 2026 (the correction pass, day 1)

> **Follows `progress_July_08.md` after a three-week gap (see §0).** Today is not a new-method day and
> not a write-up day: it is an **audit day that turned into a program-wide correction pass**. One
> incidental observation — that the free-text reasoning and direct arms had been run under *different
> system prompts* — propagated outward until it had touched Finding 1, the cost model, the verifier, and
> the headline. Five offline audits and one GPU re-run landed between 12:58 and 23:47. The arc:
> (1) the **PathVQA judging audit** that found the confound; (2) the **honest re-costing**, which
> established that the "reasoning" baseline emitted ~3 tokens on ~90% of the pool; (3) the
> **verifier-validity audit**, which found the trained verifier had been scored on 67–73% of its own
> training questions; (4) the **Finding-1 prompt-matching audit** over all 35 cross-family cells;
> (5) the **Finding-1 re-derivation**, which made the headline finding *stronger* (15/20 → 17/20); and
> (6) the **matched-prompt open-text re-run** — today's only GPU work — which cleared the confound it
> was built to test. **Compute:** four of the six items are CPU-only recomputation over existing dumps;
> the GPU work is three new Lingshu-32B open-text arms plus their judging (22:15–23:39) and the verifier
> image-ablation (13:39–13:41). Every number below is sourced to a named artifact under
> `results/cascade_methods/artifacts/`; nothing is fabricated. **Abstention remains permanently out of
> scope** — it does not appear below, as a method or otherwise.

---

## 0. The gap: 2026-07-09 → 2026-07-28

There is **no diary for July 9, and none for July 10 → July 28**. That is a genuine gap in the record,
not an omission by this entry. Exactly two artifacts exist in that window, and both are dated:

| when | what | note |
|---|---|---|
| 2026-07-09 19:44:45 | `artifacts/f8_mode_vsthink_ci.json` (+ `src/cascade_methods/f8_mode_vsthink_ci.py`, 19:43:11) | the certified-veto vs-reasoning CI that the July-8 diary flagged as its one remaining rigor gap (`progress_July_08.md` §11, loose end 2) |
| 2026-07-27 01:18:30 | `meetings/progress_report_professor_2026-07-27.html` (built by `paper/build_professor_html_2026-07-27.py`) | the most current source-cited summary deck in the repo |

Nothing else on disk changed in nineteen days. Two questions the record cannot answer, and which are
recorded here as unknown rather than guessed: **whether the CVGIP paper was submitted**, and **whether
the MMMU exclusion (Variant B) was ever ratified by the researcher**. Neither is written down anywhere.

> **A defect this gap created.** The professor deck and its builder hard-code Finding 1's **15/20** at
> `build_professor_html_2026-07-27.py:118,127` and `progress_report_professor_2026-07-27.html:216,223`.
> As of tonight that number is superseded (§6). The rendered deck is a frozen dated deliverable and is
> left alone; **the builder must be fixed before the next deck is generated.**

---

## 1. How a judging audit became a correction pass

The day began as a narrow question about one cell. PathVQA-open is the load-bearing cell of the
vs-reasoning headline — the July-8 rigor run had measured 32B-think there at **0.1087** against
no-think **0.376**, a −0.2673 collapse, and that single cell contributes **51.23%** of the pooled
+0.0245 headline (`pathvqa_judge_audit.json:headline_propagation`). A collapse that large in the cell
that carries half the claim deserved an audit before it went into a paper.

The audit did not find what it was looking for. It found something worse, in a key called
`prompt_confound`: **the two arms had never been a matched comparison.** From
`pathvqa_judge_audit.json:prompt_confound` (source: `src/labeling/run_openvqa.py` `SYS` / `SYS_THINK`,
selected by the `--think` flag):

- **direct arm:** *"You are an expert medical image analyst. Answer the question with a short, specific
  phrase. Do not explain."*
- **reasoning arm:** *"You will solve a problem/request. You should provide your thoughts within
  `<think>` `</think>` tags before providing the answer. After `</think>`, give only the short final
  answer."*

The reasoning arm loses the expert-analyst persona, the *"short, specific phrase"* constraint and the
*"Do not explain"* constraint. On free text, graded by an LLM judge, answer style and length are a
**live grading channel**. The measured think-vs-direct delta therefore conflated a reasoning effect
with an output-convention effect — and the same `run_openvqa.py:26/27` pair had produced every
open-text reasoning number the project had ever reported.

That is what turned a one-cell audit into a pass over everything that shared the defect's shape: a
comparison that had been **assumed** rather than **constructed**. Five audits followed, each asking the
same question of a different quantity — *is this comparison actually matched?*

---

## 2. The PathVQA judging audit (`pathvqa_judge_audit.json`, 13:09–13:11)

`src/cascade_methods/pathvqa_judge_audit.py --report` (seed 20260729) over
`ckpts/openvqa/strong_lingshu/ckpt_pathvqa_open_lingshu32b.jsonl` (direct) and
`.../strong_lingshu_think/ckpt_pathvqa_open_lingshu32b_think.jsonl` (reasoning), n = 1,500.

**Descriptive.** acc_direct **0.376**, acc_reason **0.1087**, Δ **−0.2673** (local judge); exact-match
0.344 vs 0.0687. The 2×2 is lopsided: direct-right/reason-wrong **449**, direct-wrong/reason-right
**48**, both-right 115, both-wrong 888. Answer length: 1.95 words / 16.1 chars (direct) against 4.28
words / 31.8 chars (reasoning). Gold answers are short — mean 2.36 words, **53.33% single-word**, 75.6%
≤ 2 words, and **0.0% binary yes/no**.

**Four things it ruled out or quantified, in order.**

1. **Not truncation.** Verdict verbatim: *"NOT truncation: 0 items reach the 512-token cap (max 261);
   4/1500 carry an unstripped trace; 0 empty. Independently re-verified."*
2. **The collapse is concentrated on short golds.** By gold length — 1 word (n=800): 0.465 → 0.0938
   (**−0.3713**); 2 words (n=334): 0.3473 → 0.1317 (−0.2156); 3–4 (n=197): 0.2792 → 0.1574 (−0.1218);
   5+ (n=169): 0.1243 → 0.0769 (−0.0473). The shorter the expected answer, the worse reasoning does.
3. **A degenerate-question family explains most of it.** Building a taxonomy vocabulary from golds seen
   ≥ 5 times gives 35 terms; **948 of 1,500 items (63.2%)** are answerable from that vocabulary. On
   them direct scores 0.4662 and reasoning 0.0949 (**Δ −0.3713**) — **81.74%** of the entire
   direct-right/reason-wrong pool. Taxonomy-token emission rate: **0.7553 direct vs 0.1508 reasoning**.
   On the non-degenerate remainder (n=552) the gap is only −0.0888. Reasoning stops emitting the
   vocabulary the benchmark rewards.
4. **The evaluated slice is a non-random prefix.** 1,500 of a 3,357-item open test set, taken as
   `run_openvqa.py items[:n]`. Degenerate fraction: **prefix 0.632 vs remainder 0.505 vs all 0.562** —
   the artifact's own note says this *"biases the magnitude, not the sign."*

**Then the judge itself was audited, by hand.** Two label sets were produced by this session (labeller
recorded verbatim in `pathvqa_judge_audit_labels.json:_meta` as *"Claude Opus 5 (this audit session),
reading question + gold + both answers, no image"*):

*(a) The 449-item disagreement pool*, stratified sample n = 90:

| class | n | share |
|---|---:|---:|
| a — genuinely wrong | 26 | 0.2889 |
| b — entails gold at a different granularity | 33 | 0.3667 |
| c — clear judge error | 7 | 0.0778 |
| d — ambiguous | 24 | 0.2667 |

`frac_not_genuinely_wrong (b+c)` = **0.4444, Wilson95 [0.3462, 0.5473]**; clear judge error alone
**0.0778 [0.0382, 0.1519]**.

*(b) A five-cell judge validation* (25 or 40 items per cell), which found the decisive asymmetry:

| cell | n | err_fair | Wilson95 (fair) | err_strict |
|---|---:|---:|---|---:|
| direct_accept | 25 | 0.04 | [0.0071, 0.1954] | 0.04 |
| direct_reject | 25 | 0.04 | [0.0071, 0.1954] | 0.00 |
| reason_accept | 25 | 0.08 | [0.0222, 0.2497] | 0.08 |
| **reason_reject** | 25 | **0.28** | [0.1428, 0.4758] | 0.04 |
| reason_reject_bothwrong (extra) | 40 | 0.05 | [0.0138, 0.1650] | 0.025 |
| reason_reject_bothwrong (merged) | 56 | 0.0714 | [0.0281, 0.1698] | 0.0357 |

The artifact's own `bias.note`: *"the judge's errors are strongly asymmetric AGAINST the reasoning
mode's answer style"* — false-reject rate 0.04 (direct) against **0.28** (reasoning).

**Corrected under two scenarios**, propagated to the cell and then to the headline:

| scenario | acc_reason | acc_direct | Δ | artifact share of collapse | pooled headline |
|---|---:|---:|---:|---:|---:|
| as measured | 0.1087 | 0.376 | −0.2673 | — | +0.0245 |
| **fair** (b+c credited) | **0.2753** | 0.3859 | **−0.1107** | **0.5861** | **+0.0190** |
| **strict** (c only) | **0.1444** | 0.3610 | **−0.2166** | 0.1899 | **+0.0226** |

**What this settled and what it did not.** The PathVQA collapse is *partly* a grading artifact — under
the fair reading **58.6%** of it is — but it is not *only* one: even under the strict reading the
reasoning arm loses 0.217. The genuinely damaging finding was the incidental one, the `prompt_confound`
key. Everything below follows from it.

> **Caveat carried forward, and it matters later.** These hand corrections were derived on the
> **unmatched** reasoning answers. They **cannot be stacked** onto the matched re-run of §7 without
> re-labelling, and are not (see `matched_prompt_reasoning_2026-07-29.json:note`).

---

## 3. The honest re-costing (`honest_recosting_2026-07-29.json`, 13:15)

`python3 src/cascade_methods/honest_recosting.py`, CPU-only, 10,000-replicate bootstrap, over Variant B
(MMMU excluded: 5 benchmarks / 8 cells / n = 42,224). The question: the cost model charges one flat
reasoning constant (10,521.6 ms / 2,001.9 J) for every item routed to the 32B "think" arm. **Is the
reasoning arm actually reasoning?**

### 3.1 Part 1 — the premise, tested before the correction

Per cell, mean generated tokens in each arm, with a "did it reason" threshold of `mean_gen_tok ≥ 50`:

| cell | n | gen (direct) | gen (reasoning) | pred agreement | classification |
|---|---:|---:|---:|---:|---|
| PMC-VQA | 33,430 | 3.00 | **3.09** | 0.92 | NOT-REASONED |
| SLAKE-closed | 836 | 3.24 | **3.33** | 0.9701 | NOT-REASONED |
| VQA-RAD-closed | 251 | 3.01 | **3.01** | 0.9721 | NOT-REASONED |
| PathVQA-closed | 3,362 | 3.00 | **— (no dump)** | — | **NO-DUMP** |
| MedXpertQA-MM | 2,000 | 3.00 | **320.33** | 0.229 | REASONED |
| MMMU-Medical-val | 150 | 2.63 | 275.37 | 0.7067 | REASONED *(excluded in Variant B)* |
| SLAKE-open | 645 | 4.76 | 122.41 | 0.4884 | REASONED |
| VQA-RAD-open | 200 | 5.63 | 104.54 | 0.395 | REASONED |
| PathVQA-open | 1,500 | 5.64 | 141.47 | 0.0527 | REASONED |

`n_reasoned` **4,345** of `n_total` **42,224** → **`frac_of_pool_genuinely_reasoned = 0.1029`**.

**Verdict verbatim:** *"PREMISE CONFIRMED. 4345 of 42224 Variant-B items (10.3%) come from cells where
the 32B reasoning-mode run emitted >= 50 generated tokens. On the other 89.7% the 'reasoning' run
emitted ~3 tokens (or does not exist at all) and agrees with the direct-mode run on 92-97% of
predictions -- it is a direct-mode run under a different name, billed at reasoning price."*

Worse, PathVQA-closed (n = 3,362, 8.0% of the pool) has **no reasoning dump at all**: the repo imputes
reasoning-accuracy = direct-accuracy in `paper_baselines.build_cells` (`okT = ok32`) **and still charges
the full reasoning latency and energy.** The field is named `acc_32b_think_measured`. It is not a
measurement.

### 3.2 Part 2 — a length-aware cost model, five variants

Calibration constants (`provenance`), from `latency_32b.jsonl` medians at n = 60 per configuration:
nothink@cap320 333.15 ms / 61.2 J; nothink@fullres 443.85 ms / 83.45 J; think@fullres 22,983.25 ms /
5,990.2 J; think@cap320 21,865.2 ms / 5,795.2 J → **decode 68.573 ms/tok and 18.261 J/tok**, prefill
280.99 ms / 24.638 J. The energy intercept cross-checks to **99.8%** (24.638 vs 24.678); the latency
intercept does **not** (280.99 vs 196.0), and the repo already carried **three mutually inconsistent
prefill decompositions of the same 665 ms — 390 / 528 / 281 ms**.

Charging each cell its own measured generation length instead of one flat constant:

| accounting | latency | energy | FLOP-eq |
|---|---:|---:|---:|
| **as charged in the paper** | **10,521.6 ms** | **2,001.9 J** | 4.57 |
| M1 (primary) | 2,018.1 ms | 487.24 J | 4.87 |
| M2 (repo φ = 0.586) | 2,126.8 ms | 487.24 J | 4.87 |
| M3 (repo quantized decomposition) | 2,265.0 ms | 487.24 J | 4.87 |
| M4 (two-anchor exact) | 2,763.1 ms | 487.24 J | 4.87 |
| M5 (binary all-or-nothing) | 1,679.3 ms | 319.84 J | 4.87 |

The method's own cost is unchanged (compute-lean 469.0 ms parallel / 83.6 J / 2.248; accuracy-max
730.6 ms / 136.7 J / 4.257). What changes is the **advantage**:

| | compute-lean lat_par / energy | accuracy-max lat_par / energy |
|---|---|---|
| **as charged** | **−95.5% / −95.8%** | **−93.1% / −93.2%** |
| M1 primary | −76.8% / −82.8% | −63.8% / −71.9% |
| M4 (most generous to the method) | −83.0% / −82.8% | −73.6% / −69.6% |
| M5 (least) | −72.1% / −73.9% | −56.5% / −57.3% |

Note the direction on **FLOPs**: because the reasoning baseline genuinely costs **4.870** FLOP-eq
under length-aware accounting rather than 4.57, the compute claim *improves* to 0.462× / 0.874×. The
latency and energy claims are the ones that were inflated.

### 3.3 Parts 4–5 — concentration, dead cells, and the macro reversal

Three findings that between them reshaped how the headline had to be reported:

- **Concentration.** Of the pooled +0.0245, PathVQA-open contributes 0.012552 and PMC-VQA 0.009426 —
  **`top2_share_pct = 89.7`**. Verdict: *"CONFIRMED: PATH_VQA_open + PMC_VQA contribute 89.7% of the
  pooled +0.0245."*
- **Dead cells.** For accuracy-max, **4 of 8** cells are *exactly* 0.0000 (SLAKE-closed, VQA-RAD-closed,
  PathVQA-closed, MedXpertQA) because the method simply runs the strong model there. Pooled
  **+0.0107 [+0.0086, +0.0127] sig**. For compute-lean, 0 of 8 zero cells; pooled **+0.0011 [−0.0028,
  +0.0052], not significant.** The verdict corrects the retrospective's own "5 of 8" heading against
  its own body: *"PARTIALLY REFUTED … not 5 … The substantive point stands and is arguably
  understated."*
- **The macro reversal — the first sighting.** Under equal weight per benchmark, compute-lean on the
  five multiple-choice cells is a **significant LOSS**: **−0.0080 [−0.0139, −0.0024]** against
  oracle-mode-32B and **−0.0070 [−0.0128, −0.0016]** against always-32B-direct. Verdict verbatim:
  *"Both are LOSSES and neither appears in any headline. 'Pareto-dominates' is a sample-weighted
  statement that is carried by one cell (PMC-VQA, 79% of the pool)."* This is the observation that
  becomes tomorrow's C26 retirement of the paper's title claim.

**`part5_honest_claim.second_sentence`, verbatim** — the sentence the project now owes its readers:

> *"Against a big model actually made to reason, the advantage is a 72-83% latency and ~83% energy
> reduction (not the previously reported 95-96%), because on ~90% of the pool the run labelled 'with
> reasoning' emitted ~3 tokens and was never a reasoning run."*

Three items are listed as `must_be_withdrawn`: the −95.5% / −96% latency-and-energy claim; the field
name `acc_32b_think_measured` on PathVQA-closed; and unqualified **"Pareto-dominates"**.

---

## 4. The verifier-validity audit (`verifier_validity_2026-07-29.json`, 13:41)

Scripts: `src/training_methods/verifier_validity_audit.py` (sections A/A2/B/B2) and
`verifier_image_ablation_v2.py` (section C; the GPU work at 13:39–13:41, `logs/img_ablation_g{0,1}.log`).
Adapter under audit: `ckpts/train/lora_verifier_pooled4`, trained by `run_lora_verifier_open.py` with a
**grouped 70/30 split by question idx**, `n_questions_in_pool 3,545` → 2,481 train / 1,064 held out.

### 4.1 The overlap

The trained free-text verifier is the component that makes the entire open-text arm work. It was scored
on the full evaluation sets — which **contain its training questions**:

| dataset | n_eval | n seen in training | **% seen** | n held-out (30%) |
|---|---:|---:|---:|---:|
| slake_open | 645 | 435 | **67.44%** | 210 |
| pathvqa_open | 1,500 | 1,065 | **71.00%** | 435 |
| vqa_rad_open | 200 | 146 | **73.00%** | 54 |
| kvasir_open | 1,200 | 835 | **69.58%** | 365 |
| radimagenet_open | 2,000 | 0 | 0.00% | — (outside pool) |

And "unseen" *understates* it, because images repeat across questions (section A2): of the 210 unseen
SLAKE questions, **210 (100.0%)** use an image the verifier trained on; PathVQA **411/435 (94.48%)**;
VQA-RAD 35/54 (64.81%); Kvasir 69/365 (18.90%).

### 4.2 What the contamination is worth

Selection gain (verifier minus greedy), split seen against unseen, pooled over the three paper open
cells (n = 2,345):

| stratum | n | greedy | verifier | oracle | **gain [95% CI]** | candidate AUROC |
|---|---:|---:|---:|---:|---|---:|
| full | 2,345 | 0.44947 | 0.55352 | 0.62601 | **+0.10405 [+0.08870, +0.11898]** | 0.94327 |
| seen | 1,646 | 0.43621 | 0.55468 | 0.61847 | **+0.11847 [+0.10085, +0.13671]** | 0.95829 |
| unseen | 699 | 0.48069 | 0.55079 | 0.64378 | **+0.07010 [+0.04435, +0.09728]** | 0.90415 |

`seen_minus_unseen_gain` = **+0.048369 [+0.015973, +0.080372], significant**, and
**`memorization_share_of_gain = 0.3263`** (n-weighted 0.3100). Per dataset the seen−unseen difference is
CI-significant only on PathVQA (**+0.055885 [+0.014959, +0.095775]**, inflation 1.4426×); SLAKE
(+0.029064, inflation 1.8233×), VQA-RAD (+0.023846) and Kvasir (+0.041982) are individually
under-powered but all point the same way.

### 4.3 Two controls that say the verifier is nonetheless real

- **Image ablation** (pooled n = 360, 1,025 candidates): with the real image the selection gain is
  **+0.063889 [+0.02778, +0.10000]** and candidate AUROC 0.93252. Replace the image and the gain
  collapses — blank-gray +0.022222, blank-black +0.019444, blank-matched +0.005556, mismatched
  +0.008333, no-image +0.016667 — every one CI-significantly below real (real − condition
  0.0417–0.0583, all `sig = true`), AUROC dropping to 0.739–0.808. **The verifier genuinely looks at
  the picture.** *(Caveat: the ablation sample is SLAKE 160 + VQA-RAD 200 and **PathVQA 0** — the
  load-bearing cell is absent.)*
- **Zero-overlap transfer** (section B2): trained on SLAKE + PathVQA only and applied to Kvasir
  (never seen), gain **+0.040833**; the pooled-4 adapter on RadImageNet (0/2000 overlap), gain
  **+0.02450 [+0.01050, +0.03801]**, AUROC 0.822335.

> **⚠ The artifact is incomplete and this is recorded, not smoothed over.** It has **no verdict**.
> `src/training_methods/verifier_validity_finalize.py` exists and would write sections **D and E (the
> verdict)**; that output is absent from the file. The audit measured the contamination but never
> concluded. The conclusion is what tomorrow's disjoint retrain has to supply.

---

## 5. The Finding-1 prompt-matching audit (`finding1_prompt_matching_audit.json`, 22:21)

If the open-text arms were unmatched, were the **multiple-choice** cross-family arms too? Finding 1 —
*reasoning hurts perception* — is the project's single most cited result, 35 (family × benchmark)
cells over 5 families. `_meta.date "2026-07-29"`, offline, read-only, no new inference. All 35
recomputed accuracies **reproduce `master_data.csv` / `generalization.json` exactly**.

The prompts are not stored in the checkpoint rows. Recovering them meant tracing each `ckpts/`
directory back to a shell variable in `runners/*.sh` or a module constant — which is the whole reason
the defect survived three weeks unseen. Twenty-four verbatim prompts were recovered and classified:

| id | source | classification | defect |
|---|---|---|---|
| A1 | MedVLThinker (`master_data.csv`) | UNMATCHED-MILD | think prompt replaces the letter-only constraint; also cap320 vs fullres; max_tokens 16 vs 2048 |
| A2 | Lingshu | **INVALID AS REASONING EVIDENCE** | the "native think" string contains **no reasoning trigger** — pure answer format |
| A3 | QoQ-Med | UNMATCHED-MILD | format constraint preserved; residual = placement + no system prompt + resolution |
| A4 | Chiron / InternVL3 | **UNMATCHED-SEVERE (format dropped)** | think instruction replaces the format constraint and nothing replaces it |
| A5 | MedGemma | **UNMATCHED-SEVERE (persona, OPPOSITE direction)** | the think arm *adds* "You are a helpful medical assistant." — and produced the one perception think-win |
| B1 | MedEvalKit (Lingshu-32B / MVT-32B / IV3-38B) | UNMATCHED-MILD | reason prompt drops the "answer … directly" clause; introduced by a **local uncommitted edit dated 2026-07-02** |
| C1 | InternVL2.5-8B, Phi-3.5-V | **MATCHED** | already clean |
| D1 | open-text arm | **UNMATCHED-SEVERE** | live style/length grading channel; not offline-repairable |

**The decisive evidence is generated-token counts.** Mean tokens, direct → think, across the seven
benchmarks:

- MedVLThinker 2.0 → 234–694 · QoQ 2.0 → 257–480 · Chiron 2.0 → 108–461 · MedGemma 2.0 → 213–841
- **Lingshu 3.0 → 3.0 / 3.3 / 3.0 / 3.1 / 3.0 / 3.0 / 3.0**

Lingshu — the headline family — **never reasoned in either arm.** Its published "think vs no-think"
comparison is two 3-token format prompts, and its quoted **1.2× latency ratio is not a reasoning
ratio.** The same applies to the pre-edit MedEvalKit `*_think` dumps (2.6–4.3 tokens); the post-edit
`*_reason` dumps do reason (275.4 / 561.2 / 368 tokens) but are format-unmatched.

**The bound on the multiple-choice half.** Single-letter gold has no style or length grading channel;
the only residual channel is extraction failure. Measured:
`max_unparsed_frac_any_think_arm_as_published` = **0.0353** (worst cells: Chiron:SLAKE 3.4%,
MedGemma:MMMU 3.5%; MedVLThinker **0.0000** in both arms on all 7). An adversarial correction that
credits every unparsed item to the think arm has `effect_on_counts`: **"NONE."**

**Verdict verbatim:** *"Finding 1 SURVIVES. The confound is real and pervasive in the prompts, but it
is BOUNDED on multiple choice and, where matched arms exist on disk, correcting it makes the finding
STRONGER (17/20 vs 15/20). Only the open-text arm is genuinely broken."*

The audit closed with five ranked re-runs. #1 (offline re-derivation, zero GPU, ~1 h) ran tonight
(§6); #2 (open-text matched re-run) ran tonight (§7); #3 (MedEvalKit matched **direct** arm with
`\boxed{}` appended, ~6,450 generations) ran tomorrow.

---

## 6. The Finding-1 re-derivation (`finding1_corrected_2026-07-29.json`, 22:45)

`src/cascade_methods/finding1_corrected.py`. 22 arms registered; 4 correction policies; significance =
exact two-sided McNemar on discordant pairs **plus** a 10,000-replicate paired-bootstrap 95% interval
(seed 20260729); noise band |Δ| ≤ 0.02. Gate: published arms reproduce `master_data.csv` with
`worst_abs_deviation = 0.0` across all 35 cells.

**The headline count went up, not down:**

| policy | perception strictly negative | within +0.02 | **CI-significant negative** | **pooled Δ (n = 30,250)** |
|---|---:|---:|---:|---|
| P0 as published | 15/20 | 19/20 | 12/20 | −0.0252 [−0.0304, −0.0199] |
| **P1 audit best-matched (primary)** | **17/20** | 19/20 | **14/20** | **−0.0401 [−0.0456, −0.0347]** |
| P2 strict resolution + format | 17/20 | 19/20 | 13/20 | −0.0408 [−0.0462, −0.0353] |
| P3 strict MVT at fullres | 17/20 | 19/20 | 13/20 | −0.0405 [−0.0459, −0.0351] |

Three independent correction policies all give **17/20**. Per family (perception pooled, n = 6,050
each, P1): MedVLThinker −0.0144 [−0.0261, −0.0030]; **Lingshu −0.0792 [−0.0902, −0.0681]**; QoQ −0.0524;
Chiron −0.0707; **MedGemma +0.0162 [+0.0028, +0.0298]** — the sole positive family.

**Two cells flipped sign** (both PMC-VQA `test_clean.csv`, n = 2,000): MedVLThinker **+0.0055 →
−0.0075** [−0.0275, +0.0120] (resolution-matched at cap320), and Lingshu **+0.0115 → −0.0425**
[−0.0625, −0.0220], p = 5.6e−5 (with a genuinely-reasoning arm). The largest re-derivations are all
Lingshu: PathVQA −0.0170 → **−0.1017**, SLAKE −0.0096 → −0.0649.

**The fully-matched subset — nothing left to correct.** Chiron and MedGemma against foreign-think, 8
medical perception cells: 6/8 strictly negative, 7/8 within noise, 5/8 CI-significant, pooled
**−0.0273 [−0.0367, −0.0176]** (n = 12,100). Non-medical peers: InternVL2.5-8B −0.0076 [−0.0208,
+0.0056]; Phi-3.5-Vision −0.0187 [−0.0336, −0.0036].

**The reasoning half did not survive intact.** 12/15 cells point-positive but only **4/15**
CI-significant and **1/15 significantly negative**:

| family | status (verbatim) |
|---|---|
| MedVLThinker-32B | "SURVIVES as evidence (2+ of 3 cells positive with 95% CI excluding 0)" — 3/3 significant |
| MedGemma-27B | "PARTIALLY SURVIVES (1 of 3 cells positive with 95% CI excluding 0)" |
| Chiron-o1-8B | "DOES NOT SURVIVE as evidence (no cell's 95% CI excludes 0)" |
| Lingshu-32B | "DOES NOT SURVIVE as evidence" |
| QoQ-Med-VL-32B | **"CONTRADICTS the reasoning-helps claim (a cell is significantly NEGATIVE)"** — MedXpert-U −0.0433, p = 0.022 |

**Six withdrawals**, of which three matter: (1) **all 7 Lingshu-32B cells**, both directions, replaced
with the foreign-think arm — repaired, perception is **4/4 strictly negative, all CIs excluding zero,
pooled −0.0866 [−0.0972, −0.0757]**, and the reasoning side is *nothing* (MMMU +0.0000, MX-R +0.0048,
MX-U +0.0271, none significant); (2) **QoQ as reasoning-side evidence** (MMMU +0.0706 → +0.0118
[−0.0588, +0.0824] matched, +0.0000 fully matched); (3) **the phrase "5 families" on the reasoning
half** — perception keeps all five. MedGemma's PathVQA exception is explicitly **not** withdrawn:
+0.0399 → **+0.0413 [+0.0220, +0.0607]**, p = 0.0000, on a fully matched pair. *"The EXCEPTION ITSELF
IS NOT WITHDRAWN."*

**The defensible statement, verbatim** (this is now the project's canonical wording of Finding 1):

> *"Chain-of-thought reasoning does not pay for itself on perception-style medical visual QA: on
> prompt- and resolution-matched arms, thinking is strictly worse than answering directly in 17/20
> (family x benchmark) perception cells across 5 medical VLM families - 14/20 with 95% CIs excluding
> zero, pooled -0.0401 [-0.0456,-0.0347] over 30250 paired samples, and 19/20 no better than +0.02 -
> and it reproduces at the same strength on the subset of arms that differ by nothing but the reasoning
> instruction; on reasoning-heavy benchmarks CoT helps some model families (MedVLThinker-32B,
> MedGemma-27B, InternVL3-38B) but not others (Lingshu-32B, QoQ-Med-VL-32B), so the reasoning-side gain
> is model-dependent rather than universal."*

And its own precision caveat, which should travel with the number every time: *"The 17/20 is a COUNT OF
SIGNS, not a measurement … at n=170 a 95% CI is roughly +/-0.07 … Report the count together with the
pooled delta and the CI-significant subcount (14/20), never the count alone."*

**A dependency problem was documented, not fixed.** `MedEvalKit/utils/question_formats.py:11` and
`utils/MMMU/data_utils.py:158` were edited locally on **2026-07-02** in a way that **deleted** the
answer-format clause instead of appending the reasoning trigger. Status verbatim: *"DOCUMENTED ONLY -
MedEvalKit is a protected dependency and was NOT modified by this script."*

---

## 7. The open-text matched-prompt re-run (`matched_prompt_reasoning_2026-07-29.json`, 23:47)

The one part of §5's damage that could not be repaired offline. Today's substantive GPU work: three new
Lingshu-32B open-text arms plus their judging, chained by `runners/run_matched_prompt_chain.sh` and
`run_matched_prompt_phase3.sh`.

**Held constant across all five arms:** same Lingshu-32B snapshot, cap320, greedy (temp 0, n_samples 1),
`max_model_len` 4096, `max_tokens` 512 on the reasoning arms, tp = 2, extraction = text after the
**last** `</think>`, same evaluated indices, same judge (`run_judge.py` `judge_ok`, a neutral
MedVLThinker/Qwen2.5-32B grader).

**The five arms** — the design point is that the reasoning trigger is *appended to* the answer-style
constraints rather than substituted for them, and that a **direct_unstyled** arm supplies the clean
contrast:

| arm | constant | GPU window (logs) |
|---|---|---|
| direct | `SYS` (persona + "short, specific phrase" + "Do not explain") | pre-existing |
| **direct_unstyled** | `SYS_DIRECT_UNSTYLED` — "You will solve a problem/request. Give only the short final answer." | 23:16–23:26 |
| reason_unmatched | `SYS_THINK` (the defective published arm) | pre-existing |
| reason_matched_A | `SYS_THINK_MATCHED` — persona first, then trigger, then style clause | 22:15–22:43 |
| **reason_matched_B (decisive)** | `SYS_THINK_MATCHED2` — trigger first, then persona, then style clause | 22:53–23:16 |

Judging ran 23:34 / 23:36 / 23:39; the analysis (`src/cascade_methods/matched_prompt_reasoning.py`,
10,000-replicate bootstrap) wrote at 23:47. `arms_missing = {}`.

**Pooled open (n = 2,345):** direct **0.5168** · direct_unstyled **0.5186** · reason_unmatched **0.3028**
· reason_matched_A **0.4235** · reason_matched_B **0.4192**.

**The 2×2 decomposition — the whole point of the experiment:**

| contrast | Δ | 95% CI | sig |
|---|---:|---|---|
| **output-convention effect, reasoning OFF** (direct − direct_unstyled) | **−0.0017** | [−0.0111, +0.0077] | **no** |
| **reasoning effect at a FIXED convention** (reason_unmatched − direct_unstyled) | **−0.2158** | [−0.2354, −0.1962] | **yes** |
| the original unmatched gap (reason_unmatched − direct) | −0.2141 | [−0.2341, −0.1949] | yes |
| reason_matched_B − direct | −0.0977 | [−0.1143, −0.0814] | yes |
| reason_matched_B − reason_unmatched | +0.1164 | [+0.0994, +0.1335] | yes |

**Attribution:** `share_output_convention` = **−0.0079**, `share_reasoning` = **+1.0079** (identity
check: lhs −0.2141, rhs −0.2141, residual −0.0). The prompt confound the audit found is **real as a
description of the two arms and worth approximately nothing as an explanation of the effect.**

> *(The naive per-arm shares — 0.5637 / 0.5438 "prompt share" — are explicitly labelled in the artifact
> as **upper bounds on the prompt's contribution, not estimates**, because the styled prompts suppress
> the trace: the trace-firing rate under arm B is 0.6698 / 0.3050 / 0.7107 across the three sets, so a
> matched-styled arm is a mixture, not a clean reasoning arm.)*

**Per dataset** — and one set does not survive:

| | SLAKE-open (645) | VQA-RAD-open (200) | PathVQA-open (1,500) |
|---|---|---|---|
| reasoning effect at unstyled | −0.1349 [−0.1674, −0.1039] **sig** | −0.0600 [−0.1250, +0.0050] **n.s.** | −0.2713 [−0.2973, −0.2460] **sig** |
| output-convention effect | +0.0047 n.s. | −0.0050 n.s. | −0.0040 n.s. |
| share convention / reasoning | 0.0337 / 0.9670 | −0.0909 / 1.0909 | −0.0150 / 1.0150 |
| **verdict** | **SURVIVES** | **COLLAPSES** (under-powered at n = 200) | **SURVIVES** |

**Headline propagation.** Against matched arm B the Variant-B headline moves **+0.0245 → +0.0180
[+0.0152, +0.0210]** (shift −0.0065); open-only **+0.2699 → +0.1535**. Per open cell against matched B:
SLAKE +0.0884 [+0.0558, +0.1209] sig; VQA-RAD +0.0500 [−0.0100, +0.1100] **n.s.**; PathVQA +0.1953
[+0.1700, +0.2200] sig.

**`interpretation.headline_sentence`, verbatim:**

> *"The prompt confound is REAL as a description of the two arms but contributes ~NOTHING to the
> measured gap: giving the direct arm the reasoning arm's unstyled wording changes its accuracy by
> -0.0017 (pooled, n=2345, not significant), while the reasoning instruction at that same fixed
> convention costs -0.2158 (CI -0.2354...-0.1962, significant). 'Reasoning hurts perception open-text
> VQA' SURVIVES matched prompts."*

**A trap closed on the way past.** §2's taxonomy-collapse mechanism was re-tested against the unstyled
arm: the **unstyled direct** arm still emits taxonomy tokens at essentially the styled direct rate on
PathVQA's degenerate family (**0.742 vs 0.755**) and scores the same. *"What collapses the
taxonomy-token rate is reasoning itself"* — not the missing style instruction.

---

## 8. The preservation commit (23:29) and the versioning fix (23:51)

Two housekeeping commits that are worth recording because the record itself was at risk.

**`9f01e27` — "Preserve the July research program + the 2026-07-29 correction pass" (23:29:41).** Until
tonight, **the entire July program was uncommitted.** This commit wrote nine progress diaries
(`progress_June_27-28`, `progress_June_29-30`, `progress_July_01-02` through `progress_July_08` — 1,857
lines in total), moved the June diaries into `progress/`, added the IEEE paper and its figures, archived
five superseded manuscripts under `paper/archive/`, added the professor decks under `meetings/`, and
rewrote the eight root documents (`CLAUDE.md`, `PROJECT_OVERVIEW.md`, `README.md`, `RESULTS.md`,
`STRUCTURE.md`, `READING_GUIDE.md`, `INCONSISTENCIES.md`) plus
`docs/current/PROJECT_RETROSPECTIVE_2026-07-29.md`.

**`da04fd0` (23:50) and `83f3e6e` (23:51).** The matched-prompt artifact, then a `.gitignore` fix so the
evidence under `results/cascade_methods/` is versioned properly rather than ignored wholesale — plus
`METHOD_IDEAS_BACKLOG.md` (1,617 lines) and the artifacts index `README.md`.

`generalization.json`'s companion `GENERALIZATION.md` was annotated at 23:09 with the superseded-Finding-1
banner. **`generalization.json` itself was not** — a known, recorded staleness.

---

## 9. Standing state (end of 2026-07-29) and what is still open

**What changed today.** The project's most-cited finding got **stronger** (15/20 → 17/20, pooled
−0.0401 on 30,250 paired samples) and its supporting cast got much weaker: the headline family's
reasoning arm never reasoned, the reasoning half of Finding 1 is model-dependent rather than universal,
the reasoning **cost** baseline was inflated by ~4–5× on latency and energy, and the free-text verifier
had been graded on 67–73% of its own training questions. The open-text reasoning effect, the one thing
the day's trigger directly threatened, **survived a matched-prompt re-run essentially untouched**.

**Corrections raised today** (entered as C20–C25 / X15–X19 in `PROJECT_RETROSPECTIVE_2026-07-29.md`
§10): the 15/20 count; the reasoning half's "5 families"; all 7 Lingshu cells; QoQ's reasoning gain; the
open-text comparison downgraded to *provisional*; and C25 — **prompts are not persisted anywhere in the
checkpoint rows.** The standing rule that came out of it: **persist the prompt in every future
checkpoint row**, and never publish a think-vs-direct pair that is not format-matched *and*
token-audited.

**Open, in the order they were picked up tomorrow:**

1. **The verifier-validity audit has no verdict** — sections D and E were never written. Measuring the
   contamination is not the same as pricing it; a **strictly disjoint retrain** is the only thing that
   can. *(Split construction began at 00:53.)*
2. **The MedEvalKit multiple-choice arms are still unmatched** — ranked re-run #3, ~6,450 generations,
   needs a GPU.
3. **The macro reversal is measured but not adopted.** compute-lean is a significant multiple-choice
   loss under equal weight per benchmark, and no headline says so.
4. **PMC-VQA carries 79% of the pool and 38% of the headline** and has never had an item-level validity
   check. *(The audit worksheet was built at 23:35 tonight and ran into tomorrow.)*
5. **The `4.57` FLOP ratio and the `522 ms` best-of-N latency** are used everywhere and derived nowhere
   — noted in passing by the re-costing's provenance block, which reproduces 4.571 while remarking that
   *"no file in the repo derives it."* Neither was chased today.
6. **PathVQA-open remains a non-random prefix** (degenerate fraction 0.632 against 0.562 for the full
   set) judged by a judge validated on SLAKE and VQA-RAD but **not on PathVQA** — and it is the cell
   carrying half the headline.
