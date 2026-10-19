# R2 — professor's review (round 2) of probe_verifier_report_2026-10-19 + script

Reviewer stance: an ML professor hearing the talk, plus a checker of DECK_SPEC.md.
Inputs read: DECK_SPEC.md, NUMBERS_T07.md, out/slide_text.md (all 72), the whole script, shots s02/25/28/30/33/37/39/44/46/52/60/62,
round-1 reviews (grep for overlaps), FINAL_METHOD_2026-10-16.md, and the artifacts named under each item.
Every number below is copied from a file named next to it. Where I add arithmetic, it is labelled "derived".

**Overall.** Round 1 fixed most of the substance. The deck is now honest and well sourced, and it meets the spec on content.
What remains is mostly **internal consistency** (the same quantity framed two ways on two slides) and **a few hard questions with
no prepared answer**. Five items are HIGH: 2, 3, 4, 5, 6.

---

## A. Compliance with DECK_SPEC.md

| # | requirement | verdict | where | what is missing → fix |
|---|---|---|---|---|
| 1 | Terminology in the field's conventional terms | **PARTLY** | 5–13 | Good: ORM/PRM, discriminative/generative, probe/control task, oracle@N = pass@N, reference-guided grading, cross-fitting = grouped CV. Still non-standard: the project words **"currency"**, **"re-read"**, **"half A/B" vs "half 0/1"**, and **"full resolution"**, which really means a 4× pixel cap. See items 9, 10, 11. |
| 2 | Key papers, methods explained | **MET** | 15–22, 72 | Each paper has a method line, a result and a "cuts against us" line. Nothing missing. |
| 3 | Method in detail, research-paper-style figures (layers, where the probe reads, probe structure, math) | **MET** (small fixes) | 6, 25, 28, 29, 30, 32, 33, 34 | Slide 30: the sigmoid is drawn **inside** the inference path but labelled "training only". Draw it as a dashed side branch off the logit. Put the 🔥 on both Linear boxes (item 21). |
| 4 | For each part: why, what was tested, why best | **MET** | 37–47 | Slide 37 row "selector" overclaims (item 5). Slide 44 shows two values for one quantity (item 7). |
| 5 | Well-reasoned comparison, protocol differences | **PARTLY** | 56–60 | Slide 60 still sets "+8.9 pts" beside BoE's "+0.26 to +0.58" without saying that our vote baseline is itself below greedy. Round 1 (R_papers A12) raised this and it is **still open** (item 8). |
| 6 | Honest gaps | **MET** | 62–65 | One honesty gap remains: the answer-prior share on OmniMedVQA and RadImageNet is about 60–67%, but the deck shows only the 25% macro figure and GEMeX (item 4). |
| 7 | Current method only; old work only as evidence | **MET** | 36+ | Old-era evidence is labelled per slide, but the eras themselves are **never defined in one place** (July / August / September / October). Item 12. |
| 8 | Single T = 0.7; per-benchmark map as a labelled variant; mixed-T test; per-modality/per-query T as future work with the GEMeX-vs-VQA-RAD caveat | **MET** | 7, 46, 64, 65, 69 | Optional: add MIX16 to the Q&A (item 17). |
| 9 | 24 probes vs N = 8; works for any N; "how does X change the mechanism" | **MET** (one gap) | 25 Q&A, 33, 45 | "What if you used only one probe?" has no prepared answer (Q12, item 18). |
| 10 | Colour scheme only from the old deck | **MET** (from shots) | all | Indigo/teal tokens; the page structure is new. |
| 11 | Presentation, not a document; balance | **PARTLY** | 21, 22, 37, 57, 62, 66 | These slides are paragraph walls. Slide 66 (summary) puts the whole claim on screen as one 90-word paragraph (item 19). |
| 12 | Word-for-word script in a format designed for this deck | **PARTLY** | script header vs body | The header promises an "On screen — where to point" cue **for each slide**, but only slides 2 and 25 have one (item 13). |
| 13 | Self-attack loop | **PARTLY** | — | 6 of the 12 questions in §C below have no prepared answer. |
| rules | judge number never alone; no abstention; sources named | **MET / PARTLY** | 29, 40, 41, 44, 46-table | Part IV design slides are judge-only. They are labelled as judge-only, which is acceptable as evidence for design choices, but slide 46's table should carry the label "judge only (August v1); EM and MedGemma not measured for this table". No abstention framing anywhere ✓. |

---

## B. Story and flow

**Arc:** vocabulary → papers → method → why → results → comparison → limits. The arc is clear, and the "what is new / not new" framing is
the same on 2, 15, 23, 62 and 66 ✓. Temperature is framed the same way on 7, 37, 46 and 69 ✓. Judge roles are the same on 2, 11, 34 and 49 ✓.
The contradictions that remain are items 2, 3, 5, 6 and 7 below. The out-of-order terms are items 12 and 15.

---

## Numbered findings

### HIGH

**1. HIGH — see items 2–6; numbering starts at 2 to keep cross-references stable.**

**2. HIGH — slides 63, 66 vs 44, 62 — "would hold on a new benchmark" contradicts "not domain-general".**
- Script 63: "A stricter question is whether it would hold on a new benchmark. Treating the eight benchmarks themselves as the sample, both judge gains stay clearly positive."
- Script 66: "The judge gains would hold on a new sample of benchmarks."
- Slides 44/62 say the opposite: a benchmark left out of training gains **less than the 4-domain probe** (−0.0056, `head_lobo_pooled_2026-08-25.json`).

The t-interval over 8 benchmarks treats them as a random sample, but **every one of those 8 had its own labelled training half**. The only
valid reading is "a new benchmark *with its own labelled training data*". A professor will catch this within seconds.
- *Fix, script 63:* "A stricter question: if we drew a new benchmark of the same kind **and labelled part of it for training**, would the gain hold? Treating the eight benchmarks as the sample, both judge gains stay positive, but the exact-match interval crosses zero. Without labels from the new benchmark, slide 44 applies: no better than the four-domain probe."
- *Fix, script 66:* replace "The judge gains would hold on a new sample of benchmarks; the exact-match gain might not." with "With benchmarks as the unit, the judge gains stay significant and the exact-match gain does not — and that assumes each new benchmark comes with labelled training data."
- *Fix, slide 66 on-screen:* "with benchmarks as the unit, the judge gains hold" → "with benchmarks as the unit (each with its own labelled training half), the judge gains hold".

**3. HIGH — slide 39 vs 27/35/62 — two incompatible cost accountings, side by side.**
Slide 39's table says "probe with states captured during generation ≈ 8 + 0" and "generate 8 + probe 14.90×". Slides 27/35/62 say
sampling 8 costs **1.13×** greedy. The 14.90× / 15.18× / "8 + 0" figures come from the August accounting, which charges 8 separate HF
forwards (`free_head_2026-08-16.json` `arms_fixed_bestof8.bo8_generation_only.flopeq_measured_geometry` = 8.0). The 1.13× figure assumes vLLM's shared prefill.
A listener will ask why sampling is 1.13× on slide 27 and 8× on slide 39.
- *Fix, slide 39 table title:* "FLOP-eq per question, HF path (8 separate generations, no shared prefill — August accounting)".
- *Fix, last row:* "≈ 8 + 0" → "≈ 8 (HF) — ≈ 1.13 with vLLM's shared prefill (slide 27)".
- *Fix, add a footnote:* "Derived, not measured end to end: shared-prefill sampling 1.13× (bestofn_vllm_2026-09-16.json) + as-run re-read 6.90× (free_head_2026-08-16.json) ≈ 8×. The two components are in slightly different units (5,831 vs 5,692 GFLOP per greedy pass)."
- *Fix, script 39:* after "about two percent less", add "— in the August accounting, where all eight samples are separate passes."

**4. HIGH (honesty) — slide 53 / script 38 — the answer-prior share is shown only where it is small.**
Slides 38/53 say the answer-string prior "recovers ~25% of the gain" and give GEMeX (+0.054 vs +0.120). The same artifact's per-benchmark
rows show the prior is much stronger on **exactly the two benchmarks that carry the exact-match gain** (slide 63: drop either and EM ties).
`audit_stats_baselines_2026-09-20.json` `per_benchmark.*.judge`, September probe, image half 0 (subtractions derived by me):

| benchmark | greedy | answer_prior_LOO_optimistic | probe | prior − greedy | probe − greedy | prior / probe |
|---|---:|---:|---:|---:|---:|---:|
| OmniMedVQA | 0.5216 | 0.6160 | 0.6799 | +0.0944 | +0.1583 | 0.60 |
| RadImageNet | 0.3337 | 0.4163 | 0.4562 | +0.0827 | +0.1225 | 0.67 |
| GEMeX | 0.3997 | 0.4535 | 0.5194 | +0.0538 | +0.1197 | 0.45 |
| Kvasir-x1 | 0.2811 | 0.3102 | 0.4051 | +0.0291 | +0.1240 | 0.23 |

A professor who asks "is OmniMed just a modality classifier?" will get this table from the student's own artifact.
- *Fix, slide 53, answer-frequency box:* append "— but per benchmark it is uneven: OmniMedVQA 60%, RadImageNet 67%, GEMeX 45%, Kvasir-x1 23% of the probe's judge gain (same file, per_benchmark; prior is leave-one-out and optimistic)."
- *Fix, script 53:* after "less than half of it" add: "On OmniMedVQA and RadImageNet, where questions ask for a modality or an organ from a small vocabulary, the prior alone gets about two-thirds of the way. So on those two, a large part of the gain is knowing which answers are usually right, and the probe adds the rest."

**5. HIGH — slide 37 vs 38 vs 21 — "beats every untrained selector" is contradicted on the next slide, and "size helps" contradicts Mirage.**
- Slide 37's row lists "32B zero-shot verifier, WED" among the alternatives with the verdict "beats every untrained selector". Slide 38 then says the untrained 32B "only ties a trained 7B" (+0.005 [−0.023, +0.032]), and slide 57 shows WED **ties** on VQA-RAD.
- *Fix, slide 37 judge-verdict cell:* "beats vote, prior, random, longest, self-verification; ties a zero-shot 32B (July, EM); vs WED 2 wins + 1 tie".
- Slide 21/script 21: "a bigger model of the same family did not help" (Mirage, p = 0.782). Slide 38: "size helps a zero-shot verifier (32B − 7B zero-shot +0.067 [+0.038, +0.095])". To a listener these sound like opposite findings.
- *Fix, script 38, one sentence after "size does help":* "That is not in conflict with Verification Mirage: they measure whether a model accepts its own wrong answers in self-verification; we measure best-of-8 selection accuracy, with the 32B judging the 7B's samples."

**6. HIGH — slide 26 + script 26 — "the gain is not coming from exposure … largest on the four benchmarks Lingshu never trained on" overclaims, and backup slide 71 contradicts it.**
Slide 71 (contamination audit, arXiv 2606.10066) says the "OmniMedVQA public mirror shows text exposure for all 5 VLMs incl. Qwen2.5-VL", and OmniMedVQA is one of the "four never trained on". Also, those four differ from the exposed ones in answer type and headroom, so the comparison is confounded.
- *Fix, slide 26 exposure box, last sentence:* "The gains are largest on the 4 benchmarks not in Lingshu's supervised training mix (OmniMedVQA text exposure is reported for its Qwen2.5-VL base, slide 71) — consistent with exposure not driving the delta, not a proof."
- *Fix, script 26:* replace "and the gain is not coming from exposure: it is largest on the four benchmarks Lingshu never trained on." with "and the gain is largest on the four benchmarks that are not in Lingshu's supervised training data — which is reassuring, but those four also differ in other ways, so I do not claim it proves anything."

### MEDIUM

**7. MED — slide 44 — one quantity, two values.** The bar labelled "leave the benchmark out" = **+0.0223** (5 seeds, `head_lobo_pooled_2026-08-25.json`). The box below says "a new benchmark with zero labels: **+0.0196** macro" (3 seeds, `head_price_from_lobo_2026-08-30.json summary.zero_shot_macro`). Both describe the same situation: no labels from that benchmark. The chart also uses a different 4-domain run (+0.0279) from the text (+0.0182), which the footnote already discloses.
- *Fix, onboarding box:* "a new benchmark with zero labels: +0.0196 macro (3-seed re-run of the leave-out bar's +0.0223; 5 of 8 beat greedy, a 6th after 50 labels) …".

**8. MED — slide 60 (still open from round 1, R_papers A12) — "+8.9 over majority vote" beside BoE's "+0.26 to +0.58" invites exactly the ranking the slide disclaims.** Our vote baseline is itself **below greedy** (−0.0094 [−0.0157, −0.0034], slide 18). BoE's vote baseline is strong on its data. Most of the gap between the two deltas reflects how weak voting is on short free-text answers.
- *Fix, row "ours, against the same baseline", reported-gain cell:* "+8.9 [+8.1, +9.7] pts judge — our majority vote is itself 0.9 pts below greedy".
- *Fix, script 60:* after "8.9 points under our judge" add "— but on our answers majority vote is already below greedy, so this delta mostly shows how weak voting is on short free text, not that we beat Best-of-Evidence."

**9. MED — terminology: "full resolution" is a 4× pixel cap, not native resolution (slides 25, 28, 35, 39, 47, 57, 62).** As run, "full resolution" means max_pixels 1,003,520 (≤ 1,280 tokens; `FINAL_METHOD_2026-10-16.md` table row "full-res regeneration … max_pixels 1,003,520"). Slide 47 says GEMeX images have a median of 7.8 MP, so a professor will ask why "full resolution" hurts an image that is downsized eightfold anyway.
- *Fix (global):* first use on slide 28 → "high-resolution cap (max_pixels 1,003,520 ≈ 1 MP; ≤ 1,280 image tokens — called 'full resolution' in our files)". After that, use "1-MP cap". On slide 47, change the row label to "1-MP cap (≤ 1,280 tokens)".

**10. MED — terminology: mixed names for the image halves.** Slides 29, 34, 37, 44 use **half A/B**. Slides 49, 53, 54, 63 and script 34/49/63 use **half 0/1** and "image half one". Slide 34 gives the mapping once (A = 0, B = 1). This mixing is hard to follow by ear.
- *Fix:* use "half A / half B" everywhere, including the script, on slide 49's column header ("image half B only") and slide 63 ("half A +0.0747, half B +0.0846"). Keep "(= fold 1 scores half B)" only in source lines.

**11. MED — terminology: "currency".** This is project jargon. The field says "metric" or "grader". It now appears in slide titles (11, 49) and throughout the script.
- *Fix:* slide 11 title → "Grading open answers: LLM judges and exact match (we call each grader a 'currency')". Slide 49 title → "Headline: significant macro gains under every grader". In the script, say "grader" or "metric" at first use on each slide.

**12. MED — the eras (July / August / September / October) are used from slide 9 on but never defined.** "August probe v1", "September pooled probe v2", "July LoRA pools", "earlier probe" and "PVFIXED" all appear without a timeline. Even with labels, a professor cannot weigh old-era evidence without one.
- *Fix, slide 36 (divider), add one strip:* "July: LoRA-verifier era (pools, self-verification, 32B verifier) · August: first probes (single layer, 4 training domains) · September: pooled probe v2, scored on one image half · October: final cross-fitted probes = every headline number".
- *Script 36, add:* "When a slide says July, August or September, it is evidence from an earlier version; every headline number is October."

**13. MED — script format: the header promises an "On screen" pointer for each slide, but only slides 2 and 25 have one.** The non-native speaker most needs "point here" cues on the figure slides.
- *Fix:* add one line each, for example:
  - 6 "left → right: image, ViT, merger, decoder".
  - 7 "left chart, then right chart".
  - 8 "the three bars, then the two brackets".
  - 28 "the dashed box, then the three violet lines, then the formula".
  - 29 "the bar values, then the ensemble line".
  - 30 "bottom to top through the boxes".
  - 32 "read one row of the grid, then the mean column".
  - 33 "three panels, left to right".
  - 34 "the two fold boxes".
  - 38 "the top two dots first".
  - 41 "the linear bar vs the rest".
  - 43 "the green tick row".
  - 46 "left chart, then the table's BEST column".
  - 50 "filled vs hollow markers".
  - 52 "grey vs teal bar, then teal vs blue".
  - 57 "the probe − WED column".
- Also slide 52 script: "The grey-to-teal step" → "the gap between the grey and teal bars" (the bars sit side by side; they are not stacked).

**14. MED — slide 57 + script: "wins in all 7 currencies" when 4 are on screen.** The slide shows judge, MedGemma, lenient EM and WED's EM. The other three are not named.
- *Fix, slide 57 result box:* "PathVQA: probe wins in all 7 graders (4 shown; also token-F1, LLaVA recall, strict EM — wed_vs_probe_2026-10-16.json)". Name whichever three the artifact actually holds. I did not open that file, so check before printing.

**15. MED — slide 17 uses the readout before it is defined (slide 32).** Slide 17 compares "probe-weighted vote" with "our readout" (EM +0.0135) 15 slides before the professor knows what the readout is. The same result is shown properly on slide 43.
- *Fix:* delete the "best-of-N weighted" box's second sentence on slide 17. Replace it with "We test Snell's aggregator against ours in Part IV (readout slide)." In the script, delete "and their default way of combining scores … while tying under both judges."

**16. MED — slide 43 vs 17 framing.** Slide 17 says probe-weighted vote "judge −0.0040, MedGemma −0.0058 (ties)". Slide 43 on screen and in the script says "but both judges down" / "lowers both judges", which implies a significant loss.
- *Fix, slide 43 and script 43:* "both judges slightly lower (ties: judge −0.0040, MedGemma −0.0058)".

**17. MED — Q&A gap: the likelihood baseline is cheaper than the script implies.** Script 9 says "Likelihood was not stored per candidate in our dumps." That is true for the shipped cap320 pools. However, per-candidate log-probabilities **were** stored from 2026-10-15 on (`src/labeling/run_openvqa.py` lines 235–241, `cumlogprob_all`), and the October 1-MP 8-sample pools have them for 7 of 8 benchmarks (`ckpts/openvqa/cheap_lingshu7b/*FR_sc8.jsonl`, all rows populated; RadImageNet's file has none).
- *Fix, script 9 Q&A:* "Per-candidate log-probabilities were not stored for the shipped cap320 pools. They exist for the October 1-MP pools on 7 of 8 benchmarks, so a likelihood-ranking baseline can be run there on CPU — it is first on the owed list."
- Also add to the slide 46 Q&A: "MIX16 (4 each at 0.2/0.4/0.7/1.0) vs 16 at 0.7: judge +0.0927 vs +0.0984, EM +0.0108 vs +0.0140 (NUMBERS_T07.md; MedGemma unreliable for 16-sample pools)."

### LOW

**18. LOW — Q&A gap: "what if you used one probe?"** *Add to the slide 33 Q&A:* "One layer instead of three: −0.0007 (head_final_stack_PVFIXED_2026-09-13.json pooled_ens − pooled_singlelayer). 2 seeds vs 8 seeds: selection efficiency 0.8120 vs 0.8283 (August; OPENTEXT_CORRECTIONS_2026-08-19.md §2). A single probe on the final recipe: not measured."

**19. LOW — slide 66: the claim is a 90-word paragraph on screen.** *Fix:* show three number tiles (+0.0710 / +0.0798 / +0.0121) plus three short lines ("6 wins, 2 ties (judge)", "beats WED on 2 of 3, ties 1", "mechanism not new"). Leave the full sentence to the script. The same balance problem affects 21, 22 and 62: cut each box to ≤ 2 lines and move the rest to the script.

**20. LOW — slide 37 "label source" row, EM cell "+0.0026 / +0.0101 / +0.0033"** is unreadable: it is not clear which number belongs to which alternative. *Fix:* "MedGemma labels +0.0026 · consensus +0.0101 · soft +0.0033". Check the order against judge_2x2 / judge_consensus before printing. Also, "pooling +0.055 (no CI)" is coloured green like the significant cells. Make it neutral grey.

**21. LOW — slide 30 figure.** The sigmoid sits in the main path, but the output is labelled "uncalibrated logit". *Fix:* move "σ(·) sigmoid — training only" to a dashed box branching off the logit. Put the 🔥 on "Linear 256 → 1" as well.

**22. LOW — slide 45: VQA-Med oracle "0.199 → 0.348"** conflicts with slide 52's VQA-Med oracle@8 = 0.21 (0.2102). Different pools. *Fix:* "VQA-Med's oracle (September pools) rises 0.199 → 0.348 …".

**23. LOW — "FLOP-eq" is used on slides 39 and 62 and in the Q&A but is not in the glossary.** *Fix, slide 13:* add "FLOP-eq — compute in units of one greedy answer of Lingshu-7B (one batch-1 forward at cap320)" (definition from `free_head_2026-08-16.json` "unit").

**24. LOW — duplicate Q&A.** "How does the judge decide?" appears on both slides 11 and 34. "Judge taste" appears on 2 and 49. *Fix:* keep the slide 11 and slide 2 versions. On 34 and 49, write "see slide 11 / slide 2 answer".

**25. LOW — slide 1 is labelled "(section divider)" in the script with ~0.5 min for 96 words.** *Fix:* label it "Title" and set ~1 min.

---

## C. The 12 hardest questions left after round 1

| # | question | answered now? | answer to give (source) |
|---|---|---|---|
| Q1 | "Will this work on a benchmark you have no labels for?" | **Contradictory** (63/66 vs 44/62) | "No better than the 4-domain probe: leave-out +0.0223 vs +0.0279 (head_lobo_pooled_2026-08-25.json); ~100 labels recover 46% / 66% of the gain on RadImageNet / OmniMedVQA (head_price_from_lobo_2026-08-30.json)." Fix per item 2. |
| Q2 | "Is OmniMed / RadImageNet just an answer prior or a modality classifier?" | **No** | "The prior alone gets 60% / 67% of the probe's judge gain there (September probe, half A, audit_stats_baselines_2026-09-20.json per_benchmark; leave-one-out, optimistic); the probe adds the rest. The image-shuffle control is owed." (item 4) |
| Q3 | "One number: total cost as run vs greedy?" | Partly (conflicting units) | "Not measured end to end. Derived sum ≈ 8× FLOP-eq: 1.13× shared-prefill sampling (bestofn_vllm_2026-09-16.json) + 6.90× re-read (free_head_2026-08-16.json). Latency 2.74× and energy 3.59× for sampling only; the re-read was never timed." (item 3) |
| Q4 | "Why not rank by the model's own likelihood? It is free." | Partly | "Not run. Log-probs exist for the October 1-MP pools on 7 of 8 benchmarks (cumlogprob_all), so it is a CPU job; it is first on the owed list." (item 17) |
| Q5 | "Mirage says bigger verifiers don't help; you say the 32B helps. Which?" | **No** | Different tasks: self-verification acceptance vs best-of-8 selection; ours is July pools, EM/substring, n = 600 (verifier_32b_gpu.json). (item 5) |
| Q6 | "How well does the probe order candidates within a question? Why does it fail on VQA-RAD / VQA-Med?" | Yes (as a gap) | "Within-question AUROC: not measured; it is the first cheap diagnostic (slide 65)." |
| Q7 | "How often do your two judges agree?" | **No** | "Not measured as an agreement rate (no artifact). Both labels exist per candidate, so Cohen's κ is a CPU computation — add it to the owed list." Only the agreement in sign of the deltas is shown (slide 50). |
| Q8 | "Your 'full resolution' is 1 MP and GEMeX is 7.8 MP — what is 'full'?" | **No** | "It is a 4× higher pixel cap (max_pixels 1,003,520), not native resolution; 'full' is our file name." (item 9) |
| Q9 | "+8.9 over vote vs BoE's +0.5 — are you claiming you beat BoE?" | Script yes, slide no | "No. Our vote is 0.9 pts below greedy, so that delta mostly measures how weak voting is here." (item 8) |
| Q10 | "Your CI holds the probes and the split fixed — what is the real variance?" | Yes | "Refit ≈ 0.003 (AUDIT_2026-09-18.md §5); fold-to-fold +0.0747 vs +0.0846; split/training resampling not measured." |
| Q11 | "Is the gain partly memorised training data?" | Partly, with an overclaim | "Exposure enters both arms; whether it changes the delta is not measured; OmniMedVQA text exposure is reported for the Qwen2.5-VL base (arXiv 2606.10066)." (item 6) |
| Q12 | "What if you used a single probe instead of 24?" | **No** | One layer −0.0007; 2 vs 8 seeds selection efficiency 0.8120 vs 0.8283 (August); single probe on final recipe not measured. (item 18) |

Also checked and **answered well** already: judge taste / paraphrase drift (2, 49); greedy as a candidate, single-distinct questions and the
tie-break (25); N changes (33); the per-benchmark map and why it is not shipped (46, 69); the WED asymmetries (57); closed-question training (58);
multiple comparisons (50); macro vs micro (2, 68); a probe on a frozen VLA (15).

---

## D. Script quality

- **Speakable?** Yes, mostly. The sentences are short and numbers are said as decimals with a "points" gloss. The speaking rate is
  ~110 words per minute overall, which suits a non-native speaker. Total Say text is 9,325 words ≈ 78 min at 120 wpm (the header's
  94 min is realistic with pauses).
- **Too dense for the minutes given (> 130 wpm):** slide 1 (192), 7 (149), 10 (149), 3 (146), 39 (134), 66 (133), 2 (132), 31 (131).
  *Fix:* raise slides 7, 10 and 39 to 2.0 min, slide 1 to 1.0 min and slide 66 to 1.5 min, or trim 20–30 words each. Slide 39 is the
  densest technical paragraph: split it into two short paragraphs (accuracy is a wash / cost).
- **Hard to say aloud:** slide 9 "Lightman and colleagues … Uesato and colleagues found outcome supervision reaches similar final-answer error with less labelling" is fine. Slide 22 "no-regret game" and "Wasserstein distance" are said without a gloss. Add "— a distance between two probability distributions that treats near-synonyms as close" after "Wasserstein distance".
- **Jargon left unexplained in Say:** "Borda count" (glossed ✓), "teacher forcing" (✓), "GELU" (says "non-linearity" ✓), "AdamW" (fine for this audience), "Bonferroni" (script 50 says "correct for testing eight benchmarks" ✓), "prefill" (script 35 says "shared prefix"; use "shared prefill" consistently with slide 27, or say "reads the image once").
- **Matches the screen?** Yes, with the exceptions in items 3, 13 (52), 14 and 16.

---

## 5-line summary
1. The deck meets the spec on content (papers, method figures, why-each-choice, gaps, T = 0.7 framing, 24-vs-8). It falls short on terminology (currency / half A vs 0 / "full resolution"), on slide balance (21, 22, 37, 62, 66) and on the script's missing "On screen" cues.
2. HIGH contradictions: "judge gains would hold on a new benchmark" (63/66) vs "not domain-general" (44/62); slide 39's 8×/14.90× HF accounting vs the 1.13× shared-prefill sampling on 27/35; "beats every untrained selector" vs the 32B tie and the WED tie.
3. HIGH honesty: the answer-prior baseline recovers 60% / 67% of the judge gain on OmniMedVQA / RadImageNet (audit_stats_baselines per_benchmark), but the deck shows only 25% and GEMeX. The "gain not from exposure" claim on slide 26 overreaches (slide 71 reports OmniMedVQA text exposure).
4. Six of 12 hard questions have no prepared answer: unlabelled benchmarks, the prior on OmniMed/RadImageNet, Mirage vs our 32B, inter-judge agreement (not measured), what "full resolution" means, and a single probe. Exact answers with sources are in §C.
5. The script is speakable (~110 wpm). Raise the minutes on slides 1, 7, 10, 39 and 66, add "On screen" cues for 16 figure slides, and fix the small mismatches on 17/43 (ties vs "lowers"), 52 (bars are not stacked) and 57 (7 graders, 4 shown).
