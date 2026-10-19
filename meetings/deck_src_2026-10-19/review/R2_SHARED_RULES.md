# Round-2 fix pass — shared rules for all three workers

Apply every finding in review/R2_numbers.md, R2_professor.md and R2_visual.md that touches your slides
(HIGH and MED required, LOW where cheap). Verify against the source before applying; reject with a reason if wrong.
Round-1 shared rules still hold (single T = 0.7 shipped; per-benchmark map reversed 2026-10-18 for the dataset-label
reason; lenient EM; MedVLThinker-32B = training judge, MedGemma-27B-it = independent cross-family judge, both
text-only and see gold; no human validation of the judge; class-weighted BCE; dedup = lower-case + strip; no
"first"; no abstention as a method; never invent numbers, write any new number to numbers_review_partN.json
with the script that computed it).

New, deck-wide decisions:
1. **Terminology.** Use the field's word "metric" (evaluation metric) on screen; the project word "currency" may
   appear once, in the vocabulary slide, as "we call these metrics currencies". Image halves are **half A / half B**
   everywhere (A = fold-0 scoring half, B = fold-1 scoring half; "half 1" = half B). "Full resolution" is replaced
   by "**1-megapixel cap** (max_pixels 1,003,520)", defined once where first used; cap320 = max_pixels 250,880.
2. **Probe versions.** Define once (worker 2, slide 26 or the decision-table slide) the probe generations the deck
   cites: August (`genframe_head_ens8`, 4 domains), September (`genframe_head_pooled_ens_v2`, shipped until Oct),
   October final (`final_probe_2026-10-16`, cross-fitted). Every older-era number must carry its version label.
3. **Generalisation wording (contradiction fix).** The claim is about *new questions and images from these 8
   benchmarks*, each of which contributed labelled training questions. Do NOT say the gain "would hold on a new
   benchmark"; say a new benchmark needs labelled questions (leave-one-benchmark-out gave no benefit).
4. **Cost — one accounting per number, never summed across harnesses.** Sampling 8 with vLLM shared prefill:
   1.13× FLOP-eq, 2.74× latency, 3.59× energy — measured on 20 VQA-RAD questions, ≤ 32 new tokens
   (`bestofn_vllm_2026-09-16.json`); say so. The HF re-read: 3.81 passes = 6.90 greedy-pass equivalents
   (`free_head_2026-08-16.json`, HF accounting with 8 separate generation passes → 14.90× / 15.18×). Label each
   with its harness; do not add them into one total.
5. **Selector claims.** Not "beats every untrained selector": say what it beats (majority vote, answer prior,
   WED on PathVQA/SLAKE) and what ties (zero-shot 32B judge as verifier; WED on VQA-RAD). Explain Mirage vs our
   32B result: Mirage = a model re-asked to verify its own answer (self-verification, same family); ours = a
   separate 32B judging candidates zero-shot vs a 7B — different question; both agree that training, not
   prompting, is what makes a selector work.
6. **Answer prior.** Show the per-benchmark share: the answer prior recovers 60% (OmniMedVQA) and 67%
   (RadImageNet) of the probe's judge gain (`audit_stats_baselines_2026-09-20.json`, September probe; ratio
   computed in review — recompute and store in numbers_review). Say plainly that the EM macro win rests on these
   two benchmarks.
7. **Exposure.** Never say the gain is "not coming from exposure". Say: largest gains are on benchmarks Lingshu
   did not train on; OmniMedVQA text exposure of the base model is reported by the audit; the effect of exposure
   on the delta is not measured.
8. **Likelihoods.** Correct "per-candidate likelihoods were not stored": not stored for the shipped cap320
   pools; the October 1-megapixel pools have them for 7 of 8 benchmarks, so likelihood ranking is runnable on CPU
   (next step).
9. **Slides as slides.** Text-heavy slides (R2_visual list, R2_professor B) → cut each card to one line; the
   explanation moves into `say`. Raise figure/table text to ≥ 13 px where R2_visual says it is small and space
   exists. Every figure slide gets a `look` cue ("On screen: point to …"). Adjust `minutes` per R2_professor D.
10. **Wilson interval** for 2/50 = [0.011, 0.135] (recompute). DOMAIN_GUIDE citation names branch
    worktree-lit-domain-package.

After editing: `python3 build_deck.py`; `SHOOT_PY=overflow.py bash shoot.sh out/probe_verifier_report_2026-10-19.html`
(it now also reports overlapping blocks — fix every OVER on your slides); screenshot and VIEW every slide you
changed (`bash shoot.sh out/probe_verifier_report_2026-10-19.html shots_wN N N …`). Report applied / rejected /
needed elsewhere / new numbers.
