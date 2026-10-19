# R_papers — professor review of Part II (key papers, slides 14–23) and Part VI (comparison, slides 55–60)

Reviewer stance: someone who knows the test-time-compute, verifier/reward-model and probing literature, reading
`out/slide_text.md` + `out/probe_verifier_report_2026-10-19_script.md` against the paper notes, the PDFs' extracted
text (`tmp/txt/`), the guide cards (`tmp/lit/cards_*.json`), the WED adapter code and the two artifacts.

Severity: **HIGH** = factually wrong or an over-claim a referee will catch; **MED** = misleading framing / omitted
caveat that changes the reading; **LOW** = precision, consistency, polish.

Numbers marked **[reviewer-computed]** are not in any artifact yet. They come from the read-only script
`tmp/deck/review/scr/wed_variants.py`, which reads `/data/dan/exp_2026_10/wed/wed_*_open_c320.jsonl`, the pool judge files and
`final_eval_FINAL_cap320_T07_records.json`. It reproduces the artifact's WED judge accuracies exactly (0.2821 / 0.7302 /
0.485). Before any of them goes on a slide, add them to `wed_vs_probe_2026-10-16.json` with bootstrap CIs.

---

## A. Part II — methods, results, attribution

**A1. HIGH — slide 23 "contribution, stated narrowly" and script 23 ("the first time this has been done for a VLM on
open-ended medical answers").**
Problem: the slide's own table contradicts this. MedProb App. H (p.27, Table 14) trains a probe on a frozen medical VLM
(Qwen2.5-VL-7B) and uses it to rerank that model's own sampled **open-ended** answers on OmniMedVQA (0.09 → 0.13, 100 test
items). The project's own audit has already withdrawn this claim: `docs/current/AUDIT_2026-09-18_appendix/lit-vlm-med.md`
correction **C2** ("First application … to vision-language / medical VQA: overreach. MedProb App. H already reranks
open-ended medical VQA generations with a probe"). CLAUDE.md §0a also says "neither is 'in a VLM'".
Exact fix (slide): *"contribution, stated narrowly: the first multi-benchmark evaluation of a hidden-state verifier
selecting among a frozen medical VLM's own open-ended answers — 8 benchmarks, 36,869 questions, image-level cross-fitting,
two judges from different families plus exact match, and a controlled head-to-head against a published selector. The
only earlier instance is MedProb App. H, a 100-item demonstration with N, grader and probe unstated. The mechanism is not
ours."*
Exact fix (script 23): replace "So the contribution is the setting and the evaluation: the first time this has been done
for a vision-language model on open-ended medical answers, with the evaluation done properly" with "So the contribution is
the evaluation, not the mechanism. MedProb's appendix tried this on a hundred questions; we measure it on eight
benchmarks, with held-out images, two independent judges and exact match, and against a published selector on identical
candidates."

**A2. HIGH — script 23: "I will show the evidence that it does not change results."**
Problem: slide 62 says "equivalence shown only for an earlier probe". The shipped probe's features come from a separate
teacher-forced pass at max_pixels 1,003,520, while generation ran at 250,880 (slide 69). No result for the shipped
probe shows that capturing during generation changes nothing.
Exact fix: "…so 'no extra pass' is true by design, not yet as run. For an earlier probe, capturing the states inside
generation left accuracy unchanged on three benchmarks. For the shipped probe, the re-extraction at the generation
resolution is running now (slide 64)."

**A3. HIGH — slide 23's table is missing the two VLM-side neighbours a referee will name.**
- arXiv:2605.28527, "What Frozen VLAs Already Know About Success". Linear probes on frozen vision-language-action
  policies are used *as a test-time selector over sampled action prefixes*. Abstract, verbatim: "on push-plate, success
  rises from 26.7% under greedy decoding to 44.3%". Source: `AUDIT_2026-09-18_appendix/lit-vlm-med.md` [V1-a]. It is not in
  the guide cards; add a card.
- arXiv:2603.22492, VHS, "Tiny Inference-Time Scaling with Latent Verifiers". A verifier on the generator's hidden
  states for best-of-N image generation: "reducing joint generation-and-verification time by 63.3%, compute FLOPs by 51%
  and VRAM usage by 14.5%" [abstract; card `cards_C` VHS].
Exact fix: add two rows:
`VLA probe 2026 (2605.28527) | ✓ (linear) | ✓ action prefixes vs greedy | — | — | vision-language-action, not VQA`
`VHS 2026 (2603.22492) | ✓ | ✓ (image generation) | — | ✓ (no decode / re-encode) | image generation, not VQA`.
Say in the script: "In vision-language models the idea also exists: for robot actions and for image generation. What is
new here is free-text medical answers at scale."

**A4. MED — slide 22, script 22, slide 60: WED's published gains.**
Problem: slide 22 and script 22 say "+1.2 to +5.0 pts over greedy". WED Table 1 (p.8) has **Qwen-8B VQA-RAD −1.2**, so the
range is −1.2 to +5.0. Slide 60 gets this right; slide 22 does not.
Exact fix (slide 22): "judge gain over greedy −1.2 to +5.0 pts (9 of 10 model × dataset cells positive) on 2–8B VLMs;
'gains diminish with size' (Table 1)". Script: "…one to five points over greedy on most small models, and a small loss
for the largest one on VQA-RAD."
In the same box, "WED is the one published selector on our exact open subsets" over-states. WED uses "the open-ended
subset of each test split, excluding yes-no questions" of VQA-RAD and PathVQA only (WED p.7); it never ran SLAKE, and its
item counts are not stated. Fix: "WED is the one published selector evaluated on two of our open benchmarks (VQA-RAD,
PathVQA open, no yes/no)". Also note on slide 60 that WED's grader is a *VLM*-as-judge (Grok 4.1 Fast, p.7), while ours
is a text-only judge.

**A5. MED — slide 15 / script 15: the speech-recognition precedents are described inaccurately.**
Problem: script 15 says "a small head reading the frozen recogniser's decoder states, trained with binary cross-entropy,
reranking the candidate transcripts". CEM (arXiv:2010.11428) **does not rerank**. It is a per-token confidence estimator
("No reranking", PAPER_NOTES_I 1.2). R-EBM (2103.14152) does rerank n-best lists, but it is a separate 2-layer
bi-LSTM (512 units) over decoder state, attention context, token embedding and top-K probabilities. Its positives include
the reference transcript itself (PAPER_NOTES_I 1.1).
Exact fix (slide box): "CEM 2020 · R-EBM 2021 — speech: trained scorers on a frozen recogniser's decoder features (CEM:
token confidence; R-EBM: n-best rescoring, BCE/NCE)". Script: "Speech recognition did the same: a trained scorer reads
the frozen recogniser's internal features to estimate confidence (2020) and to rescore the n-best list (2021)."
In the same slide, "MedProb · DualRead 2026 — probes in medical VLMs (MCQ / confidence only)" contradicts slides 21 and
23, which acknowledge App. H. Fix: "(MCQ / confidence; plus a 100-item open rerank in MedProb App. H)".

**A6. MED — slide 19, SWIFT row "MATH avg 57.5 vs 51.0 for a 7B reward model (+6.5 pts)".**
Problem: 51.0 is EurusRM-7B. The strongest baseline in the same Table 1 MATH "Avg." column is
**Skywork-Llama3.1-8B, 52.9** (SWIFT PDF p.5, Table 1). Quoting against a weaker baseline flatters the neighbour, and a
reader may think we cherry-picked.
Exact fix: "MATH avg 57.5 vs 52.9 for the strongest baseline RM (Skywork-Llama3.1-8B) and 51.0 for EurusRM-7B (Table 1); the
abstract's '12.7%' is relative to EurusRM-7B".

**A7. LOW — slide 16 / script 16, Brown.** Brown's model is "Llama-3-8B-**Instruct**", and the 82.9 → 98.44 coverage is on
**MATH** (p.3). Fix the slide label. In the script, change "a correct answer exists for 98 percent of math problems" to
"…of MATH problems, for Llama-3-8B-Instruct".

**A8. LOW — slide 17.**
- "Our analogue ('probe-weighted vote') is on the readout slide": it is on **slide 43**, not slide 32. Fix the pointer.
- Script 17 should say what slide 43 shows. Under exact match, Snell's aggregator beats our readout (+0.0135, significant).
  Under both judges it loses. This is the most direct "cuts against us" for Snell, and the script leaves it out.
- The ORM-vs-PRM "cut against us" cites only Snell App. F. Add Lightman et al. 2023 (see M1), the citation the field
  expects.

**A9. LOW — slide 20, CASE.**
- "logistic 0.596 ≥ MLP-256 0.537" has no unit. Check CASE §6.6 for the metric (the notes record only "best Qwen-7B
  cell") and write it on the slide.
- Add the second cut-against from CASE Table 4: on the medium bin, voting overtakes CASE by N = 16 (−2.0).

**A10. LOW — consistency of years and venues.**
- SWIFT is "2025" on slides 15 and 23 but "KDD 2026" on slide 19. Use "SWIFT (KDD 2026; arXiv 2505.12225)" everywhere.
- Slide 14: "194 catalogued". The cards hold 194 entries but 186 distinct papers (8 papers are carded in two categories,
  e.g. Cobbe, Brown, BoE). Write "186 papers catalogued (194 cards)".

**A11. LOW — slide 21 title, "probes so far do MCQ or confidence".** It is true of MedProb's main text and DualRead, but
App. H is on the same slide. Retitle: "The medical side: self-verification fails; medical probes so far do MCQ,
confidence, or a 100-item demo".

**A12. LOW — slide 22, Best-of-Evidence.** Two caveats are missing, and both bear on the "context only" use on slide 60.
Its VQA-Med cell is drawn from the **training partition** ("the available local copy contains only the training
partition", App. B.2). Its PMC-VQA cell is "not an official open-answer split". Add to the box: "VQA-Med items from
the training split; PMC-VQA not an official open split".

**Checked and correct.** Collins & Koo 2005 (CL 31(1)); Cobbe recipe (100 samples, outcome labels, same-size verifier,
T = 0.7, 30× quote, token-level > solution-level, decline beyond 400); Snell (proposer + verifier, MC-rollout PRM with BCE,
4×, 14× with the easy/medium caveat, ICLR 2025); self-consistency (PaLM-540B GSM8K 56.5 → 74.4, Table 2); Bay & Yearick's
modal ceiling bounds voting only; USC and EBA described correctly; Q-Probe row (layer 26, k = 48, MBPP 0.29 → 0.46,
greedy 0.38, ICML 2024); HSRM row (step boundaries, final layer with top-4 best, BT > BCE 85.93 vs 83.91, 15/16, 88.3 vs
85.2); LiLaVe (XGBoost, 16 tokens × 5 layers, AUC 0.86, BoN below voting at large n); ReProbe (ACL 2026, 810×); CASE
(0.610 → 0.542 → 0.502, Δ̂ = 100.8·AUC − 58.5, held-out r = 0.75, +19.1 MedMCQA medium bin); Mirage (FPR 62.2–72.5 from
Tables 11–15, p = 0.782, 69.5–87.1%); MedProb (> 20 pts "in several cases", linear ≈ MLP App. L, App. H 0.09 → 0.13);
DualRead (0.952 ID / 0.666 OOD); WED mechanism (T ∈ {0.5, 1.0}, SapBERT dedup, teacher-forced G/V init, no-regret
updates without further VLM calls, δ = 0.2, output rule from BDG.py); BoE (+0.26 to +0.58, 2 of 4 significant, no greedy
baseline, in-sample weights); and the majority-vote chart against `final_eval_FINAL_cap320_T07.json`.

---

## B. Papers a professor will expect — which are carded, and the proposed addition

All except Marks & Tegmark are in the guide cards (ids/arXiv in brackets). Numbers are only those the cards quote from
abstracts or the notes.

**M1. Lightman et al. 2023, "Let's Verify Step by Step" [cards_B, 2305.20050] + Uesato et al. 2022 [cards_B, 2211.14275].**
These are the canonical ORM-vs-PRM citations, and right now the deck cites only Snell's appendix. Add to slide 17's
"cuts against us" (and to slide 9's ORM/PRM row): *"Process supervision beats outcome supervision on MATH (Lightman 2023:
78% of a MATH subset, PRM800K). But outcome supervision gives similar final-answer error with less labelling (Uesato
2022: 'similar final-answer error rates with less label supervision'). Our answers are ~5 tokens with no steps, so we are
in Uesato's regime."*
The card lists Lightman's venue as "arXiv preprint". It appeared at ICLR 2024; verify, then cite it that way.

**M2. The "hidden states know correctness" lineage: Kadavath 2022 P(True)/P(IK) [cards_F 2207.05221], Burns 2022 CCS
[cards_F 2212.03827, ICLR 2023], Azaria & Mitchell 2023 [cards_C/F 2304.13734], Orgad 2024 [cards_C 2410.02707]. Marks &
Tegmark (Geometry of Truth) is NOT carded.**
A probing professor will ask why none of these appears. Add one line to slide 10 (or a box on slide 15):
*"Correctness is decodable from hidden states: Azaria & Mitchell (71–83% true/false accuracy), CCS (unsupervised,
+4% over zero-shot), Kadavath's P(IK). Each scores one statement in isolation, never ranks N answers to one question.
Orgad et al.: such 'error detectors fail to generalize across datasets', which matches our leave-one-benchmark-out
result."*
If Marks & Tegmark is mentioned, card it first. Quote no numbers from it.

**M3. Semantic Entropy Probes, Kossen et al. 2024 [cards_F 2406.15927]; INSIDE/EigenScore [cards_F 2402.03744].**
Slide 12 names "SEPs" without a citation. Add Kossen 2024 to slide 12 and to the references. Also add its directly
relevant finding to slide 62's "not domain-general" line: SEPs "generalize better to out-of-distribution data than
previous probing methods that directly predict model accuracy" [abstract]. That is a citable reason why our
accuracy-target probe transfers poorly.

**M4. Self-Certainty, Kang, Zhao & Song, NeurIPS 2025 [cards_E 2502.18581].**
This is the strongest training-free, logit-only best-of-N selector that works on open-ended output, and it is the
obvious "do you even need labels?" baseline. Add it to slide 18 next to USC/EBA. In slide 63's gap list, rename
"log-probability / perplexity ranking" to "log-probability ranking and self-certainty (Kang 2025)". Card wording usable:
"self-certainty consistently outperforms greedy decoding … and surpasses USC on Qwen-2.5-Coder-32B-Instruct"
(LiveCodeBench, Fig. 8).

**M5. Consensus Game, Jacob et al. 2023 [cards_E 2310.09139] → Bayesian Decoding Game, Zhang, Zang & Kainz 2024 [cards_E
2410.01064] → WED.**
Slide 22 presents WED as if from nowhere. Add a lineage line: *"Consensus Game (2023) → Bayesian Decoding Game (2024) →
WED (2026): WED's contribution is the semantic (Wasserstein) stopping rule. The game itself is BDG's."* This also
pre-empts "did you beat the game or just the stopping rule?" Answer: neither variant helps on our pools (see D3).

**M6. Free-text and code precedents for trained rerankers: WebGPT, Nakano 2021 [cards_A 2112.09332]; CodeRanker, Inala
et al., NeurIPS 2022 [cards_I 2206.03865]; LEVER, ICML 2023 [cards_I 2302.08468].**
Add WebGPT (2021) to the slide-15 timeline as the first best-of-n with a learned reward model on *long-form free text*.
Add LEVER to the "never tested" slide next to Collins' a₀. LEVER also "combines the verifier score with the LM's
generation probability", and that is the second independent precedent for the likelihood + probe combination we have not
run.

**M7. Reward over-optimisation: Gao, Schulman & Hilton 2022 [cards_B 2210.10760]; Huang et al. 2025 [cards_D
2503.21878].** Slide 16 already notes Cobbe's decline beyond 400. Add "(reward over-optimisation; Gao 2022)" so that the
term has its standard source. This is the field's answer to "why not N = 64?".

**M8. Medical VLM landscape: LLaVA-Med 2023 [cards_G 2306.00890], Med-PaLM M (Tu et al. 2023) [cards_G 2307.14334],
HuatuoGPT-Vision 2024 [cards_G 2406.19280], MedGemma 2025 [cards_G 2507.05201].**
Only LLaVA-Med (Part VI) and MedGemma (as judge) appear. The professor will ask "why Lingshu?". Add one line to slide 26
or the backup: *"Generators considered: LLaVA-Med, HuatuoGPT-Vision, MedGemma, Lingshu (Med-PaLM M is closed).
Lingshu-7B was chosen because it is the strongest open 7B in MedEvalKit. The MedGemma-4B replication was withdrawn (its
sampled answers ran to the token cap)."*

---

## C. Part VI — fairness and framing

The three-tier design (slide 56) is the right structure and is honestly labelled. The problems are in the details.

**V1. HIGH — slide 57 + script 57: "the probe beats WED on PathVQA and SLAKE in every currency, including WED's own exact
match".**
Problem: the artifact has 7 currencies, and the slide shows 3. On SLAKE, our lenient exact match is a **tie**:
probe − WED **+0.0109 [−0.0150, +0.0383]** (`wed_vs_probe_2026-10-16.json`, `cells.slake_open.probe_minus_wed_em`).
PathVQA is a win in all 7.
Exact fix: "the probe beats WED on PathVQA in all seven currencies and on SLAKE in six of seven (our lenient EM ties,
+0.0109 [−0.0150, +0.0383]); VQA-RAD ties in all seven." Make the same change in the script.

**V2. HIGH — slide 57 + script: "WED does not beat greedy on Lingshu-7B's pools."**
Problem: this understates. On PathVQA, WED is **significantly worse than greedy in every currency**: judge **−0.0319
[−0.0446, −0.0189]**, MedGemma −0.0363 [−0.0496, −0.0228], WED's EM −0.0363 [−0.0471, −0.0255] (artifact,
`wed_minus_greedy_*`). On VQA-RAD under WED's own EM it is also a loss, −0.0400 [−0.0761, −0.0050]. WED's paper reports
**positive** PathVQA judge gains for all five of its generators (+1.2 to +5.0, Table 1). A professor will therefore
suspect the adaptation broke WED. Saying "does not beat" invites that suspicion. Stating the loss, with the diagnostics in
D, disarms it.
Exact fix (slide): "WED is significantly below greedy on PathVQA (judge −0.0319 [−0.0446, −0.0189]) and ties greedy on
SLAKE and VQA-RAD. None of WED's four internal variants beats greedy on PathVQA either (backup)." Script: say the same,
then "because WED chooses among the same eight samples as our probe, the probe-vs-WED comparison is unaffected by how good
greedy is."

**V3. MED — slide 57 "in WED's favour": three asymmetries are missing.**
1. **Image resolution.** WED's teacher-forced scoring ran at the pool's cap320 (250,880 px; `wed_select.py` sets
   `max_pixels = HIGH_PX // CAP_DIV[cap]`). The probe's features were extracted at 1,003,520 px (slide 69). The probe saw
   the image at 4× the pixel budget. State it, and re-run after the cap320 re-extraction (slide 64).
2. **Candidate count.** WED's own pipeline draws up to 16 samples until 8 *distinct* candidates exist. Our fixed pool
   often collapses after WED's canonicalisation and dedup. Questions where WED had ≤ 1 candidate, so the game never runs:
   PathVQA 490/3,357 (478 + 12 fallbacks), SLAKE **392/645**, VQA-RAD 49/200 **[reviewer-computed from the WED picks
   files]**. On SLAKE the comparison mostly measures canonicalisation plus fallback, not the game.
3. **Power.** VQA-RAD n = 200 gives a CI half-width of ~0.055 (artifact). "Tie" there means "cannot tell", not
   "equal". Say so.
Add to the box: "WED scored the image at our generation budget (cap320); the probe's features used 4× more pixels.
On 61% of SLAKE questions our 8 samples left WED a single candidate."

**V4. MED — slide 58: two over-statements.**
- "the probe raises open recall on all three splits". On VQA-RAD it is a tie: **+0.0048 [−0.0447, +0.0524]**
  (`final_eval_FINAL_cap320_T07.json`, `cells.vqa_rad_open.all.pick_minus_greedy_llava_recall`). PathVQA (+0.0278
  [+0.0170, +0.0388]) and SLAKE (+0.0300 [+0.0054, +0.0556]) are significant. Fix: "raises open recall significantly on
  PathVQA and SLAKE-en; VQA-RAD ties". Also state that SLAKE closed accuracy dips slightly (72.12 → 71.63, tie), so the
  closed-question claim is about PathVQA only.
- "LLaVA-Med fine-tuned is higher on most columns". With the shipped T = 0.7 numbers it is higher on **all six**: VQA-RAD
  open 61.5–64.8 vs 39.00; closed 83.1–84.2 vs 78.88; SLAKE 83.1–87.1 vs 81.15; 85.3–86.8 vs 71.63; PathVQA 38.0–39.6 vs
  33.43; 91.1–91.2 vs 89.74. Fix: "higher on all six columns". COMPARISON_BRIEF's "only PathVQA open favours us" came
  from the per-benchmark-temperature variant and does not hold for the shipped method.

**V5. MED — slide 56 / script 56: "protocol alone moves scores by ~10 points" (61.9 vs 71.9).**
Problem: Liu et al. prompt "reason step by step … \boxed{}" and score against "the ground truth option" (PAPER_NOTES_GH
G4). They do not state their PathVQA item set. Lingshu's 61.9 combines open (GPT-4.1 judge) and closed (rule-based)
questions. So the gap mixes prompt, grader, answer format and possibly the item set.
Exact fix: "…71.9 in an independent benchmark (step-by-step \boxed{} prompt, option-matched scoring, item set not stated):
evaluation choices alone move the same model by ~10 points."

**V6. LOW — slide 60 details.**
- WED "N ≤ 8" → "≤ 16 draws → ≤ 8 distinct".
- WED's grader → "VLM-as-judge Grok 4.1 Fast".
- BoE row: add "VQA-Med from training split".
- The row "ours +8.0 pts" under BoE's "+0.26 to +0.58" invites a ranking even with the disclaimer. Either bold the
  "baseline" column, or move "vs majority vote +8.9 [+8.1, +9.7]" (`extra_numbers.json`, `probe_minus_vote_judge`)
  into the BoE row's own line as "matching quantity for us". The script does this well; the slide should too.

**V7. LOW — slide 59.** "their judge is GPT-4.1": in Lingshu, closed questions are rule-scored and only open ones go to
GPT-4.1 (§4.2 p.16). Our combined number uses the judge for both. Add "closed rule-scored in Lingshu".

---

## D. Is the WED re-implementation faithful? (src/exp_2026_10/wed/)

**Verdict: faithful in its components, with one adaptation that changes WED's operating regime, and one asymmetry that
should be fixed or disclosed.**

**D1. What is verbatim.** `wed_select.py` imports, unmodified, the authors' `config.ExperimentConfig` (defaults,
`use_wasserstein_update=False`, the same as main.py:315), the prompts, `canonicalize_answer`, `is_bad_candidate`,
`semantic_dedup(0.99)`, `cluster_and_select`, the SapBERT ground metric, `generator_init`, `verifier_init` and
`bdg_wasserstein`. The output is `res_w.answer` (BDG-W). fp16 on GPU matches main.py:49. The Qwen3-VL model class is
aliased to Qwen2.5-VL by a one-line rewrite, because Lingshu-7B is Qwen2.5-VL. That is necessary and correctly
documented. `wed_compare.py` imports WED's own `canonicalize_answer` for "WED's EM/F1", as WED's main.py does. All three
cells have a WED pick for every question (n_wed_picks_on_file = n).

**D2. Adaptations, all disclosed in the docstring.**
- (a) The candidate set is built from our fixed 8 samples by the post-sampling half of `build_candidate_set`, rewritten
  "line-for-line". The sampling loop (T ∈ {0.5, 1.0}, up to 16 draws until 8 distinct) is not run. This is the
  regime change: WED often gets 1–2 candidates instead of 8 (V3.2). The even-number rule in `cluster_and_select` also
  drops a candidate whenever 7, 5 or 3 distinct ones survive. That is faithful to the authors' code, but it matters more
  on small pools.
- (b) Fallback to the modal pool answer when no candidate survives (12 PathVQA, 55 SLAKE, 0 VQA-RAD). The authors
  skip such questions. Answering is required here, and this is the right choice.
- (c) Image cap set to the pool's cap320 (V3.1).

**D3. Diagnostics a referee will ask for [reviewer-computed, point estimates, judge currency].**
- WED's four internal variants on the same questions, judge accuracy:

  | benchmark | greedy | BDG-W (shipped) | BDG-classic | verifier-only | generator-only | probe |
  |---|---:|---:|---:|---:|---:|---:|
  | PathVQA | 0.3140 | 0.2821 | 0.2818 | 0.2860 | 0.2702 | 0.3664 |
  | SLAKE | 0.7302 | 0.7302 | 0.7302 | 0.7070 | 0.7349 | 0.7876 |
  | VQA-RAD | 0.490 | 0.485 | 0.485 | 0.455 | 0.470 | 0.490 |

  Even the best variant picked post hoc stays below the probe everywhere, and below greedy on PathVQA. So the
  Wasserstein stopping rule is not the cause of the result. WED's pick equals the generator-likelihood argmax in
  2,794/3,357 PathVQA questions.
- Restricted to questions where WED actually plays the game (≥ 2 candidates), probe vs WED judge accuracy:
  - PathVQA 0.3000 vs 0.2051 (n = 2,867);
  - SLAKE 0.5810 vs 0.4506 (n = 253);
  - VQA-RAD 0.3841 vs 0.3841 (n = 151).

  The probe's margin is larger where WED's mechanism is active, which argues against "the adaptation starved WED".
- Recommended: add both tables, with image-clustered CIs, to `wed_vs_probe_2026-10-16.json` and a backup slide. Re-run
  WED with its own sampling loop on PathVQA open (WED's prompts, T ∈ {0.5, 1.0}, Nmax 16) as a "WED as published"
  arm. Then the head-to-head covers both "same candidates" and "each method's own candidates".

---

## E. The hardest "how is this different from X?" questions, with the answer to give

**Q1. "This is HSRM / SWIFT / Q-Probe with a different backbone. What is new?"**
A: Nothing in the mechanism, and the deck says so. What is new is the setting and the measurement: an image-conditioned
model; short free-text answers with no exact-match checker; labels from an LLM judge; 8 benchmarks with image-level
cross-fitting; two judges from different families plus exact match; and a controlled comparison with a published
medical selector. Some of their findings cut against our choices: HSRM's BT > BCE, SWIFT's gate > mean, Q-Probe's L_PG >
L_CE.

**Q2. "MedProb App. H already did this in medicine."**
A: On 100 OmniMedVQA items, with N, grader and probe type unstated; its authors call it "an initial demonstration". Ours
is 8,883 OmniMedVQA questions (greedy 0.516 → 0.677 judge) plus seven more benchmarks, cross-fitted by image, in three
currencies. We make no priority claim.

**Q3. "Your probe is trained on MedVLThinker-32B labels and graded by MedVLThinker-32B. Isn't the gain just learning the
judge's taste?"**
A: That risk is real and measured. In this project a newly trained verifier gains ~0.006–0.009 under the training judge
from paraphrase drift alone (`coadapt_verifier_T04_2026-08-14.json`). That is why we report the cross-family MedGemma-27B
judge, never used in training: +0.0710 [+0.0622, +0.0797], 7/8 benchmarks positive. Exact match is also significant but
small: +0.0121 [+0.0030, +0.0210], with significant losses on Kvasir-x1 (−0.0381 [−0.0441, −0.0322]), VQA-Med (−0.0137 [−0.0227, −0.0049]) and GEMeX
(−0.0091 [−0.0162, −0.0021]) (`final_eval_FINAL_cap320_T07.json`, `cells.*.all.pick_minus_greedy_em`). Say that the
EM gain is about one-seventh of the judge gain.

**Q4. "Why not just ask the model (P(True), self-verification) or use its likelihood / self-certainty? No labels needed."**
A: Prompted self-verification: Mirage shows Lingshu accepts 62–73% of its own wrong answers. Likelihood: WED's
generator-only variant, which is likelihood ranking under a "correct" framing, scores 0.270 on PathVQA vs greedy 0.314
[reviewer-computed]. Self-certainty and plain log-prob ranking have not been run. Say that this is a gap (slide 63) and
an owed baseline.

**Q5. "Truthfulness probes (Azaria & Mitchell, CCS, Orgad) show hidden states encode correctness. Isn't this old?"**
A: Yes, decodability is old, and we cite it. Those works classify one statement. We rank 8 answers to the same
question. That is a harder, within-question task: CASE shows the AUC drops from 0.610 to 0.502 once question identity is
removed. We have not yet reported within-question AUROC, and it is next on our list.

**Q6. "Did you beat WED, or a crippled WED?"**
A: Same 8 candidates, authors' code for scoring, game and stopping. None of WED's four variants beats greedy on PathVQA.
Where WED's game actually runs (≥ 2 candidates), the probe's margin is larger. Asymmetries we disclose: WED's sampling
loop was not used, and WED scored images at cap320 while our features used 4× the pixels. A "WED as published" arm is
the fix.

**Q7. "Discriminative reranking is Collins & Koo 2005. Why drop the base model's log-likelihood, which Collins calls
crucial (a₀)?"**
A: It is untested. The likelihood + probe combination is on the gap list (slide 63), and LEVER (2023) is a second
precedent for it.

**Q8. "Why not fine-tune the generator on the same labels (rejection-sampling fine-tuning) instead of a probe?"**
A: The frozen-generator design keeps the model unchanged and the verifier swappable. A generator LoRA-SFT baseline exists
only from an earlier era (+0.0142 at 1× cost, `ckpts/cheapleg/scores_*`, per CLAUDE.md §0a). It has not been re-run
against the final method; slide 64 lists it as owed. Do not set +0.0142 beside +0.0798 as a ranking.

**Q9. "Snell says PRMs beat ORMs and the gap grows with N; Lightman agrees. Why an outcome scorer?"**
A: Our answers are about 5 tokens with no reasoning steps, so a process reward has nothing to score. Uesato 2022 found
outcome supervision gives similar final-answer error with less labelling. We claim nothing about long-chain settings.

**Q10. "LiLaVe and CASE found that pure best-of-n can fall below majority voting. Why does yours not?"**
A: Their answers repeat (math, MCQ). On our open answers majority vote itself loses to greedy (−0.0094 [−0.0157, −0.0034]),
and the probe beats vote by +0.0893 [+0.0811, +0.0972] judge (`extra_numbers.json`). Under exact match, probe-weighted
voting (Snell's aggregator) beats our readout (+0.0135, significant), so the readout choice depends on the currency.

**Q11. "SEPs say entropy-target probes transfer OOD better than accuracy-target probes. Yours fails leave-one-benchmark-out. Why
not train on semantic entropy?"**
A: SEPs predict uncertainty, one number per question, which cannot rank candidates within a question (slide 12). Their
transfer finding is still a fair pointer to an auxiliary target worth trying. It is untested here.
