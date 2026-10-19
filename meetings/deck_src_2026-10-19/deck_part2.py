"""deck_part2.py — Part III (the method, piece by piece, with the math) and Part IV (why each design choice:
what was tested instead, with numbers)."""
from deck_lib import C, fmt, forest, bars, lines
import deck_data as D
import deck_figs as F
import figs_v2 as F2

pm, ci = D.pm, D.ci


def build(d):
    fin = D.fin
    rows0, rows1 = fin["folds"]["0"]["train_rows"], fin["folds"]["1"]["train_rows"]

    # ================================================================== PART III
    d.part = "III · Method"
    d.add("p3", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART III</div>
<h1>Our method, piece by piece</h1><p>sample → re-read → tap three layers → 24 probes → rank-average → answer</p></div>""",
          raw=True, minutes=0.2, say="""# Our method, piece by piece
- Part three: our method, one step at a time.""")

    d.add("pipeline", "The method in one figure", F2.fig_pipeline(), "III · Method", tall=True,
          src="worked_example_12231.json (computed with ckpts/train/final_probe_2026-10-16/fold0, the probes trained without this image); shipped recipe: FINAL_METHOD_2026-10-16.md, single T = 0.7",
          minutes=2.5, look="Walk left to right: the CT and its question; step 1, the eight sampled pills; the three merged answers with their vote badges; step 2, the violet taps at blocks 22 / 20 / 18; step 3, the 24 probe tiles; step 4, the rank bars and the green returned answer.",
          say="""# The method in one figure
- Here is the whole method on one real example from SLAKE: an abdominal CT.
- The question: "what organ is the gray part on the left of the image?" The correct answer is large bowel.
# Sample
- The frozen Lingshu model writes eight answers at temperature 0.7.
- Five say small bowel, two say spleen, and only one says large bowel.
- Identical answers are merged, and we remember how many votes each got.
# Re-read & tap
- Each distinct answer is fed back through the same frozen model, together with the image and question — this is called teacher forcing.
- We read the hidden states at decoder layers 18, 20 and 22, averaged over that answer's own tokens.
- In our runs this re-read used a higher image resolution than the generation; I come back to that.
# Score
- 24 small trained probes — three layers times eight training seeds — each give every answer a score.
# Select
- Each probe's scores are turned into ranks within this question; the ranks are averaged, and the highest wins.
- Here large bowel gets a mean rank of 0.917, and 20 of the 24 probes put it first.
- Majority vote would have returned small bowel, which is wrong.
- Everything on this slide is a real number from the shipped probes, and these probes never saw this image during training.""",
          qa=[("What if two answers tie in mean rank?", "The one produced by more of the 8 samples wins (vote-count tie-break); if the votes also tie, the alphabetically first (lower-cased) answer — arbitrary but deterministic (final_method_eval.py, top[argmax(votes)]). Rank ties occurred on 0.58% of questions for the September probe (AUDIT_2026-09-18.md, statistical-validity table); double ties: not measured."),
              ("Why does the model need two passes?", "It does not have to — the states could be captured while generating (shown for an earlier probe). In our experiments we re-read each answer in a separate pass."),
              ("Is the greedy answer one of the candidates?", "No. The 8 candidates are samples only; greedy is a separate decode and the baseline. Adding greedy as a ninth candidate has not been tested."),
              ("How often is there nothing to choose?", "16.3% of the 36,869 questions have a single distinct answer (from 0.7% on VQA-Med to 56.6% on SLAKE); the method returns that answer, which may differ from greedy (numbers_review_part2.json open8_pools, from the feature-cache metas)."),
              ("Which probes would you actually deploy?", "For evaluation there are two sets of 24, one per fold, each trained without the images it scores (ckpts/train/final_probe_2026-10-16/fold0, fold1) — 48 in total. Either set is deployable; a set refit on all images has not been trained or measured.")])

    # ---- data
    bench_rows = "".join(
        f"<tr><td><b>{D.NAME[c]}</b></td><td>{mod}</td><td>{src}</td><td>{ans}</td><td class='n'>{D.cell(c)['n']:,}</td><td class='n'>{D.cell(c)['n_images']:,}</td></tr>"
        for c, mod, src, ans in [
            ("pathvqa_open", "pathology", "He et al. 2020 (textbook figures)", "short phrase"),
            ("slake_open", "CT · MRI · X-ray", "Liu et al. 2021 (physician-written)", "short phrase"),
            ("vqa_rad_open", "CT · MRI · X-ray", "Lau et al. 2018 (clinician-written)", "short phrase"),
            ("radimagenet_open", "CT · MRI", "RadImageNet-VQA 2025", "anatomy / pathology"),
            ("kvasir_x1_open", "GI endoscopy", "Kvasir-VQA-x1 2025 (LLM-composed)", "<b>sentences</b>"),
            ("omnimed_open", "7 modalities", "OmniMedVQA 2024, options removed", "modality / anatomy"),
            ("vqamed_open", "radiology", "ImageCLEF VQA-Med 2019, abnormality", "rare disease names"),
            ("gemex_open", "chest X-ray", "GEMeX (ThinkVG release)", "<b>sentences</b>")])
    d.add("data", "The data: eight open-ended medical VQA benchmarks",
          f"""
<div class="row">
 <div class="col w3"><table class="d" style="font-size:15px"><tr><th>benchmark</th><th>imaging</th><th>source</th><th>answer style</th><th class="n">questions</th><th class="n">images</th></tr>
 {bench_rows}<tr class="hi"><td><b>total</b></td><td></td><td></td><td></td><td class="n"><b>{D.N_Q:,}</b></td><td class="n"><b>{D.N_IMG:,}</b></td></tr></table></div>
 <div class="col">
  <div class="term"><b class="k">split by image</b><span>all questions about one image go to the same half (A or B)</span></div>
  <div class="term t"><b class="k">probe training data</b><span>other half of the 8 + 4 comparison sets + 4 extra sets (PathVQA/SLAKE/VQA-RAD train splits + Kvasir-VQA): <b>{rows0:,}</b> / <b>{rows1:,}</b> rows per fold</span></div>
  <div class="term w"><b class="k">exposure</b><span>Lingshu trained on PathVQA, SLAKE, VQA-RAD, VQA-Med 2019 training splits; effect on the delta not measured</span></div>
 </div>
</div>""", "III · Method", src="final_eval_FINAL_cap320_T07.json cells.*.all.{n,n_images}, folds.{0,1}.train_rows (row breakdown: numbers_review_part2.json, recomputed from the feature-cache metas, reproduces both totals); Lingshu Table 5 p.12; benchmark sources per PAPER_NOTES_GH",
          minutes=2.0, look="The table top to bottom: the three classic benchmarks, then the five that broaden coverage; then the three boxes on the right.",
          say=f"""# benchmark (table, left)
- Our data: eight open-ended medical VQA benchmarks, {D.N_Q:,} questions over {D.N_IMG:,} images.
- They cover pathology, radiology, endoscopy and chest X-ray.
- Three are the classic ones — PathVQA, SLAKE and VQA-RAD — with the official open-ended test questions. The others broaden the coverage.
- Two have whole-sentence answers written by a language model, Kvasir and GEMeX — this matters later for exact match.
# split by image
- Many questions share one image, so every split is made by image: all questions about an image go to the same half.
# probe training data
- The probe's training data is the other image half of every set we score, including three closed-question sets and the VQA-Med 2019 test.
- Plus about six thousand questions from the training splits of PathVQA, SLAKE and VQA-RAD and from Kvasir-VQA.
# exposure
- Our generator was trained on the training splits of four of these benchmarks; our VQA-Med cell is mostly that training split.
- So its absolute accuracy there is not a fair measure of the model.
- Our comparisons are always the same model against itself, so exposure affects both arms.
- The gain is largest on the four benchmarks not in Lingshu's supervised training data — reassuring, but those four also differ in other ways.
- One of them, OmniMedVQA, has reported text exposure for the base model, so I do not claim this proves anything.
- The effect of exposure on our gain is not measured.""",
          qa=[("How was OmniMedVQA made open-ended?", "Its multiple-choice options were removed; only modality and anatomy questions kept; the RadImageNet source dropped because RadImageNet-VQA is a separate benchmark of ours. "
               "Scores are therefore not comparable to published OmniMedVQA (MCQ) numbers."),
              ("Is the GEMeX cell the GEMeX test set?", "No — open-ended rows from the GEMeX-ThinkVG release (gold ≤ 12 words), not the 300-image GEMeX test. VQA-Med C4 is mostly the 2019 training split; the official 500-question test is scored separately."),
              ("What exactly is the training set?", "Per fold: the opposite image half of the 8 open benchmarks (81,331 / 81,554 rows) and of the 4 comparison sets (3,366 / 3,325 rows: closed VQA-RAD, PathVQA, SLAKE and the VQA-Med 2019 test), plus a fixed 6,029-question subsample of Kvasir-VQA and the PathVQA/SLAKE/VQA-RAD training splits (31,237 / 31,121 rows after removal). Any row whose image is in the scored half is removed (exact pixel match). Totals 115,934 / 116,000 (numbers_review_part2.json; reproduces final_eval_FINAL_cap320_T07.json folds.*.train_rows)."),
              ("Why Lingshu-7B?", "An open 7B medical VLM whose published numbers MedEvalKit reproduces, so every claim is anchored to a public baseline. The MedGemma-4B replication was withdrawn (its sampled answers ran to the token cap); Qwen2.5-VL-7B, Lingshu's base, is the second generator.")])

    # ---- what sampling does: one input, different answers (real samples, SLAKE 12231)
    import figs_sampling as FS
    d.add("sampling", "One question, many answers: temperature sampling",
          f"""<div class="center">{FS.fig_sampling()}</div>""",
          "III · Method", tall=True,
          src="sampling_example_12231.json (sampling_example.py; ckpts/openvqa/cheap_lingshu7b/ckpt_slake_open_{lingshu7bT02,lingshu7bT04,lingshu7b,lingshu7bT10}_sc8.jsonl "
              "preds/oks, greedy ckpt_slake_open_lingshu7b.jsonl seqlogprob −0.2718); middle bars schematic; distinct per question: numbers_review_part2.json open8_pools",
          minutes=1.5,
          look="Left to right: the same image and question → the same frozen model → a random draw at every token → four columns of real answers. Point at the shipped T = 0.7 column, then the greedy box.",
          say="""# same input
- One image, one question, one frozen model — nothing about the input changes between runs.
# each next token is drawn at random
- At every step the model gives a probability for each possible next token.
- Greedy decoding always takes the top one; sampling draws one at random, in proportion to those probabilities.
- Temperature T reshapes them: low T makes the top token dominate, high T flattens them.
- These bars are a schematic — we did not store the token probabilities.
# real samples for this question (8 per temperature)
- These are the model's real answers to this question, eight runs at each temperature.
- At T = 0.2 and 0.4 it says Small Bowel seven times out of eight — and never the right answer.
- At T = 0.7, our setting, three different answers appear, and one of the eight is right: Large Bowel.
- At T = 1.0 five different answers appear; Large Bowel comes up twice, but so do Liver and Kidney.
# greedy decoding
- Greedy says Small Bowel, which is wrong, and says it every single time.
- The model gives that answer about 76 percent of its probability; the right answer is in the remaining 24 percent.
- So sampling is what puts the right answer in the pool at all — the probe's job is then to pick it out.
- Across all 36,869 questions, eight samples give 4.42 different answers on average; only 16.3 percent of questions get a single answer.""",
          qa=[("Why not just raise the temperature further?", "More diversity also means more wrong answers in the pool: here T = 1.0 brings Liver and Kidney. "
               "The measured trade-off across benchmarks is on the temperature slide in Part IV: T = 0.7 was the best single setting under the judge."),
              ("Are the bars real probabilities?", "No — a schematic. Per-token probabilities were not stored for these pools; the only real number is the greedy "
               "answer's whole-sequence log-probability, −0.2718, i.e. about 0.76 (ckpt_slake_open_lingshu7b.jsonl seqlogprob).")])

    # ---- step 1 sampling
    cost = D.cost
    d.add("step1", "Step 1 — sampling: 8 answers, one image encoding",
          f"""
<div class="row">
 <div class="col">
  <div class="term"><b class="k">what is sampled</b><span>N = 8 · <b>T = 0.7</b> · cap320 (≤ 320 image tokens) · ≤ 64 new tokens · plain temperature sampling</span></div>
  <div class="term"><b class="k">shared prefill</b><span>image + prompt read once, 8 continuations: 1.13× the arithmetic of greedy (modelled)</span></div>
  <div class="term t"><b class="k">deduplicate, keep votes</b><span>merge identical answers (lower-case + trim only), keep the count \\(v_i\\): 4.42 distinct per question</span></div>
 </div>
 <div class="col">
  <table class="d"><tr><th>per question, batch 1, A100</th><th class="n">greedy</th><th class="n">best-of-8</th><th class="n">ratio</th></tr>
  <tr><td>GFLOPs (modelled, shared prefill)</td><td class="n">5,831</td><td class="n">6,589</td><td class="n">1.13×</td></tr>
  <tr><td>forward tokens</td><td class="n">328.5</td><td class="n">371.2</td><td class="n"></td></tr>
  <tr><td>latency</td><td class="n">174.0 ms</td><td class="n">476.6 ms</td><td class="n">2.74×</td></tr>
  <tr><td>energy</td><td class="n">34.3 J</td><td class="n">123.2 J</td><td class="n">3.59×</td></tr></table>
  <div class="note" style="margin-top:8px">ms and J measured on 20 VQA-RAD open questions, ≤ 32 new tokens, one request in flight — not measured on the sentence-answer benchmarks (Kvasir-x1, GEMeX). FLOPs understate the cost: decoding 8 sequences is memory-bound.</div>
 </div>
</div>""", "III · Method", src="bestofn_vllm_2026-09-16.json (ms and J measured: NVML, 20 timed calls after 3 warm-ups, vqa_rad open items, max_new 32; 1.13× FLOP-eq); GFLOPs modelled = analytic FLOP model (free_head_2026-08-16.json null_tests.flop_model_rebuild) × forward-token ratio; 4.42 distinct = 162,885 rows / 36,869 questions (numbers_review_part2.json); dedup: extract_generator_hidden.py norm(); sampling: run_openvqa.py:186",
          minutes=1.5, look="The three boxes on the left, top to bottom; then the cost table — point at the ratio column.",
          say="""# what is sampled
- For each question the model writes eight answers at temperature 0.7.
- The prompt: "Answer the question with a short, specific phrase. Do not explain."
- The image is capped at about 250 thousand pixels, so it becomes at most 320 image tokens.
- Each answer is at most 64 tokens.
# shared prefill
- Sampling eight answers is cheaper than it sounds: the serving engine reads the image and question once — the shared prefill — then continues eight times.
- In modelled arithmetic it is only 1.13 times greedy.
- But arithmetic understates the real cost. Measured on one A100: 174 milliseconds greedy, 477 milliseconds for eight answers.
- Energy goes from 34 to 123 joules — about 2.7 times the time and 3.6 times the energy.
- Those are short VQA-RAD answers; on the sentence-answer benchmarks it would be more, and we have not measured it there.
# deduplicate, keep votes
- Identical answers are merged — only case and surrounding spaces are ignored.
- We keep the count of how many samples produced each.
- On average there are 4.4 distinct answers per question.""")

    # ---- step 2 tokens
    d.add("step2", "Step 2 — where the probe reads: the answer's own tokens",
          F2.fig_tokens() + r"""
<div class="row" style="height:auto;margin-top:4px"><div class="col"><div class="eq">
$$\bar h_i^{(\ell)}=\frac{1}{|\mathcal A(y_i)|}\sum_{t\in\mathcal A(y_i)} h_t^{(\ell)}\in\mathbb R^{3584},\qquad \ell\in\{18,20,22\}$$</div></div>
<div class="col"><div class="term w"><b class="k">as run</b><span>a separate <b>teacher-forced</b> pass per distinct answer, at the <b>1-megapixel cap</b> (max_pixels 1,003,520, ≤ 1,280 image tokens; “full resolution” in our files). Generation: cap320 (250,880).</span></div></div></div>""",
          "III · Method", tall=True, src="extract_generator_hidden.py (h_span = mean over the candidate's tokens; layers index output_hidden_states, 0 = embeddings); token counts: numbers_review_part2.json (Lingshu tokenizer), mean ≈ 274 image tokens at cap320 (free_head_2026-08-16.json); bound = max_pixels/784 (2×2 merge of 14-px patches); AUDIT_2026-09-18.md §4 (two resolutions); select_policy_cap320_vs_fullres_T07.json",
          minutes=2.5, look="The dashed teal box (the answer's tokens), then the three violet lines (blocks 18, 20, 22), then the formula at the bottom.",
          say="""# system prompt · image tokens · question · answer 𝒜(y)
- This is the sequence the language model sees: the system prompt, the image tokens, the question, then one candidate answer.
- Each horizontal line is the residual stream after one of the 28 blocks.
- The model computes a hidden state at every position after every block.
# answer 𝒜(y) (dashed box)
- The probe reads only two things: the positions of the answer's own tokens — the dashed box — and only three blocks, 18, 20 and 22.
- At each of those blocks it averages the answer's vectors: one 3,584-number vector per layer — the formula at the bottom.
- The answer comes after the image and question, and the model attends to everything before it.
- So each answer vector already mixes in the image and the question — that is how a probe reading only answer tokens is image-aware.
# image tokens
- For scale: one image token covers 28 by 28 pixels, so the cap gives at most 320 image tokens — about 274 on average.
- In this example the question is 13 tokens and the answer 4.
# as run
- How we get these states: each distinct answer is fed back through the frozen model after the image and question — teacher forcing — one pass per answer.
- Precisely: in our runs this re-reading used a higher image budget than generation — a one-megapixel cap, which our files call "full resolution", instead of 250 thousand pixels.
- That is a difference between how we ran it and how it would be deployed.
- The direct test — re-extracting these states at the generation budget — is running now.
- Indirect evidence: regenerating the samples at the one-megapixel cap, so generation and features matched, gave plus 0.0745 instead of plus 0.0790.
- Matching did not help, but that test also changed the samples.""",
          qa=[("Why the answer tokens and not the last token or the question?", "Measured: mean over the answer span ≥ last token at every layer from 14 up; adding question or image token means adds ≤ +0.0018 (below the refit floor). See the feature-choice slide."),
              ("How does a probe that reads only answer tokens see the image?", "Through causal attention: every answer position attends to the image and question tokens before it. Explicitly adding image- or question-token means added at most +0.0018 (judge, 3 seeds; head_visual_features_2026-09-01.json)."),
              ("Exactly which positions are pooled?", "The answer's own tokens as the tokenizer splits the answer string, including a trailing period; no end-of-turn token (nothing is appended after the answer). extract_generator_hidden.py: the last n_ans_tok positions."),
              ("Why not capture the states while generating?", "That is the design. It was tested for an earlier (August, layer-21) probe: states read from model.generate()'s own steps vs a teacher-forced pass at the same resolution changed 24 of 2,345 picks, accuracy 0.0000 [−0.0021, +0.0021] (free_head_2026-08-16.json). Not yet repeated for the shipped probe inside vLLM.")])

    # ---- layers
    lg = [(18, 0.1036), (19, 0.1020), (20, 0.1007), (21, 0.1008), (22, 0.1028), ("ens", 0.1051)]
    d.add("layers", "Which layers: 18, 20, 22 of 28 — upper-middle depth",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 600px">{F2.fig_depth(lg)}</div>
 <div class="col">
  <div class="term"><b class="k">indexing</b><span>“layer ℓ” = <code>hidden_states[ℓ]</code> = residual stream after decoder block ℓ (index 0 = embeddings). Relative depth 0.64 / 0.71 / 0.79.</span></div>
  <div class="term t"><b class="k">why three layers</b><span>single layers 18–22 all give +0.101 to +0.104 (5 benchmarks); the 18/20/22 ensemble +0.105. The spread (≈ 0.003) is noise level, so three layers are a <b>hedge</b> against picking the wrong one, not a measured gain (pooled: +0.0007).</span></div>
  <div class="term w"><b class="k">how chosen</b><span>on image half A, 5 of 8 benchmarks, judge only, no interval. Half B is out of sample for this choice.</span></div>
  <div class="term"><b class="k">literature</b><span>HSRM: upper layers strongest, the final layer “not always the unique optimum”. MedProb: peak at layer 28 (MCQ task).</span></div>
 </div>
</div>""", "III · Method", src="head_ens_width_2026-08-25.json macro.* (5 of 8 benchmarks, 5 seeds, August-era pooled probe, judge); head_final_stack_PVFIXED_2026-09-13.json pooled_ens − pooled_singlelayer; HSRM Fig. 3 p.6; MedProb Fig. 5 p.7",
          minutes=1.5, look="The violet rows on the left (blocks 18, 20, 22), then the five bar values on the right, then the ensemble line.",
          say="""# indexing
- Why layers 18, 20 and 22? First a precise definition: layer 18 means the vector after decoder block 18 of 28.
- So these sit at about two-thirds to four-fifths of the depth.
# why three layers
- The bars show the gain over greedy for a probe trained on each single layer, on the five benchmarks where this was measured.
- All are between 0.101 and 0.104, so the differences are at noise level.
- The three-layer ensemble is 0.105.
- Honestly, the ensemble is a hedge against choosing the wrong single layer, not a real gain — on the full pool it adds 0.0007.
# literature
- The literature agrees that correctness information is spread over the upper half of the network.
- HSRM finds the upper layers best; MedProb finds a later peak for a different, multiple-choice task.""",
          qa=[("Were the layers chosen on test data?", "Yes, partly: the set was the argmax of 8 layer sets scored end-to-end on image half A, on 5 of 8 benchmarks (AUDIT_2026-09-18.md §5). All candidates lie within noise of each other, so the choice cannot have bought much. Image half B is out of sample for it: judge +0.0846 [+0.0729, +0.0965], lenient EM +0.0165 [+0.0026, +0.0299] (final_fold1_headline_2026-10-16.json).")])

    # ---- probe
    d.add("probe", "Step 3 — the probe: one hidden layer, 918,017 parameters",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 560px">{F.fig_probe()}</div>
 <div class="col">
  <div class="eq">$$\\tilde h_i^{{(\\ell)}}=\\big(\\bar h_i^{{(\\ell)}}-\\mu^{{(\\ell)}}\\big)\\oslash\\big(s^{{(\\ell)}}+\\epsilon\\big)$$</div>
  <div class="eq">$$f_{{\\ell,k}}\\big(\\tilde h_i^{{(\\ell)}}\\big)=w_2^{{\\top}}\\,\\mathrm{{GELU}}\\!\\left(W_1 \\tilde h_i^{{(\\ell)}}+b_1\\right)+b_2$$</div>
  <table class="d sm" style="width:auto;min-width:360px"><tr><th>parameter</th><th>shape</th><th class="n">count</th></tr>
  <tr><td>\\(W_1\\)</td><td>256 × 3584</td><td class="n">917,504</td></tr><tr><td>\\(b_1\\)</td><td>256</td><td class="n">256</td></tr>
  <tr><td>\\(w_2\\)</td><td>256</td><td class="n">256</td></tr><tr><td>\\(b_2\\)</td><td>1</td><td class="n">1</td></tr>
  <tr class="hi"><td><b>total</b></td><td></td><td class="n"><b>918,017</b></td></tr></table>
  <div class="note">\\(i\\) = distinct answer, \\(k\\) = seed (0–7); \\(\\mu, s\\): training-set mean / std per dimension (\\(\\epsilon=10^{{-6}}\\)), frozen with the probe.</div>
  <div class="term t" style="margin-top:8px"><b class="k">reads as</b><span>a <b>score</b>, not a calibrated probability; only its order within a question is used.</span></div>
 </div>
</div>""", "III · Method", src="ckpts/train/final_probe_2026-10-16/fold{0,1} (recipe.json: Linear(3584→256)-GELU-Linear(256→1)); 3584·256+256+256+1 = 918,017; standardiser: final_method_eval.py (std + 1e-6)",
          minutes=1.5, look="The figure bottom to top: input vector, standardise, the two trained Linear layers with GELU between, the score at the top; the dashed sigmoid box is training only. Then the two equations.",
          say="""# standardise
- The probe takes the 3,584-number vector and standardises each dimension with the mean and standard deviation from training.
# GELU
- It multiplies by a 256-by-3,584 weight matrix, applies the GELU non-linearity, and a final weighted sum gives one number: a score.
# reads as
- Higher means the probe believes more strongly that the answer is correct — but it is not a calibrated probability.
- The loss up-weights correct answers, and on our example the correct answer's scores are all negative.
- We only ever use its order within a question.
# σ(·) → class-weighted BCE
- The sigmoid in the dashed box exists only inside the training loss.
# parameter (table)
- That is 918 thousand parameters — tiny next to the 7-billion-parameter model it reads.
- It is pointwise: it looks at one answer at a time and never compares answers; the comparison happens in the next step.""")

    # ---- training
    d.add("training", "Training the probe: labels, loss, data",
          f"""
<div class="row">
 <div class="col">
  <div class="term"><b class="k">label \\(c_i\\in\\{{0,1\\}}\\)</b><span>is answer \\(y_i\\) correct? — the training judge, MedVLThinker-32B (text-only, sees the gold); one label per (question, distinct answer)</span></div>
  <div class="term t"><b class="k">loss: class-weighted binary cross-entropy</b><span>$$\\mathcal L=-\\big[w_+\\,c\\log\\sigma(f)+(1-c)\\log(1-\\sigma(f))\\big]$$ \\(w_+=(1-\\pi)/\\pi\\approx3.8\\); π = share of correct rows in the fold's training set (0.207 / 0.209). Uses every row, incl. all-right / all-wrong questions.</span></div>
  <div class="term"><b class="k">optimiser</b><span>AdamW, lr 1e-3, weight decay 1e-2, 30 epochs (no early stopping), batch 256; 8 seeds × 3 layers = 24 probes per fold (a deep ensemble × a layer ensemble).</span></div>
 </div>
 <div class="col">
  <div class="term"><b class="k">training rows</b><span><b>{rows0:,}</b> / <b>{rows1:,}</b> per fold: other image half of the 8 open benchmarks + comparison sets + 4 extra sets: PathVQA/SLAKE/VQA-RAD train splits + Kvasir-VQA (breakdown: backup recipe)</span></div>
  <div class="term w"><b class="k">what this means</b><span>the probe learns the training judge's notion of correct; training needs labels per benchmark, inference needs no dataset label</span></div>
 </div>
</div>""", "III · Method", src="final_eval_FINAL_cap320_T07.json folds.*.train_rows; head_sweep.py:172-174 (pos_weight = (1−π)/π); π and row breakdown: numbers_review_part2.json; final probes fitted on T = 0.7 pools",
          minutes=2.0, look="The loss formula in the teal box — point at the weight w₊; then the training-rows box.",
          say="""# label (top-left box)
- For every training question we have the eight sampled answers; each distinct answer is labelled correct or incorrect by the judge model.
# loss: class-weighted binary cross-entropy
- The probe is trained with binary cross-entropy — the standard loss for predicting a yes-or-no label.
- Correct answers are up-weighted, because only about one training answer in five is correct: each counts about 3.8 times.
- It uses every answer, including questions where all answers are right or all are wrong — which ranking losses have to throw away.
# optimiser
- AdamW, learning rate ten to the minus three, weight decay ten to the minus two.
- 30 epochs at batch size 256, no early stopping.
- Eight random seeds for each of the three layers gives the 24 probes.
# training rows
- The training data is the other half of the images of all eight benchmarks and of the closed-question comparison sets.
- Plus about six thousand questions from the PathVQA, SLAKE and VQA-RAD training splits and Kvasir-VQA — about 116 thousand answer rows per fold.
# what this means
- The probe learns the training judge's idea of correct — which is why we check with a second judge.
- It needs labelled data from each benchmark, which I quantify later.""")

    # ---- readout
    d.add("readout", "Step 4 — the readout: rank averaging with a vote tie-break",
          F.fig_readout() + r"""
<div class="row" style="height:auto"><div class="col" style="flex:0 0 620px"><div class="eq">
$$\begin{aligned} r_i&=\frac1{24}\sum_{\ell,k}\frac{\operatorname{rank}_{\ell,k}(i)}{\max(n-1,1)}\\ \hat y&=\arg\max_i r_i\qquad(\text{ties}\to\arg\max_i v_i)\end{aligned}$$</div>
<div class="note">n = distinct answers of this question (1 ≤ n ≤ N); rank = 0-based position in ascending score order, tied scores share the average rank (a Borda count). Ties in r → more votes; ties in votes too → first alphabetically.</div></div>
<div class="col"><div class="term t"><b class="k">why ranks</b><span>the 24 probes are trained independently; their raw scores are on different scales. Ranks are scale-free, and only the order within one question matters for an argmax. Averaging raw, standardised or calibrated scores tied (Part IV).</span></div></div></div>""",
          "III · Method", tall=True, src="worked_example_12231.json (raw scores: candidates[*].probe_scores); readout: genframe_data.rank_avg (ties averaged, ÷ max(n−1, 1)), final_method_eval.py tie-break; adopted 2026-10-15 under a pre-specified rule (readout_variants_2026-10-15.json)",
          minutes=1.5, look="Read one row of the grid — Large Bowel, 24 cells — then the mean column; then the formula.",
          say="""# Large Bowel · Small Bowel · Spleen (rank grid)
- Each of the 24 probes ranks the distinct answers of this question from worst to best; n is the number of distinct answers.
- We divide by n minus one so ranks run from 0 to 1, and average over the 24 probes — in voting-theory terms, a Borda count.
- These are the real ranks for our example: large bowel is ranked first by 20 of the 24 probes.
- Small bowel had five votes but is ranked first by only four probes. Spleen is last everywhere.
# selected
- The answer with the highest average rank wins; if two are exactly tied, the one more samples produced wins.
# why ranks
- Each probe is trained separately, and their scores live on different scales.
- On this very question, spleen scores minus 37.4 under one layer-18 probe and minus 7.2 under a layer-22 probe.
- The raw numbers are not comparable across probes, but the order is.
- Averaging raw, standardised or calibrated scores instead all tied with rank averaging under both judges — so ranks are the scale-free default, not a measured win.""")

    # ---- probes vs N
    d.add("probesN", "24 probes ≠ 8 samples — the probes work for any N",
          F.fig_probes_vs_n() + """
<div class="row" style="height:auto;margin-top:2px">
 <div class="col"><div class="term"><b class="k">24 = 3 layers × 8 seeds</b><span>“8 seeds” are 8 independent training runs per layer. They are unrelated to the 8 samples.</span></div></div>
 <div class="col"><div class="term t"><b class="k">other N, same probes</b><span>8 → 16 samples: +0.0167 macro (4 wins, 4 ties, 0 losses). RadImageNet 8/16/32: +0.122 → +0.154 → +0.193.</span></div></div>
 <div class="col"><div class="term w"><b class="k">N = 16 with the final probes</b><span>judge +0.0984 vs +0.0798 at N = 8; lenient EM +0.0140 vs +0.0121 (point estimates) — at about twice the generation. N = 4: not measured with the probe.</span></div></div>
</div>""", "III · Method", tall=True,
          src="coverage_sc16_ci_ALL_2026-09-16.json (September probe, judge); budget_conversion_sc32_2026-09-17.json (RadImageNet N = 32 sampled at T = 1.0); final_eval_FINAL_cap320_MIX_T07_16.json in the main checkout's artifacts/ (MedGemma labels incomplete for 16-sample pools)",
          minutes=1.5,
          say="""# Changing N changes how many answers are scored, never which probes are used.
- A question that always comes up: we ship 24 probes and sample 8 answers — are they paired? No.
# 24 = 3 layers × 8 seeds
- The 24 are three layers times eight training seeds; the eight seeds are just eight independent training runs.
- Each probe scores each distinct answer on its own: with four samples every probe scores at most four answers, with sixteen at most sixteen.
- Nothing is dropped and nothing is reused. N equal to four itself we have not measured with the probe.
# other N, same probes
- And it works: the same probes gave plus 0.0167 when going from 8 to 16 samples, with gains on four benchmarks and no losses.
- On RadImageNet the gain keeps rising up to 32 samples, although that 32-sample pool was drawn at a higher temperature.
# N = 16 with the final probes
- With the final probes, sixteen samples gives plus 0.098 instead of plus 0.080 under the judge, and a little more under exact match.
- That is at roughly twice the generation cost.""",
          look="Three panels, left to right: the sample column grows, the 24-probe grid stays the same.",
          qa=[("Were the probes trained on 8-sample pools?", "Yes — but the probe is pointwise and ranks are divided by (n−1), so nothing in it depends on N. The N = 16 and N = 32 results above are measured, not assumed."),
              ("What if you used one probe instead of 24?", "One layer instead of three: −0.0007 (head_final_stack_PVFIXED_2026-09-13.json, pooled_ens − pooled_singlelayer, September). Seeds: an August readout with 2 seeds gave selection efficiency 0.8120 vs 0.8283 with 8 (OPENTEXT_CORRECTIONS_2026-08-19.md §2) — but the same doc shows CPU thread count alone moves that number by up to 0.035, so it is indicative only. A single probe on the final recipe: not measured."),
              ("Is N = 16 better in every metric?", "Under the training judge (+0.0984 vs +0.0798) and lenient EM (+0.0140 vs +0.0121), yes in point estimate. The cross-family number for 16-sample pools (+0.0674) is computed on incomplete MedGemma labels, so it is not a valid comparison yet (final_eval_FINAL_cap320_MIX_T07_16.json).")])

    # ---- evaluation protocol
    d.add("evalproto", "Evaluation: scored only by probes that never saw the image",
          F.fig_crossfit() + """
<div class="row" style="height:auto">
 <div class="col"><div class="term"><b class="k">metrics</b><span>training judge (MedVLThinker-32B) · independent judge (MedGemma-27B-it) — both text-only, see the gold · lenient EM · token-F1 · LLaVA-Med recall</span></div></div>
 <div class="col"><div class="term t"><b class="k">baseline</b><span>the same model's greedy answer, same prompt, same image cap. Cross-fitting = 2-fold grouped (by image) cross-validation.</span></div></div>
 <div class="col"><div class="term w"><b class="k">macro + clustered CI</b><span>equal weight per benchmark; each benchmark resampled by image (10,000×); CI = 2.5/97.5 percentile of the mean of the 8 resampled deltas. Probes and split held fixed.</span></div></div>
</div>""", "III · Method", tall=True, src="final_eval_FINAL_cap320_T07.json (bootstrap: 10,000 resamples, clustered by image within benchmark); split: md5('nd'+pixel_md5) mod 2, fold f scores half f; run_judge.py (both judges)",
          minutes=1.5, look="The two fold rows: train on one half, score the other; then the green box on the right.",
          say="""# score every question in half A
- Every image is hashed into one of two halves, A and B.
- In fold zero we train the probes on half B and score half A; in fold one the reverse.
- So every one of the 36 thousand questions is scored by probes that never saw its image or any question about it.
- "Never saw" means an exact pixel match; near-duplicate images were not checked.
- This is called cross-fitting — in machine-learning terms, two-fold cross-validation grouped by image. We train two sets of 24 probes, one per fold.
# baseline
- We compare against the same model's greedy answer.
# metrics
- We use five metrics. The main judge also produced the training labels, so the second judge, from another model family, matters most.
# macro + clustered CI
- Averages are macro: each benchmark counts one eighth.
- Confidence intervals come from a bootstrap that resamples whole images, because questions about the same image are not independent.
- Resampling individual questions would make the intervals too narrow.""",
          qa=[("Why not just one held-out half?", "That was the September protocol (+0.0737 on one half, exact match a tie). Cross-fitting doubles the scored sample; the method also changed since. Lenient EM is significant pooled (+0.0121 [+0.0030, +0.0210]) and on half B alone (+0.0165 [+0.0026, +0.0299]; final_fold1_headline_2026-10-16.json)."),
              ("Do the intervals include the randomness of probe training?", "No — the bootstrap holds the probes and the fold split fixed. Refit noise is about 0.003; fold-split variance is not measured. The macro CI is conditional on these eight benchmarks. Listed as a gap."),
              ("Was the image matching only exact?", "Yes: images are hashed on decoded pixels, so exact duplicates across sets fall into the same half and are removed from training. Re-encoded or resized near-duplicates are not detected (AUDIT_2026-09-18 appendix DI-5 lists perceptual-hash candidates, not resolved)."),
              ("How does the judge decide?", f"See the answer prepared for slide {d.number_of('judges')} (grading open answers)."),
              ("How often do the two judges agree?", "On the 112,770 September training rows, the two judges give the same correct/incorrect label to 92.9% of candidates; Cohen's κ = 0.79, derived from that agreement and the two positive rates (0.194 vs 0.239) (judge_2x2_2026-09-20.json train_label_agreement_AB, train_label_pos_rate; κ in numbers_review_part2.json). Not recomputed on the final pools.")])

    # ---- cost
    d.add("cost", "What it costs: cheap sampling, negligible probe, costly re-read",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 330px">
  {bars(["greedy", "best-of-8"], [("latency", C["indigo_l"], [174.0, 476.6])], 0, 500, 100, w=320, h=260, ylab="ms per question", value_fmt=lambda v: f"{v:.1f}")}
  <div class="note">latency, batch 1, A100 80GB, vLLM — 20 VQA-RAD questions</div>
 </div>
 <div class="col" style="flex:0 0 330px">
  {bars(["greedy", "best-of-8"], [("energy", C["teal"], [34.3, 123.2])], 0, 140, 20, w=320, h=260, ylab="J per question", value_fmt=lambda v: f"{v:.1f}")}
  <div class="note">energy (NVML, integrated over the call) — same 20 questions</div>
 </div>
 <div class="col">
  <div class="term"><b class="k">the probe</b><span>24 × 918,017 ≈ 22 M multiply-adds per answer — negligible next to ≈ 6,589 GFLOPs (modelled) of best-of-8 generation</span></div>
  <div class="term w"><b class="k">the re-read, as run (HF)</b><span>one 1-MP-cap pass per distinct answer: 3.81 passes = 6.90 greedy-pass equivalents per question</span></div>
  <div class="term r"><b class="k">not measured</b><span>end-to-end latency of the whole system; VRAM; sentence-answer benchmarks</span></div>
 </div>
</div>""", "III · Method", src="bestofn_vllm_2026-09-16.json measured.gen1/gen8 (20 vqa_rad open items, max_new 32; vLLM shared prefill); probe = 918,017 multiply-adds per answer per probe (by design); free_head_2026-08-16.json DELIVERABLE_3_cost (HF accounting: passes_per_question 3.81 on 3 benchmarks, per_forward_flopeq.head_extraction_fullres_DEPLOYED 1.81; capture test)",
          minutes=1.0, look="Left chart (milliseconds), middle chart (joules), then the three boxes — the orange one is the honest caveat.",
          say="""# What it costs: cheap sampling, negligible probe, costly re-read
- Sampling eight answers with a shared prefill costs 2.7 times the time and 3.6 times the energy of greedy, measured on short VQA-RAD answers.
# the probe
- The probe itself is about 22 million multiply-adds per answer — negligible.
# the re-read, as run (HF)
- The honest caveat is the re-reading pass: as we ran it, every distinct answer was re-read at the one-megapixel cap.
- That pass is not small — about 6.9 greedy passes' worth of arithmetic per question.
- That figure comes from a different measurement setup, so I do not add the two into one total.
- By design those states can be captured while generating, at no extra cost; we showed for an earlier probe that this gives the same picks.
- The re-read could also stop at block 22.
# not measured
- We have not measured the whole system end to end.""",
          qa=[("Total cost as run vs greedy?", "Not measured end to end, and the two components come from different harnesses: sampling with vLLM's shared prefill 1.13× FLOP-eq (modelled), 2.74× latency, 3.59× energy on 20 VQA-RAD questions (bestofn_vllm_2026-09-16.json); the HF re-read 6.90 greedy-pass equivalents per question (free_head_2026-08-16.json, 3 benchmarks). Because the accountings differ, I do not add them into one number. The re-read pass was never timed.")])

    # ================================================================== PART IV
    d.part = "IV · Why"
    d.add("p4", "", """<div class="divider" style="position:absolute;inset:0"><div class="num">PART IV</div>
<h1>Why each choice</h1><p>For every component: what we tried instead, the measured result, and the verdict.<br>An earlier experiment is shown only where it is the evidence for a shipped choice.</p>
<div style="position:absolute;left:82px;right:80px;top:540px;display:grid;grid-template-columns:repeat(4,1fr);gap:14px;font-size:15px;line-height:1.35">
 <div style="border-top:3px solid rgba(255,255,255,.55);padding-top:8px"><b>July</b><br>LoRA-verifier era: sample pools, self-verification, zero-shot 32B verifier</div>
 <div style="border-top:3px solid rgba(255,255,255,.55);padding-top:8px"><b>August</b><br>first probe (genframe_head_ens8: layer 21, 4 domains), then pooled-probe v1 tests</div>
 <div style="border-top:3px solid rgba(255,255,255,.55);padding-top:8px"><b>September</b><br>pooled probe v2 (genframe_head_pooled_ens_v2), scored on image half A</div>
 <div style="border-top:3px solid #fff;padding-top:8px"><b>October = every headline number</b><br>final cross-fitted probes (final_probe_2026-10-16)</div>
</div></div>""",
          raw=True, minutes=0.5,
          say="""# Why each choice
- Part four: for every design choice, what we tried instead and why we chose what we did.
- Some evidence comes from earlier versions of the probe, so here is the timeline.
# July
- In July the project used a fine-tuned LoRA verifier.
# August
- In August we built the first probe on a single layer, then pooled the training data.
# September
- In September that became the pooled probe, version two, scored on one image half.
# October = every headline number
- The October probes, cross-fitted on both halves, give every headline number.
- When a slide says July, August or September, it is evidence from an earlier version.""")

    d.add("whytable", "Design decisions at a glance",
          """
<div class="note" style="margin-bottom:6px"><b style="color:var(--teal-d)">out of sample:</b> choices were made on image half A; on half B alone the method gains judge +0.0846 · MedGemma +0.0721 · lenient EM +0.0165, all significant (the cap320 check used both halves)</div>
<table class="d" style="font-size:14.5px">
<tr><th>decision</th><th>shipped</th><th>training-judge verdict</th><th>lenient EM</th></tr>
<tr><td>selector</td><td>trained probe</td><td class="win">beats vote, prior, random, longest, self-verification · ties zero-shot 32B ¹ · WED: 2 wins, 1 tie</td><td class="win">probe − vote +0.0136 (sig.)</td></tr>
<tr><td>verifier type</td><td>probe on hidden states</td><td class="win">+0.0162 vs LoRA verifier (Aug. probe)</td><td class="loss">−0.0145 → a wash</td></tr>
<tr><td>what is read</td><td>mean over answer tokens</td><td class="tie">span ≥ last (CV); extras ≤ +0.002</td><td>not measured</td></tr>
<tr><td>layers</td><td>18 / 20 / 22</td><td class="tie">within noise → a hedge (half A, 5 of 8)</td><td>not measured</td></tr>
<tr><td>probe size</td><td>1 hidden layer, 256</td><td>linear worse (CV); ≥ 128 flat</td><td>not measured</td></tr>
<tr><td>loss</td><td>class-weighted BCE</td><td class="tie">confounded — no claim</td><td>—</td></tr>
<tr><td>readout</td><td>rank average + vote tie-break</td><td class="win">tie-break +0.0006 (sig., tiny)</td><td>+0.0006 (sig.)</td></tr>
<tr><td>label source</td><td>training judge</td><td class="tie">cross-judge gain +0.057 to +0.066</td><td>Sept. L20 probe: 32B labels +0.0026 · MedGemma labels +0.0101 · consensus +0.0033 · soft +0.0057</td></tr>
<tr><td>training data</td><td>8 open + comparison + 4 domains</td><td>pooling +0.055 (no CI)</td><td>not measured</td></tr>
<tr><td>N</td><td>8 (inherited)</td><td>16: +0.0167 at 2× generation</td><td>16: +0.0140 vs +0.0121</td></tr>
<tr><td>temperature</td><td>0.7 for all</td><td>0.7 highest point estimate</td><td>prefers 0.2–0.4 (0.7 third)</td></tr>
<tr><td>image budget</td><td>cap320</td><td class="tie">not worse than the 1-MP cap</td><td class="tie">+0.0122 vs +0.0111</td></tr>
</table>
<div class="note" style="margin-top:4px">¹ July pools, EM/substring, n = 600. Eras: July = LoRA verifier · Aug. = first probe and pooled v1 · Sept. = pooled probe v2 (half A) · Oct. = final cross-fitted probes.</div>""", "IV · Why", minutes=1.5,
          look="Read the table row by row only for the rows I name: selector, verifier type, label source, temperature. The two right-hand columns are the two metrics.",
          src="final_fold1_headline_2026-10-16.json policies.T07.macro; extra_numbers.json probe_minus_vote_em; free_head_2026-08-16.json; verifier_32b_gpu.json; wed_vs_probe_2026-10-16.json; readout_variants_2026-10-15.json; judge_2x2_2026-09-20.json, judge_consensus_2026-09-20.json (layer 20, 3 seeds); temp_crossfit_2026-10-15.json; select_policy_cap320_vs_fullres_T07.json; final_eval_FINAL_cap320_MIX_T07_16.json",
          say="""# Design decisions at a glance
- This table is the summary of part four; the next slides give the evidence for each row. Two things to notice first.
- One: almost every design choice was made on one half of the images, half A. Half B played no part in those choices.
- On half B alone the method gains plus 0.085 under the judge and plus 0.017 in exact match, both significant — so the choices did not inflate the headline.
# training-judge verdict · lenient EM (columns)
- Two: most verdicts are measured with the training judge, which also produced the labels, so I show exact match next to it wherever it exists.
- Training a selector holds under both metrics: it beats majority vote, the answer prior, random and longest pick, and self-verification.
- It ties a much larger untrained verifier; against the published selector WED it wins on two benchmarks and ties on the third.
- Likelihood ranking and USC were not run.
- Some choices are ties where we picked the simpler option; the temperature is the one where the two metrics disagree.
# label source (row)
- One small surprise in this row: under exact match, a probe trained on the other judge's labels does a little better — every judge has its own taste.""",
          qa=[("Which design verdicts survive exact match?", "Measured in both: probe vs majority vote (EM +0.0136 [+0.0053, +0.0216]); readout tie-break (EM +0.0006, significant); cap320 vs the 1-MP cap (EM tie, +0.0122 vs +0.0111); probe vs LoRA verifier (EM loss −0.0145, earlier probe); temperature (EM prefers 0.2–0.4). Judge-only: token reading, layers, width, pooling. The final method is EM-significant pooled (+0.0121) and on half B alone (+0.0165)."),
              ("Which judge's labels should train the probe?", "Tested at the September stage (layer 20, 3 seeds): a probe trained on MedVLThinker labels and graded by MedGemma gains +0.0658; trained on MedGemma labels and graded by MedVLThinker +0.0565; consensus labels +0.0703 / +0.0733. Under lenient EM: MedVLThinker labels +0.0026, MedGemma labels +0.0101, consensus +0.0033, soft +0.0057 (no intervals). The choice is at the noise floor; 7–30% of a judge-measured gain is judge-specific (judge_2x2_2026-09-20.json, judge_consensus_2026-09-20.json)."),
              ("Why 8 seeds?", "Never swept. 8 seeds ≥ the best single seed (CHEAP_VERIFIER_ON_7B_2026-08-16.md §6). The intervals hold the probes fixed, so refit variance (≈ 0.003) is not in them.")])

    # ---- why a trained selector
    sel_rows = [
        ("probe (shipped, T07)", [(D.macro("judge")[0], D.macro("judge")[1], D.macro("judge")[2])]),
        ("majority vote", [(D.sc_macro("judge")[0], D.sc_macro("judge")[1], D.sc_macro("judge")[2])]),
        (f"<tspan font-style='italic' font-size='12.5' fill='{C['faint']}'>— Sept., half A ¹ —</tspan>", [None]),
        ("answer-string prior ¹", [(0.0185, -0.0295, 0.0664)]),
        ("random pick ¹", [(-0.0271, -0.0463, -0.0079)]),
        ("pick longest ¹", [(-0.0294, -0.0557, -0.0032)]),
    ]
    d.add("whysel", "Why train a selector at all",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 620px">{forest(sel_rows, -0.06, 0.10, 0.02, w=620, rowh=38, title="selector − greedy, training judge, macro over 8 benchmarks", xlabel="selector − greedy (accuracy)", value_fmt=lambda v: fmt(v,4))}
 <div class="note">top two rows: shipped method, both halves, image-clustered bootstrap. ¹ September probe v2, image half A, t-interval over the 8 benchmarks; the probe there +0.0737. Filled dot = CI excludes 0.</div></div>
 <div class="col">
  <div class="term g"><b class="k">same pools, paired</b><span>probe − vote: judge +0.0893 [+0.0811, +0.0972] · lenient EM +0.0136 [+0.0053, +0.0216]</span></div>
  <div class="term w"><b class="k">answer prior, per benchmark ¹</b><span>recovers 60% (OmniMedVQA), 67% (RadImageNet), 45% (GEMeX), 23% (Kvasir-x1) of the probe's judge gain; lenient EM on RadImageNet: prior ties the probe</span></div>
  <div class="term r"><b class="k">ask the model itself</b><span>self-verification: below random pick and greedy (0.319 vs 0.343; July)</span></div>
  <div class="term r"><b class="k">ask a bigger model</b><span>32B zero-shot beats 7B zero-shot (+0.067) but only ties a trained 7B (+0.005 [−0.023, +0.032]; July, EM)</span></div>
 </div>
</div>""", "IV · Why", src="final_eval_FINAL_cap320_T07.json; extra_numbers.json probe_minus_vote_*; audit_stats_baselines_2026-09-20.json summary_by_currency.{judge,lenientEM}.* (benchmark_t_interval_df7) and per_benchmark.* (shares: numbers_review_part2.json answer_prior_share); PROJECT_RETROSPECTIVE N32; verifier_32b_gpu.json pooled.deltas_vs_32B",
          minutes=2.0, look="The top two dots first (probe, majority vote); then the rows below the dashed September line; then the orange answer-prior box.",
          say=f"""# selector − greedy, training judge, macro over 8 benchmarks (chart)
- Why train anything? Every untrained way of choosing that we ran on these pools loses to greedy or to the probe.
- Majority vote loses to greedy. Random choice and the longest answer both lose.
# ask the model itself
- Asking Lingshu itself whether its answer is correct did worse than picking at random when we tried it.
# answer prior, per benchmark ¹
- One baseline needs care: choosing the answer string that is most often right in training.
- Over all benchmarks it recovers only about a quarter of the gain, and its interval includes zero.
- Per benchmark it is uneven: on OmniMedVQA and RadImageNet — a modality or organ from a small vocabulary — the prior alone gets about two-thirds of the way.
- So on those two, a large part of the gain is knowing which answers are usually right; the probe adds the rest.
- Under exact match on RadImageNet the prior ties the probe. I say this plainly because those are the two benchmarks the exact-match win rests on.
# ask a bigger model
- A model four and a half times bigger is a better zero-shot verifier than the small one — size does help — but untrained it only tied a small trained verifier.
- That does not conflict with Verification Mirage: they measure whether a model accepts its own wrong answers when it re-checks them.
- We measure best-of-eight selection accuracy, with the 32B judging the 7B's samples.
- Both agree that training, not prompting, is what makes a selector work.
# same pools, paired
- The probe gains plus {D.macro("judge")[0]:.3f}; on exactly the same pools it beats majority vote by 0.089 under the judge and by 0.014 in exact match, both significant.""",
          qa=[("Doesn't the answer-string prior show the probe just memorises common answers?", "Partly, on two benchmarks. Overall the prior recovers ~25% of the September probe's judge gain (+0.0185 vs +0.0737, interval includes 0). Per benchmark (judge, image half A): OmniMedVQA 60%, RadImageNet 67%, GEMeX 45%, Kvasir-x1 23% — and the probe is still significantly ahead of the prior on all four (e.g. RadImageNet +0.0398 [+0.0219, +0.0578]). Under lenient EM on RadImageNet the prior ties the probe (+0.0050 [−0.0080, +0.0179]). (audit_stats_baselines_2026-09-20.json per_benchmark; shares in numbers_review_part2.json answer_prior_share; the prior is leave-one-out and optimistic.) "
               "On gold answers never seen in training the probe still gains +0.092 judge (but loses in exact match) — see the 'does it memorise' slide."),
              ("Mirage says a bigger verifier does not help; you say the 32B helps. Which is it?", "Different questions. Mirage measures self-verification: whether a model accepts its own wrong answers when re-asked (same family, p = 0.782 for a larger verifier). We measure best-of-8 selection accuracy, with a separate 32B judging the 7B's samples zero-shot: +0.067 [+0.038, +0.095] over a 7B zero-shot verifier (July pools, EM/substring, n = 600; verifier_32b_gpu.json). Both agree that a trained small selector matches or beats an untrained large one.")])

    # ---- why hidden states vs LoRA verifier
    d.add("whyprobe", "Why a probe on hidden states instead of a fine-tuned verifier model",
          """
<div class="row">
 <div class="col">
  <table class="d sm"><tr><th>same 2,345 questions</th><th class="n">judge</th><th class="n">lenient EM</th></tr>
  <tr><td>August probe − July LoRA verifier</td><td class="n win">+0.0162 [+0.0026, +0.0299]</td><td class="n loss">−0.0145 [−0.0281, −0.0004]</td></tr></table>
  <div class="note" style="margin-top:6px">August single-layer probe (genframe_head_ens8), not the shipped one: judge win, EM loss → a wash.</div>
  <table class="d sm" style="margin-top:12px"><tr><th>cost vs greedy (HF path, measured)</th><th class="n">latency</th><th class="n">energy</th></tr>
  <tr><td>generate 8 + LoRA-verify 8</td><td class="n">3.68×</td><td class="n">5.77×</td></tr>
  <tr><td>LoRA scoring alone (verify 8)</td><td class="n">1.69×</td><td class="n">2.82×</td></tr></table>
  <table class="d sm" style="margin-top:12px"><tr><th>FLOP-eq per question — HF accounting (8 separate generation passes)</th><th class="n">vs greedy</th></tr>
  <tr><td>generate 8 + probe (separate 1-MP re-read)</td><td class="n">14.90×</td></tr>
  <tr><td>generate 8 + LoRA verifier</td><td class="n">15.18×</td></tr>
  <tr><td>probe, states captured during generation (by design)</td><td class="n">8 + 0</td></tr></table>
  <div class="note" style="margin-top:4px">the pools were generated on vLLM with a shared prefill: there sampling is 1.13× (Part III cost slide). The LoRA verifier cannot be served on vLLM, so this comparison stays on HF accounting.</div>
 </div>
 <div class="col">
  <div class="term"><b class="k">why the LoRA verifier is expensive</b><span>it re-encodes the image and runs the full 7B model once per candidate to output “Yes/No”.</span></div>
  <div class="term r"><b class="k">and it cannot be served fast</b><span>vLLM silently drops the LoRA's visual modules (same adapter: 0.775 under HF, 0.703 under vLLM).</span></div>
  <div class="term g"><b class="k">and the probe scales</b><span>training a probe on all 8 benchmarks is cheap; that pooling is what produced the large gain (later slide). A LoRA verifier trained on the pooled data was never built — a gap.</span></div>
  <div class="term v"><b class="k">literature</b><span>HSRM: hidden states beat re-reading the text (GSM8K 86.1 vs 82.2, Table 1).</span></div>
 </div>
</div>""", "IV · Why", src="free_head_2026-08-16.json DELIVERABLE_4_head_only.head_vs_incumbent, DELIVERABLE_3_cost.arms_fixed_bestof8.{bo8_head_TODAY,bo8_incumbent_LoRA}.flopeq_measured_geometry; bestofn_latency_energy_2026-08-03.json measured.{bo8_total,verify8,gen1}; CLAUDE.md §0 (vLLM LoRA landmine); HSRM Table 1 p.7",
          minutes=2.0, look="The first table (accuracy), then the two cost tables — the bottom one is in the HF accounting.",
          say="""# same 2,345 questions (table)
- Before the probe, this project used a fine-tuned verifier: Lingshu-7B with a LoRA adapter, asked "is this answer correct".
- On the same questions, an earlier single-layer probe was better under the judge and worse under exact match — on accuracy it is a wash.
# why the LoRA verifier is expensive
- The decision was about cost by design. The LoRA verifier re-reads the image and runs the whole 7-billion model once per candidate.
- Generating eight answers and verifying them costs 3.7 times the latency and 5.8 times the energy of greedy; the scoring alone is 1.7 and 2.8 times.
# and it cannot be served fast
- And the fast serving engine cannot even run its visual adapter correctly.
# FLOP-eq per question — HF accounting (8 separate generation passes)
- As we actually ran it, though, the probe needed its own re-reading pass.
- In the HF accounting, where all eight samples are separate passes, it then costs almost the same arithmetic as the LoRA verifier — about two percent less.
- The saving is real only once the states are captured during generation, which we showed equivalent only for an earlier probe.
# and the probe scales
- The probe is also cheap enough to train on all benchmarks, which turned out to be the biggest single improvement.
- A LoRA verifier trained on the same pooled data was never built, so I cannot claim the probe is more accurate than that.""")

    # ---- why answer span
    d.add("whyspan", "Why read the answer tokens, with the answering prompt",
          """
<div class="row">
 <div class="col">
  <table class="d sm"><tr><th>what the probe reads (layer 21, July features)</th><th class="n">sel. eff.</th></tr>
  <tr class="hl"><td>mean over answer tokens (span)</td><td class="n">0.7956</td></tr>
  <tr><td>last answer token</td><td class="n">0.7807</td></tr>
  <tr><td>span, but under a “is this correct?” grading prompt</td><td class="n">0.7827</td></tr></table>
  <div class="note">in-eval CV (optimistic), no interval; a clean training-pool CV agrees: span 0.680 vs last 0.655.</div>
  <table class="d sm" style="margin-top:14px"><tr><th>adding context to the span (September, judge, 3 seeds)</th><th class="n">macro</th></tr>
  <tr class="hl"><td>span only</td><td class="n">+0.0806</td></tr><tr><td>+ question-token mean</td><td class="n">+0.0825</td></tr>
  <tr><td>+ image-token mean</td><td class="n">+0.0808</td></tr><tr><td>image tokens alone</td><td class="n">−0.0361</td></tr></table>
 </div>
 <div class="col">
  <div class="term t"><b class="k">reading</b><span>the answer's own tokens carry the signal; answering vs grading prompt flips between layers 21 and 28 — no claim</span></div>
  <div class="term"><b class="k">extras are below the noise floor</b><span>question tokens +0.0018, image tokens +0.0002 (refit floor ≈ 0.003)</span></div>
  <div class="term w"><b class="k">against us</b><span>SWIFT: learned token weights beat a mean (+2.7); CASE: mean-pooling hurts on MCQ. Learned pooling untested.</span></div>
 </div>
</div>""", "IV · Why", src="verifarch_hidden_promptform_indomain_2026-08-04.json (in-eval CV, upper bound); _head_sweep_journal*.jsonl A_objbt_h256, B_L21_last (clean CV, Bradley–Terry; sd 0.016 / 0.009); head_visual_features_2026-09-01.json macro.*; SWIFT Table 8; CASE §3.2",
          minutes=1.5, look="The highlighted top row of each table; then the 'against us' box.",
          say="""# what the probe reads (layer 21, July features)
- We compared three readings: the mean over the answer tokens, the last token, and the same span when the model is asked to grade the answer.
- The mean over the answer, read while the model is answering, was best at layer 21 — but these are optimistic in-sample numbers without intervals.
# reading
- The span beats the last token at every layer from 14 up; at the last layer the grading prompt is slightly ahead of the answering prompt.
- So I read this only as: the answer's own tokens carry the signal.
- A cleaner cross-validation also puts the span ahead of the last token, 0.680 against 0.655.
# extras are below the noise floor
- Adding the question tokens or the image tokens on top changed the result by less than two thousandths — under the noise from retraining.
- So we keep one vector per layer.
# against us
- Two papers point the other way: SWIFT found learned token weights better than a plain mean; CASE found averaging harmful in a multiple-choice setting.
- Learned pooling is something we have not tried.""")

    # ---- why MLP 256
    w_rows = [("linear", 0.66426), ("128", 0.69029), ("256 ✓", 0.69396), ("512", 0.69075), ("1024", 0.69742)]
    # a dot plot, not bars: the values sit in a narrow band, and bars from a non-zero baseline would exaggerate the gaps
    w_chart = forest([(lab, [(v, None, None)]) for lab, v in w_rows], 0.62, 0.72, 0.02, w=470, rowh=40, label_w=110, zero=False,
                     title="5-fold CV selection efficiency, by probe width", xlabel="CV selection efficiency (fold sd ≈ 0.013)",
                     value_fmt=lambda v: f"{v:.3f}")
    d.add("whymlp", "Why an MLP of width 256 — the probe-size plateau",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 480px">{w_chart}<div class="note">~150-run sweep (128 distinct configurations), 5-fold CV inside the training pool, BCE, depth 1. Linear is clearly worse; ≥ 128 is a plateau.</div></div>
 <div class="col">
  <table class="d sm"><tr><th>axis</th><th>values</th><th>effect</th></tr>
  <tr><td>dropout</td><td>0 / 0.1 / 0.3 / 0.5 / 0.7</td><td>0.696 / 0.685 / 0.689 / 0.693 / 0.678</td></tr>
  <tr><td>depth</td><td>1 vs 2 hidden layers</td><td>0.697 vs 0.698</td></tr>
  <tr><td>epochs</td><td>5 / 10 / 20 / 30 / 60</td><td>0.668 / 0.687 / 0.693 / 0.696 / 0.699</td></tr>
  <tr><td>weight decay</td><td>1e-3 … 1.0</td><td>0.692 – 0.700 (flat)</td></tr></table>
  <div class="term" style="margin-top:8px"><b class="k">the sweep's best cell, re-tested end-to-end</b><span>width 1024, 60 epochs, cosine: +0.0021 over a same-session control — inside the refit floor → width 256 kept.</span></div>
  <div class="term w"><b class="k">against us</b><span>CASE, MedProb, Q-Probe, Hewitt &amp; Liang prefer linear or warn MLPs memorise; we have far more labels</span></div>
  <div class="term r"><b class="k">caveats</b><span>CV→eval selection failed once (July); LayerNorm and the BCE learning rate never tested end-to-end</span></div>
 </div>
</div>""", "IV · Why", src="_head_sweep_journal*.jsonl tags A_objbce_h{0,128,256,512,1024}, REG_drop*, REG_ep*, REG_wd*, R2_d2_dr0.0, R2_normz_ln1 (150 runs, 128 distinct _key configurations; fold sd A_objbce_h256 0.0133); head_final_stack_08{ctrl,wide}_2026-10-15.json (judge, 0.0761 vs 0.0740); VERIFIER_ARCHITECTURES_2026-08-04.md §6 item 3",
          minutes=1.5, look="The linear dot against the four MLP dots; then the axis table on the right.",
          say="""# 5-fold CV selection efficiency, by probe width (chart)
- We ran a sweep of about 150 runs, 128 distinct configurations.
- The one clear finding: a linear probe is worse by about three points of selection efficiency.
- With one hidden layer of at least 128 units, nothing else we varied — width, depth, dropout, training length, weight decay — moves the result beyond the noise.
- So we took a simple point on the plateau.
# the sweep's best cell, re-tested end-to-end
- Re-tested end to end, the sweep's best setting gained two thousandths, inside the noise, so we kept width 256.
# against us
- Several papers prefer linear probes: CASE, MedProb, and the probing-methodology literature warn that wide MLP probes can memorise.
- Our situation differs — over a hundred thousand labelled rows instead of a few hundred — and our own linear probe was clearly worse.
- But a control-task check would settle this properly.
# caveats
- Once, in July, the model this cross-validation picked was not the best on evaluation.
- LayerNorm in the probe and the learning rate were never tested end to end.""",
          qa=[("Where does sd ≈ 0.013 come from?", "The fold-to-fold standard deviation of the 5-fold CV selection efficiency in the sweep journal (e.g. A_objbce_h256 sd 0.0133)."),
              ("The sweep was on an older setup — does it transfer?", "It ran on the four-domain, layer-21 setup, so it informed rather than selected the pooled recipe — evidence of a plateau, not of an optimum. CV picked BCE/h256 in July and it scored 0.7507 on eval, below listwise at 0.7895 (VERIFIER_ARCHITECTURES_2026-08-04.md §6.3, unresolved). The one end-to-end re-test of the best cell gained +0.0021 (judge only, inside the ~0.003 refit floor).")])

    # ---- why BCE
    d.add("whybce", "Why BCE — and why we do not claim it beats ranking losses",
          """
<div class="row">
 <div class="col">
  <table class="d"><tr><th>objective (same sweep, width 256)</th><th class="n">CV sel. eff.</th></tr>
  <tr class="hl"><td>BCE (pointwise)</td><td class="n">0.694</td></tr><tr><td>Bradley–Terry (pairwise)</td><td class="n">0.680</td></tr>
  <tr><td>listwise</td><td class="n">0.674</td></tr><tr><td>hybrid</td><td class="n">0.688</td></tr></table>
  <div class="term r" style="margin-top:12px"><b class="k">confounded</b><span>ranking losses see only 38.9% of rows and 3.26× fewer steps; step-matched: BT 0.676 vs BCE 0.696</span></div>
 </div>
 <div class="col">
  <div class="term t"><b class="k">reasons we keep BCE</b><span>uses every row · matches the judge's yes/no label · the rank readout cancels per-question offsets</span></div>
  <div class="term w"><b class="k">the literature is split</b><span>SWIFT: BCE best of 5 · HSRM: BT 85.93 vs BCE 83.91 · Q-Probe: within-question 0.46 vs CE 0.40</span></div>
  <div class="term r"><b class="k">open</b><span>a ranking loss on the final recipe has not been tested</span></div>
 </div>
</div>""", "IV · Why", src="_head_sweep_journal*.jsonl A_obj{bce,bt,listwise}_h256, F_hybrid1.0, BUD_bt_ep98, BUD_bce_ep30; OPENTEXT_CORRECTIONS_2026-08-19.md §3b; SWIFT Table 7; HSRM Table 9 (GSM8K); Q-Probe Table 1, Fig. 4 (MBPP)",
          minutes=1.5, look="The table (BCE highlighted), then the red 'confounded' box — that is why there is no claim.",
          say="""# confounded
- In our sweep, binary cross-entropy scored higher than pairwise and listwise ranking losses — but that comparison is not fair.
- A ranking loss can only learn from questions with both a right and a wrong answer — 39 percent of our rows — and it took three times fewer training steps.
- So we do not claim BCE is better.
# reasons we keep BCE
- It uses every row.
- It matches the yes-or-no label the judge gives.
- Our rank-based readout cancels the one known weakness of pointwise losses — learning which questions are hard.
# the literature is split
- SWIFT found BCE best of five losses; HSRM found a pairwise loss better on math; Q-Probe found a within-question loss better on code.
# open
- A ranking loss on the final recipe is one of the experiments still owed.""",
          qa=[("Is there any controlled evidence on the loss?", "Two data points, one each way, both from the July LoRA-feature era. For BCE: training BCE on exactly the restricted rows a ranking loss can use cost −0.0020 [−0.0116, +0.0075], while swapping the loss on those same rows cost −0.0817 [−0.1042, −0.0586]. Against: listwise scored higher on eval than the CV-chosen BCE head (0.7895 vs 0.7507). VERIFIER_ARCHITECTURES_2026-08-04.md §4.3, §6.3. No claim for the final recipe.")])

    # ---- why rank + vote
    rv = D.readout["macro"]

    def g(k):
        v = rv[k]
        c = v.get("ci_image_clustered") or v.get("ci")
        return (v["macro"], c[0], c[1])
    ro_rows = [(lab, [g(f"{k}_minus_rank_j"), g(f"{k}_minus_rank_x"), g(f"{k}_minus_rank_em")])
               for lab, k in [("logit average", "logit_mean"), ("z-score average", "z_mean"), ("probability avg.", "prob_mean"),
                              ("Platt average", "platt_mean"), ("rank + vote tie ✓", "rank_vote"), ("probe-weighted vote", "vote_prob")]]
    d.add("whyreadout", "Why rank averaging with a vote tie-break",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 640px">{forest(ro_rows, -0.02, 0.03, 0.01, w=640, rowh=40,
              series=[("training judge", C["indigo"]), ("MedGemma", C["teal"]), ("lenient EM", C["tap"])],
              title="variant − rank average (macro, identical questions, image half A)", xlabel="variant − rank average (accuracy)", value_fmt=lambda v: fmt(v, 4))}</div>
 <div class="col">
  <div class="term"><b class="k">rule fixed in advance</b><span>adopt only if positive under <b>both</b> judges, significant under one, and no significant EM loss</span></div>
  <div class="term t"><b class="k">passed</b><span>vote-count tie-break: +0.0006 judge, +0.0006 EM, both significant (lower bounds +0.00005 / +0.00003)</span></div>
  <div class="term w"><b class="k">metric trades</b><span>probe-weighted vote: EM +0.0135 (sig.), both judges slightly lower (ties). z-score: EM +0.0032 (sig.), judges tie. Both fail the rule.</span></div>
 </div>
</div>""", "IV · Why", src="readout_variants_2026-10-15.json macro.*_minus_rank_{j,x,em} (rank_vote CI: judge [+0.00005, +0.00152], EM [+0.00003, +0.00154]), adoption_rule (September pooled probe, image half A, n = 18,452; not re-checked on the final probes)",
          minutes=1.5, look="The row marked with a tick — rank + vote tie — then the bottom row, probe-weighted vote: its violet EM dot is right, its two judge dots are left.",
          say="""# rule fixed in advance
- We tested the alternatives on identical held-out questions, under three metrics.
- The rule was written down before seeing the results: adopt a change only if it helps under both judges, significantly under at least one, and does not significantly hurt exact match.
# variant − rank average (macro, identical questions, image half A)
- Averaging raw scores, standardised scores, probabilities or calibrated probabilities all tie under both judges.
- z-score averaging gains a third of a point in exact match, but it fails the rule.
# passed
- Breaking exact ties by vote count passed the rule — six hundredths of a point, but significant and free — so it is part of the method.
- This test used the September probes on image half A.
# metric trades
- One result I want to show rather than hide: weighting the votes by the probe's scores — the default aggregator in Snell et al.
- It improves exact match by 1.35 points, while both judges come out slightly lower — ties, minus 0.004 and minus 0.006.
- It trades one metric for another, so under our rule it was not adopted — but if exact match were the target, it would be the better readout.""")

    # ---- why pool
    d.add("whypool", "Why train on all eight benchmarks: the largest single lever",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 520px">{bars(["3 train splits\n+ Kvasir-VQA", "leave the\nbenchmark out", "all 8 + 4\n(pooled)"], [("judge − greedy", C["teal"], [0.0279, 0.0223, 0.0707])], 0, 0.08, 0.02, w=500, h=290, value_fmt=lambda v: fmt(v, 4))}
 <div class="note">same probe recipe, training judge, macro, no interval (August, 5 seeds, head_lobo_pooled_2026-08-25.json); 4-domain set = PathVQA/SLAKE/VQA-RAD train splits + Kvasir-VQA, so the PathVQA, SLAKE, VQA-RAD and Kvasir-x1 leave-out folds keep their own source's training split (optimistic). On the 4 benchmarks without one: 4-domain +0.0147, left out +0.0033, pooled +0.0985 (mean of 4 cells, same file; numbers_review_part3.json).</div></div>
 <div class="col">
  <div class="term g"><b class="k">pooling</b><span>4 domains → pooled: +0.0279 → +0.0707 (this chart); same jump on Qwen2.5-VL-7B: +0.0227 → +0.0820</span></div>
  <div class="term r"><b class="k">breadth alone does not transfer</b><span>benchmark left out: −0.0056 vs the 4-domain probe. <b>A new benchmark needs labelled questions; inference needs no dataset label.</b></span></div>
  <div class="term"><b class="k">onboarding cost</b><span>zero labels: +0.0196 (3-seed re-run of the leave-out bar); ~100 labels recover 46% / 66% of the gain on RadImageNet / OmniMedVQA</span></div>
 </div>
</div>""", "IV · Why", src="head_lobo_pooled_2026-08-25.json macro.* (chart, −0.0056, +0.0485); head_final_stack_PVFIXED_2026-09-13.json (+0.0182 → +0.0736, September, script only); head_price_from_lobo_2026-08-30.json summary.zero_shot_macro; TRANSFER_WALL_2026-08-21.md §12 (Qwen, head_final_stack_qwen pooled ensemble)",
          minutes=1.5, look="The three bars left to right: 4 domains, left out, pooled — the left-out bar is no higher than the first.",
          say="""# pooling
- The single biggest improvement in the project was what the probe is trained on.
- In this run, a probe trained only on the four-domain set — the PathVQA, SLAKE and VQA-RAD training splits plus Kvasir-VQA — gains about 0.03.
- Trained on the training halves of all eight benchmarks: about 0.07 — two and a half times as much, and four times in a later September re-run.
- The same jump happens on Qwen2.5-VL.
# breadth alone does not transfer
- The honest limitation: leave a benchmark out of training, and the probe does no better on it than the four-domain probe.
- On the four benchmarks with no training split in that set the gap is stark: left out plus 0.003, pooled plus 0.099.
- So the claim is about new questions and images from benchmarks we have labels for: a new benchmark needs some labelled questions.
- At inference the probe needs no dataset label.
- These numbers are measured with the training judge only, without intervals.
# onboarding cost
- The cost of a new benchmark is modest: with zero labels it gains plus 0.0196 overall in a three-seed re-run.
- Five of eight benchmarks already gain, and a sixth after 50 labels.
- On the two hardest cases, RadImageNet and OmniMedVQA, about a hundred labelled questions recover about half and two-thirds of the full gain.""")

    # ---- why N=8
    d.add("whyN", "Why N = 8 — and what more samples buy",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 520px">{lines([8, 16, 32], [("RadImageNet", C["teal"], [0.1215, 0.1544, 0.1932]), ("VQA-Med", C["indigo"], [-0.0033, 0.0011, 0.0044])], -0.02, 0.22, 0.04, w=500, h=360, xlab="samples N", ylab="probe − greedy (training judge)", value_fmt=lambda v: fmt(v,4))}<div class="note">September pooled probe, September pools; N = 32 RadImageNet pool sampled at T = 1.0</div></div>
 <div class="col">
  <div class="term t"><b class="k">8 → 16, all benchmarks</b><span>+0.0167 macro: 4 significant wins, 4 ties, 0 losses — at twice the generation</span></div>
  <div class="term"><b class="k">more samples help only where selection works</b><span>VQA-Med's oracle (September pools) rises 0.199 → 0.348 from 8 to 32; accuracy barely moves</span></div>
  <div class="term w"><b class="k">over-optimisation risk</b><span>a selector's mistakes grow with N (Cobbe: decline past 400); untested beyond 32</span></div>
  <div class="term"><b class="k">so</b><span>N = 8: inherited default, cost measured · N = 4: untested with the probe · N = 16: an accuracy/cost option</span></div>
 </div>
</div>""", "IV · Why", src="coverage_sc16_ci_ALL_2026-09-16.json (judge); budget_conversion_sc32_2026-09-17.json cells.{radimagenet_open,vqamed_open} (September pooled probe; RadImageNet N = 32 sampled at T = 1.0); CHEAP_VERIFIER_ON_7B_2026-08-16.md §4 (N = 4: judge +0.00159 [−0.0051, +0.0083], EM +0.00825 [+0.0015, +0.0151])",
          minutes=1.0, look="The teal RadImageNet line rising; the indigo VQA-Med line flat.",
          say="""# 8 → 16, all benchmarks
- More samples help: doubling to sixteen gains another 0.017 overall, with significant wins on RadImageNet, Kvasir-x1, OmniMedVQA and GEMeX.
# more samples help only where selection works
- But only on the benchmarks where the probe already selects well.
- VQA-Med shows the other case: with more samples a correct answer appears more often, but the probe cannot find it, so accuracy does not move.
# over-optimisation risk
- More samples also give a selector more chances to be fooled.
# so
- Eight is our default, inherited from earlier work, where the cost was measured; sixteen is an option if accuracy matters more than time.
- We never tried fewer than eight with the probe.
- With the earlier LoRA verifier, four samples at a lower temperature did as well as eight, so that is worth testing.""")

    # ---- why T=0.7
    def _cell(v):
        # diverging fill: teal for gains, orange for losses, so a negative never looks like a small positive
        if v >= 0:
            bg, fg = f"rgba(0,137,123,{min(0.9, 0.08 + v / 0.19):.2f})", ("#fff" if v > 0.09 else "#1d2030")
        else:
            bg, fg = f"rgba(239,108,0,{min(0.85, 0.15 + -v / 0.07):.2f})", "#1d2030"
        return f"<td class='n' style='background:{bg};color:{fg}'>{pm(v,3)}</td>"
    heat = "".join(
        f"<tr><td>{D.SHORT[c]}</td>" + "".join(_cell(v) for v in D.T_PER[c]) + f"<td class='n'><b>{D.T_BEST[c]}</b></td></tr>"
        for c in D.OPEN8)
    d.add("whyT", "Why one temperature, T = 0.7",
          f"""
<div class="row">
 <div class="col" style="flex:0 0 470px"><div class="note"><span class="tag">September probe v2 · image half A</span></div>{lines([0.2, 0.4, 0.7, 1.0], [("training judge", C["indigo"], D.T_J), ("lenient EM", C["tap"], D.T_EM)], -0.02, 0.09, 0.02, w=460, h=260, xlab="sampling temperature (same T for all benchmarks)", ylab="probe − greedy, macro", value_fmt=lambda v: fmt(v,4))}
  <div class="term w" style="margin-top:4px"><b class="k">metric-dependent</b><span>judge: 0.7 highest point estimate (+0.0045 over 1.0, no CI) · lenient EM: 0.2 &gt; 0.4 &gt; 0.7 &gt; 1.0</span></div></div>
 <div class="col">
  <div class="note"><span class="tag">August probe v1 · training judge only</span></div>
  <table class="d sm"><tr><th>benchmark</th><th class="n">0.2</th><th class="n">0.4</th><th class="n">0.7</th><th class="n">1.0</th><th class="n">best</th></tr>{heat}</table>
  <div class="term r" style="margin-top:8px"><b class="k">per-benchmark T: adopted 10-16, reversed 10-18</b><span>+0.009 under both judges on the unseen half (EM tie) — but needs the dataset label at inference</span></div>
  <div class="term"><b class="k">mixed temperatures</b><span>2 each at 0.2/0.4/0.7/1.0: judge +0.0726 vs +0.0798 — lower in every metric (no paired CI)</span></div>
 </div>
</div>""", "IV · Why",
          src="chart: temp_crossfit_2026-10-15.json macro.fixed_*_minus_greedy_{judge,em} (September pooled probe v2, image half A); table: head_temp_ensemble_2026-08-30.json cells.*.{T}.minus_greedy (August pooled probe v1); tmap_vs_t07_2026-10-16.json macro.fold1_out_of_sample; final_eval_FINAL_cap320_MIX_{T07,MIX8}.json (main checkout artifacts/)",
          minutes=2.0, look="The left chart: the indigo judge line peaks at 0.7, the violet EM line falls from 0.2; then the table's 'best' column; then the red box.",
          say="""# training judge · lenient EM (left chart)
- Why one temperature of 0.7? The left chart is the gain over greedy when every benchmark uses the same temperature.
- Under the judge, 0.7 has the highest point estimate, though only by half a point over 1.0.
# metric-dependent
- Under exact match, 0.7 is third of four — colder sampling is better there. So the best temperature depends on the metric.
- 0.7 was already the project's default before we measured this, so it was not chosen by looking at the test results.
# benchmark (table)
- Each benchmark has its own preference: where the probe selects well it wants hotter sampling; the two where it struggles want colder sampling.
# per-benchmark T: adopted 10-16, reversed 10-18
- Choosing a temperature per benchmark does help, by about 0.009 on images it was not chosen on; our final-method document of October 16 adopted it.
- We reversed that on October 18: it requires knowing which dataset each incoming question comes from, which a deployed system does not know.
- Training the probe needs labelled data per domain, but the shipped probe then scores any question without a dataset label — and we keep it that way.
# mixed temperatures
- We also tried a version that needs no dataset label — two samples at each of four temperatures.
- Its point estimate was lower than eight samples at 0.7 in every metric.""",
          qa=[("Could the model choose the temperature per question, e.g. by imaging modality?", "Untested. Modality alone seems unlikely to drive it: GEMeX (chest X-ray) prefers T = 1.0 while VQA-RAD (which includes chest X-rays) prefers 0.2; "
               "the preference tracks how well the probe selects on that benchmark. A per-modality test (chosen on one image half, tested on the other) is planned, as is a per-question adaptive scheme."),
              ("What about mixing temperatures with 16 samples?", "MIX16 (4 samples each at 0.2/0.4/0.7/1.0) vs 16 samples at 0.7: judge +0.0927 vs +0.0984, lenient EM +0.0108 vs +0.0140 — lower again (point estimates; final_eval_FINAL_cap320_MIX_{MIX16,T07_16}.json in the main checkout's artifacts/; MedGemma labels incomplete for 16-sample pools)."),
              ("Why is the 0.7 here +0.0736 and the headline +0.0798?", "Different probes and protocol: this curve is the September pooled probe (v2) on image half A; the table is the August v1 probe (+0.0773 at 0.7); the headline is the final cross-fitted method on both halves."),
              ("Your final-method document adopted the per-benchmark map (+0.0897), and it passed your own out-of-sample test. Why ship +0.0798?", "The map is real: +0.0091 judge [+0.0007, +0.0182], +0.0090 MedGemma [+0.0010, +0.0175] on image half B, which played no part in choosing it; EM a tie (tmap_vs_t07_2026-10-16.json). The difference is the inference-time input: the map needs a dataset label on every incoming question; the pooled probe does not. We ship the version with fewer inference-time assumptions and report the map as a domain-calibrated variant (backup).")])

    # ---- why cap320
    d.add("whyres", "Why cap320 instead of the 1-megapixel cap",
          """
<div class="row">
 <div class="col">
  <table class="d"><tr><th>whole method, T = 0.7</th><th class="n">judge − greedy</th></tr>
  <tr class="hl"><td>cap320 (≤ 320 image tokens)</td><td class="n">+0.0790 [+0.0708, +0.0874]</td></tr>
  <tr><td>1-MP cap (≤ 1,280 tokens)</td><td class="n">+0.0745 [+0.0651, +0.0835]</td></tr>
  <tr><td>choose the cap per benchmark</td><td class="n">−0.0023 vs cap320 (tie)</td></tr>
  <tr><td>lenient EM: cap320 / 1-MP cap</td><td class="n">+0.0122 / +0.0111</td></tr></table>
  <div class="note">no paired interval; the cap320 row is an earlier probe fit (2026-10-15) — the shipped fit gives +0.0798.</div>
  <table class="d sm" style="margin-top:12px"><tr><th>greedy, 1-MP − cap320, same session</th><th class="n">judge</th></tr>
  <tr><td>GEMeX (chest X-ray, median 7.8 MP)</td><td class="n loss">−0.0414 [−0.0488, −0.0341]</td></tr>
  <tr><td>PathVQA</td><td class="n win">+0.0200</td></tr><tr><td>Kvasir-x1</td><td class="n win">+0.0091</td></tr>
  <tr><td>SLAKE, VQA-RAD, OmniMedVQA, VQA-Med</td><td class="n tie">ties</td></tr><tr><td>RadImageNet</td><td class="n">not measured</td></tr></table>
 </div>
 <div class="col">
  <div class="term"><b class="k">cost</b><span>the 1-MP cap means 2.76× the vision tokens per question (4× on GEMeX)</span></div>
  <div class="term t"><b class="k">verdict</b><span>cap320: cheaper and not worse in both metrics — not claimed better. The earlier move to the 1-MP cap was reversed. Both image halves were used.</span></div>
  <div class="term w"><b class="k">open detail</b><span>features were read at the 1-MP cap, answers generated at cap320; re-extraction at cap320 is running</span></div>
 </div>
</div>""", "IV · Why", src="select_policy_cap320_vs_fullres_T07.json macro.{cap320,fullres}_minus_ref_greedy_{judge,em}; resolution_check_2026-10-15.json (7 cells, no radimagenet_open); image_geometry_all8_2026-10-15.json pooled_question_weighted (2.76×), GEMeX 303 → 1,237 tokens",
          minutes=1.0, look="The top table's first two rows; then the GEMeX row in red.",
          say="""# whole method, T = 0.7 (table)
- Why do we cap the image at about 250 thousand pixels? We measured the whole method at a one-megapixel cap as well — what our files call "full resolution".
- Its point estimate was slightly lower under the judge, and a tie in exact match.
# cost
- The one-megapixel cap uses 2.8 times the vision tokens — four times on GEMeX.
# greedy, 1-MP − cap320, same session
- The main effect is GEMeX: its chest X-rays are around eight megapixels, and at the one-megapixel cap the model's own greedy answers get four points worse.
# verdict
- So the cheaper setting is not worse — we do not claim it is better, because we did not compute a paired interval.
# open detail
- One detail is still open: the probe's features were read at the one-megapixel cap.
- An experiment re-reading everything at the generation budget is running.""",
          qa=[("Why does a 1-MP cap hurt a 7.8-MP image that is downsized either way?", "Not explained by a measurement. At cap320 a GEMeX X-ray becomes about 303 image tokens on average; at the 1-MP cap about 1,237; at native resolution it would be about 9,092 (image_geometry_all8_2026-10-15.json benchmarks.gemex_open.mean_tokens). So neither setting is native resolution. The model's greedy accuracy is 4 points lower with the longer image sequence; why is not measured.")])
