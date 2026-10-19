"""Replace the 24 say= strings of deck_part2.py (slides 24-47, in order) with titled-bullet versions (SCRIPT_FORMAT.md)."""
import re
P = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/deck_part2.py"

NEW = [
# 24 p3
(False, r'''# Our method, piece by piece
- Part three: our method, one step at a time.'''),
# 25 pipeline
(False, r'''# The method in one figure
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
- Everything on this slide is a real number from the shipped probes, and these probes never saw this image during training.'''),
# 26 data (f-string)
(True, r'''# benchmark (table, left)
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
- The effect of exposure on our gain is not measured.'''),
# 27 step1
(False, r'''# what is sampled
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
- On average there are 4.4 distinct answers per question.'''),
# 28 step2
(False, r'''# system prompt · image tokens · question · answer 𝒜(y)
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
- Matching did not help, but that test also changed the samples.'''),
# 29 layers
(False, r'''# indexing
- Why layers 18, 20 and 22? First a precise definition: layer 18 means the vector after decoder block 18 of 28.
- So these sit at about two-thirds to four-fifths of the depth.
# why three layers
- The bars show the gain over greedy for a probe trained on each single layer, on the five benchmarks where this was measured.
- All are between 0.101 and 0.104, so the differences are at noise level.
- The three-layer ensemble is 0.105.
- Honestly, the ensemble is a hedge against choosing the wrong single layer, not a real gain — on the full pool it adds 0.0007.
# literature
- The literature agrees that correctness information is spread over the upper half of the network.
- HSRM finds the upper layers best; MedProb finds a later peak for a different, multiple-choice task.'''),
# 30 probe
(False, r'''# standardise
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
- It is pointwise: it looks at one answer at a time and never compares answers; the comparison happens in the next step.'''),
# 31 training
(False, r'''# label (top-left box)
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
- It needs labelled data from each benchmark, which I quantify later.'''),
# 32 readout
(False, r'''# Large Bowel · Small Bowel · Spleen (rank grid)
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
- Averaging raw, standardised or calibrated scores instead all tied with rank averaging under both judges — so ranks are the scale-free default, not a measured win.'''),
# 33 probesN
(False, r'''# Changing N changes how many answers are scored, never which probes are used.
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
- That is at roughly twice the generation cost.'''),
# 34 evalproto
(False, r'''# score every question in half A
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
- Resampling individual questions would make the intervals too narrow.'''),
# 35 cost
(False, r'''# What it costs: cheap sampling, negligible probe, costly re-read
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
- We have not measured the whole system end to end.'''),
# 36 p4
(False, r'''# Why each choice
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
- When a slide says July, August or September, it is evidence from an earlier version.'''),
# 37 whytable
(False, r'''# Design decisions at a glance
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
- One small surprise in this row: under exact match, a probe trained on the other judge's labels does a little better — every judge has its own taste.'''),
# 38 whysel (f-string)
(True, r'''# selector − greedy, training judge, macro over 8 benchmarks (chart)
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
- The probe gains plus {D.macro("judge")[0]:.3f}; on exactly the same pools it beats majority vote by 0.089 under the judge and by 0.014 in exact match, both significant.'''),
# 39 whyprobe
(False, r'''# same 2,345 questions (table)
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
- A LoRA verifier trained on the same pooled data was never built, so I cannot claim the probe is more accurate than that.'''),
# 40 whyspan
(False, r'''# what the probe reads (layer 21, July features)
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
- Learned pooling is something we have not tried.'''),
# 41 whymlp
(False, r'''# 5-fold CV selection efficiency, by probe width (chart)
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
- LayerNorm in the probe and the learning rate were never tested end to end.'''),
# 42 whybce
(False, r'''# confounded
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
- A ranking loss on the final recipe is one of the experiments still owed.'''),
# 43 whyreadout
(False, r'''# rule fixed in advance
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
- It trades one metric for another, so under our rule it was not adopted — but if exact match were the target, it would be the better readout.'''),
# 44 whypool
(False, r'''# pooling
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
- On the two hardest cases, RadImageNet and OmniMedVQA, about a hundred labelled questions recover about half and two-thirds of the full gain.'''),
# 45 whyN
(False, r'''# 8 → 16, all benchmarks
- More samples help: doubling to sixteen gains another 0.017 overall, with significant wins on RadImageNet, Kvasir-x1, OmniMedVQA and GEMeX.
# more samples help only where selection works
- But only on the benchmarks where the probe already selects well.
- VQA-Med shows the other case: with more samples a correct answer appears more often, but the probe cannot find it, so accuracy does not move.
# over-optimisation risk
- More samples also give a selector more chances to be fooled.
# so
- Eight is our default, inherited from earlier work, where the cost was measured; sixteen is an option if accuracy matters more than time.
- We never tried fewer than eight with the probe.
- With the earlier LoRA verifier, four samples at a lower temperature did as well as eight, so that is worth testing.'''),
# 46 whyT
(False, r'''# training judge · lenient EM (left chart)
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
- Its point estimate was lower than eight samples at 0.7 in every metric.'''),
# 47 whyres
(False, r'''# whole method, T = 0.7 (table)
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
- An experiment re-reading everything at the generation budget is running.'''),
]

src = open(P).read()
out, pos, k = [], 0, 0
for m in re.finditer(r'say=(f?)("""|")', src):
    if m.start() < pos:
        continue
    q = m.group(2)
    body_start = m.end()
    if q == '"""':
        end = src.index('"""', body_start) + 3
    else:
        i = body_start
        while True:
            i = src.index('"', i)
            if src[i - 1] != "\\":
                break
            i += 1
        end = i + 1
    is_f, text = NEW[k]
    assert "'''" not in text and '"""' not in text
    out.append(src[pos:m.start()])
    out.append(("say=f" if is_f else "say=") + '"""' + text + '"""')
    pos = end
    k += 1
out.append(src[pos:])
assert k == len(NEW) == 24, k
open(P, "w").write("".join(out))
print("replaced", k)
