"""One-off: convert the say= strings of slides 17-23 in deck_part1.py to the titled-bullet script format."""
P = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/deck_part1.py"
s = open(P).read()


def swap(start, end, new):
    """Replace the say text that begins with `start` and ends with `end` (inclusive)."""
    global s
    i = s.index(start)
    j = s.index(end, i) + len(end)
    assert s.count(start) == 1
    s = s[:i] + new + s[j:]


swap('say="""Snell and colleagues framed', 'I show it in part four.""")', '''say="""# Scaling LLM Test-Time Compute Optimally …
- Snell and colleagues framed the modern debate: spend compute at test time, or on a bigger model?
- frame: every method is a proposer plus a verifier.
- method: they choose the strategy per question according to its difficulty; their verifiers are process reward models.
- result: smart test-time search can match plain best-of-N with four times less compute.
- And it can beat a 14-times-larger model — but only, in their words, "on problems where a smaller base model attains somewhat non-trivial success rates".
- On the hardest questions, they say, pretraining is the more effective way.
# a similar shape in our data
- On OmniMedVQA, a correct answer is among the eight samples 70 percent of the time — and we gain 16 points.
- On VQA-Med, that is only 21 percent — and we gain nothing.
  - Though I will show that there the probe also fails to find the correct answers that do exist.
# against us
- Step-level verifiers beat outcome-level ones in their data, more so as N grows.
- Their method chooses the budget per question; ours uses one fixed setting.
# best-of-N weighted
- Their default way of combining scores: add up the verifier's score over identical answers.
- We tested it against our own readout; I show it in part four.""")''')

swap('say=f"""The simplest selector is majority vote', 'and that is a gap.""",', '''say=f"""# Self-consistency = majority vote
- The simplest selector is majority vote, which the field calls self-consistency.
- On math it is very strong: on grade-school math with a big model it raised accuracy from 56 to 74 percent.
- Because a correct number tends to come up again and again.
# The modal ceiling
- Bay and Yearick showed its hard ceiling.
- As you add samples, voting converges to the most frequent answer.
- So it can never beat the rate at which the most frequent answer is correct. A learned scorer is not bound by that.
# majority vote − greedy, training judge (shipped pools)
- The chart is majority vote minus greedy on our pools.
- On most benchmarks it is negative, and averaged it loses {abs(SCJ[0]):.4f}.
- Free-text medical answers rarely repeat exactly, and the right answer is often a minority.
  - Like "Large Bowel", which appeared once in our example.
# For free text
- There are training-free methods designed for free text:
  - USC asks the language model which answer is the most consistent with the others;
  - EBA picks the centre of the largest cluster of answer embeddings;
  - self-certainty scores each sample from the model's own output probabilities.
- We have not run them, and that is a gap.""",''')

swap('say="""These are our three closest relatives', 'when I justify our choices.""",', '''say="""# Hidden-state verifiers in text LLMs (table)
- These are our three closest relatives, all on text-only language models.
# Q-Probe
- Q-Probe, in 2024, put a linear probe on a frozen code model's embeddings to rerank 48 sampled programs.
# SWIFT
- SWIFT reads every token of every layer and pools them with a learned gate.
- It trains with binary cross-entropy — the same kind of loss as ours; ours additionally up-weights the correct answers.
# HSRM
- HSRM reads the states at the end of each reasoning step with a small Transformer and picks the best of eight — the same N as ours.
# ours
- The last row is us, so you can compare column by column.
- We read only the answer's own tokens, at three middle layers, with a simple mean, and a one-hidden-layer network.
# cuts against it
- I want to be honest about what these papers say against our choices.
- HSRM found a pairwise ranking loss, Bradley–Terry, better than our loss — 85.9 versus 83.9 on grade-school math.
- Q-Probe found a within-question loss better than a pointwise one, and saw MLPs overfit on preference data.
- SWIFT found that learning which tokens to weight beats a plain average, by 2.7 points.
# supports our design
- On the other side, SWIFT found our loss the best of five.
- HSRM found the reader's architecture matters much less than what you feed it.
  - A mean-pooled MLP reader did about as well as their Transformer reader.
- I will come back to each of these when I justify our choices.""",''')

swap('say="""Three more relatives, each teaching something.', 'up to 810 times larger.""")', '''say="""# CASE — a decodability criterion
- Three more relatives, each teaching something.
- CASE, from August 2026, asks exactly when choosing by hidden states beats majority voting.
- Key contribution: within-question AUC — how well the probe orders the candidates of the same question.
- Ordinary cross-validation badly overstates probes: a probe can learn which questions are easy instead of which answers are right.
  - 0.61 drops to 0.50 when measured properly.
- That metric predicts how much you gain over voting.
- On a medical multiple-choice set, the medium-difficulty questions, it gained 19 points over voting.
- against us, two findings:
  - a small logistic regression, 0.596, beat a 256-unit MLP, 0.537, in their small-data setting;
  - averaging over tokens hurt in their multiple-choice setting.
- Also, on those medium questions, voting caught up and overtook them by N of 16.
# our gap
- Our evaluation avoids the leakage they warn about by splitting by image.
- But we have not reported their within-question AUC, and we should.
# Lightweight latent verifier
- LiLaVe uses gradient-boosted trees on the hidden states of the last 16 tokens of five layers.
- In their data plain best-of-n eventually fell below voting, and weighted voting was best.
# Same head shape as ours
- ReProbe has the same classifier shape as ours.
  - One Transformer layer, a mean over the span, then linear, GELU, linear.
  - Trained with binary cross-entropy on labels from an LLM judge.
- It matches process reward models up to 810 times larger.""")''')

swap('say="""On the medical side, three papers set the context.', 'AUROC 0.95 in distribution, 0.67 outside.""",', '''say="""# Verification Mirage
- On the medical side, three papers set the context.
- Verification Mirage asked medical VLMs, including Lingshu, to check their own answers.
- Lingshu accepted 62 to 73 percent of its own wrong answers.
- A bigger model of the same family did not significantly help.
- Over four turns of conversation, the wrong answers got locked in.
- That is why our selector is trained on labels rather than prompted.
- One point to keep apart: in part four, a separate 32-billion model judging our 7B's candidates without training does better than a 7B doing the same.
  - That does not conflict with Mirage.
  - Mirage measures whether a model accepts its own wrong answers when it re-checks them.
  - We measure best-of-8 selection accuracy, with a different model as the judge.
  - Both point the same way: training, not prompting, is what makes a selector work.
# MedProb
- A simple linear probe on a medical VLM's internal state answers closed multiple-choice questions better than the model's own generated answer.
- By more than 20 points in several cases.
- But it reads the state before any answer is written, so it is a classifier, not a selector.
- In an appendix they tried reranking open answers on a hundred OmniMedVQA questions, from 0.09 to 0.13.
  - They call it an initial demonstration; the number of samples, the grader and the probe type are not stated.
# DualRead
- DualRead re-reads a frozen medical VLM's answer with teacher forcing — the same re-reading we do.
- But only to calibrate confidence.
- Its quality drops sharply on new datasets: AUROC 0.95 in distribution, 0.67 outside.""",''')

swap('say="""Two recent papers do test-time selection specifically for medical VQA.', 'I show that result in part six.""")', '''say="""# Wasserstein Equilibrium Decoding (WED)
- Two recent papers do test-time selection specifically for medical VQA.
- WED is training-free.
- It samples up to eight distinct candidates, using up to sixteen draws at temperatures 0.5 and 1.0, and merges answers with the same meaning.
- Then the same model plays two roles:
  - a generator that scores how likely each candidate is;
  - a verifier that says whether it is correct.
- The two roles adjust toward each other in a game until their rankings agree.
  - Measured with a Wasserstein distance, so that synonyms count as agreement.
- The game itself comes from earlier work, the Consensus Game and the Bayesian Decoding Game; WED's addition is the stopping rule.
- Graded by the Grok model, it reports one to five points over greedy on most small models.
- And a small loss for the largest one on VQA-RAD — so gains shrink as models get bigger.
# Best-of-Evidence
- Samples sixteen answers and uses a 235-billion-parameter judge to check individual claims — modality, anatomy and so on.
- It gains under one point over majority vote, significant on two of four datasets.
- No greedy baseline, and its VQA-Med items come from the training split.
# for us
- WED matters most for us: the only published selector evaluated on two of our open benchmarks, VQA-RAD and PathVQA.
- So rather than compare numbers across papers, we ran WED's own code on our candidates.
- I show that result in part six.""")''')

swap('say="""Putting it together. The literature', 'against a published selector on\nidentical candidates.""",', '''say="""# Where this work sits (table columns A–D)
- Putting it together. The literature's hidden-state verifiers share four properties:
  - (A) a probe on frozen states;
  - (B) used to select among samples;
  - (C) pooled over the answer's own tokens;
  - (D) in principle no extra pass through the model, because the states were already computed while generating.
- The text-only papers have most of these.
- The medical papers mostly probe the model without selecting free-text answers.
- In vision the idea also exists, for robot actions and for image generation.
# ours
- Our row has all of them except one honest asterisk, column D.
- In our experiments we re-read each answer in a separate pass, so "no extra pass" is true by design, not yet as run.
- For an earlier probe, capturing the states inside generation changed 24 of 2,345 picks and left accuracy unchanged.
- For the shipped probe, re-extracting the features at the generation resolution is running now.
# contribution, stated narrowly
- So the contribution is the evaluation, not the mechanism.
- MedProb's appendix tried this on a hundred questions.
- We measure it on eight benchmarks, with held-out images.
- With a second grader from another model family, plus string matching.
- And against a published selector on identical candidates.""",''')

open(P, "w").write(s)
print("ok")
