# Speaker script — Cheap test-time improvements for a medical VLM

Companion to `meetings/progress_deck_2026-08-17.html`. One entry per slide.
The heading is what the audience sees; the text under it is what you say.


## Act I · the system

### Slide 01 — What we report on

A cell is one benchmark paired with one answer format. Three of these datasets have both a closed split — where the answer is yes/no or a fixed option — and an open split, where the model writes free text. Those behave so differently that we score them separately. That gives us eight cells.

Worth saying out loud: the formats are not a stylistic difference. Almost everything that works on free text fails on multiple choice, and vice versa. Most of this talk is downstream of that.

### Slide 02 — Every cell counts the same

This is a reporting choice worth defending. If we averaged over questions rather than over cells, one dataset would carry 79% of the number — and that dataset is the one with no published human verification. Any improvement we reported would really be an improvement on it alone.

So every cell counts one eighth. It makes our numbers smaller, and it makes them mean something. For scale: a change of about ±0.003 on this scale is the threshold for statistical significance.

### Slide 03 — The system: answer cheap, escalate when unsure

The small model answers first. A gate decides whether to trust it. If not, the question goes to the big model.

The gate signal is the margin — the gap between the top-ranked and second-ranked answer probabilities. A large gap means the model is decided; a small gap means it is torn between two options, which is when it tends to be wrong.

Everything expensive in this system is the escalation. That is why so much of the work is about deciding when not to escalate.

### Slide 04 — On free-text questions we sample several answers and pick one

This is the part of the system most of our research went into. Rather than take the model's first answer, we sample eight and choose between them.

The third row is the important one. A correct answer is present in the eight about 63% of the time, but we only select it 49% of the time. That gap — between what's available and what we pick — is the single number this project has spent the most effort trying to close.

### Slide 05 — What the verifier actually is

Explain LoRA properly, because the next act depends on it.

A LoRA adapter is not a model. It is a set of small correction matrices that get applied on top of an existing model's weights. On its own it cannot see an image or read text — it has no meaning without the base model underneath it.

Concretely: for a weight matrix inside the 7B with about 12.8 million numbers, LoRA stores two thin matrices totalling about 114,000, and at inference the model computes (W + BA)x instead of Wx.

So you have changed which weights are used — not how many operations run. The forward pass is still a full pass through all 8.29 billion parameters. LoRA makes training cheap and storage cheap. It does nothing whatsoever for inference cost. Scoring one candidate costs a full run of the 7B.

Plant that. It's the setup for the biggest cost finding in the talk.

### Slide 06 — We made the measurement stricter three times

Each of these made our headline number smaller. I want to be explicit that we found all three ourselves, before anyone else read the work.

The prompt one is worth a sentence: we had been comparing a reasoning arm against a direct arm, but the two prompts also differed in the answer format they requested. Once we matched the format, the effect we had attributed to reasoning turned out to be an effect of the format request. Asking a model to put its answer in a box is itself enough to make it reason.

The contamination one is a straightforward audit: if removing the image barely hurts the score, the model isn't reading the image.


## Act II · the limits we found

### Slide 07 — Selection is the bottleneck, and it does not move

Define the number first: selection efficiency is the fraction of the available gain we actually capture — if a correct answer is in the pool, how often do we pick it. Random picking scores about 0.68. We sit at 0.78–0.81.

We attacked this roughly twenty-seven distinct ways. Every single one lands in the same narrow band. The important part is what we found when we looked outside: independent systems in the literature report the same conversion rate. This is a property of the problem, not a failure of our engineering.

Our own seed-to-seed variation is larger than the difference between most of those architectures — which means much of the published variation at this scale is probably noise too.

### Slide 08 — We had been charging ourselves 3.5× too much for sampling

We had assumed eight answers cost eight times one answer. They don't — the serving engine shares the expensive part.

Reading an image and a question is prefill, and it's about 82% of the total work; actually writing the answer is barely 1%. All eight candidates share the same image and question, so the engine computes that once and reuses it. We measured this with the cache deliberately switched on and off: with it off we reproduce our old 8.0, with it on we get 2.28.

This matters because it changes what is affordable. Sampling eight answers is not an eight-fold cost — it's roughly double.

### Slide 09 — The image encoder re-runs 294 tokens to recover five

This is a nice concrete one. The cache works in fixed-size blocks. The image sits at positions 31 through 325, and the cached region ends at 320 — a block boundary. Because the image runs five tokens past that boundary, the engine cannot mark it as already computed, so it re-encodes all 294 image tokens to recover the last five.

We tried every configuration route to fix it — batch size, concurrency, cache size — and got identical results to three decimals every time. It isn't a tuning problem: the engine keys that cache by request identity, and the eight candidates are eight different requests.

Passing pre-computed image features instead fixes it exactly, and that's the 1.20 on the previous slide.

### Slide 10 — The same job, for a millionth of the compute

Here is the payoff for the LoRA explanation earlier.

While the 7B processes a candidate answer, it builds an internal representation at every layer. The head is a small standalone network that takes that vector — from layer 21 — and maps it to a single score. It does not modify the model. It is not a forward pass through anything. It is about 1.8 million operations, roughly a millionth of what the adapter costs.

The gap is that large because they are different kinds of object. The adapter is the 7B with altered weights, so using it means running the whole network again. The head reads the network's output. One re-runs the model; the other reads it.

And the cheaper one scores higher. Random picking, for reference, is 0.676.

### Slide 11 — Combining the two scorers

Why ranks and not scores: the two scorers produce numbers on completely different scales, so averaging the scores directly would let one of them dominate for no good reason. Ranking each first puts them on a common footing.

Why equal weight: we tried learning the weight. Fitted with full visibility of the answers, the best weight it found was 0.5 — the parameter-free choice. So there was nothing to learn.

And the recommendation: on our primary metric the head alone captures nearly all of the combined gain — the difference between them is not statistically significant — while costing essentially nothing. That's the version worth deploying.


## Act III · changing the question

### Slide 12 — We were asking the wrong question

Two things forced this. First, the selection ceiling — we now know how much is reachable and it isn't enough. Second, and more decisive: on free-text questions the big model is about ten times more efficient per point of accuracy. Running the small model eight times and scoring the results costs more than simply asking the big model once, and is less accurate.

So the honest move was to stop competing with the big model and change the baseline. Everything after this is measured against the 7B alone, and the question is how much we can add without adding cost.

### Slide 13 — The benchmark's own instruction biases the model

The evaluation harness tells the model what the answer options are. That instruction pushes it toward answering yes — by about seven percentage points more than the data warrants.

If we simply ask the question without supplying the answer space, the bias largely disappears. The model still answers yes or no — it knows the question is binary — it just stops being pushed.

Two controls worth mentioning. We checked the generated answers didn't get longer, so this isn't the grader being fooled. And we tried merely reordering the options — "output no or yes" — which did not help, and hurt on one cell. So it is the answer space being given at all, not the order.

### Slide 14 — The result

State it plainly: a free improvement, larger than the significance threshold, and nothing regresses.

The two-grader agreement matters. We score with an automatic judge and with strict string matching, and they agree to within three ten-thousandths. When those two disagree it almost always means we've moved the grader rather than the model — here they don't.

### Slide 15 — The test that separates a real fix from a benchmark artefact

This is the slide I'd spend the most time on.

There is an obvious rival explanation for any gain on these benchmarks: you might not be improving the model at all, just exploiting the fact that the answer key is lopsided. One of these datasets has 74% of its answers concentrated in two of four options.

So we re-scored everything against a balanced answer key — reweighted so no answer is more common than any other. If a gain came from exploiting the key, it disappears. If it came from the model genuinely answering better, it survives.

The first row is a method that looked like it gained two points. Balanced, it is exactly zero. It was reading the answer key, not improving the model — so we discarded it.

Our prompt fix goes the other way: it gets larger. That's what a real improvement looks like, and it's why I'm comfortable putting it in front of you.

### Slide 16 — What didn't work

Keep this brisk. The second one is worth a sentence because it surprised us: we thought the model was constrained by being shown the options. It isn't — on a binary question it produces the same handful of answers either way. The answer space belongs to the question, not the prompt.

### Slide 17 — Where we stand

Be straight about the shape of it. We have one clean result and a lot of well-mapped territory around it. The result is free, it's statistically solid, and it got bigger under the most adversarial check we could design — which is unusual and worth trusting.

The honest limitation is that it currently rests on one cell. The obvious next step is whether the same instruction defect exists in the other benchmarks, because if it does, this stops being one result and becomes a general finding about how these systems are evaluated.
