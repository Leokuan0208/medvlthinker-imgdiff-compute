# Speaker script — Cheap test-time improvements for a medical VLM

Companion to `meetings/progress_deck_2026-08-17.html`.
The heading is what the audience sees; the text under it is what you say.

### 1. A free improvement, anda much better map of the limits. · Progress report

_(cover slide — no notes)_

### 2. Eight cells — one benchmark, one answer format · 1 · How we measure

A cell is one benchmark paired with one answer format. Three of these datasets have both a closed split — yes/no or a fixed option — and an open split where the model writes free text. Those two behave so differently that we score them separately. Eight cells in total.

Worth saying out loud: the format is not a stylistic difference. Almost everything that works on free text fails on multiple choice, and the reverse. Most of this talk is downstream of that one fact.

### 3. Every cell counts one eighth · 2 · How we measure

This is a reporting choice worth defending. PMC-VQA alone is 33,430 of the 42,224 questions. Average over questions and it carries 79% of the number, so any improvement we reported would really be an improvement on that one dataset.

So every cell counts one eighth. And for scale — a change of about three thousandths on this scale is the threshold for significance. Keep that number in mind; it comes back at the end.

### 4. Answer cheaply, escalate when unsure · 3 · The system

The small model answers first. A gate decides whether to trust it. If not, the question goes to the big model.

The gate signal is the margin — the gap between the top-ranked and second-ranked answer probabilities. A large gap means the model is decided. A small gap means it is torn between two options, which is when it tends to be wrong.

Everything expensive here is the escalation. That is why so much of the work is about deciding when not to escalate.

### 5. On free text we sample several answers and pick one · 4 · The system

This is where most of our research went. Rather than take the model's first answer, we sample eight and choose between them.

The third row is the important one. A correct answer is present in the eight about 63% of the time, but we only select it 49% of the time. That gap — between what is available and what we pick — is the number this project has spent the most effort trying to close.

### 6. What the verifier actually is · 5 · The system

Explain this properly, because the next act depends on it.

A LoRA adapter has no meaning without the base model underneath it. Concretely: for a weight matrix inside the 7B holding about 12.8 million numbers, LoRA stores two thin matrices totalling about 114,000, and at inference the model computes (W + BA)x instead of Wx.

So you have changed which weights are used — not how many operations run. The forward pass is still a full pass through the whole network. LoRA makes training cheap and storage cheap. It does nothing whatsoever for inference cost.

Plant that firmly. It is the setup for the biggest cost finding in the talk.

### 7. We made the measurement stricter three times · 6 · The system

The prompt one deserves a sentence. We had been comparing a reasoning arm against a direct arm, but the two prompts also differed in the answer format they asked for. Once we matched the format, the effect we had attributed to reasoning turned out to be an effect of the format request — asking a model to put its answer in a box is itself enough to make it reason.

The contamination one is a simple audit: if removing the image barely changes the score, the model is not reading the image.

### 8. Selection is the bottleneck, and it does not move · 7 · The limits

Define the number first. Selection efficiency is the fraction of the available gain we actually capture: if a correct answer is in the pool, how often do we pick it.

We attacked this roughly twenty-seven distinct ways and every one lands in the same narrow band. The important part is what we found looking outside: independent systems report the same number. That reframes it from a failure into a finding.

And our own seed-to-seed variation is larger than the difference between most of those architectures — which suggests much of the published variation at this scale is noise too.

### 9. We had been charging ourselves 3.5× too much for sampling · 8 · The limits

We had assumed eight answers cost eight times one answer. They do not — the serving engine shares the expensive part.

Reading the image and the question is prefill, and it is about 82% of the total work. Actually writing the answer is barely 1%. All eight candidates share the same image and question, so the engine computes that once and reuses it.

This matters because it changes what is affordable. Sampling eight answers is not an eight-fold cost. It is roughly double.

### 10. The image encoder re-runs 294 tokens to recover five · 9 · The limits

A nice concrete one. The cache works in fixed-size blocks. The image sits at positions 31 through 325 and the cached region ends at 320, a block boundary. Because the image runs five tokens past that boundary the engine cannot reuse it, so it re-encodes the whole thing.

We tried every configuration route — batch size, concurrency, cache size — and got identical results to three decimal places every time. It is not a tuning problem: the engine keys that cache by request identity, and the eight candidates are eight different requests.

Passing pre-computed image features instead fixes it exactly, and that is the 1.20 on the previous slide.

### 11. The same job, for a millionth of the compute · 10 · The limits

Here is the payoff for the LoRA explanation earlier.

While the 7B processes a candidate answer, it builds an internal representation at every layer. The head is a small standalone network that takes that vector — from layer 21 — and maps it to a single score. It does not modify the model. It is not a forward pass through anything. About 1.8 million operations, roughly a millionth of what the adapter costs.

The gap is that large because they are different kinds of object. The adapter is the 7B with altered weights, so using it means running the whole network again. The head reads the network's output.

And for reference: picking at random scores 0.676.

### 12. Combining the two scorers · 11 · The limits

Why ranks and not scores: the two scorers produce numbers on completely different scales, so averaging them directly would let one dominate for no good reason. Ranking each first puts them on a common footing.

Why equal weight: we tried learning the weight. Fitted with full visibility of the answers, the best weight it found was 0.5 — exactly the parameter-free choice. There was nothing to learn.

And the recommendation. On our primary metric the head alone captures nearly all of the combined gain while costing essentially nothing, so that is the version worth deploying.

### 13. What the head alone buys, on free text · The result

This is the head on its own, against the 7B on its own, on the three free-text cells — no big model anywhere, and no other intervention mixed in.

The total is **+0.0529**, interval +0.0371 to +0.0687, which is an **11.8% relative improvement** on free text. Nothing goes backwards.

The shape matters more than the total. It is largest exactly where the model is weakest: PathVQA is the hardest cell at 0.324, and it gains a fifth of its own accuracy. SLAKE, already at 0.736, gains about five percent relative. VQA-RAD does not move at all — with 200 questions and a narrow pool, there is often nothing to choose between.

If asked why this is not the headline number: it is measured on three of the eight cells, so on the whole-suite average it is diluted by the five cells it does not touch. Both framings are in the deck — this one shows what the component actually does.

One clarification worth volunteering. The head and the prompt fix never overlap. The head needs several candidate answers to choose between, so it only applies to free text. The prompt fix works by not supplying an answer space, so it only applies to the yes/no cells. They act on different cells — they add across the suite, they do not stack on any single cell.

### 14. We were asking the wrong question · 12 · The reframe

Two things forced this. First the selection ceiling — we now know how much is reachable and it is not enough. Second, and more decisive: on free-text questions the big model is roughly ten times more efficient per point of accuracy.

So the honest move was to stop competing with it and change the baseline. Everything after this is measured against the 7B alone, and the question is how much we can add without adding cost.

### 15. The benchmark's own instruction biases the model · 13 · The result

The model still answers yes or no — it knows the question is binary. It just stops being pushed.

Two controls worth mentioning. We checked the answers did not get longer, so this is not the grader being fooled. And we tried merely reordering the options — “output no or yes” — which did not help and hurt on one cell. So it is the answer space being given at all, not the order it is given in.

### 16. What it is worth · 14 · The result

State it plainly: a free improvement, larger than the significance threshold, and nothing regresses.

The two-grader agreement is the part I would emphasise. We score with an automatic judge and with strict string matching, and they agree to within three ten-thousandths.

### 17. The test that separates a real fix from a benchmark artefact · 15 · The result

This is the slide to spend time on.

There is an obvious rival explanation for any gain on these benchmarks: you might not be improving the model at all, only exploiting a lopsided answer key. One of these datasets has 74% of its answers concentrated in two of four options.

So we re-scored everything against a balanced answer key — reweighted so no answer is more common than any other. If a gain came from exploiting the key it disappears. If it came from the model genuinely answering better it survives.

The first row is a method that looked like it gained two points. Balanced, it is exactly zero. It was reading the answer key, not improving the model, so we threw it away.

Our prompt fix goes the other way — it gets larger. That is what a real improvement looks like, and it is why I am comfortable putting it in front of you.

### 18. What did not work · 16 · Closing

Keep this brisk. The second one is worth a sentence because it surprised us: we thought the model was constrained by being shown the options. It is not — on a binary question it produces the same handful of answers either way. The answer space belongs to the question, not to the prompt.

### 19. Where we stand · 17 · Closing

Be straight about the shape of it. One clean result, and a lot of well-mapped territory around it. The result is free, it is statistically solid, and it got bigger under the most adversarial check we could design — which is unusual and worth trusting.

The honest limitation is that it rests on one cell. So the obvious next step is whether the same defect exists elsewhere.
