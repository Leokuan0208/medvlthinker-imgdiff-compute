# Speaker script — Cheap test-time improvements for a medical VLM

Companion to `meetings/progress_deck_2026-08-17.html` (28 slides).
The heading is what the audience sees; the text under it is what you say.

**The arc in one line:** we built a cascade that matched the big model by spending more than the big
model → we worked out exactly where that money went → we stopped chasing the big model and started
making the small one better → two things fell out, and both are nearly free.

---

### 1 · Cover — We stopped trying to match the 32B and started making the 7B better

Set the frame in two sentences and move on.

Last time the headline was "our cascade matches the 32B". That is still true. The problem is that it
matches it by spending 1.74 times what the 32B costs — so as a piece of engineering it is not worth
deploying. This report is about what we did once we admitted that.

---

### 2 · Eight cells — one benchmark, one answer format

A **cell** is one benchmark paired with one answer format. Three of these datasets have both a closed
split — yes/no or a fixed option — and an open split where the model writes free text. Those two
behave so differently that we score them separately. Eight cells in total.

Worth saying out loud: the format is not a stylistic difference. Almost everything that works on free
text fails on multiple choice, and the reverse. Most of this talk is downstream of that one fact.

### 3 · Every cell counts one eighth

A reporting choice worth defending. PMC-VQA alone is 33,430 of the 42,224 questions. Average over
questions and it carries 79% of the number, so any improvement we report would really be an
improvement on that one dataset — and it is the one dataset here with no published human
verification.

So every cell counts one eighth. And for scale: a change of about **three thousandths** on this scale
is the threshold for significance. Keep that number; it comes back at the end.

---

### 4 · Answer cheaply, escalate when unsure

This is the system as it stood at the last report.

One router reads the prompt text and sends multiple-choice questions down one arm and free-text
questions down another. Each arm answers with the 7B first, and a gate decides whether to trust that
answer. If not, the question goes to the 32B.

The gate signal on the multiple-choice side is the **margin** — the gap between the top-ranked and
second-ranked answer probabilities. A large gap means the model is decided. A small gap means it is
torn between two options, which is when it tends to be wrong.

Everything expensive here is the escalation. That is why so much of the work is about deciding when
*not* to escalate.

### 5 · The three fixed ways of answering

Before any cleverness, here are the three fixed policies you could just run.

Point at the middle row and dwell on it. **The reasoning model is not the strong model.** Averaged
over all eight cells it is level with the 7B — 0.5974 against 0.5971. On free text it is far worse,
and it takes thirty times longer to be worse. The damage is concentrated in PathVQA, where the
reasoning 32B scores 0.109 against the small model's 0.324; it talks itself out of correct
perceptual answers.

That matters for reading the literature. Papers in this area usually quote the reasoning number as
the strong baseline. It is not the strong baseline — the 32B answering *directly* is. Everything in
this report is compared against both, and you will see the reasoning column carried through every
table.

### 6 · On free text we sample several answers and pick one

This is where most of our research went. Rather than take the model's first answer, sample eight and
choose between them.

The third row is the important one. A correct answer is **present** in the eight about 63% of the
time; we end up giving one 49% of the time. That gap — between what is available and what we pick —
is the number this project has spent the most effort trying to close.

### 7 · What the verifier actually is

Explain this properly, because the next act depends entirely on it.

A **LoRA adapter** has no meaning without the base model underneath it. Concretely: for a weight
matrix inside the 7B holding about 12.8 million numbers, LoRA stores two thin matrices totalling
about 114,000, and at inference the model computes `(W + BA)x` instead of `Wx`.

So you have changed *which* weights are used — not *how many operations run*. The forward pass is
still a full pass through the whole network. LoRA makes training cheap and storage cheap. It does
nothing whatsoever for inference cost.

Plant that firmly. It is the setup for the biggest cost finding in the talk.

### 8 · It worked — and that is exactly the problem

Here is the result we reported last time, and here is why we stopped being pleased with it.

The cascade reaches 0.6575 on the eight-cell average. Against the 32B *reasoning* that is a real,
significant win of six points. Against the 32B answering *directly* it is a tie. And it gets there by
spending 1.74 times what the 32B costs.

On a single free-text question it spends **11.8 times** what the 7B alone would spend. Put those
together and, on the free-text half, the 32B turns out to be roughly ten times more efficient per
point of accuracy than we are.

That is not a result you deploy. Everything after this slide follows from taking it seriously.

---

### 9 · Scoring the answers costs as much as writing them

Where does 11.8 go? Two places.

Eight of it is generating the eight candidates — that part is obvious. The other 3.8 is the verifier,
and that part is not obvious at all, so this is the slide to be slow on.

Because a LoRA adapter is a correction to the 7B's weights, *scoring* one candidate means running the
entire 8.29-billion-parameter model again — reading the image again, reading the question again, from
scratch. Eight candidates collapse to about 3.8 genuinely distinct answers after normalisation, so it
is 3.8 extra full passes rather than eight.

Judging the answers costs about half as much as producing them. That is the sentence to leave in the
room.

### 10 · We had been charging ourselves 3.4× too much for sampling

Now the same accounting applied honestly, and it goes the other way.

We had been assuming eight answers cost eight times one answer. They do not — the serving engine
shares the expensive part. Reading the image and the question is *prefill*, and prefill is about 82%
of the total work; actually writing the answer is barely 1%. All eight candidates share the same
image and the same question, so the engine computes that once and reuses it.

Measured, eight samples cost **2.37 times** one answer, not eight. We verified the instrument by
switching the cache off, where it reproduces our old 8.0 almost exactly.

### 11 · The image encoder re-runs 294 tokens to recover five

A nice concrete one, and it is worth a laugh.

The cache works in fixed-size blocks. The image sits at positions 31 through 325, and the cached
region ends at 320 — a block boundary. Because the image runs **five tokens** past that boundary the
engine cannot mark it as already computed, so it re-encodes all 294 of its tokens to recover the last
five.

The language half of the prompt shares properly, 1.16×. The image encoder does not, 4.74×. Fixing it
means encoding each image once and handing the engine the embedding. That takes the cost of eight
samples from 2.37 down to **1.20**.

---

### 12 · We were asking the wrong question

The pivot. Say it plainly.

We had been asking "can a small model plus extra computation match the big one?" The honest answer is
that it can, and it costs more than the big one, so nobody should do it.

The question we ask now is "how much can we improve the small model, for free?" — where free means at
or near the compute the small model already uses. That is a smaller claim. It is also one that
survives its own cost accounting, which the previous one did not.

### 13 · Selection is the bottleneck, and it does not move

Define the number first. **Selection efficiency** is the fraction of the available gain we actually
capture: if a correct answer is somewhere in the pool, how often do we pick it.

We have attacked this roughly twenty-seven distinct ways — prompted judges, larger judges, six other
model families, pairwise comparison, ranking objectives, set-aware scorers. Every one lands between
0.78 and 0.81. Picking at random gets 0.676, so there is real signal; there is just a ceiling on it.

The important part is what we found looking outside: independent systems in the literature report the
same conversion rate. That reframes it from *our* failure into a property of the problem.

And it is why the interesting question stopped being "which selector is best" and became **"what does
the selector cost"**. If they all score the same, take the cheap one.

### 14 · The same job, for a millionth of the compute

So here is the cheap one, and this is the technical heart of the report.

While the 7B is writing a candidate answer, it computes a hidden state at every layer — that is just
what a transformer does. The **head** is a standalone two-layer network, 3584 inputs to 256 to one
output, 918 thousand parameters, that reads the layer-21 state the generator has *already produced*
and outputs a score.

It is not part of the LoRA and it never was. It is trained separately, on the base model with no
adapter attached, in about fifteen seconds of CPU per copy — against 108 GPU-minutes for the adapter.
We train eight copies on different random seeds and average them, purely for stability; individual
seeds land between 0.788 and 0.800.

The comparison to make out loud: the adapter costs **10.7 TFLOP** per candidate because it has to
re-run the model. The head costs **1.8 MFLOP** because it reads a number that already exists. That is
a factor of about six million — **and the cheap one scores higher**, 0.8011 against 0.7752.

### 15 · Combining the two scorers

How the fusion works, since the two scorers speak different languages: the adapter emits a bounded
probability, the head emits an unbounded logit. You cannot add those.

So within each question you convert both to **ranks** — best candidate gets 1.0, worst 0.0 — add the
two rank vectors, and take the highest sum. Ranking throws away the scale and keeps only the ordering,
which is the only thing the two agree on.

Walk the worked example: the adapter would have picked A, the head picks C, the sum picks C, and C is
right.

One detail that sounds pedantic and is not: ties take the *average* of their positions rather than the
first one. That single convention is worth 0.0123 — 0.8106 against 0.7984. On free text ties are
common, because the same answer string appears in several slots.

### 16 · Why the weight is not learned

The obvious question is "why 50/50 — did you tune that?" No, and we checked hard.

We fitted a combiner with full sight of the evaluation answers, an advantage no deployable version
could have. The best weight it found was exactly 0.5, and the fitted version scored *worse* —
0.7997 against 0.8106 — and broke the guardrail. There is nothing for a combiner to learn here.

One consequence to flag: a rank only has meaning within its own question, so the fused number cannot
double as a confidence signal. When we fed it to the escalation gate it collapsed to fifteen distinct
values across 2,345 questions and the controller degenerated.

And then the turn: the fusion *is* the better selector. But you can only have it by running the
adapter — a full model pass per candidate — and that is exactly the cost the rest of this report is
about removing.

---

### 17 · What we give up, and what we stop paying

This is the trade, laid out.

The old structure — adapter only — gets 0.4853 for 3.82 scoring passes. Both together get 0.5075.
The head alone, with its states captured while the 7B generates, gets **0.5023 for zero scoring
passes**.

So dropping the adapter costs about six thousandths of accuracy and removes every scoring pass. The
head is the better of the two selectors on its own, and it is the only one whose cost can be driven
to zero, because it reads a number that already exists. Capturing it rather than recomputing it
changes 15 picks out of 2,345.

Then be explicit about the three things we are not hiding — the guardrail on VQA-RAD, the fact that
the head's edge over the adapter is specific to the judge and reverses under strict string matching,
and the resolution mismatch that costs the free capture some accuracy until the head is refitted. If
your professor asks a hard question, it will be one of these three, so get there first.

### 18 · And 8.00× was never the real number

Now put the two halves of the accounting together, and this is the payoff slide.

Verification is now **0.00002** of a forward pass — five orders of magnitude below everything else on
the table. So the open arm's cost is simply the cost of generating. Charged honestly that is 2.37, and
with the image encoded once it is **1.203**.

Eight answers and a verifier, for 1.2 times the cost of one answer. Across all eight cells that is
1.076× the always-7B baseline for +0.0091 of accuracy.

Be straight about the last row: its *cost* is measured end to end, but its *accuracy* is carried over
on the argument that moving where an image embedding is computed does not change the embedding. That
is sound in exact arithmetic and approximate in the precision we actually run at, so re-scoring the
pool through that path is the first thing on the next-steps list.

### 19 · What the head alone buys, on free text

The per-cell result, with the two 32B columns beside it.

The gain is very uneven: SLAKE +0.036, VQA-RAD flat, PathVQA +0.067 — a fifth of its own accuracy. It
is largest where the model is weakest, and absent where the pool offers nothing to choose between.

Then read the last two columns aloud. The 7B with a head that costs essentially nothing lands at
0.5023. The 32B *reasoning* is at 0.3028 — we are far past it. The 32B answering *directly* is at
0.5168, so we are within fifteen thousandths of it, at half its compute today and a quarter once the
encoder is fixed.

That gap is the whole remaining argument on the free-text side.

### 20 · How adventurous the sampling should be

One more free knob, and this one is on the generator rather than the selector.

Sampling temperature controls how varied the eight candidates are. Two forces pull opposite ways:
hotter sampling puts a correct answer in the pool more often, and makes it harder to identify.
Accuracy is *exactly* the product of those two columns, so the best setting sits in the middle — at
0.4, not the 0.7 we had been using. Worth +0.0094, no retraining, no extra compute, no cell going
backwards.

The bottom row is the one worth arguing about. Sampling at zero and keeping the single answer — the
"just be careful and deterministic" option — is a **significant loss**. Diversity is worth nothing on
its own; it is worth something only to a selector that can exploit it.

Say clearly that this ladder was measured with the LoRA verifier doing the picking. The head has never
been evaluated at 0.4, because it was trained on candidates drawn at 0.7. Retraining it there is our
first next step.

---

### 21 · The benchmark's own instruction biases the model

Now the second result, and it is a different kind of thing entirely.

On the three yes/no cells the model says "yes" about seven points more often than the data warrants.
Define that: the rate at which it answers yes, minus the rate at which yes is actually correct. A
calibrated model scores zero.

Direct attention to the **third** column, not the fourth. Two of the three cells are asked with the
two allowed answers named in the prompt. The third is not — and it is just as biased, from some other
cause we have not identified.

That distinction is what makes the next slide a test rather than a coincidence: if the sentence is the
mechanism, removing it must fix the first two and do nothing to the third.

### 22 · The fix is one sentence, and it is a deletion

Here is the whole intervention.

MedEvalKit appends `Please output 'yes' or 'no'(no extra output).` to every question on these cells.
We replace it with `Please answer the question concisely.` Same model, same weights, same image, same
question. Generated tokens go from 3.0003 to 3.0006 — so it is not a length effect, and the grader
cannot be moving. Compute is 1.000×, exactly.

The bias on PathVQA goes from +0.069 to +0.005. On VQA-RAD from +0.050 to zero. And **SLAKE does not
move at all** — which is the control, and the reason we believe the sentence is the mechanism rather
than a coincidence.

The interpretation: the model already knows the question is binary. It still answers yes or no
without being told. Naming the options does not inform it, it *pushes* it — and the push is
asymmetric.

We also separated "which options" from "in which order": keeping the sentence and only swapping the
order — "output 'no' or 'yes'" — *raised* the bias on all three cells and cost 0.0089 of accuracy on
PathVQA, a significant loss. It is the answer space being supplied at all, not the order.

### 23 · What it is worth

+0.0419 on the cell itself, which is +0.0052 on the eight-cell average — call it 1.8 times the
significance threshold — at exactly 1.000× compute, with no cell going backwards.

The line to land: two completely independent graders, a 32B judge and strict string matching, agree
to **0.0003**. When those two disagree it usually means the grader moved rather than the model. Here
they do not.

### 24 · The test that separates a real fix from a benchmark artefact

The slide I would put money on being asked about, so present it as our own scepticism rather than a
defence.

PathVQA's answer key is 54% yes. So any change that makes the model say "yes" less often will improve
the score *whether or not the model got better*. To separate those, we resample the evaluation set to
a perfectly balanced key and re-measure.

A real improvement should survive. An artefact should vanish. Read the three rows: a rival technique
that matches the model's answer rates to the data's earns +0.0202 as measured and **−0.0001** on a
balanced key — it was harvesting the skew, and we discarded it. On another benchmark it is negative
both ways. Our prompt fix goes from +0.0416 to **+0.0471** — it gets *bigger*.

That is the strongest single piece of evidence in the report.

---

### 25 · We made the measurement stricter three times

Three corrections we made to ourselves, each of which made our own headline smaller, and all three
found before anyone outside read the work.

The prompt one deserves a sentence: we had been comparing a reasoning arm against a direct arm, but
the two prompts also differed in the answer format they asked for. Once we matched the format, the
effect we had attributed to reasoning turned out to be an effect of the format request — asking a
model to put its answer in a box is itself enough to make it reason.

The contamination one is a simple audit: if removing the image barely changes the score, the model is
not reading the image.

### 26 · What did not work

Do not skip this. It is a third of the actual work and it is what makes the positive results
believable.

Making the candidate pool more diverse raised what was *available* and never reached what we *pick*.
Scoring the multiple-choice options with the verifier is beaten by the model's own answer on all four
option cells. Deciding adaptively how many answers to sample loses to a fixed number chosen once.
Voting among the samples helps on two cells and does nothing on five.

### 27 · Where we stand

Four things, briefly.

The verifier is no longer the expensive part — a 918-thousand-parameter head reading a vector the
generator already produced picks better than a 47-million-parameter adapter that costs a full model
pass. A free, significant improvement on the prompt side that survives the hardest test we have. An
honest map of the limits, with a selection ceiling that turns out to be a field constant. And what is
still open: the prompt effect lives on one cell, the head has never been evaluated at the temperature
we now want, and nothing has been run end to end.

Then the closing line. The shape of the claim has changed. It used to be "a 7B plus compute can match
a 32B", which was true and cost more than the 32B. It is now "a small trained verifier improves a 7B
on its three free-text cells at 1.2× its own compute, and a prompt correction improves it on a fourth
for free." Smaller claim. It survives its own cost accounting, which the old one did not.

### 28 · What we do next

Three items, in order.

**One — retrain the head at the temperature we actually want to sample at.** It was trained on
candidates drawn at 0.7 and the ladder says we should be drawing at 0.4. That needs one fresh capture
per generation seed and about fifteen seconds of CPU per seed to refit. Refit it on the generator's
own resolution at the same time, which is the one place the free capture currently loses accuracy.

**Two — check the other benchmarks for the same instruction defect.** Two of three yes/no cells carry
the sentence. If the pattern shows up in the multiple-choice harnesses and in other model families,
this stops being one result on one dataset and becomes a general finding about how these systems are
evaluated. That is the version of this that is worth a paper.

**Three — merge the pieces into one method and run it end to end.** Prompt fix on the yes/no cells,
head selection on free text, sampling at 0.4, states captured during generation, the image encoded
once. Right now each is measured on its own by re-scoring saved outputs. Nothing has ever been
executed as a single program — and until it is, the combined number is an expectation, not a
measurement.

Close on why we expect them to compose: they act on different cells and at different stages, and there
is no cell where two of them contend for the same gain. That is a reason to expect it, not a
substitute for measuring it — and the measurement is what the next report should carry.

---

## Likely questions, and the honest answer

**"Why is the head better than the adapter if it is so much smaller?"**

Because it is not doing the same job. The adapter re-reads the image and the question from scratch and
forms its own opinion. The head reads the generator's *own internal state* while it was writing that
answer — it is asking "did the model believe what it just said", which turns out to be a better
question than "is this answer correct".

**"Is 0.5023 versus the 32B-direct 0.5168 a real gap?"**

Yes, and we do not claim to have closed it. What we claim is that the gap now costs a quarter of the
32B's compute instead of 1.74 times it.

**"The prompt fix only works on one cell."**

Correct, and it is on the slide. Two cells carry the defective instruction; on the smaller of the two
(n=251) the bias is removed but the accuracy does not move significantly. Whether this generalises is
next step two, and it is the thing that decides whether this is a footnote or a finding.

**"You are using a 32B model as the grader. Isn't that circular?"**

It is a fair worry, and it is why every intervention here is also reported under strict string
matching. On the prompt fix the two agree to 0.0003. On the head-versus-adapter comparison they
disagree, and we say so on that slide rather than picking the flattering one.

**"Has any of this been run end to end?"**

No. Every operating point in this report is a re-scoring of saved per-candidate outputs plus a
measured re-costing. That is next step three, and it is stated on the slide.
