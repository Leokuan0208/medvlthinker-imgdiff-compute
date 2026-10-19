# Deck + script spec — everything Leo asked for (2026-10-18/19). Check every review round against this.

## Purpose
Presentation material for the professor meeting. The professor must come away understanding (1) the research
field, (2) our method in full, (3) how it compares with other methods. Proper background and context for every
term and every part of the model. No time/length limit — as robust as possible.

## Content requirements
1. **Terminology**: go through all important terms and explain them; use the **conventional terms of this
   research field** (rigour on terms is explicitly required).
2. **Key papers**: list the papers the information was gathered from; explain the **methods within the key
   papers**. Do proper research to decide which papers and what information to show.
3. **Our method in detail**, with **graphs like those in research papers** (not boxes with text): which layers
   are used, where the probes get their information, the probe structure, the math behind it, and every other
   detail needed for a full breakdown. Learn the figure style from the papers (FIGURE_STYLE_NOTES.md).
4. **For each part: why this way and not another** — show what we tested and why this is the best
   configuration (DESIGN_EVIDENCE.md).
5. **Results compared with other papers in the field** — well reasoned and detailed (COMPARISON_BRIEF.md);
   protocol differences stated; no invalid claims.
6. **Honest gaps**: where the research has gaps, say so.
7. **Current method only.** Old methods (July cascade, LoRA verifier, August heads) appear only where an old
   experiment is the evidence for a shipped design decision.
8. **Temperature**: the shipped method uses a **single T = 0.7**; the per-benchmark map is a labelled
   domain-calibrated variant (needs dataset identity → not realistic at test time). Include the mixed-temperature
   test and the per-modality / per-query temperature idea as future work (with the GEMeX vs VQA-RAD caveat).
9. **Probe-count vs sample-count questions** must be answerable from the material: 24 probes = 3 layers × 8
   seeds, unrelated to N = 8 samples; every probe scores every candidate; works for any N; nothing dropped/reused.
   Anticipate this class of "how does the mechanism behave when X changes" questions generally.

## Form requirements
10. **Colour scheme only** from the old decks (indigo/teal tokens of `meetings/progress_deck_2026-08-17.html`).
    Do NOT copy its page structure or story order. Build pages that convey everything in the best way.
11. **Deck = presentation, not a document**: don't put every word on screen. On screen: key terms, charts,
    method graphs, key numbers — the elements that make the spoken explanation easier to follow. Perfect balance.
12. **Word-for-word script** accompanying the deck, in a format **designed for this deck** (not the old script
    format).
13. **Self-attack loop**: after building, attack the deck as a reader would — every section, blind spots,
    questions — put the answers into the material; repeat until it can answer every question that our
    experiments and information can answer.

## Standing project rules
- Every number verbatim from an artifact or paper, with its source named (in the script and/or a source line
  on the slide). Never a judge-only number without the cross-family judge and exact match beside it.
- Selection efficiency = selected / oracle@N. "n/8" = positive point estimates, not significant wins.
- Judge of record = MedVLThinker-32B; cross-family judge = MedGemma-27B-it. MedGemma-4b replication withdrawn.
- Features in the headline come from a separate teacher-forced pass at 1,003,520 px (generation at cap320);
  capture-inside-generation shown only for an earlier probe (24 / 2,345 picks, accuracy 0.0000 [−0.0021,+0.0021]).
- Layer ℓ = residual stream after decoder block ℓ of 28 (hidden_states[0] = embeddings).
- No abstention framing.
