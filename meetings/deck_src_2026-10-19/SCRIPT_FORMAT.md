# Speaker-script format (Leo, 2026-10-19)

Leo wants "what to say" as **bullet points, not paragraphs**, with a way to find, for any spot on a slide, the
matching script points. The method:

1. Each slide's `say` is a list of **groups**. A group starts with a line `# <label>` where `<label>` is the
   **exact visible text** of the on-screen element the group talks about: a card/box title (the bold `b.k` heading),
   a figure stage label ("1 Sample", "Re-read & tap"), a chart or table title, a table column header, or the slide
   title for a general opening. Copy the text exactly as rendered (case included), so Leo can match by eye.
   Optionally add where it is: `# loss: class-weighted binary cross-entropy (left, 2nd box)`.
2. Order the groups in the order Leo should speak, which normally follows the slide top-left → bottom-right.
   Use one opening group `# <slide title>` only if there is something to say before pointing at any element.
3. Under each group: `- ` bullets, one idea per bullet, short (aim ≤ 20 words), speakable as written — Leo reads
   them nearly verbatim. Sub-bullets `  - ` (two spaces) only for a list inside an idea. No paragraphs, no lines
   that are neither `# ` nor `- `/`  - `.
4. **Keep every number and every claim** of the current `say` text; condense wording, do not drop qualifiers
   ("tie", "not measured", "judge only", "August probe", "half A"). Do not add numbers. `say_check.py check`
   compares the multiset of numbers per slide against `say_snapshot.json` and must report no lost/added numbers
   for your slides (if a number is a spoken word like "two and a half times", it may stay a word, but then it was a
   word before too — do not convert words to digits or digits to words).
5. Every element on the slide that matters gets a group; an element nobody needs to talk about can be skipped.
6. Do not change slide content (body HTML), `look`, `qa`, `src`, `minutes` — only `say`. Exception: if a group's
   label does not exist on screen because the slide has no heading for that element, prefer a label that does exist
   (chart title, column header) rather than inventing one.
7. Dividers and slides with empty `say` stay empty.
