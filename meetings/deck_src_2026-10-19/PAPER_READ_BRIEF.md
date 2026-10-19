# Paper-reading brief for the professor deck (read-only; write only your notes file)

We are building a professor-facing presentation on our method and its field. Read
`/home/jamesyang/.claude/jobs/61c9eff9/tmp/deck/DECK_SPEC.md` first (what the deck must achieve).

**Our method (for relevance judgements):** Lingshu-7B (a Qwen2.5-VL-7B medical finetune) samples N = 8 answers
per open-ended medical VQA question at T = 0.7 (cap320 image budget). Each distinct answer is re-read
(teacher-forced) and its hidden states at decoder layers 18/20/22 are mean-pooled over the answer tokens. 24 MLP
probes (Linear 3584→256, GELU, Linear 256→1; BCE on judge correctness labels; 3 layers × 8 seeds) score each
candidate; ranks within the question are averaged, ties broken by vote count; the top answer is returned.
Evaluated on 8 open-ended benchmarks (PathVQA, SLAKE, VQA-RAD, RadImageNet-VQA, Kvasir-VQA-x1, OmniMedVQA
converted to open, VQA-Med 2019 abnormality, GEMeX), with two-fold image cross-fitting, a MedVLThinker-32B judge,
a cross-family MedGemma-27B judge, exact match, token-F1 and LLaVA-Med recall. Headline (macro over 8, vs greedy):
judge +0.0798, MedGemma +0.0710, EM +0.0121.

**Sources.**
- Core PDFs: `/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/lit-domain-package/literature/papers/`
  (PyMuPDF: `python3 -c "import fitz"`; extract text to `/home/jamesyang/.claude/jobs/61c9eff9/tmp/txt/` — some
  extractions may already exist there; `grep -a` them). Render a page with `page.get_pixmap(dpi=110).save(...)`
  and view it with the Read tool when you need to see a figure or table layout.
- Cards (the guide's summaries of every paper, incl. non-core ones):
  `/home/jamesyang/.claude/jobs/61c9eff9/tmp/lit/cards_<X>.json` — fields like id, title, arxiv, one_line, method,
  datasets, experiments, results, conclusions, relation_to_us, priority, core.
- Existing deck research to avoid redoing (read, then go deeper): `FIGURE_STYLE_NOTES.md`, `COMPARISON_BRIEF.md`,
  `DESIGN_EVIDENCE.md` in `/home/jamesyang/.claude/jobs/61c9eff9/tmp/deck/`.
- For a non-core paper you rate as deck-worthy, you may fetch its arXiv abstract page (WebFetch
  `https://arxiv.org/abs/<id>`) to verify what the card says. Do not fetch more than ~6 per group.

**Rules.** No fabricated numbers: every number you write must be verbatim from the PDF (give table/figure/§ and
page) or from the abstract (tag [abstract]). If something isn't stated, write "not stated". Mark anything you infer
as (inferred). Use simple shell commands only (a guard refuses complex compound commands).

**Deliver** `/home/jamesyang/.claude/jobs/61c9eff9/tmp/deck/PAPER_NOTES_<GROUP>.md` with:

### Part 1 — every core PDF in your group, read in full
For each: full citation (authors, title, arXiv id/venue, year) ·
1. **Problem** (one paragraph, plain words).
2. **Method, precisely**: architecture/pipeline, equations (write them in LaTeX), what is trained vs frozen, inputs,
   outputs, loss, inference procedure, N, temperature, selection rule.
3. **Setup**: models, datasets (sizes), metrics, grader/judge, baselines.
4. **Key results**: the 3–8 numbers that matter, each with table/figure and page; include the baseline each is
   measured against.
5. **Terminology** it introduces or uses (conventional names), with its definition in the paper's words.
6. **Figures worth recreating or citing** (figure number, page, what it shows, how it is drawn).
7. **What it gives our deck**: which slide(s) it supports (terminology / background / method justification /
   comparison / limitation), the one sentence a presenter should say about it, and any **finding that cuts against
   our choices** (volunteer these).
8. **Claims about it we must not make.**

### Part 2 — every card in your categories (incl. non-core)
A table: id | title | one line | deck relevance (slide-worthy / mention / skip) | why. Then, for every card rated
slide-worthy or mention, 3–5 lines of verified content the deck can use (with source tags).

### Part 3 — recommendations
Which papers deserve their own slide, which belong in a comparison table, which in a terminology slide; terms the
deck must define; any errors you found in the cards or in the three existing deck briefs (with evidence).
