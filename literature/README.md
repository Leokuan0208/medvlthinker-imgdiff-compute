# literature/ — the domain catch-up package (2026-09-16)

**Start here → `DOMAIN_GUIDE_2026-09-16.md`** (or the `.html` for comfortable reading, or the
`.docx` if someone wants to comment inline). All three are the same document.

## What this is

A single catch-up guide for the research domain around the **live** method in this project — the MLP
probe on frozen Lingshu-7B hidden states used as a best-of-8 verifier for open-ended medical VQA.
It exists because progress reports were reading as though the work had no prior art, no external
comparisons, and non-standard vocabulary. The guide fixes all three.

## The document in one line per section

| § | what it gives you |
|---|---|
| 0 | how to read it; the provenance rules |
| 1 | **the vocabulary** — what the field calls the things we do, including why "MLP" is the wrong word for our probe |
| 2 | **where we stand** — the method, the eight benchmarks, every result, and the honest holes, each naming its artifact |
| 3 | **the literature**, nine categories, a card per paper: models, method, datasets, experiments, results, conclusions, and why it matters to us |
| 4 | **positioning** — who did what first, what is genuinely ours, ranked by how well it would survive a reviewer |
| 5 | **the professor report** — the terminology substitutions, the comparison tables, the claims to make and the four not to make, and the questions to expect |
| 6 | **what to read, in what order** |
| 7 | what is in this package |

## Files

- `DOMAIN_GUIDE_2026-09-16.{md,html,docx}` — the guide.
- `references.bib` — one BibTeX entry per carded paper, ready for the paper draft.
- `MANIFEST.md` — filename → citation → which section cites it.
- `papers/` — the ★ core PDFs. **This is a symlink** to
  `/data/dan/literature/domain_package_2026-09-16/papers/`, because the main disk must stay under
  75 %. The PDFs are not in git; everything else here is.

## Provenance

Every arXiv identifier was checked against the arXiv API on 2026-09-16 — including all 156 cited
across this project's own documents, every one of which resolved. Numbers inside a card are copied
from the paper with a tag saying where (`[abstract]`, `[§4.2]`, `[Table 3]`); `not extracted` means
the source did not state it. Nothing is estimated. Papers about abstention / selective prediction as
a method are excluded by CLAUDE.md rule 6.

## Keeping it current

Four of the closest papers appeared within ninety days of this build. To refresh, put new arXiv ids
one per line in a file and run the verifier described in §6.4 of the guide; the search terms that
surfaced the closest work are listed there too.
