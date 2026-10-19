# Build sources — professor report deck, 2026-10-19

The built deck and script sit one level up: `meetings/probe_verifier_report_2026-10-19.html`, `..._script.md`, `..._script.docx`.
Published (private) at https://claude.ai/artifact/12ySQfZKBwGHrauuRrYUtf.

## Rebuild

```bash
cd ~/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15
python3 meetings/deck_src_2026-10-19/build_deck.py
```

This writes `meetings/deck_src_2026-10-19/out/` (deck HTML + script md/docx). Copy the three files up to `meetings/` to replace the
committed versions. The docx step needs `pypandoc`.

## What is here

| file | role |
|---|---|
| `build_deck.py` | assembles the deck and the word-for-word script |
| `deck_lib.py` | slide frame, CSS, keyboard navigation, chart helpers (`forest`, `bars`, `lines`) |
| `deck_part1.py` / `deck_part2.py` / `deck_part3.py` | slides + script: opening, vocabulary, papers / method, why / results, comparison, limits, backup |
| `deck_figs.py`, `figs_v2.py` | the method figures (SVG) |
| `deck_data.py` | loads every artifact the slides quote (from `results/cascade_methods/artifacts/`) |
| `NUMBERS_T07.md/.json`, `extra_numbers.json`, `worked_example_12231.json` | the shipped method's numbers and the worked example |
| `numbers_review_part{1,2,3}.json` | numbers computed during review; each entry names the script that produced it |
| `DECK_SPEC.md` | Leo's requirements for the deck |
| `PAPER_NOTES_*.md`, `DESIGN_EVIDENCE.md`, `COMPARISON_BRIEF.md` | paper and design-evidence notes the slides cite (COMPARISON_BRIEF has some pre-T=0.7 numbers) |
| `review/R_*.md`, `R2_*.md`, `R3_*.md` | the three review rounds; `review/scr/` holds the scripts behind review numbers |
| `overflow.py`, `shoot.py`, `sheets.py`, `slide_text.py` | layout check, screenshots, contact sheets, text dump |

## Still pointing at the session's scratch folder (not needed to rebuild)

`make_publish.py` (cached KaTeX fonts), `shoot.sh` (local headless Chromium + libraries), and `review/scr/ref_lookup.py`,
`w1_cards*.py`, `w1_count.py` (the literature-guide build). Re-point them if you need them again.
