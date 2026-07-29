# Paper — current version & naming convention

## ➜ CURRENT paper
**`adaptive-cascade-medvqa_ieee_2026-07-08.pdf`** (source: `adaptive-cascade-medvqa_ieee_2026-07-08.tex`)

Rebuild: `bash build_ieee.sh adaptive-cascade-medvqa_ieee_2026-07-08.tex`
(tectonic + IEEEtran; figures come from `make_ieee_figs.py` → `figs_final/`).

## Naming convention (so the latest is always obvious)
```
<topic-slug>_<venue-or-type>_<YYYY-MM-DD>.{tex,pdf}
```
- The **date is when that version was produced** — the newest date is the current version.
- One canonical version lives at the top of `paper/`; everything superseded moves to `archive/`.
- No more "final / final2 / main" — the date disambiguates.

## Files at top level (current only)
| file | purpose |
|---|---|
| `adaptive-cascade-medvqa_ieee_2026-07-08.{tex,pdf}` | **the current paper** |
| `build_ieee.sh` | IEEE build wrapper (tectonic) |
| `make_ieee_figs.py` | generates the current paper's figures → `figs_final/` |
| `IEEEtran.cls` | IEEE document class |
| `build_report_v2.py`, `build_professor_html.py` | HTML progress-report builders (not the paper) |

## `archive/` — superseded drafts & tooling (kept for the record, not deleted)
| file | what it was |
|---|---|
| `manuscript_final_2026-07.{md,pdf}` | comprehensive Markdown draft — superseded by the IEEE rewrite |
| `manuscript_2026-07_longform.{md,pdf}` | earlier long-form Markdown |
| `conference_2026-07.{md,pdf}` | earlier short conference version |
| `cvgip2026_draft.md`, `cvgip2026_ieee.{tex,pdf}` | earliest CVGIP drafts |
| `hello_ieee.{tex,pdf}` | IEEE toolchain smoke test |
| `scripts/` | old-manuscript figure/render/util scripts |
