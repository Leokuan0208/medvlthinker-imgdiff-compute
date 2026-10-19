#!/usr/bin/env python3
"""build_deck.py — assemble the professor deck (HTML) and its word-for-word script (Markdown + DOCX)."""
import sys
sys.path.insert(0, "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19")
from deck_lib import Deck
import deck_part1, deck_part2, deck_part3

OUT = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/out"
import os
os.makedirs(OUT, exist_ok=True)
NAME = "probe_verifier_report_2026-10-19"

d = Deck()
deck_part1.build(d)
deck_part2.build(d)
deck_part3.build(d)
open(f"{OUT}/{NAME}.html", "w").write(d.html("Probe Verifier Report"))

# ------------------------------------------------------------------ script
total = sum(s["minutes"] for s in d.script)
L = [f"# Speaker script — {NAME}.html", "",
     f"{len(d.slides)} slides · main talk ≈ {total:.0f} minutes · backup slides are only for questions.", "",
     "**How to use this script.** Each slide has three parts:",
     "- **On screen** — what the audience is looking at, and where to point.",
     "- **Say** — what to say, as bullet points. Each group starts with **▸ the exact label you see on the slide** "
     "(a box title, figure stage, chart or table), so you can find the matching script for any part of the slide. "
     "Numbers are the same numbers as on the slide.",
     "- **If asked** — likely questions with a sourced answer. Don't volunteer these unless asked.", "",
     "Every number comes from a named artifact or paper; the source line at the bottom of each slide names it.", "",
     "**Finding your place.** Slide numbers match the number at the bottom-right of each slide. In Word, open "
     "View → Navigation Pane: it lists every slide and, under each, its on-screen labels — click one to jump there.", ""]
part = None
for s in d.script:
    if s["part"] != part:
        part = s["part"]
        L += ["", f"---", "", f"## {part}", ""]
    title = s["title"] or "(section divider)"
    L.append(f"### {s['n']} · {title}" + (f"  ·  ~{s['minutes']:g} min" if s["minutes"] else ""))
    if s["look"]:
        L.append(f"*On screen:* {s['look']}")
        L.append("")
    if s["say"]:
        L.append("**Say:**")
        L.append("")
        # say format: "# <exact on-screen label>" starts a group; "- " is a bullet; "  - " a sub-bullet
        for line in s["say"].strip().split("\n"):
            if not line.strip():
                continue
            if line.lstrip().startswith("# "):
                L += ["", f"#### ▸ {line.lstrip()[2:].strip()}", ""]
            elif line.startswith("  - ") or line.startswith("    - "):
                L.append("    - " + line.strip()[2:])
            elif line.lstrip().startswith("- "):
                L.append("- " + line.strip()[2:])
            else:
                L.append(line.strip())
        L.append("")
    if s["qa"]:
        L.append("**If asked:**")
        L.append("")
        for q, a in s["qa"]:
            L.append(f"- *{q}* — {a}")
        L.append("")
    if s["src"]:
        L.append(f"<sub>Source: {s['src']}</sub>")
        L.append("")
md = "\n".join(L)
open(f"{OUT}/{NAME}_script.md", "w").write(md)
try:
    import pypandoc
    pypandoc.convert_file(f"{OUT}/{NAME}_script.md", "docx", outputfile=f"{OUT}/{NAME}_script.docx",
                          extra_args=["--toc", "--toc-depth=3"])
except Exception as e:
    print("docx failed:", e)
print(f"slides {len(d.slides)}; minutes {total:.1f}; wrote {OUT}")
