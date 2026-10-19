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
     "- **Say** — the words to speak, verbatim. Numbers in the text are the same numbers as on the slide.",
     "- **If asked** — likely questions with a sourced answer. Don't volunteer these unless asked.", "",
     "Every number comes from a named artifact or paper; the source line at the bottom of each slide names it.", ""]
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
        for para in s["say"].strip().split("\n"):
            L.append(para.strip())
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
    pypandoc.convert_file(f"{OUT}/{NAME}_script.md", "docx", outputfile=f"{OUT}/{NAME}_script.docx")
except Exception as e:
    print("docx failed:", e)
print(f"slides {len(d.slides)}; minutes {total:.1f}; wrote {OUT}")
