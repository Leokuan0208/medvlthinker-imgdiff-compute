"""Dump, for slides built by deck_part2, the on-screen labels and the current say text."""
import html, re, sys
HERE = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19"
sys.path.insert(0, HERE)
from deck_lib import Deck
import deck_part1, deck_part2
d = Deck()
deck_part1.build(d)
start = len(d.slides)
deck_part2.build(d)
for (sid, title, h), s in zip(d.slides[start:], d.script[start:]):
    labs = re.findall(r'<b class="k">(.*?)</b>', h) + re.findall(r"<h3[^>]*>(.*?)</h3>", h)
    th = re.findall(r"<th[^>]*>(.*?)</th>", h)
    svg = [t for t in re.findall(r"<text[^>]*font-weight=\"(?:600|700)\"[^>]*>(.*?)</text>", h)][:20]
    clean = lambda x: html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
    print(f"=== {s['n']} [{sid}] {title}")
    print("LABELS:", " | ".join(clean(x) for x in labs))
    if th: print("TH:", " | ".join(clean(x) for x in th))
    if svg: print("SVG:", " | ".join(clean(x) for x in svg))
    print("SAY:", s["say"].strip())
    print()
