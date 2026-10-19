#!/usr/bin/env python3
"""slide_text.py — dump the visible text of every slide (incl. SVG labels) to out/slide_text.md for reviewers."""
import re, html
D = "/home/jamesyang/medvlthinker-imgdiff-compute/.claude/worktrees/exp-2026-10-15/meetings/deck_src_2026-10-19/out/"
s = open(D + "probe_verifier_report_2026-10-19.html").read()
s = re.sub(r'<image [^>]*/>', '', s)
out = []
for k, m in enumerate(re.finditer(r'<section class="slide"[^>]*data-title="([^"]*)"[^>]*>(.*?)</section>', s, re.S), 1):
    body = m.group(2)
    body = re.sub(r'<span class="katex-mathml">.*?</span>', '', body, flags=re.S)
    body = re.sub(r'<(br|/tr|/li|/div|/p|/h1|/h3|/text|/table)[^>]*>', '\n', body)
    body = re.sub(r'<td[^>]*>', ' | ', body)
    txt = html.unescape(re.sub(r'<[^>]+>', '', body))
    txt = "\n".join(l.strip() for l in txt.splitlines() if l.strip())
    out.append(f"## Slide {k} — {html.unescape(m.group(1))}\n\n{txt}\n")
open(D + "slide_text.md", "w").write("\n".join(out))
print(len(out), "slides")
