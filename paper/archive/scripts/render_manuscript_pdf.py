#!/usr/bin/env python3
"""Render paper/manuscript_final_2026-07.md -> .pdf via markdown + xhtml2pdf/pisa.

Mirrors the repo's established Markdown->PDF path (no LaTeX on this VM). DejaVu fonts are
registered for the Unicode glyphs (phi, tau, approx, minus, arrows, x, >=, <=, middot).
Usage: python3 paper/render_manuscript_pdf.py [in.md] [out.pdf]
"""
import os, sys, re, html as _html
import markdown
from xhtml2pdf import pisa

ROOT = os.path.expanduser("~/medvlthinker-imgdiff-compute")
IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper/manuscript_final_2026-07.md")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(IN)[0] + ".pdf"
FONTDIR = "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf"

md_text = open(IN, encoding="utf-8").read()

# Emoji that DejaVu cannot render -> plain-text tokens (keeps the PDF clean).
for emo, rep in [("✅", "[YES]"), ("⛔", "[RULE]"), ("\U0001f916", "[bot]"),
                 ("⭐", "*"), ("✓", "v"), ("✗", "x")]:
    md_text = md_text.replace(emo, rep)

body = markdown.markdown(
    md_text,
    extensions=["tables", "fenced_code", "sane_lists", "toc"],
    output_format="html5",
)

CSS = """
@page { size: A4; margin: 1.6cm 1.5cm 1.7cm 1.5cm; @frame footer {
    -pdf-frame-content: footerContent; bottom: 0.7cm; margin-left: 1.5cm; margin-right: 1.5cm; height: 1cm; } }
@font-face { font-family: DejaVu; src: url("%(d)s/DejaVuSans.ttf"); }
@font-face { font-family: DejaVu; font-weight: bold; src: url("%(d)s/DejaVuSans-Bold.ttf"); }
@font-face { font-family: DejaVu; font-style: italic; src: url("%(d)s/DejaVuSans-Oblique.ttf"); }
@font-face { font-family: DejaVu; font-weight: bold; font-style: italic; src: url("%(d)s/DejaVuSans-BoldOblique.ttf"); }
@font-face { font-family: DejaVuMono; src: url("%(d)s/DejaVuSansMono.ttf"); }
body { font-family: DejaVu; font-size: 8.4pt; line-height: 1.35; color: #16181d; }
h1 { font-size: 15pt; color: #10233f; margin: 0 0 6pt 0; line-height: 1.25; }
h2 { font-size: 11.5pt; color: #123a63; border-bottom: 1px solid #c9d3e0; padding-bottom: 2pt;
     margin: 13pt 0 5pt 0; -pdf-keep-with-next: true; }
h3 { font-size: 9.6pt; color: #1b4a7a; margin: 9pt 0 3pt 0; -pdf-keep-with-next: true; }
p { margin: 4pt 0; text-align: justify; }
em { font-style: italic; } strong { font-weight: bold; }
code, pre { font-family: DejaVuMono; font-size: 7.0pt; }
pre { background: #f4f6fa; border: 1px solid #dbe2ec; border-radius: 3px; padding: 4pt 6pt; margin: 5pt 0; }
code { background: #eef1f6; padding: 0 1pt; }
blockquote { background: #fbf7ec; border-left: 3px solid #d8b45a; margin: 6pt 0; padding: 3pt 9pt; color: #34302a; }
table { width: 100%%; border-collapse: collapse; margin: 5pt 0; font-size: 6.3pt; }
th { background: #123a63; color: #ffffff; font-weight: bold; padding: 1.3pt 1.8pt; border: 0.5pt solid #123a63; text-align: left; }
td { padding: 1.2pt 1.8pt; border: 0.5pt solid #cdd6e2; }
tr:nth-child(even) td { background: #f2f5fa; }
hr { border: none; border-top: 1px solid #c9d3e0; margin: 9pt 0; }
ul, ol { margin: 4pt 0 4pt 3pt; } li { margin: 2pt 0; }
a { color: #1b4a7a; text-decoration: none; }
.footer { font-size: 7pt; color: #8a94a3; text-align: center; }
""" % {"d": FONTDIR}

doc = ("<html><head><meta charset='utf-8'><style>%s</style></head><body>%s"
       "<div id='footerContent' class='footer'>Med-VLM cascade - final manuscript 2026-07 - "
       "page <pdf:pagenumber> of <pdf:pagecount></div></body></html>") % (CSS, body)

with open(OUT, "wb") as f:
    status = pisa.CreatePDF(src=doc, dest=f, encoding="utf-8")

if status.err:
    print("PDF render FAILED with %d errors" % status.err)
    sys.exit(1)
print("wrote %s (%d KiB)" % (OUT, os.path.getsize(OUT) // 1024))
