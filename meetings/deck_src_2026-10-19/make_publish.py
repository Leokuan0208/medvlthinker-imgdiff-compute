#!/usr/bin/env python3
"""make_publish.py — turn the built deck into the publishable page: no document wrapper (the publisher adds one),
KaTeX CSS inlined with its woff2 fonts as data URIs (the artifact CSP allows external stylesheets only from Google Fonts),
and an explicit dark colour-scheme for the dark stage."""
import base64, os, re, urllib.request

T = "/home/jamesyang/.claude/jobs/61c9eff9/tmp/"
SRC = T + "deck/out/probe_verifier_report_2026-10-19.html"
OUT = T + "deck/publish/probe_verifier_report.html"
KX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/"
FONTS = T + "katex/fonts/"
os.makedirs(FONTS, exist_ok=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)

css = open(T + "katex/katex.min.css").read()


def font_uri(name):
    p = FONTS + name
    if not os.path.exists(p):
        urllib.request.urlretrieve(KX + "fonts/" + name, p)
    return "data:font/woff2;base64," + base64.b64encode(open(p, "rb").read()).decode()


# keep only the woff2 source of each @font-face, embedded
def fix_face(m):
    face = m.group(0)
    w2 = re.search(r"url\(fonts/([^)]+\.woff2)\) format\(\"woff2\"\)", face)
    return re.sub(r"src:[^;}]+", f'src:url({font_uri(w2.group(1))}) format("woff2")', face) if w2 else face


css = re.sub(r"@font-face\{[^}]*\}", fix_face, css)
assert "url(fonts/" not in css

h = open(SRC).read()
head = re.search(r"<head>(.*?)</head>", h, re.S).group(1)
body = re.search(r"<body[^>]*>(.*)</body>", h, re.S).group(1)
head = re.sub(r'<meta charset[^>]*>\s*', "", head)
head = re.sub(r'<meta name="viewport"[^>]*>\s*', "", head)
head = re.sub(r'<link[^>]*katex\.min\.css[^>]*>', f"<style>{css}</style>", head)
assert "katex.min.css" not in head
title = re.search(r"<title>.*?</title>", head).group(0)
head = head.replace(title, "")
page = (title + "\n<style>:root{color-scheme:dark}</style>\n" + head.strip() + "\n" + body.strip() + "\n")
open(OUT, "w").write(page)
print(OUT, len(page.encode()) // 1024, "KB")
