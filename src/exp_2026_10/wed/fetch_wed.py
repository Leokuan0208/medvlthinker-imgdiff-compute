"""Fetch the WED (Wasserstein Equilibrium Decoding, arXiv 2605.18313) code + paper text into /data/dan/repos.

Uses urllib only (no shell git). Code: GitHub archive zip of luca-hagen/Wasserstein-BDG-medical-VQA (main).
Paper: arXiv HTML rendering, stripped to plain text at /data/dan/repos/wed_paper.txt.
"""
import html, io, os, re, urllib.request, zipfile

DEST = "/data/dan/repos"
REPO_ZIP = "https://github.com/luca-hagen/Wasserstein-BDG-medical-VQA/archive/refs/heads/main.zip"
PAPER = "https://arxiv.org/html/2605.18313"
UA = {"User-Agent": "Mozilla/5.0"}

os.makedirs(DEST, exist_ok=True)
if not os.path.isdir(os.path.join(DEST, "Wasserstein-BDG-medical-VQA-main")):
    zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(REPO_ZIP, timeout=60).read())).extractall(DEST)
    print("repo extracted")

raw = urllib.request.urlopen(urllib.request.Request(PAPER, headers=UA), timeout=60).read().decode("utf8", "ignore")
open(os.path.join(DEST, "wed_paper.html"), "w").write(raw)
t = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S)
t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t)
t = re.sub(r"[ \t]+", " ", t); t = re.sub(r"\n\s*\n+", "\n", t)
open(os.path.join(DEST, "wed_paper.txt"), "w").write(t)
print("paper chars", len(t))
